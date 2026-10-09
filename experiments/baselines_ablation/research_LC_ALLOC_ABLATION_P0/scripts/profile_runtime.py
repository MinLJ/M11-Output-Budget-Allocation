from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import platform
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from common import BUDGETS, SEEDS, add_p1_scripts, append_log, load_model, sha256_file, stable_sigmoid, verify_input_binding, write_json, write_parquet  # noqa: E402
from predict_allocate import next_slot_greedy  # noqa: E402

ORDER_SEED = 530003
ATOL, RTOL = 1e-6, 1e-5
PROFILE_BUDGET = 15


@dataclass
class RawImage:
    image_id: str
    width: int
    height: int
    candidates: pd.DataFrame
    full_logits: np.ndarray
    full_embeddings: np.ndarray


def import_file(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def qsummary(values: list[float]) -> dict:
    a = np.asarray(values, dtype=np.float64)
    return {
        "samples": len(a), "mean_ms": float(a.mean()), "median_ms": float(np.median(a)),
        "p95_ms": float(np.quantile(a, 0.95)), "min_ms": float(a.min()), "max_ms": float(a.max()),
        "std_ms": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
        "cv": float(a.std(ddof=1) / a.mean()) if len(a) > 1 and a.mean() else 0.0,
    }


def ledger_hash(path: Path, ledger_path: Path) -> str:
    ledger = pd.read_csv(ledger_path)
    target = str(path.resolve()).replace("\\", "/").lower()
    matches = []
    for row in ledger.itertuples(index=False):
        raw = Path(str(row.path))
        candidate = raw if raw.is_absolute() else ledger_path.parent / raw
        if str(candidate.resolve()).replace("\\", "/").lower() == target:
            matches.append(str(row.sha256).lower())
    if len(matches) != 1 or sha256_file(path).lower() != matches[0]:
        raise RuntimeError(f"frozen runtime dependency hash mismatch: {path}")
    return matches[0]


def load_raw_images(release_root: Path, identity: dict, target_ids: list[str]) -> tuple[dict[str, RawImage], float]:
    started = time.perf_counter()
    asset = identity["assets"]["TRAIN"]
    manifest = pq.read_table(release_root / asset["split_manifest_path"], columns=["image_id", "width", "height"]).to_pandas()
    manifest["image_id"] = manifest["image_id"].astype(str)
    meta = manifest.set_index("image_id")
    targets = set(map(str, target_ids))
    result: dict[str, RawImage] = {}
    cols = ["image_id", "candidate_record_id", "road8_rank", "score", "predicted_road8_class_id", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "bbox_cx", "bbox_cy", "bbox_w", "bbox_h", "query_index"]
    for cand_rel, native_rel in zip(asset["candidate_shards"], asset["native_state_shards"]):
        cand = pq.read_table(release_root / cand_rel, columns=cols, filters=[("road8_rank", "<=", 100)]).to_pandas()
        cand["image_id"] = cand["image_id"].astype(str)
        cand = cand[cand["image_id"].isin(targets)]
        if cand.empty:
            continue
        wanted = set(cand["image_id"].unique())
        with np.load(release_root / native_rel, allow_pickle=False) as z:
            image_ids = z["image_ids"].astype(str)
            query = z["query_index"].astype(np.int32)
            starts = {str(image_ids[ix]): ix for ix in range(0, len(image_ids), 300) if str(image_ids[ix]) in wanted}
            logits, embeddings = z["l3_road8_logits"], z["l3_query_embedding"]
            for image_id, rows in cand.groupby("image_id", sort=False):
                image_id = str(image_id)
                rows = rows.sort_values("road8_rank", kind="stable").reset_index(drop=True)
                if len(rows) != 100 or rows["road8_rank"].astype(int).tolist() != list(range(1, 101)):
                    raise RuntimeError(f"TRAIN Top100 invariant failed {image_id}")
                ix = starts[image_id]
                if not np.array_equal(query[ix:ix + 300], np.arange(300)):
                    raise RuntimeError(f"native query order failed {image_id}")
                q = rows["query_index"].to_numpy(np.int32)
                cls = rows["predicted_road8_class_id"].to_numpy(np.int16) - 1
                block_logits = np.asarray(logits[ix:ix + 300], dtype=np.float32)
                reconstructed = stable_sigmoid(block_logits[q, cls].astype(np.float64))
                if float(np.max(np.abs(reconstructed - rows["score"].to_numpy(np.float64)))) > 1e-6:
                    raise RuntimeError(f"candidate/native score mismatch {image_id}")
                m = meta.loc[image_id]
                result[image_id] = RawImage(image_id, int(m.width), int(m.height), rows, block_logits.copy(), np.asarray(embeddings[ix:ix + 300], dtype=np.float16).copy())
    if set(result) != targets:
        raise RuntimeError(f"FIT profile raw coverage failed missing={list(targets - set(result))[:3]}")
    return result, time.perf_counter() - started


def common_image_arrays(raw: RawImage) -> tuple:
    c = raw.candidates
    score = c["score"].to_numpy(np.float64)
    cls = c["predicted_road8_class_id"].to_numpy(np.int16)
    cx = c["bbox_cx"].to_numpy(np.float64) / raw.width
    cy = c["bbox_cy"].to_numpy(np.float64) / raw.height
    wn = c["bbox_w"].to_numpy(np.float64) / raw.width
    hn = c["bbox_h"].to_numpy(np.float64) / raw.height
    area = wn * hn
    quant = np.quantile(score, [0.25, 0.50, 0.75], method="linear")
    context = np.asarray([
        score.mean(), score.std(ddof=0), *quant, score[:5].mean(), score[:20].mean(), score[:50].mean(),
        np.mean(score > 0.10), np.mean(score > 0.25), np.mean(score > 0.50),
        *[np.mean(cls == z) for z in range(1, 9)], np.median(area),
    ], dtype=np.float64)
    return c, score, cls, cx, cy, wn, hn, area, context


def full_group_features(group: list[RawImage], pca, scaler, standardize: np.ndarray, iou_fn) -> tuple[np.ndarray, np.ndarray, list[list[str]]]:
    prepared, all_embeddings = [], []
    for raw in group:
        c = raw.candidates
        q = c["query_index"].to_numpy(np.int32)
        native_logits = np.asarray(raw.full_logits[q], dtype=np.float64)
        emb = np.asarray(raw.full_embeddings[q], dtype=np.float32)
        prepared.append((raw, native_logits, emb))
        all_embeddings.append(emb)
    all_pca = np.asarray(pca.transform(np.vstack(all_embeddings)), dtype=np.float64)
    rows_out, classes_out, records_out = [], [], []
    p0 = 0
    for raw, native_logits, emb in prepared:
        c, score, cls, cx, cy, wn, hn, area, context = common_image_arrays(raw)
        boxes = c[["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]].to_numpy(np.float64)
        pca32 = all_pca[p0:p0 + 100]
        p0 += 100
        emb_norm = np.linalg.norm(emb.astype(np.float64), axis=1)
        rows = np.empty((45, 90), dtype=np.float64)
        for oi, k in enumerate(range(6, 51)):
            j, prefix_n = k - 1, k - 1
            clipped = np.clip(score[j], 1e-6, 1 - 1e-6)
            local = [score[j], math.log(clipped / (1 - clipped)), k / 100.0]
            local += [float(cls[j] == z) for z in range(1, 9)]
            local += [cx[j], cy[j], wn[j], hn[j], area[j], math.log(max(wn[j], 1e-6) / max(hn[j], 1e-6))]
            local += native_logits[j].tolist() + pca32[j].tolist() + [emb_norm[j]]
            same = np.flatnonzero(cls[:prefix_n] == cls[j])
            if len(same):
                overlap = iou_fn(boxes[j:j + 1], boxes[same])[0]
                distance = np.sqrt((cx[j] - cx[same]) ** 2 + (cy[j] - cy[same]) ** 2)
                rel = [prefix_n, score[:prefix_n].mean(), score[:prefix_n].max(), score[:prefix_n].min(), len(same), len(same) / prefix_n, overlap.max(), overlap.mean(), np.sum(overlap >= 0.30), np.sum(overlap >= 0.50), distance.min(), 1.0]
            else:
                rel = [prefix_n, score[:prefix_n].mean(), score[:prefix_n].max(), score[:prefix_n].min(), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            rows[oi] = np.asarray(local + rel + context.tolist(), dtype=np.float64)
        rows_out.append(rows)
        classes_out.append(cls[5:50])
        records_out.append(c.iloc[:50]["candidate_record_id"].astype(str).tolist())
    x = np.vstack(rows_out)
    x[:, standardize] = scaler.transform(x[:, standardize])
    return x.astype(np.float32), np.concatenate(classes_out), records_out


def no_embed_group_features(group: list[RawImage], scaler, standardize: np.ndarray, embed_idx: np.ndarray, iou_fn) -> tuple[np.ndarray, np.ndarray, list[list[str]]]:
    # Deliberately does not access raw.full_embeddings and never calls PCA.
    rows_out, classes_out, records_out = [], [], []
    for raw in group:
        c, score, cls, cx, cy, wn, hn, area, context = common_image_arrays(raw)
        q = c["query_index"].to_numpy(np.int32)
        native_logits = np.asarray(raw.full_logits[q], dtype=np.float64)
        boxes = c[["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]].to_numpy(np.float64)
        rows = np.zeros((45, 90), dtype=np.float64)
        for oi, k in enumerate(range(6, 51)):
            j, prefix_n = k - 1, k - 1
            clipped = np.clip(score[j], 1e-6, 1 - 1e-6)
            local = [score[j], math.log(clipped / (1 - clipped)), k / 100.0]
            local += [float(cls[j] == z) for z in range(1, 9)]
            local += [cx[j], cy[j], wn[j], hn[j], area[j], math.log(max(wn[j], 1e-6) / max(hn[j], 1e-6))]
            local += native_logits[j].tolist() + [0.0] * 33
            same = np.flatnonzero(cls[:prefix_n] == cls[j])
            if len(same):
                overlap = iou_fn(boxes[j:j + 1], boxes[same])[0]
                distance = np.sqrt((cx[j] - cx[same]) ** 2 + (cy[j] - cy[same]) ** 2)
                rel = [prefix_n, score[:prefix_n].mean(), score[:prefix_n].max(), score[:prefix_n].min(), len(same), len(same) / prefix_n, overlap.max(), overlap.mean(), np.sum(overlap >= 0.30), np.sum(overlap >= 0.50), distance.min(), 1.0]
            else:
                rel = [prefix_n, score[:prefix_n].mean(), score[:prefix_n].max(), score[:prefix_n].min(), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            rows[oi] = np.asarray(local + rel + context.tolist(), dtype=np.float64)
        rows_out.append(rows)
        classes_out.append(cls[5:50])
        records_out.append(c.iloc[:50]["candidate_record_id"].astype(str).tolist())
    x = np.vstack(rows_out)
    x[:, standardize] = scaler.transform(x[:, standardize])
    x[:, embed_idx] = 0.0
    return x.astype(np.float32), np.concatenate(classes_out), records_out


def no_prefix_group_features(group: list[RawImage], pca, scaler, standardize: np.ndarray, prefix_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[list[str]]]:
    # Deliberately does not extract XYXY boxes, find same-class prefix rows, or compute prefix statistics.
    prepared, all_embeddings = [], []
    for raw in group:
        c = raw.candidates
        q = c["query_index"].to_numpy(np.int32)
        native_logits = np.asarray(raw.full_logits[q], dtype=np.float64)
        emb = np.asarray(raw.full_embeddings[q], dtype=np.float32)
        prepared.append((raw, native_logits, emb))
        all_embeddings.append(emb)
    all_pca = np.asarray(pca.transform(np.vstack(all_embeddings)), dtype=np.float64)
    rows_out, classes_out, records_out = [], [], []
    p0 = 0
    for raw, native_logits, emb in prepared:
        c, score, cls, cx, cy, wn, hn, area, context = common_image_arrays(raw)
        pca32 = all_pca[p0:p0 + 100]
        p0 += 100
        emb_norm = np.linalg.norm(emb.astype(np.float64), axis=1)
        rows = np.zeros((45, 90), dtype=np.float64)
        for oi, k in enumerate(range(6, 51)):
            j = k - 1
            clipped = np.clip(score[j], 1e-6, 1 - 1e-6)
            local = [score[j], math.log(clipped / (1 - clipped)), k / 100.0]
            local += [float(cls[j] == z) for z in range(1, 9)]
            local += [cx[j], cy[j], wn[j], hn[j], area[j], math.log(max(wn[j], 1e-6) / max(hn[j], 1e-6))]
            local += native_logits[j].tolist() + pca32[j].tolist() + [emb_norm[j]]
            rows[oi] = np.asarray(local + [0.0] * 12 + context.tolist(), dtype=np.float64)
        rows_out.append(rows)
        classes_out.append(cls[5:50])
        records_out.append(c.iloc[:50]["candidate_record_id"].astype(str).tolist())
    x = np.vstack(rows_out)
    x[:, standardize] = scaler.transform(x[:, standardize])
    x[:, prefix_idx] = 0.0
    return x.astype(np.float32), np.concatenate(classes_out), records_out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", required=True)
    ap.add_argument("--p1-root", required=True)
    ap.add_argument("--p1a-root", required=True)
    ap.add_argument("--p1b-root", required=True)
    ap.add_argument("--release-root", required=True)
    ap.add_argument("--r0-solver", required=True)
    args = ap.parse_args()
    root = Path(args.project_root).resolve()
    p1 = Path(args.p1_root).resolve()
    p1a = Path(args.p1a_root).resolve()
    p1b = Path(args.p1b_root).resolve()
    release = Path(args.release_root).resolve()
    t0 = time.perf_counter()
    append_log(root, "STAGE profile_runtime START")
    verify_input_binding(root)
    p1b_status = json.loads((p1b / "outputs" / "implementation_status.json").read_text(encoding="utf-8"))
    if p1b_status.get("optimization_status") != "ACCEPTED" or p1b_status.get("full_dev_K_mismatch") != 0 or p1b_status.get("full_dev_selection_mismatch") != 0:
        raise RuntimeError("P1B accepted implementation binding failed")
    p1b_ledger = p1b / "input_output_sha256.csv"
    p1a_ledger = p1a / "input_output_sha256.csv"
    p1b_profile_script = p1b / "scripts" / "profile_optimizations.py"
    p1b_environment_path = p1b / "outputs" / "runtime_environment.json"
    r0_solver_path = Path(args.r0_solver).resolve()
    profile_manifest_path = p1a / "qa" / "profile_manifest.parquet"
    ledger_hash(p1b_profile_script, p1b_ledger)
    ledger_hash(p1b_environment_path, p1b_ledger)
    ledger_hash(r0_solver_path, p1b_ledger)
    ledger_hash(profile_manifest_path, p1a_ledger)
    add_p1_scripts(p1)
    from dp_solver import group_choice_values_from_marginals, solve_group_allocations  # noqa: E402
    from p1_core import iou_xyxy  # noqa: E402
    r0 = import_file(r0_solver_path, "lc_ablation_p0_r0_solver")
    p1b_impl = import_file(p1b_profile_script, "lc_ablation_p0_p1b_profile")

    identity = json.loads((release / "CANDIDATE_ASSET_IDENTITY.json").read_text(encoding="utf-8"))
    profile_manifest = pq.read_table(profile_manifest_path).to_pandas().sort_values(["profile_group_id", "position_in_group"])
    profile_ids = profile_manifest["image_id"].astype(str).tolist()
    raw, read_seconds = load_raw_images(release, identity, profile_ids)
    groups = [[raw[x] for x in sorted(profile_manifest[profile_manifest.profile_group_id == group_id]["image_id"].astype(str))] for group_id in range(5)]
    load_start = time.perf_counter()
    pca = joblib.load(p1 / "models" / "pca32.joblib")
    scaler_bundle = joblib.load(p1 / "models" / "feature_scaler.joblib")
    scaler, standardize = scaler_bundle["scaler"], np.asarray(scaler_bundle["standardize_mask"], dtype=bool)
    weights = np.asarray(json.loads((p1 / "models" / "class_weights.json").read_text(encoding="utf-8"))["weights"], dtype=np.float64)
    masks = json.loads((root / "feature_masks.json").read_text(encoding="utf-8"))
    embed_idx = np.asarray(masks["NO_EMBED"]["indices"], dtype=np.int64)
    prefix_idx = np.asarray(masks["NO_PREFIX"]["indices"], dtype=np.int64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    expected_env = json.loads(p1b_environment_path.read_text(encoding="utf-8"))
    current_env = {
        "device": str(device), "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "torch_num_threads": torch.get_num_threads(), "torch_num_interop_threads": torch.get_num_interop_threads(),
    }
    for key, actual in current_env.items():
        if actual != expected_env.get(key):
            raise RuntimeError(f"runtime environment differs from P1B for {key}: {actual!r} != {expected_env.get(key)!r}")
    model_specs = {"FULL": (p1 / "models",), "NO_EMBED": (root / "models" / "NO_EMBED",), "NO_PREFIX": (root / "models" / "NO_PREFIX",)}
    models, temperatures = {}, {}
    for variant, (model_root,) in model_specs.items():
        for seed in SEEDS:
            models[(variant, seed)] = load_model(model_root / f"marginal_mlp_seed_{seed}.pt", device)
            temperatures[(variant, seed)] = float(json.loads((model_root / f"temperature_seed_{seed}.json").read_text(encoding="utf-8"))["temperature"])
    sync(device)
    load_seconds = time.perf_counter() - load_start

    @torch.inference_mode()
    def infer(x: np.ndarray, variant: str, seed: int) -> np.ndarray:
        xt = torch.from_numpy(np.asarray(x, dtype=np.float32)).to(device)
        logits = models[(variant, seed)](xt).float().cpu().numpy().astype(np.float64)
        sync(device)
        return stable_sigmoid(logits / temperatures[(variant, seed)])

    def allocate_from_features(x: np.ndarray, classes: np.ndarray, records: list[list[str]], variant: str, seed: int, budget: int, solver: str = "DP"):
        prob = infer(x, variant, seed)
        margins = (np.ascontiguousarray(prob).mean(axis=1) * weights[classes - 1]).reshape(40, 45)
        if solver == "DP":
            k_values, objectives = solve_group_allocations(margins, [budget])
            k = k_values[0]
            objective = float(objectives[0])
        elif solver == "GREEDY":
            k = next_slot_greedy(margins, (budget,))[budget]
            choices = group_choice_values_from_marginals(margins)
            objective = float(sum(choices[i, int(k[i]) - 5] for i in range(40)))
        else:
            raise ValueError(solver)
        selected = [records[i][:int(k[i])] for i in range(40)]
        return k, selected, objective, prob

    def full_path(group, seed, budget, solver="DP"):
        x, c, r = full_group_features(group, pca, scaler, standardize, iou_xyxy)
        return allocate_from_features(x, c, r, "FULL", seed, budget, solver)

    def no_embed_path(group, seed, budget):
        x, c, r = no_embed_group_features(group, scaler, standardize, embed_idx, iou_xyxy)
        return allocate_from_features(x, c, r, "NO_EMBED", seed, budget, "DP")

    def no_prefix_path(group, seed, budget):
        x, c, r = no_prefix_group_features(group, pca, scaler, standardize, prefix_idx)
        return allocate_from_features(x, c, r, "NO_PREFIX", seed, budget, "DP")

    def minimal_view(group):
        result = []
        for raw_image in group:
            c = raw_image.candidates
            scores = c["score"].to_numpy(np.float64)
            ids = c["candidate_record_id"].astype(str).tolist()
            result.append((raw_image.image_id, scores, ids))
        return result

    profile_views = [minimal_view(group) for group in groups]

    def score_fast(view, budget):
        ids = [x[0] for x in view]
        values = np.vstack([x[1][:50] for x in view])
        k, objective = r0.allocate_prefixes(ids, values, 40 * budget, 5, 50)
        return k, [view[i][2][:int(k[i])] for i in range(40)], float(objective)

    def score_reference(group, budget):
        ids, scores, records = [], [], []
        for raw_image in group:
            rec = raw_image.candidates.to_dict("records")
            values = np.asarray([float(x["score"]) for x in rec], dtype=np.float64)
            order = r0.rank_candidates(rec, values)
            ranked = [rec[int(j)] for j in order[:50]]
            ids.append(raw_image.image_id)
            scores.append([float(x["score"]) for x in ranked])
            records.append([str(x["candidate_record_id"]) for x in ranked])
        k, objective = r0.allocate_prefixes(ids, np.asarray(scores, dtype=np.float64), 40 * budget, 5, 50)
        return k, [records[i][:int(k[i])] for i in range(40)], float(objective)

    # Specialized-builder acceptance: reference full feature then mask, versus
    # a path that genuinely omits the masked computation.
    parity_rows = []
    upstream_rows = []
    for group_id, group in enumerate(groups):
        x_full, c_full, r_full = full_group_features(group, pca, scaler, standardize, iou_xyxy)
        x_upstream, c_upstream, r_upstream = p1b_impl.optimized_group_features(group, pca, scaler, standardize, iou_xyxy)
        full_gap = float(np.max(np.abs(x_full.astype(np.float64) - x_upstream.astype(np.float64))))
        full_ok = bool(np.allclose(x_full, x_upstream, atol=ATOL, rtol=RTOL)) and np.array_equal(c_full, c_upstream) and r_full == r_upstream
        if not full_ok:
            raise RuntimeError(f"FULL feature path differs from accepted P1B implementation group={group_id} gap={full_gap}")
        for budget in BUDGETS:
            k_ref, s_ref, o_ref = score_reference(group, budget)
            k_fast, s_fast, o_fast = score_fast(profile_views[group_id], budget)
            if not np.array_equal(k_ref, k_fast) or s_ref != s_fast or np.float64(o_ref).tobytes() != np.float64(o_fast).tobytes():
                raise RuntimeError(f"S_ADAPT fast/reference parity failed group={group_id} budget={budget}")
            upstream_rows.append({
                "profile_group_id": group_id, "budget": budget, "full_feature_max_abs_vs_p1b": full_gap,
                "full_feature_within_tolerance": full_ok, "s_adapt_K_mismatch_images": int(np.sum(k_ref != k_fast)),
                "s_adapt_selection_match": s_ref == s_fast, "s_adapt_objective_bitwise_equal": True, "pass": True,
            })
        for variant, indices, builder in (
            ("NO_EMBED", embed_idx, lambda: no_embed_group_features(group, scaler, standardize, embed_idx, iou_xyxy)),
            ("NO_PREFIX", prefix_idx, lambda: no_prefix_group_features(group, pca, scaler, standardize, prefix_idx)),
        ):
            x_reference = x_full.copy()
            x_reference[:, indices] = 0.0
            x_special, c_special, r_special = builder()
            max_abs = float(np.max(np.abs(x_reference.astype(np.float64) - x_special.astype(np.float64))))
            feature_ok = bool(np.allclose(x_reference, x_special, atol=ATOL, rtol=RTOL)) and bool(np.all(x_special[:, indices] == 0.0)) and np.array_equal(c_full, c_special) and r_full == r_special
            if not feature_ok:
                raise RuntimeError(f"specialized feature parity failed variant={variant} group={group_id} max_abs={max_abs}")
            for seed in SEEDS:
                p_ref = infer(x_reference, variant, seed)
                p_special = infer(x_special, variant, seed)
                prob_abs = float(np.max(np.abs(p_ref - p_special)))
                prob_ok = bool(np.allclose(p_ref, p_special, atol=ATOL, rtol=RTOL))
                if not prob_ok:
                    raise RuntimeError(f"specialized probability parity failed variant={variant} group={group_id} seed={seed}")
                for budget in BUDGETS:
                    kr, sr, or_, _ = allocate_from_features(x_reference, c_full, r_full, variant, seed, budget, "DP")
                    ks, ss, os, _ = allocate_from_features(x_special, c_special, r_special, variant, seed, budget, "DP")
                    k_bad = int(np.sum(kr != ks))
                    id_bad = sum(int(a != b) for a, b in zip(sr, ss))
                    if k_bad or id_bad:
                        raise RuntimeError(f"specialized selection parity failed {variant} group={group_id} seed={seed} budget={budget}")
                    parity_rows.append({
                        "variant": variant, "group_id": group_id, "seed": seed, "budget": budget,
                        "feature_max_abs": max_abs, "feature_within_tolerance": feature_ok,
                        "masked_columns_exact_zero": bool(np.all(x_special[:, indices] == 0.0)),
                        "probability_max_abs": prob_abs, "probability_within_tolerance": prob_ok,
                        "K_mismatch_images": k_bad, "selection_mismatch_images": id_bad,
                        "objective_bitwise_equal": np.float64(or_).tobytes() == np.float64(os).tobytes(), "pass": True,
                    })
    parity = pd.DataFrame(parity_rows)
    if len(parity) != 150 or not parity["pass"].all():
        raise RuntimeError("specialized runtime parity matrix incomplete")
    write_parquet(parity, root / "outputs" / "runtime_implementation_parity.parquet")
    upstream_parity = pd.DataFrame(upstream_rows)
    if len(upstream_parity) != 25 or not upstream_parity["pass"].all():
        raise RuntimeError("runtime upstream-reference parity matrix incomplete")
    write_parquet(upstream_parity, root / "outputs" / "runtime_upstream_parity.parquet")

    policies = ("FULL_DP", "NO_EMBED_DP", "NO_PREFIX_DP", "FULL_NEXT_SLOT_GREEDY", "S_ADAPT_FAST")

    def timed_call(policy: str, group_id: int):
        group = groups[group_id]
        if policy == "FULL_DP":
            k, selected, objective, _ = full_path(group, 530101, PROFILE_BUDGET, "DP")
            return k, selected, objective
        if policy == "NO_EMBED_DP":
            k, selected, objective, _ = no_embed_path(group, 530101, PROFILE_BUDGET)
            return k, selected, objective
        if policy == "NO_PREFIX_DP":
            k, selected, objective, _ = no_prefix_path(group, 530101, PROFILE_BUDGET)
            return k, selected, objective
        if policy == "FULL_NEXT_SLOT_GREEDY":
            k, selected, objective, _ = full_path(group, 530101, PROFILE_BUDGET, "GREEDY")
            return k, selected, objective
        return score_fast(profile_views[group_id], PROFILE_BUDGET)

    samples = []
    rng = random.Random(ORDER_SEED)
    conditions = [(policy, group_id) for policy in policies for group_id in range(5)]
    warm = [x for x in conditions for _ in range(5)]
    measured = [x for x in conditions for _ in range(20)]
    rng.shuffle(warm)
    rng.shuffle(measured)
    sample_order = 0
    for phase, calls in (("WARMUP", warm), ("MEASURED", measured)):
        for policy, group_id in calls:
            sync(device)
            started = time.perf_counter_ns()
            k, selected, objective = timed_call(policy, group_id)
            sync(device)
            elapsed = (time.perf_counter_ns() - started) / 1e6
            if int(np.sum(k)) != 40 * PROFILE_BUDGET or sum(map(len, selected)) != 40 * PROFILE_BUDGET:
                raise RuntimeError("runtime output capacity invariant failed")
            if phase == "MEASURED":
                digest = hashlib.sha256("|".join(x for sub in selected for x in sub).encode("utf-8")).hexdigest()
                samples.append({
                    "sample_order": sample_order, "phase": phase, "timing_scope": "FULL_POLICY_PATH",
                    "policy": policy, "seed": 530101 if policy != "S_ADAPT_FAST" else -1,
                    "budget": PROFILE_BUDGET, "profile_group_id": group_id, "elapsed_ms": elapsed,
                    "amortized_ms_per_image": elapsed / 40, "output_records": int(np.sum(k)),
                    "predicted_objective": objective, "selection_digest": digest,
                })
                sample_order += 1

    # Component-only diagnostics use cached intermediates and are not summed to
    # manufacture the full-path median.
    component_order = 2_000_000
    for group_id, group in enumerate(groups):
        x_full, c_full, r_full = full_group_features(group, pca, scaler, standardize, iou_xyxy)
        p_full = infer(x_full, "FULL", 530101)
        margins = (p_full.mean(axis=1) * weights[c_full - 1]).reshape(40, 45)
        for _ in range(10):
            components = (
                ("COMPONENT_FEATURE_FULL", lambda: full_group_features(group, pca, scaler, standardize, iou_xyxy)),
                ("COMPONENT_FEATURE_NO_EMBED", lambda: no_embed_group_features(group, scaler, standardize, embed_idx, iou_xyxy)),
                ("COMPONENT_FEATURE_NO_PREFIX", lambda: no_prefix_group_features(group, pca, scaler, standardize, prefix_idx)),
                ("COMPONENT_SINGLE_MODEL_TEMPERATURE", lambda: infer(x_full, "FULL", 530101)),
                ("COMPONENT_EXACT_DP", lambda: solve_group_allocations(margins, [PROFILE_BUDGET])),
                ("COMPONENT_NEXT_SLOT_GREEDY", lambda: next_slot_greedy(margins, (PROFILE_BUDGET,))),
                ("COMPONENT_S_ADAPT_FAST", lambda: score_fast(profile_views[group_id], PROFILE_BUDGET)),
            )
            for scope, function in components:
                sync(device)
                started = time.perf_counter_ns()
                _ = function()
                sync(device)
                elapsed = (time.perf_counter_ns() - started) / 1e6
                samples.append({
                    "sample_order": component_order, "phase": "MEASURED", "timing_scope": scope,
                    "policy": "DIAGNOSTIC", "seed": 530101, "budget": PROFILE_BUDGET,
                    "profile_group_id": group_id, "elapsed_ms": elapsed,
                    "amortized_ms_per_image": elapsed / 40, "output_records": 0,
                    "predicted_objective": np.nan, "selection_digest": "",
                })
                component_order += 1
    sample_df = pd.DataFrame(samples)
    if len(sample_df[sample_df.timing_scope == "FULL_POLICY_PATH"]) != 500:
        raise RuntimeError("main runtime sample count mismatch")
    write_parquet(sample_df, root / "outputs" / "runtime_samples.parquet")
    summary = []
    for key, d in sample_df.groupby(["timing_scope", "policy", "seed", "budget"], sort=True):
        stats = qsummary(d["elapsed_ms"].tolist())
        summary.append({
            "timing_scope": key[0], "policy": key[1], "seed": int(key[2]), "budget": int(key[3]),
            "input_groups": int(d["profile_group_id"].nunique()), **stats,
            "amortized_median_ms_per_image": stats["median_ms"] / 40,
            "boundary": "host-resident policy-required raw candidate/native state through K and ordered record IDs" if key[0] == "FULL_POLICY_PATH" else "component-only cached-intermediate diagnostic",
        })
    for scope, seconds, description in (
        ("EXCLUDED_TRAIN_PROFILE_FILE_READ", read_seconds, "one-pass frozen release read for FIT200"),
        ("EXCLUDED_MODEL_PREPROCESSOR_LOAD", load_seconds, "PCA/scaler/model load and GPU placement"),
    ):
        summary.append({
            "timing_scope": scope, "policy": "INPUT", "seed": -1, "budget": -1, "input_groups": 5,
            "samples": 1, "mean_ms": seconds * 1000, "median_ms": seconds * 1000, "p95_ms": seconds * 1000,
            "min_ms": seconds * 1000, "max_ms": seconds * 1000, "std_ms": 0.0, "cv": 0.0,
            "amortized_median_ms_per_image": np.nan, "boundary": description,
        })
    summary_df = pd.DataFrame(summary)
    write_parquet(summary_df, root / "outputs" / "runtime_summary.parquet")
    write_json(root / "outputs" / "runtime_environment.json", {
        "device": str(device), "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "torch": torch.__version__, "cuda": torch.version.cuda, "python": sys.version, "os": platform.platform(),
        "torch_num_threads": torch.get_num_threads(), "torch_num_interop_threads": torch.get_num_interop_threads(),
        "order_seed": ORDER_SEED, "profile_images": 200, "profile_groups": 5, "source_role": "FIT",
        "budget": PROFILE_BUDGET, "warmups_per_condition_group": 5, "measured_per_condition_group": 20,
        "main_boundary": "host-resident policy-required raw state through K and ordered record IDs",
        "no_embed_skips": ["embedding read", "embedding gather", "PCA transform", "embedding norm"],
        "no_prefix_skips": ["prefix XYXY extraction", "same-class search", "prefix IoU", "prefix center distance", "all 12 explicit prefix statistics"],
        "detector_timing_included": False, "elapsed_seconds": time.perf_counter() - t0,
    })
    append_log(root, f"STAGE profile_runtime COMPLETE elapsed_seconds={time.perf_counter()-t0:.6f}")


if __name__ == "__main__":
    main()
