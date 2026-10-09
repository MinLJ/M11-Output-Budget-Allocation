"""Profile one frozen P1 model and the frozen score allocator on FIT identities.

Primary full-path timing starts with candidate/native tensors already resident in
host memory and includes query-index gather, feature construction, FIT-frozen
PCA/scaler, one model forward with its frozen temperature, QUALITY utility,
one exact budget solve, and selected-record materialization. File/model loading,
GT, detector inference, and result serialization are outside this boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import random
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch

sys.dont_write_bytecode = True

SEEDS = (530101, 530102, 530103)
CORE_BUDGETS = (10, 15, 20)
ORDER_SEED = 530003


@dataclass
class RawImage:
    image_id: str
    width: int
    height: int
    candidates: pd.DataFrame
    full_road8_logits: np.ndarray
    full_embeddings: np.ndarray


def hash_key(image_id: str) -> tuple[str, str]:
    return hashlib.sha256(("LC_ALLOC_P1A_PROFILE_V1|" + image_id).encode()).hexdigest(), image_id


def append_log(root: Path, text: str) -> None:
    with (root / "runtime.log").open("a", encoding="utf-8") as f:
        f.write(text.rstrip() + "\n")


def import_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def qsummary(values_ms: list[float]) -> dict:
    a = np.asarray(values_ms, dtype=np.float64)
    return {
        "samples": int(len(a)), "mean_ms": float(a.mean()), "median_ms": float(np.median(a)),
        "p95_ms": float(np.quantile(a, 0.95)), "min_ms": float(a.min()), "max_ms": float(a.max()),
        "std_ms": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
        "cv": float(a.std(ddof=1) / a.mean()) if len(a) > 1 and a.mean() else 0.0,
    }


def load_raw_images(release_root: Path, target_ids: list[str]) -> tuple[dict[str, RawImage], float]:
    start = time.perf_counter()
    identity = json.loads((release_root / "CANDIDATE_ASSET_IDENTITY.json").read_text(encoding="utf-8"))
    asset = identity["assets"]["TRAIN"]
    manifest = pq.read_table(release_root / asset["split_manifest_path"], columns=["image_id", "width", "height"]).to_pandas()
    manifest["image_id"] = manifest["image_id"].astype(str)
    meta = manifest.set_index("image_id")
    targets = set(target_ids)
    result: dict[str, RawImage] = {}
    for cand_rel, native_rel in zip(asset["candidate_shards"], asset["native_state_shards"]):
        cand = pq.read_table(
            release_root / cand_rel,
            columns=["image_id", "candidate_record_id", "road8_rank", "score", "predicted_road8_class_id",
                     "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "bbox_cx", "bbox_cy", "bbox_w", "bbox_h", "query_index"],
            filters=[("road8_rank", "<=", 100)],
        ).to_pandas()
        cand["image_id"] = cand["image_id"].astype(str)
        cand = cand[cand["image_id"].isin(targets)]
        if cand.empty:
            continue
        wanted = set(cand["image_id"].unique())
        with np.load(release_root / native_rel, allow_pickle=False) as state:
            image_ids = state["image_ids"].astype(str)
            query_index = state["query_index"].astype(np.int32)
            starts = {str(image_ids[ix]): ix for ix in range(0, len(image_ids), 300) if str(image_ids[ix]) in wanted}
            logits_all = state["l3_road8_logits"]
            embeddings_all = state["l3_query_embedding"]
            for image_id, rows in cand.groupby("image_id", sort=False):
                image_id = str(image_id)
                rows = rows.copy().reset_index(drop=True)
                rows_sorted = rows.sort_values("road8_rank", kind="stable")
                if len(rows_sorted) != 100 or rows_sorted["road8_rank"].astype(int).tolist() != list(range(1, 101)):
                    raise RuntimeError(f"profile candidate Top100 invariant failed: {image_id}")
                start_ix = starts[image_id]
                block_q = query_index[start_ix:start_ix + 300]
                block_ids = image_ids[start_ix:start_ix + 300]
                if not np.array_equal(block_q, np.arange(300)) or len(set(block_ids)) != 1 or str(block_ids[0]) != image_id:
                    raise RuntimeError(f"native query order invariant failed: {image_id}")
                q = rows_sorted["query_index"].to_numpy(np.int32)
                cls = rows_sorted["predicted_road8_class_id"].to_numpy(np.int16) - 1
                gathered = np.asarray(logits_all[start_ix:start_ix + 300], dtype=np.float32)[q, cls]
                reconstructed = 1.0 / (1.0 + np.exp(-gathered.astype(np.float64)))
                if float(np.max(np.abs(reconstructed - rows_sorted["score"].to_numpy(np.float64)))) > 1e-6:
                    raise RuntimeError(f"profile score/native reconstruction failed: {image_id}")
                m = meta.loc[image_id]
                result[image_id] = RawImage(
                    image_id=image_id, width=int(m["width"]), height=int(m["height"]), candidates=rows,
                    full_road8_logits=np.asarray(logits_all[start_ix:start_ix + 300], dtype=np.float32).copy(),
                    full_embeddings=np.asarray(embeddings_all[start_ix:start_ix + 300], dtype=np.float16).copy(),
                )
    if set(result) != targets:
        raise RuntimeError(f"profile raw input coverage missing={sorted(targets-set(result))[:5]}")
    return result, time.perf_counter() - start


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", required=True)
    ap.add_argument("--p1-root", required=True)
    ap.add_argument("--r0-solver", required=True)
    args = ap.parse_args()
    root, p1 = Path(args.project_root).resolve(), Path(args.p1_root).resolve()
    t0 = time.perf_counter()
    append_log(root, "STAGE profile_single_model START")
    p1_scripts = p1 / "scripts"
    sys.path.insert(0, str(p1_scripts))
    from dp_solver import solve_group_allocations  # type: ignore
    from p1_core import build_raw_features, sigmoid  # type: ignore
    from train_models import MarginalMLP  # type: ignore
    r0_solver = import_module(Path(args.r0_solver).resolve(), "lc_alloc_p1a_r0_solver")

    config = json.loads((root / "run_config.json").read_text(encoding="utf-8"))
    release_root = Path(config["release_root"])
    roles = pq.read_table(p1 / "train_role_split.parquet").to_pandas()
    fit_ids = roles[roles["role"] == "FIT"]["image_id"].astype(str).tolist()
    if len(fit_ids) != 8000:
        raise RuntimeError("frozen FIT identity count changed")
    profile_ids = sorted(fit_ids, key=hash_key)[:200]
    profile_rows = [{"profile_group_id": ix // 40, "position_in_group": ix % 40, "image_id": image_id,
                     "profile_key": hash_key(image_id)[0]} for ix, image_id in enumerate(profile_ids)]
    profile_manifest = pd.DataFrame(profile_rows)
    pq.write_table(pa.Table.from_pandas(profile_manifest, preserve_index=False), root / "qa" / "profile_manifest.parquet", compression="zstd")

    raw, file_read_seconds = load_raw_images(release_root, profile_ids)
    # Hash order fixes group membership; frozen P1 DP tie semantics then use
    # image_id ascending within each group.
    groups = [[raw[x] for x in sorted(profile_ids[g * 40:(g + 1) * 40])] for g in range(5)]

    load_start = time.perf_counter()
    pca = joblib.load(p1 / "models" / "pca32.joblib")
    scaler_bundle = joblib.load(p1 / "models" / "feature_scaler.joblib")
    scaler = scaler_bundle["scaler"]
    standardize = np.asarray(scaler_bundle["standardize_mask"], dtype=bool)
    weights = np.asarray(json.loads((p1 / "models" / "class_weights.json").read_text(encoding="utf-8"))["weights"], np.float64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models: dict[int, MarginalMLP] = {}
    temperatures: dict[int, float] = {}
    for seed in SEEDS:
        snap = torch.load(p1 / "models" / f"marginal_mlp_seed_{seed}.pt", map_location="cpu", weights_only=True)
        model = MarginalMLP(int(snap["input_dim"]))
        model.load_state_dict(snap["state_dict"])
        model.eval().to(device)
        models[seed] = model
        temperatures[seed] = float(json.loads((p1 / "models" / f"temperature_seed_{seed}.json").read_text(encoding="utf-8"))["temperature"])
    sync(device)
    model_load_seconds = time.perf_counter() - load_start

    def features(group: list[RawImage]) -> tuple[np.ndarray, np.ndarray, list[list[str]]]:
        xs, classes, records = [], [], []
        for raw_image in group:
            cand = raw_image.candidates.sort_values("road8_rank", kind="stable").reset_index(drop=True)
            q = cand["query_index"].to_numpy(np.int32)
            if np.any((q < 0) | (q >= 300)):
                raise RuntimeError("profile query index out of range")
            gathered_logits = raw_image.full_road8_logits[q]
            gathered_embeddings = raw_image.full_embeddings[q]
            x = build_raw_features(cand, gathered_logits, gathered_embeddings, pca, raw_image.width, raw_image.height)
            x[:, standardize] = scaler.transform(x[:, standardize])
            xs.append(x.astype(np.float32))
            classes.append(cand.iloc[5:50]["predicted_road8_class_id"].to_numpy(np.int16))
            records.append(cand.iloc[:50]["candidate_record_id"].astype(str).tolist())
        return np.vstack(xs), np.concatenate(classes), records

    @torch.inference_mode()
    def infer(x: np.ndarray, seed: int) -> np.ndarray:
        xt = torch.from_numpy(np.asarray(x, dtype=np.float32)).to(device)
        logits = models[seed](xt).float().cpu().numpy().astype(np.float64)
        sync(device)
        return sigmoid(logits / temperatures[seed])

    def quality_dp(prob: np.ndarray, classes: np.ndarray, records: list[list[str]], budget: int) -> tuple[np.ndarray, list[list[str]], float]:
        margins = (prob.mean(axis=1) * weights[classes - 1]).reshape(40, 45)
        kvals, objective = solve_group_allocations(margins, [budget])
        k = kvals[0]
        selected = [records[i][: int(k[i])] for i in range(40)]
        return k, selected, float(objective[0])

    def learned_full(group: list[RawImage], seed: int, budget: int) -> tuple[np.ndarray, list[list[str]], float]:
        x, classes, records = features(group)
        prob = infer(x, seed)
        return quality_dp(prob, classes, records, budget)

    def score_full(group: list[RawImage], budget: int) -> tuple[np.ndarray, list[list[str]], float]:
        image_ids, ordered_scores, ordered_records = [], [], []
        for raw_image in group:
            records = raw_image.candidates.to_dict("records")
            score = np.asarray([float(x["score"]) for x in records], dtype=np.float64)
            order = r0_solver.rank_candidates(records, score)
            ranked = [records[int(j)] for j in order[:50]]
            image_ids.append(raw_image.image_id)
            ordered_scores.append([float(x["score"]) for x in ranked])
            ordered_records.append([str(x["candidate_record_id"]) for x in ranked])
        k, objective = r0_solver.allocate_prefixes(image_ids, np.asarray(ordered_scores, np.float64), 40 * budget, 5, 50)
        selected = [ordered_records[i][: int(k[i])] for i in range(40)]
        return k, selected, float(objective)

    # Semantic checks for single-budget wrappers and prefix materialization.
    for group_id, group in enumerate(groups):
        for budget in CORE_BUDGETS:
            k, selected, _ = learned_full(group, 530101, budget)
            if int(k.sum()) != 40 * budget or any(len(selected[i]) != int(k[i]) for i in range(40)):
                raise RuntimeError("learned runtime wrapper budget/prefix mismatch")
            sk, ssel, _ = score_full(group, budget)
            if int(sk.sum()) != 40 * budget or any(len(ssel[i]) != int(sk[i]) for i in range(40)):
                raise RuntimeError("score runtime wrapper budget/prefix mismatch")
        x_check, cls_check, records_check = features(group)
        p_check = infer(x_check, 530101)
        margins_check = (p_check.mean(axis=1) * weights[cls_check - 1]).reshape(40, 45)
        all_k, all_obj = solve_group_allocations(margins_check, CORE_BUDGETS)
        for bi, budget in enumerate(CORE_BUDGETS):
            one_k, one_sel, one_obj = quality_dp(p_check, cls_check, records_check, budget)
            if not np.array_equal(one_k, all_k[bi]) or one_obj != float(all_obj[bi]):
                raise RuntimeError("single-budget wrapper differs from frozen multi-budget solver")

    sample_rows: list[dict] = []
    order_rng = random.Random(ORDER_SEED)

    def run_full(policy: str, seed: int, budget: int, group_id: int, phase: str, order_ix: int) -> None:
        sync(device)
        start = time.perf_counter_ns()
        if policy == "LEARN_QUALITY":
            k, selected, objective = learned_full(groups[group_id], seed, budget)
        else:
            k, selected, objective = score_full(groups[group_id], budget)
        sync(device)
        elapsed_ms = (time.perf_counter_ns() - start) / 1e6
        if int(k.sum()) != 40 * budget or sum(map(len, selected)) != 40 * budget:
            raise RuntimeError("profiled full call capacity mismatch")
        if phase == "MEASURED":
            digest = hashlib.sha256("|".join(x for sub in selected for x in sub).encode()).hexdigest()
            sample_rows.append({
                "sample_order": order_ix, "phase": phase, "timing_scope": "FULL_POLICY_PATH",
                "policy": policy, "seed": seed, "budget": budget, "profile_group_id": group_id,
                "elapsed_ms": elapsed_ms, "amortized_ms_per_image": elapsed_ms / 40.0,
                "output_records": int(k.sum()), "predicted_objective": objective, "selection_digest": digest,
            })

    # Warm-up every measured condition, excluded from raw samples.
    conditions = []
    for group_id in range(5):
        for budget in CORE_BUDGETS:
            conditions.append(("LEARN_QUALITY", 530101, budget, group_id, 20))
            conditions.append(("S_ADAPT", -1, budget, group_id, 20))
        for seed in (530102, 530103):
            conditions.append(("LEARN_QUALITY", seed, 15, group_id, 10))
    warm = [(p, s, b, g) for p, s, b, g, _ in conditions for _ in range(5)]
    order_rng.shuffle(warm)
    for ix, (policy, seed, budget, group_id) in enumerate(warm):
        run_full(policy, seed, budget, group_id, "WARMUP", ix)
    measured = [(p, s, b, g) for p, s, b, g, repeats in conditions for _ in range(repeats)]
    order_rng.shuffle(measured)
    for ix, (policy, seed, budget, group_id) in enumerate(measured):
        run_full(policy, seed, budget, group_id, "MEASURED", ix)

    # Component measurements use cached intermediates by design and are not
    # summed to fabricate a full-path latency.
    component_order = 1_000_000
    for group_id, group in enumerate(groups):
        cached_x, cached_cls, cached_records = features(group)
        cached_prob = infer(cached_x, 530101)
        for rep in range(10):
            sync(device); start = time.perf_counter_ns(); _ = features(group); sync(device)
            elapsed = (time.perf_counter_ns() - start) / 1e6
            sample_rows.append({"sample_order": component_order, "phase": "MEASURED", "timing_scope": "COMPONENT_FEATURE_PCA_SCALER",
                                "policy": "LEARN_QUALITY", "seed": 530101, "budget": 15, "profile_group_id": group_id,
                                "elapsed_ms": elapsed, "amortized_ms_per_image": elapsed / 40, "output_records": 0,
                                "predicted_objective": np.nan, "selection_digest": ""})
            component_order += 1
            sync(device); start = time.perf_counter_ns(); _ = infer(cached_x, 530101); sync(device)
            elapsed = (time.perf_counter_ns() - start) / 1e6
            sample_rows.append({"sample_order": component_order, "phase": "MEASURED", "timing_scope": "COMPONENT_SINGLE_MODEL_INFERENCE_TEMPERATURE",
                                "policy": "LEARN_QUALITY", "seed": 530101, "budget": 15, "profile_group_id": group_id,
                                "elapsed_ms": elapsed, "amortized_ms_per_image": elapsed / 40, "output_records": 0,
                                "predicted_objective": np.nan, "selection_digest": ""})
            component_order += 1
            start = time.perf_counter_ns(); k, selected, objective = quality_dp(cached_prob, cached_cls, cached_records, 15)
            elapsed = (time.perf_counter_ns() - start) / 1e6
            sample_rows.append({"sample_order": component_order, "phase": "MEASURED", "timing_scope": "COMPONENT_QUALITY_UTILITY_SINGLE_BUDGET_DP",
                                "policy": "LEARN_QUALITY", "seed": 530101, "budget": 15, "profile_group_id": group_id,
                                "elapsed_ms": elapsed, "amortized_ms_per_image": elapsed / 40, "output_records": int(k.sum()),
                                "predicted_objective": objective, "selection_digest": hashlib.sha256("|".join(x for sub in selected for x in sub).encode()).hexdigest()})
            component_order += 1

    samples = pd.DataFrame(sample_rows)
    pq.write_table(pa.Table.from_pandas(samples, preserve_index=False), root / "outputs" / "runtime_samples.parquet", compression="zstd")
    summary_rows: list[dict] = []
    for key, d in samples.groupby(["timing_scope", "policy", "seed", "budget"], sort=True):
        stats = qsummary(d["elapsed_ms"].tolist())
        summary_rows.append({"timing_scope": key[0], "policy": key[1], "seed": int(key[2]), "budget": int(key[3]),
                             "input_groups": int(d["profile_group_id"].nunique()), **stats,
                             "amortized_median_ms_per_image": stats["median_ms"] / 40.0,
                             "boundary": "candidate/native tensors resident in host memory through K and selected record IDs" if key[0] == "FULL_POLICY_PATH" else "component-only cached-intermediate diagnostic"})
    summary_rows.extend([
        {"timing_scope": "EXCLUDED_FILE_READ", "policy": "INPUT", "seed": -1, "budget": -1, "input_groups": 5,
         "samples": 1, "mean_ms": file_read_seconds * 1000, "median_ms": file_read_seconds * 1000, "p95_ms": file_read_seconds * 1000,
         "min_ms": file_read_seconds * 1000, "max_ms": file_read_seconds * 1000, "std_ms": 0.0, "cv": 0.0,
         "amortized_median_ms_per_image": file_read_seconds * 1000 / 200, "boundary": "excluded one-pass release file read into raw host-memory state"},
        {"timing_scope": "EXCLUDED_MODEL_PREPROCESSOR_LOAD", "policy": "INPUT", "seed": -1, "budget": -1, "input_groups": 5,
         "samples": 1, "mean_ms": model_load_seconds * 1000, "median_ms": model_load_seconds * 1000, "p95_ms": model_load_seconds * 1000,
         "min_ms": model_load_seconds * 1000, "max_ms": model_load_seconds * 1000, "std_ms": 0.0, "cv": 0.0,
         "amortized_median_ms_per_image": model_load_seconds * 1000 / 200, "boundary": "excluded PCA/scaler/model load and GPU placement"},
    ])
    summary_df = pd.DataFrame(summary_rows)
    pq.write_table(pa.Table.from_pandas(summary_df, preserve_index=False), root / "outputs" / "runtime_summary.parquet", compression="zstd")
    environment = {
        "status": "PASS", "profile_images": 200, "profile_groups": 5, "source_role": "FIT",
        "selection_rule": "SHA256('LC_ALLOC_P1A_PROFILE_V1|'+image_id), then image_id; first 200",
        "order_seed": ORDER_SEED, "device": str(device), "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "torch": torch.__version__, "cuda": torch.version.cuda, "python": sys.version, "os": platform.platform(),
        "torch_num_threads": torch.get_num_threads(), "torch_num_interop_threads": torch.get_num_interop_threads(),
        "file_read_seconds_excluded": file_read_seconds, "model_preprocessor_load_seconds_excluded": model_load_seconds,
        "elapsed_seconds": time.perf_counter() - t0,
    }
    (root / "outputs" / "runtime_environment.json").write_text(json.dumps(environment, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    append_log(root, f"STAGE profile_single_model COMPLETE elapsed_seconds={time.perf_counter()-t0:.6f}")


if __name__ == "__main__":
    main()
