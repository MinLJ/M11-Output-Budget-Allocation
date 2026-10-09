"""LC-ALLOC-EFF-AUDIT full detector + frozen M11 latency audit.

The scientific method is unchanged. This script only measures a fixed frozen
pipeline on TRAIN/FIT images and never opens GT, TEST, RESERVE, holdout, or
Road1000 assets. The timed boundary starts from host-resident, preprocessed
1x3x640x640 float32 tensors. Disk I/O, image decode/resize, model loading, and
hook registration are excluded.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
import winreg
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True

import joblib
import numpy as np
import pandas as pd
import psutil
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from PIL import Image


PROJECT = Path(r"D:\AOP_DETR\research_LC_ALLOC_EFF_AUDIT")
P1 = Path(r"D:\AOP_DETR\research_LC_ALLOC_P1")
P1A = Path(r"D:\AOP_DETR\research_LC_ALLOC_P1A")
P1B = Path(r"D:\AOP_DETR\research_LC_ALLOC_P1B")
EXPORT = Path(r"D:\AOP_DETR\shared_benchmark\AOP_ROAD8_LARGECLEAN_V1\P1_SHARED_EXPORT")
RELEASE = EXPORT / "release" / "AOP_Road8_LC_v1_RTDERTv2_R18VD_RELEASE_v1"
EXPORT_CONFIG = EXPORT / "configs" / "RTDETRV2_R18VD_EXPORT_CONFIG.json"

SEED = 530101  # inherited P1A/P1B primary timing seed; not DEV-selected
ORDER_SEED = 530004
IMAGE_COUNTS = (20, 40, 80)
BUDGETS = (10, 20, 40)
ROAD8_INDICES = (0, 1, 2, 3, 5, 6, 7, 9)
ATOL = 1e-6
RTOL = 1e-5

EXPECTED = {
    P1A / "candidate_freeze.json": "8889fe172a90a98572fdd9e1fddced2fbf9a4bf0ded65841f5a19b1c625180d7",
    P1B / "run_config.json": "5b103264c1b48a4ae28e060dac35a0d5c7f33e89c9726bf0205ee2ba2282ca3f",
    P1B / "LC_ALLOC_P1B_REPORT.md": "b3c0551b35c6bcd9786ea9e74d4e5186d2af1ca393730368aaf12cd012481878",
    P1A / "qa" / "profile_manifest.parquet": "3841aaf4de5e2de031b3fa2154c5a353b6bb1f9f17527f25a47b2f07562b0b47",
    P1 / "models" / "marginal_mlp_seed_530101.pt": "cb9732dbc828eff968b96294ab541b925eacc0d57472faf5a11f4bb246597aa8",
    P1 / "models" / "marginal_mlp_seed_530102.pt": "7b60f0f4d9ef5fc63941ed61398086ee7c82abf3c928a0db7b8c905f27c20996",
    P1 / "models" / "marginal_mlp_seed_530103.pt": "890cf82a9cf654d6b44816b2111d51e86ee84ba1f35050b299c3473a17d0b6b8",
    P1 / "models" / "pca32.joblib": "12d8e656d53bdad54e498125993f437f77d874e97be38ffe29a0c969e1ff41af",
    P1 / "models" / "feature_scaler.joblib": "9d3da8c36b3b5a2cf8c8ab2deb963e3595c313b4d13f6d3e7fb8c27d923ced6d",
    P1 / "models" / "class_weights.json": "2ee6f8d1a7daddd298c7874429d954d21885aefcab9a7a67bb1b49eba7f9c237",
    P1 / "feature_schema.json": "8613d979ab6393c5d0524f1451e80329f60d0caa7a514ae256be4022d70096f7",
    P1 / "scripts" / "dp_solver.py": "f230626580575e256576e1be92ca1ad409a57c9d725202f2b364f78dbca0c317",
    P1A / "qa" / "profile_manifest.parquet": "3841aaf4de5e2de031b3fa2154c5a353b6bb1f9f17527f25a47b2f07562b0b47",
}


@dataclass
class TimedImage:
    image_id: str
    width: int
    height: int
    image_sha256: str
    host_tensor: torch.Tensor
    expected_candidates: pd.DataFrame
    expected_road8_logits: np.ndarray
    expected_embedding_f16: np.ndarray


@dataclass
class RawImage:
    image_id: str
    width: int
    height: int
    candidates: pd.DataFrame
    full_logits: np.ndarray
    full_embeddings: np.ndarray


def sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk_size), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_log(message: str) -> None:
    with (PROJECT / "runtime.log").open("a", encoding="utf-8") as f:
        f.write(message.rstrip() + "\n")


def import_file(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def verify_frozen_inputs() -> dict[str, Any]:
    checks = []
    for path, expected in EXPECTED.items():
        actual = sha256_file(path)
        checks.append({"path": str(path), "expected_sha256": expected, "actual_sha256": actual, "pass": actual == expected})
    freeze = json.loads((P1A / "candidate_freeze.json").read_text(encoding="utf-8"))
    p1b = json.loads((P1B / "run_config.json").read_text(encoding="utf-8"))
    for model in freeze["models"]:
        checks.append({
            "path": model["model_path"],
            "expected_sha256": model["model_sha256"],
            "actual_sha256": sha256_file(Path(model["model_path"])),
            "pass": sha256_file(Path(model["model_path"])) == model["model_sha256"],
        })
        checks.append({
            "path": model["temperature_path"],
            "expected_sha256": model["temperature_sha256"],
            "actual_sha256": sha256_file(Path(model["temperature_path"])),
            "pass": sha256_file(Path(model["temperature_path"])) == model["temperature_sha256"],
        })
    requirements = {
        "candidate": freeze["candidate"] == "LEARN_QUALITY",
        "candidate_identity_not_test_authorization": freeze["confirmation_status"] == "DRAFT_ONLY_NOT_AUTHORIZED",
        "p1b_complete": p1b["execution_status"] == "COMPLETE",
        "p1b_optimization_accepted": p1b["optimization_status"] == "ACCEPTED",
        "p1b_confirmation_not_authorized": p1b["confirmation_status"] == "NOT_AUTHORIZED",
        "three_seeds_bound": [x["seed"] for x in freeze["models"]] == [530101, 530102, 530103],
        "test_in_release_false": json.loads((RELEASE / "RELEASE_MANIFEST.json").read_text(encoding="utf-8"))["TEST_INCLUDED"] is False,
    }
    if not all(x["pass"] for x in checks) or not all(requirements.values()):
        raise RuntimeError("FROZEN_INPUT_BINDING_FAIL")
    return {"status": "PASS", "candidate_freeze_inherited_path": str(P1A / "candidate_freeze.json"), "checks": checks, "requirements": requirements}


def query_nvidia_smi() -> dict[str, Any]:
    fields = "name,memory.total,driver_version,temperature.gpu,pstate,clocks.current.graphics,clocks.current.memory,power.draw,utilization.gpu,memory.used"
    try:
        output = subprocess.check_output(
            ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.STDOUT, timeout=10,
        ).strip().splitlines()[0]
        values = [x.strip() for x in output.split(",")]
        return dict(zip(fields.split(","), values))
    except Exception as exc:
        return {"unavailable": repr(exc)}


def cpu_name() -> str:
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
            return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
    except Exception:
        return platform.processor() or "unavailable"


def candidate_record_id(asset_id: str, image_id: str, rank: int) -> str:
    fields = (asset_id.encode("utf-8"), image_id.encode("utf-8"), str(rank).encode("ascii"))
    payload = b"".join(len(v).to_bytes(8, "big") + v for v in fields)
    return hashlib.sha256(payload).hexdigest()


def build_candidates(
    road8_scores: np.ndarray,
    boxes: np.ndarray,
    meta: TimedImage,
    asset_id: str,
    detector_id: str,
    checkpoint_sha: str,
    export_config_sha: str,
) -> pd.DataFrame:
    scores = np.asarray(road8_scores, dtype=np.float32).reshape(-1)
    q = np.repeat(np.arange(300, dtype=np.int32), 8)
    cls = np.tile(np.arange(1, 9, dtype=np.int32), 300)
    original = np.arange(2400, dtype=np.int32)
    order = np.lexsort((original, cls, q, -scores))[:300]
    boxes = np.asarray(boxes, dtype=np.float32)
    cx, cy, w, h = [boxes[:, j] for j in range(4)]
    xy = np.stack((cx - w / np.float32(2), cy - h / np.float32(2), cx + w / np.float32(2), cy + h / np.float32(2)), axis=1)
    xy *= np.asarray([meta.width, meta.height, meta.width, meta.height], dtype=np.float32)
    rows = []
    for rank, original_index in enumerate(order, 1):
        qi = int(q[original_index]); cid = int(cls[original_index])
        x1, y1, x2, y2 = map(float, xy[qi]); score = float(scores[original_index])
        rows.append({
            "image_id": meta.image_id,
            "image_sha256": meta.image_sha256,
            "detector_id": detector_id,
            "checkpoint_sha256": checkpoint_sha,
            "export_config_sha256": export_config_sha,
            "candidate_asset_id": asset_id,
            "candidate_record_id": candidate_record_id(asset_id, meta.image_id, rank),
            "road8_rank": rank,
            "predicted_road8_class_id": cid,
            "score": score,
            "bbox_x1": x1,
            "bbox_y1": y1,
            "bbox_x2": x2,
            "bbox_y2": y2,
            "bbox_cx": (x1 + x2) / 2,
            "bbox_cy": (y1 + y2) / 2,
            "bbox_w": x2 - x1,
            "bbox_h": y2 - y1,
            "query_index": qi,
            "source_order": int(original_index),
        })
    frame = pd.DataFrame(rows)
    if frame["road8_rank"].tolist() != list(range(1, 301)) or frame["candidate_record_id"].duplicated().any():
        raise RuntimeError("CANDIDATE_OUTPUT_INVARIANT_FAIL")
    return frame


def preprocess_host(snapshot: Any, image_path: Path) -> torch.Tensor:
    with Image.open(image_path) as image:
        rgb = image.convert("RGB") if image.mode != "RGB" else image.copy()
    resized = snapshot.TF.resize(rgb, list(snapshot.input_size), interpolation=snapshot.interpolation, antialias=True)
    tensor = snapshot.TF.pil_to_tensor(resized).to(dtype=torch.float32).div(255.0)
    return tensor.contiguous().unsqueeze(0).cpu()


def load_expected_release(identity: dict, target_ids: list[str]) -> tuple[dict[str, pd.DataFrame], dict[str, tuple[np.ndarray, np.ndarray]]]:
    asset = identity["assets"]["TRAIN"]
    targets = set(target_ids)
    asset_index = pq.read_table(RELEASE / asset["asset_index_path"]).to_pandas()
    asset_index["image_id"] = asset_index["image_id"].astype(str)
    target_index = asset_index[asset_index["image_id"].isin(targets)]
    expected_candidates: dict[str, pd.DataFrame] = {}
    expected_native: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for shard_name, rows in target_index.groupby("candidate_shard"):
        ids = rows["image_id"].astype(str).tolist()
        cand_rel = next(x for x in asset["candidate_shards"] if Path(x).name == shard_name)
        table = pq.read_table(RELEASE / cand_rel, filters=[("image_id", "in", ids)]).to_pandas()
        table["image_id"] = table["image_id"].astype(str)
        for image_id, group in table.groupby("image_id", sort=False):
            expected_candidates[str(image_id)] = group.sort_values("road8_rank", kind="stable").reset_index(drop=True)
    for native_name, rows in target_index.groupby("native_state_shard"):
        ids = set(rows["image_id"].astype(str))
        native_rel = asset["native_shard_map"][native_name]
        with np.load(RELEASE / native_rel, allow_pickle=False) as z:
            image_ids = z["image_ids"].astype(str)
            for start in range(0, len(image_ids), 300):
                image_id = str(image_ids[start])
                if image_id in ids:
                    expected_native[image_id] = (
                        np.asarray(z["l3_road8_logits"][start:start + 300], dtype=np.float32).copy(),
                        np.asarray(z["l3_query_embedding"][start:start + 300], dtype=np.float16).copy(),
                    )
    if set(expected_candidates) != targets or set(expected_native) != targets:
        raise RuntimeError("RELEASE_EXPECTED_INPUT_COVERAGE_FAIL")
    return expected_candidates, expected_native


def summary(values: list[float]) -> dict[str, float | int]:
    a = np.asarray(values, dtype=np.float64)
    return {
        "samples": int(len(a)),
        "mean_ms": float(a.mean()),
        "median_ms": float(np.median(a)),
        "p95_ms": float(np.quantile(a, 0.95)),
        "std_ms": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
        "min_ms": float(a.min()),
        "max_ms": float(a.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--measurements", type=int, default=50)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    PROJECT.mkdir(parents=True, exist_ok=True)
    (PROJECT / "outputs").mkdir(parents=True, exist_ok=True)
    (PROJECT / "qa").mkdir(parents=True, exist_ok=True)
    start_all = time.perf_counter()
    append_log(f"STAGE full_pipeline START warmups={args.warmups} measurements={args.measurements}")

    binding = verify_frozen_inputs()
    write_json(PROJECT / "qa" / "frozen_binding.json", binding)
    config = json.loads(EXPORT_CONFIG.read_text(encoding="utf-8"))
    identity = json.loads((RELEASE / "CANDIDATE_ASSET_IDENTITY.json").read_text(encoding="utf-8"))
    train_asset = identity["assets"]["TRAIN"]
    export_config_sha = sha256_file(EXPORT_CONFIG)
    release_manifest = json.loads((RELEASE / "RELEASE_MANIFEST.json").read_text(encoding="utf-8"))
    if export_config_sha != release_manifest["export_config_sha256"]:
        raise RuntimeError("EXPORT_CONFIG_HASH_FAIL")

    sys.path.insert(0, str(EXPORT / "scripts"))
    from detector_snapshot import DetectorSnapshot  # type: ignore
    sys.path.insert(0, str(P1 / "scripts"))
    from dp_solver import choice_values_from_optional_marginals, solve_exact_multiple_choice, solve_group_allocations  # type: ignore
    from p1_core import iou_xyxy, sigmoid  # type: ignore
    from train_models import MarginalMLP  # type: ignore
    p1b_profile = import_file(P1B / "scripts" / "profile_optimizations.py", "lc_eff_p1b_profile")

    snapshot = DetectorSnapshot(config)
    device = snapshot.device
    torch.set_grad_enabled(False)
    profile = pq.read_table(P1A / "qa" / "profile_manifest.parquet").to_pandas().sort_values(["profile_group_id", "position_in_group"], kind="stable")
    chosen_ids = profile.head(80)["image_id"].astype(str).tolist()
    train_manifest = pq.read_table(RELEASE / train_asset["split_manifest_path"], columns=["image_id", "image_path", "width", "height"]).to_pandas()
    hashes = pq.read_table(RELEASE / train_asset["image_hash_manifest_path"], columns=["image_id", "sha256"]).to_pandas()
    train_manifest["image_id"] = train_manifest["image_id"].astype(str); hashes["image_id"] = hashes["image_id"].astype(str)
    meta = train_manifest.merge(hashes, on="image_id", validate="one_to_one").set_index("image_id")
    expected_candidates, expected_native = load_expected_release(identity, chosen_ids)

    preprocess_start = time.perf_counter()
    timed_images: list[TimedImage] = []
    for image_id in chosen_ids:
        row = meta.loc[image_id]
        image_path = Path(str(row.image_path))
        if not image_path.is_file():
            raise RuntimeError(f"TRAIN_IMAGE_MISSING {image_id}")
        host_tensor = preprocess_host(snapshot, image_path)
        logits_expected, embedding_expected = expected_native[image_id]
        timed_images.append(TimedImage(
            image_id=image_id,
            width=int(row.width),
            height=int(row.height),
            image_sha256=str(row.sha256),
            host_tensor=host_tensor,
            expected_candidates=expected_candidates[image_id],
            expected_road8_logits=logits_expected,
            expected_embedding_f16=embedding_expected,
        ))
    preprocess_seconds = time.perf_counter() - preprocess_start

    pca = joblib.load(P1 / "models" / "pca32.joblib")
    scaler_bundle = joblib.load(P1 / "models" / "feature_scaler.joblib")
    scaler = scaler_bundle["scaler"]
    standardize = np.asarray(scaler_bundle["standardize_mask"], dtype=bool)
    weights = np.asarray(json.loads((P1 / "models" / "class_weights.json").read_text(encoding="utf-8"))["weights"], dtype=np.float64)
    model_snapshot = torch.load(P1 / "models" / f"marginal_mlp_seed_{SEED}.pt", map_location="cpu", weights_only=True)
    model = MarginalMLP(int(model_snapshot["input_dim"])); model.load_state_dict(model_snapshot["state_dict"])
    model.eval().requires_grad_(False).to(device)
    temperature = float(json.loads((P1 / "models" / f"temperature_seed_{SEED}.json").read_text(encoding="utf-8"))["temperature"])

    capture: dict[str, torch.Tensor] = {}
    def layer_hook(module: Any, inputs: tuple[Any, ...], output: torch.Tensor) -> None:
        capture["embedding"] = output.detach()
    hook = snapshot.decoder.layers[2].register_forward_hook(layer_hook)

    @torch.inference_mode()
    def forward_one(meta_image: TimedImage) -> tuple[pd.DataFrame, torch.Tensor, torch.Tensor]:
        capture.clear()
        x = meta_image.host_tensor.to(device)
        with torch.autocast(device_type="cuda", enabled=False):
            output = snapshot.model(x)
        if "embedding" not in capture or tuple(capture["embedding"].shape) != (1, 300, 256):
            raise RuntimeError("L3_EMBEDDING_CAPTURE_FAIL")
        logits = output["pred_logits"][0]
        boxes = output["pred_boxes"][0]
        road8_logits = logits[:, ROAD8_INDICES]
        scores_cpu = road8_logits.sigmoid().detach().cpu().contiguous().numpy().copy()
        boxes_cpu = boxes.detach().cpu().contiguous().numpy().copy()
        candidates = build_candidates(
            scores_cpu, boxes_cpu, meta_image, train_asset["candidate_asset_id"], config["detector_id"],
            config["checkpoint_sha256"], export_config_sha,
        )
        return candidates, road8_logits.detach(), capture["embedding"][0].detach()

    @torch.inference_mode()
    def model_probabilities(x: np.ndarray) -> np.ndarray:
        xt = torch.from_numpy(np.asarray(x, dtype=np.float32)).to(device)
        out = model(xt).float().cpu().numpy().astype(np.float64)
        torch.cuda.synchronize(device)
        return sigmoid(out / temperature)

    def solve_policy(raw_images: list[RawImage], budget: int) -> tuple[np.ndarray, list[list[str]], float]:
        all_k: list[np.ndarray] = []
        all_selected: list[list[str]] = []
        objective = np.float64(0.0)
        chunks = [raw_images] if len(raw_images) <= 40 else [raw_images[:40], raw_images[40:80]]
        for chunk in chunks:
            x, classes, records = p1b_profile.optimized_group_features(chunk, pca, scaler, standardize, iou_xyxy)
            probabilities = model_probabilities(x)
            margins = (np.ascontiguousarray(probabilities).mean(axis=1) * weights[classes - 1]).reshape(len(chunk), 45)
            if len(chunk) == 40:
                kval, obj = solve_group_allocations(margins, [budget])
            else:
                choices = choice_values_from_optional_marginals(margins)
                kval, obj = solve_exact_multiple_choice(choices, [len(chunk) * budget], k_min=5)
            k = kval[0]
            all_k.append(k)
            all_selected.extend([records[i][:int(k[i])] for i in range(len(chunk))])
            objective = np.float64(objective + obj[0])
        combined_k = np.concatenate(all_k)
        if int(combined_k.sum()) != len(raw_images) * budget or sum(len(x) for x in all_selected) != len(raw_images) * budget:
            raise RuntimeError("EXACT_OUTPUT_BUDGET_FAIL")
        return combined_k, all_selected, float(objective)

    # Preflight current forward against the immutable release for three fixed TRAIN/FIT images.
    preflight_rows = []
    for meta_image in timed_images[:3]:
        candidates, road8_logits_gpu, embedding_gpu = forward_one(meta_image)
        torch.cuda.synchronize(device)
        road8_logits = road8_logits_gpu.cpu().contiguous().numpy().copy()
        embedding_f16 = embedding_gpu.cpu().contiguous().numpy().astype(np.float16)
        expected = meta_image.expected_candidates.sort_values("road8_rank", kind="stable").reset_index(drop=True)
        exact_cols = ["road8_rank", "query_index", "predicted_road8_class_id", "candidate_record_id"]
        exact_ok = all(candidates[c].astype(str).tolist() == expected[c].astype(str).tolist() for c in exact_cols)
        score_gap = float(np.max(np.abs(candidates["score"].to_numpy(np.float64) - expected["score"].to_numpy(np.float64))))
        bbox_cols = ["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]
        bbox_gap = float(np.max(np.abs(candidates[bbox_cols].to_numpy(np.float64) - expected[bbox_cols].to_numpy(np.float64))))
        logit_gap = float(np.max(np.abs(road8_logits.astype(np.float64) - meta_image.expected_road8_logits.astype(np.float64))))
        embedding_exact = bool(np.array_equal(embedding_f16, meta_image.expected_embedding_f16))
        passed = exact_ok and score_gap <= ATOL and bbox_gap <= ATOL and logit_gap <= ATOL and embedding_exact
        preflight_rows.append({
            "image_id": meta_image.image_id, "candidate_identity_exact": exact_ok, "score_max_abs": score_gap,
            "bbox_max_abs": bbox_gap, "road8_logit_max_abs": logit_gap, "embedding_float16_exact": embedding_exact,
            "pass": passed,
        })
    if not all(x["pass"] for x in preflight_rows):
        raise RuntimeError("DETECTOR_RELEASE_PARITY_FAIL")
    write_json(PROJECT / "qa" / "runtime_preflight.json", {
        "status": "PASS", "images": preflight_rows, "atol": ATOL, "rtol": RTOL,
        "boundary": "host-resident preprocessed tensor; H2D included; disk/decode/resize excluded",
        "candidate_freeze_path_note": "candidate_freeze.json is inherited from P1A; P1B retains it without a duplicate copy",
    })
    append_log("STAGE preflight PASS images=3")
    if args.preflight_only:
        hook.remove(); append_log("STAGE full_pipeline PRECHECK_ONLY COMPLETE"); return

    image_sets = {n: timed_images[:n] for n in IMAGE_COUNTS}
    def timed_pipeline(n_images: int, budget: int) -> dict[str, Any]:
        metas = image_sets[n_images]
        torch.cuda.synchronize(device)
        t0 = time.perf_counter_ns()
        detector_outputs = []
        for meta_image in metas:
            candidates, road8_logits_gpu, embedding_gpu = forward_one(meta_image)
            detector_outputs.append((meta_image, candidates, road8_logits_gpu, embedding_gpu))
        torch.cuda.synchronize(device)
        t_a = time.perf_counter_ns()
        raw_images = []
        for meta_image, candidates, road8_logits_gpu, embedding_gpu in detector_outputs:
            road8_logits = road8_logits_gpu.cpu().contiguous().numpy().copy()
            embedding_f16 = embedding_gpu.cpu().contiguous().numpy().astype(np.float16, copy=True)
            raw_images.append(RawImage(
                image_id=meta_image.image_id, width=meta_image.width, height=meta_image.height,
                candidates=candidates.iloc[:100].copy(), full_logits=road8_logits,
                full_embeddings=embedding_f16,
            ))
        k, selected, objective = solve_policy(raw_images, budget)
        torch.cuda.synchronize(device)
        t_b = time.perf_counter_ns()
        selection_digest = hashlib.sha256("|".join(x for rows in selected for x in rows).encode("utf-8")).hexdigest()
        detector_ms = (t_a - t0) / 1e6
        allocator_ms = (t_b - t_a) / 1e6
        total_ms = (t_b - t0) / 1e6
        return {
            "detector_ms": detector_ms, "allocator_ms": allocator_ms, "total_ms": total_ms,
            "output_records": int(k.sum()), "predicted_objective": objective,
            "selection_digest": selection_digest,
        }

    conditions = [(n, k) for n in IMAGE_COUNTS for k in BUDGETS]
    rng = random.Random(ORDER_SEED)
    warm = [x for x in conditions for _ in range(args.warmups)]
    measured = [x for x in conditions for _ in range(args.measurements)]
    rng.shuffle(warm); rng.shuffle(measured)
    samples = []
    pre_gpu = query_nvidia_smi()
    for phase, calls in (("WARMUP", warm), ("MEASURED", measured)):
        counters = {x: 0 for x in conditions}
        for order, (n_images, budget) in enumerate(calls):
            result = timed_pipeline(n_images, budget)
            repeat = counters[(n_images, budget)]; counters[(n_images, budget)] += 1
            if phase == "MEASURED":
                samples.append({
                    "sample_order": order, "phase": phase, "repeat": repeat, "images": n_images,
                    "budget": budget, "seed": SEED, "detector_ms": result["detector_ms"],
                    "allocator_ms": result["allocator_ms"], "total_ms": result["total_ms"],
                    "allocator_over_detector_pct": 100.0 * result["allocator_ms"] / result["detector_ms"],
                    "allocator_over_total_pct": 100.0 * result["allocator_ms"] / result["total_ms"],
                    "detector_ms_per_image": result["detector_ms"] / n_images,
                    "allocator_ms_per_image": result["allocator_ms"] / n_images,
                    "total_ms_per_image": result["total_ms"] / n_images,
                    "output_records": result["output_records"], "predicted_objective": result["predicted_objective"],
                    "selection_digest": result["selection_digest"],
                })
        append_log(f"STAGE {phase.lower()} COMPLETE calls={len(calls)}")
    post_gpu = query_nvidia_smi()
    hook.remove()

    sample_df = pd.DataFrame(samples)
    pq.write_table(pa.Table.from_pandas(sample_df, preserve_index=False), PROJECT / "outputs" / "runtime_pipeline_samples.parquet", compression="zstd")
    summary_rows = []
    for (n_images, budget), group in sample_df.groupby(["images", "budget"], sort=True):
        d = summary(group["detector_ms"].tolist()); a = summary(group["allocator_ms"].tolist()); t = summary(group["total_ms"].tolist())
        summary_rows.append({
            "images": int(n_images), "K": int(budget), "timing_seed": SEED, "warmups": args.warmups,
            "measurements": args.measurements,
            "detector_ms_mean": d["mean_ms"], "detector_ms_median": d["median_ms"], "detector_ms_p95": d["p95_ms"], "detector_ms_std": d["std_ms"],
            "allocator_ms_mean": a["mean_ms"], "allocator_ms_median": a["median_ms"], "allocator_ms_p95": a["p95_ms"], "allocator_ms_std": a["std_ms"],
            "total_ms_mean": t["mean_ms"], "total_ms_median": t["median_ms"], "total_ms_p95": t["p95_ms"], "total_ms_std": t["std_ms"],
            "allocator_over_detector_pct_median": 100.0 * a["median_ms"] / d["median_ms"],
            "allocator_over_total_pct_median": 100.0 * a["median_ms"] / t["median_ms"],
            "detector_ms_per_image_median": d["median_ms"] / int(n_images),
            "allocator_ms_per_image_median": a["median_ms"] / int(n_images),
            "total_ms_per_image_median": t["median_ms"] / int(n_images),
            "boundary_A": "host tensor -> H2D -> RT-DETRv2 forward -> canonical Top300 candidate records",
            "boundary_B": "Boundary A + native-state host materialization -> 90D/PCA/scaler -> M11/temp/QUALITY -> exact DP -> IDs",
            "allocation_grouping": "20-image scaling diagnostic" if int(n_images) == 20 else ("one frozen 40-image group" if int(n_images) == 40 else "two independent frozen 40-image groups"),
        })
    summary_df = pd.DataFrame(summary_rows)
    pq.write_table(pa.Table.from_pandas(summary_df, preserve_index=False), PROJECT / "outputs" / "runtime_pipeline_summary.parquet", compression="zstd")

    run_config = {
        "task": "LC-ALLOC-EFF-AUDIT", "candidate": "LEARN_QUALITY_M11", "timing_seed": SEED,
        "timing_seed_reason": "inherited P1A/P1B primary timing seed by fixed seed order; not selected by DEV",
        "all_scientific_seeds_bound": [530101, 530102, 530103], "image_counts": list(IMAGE_COUNTS),
        "budgets": list(BUDGETS), "warmups": args.warmups, "measurements": args.measurements,
        "order_seed": ORDER_SEED, "input_role": "FIT only; no GT read", "batch_size": 1,
        "input_resolution": [640, 640], "precision": "FP32", "model_eval": True, "no_grad": True,
        "autocast": False, "tf32": False, "candidate_pool": "canonical Road8 Top300 output; allocator consumes original-rank Top100",
        "action": "fixed original-score prefix length K only", "k_bounds": [5, 50], "group_size": 40,
        "twenty_image_note": "latency scaling diagnostic only; no scientific group or result changed",
        "eighty_image_note": "two independent 40-image frozen allocation groups",
        "test_access": 0, "candidate_freeze": {"path": str(P1A / "candidate_freeze.json"), "sha256": EXPECTED[P1A / "candidate_freeze.json"]},
    }
    write_json(PROJECT / "run_config.json", run_config)
    environment = {
        "gpu": torch.cuda.get_device_name(device), "gpu_memory_bytes": int(torch.cuda.get_device_properties(device).total_memory),
        "driver_and_gpu_pre": pre_gpu, "driver_and_gpu_post": post_gpu,
        "cpu": cpu_name(), "logical_cpu_count": psutil.cpu_count(logical=True), "physical_cpu_count": psutil.cpu_count(logical=False),
        "ram_bytes": int(psutil.virtual_memory().total), "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
        "pytorch": torch.__version__, "python": sys.version, "os": platform.platform(),
        "batch_size": 1, "input_resolution": [640, 640], "precision": "FP32", "model_eval": True,
        "inference_mode": True, "autocast": False, "tf32": False, "cuda_synchronize_at_boundaries": True,
        "timing_clock": "time.perf_counter_ns wall clock", "host_tensor_start": True,
        "excluded": ["disk read", "JPEG decode", "resize/to-tensor", "model/PCA/scaler load", "hook registration", "GT evaluation", "result save"],
        "preprocessed_host_tensor_seconds_excluded": preprocess_seconds,
        "profile_manifest_sha256": EXPECTED[P1A / "qa" / "profile_manifest.parquet"],
        "profile_image_count_loaded": 80,
    }
    write_json(PROJECT / "environment.json", environment)
    append_log(f"STAGE full_pipeline COMPLETE measured_samples={len(sample_df)} elapsed_seconds={time.perf_counter()-start_all:.6f}")


if __name__ == "__main__":
    main()
