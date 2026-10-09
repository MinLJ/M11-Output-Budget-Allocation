"""Profile P2 calibration allocators against the accepted P1B M11 path.

This script is deliberately limited to the fixed P1A FIT200 profiling manifest.
It never accepts a GT path, opens an image, runs the detector, or touches DEV/TEST.
The timed boundary is policy-specific host-resident input through value estimation,
the frozen exact float64 DP, and ordered candidate-record IDs.

Expected calibration-parameter JSON (minor nesting aliases are accepted)::

    {
      "cal_temp": {"temperature": 1.23},
      "cal_iso": {
        "x_thresholds": [0.0, 0.1, 1.0],
        "y_thresholds": [0.0, 0.02, 0.3]
      }
    }

The P2 fit script's canonical alternative is also accepted: top-level
``temperature`` plus ``isotonic_parameters_npz`` pointing to arrays named
``isotonic_x_thresholds`` and ``isotonic_y_thresholds``.

The default parameter path is ``<project-root>/models/calibration_parameters.json``.
Raw samples and aggregate summaries are Parquet only; the final user-facing CSV is
intentionally left to the parent P2 reporting pipeline.
"""

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
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch


sys.dont_write_bytecode = True

POLICIES = ("CAL_TEMP_ALLOC", "CAL_ISO_ALLOC", "M11_OPTIMIZED")
BUDGETS = (10, 20, 40)
PROFILE_GROUPS = 5
GROUP_SIZE = 40
WARMUPS_PER_GROUP_CONDITION = 5
MEASUREMENTS_PER_GROUP_CONDITION = 20
ORDER_SEED = 530003
M11_SEED = 530101
SCORE_EPSILON = 1e-6


def import_file(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    out = np.empty_like(values)
    nonnegative = values >= 0
    out[nonnegative] = 1.0 / (1.0 + np.exp(-values[nonnegative]))
    exp_values = np.exp(values[~nonnegative])
    out[~nonnegative] = exp_values / (1.0 + exp_values)
    return out


def nested_value(payload: dict[str, Any], paths: tuple[tuple[str, ...], ...]) -> Any:
    for path in paths:
        value: Any = payload
        for key in path:
            if not isinstance(value, dict) or key not in value:
                break
            value = value[key]
        else:
            return value
    joined = [".".join(path) for path in paths]
    raise KeyError(f"none of the expected parameter keys exist: {joined}")


def load_calibration_parameters(path: Path) -> tuple[float, np.ndarray, np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    temperature = float(
        nested_value(
            payload,
            (
                ("cal_temp", "temperature"),
                ("temperature_scaling", "temperature"),
                ("cal_temp_temperature",),
                ("temperature",),
            ),
        )
    )
    try:
        iso_x_value = nested_value(
            payload,
            (
                ("cal_iso", "x_thresholds"),
                ("isotonic", "x_thresholds"),
                ("isotonic", "x_thresholds_"),
                ("isotonic_x_thresholds",),
            ),
        )
        iso_y_value = nested_value(
            payload,
            (
                ("cal_iso", "y_thresholds"),
                ("isotonic", "y_thresholds"),
                ("isotonic", "y_thresholds_"),
                ("isotonic_y_thresholds",),
            ),
        )
        iso_x = np.asarray(iso_x_value, dtype=np.float64)
        iso_y = np.asarray(iso_y_value, dtype=np.float64)
    except KeyError:
        npz_value = nested_value(
            payload,
            (
                ("isotonic_parameters_npz",),
                ("cal_iso", "parameters_npz"),
                ("isotonic", "parameters_npz"),
            ),
        )
        npz_path = Path(str(npz_value))
        if not npz_path.is_absolute():
            npz_path = (path.parent / npz_path).resolve()
        if not npz_path.is_file():
            raise FileNotFoundError(f"isotonic parameter sidecar missing: {npz_path}")
        declared_npz_sha = payload.get("isotonic_parameters_npz_sha256")
        if declared_npz_sha is not None and sha256_file(npz_path) != str(declared_npz_sha):
            raise RuntimeError("isotonic parameter sidecar fails its declared SHA256")
        with np.load(npz_path, allow_pickle=False) as sidecar:
            iso_x = np.asarray(sidecar["isotonic_x_thresholds"], dtype=np.float64)
            iso_y = np.asarray(sidecar["isotonic_y_thresholds"], dtype=np.float64)
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("global temperature must be finite and strictly positive")
    if iso_x.ndim != 1 or iso_y.ndim != 1 or len(iso_x) != len(iso_y) or len(iso_x) < 2:
        raise ValueError("isotonic thresholds must be equal-length 1-D arrays with >=2 entries")
    if not np.all(np.isfinite(iso_x)) or not np.all(np.isfinite(iso_y)):
        raise ValueError("isotonic thresholds must be finite")
    if np.any(np.diff(iso_x) <= 0):
        raise ValueError("isotonic x thresholds must be strictly increasing")
    if np.any(np.diff(iso_y) < 0) or np.any((iso_y < 0) | (iso_y > 1)):
        raise ValueError("isotonic y thresholds must be monotone and lie in [0,1]")
    return temperature, iso_x, iso_y


def summary(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "samples": int(len(array)),
        "mean_ms": float(array.mean()),
        "median_ms": float(np.median(array)),
        "p95_ms": float(np.quantile(array, 0.95)),
        "std_ms": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "min_ms": float(array.min()),
        "max_ms": float(array.max()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--p1-root", default=r"D:\AOP_DETR\research_LC_ALLOC_P1")
    parser.add_argument("--p1a-root", default=r"D:\AOP_DETR\research_LC_ALLOC_P1A")
    parser.add_argument("--p1b-root", default=r"D:\AOP_DETR\research_LC_ALLOC_P1B")
    parser.add_argument(
        "--release-root",
        default=(
            r"D:\AOP_DETR\shared_benchmark\AOP_ROAD8_LARGECLEAN_V1"
            r"\P1_SHARED_EXPORT\release\AOP_Road8_LC_v1_RTDERTv2_R18VD_RELEASE_v1"
        ),
    )
    parser.add_argument("--calibration-params", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    p1_root = Path(args.p1_root).resolve()
    p1a_root = Path(args.p1a_root).resolve()
    p1b_root = Path(args.p1b_root).resolve()
    release_root = Path(args.release_root).resolve()
    params_path = (
        Path(args.calibration_params).resolve()
        if args.calibration_params
        else project_root / "models" / "calibration_parameters.json"
    )
    output_root = project_root / "outputs"
    output_root.mkdir(parents=True, exist_ok=True)

    # These are the only data/model dependencies. There is intentionally no GT,
    # image-root, detector, DEV, TEST, RESERVE, or historical-dataset argument.
    profile_manifest_path = p1a_root / "qa" / "profile_manifest.parquet"
    p1b_script_path = p1b_root / "scripts" / "profile_optimizations.py"
    implementation_status_path = p1b_root / "outputs" / "implementation_status.json"
    p1b_ledger_path = p1b_root / "input_output_sha256.csv"
    identity_path = release_root / "CANDIDATE_ASSET_IDENTITY.json"
    required = (
        params_path,
        profile_manifest_path,
        p1b_script_path,
        implementation_status_path,
        p1b_ledger_path,
        identity_path,
        p1_root / "scripts" / "dp_solver.py",
        p1_root / "scripts" / "p1_core.py",
        p1_root / "scripts" / "train_models.py",
        p1_root / "models" / "pca32.joblib",
        p1_root / "models" / "feature_scaler.joblib",
        p1_root / "models" / "class_weights.json",
        p1_root / "models" / f"marginal_mlp_seed_{M11_SEED}.pt",
        p1_root / "models" / f"temperature_seed_{M11_SEED}.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"required frozen inputs missing: {missing}")

    implementation_status = json.loads(implementation_status_path.read_text(encoding="utf-8"))
    if implementation_status.get("optimization_status") != "ACCEPTED":
        raise RuntimeError("P1B M11 optimized implementation is not frozen as ACCEPTED")
    if int(implementation_status.get("full_dev_K_mismatch", -1)) != 0:
        raise RuntimeError("P1B accepted implementation has nonzero K mismatch")
    if int(implementation_status.get("full_dev_selection_mismatch", -1)) != 0:
        raise RuntimeError("P1B accepted implementation has nonzero selection mismatch")
    p1b_ledger = pd.read_csv(p1b_ledger_path, dtype=str, keep_default_na=False)
    normalized_ledger_paths = p1b_ledger["path"].str.replace("\\", "/", regex=False)
    source_rows = p1b_ledger[
        normalized_ledger_paths.str.endswith("scripts/profile_optimizations.py")
    ]
    if len(source_rows) != 1:
        raise RuntimeError("P1B ledger does not uniquely bind profile_optimizations.py")
    expected_p1b_script_sha = str(source_rows.iloc[0]["sha256"]).lower()
    actual_p1b_script_sha = sha256_file(p1b_script_path)
    if actual_p1b_script_sha != expected_p1b_script_sha:
        raise RuntimeError("P1B accepted runtime implementation fails frozen-ledger SHA")

    temperature, iso_x, iso_y = load_calibration_parameters(params_path)
    params_sha256 = sha256_file(params_path)

    sys.path.insert(0, str(p1_root / "scripts"))
    from dp_solver import solve_group_allocations  # type: ignore
    from p1_core import iou_xyxy  # type: ignore
    from train_models import MarginalMLP  # type: ignore

    p1b = import_file(p1b_script_path, "lc_alloc_p2_frozen_p1b_profile")
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    profile_manifest = pq.read_table(profile_manifest_path).to_pandas()
    required_manifest_columns = {"image_id", "profile_group_id", "position_in_group"}
    if not required_manifest_columns.issubset(profile_manifest.columns):
        raise RuntimeError("P1A profile manifest lacks required identity/group columns")
    profile_manifest["image_id"] = profile_manifest["image_id"].astype(str)
    profile_manifest = profile_manifest.sort_values(
        ["profile_group_id", "position_in_group"], kind="stable"
    )
    if len(profile_manifest) != PROFILE_GROUPS * GROUP_SIZE:
        raise RuntimeError("P1A profile manifest must contain exactly 200 FIT identities")
    counts = profile_manifest.groupby("profile_group_id").size().to_dict()
    if set(map(int, counts)) != set(range(PROFILE_GROUPS)) or any(
        int(value) != GROUP_SIZE for value in counts.values()
    ):
        raise RuntimeError(f"invalid P1A profile-group structure: {counts}")

    profile_ids = profile_manifest["image_id"].tolist()
    raw_images, excluded_read_seconds = p1b.load_raw_images(
        release_root, identity, "TRAIN", profile_ids
    )
    # Frozen DP row order is lexicographic image_id within each fixed group, as
    # used by P1B's accepted implementation.
    groups = [
        [
            raw_images[image_id]
            for image_id in sorted(
                profile_manifest.loc[
                    profile_manifest["profile_group_id"] == group_id, "image_id"
                ].tolist()
            )
        ]
        for group_id in range(PROFILE_GROUPS)
    ]

    load_start = time.perf_counter()
    pca = joblib.load(p1_root / "models" / "pca32.joblib")
    scaler_bundle = joblib.load(p1_root / "models" / "feature_scaler.joblib")
    scaler = scaler_bundle["scaler"]
    standardize = np.asarray(scaler_bundle["standardize_mask"], dtype=bool)
    class_weights = np.asarray(
        json.loads((p1_root / "models" / "class_weights.json").read_text(encoding="utf-8"))[
            "weights"
        ],
        dtype=np.float64,
    )
    if class_weights.shape != (8,):
        raise RuntimeError("frozen M11 class-weight vector must have length 8")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    snapshot = torch.load(
        p1_root / "models" / f"marginal_mlp_seed_{M11_SEED}.pt",
        map_location="cpu",
        weights_only=True,
    )
    model = MarginalMLP(int(snapshot["input_dim"]))
    model.load_state_dict(snapshot["state_dict"])
    model.eval().to(device)
    m11_temperature = float(
        json.loads(
            (p1_root / "models" / f"temperature_seed_{M11_SEED}.json").read_text(
                encoding="utf-8"
            )
        )["temperature"]
    )
    sync(device)
    excluded_load_seconds = time.perf_counter() - load_start

    @torch.inference_mode()
    def infer_m11(features: np.ndarray) -> np.ndarray:
        tensor = torch.from_numpy(np.asarray(features, dtype=np.float32)).to(device)
        logits = model(tensor).float().cpu().numpy().astype(np.float64)
        sync(device)
        return sigmoid(logits / m11_temperature)

    def selected_from_k(
        records: list[list[str]], allocations: np.ndarray
    ) -> list[list[str]]:
        return [records[index][: int(allocations[index])] for index in range(GROUP_SIZE)]

    def validate_output(
        allocations: np.ndarray, selected: list[list[str]], budget: int
    ) -> None:
        if allocations.shape != (GROUP_SIZE,):
            raise RuntimeError("allocator returned an invalid K shape")
        if np.any((allocations < 5) | (allocations > 50)):
            raise RuntimeError("allocator returned K outside [5,50]")
        if int(allocations.sum(dtype=np.int64)) != GROUP_SIZE * budget:
            raise RuntimeError("allocator failed the exact group budget")
        if len(selected) != GROUP_SIZE or sum(map(len, selected)) != GROUP_SIZE * budget:
            raise RuntimeError("returned record IDs do not match the exact group budget")
        if any(len(values) != len(set(values)) for values in selected):
            raise RuntimeError("a selected prefix contains duplicate candidate_record_id")

    # Host-resident baseline views contain precisely the policy-required source
    # columns. Calibration policies do not pay for native tensors they do not use.
    baseline_views: list[tuple[np.ndarray, list[list[str]]]] = []
    for group in groups:
        score_rows: list[np.ndarray] = []
        record_rows: list[list[str]] = []
        for raw in group:
            candidates = raw.candidates.sort_values("road8_rank", kind="stable").reset_index(
                drop=True
            )
            ranks = candidates["road8_rank"].to_numpy(dtype=np.int64)
            if not np.array_equal(ranks, np.arange(1, 101, dtype=np.int64)):
                raise RuntimeError(f"Top100 rank invariant failed for {raw.image_id}")
            score_rows.append(candidates["score"].to_numpy(dtype=np.float64))
            record_rows.append(candidates.iloc[:50]["candidate_record_id"].astype(str).tolist())
        baseline_views.append((np.vstack(score_rows), record_rows))

    def calibration_policy(
        group_id: int, budget: int, transform: Callable[[np.ndarray], np.ndarray]
    ) -> tuple[np.ndarray, list[list[str]], float]:
        scores, records = baseline_views[group_id]
        # Only optional prefix additions k=6..50 enter the common U_i(5)=0 DP.
        marginal_values = np.asarray(transform(scores[:, 5:50]), dtype=np.float64)
        if marginal_values.shape != (GROUP_SIZE, 45) or not np.all(
            np.isfinite(marginal_values)
        ):
            raise RuntimeError("calibration transform returned invalid optional-slot values")
        allocation_rows, objectives = solve_group_allocations(marginal_values, [budget])
        allocations = allocation_rows[0]
        selected = selected_from_k(records, allocations)
        validate_output(allocations, selected, budget)
        return allocations, selected, float(objectives[0])

    def temperature_transform(scores: np.ndarray) -> np.ndarray:
        clipped = np.clip(np.asarray(scores, dtype=np.float64), SCORE_EPSILON, 1-SCORE_EPSILON)
        logits = np.log(clipped / (1.0 - clipped))
        return sigmoid(logits / temperature)

    def isotonic_transform(scores: np.ndarray) -> np.ndarray:
        return np.interp(
            np.asarray(scores, dtype=np.float64),
            iso_x,
            iso_y,
            left=float(iso_y[0]),
            right=float(iso_y[-1]),
        )

    def m11_policy(
        group_id: int, budget: int
    ) -> tuple[np.ndarray, list[list[str]], float]:
        group = groups[group_id]
        features, classes, records = p1b.optimized_group_features(
            group, pca, scaler, standardize, iou_xyxy
        )
        probabilities = infer_m11(features)
        if probabilities.shape != (GROUP_SIZE * 45, 10):
            raise RuntimeError(f"M11 output shape mismatch: {probabilities.shape}")
        marginal_values = (
            np.ascontiguousarray(probabilities).mean(axis=1) * class_weights[classes - 1]
        ).reshape(GROUP_SIZE, 45)
        allocation_rows, objectives = solve_group_allocations(marginal_values, [budget])
        allocations = allocation_rows[0]
        selected = selected_from_k(records, allocations)
        validate_output(allocations, selected, budget)
        return allocations, selected, float(objectives[0])

    def run_policy(
        policy: str, group_id: int, budget: int
    ) -> tuple[np.ndarray, list[list[str]], float]:
        if policy == "CAL_TEMP_ALLOC":
            return calibration_policy(group_id, budget, temperature_transform)
        if policy == "CAL_ISO_ALLOC":
            return calibration_policy(group_id, budget, isotonic_transform)
        if policy == "M11_OPTIMIZED":
            return m11_policy(group_id, budget)
        raise KeyError(policy)

    # Run one untimed identity/capacity check for every condition before timing.
    frozen_digest: dict[tuple[str, int, int], str] = {}
    for policy in POLICIES:
        for group_id in range(PROFILE_GROUPS):
            for budget in BUDGETS:
                allocations, selected, _ = run_policy(policy, group_id, budget)
                validate_output(allocations, selected, budget)
                frozen_digest[(policy, group_id, budget)] = hashlib.sha256(
                    "|".join(record for rows in selected for record in rows).encode("utf-8")
                ).hexdigest()

    conditions = [
        (policy, group_id, budget)
        for policy in POLICIES
        for group_id in range(PROFILE_GROUPS)
        for budget in BUDGETS
    ]
    rng = random.Random(ORDER_SEED)
    warmups = [condition for condition in conditions for _ in range(WARMUPS_PER_GROUP_CONDITION)]
    measurements = [
        condition for condition in conditions for _ in range(MEASUREMENTS_PER_GROUP_CONDITION)
    ]
    rng.shuffle(warmups)
    rng.shuffle(measurements)

    samples: list[dict[str, Any]] = []
    measured_order = 0
    for phase, calls in (("WARMUP", warmups), ("MEASURED", measurements)):
        for phase_order, (policy, group_id, budget) in enumerate(calls):
            sync(device)
            start = time.perf_counter_ns()
            allocations, selected, objective = run_policy(policy, group_id, budget)
            sync(device)
            elapsed_ms = (time.perf_counter_ns() - start) / 1e6
            digest = hashlib.sha256(
                "|".join(record for rows in selected for record in rows).encode("utf-8")
            ).hexdigest()
            if digest != frozen_digest[(policy, group_id, budget)]:
                raise RuntimeError(
                    f"nondeterministic selection policy={policy} group={group_id} budget={budget}"
                )
            if phase == "MEASURED":
                samples.append(
                    {
                        "sample_order": measured_order,
                        "interleaved_phase_order": phase_order,
                        "phase": phase,
                        "timing_scope": "FULL_ALLOCATOR_PATH",
                        "policy": policy,
                        "implementation": (
                            "M11_OPT_V1_BATCHED_PCA_SCALER"
                            if policy == "M11_OPTIMIZED"
                            else "P2_SCORE_CALIBRATION_EXACT_DP"
                        ),
                        "seed": M11_SEED if policy == "M11_OPTIMIZED" else -1,
                        "budget": budget,
                        "profile_group_id": group_id,
                        "elapsed_ms": elapsed_ms,
                        "amortized_ms_per_image": elapsed_ms / GROUP_SIZE,
                        "output_records": int(allocations.sum(dtype=np.int64)),
                        "predicted_objective": objective,
                        "selection_digest": digest,
                        "calibration_parameters_sha256": params_sha256,
                    }
                )
                measured_order += 1

    samples_frame = pd.DataFrame(samples)
    expected_samples = (
        len(POLICIES)
        * len(BUDGETS)
        * PROFILE_GROUPS
        * MEASUREMENTS_PER_GROUP_CONDITION
    )
    if len(samples_frame) != expected_samples:
        raise RuntimeError(f"runtime sample count mismatch: {len(samples_frame)} != {expected_samples}")
    samples_path = output_root / "calibration_runtime_samples.parquet"
    pq.write_table(
        pa.Table.from_pandas(samples_frame, preserve_index=False),
        samples_path,
        compression="zstd",
    )

    summaries: list[dict[str, Any]] = []
    for (policy, seed, budget), rows in samples_frame.groupby(
        ["policy", "seed", "budget"], sort=True
    ):
        stats = summary(rows["elapsed_ms"].tolist())
        summaries.append(
            {
                "timing_scope": "FULL_ALLOCATOR_PATH",
                "policy": str(policy),
                "implementation": str(rows["implementation"].iloc[0]),
                "seed": int(seed),
                "budget": int(budget),
                "input_images": PROFILE_GROUPS * GROUP_SIZE,
                "input_groups": int(rows["profile_group_id"].nunique()),
                **stats,
                "amortized_median_ms_per_image": float(stats["median_ms"]) / GROUP_SIZE,
                "boundary": (
                    "host-resident policy-required candidate/native tensors through "
                    "value estimator, exact float64 DP, and ordered record IDs"
                ),
                "warmups_per_group_condition": WARMUPS_PER_GROUP_CONDITION,
                "measurements_per_group_condition": MEASUREMENTS_PER_GROUP_CONDITION,
                "order_seed": ORDER_SEED,
                "calibration_parameters_sha256": params_sha256,
            }
        )
    summary_frame = pd.DataFrame(summaries)
    if len(summary_frame) != len(POLICIES) * len(BUDGETS):
        raise RuntimeError("runtime summary must have exactly nine policy-budget rows")
    summary_path = output_root / "calibration_runtime_summary.parquet"
    pq.write_table(
        pa.Table.from_pandas(summary_frame, preserve_index=False),
        summary_path,
        compression="zstd",
    )

    metadata = {
        "status": "COMPLETE",
        "script": str(Path(__file__).resolve()),
        "profile_manifest": str(profile_manifest_path),
        "profile_manifest_sha256": sha256_file(profile_manifest_path),
        "profile_role": "FIT",
        "profile_images": PROFILE_GROUPS * GROUP_SIZE,
        "profile_groups": PROFILE_GROUPS,
        "group_size": GROUP_SIZE,
        "policies": list(POLICIES),
        "budgets": list(BUDGETS),
        "m11_seed": M11_SEED,
        "m11_implementation": "M11_OPT_V1_BATCHED_PCA_SCALER",
        "m11_implementation_sha256": actual_p1b_script_sha,
        "m11_accepted_status_sha256": sha256_file(implementation_status_path),
        "calibration_parameters": str(params_path),
        "calibration_parameters_sha256": params_sha256,
        "warmups_per_group_condition": WARMUPS_PER_GROUP_CONDITION,
        "measurements_per_group_condition": MEASUREMENTS_PER_GROUP_CONDITION,
        "measured_samples": expected_samples,
        "order_seed": ORDER_SEED,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "python": sys.version,
        "os": platform.platform(),
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "excluded_release_read_seconds": excluded_read_seconds,
        "excluded_model_preprocessor_load_seconds": excluded_load_seconds,
        "gt_access": 0,
        "detector_forward": 0,
        "test_access": 0,
        "raw_samples": str(samples_path),
        "summary": str(summary_path),
    }
    (output_root / "calibration_runtime_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
