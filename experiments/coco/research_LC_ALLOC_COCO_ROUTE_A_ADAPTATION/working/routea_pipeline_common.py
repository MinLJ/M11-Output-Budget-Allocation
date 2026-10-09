"""Shared, non-scientific helpers for the COCO Route-A pipeline.

This module contains no dataset or ground-truth entry point.  It centralizes
the immutable condition matrix, deterministic serialization helpers, model
asset parsing, and the relative-utility curve convention used by selection
and evaluation.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


DATASET_VERSION = "COCO2017-Road8-v1"
SPLIT = "VAL2017"
CANDIDATE_ASSET_ID = "5ff6497669d5b02c888f90451882d3c8e594b74a3aedd136639a9c0c8621c707"
EXPECTED_IMAGES = 5_000
GROUP_SIZE = 40
EXPECTED_GROUPS = 125
TOP_N = 100
K_MIN = 5
K_MAX = 50
BUDGETS = (10, 15, 20, 30, 40)
ROUTE_A_SEEDS = (630101, 630102, 630103)
ROUTE_B_SEEDS = (530101, 530102, 530103)
BASELINE_SEED = -1
CAL_TEMP = 0.389210829318298
ROAD8 = ("person", "bicycle", "car", "motorcycle", "bus", "train", "truck", "traffic light")
METHOD_ORDER = {"S_FIXED": 0, "S_ADAPT": 1, "CAL_TEMP_ALLOC": 2, "M11-COCO": 3}

ROOT_DEFAULT = (Path(os.environ.get('M11_EXPERIMENT_ROOT', "external_assets")) / 'research_LC_ALLOC_COCO_ROUTE_A_ADAPTATION')
ASSET_ROOT_DEFAULT = (Path(os.environ.get('M11_EXPERIMENT_ROOT', "external_assets")) / 'research_LC_ALLOC_COCO_CANONICAL_EXPORT')
GROUP_MANIFEST_DEFAULT = (Path(os.environ.get('M11_EXPERIMENT_ROOT', "external_assets")) / 'research_LC_ALLOC_COCO_EXECUTION_GATE/manifests/VAL2017_groups.csv')
P1_ROOT_DEFAULT = (Path(os.environ.get('M11_EXPERIMENT_ROOT', "external_assets")) / 'research_LC_ALLOC_P1')
ROUTE_B_SELECTION_ROOT_DEFAULT = (Path(os.environ.get('M11_EXPERIMENT_ROOT', "external_assets")) / 'research_LC_ALLOC_COCO_ROUTE_B_SELECTION')
ROUTE_B_EVALUATION_ROOT_DEFAULT = (Path(os.environ.get('M11_EXPERIMENT_ROOT', "external_assets")) / 'research_LC_ALLOC_COCO_ROUTE_B_EVALUATION')
ROUTE_B_RUNNER_DEFAULT = ROUTE_B_SELECTION_ROOT_DEFAULT / "working" / "route_b_selection_runner.py"
ROUTE_B_EVALUATOR_DEFAULT = ROUTE_B_EVALUATION_ROOT_DEFAULT / "working" / "evaluate_route_b.py"
PYCOCO_SITE_DEFAULT = (Path(os.environ.get('M11_ENVIRONMENT_ROOT', "external_assets")) / 'envs/aop_detr/Lib/site-packages')
VAL_GT_DEFAULT = (Path(os.environ.get('M11_COCO_ROOT', "external_assets")) / 'annotations/instances_val2017.json')

FROZEN_CODE_SHA256 = {
    "p1_core": "fd3daa06e363848c48685511ac4bd91189dea3cd82b7dbcc83748a1b3d0292f0",
    "dp_solver": "f230626580575e256576e1be92ca1ad409a57c9d725202f2b364f78dbca0c317",
    "train_models": "58358b2bcc371d4a4724da90a9bbe6a3199f157d90ec5c5f40e98a793b92e1bc",
}
LC_CLASS_WEIGHT_SHA256 = "2ee6f8d1a7daddd298c7874429d954d21885aefcab9a7a67bb1b49eba7f9c237"
VAL_GT_SHA256 = "e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f"
BOOTSTRAP_INDICES_SHA256 = "895fbbbe33169a2567aee34422d1dfdb8445ec5b6f64734ae819af06816af84a"
ROUTE_B_LEDGER_SHA256 = "37c7f0db40e916e64e1934db29692c52975d6ecedb51a755a2453ea3730992d8"
ROUTE_B_EVALUATOR_SHA256 = "e3b3c229ffc7f75e80ab66de2821ea4971cd35e4c692ddeb9959f7f65e1737fd"
ROUTE_B_SELECTION_RUNNER_SHA256 = "b2c247d71498f941f111f6434172d0e18e52462cfb91da5f203867a2e8470c1a"


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def sha256_file(path: str | os.PathLike[str], chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            block = stream.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def utc_now() -> str:
    import datetime as dt

    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def write_bytes_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        temporary.unlink()
    with temporary.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def write_json_atomic(path: Path, value: Any) -> None:
    write_bytes_atomic(path, canonical_json_bytes(value))


def write_text_atomic(path: Path, value: str) -> None:
    write_bytes_atomic(path, value.encode("utf-8"))


def write_csv_atomic(path: Path, frame: pd.DataFrame | Sequence[Mapping[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    buffer = io.StringIO(newline="")
    if isinstance(frame, pd.DataFrame):
        frame.to_csv(buffer, index=False, lineterminator="\n")
    else:
        require(fieldnames is not None, "fieldnames required for row mappings")
        writer = csv.DictWriter(buffer, fieldnames=list(fieldnames), extrasaction="raise", lineterminator="\n")
        writer.writeheader()
        for row in frame:
            writer.writerow(dict(row))
    write_text_atomic(path, buffer.getvalue())


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, f"cannot import module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_frozen_p1_modules(p1_root: Path):
    scripts = p1_root / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    import dp_solver  # type: ignore
    import p1_core  # type: ignore
    import train_models  # type: ignore

    modules = {"p1_core": p1_core, "dp_solver": dp_solver, "train_models": train_models}
    for name, module in modules.items():
        path = Path(module.__file__).resolve(strict=True)
        require(sha256_file(path) == FROZEN_CODE_SHA256[name], f"FROZEN_CODE_SHA_MISMATCH:{name}:{path}")
    result = dp_solver.self_test()
    require(result.get("status") == "PASS", f"DP_SELF_TEST_FAIL:{result}")
    return p1_core, dp_solver, train_models


def condition_instances() -> list[tuple[str, int, str]]:
    result = [
        ("S_FIXED", BASELINE_SEED, "FIXED_PREFIX_LENGTH"),
        ("S_ADAPT", BASELINE_SEED, "RAW_SCORE_MARGINAL"),
        ("CAL_TEMP_ALLOC", BASELINE_SEED, "GLOBAL_TEMPERATURE_SCORE_MARGINAL"),
    ]
    result.extend(("M11-COCO", seed, "COCO_ADAPTED_M11_QUALITY_MARGINAL") for seed in ROUTE_A_SEEDS)
    return result


def expected_conditions() -> set[tuple[str, int, int]]:
    result = {(method, seed, budget) for method, seed, _ in condition_instances() for budget in BUDGETS}
    return result


def policy_id(method: str, seed: int) -> str:
    return method if seed == BASELINE_SEED else f"{method}_SEED_{seed}"


def condition_label(method: str, seed: int, budget: int) -> str:
    seed_text = "DET" if seed < 0 else str(seed)
    return f"{method}|{seed_text}|K{budget}"


def stable_condition_sort(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["_method_order"] = out["method"].map(METHOD_ORDER)
    require(out["_method_order"].notna().all(), "unknown method in condition sort")
    return out.sort_values(["_method_order", "seed", "budget"], kind="mergesort").drop(columns="_method_order").reset_index(drop=True)


def load_class_weights(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    payload = read_json(path)
    strategy = str(payload.get("strategy", payload.get("choice", payload.get("policy", payload.get("class_weight_policy", ""))))).upper()
    all_sha_values = {
        str(value).lower()
        for key, value in _walk_items(payload)
        if "sha" in key.lower() and isinstance(value, str)
    }
    require("LC" in strategy or "KEEP" in strategy or "RETAIN" in strategy or LC_CLASS_WEIGHT_SHA256 in all_sha_values, "class-weight strategy is not frozen LC retention")
    weights = np.asarray(payload.get("weights"), dtype=np.float64)
    require(weights.shape == (8,) and np.isfinite(weights).all() and np.all(weights > 0), "invalid class weights")
    require(tuple(payload.get("classes", ())) == ROAD8, "class-weight class order mismatch")
    require(payload.get("frozen_before_training_and_val") is True, "class weights were not frozen before training/VAL")
    require(payload.get("training_loss_weighted") is False, "M11 training loss must remain unweighted")
    require(payload.get("allocation_utility_weighted") is True, "allocation utility must apply class weights")
    source_sha = str(payload.get("source_sha256", payload.get("class_weights_sha256", LC_CLASS_WEIGHT_SHA256 if LC_CLASS_WEIGHT_SHA256 in all_sha_values else "")))
    require(source_sha == LC_CLASS_WEIGHT_SHA256, f"LC class-weight SHA mismatch: {source_sha}")
    source_path = Path(str(payload.get("source_path", "")))
    require(source_path.is_file() and sha256_file(source_path) == source_sha, "LC class-weight source file/hash mismatch")
    return weights, payload


def load_temperatures(path: Path) -> tuple[dict[int, float], dict[str, Any]]:
    payload = read_json(path)
    require(payload.get("fit_role") == "CALIBRATION", "temperature fit role must be CALIBRATION")
    require(payload.get("val_accessed") is False, "temperature fitting accessed VAL")
    require(payload.get("shared_across_outputs") is True, "temperature must be shared across ten outputs")
    require(payload.get("one_parameter_per_seed") is True, "temperature must have one parameter per seed")
    require([float(value) for value in payload.get("bounds", [])] == [0.25, 4.0], "temperature bounds changed")
    items = payload.get("seeds")
    require(isinstance(items, list), "temperature_params.json must contain a seeds array")
    result: dict[int, float] = {}
    for item in items:
        require(isinstance(item, dict), "temperature seed entry must be an object")
        seed = int(item["seed"])
        temperature = float(item["temperature"])
        require(seed in ROUTE_A_SEEDS and seed not in result, f"unexpected/duplicate temperature seed: {seed}")
        require(math.isfinite(temperature) and temperature > 0, f"invalid temperature: {seed}")
        result[seed] = temperature
    require(set(result) == set(ROUTE_A_SEEDS), "temperature seed set mismatch")
    return result, payload


def validate_feature_config(path: Path, root: Path) -> dict[str, Any]:
    payload = read_json(path)
    dimension = payload.get("dimension", payload.get("feature_dimension", payload.get("input_dimension")))
    require(int(dimension) == 90, "feature_config dimension must be 90")
    text = json.dumps(payload, ensure_ascii=False).lower()
    require("964077b8b128c693d1bffeb627609d5a3b1da50b88e0bdcf0d6d469947f84b99" in text, "feature_config does not bind TRAIN2017 CandidateAssetID")
    pca_path = root / "pca_model.joblib"
    scaler_path = root / "scaler.joblib"
    require(pca_path.is_file() and scaler_path.is_file(), "Route-A PCA/scaler files are missing")
    for label, artifact in (("pca", pca_path), ("scaler", scaler_path)):
        actual = sha256_file(artifact)
        hashes = [str(v).lower() for k, v in _walk_items(payload) if "sha" in k.lower() and label in k.lower()]
        require(actual in hashes, f"feature_config does not bind {label} SHA: {actual}")
    return payload


def _walk_items(value: Any, prefix: str = "") -> Iterable[tuple[str, Any]]:
    if isinstance(value, dict):
        for key, item in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            yield name, item
            yield from _walk_items(item, name)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _walk_items(item, f"{prefix}[{index}]")


def curve_frame(image_ids: Sequence[str], margins: Mapping[int, Mapping[str, np.ndarray]]) -> pd.DataFrame:
    """Build the frozen relative-utility curve representation.

    M11 is supervised and evaluated only for extension ranks 6..50.  Because
    every feasible action includes ranks 1..5, the allocation objective is
    identifiable only up to that common additive constant.  We therefore
    encode k<5 as undefined, anchor U(5)=0, and cumulatively sum modeled
    deltas from k=6 onward.  No out-of-support prediction is manufactured.
    """

    rows: list[dict[str, Any]] = []
    for seed in ROUTE_A_SEEDS:
        per_image = margins[seed]
        require(set(per_image) == set(image_ids), f"curve image set mismatch for seed {seed}")
        for image_id in image_ids:
            delta = np.asarray(per_image[image_id], dtype=np.float64)
            require(delta.shape == (45,) and np.isfinite(delta).all(), f"curve margin shape/nonfinite: {seed}:{image_id}")
            cumulative = np.cumsum(delta, dtype=np.float64)
            for k in range(1, 51):
                if k < 5:
                    predicted_delta = math.nan
                    predicted_u = math.nan
                    modeled = False
                    origin = "UNDEFINED_BELOW_MANDATORY_K_MIN"
                elif k == 5:
                    predicted_delta = math.nan
                    predicted_u = 0.0
                    modeled = False
                    origin = "MANDATORY_K5_RELATIVE_UTILITY_ORIGIN"
                else:
                    predicted_delta = float(delta[k - 6])
                    predicted_u = float(cumulative[k - 6])
                    modeled = True
                    origin = "COCO_ADAPTED_M11_CALIBRATED_MARGIN"
                rows.append({
                    "image_id": image_id,
                    "seed": seed,
                    "k": k,
                    "predicted_delta": predicted_delta,
                    "predicted_U": predicted_u,
                    "method": "M11-COCO",
                    "is_model_predicted": modeled,
                    "curve_origin": origin,
                })
    frame = pd.DataFrame(rows)
    require(len(frame) == len(image_ids) * len(ROUTE_A_SEEDS) * 50, "predicted curve row count mismatch")
    return frame


def frozen_bootstrap_indices() -> tuple[np.ndarray, str]:
    rng = np.random.default_rng(530002)
    indices = rng.integers(0, 125, size=(5000, 125), endpoint=False, dtype=np.int16).astype("<i2", copy=False)
    header = (
        "schema=COCO2017_PAIRED_GROUP_BOOTSTRAP_V1"
        "\tgenerator=numpy.default_rng(PCG64)\tseed=530002"
        "\tshape=5000x125\tdtype=int16_little_endian\torder=C\tendpoint=false\n"
    ).encode("utf-8")
    digest = hashlib.sha256(header + indices.tobytes(order="C")).hexdigest()
    require(digest == BOOTSTRAP_INDICES_SHA256, f"bootstrap identity mismatch: {digest}")
    return indices, digest


def self_test_common() -> dict[str, Any]:
    images = ["image-a", "image-b"]
    margins = {
        seed: {image: np.full(45, (seed % 1000) / 1_000.0 + index / 100.0, dtype=np.float64) for index, image in enumerate(images)}
        for seed in ROUTE_A_SEEDS
    }
    curves = curve_frame(images, margins)
    require(len(curves) == 300, "synthetic curve cardinality")
    k5 = curves[curves.k == 5]
    require(k5["predicted_delta"].isna().all() and np.array_equal(k5["predicted_U"].to_numpy(), np.zeros(len(k5))), "curve K5 origin")
    modeled = curves[curves.k >= 6]
    require(modeled["predicted_delta"].notna().all() and modeled["predicted_U"].notna().all(), "modeled curve null")
    _, digest = frozen_bootstrap_indices()
    return {"status": "PASS", "curve_rows": len(curves), "bootstrap_indices_sha256": digest}


if __name__ == "__main__":
    print(json.dumps(self_test_common(), sort_keys=True))
