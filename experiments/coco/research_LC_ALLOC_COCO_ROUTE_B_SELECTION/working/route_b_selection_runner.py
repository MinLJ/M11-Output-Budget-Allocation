"""GT-free frozen LC M11 Route-B selection on COCO2017 VAL2017.

This entrypoint deliberately has no annotation/GT/evaluator argument.  It
consumes only the frozen canonical candidate/native asset, frozen grouping,
and frozen LC allocator assets.  It publishes selections only after complete
structural QA and writes a final logical completion marker last.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import sklearn
import torch


sys.dont_write_bytecode = True

TASK = "LC-ALLOC-COCO-ROUTE-B-SELECTION"
DATASET_VERSION = "COCO2017-Road8-v1"
SPLIT = "VAL2017"
CANDIDATE_ASSET_ID = "5ff6497669d5b02c888f90451882d3c8e594b74a3aedd136639a9c0c8621c707"
EXPECTED_IMAGES = 5000
GROUP_SIZE = 40
EXPECTED_GROUPS = 125
TOP_N = 100
K_MIN = 5
K_MAX = 50
BUDGETS = (10, 15, 20, 30, 40)
SEEDS = (530101, 530102, 530103)
BASELINE_SEED = -1
CAL_TEMP = 0.389210829318298
ROAD8 = ("person", "bicycle", "car", "motorcycle", "bus", "train", "truck", "traffic light")
EXPECTED_DETECTOR_ID = "RT-DETRv2-R18VD"
EXPECTED_CHECKPOINT_SHA = "2ace52184b620204004509b72752ac7bfe64aadaf7fc1d076b18df8ab5a5c77e"
EXPECTED_EXPORT_CONFIG_SHA = "f6a91e9d6879357fc10095c182689e43ee73bca2c10ec6926f93960cc68d98f2"
EXPECTED_CANDIDATE_SCHEMA_SHA = "e977e02fe638a8ac44e98919d76225306eb88fa7385d3ec0793342112ec8c53b"

OUTPUT_ROOT_DEFAULT = (Path(os.environ.get('M11_EXPERIMENT_ROOT', "external_assets")) / 'research_LC_ALLOC_COCO_ROUTE_B_SELECTION')
ASSET_ROOT_DEFAULT = (Path(os.environ.get('M11_EXPERIMENT_ROOT', "external_assets")) / 'research_LC_ALLOC_COCO_CANONICAL_EXPORT')
GROUP_MANIFEST_DEFAULT = (Path(os.environ.get('M11_EXPERIMENT_ROOT', "external_assets")) / 'research_LC_ALLOC_COCO_EXECUTION_GATE/manifests/VAL2017_groups.csv')
P1_ROOT_DEFAULT = (Path(os.environ.get('M11_EXPERIMENT_ROOT', "external_assets")) / 'research_LC_ALLOC_P1')
P1A_ROOT_DEFAULT = (Path(os.environ.get('M11_EXPERIMENT_ROOT', "external_assets")) / 'research_LC_ALLOC_P1A')
P2_ROOT_DEFAULT = (Path(os.environ.get('M11_EXPERIMENT_ROOT', "external_assets")) / 'research_LC_ALLOC_P2')

EXPECTED_STATIC_SHA = {
    "group_manifest": "b1346893ba70c1b6f1904f83530aa956253badefc8b227473cc8660ab78a5cc6",
    "feature_schema": "8613d979ab6393c5d0524f1451e80329f60d0caa7a514ae256be4022d70096f7",
    "pca32": "12d8e656d53bdad54e498125993f437f77d874e97be38ffe29a0c969e1ff41af",
    "feature_scaler": "9d3da8c36b3b5a2cf8c8ab2deb963e3595c313b4d13f6d3e7fb8c27d923ced6d",
    "class_weights": "2ee6f8d1a7daddd298c7874429d954d21885aefcab9a7a67bb1b49eba7f9c237",
    "model_530101": "cb9732dbc828eff968b96294ab541b925eacc0d57472faf5a11f4bb246597aa8",
    "model_530102": "7b60f0f4d9ef5fc63941ed61398086ee7c82abf3c928a0db7b8c905f27c20996",
    "model_530103": "890cf82a9cf654d6b44816b2111d51e86ee84ba1f35050b299c3473a17d0b6b8",
    "temperature_530101": "e321ba3385d8eb7260d36164f13c401e89750ec35e5d590ebffe29a98aee4b03",
    "temperature_530102": "d54c68323d16179d8e78a125cabaa247d1303c8f3867871f1967ec1996827501",
    "temperature_530103": "ad4805b6566ab24eb15e0b44f9a01d01908e493dd4307c27c2e52e0afd1341da",
    "p1_core": "fd3daa06e363848c48685511ac4bd91189dea3cd82b7dbcc83748a1b3d0292f0",
    "dp_solver": "f230626580575e256576e1be92ca1ad409a57c9d725202f2b364f78dbca0c317",
    "train_models": "58358b2bcc371d4a4724da90a9bbe6a3199f157d90ec5c5f40e98a793b92e1bc",
    "candidate_freeze": "8889fe172a90a98572fdd9e1fddced2fbf9a4bf0ded65841f5a19b1c625180d7",
    "calibration_parameters": "f36ef64e62de555464bf0eb3ee865e8f97941fa287380e6bc538ea1c00ab3c92",
    "train_X_raw": "5ed3b351f4cc416f5dfc72e1643ab303cfb130a6066758046f24e229f9f4adea",
    "train_role": "d28cd903cff297ea9e46e30b24f01ae620fbf307de7fa9fbf6dde5d34e4dc231",
}

EXPECTED_TEMPERATURES = {
    530101: 1.0053284330481267,
    530102: 1.017592050678593,
    530103: 1.0074616372046543,
}

CANDIDATE_COLUMNS = [
    "dataset_version", "split", "coco_image_id", "image_id", "canonical_image_id",
    "image_sha256", "candidate_asset_id", "detector_id", "checkpoint_sha256",
    "export_config_sha256", "schema_sha256", "candidate_record_id", "road8_rank",
    "source_order", "query_index", "predicted_road8_class_id",
    "predicted_road8_class_name", "predicted_coco_category_id",
    "detector_class_index", "score", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
    "bbox_cx", "bbox_cy", "bbox_w", "bbox_h",
]

SELECTED_SCHEMA = pa.schema([
    pa.field("dataset_version", pa.string(), nullable=False),
    pa.field("split", pa.string(), nullable=False),
    pa.field("candidate_asset_id", pa.string(), nullable=False),
    pa.field("method", pa.string(), nullable=False),
    pa.field("seed", pa.int64(), nullable=False),
    pa.field("policy_instance_id", pa.string(), nullable=False),
    pa.field("value_semantics", pa.string(), nullable=False),
    pa.field("budget", pa.int16(), nullable=False),
    pa.field("group_id", pa.int16(), nullable=False),
    pa.field("image_id", pa.string(), nullable=False),
    pa.field("canonical_image_id", pa.string(), nullable=False),
    pa.field("coco_image_id", pa.int64(), nullable=False),
    pa.field("K_i", pa.int16(), nullable=False),
    pa.field("selection_rank", pa.int16(), nullable=False),
    pa.field("candidate_record_id", pa.string(), nullable=False),
    pa.field("road8_rank", pa.int16(), nullable=False),
    pa.field("query_index", pa.int16(), nullable=False),
    pa.field("predicted_road8_class_id", pa.int8(), nullable=False),
    pa.field("score", pa.float64(), nullable=False),
])

CURRENT_STAGE = "initialization"


def progress(stage: str, message: str) -> None:
    print(f"[{utc_now()}] {stage}: {message}", flush=True)


def utc_now() -> str:
    import datetime as _dt
    return _dt.datetime.now(_dt.timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


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


def write_csv_atomic(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    import io
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(fieldnames), lineterminator="\n", extrasaction="raise")
    writer.writeheader()
    for row in rows:
        writer.writerow(dict(row))
    write_text_atomic(path, buffer.getvalue())


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def quantiles(values: np.ndarray, points: Sequence[float] = (0, .01, .05, .25, .5, .75, .95, .99, 1)) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("DIAGNOSTIC_ARRAY_EMPTY_OR_NONFINITE")
    result = np.quantile(array, points, method="linear")
    return {f"q{int(round(point * 100)):02d}": float(value) for point, value in zip(points, result)}


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def static_paths(p1_root: Path, p1a_root: Path, p2_root: Path, group_manifest: Path) -> dict[str, Path]:
    return {
        "group_manifest": group_manifest,
        "feature_schema": p1_root / "feature_schema.json",
        "pca32": p1_root / "models" / "pca32.joblib",
        "feature_scaler": p1_root / "models" / "feature_scaler.joblib",
        "class_weights": p1_root / "models" / "class_weights.json",
        **{f"model_{seed}": p1_root / "models" / f"marginal_mlp_seed_{seed}.pt" for seed in SEEDS},
        **{f"temperature_{seed}": p1_root / "models" / f"temperature_seed_{seed}.json" for seed in SEEDS},
        "p1_core": p1_root / "scripts" / "p1_core.py",
        "dp_solver": p1_root / "scripts" / "dp_solver.py",
        "train_models": p1_root / "scripts" / "train_models.py",
        "candidate_freeze": p1a_root / "candidate_freeze.json",
        "calibration_parameters": p2_root / "models" / "calibration_parameters.json",
        "train_X_raw": p1_root / "cache" / "train_X_raw.npy",
        "train_role": p1_root / "cache" / "train_role.npy",
    }


def verify_static_bindings(paths: Mapping[str, Path]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for key, path in paths.items():
        path = path.resolve(strict=True)
        actual = sha256_file(path)
        expected = EXPECTED_STATIC_SHA.get(key)
        if expected is not None and actual != expected:
            raise ValueError(f"FROZEN_INPUT_SHA_MISMATCH:{key}:{actual}")
        result[key] = {"path": str(path), "bytes": path.stat().st_size, "sha256": actual}
    return result


def verify_canonical_asset(asset_root: Path) -> tuple[dict[str, Any], list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    marker_path = asset_root / "CANONICAL_EXPORT_COMPLETE.json"
    marker = read_json(marker_path)
    if marker.get("status") != "COCO_CANONICAL_EXPORT_COMPLETE" or not marker.get("qa_all_checks_pass"):
        raise ValueError("CANONICAL_ASSET_NOT_COMPLETE")
    if marker.get("candidate_asset_id") != CANDIDATE_ASSET_ID or int(marker.get("image_count", -1)) != EXPECTED_IMAGES:
        raise ValueError("CANONICAL_ASSET_IDENTITY_MISMATCH")
    if marker.get("dataset_version") != DATASET_VERSION or marker.get("split") != SPLIT:
        raise ValueError("CANONICAL_ASSET_DATASET_SPLIT_MISMATCH")
    qa_path = asset_root / "qa_validation_result.json"
    ledger_path = asset_root / "sha256_ledger.csv"
    if sha256_file(qa_path) != marker.get("qa_validation_result_sha256"):
        raise ValueError("CANONICAL_QA_MARKER_SHA_MISMATCH")
    ledger_meta = marker.get("sha256_ledger", {})
    if sha256_file(ledger_path) != ledger_meta.get("sha256"):
        raise ValueError("CANONICAL_LEDGER_MARKER_SHA_MISMATCH")
    qa = read_json(qa_path)
    if not qa.get("all_checks_pass") or qa.get("annotation_or_gt_read") or qa.get("scientific_evaluation_performed"):
        raise ValueError("CANONICAL_ASSET_QA_BOUNDARY_FAIL")
    if int(qa.get("candidate_rows", -1)) != 1_500_000 or int(qa.get("native_rows", -1)) != 1_500_000:
        raise ValueError("CANONICAL_ASSET_ROW_COUNT_FAIL")

    candidate_manifest = read_csv_rows(asset_root / "candidate_manifest.csv")
    native_manifest = read_csv_rows(asset_root / "native_manifest.csv")
    image_manifest = read_csv_rows(asset_root / "image_manifest.csv")
    if len(candidate_manifest) != 10 or len(native_manifest) != 10 or len(image_manifest) != EXPECTED_IMAGES:
        raise ValueError("CANONICAL_MANIFEST_CARDINALITY_FAIL")

    marker_manifests = marker.get("manifest_sha256", {})
    for name in ("candidate_manifest.csv", "native_manifest.csv", "image_manifest.csv", "shard_manifest.csv"):
        if sha256_file(asset_root / name) != marker_manifests.get(name):
            raise ValueError(f"CANONICAL_MANIFEST_MARKER_SHA_MISMATCH:{name}")

    ledger_rows = read_csv_rows(ledger_path)
    if len(ledger_rows) != int(ledger_meta.get("entries", -1)):
        raise ValueError("CANONICAL_LEDGER_ENTRY_COUNT_MISMATCH")
    ledger_lookup: dict[str, set[str]] = {}
    for row in ledger_rows:
        ledger_lookup.setdefault(row["path"].replace("\\", "/"), set()).add(row["sha256"])
    required_ledger_bindings: list[tuple[str, str]] = [
        (name, marker_manifests[name])
        for name in ("candidate_manifest.csv", "native_manifest.csv", "image_manifest.csv", "shard_manifest.csv")
    ]
    required_ledger_bindings.extend((row["path"], row["sha256"]) for row in candidate_manifest)
    required_ledger_bindings.extend((row["path"], row["sha256"]) for row in native_manifest)
    required_ledger_bindings.extend((row["index_path"], row["index_sha256"]) for row in native_manifest)
    for relative_path, expected_sha in required_ledger_bindings:
        if expected_sha not in ledger_lookup.get(relative_path.replace("\\", "/"), set()):
            raise ValueError(f"CANONICAL_LEDGER_DEPENDENCY_BINDING_FAIL:{relative_path}")

    for row in candidate_manifest:
        path = asset_root / row["path"]
        if path.stat().st_size != int(row["bytes"]) or sha256_file(path) != row["sha256"]:
            raise ValueError(f"CANDIDATE_SHARD_SHA_FAIL:{path.name}")
    for row in native_manifest:
        native = asset_root / row["path"]
        index = asset_root / row["index_path"]
        if native.stat().st_size != int(row["bytes"]) or sha256_file(native) != row["sha256"]:
            raise ValueError(f"NATIVE_SHARD_SHA_FAIL:{native.name}")
        if index.stat().st_size != int(row["index_bytes"]) or sha256_file(index) != row["index_sha256"]:
            raise ValueError(f"NATIVE_INDEX_SHA_FAIL:{index.name}")
    return marker, candidate_manifest, native_manifest, image_manifest


def verify_groups(group_manifest: Path, image_manifest: Sequence[Mapping[str, str]]) -> tuple[pd.DataFrame, dict[str, int]]:
    groups = pd.read_csv(group_manifest, dtype={"canonical_image_id": str, "group_key": str})
    required = ["group_id", "position", "coco_image_id", "canonical_image_id", "group_key"]
    if groups.columns.tolist() != required or len(groups) != EXPECTED_IMAGES:
        raise ValueError("GROUP_MANIFEST_SCHEMA_OR_COUNT_FAIL")
    if groups["canonical_image_id"].nunique() != EXPECTED_IMAGES or groups["group_id"].nunique() != EXPECTED_GROUPS:
        raise ValueError("GROUP_MANIFEST_IDENTITY_FAIL")
    if not (groups.groupby("group_id", sort=True).size().to_numpy() == GROUP_SIZE).all():
        raise ValueError("GROUP_SIZE_FAIL")
    expected_keys = groups["canonical_image_id"].map(lambda value: sha256_text("LC_ALLOC_COCO2017_GROUP_V1|" + value))
    if not np.array_equal(expected_keys.to_numpy(), groups["group_key"].to_numpy()):
        raise ValueError("GROUP_KEY_FAIL")
    sorted_copy = groups.sort_values(["group_key", "canonical_image_id"], kind="stable").reset_index(drop=True)
    if not np.array_equal(sorted_copy["canonical_image_id"].to_numpy(), groups["canonical_image_id"].to_numpy()):
        raise ValueError("GROUP_GLOBAL_ORDER_FAIL")
    expected_group = np.arange(EXPECTED_IMAGES, dtype=np.int64) // GROUP_SIZE
    expected_position = np.arange(EXPECTED_IMAGES, dtype=np.int64) % GROUP_SIZE
    if not np.array_equal(groups["group_id"].to_numpy(np.int64), expected_group) or not np.array_equal(groups["position"].to_numpy(np.int64), expected_position):
        raise ValueError("GROUP_NUMBERING_FAIL")
    image_ids = {row["canonical_image_id"] for row in image_manifest}
    if set(groups["canonical_image_id"].astype(str)) != image_ids:
        raise ValueError("GROUP_CANONICAL_ASSET_ID_SET_FAIL")
    image_coco_ids = {row["canonical_image_id"]: int(row["coco_image_id"]) for row in image_manifest}
    expected_coco_ids = groups["canonical_image_id"].astype(str).map(image_coco_ids).to_numpy(np.int64)
    if not np.array_equal(groups["coco_image_id"].to_numpy(np.int64), expected_coco_ids):
        raise ValueError("GROUP_COCO_IMAGE_ID_JOIN_FAIL")
    return groups, dict(zip(groups["canonical_image_id"].astype(str), groups["group_id"].astype(int)))


def fit_feature_bounds(raw_path: Path, role_path: Path) -> tuple[np.ndarray, np.ndarray, int]:
    raw = np.load(raw_path, mmap_mode="r", allow_pickle=False)
    role = np.load(role_path, mmap_mode="r", allow_pickle=False)
    if raw.shape != (450000, 90) or role.shape != (450000,):
        raise ValueError(f"FIT_REFERENCE_SHAPE_FAIL:{raw.shape}:{role.shape}")
    minimum = np.full(90, np.inf, dtype=np.float64)
    maximum = np.full(90, -np.inf, dtype=np.float64)
    fit_rows = 0
    for start in range(0, len(role), 32768):
        stop = min(start + 32768, len(role))
        mask = np.asarray(role[start:stop]) == 0
        if not np.any(mask):
            continue
        block = np.asarray(raw[start:stop][mask], dtype=np.float64)
        if not np.isfinite(block).all():
            raise ValueError("FIT_REFERENCE_NONFINITE")
        minimum = np.minimum(minimum, block.min(axis=0))
        maximum = np.maximum(maximum, block.max(axis=0))
        fit_rows += len(block)
    if fit_rows != 360000 or not np.isfinite(minimum).all() or not np.isfinite(maximum).all():
        raise ValueError(f"FIT_REFERENCE_ROWS_FAIL:{fit_rows}")
    return minimum, maximum, fit_rows


def load_frozen_assets(p1_root: Path, p2_root: Path, p1_core: Any) -> tuple[Any, Any, np.ndarray, np.ndarray, dict[int, float]]:
    pca = joblib.load(p1_root / "models" / "pca32.joblib")
    scaler_bundle = joblib.load(p1_root / "models" / "feature_scaler.joblib")
    if int(getattr(pca, "n_components_", -1)) != 32 or int(getattr(pca, "n_features_in_", -1)) != 256 or bool(getattr(pca, "whiten", True)):
        raise ValueError("PCA_METADATA_FAIL")
    if not isinstance(scaler_bundle, dict) or "scaler" not in scaler_bundle or "standardize_mask" not in scaler_bundle:
        raise ValueError("SCALER_BUNDLE_FAIL")
    scaler = scaler_bundle["scaler"]
    standardize = np.asarray(scaler_bundle["standardize_mask"], dtype=bool)
    if standardize.shape != (90,) or int(standardize.sum()) != 81 or int(getattr(scaler, "n_features_in_", -1)) != 81:
        raise ValueError("SCALER_METADATA_FAIL")
    if np.flatnonzero(~standardize).tolist() != [3, 4, 5, 6, 7, 8, 9, 10, 69]:
        raise ValueError("SCALER_MASK_FAIL")

    weights_payload = read_json(p1_root / "models" / "class_weights.json")
    if tuple(weights_payload.get("classes", [])) != ROAD8:
        raise ValueError("CLASS_WEIGHT_ORDER_FAIL")
    class_weights = np.asarray(weights_payload["weights"], dtype=np.float64)
    if class_weights.shape != (8,) or not np.isfinite(class_weights).all() or np.any(class_weights <= 0):
        raise ValueError("CLASS_WEIGHTS_FAIL")

    temperatures: dict[int, float] = {}
    for seed in SEEDS:
        value = float(read_json(p1_root / "models" / f"temperature_seed_{seed}.json")["temperature"])
        if value != EXPECTED_TEMPERATURES[seed]:
            raise ValueError(f"M11_TEMPERATURE_FAIL:{seed}:{value}")
        temperatures[seed] = value

    calibration = read_json(p2_root / "models" / "calibration_parameters.json")
    if float(calibration["temperature"]) != CAL_TEMP or calibration.get("class_weights_applied_to_baseline_slot_value") is not False:
        raise ValueError("CAL_TEMP_BINDING_FAIL")
    if calibration.get("target") != "mean of ten frozen maximum-matching prefix marginal labels":
        raise ValueError("CAL_TEMP_TARGET_FAIL")

    schema = read_json(p1_root / "feature_schema.json")
    if int(schema.get("dimension", -1)) != 90 or schema.get("columns") != p1_core.feature_spec():
        raise ValueError("FEATURE_SCHEMA_CODE_PARITY_FAIL")
    return pca, scaler, standardize, class_weights, temperatures


def verify_candidate_freeze(path: Path) -> None:
    freeze = read_json(path)
    if freeze.get("candidate") != "LEARN_QUALITY":
        raise ValueError("CANDIDATE_FREEZE_METHOD_FAIL")
    if freeze.get("K_bounds") != [K_MIN, K_MAX] or freeze.get("budgets") != list(BUDGETS) or int(freeze.get("group_size", -1)) != GROUP_SIZE:
        raise ValueError("CANDIDATE_FREEZE_ACTION_FAIL")
    if freeze.get("feature_schema", {}).get("sha256") != EXPECTED_STATIC_SHA["feature_schema"]:
        raise ValueError("CANDIDATE_FREEZE_SCHEMA_FAIL")
    if freeze.get("pca", {}).get("sha256") != EXPECTED_STATIC_SHA["pca32"] or freeze.get("scaler", {}).get("sha256") != EXPECTED_STATIC_SHA["feature_scaler"]:
        raise ValueError("CANDIDATE_FREEZE_PREPROCESSOR_FAIL")
    if freeze.get("class_weights", {}).get("sha256") != EXPECTED_STATIC_SHA["class_weights"]:
        raise ValueError("CANDIDATE_FREEZE_WEIGHTS_FAIL")
    frozen_models = {int(row["seed"]): row for row in freeze.get("models", [])}
    if set(frozen_models) != set(SEEDS):
        raise ValueError("CANDIDATE_FREEZE_SEEDS_FAIL")
    for seed in SEEDS:
        row = frozen_models[seed]
        if row.get("model_sha256") != EXPECTED_STATIC_SHA[f"model_{seed}"] or row.get("temperature_sha256") != EXPECTED_STATIC_SHA[f"temperature_{seed}"]:
            raise ValueError(f"CANDIDATE_FREEZE_MODEL_BINDING_FAIL:{seed}")
        if float(row.get("temperature")) != EXPECTED_TEMPERATURES[seed]:
            raise ValueError(f"CANDIDATE_FREEZE_TEMPERATURE_FAIL:{seed}")


def load_models(p1_root: Path, train_models: Any, device: torch.device) -> dict[int, torch.nn.Module]:
    models: dict[int, torch.nn.Module] = {}
    for seed in SEEDS:
        path = p1_root / "models" / f"marginal_mlp_seed_{seed}.pt"
        snapshot = torch.load(path, map_location="cpu", weights_only=True)
        if int(snapshot.get("input_dim", -1)) != 90 or int(snapshot.get("seed", seed)) != seed:
            raise ValueError(f"MODEL_METADATA_FAIL:{seed}")
        model = train_models.MarginalMLP(90)
        model.load_state_dict(snapshot["state_dict"], strict=True)
        model.eval().to(device)
        if model.training or any(parameter.requires_grad for parameter in model.parameters()):
            # Training flags are irrelevant under inference_mode, but freezing
            # gradients makes the selection-only boundary explicit.
            for parameter in model.parameters():
                parameter.requires_grad_(False)
        model.eval()
        models[seed] = model
    return models


def content_sequence_sha(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    for image_id, record_id, rank in zip(
        frame["canonical_image_id"].astype(str),
        frame["candidate_record_id"].astype(str),
        frame["road8_rank"].astype(int),
    ):
        digest.update(image_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(record_id.encode("ascii"))
        digest.update(b"\0")
        digest.update(str(rank).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def build_distribution_report(diag: Mapping[str, Any]) -> str:
    class_lines = "\n".join(
        f"- {name}: {diag['predicted_class_proportions'][name]:.8f}"
        for name in ROAD8
    )
    geometry_lines = "\n".join(
        f"- `{name}`: " + ", ".join(f"{key}={value:.6g}" for key, value in values.items())
        for name, values in diag["geometry_summary"].items()
    )
    top_exceed = "\n".join(
        f"- `{item['index']:02d} {item['name']}`: {item['exceed_ratio']:.8f}"
        for item in diag["top_feature_exceed_ratios"]
    )
    return f"""# COCO Route-B input distribution shift diagnostic

本报告只描述冻结 M11 输入分布，不参与分配决策，也不触发 refit、clip、模型切换或参数调整。

## 绑定范围

- Dataset：`{DATASET_VERSION} / {SPLIT}`
- CandidateAssetID：`{CANDIDATE_ASSET_ID}`
- 模型输入行：{diag['model_feature_rows']:,}（5,000 图 × ranks 6..50）
- 候选池记录：{diag['candidate_pool_rows']:,}（每图原始 Top100）
- LC FIT 参考行：{diag['fit_reference_rows']:,}

## 90D 范围漂移

- 超出 LC FIT 每列 raw min/max 的 feature cells 比例：{diag['feature_cells_outside_fit_minmax_ratio']:.8f}
- 至少一列超界的输入行比例：{diag['feature_rows_with_any_outside_ratio']:.8f}
- 出现过超界值的特征维数：{diag['feature_dimensions_with_any_outside']}/90

超界比例最高的列：

{top_exceed}

## 标准化输入与 PCA

- 标准化后 90D 绝对值分位数：`{json.dumps(diag['standardized_abs_quantiles'], ensure_ascii=False, sort_keys=True)}`
- PCA32 向量 L2 norm：`{json.dumps(diag['pca_norm_quantiles'], ensure_ascii=False, sort_keys=True)}`
- stored float16 embedding 转 float32 后 norm：`{json.dumps(diag['embedding_norm_quantiles'], ensure_ascii=False, sort_keys=True)}`

## Candidate pool

- Raw score：`{json.dumps(diag['raw_score_quantiles'], ensure_ascii=False, sort_keys=True)}`
- Rank：min={diag['rank_distribution']['min']}，max={diag['rank_distribution']['max']}，每个 rank count={diag['rank_distribution']['count_per_rank']}。
- 类别比例：

{class_lines}

Geometry（按原图宽高归一化，未 clip）：

{geometry_lines}

## 数值检查

- NaN count：{diag['nan_count']}
- Inf count：{diag['inf_count']}
- candidate/native score reconstruction max abs error：{diag['score_reconstruction_max_abs_error']:.12g}

这些统计未使用 annotation/GT，未改变任何选择值。
"""


def build_qa_report(qa: Mapping[str, Any]) -> str:
    return f"""# COCO Route-B selection QA report

## 状态

```text
execution_status = COMPLETE
selection_status = COMPLETE
gt_access_status = NOT_ACCESSED
qa_status = PASS
```

## 冻结输入与覆盖

- Dataset：`{DATASET_VERSION} / {SPLIT}`
- CandidateAssetID：`{CANDIDATE_ASSET_ID}`
- Images：{qa['images']}/{EXPECTED_IMAGES}
- Groups：{qa['groups']}/{EXPECTED_GROUPS}，每组 {GROUP_SIZE} 图
- Conditions：{qa['conditions']}（确定性 baseline 使用 `seed=-1`；M11 三个 seed 独立）
- Allocation rows：{qa['allocation_rows']:,}
- Selected rows：{qa['selected_rows']:,}

## 结构 QA

- K range failures：{qa['k_range_failures']}
- Exact group-budget failures：{qa['group_budget_failures']}
- Prefix failures：{qa['prefix_failures']}
- Candidate identity/image failures：{qa['candidate_identity_failures']}
- Duplicate selected records within condition：{qa['duplicate_selection_failures']}
- NaN/Inf：{qa['nan_count']}/{qa['inf_count']}
- Serialized image-count failures：{qa['serialized_image_count_failures']}
- Serialized K consistency failures：{qa['serialized_k_consistency_failures']}
- Serialized group identity failures：{qa['serialized_group_id_failures']}
- Serialized exact group-budget failures：{qa['serialized_group_budget_failures']}
- CAL_TEMP vs S_ADAPT K mismatches：{qa['cal_temp_k_mismatches']}
- CAL_TEMP vs S_ADAPT selected-record symmetric difference：{qa['cal_temp_selection_symmetric_difference']}
- CAL_TEMP vs S_ADAPT serialized-order mismatch budgets：{qa['cal_temp_selection_order_mismatch_budgets']}
- CAL_TEMP optional scores clipped low/high：{qa['cal_temp_clip_low_count']}/{qa['cal_temp_clip_high_count']}
- CAL_TEMP strict-equivalence expectation：{qa['cal_temp_strict_equivalence_expected']}
- Selected rows participating in retained same-query multi-class groups：{qa['retained_same_query_multiclass_rows']:,}

所有最终选集均严格为原 canonical `road8_rank = 1..K_i` 前缀；没有 rerank、NMS、query dedup、score 修改或跨组借预算。

## 访问边界

Selection entrypoint 不接受 annotation/GT 参数，没有导入 evaluator，没有执行 matching、AP/AR、Coverage 或 QUALITY 评价。QUALITY 仅作为冻结 M11 的**预测效用定义**，未计算任何真实标注指标。
"""


def condition_instances() -> list[tuple[str, int, str]]:
    result = [
        ("S_FIXED", BASELINE_SEED, "FIXED_PREFIX_LENGTH"),
        ("S_ADAPT", BASELINE_SEED, "RAW_SCORE_MARGINAL"),
        ("CAL_TEMP_ALLOC", BASELINE_SEED, "GLOBAL_TEMPERATURE_SCORE_MARGINAL"),
    ]
    result.extend(("M11", seed, "FROZEN_M11_QUALITY_MARGINAL") for seed in SEEDS)
    return result


def policy_id(method: str, seed: int) -> str:
    return method if seed == BASELINE_SEED else f"{method}_SEED_{seed}"


def execute(args: argparse.Namespace) -> dict[str, Any]:
    global CURRENT_STAGE
    started_total = time.perf_counter()
    output_root = args.output_root.resolve()
    asset_root = args.asset_root.resolve(strict=True)
    group_manifest = args.group_manifest.resolve(strict=True)
    p1_root = args.p1_root.resolve(strict=True)
    p1a_root = args.p1a_root.resolve(strict=True)
    p2_root = args.p2_root.resolve(strict=True)
    working = output_root / "working"
    staging = working / ("staging_" + CANDIDATE_ASSET_ID[:12])
    output_root.mkdir(parents=True, exist_ok=True)
    working.mkdir(parents=True, exist_ok=True)
    if (output_root / "SELECTION_COMPLETE.json").exists():
        raise FileExistsError("SELECTION_ALREADY_COMPLETE")
    if staging.exists() and any(staging.iterdir()):
        raise FileExistsError("NONEMPTY_STAGING_REFUSED")
    planned_public_files = {
        "run_config.json", "selection_manifest.csv", "selected_records.parquet",
        "allocation_summary.csv", "distribution_shift_report.md",
        "selection_QA_report.md", "runtime_summary.csv", "sha256_ledger.csv",
    }
    existing_public_files = sorted(name for name in planned_public_files if (output_root / name).exists())
    if existing_public_files:
        raise FileExistsError(f"PARTIAL_PUBLICATION_REFUSED:{existing_public_files}")
    staging.mkdir(parents=True, exist_ok=True)
    runner_path = Path(__file__).resolve(strict=True)
    runtime_rows: list[dict[str, Any]] = []

    CURRENT_STAGE = "frozen_preflight"
    t0 = time.perf_counter()
    progress(CURRENT_STAGE, "verifying frozen hashes, canonical manifests, groups, and solver")
    paths = static_paths(p1_root, p1a_root, p2_root, group_manifest)
    bindings = verify_static_bindings(paths)
    verify_candidate_freeze(paths["candidate_freeze"])
    marker, candidate_manifest, native_manifest, image_manifest_rows = verify_canonical_asset(asset_root)
    groups, group_map = verify_groups(group_manifest, image_manifest_rows)
    image_manifest = pd.DataFrame(image_manifest_rows)
    image_manifest["canonical_image_id"] = image_manifest["canonical_image_id"].astype(str)
    image_manifest["coco_image_id"] = image_manifest["coco_image_id"].astype(np.int64)
    image_manifest["width"] = image_manifest["width"].astype(np.int64)
    image_manifest["height"] = image_manifest["height"].astype(np.int64)
    image_meta = image_manifest.set_index("canonical_image_id", drop=False)
    if image_manifest["canonical_image_id"].nunique() != EXPECTED_IMAGES:
        raise ValueError("IMAGE_MANIFEST_IDENTITY_FAIL")
    if (image_manifest["width"] <= 0).any() or (image_manifest["height"] <= 0).any():
        raise ValueError("IMAGE_MANIFEST_DIMENSION_FAIL")

    runner_sha = sha256_file(runner_path)
    bindings["runner"] = {"path": str(runner_path), "bytes": runner_path.stat().st_size, "sha256": runner_sha}
    bindings["canonical_completion_marker"] = {
        "path": str(asset_root / "CANONICAL_EXPORT_COMPLETE.json"),
        "bytes": (asset_root / "CANONICAL_EXPORT_COMPLETE.json").stat().st_size,
        "sha256": sha256_file(asset_root / "CANONICAL_EXPORT_COMPLETE.json"),
    }
    bindings["canonical_qa"] = {
        "path": str(asset_root / "qa_validation_result.json"),
        "bytes": (asset_root / "qa_validation_result.json").stat().st_size,
        "sha256": sha256_file(asset_root / "qa_validation_result.json"),
    }
    bindings["canonical_sha256_ledger"] = {
        "path": str(asset_root / "sha256_ledger.csv"),
        "bytes": (asset_root / "sha256_ledger.csv").stat().st_size,
        "sha256": sha256_file(asset_root / "sha256_ledger.csv"),
    }
    for row in candidate_manifest:
        path = (asset_root / row["path"]).resolve(strict=True)
        bindings[f"candidate_shard_{row['shard_id']}"] = {"path": str(path), "bytes": path.stat().st_size, "sha256": row["sha256"]}
    for row in native_manifest:
        path = (asset_root / row["path"]).resolve(strict=True)
        index = (asset_root / row["index_path"]).resolve(strict=True)
        bindings[f"native_shard_{row['shard_id']}"] = {"path": str(path), "bytes": path.stat().st_size, "sha256": row["sha256"]}
        bindings[f"native_index_{row['shard_id']}"] = {"path": str(index), "bytes": index.stat().st_size, "sha256": row["index_sha256"]}
    for name in ("candidate_manifest.csv", "native_manifest.csv", "image_manifest.csv", "shard_manifest.csv"):
        path = (asset_root / name).resolve(strict=True)
        bindings[f"canonical_{name}"] = {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}

    p1_scripts = p1_root / "scripts"
    if str(p1_scripts) not in sys.path:
        sys.path.insert(0, str(p1_scripts))
    import p1_core  # type: ignore
    import dp_solver  # type: ignore
    import train_models  # type: ignore

    for module, binding_key in ((p1_core, "p1_core"), (dp_solver, "dp_solver"), (train_models, "train_models")):
        imported_path = Path(module.__file__).resolve(strict=True)
        if imported_path != paths[binding_key].resolve(strict=True) or sha256_file(imported_path) != EXPECTED_STATIC_SHA[binding_key]:
            raise ValueError(f"FROZEN_MODULE_RESOLUTION_FAIL:{binding_key}:{imported_path}")

    dp_test = dp_solver.self_test()
    if dp_test.get("status") != "PASS":
        raise ValueError(f"FROZEN_DP_SELF_TEST_FAIL:{dp_test}")
    pca, scaler, standardize, class_weights, temperatures = load_frozen_assets(p1_root, p2_root, p1_core)
    if not torch.cuda.is_available():
        raise RuntimeError("FROZEN_ROUTE_B_REQUIRES_CUDA_ENVIRONMENT")
    device = torch.device("cuda:0")
    runtime_rows.append({"stage": "frozen_preflight", "method": "ALL", "seed": -1, "budget": -1, "seconds": time.perf_counter() - t0, "images": EXPECTED_IMAGES, "groups": EXPECTED_GROUPS, "notes": "hash all direct frozen inputs; no GT"})
    progress(CURRENT_STAGE, "PASS; 5,000 images and 125 frozen groups bound")

    run_config = {
        "task": TASK,
        "created_at_utc": utc_now(),
        "dataset_version": DATASET_VERSION,
        "split": SPLIT,
        "candidate_asset_id": CANDIDATE_ASSET_ID,
        "route": "B_FROZEN_LC_M11_TRANSFER",
        "methods": {
            "deterministic_baselines": ["S_FIXED", "S_ADAPT", "CAL_TEMP_ALLOC"],
            "deterministic_seed_sentinel": BASELINE_SEED,
            "m11_seeds": list(SEEDS),
            "m11_seed_predictions_averaged": False,
        },
        "candidate_action": {"pool": "canonical original road8_rank Top100", "selection": "rank 1..K_i prefix only", "rerank": False, "nms": False, "query_dedup": False},
        "allocation": {"group_size": GROUP_SIZE, "group_count": EXPECTED_GROUPS, "caller_order": "canonical_image_id ascending within group", "budgets": list(BUDGETS), "k_bounds": [K_MIN, K_MAX], "exact_budget": "sum K_i = group_size * budget", "solver": "frozen exact float64 multiple-choice DP; lexicographically smallest K vector on exact objective tie"},
        "m11": {"feature_dimension": 90, "pca_components": 32, "probability": "sigmoid(per-seed logits / per-seed frozen temperature)", "value": "frozen FIT class weight times mean of ten calibrated output probabilities", "temperatures": {str(k): v for k, v in temperatures.items()}, "class_weights": class_weights.tolist()},
        "cal_temp": {"temperature": CAL_TEMP, "formula": "sigmoid(logit(clip(score,1e-6,1-1e-6))/T)", "class_weight": False, "refit": False},
        "distribution_shift": {"diagnostic_only": True, "decision_effect": False, "fit_reference": "LC FIT raw 90D feature min/max"},
        "access_boundary": {"annotation_or_gt_argument": False, "annotation_or_gt_read": False, "matching": False, "scientific_evaluation": False},
        "environment": {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__, "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(0), "numpy": np.__version__, "pandas": pd.__version__, "pyarrow": pa.__version__, "sklearn": sklearn.__version__, "joblib": joblib.__version__, "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32), "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32)},
        "input_bindings": bindings,
        "runner_source_sha256": runner_sha,
    }
    run_config_path = staging / "run_config.json"
    write_json_atomic(run_config_path, run_config)

    CURRENT_STAGE = "fit_reference_bounds"
    t0 = time.perf_counter()
    progress(CURRENT_STAGE, "reading frozen LC FIT feature cache for diagnostic bounds only")
    fit_min, fit_max, fit_rows = fit_feature_bounds(paths["train_X_raw"], paths["train_role"])
    runtime_rows.append({"stage": "fit_reference_bounds", "method": "DIAGNOSTIC", "seed": -1, "budget": -1, "seconds": time.perf_counter() - t0, "images": 8000, "groups": 0, "notes": "read frozen LC feature cache only; no GT"})

    CURRENT_STAGE = "canonical_asset_read_and_feature_construction"
    t0 = time.perf_counter()
    feature_count = EXPECTED_IMAGES * (K_MAX - K_MIN)
    features = np.empty((feature_count, 90), dtype=np.float32)
    feature_classes = np.empty(feature_count, dtype=np.int16)
    image_slices: dict[str, slice] = {}
    candidate_parts: list[pd.DataFrame] = []
    score_parts: list[np.ndarray] = []
    geometry_parts: dict[str, list[np.ndarray]] = {name: [] for name in ("cx_norm", "cy_norm", "w_norm", "h_norm", "area_norm", "log_aspect")}
    pca_norm_parts: list[np.ndarray] = []
    embedding_norm_parts: list[np.ndarray] = []
    class_counts = np.zeros(8, dtype=np.int64)
    rank_counts = np.zeros(TOP_N, dtype=np.int64)
    feature_exceed_counts = np.zeros(90, dtype=np.int64)
    feature_rows_any_exceed = 0
    numeric_nan_count = 0
    numeric_inf_count = 0
    score_reconstruction_max_abs_error = 0.0
    observed_detector_ids: set[str] = set()
    observed_checkpoint_shas: set[str] = set()
    observed_export_config_shas: set[str] = set()
    observed_schema_shas: set[str] = set()
    row0 = 0

    native_by_id = {row["shard_id"]: row for row in native_manifest}
    for shard_number, candidate_manifest_row in enumerate(candidate_manifest, start=1):
        shard_id = candidate_manifest_row["shard_id"]
        if shard_id not in native_by_id:
            raise ValueError(f"CANDIDATE_NATIVE_MANIFEST_PAIR_FAIL:{shard_id}")
        candidate_path = asset_root / candidate_manifest_row["path"]
        native_path = asset_root / native_by_id[shard_id]["path"]
        table = pq.read_table(candidate_path, columns=CANDIDATE_COLUMNS, filters=[("road8_rank", "<=", TOP_N)])
        frame = table.to_pandas()
        if len(frame) != 500 * TOP_N:
            raise ValueError(f"TOP100_SHARD_ROW_COUNT_FAIL:{shard_id}:{len(frame)}")
        if set(frame["candidate_asset_id"].astype(str).unique()) != {CANDIDATE_ASSET_ID} or set(frame["dataset_version"].astype(str).unique()) != {DATASET_VERSION} or set(frame["split"].astype(str).unique()) != {SPLIT}:
            raise ValueError(f"TOP100_SHARD_IDENTITY_FAIL:{shard_id}")
        if frame["candidate_record_id"].duplicated().any():
            raise ValueError(f"TOP100_SHARD_DUPLICATE_RECORD_FAIL:{shard_id}")
        observed_detector_ids.update(frame["detector_id"].astype(str).unique())
        observed_checkpoint_shas.update(frame["checkpoint_sha256"].astype(str).unique())
        observed_export_config_shas.update(frame["export_config_sha256"].astype(str).unique())
        observed_schema_shas.update(frame["schema_sha256"].astype(str).unique())

        with np.load(native_path, allow_pickle=False) as native:
            native_image_ids = native["image_ids"].astype(str)
            native_query = native["query_index"].astype(np.int32)
            native_logits = native["l3_road8_logits"]
            native_embedding = native["l3_query_embedding"]
            if native_logits.shape != (150000, 8) or native_embedding.shape != (150000, 256):
                raise ValueError(f"NATIVE_SHAPE_FAIL:{shard_id}")
            starts: dict[str, int] = {}
            for start in range(0, len(native_image_ids), 300):
                ids = native_image_ids[start:start + 300]
                query = native_query[start:start + 300]
                if len(ids) != 300 or len(set(ids)) != 1 or not np.array_equal(query, np.arange(300, dtype=np.int32)):
                    raise ValueError(f"NATIVE_BLOCK_FAIL:{shard_id}:{start}")
                starts[str(ids[0])] = start

            for image_id, rows in frame.groupby("canonical_image_id", sort=False):
                image_id = str(image_id)
                rows = rows.sort_values("road8_rank", kind="stable").reset_index(drop=True)
                if len(rows) != TOP_N or rows["road8_rank"].astype(int).tolist() != list(range(1, TOP_N + 1)):
                    raise ValueError(f"TOP100_RANK_FAIL:{image_id}")
                if not (rows["image_id"].astype(str) == image_id).all() or image_id not in starts or image_id not in image_meta.index or image_id not in group_map:
                    raise ValueError(f"CANDIDATE_NATIVE_MANIFEST_JOIN_FAIL:{image_id}")
                meta = image_meta.loc[image_id]
                if not (rows["coco_image_id"].to_numpy(np.int64) == int(meta["coco_image_id"])).all():
                    raise ValueError(f"CANDIDATE_COCO_IMAGE_ID_FAIL:{image_id}")
                queries = rows["query_index"].to_numpy(np.int32)
                if np.any((queries < 0) | (queries >= 300)):
                    raise ValueError(f"QUERY_RANGE_FAIL:{image_id}")
                native_rows = starts[image_id] + queries
                if not np.all(native_image_ids[native_rows] == image_id) or not np.array_equal(native_query[native_rows], queries):
                    raise ValueError(f"NATIVE_GATHER_FAIL:{image_id}")
                logits = np.asarray(native_logits[native_rows], dtype=np.float32)
                embeddings = np.asarray(native_embedding[native_rows], dtype=np.float16)
                classes = rows["predicted_road8_class_id"].to_numpy(np.int16)
                if np.any((classes < 1) | (classes > 8)) or rows["predicted_road8_class_name"].astype(str).tolist() != [ROAD8[value - 1] for value in classes]:
                    raise ValueError(f"CANDIDATE_CLASS_IDENTITY_FAIL:{image_id}")
                reconstructed = p1_core.sigmoid(logits[np.arange(TOP_N), classes - 1])
                score_diff = float(np.max(np.abs(reconstructed - rows["score"].to_numpy(np.float64))))
                score_reconstruction_max_abs_error = max(score_reconstruction_max_abs_error, score_diff)
                if score_diff > 1e-6:
                    raise ValueError(f"SCORE_RECONSTRUCTION_FAIL:{image_id}:{score_diff}")
                raw = p1_core.build_raw_features(rows, logits, embeddings, pca, float(meta["width"]), float(meta["height"]))
                if raw.shape != (45, 90) or not np.isfinite(raw).all():
                    raise ValueError(f"RAW_FEATURE_FAIL:{image_id}")
                outside = (raw < fit_min[None, :]) | (raw > fit_max[None, :])
                feature_exceed_counts += outside.sum(axis=0, dtype=np.int64)
                feature_rows_any_exceed += int(np.any(outside, axis=1).sum())
                pca_norm_parts.append(np.linalg.norm(raw[:, 25:57], axis=1))
                embedding_norm_parts.append(raw[:, 57].copy())
                scaled = raw.copy()
                scaled[:, standardize] = scaler.transform(scaled[:, standardize])
                if not np.isfinite(scaled).all():
                    raise ValueError(f"SCALED_FEATURE_FAIL:{image_id}")
                features[row0:row0 + 45] = scaled.astype(np.float32)
                feature_classes[row0:row0 + 45] = classes[5:50]
                image_slices[image_id] = slice(row0, row0 + 45)
                row0 += 45

                width = float(meta["width"])
                height = float(meta["height"])
                wn = rows["bbox_w"].to_numpy(np.float64) / width
                hn = rows["bbox_h"].to_numpy(np.float64) / height
                geometry_parts["cx_norm"].append(rows["bbox_cx"].to_numpy(np.float64) / width)
                geometry_parts["cy_norm"].append(rows["bbox_cy"].to_numpy(np.float64) / height)
                geometry_parts["w_norm"].append(wn)
                geometry_parts["h_norm"].append(hn)
                geometry_parts["area_norm"].append(wn * hn)
                geometry_parts["log_aspect"].append(np.log(np.maximum(wn, 1e-6) / np.maximum(hn, 1e-6)))
                scores = rows["score"].to_numpy(np.float64)
                score_parts.append(scores)
                class_counts += np.bincount(classes - 1, minlength=8)[:8]
                rank_counts += np.bincount(rows["road8_rank"].to_numpy(np.int64) - 1, minlength=TOP_N)[:TOP_N]

        keep = frame[["image_id", "canonical_image_id", "coco_image_id", "candidate_record_id", "road8_rank", "query_index", "predicted_road8_class_id", "score"]].copy()
        keep["image_id"] = keep["image_id"].astype(str)
        keep["canonical_image_id"] = keep["canonical_image_id"].astype(str)
        candidate_parts.append(keep)
        progress(CURRENT_STAGE, f"processed shard {shard_number}/{len(candidate_manifest)} ({shard_id})")

    if row0 != feature_count or len(image_slices) != EXPECTED_IMAGES:
        raise ValueError(f"FEATURE_COVERAGE_FAIL:{row0}:{len(image_slices)}")
    candidates = pd.concat(candidate_parts, ignore_index=True)
    if len(candidates) != EXPECTED_IMAGES * TOP_N or candidates["candidate_record_id"].duplicated().any():
        raise ValueError("CANDIDATE_POOL_FAIL")
    if set(candidates["canonical_image_id"].astype(str)) != set(group_map):
        raise ValueError("CANDIDATE_GROUP_ID_SET_FAIL")
    if observed_detector_ids != {EXPECTED_DETECTOR_ID} or observed_checkpoint_shas != {EXPECTED_CHECKPOINT_SHA} or observed_export_config_shas != {EXPECTED_EXPORT_CONFIG_SHA} or observed_schema_shas != {EXPECTED_CANDIDATE_SCHEMA_SHA}:
        raise ValueError(f"CANDIDATE_PROVENANCE_FAIL:{observed_detector_ids}:{observed_checkpoint_shas}:{observed_export_config_shas}:{observed_schema_shas}")
    candidates["group_id"] = candidates["canonical_image_id"].map(group_map).astype(np.int16)
    candidates = candidates.sort_values(["canonical_image_id", "road8_rank"], kind="stable").reset_index(drop=True)
    runtime_rows.append({"stage": "asset_read_and_feature_construction", "method": "M11", "seed": -1, "budget": -1, "seconds": time.perf_counter() - t0, "images": EXPECTED_IMAGES, "groups": EXPECTED_GROUPS, "notes": "Top100 candidate/native gather plus frozen 90D/PCA/scaler"})
    progress(CURRENT_STAGE, "PASS; constructed 225,000 frozen 90D inputs")

    CURRENT_STAGE = "distribution_shift_diagnostic"
    t0 = time.perf_counter()
    score_values = np.concatenate(score_parts)
    pca_norm_values = np.concatenate(pca_norm_parts)
    embedding_norm_values = np.concatenate(embedding_norm_parts)
    geometry_values = {name: np.concatenate(parts) for name, parts in geometry_parts.items()}
    numeric_arrays = [features, score_values, pca_norm_values, embedding_norm_values, *geometry_values.values()]
    numeric_nan_count += sum(int(np.isnan(np.asarray(array)).sum()) for array in numeric_arrays)
    numeric_inf_count += sum(int(np.isinf(np.asarray(array)).sum()) for array in numeric_arrays)
    schema_columns = read_json(p1_root / "feature_schema.json")["columns"]
    exceed_ratios = feature_exceed_counts.astype(np.float64) / float(feature_count)
    top_indices = np.argsort(-exceed_ratios, kind="stable")[:10]
    diagnostics: dict[str, Any] = {
        "decision_effect": False,
        "candidate_pool_rows": int(len(candidates)),
        "model_feature_rows": int(feature_count),
        "fit_reference_rows": int(fit_rows),
        "feature_cells_outside_fit_minmax_ratio": float(feature_exceed_counts.sum() / (feature_count * 90)),
        "feature_rows_with_any_outside_ratio": float(feature_rows_any_exceed / feature_count),
        "feature_dimensions_with_any_outside": int(np.count_nonzero(feature_exceed_counts)),
        "feature_exceed_ratio_by_column": [
            {"index": int(index), "name": str(schema_columns[index]["name"]), "exceed_ratio": float(exceed_ratios[index])}
            for index in range(90)
        ],
        "top_feature_exceed_ratios": [
            {"index": int(index), "name": str(schema_columns[index]["name"]), "exceed_ratio": float(exceed_ratios[index])}
            for index in top_indices
        ],
        "standardized_abs_quantiles": quantiles(np.abs(features), (.5, .9, .95, .99, 1.0)),
        "pca_norm_quantiles": quantiles(pca_norm_values),
        "embedding_norm_quantiles": quantiles(embedding_norm_values),
        "raw_score_quantiles": quantiles(score_values),
        "predicted_class_proportions": {ROAD8[index]: float(class_counts[index] / class_counts.sum()) for index in range(8)},
        "rank_distribution": {"min": 1, "max": TOP_N, "count_per_rank": int(rank_counts[0]), "all_ranks_equal_count": bool(np.all(rank_counts == EXPECTED_IMAGES))},
        "geometry_summary": {name: quantiles(values) for name, values in geometry_values.items()},
        "nan_count": int(numeric_nan_count),
        "inf_count": int(numeric_inf_count),
        "score_reconstruction_max_abs_error": score_reconstruction_max_abs_error,
    }
    if numeric_nan_count or numeric_inf_count or not diagnostics["rank_distribution"]["all_ranks_equal_count"]:
        raise ValueError("DISTRIBUTION_DIAGNOSTIC_NUMERICAL_OR_RANK_FAIL")
    write_text_atomic(staging / "distribution_shift_report.md", build_distribution_report(diagnostics))
    write_json_atomic(working / "distribution_shift_diagnostics.json", diagnostics)
    runtime_rows.append({"stage": "distribution_shift_diagnostic", "method": "DIAGNOSTIC", "seed": -1, "budget": -1, "seconds": time.perf_counter() - t0, "images": EXPECTED_IMAGES, "groups": EXPECTED_GROUPS, "notes": "descriptive only; no decision effect"})
    progress(CURRENT_STAGE, "diagnostics saved; no decision parameters changed")

    CURRENT_STAGE = "frozen_model_inference"
    t0 = time.perf_counter()
    models = load_models(p1_root, train_models, device)
    torch.cuda.synchronize()
    model_load_seconds = time.perf_counter() - t0
    runtime_rows.append({"stage": "model_load", "method": "M11", "seed": -1, "budget": -1, "seconds": model_load_seconds, "images": 0, "groups": 0, "notes": "three trusted weights_only snapshots; eval/inference only"})
    tensor = torch.from_numpy(features).to(device)
    probabilities: dict[int, np.ndarray] = {}
    prediction_content_sha256: dict[str, str] = {}
    inference_repeat_exact: dict[str, bool] = {}
    for seed in SEEDS:
        torch.cuda.synchronize()
        seed_started = time.perf_counter()
        parts: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(tensor), 16384):
                parts.append(models[seed](tensor[start:start + 16384]).float().cpu().numpy())
        torch.cuda.synchronize()
        logits = np.vstack(parts).astype(np.float64)
        repeat_count = min(16384, len(tensor))
        with torch.inference_mode():
            # Repeat the identical first production batch shape.  Changing the
            # GEMM row dimension is not a determinism test because kernels may
            # use a different numerically equivalent reduction path.
            repeat_logits = models[seed](tensor[:repeat_count]).float().cpu().numpy().astype(np.float64)
        torch.cuda.synchronize()
        inference_repeat_exact[str(seed)] = bool(np.array_equal(logits[:repeat_count], repeat_logits))
        if not inference_repeat_exact[str(seed)]:
            raise ValueError(f"M11_REPEAT_INFERENCE_MISMATCH:{seed}")
        prob = p1_core.sigmoid(logits / temperatures[seed])
        if prob.shape != (feature_count, 10) or not np.isfinite(prob).all() or np.any((prob < 0) | (prob > 1)):
            raise ValueError(f"M11_PREDICTION_FAIL:{seed}")
        probabilities[seed] = prob
        prediction_content_sha256[str(seed)] = sha256_array(prob)
        runtime_rows.append({"stage": "model_inference_and_temperature", "method": "M11", "seed": seed, "budget": -1, "seconds": time.perf_counter() - seed_started, "images": EXPECTED_IMAGES, "groups": EXPECTED_GROUPS, "notes": "independent seed; probabilities not averaged"})
        progress(CURRENT_STAGE, f"seed {seed} complete; no cross-seed averaging")
    del tensor
    for model in models.values():
        model.to("cpu")
    del models
    torch.cuda.empty_cache()
    run_config["m11"]["prediction_probability_sha256"] = prediction_content_sha256
    run_config["m11"]["repeat_inference_exact_first_production_batch"] = inference_repeat_exact
    write_json_atomic(run_config_path, run_config)

    CURRENT_STAGE = "exact_allocation"
    candidate_by_image = {
        str(image_id): rows.sort_values("road8_rank", kind="stable").reset_index(drop=True)
        for image_id, rows in candidates.groupby("canonical_image_id", sort=False)
    }
    score_margins = {
        image_id: rows.iloc[5:50]["score"].to_numpy(np.float64)
        for image_id, rows in candidate_by_image.items()
    }
    cal_margins: dict[str, np.ndarray] = {}
    cal_temp_clip_low_count = 0
    cal_temp_clip_high_count = 0
    for image_id, scores in score_margins.items():
        cal_temp_clip_low_count += int(np.count_nonzero(scores < 1e-6))
        cal_temp_clip_high_count += int(np.count_nonzero(scores > 1 - 1e-6))
        clipped = np.clip(scores, 1e-6, 1 - 1e-6)
        cal_margins[image_id] = p1_core.sigmoid(np.log(clipped / (1 - clipped)) / CAL_TEMP)
    cal_temp_strict_equivalence_expected = cal_temp_clip_low_count == 0 and cal_temp_clip_high_count == 0
    m11_margins: dict[int, dict[str, np.ndarray]] = {}
    for seed in SEEDS:
        mean_probability = probabilities[seed].mean(axis=1)
        values = mean_probability * class_weights[feature_classes - 1]
        if not np.isfinite(values).all():
            raise ValueError(f"M11_VALUE_NONFINITE:{seed}")
        m11_margins[seed] = {image_id: np.asarray(values[image_slices[image_id]], dtype=np.float64) for image_id in image_slices}

    allocations: list[dict[str, Any]] = []
    allocation_map: dict[tuple[str, int, int, str], int] = {}
    group_objectives: dict[tuple[str, int, int, int], float] = {}
    group_members = {
        int(group_id): sorted(frame["canonical_image_id"].astype(str).tolist())
        for group_id, frame in groups.groupby("group_id", sort=True)
    }
    for group_id, image_ids in group_members.items():
        if len(image_ids) != GROUP_SIZE:
            raise ValueError(f"GROUP_CALLER_SIZE_FAIL:{group_id}")
        for budget in BUDGETS:
            for image_id in image_ids:
                key = ("S_FIXED", BASELINE_SEED, budget, image_id)
                allocation_map[key] = budget
                allocations.append({"method": "S_FIXED", "seed": BASELINE_SEED, "budget": budget, "group_id": group_id, "canonical_image_id": image_id, "K_i": budget, "group_predicted_objective": math.nan})
        policies: list[tuple[str, int, Mapping[str, np.ndarray]]] = [
            ("S_ADAPT", BASELINE_SEED, score_margins),
            ("CAL_TEMP_ALLOC", BASELINE_SEED, cal_margins),
            *[("M11", seed, m11_margins[seed]) for seed in SEEDS],
        ]
        for method, seed, per_image in policies:
            policy_started = time.perf_counter()
            margins = np.vstack([per_image[image_id] for image_id in image_ids]).astype(np.float64)
            solved, objectives = dp_solver.solve_group_allocations(margins, BUDGETS)
            for budget_index, budget in enumerate(BUDGETS):
                objective = float(objectives[budget_index])
                group_objectives[(method, seed, budget, group_id)] = objective
                for image_index, image_id in enumerate(image_ids):
                    k_value = int(solved[budget_index, image_index])
                    allocation_map[(method, seed, budget, image_id)] = k_value
                    allocations.append({"method": method, "seed": seed, "budget": budget, "group_id": group_id, "canonical_image_id": image_id, "K_i": k_value, "group_predicted_objective": objective})
            runtime_rows.append({"stage": "exact_dp_group", "method": method, "seed": seed, "budget": -1, "seconds": time.perf_counter() - policy_started, "images": GROUP_SIZE, "groups": 1, "notes": f"group_id={group_id}; all five budgets in one DP"})
        if (group_id + 1) % 25 == 0 or group_id == 0:
            progress(CURRENT_STAGE, f"completed group {group_id + 1}/{EXPECTED_GROUPS} for all policies and budgets")

    allocation_frame = pd.DataFrame(allocations)
    expected_allocation_rows = len(condition_instances()) * len(BUDGETS) * EXPECTED_IMAGES
    if len(allocation_frame) != expected_allocation_rows:
        raise ValueError(f"ALLOCATION_ROW_COUNT_FAIL:{len(allocation_frame)}:{expected_allocation_rows}")
    if not allocation_frame["K_i"].between(K_MIN, K_MAX).all():
        raise ValueError("ALLOCATION_K_RANGE_FAIL")
    sums = allocation_frame.groupby(["method", "seed", "budget", "group_id"], sort=True)["K_i"].sum()
    expected_sums = sums.index.get_level_values("budget").to_numpy(np.int64) * GROUP_SIZE
    if not np.array_equal(sums.to_numpy(np.int64), expected_sums):
        raise ValueError("ALLOCATION_EXACT_GROUP_BUDGET_FAIL")
    fixed = allocation_frame[allocation_frame["method"] == "S_FIXED"]
    if not np.array_equal(fixed["K_i"].to_numpy(np.int64), fixed["budget"].to_numpy(np.int64)):
        raise ValueError("S_FIXED_FAIL")
    adapt_k = {(int(row.budget), str(row.canonical_image_id)): int(row.K_i) for row in allocation_frame[allocation_frame["method"] == "S_ADAPT"].itertuples(index=False)}
    cal_k = {(int(row.budget), str(row.canonical_image_id)): int(row.K_i) for row in allocation_frame[allocation_frame["method"] == "CAL_TEMP_ALLOC"].itertuples(index=False)}
    cal_temp_k_mismatches = sum(adapt_k[key] != cal_k[key] for key in adapt_k)

    CURRENT_STAGE = "selection_materialization"
    selected_path = staging / "selected_records.parquet"
    writer: pq.ParquetWriter | None = None
    selection_manifest_rows: list[dict[str, Any]] = []
    allocation_summary_rows: list[dict[str, Any]] = []
    selected_total = 0
    retained_same_query_multiclass_rows = 0
    condition_order = condition_instances()
    materialize_started = time.perf_counter()
    try:
        for method, seed, value_semantics in condition_order:
            instance = policy_id(method, seed)
            for budget in BUDGETS:
                k_by_image = {
                    image_id: allocation_map[(method, seed, budget, image_id)]
                    for image_id in candidate_by_image
                }
                k_values = candidates["canonical_image_id"].map(k_by_image).to_numpy(np.int16)
                selected = candidates[candidates["road8_rank"].to_numpy(np.int16) <= k_values].copy()
                selected["K_i"] = selected["canonical_image_id"].map(k_by_image).astype(np.int16)
                selected.insert(0, "selection_rank", selected["road8_rank"].astype(np.int16))
                selected.insert(0, "budget", np.int16(budget))
                selected.insert(0, "value_semantics", value_semantics)
                selected.insert(0, "policy_instance_id", instance)
                selected.insert(0, "seed", np.int64(seed))
                selected.insert(0, "method", method)
                selected.insert(0, "candidate_asset_id", CANDIDATE_ASSET_ID)
                selected.insert(0, "split", SPLIT)
                selected.insert(0, "dataset_version", DATASET_VERSION)
                selected = selected.sort_values(["group_id", "canonical_image_id", "road8_rank"], kind="stable").reset_index(drop=True)
                expected_rows = budget * EXPECTED_IMAGES
                if len(selected) != expected_rows:
                    raise ValueError(f"SELECTION_CONDITION_ROW_COUNT_FAIL:{instance}:{budget}:{len(selected)}")
                duplicate_query_mask = selected.duplicated(["canonical_image_id", "query_index"], keep=False)
                retained_same_query_multiclass_rows += int(duplicate_query_mask.sum())
                output = selected[[field.name for field in SELECTED_SCHEMA]].copy()
                table = pa.Table.from_pandas(output, schema=SELECTED_SCHEMA, preserve_index=False, safe=True)
                if writer is None:
                    writer = pq.ParquetWriter(selected_path, SELECTED_SCHEMA, compression="zstd", use_dictionary=True)
                writer.write_table(table, row_group_size=len(table))
                condition_sha = content_sequence_sha(output)
                selected_total += len(output)
                k_condition = allocation_frame[(allocation_frame["method"] == method) & (allocation_frame["seed"] == seed) & (allocation_frame["budget"] == budget)]
                selection_manifest_rows.append({
                    "dataset": DATASET_VERSION,
                    "split": SPLIT,
                    "candidate_asset_id": CANDIDATE_ASSET_ID,
                    "method": method,
                    "seed": seed,
                    "policy_instance_id": instance,
                    "value_semantics": value_semantics,
                    "budget": budget,
                    "group_count": EXPECTED_GROUPS,
                    "image_count": EXPECTED_IMAGES,
                    "allocation_rows": EXPECTED_IMAGES,
                    "selected_rows": len(output),
                    "condition_content_sha256": condition_sha,
                    "status": "FROZEN_BEFORE_GT_EVALUATION",
                    "gt_path_available_to_selection_entrypoint": False,
                })
                allocation_summary_rows.append({
                    "dataset": DATASET_VERSION,
                    "method": method,
                    "seed": seed,
                    "budget": budget,
                    "group_count": EXPECTED_GROUPS,
                    "mean_K": float(k_condition["K_i"].mean()),
                    "total_records": int(k_condition["K_i"].sum()),
                    "image_count": EXPECTED_IMAGES,
                    "min_K": int(k_condition["K_i"].min()),
                    "max_K": int(k_condition["K_i"].max()),
                    "images_at_K_min": int((k_condition["K_i"] == K_MIN).sum()),
                    "images_at_K_max": int((k_condition["K_i"] == K_MAX).sum()),
                })
                progress(CURRENT_STAGE, f"wrote {instance} budget={budget} ({len(output):,} rows)")
    finally:
        if writer is not None:
            writer.close()
    expected_selected_rows = EXPECTED_IMAGES * sum(BUDGETS) * len(condition_instances())
    if selected_total != expected_selected_rows:
        raise ValueError(f"SELECTION_TOTAL_ROW_COUNT_FAIL:{selected_total}:{expected_selected_rows}")
    selected_sha = sha256_file(selected_path)
    for row in selection_manifest_rows:
        row["selected_records_path"] = "selected_records.parquet"
        row["selected_records_sha256"] = selected_sha
    runtime_rows.append({"stage": "selection_materialization", "method": "ALL", "seed": -1, "budget": -1, "seconds": time.perf_counter() - materialize_started, "images": EXPECTED_IMAGES, "groups": EXPECTED_GROUPS, "notes": f"{selected_total} selected reference rows; one parquet row group per condition"})

    CURRENT_STAGE = "selection_qa"
    qa_started = time.perf_counter()
    reference = candidates.set_index("candidate_record_id", drop=False)
    prefix_failures = 0
    identity_failures = 0
    duplicate_failures = 0
    serialized_image_count_failures = 0
    serialized_k_consistency_failures = 0
    serialized_group_id_failures = 0
    serialized_group_budget_failures = 0
    selection_nan = 0
    selection_inf = 0
    condition_seen: set[tuple[str, int, int]] = set()
    parquet_file = pq.ParquetFile(selected_path)
    try:
        if parquet_file.metadata.num_rows != selected_total or parquet_file.num_row_groups != len(selection_manifest_rows):
            raise ValueError("SELECTION_PARQUET_METADATA_FAIL")
        cal_selection_ids: dict[int, list[str]] = {}
        adapt_selection_ids: dict[int, list[str]] = {}
        for row_group_index in range(parquet_file.num_row_groups):
            frame = parquet_file.read_row_group(row_group_index).to_pandas()
            method_values = frame["method"].unique()
            seed_values = frame["seed"].unique()
            budget_values = frame["budget"].unique()
            if len(method_values) != 1 or len(seed_values) != 1 or len(budget_values) != 1:
                raise ValueError(f"SELECTION_ROW_GROUP_CONDITION_FAIL:{row_group_index}")
            condition = (str(method_values[0]), int(seed_values[0]), int(budget_values[0]))
            if condition in condition_seen:
                raise ValueError(f"SELECTION_DUPLICATE_CONDITION:{condition}")
            condition_seen.add(condition)
            duplicate_failures += int(frame.duplicated(["canonical_image_id", "candidate_record_id"]).sum())
            selection_nan += int(frame.select_dtypes(include=[np.number]).isna().sum().sum())
            numeric = frame.select_dtypes(include=[np.number]).to_numpy(np.float64)
            selection_inf += int(np.isinf(numeric).sum())
            per_image = frame.groupby("canonical_image_id", sort=False).agg(
                K_i=("K_i", "first"),
                group_id=("group_id", "first"),
                rows=("candidate_record_id", "size"),
                k_unique=("K_i", "nunique"),
                group_unique=("group_id", "nunique"),
            )
            if len(per_image) != EXPECTED_IMAGES:
                serialized_image_count_failures += abs(len(per_image) - EXPECTED_IMAGES)
            serialized_k_consistency_failures += int(((per_image["rows"] != per_image["K_i"]) | (per_image["k_unique"] != 1)).sum())
            expected_serialized_group = per_image.index.to_series().map(group_map).to_numpy(np.int64)
            serialized_group_id_failures += int(np.count_nonzero(per_image["group_id"].to_numpy(np.int64) != expected_serialized_group))
            serialized_group_id_failures += int((per_image["group_unique"] != 1).sum())
            serialized_group_sums = per_image.groupby("group_id", sort=True)["K_i"].sum()
            if len(serialized_group_sums) != EXPECTED_GROUPS:
                serialized_group_budget_failures += abs(len(serialized_group_sums) - EXPECTED_GROUPS)
            serialized_group_budget_failures += int(np.count_nonzero(serialized_group_sums.to_numpy(np.int64) != int(condition[2]) * GROUP_SIZE))
            for image_id, image_rows in frame.groupby("canonical_image_id", sort=False):
                image_rows = image_rows.sort_values("road8_rank", kind="stable")
                k = int(image_rows["K_i"].iloc[0])
                ranks = image_rows["road8_rank"].astype(int).tolist()
                if len(image_rows) != k or ranks != list(range(1, k + 1)) or image_rows["selection_rank"].astype(int).tolist() != ranks:
                    prefix_failures += 1
                record_ids = image_rows["candidate_record_id"].astype(str).tolist()
                if any(record_id not in reference.index for record_id in record_ids):
                    identity_failures += 1
                    continue
                source = reference.loc[record_ids]
                if isinstance(source, pd.Series):
                    source = source.to_frame().T
                if not (source["canonical_image_id"].astype(str).to_numpy() == str(image_id)).all():
                    identity_failures += 1
                if not (image_rows["image_id"].astype(str).to_numpy() == str(image_id)).all():
                    identity_failures += 1
                if not np.array_equal(source["road8_rank"].to_numpy(np.int64), image_rows["road8_rank"].to_numpy(np.int64)):
                    identity_failures += 1
                if not np.array_equal(source["query_index"].to_numpy(np.int64), image_rows["query_index"].to_numpy(np.int64)):
                    identity_failures += 1
                if not np.array_equal(source["predicted_road8_class_id"].to_numpy(np.int64), image_rows["predicted_road8_class_id"].to_numpy(np.int64)):
                    identity_failures += 1
                if not np.array_equal(source["score"].to_numpy(np.float64), image_rows["score"].to_numpy(np.float64)):
                    identity_failures += 1
            if condition[0] == "S_ADAPT":
                adapt_selection_ids[condition[2]] = frame["candidate_record_id"].astype(str).tolist()
            elif condition[0] == "CAL_TEMP_ALLOC":
                cal_selection_ids[condition[2]] = frame["candidate_record_id"].astype(str).tolist()
    finally:
        parquet_file.close()
    if condition_seen != {(method, seed, budget) for method, seed, _ in condition_instances() for budget in BUDGETS}:
        raise ValueError("SELECTION_CONDITION_SET_FAIL")
    cal_selection_symmetric_difference = 0
    cal_selection_order_mismatch_budgets = 0
    for budget in BUDGETS:
        if adapt_selection_ids[budget] != cal_selection_ids[budget]:
            cal_selection_order_mismatch_budgets += 1
            cal_selection_symmetric_difference += len(set(adapt_selection_ids[budget]).symmetric_difference(set(cal_selection_ids[budget])))
    cal_temp_equivalence_violation = int(
        cal_temp_strict_equivalence_expected
        and (cal_temp_k_mismatches != 0 or cal_selection_order_mismatch_budgets != 0)
    )

    qa = {
        "images": EXPECTED_IMAGES,
        "groups": EXPECTED_GROUPS,
        "conditions": len(selection_manifest_rows),
        "allocation_rows": len(allocation_frame),
        "selected_rows": selected_total,
        "k_range_failures": int((~allocation_frame["K_i"].between(K_MIN, K_MAX)).sum()),
        "group_budget_failures": int(np.count_nonzero(sums.to_numpy(np.int64) != expected_sums)),
        "prefix_failures": prefix_failures,
        "candidate_identity_failures": identity_failures,
        "duplicate_selection_failures": duplicate_failures,
        "serialized_image_count_failures": serialized_image_count_failures,
        "serialized_k_consistency_failures": serialized_k_consistency_failures,
        "serialized_group_id_failures": serialized_group_id_failures,
        "serialized_group_budget_failures": serialized_group_budget_failures,
        "nan_count": numeric_nan_count + selection_nan,
        "inf_count": numeric_inf_count + selection_inf,
        "cal_temp_k_mismatches": cal_temp_k_mismatches,
        "cal_temp_selection_symmetric_difference": cal_selection_symmetric_difference,
        "cal_temp_selection_order_mismatch_budgets": cal_selection_order_mismatch_budgets,
        "cal_temp_clip_low_count": cal_temp_clip_low_count,
        "cal_temp_clip_high_count": cal_temp_clip_high_count,
        "cal_temp_strict_equivalence_expected": cal_temp_strict_equivalence_expected,
        "cal_temp_equivalence_violation": cal_temp_equivalence_violation,
        "retained_same_query_multiclass_rows": retained_same_query_multiclass_rows,
        "annotation_or_gt_read": False,
        "scientific_evaluation_performed": False,
        "all_checks_pass": False,
    }
    hard_failures = [qa[key] for key in (
        "k_range_failures", "group_budget_failures", "prefix_failures",
        "candidate_identity_failures", "duplicate_selection_failures",
        "serialized_image_count_failures", "serialized_k_consistency_failures",
        "serialized_group_id_failures", "serialized_group_budget_failures",
        "nan_count", "inf_count", "cal_temp_equivalence_violation",
    )]
    if any(int(value) != 0 for value in hard_failures):
        raise ValueError(f"SELECTION_QA_FAIL:{qa}")
    qa["all_checks_pass"] = True
    write_json_atomic(working / "selection_qa.json", qa)
    write_text_atomic(staging / "selection_QA_report.md", build_qa_report(qa))
    runtime_rows.append({"stage": "selection_qa", "method": "ALL", "seed": -1, "budget": -1, "seconds": time.perf_counter() - qa_started, "images": EXPECTED_IMAGES, "groups": EXPECTED_GROUPS, "notes": "serialized selection identity/prefix/budget QA; no GT"})
    progress(CURRENT_STAGE, "PASS; prefix, identity, exact-budget, and serialization checks complete")

    CURRENT_STAGE = "final_metadata"
    selection_manifest_fields = [
        "dataset", "split", "candidate_asset_id", "method", "seed", "policy_instance_id",
        "value_semantics", "budget", "group_count", "image_count", "allocation_rows",
        "selected_rows", "condition_content_sha256", "selected_records_path",
        "selected_records_sha256", "status", "gt_path_available_to_selection_entrypoint",
    ]
    allocation_summary_fields = [
        "dataset", "method", "seed", "budget", "group_count", "mean_K", "total_records",
        "image_count", "min_K", "max_K", "images_at_K_min", "images_at_K_max",
    ]
    write_csv_atomic(staging / "selection_manifest.csv", selection_manifest_rows, selection_manifest_fields)
    write_csv_atomic(staging / "allocation_summary.csv", allocation_summary_rows, allocation_summary_fields)
    runtime_rows.append({"stage": "total_before_publication", "method": "ALL", "seed": -1, "budget": -1, "seconds": time.perf_counter() - started_total, "images": EXPECTED_IMAGES, "groups": EXPECTED_GROUPS, "notes": "includes input hashing/read/features/inference/DP/materialization/QA; excludes GT/evaluation"})
    runtime_fields = ["stage", "method", "seed", "budget", "seconds", "images", "groups", "notes"]
    write_csv_atomic(staging / "runtime_summary.csv", runtime_rows, runtime_fields)

    output_files = [
        "run_config.json", "selection_manifest.csv", "selected_records.parquet",
        "allocation_summary.csv", "distribution_shift_report.md", "selection_QA_report.md",
        "runtime_summary.csv",
    ]
    ledger_rows: list[dict[str, Any]] = []
    for name, binding in sorted(bindings.items()):
        ledger_rows.append({"role": "frozen_input:" + name, "path": binding["path"], "bytes": binding["bytes"], "sha256": binding["sha256"]})
    for name in output_files:
        path = staging / name
        ledger_rows.append({"role": "selection_output", "path": name, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    write_csv_atomic(staging / "sha256_ledger.csv", ledger_rows, ["role", "path", "bytes", "sha256"])

    CURRENT_STAGE = "atomic_publication"
    progress(CURRENT_STAGE, "publishing validated outputs and verifying the SHA ledger")
    for name in output_files + ["sha256_ledger.csv"]:
        destination = output_root / name
        if destination.exists():
            raise FileExistsError(f"REFUSE_OUTPUT_OVERWRITE:{destination}")
        os.replace(staging / name, destination)

    mismatches: list[str] = []
    for row in read_csv_rows(output_root / "sha256_ledger.csv"):
        listed = Path(row["path"])
        path = listed if listed.is_absolute() else output_root / listed
        if not path.is_file() or path.stat().st_size != int(row["bytes"]) or sha256_file(path) != row["sha256"]:
            mismatches.append(row["path"])
    if mismatches:
        raise ValueError(f"POST_PUBLICATION_LEDGER_MISMATCH:{mismatches}")
    completion = {
        "contract_type": "COCO2017_ROUTE_B_SELECTION_COMPLETION_V1",
        "status": "COMPLETE",
        "execution_status": "COMPLETE",
        "selection_status": "COMPLETE",
        "gt_access_status": "NOT_ACCESSED",
        "qa_status": "PASS",
        "dataset_version": DATASET_VERSION,
        "split": SPLIT,
        "candidate_asset_id": CANDIDATE_ASSET_ID,
        "conditions": len(selection_manifest_rows),
        "selected_rows": selected_total,
        "m11_seeds": list(SEEDS),
        "seed_prediction_averaging": False,
        "sha256_ledger": {"path": "sha256_ledger.csv", "sha256": sha256_file(output_root / "sha256_ledger.csv"), "entries": len(ledger_rows), "mismatches": 0},
        "selected_records_sha256": selected_sha,
        "published_at_utc": utc_now(),
        "publication_rule": "This marker is the final logical commit; absence means selections are unpublished.",
    }
    write_json_atomic(output_root / "SELECTION_COMPLETE.json", completion)
    progress(CURRENT_STAGE, "COMPLETE; final marker written")
    return completion


def write_failure(output_root: Path, stage: str, error: BaseException) -> None:
    working = output_root / "working"
    working.mkdir(parents=True, exist_ok=True)
    write_json_atomic(working / "failure_detail.json", {
        "execution_status": "BLOCKED",
        "selection_status": "NOT_PUBLISHED",
        "gt_access_status": "NOT_ACCESSED",
        "stage": stage,
        "error_type": type(error).__name__,
        "error": str(error),
        "traceback": traceback.format_exc(),
        "timestamp_utc": utc_now(),
    })


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GT-free frozen M11 Route-B selection on COCO2017 VAL2017")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT_DEFAULT)
    parser.add_argument("--asset-root", type=Path, default=ASSET_ROOT_DEFAULT)
    parser.add_argument("--group-manifest", type=Path, default=GROUP_MANIFEST_DEFAULT)
    parser.add_argument("--p1-root", type=Path, default=P1_ROOT_DEFAULT)
    parser.add_argument("--p1a-root", type=Path, default=P1A_ROOT_DEFAULT)
    parser.add_argument("--p2-root", type=Path, default=P2_ROOT_DEFAULT)
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    try:
        result = execute(parsed)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    except BaseException as exc:
        write_failure(parsed.output_root.resolve(), CURRENT_STAGE, exc)
        raise
