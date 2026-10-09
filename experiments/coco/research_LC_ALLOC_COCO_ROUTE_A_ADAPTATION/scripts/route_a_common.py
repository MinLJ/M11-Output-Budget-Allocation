"""Shared, detector-free contracts for COCO Route-A allocator training.

This module reads already materialized canonical candidate/native assets.  It
contains no detector import or forward path and never reads VAL annotations.
The frozen LC P1 feature/matching implementation is imported by exact hash so
the 90D feature and marginal-label definitions do not drift.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
REPO_ROOT = PROJECT_ROOT.parent
DEFAULT_CONFIG = PROJECT_ROOT / "working" / "route_a_pipeline_config.json"

ROAD8 = ("person", "bicycle", "car", "motorcycle", "bus", "train", "truck", "traffic light")
COCO_CATEGORY_IDS = (1, 2, 3, 4, 6, 7, 8, 10)
COCO_TO_ROAD8 = dict(zip(COCO_CATEGORY_IDS, range(1, 9)))
EXPECTED_TRAIN_ASSET_ID = "964077b8b128c693d1bffeb627609d5a3b1da50b88e0bdcf0d6d469947f84b99"
EXPECTED_TRAIN_IMAGES = 118_287
EXPECTED_TRAIN_GT_SHA256 = "610fce4944abdeb15354cc765333805529359d12d88f2f711393ca586901d01d"
EXPECTED_CANDIDATE_SCHEMA_SHA256 = "e977e02fe638a8ac44e98919d76225306eb88fa7385d3ec0793342112ec8c53b"
EXPECTED_EXPORT_CONFIG_SHA256 = "f6a91e9d6879357fc10095c182689e43ee73bca2c10ec6926f93960cc68d98f2"
TOP_N = 100
QUERY_COUNT = 300
K_MIN = 5
K_MAX = 50
MODEL_RANKS = tuple(range(6, 51))
# Preserve the historical P1 construction exactly.  JSON round-tripping may spell
# the same decimal grid with the adjacent binary float (for example 0.85 versus
# 0.8500000000000001), so config validation below uses a tight numeric check.
THRESHOLDS = np.asarray([0.50 + 0.05 * index for index in range(10)], dtype=np.float64)
SEEDS = (630101, 630102, 630103)
ROLE_ORDER = ("FIT", "EARLY_STOP", "CALIBRATION")
ROLE_TO_CODE = {"FIT": 0, "EARLY_STOP": 1, "CALIBRATION": 2}

CANDIDATE_COLUMNS = [
    "dataset_version", "split", "image_id", "canonical_image_id", "coco_image_id",
    "image_sha256", "detector_id", "checkpoint_sha256", "export_config_sha256",
    "schema_sha256", "candidate_asset_id", "candidate_record_id", "road8_rank",
    "predicted_road8_class_id", "predicted_road8_class_name", "score",
    "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "bbox_cx", "bbox_cy",
    "bbox_w", "bbox_h", "query_index", "source_order",
]


def sha256_file(path: os.PathLike[str] | str, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            block = stream.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def write_bytes_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def write_json_atomic(path: Path, value: Any) -> None:
    write_bytes_atomic(path, canonical_json_bytes(value))


def write_dataframe_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    os.close(handle)
    try:
        frame.to_csv(temporary, index=False, lineterminator="\n")
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    validate_config(config)
    return config


def hamilton_counts(total: int, ratios: Mapping[str, float], role_order: Sequence[str]) -> dict[str, int]:
    if total <= 0:
        raise ValueError("split total must be positive")
    if tuple(role_order) != ROLE_ORDER:
        raise ValueError(f"role order must remain {ROLE_ORDER}")
    if set(ratios) != set(role_order) or not math.isclose(sum(float(ratios[x]) for x in role_order), 1.0, abs_tol=1e-12):
        raise ValueError("split ratios must cover the three roles and sum to one")
    ideal = {role: total * float(ratios[role]) for role in role_order}
    counts = {role: int(math.floor(ideal[role])) for role in role_order}
    residual = total - sum(counts.values())
    priority = sorted(role_order, key=lambda role: (-(ideal[role] - counts[role]), role_order.index(role)))
    for role in priority[:residual]:
        counts[role] += 1
    if sum(counts.values()) != total:
        raise AssertionError("Hamilton rounding did not preserve the image count")
    return counts


def validate_config(config: Mapping[str, Any]) -> None:
    if config.get("task") != "LC-ALLOC-COCO-ROUTE-A-ADAPTATION" or config.get("route") != "A_COCO_ADAPTED_M11":
        raise ValueError("wrong Route-A task identity")
    if config.get("train_split") != "TRAIN2017" or config.get("validation_split") != "VAL2017":
        raise ValueError("COCO split identity changed")
    if config.get("train_ground_truth", {}).get("sha256") != EXPECTED_TRAIN_GT_SHA256:
        raise ValueError("TRAIN2017 annotation identity changed")
    candidate = config["candidate_protocol"]
    if candidate["expected_train_candidate_asset_id"] != EXPECTED_TRAIN_ASSET_ID or int(candidate["expected_train_images"]) != EXPECTED_TRAIN_IMAGES:
        raise ValueError("TRAIN2017 CandidateAssetID/count changed")
    if any(bool(candidate[key]) for key in ("query_deduplication", "rerank", "nms")):
        raise ValueError("candidate protocol mutation is forbidden")
    split = config["role_split"]
    ratios = {key: float(value) for key, value in split["ratios"].items()}
    counts = hamilton_counts(EXPECTED_TRAIN_IMAGES, ratios, tuple(split["role_order"]))
    if counts != {key: int(value) for key, value in split["expected_counts"].items()}:
        raise ValueError(f"frozen split counts do not match Hamilton rule: {counts}")
    if not str(split["namespace"]).startswith("LC_ALLOC_COCO_ROUTE_A_"):
        raise ValueError("Route-A needs an independent split namespace")
    labels = config["labels"]
    configured_thresholds = np.asarray(labels["iou_thresholds"], np.float64)
    if (
        labels["ranks"] != [6, 50]
        or configured_thresholds.shape != THRESHOLDS.shape
        or not np.allclose(configured_thresholds, THRESHOLDS, rtol=0.0, atol=1e-12)
    ):
        raise ValueError("label rank/IoU protocol changed")
    if bool(labels.get("candidate_matchability_target", True)):
        raise ValueError("candidate matchability is not the M11 target")
    features = config["features"]
    if int(features["dimension"]) != 90 or int(features["pca"]["components"]) != 32:
        raise ValueError("M11 feature dimension/PCA changed")
    if config["class_weights"]["strategy"] != "KEEP_LC_CLASS_WEIGHTS":
        raise ValueError("this frozen Route-A pipeline retains LC class weights")
    model = config["model"]
    exact = {
        "optimizer": "AdamW", "learning_rate": 0.001, "weight_decay": 0.0001,
        "batch_size": 4096, "max_epochs": 30, "early_stop_patience": 5,
        "early_stop_min_delta": 0.0, "final_refit": False,
    }
    for key, value in exact.items():
        if model[key] != value:
            raise ValueError(f"frozen model recipe changed: {key}")
    if tuple(int(value) for value in model["seeds"]) != SEEDS:
        raise ValueError("Route-A seeds changed")
    if config["calibration"]["role"] != "CALIBRATION" or bool(config["calibration"]["validation_access"]):
        raise ValueError("temperature calibration boundary changed")
    allocation = config["allocation_contract"]
    if allocation["k_bounds"] != [5, 50] or allocation["budgets"] != [10, 15, 20, 30, 40] or int(allocation["group_size"]) != 40:
        raise ValueError("allocation protocol changed")


def resolve_reference(config: Mapping[str, Any], key: str, sha_key: str) -> Path:
    path = (REPO_ROOT / str(config["references"][key])).resolve(strict=True)
    observed = sha256_file(path)
    expected = str(config["references"][sha_key])
    if observed != expected:
        raise ValueError(f"frozen reference hash mismatch for {key}: {observed} != {expected}")
    return path


def import_file(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_p1_core(config: Mapping[str, Any]):
    return import_file(resolve_reference(config, "p1_core", "p1_core_sha256"), "route_a_frozen_p1_core")


def load_dp_solver(config: Mapping[str, Any]):
    return import_file(resolve_reference(config, "dp_solver", "dp_solver_sha256"), "route_a_frozen_dp_solver")


def build_split_manifest(image_ids: Iterable[str], config: Mapping[str, Any]) -> pd.DataFrame:
    ids = [str(value) for value in image_ids]
    if len(ids) != len(set(ids)):
        raise ValueError("image identities must be unique before role assignment")
    namespace = str(config["role_split"]["namespace"])
    hashed = [(image_id, hashlib.sha256((namespace + image_id).encode("utf-8")).hexdigest()) for image_id in ids]
    ordered = sorted(hashed, key=lambda item: (item[1], item[0]))
    counts = {key: int(value) for key, value in config["role_split"]["expected_counts"].items()}
    role_by_id: dict[str, str] = {}
    cursor = 0
    for role in ROLE_ORDER:
        for image_id, _ in ordered[cursor:cursor + counts[role]]:
            role_by_id[image_id] = role
        cursor += counts[role]
    if cursor != len(ids) or len(role_by_id) != len(ids):
        raise AssertionError("role assignment coverage failure")
    result = pd.DataFrame({"image_id": [x[0] for x in hashed], "split": [role_by_id[x[0]] for x in hashed], "hash": [x[1] for x in hashed]})
    observed = result["split"].value_counts().to_dict()
    if observed != counts:
        raise AssertionError(f"role count mismatch: {observed} != {counts}")
    return result.sort_values("image_id", kind="mergesort").reset_index(drop=True)


@dataclass(frozen=True)
class CanonicalBundle:
    image_id: str
    coco_image_id: int
    width: int
    height: int
    candidates: pd.DataFrame
    road8_logits: np.ndarray
    embeddings: np.ndarray


class CanonicalTrainReader:
    """Read-only adapter for a completed COCO TRAIN2017 canonical asset."""

    def __init__(self, asset_root: Path, verify_shard_hashes: bool = True):
        self.root = asset_root.resolve(strict=True)
        marker_path = self.root / "CANONICAL_EXPORT_COMPLETE.json"
        if not marker_path.is_file():
            raise FileNotFoundError(f"TRAIN canonical completion marker missing: {marker_path}")
        self.marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if (
            self.marker.get("contract_type") != "COCO2017_TRAIN2017_CANONICAL_EXPORT_COMPLETION_V1"
            or self.marker.get("status") != "COCO_CANONICAL_EXPORT_COMPLETE"
            or self.marker.get("split") != "TRAIN2017"
            or self.marker.get("qa_all_checks_pass") is not True
        ):
            raise ValueError("canonical marker is not a completed TRAIN2017 export")
        if self.marker.get("candidate_asset_id") != EXPECTED_TRAIN_ASSET_ID or int(self.marker.get("image_count", -1)) != EXPECTED_TRAIN_IMAGES:
            raise ValueError("TRAIN canonical marker identity/count mismatch")
        manifest_hashes = self.marker.get("manifest_sha256")
        required_manifest_names = (
            "candidate_manifest.csv", "native_manifest.csv", "image_manifest.csv", "shard_manifest.csv",
        )
        if not isinstance(manifest_hashes, dict) or set(manifest_hashes) != set(required_manifest_names):
            raise ValueError("TRAIN completion marker manifest binding is incomplete")
        for name in required_manifest_names:
            path = self.root / name
            if not path.is_file() or sha256_file(path) != str(manifest_hashes[name]):
                raise ValueError(f"TRAIN completion marker manifest SHA mismatch: {name}")
        ledger_binding = self.marker.get("sha256_ledger")
        ledger_path = self.root / "sha256_ledger.csv"
        if (
            not isinstance(ledger_binding, dict)
            or int(ledger_binding.get("mismatches", -1)) != 0
            or not ledger_path.is_file()
            or sha256_file(ledger_path) != str(ledger_binding.get("sha256"))
        ):
            raise ValueError("TRAIN canonical ledger binding mismatch")
        asset_manifest_path = self.root / "manifest" / "asset_manifest.json"
        if not asset_manifest_path.is_file() or sha256_file(asset_manifest_path) != str(self.marker.get("asset_manifest_sha256")):
            raise ValueError("TRAIN canonical asset manifest binding mismatch")
        asset_manifest = json.loads(asset_manifest_path.read_text(encoding="utf-8"))
        bindings = asset_manifest.get("input_bindings", {})
        if (
            asset_manifest.get("candidate_asset_id") != EXPECTED_TRAIN_ASSET_ID
            or asset_manifest.get("split") != "TRAIN2017"
            or asset_manifest.get("qa_validation_result", {}).get("all_checks_pass") is not True
            or bindings.get("candidate_schema", {}).get("sha256") != EXPECTED_CANDIDATE_SCHEMA_SHA256
            or bindings.get("export_config", {}).get("sha256") != EXPECTED_EXPORT_CONFIG_SHA256
        ):
            raise ValueError("TRAIN canonical schema/export/QA binding mismatch")
        self.candidate_manifest = pd.read_csv(self.root / "candidate_manifest.csv", dtype={"shard_id": str, "path": str, "sha256": str})
        self.native_manifest = pd.read_csv(self.root / "native_manifest.csv", dtype={"shard_id": str, "path": str, "sha256": str})
        self.image_manifest = pd.read_csv(self.root / "image_manifest.csv", dtype={"canonical_image_id": str, "image_id": str})
        required_images = {"manifest_index", "coco_image_id", "canonical_image_id", "width", "height", "candidate_asset_id", "split"}
        if not required_images.issubset(self.image_manifest.columns):
            raise ValueError(f"TRAIN image manifest columns missing: {sorted(required_images - set(self.image_manifest.columns))}")
        if len(self.image_manifest) != EXPECTED_TRAIN_IMAGES or self.image_manifest["canonical_image_id"].nunique() != EXPECTED_TRAIN_IMAGES:
            raise ValueError("TRAIN image manifest coverage mismatch")
        if set(self.image_manifest["candidate_asset_id"].astype(str)) != {EXPECTED_TRAIN_ASSET_ID} or set(self.image_manifest["split"].astype(str)) != {"TRAIN2017"}:
            raise ValueError("TRAIN image manifest identity mismatch")
        self.image_manifest["canonical_image_id"] = self.image_manifest["canonical_image_id"].astype(str)
        self.image_manifest["coco_image_id"] = self.image_manifest["coco_image_id"].astype(np.int64)
        self.image_manifest["manifest_index"] = self.image_manifest["manifest_index"].astype(np.int64)
        self.image_manifest = self.image_manifest.sort_values("manifest_index", kind="mergesort").reset_index(drop=True)
        self.image_meta = self.image_manifest.set_index("canonical_image_id", drop=False)
        if not np.array_equal(self.image_manifest["manifest_index"].to_numpy(), np.arange(EXPECTED_TRAIN_IMAGES)):
            raise ValueError("TRAIN manifest_index must be exactly 0..118286")
        if set(self.candidate_manifest["shard_id"].astype(str)) != set(self.native_manifest["shard_id"].astype(str)):
            raise ValueError("candidate/native shard identity sets differ")
        if verify_shard_hashes:
            for frame in (self.candidate_manifest, self.native_manifest):
                for row in frame.itertuples(index=False):
                    path = (self.root / str(row.path)).resolve(strict=True)
                    if sha256_file(path) != str(row.sha256):
                        raise ValueError(f"canonical shard SHA mismatch: {path}")

    def iter_candidate_frames(self) -> Iterator[tuple[str, pd.DataFrame]]:
        for row in self.candidate_manifest.sort_values("first_manifest_index", kind="mergesort").itertuples(index=False):
            path = self.root / str(row.path)
            frame = pq.read_table(path, columns=CANDIDATE_COLUMNS, filters=[("road8_rank", "<=", TOP_N)]).to_pandas()
            expected = int(row.image_count) * TOP_N
            if len(frame) != expected:
                raise ValueError(f"Top100 row count mismatch in {row.shard_id}: {len(frame)} != {expected}")
            if set(frame["candidate_asset_id"].astype(str)) != {EXPECTED_TRAIN_ASSET_ID} or set(frame["split"].astype(str)) != {"TRAIN2017"}:
                raise ValueError(f"candidate identity mismatch in {row.shard_id}")
            if frame["candidate_record_id"].astype(str).duplicated().any():
                raise ValueError(f"duplicate candidate record in {row.shard_id}")
            yield str(row.shard_id), frame

    def iter_bundles(self) -> Iterator[CanonicalBundle]:
        native_by_shard = {str(row.shard_id): row for row in self.native_manifest.itertuples(index=False)}
        seen: set[str] = set()
        for shard_id, frame in self.iter_candidate_frames():
            native_row = native_by_shard[shard_id]
            with np.load(self.root / str(native_row.path), allow_pickle=False) as native:
                native_ids = native["image_ids"].astype(str)
                native_query = native["query_index"].astype(np.int32)
                logits_all = native["l3_road8_logits"]
                embeddings_all = native["l3_query_embedding"]
                expected_native = int(native_row.image_count) * QUERY_COUNT
                if logits_all.shape != (expected_native, 8) or embeddings_all.shape != (expected_native, 256):
                    raise ValueError(f"native tensor shape mismatch in {shard_id}")
                starts: dict[str, int] = {}
                for start in range(0, expected_native, QUERY_COUNT):
                    ids = native_ids[start:start + QUERY_COUNT]
                    queries = native_query[start:start + QUERY_COUNT]
                    if len(ids) != QUERY_COUNT or len(set(ids)) != 1 or not np.array_equal(queries, np.arange(QUERY_COUNT, dtype=np.int32)):
                        raise ValueError(f"native 300-query block mismatch in {shard_id}:{start}")
                    starts[str(ids[0])] = start
                for image_id, rows in frame.groupby("canonical_image_id", sort=False):
                    image_id = str(image_id)
                    if image_id in seen:
                        raise ValueError(f"duplicate image across TRAIN shards: {image_id}")
                    seen.add(image_id)
                    rows = rows.sort_values("road8_rank", kind="mergesort").reset_index(drop=True)
                    if len(rows) != TOP_N or rows["road8_rank"].astype(int).tolist() != list(range(1, TOP_N + 1)):
                        raise ValueError(f"Top100 rank invariant failed: {image_id}")
                    if image_id not in starts or image_id not in self.image_meta.index or not (rows["image_id"].astype(str) == image_id).all():
                        raise ValueError(f"candidate/native/image join failed: {image_id}")
                    queries = rows["query_index"].to_numpy(np.int32)
                    if np.any((queries < 0) | (queries >= QUERY_COUNT)):
                        raise ValueError(f"query index outside 0..299: {image_id}")
                    indices = starts[image_id] + queries
                    if not np.all(native_ids[indices] == image_id) or not np.array_equal(native_query[indices], queries):
                        raise ValueError(f"native gather identity failed: {image_id}")
                    classes = rows["predicted_road8_class_id"].to_numpy(np.int16)
                    if np.any((classes < 1) | (classes > 8)):
                        raise ValueError(f"Road8 class id out of range: {image_id}")
                    logits = np.asarray(logits_all[indices], dtype=np.float32)
                    embeddings = np.asarray(embeddings_all[indices], dtype=np.float16)
                    reconstructed = 1.0 / (1.0 + np.exp(-logits[np.arange(TOP_N), classes - 1].astype(np.float64)))
                    max_error = float(np.max(np.abs(reconstructed - rows["score"].to_numpy(np.float64))))
                    if max_error > 1e-6:
                        raise ValueError(f"candidate/native score reconstruction failed: {image_id}, {max_error}")
                    meta = self.image_meta.loc[image_id]
                    if not (rows["coco_image_id"].to_numpy(np.int64) == int(meta["coco_image_id"])).all():
                        raise ValueError(f"COCO numeric id mismatch: {image_id}")
                    yield CanonicalBundle(
                        image_id=image_id,
                        coco_image_id=int(meta["coco_image_id"]),
                        width=int(meta["width"]),
                        height=int(meta["height"]),
                        candidates=rows,
                        road8_logits=logits,
                        embeddings=embeddings,
                    )
        if len(seen) != EXPECTED_TRAIN_IMAGES:
            raise ValueError(f"TRAIN canonical image coverage mismatch: {len(seen)}")


@dataclass(frozen=True)
class GTImage:
    classes: np.ndarray
    boxes: np.ndarray


def load_coco_train_gt(path: Path, image_manifest: pd.DataFrame) -> tuple[dict[str, GTImage], dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    expected_numeric = set(image_manifest["coco_image_id"].astype(int))
    if {int(row["id"]) for row in raw["images"]} != expected_numeric:
        raise ValueError("TRAIN2017 annotation/image manifest identity mismatch")
    categories = {int(row["id"]): str(row["name"]) for row in raw["categories"]}
    if {cid: categories.get(cid) for cid in COCO_CATEGORY_IDS} != dict(zip(COCO_CATEGORY_IDS, ROAD8)):
        raise ValueError("COCO Road8 category mapping mismatch")
    canonical_by_numeric = dict(zip(image_manifest["coco_image_id"].astype(int), image_manifest["canonical_image_id"].astype(str)))
    classes: dict[str, list[int]] = {value: [] for value in canonical_by_numeric.values()}
    boxes: dict[str, list[list[float]]] = {value: [] for value in canonical_by_numeric.values()}
    counters = {"official_road8": 0, "valid_noncrowd": 0, "crowd": 0, "ignore": 0, "invalid": 0}
    for annotation in raw["annotations"]:
        coco_class = int(annotation["category_id"])
        if coco_class not in COCO_TO_ROAD8:
            continue
        counters["official_road8"] += 1
        if int(annotation.get("iscrowd", 0)):
            counters["crowd"] += 1
            continue
        if int(annotation.get("ignore", 0)):
            counters["ignore"] += 1
            continue
        bbox = annotation.get("bbox", [])
        if len(bbox) != 4:
            counters["invalid"] += 1
            continue
        x, y, width, height = [float(value) for value in bbox]
        area = float(annotation.get("area", width * height))
        if not all(math.isfinite(value) for value in (x, y, width, height, area)) or width <= 0 or height <= 0 or area <= 0:
            counters["invalid"] += 1
            continue
        image_id = canonical_by_numeric[int(annotation["image_id"])]
        classes[image_id].append(COCO_TO_ROAD8[coco_class])
        boxes[image_id].append([x, y, x + width, y + height])
        counters["valid_noncrowd"] += 1
    result = {
        image_id: GTImage(
            np.asarray(classes[image_id], dtype=np.int16),
            np.asarray(boxes[image_id], dtype=np.float64).reshape(-1, 4),
        )
        for image_id in classes
    }
    summary = {
        **counters,
        "image_count": len(result),
        "images_with_valid_road8_gt": int(sum(bool(value) for value in classes.values())),
        "empty_valid_road8_gt_images": int(sum(not value for value in classes.values())),
    }
    return result, summary


def build_predicted_curve_rows(image_id: str, seed: int, modeled_deltas: np.ndarray) -> list[dict[str, Any]]:
    """Serialize the frozen k=1..50 curve semantics without inventing k<6 predictions."""
    values = np.asarray(modeled_deltas, dtype=np.float64)
    if values.shape != (45,) or not np.isfinite(values).all():
        raise ValueError("modeled_deltas must contain 45 finite values for ranks 6..50")
    rows: list[dict[str, Any]] = []
    for k in range(1, 5):
        rows.append({"image_id": str(image_id), "seed": int(seed), "k": k, "predicted_delta": None, "predicted_U": None})
    rows.append({"image_id": str(image_id), "seed": int(seed), "k": 5, "predicted_delta": None, "predicted_U": 0.0})
    cumulative = np.cumsum(values, dtype=np.float64)
    for offset, k in enumerate(range(6, 51)):
        rows.append({"image_id": str(image_id), "seed": int(seed), "k": k, "predicted_delta": float(values[offset]), "predicted_U": float(cumulative[offset])})
    return rows


def write_sha256_ledger(root: Path, relative_paths: Iterable[Path], ledger_path: Path | None = None) -> pd.DataFrame:
    records = []
    for relative in sorted({Path(value) for value in relative_paths}, key=lambda value: value.as_posix()):
        path = (root / relative).resolve(strict=True)
        records.append({"path": relative.as_posix(), "sha256": sha256_file(path), "bytes": path.stat().st_size})
    frame = pd.DataFrame(records, columns=["path", "sha256", "bytes"])
    write_dataframe_csv_atomic(frame, ledger_path or (root / "sha256_ledger.csv"))
    return frame
