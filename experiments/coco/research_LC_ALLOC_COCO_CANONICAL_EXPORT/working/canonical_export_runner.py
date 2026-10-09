"""Canonical COCO2017 VAL2017 detector export runner.

This executable implements the exact frozen COCO2017 Road8 export contract.
It accepts only an image-only manifest and detector/export identities.  It has
no annotation, ground-truth, allocator, matching, or evaluator input.  The
runner writes ten image-aligned staging shards, performs a fixed 20-image
repeat-forward determinism check, delegates complete readback validation to an
independent validator, and publishes only after ``all_checks_pass`` is true.

The publication transaction is logically committed by the final atomic
``CANONICAL_EXPORT_COMPLETE.json`` marker.  Consumers must refuse an output
tree without that marker.  Interrupted shard writes are resumable inside the
private staging tree; canonical files are never overwritten.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback
from typing import Any, Iterable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
import torch


OUTPUT_ROOT_DEFAULT = Path(r"D:\AOP_DETR\research_LC_ALLOC_COCO_CANONICAL_EXPORT")
FREEZE_ROOT_DEFAULT = Path(r"D:\AOP_DETR\research_LC_ALLOC_COCO_EXPORT_FREEZE")
DATASET_ROOT_DEFAULT = Path(r"D:\COCO2017")
DETECTOR_SNAPSHOT_DEFAULT = Path(
    r"D:\AOP_DETR\shared_benchmark\AOP_ROAD8_LARGECLEAN_V1"
    r"\P1_SHARED_EXPORT\scripts\detector_snapshot.py"
)
VALIDATOR_DEFAULT = OUTPUT_ROOT_DEFAULT / "working" / "canonical_export_validator.py"
DEPENDENCY_FALLBACK_DEFAULT = Path(r"D:\miniconada\envs\aop_detr\Lib\site-packages")

DATASET_VERSION = "COCO2017-Road8-v1"
SPLIT = "VAL2017"
CANDIDATE_ASSET_ID = "5ff6497669d5b02c888f90451882d3c8e594b74a3aedd136639a9c0c8621c707"
EXPECTED_IMAGE_COUNT = 5000
IMAGES_PER_SHARD = 500
EXPECTED_SHARD_COUNT = 10
QUERY_COUNT = 300
ROAD8_CLASS_COUNT = 8
RAW_HYPOTHESES_PER_IMAGE = QUERY_COUNT * ROAD8_CLASS_COUNT
STORED_CANDIDATES_PER_IMAGE = 300
NATIVE_EMBEDDING_DIM = 256
DETERMINISM_SAMPLE_COUNT = 20
DETERMINISM_NAMESPACE = "LC_ALLOC_COCO_CANONICAL_EXPORT_DETERMINISM_V1|"

SCHEMA_FILE = "COCO2017_Road8_CandidateSchema_v1.json"
EXPORT_CONFIG_FILE = "COCO2017_Road8_RTDETRv2_R18VD_EXPORT_CONFIG.json"
IDENTITY_FILE = "COCO2017_candidate_asset_identity.json"
RUNNER_CONTRACT_FILE = "export_runner_contract.md"
QA_SPEC_FILE = "coco_export_QA_spec.md"
SCHEMA_SHA = "e977e02fe638a8ac44e98919d76225306eb88fa7385d3ec0793342112ec8c53b"
EXPORT_CONFIG_SHA = "f6a91e9d6879357fc10095c182689e43ee73bca2c10ec6926f93960cc68d98f2"
IDENTITY_SHA = "3776105cf88fd05c4f70a2de82ea610c3e98ef8bf47b7d80e1a4f18f5b5e7a1b"
DETECTOR_SNAPSHOT_SHA = "4714b925380ce62075318331d4078ddc827c25c217788cbb81993b2c75efd1b0"
RUNNER_CONTRACT_SHA = "8c67226cd7530d982b99589ef8be0d21955d79a1098fc81d645f6567cb320f27"
QA_SPEC_SHA = "6ba34bc20d694533892147868020a0838872caf677626e0a07626f0ec1287e9c"
VAL_MANIFEST_SHA = "9e0c1ac3d4a2c3b8e61246f8ec147745ca937c8830dc88b98bd4879342503ad2"
CHECKPOINT_SHA = "2ace52184b620204004509b72752ac7bfe64aadaf7fc1d076b18df8ab5a5c77e"
MODEL_CONFIG_SHA = "3fc6fda05f01ac16a90cf4116bd3793682fa780979ac37b09f459a66bd21cc54"
REPOSITORY_COMMIT = "1c8ac3f7ba84f14bd5651ab7b1b70d69a5f55f47"

ROAD8_NAMES = ("person", "bicycle", "car", "motorcycle", "bus", "train", "truck", "traffic light")
COCO_CATEGORY_IDS = (1, 2, 3, 4, 6, 7, 8, 10)
DETECTOR_CLASS_INDICES = (0, 1, 2, 3, 5, 6, 7, 9)

IMAGE_MANIFEST_COLUMNS = (
    "dataset_version",
    "split",
    "coco_image_id",
    "canonical_image_id",
    "relative_path",
    "file_name",
    "width",
    "height",
    "image_sha256",
    "image_archive_sha256",
)

FINAL_IMAGE_MANIFEST_COLUMNS = (
    "dataset_version",
    "split",
    "manifest_index",
    "coco_image_id",
    "canonical_image_id",
    "relative_path",
    "file_name",
    "width",
    "height",
    "image_sha256",
    "image_archive_sha256",
    "shard_id",
    "candidate_asset_id",
)

CANDIDATE_MANIFEST_COLUMNS = (
    "shard_id",
    "split",
    "path",
    "rows",
    "image_count",
    "first_manifest_index",
    "last_manifest_index",
    "first_image_id",
    "last_image_id",
    "sha256",
    "bytes",
    "semantic_sha256",
)

NATIVE_MANIFEST_COLUMNS = (
    "shard_id",
    "split",
    "path",
    "rows",
    "image_count",
    "first_manifest_index",
    "last_manifest_index",
    "first_image_id",
    "last_image_id",
    "sha256",
    "bytes",
    "content_sha256",
    "index_path",
    "index_rows",
    "index_sha256",
    "index_bytes",
)

SHARD_MANIFEST_COLUMNS = (
    "shard_id",
    "split",
    "first_manifest_index",
    "last_manifest_index",
    "first_image_id",
    "last_image_id",
    "image_count",
    "candidate_path",
    "candidate_rows",
    "candidate_sha256",
    "native_path",
    "native_rows",
    "native_sha256",
    "index_path",
    "index_rows",
    "index_sha256",
)

NATIVE_KEYS = (
    "image_ids",
    "query_index",
    "l3_road8_logits",
    "l3_road8_scores",
    "l3_query_embedding",
    "l3_pred_box_cxcywh",
)


CURRENT_STAGE = "startup"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def nvidia_driver_version() -> str | None:
    try:
        output = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        return output[0].strip() if output else None
    except (OSError, subprocess.SubprocessError):
        return None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        temporary.unlink()
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def write_csv_atomic(path: Path, fieldnames: Iterable[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        temporary.unlink()
    with temporary.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fieldnames), extrasaction="raise", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def write_text_atomic(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        temporary.unlink()
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(value)
        if not value.endswith("\n"):
            stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON_OBJECT_REQUIRED: {path}")
    return value


def load_csv_exact(path: Path, expected_columns: Iterable[str]) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != tuple(expected_columns):
            raise ValueError(f"IMAGE_MANIFEST_COLUMNS_MISMATCH: {reader.fieldnames}")
        return list(reader)


def canonical_json_hash(value: Any) -> str:
    digest = hashlib.sha256()
    encoder = json.JSONEncoder(
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    for chunk in encoder.iterencode(value):
        digest.update(chunk.encode("utf-8"))
    return digest.hexdigest()


def canonical_record_sequence_hash(records: Iterable[dict[str, Any]]) -> str:
    """Hash records as the canonical compact JSON array without materializing it."""
    digest = hashlib.sha256()
    encoder = json.JSONEncoder(
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    digest.update(b"[")
    first = True
    for record in records:
        if not first:
            digest.update(b",")
        first = False
        for chunk in encoder.iterencode(record):
            digest.update(chunk.encode("utf-8"))
    digest.update(b"]")
    return digest.hexdigest()


def array_bundle_hash(arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        array = np.ascontiguousarray(arrays[name])
        tokens = (name, array.dtype.str, json.dumps(list(array.shape), separators=(",", ":")))
        for token in tokens:
            encoded = token.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
        raw = array.tobytes(order="C")
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def length_prefixed_record_id(asset_id: str, image_id: str, road8_rank: int) -> str:
    fields = (
        asset_id.encode("utf-8"),
        image_id.encode("utf-8"),
        str(road8_rank).encode("ascii"),
    )
    payload = b"".join(len(field).to_bytes(8, "big") + field for field in fields)
    return hashlib.sha256(payload).hexdigest()


def compute_candidate_asset_id(identity: dict[str, Any]) -> str:
    split_binding = identity["split_assets"][SPLIT]
    values = (
        identity["dataset_version"],
        split_binding["split_manifest_sha256"],
        identity["detector_id"],
        identity["checkpoint_sha256"],
        identity["export_config"]["sha256"],
        identity["candidate_and_native_schema"]["sha256"],
    )
    return sha256_text("".join(values))


def import_symbol(path: Path, module_name: str, symbol: str) -> Any:
    specification = importlib.util.spec_from_file_location(module_name, path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"MODULE_IMPORT_SPEC_FAIL: {path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    if not hasattr(module, symbol):
        raise RuntimeError(f"MODULE_SYMBOL_MISSING: {symbol}: {path}")
    return getattr(module, symbol)


def candidate_arrow_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("dataset_version", pa.string(), nullable=False),
            pa.field("split", pa.string(), nullable=False),
            pa.field("coco_image_id", pa.int64(), nullable=False),
            pa.field("image_id", pa.string(), nullable=False),
            pa.field("canonical_image_id", pa.string(), nullable=False),
            pa.field("image_sha256", pa.string(), nullable=False),
            pa.field("candidate_asset_id", pa.string(), nullable=False),
            pa.field("detector_id", pa.string(), nullable=False),
            pa.field("checkpoint_sha256", pa.string(), nullable=False),
            pa.field("export_config_sha256", pa.string(), nullable=False),
            pa.field("schema_sha256", pa.string(), nullable=False),
            pa.field("candidate_record_id", pa.string(), nullable=False),
            pa.field("road8_rank", pa.int32(), nullable=False),
            pa.field("source_order", pa.int32(), nullable=False),
            pa.field("query_index", pa.int32(), nullable=False),
            pa.field("predicted_road8_class_id", pa.int16(), nullable=False),
            pa.field("predicted_road8_class_name", pa.string(), nullable=False),
            pa.field("predicted_coco_category_id", pa.int16(), nullable=False),
            pa.field("detector_class_index", pa.int16(), nullable=False),
            pa.field("score", pa.float64(), nullable=False),
            pa.field("bbox_x1", pa.float64(), nullable=False),
            pa.field("bbox_y1", pa.float64(), nullable=False),
            pa.field("bbox_x2", pa.float64(), nullable=False),
            pa.field("bbox_y2", pa.float64(), nullable=False),
            pa.field("bbox_cx", pa.float64(), nullable=False),
            pa.field("bbox_cy", pa.float64(), nullable=False),
            pa.field("bbox_w", pa.float64(), nullable=False),
            pa.field("bbox_h", pa.float64(), nullable=False),
            pa.field("bbox_xyxy", pa.list_(pa.float64(), 4), nullable=False),
            pa.field("bbox_cxcywh", pa.list_(pa.float64(), 4), nullable=False),
        ]
    )


def native_index_arrow_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("canonical_image_id", pa.string(), nullable=False),
            pa.field("query_index", pa.int32(), nullable=False),
            pa.field("shard_id", pa.string(), nullable=False),
            pa.field("row_offset", pa.int64(), nullable=False),
        ]
    )


def native_for_storage(native: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    stored = {
        "query_index": np.asarray(native["query_index"], dtype=np.int32),
        "l3_road8_logits": np.asarray(native["road8_logits"], dtype=np.float32),
        "l3_road8_scores": np.asarray(native["road8_scores"], dtype=np.float32),
        "l3_query_embedding": np.asarray(native["query_embedding"], dtype=np.float16),
        "l3_pred_box_cxcywh": np.asarray(native["boxes_cxcywh"], dtype=np.float32),
    }
    expected = {
        "query_index": ((300,), np.dtype(np.int32)),
        "l3_road8_logits": ((300, 8), np.dtype(np.float32)),
        "l3_road8_scores": ((300, 8), np.dtype(np.float32)),
        "l3_query_embedding": ((300, 256), np.dtype(np.float16)),
        "l3_pred_box_cxcywh": ((300, 4), np.dtype(np.float32)),
    }
    for name, (shape, dtype) in expected.items():
        array = stored[name]
        if array.shape != shape or array.dtype != dtype or not np.isfinite(array).all():
            raise ValueError(f"NATIVE_SHAPE_DTYPE_FINITE_FAIL: {name}")
    if not np.array_equal(stored["query_index"], np.arange(300, dtype=np.int32)):
        raise ValueError("NATIVE_QUERY_INDEX_FAIL")
    reconstructed = torch.sigmoid(torch.from_numpy(stored["l3_road8_logits"])).numpy()
    if not np.allclose(reconstructed, stored["l3_road8_scores"], atol=1e-7, rtol=1e-6):
        maximum = float(np.max(np.abs(reconstructed - stored["l3_road8_scores"])))
        raise ValueError(f"SCORE_RECONSTRUCTION_FAILURE: max_abs={maximum}")
    return stored


def build_candidates(
    native: dict[str, np.ndarray],
    row: dict[str, str],
) -> list[dict[str, Any]]:
    scores_2d = np.asarray(native["road8_scores"], dtype=np.float32)
    boxes = np.asarray(native["boxes_cxcywh"], dtype=np.float32)
    if scores_2d.shape != (300, 8) or boxes.shape != (300, 4):
        raise ValueError("QUERY_SHAPE_MISMATCH")
    scores = scores_2d.reshape(-1)
    query_index = np.repeat(np.arange(300, dtype=np.int32), 8)
    road8_id = np.tile(np.arange(1, 9, dtype=np.int32), 300)
    source_order = np.arange(2400, dtype=np.int32)
    order = np.lexsort((source_order, road8_id, query_index, -scores))
    selected = order[:300]
    if selected.size != 300:
        raise ValueError("TOP300_CONSTRUCTION_FAIL")

    cx, cy, width, height = (boxes[:, index] for index in range(4))
    xyxy = np.stack(
        (
            cx - width / np.float32(2),
            cy - height / np.float32(2),
            cx + width / np.float32(2),
            cy + height / np.float32(2),
        ),
        axis=1,
    )
    xyxy *= np.asarray(
        [int(row["width"]), int(row["height"]), int(row["width"]), int(row["height"])],
        dtype=np.float32,
    )

    records: list[dict[str, Any]] = []
    previous_key: tuple[float, int, int, int] | None = None
    for rank, flat_value in enumerate(selected, start=1):
        flat = int(flat_value)
        query = int(query_index[flat])
        class_id = int(road8_id[flat])
        source = int(source_order[flat])
        score = float(scores[flat])
        key = (-score, query, class_id, source)
        if previous_key is not None and key < previous_key:
            raise ValueError("CANDIDATE_ORDERING_MISMATCH")
        previous_key = key
        if source != query * 8 + class_id - 1:
            raise ValueError("SOURCE_ORDER_MISMATCH")
        x1, y1, x2, y2 = map(float, xyxy[query])
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        box_width = x2 - x1
        box_height = y2 - y1
        records.append(
            {
                "dataset_version": DATASET_VERSION,
                "split": SPLIT,
                "coco_image_id": int(row["coco_image_id"]),
                "image_id": row["canonical_image_id"],
                "canonical_image_id": row["canonical_image_id"],
                "image_sha256": row["image_sha256"],
                "candidate_asset_id": CANDIDATE_ASSET_ID,
                "detector_id": "RT-DETRv2-R18VD",
                "checkpoint_sha256": CHECKPOINT_SHA,
                "export_config_sha256": EXPORT_CONFIG_SHA,
                "schema_sha256": SCHEMA_SHA,
                "candidate_record_id": length_prefixed_record_id(
                    CANDIDATE_ASSET_ID, row["canonical_image_id"], rank
                ),
                "road8_rank": rank,
                "source_order": source,
                "query_index": query,
                "predicted_road8_class_id": class_id,
                "predicted_road8_class_name": ROAD8_NAMES[class_id - 1],
                "predicted_coco_category_id": COCO_CATEGORY_IDS[class_id - 1],
                "detector_class_index": DETECTOR_CLASS_INDICES[class_id - 1],
                "score": score,
                "bbox_x1": x1,
                "bbox_y1": y1,
                "bbox_x2": x2,
                "bbox_y2": y2,
                "bbox_cx": center_x,
                "bbox_cy": center_y,
                "bbox_w": box_width,
                "bbox_h": box_height,
                "bbox_xyxy": [x1, y1, x2, y2],
                "bbox_cxcywh": [center_x, center_y, box_width, box_height],
            }
        )
    if [record["road8_rank"] for record in records] != list(range(1, 301)):
        raise ValueError("CANDIDATE_RANK_CONTINUITY_FAIL")
    if len({record["candidate_record_id"] for record in records}) != 300:
        raise ValueError("CANDIDATE_RECORD_ID_WITHIN_IMAGE_DUPLICATE")
    return records


def load_rgb_verified(dataset_root: Path, row: dict[str, str]) -> Image.Image:
    candidate = (dataset_root / row["relative_path"]).resolve(strict=True)
    if not candidate.is_relative_to(dataset_root):
        raise ValueError("UNSAFE_IMAGE_PATH")
    raw = candidate.read_bytes()
    actual_sha = hashlib.sha256(raw).hexdigest()
    if actual_sha != row["image_sha256"]:
        raise ValueError(f"IMAGE_SHA256_MISMATCH: {row['canonical_image_id']}")
    with Image.open(io.BytesIO(raw)) as source:
        source.load()
        if source.size != (int(row["width"]), int(row["height"])):
            raise ValueError(f"IMAGE_DIMENSION_MISMATCH: {row['canonical_image_id']}")
        image = source.convert("RGB")
    return image


def forward_one(detector: Any, dataset_root: Path, row: dict[str, str]) -> tuple[list[dict[str, Any]], dict[str, np.ndarray], float]:
    started = time.perf_counter()
    image = load_rgb_verified(dataset_root, row)
    try:
        native = detector.infer(image)
    finally:
        image.close()
    stored = native_for_storage(native)
    candidates = build_candidates(native, row)
    elapsed = time.perf_counter() - started
    return candidates, stored, elapsed


def shard_paths(staging_root: Path, ordinal: int) -> dict[str, Path]:
    suffix = f"{ordinal:04d}"
    return {
        "candidate": staging_root / "candidate" / f"VAL2017_candidates_{suffix}.parquet",
        "native": staging_root / "native" / f"VAL2017_native_{suffix}.npz",
        "index": staging_root / "native" / f"VAL2017_native_index_{suffix}.parquet",
        "metadata": staging_root / "shards" / f"VAL2017_shard_{suffix}.json",
    }


def verify_resumable_shard(metadata_path: Path, staging_root: Path, ordinal: int) -> dict[str, Any]:
    metadata = read_json(metadata_path)
    expected_id = f"VAL2017_shard_{ordinal:04d}"
    if metadata.get("status") != "SHARD_COMMITTED" or metadata.get("shard_id") != expected_id:
        raise ValueError(f"RESUME_SHARD_METADATA_FAIL: {expected_id}")
    if metadata.get("candidate_asset_id") != CANDIDATE_ASSET_ID:
        raise ValueError(f"RESUME_ASSET_ID_MISMATCH: {expected_id}")
    if metadata.get("runner_source_sha256") != sha256_file(Path(__file__).resolve()):
        raise ValueError(f"RESUME_RUNNER_SOURCE_MISMATCH: {expected_id}")
    for section in ("candidate", "native", "native_index"):
        item = metadata[section]
        path = staging_root / item["path"]
        if not path.is_file() or sha256_file(path) != item["sha256"] or path.stat().st_size != item["bytes"]:
            raise ValueError(f"RESUME_SHARD_BYTES_MISMATCH: {expected_id}:{section}")
    if metadata["candidate"]["rows"] != IMAGES_PER_SHARD * STORED_CANDIDATES_PER_IMAGE:
        raise ValueError(f"RESUME_CANDIDATE_ROW_COUNT_FAIL: {expected_id}")
    if metadata["native"]["rows"] != IMAGES_PER_SHARD * QUERY_COUNT:
        raise ValueError(f"RESUME_NATIVE_ROW_COUNT_FAIL: {expected_id}")
    if metadata["native_index"]["rows"] != IMAGES_PER_SHARD * QUERY_COUNT:
        raise ValueError(f"RESUME_NATIVE_INDEX_ROW_COUNT_FAIL: {expected_id}")
    return metadata


def purge_incomplete_shard(paths: dict[str, Path]) -> None:
    if paths["metadata"].exists():
        raise RuntimeError("REFUSE_PURGE_COMMITTED_SHARD")
    for key in ("candidate", "native", "index"):
        path = paths[key]
        temporary = path.with_name(path.name + ".tmp")
        if path.exists():
            path.unlink()
        if temporary.exists():
            temporary.unlink()


def write_candidate_shard(path: Path, records: list[dict[str, Any]]) -> dict[str, Any]:
    expected_rows = IMAGES_PER_SHARD * STORED_CANDIDATES_PER_IMAGE
    if len(records) != expected_rows:
        raise ValueError(f"CANDIDATE_SHARD_ROW_COUNT_FAIL: {len(records)}")
    schema = candidate_arrow_schema()
    semantic_sha_before = canonical_record_sequence_hash(records)
    table = pa.Table.from_pylist(records, schema=schema)
    if table.num_rows != expected_rows or any(column.null_count for column in table.columns):
        raise ValueError("CANDIDATE_SHARD_TABLE_FAIL")
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists() or path.exists():
        raise FileExistsError(f"REFUSE_CANDIDATE_OVERWRITE: {path}")
    pq.write_table(table, temporary, compression="zstd", use_dictionary=True)
    del table
    readback_rows = 0
    null_count = 0
    # ParquetFile keeps an open Windows handle.  Keep validation inside the
    # context so the handle is closed before the atomic rename below.
    with pq.ParquetFile(temporary) as parquet_file:
        if not parquet_file.schema_arrow.equals(schema, check_metadata=False):
            raise ValueError("CANDIDATE_PARQUET_SCHEMA_READBACK_FAIL")
        if parquet_file.metadata.num_rows != expected_rows:
            raise ValueError("CANDIDATE_PARQUET_READBACK_FAIL")

        def iter_readback() -> Iterable[dict[str, Any]]:
            nonlocal readback_rows, null_count
            for batch in parquet_file.iter_batches(batch_size=4096):
                readback_rows += batch.num_rows
                null_count += sum(batch.column(index).null_count for index in range(batch.num_columns))
                yield from batch.to_pylist()

        semantic_sha = canonical_record_sequence_hash(iter_readback())
    if readback_rows != expected_rows or null_count != 0:
        raise ValueError("CANDIDATE_PARQUET_NULL_OR_ROW_READBACK_FAIL")
    if semantic_sha != semantic_sha_before:
        raise ValueError("CANDIDATE_PARQUET_SEMANTIC_READBACK_FAIL")
    os.replace(temporary, path)
    return {
        "path": path.relative_to(path.parents[1]).as_posix(),
        "format": "parquet",
        "rows": expected_rows,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "semantic_sha256": semantic_sha,
    }


def write_native_shard(
    path: Path,
    image_ids: list[str],
    arrays_per_image: list[dict[str, np.ndarray]],
) -> dict[str, Any]:
    if len(image_ids) != IMAGES_PER_SHARD or len(arrays_per_image) != IMAGES_PER_SHARD:
        raise ValueError("NATIVE_IMAGE_COUNT_FAIL")
    max_chars = max(len(value) for value in image_ids)
    payload: dict[str, np.ndarray] = {
        "image_ids": np.concatenate(
            [np.repeat(np.asarray(image_id, dtype=f"<U{max_chars}"), 300) for image_id in image_ids]
        ),
        "query_index": np.concatenate([item["query_index"] for item in arrays_per_image]),
        "l3_road8_logits": np.concatenate([item["l3_road8_logits"] for item in arrays_per_image]),
        "l3_road8_scores": np.concatenate([item["l3_road8_scores"] for item in arrays_per_image]),
        "l3_query_embedding": np.concatenate([item["l3_query_embedding"] for item in arrays_per_image]),
        "l3_pred_box_cxcywh": np.concatenate([item["l3_pred_box_cxcywh"] for item in arrays_per_image]),
    }
    expected_rows = IMAGES_PER_SHARD * QUERY_COUNT
    if tuple(payload) != NATIVE_KEYS or any(array.shape[0] != expected_rows for array in payload.values()):
        raise ValueError("NATIVE_PAYLOAD_KEY_OR_ROW_FAIL")
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists() or path.exists():
        raise FileExistsError(f"REFUSE_NATIVE_OVERWRITE: {path}")
    with temporary.open("xb") as stream:
        np.savez_compressed(stream, **payload)
        stream.flush()
        os.fsync(stream.fileno())
    with np.load(temporary, allow_pickle=False) as loaded:
        if tuple(loaded.files) != NATIVE_KEYS:
            raise ValueError("NATIVE_NPZ_KEYS_READBACK_FAIL")
        reread_payload: dict[str, np.ndarray] = {}
        for name in NATIVE_KEYS:
            actual = loaded[name]
            expected = payload[name]
            if actual.dtype != expected.dtype or actual.shape != expected.shape or not np.array_equal(actual, expected):
                raise ValueError(f"NATIVE_NPZ_READBACK_FAIL: {name}")
            reread_payload[name] = actual.copy()
    content_sha = array_bundle_hash(reread_payload)
    os.replace(temporary, path)
    return {
        "path": path.relative_to(path.parents[1]).as_posix(),
        "format": "npz",
        "rows": expected_rows,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "content_sha256": content_sha,
        "keys": list(NATIVE_KEYS),
    }


def write_native_index(path: Path, image_ids: list[str], shard_id: str) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for image_offset, image_id in enumerate(image_ids):
        base = image_offset * QUERY_COUNT
        rows.extend(
            {
                "canonical_image_id": image_id,
                "query_index": query_index,
                "shard_id": shard_id,
                "row_offset": base + query_index,
            }
            for query_index in range(QUERY_COUNT)
        )
    table = pa.Table.from_pylist(rows, schema=native_index_arrow_schema())
    expected_rows = IMAGES_PER_SHARD * QUERY_COUNT
    if table.num_rows != expected_rows or any(column.null_count for column in table.columns):
        raise ValueError("NATIVE_INDEX_TABLE_FAIL")
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists() or path.exists():
        raise FileExistsError(f"REFUSE_NATIVE_INDEX_OVERWRITE: {path}")
    pq.write_table(table, temporary, compression="zstd", use_dictionary=True)
    with pq.ParquetFile(temporary) as parquet_file:
        reread = parquet_file.read()
        if not reread.schema.equals(native_index_arrow_schema(), check_metadata=False):
            raise ValueError("NATIVE_INDEX_SCHEMA_READBACK_FAIL")
        if reread.num_rows != expected_rows or reread.column("row_offset").to_pylist() != list(range(expected_rows)):
            raise ValueError("NATIVE_INDEX_READBACK_FAIL")
    os.replace(temporary, path)
    return {
        "path": path.relative_to(path.parents[1]).as_posix(),
        "format": "parquet",
        "rows": expected_rows,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def select_determinism_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    ranked = sorted(
        rows,
        key=lambda row: (
            sha256_text(DETERMINISM_NAMESPACE + row["canonical_image_id"]),
            row["canonical_image_id"],
        ),
    )
    return ranked[:DETERMINISM_SAMPLE_COUNT]


def derive_manifest_rows(shards: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    candidate_rows: list[dict[str, Any]] = []
    native_rows: list[dict[str, Any]] = []
    shard_rows: list[dict[str, Any]] = []
    for item in shards:
        common = {
            "shard_id": item["shard_id"],
            "split": item["split"],
            "image_count": item["image_count"],
            "first_manifest_index": item["first_manifest_index"],
            "last_manifest_index": item["last_manifest_index"],
            "first_image_id": item["first_image_id"],
            "last_image_id": item["last_image_id"],
        }
        candidate_rows.append(
            {
                **common,
                "path": item["candidate"]["path"],
                "rows": item["candidate"]["rows"],
                "sha256": item["candidate"]["sha256"],
                "bytes": item["candidate"]["bytes"],
                "semantic_sha256": item["candidate"]["semantic_sha256"],
            }
        )
        native_rows.append(
            {
                **common,
                "path": item["native"]["path"],
                "rows": item["native"]["rows"],
                "sha256": item["native"]["sha256"],
                "bytes": item["native"]["bytes"],
                "content_sha256": item["native"]["content_sha256"],
                "index_path": item["native_index"]["path"],
                "index_rows": item["native_index"]["rows"],
                "index_sha256": item["native_index"]["sha256"],
                "index_bytes": item["native_index"]["bytes"],
            }
        )
        shard_rows.append(
            {
                **common,
                "candidate_path": item["candidate"]["path"],
                "candidate_rows": item["candidate"]["rows"],
                "candidate_sha256": item["candidate"]["sha256"],
                "native_path": item["native"]["path"],
                "native_rows": item["native"]["rows"],
                "native_sha256": item["native"]["sha256"],
                "index_path": item["native_index"]["path"],
                "index_rows": item["native_index"]["rows"],
                "index_sha256": item["native_index"]["sha256"],
            }
        )
    return candidate_rows, native_rows, shard_rows


def create_sha_ledger(staging_root: Path, run_config_path: Path) -> None:
    rows: list[dict[str, Any]] = []
    for role, base in (
        ("candidate_shard", staging_root / "candidate"),
        ("native_or_index_shard", staging_root / "native"),
        ("shard_metadata", staging_root / "shards"),
    ):
        for path in sorted(base.glob("*")):
            if path.is_file():
                rows.append(
                    {
                        "role": role,
                        "path": path.relative_to(staging_root).as_posix(),
                        "bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                    }
                )
    for name in (
        "candidate_manifest.csv",
        "native_manifest.csv",
        "image_manifest.csv",
        "shard_manifest.csv",
        "runtime_environment.json",
        "sha256_ledger_scope.json",
    ):
        path = staging_root / "manifest" / name
        rows.append(
            {
                "role": "manifest_copy",
                "path": path.relative_to(staging_root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    for name in (
        "candidate_manifest.csv",
        "native_manifest.csv",
        "image_manifest.csv",
        "shard_manifest.csv",
        "native_state_index.parquet",
        "determinism_report.json",
        "runtime_summary.csv",
    ):
        path = staging_root / name
        rows.append(
            {
                "role": "staging_root_metadata",
                "path": name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    rows.append(
        {
            "role": "run_config",
            "path": str(run_config_path),
            "bytes": run_config_path.stat().st_size,
            "sha256": sha256_file(run_config_path),
        }
    )
    for name in ("preflight_report.json", "model_load_report.json"):
        path = run_config_path.parent / name
        rows.append(
            {
                "role": "run_evidence",
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    write_csv_atomic(staging_root / "sha256_ledger.csv", ("role", "path", "bytes", "sha256"), rows)


def verify_sha_ledger(asset_root: Path, ledger_path: Path) -> dict[str, int]:
    rows = load_csv_exact(ledger_path, ("role", "path", "bytes", "sha256"))
    seen: set[str] = set()
    mismatches = 0
    for row in rows:
        identity = row["path"]
        if identity in seen:
            raise ValueError(f"SHA_LEDGER_DUPLICATE_PATH: {identity}")
        seen.add(identity)
        listed = Path(identity)
        path = listed if listed.is_absolute() else asset_root / listed
        if (
            not path.is_file()
            or path.stat().st_size != int(row["bytes"])
            or sha256_file(path) != row["sha256"]
        ):
            mismatches += 1
    if mismatches:
        raise ValueError(f"SHA_LEDGER_MISMATCH: {mismatches}")
    return {"entries": len(rows), "mismatches": 0}


def write_global_native_index(staging_root: Path) -> dict[str, Any]:
    path = staging_root / "native_state_index.parquet"
    temporary = path.with_name(path.name + ".tmp")
    # This file is a deterministic projection of the ten committed per-shard
    # indexes, not an independently generated scientific state.  A resumed
    # transaction rebuilds it atomically from those hash-verified sources.
    if temporary.exists():
        temporary.unlink()
    total_rows = 0
    writer = pq.ParquetWriter(temporary, native_index_arrow_schema(), compression="zstd")
    try:
        for ordinal in range(EXPECTED_SHARD_COUNT):
            source = staging_root / "native" / f"VAL2017_native_index_{ordinal:04d}.parquet"
            table = pq.read_table(source)
            if not table.schema.equals(native_index_arrow_schema(), check_metadata=False):
                raise ValueError(f"GLOBAL_NATIVE_INDEX_SOURCE_SCHEMA_FAIL: {source}")
            writer.write_table(table)
            total_rows += table.num_rows
    finally:
        writer.close()
    if total_rows != EXPECTED_IMAGE_COUNT * QUERY_COUNT:
        raise ValueError(f"GLOBAL_NATIVE_INDEX_ROW_COUNT_FAIL: {total_rows}")
    # Close the ParquetFile before replacing its path on Windows.
    with pq.ParquetFile(temporary) as parquet_file:
        if (
            not parquet_file.schema_arrow.equals(native_index_arrow_schema(), check_metadata=False)
            or parquet_file.metadata.num_rows != total_rows
        ):
            raise ValueError("GLOBAL_NATIVE_INDEX_READBACK_FAIL")
    os.replace(temporary, path)
    return {
        "path": path.name,
        "rows": total_rows,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def write_post_qa_metadata(
    staging_root: Path,
    run_config_path: Path,
    validator_result: dict[str, Any],
    shard_metadata: list[dict[str, Any]],
    global_index: dict[str, Any],
) -> dict[str, Any]:
    qa_text = f"""# CANONICAL EXPORT QA REPORT

任务：`LC-ALLOC-COCO-CANONICAL-EXPORT`

```text
qa_status = PASS
images = {validator_result['images_validated']}/{validator_result['images_expected']}
candidate_rows = {validator_result['candidate_rows']}
native_rows = {validator_result['native_rows']}
candidate_record_id_duplicates = {validator_result['candidate_record_id_duplicates']}
candidate_native_join_failures = {validator_result['candidate_native_join_failures']}
sorting_mismatches = {validator_result['sorting_mismatches']}
sha_mismatches = {validator_result['sha_mismatches']}
determinism = {validator_result['determinism']['candidate_exact']}/20 candidate, {validator_result['determinism']['native_exact']}/20 native
```

独立只读 validator 对 5,000 张 VAL2017、10 组 candidate/native/index shards、全局 native index、四份 manifest、身份、排序、score/bbox 重构、join、物理 schema、SHA ledger 与固定 20 图复跑执行了完整检查。所有 hard counts 为 0。合法但未进入 Top300 的 native query 不计为 join failure；每条 candidate 均通过 `(canonical_image_id, query_index)` 唯一连接。

本 QA 未读取 annotation/GT，未运行 allocator，也未计算 AP/AR/Coverage/QUALITY。
"""
    write_text_atomic(staging_root / "qa_report.md", qa_text)
    write_text_atomic(staging_root / "CANONICAL_EXPORT_QA_REPORT.md", qa_text)

    total_candidate_bytes = sum(item["candidate"]["bytes"] for item in shard_metadata)
    total_native_bytes = sum(item["native"]["bytes"] for item in shard_metadata)
    total_index_bytes = sum(item["native_index"]["bytes"] for item in shard_metadata)
    report_text = f"""# COCO canonical export report

任务：`LC-ALLOC-COCO-CANONICAL-EXPORT`

## 状态

```text
execution_status = COMPLETE
export_status = COCO_CANONICAL_EXPORT_COMPLETE
qa_status = PASS
COCO_CANONICAL_EXPORT = COMPLETE
```

## 冻结身份与范围

- Dataset：`COCO2017-Road8-v1 / VAL2017`
- Images：5,000/5,000，严格保持 image-only canonical manifest 顺序。
- Detector：`RT-DETRv2-R18VD`，checkpoint `{CHECKPOINT_SHA}`。
- CandidateAssetID：`{CANDIDATE_ASSET_ID}`。
- Candidate：每图从 300 queries × 8 Road8 hypotheses 原位构造，按冻结四键排序，仅保存 Top300；总计 1,500,000 行。
- Native：每图 300 行，保存 Road8 logits/scores、float16 query embedding、L3 normalized boxes 与 query index；总计 1,500,000 行。
- Shards：10，每片 500 图；candidate/native/index bytes 分别为 {total_candidate_bytes}/{total_native_bytes}/{total_index_bytes}。

## QA 与边界

- candidate record ID duplicate = 0；candidate/native join failure = 0；NaN/Inf = 0；sorting mismatch = 0；ledger mismatch = 0。
- 固定 20 张 VAL 图重复 forward，native/candidate semantic hashes、record IDs 与排序全部 exact 一致。
- 没有读取 annotation/GT，没有执行 allocator，没有生成 selection，也没有计算 AP/AR/Coverage/QUALITY。

资产只有在 `CANONICAL_EXPORT_COMPLETE.json` 最终逻辑提交标记写入后才可消费。
"""
    write_text_atomic(staging_root / "COCO_CANONICAL_EXPORT_REPORT.md", report_text)

    run_config = read_json(run_config_path)
    ledger_path = staging_root / "sha256_ledger.csv"
    qa_result_path = staging_root / "qa_validation_result.json"
    asset_manifest = {
        "contract_type": "COCO2017_VAL2017_CANONICAL_ASSET_MANIFEST_V1",
        "status": "QA_PASS_AWAITING_ATOMIC_PUBLICATION",
        "dataset_version": DATASET_VERSION,
        "split": SPLIT,
        "image_count": EXPECTED_IMAGE_COUNT,
        "candidate_asset_id": CANDIDATE_ASSET_ID,
        "candidate_rows": EXPECTED_IMAGE_COUNT * STORED_CANDIDATES_PER_IMAGE,
        "native_rows": EXPECTED_IMAGE_COUNT * QUERY_COUNT,
        "candidate_shards": EXPECTED_SHARD_COUNT,
        "native_shards": EXPECTED_SHARD_COUNT,
        "input_bindings": run_config["input_bindings"],
        "runtime_environment": {
            "path": "manifest/runtime_environment.json",
            "sha256": sha256_file(staging_root / "manifest" / "runtime_environment.json"),
        },
        "global_native_index": global_index,
        "sha256_ledger": {
            "path": "sha256_ledger.csv",
            "sha256": sha256_file(ledger_path),
            "scope": "non-self-referential data ledger; exclusions are frozen in manifest/sha256_ledger_scope.json",
        },
        "qa_validation_result": {
            "path": "qa_validation_result.json",
            "sha256": sha256_file(qa_result_path),
            "all_checks_pass": True,
        },
        "reports": {
            name: sha256_file(staging_root / name)
            for name in (
                "qa_report.md",
                "CANONICAL_EXPORT_QA_REPORT.md",
                "COCO_CANONICAL_EXPORT_REPORT.md",
            )
        },
        "shards": validator_result["verified_shards"],
        "scientific_evaluation_performed": False,
        "annotation_or_GT_read": False,
        "created_at_utc": utc_now(),
    }
    path = staging_root / "manifest" / "asset_manifest.json"
    write_json_atomic(path, asset_manifest)
    return asset_manifest


def publish_after_qa(output_root: Path, staging_root: Path, validator_result: dict[str, Any], run_config_path: Path) -> dict[str, Any]:
    completion_marker = output_root / "CANONICAL_EXPORT_COMPLETE.json"
    if completion_marker.exists():
        raise FileExistsError("CANONICAL_EXPORT_ALREADY_PUBLISHED")
    directory_names = ("candidate", "native", "shards", "manifest")
    top_files = (
        "candidate_manifest.csv",
        "native_manifest.csv",
        "image_manifest.csv",
        "shard_manifest.csv",
        "native_state_index.parquet",
        "determinism_report.json",
        "runtime_summary.csv",
        "sha256_ledger.csv",
        "qa_validation_result.json",
        "qa_report.md",
        "CANONICAL_EXPORT_QA_REPORT.md",
        "COCO_CANONICAL_EXPORT_REPORT.md",
    )
    for name in directory_names:
        if (output_root / name).exists():
            raise FileExistsError(f"REFUSE_FINAL_DIRECTORY_OVERWRITE: {name}")
    for name in top_files:
        if (output_root / name).exists():
            raise FileExistsError(f"REFUSE_FINAL_FILE_OVERWRITE: {name}")

    moved_directories: list[str] = []
    published_files: list[Path] = []
    try:
        for name in directory_names:
            source = staging_root / name
            if not source.is_dir():
                raise FileNotFoundError(f"STAGING_PUBLICATION_DIRECTORY_MISSING: {name}")
            os.replace(source, output_root / name)
            moved_directories.append(name)
        for name in top_files:
            source = staging_root / name
            if not source.is_file():
                if name in ("qa_report.md", "CANONICAL_EXPORT_QA_REPORT.md"):
                    continue
                raise FileNotFoundError(f"STAGING_PUBLICATION_FILE_MISSING: {name}")
            temporary = output_root / (".publishing." + name)
            if temporary.exists():
                temporary.unlink()
            shutil.copyfile(source, temporary)
            if sha256_file(temporary) != sha256_file(source):
                raise RuntimeError(f"PUBLICATION_COPY_SHA_FAIL: {name}")
            os.replace(temporary, output_root / name)
            published_files.append(output_root / name)

        ledger_verification = verify_sha_ledger(
            output_root, output_root / "sha256_ledger.csv"
        )
        asset_manifest_path = output_root / "manifest" / "asset_manifest.json"
        runtime_environment_path = output_root / "manifest" / "runtime_environment.json"
        if not asset_manifest_path.is_file() or not runtime_environment_path.is_file():
            raise FileNotFoundError("PUBLISHED_ASSET_MANIFEST_OR_RUNTIME_ENVIRONMENT_MISSING")

        marker = {
            "contract_type": "COCO2017_VAL2017_CANONICAL_EXPORT_COMPLETION_V1",
            "status": "COCO_CANONICAL_EXPORT_COMPLETE",
            "dataset_version": DATASET_VERSION,
            "split": SPLIT,
            "image_count": EXPECTED_IMAGE_COUNT,
            "candidate_asset_id": CANDIDATE_ASSET_ID,
            "runner_source_sha256": sha256_file(Path(__file__).resolve()),
            "validator_source_sha256": read_json(run_config_path)["input_bindings"]["qa_validator"]["sha256"],
            "qa_all_checks_pass": validator_result["all_checks_pass"],
            "sha256_ledger": {
                "sha256": sha256_file(output_root / "sha256_ledger.csv"),
                **ledger_verification,
            },
            "qa_validation_result_sha256": sha256_file(output_root / "qa_validation_result.json"),
            "asset_manifest_sha256": sha256_file(asset_manifest_path),
            "runtime_environment_sha256": sha256_file(runtime_environment_path),
            "final_report_sha256": sha256_file(output_root / "COCO_CANONICAL_EXPORT_REPORT.md"),
            "published_at_utc": utc_now(),
            "manifest_sha256": {
                name: sha256_file(output_root / name)
                for name in ("candidate_manifest.csv", "native_manifest.csv", "image_manifest.csv", "shard_manifest.csv")
            },
            "publication_rule": "This marker is the final atomic logical commit. Absence means unpublished.",
        }
        temporary_marker = output_root / ".CANONICAL_EXPORT_COMPLETE.json.tmp"
        write_json_atomic(temporary_marker, marker)
        os.replace(temporary_marker, completion_marker)
        return marker
    except BaseException:
        for path in reversed(published_files):
            if path.exists():
                path.unlink()
        for name in reversed(moved_directories):
            final_path = output_root / name
            staging_path = staging_root / name
            if final_path.exists() and not staging_path.exists():
                os.replace(final_path, staging_path)
        raise


def build_run_config(
    output_root: Path,
    staging_root: Path,
    freeze_root: Path,
    dataset_root: Path,
    validator_path: Path,
    detector_snapshot_path: Path,
    export_config: dict[str, Any],
    identity_sha: str,
    dependency_fallback_path: Path | None,
) -> dict[str, Any]:
    manifest_path = Path(export_config["split_manifest_registry"][SPLIT]["path"])
    checkpoint_path = Path(export_config["detector"]["checkpoint_path"])
    model_config_path = Path(export_config["detector"]["model_config_path"])
    schema_path = freeze_root / SCHEMA_FILE
    export_config_path = freeze_root / EXPORT_CONFIG_FILE
    identity_path = freeze_root / IDENTITY_FILE
    runner_contract_path = freeze_root / RUNNER_CONTRACT_FILE
    qa_spec_path = freeze_root / QA_SPEC_FILE
    runner_source_path = Path(__file__).resolve()
    manifest_rows = load_csv_exact(manifest_path, IMAGE_MANIFEST_COLUMNS)
    determinism_image_ids = [
        row["canonical_image_id"] for row in select_determinism_rows(manifest_rows)
    ]
    reference_runtime = export_config["reference_runtime"]
    actual_runtime = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": importlib.metadata.version("torchvision"),
        "numpy": np.__version__,
        "pyarrow": pa.__version__,
        "Pillow": importlib.metadata.version("Pillow"),
    }
    expected_runtime = {
        name: reference_runtime[name]
        for name in ("python", "torch", "torchvision", "numpy", "pyarrow", "Pillow")
    }
    return {
        "contract_type": "COCO2017_VAL2017_CANONICAL_EXPORT_RUN_V1",
        "status": "STAGING_NOT_PUBLISHED",
        "created_at_utc": utc_now(),
        "created_before_model_load": True,
        "immutable_after_creation": True,
        "output_root": str(output_root),
        "staging_root": str(staging_root),
        "dataset_version": DATASET_VERSION,
        "split": SPLIT,
        "expected_image_count": EXPECTED_IMAGE_COUNT,
        "images_per_shard": IMAGES_PER_SHARD,
        "expected_shard_count": EXPECTED_SHARD_COUNT,
        "candidate_asset_id": CANDIDATE_ASSET_ID,
        "input_bindings": {
            "candidate_schema": {"path": str(schema_path), "sha256": SCHEMA_SHA},
            "export_config": {"path": str(export_config_path), "sha256": EXPORT_CONFIG_SHA},
            "candidate_asset_identity": {"path": str(identity_path), "sha256": identity_sha},
            "runner_contract": {"path": str(runner_contract_path), "sha256": RUNNER_CONTRACT_SHA},
            "qa_spec": {"path": str(qa_spec_path), "sha256": QA_SPEC_SHA},
            "image_manifest": {"path": str(manifest_path), "sha256": VAL_MANIFEST_SHA},
            "checkpoint": {"path": str(checkpoint_path), "sha256": CHECKPOINT_SHA},
            "model_config": {"path": str(model_config_path), "sha256": MODEL_CONFIG_SHA},
            "detector_snapshot": {"path": str(detector_snapshot_path), "sha256": DETECTOR_SNAPSHOT_SHA},
            "runner_source": {"path": str(runner_source_path), "sha256": sha256_file(runner_source_path)},
            "qa_validator": {"path": str(validator_path), "sha256": sha256_file(validator_path)},
        },
        "detector_runtime": {
            "detector_id": "RT-DETRv2-R18VD",
            "repository_path": export_config["detector"]["repository_path"],
            "repository_commit": REPOSITORY_COMMIT,
            "checkpoint_state_key": "ema.module",
            "strict_checkpoint_load": True,
            "device": "cuda:0",
            "batch_size": 1,
            "eval": True,
            "inference_mode": True,
            "autocast": False,
            "TF32": False,
            "model_dtype": "float32",
            "preprocess": export_config["preprocessing"],
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "numpy": np.__version__,
            "pyarrow": pa.__version__,
            "pillow": importlib.metadata.version("Pillow"),
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name(torch.device("cuda:0")) if torch.cuda.is_available() else None,
            "gpu_driver": nvidia_driver_version(),
            "dependency_fallback_site_packages": (
                str(dependency_fallback_path) if dependency_fallback_path else None
            ),
            "reference_runtime": expected_runtime,
            "reference_runtime_exact_match": actual_runtime == expected_runtime,
        },
        "serialization": {
            "candidate_pattern": "candidate/VAL2017_candidates_{ordinal:04d}.parquet",
            "native_pattern": "native/VAL2017_native_{ordinal:04d}.npz",
            "native_index_pattern": "native/VAL2017_native_index_{ordinal:04d}.parquet",
            "shard_metadata_pattern": "shards/VAL2017_shard_{ordinal:04d}.json",
            "native_keys": list(NATIVE_KEYS),
            "candidate_rows_per_full_shard": 150000,
            "native_rows_per_full_shard": 150000,
            "raw_hypotheses_per_image": 2400,
            "stored_candidates_per_image": 300,
            "m11_consumed_prefix": 100,
        },
        "resume_policy": {
            "enabled": True,
            "skip_only_when_committed_shard_metadata_and_all_file_hashes_match": True,
            "incomplete_staging_shard_is_rebuilt": True,
            "committed_shard_is_never_overwritten": True,
        },
        "determinism": {
            "selection_namespace": DETERMINISM_NAMESPACE,
            "sample_count": DETERMINISM_SAMPLE_COUNT,
            "selection_rule": "sort SHA256(namespace+canonical_image_id), canonical_image_id; take first 20",
            "repeat_forward_count": 20,
            "comparison": "exact stored native arrays, candidate semantic records, order and record IDs",
            "sample_image_ids": determinism_image_ids,
        },
        "expected_counts": {
            "images": 5000,
            "shards": 10,
            "queries_per_image": 300,
            "native_rows": 1500000,
            "raw_hypotheses_per_image": 2400,
            "raw_hypotheses_total": 12000000,
            "stored_candidates_per_image": 300,
            "candidate_rows": 1500000,
            "m11_top100_rows": 500000,
            "canonical_forwards": 5000,
            "determinism_repeat_forwards": 20,
            "clean_run_total_forwards": 5020,
        },
        "publication_policy": {
            "validator_interface": "validate_export(staging_root: Path, run_config_path: Path)->dict",
            "requires_all_checks_pass_true": True,
            "final_logical_commit_marker": "CANONICAL_EXPORT_COMPLETE.json",
            "refuse_overwrite": True,
        },
        "forbidden_inputs": [
            "annotation_path",
            "ground_truth_path",
            "evaluation_result_path",
            "TEST",
            "RESERVE",
            "old_holdout",
        ],
        "forbidden_operations": [
            "allocator inference or training",
            "matching",
            "AP/AR/Coverage/QUALITY evaluation",
            "NMS",
            "thresholding",
            "query deduplication",
            "score/box/class modification",
        ],
        "dataset_root": str(dataset_root),
    }


def verify_preflight(run_config: dict[str, Any], export_config: dict[str, Any], identity: dict[str, Any]) -> list[dict[str, str]]:
    for name, binding in run_config["input_bindings"].items():
        path = Path(binding["path"])
        actual = sha256_file(path)
        if actual != binding["sha256"]:
            raise ValueError(f"FROZEN_INPUT_SHA_MISMATCH: {name}: {actual}")
    if compute_candidate_asset_id(identity) != CANDIDATE_ASSET_ID:
        raise ValueError("CANDIDATE_ASSET_ID_RECOMPUTE_FAIL")
    if identity["split_assets"][SPLIT]["candidate_asset_id"] != CANDIDATE_ASSET_ID:
        raise ValueError("CANDIDATE_ASSET_ID_CONTRACT_FAIL")
    if export_config["dataset_version"] != DATASET_VERSION or SPLIT not in export_config["allowed_splits"]:
        raise ValueError("EXPORT_CONFIG_DATASET_OR_SPLIT_FAIL")
    if export_config["serialization"]["shard_images"] != IMAGES_PER_SHARD:
        raise ValueError("EXPORT_CONFIG_SHARD_SIZE_FAIL")
    if run_config["detector_runtime"].get("reference_runtime_exact_match") is not True:
        raise ValueError(
            "REFERENCE_RUNTIME_MISMATCH: "
            f"expected={run_config['detector_runtime'].get('reference_runtime')}"
        )
    output_root = Path(run_config["output_root"])
    final_targets = (
        "candidate",
        "native",
        "shards",
        "manifest",
        "candidate_manifest.csv",
        "native_manifest.csv",
        "image_manifest.csv",
        "shard_manifest.csv",
        "native_state_index.parquet",
        "determinism_report.json",
        "runtime_summary.csv",
        "sha256_ledger.csv",
        "qa_validation_result.json",
        "qa_report.md",
        "CANONICAL_EXPORT_QA_REPORT.md",
        "COCO_CANONICAL_EXPORT_REPORT.md",
        "CANONICAL_EXPORT_COMPLETE.json",
    )
    collisions = [name for name in final_targets if (output_root / name).exists()]
    if collisions:
        raise FileExistsError(f"CANONICAL_OUTPUT_COLLISION: {collisions}")
    manifest_path = Path(run_config["input_bindings"]["image_manifest"]["path"])
    rows = load_csv_exact(manifest_path, IMAGE_MANIFEST_COLUMNS)
    if len(rows) != EXPECTED_IMAGE_COUNT:
        raise ValueError(f"IMAGE_MANIFEST_COUNT_FAIL: {len(rows)}")
    seen: set[str] = set()
    for index, row in enumerate(rows):
        if row["dataset_version"] != DATASET_VERSION or row["split"] != SPLIT:
            raise ValueError(f"IMAGE_MANIFEST_IDENTITY_FAIL: {index}")
        expected_id = f"coco2017:VAL2017:{int(row['coco_image_id']):012d}"
        if row["canonical_image_id"] != expected_id or row["file_name"] != f"{int(row['coco_image_id']):012d}.jpg":
            raise ValueError(f"CANONICAL_IMAGE_ID_FAIL: {index}")
        if row["canonical_image_id"] in seen:
            raise ValueError(f"DUPLICATE_CANONICAL_IMAGE_ID: {row['canonical_image_id']}")
        seen.add(row["canonical_image_id"])
        if not row["relative_path"].startswith("val2017/") or Path(row["relative_path"]).is_absolute():
            raise ValueError(f"UNSAFE_RELATIVE_IMAGE_PATH: {index}")
        if not (len(row["image_sha256"]) == 64 and all(c in "0123456789abcdef" for c in row["image_sha256"])):
            raise ValueError(f"IMAGE_SHA_FORMAT_FAIL: {index}")
        # The frozen runner contract requires every image byte identity to be
        # checked before model load.  Forward-time and validator readback
        # intentionally repeat this check at their own trust boundaries.
        image = load_rgb_verified(Path(run_config["dataset_root"]), row)
        image.close()
    return rows


def make_final_image_manifest(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        result.append(
            {
                "dataset_version": row["dataset_version"],
                "split": row["split"],
                "manifest_index": index,
                "coco_image_id": int(row["coco_image_id"]),
                "canonical_image_id": row["canonical_image_id"],
                "relative_path": row["relative_path"],
                "file_name": row["file_name"],
                "width": int(row["width"]),
                "height": int(row["height"]),
                "image_sha256": row["image_sha256"],
                "image_archive_sha256": row["image_archive_sha256"],
                "shard_id": f"VAL2017_shard_{index // IMAGES_PER_SHARD:04d}",
                "candidate_asset_id": CANDIDATE_ASSET_ID,
            }
        )
    return result


def execute(args: argparse.Namespace) -> dict[str, Any]:
    global CURRENT_STAGE
    output_root = args.output_root.resolve()
    freeze_root = args.freeze_root.resolve(strict=True)
    dataset_root = args.dataset_root.resolve(strict=True)
    validator_path = args.validator.resolve(strict=True)
    detector_snapshot_path = args.detector_snapshot.resolve(strict=True)
    working_root = output_root / "working"
    staging_root = working_root / ("staging_VAL2017_" + CANDIDATE_ASSET_ID[:12])
    run_config_path = working_root / "run_config.json"
    output_root.mkdir(parents=True, exist_ok=True)
    working_root.mkdir(parents=True, exist_ok=True)
    for name in ("candidate", "native", "shards", "manifest"):
        (staging_root / name).mkdir(parents=True, exist_ok=True)

    CURRENT_STAGE = "frozen_input_preflight"
    schema_path = freeze_root / SCHEMA_FILE
    export_config_path = freeze_root / EXPORT_CONFIG_FILE
    identity_path = freeze_root / IDENTITY_FILE
    if sha256_file(schema_path) != SCHEMA_SHA or sha256_file(export_config_path) != EXPORT_CONFIG_SHA:
        raise ValueError("FROZEN_CONTRACT_SHA_MISMATCH")
    identity_sha = sha256_file(identity_path)
    if identity_sha != IDENTITY_SHA:
        raise ValueError("FROZEN_IDENTITY_CONTRACT_SHA_MISMATCH")
    export_config = read_json(export_config_path)
    identity = read_json(identity_path)
    dependency_fallback = (
        args.dependency_fallback.resolve(strict=True)
        if args.dependency_fallback
        else None
    )
    run_config = build_run_config(
        output_root,
        staging_root,
        freeze_root,
        dataset_root,
        validator_path,
        detector_snapshot_path,
        export_config,
        identity_sha,
        dependency_fallback,
    )
    if run_config_path.exists():
        previous = read_json(run_config_path)
        immutable_keys = (
            "contract_type",
            "output_root",
            "staging_root",
            "dataset_version",
            "split",
            "expected_image_count",
            "images_per_shard",
            "expected_shard_count",
            "candidate_asset_id",
            "input_bindings",
            "serialization",
            "determinism",
            "publication_policy",
            "detector_runtime",
        )
        if any(previous.get(key) != run_config.get(key) for key in immutable_keys):
            raise ValueError("RESUME_RUN_CONFIG_IDENTITY_MISMATCH")
        run_config = previous
    else:
        write_json_atomic(run_config_path, run_config)
    preflight_started = time.perf_counter()
    manifest_rows = verify_preflight(run_config, export_config, identity)
    preflight_image_seconds = time.perf_counter() - preflight_started
    if (output_root / "CANONICAL_EXPORT_COMPLETE.json").exists():
        raise FileExistsError("CANONICAL_EXPORT_ALREADY_COMPLETE")

    repo_path = Path(export_config["detector"]["repository_path"])
    repo_head = subprocess.run(
        [
            "git",
            "-c",
            "safe.directory=" + repo_path.as_posix(),
            "-C",
            str(repo_path),
            "rev-parse",
            "HEAD",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if repo_head != REPOSITORY_COMMIT:
        raise ValueError(f"DETECTOR_REPO_COMMIT_MISMATCH: {repo_head}")
    repo_status = subprocess.run(
        [
            "git",
            "-c",
            "safe.directory=" + repo_path.as_posix(),
            "-C",
            str(repo_path),
            "status",
            "--porcelain",
            "--untracked-files=no",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if repo_status:
        raise ValueError(f"DETECTOR_REPO_TRACKED_DIFF: {repo_status}")
    write_json_atomic(
        working_root / "preflight_report.json",
        {
            "status": "PREFLIGHT_PASS",
            "completed_before_model_load": True,
            "images_byte_sha_and_dimensions_verified": len(manifest_rows),
            "image_preflight_seconds": preflight_image_seconds,
            "repository_commit": repo_head,
            "repository_tracked_diff_count": 0,
            "frozen_binding_count": len(run_config["input_bindings"]),
            "candidate_asset_id": CANDIDATE_ASSET_ID,
            "annotation_or_GT_read": False,
            "completed_at_utc": utc_now(),
        },
    )

    CURRENT_STAGE = "detector_load"
    DetectorSnapshot = import_symbol(detector_snapshot_path, "lc_coco_canonical_detector_snapshot", "DetectorSnapshot")
    detector_config = {
        "repository_path": export_config["detector"]["repository_path"],
        "runtime_source_path": str(repo_path / "rtdetrv2_pytorch"),
        "checkpoint_path": export_config["detector"]["checkpoint_path"],
        "checkpoint_sha256": CHECKPOINT_SHA,
        "checkpoint_state_key": "ema.module",
        "model_config_path": export_config["detector"]["model_config_path"],
        "model_config_sha256": MODEL_CONFIG_SHA,
        "input_size": [640, 640],
        "device": "cuda:0",
        "deploy": True,
        "road8_full_class_indices": list(DETECTOR_CLASS_INDICES),
        "model_execution": export_config["model_execution"],
        "seed": export_config["model_execution"]["seed"],
        "optional_dependency_site_packages": str(dependency_fallback) if dependency_fallback else None,
    }
    load_started = time.perf_counter()
    detector = DetectorSnapshot(detector_config)
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_started
    write_json_atomic(
        working_root / "model_load_report.json",
        {
            "status": "MODEL_LOAD_PASS",
            "checkpoint_sha256": sha256_file(Path(export_config["detector"]["checkpoint_path"])),
            "model_config_sha256": sha256_file(Path(export_config["detector"]["model_config_path"])),
            "repository_commit": repo_head,
            "strict_checkpoint_load": True,
            "model_eval": not detector.model.training,
            "requires_grad_parameters": sum(parameter.requires_grad for parameter in detector.model.parameters()),
            "load_seconds": load_seconds,
            "runtime_metadata": detector.runtime_metadata,
        },
    )

    determinism_ids = {row["canonical_image_id"] for row in select_determinism_rows(manifest_rows)}
    shard_metadata: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    CURRENT_STAGE = "canonical_forward_and_shard_write"
    export_started = time.perf_counter()
    for ordinal in range(EXPECTED_SHARD_COUNT):
        paths = shard_paths(staging_root, ordinal)
        shard_id = f"VAL2017_shard_{ordinal:04d}"
        first_index = ordinal * IMAGES_PER_SHARD
        last_index = first_index + IMAGES_PER_SHARD - 1
        selected_rows = manifest_rows[first_index : last_index + 1]
        if paths["metadata"].is_file():
            metadata = verify_resumable_shard(paths["metadata"], staging_root, ordinal)
            shard_metadata.append(metadata)
            print(f"[{utc_now()}] resume verified {shard_id} ({ordinal + 1}/{EXPECTED_SHARD_COUNT})", flush=True)
            continue
        purge_incomplete_shard(paths)
        print(f"[{utc_now()}] start {shard_id} images {first_index}..{last_index}", flush=True)
        candidates: list[dict[str, Any]] = []
        native_by_image: list[dict[str, np.ndarray]] = []
        image_ids: list[str] = []
        deterministic_baselines: list[dict[str, Any]] = []
        shard_started = time.perf_counter()
        forward_seconds = 0.0
        for local_index, row in enumerate(selected_rows):
            candidate_rows, native_arrays, elapsed = forward_one(detector, dataset_root, row)
            forward_seconds += elapsed
            candidates.extend(candidate_rows)
            native_by_image.append(native_arrays)
            image_ids.append(row["canonical_image_id"])
            if row["canonical_image_id"] in determinism_ids:
                deterministic_baselines.append(
                    {
                        "manifest_index": first_index + local_index,
                        "canonical_image_id": row["canonical_image_id"],
                        "shard_id": shard_id,
                        "native_content_sha256": array_bundle_hash(native_arrays),
                        "candidate_content_sha256": canonical_json_hash(candidate_rows),
                        "candidate_record_ids": [record["candidate_record_id"] for record in candidate_rows],
                        "candidate_order": [
                            [record["road8_rank"], record["query_index"], record["predicted_road8_class_id"], record["source_order"]]
                            for record in candidate_rows
                        ],
                    }
                )
            if (local_index + 1) % 50 == 0:
                print(
                    f"[{utc_now()}] {shard_id} {local_index + 1}/{IMAGES_PER_SHARD} "
                    f"global={first_index + local_index + 1}/{EXPECTED_IMAGE_COUNT}",
                    flush=True,
                )
        torch.cuda.synchronize()
        candidate_info = write_candidate_shard(paths["candidate"], candidates)
        native_info = write_native_shard(paths["native"], image_ids, native_by_image)
        index_info = write_native_index(paths["index"], image_ids, shard_id)
        total_seconds = time.perf_counter() - shard_started
        metadata = {
            "contract_type": "COCO2017_VAL2017_CANONICAL_SHARD_V1",
            "status": "SHARD_COMMITTED",
            "dataset_version": DATASET_VERSION,
            "split": SPLIT,
            "candidate_asset_id": CANDIDATE_ASSET_ID,
            "shard_id": shard_id,
            "shard_ordinal": ordinal,
            "first_manifest_index": first_index,
            "last_manifest_index": last_index,
            "first_image_id": image_ids[0],
            "last_image_id": image_ids[-1],
            "image_count": len(image_ids),
            "candidate": candidate_info,
            "native": native_info,
            "native_index": index_info,
            "invariants": {
                "raw_hypotheses_per_image": 2400,
                "stored_candidates_per_image": 300,
                "native_rows_per_image": 300,
                "candidate_to_native_join_key": ["canonical_image_id", "query_index"],
                "image_sha_verified_count": 500,
                "image_dimension_verified_count": 500,
                "annotation_or_GT_read": False,
            },
            "determinism_baselines": deterministic_baselines,
            "runner_source_sha256": sha256_file(Path(__file__).resolve()),
            "detector_forward_count": 500,
            "detector_forward_count_cumulative_diagnostic": detector.forward_count,
            "runtime": {
                "forward_and_input_seconds": forward_seconds,
                "total_shard_seconds": total_seconds,
                "seconds_per_image": total_seconds / len(image_ids),
            },
            "completed_at_utc": utc_now(),
        }
        write_json_atomic(paths["metadata"], metadata)
        shard_metadata.append(metadata)
        runtime_rows.append(
            {
                "shard_id": shard_id,
                "image_count": len(image_ids),
                "forward_and_input_seconds": forward_seconds,
                "total_shard_seconds": total_seconds,
                "seconds_per_image": total_seconds / len(image_ids),
                "resumed": False,
            }
        )
        del candidates, native_by_image
        print(f"[{utc_now()}] committed {shard_id} in {total_seconds:.3f}s", flush=True)

    if len(shard_metadata) != EXPECTED_SHARD_COUNT:
        raise RuntimeError("SHARD_COUNT_FAIL")
    shard_metadata.sort(key=lambda item: item["shard_ordinal"])

    CURRENT_STAGE = "determinism_repeat"
    baseline_by_id: dict[str, dict[str, Any]] = {}
    for metadata in shard_metadata:
        for item in metadata.get("determinism_baselines", []):
            baseline_by_id[item["canonical_image_id"]] = item
    deterministic_rows = select_determinism_rows(manifest_rows)
    if set(baseline_by_id) != {row["canonical_image_id"] for row in deterministic_rows}:
        raise ValueError("DETERMINISM_BASELINE_SET_FAIL")
    repeat_records: list[dict[str, Any]] = []
    repeat_started = time.perf_counter()
    for row in deterministic_rows:
        candidate_rows, native_arrays, _ = forward_one(detector, dataset_root, row)
        baseline = baseline_by_id[row["canonical_image_id"]]
        native_hash = array_bundle_hash(native_arrays)
        candidate_hash = canonical_json_hash(candidate_rows)
        record_ids = [record["candidate_record_id"] for record in candidate_rows]
        order = [
            [record["road8_rank"], record["query_index"], record["predicted_road8_class_id"], record["source_order"]]
            for record in candidate_rows
        ]
        record = {
            "manifest_index": baseline["manifest_index"],
            "canonical_image_id": row["canonical_image_id"],
            "shard_id": baseline["shard_id"],
            "first_native_content_sha256": baseline["native_content_sha256"],
            "repeat_native_content_sha256": native_hash,
            "native_exact": native_hash == baseline["native_content_sha256"],
            "first_candidate_content_sha256": baseline["candidate_content_sha256"],
            "repeat_candidate_content_sha256": candidate_hash,
            "candidate_exact": candidate_hash == baseline["candidate_content_sha256"],
            "record_ids_exact": record_ids == baseline["candidate_record_ids"],
            "sorting_exact": order == baseline["candidate_order"],
        }
        record["all_exact"] = all(
            record[key] for key in ("native_exact", "candidate_exact", "record_ids_exact", "sorting_exact")
        )
        repeat_records.append(record)
    torch.cuda.synchronize()
    repeat_seconds = time.perf_counter() - repeat_started
    if not all(record["all_exact"] for record in repeat_records):
        raise ValueError("DETERMINISM_REPEAT_FAIL")
    determinism_report = {
        "contract_type": "COCO2017_VAL2017_CANONICAL_DETERMINISM_V1",
        "status": "DETERMINISM_PASS",
        "selection_namespace": DETERMINISM_NAMESPACE,
        "sample_count": DETERMINISM_SAMPLE_COUNT,
        "sample_image_ids": [row["canonical_image_id"] for row in deterministic_rows],
        "comparison": "exact stored native arrays, candidate records, record IDs and frozen sorting identity",
        "records": repeat_records,
        "all_checks_pass": True,
    }
    write_json_atomic(staging_root / "determinism_report.json", determinism_report)

    CURRENT_STAGE = "manifest_materialization"
    candidate_manifest_rows, native_manifest_rows, shard_manifest_rows = derive_manifest_rows(shard_metadata)
    final_image_rows = make_final_image_manifest(manifest_rows)
    write_csv_atomic(staging_root / "candidate_manifest.csv", CANDIDATE_MANIFEST_COLUMNS, candidate_manifest_rows)
    write_csv_atomic(staging_root / "native_manifest.csv", NATIVE_MANIFEST_COLUMNS, native_manifest_rows)
    write_csv_atomic(staging_root / "image_manifest.csv", FINAL_IMAGE_MANIFEST_COLUMNS, final_image_rows)
    write_csv_atomic(staging_root / "shard_manifest.csv", SHARD_MANIFEST_COLUMNS, shard_manifest_rows)
    global_native_index = write_global_native_index(staging_root)
    for name in ("candidate_manifest.csv", "native_manifest.csv", "image_manifest.csv", "shard_manifest.csv"):
        shutil.copyfile(staging_root / name, staging_root / "manifest" / name)

    total_export_seconds = time.perf_counter() - export_started
    runtime_rows.extend(
        {
            "shard_id": metadata["shard_id"],
            "image_count": metadata["image_count"],
            "forward_and_input_seconds": "",
            "total_shard_seconds": "",
            "seconds_per_image": "",
            "resumed": True,
        }
        for metadata in shard_metadata
        if not any(row["shard_id"] == metadata["shard_id"] for row in runtime_rows)
    )
    write_csv_atomic(
        working_root / "runtime_records.csv",
        ("shard_id", "image_count", "forward_and_input_seconds", "total_shard_seconds", "seconds_per_image", "resumed"),
        runtime_rows,
    )
    write_csv_atomic(
        staging_root / "runtime_summary.csv",
        ("metric", "value", "unit", "scope"),
        [
            {"metric": "model_load", "value": load_seconds, "unit": "seconds", "scope": "one load"},
            {"metric": "export_wall_current_invocation", "value": total_export_seconds, "unit": "seconds", "scope": "resume-aware invocation"},
            {"metric": "preflight_all_image_bytes_and_dimensions", "value": preflight_image_seconds, "unit": "seconds", "scope": "5000 images before model load"},
            {"metric": "determinism_repeat", "value": repeat_seconds, "unit": "seconds", "scope": "20 images"},
            {"metric": "detector_forward_count_current_invocation", "value": detector.forward_count, "unit": "forwards", "scope": "batch_size_1"},
            {"metric": "canonical_image_count", "value": EXPECTED_IMAGE_COUNT, "unit": "images", "scope": "VAL2017"},
        ],
    )
    write_json_atomic(
        staging_root / "manifest" / "runtime_environment.json",
        {
            "contract_type": "COCO2017_VAL2017_CANONICAL_RUNTIME_ENVIRONMENT_V1",
            "recorded_at_utc": utc_now(),
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torchvision": importlib.metadata.version("torchvision"),
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name(torch.device("cuda:0")),
            "gpu_driver": nvidia_driver_version(),
            "numpy": np.__version__,
            "pyarrow": pa.__version__,
            "Pillow": importlib.metadata.version("Pillow"),
            "batch_size": 1,
            "model_dtype": "float32",
            "autocast": False,
            "TF32": False,
            "cudnn_benchmark": False,
            "eval": True,
            "inference_mode": True,
            "no_grad": True,
            "strict_checkpoint_load": True,
            "repository_commit": REPOSITORY_COMMIT,
            "repository_tracked_diff_count": 0,
            "checkpoint_sha256": CHECKPOINT_SHA,
            "model_config_sha256": MODEL_CONFIG_SHA,
            "python_dont_write_bytecode": bool(sys.dont_write_bytecode),
        },
    )
    write_json_atomic(
        staging_root / "manifest" / "sha256_ledger_scope.json",
        {
            "contract_type": "COCO2017_VAL2017_CANONICAL_SHA_LEDGER_SCOPE_V1",
            "ledger": "sha256_ledger.csv",
            "covered": "all scientific shards, shard metadata, four manifest mirrors, runtime evidence, determinism report, runtime summary and run_config",
            "excluded_to_avoid_self_reference": [
                "sha256_ledger.csv",
                "manifest/asset_manifest.json",
                "qa_validation_result.json",
                "qa_report.md",
                "CANONICAL_EXPORT_QA_REPORT.md",
                "COCO_CANONICAL_EXPORT_REPORT.md",
                "CANONICAL_EXPORT_COMPLETE.json",
            ],
            "post_publication_rule": "all ledger rows are rehashed after promotion and before the completion marker",
        },
    )
    create_sha_ledger(staging_root, run_config_path)

    CURRENT_STAGE = "independent_qa"
    validate_export = import_symbol(validator_path, "lc_coco_canonical_export_validator", "validate_export")
    validator_result = validate_export(staging_root=staging_root, run_config_path=run_config_path)
    if not isinstance(validator_result, dict) or validator_result.get("all_checks_pass") is not True:
        raise ValueError(f"INDEPENDENT_QA_FAIL: {validator_result!r}")
    write_json_atomic(staging_root / "qa_validation_result.json", validator_result)
    verify_sha_ledger(staging_root, staging_root / "sha256_ledger.csv")
    write_post_qa_metadata(
        staging_root,
        run_config_path,
        validator_result,
        shard_metadata,
        global_native_index,
    )

    CURRENT_STAGE = "atomic_publication"
    marker = publish_after_qa(output_root, staging_root, validator_result, run_config_path)
    print(f"[{utc_now()}] canonical export published: {marker['status']}", flush=True)
    return {
        "execution_status": "COMPLETE",
        "export_status": "COCO_CANONICAL_EXPORT_COMPLETE",
        "qa_status": "PASS",
        "candidate_asset_id": CANDIDATE_ASSET_ID,
        "image_count": EXPECTED_IMAGE_COUNT,
        "shard_count": EXPECTED_SHARD_COUNT,
        "detector_forward_count_current_invocation": detector.forward_count,
        "completion_marker": str(output_root / "CANONICAL_EXPORT_COMPLETE.json"),
    }


def write_failure(output_root: Path, stage: str, error: BaseException) -> None:
    working = output_root / "working"
    working.mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        working / "failure_detail.json",
        {
            "execution_status": "BLOCKED",
            "export_status": "NOT_PUBLISHED",
            "stage": stage,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "timestamp_utc": utc_now(),
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Frozen COCO2017 VAL2017 canonical export")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT_DEFAULT)
    parser.add_argument("--freeze-root", type=Path, default=FREEZE_ROOT_DEFAULT)
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT_DEFAULT)
    parser.add_argument("--detector-snapshot", type=Path, default=DETECTOR_SNAPSHOT_DEFAULT)
    parser.add_argument("--validator", type=Path, default=VALIDATOR_DEFAULT)
    parser.add_argument("--dependency-fallback", type=Path, default=DEPENDENCY_FALLBACK_DEFAULT)
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    try:
        result = execute(parsed)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    except BaseException as failure:
        write_failure(parsed.output_root.resolve(), CURRENT_STAGE, failure)
        traceback.print_exc()
        raise SystemExit(1)
