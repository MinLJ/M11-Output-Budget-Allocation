"""Read-only validator for the COCO2017 Road8 canonical export.

This module validates an already serialized VAL2017 staging (or published)
root.  It never imports detector code, loads a checkpoint, reads annotations or
GT, runs inference, performs matching, or writes into the asset root.

The runner-facing API is::

    validate_export(staging_root: Path, run_config_path: Path) -> dict

The returned object is JSON serializable.  A caller must publish nothing unless
``all_checks_pass`` is exactly ``True``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import sys
from typing import Any, Iterable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
import torch


DATASET_VERSION = "COCO2017-Road8-v1"
SPLIT = "VAL2017"
DETECTOR_ID = "RT-DETRv2-R18VD"
EXPECTED_IMAGE_COUNT = 5_000
IMAGES_PER_SHARD = 500
EXPECTED_SHARD_COUNT = 10
QUERIES_PER_IMAGE = 300
ROAD8_CLASS_COUNT = 8
CANDIDATES_PER_IMAGE = 300
RAW_HYPOTHESES_PER_IMAGE = 2_400

EXPECTED_CANDIDATE_ASSET_ID = (
    "5ff6497669d5b02c888f90451882d3c8e594b74a3aedd136639a9c0c8621c707"
)
EXPECTED_SCHEMA_SHA256 = (
    "e977e02fe638a8ac44e98919d76225306eb88fa7385d3ec0793342112ec8c53b"
)
EXPECTED_EXPORT_CONFIG_SHA256 = (
    "f6a91e9d6879357fc10095c182689e43ee73bca2c10ec6926f93960cc68d98f2"
)
EXPECTED_IDENTITY_CONTRACT_SHA256 = (
    "3776105cf88fd05c4f70a2de82ea610c3e98ef8bf47b7d80e1a4f18f5b5e7a1b"
)
EXPECTED_RUNNER_CONTRACT_SHA256 = (
    "8c67226cd7530d982b99589ef8be0d21955d79a1098fc81d645f6567cb320f27"
)
EXPECTED_QA_SPEC_SHA256 = (
    "6ba34bc20d694533892147868020a0838872caf677626e0a07626f0ec1287e9c"
)
EXPECTED_DETECTOR_SNAPSHOT_SHA256 = (
    "4714b925380ce62075318331d4078ddc827c25c217788cbb81993b2c75efd1b0"
)
EXPECTED_SPLIT_MANIFEST_SHA256 = (
    "9e0c1ac3d4a2c3b8e61246f8ec147745ca937c8830dc88b98bd4879342503ad2"
)
EXPECTED_CHECKPOINT_SHA256 = (
    "2ace52184b620204004509b72752ac7bfe64aadaf7fc1d076b18df8ab5a5c77e"
)
EXPECTED_MODEL_CONFIG_SHA256 = (
    "3fc6fda05f01ac16a90cf4116bd3793682fa780979ac37b09f459a66bd21cc54"
)

FREEZE_ROOT = Path(r"D:\AOP_DETR\research_LC_ALLOC_COCO_EXPORT_FREEZE")
SCHEMA_PATH = FREEZE_ROOT / "COCO2017_Road8_CandidateSchema_v1.json"
EXPORT_CONFIG_PATH = (
    FREEZE_ROOT / "COCO2017_Road8_RTDETRv2_R18VD_EXPORT_CONFIG.json"
)
IDENTITY_CONTRACT_PATH = FREEZE_ROOT / "COCO2017_candidate_asset_identity.json"
SOURCE_IMAGE_MANIFEST_PATH = Path(
    r"D:\AOP_DETR\research_LC_ALLOC_COCO_EXECUTION_GATE\manifests\VAL2017_images.csv"
)

ROAD8_NAMES = np.asarray(
    ("person", "bicycle", "car", "motorcycle", "bus", "train", "truck", "traffic light"),
    dtype=object,
)
COCO_CATEGORY_IDS = np.asarray((1, 2, 3, 4, 6, 7, 8, 10), dtype=np.int16)
DETECTOR_CLASS_INDICES = np.asarray((0, 1, 2, 3, 5, 6, 7, 9), dtype=np.int16)
HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
NATIVE_KEY_ORDER = (
    "image_ids",
    "query_index",
    "l3_road8_logits",
    "l3_road8_scores",
    "l3_query_embedding",
    "l3_pred_box_cxcywh",
)


class ValidationFailure(RuntimeError):
    """A fail-closed, expected validation failure."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        expected: Any | None = None,
        actual: Any | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.expected = expected
        self.actual = actual


def require(
    condition: bool,
    code: str,
    message: str,
    *,
    expected: Any | None = None,
    actual: Any | None = None,
) -> None:
    if not condition:
        raise ValidationFailure(code, message, expected=expected, actual=actual)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    require(isinstance(value, dict), "SERIALIZED_READBACK_FAIL", f"JSON root is not an object: {path}")
    return value


def load_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames is not None, "SERIALIZED_READBACK_FAIL", f"CSV has no header: {path}")
        return list(reader.fieldnames), list(reader)


def int_field(row: dict[str, str], name: str, source: str) -> int:
    try:
        return int(row[name])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValidationFailure(
            "SERIALIZED_READBACK_FAIL", f"invalid integer {source}.{name}", actual=row.get(name)
        ) from exc


def length_prefixed_record_id(asset_id: str, image_id: str, road8_rank: int) -> str:
    fields = (
        asset_id.encode("utf-8"),
        image_id.encode("utf-8"),
        str(road8_rank).encode("ascii"),
    )
    payload = b"".join(len(field).to_bytes(8, "big") + field for field in fields)
    return hashlib.sha256(payload).hexdigest()


def candidate_asset_id(
    dataset_version: str,
    split_manifest_sha256: str,
    detector_id: str,
    checkpoint_sha256: str,
    export_config_sha256: str,
    schema_sha256: str,
) -> str:
    # Deliberately no delimiter, trim, case folding or Unicode normalization.
    payload = (
        dataset_version
        + split_manifest_sha256
        + detector_id
        + checkpoint_sha256
        + export_config_sha256
        + schema_sha256
    )
    return sha256_text(payload)


def array_bundle_hash(arrays: dict[str, np.ndarray]) -> str:
    """Frozen native semantic hash used by the canonical runner."""
    digest = hashlib.sha256()
    for name in sorted(arrays):
        array = np.ascontiguousarray(arrays[name])
        tokens = (
            name,
            array.dtype.str,
            json.dumps(list(array.shape), separators=(",", ":")),
        )
        for token in tokens:
            encoded = token.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
        raw = array.tobytes(order="C")
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def canonical_table_rows_hash(table: pa.Table) -> str:
    """Hash canonical JSON for ``table.to_pylist()`` without materializing it all."""
    digest = hashlib.sha256()
    digest.update(b"[")
    first = True
    for batch in table.to_batches(max_chunksize=4096):
        for row in batch.to_pylist():
            if first:
                first = False
            else:
                digest.update(b",")
            payload = json.dumps(
                row,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
            digest.update(payload)
    digest.update(b"]")
    return digest.hexdigest()


def safe_artifact_path(root: Path, relative: str) -> Path:
    require(relative == relative.replace("\\", "/"), "MANIFEST_PATH_FAIL", "backslash in artifact path", actual=relative)
    pure = PurePosixPath(relative)
    require(not pure.is_absolute(), "MANIFEST_PATH_FAIL", "absolute artifact path", actual=relative)
    require(".." not in pure.parts and "." not in pure.parts, "MANIFEST_PATH_FAIL", "unsafe artifact path", actual=relative)
    candidate = root.joinpath(*pure.parts)
    resolved_root = root.resolve()
    resolved = candidate.resolve()
    require(
        resolved == resolved_root or resolved_root in resolved.parents,
        "MANIFEST_PATH_FAIL",
        "artifact escapes root",
        actual=relative,
    )
    return candidate


def require_exact_columns(actual: Iterable[str], expected: list[str], source: str) -> None:
    actual_list = list(actual)
    require(
        actual_list == expected,
        "SERIALIZED_SCHEMA_FAIL",
        f"column mismatch: {source}",
        expected=expected,
        actual=actual_list,
    )


def _validate_sha_ledger(
    root: Path, run_config_path: Path
) -> dict[str, int]:
    path = root / "sha256_ledger.csv"
    require(path.is_file(), "SHA_MISMATCH", "sha256_ledger.csv missing")
    fields, rows = load_csv(path)
    require_exact_columns(fields, ["role", "path", "bytes", "sha256"], "sha256_ledger.csv")
    require(rows, "SHA_MISMATCH", "empty SHA ledger")
    seen: set[str] = set()
    mismatches = 0
    for row in rows:
        identity = row["path"]
        require(identity not in seen, "SHA_MISMATCH", "duplicate ledger path", actual=identity)
        seen.add(identity)
        listed = Path(identity)
        artifact = listed if listed.is_absolute() else safe_artifact_path(root, identity)
        expected_sha = row["sha256"]
        require(HEX64_RE.fullmatch(expected_sha) is not None, "SHA_MISMATCH", "invalid ledger SHA", actual=expected_sha)
        expected_bytes = int_field(row, "bytes", "sha256_ledger")
        if (
            not artifact.is_file()
            or artifact.stat().st_size != expected_bytes
            or sha256_file(artifact) != expected_sha
        ):
            mismatches += 1
    require("sha256_ledger.csv" not in seen, "SHA_MISMATCH", "ledger must not self-reference")
    require(str(run_config_path) in seen, "SHA_MISMATCH", "run_config absent from ledger")
    required_relative = {
        *(f"candidate/{SPLIT}_candidates_{index:04d}.parquet" for index in range(10)),
        *(f"native/{SPLIT}_native_{index:04d}.npz" for index in range(10)),
        *(f"native/{SPLIT}_native_index_{index:04d}.parquet" for index in range(10)),
        *(f"shards/{SPLIT}_shard_{index:04d}.json" for index in range(10)),
        "candidate_manifest.csv",
        "native_manifest.csv",
        "image_manifest.csv",
        "shard_manifest.csv",
        "native_state_index.parquet",
        "determinism_report.json",
        "runtime_summary.csv",
        "manifest/candidate_manifest.csv",
        "manifest/native_manifest.csv",
        "manifest/image_manifest.csv",
        "manifest/shard_manifest.csv",
        "manifest/runtime_environment.json",
        "manifest/sha256_ledger_scope.json",
    }
    missing = sorted(required_relative - seen)
    require(not missing, "SHA_MISMATCH", "required artifact absent from ledger", actual=missing)
    require(mismatches == 0, "SHA_MISMATCH", "ledger byte/hash mismatch", actual=mismatches)
    return {"entries": len(rows), "mismatches": 0}


def _validate_global_native_index(root: Path, image_rows: list[dict[str, str]]) -> dict[str, Any]:
    path = root / "native_state_index.parquet"
    require(path.is_file(), "SERIALIZED_READBACK_FAIL", "global native_state_index.parquet missing")
    parquet_file = pq.ParquetFile(path)
    require(
        parquet_file.schema_arrow.remove_metadata() == native_index_arrow_schema(),
        "SERIALIZED_SCHEMA_FAIL",
        "global native index physical schema",
    )
    expected_rows = EXPECTED_IMAGE_COUNT * QUERIES_PER_IMAGE
    require(parquet_file.metadata.num_rows == expected_rows, "CANDIDATE_NATIVE_JOIN_FAIL", "global native index row count")
    cursor = 0
    for batch in parquet_file.iter_batches(batch_size=65536):
        table = pa.Table.from_batches([batch])
        count = table.num_rows
        positions = np.arange(cursor, cursor + count, dtype=np.int64)
        image_indices = positions // QUERIES_PER_IMAGE
        expected_ids = np.asarray(
            [image_rows[int(index)]["canonical_image_id"] for index in image_indices],
            dtype=object,
        )
        expected_queries = (positions % QUERIES_PER_IMAGE).astype(np.int32)
        expected_shards = np.asarray(
            [f"{SPLIT}_shard_{int(index) // IMAGES_PER_SHARD:04d}" for index in image_indices],
            dtype=object,
        )
        expected_offsets = (
            (image_indices % IMAGES_PER_SHARD) * QUERIES_PER_IMAGE
            + expected_queries.astype(np.int64)
        )
        require(np.array_equal(_column_strings(table, "canonical_image_id"), expected_ids), "CANDIDATE_NATIVE_JOIN_FAIL", "global index image order")
        require(np.array_equal(_column_numpy(table, "query_index", np.int32), expected_queries), "CANDIDATE_NATIVE_JOIN_FAIL", "global index query order")
        require(np.array_equal(_column_strings(table, "shard_id"), expected_shards), "CANDIDATE_NATIVE_JOIN_FAIL", "global index shard IDs")
        require(np.array_equal(_column_numpy(table, "row_offset", np.int64), expected_offsets), "CANDIDATE_NATIVE_JOIN_FAIL", "global index row offsets")
        cursor += count
    require(cursor == expected_rows, "CANDIDATE_NATIVE_JOIN_FAIL", "global index readback rows")
    return {"rows": cursor, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def candidate_arrow_schema() -> pa.Schema:
    nullable = False
    return pa.schema(
        [
            pa.field("dataset_version", pa.string(), nullable=nullable),
            pa.field("split", pa.string(), nullable=nullable),
            pa.field("coco_image_id", pa.int64(), nullable=nullable),
            pa.field("image_id", pa.string(), nullable=nullable),
            pa.field("canonical_image_id", pa.string(), nullable=nullable),
            pa.field("image_sha256", pa.string(), nullable=nullable),
            pa.field("candidate_asset_id", pa.string(), nullable=nullable),
            pa.field("detector_id", pa.string(), nullable=nullable),
            pa.field("checkpoint_sha256", pa.string(), nullable=nullable),
            pa.field("export_config_sha256", pa.string(), nullable=nullable),
            pa.field("schema_sha256", pa.string(), nullable=nullable),
            pa.field("candidate_record_id", pa.string(), nullable=nullable),
            pa.field("road8_rank", pa.int32(), nullable=nullable),
            pa.field("source_order", pa.int32(), nullable=nullable),
            pa.field("query_index", pa.int32(), nullable=nullable),
            pa.field("predicted_road8_class_id", pa.int16(), nullable=nullable),
            pa.field("predicted_road8_class_name", pa.string(), nullable=nullable),
            pa.field("predicted_coco_category_id", pa.int16(), nullable=nullable),
            pa.field("detector_class_index", pa.int16(), nullable=nullable),
            pa.field("score", pa.float64(), nullable=nullable),
            pa.field("bbox_x1", pa.float64(), nullable=nullable),
            pa.field("bbox_y1", pa.float64(), nullable=nullable),
            pa.field("bbox_x2", pa.float64(), nullable=nullable),
            pa.field("bbox_y2", pa.float64(), nullable=nullable),
            pa.field("bbox_cx", pa.float64(), nullable=nullable),
            pa.field("bbox_cy", pa.float64(), nullable=nullable),
            pa.field("bbox_w", pa.float64(), nullable=nullable),
            pa.field("bbox_h", pa.float64(), nullable=nullable),
            pa.field("bbox_xyxy", pa.list_(pa.float64(), 4), nullable=nullable),
            pa.field("bbox_cxcywh", pa.list_(pa.float64(), 4), nullable=nullable),
        ]
    )


def native_index_arrow_schema() -> pa.Schema:
    nullable = False
    return pa.schema(
        [
            pa.field("canonical_image_id", pa.string(), nullable=nullable),
            pa.field("query_index", pa.int32(), nullable=nullable),
            pa.field("shard_id", pa.string(), nullable=nullable),
            pa.field("row_offset", pa.int64(), nullable=nullable),
        ]
    )


CANDIDATE_MANIFEST_COLUMNS = [
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
]
NATIVE_MANIFEST_COLUMNS = [
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
]
IMAGE_MANIFEST_COLUMNS = [
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
]
SHARD_MANIFEST_COLUMNS = [
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
]
SOURCE_IMAGE_MANIFEST_COLUMNS = [
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
]


def _validate_frozen_inputs(run_config: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, str]]]:
    frozen_files = (
        (SCHEMA_PATH, EXPECTED_SCHEMA_SHA256, "SCHEMA_MISMATCH"),
        (EXPORT_CONFIG_PATH, EXPECTED_EXPORT_CONFIG_SHA256, "EXPORT_CONFIG_MISMATCH"),
        (IDENTITY_CONTRACT_PATH, EXPECTED_IDENTITY_CONTRACT_SHA256, "IDENTITY_CONTRACT_MISMATCH"),
        (SOURCE_IMAGE_MANIFEST_PATH, EXPECTED_SPLIT_MANIFEST_SHA256, "IMAGE_MANIFEST_MISMATCH"),
    )
    for path, expected_sha, code in frozen_files:
        require(path.is_file(), code, f"frozen input missing: {path}")
        actual_sha = sha256_file(path)
        require(actual_sha == expected_sha, code, f"frozen input SHA mismatch: {path}", expected=expected_sha, actual=actual_sha)

    schema_sidecar = (FREEZE_ROOT / "schema_sha256.txt").read_text(encoding="ascii").strip()
    config_sidecar = (FREEZE_ROOT / "export_config_sha256.txt").read_text(encoding="ascii").strip()
    require(schema_sidecar == EXPECTED_SCHEMA_SHA256, "SCHEMA_MISMATCH", "schema sidecar mismatch")
    require(config_sidecar == EXPECTED_EXPORT_CONFIG_SHA256, "EXPORT_CONFIG_MISMATCH", "config sidecar mismatch")

    export_config = load_json(EXPORT_CONFIG_PATH)
    identity_contract = load_json(IDENTITY_CONTRACT_PATH)
    require(export_config["dataset_version"] == DATASET_VERSION, "EXPORT_CONFIG_MISMATCH", "dataset version")
    require(export_config["detector"]["detector_id"] == DETECTOR_ID, "EXPORT_CONFIG_MISMATCH", "detector ID")
    require(export_config["detector"]["checkpoint_sha256"] == EXPECTED_CHECKPOINT_SHA256, "EXPORT_CONFIG_MISMATCH", "checkpoint binding")
    require(export_config["detector"]["model_config_sha256"] == EXPECTED_MODEL_CONFIG_SHA256, "EXPORT_CONFIG_MISMATCH", "model config binding")
    require(export_config["schema_binding"]["sha256"] == EXPECTED_SCHEMA_SHA256, "EXPORT_CONFIG_MISMATCH", "schema binding")
    split_cfg = export_config["split_manifest_registry"][SPLIT]
    require(split_cfg["sha256"] == EXPECTED_SPLIT_MANIFEST_SHA256, "EXPORT_CONFIG_MISMATCH", "split binding")
    require(split_cfg["image_count"] == EXPECTED_IMAGE_COUNT, "EXPORT_CONFIG_MISMATCH", "split count")

    identity_split = identity_contract["split_assets"][SPLIT]
    require(identity_split["candidate_asset_id"] == EXPECTED_CANDIDATE_ASSET_ID, "NONDETERMINISTIC_IDENTITY", "identity contract asset ID")
    recomputed_asset = candidate_asset_id(
        DATASET_VERSION,
        EXPECTED_SPLIT_MANIFEST_SHA256,
        DETECTOR_ID,
        EXPECTED_CHECKPOINT_SHA256,
        EXPECTED_EXPORT_CONFIG_SHA256,
        EXPECTED_SCHEMA_SHA256,
    )
    require(recomputed_asset == EXPECTED_CANDIDATE_ASSET_ID, "NONDETERMINISTIC_IDENTITY", "recomputed CandidateAssetID")

    require(run_config.get("dataset_version") == DATASET_VERSION, "RUN_CONFIG_MISMATCH", "run dataset version")
    require(run_config.get("split") == SPLIT, "RUN_CONFIG_MISMATCH", "run split")
    require(run_config.get("expected_image_count") == EXPECTED_IMAGE_COUNT, "RUN_CONFIG_MISMATCH", "run image count")
    require(run_config.get("images_per_shard") == IMAGES_PER_SHARD, "RUN_CONFIG_MISMATCH", "run shard size")
    require(run_config.get("expected_shard_count") == EXPECTED_SHARD_COUNT, "RUN_CONFIG_MISMATCH", "run shard count")
    require(run_config.get("candidate_asset_id") == EXPECTED_CANDIDATE_ASSET_ID, "RUN_CONFIG_MISMATCH", "run CandidateAssetID")

    bindings = run_config.get("input_bindings")
    require(isinstance(bindings, dict), "RUN_CONFIG_MISMATCH", "input_bindings missing")
    expected_binding_shas = {
        "candidate_schema": EXPECTED_SCHEMA_SHA256,
        "export_config": EXPECTED_EXPORT_CONFIG_SHA256,
        "image_manifest": EXPECTED_SPLIT_MANIFEST_SHA256,
        "checkpoint": EXPECTED_CHECKPOINT_SHA256,
        "model_config": EXPECTED_MODEL_CONFIG_SHA256,
        "candidate_asset_identity": EXPECTED_IDENTITY_CONTRACT_SHA256,
        "runner_contract": EXPECTED_RUNNER_CONTRACT_SHA256,
        "qa_spec": EXPECTED_QA_SPEC_SHA256,
        "detector_snapshot": EXPECTED_DETECTOR_SNAPSHOT_SHA256,
    }
    for key, expected_sha in expected_binding_shas.items():
        binding = bindings.get(key)
        require(isinstance(binding, dict), "RUN_CONFIG_MISMATCH", f"missing input binding {key}")
        require(binding.get("sha256") == expected_sha, "RUN_CONFIG_MISMATCH", f"binding SHA mismatch: {key}")
    identity_binding = bindings.get("candidate_asset_identity")
    require(isinstance(identity_binding, dict), "RUN_CONFIG_MISMATCH", "missing identity binding")
    require(identity_binding.get("sha256") == sha256_file(IDENTITY_CONTRACT_PATH), "RUN_CONFIG_MISMATCH", "identity binding SHA")
    # Re-hash every bound implementation/input file, including the checkpoint,
    # without importing or deserializing any of them.
    for key, binding in bindings.items():
        require(isinstance(binding, dict), "RUN_CONFIG_MISMATCH", f"invalid input binding {key}")
        bound_path = Path(str(binding.get("path", "")))
        bound_sha = binding.get("sha256")
        require(bound_path.is_file(), "RUN_CONFIG_MISMATCH", f"bound file missing: {key}", actual=str(bound_path))
        require(isinstance(bound_sha, str) and HEX64_RE.fullmatch(bound_sha) is not None, "RUN_CONFIG_MISMATCH", f"invalid bound SHA: {key}")
        actual_bound_sha = sha256_file(bound_path)
        require(actual_bound_sha == bound_sha, "RUN_CONFIG_MISMATCH", f"bound file SHA mismatch: {key}", expected=bound_sha, actual=actual_bound_sha)

    source_fields, source_rows = load_csv(SOURCE_IMAGE_MANIFEST_PATH)
    require_exact_columns(source_fields, SOURCE_IMAGE_MANIFEST_COLUMNS, "frozen image manifest")
    require(len(source_rows) == EXPECTED_IMAGE_COUNT, "IMAGE_MANIFEST_MISMATCH", "frozen image count")
    return export_config, source_rows


def _validate_output_image_manifest(
    root: Path, source_rows: list[dict[str, str]], dataset_root: Path
) -> list[dict[str, str]]:
    path = root / "image_manifest.csv"
    require(path.is_file(), "SERIALIZED_READBACK_FAIL", "image_manifest.csv missing")
    fields, rows = load_csv(path)
    require_exact_columns(fields, IMAGE_MANIFEST_COLUMNS, "image_manifest.csv")
    require(len(rows) == EXPECTED_IMAGE_COUNT, "IMAGE_MANIFEST_MISMATCH", "output image count")
    seen_coco: set[int] = set()
    seen_canonical: set[str] = set()
    for index, (row, source) in enumerate(zip(rows, source_rows)):
        expected_shard_id = f"{SPLIT}_shard_{index // IMAGES_PER_SHARD:04d}"
        require(int_field(row, "manifest_index", "image_manifest") == index, "IMAGE_MANIFEST_MISMATCH", "manifest index")
        for name in SOURCE_IMAGE_MANIFEST_COLUMNS:
            require(row[name] == source[name], "IMAGE_MANIFEST_MISMATCH", f"source field mismatch at {index}: {name}")
        require(row["shard_id"] == expected_shard_id, "IMAGE_MANIFEST_MISMATCH", "shard assignment")
        require(row["candidate_asset_id"] == EXPECTED_CANDIDATE_ASSET_ID, "IMAGE_MANIFEST_MISMATCH", "asset ID")
        coco_id = int_field(row, "coco_image_id", "image_manifest")
        canonical = row["canonical_image_id"]
        require(coco_id not in seen_coco, "IMAGE_MANIFEST_MISMATCH", "duplicate COCO image ID", actual=coco_id)
        require(canonical not in seen_canonical, "IMAGE_MANIFEST_MISMATCH", "duplicate canonical image ID", actual=canonical)
        seen_coco.add(coco_id)
        seen_canonical.add(canonical)
        relative = PurePosixPath(row["relative_path"])
        require(not relative.is_absolute() and ".." not in relative.parts and "." not in relative.parts, "IMAGE_MANIFEST_MISMATCH", "unsafe source image path", actual=row["relative_path"])
        image_path = dataset_root.joinpath(*relative.parts).resolve()
        require(dataset_root in image_path.parents and image_path.is_file(), "IMAGE_MANIFEST_MISMATCH", "source image missing or outside root", actual=str(image_path))
        actual_image_sha = sha256_file(image_path)
        require(actual_image_sha == row["image_sha256"], "IMAGE_MANIFEST_MISMATCH", "source image SHA mismatch", expected=row["image_sha256"], actual=actual_image_sha)
        with Image.open(image_path) as image:
            actual_size = image.size
        expected_size = (int_field(row, "width", "image_manifest"), int_field(row, "height", "image_manifest"))
        require(actual_size == expected_size, "IMAGE_MANIFEST_MISMATCH", "source image dimensions", expected=list(expected_size), actual=list(actual_size))
    return rows


def _read_artifact_manifests(root: Path) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    files = (
        ("candidate_manifest.csv", CANDIDATE_MANIFEST_COLUMNS),
        ("native_manifest.csv", NATIVE_MANIFEST_COLUMNS),
        ("shard_manifest.csv", SHARD_MANIFEST_COLUMNS),
    )
    loaded: list[list[dict[str, str]]] = []
    for filename, expected_columns in files:
        path = root / filename
        require(path.is_file(), "SERIALIZED_READBACK_FAIL", f"missing {filename}")
        columns, rows = load_csv(path)
        require_exact_columns(columns, expected_columns, filename)
        require(len(rows) == EXPECTED_SHARD_COUNT, "SERIALIZED_READBACK_FAIL", f"wrong row count: {filename}")
        loaded.append(rows)
    return loaded[0], loaded[1], loaded[2]


def _column_numpy(table: pa.Table, name: str, dtype: np.dtype[Any] | type[Any]) -> np.ndarray:
    return np.asarray(table[name].combine_chunks().to_numpy(zero_copy_only=False), dtype=dtype)


def _column_strings(table: pa.Table, name: str) -> np.ndarray:
    return np.asarray(table[name].combine_chunks().to_pylist(), dtype=object)


def _column_fixed_list(table: pa.Table, name: str, dtype: np.dtype[Any] | type[Any]) -> np.ndarray:
    return np.asarray(table[name].combine_chunks().to_pylist(), dtype=dtype)


def _validate_native(
    native_path: Path,
    index_path: Path,
    expected_image_ids: list[str],
    shard_id: str,
) -> tuple[dict[str, np.ndarray], str, int, float]:
    expected_rows = len(expected_image_ids) * QUERIES_PER_IMAGE
    expected_keys = set(NATIVE_KEY_ORDER)
    with np.load(native_path, allow_pickle=False) as archive:
        require(set(archive.files) == expected_keys, "SERIALIZED_SCHEMA_FAIL", "native NPZ keys", expected=sorted(expected_keys), actual=sorted(archive.files))
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}

    expected_specs = {
        "image_ids": ((expected_rows,), "U"),
        "query_index": ((expected_rows,), np.dtype(np.int32)),
        "l3_road8_logits": ((expected_rows, 8), np.dtype(np.float32)),
        "l3_road8_scores": ((expected_rows, 8), np.dtype(np.float32)),
        "l3_query_embedding": ((expected_rows, 256), np.dtype(np.float16)),
        "l3_pred_box_cxcywh": ((expected_rows, 4), np.dtype(np.float32)),
    }
    for name, (shape, dtype_or_kind) in expected_specs.items():
        value = arrays[name]
        require(value.shape == shape, "QUERY_SHAPE_MISMATCH", f"native shape {name}", expected=list(shape), actual=list(value.shape))
        if isinstance(dtype_or_kind, str):
            require(value.dtype.kind == dtype_or_kind, "SERIALIZED_SCHEMA_FAIL", f"native dtype kind {name}", expected=dtype_or_kind, actual=value.dtype.str)
        else:
            require(value.dtype == dtype_or_kind, "SERIALIZED_SCHEMA_FAIL", f"native dtype {name}", expected=str(dtype_or_kind), actual=str(value.dtype))
            require(np.isfinite(value).all(), "NONFINITE_NATIVE_STATE", f"non-finite native values: {name}")
        require(value.dtype.kind != "O", "SERIALIZED_SCHEMA_FAIL", f"object array forbidden: {name}")

    expected_ids = np.repeat(np.asarray(expected_image_ids, dtype=arrays["image_ids"].dtype), 300)
    expected_queries = np.tile(np.arange(300, dtype=np.int32), len(expected_image_ids))
    require(np.array_equal(arrays["image_ids"], expected_ids), "CANDIDATE_NATIVE_JOIN_FAIL", "native image block/order")
    require(np.array_equal(arrays["query_index"], expected_queries), "CANDIDATE_NATIVE_JOIN_FAIL", "native query block/order")
    require(np.all((arrays["l3_road8_scores"] >= 0.0) & (arrays["l3_road8_scores"] <= 1.0)), "SCORE_RECONSTRUCTION_FAILURE", "native score range")
    require(np.all((arrays["l3_pred_box_cxcywh"] >= 0.0) & (arrays["l3_pred_box_cxcywh"] <= 1.0)), "SERIALIZED_SCHEMA_FAIL", "normalized native box range")

    reconstructed = torch.sigmoid(torch.from_numpy(arrays["l3_road8_logits"])).numpy()
    max_abs = float(np.max(np.abs(reconstructed - arrays["l3_road8_scores"])))
    require(
        np.allclose(reconstructed, arrays["l3_road8_scores"], atol=1e-7, rtol=1e-6),
        "SCORE_RECONSTRUCTION_FAILURE",
        "stored Road8 scores do not reconstruct from float32 logits",
        actual=max_abs,
    )

    index_table = pq.read_table(index_path)
    require(index_table.schema.remove_metadata() == native_index_arrow_schema(), "SERIALIZED_SCHEMA_FAIL", "native index physical schema")
    require(index_table.num_rows == expected_rows, "CANDIDATE_NATIVE_JOIN_FAIL", "native index row count")
    require(sum(index_table[name].null_count for name in index_table.column_names) == 0, "SERIALIZED_SCHEMA_FAIL", "native index null")
    index_ids = _column_strings(index_table, "canonical_image_id")
    index_queries = _column_numpy(index_table, "query_index", np.int32)
    index_shards = _column_strings(index_table, "shard_id")
    index_offsets = _column_numpy(index_table, "row_offset", np.int64)
    require(np.array_equal(index_ids, expected_ids.astype(object)), "CANDIDATE_NATIVE_JOIN_FAIL", "native index image IDs")
    require(np.array_equal(index_queries, expected_queries), "CANDIDATE_NATIVE_JOIN_FAIL", "native index query IDs")
    require(np.all(index_shards == shard_id), "CANDIDATE_NATIVE_JOIN_FAIL", "native index shard ID")
    require(np.array_equal(index_offsets, np.arange(expected_rows, dtype=np.int64)), "CANDIDATE_NATIVE_JOIN_FAIL", "native index offsets")
    return arrays, array_bundle_hash(arrays), expected_rows, max_abs


def _validate_candidates(
    table: pa.Table,
    native: dict[str, np.ndarray],
    image_rows: list[dict[str, str]],
    global_record_ids: set[bytes],
) -> dict[str, Any]:
    expected_rows = len(image_rows) * CANDIDATES_PER_IMAGE
    require(table.schema.remove_metadata() == candidate_arrow_schema(), "SERIALIZED_SCHEMA_FAIL", "candidate physical schema")
    require(table.num_rows == expected_rows, "SERIALIZED_SCHEMA_FAIL", "candidate shard row count")
    require(sum(table[name].null_count for name in table.column_names) == 0, "SERIALIZED_SCHEMA_FAIL", "candidate required-field null")

    image_offsets = np.repeat(np.arange(len(image_rows), dtype=np.int32), CANDIDATES_PER_IMAGE)
    expected_image_ids = np.repeat(np.asarray([row["canonical_image_id"] for row in image_rows], dtype=object), 300)
    expected_coco_ids = np.repeat(np.asarray([int(row["coco_image_id"]) for row in image_rows], dtype=np.int64), 300)
    expected_image_shas = np.repeat(np.asarray([row["image_sha256"] for row in image_rows], dtype=object), 300)
    expected_ranks = np.tile(np.arange(1, 301, dtype=np.int32), len(image_rows))

    strings = {
        name: _column_strings(table, name)
        for name in (
            "dataset_version", "split", "image_id", "canonical_image_id", "image_sha256",
            "candidate_asset_id", "detector_id", "checkpoint_sha256",
            "export_config_sha256", "schema_sha256", "candidate_record_id",
            "predicted_road8_class_name",
        )
    }
    require(np.all(strings["dataset_version"] == DATASET_VERSION), "SERIALIZED_SCHEMA_FAIL", "candidate dataset version")
    require(np.all(strings["split"] == SPLIT), "SERIALIZED_SCHEMA_FAIL", "candidate split")
    require(np.array_equal(strings["image_id"], expected_image_ids), "CANDIDATE_NATIVE_JOIN_FAIL", "candidate image_id order")
    require(np.array_equal(strings["canonical_image_id"], expected_image_ids), "CANDIDATE_NATIVE_JOIN_FAIL", "candidate canonical IDs")
    require(np.array_equal(strings["image_sha256"], expected_image_shas), "IMAGE_MANIFEST_MISMATCH", "candidate image SHA")
    require(np.all(strings["candidate_asset_id"] == EXPECTED_CANDIDATE_ASSET_ID), "NONDETERMINISTIC_IDENTITY", "candidate asset ID")
    require(np.all(strings["detector_id"] == DETECTOR_ID), "DETECTOR_ASSET_MISMATCH", "candidate detector ID")
    require(np.all(strings["checkpoint_sha256"] == EXPECTED_CHECKPOINT_SHA256), "DETECTOR_ASSET_MISMATCH", "candidate checkpoint")
    require(np.all(strings["export_config_sha256"] == EXPECTED_EXPORT_CONFIG_SHA256), "EXPORT_CONFIG_MISMATCH", "candidate config")
    require(np.all(strings["schema_sha256"] == EXPECTED_SCHEMA_SHA256), "SCHEMA_MISMATCH", "candidate schema")

    coco_ids = _column_numpy(table, "coco_image_id", np.int64)
    ranks = _column_numpy(table, "road8_rank", np.int32)
    source_order = _column_numpy(table, "source_order", np.int32)
    query_index = _column_numpy(table, "query_index", np.int32)
    class_id = _column_numpy(table, "predicted_road8_class_id", np.int16)
    coco_class_id = _column_numpy(table, "predicted_coco_category_id", np.int16)
    detector_class = _column_numpy(table, "detector_class_index", np.int16)
    score = _column_numpy(table, "score", np.float64)
    require(np.array_equal(coco_ids, expected_coco_ids), "IMAGE_MANIFEST_MISMATCH", "candidate COCO IDs")
    require(np.array_equal(ranks, expected_ranks), "CANDIDATE_ORDERING_MISMATCH", "candidate rank continuity")
    require(np.all((query_index >= 0) & (query_index < 300)), "CANDIDATE_NATIVE_JOIN_FAIL", "query index domain")
    require(np.all((class_id >= 1) & (class_id <= 8)), "CLASS_MAPPING_INCOMPATIBLE", "Road8 class domain")
    expected_source = query_index.astype(np.int32) * 8 + (class_id.astype(np.int32) - 1)
    require(np.array_equal(source_order, expected_source), "CANDIDATE_ORDERING_MISMATCH", "source order formula")
    class_zero = class_id.astype(np.int64) - 1
    require(np.array_equal(strings["predicted_road8_class_name"], ROAD8_NAMES[class_zero]), "CLASS_MAPPING_INCOMPATIBLE", "Road8 class names")
    require(np.array_equal(coco_class_id, COCO_CATEGORY_IDS[class_zero]), "CLASS_MAPPING_INCOMPATIBLE", "COCO class IDs")
    require(np.array_equal(detector_class, DETECTOR_CLASS_INDICES[class_zero]), "CLASS_MAPPING_INCOMPATIBLE", "detector class indices")

    numeric_names = [
        "score", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "bbox_cx", "bbox_cy", "bbox_w", "bbox_h"
    ]
    numeric = {name: _column_numpy(table, name, np.float64) for name in numeric_names}
    for name, values in numeric.items():
        require(np.isfinite(values).all(), "NONFINITE_CANDIDATE", f"non-finite candidate values: {name}")
    require(np.all((score >= 0.0) & (score <= 1.0)), "SCORE_RECONSTRUCTION_FAILURE", "candidate score range")
    require(np.all(numeric["bbox_w"] >= 0.0) and np.all(numeric["bbox_h"] >= 0.0), "SERIALIZED_SCHEMA_FAIL", "negative bbox extent")

    xyxy_alias = _column_fixed_list(table, "bbox_xyxy", np.float64)
    cxcywh_alias = _column_fixed_list(table, "bbox_cxcywh", np.float64)
    scalar_xyxy = np.column_stack([numeric[name] for name in ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2")])
    scalar_cxcywh = np.column_stack([numeric[name] for name in ("bbox_cx", "bbox_cy", "bbox_w", "bbox_h")])
    require(np.array_equal(xyxy_alias, scalar_xyxy), "SERIALIZED_SCHEMA_FAIL", "bbox xyxy aliases")
    require(np.array_equal(cxcywh_alias, scalar_cxcywh), "SERIALIZED_SCHEMA_FAIL", "bbox cxcywh aliases")
    derived = np.column_stack(
        [
            (scalar_xyxy[:, 0] + scalar_xyxy[:, 2]) / 2.0,
            (scalar_xyxy[:, 1] + scalar_xyxy[:, 3]) / 2.0,
            scalar_xyxy[:, 2] - scalar_xyxy[:, 0],
            scalar_xyxy[:, 3] - scalar_xyxy[:, 1],
        ]
    )
    require(np.array_equal(derived, scalar_cxcywh), "SERIALIZED_SCHEMA_FAIL", "bbox derived cxcywh")

    native_row = image_offsets.astype(np.int64) * 300 + query_index.astype(np.int64)
    expected_scores = native["l3_road8_scores"][native_row, class_zero].astype(np.float64)
    require(np.array_equal(score, expected_scores), "SCORE_RECONSTRUCTION_FAILURE", "candidate/native score exact equality")

    native_box = native["l3_pred_box_cxcywh"][native_row]
    cx = native_box[:, 0]
    cy = native_box[:, 1]
    width = native_box[:, 2]
    height = native_box[:, 3]
    expected_xyxy = np.stack(
        (cx - width / np.float32(2), cy - height / np.float32(2), cx + width / np.float32(2), cy + height / np.float32(2)),
        axis=1,
    )
    image_widths = np.asarray([int(row["width"]) for row in image_rows], dtype=np.float32)[image_offsets]
    image_heights = np.asarray([int(row["height"]) for row in image_rows], dtype=np.float32)[image_offsets]
    expected_xyxy *= np.column_stack((image_widths, image_heights, image_widths, image_heights)).astype(np.float32)
    require(np.array_equal(scalar_xyxy, expected_xyxy.astype(np.float64)), "BBOX_RECONSTRUCTION_FAILURE", "candidate/native bbox exact equality")

    record_ids = strings["candidate_record_id"]
    require(all(isinstance(item, str) and HEX64_RE.fullmatch(item) for item in record_ids), "NONDETERMINISTIC_IDENTITY", "record ID format")
    for row_index, record_id in enumerate(record_ids):
        expected_id = length_prefixed_record_id(EXPECTED_CANDIDATE_ASSET_ID, str(expected_image_ids[row_index]), int(ranks[row_index]))
        require(record_id == expected_id, "NONDETERMINISTIC_IDENTITY", "candidate record ID", expected=expected_id, actual=record_id)
        key = bytes.fromhex(record_id)
        require(key not in global_record_ids, "NONDETERMINISTIC_IDENTITY", "duplicate candidate record ID", actual=record_id)
        global_record_ids.add(key)

    sorting_mismatches = 0
    for image_offset in range(len(image_rows)):
        native_start = image_offset * 300
        candidate_start = image_offset * 300
        scores_2d = native["l3_road8_scores"][native_start : native_start + 300]
        flat_scores = scores_2d.reshape(-1)
        all_query = np.repeat(np.arange(300, dtype=np.int32), 8)
        all_class = np.tile(np.arange(1, 9, dtype=np.int16), 300)
        all_source = np.arange(2400, dtype=np.int32)
        order = np.lexsort((all_source, all_class, all_query, -flat_scores))[:300]
        block = slice(candidate_start, candidate_start + 300)
        if not (
            np.array_equal(query_index[block], all_query[order])
            and np.array_equal(class_id[block], all_class[order])
            and np.array_equal(source_order[block], all_source[order])
            and np.array_equal(score[block], flat_scores[order].astype(np.float64))
        ):
            sorting_mismatches += 1
    require(sorting_mismatches == 0, "CANDIDATE_ORDERING_MISMATCH", "Top300 reconstruction mismatch", actual=sorting_mismatches)

    repeated_query_class_records = int(
        sum(
            len(set(query_index[offset : offset + 300].tolist())) < 300
            for offset in range(0, expected_rows, 300)
        )
    )
    return {
        "rows": expected_rows,
        "sorting_mismatches": sorting_mismatches,
        "candidate_native_join_failures": 0,
        "images_with_repeated_query_different_class_records": repeated_query_class_records,
    }


def _validate_shard_json(
    path: Path,
    *,
    ordinal: int,
    shard_id: str,
    image_rows: list[dict[str, str]],
    candidate_row: dict[str, str],
    native_row: dict[str, str],
    candidate_path: str,
    native_path: str,
    index_path: str,
    expected_runner_sha256: str,
) -> list[dict[str, Any]]:
    value = load_json(path)
    required = {
        "contract_type", "status", "dataset_version", "split", "candidate_asset_id",
        "shard_id", "shard_ordinal", "first_manifest_index", "last_manifest_index",
        "first_image_id", "last_image_id", "image_count", "candidate", "native",
        "native_index", "invariants", "runner_source_sha256", "detector_forward_count",
        "detector_forward_count_cumulative_diagnostic", "determinism_baselines", "runtime",
        "completed_at_utc",
    }
    require(set(value) == required, "SERIALIZED_SCHEMA_FAIL", "shard JSON fields", expected=sorted(required), actual=sorted(value))
    require(value["contract_type"] == "COCO2017_VAL2017_CANONICAL_SHARD_V1", "SERIALIZED_SCHEMA_FAIL", "shard contract type")
    require(value["dataset_version"] == DATASET_VERSION and value["split"] == SPLIT, "SERIALIZED_SCHEMA_FAIL", "shard identity")
    require(value["candidate_asset_id"] == EXPECTED_CANDIDATE_ASSET_ID, "NONDETERMINISTIC_IDENTITY", "shard asset ID")
    require(value["shard_id"] == shard_id and value["shard_ordinal"] == ordinal, "SERIALIZED_SCHEMA_FAIL", "shard ordinal")
    require(value["first_manifest_index"] == ordinal * 500 and value["last_manifest_index"] == ordinal * 500 + 499, "SERIALIZED_SCHEMA_FAIL", "shard range")
    require(value["first_image_id"] == image_rows[0]["canonical_image_id"] and value["last_image_id"] == image_rows[-1]["canonical_image_id"], "SERIALIZED_SCHEMA_FAIL", "shard image range")
    require(value["image_count"] == 500 and value["detector_forward_count"] == 500, "SERIALIZED_SCHEMA_FAIL", "shard counts")
    require(isinstance(value["detector_forward_count_cumulative_diagnostic"], int) and value["detector_forward_count_cumulative_diagnostic"] >= 500, "SERIALIZED_SCHEMA_FAIL", "cumulative detector forward diagnostic")
    require(value["status"] == "SHARD_COMMITTED", "SERIALIZED_SCHEMA_FAIL", "shard completion status")
    require(value["runner_source_sha256"] == expected_runner_sha256, "SERIALIZED_SCHEMA_FAIL", "runner source SHA binding")
    require(isinstance(value["completed_at_utc"], str) and value["completed_at_utc"], "SERIALIZED_SCHEMA_FAIL", "shard completion time")
    invariants = value["invariants"]
    expected_invariants = {
        "raw_hypotheses_per_image": 2400,
        "stored_candidates_per_image": 300,
        "native_rows_per_image": 300,
        "candidate_to_native_join_key": ["canonical_image_id", "query_index"],
        "annotation_or_GT_read": False,
        "image_sha_verified_count": 500,
        "image_dimension_verified_count": 500,
    }
    require(invariants == expected_invariants, "QA_AGGREGATE_FAIL", "shard invariants", expected=expected_invariants, actual=invariants)
    baselines = value["determinism_baselines"]
    require(isinstance(baselines, list), "SERIALIZED_SCHEMA_FAIL", "determinism baselines type")
    baseline_fields = {
        "manifest_index", "canonical_image_id", "shard_id",
        "native_content_sha256", "candidate_content_sha256",
        "candidate_record_ids", "candidate_order",
    }
    for baseline in baselines:
        require(isinstance(baseline, dict) and set(baseline) == baseline_fields, "SERIALIZED_SCHEMA_FAIL", "determinism baseline fields")
        require(baseline["shard_id"] == shard_id, "DETERMINISM_FAIL", "baseline shard ID")
        require(isinstance(baseline["manifest_index"], int), "DETERMINISM_FAIL", "baseline manifest index")
        require(isinstance(baseline["canonical_image_id"], str), "DETERMINISM_FAIL", "baseline image ID")
        require(HEX64_RE.fullmatch(str(baseline["native_content_sha256"])) is not None, "DETERMINISM_FAIL", "baseline native hash")
        require(HEX64_RE.fullmatch(str(baseline["candidate_content_sha256"])) is not None, "DETERMINISM_FAIL", "baseline candidate hash")
        require(isinstance(baseline["candidate_record_ids"], list) and len(baseline["candidate_record_ids"]) == 300, "DETERMINISM_FAIL", "baseline record IDs")
        require(isinstance(baseline["candidate_order"], list) and len(baseline["candidate_order"]) == 300, "DETERMINISM_FAIL", "baseline candidate order")
    runtime = value["runtime"]
    require(
        isinstance(runtime, dict)
        and set(runtime) == {"forward_and_input_seconds", "total_shard_seconds", "seconds_per_image"}
        and all(isinstance(item, (int, float)) and not isinstance(item, bool) and math.isfinite(float(item)) and float(item) >= 0.0 for item in runtime.values()),
        "SERIALIZED_SCHEMA_FAIL",
        "shard runtime diagnostics",
    )

    expected_candidate = {
        "path": candidate_path,
        "format": "parquet",
        "rows": int(candidate_row["rows"]),
        "bytes": int(candidate_row["bytes"]),
        "sha256": candidate_row["sha256"],
        "semantic_sha256": candidate_row["semantic_sha256"],
    }
    expected_native = {
        "path": native_path,
        "format": "npz",
        "rows": int(native_row["rows"]),
        "bytes": int(native_row["bytes"]),
        "sha256": native_row["sha256"],
        "content_sha256": native_row["content_sha256"],
        "keys": list(NATIVE_KEY_ORDER),
    }
    expected_index = {
        "path": index_path,
        "format": "parquet",
        "rows": int(native_row["index_rows"]),
        "bytes": int(native_row["index_bytes"]),
        "sha256": native_row["index_sha256"],
    }
    require(value["candidate"] == expected_candidate, "SERIALIZED_SCHEMA_FAIL", "shard candidate binding")
    # Key order has no meaning; require list membership order as frozen lexical order.
    require(value["native"] == expected_native, "SERIALIZED_SCHEMA_FAIL", "shard native binding")
    require(value["native_index"] == expected_index, "SERIALIZED_SCHEMA_FAIL", "shard index binding")
    return baselines


def _validate_determinism(
    root: Path,
    image_rows: list[dict[str, str]],
    run_config: dict[str, Any],
    serialized: dict[str, dict[str, Any]],
    baselines: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    path = root / "determinism_report.json"
    require(path.is_file(), "DETERMINISM_FAIL", "determinism_report.json missing")
    report = load_json(path)
    required = {
        "contract_type", "status", "selection_namespace", "sample_count",
        "sample_image_ids", "comparison", "records", "all_checks_pass",
    }
    require(set(report) == required, "DETERMINISM_FAIL", "determinism report fields")
    require(report["contract_type"] == "COCO2017_VAL2017_CANONICAL_DETERMINISM_V1", "DETERMINISM_FAIL", "determinism contract type")
    require(report["sample_count"] == 20 and report["all_checks_pass"] is True, "DETERMINISM_FAIL", "determinism aggregate")
    require(isinstance(report["status"], str) and report["status"].endswith("PASS"), "DETERMINISM_FAIL", "determinism status")
    run_det = run_config.get("determinism")
    require(isinstance(run_det, dict), "RUN_CONFIG_MISMATCH", "run determinism config")
    namespace = report["selection_namespace"]
    require(namespace == run_det.get("selection_namespace"), "DETERMINISM_FAIL", "selection namespace binding")
    ranked = sorted(
        ((sha256_text(namespace + row["canonical_image_id"]), row["canonical_image_id"]) for row in image_rows),
        key=lambda item: (item[0], item[1]),
    )
    expected_ids = [item[1] for item in ranked[:20]]
    require(run_det.get("sample_image_ids") == expected_ids, "RUN_CONFIG_MISMATCH", "frozen determinism sample IDs")
    require(report["sample_image_ids"] == expected_ids, "DETERMINISM_FAIL", "determinism sample selection")
    records = report["records"]
    require(isinstance(records, list) and len(records) == 20, "DETERMINISM_FAIL", "determinism records")
    record_fields = {
        "manifest_index", "canonical_image_id", "shard_id",
        "first_native_content_sha256", "repeat_native_content_sha256", "native_exact",
        "first_candidate_content_sha256", "repeat_candidate_content_sha256", "candidate_exact",
        "record_ids_exact", "sorting_exact", "all_exact",
    }
    image_lookup = {row["canonical_image_id"]: index for index, row in enumerate(image_rows)}
    require(set(serialized) == set(expected_ids), "DETERMINISM_FAIL", "serialized determinism sample set")
    require(set(baselines) == set(expected_ids), "DETERMINISM_FAIL", "shard baseline sample set")
    for expected_id, record in zip(expected_ids, records):
        require(set(record) == record_fields, "DETERMINISM_FAIL", "determinism record fields")
        require(record["canonical_image_id"] == expected_id, "DETERMINISM_FAIL", "determinism record order")
        expected_index = image_lookup[expected_id]
        require(record["manifest_index"] == expected_index, "DETERMINISM_FAIL", "determinism manifest index")
        require(record["shard_id"] == f"{SPLIT}_shard_{expected_index // 500:04d}", "DETERMINISM_FAIL", "determinism shard ID")
        for name in ("first_native_content_sha256", "repeat_native_content_sha256", "first_candidate_content_sha256", "repeat_candidate_content_sha256"):
            require(isinstance(record[name], str) and HEX64_RE.fullmatch(record[name]), "DETERMINISM_FAIL", f"invalid hash {name}")
        require(record["first_native_content_sha256"] == record["repeat_native_content_sha256"], "DETERMINISM_FAIL", "native repeat hash")
        require(record["first_candidate_content_sha256"] == record["repeat_candidate_content_sha256"], "DETERMINISM_FAIL", "candidate repeat hash")
        persisted = serialized[expected_id]
        baseline = baselines[expected_id]
        require(baseline["manifest_index"] == expected_index and baseline["shard_id"] == record["shard_id"], "DETERMINISM_FAIL", "baseline identity binding")
        require(baseline["native_content_sha256"] == persisted["native_content_sha256"] == record["first_native_content_sha256"], "DETERMINISM_FAIL", "serialized native determinism binding")
        require(baseline["candidate_content_sha256"] == persisted["candidate_content_sha256"] == record["first_candidate_content_sha256"], "DETERMINISM_FAIL", "serialized candidate determinism binding")
        require(baseline["candidate_record_ids"] == persisted["candidate_record_ids"], "DETERMINISM_FAIL", "serialized record ID binding")
        require(baseline["candidate_order"] == persisted["candidate_order"], "DETERMINISM_FAIL", "serialized order binding")
        for name in ("native_exact", "candidate_exact", "record_ids_exact", "sorting_exact", "all_exact"):
            require(record[name] is True, "DETERMINISM_FAIL", f"determinism boolean {name}")
    return {
        "sample_count": 20,
        "native_exact": 20,
        "candidate_exact": 20,
        "record_ids_exact": 20,
        "sorting_exact": 20,
    }


def _validate_export(staging_root: Path, run_config_path: Path) -> dict[str, Any]:
    root = staging_root.resolve()
    require(root.is_dir(), "SERIALIZED_READBACK_FAIL", "staging root missing", actual=str(root))
    require(run_config_path.is_file(), "RUN_CONFIG_MISMATCH", "run_config missing", actual=str(run_config_path))
    run_config = load_json(run_config_path)
    configured_staging = run_config.get("staging_root")
    require(isinstance(configured_staging, str), "RUN_CONFIG_MISMATCH", "staging_root missing")
    require(Path(configured_staging).resolve() == root, "RUN_CONFIG_MISMATCH", "staging_root path binding", expected=str(root), actual=configured_staging)

    _, source_rows = _validate_frozen_inputs(run_config)
    configured_dataset_root = run_config.get("dataset_root")
    require(isinstance(configured_dataset_root, str), "RUN_CONFIG_MISMATCH", "dataset_root missing")
    dataset_root = Path(configured_dataset_root).resolve()
    require(dataset_root.is_dir(), "RUN_CONFIG_MISMATCH", "dataset_root does not exist", actual=str(dataset_root))
    image_rows = _validate_output_image_manifest(root, source_rows, dataset_root)
    candidate_manifest, native_manifest, shard_manifest = _read_artifact_manifests(root)
    global_record_ids: set[bytes] = set()

    total_candidate_rows = 0
    total_native_rows = 0
    total_index_rows = 0
    total_sorting_mismatches = 0
    total_join_failures = 0
    total_repeated_query_images = 0
    sigmoid_max_abs = 0.0
    verified_shard_files: list[dict[str, Any]] = []
    determinism_ids = set(run_config["determinism"]["sample_image_ids"])
    serialized_determinism: dict[str, dict[str, Any]] = {}
    baseline_determinism: dict[str, dict[str, Any]] = {}

    for ordinal in range(EXPECTED_SHARD_COUNT):
        shard_id = f"{SPLIT}_shard_{ordinal:04d}"
        first_index = ordinal * IMAGES_PER_SHARD
        last_index = first_index + IMAGES_PER_SHARD - 1
        shard_images = image_rows[first_index : last_index + 1]
        first_id = shard_images[0]["canonical_image_id"]
        last_id = shard_images[-1]["canonical_image_id"]
        candidate_rel = f"candidate/{SPLIT}_candidates_{ordinal:04d}.parquet"
        native_rel = f"native/{SPLIT}_native_{ordinal:04d}.npz"
        index_rel = f"native/{SPLIT}_native_index_{ordinal:04d}.parquet"
        shard_rel = f"shards/{SPLIT}_shard_{ordinal:04d}.json"

        candidate_row = candidate_manifest[ordinal]
        native_row = native_manifest[ordinal]
        combined_row = shard_manifest[ordinal]
        common_expected = {
            "shard_id": shard_id,
            "split": SPLIT,
            "first_manifest_index": str(first_index),
            "last_manifest_index": str(last_index),
            "first_image_id": first_id,
            "last_image_id": last_id,
            "image_count": str(IMAGES_PER_SHARD),
        }
        for manifest_name, row in (("candidate_manifest", candidate_row), ("native_manifest", native_row), ("shard_manifest", combined_row)):
            for key, expected in common_expected.items():
                require(row[key] == expected, "SERIALIZED_SCHEMA_FAIL", f"{manifest_name} {ordinal} {key}", expected=expected, actual=row[key])

        expected_candidate_rows = IMAGES_PER_SHARD * 300
        expected_native_rows = IMAGES_PER_SHARD * 300
        require(candidate_row["path"] == candidate_rel, "MANIFEST_PATH_FAIL", "candidate path")
        require(native_row["path"] == native_rel and native_row["index_path"] == index_rel, "MANIFEST_PATH_FAIL", "native paths")
        require(int_field(candidate_row, "rows", "candidate_manifest") == expected_candidate_rows, "SERIALIZED_SCHEMA_FAIL", "candidate manifest rows")
        require(int_field(native_row, "rows", "native_manifest") == expected_native_rows, "SERIALIZED_SCHEMA_FAIL", "native manifest rows")
        require(int_field(native_row, "index_rows", "native_manifest") == expected_native_rows, "SERIALIZED_SCHEMA_FAIL", "index manifest rows")

        candidate_path = safe_artifact_path(root, candidate_rel)
        native_path = safe_artifact_path(root, native_rel)
        index_path = safe_artifact_path(root, index_rel)
        shard_path = safe_artifact_path(root, shard_rel)
        for path in (candidate_path, native_path, index_path, shard_path):
            require(path.is_file(), "SERIALIZED_READBACK_FAIL", f"missing shard artifact: {path}")

        file_checks = (
            (candidate_path, candidate_row["sha256"], int_field(candidate_row, "bytes", "candidate_manifest")),
            (native_path, native_row["sha256"], int_field(native_row, "bytes", "native_manifest")),
            (index_path, native_row["index_sha256"], int_field(native_row, "index_bytes", "native_manifest")),
        )
        for path, expected_sha, expected_bytes in file_checks:
            require(HEX64_RE.fullmatch(expected_sha) is not None, "SHA_MISMATCH", f"invalid manifest SHA: {path}")
            require(path.stat().st_size == expected_bytes, "SHA_MISMATCH", f"byte size mismatch: {path}")
            actual_sha = sha256_file(path)
            require(actual_sha == expected_sha, "SHA_MISMATCH", f"file SHA mismatch: {path}", expected=expected_sha, actual=actual_sha)

        native_arrays, native_content_sha, native_rows, shard_sigmoid_max = _validate_native(
            native_path, index_path, [row["canonical_image_id"] for row in shard_images], shard_id
        )
        require(native_content_sha == native_row["content_sha256"], "SHA_MISMATCH", "native content SHA")
        sigmoid_max_abs = max(sigmoid_max_abs, shard_sigmoid_max)

        candidate_table = pq.read_table(candidate_path)
        candidate_diag = _validate_candidates(candidate_table, native_arrays, shard_images, global_record_ids)
        candidate_semantic_sha = canonical_table_rows_hash(candidate_table)
        require(candidate_semantic_sha == candidate_row["semantic_sha256"], "SHA_MISMATCH", "candidate semantic SHA")

        expected_combined = {
            **common_expected,
            "candidate_path": candidate_rel,
            "candidate_rows": str(expected_candidate_rows),
            "candidate_sha256": candidate_row["sha256"],
            "native_path": native_rel,
            "native_rows": str(expected_native_rows),
            "native_sha256": native_row["sha256"],
            "index_path": index_rel,
            "index_rows": str(expected_native_rows),
            "index_sha256": native_row["index_sha256"],
        }
        require(combined_row == expected_combined, "SERIALIZED_SCHEMA_FAIL", "combined shard manifest row", expected=expected_combined, actual=combined_row)
        shard_baselines = _validate_shard_json(
            shard_path,
            ordinal=ordinal,
            shard_id=shard_id,
            image_rows=shard_images,
            candidate_row=candidate_row,
            native_row=native_row,
            candidate_path=candidate_rel,
            native_path=native_rel,
            index_path=index_rel,
            expected_runner_sha256=run_config["input_bindings"]["runner_source"]["sha256"],
        )
        for baseline in shard_baselines:
            image_id = baseline["canonical_image_id"]
            require(image_id in determinism_ids and image_id not in baseline_determinism, "DETERMINISM_FAIL", "unexpected or duplicate shard baseline", actual=image_id)
            baseline_determinism[image_id] = baseline
        for local_index, image_row in enumerate(shard_images):
            image_id = image_row["canonical_image_id"]
            if image_id not in determinism_ids:
                continue
            native_start = local_index * QUERIES_PER_IMAGE
            native_stop = native_start + QUERIES_PER_IMAGE
            native_sample = {
                name: native_arrays[name][native_start:native_stop]
                for name in NATIVE_KEY_ORDER
                if name != "image_ids"
            }
            candidate_sample = candidate_table.slice(
                local_index * CANDIDATES_PER_IMAGE, CANDIDATES_PER_IMAGE
            )
            serialized_determinism[image_id] = {
                "native_content_sha256": array_bundle_hash(native_sample),
                "candidate_content_sha256": canonical_table_rows_hash(candidate_sample),
                "candidate_record_ids": _column_strings(candidate_sample, "candidate_record_id").tolist(),
                "candidate_order": np.column_stack(
                    [
                        _column_numpy(candidate_sample, "road8_rank", np.int32),
                        _column_numpy(candidate_sample, "query_index", np.int32),
                        _column_numpy(candidate_sample, "predicted_road8_class_id", np.int16),
                        _column_numpy(candidate_sample, "source_order", np.int32),
                    ]
                ).astype(np.int64).tolist(),
            }

        total_candidate_rows += candidate_table.num_rows
        total_native_rows += native_rows
        total_index_rows += native_rows
        total_sorting_mismatches += candidate_diag["sorting_mismatches"]
        total_join_failures += candidate_diag["candidate_native_join_failures"]
        total_repeated_query_images += candidate_diag["images_with_repeated_query_different_class_records"]
        verified_shard_files.append(
            {
                "shard_id": shard_id,
                "candidate": {
                    "path": candidate_rel,
                    "rows": expected_candidate_rows,
                    "bytes": int(candidate_row["bytes"]),
                    "sha256": candidate_row["sha256"],
                },
                "native": {
                    "path": native_rel,
                    "rows": expected_native_rows,
                    "bytes": int(native_row["bytes"]),
                    "sha256": native_row["sha256"],
                },
                "native_index": {
                    "path": index_rel,
                    "rows": expected_native_rows,
                    "bytes": int(native_row["index_bytes"]),
                    "sha256": native_row["index_sha256"],
                },
                "candidate_semantic_sha256": candidate_semantic_sha,
                "native_content_sha256": native_content_sha,
            }
        )

    require(total_candidate_rows == EXPECTED_IMAGE_COUNT * 300, "QA_AGGREGATE_FAIL", "aggregate candidate rows")
    require(total_native_rows == EXPECTED_IMAGE_COUNT * 300, "QA_AGGREGATE_FAIL", "aggregate native rows")
    require(total_index_rows == EXPECTED_IMAGE_COUNT * 300, "QA_AGGREGATE_FAIL", "aggregate index rows")
    require(len(global_record_ids) == EXPECTED_IMAGE_COUNT * 300, "NONDETERMINISTIC_IDENTITY", "global record ID count")
    require(total_sorting_mismatches == 0, "CANDIDATE_ORDERING_MISMATCH", "aggregate sorting mismatches")
    require(total_join_failures == 0, "CANDIDATE_NATIVE_JOIN_FAIL", "aggregate join failures")
    determinism = _validate_determinism(
        root,
        image_rows,
        run_config,
        serialized_determinism,
        baseline_determinism,
    )
    global_native_index = _validate_global_native_index(root, image_rows)
    ledger = _validate_sha_ledger(root, run_config_path)

    manifest_copy_root = root / "manifest"
    require(manifest_copy_root.is_dir(), "SERIALIZED_READBACK_FAIL", "manifest mirror directory missing")
    for filename in ("candidate_manifest.csv", "native_manifest.csv", "image_manifest.csv", "shard_manifest.csv"):
        mirror = manifest_copy_root / filename
        source = root / filename
        require(mirror.is_file(), "SERIALIZED_READBACK_FAIL", f"manifest mirror missing: {filename}")
        require(sha256_file(mirror) == sha256_file(source), "SHA_MISMATCH", f"manifest mirror mismatch: {filename}")

    # Exact file inventory for canonical scientific shards.  Reports/logs may be
    # added later by the caller, but extra candidate/native/shard payloads fail.
    expected_candidate_files = {f"{SPLIT}_candidates_{index:04d}.parquet" for index in range(10)}
    expected_native_files = {
        *(f"{SPLIT}_native_{index:04d}.npz" for index in range(10)),
        *(f"{SPLIT}_native_index_{index:04d}.parquet" for index in range(10)),
    }
    expected_shard_files = {f"{SPLIT}_shard_{index:04d}.json" for index in range(10)}
    require({path.name for path in (root / "candidate").iterdir() if path.is_file()} == expected_candidate_files, "SERIALIZED_SCHEMA_FAIL", "candidate file inventory")
    require({path.name for path in (root / "native").iterdir() if path.is_file()} == expected_native_files, "SERIALIZED_SCHEMA_FAIL", "native file inventory")
    require({path.name for path in (root / "shards").iterdir() if path.is_file()} == expected_shard_files, "SERIALIZED_SCHEMA_FAIL", "shard metadata inventory")

    return {
        "qa_contract": "COCO2017_ROAD8_EXPORT_QA_V1",
        "validator": "canonical_export_validator.py",
        "validation_mode": "read_only_serialized_asset_validation",
        "split": SPLIT,
        "dataset_version": DATASET_VERSION,
        "candidate_asset_id": EXPECTED_CANDIDATE_ASSET_ID,
        "all_checks_pass": True,
        "image_coverage_fraction": 1.0,
        "images_expected": EXPECTED_IMAGE_COUNT,
        "images_validated": EXPECTED_IMAGE_COUNT,
        "image_byte_sha_verified": EXPECTED_IMAGE_COUNT,
        "image_dimensions_verified": EXPECTED_IMAGE_COUNT,
        "missing_images": 0,
        "extra_images": 0,
        "candidate_rows": total_candidate_rows,
        "native_rows": total_native_rows,
        "native_index_rows": total_index_rows,
        "global_native_index": global_native_index,
        "raw_hypotheses_reconstructed": EXPECTED_IMAGE_COUNT * RAW_HYPOTHESES_PER_IMAGE,
        "candidate_record_id_duplicates": 0,
        "candidate_native_join_failures": 0,
        "sorting_mismatches": 0,
        "score_reconstruction_max_abs_error": sigmoid_max_abs,
        "nan_count": 0,
        "inf_count": 0,
        "sha_mismatches": ledger["mismatches"],
        "sha_ledger_entries": ledger["entries"],
        "schema_sha256": EXPECTED_SCHEMA_SHA256,
        "export_config_sha256": EXPECTED_EXPORT_CONFIG_SHA256,
        "split_manifest_sha256": EXPECTED_SPLIT_MANIFEST_SHA256,
        "shard_count": EXPECTED_SHARD_COUNT,
        "candidate_shard_count": EXPECTED_SHARD_COUNT,
        "native_shard_count": EXPECTED_SHARD_COUNT,
        "native_index_shard_count": EXPECTED_SHARD_COUNT,
        "images_with_retained_same_query_multiple_class_records": total_repeated_query_images,
        "determinism": determinism,
        "verified_shards": verified_shard_files,
        "scientific_evaluation_performed": False,
        "annotation_or_gt_read": False,
    }


def validate_export(staging_root: Path, run_config_path: Path) -> dict[str, Any]:
    """Validate one exact canonical export root without mutating it.

    Known validation failures are returned as a JSON-serializable fail-closed
    result.  Unexpected programming/runtime errors are also converted to a
    blocked result so a caller cannot accidentally publish after validator
    failure.
    """
    try:
        return _validate_export(Path(staging_root), Path(run_config_path))
    except ValidationFailure as failure:
        return {
            "qa_contract": "COCO2017_ROAD8_EXPORT_QA_V1",
            "validator": "canonical_export_validator.py",
            "all_checks_pass": False,
            "status": "COCO_EXPORT_QA_FAIL",
            "stage": failure.code,
            "error": str(failure),
            "expected": failure.expected,
            "actual": failure.actual,
            "scientific_evaluation_performed": False,
            "annotation_or_gt_read": False,
        }
    except BaseException as failure:  # fail closed, including dependency/readback bugs
        return {
            "qa_contract": "COCO2017_ROAD8_EXPORT_QA_V1",
            "validator": "canonical_export_validator.py",
            "all_checks_pass": False,
            "status": "COCO_EXPORT_QA_FAIL",
            "stage": "VALIDATOR_INTERNAL_ERROR",
            "error": f"{type(failure).__name__}: {failure}",
            "expected": "validator completes without an unexpected exception",
            "actual": type(failure).__name__,
            "scientific_evaluation_performed": False,
            "annotation_or_gt_read": False,
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging-root", type=Path, required=True)
    parser.add_argument("--run-config", type=Path, required=True)
    args = parser.parse_args(argv)
    result = validate_export(args.staging_root, args.run_config)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))
    return 0 if result.get("all_checks_pass") is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
