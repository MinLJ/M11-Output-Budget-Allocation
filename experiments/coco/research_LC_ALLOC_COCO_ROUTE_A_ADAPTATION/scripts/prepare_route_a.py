"""Prepare COCO TRAIN2017 Route-A labels, PCA32, scaler, and training arrays.

This entrypoint is intentionally TRAIN-only.  It accepts no VAL asset, VAL GT,
selection, or evaluation argument and contains no detector execution path.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from route_a_common import (
    DEFAULT_CONFIG,
    EXPECTED_TRAIN_IMAGES,
    K_MAX,
    MODEL_RANKS,
    PROJECT_ROOT,
    REPO_ROOT,
    ROLE_TO_CODE,
    ROAD8,
    THRESHOLDS,
    CanonicalTrainReader,
    build_split_manifest,
    load_coco_train_gt,
    load_config,
    load_p1_core,
    sha256_file,
    write_dataframe_csv_atomic,
    write_json_atomic,
)


Y_COLUMNS = [f"y_iou_{threshold:.2f}".replace(".", "_") for threshold in THRESHOLDS]


def progress(message: str) -> None:
    print(f"[Route-A prepare] {message}", flush=True)


def refuse_published_outputs(root: Path) -> None:
    protected = [
        "route_a_config.json", "split_manifest.csv", "coco_allocator_split_manifest.csv",
        "feature_config.json", "pca_model.joblib", "scaler.joblib", "class_weight_config.json",
        "training_history.csv", "temperature_params.json", "sha256_ledger.csv",
    ]
    existing = [name for name in protected if (root / name).exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite existing Route-A scientific outputs: {existing}")
    checkpoint_dir = root / "model_checkpoints"
    if checkpoint_dir.exists() and any(checkpoint_dir.iterdir()):
        raise FileExistsError("refusing nonempty model_checkpoints directory")


def dump_joblib_atomic(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    os.close(handle)
    try:
        joblib.dump(value, temporary)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def select_pca_identities(
    reader: CanonicalTrainReader,
    fit_ids: set[str],
    namespace: str,
    cap: int,
) -> pd.DataFrame:
    """Keep the globally smallest hash/image/query keys with bounded memory."""
    heap: list[tuple[int, str, str, int]] = []
    visited_images = 0
    for shard_id, frame in reader.iter_candidate_frames():
        frame = frame[frame["canonical_image_id"].astype(str).isin(fit_ids)]
        for image_id, rows in frame.groupby("canonical_image_id", sort=False):
            image_id = str(image_id)
            numeric_id = int(image_id.rsplit(":", 1)[1])
            for query_index in np.unique(rows["query_index"].to_numpy(np.int16)):
                identity = f"{namespace}{image_id}|{int(query_index)}"
                hex_digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
                # canonical ids use a fixed 12-digit numeric suffix.  Appending
                # numeric id/query bits implements the documented hash/id/query
                # tie-break exactly while a negative key gives a max-heap.
                composite = (int(hex_digest, 16) << 30) | (numeric_id << 9) | int(query_index)
                entry = (-composite, hex_digest, image_id, int(query_index))
                if len(heap) < cap:
                    heapq.heappush(heap, entry)
                elif composite < -heap[0][0]:
                    heapq.heapreplace(heap, entry)
            visited_images += 1
        progress(f"PCA identity pass completed {shard_id}; FIT images visited={visited_images}")
    if visited_images != len(fit_ids):
        raise RuntimeError(f"PCA identity FIT coverage mismatch: {visited_images} != {len(fit_ids)}")
    selected = sorted(((item[1], item[2], item[3]) for item in heap), key=lambda item: (item[0], item[1], item[2]))
    if len(selected) != cap:
        raise RuntimeError(f"PCA sample has {len(selected)} rows, expected {cap}")
    result = pd.DataFrame({
        "selection_order": np.arange(1, len(selected) + 1, dtype=np.int32),
        "identity_hash": [item[0] for item in selected],
        "image_id": [item[1] for item in selected],
        "query_index": np.asarray([item[2] for item in selected], dtype=np.int16),
    })
    if result.duplicated(["image_id", "query_index"]).any():
        raise RuntimeError("PCA unique-query sample contains duplicates")
    return result


def collect_pca_embeddings(reader: CanonicalTrainReader, sample: pd.DataFrame) -> np.ndarray:
    wanted: dict[str, dict[int, int]] = {}
    for row in sample.itertuples(index=False):
        wanted.setdefault(str(row.image_id), {})[int(row.query_index)] = int(row.selection_order) - 1
    output = np.empty((len(sample), 256), dtype=np.float32)
    filled = np.zeros(len(sample), dtype=bool)
    seen = 0
    for bundle in reader.iter_bundles():
        targets = wanted.get(bundle.image_id)
        if targets:
            first: dict[int, int] = {}
            for index, query in enumerate(bundle.candidates["query_index"].to_numpy(np.int16)):
                first.setdefault(int(query), index)
            for query, destination in targets.items():
                if query not in first:
                    raise RuntimeError(f"PCA query missing from Top100: {bundle.image_id}/{query}")
                output[destination] = bundle.embeddings[first[query]].astype(np.float32)
                filled[destination] = True
        seen += 1
        if seen % 10_000 == 0:
            progress(f"PCA embedding pass images={seen}/{EXPECTED_TRAIN_IMAGES}")
    if not np.all(filled) or not np.isfinite(output).all():
        raise RuntimeError(f"PCA embedding gather incomplete/nonfinite: missing={int((~filled).sum())}")
    return output


def label_audit_specs(fit_ids: set[str]) -> dict[str, tuple[int, int]]:
    selected = sorted(
        ((hashlib.sha256(("LC_ALLOC_COCO_ROUTE_A_LABEL_AUDIT_V1|" + image_id).encode("utf-8")).digest(), image_id) for image_id in fit_ids),
        key=lambda item: (item[0], item[1]),
    )[:64]
    return {image_id: (6 + digest[0] % 45, digest[1] % 10) for digest, image_id in selected}


def copy_lc_class_weights(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    source = (REPO_ROOT / str(config["class_weights"]["source"])).resolve(strict=True)
    expected_sha = str(config["class_weights"]["source_sha256"])
    observed_sha = sha256_file(source)
    if observed_sha != expected_sha:
        raise ValueError(f"LC class-weight SHA mismatch: {observed_sha} != {expected_sha}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    weights = np.asarray(payload["weights"], dtype=np.float64)
    if tuple(payload["classes"]) != ROAD8 or weights.shape != (8,) or not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("LC class-weight payload is incompatible")
    output = {
        "task": config["task"],
        "strategy": "KEEP_LC_CLASS_WEIGHTS",
        "frozen_before_training_and_val": True,
        "source_path": str(source),
        "source_sha256": observed_sha,
        "classes": list(ROAD8),
        "weights": weights.tolist(),
        "source_fit_gt_counts": payload.get("fit_gt_counts"),
        "source_formula": payload.get("formula"),
        "copied_exactly": bool(np.array_equal(weights, np.asarray(payload["weights"], dtype=np.float64))),
        "training_loss_weighted": False,
        "allocation_utility_weighted": True,
    }
    write_json_atomic(root / "class_weight_config.json", output)
    return output


def execute(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    (root / "working" / "cache").mkdir(parents=True, exist_ok=True)
    (root / "model_checkpoints").mkdir(parents=True, exist_ok=True)
    refuse_published_outputs(root)

    config_path = args.config.resolve(strict=True)
    config = load_config(config_path)
    p1_core = load_p1_core(config)
    if len(p1_core.feature_spec()) != 90 or not np.array_equal(np.asarray(p1_core.THRESHOLDS, np.float64), THRESHOLDS):
        raise ValueError("frozen P1 feature/threshold implementation mismatch")
    write_json_atomic(root / "route_a_config.json", config)
    config_sha = sha256_file(root / "route_a_config.json")

    progress("validating completed TRAIN2017 canonical asset and all shard hashes")
    reader = CanonicalTrainReader(args.train_asset_root, verify_shard_hashes=True)
    split_manifest = build_split_manifest(reader.image_manifest["canonical_image_id"].astype(str), config)
    write_dataframe_csv_atomic(split_manifest, root / "split_manifest.csv")
    write_dataframe_csv_atomic(split_manifest, root / "coco_allocator_split_manifest.csv")
    if (root / "split_manifest.csv").read_bytes() != (root / "coco_allocator_split_manifest.csv").read_bytes():
        raise AssertionError("split manifest aliases differ")
    role_by_id = dict(zip(split_manifest["image_id"].astype(str), split_manifest["split"].astype(str)))
    fit_ids = {image_id for image_id, role in role_by_id.items() if role == "FIT"}

    progress("loading TRAIN2017 annotations under the frozen noncrowd Road8 rule")
    gt_path = args.train_gt.resolve(strict=True)
    gt_sha = sha256_file(gt_path)
    expected_gt = config.get("train_ground_truth", {})
    if gt_sha != str(expected_gt.get("sha256")):
        raise ValueError(f"TRAIN2017 annotation SHA mismatch: {gt_sha} != {expected_gt.get('sha256')}")
    gt, gt_summary = load_coco_train_gt(gt_path, reader.image_manifest)
    if set(gt) != set(role_by_id):
        raise ValueError("GT/split/canonical identity sets differ")
    write_json_atomic(root / "working" / "train_gt_summary.json", {**gt_summary, "gt_path": str(gt_path), "gt_sha256": gt_sha})
    copy_lc_class_weights(root, config)

    pca_cfg = config["features"]["pca"]
    progress("selecting deterministic FIT-only unique-query PCA sample")
    pca_sample = select_pca_identities(reader, fit_ids, str(pca_cfg["namespace"]), int(pca_cfg["sample_cap"]))
    sample_path = root / "working" / "pca_sample_manifest.parquet"
    pq.write_table(pa.Table.from_pandas(pca_sample, preserve_index=False), sample_path, compression="zstd")
    pca_input = collect_pca_embeddings(reader, pca_sample)
    pca = PCA(
        n_components=int(pca_cfg["components"]),
        svd_solver=str(pca_cfg["svd_solver"]),
        random_state=int(pca_cfg["random_state"]),
        whiten=bool(pca_cfg["whiten"]),
    )
    pca.fit(pca_input)
    del pca_input
    dump_joblib_atomic(pca, root / "pca_model.joblib")
    progress("COCO FIT PCA32 fitted and serialized")

    image_count = len(reader.image_manifest)
    rows_per_image = len(MODEL_RANKS)
    total_rows = image_count * rows_per_image
    cache = root / "working" / "cache"
    x_raw = np.lib.format.open_memmap(cache / "train_X_raw.npy", mode="w+", dtype=np.float32, shape=(total_rows, 90))
    targets = np.lib.format.open_memmap(cache / "train_y.npy", mode="w+", dtype=np.uint8, shape=(total_rows, 10))
    roles = np.lib.format.open_memmap(cache / "train_role.npy", mode="w+", dtype=np.int8, shape=(total_rows,))
    audit_specs = label_audit_specs(fit_ids)
    audit_rows: list[dict[str, Any]] = []
    label_temp = root / "working" / "train_labels.parquet.tmp"
    label_final = root / "working" / "train_labels.parquet"
    writer: pq.ParquetWriter | None = None
    label_batches: list[pd.DataFrame] = []
    row0 = 0
    images_processed = 0
    try:
        for bundle in reader.iter_bundles():
            candidates = bundle.candidates
            gt_image = gt[bundle.image_id]
            classes = candidates["predicted_road8_class_id"].to_numpy(np.int16)
            boxes = candidates[["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]].to_numpy(np.float64)
            totals, _ = p1_core.prefix_matching_counts(classes, boxes, gt_image.classes, gt_image.boxes, max_k=K_MAX)
            y = (totals[6:51] - totals[5:50]).astype(np.int8)
            if y.shape != (45, 10) or np.any((y < 0) | (y > 1)):
                raise ValueError(f"invalid M11 marginal targets: {bundle.image_id}")
            if not np.array_equal(totals[50], totals[5] + y.sum(axis=0)):
                raise ValueError(f"cumulative target identity failed: {bundle.image_id}")
            raw = p1_core.build_raw_features(candidates, bundle.road8_logits, bundle.embeddings, pca, bundle.width, bundle.height)
            if raw.shape != (45, 90) or not np.isfinite(raw).all():
                raise ValueError(f"invalid 90D features: {bundle.image_id}")
            role = role_by_id[bundle.image_id]
            stop = row0 + rows_per_image
            x_raw[row0:stop] = raw.astype(np.float32)
            targets[row0:stop] = y.astype(np.uint8)
            roles[row0:stop] = ROLE_TO_CODE[role]

            current = candidates.iloc[5:50]
            label_frame = pd.DataFrame({
                "image_id": bundle.image_id,
                "split": role,
                "road8_rank": current["road8_rank"].to_numpy(np.int16),
                "candidate_record_id": current["candidate_record_id"].astype(str).to_numpy(),
                "query_index": current["query_index"].to_numpy(np.int16),
                "predicted_road8_class_id": current["predicted_road8_class_id"].to_numpy(np.int8),
                "raw_score": current["score"].to_numpy(np.float64),
            })
            for threshold_index, column in enumerate(Y_COLUMNS):
                label_frame[column] = y[:, threshold_index].astype(np.uint8)
            label_batches.append(label_frame)

            if bundle.image_id in audit_specs:
                rank, threshold_index = audit_specs[bundle.image_id]
                before = p1_core.independent_prefix_count(classes, boxes, gt_image.classes, gt_image.boxes, rank - 1, float(THRESHOLDS[threshold_index]))
                after = p1_core.independent_prefix_count(classes, boxes, gt_image.classes, gt_image.boxes, rank, float(THRESHOLDS[threshold_index]))
                stored = int(y[rank - 6, threshold_index])
                audit_rows.append({
                    "image_id": bundle.image_id, "rank": rank, "threshold": float(THRESHOLDS[threshold_index]),
                    "independent_before": before, "independent_after": after,
                    "independent_delta": after - before, "stored_delta": stored,
                    "pass": bool(after - before == stored),
                })

            if len(label_batches) >= 250:
                table = pa.Table.from_pandas(pd.concat(label_batches, ignore_index=True), preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(label_temp, table.schema, compression="zstd")
                writer.write_table(table)
                label_batches = []
            row0 = stop
            images_processed += 1
            if images_processed % 5_000 == 0:
                progress(f"label/feature pass images={images_processed}/{image_count}")
        if label_batches:
            table = pa.Table.from_pandas(pd.concat(label_batches, ignore_index=True), preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(label_temp, table.schema, compression="zstd")
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    if row0 != total_rows:
        raise RuntimeError(f"training row coverage mismatch: {row0} != {total_rows}")
    if len(audit_rows) != 64 or not all(bool(row["pass"]) for row in audit_rows):
        raise RuntimeError("64-unit independent maximum-matching label audit failed")
    os.replace(label_temp, label_final)
    x_raw.flush(); targets.flush(); roles.flush()

    spec = p1_core.feature_spec()
    standardize = np.asarray([bool(column["standardize_fit_only"]) for column in spec], dtype=bool)
    if standardize.shape != (90,) or int(standardize.sum()) != 81:
        raise ValueError("frozen P1 standardization mask changed")
    scaler = StandardScaler()
    for start in range(0, total_rows, 50_000):
        stop = min(start + 50_000, total_rows)
        fit_mask = np.asarray(roles[start:stop]) == ROLE_TO_CODE["FIT"]
        if fit_mask.any():
            scaler.partial_fit(np.asarray(x_raw[start:stop])[fit_mask][:, standardize])
    scaler.scale_[scaler.scale_ == 0] = 1.0
    scaler_bundle = {"scaler": scaler, "standardize_mask": standardize, "feature_names": [column["name"] for column in spec]}
    dump_joblib_atomic(scaler_bundle, root / "scaler.joblib")

    x_scaled = np.lib.format.open_memmap(cache / "train_X_scaled.npy", mode="w+", dtype=np.float32, shape=(total_rows, 90))
    for start in range(0, total_rows, 50_000):
        stop = min(start + 50_000, total_rows)
        values = np.asarray(x_raw[start:stop], dtype=np.float64).copy()
        values[:, standardize] = scaler.transform(values[:, standardize])
        x_scaled[start:stop] = values.astype(np.float32)
    x_scaled.flush()
    for start in range(0, total_rows, 50_000):
        stop = min(start + 50_000, total_rows)
        if not np.isfinite(np.asarray(x_scaled[start:stop])).all():
            raise RuntimeError(f"scaled training features contain NaN/Inf in rows {start}:{stop}")

    split_counts = split_manifest["split"].value_counts().to_dict()
    feature_config = {
        "task": config["task"],
        "route": config["route"],
        "dataset_version": config["dataset_version"],
        "train_split": "TRAIN2017",
        "train_candidate_asset_id": reader.marker["candidate_asset_id"],
        "route_a_config_sha256": config_sha,
        "split_manifest_sha256": sha256_file(root / "split_manifest.csv"),
        "dimension": 90,
        "columns": spec,
        "candidate_pool": "canonical original road8_rank Top100; ranks 6..50 form model rows",
        "label_target": config["labels"],
        "role_counts_images": {key: int(value) for key, value in split_counts.items()},
        "role_counts_rows": {key: int(value) * 45 for key, value in split_counts.items()},
        "pca": {
            **pca_cfg,
            "model_path": "pca_model.joblib",
            "model_sha256": sha256_file(root / "pca_model.joblib"),
            "sample_manifest_path": "working/pca_sample_manifest.parquet",
            "sample_manifest_sha256": sha256_file(sample_path),
            "explained_variance_ratio_sum_diagnostic_only": float(pca.explained_variance_ratio_.sum()),
        },
        "scaler": {
            **config["features"]["scaler"],
            "model_path": "scaler.joblib",
            "model_sha256": sha256_file(root / "scaler.joblib"),
            "standardized_columns": int(standardize.sum()),
            "unstandardized_columns": int((~standardize).sum()),
        },
        "input_bindings": {
            "train_gt": {"path": str(gt_path), "sha256": sha256_file(gt_path)},
            "canonical_completion_marker": {
                "path": str(reader.root / "CANONICAL_EXPORT_COMPLETE.json"),
                "sha256": sha256_file(reader.root / "CANONICAL_EXPORT_COMPLETE.json"),
            },
            "p1_core": {
                "path": str((REPO_ROOT / config["references"]["p1_core"]).resolve()),
                "sha256": config["references"]["p1_core_sha256"],
            },
        },
    }
    write_json_atomic(root / "feature_config.json", feature_config)
    write_json_atomic(root / "working" / "label_audit.json", {
        "label_rows": total_rows,
        "target_columns": Y_COLUMNS,
        "values_only_0_or_1": True,
        "cumulative_identity_all_images": True,
        "independent_units": audit_rows,
        "independent_units_pass": True,
        "gt_summary": gt_summary,
    })
    complete = {
        "status": "ROUTE_A_TRAIN_PREPARATION_COMPLETE",
        "elapsed_seconds": time.perf_counter() - started,
        "images": image_count,
        "rows": total_rows,
        "route_a_config_sha256": config_sha,
        "split_manifest_sha256": sha256_file(root / "split_manifest.csv"),
        "feature_config_sha256": sha256_file(root / "feature_config.json"),
        "pca_sha256": sha256_file(root / "pca_model.joblib"),
        "scaler_sha256": sha256_file(root / "scaler.joblib"),
        "class_weight_config_sha256": sha256_file(root / "class_weight_config.json"),
        "train_labels_sha256": sha256_file(label_final),
        "train_X_scaled_sha256": sha256_file(cache / "train_X_scaled.npy"),
        "train_y_sha256": sha256_file(cache / "train_y.npy"),
        "train_role_sha256": sha256_file(cache / "train_role.npy"),
        "train_gt_sha256": gt_sha,
        "train_canonical_completion_marker_sha256": sha256_file(reader.root / "CANONICAL_EXPORT_COMPLETE.json"),
        "cache_shapes": {"X": [total_rows, 90], "y": [total_rows, 10], "role": [total_rows]},
        "val_accessed": False,
        "detector_forward": False,
    }
    write_json_atomic(root / "working" / "prepare_complete.json", complete)
    progress("preparation complete; no model training, VAL access, or detector forward was performed by this entrypoint")
    return complete


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare COCO TRAIN2017 data for the Route-A M11 allocator")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--train-asset-root", type=Path, required=True, help="Completed TRAIN2017 canonical candidate/native asset root")
    parser.add_argument("--train-gt", type=Path, required=True, help="Official instances_train2017.json")
    return parser.parse_args()


if __name__ == "__main__":
    result = execute(parse_args())
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
