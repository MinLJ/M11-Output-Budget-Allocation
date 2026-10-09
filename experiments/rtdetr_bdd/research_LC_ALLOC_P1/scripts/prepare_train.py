"""Prepare frozen TRAIN roles, prefix labels, PCA/scaler, TABLE, and model rows.

No DEV asset or DEV ground truth is opened by this entrypoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from p1_core import (  # noqa: E402
    ROAD8, THRESHOLDS, build_raw_features, compute_class_weights,
    feature_spec, fit_table, independent_prefix_count, load_coco_gt,
    prefix_matching_counts, sha256_file, write_json,
)
from release_io import ReleaseReader  # noqa: E402


ROLE_TO_CODE = {"FIT": 0, "EARLY_STOP": 1, "CALIBRATION": 2}
Y_COLS = [f"y_iou_{t:.2f}".replace(".", "_") for t in THRESHOLDS]


def append_log(root: Path, text: str) -> None:
    with (root / "runtime.log").open("a", encoding="utf-8") as f:
        f.write(text.rstrip() + "\n")


def role_split(manifest: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for image_id in manifest["image_id"].astype(str):
        key = hashlib.sha256(("LC_ALLOC_P1_SPLIT|" + image_id).encode("utf-8")).hexdigest()
        rows.append((image_id, key))
    ordered = sorted(rows, key=lambda z: (z[1], z[0]))
    role = {}
    for ix, (image_id, _) in enumerate(ordered):
        role[image_id] = "FIT" if ix < 8000 else ("EARLY_STOP" if ix < 9000 else "CALIBRATION")
    out = pd.DataFrame(rows, columns=["image_id", "split_key"])
    out["role"] = out["image_id"].map(role)
    out["role_rank"] = out["image_id"].map({x[0]: i + 1 for i, x in enumerate(ordered)}).astype(np.int32)
    counts = out["role"].value_counts().to_dict()
    if counts != {"FIT": 8000, "EARLY_STOP": 1000, "CALIBRATION": 1000}:
        raise RuntimeError(f"role split count mismatch: {counts}")
    return out


def select_pca_identities(reader: ReleaseReader, fit_ids: set[str], cap: int = 200_000) -> pd.DataFrame:
    max_rows = len(fit_ids) * 100
    hashes = np.empty(max_rows, dtype="S64")
    image_ids = np.empty(max_rows, dtype="S17")
    query_ids = np.empty(max_rows, dtype=np.int16)
    n = 0
    for candidate_path, _ in reader.shard_paths("TRAIN"):
        df = pq.read_table(
            candidate_path, columns=["image_id", "query_index", "road8_rank"],
            filters=[("road8_rank", "<=", 100)],
        ).to_pandas()
        df = df[df["image_id"].astype(str).isin(fit_ids)]
        for image_id, rows in df.groupby("image_id", sort=False):
            for q in np.unique(rows["query_index"].to_numpy(np.int16)):
                identity = f"LC_ALLOC_P1_PCA|{image_id}|{int(q)}"
                hashes[n] = hashlib.sha256(identity.encode("utf-8")).hexdigest().encode("ascii")
                image_ids[n] = str(image_id).encode("ascii")
                query_ids[n] = q
                n += 1
    hashes, image_ids, query_ids = hashes[:n], image_ids[:n], query_ids[:n]
    order = np.lexsort((query_ids, image_ids, hashes))
    chosen = order[: min(cap, n)]
    out = pd.DataFrame({
        "selection_order": np.arange(1, len(chosen) + 1, dtype=np.int32),
        "identity_hash": np.char.decode(hashes[chosen], "ascii"),
        "image_id": np.char.decode(image_ids[chosen], "ascii"),
        "query_index": query_ids[chosen].astype(np.int16),
    })
    if out.duplicated(["image_id", "query_index"]).any():
        raise RuntimeError("PCA unique-query sample contains duplicates")
    return out


def collect_pca_embeddings(reader: ReleaseReader, sample: pd.DataFrame) -> np.ndarray:
    wanted: dict[str, dict[int, int]] = {}
    for row in sample.itertuples(index=False):
        wanted.setdefault(str(row.image_id), {})[int(row.query_index)] = int(row.selection_order) - 1
    out = np.empty((len(sample), 256), dtype=np.float32)
    filled = np.zeros(len(sample), dtype=bool)
    for bundle in reader.iter_bundles("TRAIN"):
        targets = wanted.get(bundle.image_id)
        if not targets:
            continue
        first: dict[int, int] = {}
        for ix, q in enumerate(bundle.candidates["query_index"].to_numpy(np.int16)):
            first.setdefault(int(q), ix)
        for q, dst in targets.items():
            if q not in first:
                raise RuntimeError(f"PCA query absent from Top100: {bundle.image_id}/{q}")
            out[dst] = bundle.embeddings[first[q]].astype(np.float32)
            filled[dst] = True
    if not np.all(filled) or not np.all(np.isfinite(out)):
        raise RuntimeError(f"PCA embedding gather incomplete: {int((~filled).sum())}")
    return out


def label_audit_specs(fit_ids: set[str]) -> dict[str, tuple[int, int]]:
    ordered = sorted(
        ((hashlib.sha256(("LC_ALLOC_P1_LABEL_AUDIT|" + x).encode("utf-8")).digest(), x) for x in fit_ids),
        key=lambda z: (z[0], z[1]),
    )[:64]
    result = {}
    for digest, image_id in ordered:
        result[image_id] = (6 + digest[0] % 45, digest[1] % 10)
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", required=True)
    ap.add_argument("--release-root", required=True)
    args = ap.parse_args()
    root = Path(args.project_root).resolve()
    release_root = Path(args.release_root).resolve()
    t0 = time.perf_counter()
    for d in ("configs", "models", "outputs", "cache", "qa", "figures"):
        (root / d).mkdir(parents=True, exist_ok=True)
    append_log(root, "STAGE prepare_train START")

    reader = ReleaseReader(release_root)
    train_manifest = reader.split_manifest("TRAIN")
    roles = role_split(train_manifest)
    pq.write_table(pa.Table.from_pandas(roles, preserve_index=False), root / "train_role_split.parquet", compression="zstd")
    role_map = dict(zip(roles["image_id"].astype(str), roles["role"].astype(str)))
    fit_ids = {x for x, role in role_map.items() if role == "FIT"}

    spec = feature_spec()
    feature_schema = {
        "schema_name": "LC_ALLOC_P1_FEATURE_SCHEMA_V1",
        "dimension": len(spec),
        "columns": spec,
        "candidate_pool": "release raw road8_rank Top100 candidate records; no query deduplication",
        "action_state": "fixed raw-score prefix ranks 1..k-1",
        "pca": {
            "fit_role": "FIT only", "unique_key": ["image_id", "query_index"],
            "sample_rule": "first 200000 after SHA256('LC_ALLOC_P1_PCA|'+image_id+'|'+query_index), then image_id/query_index",
            "n_components": 32, "solver": "randomized", "random_state": 530100, "whiten": False,
            "input_storage": "release float16 embedding cast to float32; original float32 precision is not restored",
        },
        "standardization": {
            "fit_role": "FIT only", "population_variance": True,
            "excluded": "8 class one-hot fields and same_class_prefix_exists",
            "zero_variance_scale": 1.0,
        },
        "quantile_method": "linear", "score_std_ddof": 0,
        "bbox_policy": "no clipping; 1e-6 protection applies only inside log-aspect feature",
    }
    write_json(root / "feature_schema.json", feature_schema)
    feature_schema_sha = sha256_file(root / "feature_schema.json")

    release_manifest_sha = sha256_file(release_root / "RELEASE_MANIFEST.json")
    config = {
        "task": "LC-ALLOC-P1", "execution_stage": "TRAIN_PREPARATION",
        "release_root": str(release_root), "release_manifest_sha256": release_manifest_sha,
        "release_verify_status": "SHARED_ASSET_VERIFY_PASS",
        "candidate_asset_ids": {
            "TRAIN": reader.asset("TRAIN")["candidate_asset_id"],
            "DEV": reader.asset("DEV")["candidate_asset_id"],
        },
        "train_manifest_sha256": sha256_file(release_root / reader.asset("TRAIN")["split_manifest_path"]),
        "train_gt_sha256": sha256_file(release_root / "gt" / "TRAIN10K_ROAD8_GT.json"),
        "dev_gt_declared_sha256": "aead664f0d826b61a2f0ef190bb71b43af5b9ca45729ea467353680d3dcbe437",
        "feature_schema_sha256": feature_schema_sha,
        "roles": {"FIT": 8000, "EARLY_STOP": 1000, "CALIBRATION": 1000},
        "role_rule": "SHA256('LC_ALLOC_P1_SPLIT|'+image_id), image_id; first8000/next1000/last1000",
        "candidate_pool": "raw road8_rank 1..100", "variable_label_ranks": [6, 50],
        "iou_thresholds": THRESHOLDS.tolist(), "k_bounds": [5, 50], "budgets": [10, 15, 20, 30, 40],
        "model": {
            "architecture": "Linear(90,128)-LayerNorm-GELU-Linear(128,64)-GELU-Linear(64,10)",
            "loss": "unweighted natural BCEWithLogitsLoss mean over rows and outputs",
            "optimizer": "AdamW", "learning_rate": 0.001, "weight_decay": 0.0001,
            "batch_size": 4096, "max_epochs": 30, "early_stop_patience": 5,
            "early_stop_metric": "EARLY_STOP natural BCE; min_delta=0; earliest strict best",
            "seeds": [530101, 530102, 530103],
        },
        "calibration": {
            "role": "CALIBRATION", "temperature": "one shared positive scalar per seed",
            "bounds": [0.25, 4.0], "fallback": "T=1 if BCE improvement <=1e-12",
            "reliability_bins": "10 equal-width probability bins",
        },
        "table": {
            "fit_role": "FIT", "rank_bins": ["6-10", "11-15", "16-20", "21-30", "31-50"],
            "score_bins": ["[0,.05)", "[.05,.1)", "[.1,.2)", "[.2,.5)", "[.5,1]"],
            "rank_prior": "(positive_count+1)/(row_count+2)",
            "cell": "(positive_count+20*rank_prior)/(row_count+20)",
        },
        "quality_weight": "min(4,sqrt(nmax/max(nc,1))) normalized by FIT GT-weighted mean",
        "dp": {
            "solver": "exact multiple-choice dynamic programming float64",
            "tie_break": "process image_id ascending; exact objective ties choose the lexicographically smallest complete K vector",
            "group_rule": "frozen R0 group_manifest; no cross-group borrowing",
        },
        "bootstrap": {"unit": "frozen 40-image group", "groups": 50, "resamples": 5000, "seed": 530002},
        "objective_negative_rule": "97.5% CI upper bound <0; otherwise unsupported result is UNCONFIRMED",
        "high_budget_warning_rule": "apply the same AP/AR/class-recall engineering thresholds at K30 and K40",
        "runtime_repeats": {"cached_40_image_warmup": 5, "cached_40_image_measured": 20},
        "dev_boundary": "prediction/allocation entrypoint has no DEV GT argument; selections hashed before separate evaluation",
        "restricted_scientific_splits_accessed": 0,
    }
    write_json(root / "run_config.json", config)

    gt, gt_meta = load_coco_gt(release_root / "gt" / "TRAIN10K_ROAD8_GT.json")
    if set(gt) != set(train_manifest["image_id"].astype(str)):
        raise RuntimeError("TRAIN GT/manifest identity mismatch")
    class_counts, class_weights = compute_class_weights(gt, fit_ids)
    write_json(root / "models" / "class_weights.json", {
        "classes": list(ROAD8), "fit_gt_counts": class_counts.tolist(),
        "weights": class_weights.tolist(), "weighted_mean_check": float(np.sum(class_counts * class_weights) / np.sum(class_counts)),
        "formula": "a_c=min(4,sqrt(n_max/max(n_c,1))); Z=sum(n_c*a_c)/sum(n_c); w_c=a_c/Z",
    })

    sample = select_pca_identities(reader, fit_ids)
    pq.write_table(pa.Table.from_pandas(sample, preserve_index=False), root / "models" / "pca_sample_manifest.parquet", compression="zstd")
    append_log(root, f"PCA unique-query deterministic sample rows={len(sample)}")
    pca_input = collect_pca_embeddings(reader, sample)
    pca = PCA(n_components=32, svd_solver="randomized", random_state=530100, whiten=False)
    pca.fit(pca_input)
    joblib.dump(pca, root / "models" / "pca32.joblib")
    write_json(root / "qa" / "pca_fit.json", {
        "sample_rows": len(sample), "input_dtype_storage": "float16", "fit_cast_dtype": "float32",
        "components": 32, "explained_variance_ratio_sum_diagnostic_only": float(pca.explained_variance_ratio_.sum()),
        "sample_manifest_sha256": sha256_file(root / "models" / "pca_sample_manifest.parquet"),
    })
    del pca_input

    n_rows = 10_000 * 45
    x_raw = np.lib.format.open_memmap(root / "cache" / "train_X_raw.npy", mode="w+", dtype=np.float32, shape=(n_rows, 90))
    targets = np.lib.format.open_memmap(root / "cache" / "train_y.npy", mode="w+", dtype=np.uint8, shape=(n_rows, 10))
    role_codes = np.lib.format.open_memmap(root / "cache" / "train_role.npy", mode="w+", dtype=np.int8, shape=(n_rows,))
    audit_specs = label_audit_specs(fit_ids)
    audit_rows = []
    parity_fail = 0
    score_max_diff = 0.0
    row0 = 0
    label_path = root / "train_labels.parquet"
    writer = None
    try:
        batch_rows = []
        batch_x = []
        batch_y = []
        batch_roles = []
        for bundle in reader.iter_bundles("TRAIN"):
            c = bundle.candidates
            g = gt[bundle.image_id]
            classes = c["predicted_road8_class_id"].to_numpy(np.int16)
            boxes = c[["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]].to_numpy(np.float64)
            totals, _ = prefix_matching_counts(classes, boxes, g.classes, g.boxes, max_k=50)
            y = (totals[6:51] - totals[5:50]).astype(np.int8)
            if y.shape != (45, 10) or np.any((y < 0) | (y > 1)):
                raise RuntimeError(f"invalid prefix targets: {bundle.image_id}")
            if not np.array_equal(totals[50], totals[5] + y.sum(axis=0)):
                raise RuntimeError(f"cumulative label identity failed: {bundle.image_id}")
            direct50 = independent_prefix_count(classes, boxes, g.classes, g.boxes, 50, 0.50)
            parity_fail += int(direct50 != int(totals[50, 0]))

            cls0 = classes - 1
            reconstructed = 1.0 / (1.0 + np.exp(-bundle.road8_logits[np.arange(100), cls0].astype(np.float64)))
            score_max_diff = max(score_max_diff, float(np.max(np.abs(reconstructed - c["score"].to_numpy(np.float64)))))
            x = build_raw_features(c, bundle.road8_logits, bundle.embeddings, pca, bundle.width, bundle.height)
            role = role_map[bundle.image_id]
            current = c.iloc[5:50]
            local = pd.DataFrame({
                "image_id": bundle.image_id, "role": role,
                "road8_rank": current["road8_rank"].to_numpy(np.int16),
                "candidate_record_id": current["candidate_record_id"].astype(str).to_numpy(),
                "query_index": current["query_index"].to_numpy(np.int16),
                "predicted_road8_class_id": current["predicted_road8_class_id"].to_numpy(np.int8),
                "raw_score": current["score"].to_numpy(np.float64),
            })
            for ti, col in enumerate(Y_COLS):
                local[col] = y[:, ti].astype(np.uint8)
            batch_rows.append(local)
            batch_x.append(x.astype(np.float32))
            batch_y.append(y.astype(np.uint8))
            batch_roles.append(np.full(45, ROLE_TO_CODE[role], dtype=np.int8))

            if bundle.image_id in audit_specs:
                k, ti = audit_specs[bundle.image_id]
                before = independent_prefix_count(classes, boxes, g.classes, g.boxes, k - 1, float(THRESHOLDS[ti]))
                after = independent_prefix_count(classes, boxes, g.classes, g.boxes, k, float(THRESHOLDS[ti]))
                observed = int(y[k - 6, ti])
                audit_rows.append({
                    "image_id": bundle.image_id, "rank": k, "threshold": float(THRESHOLDS[ti]),
                    "independent_before": before, "independent_after": after,
                    "independent_delta": after - before, "stored_delta": observed,
                    "pass": bool(after - before == observed),
                })

            if len(batch_rows) >= 500:
                dfb = pd.concat(batch_rows, ignore_index=True)
                xb, yb, rb = np.vstack(batch_x), np.vstack(batch_y), np.concatenate(batch_roles)
                n = len(dfb)
                x_raw[row0:row0 + n] = xb
                targets[row0:row0 + n] = yb
                role_codes[row0:row0 + n] = rb
                table = pa.Table.from_pandas(dfb, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(label_path, table.schema, compression="zstd")
                writer.write_table(table)
                row0 += n
                batch_rows, batch_x, batch_y, batch_roles = [], [], [], []
        if batch_rows:
            dfb = pd.concat(batch_rows, ignore_index=True)
            xb, yb, rb = np.vstack(batch_x), np.vstack(batch_y), np.concatenate(batch_roles)
            n = len(dfb)
            x_raw[row0:row0 + n] = xb
            targets[row0:row0 + n] = yb
            role_codes[row0:row0 + n] = rb
            table = pa.Table.from_pandas(dfb, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(label_path, table.schema, compression="zstd")
            writer.write_table(table)
            row0 += n
    finally:
        if writer is not None:
            writer.close()
    x_raw.flush(); targets.flush(); role_codes.flush()
    if row0 != n_rows or parity_fail or score_max_diff > 1e-6:
        raise RuntimeError(f"TRAIN preparation invariant failed rows={row0}, parity={parity_fail}, score_diff={score_max_diff}")
    if len(audit_rows) != 64 or not all(x["pass"] for x in audit_rows):
        raise RuntimeError("64-unit independent label audit failed")
    write_json(root / "qa" / "label_audit.json", {
        "label_rows": n_rows, "role_rows": {k: int(np.sum(np.asarray(role_codes) == v)) for k, v in ROLE_TO_CODE.items()},
        "targets": Y_COLS, "values_only_0_or_1": True, "cumulative_identity_all_images": True,
        "independent_units": audit_rows, "independent_units_pass": True,
        "full_train_iou050_k50_independent_parity_mismatch": parity_fail,
        "candidate_native_score_max_abs_diff": score_max_diff,
        "train_gt_images": gt_meta["image_count"], "train_gt_annotations": gt_meta["annotation_count"],
    })

    standardize = np.asarray([bool(x["standardize_fit_only"]) for x in spec], dtype=bool)
    scaler = StandardScaler()
    for start in range(0, n_rows, 50_000):
        stop = min(start + 50_000, n_rows)
        mask = np.asarray(role_codes[start:stop]) == ROLE_TO_CODE["FIT"]
        if mask.any():
            scaler.partial_fit(np.asarray(x_raw[start:stop])[mask][:, standardize])
    scaler.scale_[scaler.scale_ == 0] = 1.0
    joblib.dump({"scaler": scaler, "standardize_mask": standardize, "feature_names": [x["name"] for x in spec]}, root / "models" / "feature_scaler.joblib")
    x_scaled = np.lib.format.open_memmap(root / "cache" / "train_X_scaled.npy", mode="w+", dtype=np.float32, shape=(n_rows, 90))
    for start in range(0, n_rows, 50_000):
        stop = min(start + 50_000, n_rows)
        z = np.asarray(x_raw[start:stop], dtype=np.float64).copy()
        z[:, standardize] = scaler.transform(z[:, standardize])
        x_scaled[start:stop] = z.astype(np.float32)
    x_scaled.flush()
    write_json(root / "qa" / "feature_audit.json", {
        "dimension": 90, "rows": n_rows, "all_finite_raw": bool(np.isfinite(np.asarray(x_raw)).all()),
        "all_finite_scaled": bool(np.isfinite(np.asarray(x_scaled)).all()),
        "standardized_columns": int(standardize.sum()), "unstandardized_columns": int((~standardize).sum()),
        "train_embedding_storage": "float16", "model_feature_cast": "float32",
    })

    labels_fit = pq.read_table(label_path, filters=[("role", "=", "FIT")]).to_pandas()
    table_model = fit_table(
        labels_fit["predicted_road8_class_id"].to_numpy(),
        labels_fit["road8_rank"].to_numpy(), labels_fit["raw_score"].to_numpy(),
        labels_fit[Y_COLS].to_numpy(np.uint8),
    )
    np.savez(
        root / "models" / "table_model.npz",
        rank_rows=table_model["rank_rows"], rank_pos=table_model["rank_pos"], rank_prior=table_model["rank_prior"],
        cell_rows=table_model["cell_rows"], cell_pos=table_model["cell_pos"], probabilities=table_model["probabilities"],
    )
    table_rows = []
    rank_names = ["6-10", "11-15", "16-20", "21-30", "31-50"]
    score_names = ["[0,.05)", "[.05,.1)", "[.1,.2)", "[.2,.5)", "[.5,1]"]
    for ci, cname in enumerate(ROAD8):
        for rb, rname in enumerate(rank_names):
            for sb, sname in enumerate(score_names):
                for ti, threshold in enumerate(THRESHOLDS):
                    table_rows.append({
                        "class_id": ci + 1, "class_name": cname, "rank_bin": rname, "score_bin": sname,
                        "threshold": float(threshold), "rank_row_count": int(table_model["rank_rows"][rb]),
                        "rank_positive_count": int(table_model["rank_pos"][rb, ti]),
                        "rank_prior": float(table_model["rank_prior"][rb, ti]),
                        "cell_row_count": int(table_model["cell_rows"][ci, rb, sb]),
                        "cell_positive_count": int(table_model["cell_pos"][ci, rb, sb, ti]),
                        "cell_probability": float(table_model["probabilities"][ci, rb, sb, ti]),
                    })
    pq.write_table(pa.Table.from_pylist(table_rows), root / "models" / "table_cells.parquet", compression="zstd")
    write_json(root / "qa" / "table_audit.json", {
        "fit_rows": len(labels_fit), "cells_by_threshold": 8 * 5 * 5,
        "probability_min": float(table_model["probabilities"].min()),
        "probability_max": float(table_model["probabilities"].max()),
        "empty_cell_count": int(np.sum(table_model["cell_rows"] == 0)),
        "empty_cells_equal_rank_prior": bool(np.allclose(
            table_model["probabilities"][table_model["cell_rows"] == 0],
            np.broadcast_to(table_model["rank_prior"][None, :, None, :], (8, 5, 5, 10))[table_model["cell_rows"] == 0],
            atol=0, rtol=0,
        )),
    })

    append_log(root, f"STAGE prepare_train COMPLETE elapsed_seconds={time.perf_counter()-t0:.6f}")
    write_json(root / "qa" / "prepare_train_complete.json", {
        "status": "PASS", "elapsed_seconds": time.perf_counter() - t0,
        "feature_schema_sha256": feature_schema_sha, "train_labels_sha256": sha256_file(label_path),
        "pca_sha256": sha256_file(root / "models" / "pca32.joblib"),
        "scaler_sha256": sha256_file(root / "models" / "feature_scaler.joblib"),
        "table_sha256": sha256_file(root / "models" / "table_model.npz"),
    })


if __name__ == "__main__":
    main()
