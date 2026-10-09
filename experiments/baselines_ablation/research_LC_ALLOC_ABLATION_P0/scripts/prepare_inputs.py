from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from common import THRESHOLDS, add_p1_scripts, append_log, sha256_file, write_json  # noqa: E402


def role_codes(values: pd.Series) -> np.ndarray:
    mapping = {"FIT": 0, "EARLY_STOP": 1, "CALIBRATION": 2}
    out = values.map(mapping)
    if out.isna().any():
        raise RuntimeError("unknown TRAIN role in label metadata")
    return out.to_numpy(np.int8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", required=True)
    ap.add_argument("--p1-root", required=True)
    ap.add_argument("--p1b-root", required=True)
    ap.add_argument("--audit-root", required=True)
    ap.add_argument("--release-root", required=True)
    args = ap.parse_args()
    root = Path(args.project_root).resolve()
    p1 = Path(args.p1_root).resolve()
    p1b = Path(args.p1b_root).resolve()
    audit = Path(args.audit_root).resolve()
    release = Path(args.release_root).resolve()
    start = time.perf_counter()
    append_log(root, "STAGE prepare_inputs START (TRAIN GT permitted; TEST/RESERVE inaccessible by interface)")

    add_p1_scripts(p1)
    from p1_core import iou_xyxy, load_coco_gt  # noqa: E402

    required = [
        p1 / "run_config.json", p1 / "feature_schema.json",
        p1 / "cache" / "train_X_scaled.npy", p1 / "cache" / "train_y.npy",
        p1 / "cache" / "train_role.npy", p1 / "cache" / "dev_X_scaled.npy",
        p1 / "train_labels.parquet", p1 / "dev_predictions.parquet",
        p1 / "models" / "class_weights.json", p1 / "models" / "pca32.joblib",
        p1 / "models" / "feature_scaler.joblib", p1 / "cache" / "dev_allocations.parquet",
        p1 / "input_output_sha256.csv", p1 / "outputs" / "dev_selection_commit.json",
        p1 / "per_image_results.parquet", p1 / "main_results.parquet", p1 / "class_results.parquet",
        p1 / "outputs" / "bootstrap_group_indices.npy",
        p1 / "scripts" / "p1_core.py", p1 / "scripts" / "dp_solver.py",
        p1b / "outputs" / "implementation_status.json",
        p1b / "input_output_sha256.csv", p1b / "scripts" / "profile_optimizations.py",
        audit / "verified_method_spec.json", audit / "feature_schema_audited.csv",
        release / "CANDIDATE_ASSET_IDENTITY.json", release / "gt" / "TRAIN10K_ROAD8_GT.json",
    ]
    for seed in (530101, 530102, 530103):
        required.extend([
            p1 / "models" / f"marginal_mlp_seed_{seed}.pt",
            p1 / "models" / f"temperature_seed_{seed}.json",
        ])
    missing = [str(x) for x in required if not x.is_file()]
    if missing:
        raise FileNotFoundError(f"missing required inputs: {missing}")

    schema = json.loads((p1 / "feature_schema.json").read_text(encoding="utf-8"))
    cols = sorted(schema["columns"], key=lambda x: int(x["index"]))
    if [int(x["index"]) for x in cols] != list(range(90)):
        raise RuntimeError("frozen feature schema is not exact ordered 90-D")
    embed_names = {f"embedding_pca_{j:02d}" for j in range(32)} | {"embedding_norm"}
    embed = [x for x in cols if str(x["name"]) in embed_names]
    prefix = [x for x in cols if str(x["block"]) == "prefix"]
    if len(embed) != 33 or len(prefix) != 12:
        raise RuntimeError(f"mask cardinality mismatch embedding={len(embed)} prefix={len(prefix)}")
    if {str(x["name"]) for x in embed} != embed_names:
        raise RuntimeError("embedding mask names are incomplete")
    masks = {
        "schema_path": str(p1 / "feature_schema.json"),
        "schema_sha256": sha256_file(p1 / "feature_schema.json"),
        "application_stage": "after frozen FIT-only standardization",
        "fill_value": 0.0,
        "model_input_dimension": 90,
        "NO_EMBED": {
            "indices": [int(x["index"]) for x in embed],
            "names": [str(x["name"]) for x in embed],
            "masked_dimensions": 33,
            "effective_dimensions": 57,
            "does_not_remove_native_logits": True,
        },
        "NO_PREFIX": {
            "indices": [int(x["index"]) for x in prefix],
            "names": [str(x["name"]) for x in prefix],
            "masked_dimensions": 12,
            "effective_dimensions": 78,
            "retains_rank_and_image_context": True,
        },
        "MATCHABILITY_TARGET": {"indices": [], "names": [], "masked_dimensions": 0, "effective_dimensions": 90},
    }
    write_json(root / "feature_masks.json", masks)

    x_train = np.load(p1 / "cache" / "train_X_scaled.npy", mmap_mode="r")
    y_marginal = np.load(p1 / "cache" / "train_y.npy", mmap_mode="r")
    role = np.load(p1 / "cache" / "train_role.npy", mmap_mode="r")
    x_dev = np.load(p1 / "cache" / "dev_X_scaled.npy", mmap_mode="r")
    if x_train.shape != (450000, 90) or y_marginal.shape != (450000, 10) or role.shape != (450000,) or x_dev.shape != (90000, 90):
        raise RuntimeError("frozen cache shapes do not match audited P1 specification")
    if not np.isfinite(x_train).all() or not np.isfinite(x_dev).all():
        raise RuntimeError("nonfinite value in frozen feature cache")

    label_path = p1 / "train_labels.parquet"
    labels = pq.read_table(label_path).to_pandas()
    y_cols = [f"y_iou_{v:.2f}".replace(".", "_") for v in THRESHOLDS]
    if len(labels) != 450000 or labels.groupby("image_id", sort=False).size().nunique() != 1 or int(labels.groupby("image_id", sort=False).size().iloc[0]) != 45:
        raise RuntimeError("P1 train label row identity invariant failed")
    if not np.array_equal(labels[y_cols].to_numpy(np.uint8), np.asarray(y_marginal)):
        raise RuntimeError("unledgered train_y cache differs from frozen label parquet")
    if not np.array_equal(role_codes(labels["role"]), np.asarray(role)):
        raise RuntimeError("unledgered train_role cache differs from frozen label parquet")

    identity = json.loads((release / "CANDIDATE_ASSET_IDENTITY.json").read_text(encoding="utf-8"))
    candidate_cols = [
        "image_id", "road8_rank", "candidate_record_id", "query_index", "predicted_road8_class_id", "score",
        "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
    ]
    candidate_parts = []
    for rel in identity["assets"]["TRAIN"]["candidate_shards"]:
        part = pq.read_table(release / rel, columns=candidate_cols).to_pandas()
        part = part[(part["road8_rank"] >= 6) & (part["road8_rank"] <= 50)].copy()
        candidate_parts.append(part)
    candidates = pd.concat(candidate_parts, ignore_index=True)
    if len(candidates) != 450000 or candidates["candidate_record_id"].duplicated().any():
        raise RuntimeError("TRAIN candidate rank6..50 identity/count invariant failed")
    candidate_by_id = candidates.set_index("candidate_record_id", verify_integrity=True)
    ids = labels["candidate_record_id"].astype(str)
    if not ids.isin(candidate_by_id.index.astype(str)).all():
        raise RuntimeError("P1 label candidate identities do not map to release")
    aligned = candidate_by_id.loc[ids].reset_index()
    exact_fields = ["image_id", "road8_rank", "candidate_record_id", "query_index", "predicted_road8_class_id"]
    for field in exact_fields:
        if not np.array_equal(labels[field].astype(str if field in ("image_id", "candidate_record_id") else np.int64).to_numpy(), aligned[field].astype(str if field in ("image_id", "candidate_record_id") else np.int64).to_numpy()):
            raise RuntimeError(f"TRAIN label/candidate identity mismatch: {field}")
    raw_diff = float(np.max(np.abs(labels["raw_score"].to_numpy(np.float64) - aligned["score"].to_numpy(np.float64))))
    if raw_diff > 1e-7:
        raise RuntimeError(f"TRAIN label/release raw score mismatch {raw_diff}")

    gt, gt_meta = load_coco_gt(release / "gt" / "TRAIN10K_ROAD8_GT.json")
    if gt_meta["image_count"] != 10000 or set(labels["image_id"].astype(str).unique()) != set(gt):
        raise RuntimeError("TRAIN GT identity coverage mismatch")
    match = np.zeros((450000, 10), dtype=np.uint8)
    processed = 0
    for image_id, index in labels.groupby("image_id", sort=False).groups.items():
        ix = np.asarray(index, dtype=np.int64)
        rows = aligned.iloc[ix]
        if rows["road8_rank"].astype(int).tolist() != list(range(6, 51)):
            raise RuntimeError(f"TRAIN rank order mismatch {image_id}")
        g = gt[str(image_id)]
        boxes = rows[["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]].to_numpy(np.float64)
        cls = rows["predicted_road8_class_id"].to_numpy(np.int16)
        overlap = iou_xyxy(boxes, g.boxes)
        same = cls[:, None] == g.classes[None, :] if len(g.classes) else np.zeros((45, 0), dtype=bool)
        for ti, threshold in enumerate(THRESHOLDS):
            match[ix, ti] = np.any(same & (overlap >= float(threshold)), axis=1).astype(np.uint8) if len(g.classes) else 0
        processed += 1
        if processed % 1000 == 0:
            append_log(root, f"MATCHABILITY_LABEL_PROGRESS images={processed}")
    if not np.isin(match, [0, 1]).all() or not np.all(match[:, 1:] <= match[:, :-1]):
        raise RuntimeError("matchability target binary/threshold monotonicity invariant failed")
    marginal_not_matchable = int(np.sum(np.asarray(y_marginal, dtype=np.uint8) > match))
    if marginal_not_matchable:
        raise RuntimeError(f"prefix marginal positive without candidate matchability: {marginal_not_matchable}")
    match_path = root / "cache" / "train_y_matchability.npy"
    np.save(match_path, match, allow_pickle=False)

    role_names = {0: "FIT", 1: "EARLY_STOP", 2: "CALIBRATION"}
    stats = []
    for code, name in role_names.items():
        mask = np.asarray(role) == code
        for ti, threshold in enumerate(THRESHOLDS):
            stats.append({
                "role": name, "threshold": float(threshold), "rows": int(mask.sum()),
                "marginal_positives": int(np.asarray(y_marginal)[mask, ti].sum()),
                "marginal_positive_rate": float(np.asarray(y_marginal)[mask, ti].mean()),
                "matchability_positives": int(match[mask, ti].sum()),
                "matchability_positive_rate": float(match[mask, ti].mean()),
            })
    write_json(root / "outputs" / "matchability_label_summary.json", {
        "status": "PASS", "rows": 450000, "images": 10000,
        "target_definition": "candidate matches any valid same-class GT at threshold; independent of prefix coverage",
        "thresholds": THRESHOLDS.tolist(), "binary": True, "threshold_axis_nonincreasing": True,
        "marginal_positive_not_matchable": marginal_not_matchable,
        "release_candidate_score_max_abs_diff": raw_diff,
        "statistics": stats,
        "matchability_target_path": str(match_path), "matchability_target_sha256": sha256_file(match_path),
    })

    binding = {str(p): sha256_file(p) for p in required}
    binding.update({
        str(p1 / "cache" / "train_X_scaled.npy"): sha256_file(p1 / "cache" / "train_X_scaled.npy"),
        str(p1 / "cache" / "train_y.npy"): sha256_file(p1 / "cache" / "train_y.npy"),
        str(p1 / "cache" / "train_role.npy"): sha256_file(p1 / "cache" / "train_role.npy"),
        str(p1 / "cache" / "dev_X_scaled.npy"): sha256_file(p1 / "cache" / "dev_X_scaled.npy"),
    })
    write_json(root / "outputs" / "input_binding.json", {"files": binding, "restricted_scientific_splits_accessed": 0})
    run_config = {
        "task": "LC-ALLOC-ABLATION-P0", "analysis_type": "post-confirmation DEV ablation",
        "project_root": str(root), "p1_root": str(p1), "p1b_root": str(p1b), "method_audit_root": str(audit),
        "release_root": str(release), "candidate_pool": "original road8_rank Top100 candidate records",
        "action": "output raw-score prefix ranks 1..K_i only", "groups": "frozen DEV 50x40 groups",
        "budgets": [10, 15, 20, 30, 40], "k_bounds": [5, 50], "exact_group_budget": True,
        "seeds": [530101, 530102, 530103], "new_training_variants": ["NO_EMBED", "NO_PREFIX", "MATCHABILITY_TARGET"],
        "solver_control": "NEXT_SLOT_GREEDY on frozen FULL/M11 probabilities",
        "training": {"architecture": "90-128-LayerNorm-GELU-64-GELU-10", "optimizer": "AdamW", "lr": 0.001, "weight_decay": 0.0001, "batch_size": 4096, "max_epochs": 30, "patience": 5, "loss": "unweighted natural BCE mean"},
        "temperature": {"one_per_variant_seed": True, "bounds": [0.25, 4.0], "fit_role": "CALIBRATION"},
        "bootstrap": {"groups": 50, "resamples": 5000, "seed": 530002},
        "runtime": {"seed": 530101, "budget": 15, "groups": 5, "warmups": 5, "measurements_per_group": 20, "order_seed": 530003},
        "restricted_splits_forbidden": ["TEST", "RESERVE", "old holdout", "Road1000"],
        "test_access": 0, "detector_forward": False, "candidate_export": False,
    }
    write_json(root / "run_config.json", run_config)
    append_log(root, f"STAGE prepare_inputs COMPLETE elapsed_seconds={time.perf_counter()-start:.6f}")


if __name__ == "__main__":
    main()
