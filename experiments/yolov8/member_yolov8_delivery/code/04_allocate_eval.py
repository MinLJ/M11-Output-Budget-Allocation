# -*- coding: utf-8 -*-
"""04 — allocate (M11 / S_ADAPT / S_FIXED) on DEV and evaluate coverage/QUALITY.

Loads the trained YOLO asset (PCA/scaler/3-seed MLP/temperature/class weights),
builds DEV features, solves exact-prefix allocations per 40-image group, then
evaluates coverage + QUALITY and runs the paired-group bootstrap (M11 vs
S_ADAPT).

The per-image prefix matching (the expensive part) is computed ONCE and cached;
each allocation condition only does an O(1) lookup per image, mirroring the
semantics of ``evaluate_prefix_allocations`` without re-running matching 25x.
"""
from __future__ import annotations
import os

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

BASE = Path(__file__).resolve().parent
M11 = Path(os.environ.get("M11_EXPERIMENT_ROOT", "external_assets")) / "material from memberA" / "LC_ALLOC_M11_HANDOFF_v1" / "handoff_LC_ALLOC_M11_v1"
if str(M11) not in sys.path:
    sys.path.insert(0, str(M11))
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from lc_alloc.allocation.policies import allocate_m11, allocate_s_adapt, allocate_s_fixed  # noqa: E402
from lc_alloc.constants import K_MAX, THRESHOLDS  # noqa: E402
from lc_alloc.data.io import read_candidates, read_native  # noqa: E402
from lc_alloc.data.schema import NativeTable  # noqa: E402
from lc_alloc.evaluation.bootstrap import paired_group_bootstrap  # noqa: E402
from lc_alloc.features._executed_p1_core import prefix_matching_counts  # noqa: E402
from lc_alloc.features.r18 import transform_features  # noqa: E402
from lc_alloc.labels.prefix import load_normalized_gt  # noqa: E402
from lc_alloc.models.assets import predict_probabilities  # noqa: E402
from lc_alloc.models.mlp import MarginalMLP  # noqa: E402
from lc_alloc.utils import read_json  # noqa: E402

from m11_yolo.config import SEEDS, YOLO_NATIVE_SCHEMA_ID  # noqa: E402
from m11_yolo.features import build_image_features  # noqa: E402

BUDGETS = (10, 15, 20, 30, 40)


def _index_native(native: NativeTable):
    iid = np.asarray(native.image_ids).astype(str)
    sid = np.asarray(native.source_ids).astype(str)
    vec = np.asarray(native.native_vectors)
    sig = np.asarray(native.class_signals)
    groups = pd.DataFrame({"iid": iid}).groupby("iid", sort=False).indices
    return iid, sid, vec, sig, groups, native.native_schema_id


def _native_subset(ndx, image_id: str) -> NativeTable:
    iid, sid, vec, sig, groups, schema = ndx
    idx = groups[image_id]
    return NativeTable(iid[idx], sid[idx], vec[idx], sig[idx], schema)


def load_asset(asset_dir: Path, device: str):
    pca = joblib.load(asset_dir / "pca32.joblib")
    scaler_bundle = joblib.load(asset_dir / "scaler.joblib")
    class_weights = np.asarray(read_json(asset_dir / "class_weights.json")["weights"], dtype=np.float64)
    models, temps = {}, {}
    for seed in SEEDS:
        snap = torch.load(asset_dir / "models" / f"marginal_mlp_seed_{seed}.pt", map_location="cpu", weights_only=True)
        model = MarginalMLP(int(snap["input_dim"]))
        model.load_state_dict(snap["state_dict"], strict=True)
        model.eval().to(torch.device(device))
        for p in model.parameters():
            p.requires_grad_(False)
        models[seed] = model
        temps[seed] = float(read_json(asset_dir / "models" / f"temperature_seed_{seed}.json")["temperature"])
    return pca, scaler_bundle, class_weights, models, temps


def _evaluate(cache, allocations, class_weights):
    """Replicate evaluate_prefix_allocations semantics from a cached match table."""
    weights = np.asarray(class_weights, dtype=np.float64)
    a = allocations.copy()
    if a["image_id"].astype(str).duplicated().any():
        raise ValueError("allocation must contain one row per image")
    k_num = pd.to_numeric(a["K_i"], errors="coerce").to_numpy(np.float64)
    if not np.isfinite(k_num).all() or not np.equal(k_num, np.floor(k_num)).all():
        raise ValueError("K must be finite integers")
    a["K_i"] = k_num.astype(np.int64)
    if not a["K_i"].between(5, 50).all():
        raise ValueError("K outside [5,50]")
    if {"group_id", "budget"}.issubset(a.columns):
        for (group_id, budget), group in a.groupby(["group_id", "budget"], dropna=False, sort=False):
            expected = len(group) * int(budget)
            if int(group["K_i"].sum()) != expected:
                raise ValueError(f"exact group budget failed for group={group_id}: {int(group['K_i'].sum())} != {expected}")

    rows = []
    coverage_total = 0
    quality_total = 0.0
    valid_gt_total = 0
    class_coverage = np.zeros(8, dtype=np.int64)
    class_gt = np.zeros(8, dtype=np.int64)
    for allocation in a.itertuples(index=False):
        image_id = str(allocation.image_id)
        k = int(allocation.K_i)
        totals, by_class, gt_count, gt_bincount = cache[image_id]
        coverage = int(totals[k, 0])
        quality = float((weights * by_class[k].mean(axis=1).astype(np.float64)).sum())
        coverage_total += coverage
        quality_total += quality
        valid_gt_total += gt_count
        class_coverage += by_class[k, :, 0]
        class_gt += gt_bincount
        rows.append({"image_id": image_id, "K_i": k, "valid_gt": gt_count, "coverage": coverage, "quality": quality, "coverage_recall_image": coverage / gt_count if gt_count else np.nan})
    result = pd.DataFrame(rows).sort_values("image_id", kind="mergesort").reset_index(drop=True)
    summary = {
        "image_count": len(result),
        "output_records": int(result["K_i"].sum()),
        "coverage_total": int(coverage_total),
        "coverage_per_image": float(coverage_total / len(result)),
        "coverage_recall": float(coverage_total / valid_gt_total) if valid_gt_total else None,
        "quality_total": float(quality_total),
        "quality_per_image": float(quality_total / len(result)),
        "valid_gt_total": int(valid_gt_total),
        "empty_valid_gt_images": int((result["valid_gt"] == 0).sum()),
        "class_coverage": class_coverage.tolist(),
        "class_gt": class_gt.tolist(),
    }
    return result, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev-candidates", required=True)
    ap.add_argument("--dev-native", required=True)
    ap.add_argument("--dev-gt", required=True)
    ap.add_argument("--groups", required=True)
    ap.add_argument("--assets", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = args.device

    pca, scaler_bundle, class_weights, models, temps = load_asset(Path(args.assets), device)
    candidates = read_candidates(args.dev_candidates)
    candidates["image_id"] = candidates["image_id"].astype(str)
    native = read_native(args.dev_native, expected_schema_id=YOLO_NATIVE_SCHEMA_ID)
    gt = load_normalized_gt(args.dev_gt)
    groups = pd.read_csv(args.groups)
    groups["image_id"] = groups["image_id"].astype(str)

    frames = {iid: g for iid, g in candidates.groupby("image_id", sort=False)}
    image_ids = sorted(frames)
    ndx = _index_native(native)

    # --- per-image cached prefix matching (once) ---
    cache = {}
    s_adapt_scores = {}
    for image_id in image_ids:
        frame = frames[image_id].sort_values("rank", kind="mergesort")
        cls = frame["predicted_road8_class_id"].to_numpy(np.int16)
        boxes = frame[["box_x1", "box_y1", "box_x2", "box_y2"]].to_numpy(np.float64)
        truth = gt[image_id]
        totals, by_class = prefix_matching_counts(cls, boxes, truth.classes, truth.boxes, max_k=K_MAX, thresholds=THRESHOLDS)
        cache[image_id] = (totals, by_class, len(truth.classes), np.bincount(truth.classes, minlength=9)[1:9])
        s_adapt_scores[image_id] = frame["original_score"].to_numpy(np.float64)[:K_MAX]

    # --- features + predicted marginals ---
    scaled = {}
    cls_by_image = {}
    for image_id in image_ids:
        raw = build_image_features(frames[image_id], _native_subset(ndx, image_id), pca)
        scaled[image_id] = transform_features(raw, scaler_bundle)
        cls_by_image[image_id] = frames[image_id].sort_values("rank", kind="mergesort").iloc[5:50]["predicted_road8_class_id"].to_numpy(np.int64)

    marginals = {seed: {} for seed in models}
    for seed, model in models.items():
        with torch.inference_mode():
            for image_id in image_ids:
                prob = predict_probabilities(model, scaled[image_id], temps[seed], device=device)
                marginals[seed][image_id] = prob.mean(axis=1) * class_weights[cls_by_image[image_id] - 1]

    # --- allocations ---
    alloc_parts = []
    for group_id, group in groups.groupby("group_id", sort=True):
        gids = sorted(group.image_id.astype(str))
        for method in ("M11", "S_ADAPT", "S_FIXED"):
            if method == "M11":
                for seed in SEEDS:
                    solved, _ = allocate_m11(np.vstack([marginals[seed][iid] for iid in gids]), BUDGETS)
                    for bi, budget in enumerate(BUDGETS):
                        alloc_parts.append(pd.DataFrame({"group_id": group_id, "image_id": gids, "K_i": solved[bi], "method": "M11", "seed": seed, "budget": budget}))
            elif method == "S_ADAPT":
                sub = pd.concat([frames[iid] for iid in gids], ignore_index=True)
                solved, _ = allocate_s_adapt(sub, gids, BUDGETS)
                for bi, budget in enumerate(BUDGETS):
                    alloc_parts.append(pd.DataFrame({"group_id": group_id, "image_id": gids, "K_i": solved[bi], "method": "S_ADAPT", "seed": -1, "budget": budget}))
            else:
                solved = allocate_s_fixed(len(gids), BUDGETS)
                for bi, budget in enumerate(BUDGETS):
                    alloc_parts.append(pd.DataFrame({"group_id": group_id, "image_id": gids, "K_i": solved[bi], "method": "S_FIXED", "seed": -1, "budget": budget}))
    allocations = pd.concat(alloc_parts, ignore_index=True)
    allocations.to_parquet(out / "allocations.parquet", index=False)

    # --- evaluate (per-image + summary per condition) from cache ---
    group_map = dict(zip(groups.image_id, groups.group_id))
    condition_columns = ["method", "seed", "budget"]
    per_image_parts = []
    summaries = []
    for key, frame in allocations.groupby(condition_columns, dropna=False, sort=True):
        eval_cols = [c for c in ("image_id", "K_i", "group_id", "budget") if c in frame.columns]
        per_image, summary = _evaluate(cache, frame[eval_cols], class_weights)
        key_vals = key if isinstance(key, tuple) else (key,)
        for col, val in zip(condition_columns, key_vals):
            per_image[col] = val
            summary[col] = val
        per_image["group_id"] = per_image["image_id"].map(group_map)
        per_image_parts.append(per_image)
        summaries.append(summary)
    per_image = pd.concat(per_image_parts, ignore_index=True)
    per_image.to_csv(out / "per_image_results.csv", index=False)
    main = pd.DataFrame(summaries)
    main.to_csv(out / "main_results.csv", index=False)

    # --- paired-group bootstrap: M11 vs S_ADAPT per seed per budget ---
    boot_rows = []
    for seed in SEEDS:
        for budget in BUDGETS:
            m = per_image[(per_image.method == "M11") & (per_image.seed == seed) & (per_image.budget == budget)]
            s = per_image[(per_image.method == "S_ADAPT") & (per_image.budget == budget)]
            if m.empty or s.empty:
                continue
            m_g = m.groupby("group_id")["coverage"].mean().sort_index()
            s_g = s.groupby("group_id")["coverage"].mean().sort_index()
            common = m_g.index.intersection(s_g.index)
            if len(common) != len(m_g):
                raise ValueError("group identity mismatch in bootstrap")
            cov = paired_group_bootstrap((m_g.loc[common] - s_g.loc[common]).to_numpy(np.float64), resamples=5000, seed=530002)
            mq = m.groupby("group_id")["quality"].mean().sort_index()
            sq = s.groupby("group_id")["quality"].mean().sort_index()
            qual = paired_group_bootstrap((mq.loc[common] - sq.loc[common]).to_numpy(np.float64), resamples=5000, seed=530002)
            boot_rows.append({
                "seed": int(seed), "budget": int(budget),
                "m11_coverage_per_image": float(m_g.mean()), "s_adapt_coverage_per_image": float(s_g.mean()),
                "coverage_diff": cov["point_estimate"], "coverage_ci95_low": cov["ci95_low"], "coverage_ci95_high": cov["ci95_high"],
                "quality_diff": qual["point_estimate"], "quality_ci95_low": qual["ci95_low"], "quality_ci95_high": qual["ci95_high"],
            })
    boot = pd.DataFrame(boot_rows)
    boot.to_csv(out / "paired_group_bootstrap.csv", index=False)

    key_cols = ["method", "seed", "budget", "coverage_per_image", "quality_per_image", "coverage_total", "output_records"]
    print(main[[c for c in key_cols if c in main.columns]].to_string(index=False))
    print(json.dumps({"status": "PASS", "conditions": len(main), "bootstrap_rows": len(boot)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
