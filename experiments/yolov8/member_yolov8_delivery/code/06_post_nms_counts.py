# -*- coding: utf-8 -*-
"""06 — NMS-survivor counts for the frozen prefix selections (guideline Y2).

Post-processing record only: reads the FROZEN allocations from 04, takes the
prefix ranks 1..K_i, applies the SAME class-aware NMS (IoU=0.70) as 05 via
``m11_yolo.nms``, and counts survivors.  It does NOT load a trainer, does NOT
modify K_i, and does NOT search NMS thresholds.

Outputs (all per the guideline 表7 / §7):
  post_nms_counts.csv        per method/seed/budget/image: n_before, n_after,
                             removed_count, surviving record IDs
  post_nms_group_summary.csv per group totals
  post_nms_overall.csv       overall mean / min / max + empty-output image count

Invariants enforced: removed_count == n_before - n_after and n_after <= n_before.
Removed slots are NOT backfilled.
"""
from __future__ import annotations
import os

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
M11 = Path(os.environ.get("M11_EXPERIMENT_ROOT", "external_assets")) / "material from memberA" / "LC_ALLOC_M11_HANDOFF_v1" / "handoff_LC_ALLOC_M11_v1"
if str(M11) not in sys.path:
    sys.path.insert(0, str(M11))
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from m11_yolo.nms import class_aware_nms  # noqa: E402

NMS_IOU = 0.70


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev-candidates", required=True)
    ap.add_argument("--allocations", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--groups", default="", help="dev_groups.csv for group_id mapping")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cand = pd.read_parquet(args.dev_candidates)
    cand["image_id"] = cand["image_id"].astype(str)
    cand["rank"] = cand["rank"].astype(int)
    frames = {
        iid: g.sort_values("rank", kind="mergesort").reset_index(drop=True)
        for iid, g in cand.groupby("image_id", sort=False)
    }

    group_map = {}
    if args.groups:
        gdf = pd.read_csv(args.groups)
        gdf["image_id"] = gdf["image_id"].astype(str)
        group_map = dict(zip(gdf.image_id, gdf.group_id))

    alloc = pd.read_parquet(args.allocations)
    alloc["image_id"] = alloc["image_id"].astype(str)
    alloc["K_i"] = alloc["K_i"].astype(int)

    rows = []
    for (method, seed, budget), sub in alloc.groupby(["method", "seed", "budget"], sort=True):
        for image_id, k in zip(sub["image_id"], sub["K_i"]):
            f = frames[image_id]
            f = f[f["rank"] <= k]
            n_before = len(f)
            if n_before == 0:
                rows.append({"method": method, "seed": int(seed), "budget": int(budget),
                             "image_id": image_id, "group_id": group_map.get(image_id),
                             "n_before": 0, "n_after": 0, "removed_count": 0,
                             "surviving_record_ids": ""})
                continue
            boxes = f[["box_x1", "box_y1", "box_x2", "box_y2"]].to_numpy(np.float64)
            scores = f["original_score"].to_numpy(np.float64)
            classes = f["predicted_road8_class_id"].to_numpy(np.int64)
            keep = class_aware_nms(boxes, scores, classes, NMS_IOU)
            n_after = int(len(keep))
            surv_ids = f["candidate_record_id"].to_numpy()[keep]
            rows.append({
                "method": method, "seed": int(seed), "budget": int(budget),
                "image_id": image_id, "group_id": group_map.get(image_id),
                "n_before": n_before, "n_after": n_after,
                "removed_count": n_before - n_after,
                "surviving_record_ids": "|".join(str(x) for x in surv_ids),
            })
        print(f"  {method} seed={seed} b={budget} done", flush=True)

    df = pd.DataFrame(rows)
    # --- invariants ---
    assert (df["removed_count"] == df["n_before"] - df["n_after"]).all(), "removed != before - after"
    assert (df["n_after"] <= df["n_before"]).all(), "n_after > n_before"
    assert df["n_before"].sum() == alloc["K_i"].sum(), "n_before total != exact budget total"
    df.to_csv(out / "post_nms_counts.csv", index=False)

    grp = df.groupby(["method", "seed", "budget", "group_id"], dropna=False).agg(
        images=("image_id", "size"),
        n_before_total=("n_before", "sum"),
        n_after_total=("n_after", "sum"),
        removed_total=("removed_count", "sum"),
    ).reset_index()
    grp.to_csv(out / "post_nms_group_summary.csv", index=False)

    ovr = df.groupby(["method", "seed", "budget"]).agg(
        images=("image_id", "size"),
        n_before_mean=("n_before", "mean"), n_before_min=("n_before", "min"), n_before_max=("n_before", "max"),
        n_after_mean=("n_after", "mean"), n_after_min=("n_after", "min"), n_after_max=("n_after", "max"),
        removed_total=("removed_count", "sum"),
        empty_output_images=("n_after", lambda s: int((s == 0).sum())),
    ).reset_index()
    ovr.to_csv(out / "post_nms_overall.csv", index=False)

    print(json.dumps({
        "status": "PASS", "nms_iou": NMS_IOU, "rows": len(df),
        "n_before_total": int(df.n_before.sum()), "n_after_total": int(df.n_after.sum()),
        "removed_total": int(df.removed_count.sum()),
        "empty_output_images": int((df.n_after == 0).sum()),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
