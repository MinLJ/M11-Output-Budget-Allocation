# -*- coding: utf-8 -*-
"""05 — standard COCO AP/AP50/AP75/AR100 on the FROZEN selected prefix sets.

GPT review #2 item 5.  Coverage does not reflect the precision cost of
duplicate hypotheses, so we add the standard detection metrics.

Protocol (IMPORTANT, must be disclosed):
  * The selected set for a condition is the frozen prefix ranks 1..K_i of the
    pre-NMS Top-100 candidate stream (recovered from work/eval/allocations.parquet
    + candidates.parquet; NO re-solving, NO re-training, NO detector re-run).
  * Scores are the ORIGINAL detector scores, unchanged.
  * Action space A (primary): the frozen selection as-is (no NMS) — this is the
    exact action space M11/S_ADAPT/S_FIXED allocate over.
  * Action space B (reference only, SEPARATE table): the same selection with NMS
    applied per image.  This CHANGES the candidate set, so it is NOT merged with
    the primary prefix table.

Evaluated with pycocotools COCOeval (iouThrs 0.50:0.05:0.95, 101-point
interpolation, maxDets 100) on all 2000 DEV images / 29966 GT boxes.
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

from lc_alloc.labels.prefix import load_normalized_gt  # noqa: E402

ROAD8_CN = ["人", "自行车", "轿车", "摩托车", "公交车", "火车", "卡车", "交通灯"]


def build_gt(gt_map) -> tuple[list[dict], dict[str, int]]:
    ids = sorted(gt_map)
    id2int = {iid: i + 1 for i, iid in enumerate(ids)}
    anns = []
    ann_id = 1
    for iid in ids:
        g = gt_map[iid]
        for cls, box in zip(g.classes, g.boxes):
            x1, y1, x2, y2 = (float(v) for v in box)
            anns.append({
                "id": ann_id,
                "image_id": id2int[iid],
                "category_id": int(cls),
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "area": (x2 - x1) * (y2 - y1),
                "iscrowd": 0,
            })
            ann_id += 1
    return anns, id2int


def to_coco_gt(anns, id2int) -> dict:
    return {
        "images": [{"id": v, "width": 1280, "height": 720} for v in id2int.values()],
        "annotations": anns,
        "categories": [{"id": c, "name": ROAD8_CN[c - 1]} for c in range(1, 9)],
    }


def to_coco_dt(dets: list[dict]) -> list[dict]:
    return [{"image_id": d["image_id"], "category_id": d["category_id"],
             "bbox": d["bbox"], "score": d["score"]} for d in dets]


def build_coco(coco_gt):
    from pycocotools.coco import COCO
    c = COCO()
    c.dataset = coco_gt
    c.createIndex()
    return c


def run_cocoeval(c, dets):
    """Return overall stats + per-category AP/AR (useCats=1)."""
    from pycocotools.cocoeval import COCOeval
    if not dets:
        return dict(AP=np.nan, AP50=np.nan, AP75=np.nan, AR100=np.nan, detections=0), {}
    e = COCOeval(c, c.loadRes(dets), "bbox")
    e.params.maxDets = [1, 10, 100]
    e.evaluate()
    e.accumulate()
    e.summarize()
    # per-category: precision[TxRxKxAxM], recall[TxKxAxM]; M index 2 == maxDets 100
    per_cat = {}
    cats = list(e.params.catIds)
    for ki, cat in enumerate(cats):
        p = e.eval["precision"][:, :, ki, 0, 2]
        p = p[p > -1]
        r = e.eval["recall"][:, ki, 0, 2]
        r = r[r > -1]
        per_cat[int(cat)] = {
            "AP": float(p.mean()) if p.size else np.nan,
            "AP50": float(np.mean(e.eval["precision"][0, :, ki, 0, 2][e.eval["precision"][0, :, ki, 0, 2] > -1])) if (e.eval["precision"][0, :, ki, 0, 2] > -1).any() else np.nan,
            "AP75": float(np.mean(e.eval["precision"][5, :, ki, 0, 2][e.eval["precision"][5, :, ki, 0, 2] > -1])) if (e.eval["precision"][5, :, ki, 0, 2] > -1).any() else np.nan,
            "AR100": float(r.mean()) if r.size else np.nan,
        }
    return dict(AP=float(e.stats[0]), AP50=float(e.stats[1]), AP75=float(e.stats[2]),
                AR100=float(e.stats[8]), detections=len(dets)), per_cat


def select_and_eval(cand_frames, allocations, id2int, coco, *, nms_iou: float | None = None):
    """Return (summary_with_metrics, per_class_AP dict) for one allocation condition."""
    dets = []
    for image_id, k in zip(allocations["image_id"].astype(str), allocations["K_i"].astype(int)):
        f = cand_frames[image_id]
        f = f[f["rank"].astype(int) <= k]
        if f.empty:
            continue
        boxes = f[["box_x1", "box_y1", "box_x2", "box_y2"]].to_numpy(np.float64)
        scores = f["original_score"].to_numpy(np.float64)
        classes = f["predicted_road8_class_id"].to_numpy(np.int64)
        if nms_iou is not None:
            keep = _nms(boxes, scores, classes, nms_iou)
            boxes, scores, classes = boxes[keep], scores[keep], classes[keep]
        iid = id2int[image_id]
        for (x1, y1, x2, y2), s, cc in zip(boxes, scores, classes):
            dets.append({"image_id": iid, "category_id": int(cc),
                         "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                         "score": float(s)})
    return _finish(coco, dets)


def _finish(coco, dets):
    return run_cocoeval(coco, to_coco_dt(dets))
    return run_cocoeval(coco, to_coco_dt(dets))


def _nms(boxes, scores, classes, iou_thr):
    """Per-image class-aware NMS — canonical implementation in m11_yolo.nms."""
    from m11_yolo.nms import class_aware_nms
    return class_aware_nms(boxes, scores, classes, iou_thr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev-candidates", required=True)
    ap.add_argument("--dev-gt", required=True)
    ap.add_argument("--allocations", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--nms-iou", type=float, default=0.70, help="IoU for the reference NMS variant")
    ap.add_argument("--budgets", default="10,15,20,30,40")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    budgets = [int(x) for x in args.budgets.split(",")]

    cand = pd.read_parquet(args.dev_candidates)
    cand["image_id"] = cand["image_id"].astype(str)
    frames = {iid: g for iid, g in cand.groupby("image_id", sort=False)}

    gt_map = load_normalized_gt(args.dev_gt)
    anns, id2int = build_gt(gt_map)
    coco_gt = to_coco_gt(anns, id2int)
    coco = build_coco(coco_gt)
    print(f"GT: images={len(id2int)} boxes={len(anns)}", flush=True)

    alloc = pd.read_parquet(args.allocations)
    alloc["image_id"] = alloc["image_id"].astype(str)

    rows, pc_rows = [], []
    for (method, seed, budget), sub in alloc.groupby(["method", "seed", "budget"], sort=True):
        if int(budget) not in budgets:
            continue
        sub = sub.sort_values("image_id", kind="mergesort").reset_index(drop=True)
        m, per_cat = select_and_eval(frames, sub, id2int, coco, nms_iou=None)
        rows.append({"action_space": "frozen_prefix_no_nms", "method": method, "seed": int(seed),
                     "budget": int(budget), "mean_K": float(sub["K_i"].mean()), **m})
        for cat, v in per_cat.items():
            pc_rows.append({"action_space": "frozen_prefix_no_nms", "method": method, "seed": int(seed),
                            "budget": int(budget), "category_id": cat, "class_name": ROAD8_CN[cat - 1], **v})
        print(f"  [prefix ] {method} seed={seed} b={budget}: AP={m['AP']:.4f} AP50={m['AP50']:.4f} "
              f"AP75={m['AP75']:.4f} AR100={m['AR100']:.4f}", flush=True)
    primary = pd.DataFrame(rows)
    primary.to_csv(out / "ap_ar_prefix.csv", index=False)
    pd.DataFrame(pc_rows).to_csv(out / "ap_ar_per_class.csv", index=False)

    # --- reference: same selection + NMS (SEPARATE action space) ---
    rows_nms = []
    for budget in budgets:
        for method, seed in (("M11", 830101), ("S_ADAPT", -1), ("S_FIXED", -1)):
            sub = alloc[(alloc.method == method) & (alloc.seed == seed) & (alloc.budget == budget)]
            if sub.empty:
                continue
            sub = sub.sort_values("image_id", kind="mergesort").reset_index(drop=True)
            m, _ = select_and_eval(frames, sub, id2int, coco, nms_iou=args.nms_iou)
            rows_nms.append({"action_space": f"frozen_prefix_plus_nms{args.nms_iou}", "method": method,
                             "seed": int(seed), "budget": int(budget), "mean_K": float(sub["K_i"].mean()), **m})
            print(f"  [nms    ] {method} seed={seed} b={budget}: AP={m['AP']:.4f} AR100={m['AR100']:.4f}", flush=True)
    pd.DataFrame(rows_nms).to_csv(out / "ap_ar_reference_nms.csv", index=False)

    print(json.dumps({"status": "PASS", "primary_rows": len(primary)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
