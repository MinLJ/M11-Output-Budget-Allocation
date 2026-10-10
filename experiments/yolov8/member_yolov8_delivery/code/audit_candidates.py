# -*- coding: utf-8 -*-
"""Candidate availability audit for the YOLO M11 cross-detector experiment."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent

for split in ("DEV", "TRAIN"):
    cand = pd.read_parquet(BASE / "work" / split / "candidates.parquet")
    cand["image_id"] = cand["image_id"].astype(str)
    n_imgs = cand["image_id"].nunique()
    per_img = cand.groupby("image_id", sort=False).size()
    n_records = len(cand)
    topk = int(cand["rank"].max())
    n_lt = int((per_img < topk).sum())
    n_gt = int((per_img > topk).sum())
    print(f"===== {split} =====")
    print(f"images: {n_imgs}")
    print(f"candidate_records: {n_records}")
    print(f"top_k (max rank): {topk}")
    print(f"per-image candidates: mean={per_img.mean():.3f}  min={per_img.min()}  max={per_img.max()}  median={per_img.median():.1f}")
    print(f"images with < {topk} candidates: {n_lt}  ({n_lt / n_imgs * 100:.3f}%)")
    print(f"images with > {topk} candidates: {n_gt}  ({n_gt / n_imgs * 100:.3f}%)")
    if n_lt:
        print("  min-count tail (count -> #images):")
        print("  " + str(per_img[per_img < topk].value_counts().sort_index().to_dict()))
    # native row availability (unique anchors per image)
    if (BASE / "work" / split / "native.npz").exists():
        nat = np.load(BASE / "work" / split / "native.npz", allow_pickle=True)
        nids = pd.Series(nat["image_ids"].astype(str))
        per_img_nat = nids.groupby(nids).size()
        print(f"native rows: {len(nids)}  per-image: mean={per_img_nat.mean():.3f} min={per_img_nat.min()} max={per_img_nat.max()}")
    # export log runtime
    log = json.loads((BASE / "work" / split / "export_log.json").read_text(encoding="utf-8"))
    print(f"export wall-clock: {log.get('seconds')}s  -> {log.get('seconds', 0) / n_imgs * 1000:.1f} ms/image (batch incl. IO+decode)")
    print()

# DEV candidate <100 count by rank if any image has fewer
cand = pd.read_parquet(BASE / "work" / "DEV" / "candidates.parquet")
cand["image_id"] = cand["image_id"].astype(str)
per_img = cand.groupby("image_id", sort=False).size()
print("DEV per-image candidate count distribution (histogram):")
bins = [0, 50, 90, 95, 99, 100]
labels = ["1-50", "51-90", "91-95", "96-99", "=100"]
hist = pd.cut(per_img, bins=bins, right=True, include_lowest=True, labels=labels)
print(hist.value_counts().sort_index().to_string())
