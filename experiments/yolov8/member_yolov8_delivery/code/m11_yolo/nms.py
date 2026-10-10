# -*- coding: utf-8 -*-
"""Canonical NMS used by the YOLO branch (shared by 05_ap_ar.py and 06_post_nms_counts.py).

Frozen semantics (recorded here because the guideline requires the actual
implementation semantics be documented, not inferred):

  * class-aware  : one NMS pass per predicted Road8 class, then concatenate.
  * library      : torchvision.ops.nms.
  * IoU          : caller-supplied; the YOLO reference uses 0.70.
  * tie-breaking : torchvision.ops.nms keeps the higher score; for exactly
                   equal scores the lower input index wins (its internal sort
                   is on descending score).
  * input order  : the caller passes candidates sorted by rank ascending
                   (rank 1..K_i), so the surviving index set is a subset of the
                   original prefix slot indices.
  * no refill    : removed slots are NOT backfilled; n_after <= n_before always.
"""
from __future__ import annotations

import numpy as np


def class_aware_nms(boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray, iou_thr: float) -> np.ndarray:
    """Return the indices (into the input arrays) surviving class-aware NMS.

    Inputs must be rank-ascending (slot order).  Index order in the output is
    ascending by class group, so callers that need slot order should sort.
    """
    import torch
    from torchvision.ops import nms

    if len(boxes) == 0:
        return np.array([], dtype=np.int64)
    keep_all = []
    t_boxes = torch.from_numpy(np.asarray(boxes, dtype=np.float64)).float()
    t_scores = torch.from_numpy(np.asarray(scores, dtype=np.float64)).float()
    for c in np.unique(classes):
        idx = np.flatnonzero(classes == c)
        keep = nms(t_boxes[idx], t_scores[idx], iou_thr).numpy()
        if len(keep):
            keep_all.append(idx[keep])
    if not keep_all:
        return np.array([], dtype=np.int64)
    return np.sort(np.concatenate(keep_all)).astype(np.int64)
