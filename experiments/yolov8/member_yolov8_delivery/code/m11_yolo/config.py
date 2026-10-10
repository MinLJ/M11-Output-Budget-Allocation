# -*- coding: utf-8 -*-
"""Frozen YOLO adapter identity for the M11 cross-detector validation.

Every value here is a declared, reproducible detector/asset identity.  The
candidate/native export and the downstream feature builder read from these
constants only; nothing is inferred from the runtime environment.
"""
from __future__ import annotations
import os
from pathlib import Path

# --- detector freeze ---------------------------------------------------------
DETECTOR_FAMILY = "YOLO"
DETECTOR_VERSION = "YOLOv8n (ultralytics 8.4.77)"
CHECKPOINT_PATH = str(Path(os.environ.get("M11_EXPERIMENT_ROOT", "external_assets")) / "CW2" / "weights" / "yolov8n.pt")
CHECKPOINT_SHA256 = "f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36"

# --- preprocessing freeze ----------------------------------------------------
# RGB -> LetterBox(640x640, auto=False, stride=32) -> /255.0 float32 -> CUDA.
# No ImageNet mean/std subtraction; no augmentation; no TTA.
IMGSZ = (640, 640)
STRIDE = 32
NORMALIZE = "divide_by_255"
PRECISION = "float32"

# --- Road8 mapping (frozen, identical to RT-DETRv2 reference) ---------------
# COCO 0-indexed class indices -> Road8 id 1..8.
ROAD8_COCO_IDX = [0, 1, 2, 3, 5, 6, 7, 9]
ROAD8_ID = [1, 2, 3, 4, 5, 6, 7, 8]
N_CLASSES = 80  # full COCO class head width

# --- candidate export point --------------------------------------------------
# PRE-NMS dense anchor stream: 8400 anchors x 8 Road8 classes = 67200 hypoth.
# Positive-area box filter (drop x2<=x1 or y2<=y1), no conf threshold, no NMS,
# no objectness, top-100 by (score desc, anchor asc, Road8 class id asc).
N_ANCHORS = 8400
TOP_K = 100
EXPORT_POINT = "pre-nms dense anchor stream (8400x8), positive-area filter, top-100"

# --- score semantics ----------------------------------------------------------
# Candidate score = post-sigmoid per-class score of the anchor's Road8 class.
# No objectness multiplication (YOLOv8 has no separate objectness branch).
SCORE_SEMANTICS = "post-sigmoid per-class score; no objectness; no NMS"

# --- source_id / native association ------------------------------------------
# source_id = anchor index 0..8399 (row index in the concatenated 8400 grid).
# 8 Road8 class candidates from one anchor share one source_id.
# native layer = Detect head final concatenated output (84, 8400) =
#   [4 box (xywh letterboxed px) | 80 post-sigmoid class scores].
# native_vector = that anchor's 84-dim raw head row (float32).
# class_signal   = that anchor's 8 Road8 class scores (float32).
NATIVE_LAYER = "yolov8n Detect head final concat output (84, 8400)"
YOLO_NATIVE_SCHEMA_ID = "yolov8n_detect_head_raw84_road8_scores_v1"
NATIVE_VECTOR_DIM = 84
CLASS_SIGNAL_DIM = 8
RECORD_TO_NATIVE = "one-to-8: each native row (anchor) <-> up to 8 Road8 class records"

# --- downstream asset identity ----------------------------------------------
# The 90-dim input schema is reused structurally (58 candidate + 12 prefix +
# 20 image context).  The 8 "native logit" slots hold YOLO Road8 class scores,
# the 32 "embedding pca" slots hold PCA32 of the 84-dim native_vector, and the
# "embedding norm" slot holds its L2 norm.  PCA/scaler/MLP/temperature are
# re-trained on YOLO data -> a NEW asset identity, not the RT-DETR asset.
FEATURE_DIM = 90
PCA_COMPONENTS = 32
NETWORK = "Linear(90,128)-LayerNorm-GELU-Linear(128,64)-GELU-Linear(64,10)"
SEEDS = (830101, 830102, 830103)

# --- record identity ---------------------------------------------------------
CANDIDATE_ASSET_ID = "yolov8n_bdd100k_road8_top100_v1"
