# -*- coding: utf-8 -*-
"""YOLO native -> 90-dim feature builder (the YOLO analog of features.r18).

The 90-dim schema is reused structurally (the shared candidate/prefix/image
blocks are detector-independent).  The only YOLO-specific part is the native
gather: class_signal is 8 Road8 class scores, native_vector is the anchor's
84-dim raw Detect-head row.  ``build_raw_features`` / ``transform_features`` /
``fit_pca`` / ``fit_scaler`` from the shared package are reused unchanged.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from lc_alloc.data.schema import NativeTable, gather_native, normalize_candidate_columns, validate_native
from lc_alloc.errors import LCAllocError
from lc_alloc.features._executed_p1_core import build_raw_features
from lc_alloc.features.r18 import transform_features  # schema-agnostic scaler apply

from .config import CLASS_SIGNAL_DIM, NATIVE_VECTOR_DIM, YOLO_NATIVE_SCHEMA_ID


def _to_executed_schema(candidates: pd.DataFrame) -> pd.DataFrame:
    c = normalize_candidate_columns(candidates).sort_values("rank", kind="mergesort").copy()
    c["road8_rank"] = c["rank"].astype(np.int32)
    c["score"] = c["original_score"].astype(np.float64)
    c["bbox_x1"] = c["box_x1"].astype(np.float64)
    c["bbox_y1"] = c["box_y1"].astype(np.float64)
    c["bbox_x2"] = c["box_x2"].astype(np.float64)
    c["bbox_y2"] = c["box_y2"].astype(np.float64)
    c["bbox_w"] = c["bbox_x2"] - c["bbox_x1"]
    c["bbox_h"] = c["bbox_y2"] - c["bbox_y1"]
    c["bbox_cx"] = (c["bbox_x1"] + c["bbox_x2"]) / 2.0
    c["bbox_cy"] = (c["bbox_y1"] + c["bbox_y2"]) / 2.0
    return c


def build_image_features(candidates: pd.DataFrame, native: NativeTable, pca) -> np.ndarray:
    """Build the (45, 90) raw feature matrix for ranks 6..50 of one image."""
    validate_native(native, expected_schema_id=YOLO_NATIVE_SCHEMA_ID)
    c = _to_executed_schema(candidates)
    if c["image_id"].astype(str).nunique() != 1:
        raise LCAllocError("FEATURE_IMAGE_MIX", "build_image_features accepts exactly one image")
    class_signal, native_vector = gather_native(
        c, native, expected_vector_dim=NATIVE_VECTOR_DIM, expected_class_signal_dim=CLASS_SIGNAL_DIM
    )
    width = float(c["image_width"].iloc[0])
    height = float(c["image_height"].iloc[0])
    if width <= 0 or height <= 0:
        raise LCAllocError("INVALID_IMAGE_SIZE", f"width={width}, height={height}")
    return build_raw_features(c, class_signal, native_vector, pca, width, height)
