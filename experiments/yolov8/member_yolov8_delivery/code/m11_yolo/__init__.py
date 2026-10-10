"""YOLO adapter + feature builder for the M11 allocation method (member 2 task).

Follows the LC_ALLOC_M11_HANDOFF_v1 adapter contract.  Freezes YOLOv8n
(ultralytics 8.4.77) as a second detector and reuses the shared M11 modules
(max-cardinality prefix labels, exact float64 DP, training/temperature
calibration, evaluation) from ``lc_alloc``.
"""
from .adapter import YOLOAdapter
from .config import (
    CLASS_SIGNAL_DIM,
    NATIVE_VECTOR_DIM,
    SEEDS,
    YOLO_NATIVE_SCHEMA_ID,
)

__all__ = [
    "YOLOAdapter",
    "YOLO_NATIVE_SCHEMA_ID",
    "NATIVE_VECTOR_DIM",
    "CLASS_SIGNAL_DIM",
    "SEEDS",
]
