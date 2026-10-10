# -*- coding: utf-8 -*-
"""YOLOAdapter — frozen YOLOv8n candidate/native export for the M11 method.

Implements the LC_ALLOC_M11_HANDOFF_v1 DetectorAdapter contract for YOLO:

  * candidate stream: pre-NMS dense anchor hypotheses, top-100 by original
    per-class sigmoid score (positive-area box filter, no conf threshold, no
    NMS, no objectness);
  * source_id = anchor index 0..8399 (8 Road8 class candidates share it);
  * native sidecar = the anchor's 84-dim raw Detect-head row + 8 Road8 scores,
    joined only by (image_id, source_id).

Writes ``candidates.parquet`` + ``native.npz`` + ``export_log.json``.  Never
touches the immutable RT-DETR release directory.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from PIL import Image
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from ultralytics.utils.ops import scale_boxes, xywh2xyxy

from lc_alloc.adapters.base import AdapterMetadata, DetectorAdapter
from lc_alloc.data.identity import candidate_record_id

from .config import (
    CANDIDATE_ASSET_ID,
    CHECKPOINT_PATH,
    CHECKPOINT_SHA256,
    CLASS_SIGNAL_DIM,
    DETECTOR_FAMILY,
    DETECTOR_VERSION,
    EXPORT_POINT,
    IMGSZ,
    N_ANCHORS,
    NATIVE_VECTOR_DIM,
    RECORD_TO_NATIVE,
    ROAD8_COCO_IDX,
    ROAD8_ID,
    SCORE_SEMANTICS,
    STRIDE,
    TOP_K,
    YOLO_NATIVE_SCHEMA_ID,
)


class YOLOAdapter(DetectorAdapter):
    def describe(self) -> AdapterMetadata:
        return AdapterMetadata(
            detector_family=DETECTOR_FAMILY,
            detector_version=DETECTOR_VERSION,
            checkpoint_identity=f"yolov8n.pt sha256:{CHECKPOINT_SHA256}",
            candidate_export_implemented=True,
            native_export_implemented=True,
            status="IMPLEMENTED",
            notes=(
                f"preprocess: RGB -> LetterBox({IMGSZ}, auto=False, stride={STRIDE}) -> /255.0 float32 -> CUDA",
                f"candidate export point: {EXPORT_POINT}",
                f"score semantics: {SCORE_SEMANTICS}",
                f"source_id: anchor index 0..{N_ANCHORS - 1}; {RECORD_TO_NATIVE}",
                f"native layer: {DETECTOR_VERSION} Detect head (84, 8400); vector dim {NATIVE_VECTOR_DIM} float32, signal dim {CLASS_SIGNAL_DIM} float32",
            ),
        )

    def export(
        self,
        manifest_path: str | Path,
        image_dir: str | Path,
        output_dir: str | Path,
        *,
        split: str = "DEV",
        max_images: int = 0,
        batch: int = 32,
        device: str = "cuda",
    ) -> dict:
        """Run frozen YOLOv8n inference and write candidate/native sidecars."""
        manifest_path = Path(manifest_path)
        image_dir = Path(image_dir)
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        manifest = pd.read_parquet(manifest_path) if manifest_path.suffix == ".parquet" else pd.read_csv(manifest_path)
        image_ids = manifest["image_id"].astype(str).to_numpy()
        file_names = manifest["file_name"].astype(str).to_numpy()
        widths = manifest["width"].to_numpy(dtype=np.float64)
        heights = manifest["height"].to_numpy(dtype=np.float64)
        n = len(image_ids)
        if max_images:
            n = min(n, int(max_images))
        image_ids = image_ids[:n]
        file_names = file_names[:n]
        widths = widths[:n]
        heights = heights[:n]

        model = YOLO(str(Path(CHECKPOINT_PATH)))
        mm = model.model.to(torch.device(device)).eval()
        lb = LetterBox(new_shape=IMGSZ, auto=False, stride=STRIDE)

        cand_rows: list[dict] = []
        native_image_ids: list[str] = []
        native_source_ids: list[str] = []
        native_vectors: list[np.ndarray] = []
        class_signals: list[np.ndarray] = []

        t0 = time.time()
        for b0 in range(0, n, batch):
            b1 = min(b0 + batch, n)
            ims, shapes = [], []
            for i in range(b0, b1):
                arr = np.asarray(Image.open(image_dir / file_names[i]).convert("RGB"))
                shapes.append((arr.shape[0], arr.shape[1]))  # H, W
                ims.append(lb(image=arr).transpose(2, 0, 1))
            t = torch.from_numpy(np.stack(ims)).float().to(device) / 255.0
            with torch.no_grad():
                raw = mm(t)
            p = raw[0].cpu().numpy()  # (B, 84, 8400)
            for j in range(b1 - b0):
                i = b0 + j
                image_id = str(image_ids[i])
                boxes, cls_road8 = _decode(p[j], shapes[j])
                sel = _topk(boxes, cls_road8, TOP_K)
                anchors = np.repeat(np.arange(N_ANCHORS), 8)[sel]
                classes = np.tile(np.asarray(ROAD8_ID, dtype=np.int64), N_ANCHORS)[sel]
                scores = cls_road8.T.reshape(-1)[sel]
                unique_anchors = np.unique(anchors)
                for rank, (anchor, cls, score) in enumerate(zip(anchors, classes, scores), start=1):
                    cand_rows.append({
                        "image_id": image_id,
                        "candidate_record_id": candidate_record_id(CANDIDATE_ASSET_ID, image_id, rank),
                        "rank": rank,
                        "predicted_road8_class_id": int(cls),
                        "original_score": float(score),
                        "box_x1": float(boxes[anchor, 0]),
                        "box_y1": float(boxes[anchor, 1]),
                        "box_x2": float(boxes[anchor, 2]),
                        "box_y2": float(boxes[anchor, 3]),
                        "source_id": str(anchor),
                        "image_width": float(widths[i]),
                        "image_height": float(heights[i]),
                    })
                for anchor in unique_anchors:
                    native_image_ids.append(image_id)
                    native_source_ids.append(str(anchor))
                    native_vectors.append(p[j, :, anchor].astype(np.float32))
                    class_signals.append(cls_road8[:, anchor].astype(np.float32))
            done = b1
            if done % (10 * batch) < batch or done == n:
                print(f"  [{split}] {done}/{n} images ({time.time() - t0:.1f}s)", flush=True)

        candidates = pd.DataFrame(cand_rows)
        candidates.to_parquet(out / "candidates.parquet", index=False)

        np.savez(
            out / "native.npz",
            image_ids=np.asarray(native_image_ids),
            source_ids=np.asarray(native_source_ids),
            native_vectors=np.stack(native_vectors),
            class_signals=np.stack(class_signals),
            native_schema_id=np.asarray(YOLO_NATIVE_SCHEMA_ID),
        )

        log = {
            "detector_family": DETECTOR_FAMILY,
            "detector_version": DETECTOR_VERSION,
            "checkpoint_path": CHECKPOINT_PATH,
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "native_schema_id": YOLO_NATIVE_SCHEMA_ID,
            "candidate_asset_id": CANDIDATE_ASSET_ID,
            "split": split,
            "images": int(n),
            "candidate_records": int(len(candidates)),
            "native_rows": int(len(native_image_ids)),
            "top_k": TOP_K,
            "export_point": EXPORT_POINT,
            "score_semantics": SCORE_SEMANTICS,
            "seconds": round(time.time() - t0, 1),
        }
        (out / "export_log.json").write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(log, ensure_ascii=False), flush=True)
        return log


def _decode(p: np.ndarray, orig_hw: tuple[int, int]):
    """p: (84, 8400). Returns (boxes (A,4) orig xyxy, cls_road8 (8,A))."""
    box = p[:4]
    cls_road8 = p[4:][ROAD8_COCO_IDX]
    with torch.no_grad():
        b_xyxy = xywh2xyxy(torch.from_numpy(box.T).float().cuda())
        b_orig = scale_boxes(IMGSZ, b_xyxy, orig_hw).cpu().numpy()
    return b_orig, cls_road8


def _topk(boxes: np.ndarray, cls_road8: np.ndarray, k: int) -> np.ndarray:
    """Indices (into flattened (anchor, class) hypotheses) of the top-k valid ones."""
    a = cls_road8.shape[1]
    anchor_valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    scores = cls_road8.T.reshape(-1)
    anchor_idx = np.repeat(np.arange(a), 8)
    cls_id = np.tile(np.asarray(ROAD8_ID, dtype=np.int64), a)
    valid = np.flatnonzero(np.repeat(anchor_valid, 8))
    sc = scores[valid]
    m = min(k, len(sc))
    part = np.argpartition(-sc, m - 1)[:m]
    sel = valid[part]
    order = np.lexsort((cls_id[sel], anchor_idx[sel], -scores[sel]))
    return sel[order]
