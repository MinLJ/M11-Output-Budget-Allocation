# -*- coding: utf-8 -*-
u"""`FasterRCNNAdapter` 的实现版本 —— 可直接 drop-in 替换交接包里那份 SCAFFOLD_ONLY。

对应合同要求:
  * 候选来自**完成检测头之后的记录**; RPN proposal 预算不是输出记录预算。
    主用流 `PRE_NMS_TOPN100` 的导出点在最终分数阈值 / 退化框过滤 / NMS / top-N **之前**;
    官方后处理之后的 `POST_NMS_TOPN300` 只作登记备查, 不参与 M11 比较 (§03);
  * 最终 record 与 RoI/native feature 的稳定对应;
  * 阈值 / NMS / top-N 顺序、score 语义、source_id 生成规则的说明。

import 一律用**绝对路径** `from lc_alloc...`, 这样这份文件既能放在覆盖层里跑,
也能原样拷进 `lc_alloc/adapters/faster_rcnn.py` 而不改一个字。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from lc_alloc.adapters.base import AdapterMetadata, DetectorAdapter
from lc_alloc.errors import LCAllocError

NATIVE_SCHEMA_ID = "frcnn_r18fpn_road8_roihead1024_logits9_v1"

# 本次适配冻结的检测器身份 —— 与 `configs/faster_rcnn_adapter.json` 必须一致
DETECTOR_ID = "FasterRCNN-resnet18-FPN-Road8"
DETECTOR_VERSION = "torchvision 0.27.0 FasterRCNN(backbone=resnet18-FPN, weights=IMAGENET1K_V1, min_size=max_size=640)"


@dataclass(frozen=True)
class ExportReport:
    u"""一次导出的自检结果。任何一项不过都直接失败, 不做静默回退。"""
    images: int
    records: int
    native_rows: int
    records_per_image_min: int
    records_per_image_max: int
    records_per_image_median: float
    # 在**传入的候选表**上数出来的。主用流已被 top-100 截断, 所以这个值恒为 0 ——
    # 它只能证明"截断后每图都够 100 条", **不能**用来论证候选池充足。
    # 真实候选池规模 (pre-NMS, = RoI 数 × 前景类数) 在导出的 export_summary.json 里。
    images_below_100: int
    join_missing: int
    join_ambiguous: int
    score_reconstruction_max_abs_error: float
    shared_source_records: int
    native_schema_id: str

    def to_dict(self) -> dict:
        return dict(self.__dict__)


class FasterRCNNAdapter(DetectorAdapter):
    u"""两阶段 anchor-based Faster R-CNN —— 与参照臂 RT-DETRv2-R18VD 结构上完全不同族。

    原生证据:
      * `native_vector` = `roi_heads.box_head` 输出 (1024 维), 每个 RoI 一行;
      * `class_signal`  = `box_predictor.cls_score` 的 **pre-softmax** logits 全部 9 列
        (0=background, 1..8=Road8)。存的是真 logits, 不是从 score 反推的 ——
        所以 score 可以**正向**重建并逐条校验。
    """

    def describe(self) -> AdapterMetadata:
        return AdapterMetadata(
            detector_family="Faster R-CNN",
            detector_version=DETECTOR_VERSION,
            checkpoint_identity="outputs/frcnn_r18fpn_road8/best.pth (sha256 见 export_config.json)",
            candidate_export_implemented=True,
            native_export_implemented=True,
            status="ADAPTED",
            notes=(
                "Primary candidate stream is PRE_NMS_TOPN100: RoI-head classification and box "
                "regression, clipping and background removal, then the original-score prefix "
                "BEFORE the final score threshold / degenerate-box filter / NMS / top-N. "
                "A second POST_NMS_TOPN300 stream (the official postprocess output) is exported "
                "for reference only and is not used by the M11 comparison. Both come from the "
                "same forward pass. RPN proposals are never used as output records.",
                "source_id = 'roi{index:05d}' is the stable row index of the RoI inside the "
                "box_head input batch; one RoI may carry several class records sharing it.",
                "Join is by (image_id, source_id) only; rank, array row order and reshape position "
                "are never used for joining.",
                "class_signal stores true pre-softmax RoI-head class logits, so the detector score "
                "is reconstructed forward as softmax(logits)[class] and checked per record.",
                "native_schema_id is Faster R-CNN specific; RT-DETR R18 PCA/scaler/model assets "
                "must not be cross-loaded.",
            ),
        )

    # ── 导出 ────────────────────────────────────────────────────────────
    def export(self, candidates, native, *, expected_native_schema_id: str | None = NATIVE_SCHEMA_ID,
               check_score_reconstruction: bool = True) -> ExportReport:
        u"""读取一份**冻结**的 Faster R-CNN 导出并做全量自检。

        参数
        ----
        candidates : DataFrame 或路径 (parquet/csv)
            M11 candidate 表, 必须含 image_id / rank / original_score /
            predicted_road8_class_id / source_id / box_*.
        native : NativeTable 或路径 (npz)
            native sidecar, 列 image_ids/source_ids/native_vectors/class_signals/native_schema_id.

        这里**不跑检测器**: 检测器只在 `export_frcnn.py` 里跑一次并冻结。
        适配器的职责是把冻结结果接进 M11, 并证明这份接法自洽。
        """
        import pandas as pd

        from lc_alloc.data.io import read_native
        from lc_alloc.data.schema import normalize_candidate_columns, validate_candidates

        if isinstance(candidates, (str, Path)):
            path = Path(candidates)
            frame = pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)
        else:
            frame = candidates
        frame = validate_candidates(normalize_candidate_columns(frame))

        if isinstance(native, (str, Path)):
            native = read_native(native, expected_schema_id=expected_native_schema_id)
        elif expected_native_schema_id is not None and native.native_schema_id != expected_native_schema_id:
            raise LCAllocError("NATIVE_SCHEMA_MISMATCH",
                               f"{native.native_schema_id!r} != {expected_native_schema_id!r}")

        image_ids = np.asarray(native.image_ids).astype(str)
        source_ids = np.asarray(native.source_ids).astype(str)
        vectors = np.asarray(native.native_vectors)
        signals = np.asarray(native.class_signals)
        if vectors.ndim != 2 or signals.ndim != 2:
            raise LCAllocError("NATIVE_DIMENSION_MISMATCH", f"vector={vectors.shape} signal={signals.shape}")

        lookup: dict[tuple[str, str], int] = {}
        ambiguous = 0
        for i, key in enumerate(zip(image_ids.tolist(), source_ids.tolist())):
            if key in lookup:
                ambiguous += 1
            else:
                lookup[key] = i
        if ambiguous:
            raise LCAllocError("NATIVE_JOIN_AMBIGUOUS", f"{ambiguous} duplicated (image_id, source_id) keys")

        rows = []
        missing = 0
        for row in frame.itertuples(index=False):
            key = (str(row.image_id), str(row.source_id))
            if key not in lookup:
                missing += 1
                continue
            rows.append((lookup[key], int(row.predicted_road8_class_id), float(row.original_score)))
        if missing:
            raise LCAllocError("NATIVE_JOIN_MISSING", f"{missing} candidate records have no native row")

        idx = np.asarray([r[0] for r in rows], dtype=np.int64)
        cls = np.asarray([r[1] for r in rows], dtype=np.int64)
        score = np.asarray([r[2] for r in rows], dtype=np.float64)

        max_err = float("nan")
        if check_score_reconstruction:
            # class_signal 是**完整**的 pre-softmax class logits (0=background, 1..8=Road8),
            # 所以 score = softmax(logits)[class] 可以正向重建并逐条校验。
            # 这条检查只有在存了真 logits 时才做得成 —— 拿 score 的逆 sigmoid 冒充 logits
            # 会在这一步直接暴露。
            logits = np.asarray(signals, dtype=np.float64)[idx]
            if logits.shape[1] != 9:
                raise LCAllocError("NATIVE_SCHEMA_MISMATCH",
                                   f"score reconstruction expects 9 pre-softmax columns, got {logits.shape[1]}")
            shifted = logits - logits.max(axis=1, keepdims=True)
            prob = np.exp(shifted)
            prob /= prob.sum(axis=1, keepdims=True)
            max_err = float(np.max(np.abs(prob[np.arange(len(cls)), cls] - score)))

        counts = frame.groupby("image_id", sort=False).size().to_numpy()
        pairs = frame.groupby(["image_id", "source_id"], sort=False).size().to_numpy()
        return ExportReport(
            images=int(frame["image_id"].nunique()),
            records=int(len(frame)),
            native_rows=int(len(image_ids)),
            records_per_image_min=int(counts.min()) if len(counts) else 0,
            records_per_image_max=int(counts.max()) if len(counts) else 0,
            records_per_image_median=float(np.median(counts)) if len(counts) else 0.0,
            images_below_100=int((counts < 100).sum()),
            join_missing=int(missing),
            join_ambiguous=int(ambiguous),
            score_reconstruction_max_abs_error=max_err,
            shared_source_records=int((pairs > 1).sum()),
            native_schema_id=str(native.native_schema_id),
        )
