# -*- coding: utf-8 -*-
u"""Faster R-CNN 臂的 native → feature builder。

合同(`docs/ADAPTER_CONTRACT.md`)要求: 新检测器可以复用标签/训练/校准/分配/评价模块,
**但必须先实现并登记自己的 native→feature builder、最终输入维度及网络配置**。
包内现成的 `lc_alloc.features.r18` 是 RT-DETR R18 那条配方的参考实现, 不是万能特征器,
所以这里独立实现一份。

【为什么维度还是 90】
参照臂的 90 维 = 58 (当前候选) + 12 (k-1 前缀) + 20 (Top100 图级上下文)。
Faster R-CNN 臂**刻意沿用同一布局与同一维度**, 目的是把分配器(网络结构/损失/训练配方)
整个固定住 —— 于是两条臂之间唯一的变量就是**检测器本身**。
这是设计选择, 不是复用旧配方: 输入的原生证据完全不同
(1024 维 RoI head 特征 vs 256 维 L3 query embedding;
 9 维 softmax head 的 pre-softmax logits vs 80 类 sigmoid logits),
所以 PCA/scaler/模型/温度全部需要为本检测器单独拟合并登记新资产身份。
"""
from __future__ import annotations

import hashlib
import math

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from lc_alloc.constants import FEATURE_TOP_N, ROAD8
from lc_alloc.data.schema import NativeTable, gather_native, normalize_candidate_columns, validate_native
from lc_alloc.errors import LCAllocError
from lc_alloc.features._executed_p1_core import iou_xyxy

# ── 本检测器自己的原生契约 ────────────────────────────────────────────────
NATIVE_SCHEMA_ID = "frcnn_r18fpn_road8_roihead1024_logits9_v1"
NATIVE_VECTOR_DIM = 1024        # roi_heads.box_head 输出
CLASS_SIGNAL_DIM = 9            # 完整 pre-softmax class logits: 0=background, 1..8=Road8
ROAD8_SIGNAL_SLICE = slice(1, 9)
FEATURE_DIM = 90
PCA_COMPONENTS = 32

_EXPECTED_NAMES: list[str] | None = None


def feature_spec() -> list[dict]:
    u"""Faster R-CNN 臂的冻结 90 列 schema, 顺序即模型输入顺序。

    列名与分块和参照臂对齐, 但 `native_*` 与 `embedding_pca_*` 的来源是 **RoI head**,
    不是 decoder query。名称里带 `native_` 而不是 `l3_`, 就是为了不让人误以为可以
    拿 RT-DETR 的资产来套。
    """
    rows: list[dict] = []

    def add(name: str, block: str, formula: str, standardize: bool = True, unit: str = "unitless") -> None:
        rows.append({
            "index": len(rows), "name": name, "block": block, "formula": formula,
            "source": "Faster R-CNN export (candidate table + RoI-head native sidecar)",
            "unit": unit, "missing_rule": "none; explicit zero only where formula states it",
            "standardize_fit_only": bool(standardize),
        })

    add("raw_score", "candidate", "detector softmax score of the predicted class at this RoI")
    add("score_logit", "candidate", "logit(clip(raw_score,1e-6,1-1e-6))")
    add("rank_normalized", "candidate", "rank / 100")
    for c in ROAD8:
        add(f"class_onehot_{c.replace(' ', '_')}", "candidate", f"1[predicted class is {c}]", False)
    for n, f in (
        ("bbox_cx_norm", "bbox_cx / image_width"), ("bbox_cy_norm", "bbox_cy / image_height"),
        ("bbox_w_norm", "bbox_w / image_width"), ("bbox_h_norm", "bbox_h / image_height"),
        ("bbox_area_norm", "bbox_w*bbox_h/(image_width*image_height)"),
        ("bbox_log_aspect", "log(max(bbox_w/image_width,1e-6)/max(bbox_h/image_height,1e-6))"),
    ):
        add(n, "candidate", f)
    for c in ROAD8:
        add(f"native_road8_logit_{c.replace(' ', '_')}", "candidate",
            "RoI-head pre-softmax Road8 logit gathered by (image_id, source_id)")
    for j in range(PCA_COMPONENTS):
        add(f"embedding_pca_{j:02d}", "candidate",
            "FIT-only PCA32 transform of stored float16 box_head feature cast to float32")
    add("embedding_norm", "candidate", "L2 norm of stored float16 box_head feature after float32 cast")

    for n, f, std in (
        ("prefix_length", "k-1", True),
        ("prefix_score_mean", "mean score ranks 1..k-1", True),
        ("prefix_score_max", "max score ranks 1..k-1", True),
        ("prefix_score_min", "min score ranks 1..k-1", True),
        ("same_class_prefix_count", "count prefix candidates with current predicted class", True),
        ("same_class_prefix_fraction", "same_class_prefix_count/(k-1)", True),
        ("same_class_iou_max", "max IoU(current,prefix same class); 0 if empty", True),
        ("same_class_iou_mean", "mean IoU(current,prefix same class); 0 if empty", True),
        ("same_class_iou_count_ge_030", "count same-class IoU >= .30", True),
        ("same_class_iou_count_ge_050", "count same-class IoU >= .50", True),
        ("same_class_center_distance_min", "min Euclidean distance in normalized xy; 0 if empty", True),
        ("same_class_prefix_exists", "1 iff same-class prefix exists", False),
    ):
        add(n, "prefix", f, std)

    for n, f in (
        ("image_score_mean", "Top100 mean score"), ("image_score_std", "Top100 population std ddof=0"),
        ("image_score_p25", "Top100 25th percentile, linear"), ("image_score_p50", "Top100 median, linear"),
        ("image_score_p75", "Top100 75th percentile, linear"), ("image_top5_score_mean", "Top5 mean score"),
        ("image_top20_score_mean", "Top20 mean score"), ("image_top50_score_mean", "Top50 mean score"),
        ("image_score_fraction_gt_010", "Top100 fraction score > .10"),
        ("image_score_fraction_gt_025", "Top100 fraction score > .25"),
        ("image_score_fraction_gt_050", "Top100 fraction score > .50"),
    ):
        add(n, "image", f)
    for c in ROAD8:
        add(f"image_class_fraction_{c.replace(' ', '_')}", "image", f"Top100 fraction predicted class {c}")
    add("image_bbox_area_median", "image", "Top100 median normalized bbox area, linear")
    if len(rows) != FEATURE_DIM:
        raise AssertionError(f"feature schema length {len(rows)} != {FEATURE_DIM}")
    return rows


def _expected_names() -> list[str]:
    global _EXPECTED_NAMES
    if _EXPECTED_NAMES is None:
        _EXPECTED_NAMES = [entry["name"] for entry in feature_spec()]
    return _EXPECTED_NAMES


def _to_executed_schema(candidates: pd.DataFrame) -> pd.DataFrame:
    c = normalize_candidate_columns(candidates).sort_values("rank", kind="mergesort").head(FEATURE_TOP_N).copy()
    c["road8_rank"] = c["rank"].astype(np.int32)
    c["score"] = c["original_score"].astype(np.float64)
    for target, source in (("bbox_x1", "box_x1"), ("bbox_y1", "box_y1"), ("bbox_x2", "box_x2"), ("bbox_y2", "box_y2")):
        c[target] = c[source].astype(np.float64)
    c["bbox_w"] = c["bbox_x2"] - c["bbox_x1"]
    c["bbox_h"] = c["bbox_y2"] - c["bbox_y1"]
    c["bbox_cx"] = (c["bbox_x1"] + c["bbox_x2"]) / 2.0
    c["bbox_cy"] = (c["bbox_y1"] + c["bbox_y2"]) / 2.0
    return c


def build_raw_features(candidates, native_logits, embeddings, pca, image_width, image_height) -> np.ndarray:
    u"""单图 45×90 原始特征 (rank 6..50)。公式与参照臂逐项同构。"""
    c = candidates.sort_values("road8_rank", kind="stable").reset_index(drop=True)
    if len(c) != FEATURE_TOP_N or c["road8_rank"].tolist() != list(range(1, FEATURE_TOP_N + 1)):
        raise LCAllocError("CANDIDATE_COUNT_BELOW_FEATURE_TOP_N",
                           f"FRCNN feature builder requires exact ranks 1..{FEATURE_TOP_N}")
    score = c["score"].to_numpy(np.float64)
    cls = c["predicted_road8_class_id"].to_numpy(np.int16)
    boxes = c[["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]].to_numpy(np.float64)
    cx = c["bbox_cx"].to_numpy(np.float64) / float(image_width)
    cy = c["bbox_cy"].to_numpy(np.float64) / float(image_height)
    wn = c["bbox_w"].to_numpy(np.float64) / float(image_width)
    hn = c["bbox_h"].to_numpy(np.float64) / float(image_height)
    area = wn * hn
    emb32 = np.asarray(embeddings, dtype=np.float32)
    pca32 = np.asarray(pca.transform(emb32), dtype=np.float64)
    emb_norm = np.linalg.norm(emb32.astype(np.float64), axis=1)
    native_logits = np.asarray(native_logits, dtype=np.float64)
    if native_logits.shape != (FEATURE_TOP_N, CLASS_SIGNAL_DIM):
        raise LCAllocError("NATIVE_DIMENSION_MISMATCH", f"class_signal {native_logits.shape}")
    # 特征里只用 8 个 Road8 列 —— 与参照臂 ClassSignal(8) 的 block 宽度对齐,
    # background 列(第 0 列)保留在 native 里但不进特征。
    native_logits = native_logits[:, ROAD8_SIGNAL_SLICE]
    if pca32.shape != (FEATURE_TOP_N, PCA_COMPONENTS):
        raise LCAllocError("FEATURE_SCHEMA_MISMATCH", f"pca {pca32.shape}")

    quant = np.quantile(score, [0.25, 0.50, 0.75], method="linear")
    context = np.asarray([
        score.mean(), score.std(ddof=0), quant[0], quant[1], quant[2],
        score[:5].mean(), score[:20].mean(), score[:50].mean(),
        np.mean(score > 0.10), np.mean(score > 0.25), np.mean(score > 0.50),
        *[np.mean(cls == j) for j in range(1, 9)], np.median(area),
    ], dtype=np.float64)

    rows = np.empty((45, FEATURE_DIM), dtype=np.float64)
    for oi, k in enumerate(range(6, 51)):
        j = k - 1
        s_j = np.clip(score[j], 1e-6, 1 - 1e-6)
        local = [score[j], math.log(s_j / (1 - s_j)), k / 100.0]
        local += [float(cls[j] == z) for z in range(1, 9)]
        local += [cx[j], cy[j], wn[j], hn[j], area[j], math.log(max(wn[j], 1e-6) / max(hn[j], 1e-6))]
        local += native_logits[j].tolist() + pca32[j].tolist() + [emb_norm[j]]

        prefix_n = k - 1
        same_idx = np.flatnonzero(cls[:prefix_n] == cls[j])
        if len(same_idx):
            overlaps = iou_xyxy(boxes[j:j + 1], boxes[same_idx])[0]
            distances = np.sqrt((cx[j] - cx[same_idx]) ** 2 + (cy[j] - cy[same_idx]) ** 2)
            rel = [
                prefix_n, score[:prefix_n].mean(), score[:prefix_n].max(), score[:prefix_n].min(),
                len(same_idx), len(same_idx) / prefix_n, overlaps.max(), overlaps.mean(),
                np.sum(overlaps >= 0.30), np.sum(overlaps >= 0.50), distances.min(), 1.0,
            ]
        else:
            rel = [prefix_n, score[:prefix_n].mean(), score[:prefix_n].max(), score[:prefix_n].min(),
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        rows[oi] = np.asarray(local + rel + context.tolist(), dtype=np.float64)

    if not np.all(np.isfinite(rows)):
        raise LCAllocError("NONFINITE_FEATURE", "nonfinite feature generated")
    return rows


def build_image_features(candidates: pd.DataFrame, native: NativeTable, pca) -> np.ndarray:
    validate_native(native, expected_schema_id=NATIVE_SCHEMA_ID)
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


def transform_features(raw_features: np.ndarray, scaler_bundle: dict) -> np.ndarray:
    values = np.asarray(raw_features, dtype=np.float64).copy()
    mask = np.asarray(scaler_bundle["standardize_mask"], dtype=bool)
    names = list(scaler_bundle["feature_names"])
    if values.shape[1] != FEATURE_DIM or mask.shape != (FEATURE_DIM,) or names != _expected_names():
        raise LCAllocError("FEATURE_SCHEMA_MISMATCH", "scaler and FRCNN 90D schema do not match")
    values[:, mask] = scaler_bundle["scaler"].transform(values[:, mask])
    if not np.isfinite(values).all():
        raise LCAllocError("NONFINITE_FEATURE", "scaled features contain NaN or Inf")
    return values.astype(np.float32)


def fit_pca(image_ids, source_ids, native_vectors, *, namespace: str,
            sample_cap: int = 200_000, random_state: int = 530100) -> PCA:
    u"""FIT-only PCA32。抽样顺序用 namespace 哈希固定 —— 与参照臂同一套确定性规则。"""
    keys: dict[tuple[str, str], int] = {}
    for i, (image_id, source_id) in enumerate(zip(image_ids, source_ids)):
        keys.setdefault((str(image_id), str(source_id)), i)
    ordered = sorted(
        keys,
        key=lambda key: (hashlib.sha256(f"{namespace}{key[0]}|{key[1]}".encode("utf-8")).hexdigest(), key),
    )[:sample_cap]
    matrix = np.asarray(native_vectors, dtype=np.float32)[[keys[key] for key in ordered]]
    if len(matrix) < 32:
        raise ValueError("PCA32 requires at least 32 unique FIT native rows")
    pca = PCA(n_components=PCA_COMPONENTS, svd_solver="randomized", whiten=False, random_state=random_state)
    pca.fit(matrix)
    return pca


def fit_scaler(raw_fit_features: np.ndarray) -> dict:
    spec = feature_spec()
    mask = np.asarray([bool(column["standardize_fit_only"]) for column in spec], dtype=bool)
    raw = np.asarray(raw_fit_features, dtype=np.float64)
    scaler = StandardScaler().fit(raw[:, mask])
    scaler.scale_[np.asarray(scaler.scale_) == 0] = 1.0
    return {"scaler": scaler, "standardize_mask": mask, "feature_names": [x["name"] for x in spec]}
