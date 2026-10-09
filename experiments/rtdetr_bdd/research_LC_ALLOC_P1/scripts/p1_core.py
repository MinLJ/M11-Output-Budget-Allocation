"""Core, side-effect-free utilities for LC-ALLOC-P1.

The module deliberately contains no top-level file access.  Every caller must
pass explicit paths, which keeps DEV ground truth out of prediction/allocation
entry points.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


ROAD8 = ("person", "bicycle", "car", "motorcycle", "bus", "train", "truck", "traffic light")
THRESHOLDS = np.asarray([0.50 + 0.05 * i for i in range(10)], dtype=np.float64)
RANKS = np.arange(6, 51, dtype=np.int16)
BUDGETS = (10, 15, 20, 30, 40)
SEEDS = (530101, 530102, 530103)


def sha256_file(path: os.PathLike[str] | str, chunk: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def canonical_json_sha(obj: object) -> str:
    payload = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_json(path: os.PathLike[str] | str, obj: object) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU using the frozen half-open continuous-XYXY semantics."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size == 0 or b.size == 0:
        return np.zeros((len(a), len(b)), dtype=np.float64)
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    iw = np.maximum(0.0, ix2 - ix1)
    ih = np.maximum(0.0, iy2 - iy1)
    inter = iw * ih
    aa = np.maximum(0.0, a[:, 2] - a[:, 0]) * np.maximum(0.0, a[:, 3] - a[:, 1])
    bb = np.maximum(0.0, b[:, 2] - b[:, 0]) * np.maximum(0.0, b[:, 3] - b[:, 1])
    union = aa[:, None] + bb[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def _augment(left: int, adjacency: Sequence[np.ndarray], match_right: np.ndarray, seen: np.ndarray) -> bool:
    for right in adjacency[left]:
        r = int(right)
        if seen[r]:
            continue
        seen[r] = True
        if match_right[r] < 0 or _augment(int(match_right[r]), adjacency, match_right, seen):
            match_right[r] = left
            return True
    return False


def maximum_matching_count(adjacency: Sequence[np.ndarray], n_right: int) -> int:
    if n_right == 0 or len(adjacency) == 0:
        return 0
    match_right = np.full(n_right, -1, dtype=np.int32)
    result = 0
    for left in range(len(adjacency)):
        seen = np.zeros(n_right, dtype=bool)
        result += int(_augment(left, adjacency, match_right, seen))
    return int(result)


def prefix_matching_counts(
    cand_classes: np.ndarray,
    cand_boxes: np.ndarray,
    gt_classes: np.ndarray,
    gt_boxes: np.ndarray,
    max_k: int = 50,
    thresholds: np.ndarray = THRESHOLDS,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return total and per-class prefix matching counts for K=0..max_k.

    Every threshold maintains an independent matching state.  Adding a left
    vertex can increase maximum cardinality only by zero or one.  Since graph
    components are class-disjoint, a successful increment belongs to the new
    candidate's predicted class.
    """
    cand_classes = np.asarray(cand_classes, dtype=np.int16)[:max_k]
    cand_boxes = np.asarray(cand_boxes, dtype=np.float64)[:max_k]
    gt_classes = np.asarray(gt_classes, dtype=np.int16)
    gt_boxes = np.asarray(gt_boxes, dtype=np.float64)
    n_t = len(thresholds)
    totals = np.zeros((max_k + 1, n_t), dtype=np.int16)
    by_class = np.zeros((max_k + 1, len(ROAD8), n_t), dtype=np.int16)
    if len(cand_classes) < max_k:
        raise ValueError(f"candidate pool has {len(cand_classes)} rows, expected at least {max_k}")

    overlaps = iou_xyxy(cand_boxes, gt_boxes)
    same = cand_classes[:, None] == gt_classes[None, :] if len(gt_classes) else np.zeros((max_k, 0), bool)
    for ti, threshold in enumerate(thresholds):
        adjacency: List[np.ndarray] = []
        match_right = np.full(len(gt_classes), -1, dtype=np.int32)
        count = 0
        class_count = np.zeros(len(ROAD8), dtype=np.int16)
        for k0 in range(max_k):
            adjacency.append(np.flatnonzero(same[k0] & (overlaps[k0] >= float(threshold))).astype(np.int32))
            seen = np.zeros(len(gt_classes), dtype=bool)
            gained = int(_augment(k0, adjacency, match_right, seen)) if len(gt_classes) else 0
            count += gained
            if gained:
                class_count[int(cand_classes[k0]) - 1] += 1
            totals[k0 + 1, ti] = count
            by_class[k0 + 1, :, ti] = class_count
    return totals, by_class


def independent_prefix_count(
    cand_classes: np.ndarray,
    cand_boxes: np.ndarray,
    gt_classes: np.ndarray,
    gt_boxes: np.ndarray,
    k: int,
    threshold: float,
) -> int:
    cc = np.asarray(cand_classes)[:k]
    cb = np.asarray(cand_boxes)[:k]
    gc = np.asarray(gt_classes)
    gb = np.asarray(gt_boxes)
    ov = iou_xyxy(cb, gb)
    adjacency = [np.flatnonzero((gc == cc[j]) & (ov[j] >= threshold)).astype(np.int32) for j in range(k)]
    return maximum_matching_count(adjacency, len(gc))


def legacy_prefix_counts(
    cand_classes: np.ndarray,
    cand_boxes: np.ndarray,
    gt_classes: np.ndarray,
    gt_boxes: np.ndarray,
    max_k: int = 50,
) -> Tuple[np.ndarray, np.ndarray]:
    """Frozen legacy rank-greedy matching at IoU .50, prefix K=0..max_k."""
    out = np.zeros(max_k + 1, dtype=np.int16)
    by_class = np.zeros((max_k + 1, len(ROAD8)), dtype=np.int16)
    matched = np.zeros(len(gt_classes), dtype=bool)
    ov = iou_xyxy(np.asarray(cand_boxes)[:max_k], np.asarray(gt_boxes))
    count = 0
    class_count = np.zeros(len(ROAD8), dtype=np.int16)
    for j in range(max_k):
        valid = np.flatnonzero((np.asarray(gt_classes) == int(cand_classes[j])) & (~matched) & (ov[j] >= 0.5))
        if len(valid):
            best = int(valid[np.argmax(ov[j, valid])])
            matched[best] = True
            count += 1
            class_count[int(cand_classes[j]) - 1] += 1
        out[j + 1] = count
        by_class[j + 1] = class_count
    return out, by_class


@dataclass(frozen=True)
class GTImage:
    classes: np.ndarray
    boxes: np.ndarray


def load_coco_gt(path: os.PathLike[str] | str) -> Tuple[Dict[str, GTImage], dict]:
    """Load frozen derived Road8 GT; this function is called only by label/eval entrypoints."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    image_key = {str(x["id"]): str(x.get("source_image_id", x.get("image_id", x["id"]))) for x in raw["images"]}
    classes: Dict[str, List[int]] = {v: [] for v in image_key.values()}
    boxes: Dict[str, List[List[float]]] = {v: [] for v in image_key.values()}
    for ann in raw["annotations"]:
        image_id = image_key[str(ann["image_id"])]
        x, y, w, h = map(float, ann["bbox"])
        classes[image_id].append(int(ann["category_id"]))
        boxes[image_id].append([x, y, x + w, y + h])
    result = {
        image_id: GTImage(
            np.asarray(classes[image_id], dtype=np.int16),
            np.asarray(boxes[image_id], dtype=np.float64).reshape(-1, 4),
        )
        for image_id in classes
    }
    meta = {
        "image_count": len(raw["images"]),
        "annotation_count": len(raw["annotations"]),
        "categories": raw.get("categories", []),
    }
    return result, meta


def feature_spec() -> List[dict]:
    """Frozen 90-column P1 schema in exact model-input order."""
    rows: List[dict] = []

    def add(name: str, block: str, formula: str, standardize: bool = True, unit: str = "unitless") -> None:
        rows.append({
            "index": len(rows), "name": name, "block": block, "formula": formula,
            "source": "release candidate/native state and fixed raw-score prefix",
            "unit": unit, "missing_rule": "none; explicit zero only where formula states it",
            "standardize_fit_only": bool(standardize),
        })

    add("raw_score", "candidate", "release detector sigmoid score")
    add("score_logit", "candidate", "logit(clip(raw_score,1e-6,1-1e-6))")
    add("rank_normalized", "candidate", "road8_rank / 100")
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
        add(f"native_road8_logit_{c.replace(' ', '_')}", "candidate", "L3 native Road8 logit gathered by (image_id,query_index)")
    for j in range(32):
        add(f"embedding_pca_{j:02d}", "candidate", "FIT-only PCA32 transform of stored float16 embedding cast to float32")
    add("embedding_norm", "candidate", "L2 norm of stored float16 embedding after float32 cast")

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
    if len(rows) != 90:
        raise AssertionError(f"feature schema length {len(rows)} != 90")
    return rows


def build_raw_features(
    candidates: pd.DataFrame,
    native_logits: np.ndarray,
    embeddings: np.ndarray,
    pca,
    image_width: float,
    image_height: float,
) -> np.ndarray:
    """Build 45x90 raw feature matrix for ranks 6..50 for one image."""
    c = candidates.sort_values("road8_rank", kind="stable").reset_index(drop=True)
    if len(c) != 100 or c["road8_rank"].tolist() != list(range(1, 101)):
        raise ValueError("each image must have exact ranks 1..100")
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
    if native_logits.shape != (100, 8) or pca32.shape != (100, 32):
        raise ValueError(f"native shapes invalid logits={native_logits.shape} pca={pca32.shape}")

    quant = np.quantile(score, [0.25, 0.50, 0.75], method="linear")
    context = np.asarray([
        score.mean(), score.std(ddof=0), quant[0], quant[1], quant[2],
        score[:5].mean(), score[:20].mean(), score[:50].mean(),
        np.mean(score > 0.10), np.mean(score > 0.25), np.mean(score > 0.50),
        *[np.mean(cls == j) for j in range(1, 9)], np.median(area),
    ], dtype=np.float64)
    rows = np.empty((45, 90), dtype=np.float64)
    for oi, k in enumerate(range(6, 51)):
        j = k - 1
        local = [score[j], math.log(np.clip(score[j], 1e-6, 1 - 1e-6) / (1 - np.clip(score[j], 1e-6, 1 - 1e-6))), k / 100.0]
        onehot = [float(cls[j] == z) for z in range(1, 9)]
        geom = [cx[j], cy[j], wn[j], hn[j], area[j], math.log(max(wn[j], 1e-6) / max(hn[j], 1e-6))]
        local.extend(onehot + geom + native_logits[j].tolist() + pca32[j].tolist() + [emb_norm[j]])

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
            rel = [
                prefix_n, score[:prefix_n].mean(), score[:prefix_n].max(), score[:prefix_n].min(),
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ]
        rows[oi] = np.asarray(local + rel + context.tolist(), dtype=np.float64)
    if not np.all(np.isfinite(rows)):
        raise ValueError("nonfinite feature generated")
    return rows


def table_bin_indices(classes: np.ndarray, ranks: np.ndarray, scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    classes0 = np.asarray(classes, dtype=np.int16) - 1
    ranks = np.asarray(ranks, dtype=np.int16)
    scores = np.asarray(scores, dtype=np.float64)
    rank_bin = np.select(
        [ranks <= 10, ranks <= 15, ranks <= 20, ranks <= 30, ranks <= 50],
        [0, 1, 2, 3, 4], default=-1,
    ).astype(np.int8)
    score_bin = np.searchsorted(np.asarray([0.05, 0.10, 0.20, 0.50], dtype=np.float64), scores, side="right").astype(np.int8)
    if np.any((classes0 < 0) | (classes0 >= 8) | (rank_bin < 0) | (score_bin < 0) | (score_bin >= 5)):
        raise ValueError("invalid TABLE bins")
    return classes0, rank_bin, score_bin


def fit_table(classes: np.ndarray, ranks: np.ndarray, scores: np.ndarray, targets: np.ndarray) -> dict:
    c, r, s = table_bin_indices(classes, ranks, scores)
    y = np.asarray(targets, dtype=np.float64)
    rank_rows = np.zeros((5,), dtype=np.int64)
    rank_pos = np.zeros((5, 10), dtype=np.int64)
    cell_rows = np.zeros((8, 5, 5), dtype=np.int64)
    cell_pos = np.zeros((8, 5, 5, 10), dtype=np.int64)
    for rb in range(5):
        mask = r == rb
        rank_rows[rb] = int(mask.sum())
        rank_pos[rb] = y[mask].sum(axis=0).astype(np.int64)
    for ci in range(8):
        for rb in range(5):
            for sb in range(5):
                mask = (c == ci) & (r == rb) & (s == sb)
                cell_rows[ci, rb, sb] = int(mask.sum())
                cell_pos[ci, rb, sb] = y[mask].sum(axis=0).astype(np.int64)
    rank_prior = (rank_pos + 1.0) / (rank_rows[:, None] + 2.0)
    probabilities = (cell_pos + 20.0 * rank_prior[None, :, None, :]) / (cell_rows[..., None] + 20.0)
    return {
        "rank_rows": rank_rows, "rank_pos": rank_pos, "rank_prior": rank_prior,
        "cell_rows": cell_rows, "cell_pos": cell_pos, "probabilities": probabilities,
    }


def predict_table(table: Mapping[str, np.ndarray], classes: np.ndarray, ranks: np.ndarray, scores: np.ndarray) -> np.ndarray:
    c, r, s = table_bin_indices(classes, ranks, scores)
    return np.asarray(table["probabilities"], dtype=np.float64)[c, r, s]


def compute_class_weights(gt: Mapping[str, GTImage], fit_ids: Iterable[str]) -> Tuple[np.ndarray, np.ndarray]:
    counts = np.zeros(8, dtype=np.int64)
    for image_id in fit_ids:
        g = gt[str(image_id)].classes
        for c in range(1, 9):
            counts[c - 1] += int(np.sum(g == c))
    nmax = int(counts.max()) if len(counts) else 0
    a = np.minimum(4.0, np.sqrt(nmax / np.maximum(counts, 1)))
    z = float(np.sum(counts * a) / max(np.sum(counts), 1))
    weights = a / z
    if not np.isclose(np.sum(counts * weights) / max(np.sum(counts), 1), 1.0, atol=1e-12):
        raise AssertionError("class weight normalization failed")
    return counts, weights.astype(np.float64)


def reliability_rows(prob: np.ndarray, target: np.ndarray, seed: int, calibrated: bool) -> List[dict]:
    p = np.asarray(prob, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    rows: List[dict] = []
    for tmode, ti in [("pooled", None)] + [(f"iou_{v:.2f}", j) for j, v in enumerate(THRESHOLDS)]:
        pp = p.reshape(-1) if ti is None else p[:, ti]
        yy = y.reshape(-1) if ti is None else y[:, ti]
        bins = np.minimum((pp * 10).astype(np.int16), 9)
        for b in range(10):
            mask = bins == b
            rows.append({
                "seed": seed, "calibrated": calibrated, "target": tmode, "bin": b,
                "lower": b / 10.0, "upper": (b + 1) / 10.0,
                "count": int(mask.sum()),
                "mean_probability": float(pp[mask].mean()) if mask.any() else math.nan,
                "event_rate": float(yy[mask].mean()) if mask.any() else math.nan,
            })
    return rows


def bce_numpy(logits: np.ndarray, targets: np.ndarray, temperature: float = 1.0) -> float:
    z = np.asarray(logits, dtype=np.float64) / float(temperature)
    y = np.asarray(targets, dtype=np.float64)
    loss = np.maximum(z, 0) - z * y + np.log1p(np.exp(-np.abs(z)))
    return float(loss.mean())


def brier_numpy(logits: np.ndarray, targets: np.ndarray, temperature: float = 1.0) -> float:
    p = sigmoid(np.asarray(logits, dtype=np.float64) / float(temperature))
    return float(np.mean((p - np.asarray(targets, dtype=np.float64)) ** 2))


def fit_temperature(logits: np.ndarray, targets: np.ndarray) -> Tuple[float, dict]:
    """Fit one shared positive temperature on CAL with fixed bounded scalar optimization."""
    from scipy.optimize import minimize_scalar

    base = bce_numpy(logits, targets, 1.0)
    result = minimize_scalar(lambda t: bce_numpy(logits, targets, float(t)), bounds=(0.25, 4.0), method="bounded", options={"xatol": 1e-8})
    candidates = [(1.0, base), (0.25, bce_numpy(logits, targets, 0.25)), (4.0, bce_numpy(logits, targets, 4.0))]
    if bool(result.success):
        candidates.append((float(result.x), float(result.fun)))
    best_t, best_loss = min(candidates, key=lambda x: (x[1], abs(x[0] - 1.0), x[0]))
    if base - best_loss <= 1e-12:
        best_t, best_loss = 1.0, base
    return float(best_t), {
        "temperature": float(best_t), "raw_bce": base, "calibrated_bce": float(best_loss),
        "raw_brier": brier_numpy(logits, targets, 1.0),
        "calibrated_brier": brier_numpy(logits, targets, best_t),
        "optimizer_success": bool(result.success), "optimizer_message": str(result.message),
        "bounds": [0.25, 4.0], "fallback_tolerance": 1e-12,
    }


def dataframe_to_parquet_atomic(df: pd.DataFrame, path: os.PathLike[str] | str, compression: str = "zstd") -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), tmp, compression=compression)
    os.replace(tmp, p)
