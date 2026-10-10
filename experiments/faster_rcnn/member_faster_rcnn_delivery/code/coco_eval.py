# -*- coding: utf-8 -*-
u"""标准 COCO bbox 评价, 跑在**已冻结的选集**上, 而不是重新求解配额。

为什么不用包里的 `evaluate_coco_bbox`: 它要求候选表里有整数 `coco_image_id` 并把 Road8
类别映射回 COCO 的 category id。本分支用的是 release 的冻结 Road8 GT —— image_id 是字符串,
类别 id 本来就是 1..8, 两边同一个 ID 空间, 直接对齐即可, 不做任何重映射。

口径按指导文档 §06 逐条落定:
  * iouType="bbox", useCats=1, IoU=.50:.05:.95, 101 个 recall 点, maxDets=[1,10,100]
  * 评价分数**始终是原 detector score**, 不重排、不改分
  * GT 的 area / iscrowd / ignore 语义原样交给 pycocotools
  * 逐 seed 在完整 DEV 上算, 不平均各 40 图组的 AP
  * **有 GT 而 AP=-1 → 记为 0.0; 没有有效 GT → 标 missing=None** —— 两者绝不能混
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from lc_alloc.constants import ROAD8
from lc_alloc.data.schema import normalize_candidate_columns

# 实际生效的参数 —— 被测过的值就是这个, 不是"建议值"
COCO_EVAL_PARAMS = {
    "iouType": "bbox",
    "useCats": 1,
    "iouThrs": "0.50:0.05:0.95 (10 thresholds)",
    "recThrs": "101 points (pycocotools default)",
    "maxDets": [1, 10, 100],
    "areaRng": "pycocotools default all/small/medium/large",
    "score_threshold": "original detector score, no rescaling, no re-ranking",
}


def select_records(candidates: pd.DataFrame, allocation: pd.DataFrame) -> pd.DataFrame:
    u"""按冻结的 K_i 取原分数前缀。输出恒为 rank 1..K_i, 不做任何别的动作。"""
    c = normalize_candidate_columns(candidates)
    k_map = dict(zip(allocation["image_id"].astype(str), allocation["K_i"].astype(int)))
    wanted = set(k_map)
    sub = c[c["image_id"].astype(str).isin(wanted)].copy()
    sub["_k"] = sub["image_id"].astype(str).map(k_map)
    selected = sub[sub["rank"].astype(int) <= sub["_k"]].drop(columns=["_k"])
    return selected.sort_values(["image_id", "rank"], kind="mergesort")


def _gt_support(gt_json: Path) -> dict[int, dict]:
    raw = json.loads(Path(gt_json).read_text(encoding="utf-8"))
    support = {int(c["id"]): {"gt_count": 0, "images": set()} for c in raw["categories"]}
    for ann in raw["annotations"]:
        cid = int(ann["category_id"])
        if cid not in support:
            continue
        support[cid]["gt_count"] += 1
        support[cid]["images"].add(str(ann["image_id"]))
    return {cid: {"gt_count": v["gt_count"], "image_count": len(v["images"])} for cid, v in support.items()}


def evaluate_selection(candidates: pd.DataFrame, allocation: pd.DataFrame,
                       gt_json: str | Path, *, with_per_class: bool = False) -> dict:
    u"""一个条件 (method, seed, budget) 的一组 K_i -> 标准 COCO AP/AR。"""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    import tempfile

    gt_json = Path(gt_json)
    selected = select_records(candidates, allocation)
    results = []
    for row in selected.itertuples(index=False):
        x1, y1, x2, y2 = (float(row.box_x1), float(row.box_y1), float(row.box_x2), float(row.box_y2))
        results.append({
            "image_id": str(row.image_id),
            "category_id": int(row.predicted_road8_class_id),
            "bbox": [x1, y1, x2 - x1, y2 - y1],
            "score": float(row.original_score),
        })

    gt = COCO(str(gt_json))
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "results.json"
        path.write_text(json.dumps(results), encoding="utf-8")
        dt = gt.loadRes(str(path))

        def run(cat_ids: list[int] | None) -> np.ndarray:
            u"""**必须调 summarize()**: pycocotools 把 12 个汇总量写进 `evaluator.stats` 是在
            summarize() 里做的, 只 accumulate() 拿到的是空数组 —— 那会让所有 AP 静默变成
            "无法评价", 看起来像一条都没命中。summarize() 会打 12 行到 stdout, 这里吞掉。
            """
            import contextlib
            import io

            evaluator = COCOeval(gt, dt, "bbox")
            if cat_ids is not None:
                evaluator.params.catIds = cat_ids
            evaluator.params.maxDets = [1, 10, 100]
            with contextlib.redirect_stdout(io.StringIO()):
                evaluator.evaluate()
                evaluator.accumulate()
                evaluator.summarize()
            return np.asarray(evaluator.stats, dtype=np.float64)

        stats = run(None)

    def pick(values: np.ndarray, index: int) -> float | None:
        if values.size <= index or values[index] < 0:
            return None
        return float(values[index])

    payload = {
        "image_count": int(allocation["image_id"].nunique()),
        "total_records": int(len(selected)),
        "AP": pick(stats, 0), "AP50": pick(stats, 1), "AP75": pick(stats, 2),
        "AP_small": pick(stats, 3), "AP_medium": pick(stats, 4), "AP_large": pick(stats, 5),
        "AR1": pick(stats, 6), "AR10": pick(stats, 7), "AR100": pick(stats, 8),
        "eval_params": COCO_EVAL_PARAMS,
    }

    if not with_per_class:
        return payload

    support = _gt_support(gt_json)
    rows = []
    for class_index, name in enumerate(ROAD8, start=1):
        class_stats = run([class_index])
        has_gt = support.get(class_index, {}).get("gt_count", 0) > 0
        ap = class_stats[0] if class_stats.size > 0 else -1.0
        values = {
            "AP": float(ap) if ap >= 0 else (0.0 if has_gt else None),
            "AP50": (lambda v: float(v) if v >= 0 else (0.0 if has_gt else None))(class_stats[1] if class_stats.size > 1 else -1.0),
            "AP75": (lambda v: float(v) if v >= 0 else (0.0 if has_gt else None))(class_stats[2] if class_stats.size > 2 else -1.0),
            "AR100": (lambda v: float(v) if v >= 0 else (0.0 if has_gt else None))(class_stats[8] if class_stats.size > 8 else -1.0),
        }
        rows.append({
            "road8_class_id": class_index,
            "class_name": name,
            "gt_count": support.get(class_index, {}).get("gt_count", 0),
            "gt_image_count": support.get(class_index, {}).get("image_count", 0),
            "class_evaluable": bool(has_gt),
            **values,
        })
    payload["by_class"] = rows
    return payload
