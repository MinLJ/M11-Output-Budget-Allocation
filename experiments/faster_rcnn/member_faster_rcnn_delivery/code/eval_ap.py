# -*- coding: utf-8 -*-
u"""Faster R-CNN 臂自己的检测质量: 在 DEV2K 冻结 Road8 GT 上的标准 COCO AP。

为什么要单独跑这个:
  两条检测器臂要比的不只是 M11 的覆盖/质量, 还有**检测器本身是否可比**。
  如果 Faster R-CNN 的 AP 远低于参照臂, 那 M11 的差异就说不清是分配器还是检测器。
  所以这里把每条臂在自己候选流上的 AP 报出来, 并给出 **AP vs 前缀预算 K** 的曲线 ——
  这条曲线就是「原分数前缀截断在不同 K 下的检测质量」, 与 M11 的动作口径完全一致。

**评价的是导出文件, 不是重新推理一遍。** 导出脚本已经做过逐位一致性校验, 重新推理一遍
只会引入"评价用的记录"和"M11 实际截断的记录"不一致的风险 —— 那样 AP-K 曲线就不再
描述 M11 看到的东西了。

预测与 GT 的类别 id 都是冻结 Road8 的 1..8, image_id 都是 release 的字符串 id,
两边直接对齐, 不做任何类别重映射。

用法:
  python -B src/eval_ap.py --split DEV2K \\
      --primary   outputs/frcnn_r18fpn_road8/export_dev/candidates/dev2k_candidates.parquet \\
      --post-nms  outputs/frcnn_r18fpn_road8/export_dev/candidates/dev2k_candidates_post_nms.parquet \\
      --out outputs/frcnn_r18fpn_road8/ap_dev2k
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model import declaration  # noqa: E402
from road8_data import ROAD8_NAMES  # noqa: E402

EXPERIMENT_ROOT = Path(os.environ.get('M11_EXPERIMENT_ROOT', "external_assets"))
RELEASE = EXPERIMENT_ROOT / 'AOP_Road8_LC_v1_RTDERTv2_R18VD_RELEASE_v1'
GT_FILES = {"DEV2K": "DEV2K_ROAD8_GT.json", "TRAIN10K": "TRAIN10K_ROAD8_GT.json"}
CUTOFFS = (1, 10, 15, 20, 30, 40, 50, 100, 300)


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def load_stream(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    needed = {"image_id", "rank", "original_score", "predicted_road8_class_id",
              "box_x1", "box_y1", "box_x2", "box_y2"}
    missing = needed - set(frame.columns)
    if missing:
        raise ValueError(f"{path.name}: missing columns {sorted(missing)}")
    return frame


def to_coco_records(frame: pd.DataFrame) -> list[dict]:
    return [{
        "image_id": str(row.image_id),
        "category_id": int(row.predicted_road8_class_id),
        "bbox": [float(row.box_x1), float(row.box_y1),
                 float(row.box_x2) - float(row.box_x1), float(row.box_y2) - float(row.box_y1)],
        "score": float(row.original_score),
    } for row in frame.itertuples(index=False)]


def evaluate(gt_json: Path, records: list[dict], cat_ids: list[int] | None = None) -> np.ndarray:
    u"""**必须调 summarize()**: pycocotools 是在 summarize() 里才把 12 个汇总量写进
    `evaluator.stats` 的, 只调 accumulate() 拿到的 stats 是空的 —— 那会让所有 AP 静默变成
    "无法评价", 看起来像检测器一条都没命中。summarize() 会往 stdout 打 12 行, 这里吞掉。
    """
    import contextlib
    import io

    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    gt = COCO(str(gt_json))
    if not records:
        return np.full(12, -1.0)
    loaded = gt.loadRes(records)
    evaluator = COCOeval(gt, loaded, "bbox")
    if cat_ids is not None:
        evaluator.params.catIds = cat_ids
    evaluator.params.maxDets = [1, 10, 100]
    with contextlib.redirect_stdout(io.StringIO()):
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    return np.asarray(evaluator.stats, dtype=np.float64)


def pick(stats: np.ndarray, index: int) -> float | None:
    u"""pycocotools 在"这一类没有 GT 或没有检测"时返回空/负数组 —— 那是"无法评价", 是 None。
    有 GT 却算出 -1 的情况由调用方决定记 0.0 还是 None, 绝不在这里悄悄合并。"""
    if stats.size <= index or stats[index] < 0:
        return None
    return float(stats[index])


def gt_support(gt_json: Path) -> dict[int, dict]:
    raw = json.loads(gt_json.read_text(encoding="utf-8"))
    support = {int(c["id"]): {"gt_count": 0, "images": set()} for c in raw["categories"]}
    for ann in raw["annotations"]:
        cid = int(ann["category_id"])
        if cid in support:
            support[cid]["gt_count"] += 1
            support[cid]["images"].add(str(ann["image_id"]))
    return {cid: {"gt_count": v["gt_count"], "image_count": len(v["images"])}
            for cid, v in support.items()}


def curve_for(stream: str, frame: pd.DataFrame, gt_json: Path, image_count: int,
              cutoffs: list[int]) -> list[dict]:
    depth = int(frame["rank"].max())
    rows = []
    for cutoff in cutoffs:
        if cutoff > depth:
            continue
        subset = frame[frame["rank"].astype(int) <= cutoff]
        records = to_coco_records(subset)
        stats = evaluate(gt_json, records)
        per_image = subset.groupby("image_id", sort=False).size()
        rows.append({
            "stream": stream, "cutoff": cutoff, "depth_available": depth,
            "images": image_count,
            "images_short_of_cutoff": int((per_image.reindex(
                sorted(frame["image_id"].unique())).fillna(0) < cutoff).sum()),
            "detections": len(records),
            "AP": pick(stats, 0), "AP50": pick(stats, 1), "AP75": pick(stats, 2),
            "AP_small": pick(stats, 3), "AP_medium": pick(stats, 4), "AP_large": pick(stats, 5),
            "AR_1": pick(stats, 6), "AR_10": pick(stats, 7), "AR_100": pick(stats, 8),
        })
        log(f"  [{stream}] cutoff={cutoff:>4}  AP={rows[-1]['AP']}  AP50={rows[-1]['AP50']}  "
            f"AR100={rows[-1]['AR_100']}  dets={len(records)}")
    return rows


def per_class(stream: str, frame: pd.DataFrame, gt_json: Path, cutoff: int) -> list[dict]:
    support = gt_support(gt_json)
    subset = frame[frame["rank"].astype(int) <= cutoff]
    records = to_coco_records(subset)
    rows = []
    for class_id, name in enumerate(ROAD8_NAMES, start=1):
        stats = evaluate(gt_json, records, cat_ids=[class_id])
        has_gt = support.get(class_id, {}).get("gt_count", 0) > 0
        ap = pick(stats, 0)
        rows.append({
            "stream": stream, "cutoff": cutoff,
            "road8_class_id": class_id, "class_name": name,
            "gt_count": support.get(class_id, {}).get("gt_count", 0),
            "gt_image_count": support.get(class_id, {}).get("image_count", 0),
            "detections": int((subset["predicted_road8_class_id"].astype(int) == class_id).sum()),
            # 有 GT 而 pycocotools 仍给 -1 → 该类一条都没命中, AP 就是 0.0;
            # 没有有效 GT → None。两者绝不混。
            "AP": ap if ap is not None else (0.0 if has_gt else None),
            "AP50": pick(stats, 1) if pick(stats, 1) is not None else (0.0 if has_gt else None),
            "AP75": pick(stats, 2) if pick(stats, 2) is not None else (0.0 if has_gt else None),
            "AR100": pick(stats, 8) if pick(stats, 8) is not None else (0.0 if has_gt else None),
            "evaluable": bool(has_gt),
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="DEV2K", choices=sorted(GT_FILES))
    ap.add_argument("--primary", required=True, help="主用流 (PRE_NMS_TOPN100) 的候选 parquet")
    ap.add_argument("--post-nms", default="", help="登记备查流 (POST_NMS_TOPN300) 的候选 parquet")
    ap.add_argument("--cutoffs", nargs="+", type=int, default=list(CUTOFFS))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    gt_json = RELEASE / "gt" / GT_FILES[args.split]
    image_count = len(json.loads(gt_json.read_text(encoding="utf-8"))["images"])

    primary = load_stream(Path(args.primary))
    log(f"split={args.split} gt_images={image_count} primary={Path(args.primary).name} "
        f"rows={len(primary)} depth={int(primary['rank'].max())} "
        f"protocol={sorted(primary['protocol_id'].unique()) if 'protocol_id' in primary else 'n/a'}")

    curves = curve_for("primary_pre_nms", primary, gt_json, image_count, args.cutoffs)
    classes = per_class("primary_pre_nms", primary, gt_json,
                        min(100, int(primary["rank"].max())))

    if args.post_nms:
        post = load_stream(Path(args.post_nms))
        log(f"post_nms={Path(args.post_nms).name} rows={len(post)} "
            f"depth={int(post['rank'].max())} "
            f"protocol={sorted(post['protocol_id'].unique()) if 'protocol_id' in post else 'n/a'}")
        curves += curve_for("post_nms", post, gt_json, image_count, args.cutoffs)
        classes += per_class("post_nms", post, gt_json, int(post["rank"].max()))

    curve_frame = pd.DataFrame(curves)
    class_frame = pd.DataFrame(classes)
    curve_frame.to_csv(out / "ap_vs_budget.csv", index=False)
    class_frame.to_csv(out / "ap_per_class.csv", index=False)

    checkpoint_sha = sorted(set(primary["checkpoint_sha256"].astype(str)))[0] \
        if "checkpoint_sha256" in primary else None
    payload = {
        "split": args.split,
        "images": image_count,
        "checkpoint_sha256": checkpoint_sha,
        "declaration": declaration("resnet18"),
        "streams": {
            "primary_pre_nms": {
                "file": str(Path(args.primary).resolve()),
                "protocol_id": sorted(set(primary["protocol_id"].astype(str))) if "protocol_id" in primary else None,
                "depth": int(primary["rank"].max()),
                "note": ("M11 实际截断的就是这条流: 检测头分类+回归之后, 分数阈值/退化框过滤/"
                         "NMS/top-N 之前, 按原分数取前缀。"),
            },
        },
        "ap_vs_prefix_budget": curves,
        "per_class": classes,
        "note": ("cutoff = K 前缀预算下的原分数截断, 与 M11 的候选动作完全同口径。"
                 "pre-NMS 流没有做逐类 NMS, 所以同一目标可能有多条近重复记录 —— "
                 "它的 AP 会比 post-NMS 流低, 这是协议选择的后果, 不是检测器缺陷; "
                 "两条数字都要报, 不能只报高的那条。"),
    }
    if args.post_nms:
        post = load_stream(Path(args.post_nms))
        payload["streams"]["post_nms"] = {
            "file": str(Path(args.post_nms).resolve()),
            "protocol_id": sorted(set(post["protocol_id"].astype(str))) if "protocol_id" in post else None,
            "depth": int(post["rank"].max()),
            "note": "官方 postprocess_detections 的最终输出, 逐位等于 model(images); 只作对照。",
        }
    (out / "ap_report.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"-> {out/'ap_vs_budget.csv'}, {out/'ap_per_class.csv'}, {out/'ap_report.json'}")


if __name__ == "__main__":
    main()
