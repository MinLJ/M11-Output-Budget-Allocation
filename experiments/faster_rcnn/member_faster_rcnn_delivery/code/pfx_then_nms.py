# -*- coding: utf-8 -*-
u"""指导文档 §07 的 **PFX_THEN_NMS** 参考协议与计数表。

主比较是 **PFX_EXACT**: 固定候选 → 原序前缀分配 → 评价。它的最终记录数**精确等于**
组预算 (每图 rank ≤ K_i), 没有任何后置删除。本模块补的是另一个协议:

    PFX_THEN_NMS: 固定候选 → 前缀分配 → **对选集追加 NMS** → 评价
    仅 NMS 前精确; NMS 后实际条数另报。

§07 要求的那张唯一新计数表, 这里逐条落地:
  每个 method/seed/budget/image 记录 n_before、n_after、removed_count、NMS 后 record IDs;
  按组汇总总数, 再报告全体均值/最小值/最大值及空输出图数。
  必须满足 removed == n_before − n_after 且 n_after ≤ n_before, **不回填槽位**。

两条边界, 写进代码而不是只写在文档里:
  * **PFX_THEN_NMS 不能证明 NMS_THEN_PFX 也有效。** 后置 NMS 删掉重复记录后, 不会给其他
    图像补回原先失去的预算; 相对次序保留 ≠ 已排除重复候选的作用。这是处理顺序的逻辑解释,
    不是已完成的因果实验。本模块不产出 NMS_THEN_PFX 的任何数字。
  * **IoU 不套用 YOLO 的 0.70。** §07 明确说 0.70 只沿用 YOLO 既有参考, 不自动作为
    Faster R-CNN 所有 NMS 的参数。这里用的是**本检测器声明里冻结的那个 NMS 阈值**,
    从实际实现记录, 不通过 DEV 质量搜索。

用法:
  python -B lc_frcnn/pfx_then_nms.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
M11_ROOT = HERE.parent
EXPERIMENT_ROOT = Path(os.environ.get('M11_EXPERIMENT_ROOT', "external_assets"))
ROAD8 = EXPERIMENT_ROOT / 'frcnn_road8'
DECLARATION = ROAD8 / "outputs/frcnn_r18fpn_road8/detector_declaration.json"


def log(message: str) -> None:
    print(f"[pfx_nms] {message}", flush=True)


def declared_nms() -> dict:
    u"""从冻结声明里读 NMS 语义 —— 不在这里另立一套参数。"""
    import re

    decl = json.loads(DECLARATION.read_text(encoding="utf-8"))
    steps = decl["threshold_nms_topn_order"]
    text = " | ".join(steps)
    iou = None
    match = re.search(r"IoU\s*阈值\s*=\s*([0-9.]+)", text)
    if match:
        iou = float(match.group(1))
    per_class = "per-class" in text or "batched_nms" in text
    return {
        "iou_threshold": iou,
        "class_aware": per_class,
        "source": f"{DECLARATION} → threshold_nms_topn_order",
        "declared_steps": steps,
    }


def apply_nms(frame: pd.DataFrame, iou: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    u"""对**已冻结的选集**逐图逐类做 NMS。返回 (保留的记录, 逐图计数)。

    不回填槽位: 被删掉的位置留空, 不从候选池里补新的进来。
    """
    import torch
    from torchvision.ops import boxes as box_ops

    kept_parts = []
    counts = []
    if "group_id" not in frame.columns:
        raise ValueError("selection lacks group_id; §07 requires the counting table to be aggregatable by group")
    for image_id, group in frame.groupby("image_id", sort=False):
        boxes = torch.as_tensor(
            group[["box_x1", "box_y1", "box_x2", "box_y2"]].to_numpy(np.float32))
        scores = torch.as_tensor(group["original_score"].to_numpy(np.float32))
        # 与导出链同一个约定: batched_nms 传 1 基类别号 (见 export_frcnn.py 的说明)
        labels = torch.as_tensor(group["predicted_road8_class_id"].to_numpy(np.int64))
        keep = box_ops.batched_nms(boxes, scores, labels, iou)
        n_before = int(len(group))
        n_after = int(len(keep))
        counts.append({
            "image_id": str(image_id),
            # group_id 在同一图内是常量 (冻结的 40 图组), 取首行即可。
            # §07 要求这张表**按组汇总**, 所以组号必须一路带进计数表 ——
            # 早先只在最后回头查组, 结果 group_id 不在表里, 逐组汇总被静默跳过了。
            "group_id": int(group["group_id"].iloc[0]),
            "n_before": n_before,
            "n_after": n_after,
            "removed_count": n_before - n_after,
            "kept_record_ids": "|".join(group.iloc[keep.numpy()]["candidate_record_id"].astype(str)),
        })
        kept_parts.append(group.iloc[keep.numpy()])
    kept = pd.concat(kept_parts, ignore_index=True) if kept_parts else frame.iloc[:0]
    return kept, pd.DataFrame(counts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=str(M11_ROOT / "work"))
    ap.add_argument("--out", default=str(M11_ROOT / "work" / "pfx_then_nms"))
    args = ap.parse_args()

    work = Path(args.work)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    nms = declared_nms()
    if nms["iou_threshold"] is None:
        raise SystemExit("冻结声明里没有可解析的 NMS IoU 阈值 —— 不能凭空选一个。")
    log(f"declared NMS: IoU={nms['iou_threshold']} class_aware={nms['class_aware']} "
        f"({nms['source']})")
    log("注意: 这里**不用** YOLO 参考里的 0.70 (§07 明确不自动套用); "
        "用的是本检测器声明里冻结的阈值。")

    selected = pd.read_parquet(work / "selected_records.parquet")
    log(f"selected_records: {len(selected)} rows, "
        f"{selected.groupby(['method','seed','budget']).ngroups} conditions")

    per_image_rows = []
    for (method, seed, budget), group in selected.groupby(["method", "seed", "budget"], sort=False):
        _, counts = apply_nms(group, float(nms["iou_threshold"]))
        counts.insert(0, "budget", int(budget))
        counts.insert(0, "seed", int(seed))
        counts.insert(0, "method", method)
        per_image_rows.append(counts)

    per_image = pd.concat(per_image_rows, ignore_index=True)

    # §07 的硬不变量 —— 不成立就直接炸, 不许"大约对"。
    bad = per_image[per_image["removed_count"] != per_image["n_before"] - per_image["n_after"]]
    if len(bad):
        raise AssertionError(f"removed_count != n_before - n_after on {len(bad)} rows")
    bad = per_image[per_image["n_after"] > per_image["n_before"]]
    if len(bad):
        raise AssertionError(f"n_after > n_before on {len(bad)} rows")
    log("invariants OK: removed == n_before - n_after 且 n_after <= n_before (全部行)")

    per_image.to_csv(out / "pfx_then_nms_per_image.csv", index=False)

    grouped = (per_image.groupby(["method", "seed", "budget"], sort=False)
               .agg(image_count=("image_id", "nunique"),
                    n_before_total=("n_before", "sum"),
                    n_after_total=("n_after", "sum"),
                    removed_total=("removed_count", "sum"),
                    n_before_mean=("n_before", "mean"),
                    n_before_min=("n_before", "min"),
                    n_before_max=("n_before", "max"),
                    n_after_mean=("n_after", "mean"),
                    n_after_min=("n_after", "min"),
                    n_after_max=("n_after", "max"),
                    empty_images=("n_after", lambda s: int((s == 0).sum())))
               .reset_index())
    grouped["group_size"] = 40
    grouped["budget_closure_before"] = grouped["n_before_total"] == grouped["image_count"] * grouped["budget"]
    grouped["slot_backfilling"] = False
    grouped.to_csv(out / "pfx_then_nms_by_condition.csv", index=False)

    # §07 要求逐组汇总, 所以这里**不再用 if 静默跳过** —— 缺 group_id 就会在上面直接炸,
    # 而不是安静地少交一张表、让交付阶段到拷文件时才发现。
    per_group = (per_image.groupby(["method", "seed", "budget", "group_id"], sort=False)
                 .agg(image_count=("image_id", "nunique"),
                      n_before=("n_before", "sum"), n_after=("n_after", "sum"),
                      removed=("removed_count", "sum"))
                 .reset_index())
    per_group["group_size"] = 40
    per_group["budget_closure_before"] = per_group["n_before"] == per_group["image_count"] * per_group["budget"]

    summary = {
        "protocol_id": "PFX_THEN_NMS",
        "role": "reference only; the main comparison is PFX_EXACT",
        "processing_order": "fixed candidates -> prefix allocation -> append NMS to the selection -> evaluate",
        "budget_semantics": "exact only before NMS; the actual post-NMS count is reported separately",
        "nms": nms,
        "not_yolo_iou": ("IoU=0.70 只沿用 YOLO 既有参考; §07 明确不自动套用为 Faster R-CNN 的 NMS 参数。"
                         "此处用的是本检测器冻结声明里的阈值。"),
        "invariants": {
            "removed_equals_before_minus_after": True,
            "after_le_before": True,
            "slot_backfilling": False,
            "verified_rows": int(len(per_image)),
        },
        "does_not_establish": (
            "PFX_THEN_NMS 不能证明 NMS_THEN_PFX 也有效: 后置 NMS 删掉重复记录后, 不会给其他图像"
            "补回原先失去的预算; 相对次序保留不等于已排除重复候选的作用。"
            "这是对处理顺序的逻辑解释, 不是已完成的因果实验。本文件不含 NMS_THEN_PFX 的任何数字。"
        ),
        "conditions": grouped.to_dict(orient="records"),
    }
    (out / "pfx_then_nms_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    per_group.to_csv(out / "pfx_then_nms_by_group.csv", index=False)

    log(f"conditions={len(grouped)}  rows={len(per_image)}  groups={len(per_group)}")
    log(f"-> {out/'pfx_then_nms_per_image.csv'}, {out/'pfx_then_nms_by_condition.csv'}, "
        f"{out/'pfx_then_nms_by_group.csv'}, {out/'pfx_then_nms_summary.json'}")


if __name__ == "__main__":
    main()
