# -*- coding: utf-8 -*-
u"""M11 跨检测器实验指导文档 §04 要求的小样本闭环交付物。

一次闭环, 不拆成多轮审批: 从固定身份顺序取少量 TRAIN 图 + 一个完整 DEV40 图组,
**仅用于工程验证** —— 原图/网络坐标变换、类别映射、候选数、record→native 连接、
数值有限性、标签增量、训练一步、模型保存加载、DP 与评价入口。同一小样本重复
运行, 分别报告数值差与选集一致性。

【为什么闭环样本里除 FIT 之外还要带 EARLY_STOP / CALIBRATION】
冻结的 `command_train` 把 `role == "EARLY_STOP"` 当验证集、`command_calibrate`
把 `role == "CALIBRATION"` 当拟合集。这两个切片为空时 `BCEWithLogitsLoss` 在空
张量上算, 损失是 NaN, `train_allocator` 会直接抛 "no best model captured" ——
第 6 节的入口根本进不去。所以样本必须覆盖三种角色, 这是冻结接口的硬约束,
不是这里擅自扩大样本量。DEV40 只进评价, 绝不进任何拟合。

【本次产物不是科学结果】
这一遍拟合出来的 PCA/scaler/MLP/温度以及那张合成 marginals, 全部是小样本上的
工程脚手架, 只为证明接口能闭环, 不得进论文表格。正式结果仍要全量导出 + 全量训练。

用法:
  python -B lc_frcnn/qa_closed_loop.py
  python -B lc_frcnn/qa_closed_loop.py --train-images 8 --dev-group-index 0

报告结尾是 `CLOSED_LOOP_PASS` 或 `CLOSED_LOOP_FAIL: <原因>`; 任一硬检查不过即失败,
不静默跳过、不补假数据。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import traceback
import unittest
from argparse import Namespace
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
M11_ROOT = HERE.parent
EXPERIMENT_ROOT = Path(os.environ.get('M11_EXPERIMENT_ROOT', "external_assets"))
HANDOFF = EXPERIMENT_ROOT / 'LC_ALLOC_M11_HANDOFF_v1/handoff_LC_ALLOC_M11_v1'
DECLARATION = EXPERIMENT_ROOT / 'frcnn_road8/outputs/frcnn_r18fpn_road8/detector_declaration.json'

# 先 `import lc_frcnn`, 它自己会把覆盖层与交接包装进 sys.path —— 否则 lc_alloc 还看不到。
sys.path.insert(0, str(M11_ROOT))

import lc_frcnn  # noqa: E402,F401
from lc_frcnn.features_frcnn import (  # noqa: E402
    NATIVE_SCHEMA_ID, build_image_features, fit_pca, fit_scaler,
)
from lc_frcnn.run_pipeline import feature_frame_frcnn  # noqa: E402
from lc_alloc.allocation.policies import allocate_m11, allocate_s_fixed  # noqa: E402
from lc_alloc.cli import (  # noqa: E402
    command_calibrate, command_evaluate, command_make_labels, command_train,
)
from lc_alloc.constants import K_MAX, K_MIN, OUTPUT_DIM, ROAD8  # noqa: E402
from lc_alloc.data.io import read_candidates, read_native  # noqa: E402
from lc_alloc.data.schema import NativeTable  # noqa: E402
from lc_alloc.features._executed_p1_core import compute_class_weights  # noqa: E402
from lc_alloc.labels.prefix import load_normalized_gt  # noqa: E402
from lc_alloc.models.mlp import MarginalMLP  # noqa: E402

BUDGETS = (10, 15, 20, 30, 40)
# 5/50 是 K 的上下界: 40*5=200 与 40*50=2000 正好把 DP 的可行容量区间压到头,
# 所以预算边界单独测一遍, 只看中间那几个锚点验不出边界行为。
EDGE_BUDGETS = (K_MIN, 10, 15, 20, 30, 40, K_MAX)
PCA_NAMESPACE = "LC_ALLOC_FRCNN_CLOSED_LOOP|"   # 闭环专用命名空间, 与正式资产的 PCA 身份分开

OUT: list[str] = []
CHECKS: list[dict] = []


class ClosedLoopFailure(RuntimeError):
    u"""任一硬检查不过就抛这个 —— 闭环不允许带伤继续跑后面的步骤。"""


def say(line: str = "") -> None:
    print(line, flush=True)
    OUT.append(line)


def check(name: str, ok: bool, detail: str = "") -> bool:
    CHECKS.append({"name": name, "pass": bool(ok), "detail": str(detail)})
    say(f"- **{'PASS' if ok else 'FAIL'}** · {name}" + (f" —— {detail}" if detail else ""))
    return bool(ok)


def require(name: str, ok: bool, detail: str = "") -> None:
    u"""硬检查: 不过就当场终止, 后面的步骤不再产生任何像结果的数字。"""
    if not check(name, ok, detail):
        raise ClosedLoopFailure(f"{name}({detail})" if detail else name)


def jnum(value):
    u"""把 numpy 标量换成 JSON 能写的 python 标量; NaN/Inf 换成 None 而不是假 0。"""
    if value is None:
        return None
    number = float(value)
    return number if np.isfinite(number) else None


def gt_image_sizes(path: Path) -> dict[str, tuple[int, int]]:
    u"""直接读 normalized GT 的 images 段 —— 原图宽高只在这里声明。"""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return {str(row["image_id"]): (int(row["width"]), int(row["height"])) for row in raw["images"]}


def gt_classes_for(gt: dict, image_ids) -> np.ndarray:
    parts = [gt[str(image_id)].classes for image_id in image_ids if str(image_id) in gt]
    parts = [part for part in parts if len(part)]
    return np.concatenate(parts).astype(np.int64) if parts else np.zeros(0, dtype=np.int64)


def parse_declared_resize() -> tuple[int, int, str]:
    u"""从检测器声明里取 min_size/max_size; 取不到退回 640/640 并说明来源。"""
    try:
        raw = json.loads(DECLARATION.read_text(encoding="utf-8"))
        text = str(raw.get("transform", ""))
        match = re.search(r"min_size=\((\d+),\).*?max_size=(\d+)", text)
        if match:
            return int(match.group(1)), int(match.group(2)), "detector_declaration.json 的 transform 行"
    except Exception as exc:  # noqa: BLE001 —— 声明读不到不该让闭环崩, 但必须如实写出来
        return 640, 640, f"declaration 不可读({type(exc).__name__}), 退回覆盖层声明的 640/640"
    return 640, 640, "declaration 里没有 Resize 行, 退回覆盖层声明的 640/640"


def probe_transform(width: int, height: int, min_size: int, max_size: int) -> dict:
    u"""用真 torchvision transform 在一张全零假图上跑一次 —— 这是"无 letterbox"的实测证据。

    只喂合成张量, 不加载检测器权重、不碰任何真图。
    """
    from torchvision.models.detection.transform import GeneralizedRCNNTransform

    transform = GeneralizedRCNNTransform(
        min_size=min_size, max_size=max_size,
        image_mean=[0.485, 0.456, 0.406], image_std=[0.229, 0.224, 0.225])
    with torch.no_grad():
        image, _ = transform([torch.zeros(3, int(height), int(width))])
    _, _, padded_h, padded_w = (int(value) for value in image.tensors.shape)
    scale = float(max_size) / float(max(width, height))
    content_w, content_h = int(round(width * scale)), int(round(height * scale))
    return {"width": int(width), "height": int(height), "scale": scale,
            "content_w": content_w, "content_h": content_h,
            "padded_w": padded_w, "padded_h": padded_h,
            "pad_right": padded_w - content_w, "pad_bottom": padded_h - content_h}


def native_subset(native: NativeTable, image_id: str) -> NativeTable:
    u"""按 image_id 切出单图原生表 —— 包内私有函数在 cli 里, 这里不跨模块借用私名。"""
    mask = np.asarray(native.image_ids).astype(str) == str(image_id)
    return NativeTable(
        image_ids=np.asarray(native.image_ids)[mask],
        source_ids=np.asarray(native.source_ids)[mask],
        native_vectors=np.asarray(native.native_vectors)[mask],
        class_signals=np.asarray(native.class_signals)[mask],
        native_schema_id=native.native_schema_id,
    )


# ── 1. 小样本选取 ────────────────────────────────────────────────────────
def select_sample(cand_tr: pd.DataFrame, cand_dv: pd.DataFrame, roles: pd.DataFrame,
                  groups: pd.DataFrame, n_train: int, n_support: int, group_index: int) -> dict:
    say("## 1. 小样本选取 (固定身份顺序)")
    say()
    say("选取规则只有一条: **image_id 字典序取前 N 个**。不抽样、不打乱、不按结果挑 —— "
        "同一份 prepared 输入永远得到同一组图, 否则第 7 节的重复一致性无从谈起。")
    say()

    present = set(cand_tr.image_id.astype(str))
    fit_ids = sorted(roles.loc[roles.role == "FIT", "image_id"].astype(str))
    train_ids = fit_ids[:n_train]
    require("1.1 TRAIN FIT 图足以取满 n 张", len(train_ids) == n_train,
            f"FIT 共 {len(fit_ids)} 张, 取到 {len(train_ids)} 张")
    require("1.2 选中的 FIT 图全部在 TRAIN candidate 表里", set(train_ids) <= present,
            f"缺失 {sorted(set(train_ids) - present)[:5]}")

    # EARLY_STOP / CALIBRATION 只为让冻结的 train/calibrate 入口有非空切片, 见模块 docstring。
    support = {}
    for role in ("EARLY_STOP", "CALIBRATION"):
        ids = sorted(roles.loc[roles.role == role, "image_id"].astype(str))
        take = ids[:n_support]
        require(f"1.3 {role} 支持图取满且都在 TRAIN candidate 表里",
                len(take) == n_support and set(take) <= present,
                f"取到 {len(take)} 张, 缺失 {sorted(set(take) - present)[:5]}")
        support[role] = take

    group_sizes = groups.groupby("group_id", sort=True).size()
    group_ids = sorted(group_sizes.index.astype(str))
    require("1.4 DEV 组索引在范围内", 0 <= group_index < len(group_ids),
            f"共 {len(group_ids)} 组, 请求第 {group_index} 组")
    group_id = group_ids[group_index]
    dev_ids = sorted(groups.loc[groups.group_id.astype(str) == group_id, "image_id"].astype(str))
    require("1.5 DEV 组是完整 40 图", len(dev_ids) == 40, f"group_id={group_id} 实际 {len(dev_ids)} 图")
    dev_present = set(cand_dv.image_id.astype(str))
    require("1.6 DEV 组图全部在 DEV candidate 表里", set(dev_ids) <= dev_present,
            f"缺失 {sorted(set(dev_ids) - dev_present)[:5]}")

    say(f"- TRAIN 角色计数: FIT {int((roles.role == 'FIT').sum())}, "
        f"EARLY_STOP {int((roles.role == 'EARLY_STOP').sum())}, "
        f"CALIBRATION {int((roles.role == 'CALIBRATION').sum())}")
    say(f"- 取 FIT 前 **{n_train}** 张 (字典序) —— 这 {n_train} 张就是「固定身份顺序的小样本」:")
    say()
    say("  ```")
    for image_id in train_ids:
        say(f"  {image_id}")
    say("  ```")
    say()
    for role, ids in support.items():
        say(f"- 取 {role} 前 **{len(ids)}** 张 (字典序, 仅为让冻结入口有非空切片): "
            f"`{', '.join(ids)}`")
    say(f"- DEV 组: 按 group_id 字典序取第 {group_index} 个 (共 {len(group_ids)} 组), "
        f"group_id=`{group_id}`, 完整 **{len(dev_ids)}** 图:")
    say()
    say("  ```")
    for start in range(0, len(dev_ids), 4):
        say("  " + "  ".join(dev_ids[start:start + 4]))
    say("  ```")
    say()
    say("- DEV40 **只用于评价**: PCA/scaler/温度/DP 值一律不碰这 40 张。")
    say()
    return {"train_ids": train_ids, "support": support, "group_id": group_id,
            "group_ids_total": len(group_ids), "dev_ids": dev_ids}


# ── 2. 原图 / 网络坐标变换 ───────────────────────────────────────────────
def section_coordinates(records: pd.DataFrame, gt_sizes: dict, min_size: int, max_size: int,
                        source: str, native_extra: dict) -> dict:
    say("## 2. 原图 / 网络坐标变换")
    say()
    say(f"本臂预处理 = `GeneralizedRCNNTransform(min_size={min_size}, max_size={max_size})`, "
        f"声明 `letterbox = False`; 来源: {source}")
    say()
    say("候选框 `box_x1..y2` 按声明的 `bbox_policy` **已经映射回原图坐标系** "
        "(postprocess 的逆变换), 所以下面直接拿原图宽高判边界。")
    say()

    width = records.image_width.to_numpy(np.float64)
    height = records.image_height.to_numpy(np.float64)
    x1 = records.box_x1.to_numpy(np.float64)
    y1 = records.box_y1.to_numpy(np.float64)
    x2 = records.box_x2.to_numpy(np.float64)
    y2 = records.box_y2.to_numpy(np.float64)

    in_bounds = (x1 >= 0.0) & (y1 >= 0.0) & (x2 <= width) & (y2 <= height)
    require("2.1 候选框全部落在 [0,W]x[0,H] 内", bool(in_bounds.all()),
            f"越界 {int((~in_bounds).sum())} / {len(records)} 条")

    box_w, box_h = x2 - x1, y2 - y1
    non_degenerate = (box_w > 0) & (box_h > 0)
    area = box_w * box_h
    require("2.2 无退化框, 宽高与面积均为正且有限",
            bool(non_degenerate.all() and np.isfinite(area).all()),
            f"退化 {int((~non_degenerate).sum())} 条, min_w={box_w.min():.4f} min_h={box_h.min():.4f}")

    # 贴边是合法的(检测器可以把框顶到图像边), 这条按"报告"处理, 不做判据。
    eps = 1e-9
    clipped = (x1 <= eps) | (y1 <= eps) | (x2 >= width - eps) | (y2 >= height - eps)

    checked, mismatch = 0, []
    for image_id, (gt_w, gt_h) in gt_sizes.items():
        rows = records[records.image_id.astype(str) == image_id]
        if rows.empty:
            continue
        checked += 1
        if not (int(rows.image_width.iloc[0]) == gt_w and int(rows.image_height.iloc[0]) == gt_h):
            mismatch.append(image_id)
    require("2.3 candidate 的 image_width/height 与 GT 声明逐图一致",
            bool(checked == len(gt_sizes) and not mismatch),
            f"核对 {checked}/{len(gt_sizes)} 张, 不一致 {mismatch[:5]}")

    say(f"- 记录数 **{len(records)}**, 图数 **{records.image_id.nunique()}**")
    say(f"- 框宽 > 0 比例 **{(box_w > 0).mean() * 100:.4f}%**, "
        f"框高 > 0 比例 **{(box_h > 0).mean() * 100:.4f}%**")
    say(f"- 被夹到图像边界的框 **{clipped.mean() * 100:.4f}%** ({int(clipped.sum())} 条) —— "
        f"报告项: 贴边本身合法, 但这个比例异常高就意味着逆变换把框顶出了原图, 要人看一眼")
    say(f"- 框面积 (原图像素): min **{area.min():.1f}**, 中位 **{np.median(area):.1f}**, "
        f"均值 {area.mean():.1f}, max **{area.max():.1f}**")
    say(f"- 框面积占整图比例: min {float((area / (width * height)).min()):.3e}, "
        f"中位 **{np.median(area / (width * height)):.6f}**, max {float((area / (width * height)).max()):.6f}")
    say()

    probes = []
    say("| 原图尺寸 | 图数 | 缩放比例 | 预计内容尺寸 | 实测张量 (C×H×W) | 对齐 padding (底,右) |")
    say("|---|---|---|---|---|---|")
    for (image_w, image_h) in sorted({(int(w), int(h)) for w, h in zip(width, height)}):
        count = int(((width == image_w) & (height == image_h)).sum())
        probe = {**probe_transform(image_w, image_h, min_size, max_size), "images": count}
        probes.append(probe)
        say(f"| {image_w}×{image_h} | {count} | {probe['scale']:.6f} | "
            f"{probe['content_w']}×{probe['content_h']} | 3×{probe['padded_h']}×{probe['padded_w']} | "
            f"{probe['pad_bottom']},{probe['pad_right']} |")
    say()
    require("2.4 实测长边缩到固定值且无 letterbox 补边",
            bool(all(abs(max(p["padded_h"], p["padded_w"]) - max_size) < 32 for p in probes)
                 and all(0 <= p["pad_bottom"] < 32 and 0 <= p["pad_right"] < 32 for p in probes)),
            "实测: " + "; ".join(f"{p['width']}x{p['height']}->{p['padded_h']}x{p['padded_w']}" for p in probes))
    landscape = [p for p in probes if (p["width"], p["height"]) == (1280, 720)]
    if landscape:
        require("2.5 1280×720 的缩放比例 == 0.5", bool(abs(landscape[0]["scale"] - 0.5) < 1e-12),
                f"实测 {landscape[0]['scale']:.6f}")
        say("> 1280×720 → 内容 **640×360**, 张量再被 `size_divisible=32` 对齐到 3×384×640。"
            "多出来的 **24 行**只是右/下对齐 padding, **不是**为保持长宽比补的边 —— "
            "letterbox 会两个方向同时补、且比例不守恒。")
    else:
        say("> 本样本里没有 1280×720 的图, 0.5 这个具体数值不适用; 缩放比例按上表逐尺寸给出。")
    say()

    resized_column = None
    for name, array in native_extra.items():
        if array.ndim == 2 and array.shape[1] == 4 and re.search(r"resize|xyxy|proposal|box", name, re.I):
            resized_column = name
            break
    if resized_column is None:
        say(f"- native sidecar 的额外数组: {sorted(native_extra) or '无'}; **没有** resized/proposal "
            f"坐标列, 该子项不适用 —— 但「无 letterbox」已经由上面真跑 transform 的实测覆盖, "
            f"不是静默跳过。")
    else:
        say(f"- 发现原生坐标列 `{resized_column}` —— 见 JSON 里的逐条比值核对。")
    say()

    return {"records": int(len(records)), "images": int(records.image_id.nunique()),
            "positive_width_fraction": jnum((box_w > 0).mean()),
            "positive_height_fraction": jnum((box_h > 0).mean()),
            "clipped_fraction": jnum(clipped.mean()), "clipped_records": int(clipped.sum()),
            "area_min": jnum(area.min()), "area_median": jnum(np.median(area)),
            "area_max": jnum(area.max()),
            "sizes": probes, "resized_coordinate_column": resized_column,
            "min_size": int(min_size), "max_size": int(max_size), "resize_source": source}


# ── 3. 类别映射 ──────────────────────────────────────────────────────────
def section_classes(records: pd.DataFrame, gt_tr: dict, gt_dv: dict, sample_ids: list[str]) -> dict:
    say("## 3. 类别映射")
    say()
    cand_class = records.predicted_road8_class_id.to_numpy(np.int64)
    require("3.1 候选 predicted_road8_class_id 全在 1..8",
            bool((cand_class >= 1).all() and (cand_class <= len(ROAD8)).all()),
            f"取值范围 [{cand_class.min()}, {cand_class.max()}]")

    # 取值域看**整份** GT(全部 TRAIN10K + DEV2K), 计数表只看本样本 —— 前者证 ID 空间,
    # 后者才是和候选分布可比的分母。
    gt_all = np.concatenate([gt_classes_for(gt_tr, list(gt_tr)), gt_classes_for(gt_dv, list(gt_dv))])
    require("3.2 GT road8_class_id 全在 1..8 (与候选同一 ID 空间, 无重映射)",
            bool(len(gt_all) > 0 and (gt_all >= 1).all() and (gt_all <= len(ROAD8)).all()),
            f"GT 标注 {len(gt_all)} 条, 取值域 {sorted(set(gt_all.tolist()))}")
    gt_sample = np.concatenate([gt_classes_for(gt_tr, sample_ids), gt_classes_for(gt_dv, sample_ids)])

    cand_counts = np.bincount(cand_class, minlength=len(ROAD8) + 1)[1:9]
    gt_counts = np.bincount(gt_sample, minlength=len(ROAD8) + 1)[1:9]
    say()
    say("| class_id | ROAD8 类名 | 候选记录数 | 候选占比 | 本样本 GT 实例数 |")
    say("|---|---|---|---|---|")
    for index, name in enumerate(ROAD8):
        say(f"| {index + 1} | `{name}` | {int(cand_counts[index])} | "
            f"{cand_counts[index] / max(len(records), 1) * 100:.4f}% | {int(gt_counts[index])} |")
    say()
    say(f"候选每条记录的类名就是 `lc_alloc.constants.ROAD8[id-1]`, "
        f"例如第 1 条记录 id={int(cand_class[0])} → `{ROAD8[int(cand_class[0]) - 1]}`。")
    say()

    if "predicted_road8_class_name" in records.columns:
        expected = [ROAD8[int(value) - 1] for value in cand_class]
        observed = records.predicted_road8_class_name.astype(str).tolist()
        require("3.3 候选表里的 class_name 与 ROAD8 常量逐条一致",
                expected == observed, f"不一致 {sum(a != b for a, b in zip(expected, observed))} 条")
    else:
        say("- 候选表没有 `predicted_road8_class_name` 列, 3.3 不适用; "
            "类名一律由 ROAD8 常量现场解析。")
    say()

    return {"class_names": list(ROAD8), "candidate_counts": cand_counts.astype(int).tolist(),
            "gt_counts": gt_counts.astype(int).tolist(),
            "gt_instances_full": int(len(gt_all)), "gt_instances_sample": int(len(gt_sample)),
            "candidate_class_range": [int(cand_class.min()), int(cand_class.max())],
            "gt_class_range": [int(gt_all.min()), int(gt_all.max())]}


# ── 4. 候选数与 record → native 连接 ─────────────────────────────────────
def section_counts_and_join(cand_sample: pd.DataFrame, cand_dev_group: pd.DataFrame,
                            native_tr: NativeTable, native_dv: NativeTable,
                            sample_ids: list[str], dev_ids: list[str],
                            verify: bool = True) -> dict:
    u"""`verify=False`: 只重算数值, 不加叙述、不重复登记同名硬检查 —— 第 7 节用它做
    第二遍复算, 关键字段(含逐图候选数)的差异由 7.9 硬检查兜底。"""
    if verify:
        say("## 4. 候选数与 record → native 连接")
        say()
    counts = {}
    for label, frame, ids in (("TRAIN 小样本", cand_sample, sample_ids), ("DEV40 组", cand_dev_group, dev_ids)):
        per_image = frame.groupby("image_id", sort=False).size().reindex(ids)
        counts[label] = per_image
        if verify:
            require(f"4.1 每图候选数 ≥ 100 ({label})", bool((per_image >= 100).all()),
                    f"min={int(per_image.min())} 中位={int(per_image.median())} max={int(per_image.max())}")
    if verify:
        say("| 范围 | 图数 | min | 中位 | max | 记录合计 |")
        say("|---|---|---|---|---|---|")
        for label, series in counts.items():
            say(f"| {label} | {len(series)} | {int(series.min())} | {int(series.median())} | "
                f"{int(series.max())} | {int(series.sum())} |")
        say()

    join = {}
    for label, frame, native in (("TRAIN", cand_sample, native_tr), ("DEV", cand_dev_group, native_dv)):
        image_ids = np.asarray(native.image_ids).astype(str)
        source_ids = np.asarray(native.source_ids).astype(str)
        vectors = np.asarray(native.native_vectors)
        signals = np.asarray(native.class_signals)

        duplicated = int(pd.DataFrame({"image_id": image_ids, "source_id": source_ids})
                         .duplicated().sum())
        if verify:
            require(f"4.2 native (image_id, source_id) 唯一 ({label})", duplicated == 0,
                    f"重复键 {duplicated} 条")

        lookup = {(a, b): index for index, (a, b) in enumerate(zip(image_ids.tolist(), source_ids.tolist()))}
        hits, missing = [], 0
        for row in frame.itertuples(index=False):
            key = (str(row.image_id), str(row.source_id))
            if key in lookup:
                hits.append(lookup[key])
            else:
                missing += 1
        pairs = frame.groupby(["image_id", "source_id"], sort=False).size()
        shared = int((pairs > 1).sum())
        scores = frame.original_score.to_numpy(np.float64)
        if verify:
            require(f"4.3 join missing == 0 ({label})", missing == 0, f"missing {missing} 条")
            require(f"4.4 join ambiguous == 0 ({label})", duplicated == 0, f"ambiguous {duplicated} 条")
            require(f"4.5 join 后记录数不变, 没有按 source_id 去重 ({label})",
                    len(hits) == len(frame), f"join 命中 {len(hits)} == 候选 {len(frame)}")
            require(f"4.6 original_score 全有限且 ∈ (0,1] ({label})",
                    bool(np.isfinite(scores).all() and (scores > 0).all() and (scores <= 1).all()),
                    f"min={scores.min():.3e} max={scores.max():.6f}")
            require(f"4.7 native_vectors / class_signals 全有限 ({label})",
                    bool(np.isfinite(vectors.astype(np.float64)).all()
                         and np.isfinite(signals.astype(np.float64)).all()),
                    f"vectors {vectors.shape} {vectors.dtype}, signals {signals.shape} {signals.dtype}")

        join[label] = {"records": int(len(frame)), "native_rows": int(len(image_ids)),
                       "missing": int(missing), "ambiguous": int(duplicated),
                       "shared_source_records": shared,
                       "unique_sources": int(len(pairs)),
                       "records_per_source_max": int(pairs.max()),
                       "records_per_source_median": jnum(pairs.median()),
                       "records_per_source_mean": jnum(len(frame) / max(len(pairs), 1)),
                       "score_min": jnum(scores.min()), "score_max": jnum(scores.max())}
        if verify:
            say(f"- **{label}**: 候选 {len(frame)} 条 / 原生行 {len(image_ids)} 条 · "
                f"missing **{missing}** · ambiguous **{duplicated}**")
            say(f"  - 共享同一 `source_id` 的来源 **{shared}** 条 "
                f"({shared / max(len(pairs), 1) * 100:.2f}%), 每条来源承载 1..{int(pairs.max())} 条记录 "
                f"(中位 {pairs.median():.0f}) —— 多类别共享一个 RoI 是声明过的 one-to-many, "
                f"**全程不做任何去重**")
            say(f"  - native_vectors {vectors.shape} {vectors.dtype}, class_signals {signals.shape} "
                f"{signals.dtype}; score∈({scores.min():.3e}, {scores.max():.6f}]")
    if verify:
        say()
        say("> native 张量是真的被逐条 join 过的(上面的命中数就是物证)。join **只有** "
            "`(image_id, source_id)` 一个键, rank、数组行号、reshape 位置一律不参与。")
        say()
    return {"counts": {label: series.astype(int).tolist() for label, series in counts.items()},
            "counts_by_image": {label: {str(key): int(value) for key, value in series.items()}
                                for label, series in counts.items()},
            "join": join}


# ── 5+6. 标签增量 / 训练一步 / DP / 评价 ─────────────────────────────────
def labels_matrix(labels_frame: pd.DataFrame) -> tuple[np.ndarray, list[str], list[str]]:
    u"""按 (image_id, rank) 排序后拼成 (图, 45, 10) 的增量张量。"""
    y_columns = sorted(column for column in labels_frame.columns if column.startswith("y_"))
    image_ids = sorted(set(labels_frame.image_id.astype(str)))
    blocks = []
    for image_id in image_ids:
        part = labels_frame[labels_frame.image_id.astype(str) == image_id].sort_values("rank", kind="mergesort")
        blocks.append(part[y_columns].to_numpy(np.float64))
    return np.asarray(blocks), image_ids, y_columns


def build_sample_features(tag: str, stage: Path, cand_sample: pd.DataFrame, native_tr: NativeTable,
                          roles_sample: pd.DataFrame, fit_ids: list[str]) -> tuple[pd.DataFrame, object]:
    u"""FIT-only PCA32 + scaler, 再给整个小样本算 90 维特征。

    PCA/scaler **只在 FIT 图上拟合**, 这条不因为样本小就放弃 —— 否则 EARLY_STOP /
    CALIBRATION 的信息从预处理那一步就漏进来了, 后面所有一致性检查都白做。
    """
    fit_keys = {(str(row.image_id), str(row.source_id))
                for row in cand_sample[cand_sample.image_id.astype(str).isin(fit_ids)].itertuples(index=False)}
    native_ids = np.asarray(native_tr.image_ids).astype(str)
    native_src = np.asarray(native_tr.source_ids).astype(str)
    mask = np.asarray([(a, b) in fit_keys for a, b in zip(native_ids.tolist(), native_src.tolist())], dtype=bool)
    require(f"{tag} 特征: FIT 原生行足够拟合 PCA32", int(mask.sum()) >= 32,
            f"FIT native 行 {int(mask.sum())}")

    pca = fit_pca(native_ids[mask], native_src[mask], np.asarray(native_tr.native_vectors)[mask],
                  namespace=PCA_NAMESPACE, random_state=730100)
    raw_fit = [build_image_features(cand_sample[cand_sample.image_id.astype(str) == image_id],
                                    native_subset(native_tr, image_id), pca)
               for image_id in sorted(fit_ids)]
    scaler_bundle = fit_scaler(np.vstack(raw_fit))
    features = feature_frame_frcnn(cand_sample, native_tr, pca, scaler_bundle, roles_sample)
    features.to_parquet(stage / "features_train_sample.parquet", index=False)
    joblib.dump(pca, stage / "pca32_closed_loop.joblib")
    joblib.dump(scaler_bundle, stage / "scaler_closed_loop.joblib")

    matrix = features.filter(like="f").to_numpy(np.float64)
    require(f"{tag} 特征: 维数 == 90", int(matrix.shape[1]) == 90, f"实际 {matrix.shape[1]}")
    require(f"{tag} 特征: 矩阵全有限", bool(np.isfinite(matrix).all()), f"shape {matrix.shape}")
    say(f"- 特征 `{matrix.shape}` = {features.image_id.nunique()} 图 × 45 rank × 90 维; "
        f"PCA32 解释方差 {float(pca.explained_variance_ratio_.sum()):.4f} (小样本, 只作工程验证)")
    say()
    return features, pca


def synthetic_marginals(cand_dev_group: pd.DataFrame, image_ids: list[str]) -> np.ndarray:
    u"""合成 marginals: 直接取每图 rank 6..50 的原分数当槽位价值。

    它不是 M11 模型的输出 —— 但来自真候选流, 比随机数更容易压到并列与边界。
    只允许用于工程验证, 不得当成科学结果。
    """
    return np.asarray([
        cand_dev_group[cand_dev_group.image_id.astype(str) == image_id]
        .sort_values("rank", kind="mergesort").original_score.to_numpy(np.float64)[K_MIN:K_MAX]
        for image_id in image_ids], dtype=np.float64)


def allocation_checks(prefix: str, marginals: np.ndarray, budgets) -> dict:
    u"""每个预算都必须恰好花完 `N * budget`, 且每个 K_i 落在 [K_MIN, K_MAX]。"""
    result = {}
    solved, objectives = allocate_m11(marginals, budgets)
    for name, matrix in (("allocate_m11", solved),
                         ("allocate_s_fixed", allocate_s_fixed(marginals.shape[0], budgets))):
        require(f"{prefix} {name}: K_i ∈ [5,50]",
                bool((matrix >= K_MIN).all() and (matrix <= K_MAX).all()),
                f"min={int(matrix.min())} max={int(matrix.max())} dtype={matrix.dtype}")
        expected = np.asarray(budgets, dtype=np.int64) * matrix.shape[1]
        require(f"{prefix} {name}: 整数解且 sum(K_i) == N*budget",
                bool(matrix.dtype == np.int64 and np.array_equal(matrix.sum(axis=1, dtype=np.int64), expected)),
                f"sums={matrix.sum(axis=1).astype(int).tolist()} expected={expected.tolist()}")
        result[name] = {"K": matrix.tolist(), "sums": matrix.sum(axis=1).astype(int).tolist(),
                        "min": int(matrix.min()), "max": int(matrix.max())}
    result["allocate_m11"]["objectives"] = [jnum(value) for value in objectives]
    result["budgets"] = [int(value) for value in budgets]
    return result


def run_closure(tag: str, stage: Path, ctx: dict, seed: int, device: str) -> dict:
    u"""跑一遍完整闭环: 特征 → 标签 → 训练一步 → 重载 → 温度 → DP → 评价。"""
    stage.mkdir(parents=True, exist_ok=True)
    say(f"### {tag}")
    say()
    say(f"#### {tag} · 特征构造 (第 6 节输入; PCA/scaler 只用 FIT 拟合)")
    say()

    features, _pca = build_sample_features(tag, stage, ctx["cand_sample"], ctx["native_tr"],
                                           ctx["roles_sample"], ctx["fit_ids"])
    dev_ids = ctx["dev_ids"]

    # --- 标签增量: 直接调交接包的 command_make_labels, 不自己重写一遍算法 ---
    say(f"#### {tag} · 标签增量 (第 5 节)")
    say()
    labels_root = stage / "labels_out"
    label_info = command_make_labels(Namespace(
        candidates=str(ctx["sample_candidates"]), gt=str(ctx["gt_train"]), gt_format="normalized",
        roles=str(ctx["sample_roles"]), output_root=str(labels_root)))
    labels_frame = pd.read_parquet(labels_root / "labels.parquet")
    matrix, label_ids, y_columns = labels_matrix(labels_frame)
    require(f"{tag} 标签: shape == (图数, 45, 10)", matrix.shape[1:] == (45, OUTPUT_DIM),
            f"实际 {matrix.shape}")
    require(f"{tag} 标签: 取值只有 0/1", bool(np.isin(matrix, (0.0, 1.0)).all()),
            f"取值 {sorted(np.unique(matrix).tolist())}")
    cumulative = np.cumsum(matrix, axis=1)
    require(f"{tag} 标签: 累计 m(k) 沿 k 单调不减",
            bool((np.diff(cumulative, axis=1) >= 0).all() and (cumulative >= 0).all()),
            "labels 是 m(k)-m(k-1) 的增量, 累计后必须非降")
    require(f"{tag} 标签: 单步增量 ≤ 1", bool(matrix.max() <= 1.0), f"max={matrix.max()}")
    positive = float(matrix.sum() / matrix.size)
    say(f"- 标签: rows={len(labels_frame)}, 图数={len(label_ids)}, "
        f"正标签比例 **{positive * 100:.4f}%** ({int(matrix.sum())}/{int(matrix.size)}), "
        f"每图累计峰值 {int(cumulative.max())}")
    say(f"- y 列 `{y_columns[0]}` … `{y_columns[-1]}` 共 {len(y_columns)} 个 IoU 阈值, "
        f"与 `THRESHOLDS` 一致; 正标签比例低是正常的(小样本 + 前缀匹配本来就稀疏), "
        f"这里只要求它算得出来且自洽。")
    say()

    # --- 训练一步 + 保存后重载 ---
    say(f"#### {tag} · 训练一步 / 重载 / 温度 / DP / 评价入口 (第 6 节)")
    say()
    train_root = stage / "train_out"
    train_info = command_train(Namespace(
        features=str(stage / "features_train_sample.parquet"), labels=str(labels_root / "labels.parquet"),
        seed=seed, output_root=str(train_root), device=device,
        max_epochs=1, patience=1, batch_size=4096))
    model_path = train_root / f"marginal_mlp_seed_{seed}.pt"
    require(f"{tag} 训练: 一步训练产出 checkpoint",
            bool(model_path.is_file() and train_info.get("status") == "PASS"),
            f"status={train_info.get('status')} epochs_run={train_info.get('epochs_run')} "
            f"best_epoch={train_info.get('best_epoch')}")

    snapshot = torch.load(model_path, map_location="cpu", weights_only=True)
    reloaded = MarginalMLP(int(snapshot["input_dim"]))
    reloaded.load_state_dict(snapshot["state_dict"], strict=True)
    reloaded.eval()
    with torch.inference_mode():
        logits = reloaded(torch.from_numpy(features.filter(like="f").to_numpy(np.float32)[:64])).numpy()
    require(f"{tag} 训练: torch.load(weights_only=True) + MarginalMLP 重载成功且前向有限",
            bool(np.isfinite(logits).all() and logits.shape[1] == OUTPUT_DIM),
            f"input_dim={snapshot['input_dim']} 前向 shape={logits.shape}")
    state = {name: tensor.detach().cpu().clone() for name, tensor in snapshot["state_dict"].items()}
    say(f"- 训练: epochs_run={train_info['epochs_run']} best_epoch={train_info['best_epoch']}, "
        f"重载通过, 前向 {logits.shape} 全有限")
    say()

    # --- 温度 ---
    calib_info = command_calibrate(Namespace(
        model=str(model_path), features=str(stage / "features_train_sample.parquet"),
        labels=str(labels_root / "labels.parquet"), output_root=str(stage / "calib_out"), device=device))
    temperature = float(calib_info["temperature"])
    require(f"{tag} 温度: 有限正数", bool(np.isfinite(temperature) and temperature > 0.0),
            f"T={temperature:.6f} raw_bce={jnum(calib_info.get('raw_bce'))} "
            f"cal_bce={jnum(calib_info.get('calibrated_bce'))}")
    say(f"- 温度: T={temperature:.6f} (搜索边界 {calib_info.get('bounds')}), "
        f"BCE {jnum(calib_info.get('raw_bce'))} → {jnum(calib_info.get('calibrated_bce'))}")
    say()

    # --- DP: 真值合成 marginals + 全零并列 + 预算边界 ---
    marginals = synthetic_marginals(ctx["cand_dev_group"], dev_ids)
    require(f"{tag} DP: 合成 marginals shape == (40,45)",
            marginals.shape == (len(dev_ids), K_MAX - K_MIN), f"实际 {marginals.shape}")
    say(f"- DP 输入 = 合成 marginals `{marginals.shape}` (每图 rank 6..50 的原分数), "
        f"**工程验证用, 不是 M11 模型的输出**")
    allocation = allocation_checks(f"{tag} 真值·", marginals, BUDGETS)
    tie = allocation_checks(f"{tag} 全零并列·", np.zeros((len(dev_ids), K_MAX - K_MIN)), EDGE_BUDGETS)
    say(f"- 全零 marginals (全并列) 仍返回精确整数解: sum={tie['allocate_m11']['sums']}, "
        f"K ∈ [{tie['allocate_m11']['min']}, {tie['allocate_m11']['max']}]")
    say(f"- 预算边界(含 K_MIN=5 与 K_MAX=50 两个端点预算)同样精确花完: "
        f"sum={tie['allocate_s_fixed']['sums']}")
    say()

    # --- 评价入口 ---
    n_images = len(dev_ids)
    allocations = pd.DataFrame({
        "group_id": ["closed_loop"] * (n_images * len(BUDGETS) * 2),
        "image_id": dev_ids * len(BUDGETS) * 2,
        "K_i": np.concatenate([np.concatenate([allocation["allocate_m11"]["K"][index]
                                               for index in range(len(BUDGETS))]),
                               np.concatenate([allocation["allocate_s_fixed"]["K"][index]
                                               for index in range(len(BUDGETS))])]),
        "method": ["M11_SYNTHETIC"] * (n_images * len(BUDGETS)) + ["S_FIXED"] * (n_images * len(BUDGETS)),
        "seed": [seed] * (n_images * len(BUDGETS)) + [-1] * (n_images * len(BUDGETS)),
        # 预算列必须与 K_i 同序: 每个预算连续 n_images 行 (预算外层、图内层),
        # 顺序错了会让同一个 (method, seed, budget) 组里出现重复 image_id。
        "budget": [int(budget) for budget in BUDGETS for _ in range(n_images)] * 2,
    })
    allocations_path = stage / "allocations_synthetic.parquet"
    allocations.to_parquet(allocations_path, index=False)

    eval_info = command_evaluate(Namespace(
        candidates=str(ctx["candidates_dev_group"]), allocations=str(allocations_path),
        gt=str(ctx["gt_dev"]), gt_format="normalized", class_weights=str(ctx["class_weights"]),
        coco_ap=False, output_root=str(stage / "eval_out")))
    main = pd.read_csv(stage / "eval_out" / "main_results.csv")
    per_image = pd.read_csv(stage / "eval_out" / "per_image_results.csv")
    require(f"{tag} 评价: 条件数 == 2 方法 × {len(BUDGETS)} 预算",
            len(main) == len(BUDGETS) * 2, f"conditions={len(main)}")
    require(f"{tag} 评价: 每条件 40 图且 output_records == 40×budget",
            bool((main.image_count == n_images).all()
                 and (main.output_records == main.budget * n_images).all()),
            f"image_count={sorted(set(main.image_count.tolist()))} "
            f"output_records={sorted(set(main.output_records.tolist()))}")
    require(f"{tag} 评价: 数值全有限",
            bool(np.isfinite(main.coverage_per_image.to_numpy(np.float64)).all()
                 and np.isfinite(main.quality_per_image.to_numpy(np.float64)).all()),
            f"coverage/图 ∈ [{main.coverage_per_image.min():.4f}, {main.coverage_per_image.max():.4f}], "
            f"quality/图 ∈ [{main.quality_per_image.min():.6f}, {main.quality_per_image.max():.6f}]")
    say(f"- 评价: {len(main)} 个条件 × {n_images} 图 = {len(per_image)} 条 per-image 结果; "
        f"coverage/图 {main.coverage_per_image.min():.4f}…{main.coverage_per_image.max():.4f}, "
        f"quality/图 {main.quality_per_image.min():.6f}…{main.quality_per_image.max():.6f}")
    say("  (高/低都不作结论 —— 这是小样本脚手架上的数, 覆盖与质量随预算怎么走只是观察)")
    say()

    return {"label_info": label_info, "labels_path": str(labels_root / "labels.parquet"),
            "labels_matrix": matrix, "labels_image_ids": label_ids,
            "labels_positive_fraction": positive,
            "features": features.filter(like="f").to_numpy(np.float64),
            "train_info": train_info, "state_dict": state, "temperature": temperature,
            # 只收数值项 —— calib_info 里还混着 status 字符串和 bounds 列表。
            "calibration": {key: jnum(value) for key, value in calib_info.items()
                            if isinstance(value, (int, float, np.integer, np.floating))
                            and not isinstance(value, bool)},
            "allocation": allocation, "allocation_all_zero": tie, "eval_info": eval_info,
            "eval_signature": {
                "conditions": int(len(main)), "per_image_rows": int(len(per_image)),
                "coverage_per_image": {f"{row.method}|{int(row.budget)}": jnum(row.coverage_per_image)
                                       for row in main.itertuples()},
                "quality_per_image": {f"{row.method}|{int(row.budget)}": jnum(row.quality_per_image)
                                      for row in main.itertuples()},
                "output_records": {f"{row.method}|{int(row.budget)}": int(row.output_records)
                                   for row in main.itertuples()}}}


# ── 7. 重复运行 ──────────────────────────────────────────────────────────
def compare_runs(run1: dict, run2: dict, section1: dict, section2: dict) -> dict:
    join1, join2 = section1["join"], section2["join"]
    say("## 7. 重复运行一致性 (同一输入跑两遍)")
    say()
    say("两遍是**同一份输入、同一个 seed**。分别写进不同输出目录 —— `command_*` 系列"
        "(`_ensure_new_output`) 会拒绝覆盖非空目录, 用同一路径根本跑不了第二遍。"
        "第 4 节的候选数与 join 数值在第二遍里也原样重算了一遍, 逐字段差记在 7.9。")
    say()

    same_shape = run1["labels_matrix"].shape == run2["labels_matrix"].shape
    require("7.1 两遍标签矩阵 shape 相同 (选集一致)", same_shape,
            f"{run1['labels_matrix'].shape} vs {run2['labels_matrix'].shape}")
    label_diff = float(np.max(np.abs(run1["labels_matrix"] - run2["labels_matrix"])))
    label_same = bool(np.array_equal(run1["labels_matrix"], run2["labels_matrix"])
                      and run1["labels_image_ids"] == run2["labels_image_ids"])
    require("7.2 两遍标签数值差 == 0 且选集一致", label_diff == 0.0 and label_same,
            f"max|Δ|={label_diff:.3e}, 选集逐元素一致={label_same}")

    feature_diff = float(np.max(np.abs(run1["features"] - run2["features"]))) \
        if run1["features"].shape == run2["features"].shape else float("nan")
    require("7.3 两遍特征数值差 == 0", feature_diff == 0.0, f"max|Δ|={feature_diff:.3e}")

    require("7.4 两遍 checkpoint 参数名一致", run1["state_dict"].keys() == run2["state_dict"].keys(),
            f"{len(run1['state_dict'])} vs {len(run2['state_dict'])} 个张量")
    weight_diff = max(float(torch.max(torch.abs(run1["state_dict"][name]
                                                - run2["state_dict"][name])).item())
                      for name in run1["state_dict"])
    require("7.5 两遍模型权重数值差 == 0", weight_diff == 0.0, f"max|Δ|={weight_diff:.3e}")

    temperature_diff = abs(run1["temperature"] - run2["temperature"])
    require("7.6 两遍温度数值差 == 0", temperature_diff == 0.0, f"|Δ|={temperature_diff:.3e}")

    k1 = np.asarray(run1["allocation"]["allocate_m11"]["K"], dtype=np.int64)
    k2 = np.asarray(run2["allocation"]["allocate_m11"]["K"], dtype=np.int64)
    zero1 = np.asarray(run1["allocation_all_zero"]["allocate_m11"]["K"], dtype=np.int64)
    zero2 = np.asarray(run2["allocation_all_zero"]["allocate_m11"]["K"], dtype=np.int64)
    k_same = bool(np.array_equal(k1, k2))
    zero_same = bool(np.array_equal(zero1, zero2))
    require("7.7 两遍分配的 K_i 逐元素相同 (真值 + 全零并列)", k_same and zero_same,
            f"真值 {k1.shape}=={k2.shape}: {k_same}; 全零: {zero_same}")

    keys = sorted(run1["eval_signature"]["coverage_per_image"])
    coverage_diff = max(abs(run1["eval_signature"]["coverage_per_image"][key]
                            - run2["eval_signature"]["coverage_per_image"][key]) for key in keys)
    quality_diff = max(abs(run1["eval_signature"]["quality_per_image"][key]
                           - run2["eval_signature"]["quality_per_image"][key]) for key in keys)
    require("7.8 两遍评价数值差 == 0", coverage_diff == 0.0 and quality_diff == 0.0,
            f"max|Δcoverage|={coverage_diff:.3e}, max|Δquality|={quality_diff:.3e}")

    fields = ["records", "native_rows", "missing", "ambiguous", "shared_source_records",
              "unique_sources", "records_per_source_max", "score_min", "score_max"]
    join_diff = {label: {field: abs(float(join1[label][field]) - float(join2[label][field]))
                         for field in fields} for label in join1}
    counts_diff = {label: max(abs(int(value) - int(section2["counts_by_image"][label][image_id]))
                              for image_id, value in section1["counts_by_image"][label].items())
                   for label in section1["counts_by_image"]}
    worst_join = max(value for per_label in join_diff.values() for value in per_label.values())
    worst_counts = max(counts_diff.values())
    require("7.9 两遍逐图候选数 / join 关键数值差 == 0",
            worst_join == 0.0 and worst_counts == 0,
            json.dumps({"join_max": {label: max(per_label.values())
                                     for label, per_label in join_diff.items()},
                        "counts_max": counts_diff}, ensure_ascii=False))

    say("| 对象 | 两遍数值差 | 选集一致性 |")
    say("|---|---|---|")
    say(f"| 前缀标签增量矩阵 | {label_diff:.1e} | 逐元素相同 = {label_same} |")
    say(f"| 90 维特征矩阵 | {feature_diff:.1e} | — |")
    say(f"| checkpoint 全部参数 | {weight_diff:.1e} | — |")
    say(f"| 温度 | {temperature_diff:.1e} | — |")
    say(f"| DP K_i (真值合成 marginals) | 0 (整数) | 逐元素相同 = {k_same} |")
    say(f"| DP K_i (全零并列) | 0 (整数) | 逐元素相同 = {zero_same} |")
    say(f"| 评价 coverage / quality | {coverage_diff:.1e} / {quality_diff:.1e} | — |")
    say(f"| 逐图候选数 / join 关键值 | {max(float(worst_join), float(worst_counts)):.1e} | — |")
    say()
    say("> 全 0 是这里的**预期**, 不是「结果必须为正」的假设: 固定身份顺序 + 固定 seed 下"
        "同一份输入必须给出逐位相同的结果。任何非零都说明存在隐藏随机性或输入顺序依赖,"
        "那时这张表会如实报出非零值并让闭环失败。")
    say()

    return {"label_diff": label_diff, "label_selection_identical": label_same,
            "feature_diff": feature_diff, "weight_diff": weight_diff,
            "temperature_diff": temperature_diff, "allocation_identical": k_same,
            "allocation_all_zero_identical": zero_same, "coverage_diff": coverage_diff,
            "quality_diff": quality_diff, "join_diff": join_diff, "counts_diff": counts_diff}


def run_handoff_synthetic_tests() -> dict:
    u"""沿用交接包的公共合成测试: 匹配增广路径、并列、预算边界、one-to-many 保留、schema 身份。

    这些是**包自带**的用例, 不是本臂新写的自证; 只挑与本次闭环相关的那几条跑。
    """
    from lc_alloc.allocation import dp as dp_module
    from tests import test_synthetic

    names = [
        "test_augmenting_path",                              # 匹配增广路径
        "test_prefix_first_five_participate",                # 前缀 K=5 参与匹配, 增量为 0/1
        "test_same_source_multi_class_is_preserved",         # 一个 source 多类别不得被去重
        "test_dp_matches_brute_force_and_ties",              # 并列下与穷举结果一致
        "test_zero_tie_uses_lexicographically_smallest_k",   # 全零并列的词典序规则
        "test_score_fast_is_exact_for_nonincreasing_scores",  # 快速分配与 DP 目标一致
        "test_candidate_shortfall_is_explicit",              # 候选不足必须显式失败, 不许静默降级
        "test_native_schema_identity_is_enforced",           # 原生 schema 身份必须强校验
    ]
    result = unittest.TestResult()
    unittest.TestSuite([test_synthetic.SyntheticContractTests(name) for name in names]).run(result)
    failures = [f"{case.id().split('.')[-1]} -> {(text or '').strip().splitlines()[-1]}"
                for case, text in list(result.failures) + list(result.errors)]
    require("6.8 交接包公共合成测试全部通过", not failures,
            f"通过 {result.testsRun - len(failures)}/{result.testsRun}; 失败 {failures}")
    say("  - " + ", ".join(f"`{name}`" for name in names))
    dp_self = dp_module.self_test()
    require("6.9 DP 自检与穷举逐条对齐", str(dp_self.get("status")) == "PASS",
            f"tiny_instances={dp_self.get('tiny_instances')} "
            f"capacity_checks={dp_self.get('tiny_capacity_checks')} tie_break={dp_self.get('tie_break')}")
    say(f"  - `dp.self_test()`: {dp_self['tiny_instances']} 个小实例 × "
        f"{dp_self['tiny_capacity_checks']} 次容量校验全与穷举对齐; "
        f"并列规则 = {dp_self['tie_break']}")
    say()
    return {"unittest_total": result.testsRun, "unittest_failed": len(failures),
            "unittest_names": names, "dp_self_test": dp_self}


def main() -> int:
    ap = argparse.ArgumentParser(description="M11 小样本闭环 QA (Faster R-CNN 臂)")
    ap.add_argument("--prepared", default=str(M11_ROOT / "prepared"))
    ap.add_argument("--out", default=str(M11_ROOT / "qa"))
    ap.add_argument("--work", default=str(M11_ROOT / "qa" / "closed_loop_work"))
    ap.add_argument("--train-images", type=int, default=8)
    ap.add_argument("--support-images", type=int, default=2)
    ap.add_argument("--dev-group-index", type=int, default=0)
    ap.add_argument("--seed", type=int, default=730101)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    prepared = Path(args.prepared)
    out_dir = Path(args.out)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    work_root = Path(args.work) / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    report: dict = {"status": None, "failure": None, "stamp": stamp,
                    "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "prepared": str(prepared), "work_root": str(work_root),
                    "arguments": dict(vars(args)),
                    "checks": CHECKS, "sections": {}}

    try:
        say("# Faster R-CNN 臂 · M11 小样本闭环 QA")
        say()
        say(f"生成时间 {report['generated_at']} · prepared = `{prepared}` · work = `{work_root}`")
        say()
        say("## 0. 输入身份")
        say()
        required = ["candidates_train.parquet", "native_train.npz", "gt_train_normalized.json",
                    "roles.csv", "candidates_dev.parquet", "native_dev.npz",
                    "gt_dev_normalized.json", "groups_dev.csv"]
        missing_files = [name for name in required if not (prepared / name).is_file()]
        require("0.1 prepared 输入齐备 (缺就先跑 prepare_inputs.py)", not missing_files,
                f"缺少 {missing_files}" if missing_files else f"{len(required)} 个文件全在")

        roles = pd.read_csv(prepared / "roles.csv")
        groups = pd.read_csv(prepared / "groups_dev.csv")
        train_sizes = gt_image_sizes(prepared / "gt_train_normalized.json")
        dev_sizes = gt_image_sizes(prepared / "gt_dev_normalized.json")
        gt_tr = load_normalized_gt(prepared / "gt_train_normalized.json")
        gt_dv = load_normalized_gt(prepared / "gt_dev_normalized.json")
        cand_tr = read_candidates(prepared / "candidates_train.parquet")
        cand_dv = read_candidates(prepared / "candidates_dev.parquet")
        native_tr = read_native(prepared / "native_train.npz", expected_schema_id=NATIVE_SCHEMA_ID)
        native_dv = read_native(prepared / "native_dev.npz", expected_schema_id=NATIVE_SCHEMA_ID)
        native_extra = {}
        with np.load(prepared / "native_train.npz", allow_pickle=False) as raw:
            for name in raw.files:
                if name not in {"image_ids", "source_ids", "native_vectors", "class_signals",
                                "native_schema_id"}:
                    native_extra[name] = np.asarray(raw[name])
        require("0.2 原生 schema 身份与覆盖层声明一致",
                native_tr.native_schema_id == native_dv.native_schema_id == NATIVE_SCHEMA_ID,
                f"{native_tr.native_schema_id!r} / {native_dv.native_schema_id!r}")
        say(f"- native schema: `{native_tr.native_schema_id}`")
        say(f"- TRAIN 候选 {len(cand_tr)} 条 / {cand_tr.image_id.nunique()} 图; "
            f"DEV 候选 {len(cand_dv)} 条 / {cand_dv.image_id.nunique()} 图")
        say()

        sample = select_sample(cand_tr, cand_dv, roles, groups, args.train_images,
                               args.support_images, args.dev_group_index)
        sample_ids = sample["train_ids"] + sample["support"]["EARLY_STOP"] + sample["support"]["CALIBRATION"]
        roles_sample = roles.loc[roles.image_id.astype(str).isin(sample_ids), ["image_id", "role"]].copy()
        roles_sample["image_id"] = roles_sample.image_id.astype(str)
        require("1.7 闭环样本覆盖 FIT / EARLY_STOP / CALIBRATION 三个角色",
                set(roles_sample.role.astype(str)) >= {"FIT", "EARLY_STOP", "CALIBRATION"},
                f"roles={roles_sample.role.value_counts().to_dict()}")
        cand_sample = cand_tr[cand_tr.image_id.astype(str).isin(sample_ids)].copy()
        cand_dev_group = cand_dv[cand_dv.image_id.astype(str).isin(sample["dev_ids"])].copy()
        say()

        sample_dir = work_root / "sample"
        sample_dir.mkdir(parents=True, exist_ok=True)
        paths = {"sample_candidates": sample_dir / "candidates_train_sample.parquet",
                 "sample_roles": sample_dir / "roles_sample.csv",
                 "candidates_dev_group": sample_dir / "candidates_dev_group.parquet"}
        cand_sample.to_parquet(paths["sample_candidates"], index=False)
        roles_sample.to_csv(paths["sample_roles"], index=False)
        cand_dev_group.to_parquet(paths["candidates_dev_group"], index=False)

        report["sample"] = {
            "train_ids": sample["train_ids"], "support": sample["support"],
            "group_id": sample["group_id"], "group_index": args.dev_group_index,
            "groups_total": sample["group_ids_total"], "dev_ids": sample["dev_ids"],
            "role_counts": {key: int(value) for key, value in
                            roles_sample.role.value_counts().items()},
            "train_records": int(len(cand_sample)), "dev_records": int(len(cand_dev_group)),
        }

        all_records = pd.concat([cand_sample, cand_dev_group], ignore_index=True)
        all_sizes = {**{image_id: train_sizes[image_id] for image_id in sample_ids},
                     **{image_id: dev_sizes[image_id] for image_id in sample["dev_ids"]}}
        min_size, max_size, resize_source = parse_declared_resize()

        report["sections"]["coordinates"] = section_coordinates(
            all_records, all_sizes, min_size, max_size, resize_source, native_extra)
        report["sections"]["classes"] = section_classes(all_records, gt_tr, gt_dv,
                                                        sample_ids + sample["dev_ids"])
        counts_join_1 = section_counts_and_join(cand_sample, cand_dev_group, native_tr, native_dv,
                                                sample_ids, sample["dev_ids"])
        report["sections"]["counts_join"] = counts_join_1

        say("---")
        say()
        say("## 5. 标签增量 (交接包 `command_make_labels`)")
        say()
        say("标签是**前缀匹配计数的一阶差分** `m(k) - m(k-1)` (k=6..50), 每个 IoU 阈值一列, "
            "共 10 列 —— 与参照臂同一实现、同一冻结语义, 这里只调不改。")
        say()

        say("---")
        say()
        say("## 6. 训练一步 + 温度 + DP + 评价入口")
        say()
        fit_ids = sample["train_ids"]
        counts, weights = compute_class_weights(gt_tr, fit_ids)
        class_weights_path = sample_dir / "class_weights.json"
        class_weights_path.write_text(json.dumps(
            {"counts": counts.tolist(), "weights": weights.tolist(),
             "scope": "closed-loop engineering only; counts from the FIT sample"},
            ensure_ascii=False, indent=2), encoding="utf-8")
        say(f"- 类别权重用本样本 FIT {len(fit_ids)} 张的 GT 计数: {counts.tolist()} → "
            f"weights {[round(float(value), 4) for value in weights]}")
        say()

        ctx = {"cand_sample": cand_sample, "native_tr": native_tr, "roles_sample": roles_sample,
               "fit_ids": fit_ids, "dev_ids": sample["dev_ids"], "gt_train": prepared / "gt_train_normalized.json",
               "gt_dev": prepared / "gt_dev_normalized.json", "cand_dev_group": cand_dev_group,
               "class_weights": class_weights_path, **paths}

        run1 = run_closure("run1", work_root / "run1", ctx, args.seed, args.device)

        say("---")
        say()
        run2 = run_closure("run2", work_root / "run2", ctx, args.seed, args.device)
        # 第二遍只重算数值 (verify=False): 不与首测重复登记同名硬检查, 差异由 7.9 兜底。
        counts_join_2 = section_counts_and_join(cand_sample, cand_dev_group, native_tr, native_dv,
                                                sample_ids, sample["dev_ids"], verify=False)

        say("---")
        say()
        say("### 交接包公共合成测试与 DP 自检 (检查 6.8 / 6.9)")
        say()
        report["sections"]["handoff_tests"] = run_handoff_synthetic_tests()

        say("---")
        say()
        report["sections"]["repeat"] = compare_runs(run1, run2, counts_join_1, counts_join_2)
        report["sections"]["labels"] = {
            "rows": int(run1["labels_matrix"].shape[0] * 45),
            "images": len(run1["labels_image_ids"]),
            "shape": list(run1["labels_matrix"].shape),
            "positive_fraction": run1["labels_positive_fraction"],
            "positive_labels": int(run1["labels_matrix"].sum()),
            "image_ids": run1["labels_image_ids"],
            "command_result": run1["label_info"],
        }
        report["sections"]["train"] = {
            "train_info": run1["train_info"], "temperature": jnum(run1["temperature"]),
            "calibration": run1["calibration"], "allocation": run1["allocation"],
            "allocation_all_zero": run1["allocation_all_zero"],
            "eval_signature": run1["eval_signature"], "eval_info": run1["eval_info"],
        }

        report["status"] = "CLOSED_LOOP_PASS"
        say("---")
        say()
        say("## 8. 结论")
        say()
        say(f"通过 **{sum(1 for item in CHECKS if item['pass'])} / {len(CHECKS)}** 项硬检查。"
            f"闭环覆盖: 原图/网络坐标变换、类别映射、候选数、record→native 连接、数值有限性、"
            f"标签增量、训练一步与保存加载、温度、DP(并列与预算边界)、评价入口、重复运行一致性。")
        say()
        say("所有产物均为小样本工程验证用, 不构成任何科学结论;"
            "下一步是全量导出 + 全量训练, 不再追加重复的「准备审计」。")
        say()
        say("CLOSED_LOOP_PASS")
    except ClosedLoopFailure as exc:
        report["status"] = "CLOSED_LOOP_FAIL"
        report["failure"] = str(exc)
        say()
        say("## 失败")
        say()
        say(f"硬检查未通过: `{exc}`")
        say()
        say(f"CLOSED_LOOP_FAIL: {exc}")
    except Exception as exc:  # noqa: BLE001 —— 未预期异常同样是失败, 不许变成"没跑"
        report["status"] = "CLOSED_LOOP_FAIL"
        report["failure"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
        say()
        say("## 失败")
        say()
        say(f"未预期异常 `{type(exc).__name__}: {exc}`")
        say()
        say("```")
        for line in report["traceback"].strip().splitlines()[-14:]:
            say(line)
        say("```")
        say()
        say(f"CLOSED_LOOP_FAIL: {type(exc).__name__}: {exc}")

    report["checks"] = CHECKS
    report["checks_total"] = len(CHECKS)
    report["checks_passed"] = sum(1 for item in CHECKS if item["pass"])
    md_path = out_dir / "qa_closed_loop.md"
    md_path.write_text("\n".join(OUT) + "\n", encoding="utf-8")
    (out_dir / "qa_closed_loop.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\n[qa] -> {md_path}")
    return 0 if report["status"] == "CLOSED_LOOP_PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
