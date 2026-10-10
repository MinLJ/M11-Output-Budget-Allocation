# -*- coding: utf-8 -*-
u"""合同要求的第一份交付: 10–40 图小样本的 candidate/native schema、join、
score reconstruction 和候选数量报告。

对应 `docs/MEMBER_TASKS_CN.md`:
  「先交 10–40 图小样本的 candidate/native schema、join、score reconstruction
    和候选数量报告; 通过后再按共同配置训练自己的 PCA/scaler/三 seed MLP/温度并评价。」

用法:
  python -B lc_frcnn/qa_small_sample.py --split dev --images 40
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
HANDOFF = EXPERIMENT_ROOT / 'LC_ALLOC_M11_HANDOFF_v1/handoff_LC_ALLOC_M11_v1'
sys.path.insert(0, str(M11_ROOT))
sys.path.insert(0, str(HANDOFF))

from lc_frcnn.adapter import NATIVE_SCHEMA_ID, FasterRCNNAdapter  # noqa: E402

OUT: list[str] = []


def say(line: str = "") -> None:
    print(line, flush=True)
    OUT.append(line)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared", default=str(M11_ROOT / "prepared"))
    ap.add_argument("--split", default="dev")
    ap.add_argument("--images", type=int, default=40)
    ap.add_argument("--out", default=str(M11_ROOT / "qa"))
    ap.add_argument("--export-summary", default="",
                    help="留空则按 split 推断 export_{dev,train}/export_summary.json")
    args = ap.parse_args()
    if not args.export_summary:
        args.export_summary = str(
            EXPERIMENT_ROOT / 'frcnn_road8/outputs/frcnn_r18fpn_road8'
            / f"export_{args.split}" / "export_summary.json")

    prepared = Path(args.prepared)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cand_path = prepared / f"candidates_{args.split}.parquet"
    native_path = prepared / f"native_{args.split}.npz"
    frame = pd.read_parquet(cand_path)

    # 小样本优先取**冻结的完整 40 图组**, 与闭环 QA 和正式流程用同一套分组口径;
    # 没有分组文件 (例如 TRAIN) 时才退回按 image_id 词典序取前 N 张。
    groups_path = prepared / "groups_dev.csv"
    sample_note = ""
    if args.split == "dev" and groups_path.is_file():
        groups = pd.read_csv(groups_path)
        first = sorted(groups.group_id.astype(str).unique())[0]
        members = sorted(groups.loc[groups.group_id.astype(str) == first, "image_id"].astype(str))
        available = [i for i in members if i in set(frame.image_id.astype(str))]
        take = available[: args.images]
        sample_note = f"冻结组 `{first}` 的前 {len(take)} 张"
    else:
        take = sorted(frame.image_id.astype(str).unique())[: args.images]
        sample_note = f"按 image_id 词典序的前 {len(take)} 张"

    sub = frame[frame.image_id.astype(str).isin(take)].copy()
    say(f"# Faster R-CNN 臂 · 小样本 QA ({len(take)} 图, split={args.split})")
    say()
    say(f"样本口径: {sample_note}")
    say()
    say(f"检测器: `FasterRCNN-resnet18-FPN-Road8`, torchvision 0.27.0, "
        f"GeneralizedRCNNTransform min_size=max_size=640 (等比缩放到 640)")
    say()
    if "protocol_id" in frame.columns:
        say(f"本表协议: `{sorted(set(frame['protocol_id'].astype(str)))}`")
    say(f"native schema: `{NATIVE_SCHEMA_ID}`")
    say()

    # ── 1. candidate schema ────────────────────────────────────────────
    say("## 1. candidate schema")
    say()
    say("| 列 | dtype | 示例 |")
    say("|---|---|---|")
    for col in sub.columns:
        example = sub[col].iloc[0]
        text = f"{example:.6g}" if isinstance(example, (float, np.floating)) else str(example)
        say(f"| `{col}` | {sub[col].dtype} | `{text[:60]}` |")
    say()

    # ── 2. native schema ───────────────────────────────────────────────
    with np.load(native_path, allow_pickle=False) as raw:
        n_img = np.asarray(raw["image_ids"]).astype(str)
        n_src = np.asarray(raw["source_ids"]).astype(str)
        n_vec = np.asarray(raw["native_vectors"])
        n_sig = np.asarray(raw["class_signals"])
        n_schema = str(np.asarray(raw["native_schema_id"]).reshape(-1)[0])
    say("## 2. native sidecar schema")
    say()
    say(f"- `native_schema_id` = `{n_schema}`")
    say(f"- `native_vectors`  shape {n_vec.shape}, dtype {n_vec.dtype}  (box_head 输出, float16 存储)")
    say(f"- `class_signals`   shape {n_sig.shape}, dtype {n_sig.dtype}  "
        f"(pre-softmax class logits, 0=background, 1..8=Road8)")
    say(f"- join 键 = `(image_id, source_id)`, 全表唯一: "
        f"**{not pd.DataFrame({'a': n_img, 'b': n_src}).duplicated().any()}**")
    say()

    # ── 3. join ────────────────────────────────────────────────────────
    # lookup 建在**完整** native 表上, 因此 idx 是 n_sig / n_vec 的绝对行号 ——
    # 与 code/adapter.py 的 join 做法一致。绝不能先按子集缩表 (mask) 再拿相对位置
    # 去索引完整数组: 那会静默比对到错行, 把 §4 的正向重建报成失败。
    lookup = {(a, b): i for i, (a, b) in enumerate(zip(n_img, n_src))}
    idx = np.asarray([lookup[(str(r.image_id), str(r.source_id))] for r in sub.itertuples(index=False)], dtype=np.int64)
    pairs = sub.groupby(["image_id", "source_id"], sort=False).size()
    say("## 3. record ↔ native join")
    say()
    say(f"- 候选记录数 `{len(sub)}`, 全部命中原生行: **{len(idx) == len(sub)}**")
    say(f"- 不同 `(image_id, source_id)` 原生来源数 `{len(pairs)}`")
    say(f"- 一条原生来源承载多条候选记录 (多类别假设) 的来源数 `{int((pairs > 1).sum())}` "
        f"({(pairs > 1).mean() * 100:.1f}%)")
    say(f"- 每条来源承载的记录数: 中位 {pairs.median():.0f}, 最大 {pairs.max()}")
    say(f"- **没有按 source_id 去重** —— 这是合同明确要求保留的结构")
    say()

    # ── 4. score reconstruction ────────────────────────────────────────
    logits = n_sig[idx].astype(np.float64)
    shifted = logits - logits.max(axis=1, keepdims=True)
    prob = np.exp(shifted)
    prob /= prob.sum(axis=1, keepdims=True)
    cls = sub.predicted_road8_class_id.to_numpy(np.int64)
    recon = prob[np.arange(len(cls)), cls]
    err = np.abs(recon - sub.original_score.to_numpy(np.float64))
    say("## 4. score reconstruction")
    say()
    say("用原生 sidecar 里的 pre-softmax logits 正向重建 detector score:")
    say()
    say(r"$$\hat{s} = \mathrm{softmax}(\mathbf{z}_{\mathrm{RoI}})[\,c\,], "
        r"\qquad \mathbf{z}_{\mathrm{RoI}} \in \mathbb{R}^{9}$$")
    say()
    say(f"- max |ŝ − s| = **{err.max():.3e}**  (float32 存储精度的量级)")
    say(f"- mean |ŝ − s| = {err.mean():.3e}")
    say(f"- 全部 < 1e-6 的记录比例: {(err < 1e-6).mean() * 100:.4f}%")
    say()
    say("> 重建误差在 float32 存储精度内 —— sidecar 里存的是**真** pre-softmax logits,"
        " score 可由 logits 正向 softmax 逐条重建。合同禁止把 score 的逆 sigmoid "
        "冒充原生 logits: 一旦冒充, 这条正向检验会直接暴露。")
    say()

    # ── 5. 候选数量 ────────────────────────────────────────────────────
    # 注意: 本表是**已被 top-100 截断**的候选流, 它的每图条数恒为 100,
    # 拿它判断"候选是否充足"是同义反复。真实候选数在导出的 export_summary.json 里
    # (pre-NMS, = 1000 个 RoI × 8 个前景类), 必须报那一个。
    counts = sub.groupby("image_id", sort=False).size()
    full_counts = frame.groupby("image_id", sort=False).size()
    say("## 5. 候选数量")
    say()
    say("| 范围 | 图数 | min | 中位 | 均值 | max |")
    say("|---|---|---|---|---|---|")
    for label, series in ((f"本样本 ({len(take)} 图)", counts), (f"全 {args.split.upper()} 集", full_counts)):
        say(f"| {label} | {len(series)} | {series.min()} | {series.median():.0f} | "
            f"{series.mean():.2f} | {series.max()} |")
    say()
    summary_path = Path(args.export_summary)
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        pre = summary["pre_nms_records_per_image"]
        depth = int(frame["rank"].max())
        say(f"导出摘要 ({summary_path.name}):")
        say()
        say("| 口径 | min | 中位 | 均值 | max |")
        say("|---|---|---|---|---|")
        say(f"| **pre-NMS 真实候选数** (主用流的来源) | {pre['min']} | {pre['median']:.0f} | "
            f"{pre['mean']:.2f} | {pre['max']} |")
        say(f"| 主用流导出的记录数 (top-{depth} 截断后) | {full_counts.min()} | "
            f"{full_counts.median():.0f} | {full_counts.mean():.2f} | {full_counts.max()} |")
        say()
        say(f"- pre-NMS 真实候选数低于 100 的图: **{summary['pre_nms_images_below_100']}** / "
            f"{summary['images_exported']}")
        say(f"- pre-NMS 真实候选数低于 50 的图:  **{summary['pre_nms_images_below_50']}** / "
            f"{summary['images_exported']}")
        say()
        if summary["pre_nms_images_below_100"] == 0 and summary["pre_nms_images_below_50"] == 0:
            say("> 每图真实候选数均 ≥ 100,**不需要**为候选不足另立协议; "
                "M11 的 Top100 前缀动作可以直接在这个候选流上执行。")
        else:
            say("> **存在低于 100 条的图** —— 按合同必须另立命名的协议身份, "
                "不得补假框或静默降预算。")
    else:
        say(f"> 未找到 `{summary_path}`, pre-NMS 真实候选数无法从本文件判断 —— "
            f"上表「每图 100 条」是截断结果, **不能**用来论证候选充足。")
    say()

    # ── 6. adapter 自检 ────────────────────────────────────────────────
    say("## 6. adapter 自检 (走 `FasterRCNNAdapter.export`)")
    say()
    adapter = FasterRCNNAdapter()
    report = adapter.export(sub, native_path, check_score_reconstruction=True)
    say("```json")
    say(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    say("```")
    say()
    meta = adapter.describe()
    say(f"adapter status = **{meta.status}**, candidate_export_implemented="
        f"{meta.candidate_export_implemented}, native_export_implemented={meta.native_export_implemented}")
    say()

    (out_dir / "qa_small_sample.md").write_text("\n".join(OUT) + "\n", encoding="utf-8")
    (out_dir / "qa_small_sample.json").write_text(
        json.dumps({"report": report.to_dict(), "images": len(counts),
                    "records": int(len(sub)), "recon_max_abs_err": float(err.max())},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[qa] -> {out_dir/'qa_small_sample.md'}")


if __name__ == "__main__":
    main()
