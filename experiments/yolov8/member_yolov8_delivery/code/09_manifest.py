# -*- coding: utf-8 -*-
"""09 — write manifest.csv for the assembled delivery package.

Guideline §08 requires a manifest so the receiver can verify every file.  The
manifest lists relative path, size, SHA256 and a one-line purpose.  It excludes
itself (a file cannot carry its own hash).

Run last, after 08.  Purely descriptive: it reads the assembled tree and hashes
it; it never writes into results/ or assets/.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

REL = Path("yolo交付物") / "member_yolo_delivery"


def _find_deliv(start: Path) -> Path:
    """Locate the package whether this script sits next to it or inside code/."""
    for base in (start, *start.parents):
        cand = base / REL
        if cand.is_dir():
            return cand
    raise SystemExit(f"member_yolo_delivery not found above {start}")


BASE = Path(__file__).resolve().parent
DELIV = _find_deliv(BASE)

PURPOSE = {
    "README_CN.md": "交付说明：环境、入口、边界、复现顺序",
    # --- assets ---
    "assets/asset.json": "资产清单（可经 M11 契约 loader 校验 SHA/维度）",
    "assets/class_weights.json": "类别权重（FIT 计数）",
    "assets/pca32.joblib": "FIT-only PCA（32 维）",
    "assets/scaler.joblib": "FIT-only StandardScaler",
    "assets/models/marginal_mlp_seed_830101.pt": "种子 830101 分配器权重",
    "assets/models/marginal_mlp_seed_830102.pt": "种子 830102 分配器权重",
    "assets/models/marginal_mlp_seed_830103.pt": "种子 830103 分配器权重",
    "assets/models/temperature_seed_830101.json": "种子 830101 温度校准",
    "assets/models/temperature_seed_830102.json": "种子 830102 温度校准",
    "assets/models/temperature_seed_830103.json": "种子 830103 温度校准",
    "assets/training_history_seed_830101.csv": "种子 830101 完整训练历史",
    "assets/training_history_seed_830102.csv": "种子 830102 完整训练历史",
    "assets/training_history_seed_830103.csv": "种子 830103 完整训练历史",
    # --- code ---
    "code/01_export.py": "候选/native 导出入口",
    "code/02_prepare.py": "GT 转换 + 角色/组划分",
    "code/03_train.py": "PCA/scaler/3 种子训练与温度校准",
    "code/04_allocate_eval.py": "分配（M11/S_ADAPT/S_FIXED）+ 评价 + bootstrap",
    "code/05_ap_ar.py": "标准 COCO AP/AP50/AP75/AR100（冻结选集，不重新求解）",
    "code/06_post_nms_counts.py": "NMS 后条数计数（后处理补记录）",
    "code/07_predicted_utility.py": "逐槽预测效用重放（k=6..50）",
    "code/08_build_delivery.py": "本交付包组装脚本",
    "code/09_manifest.py": "本清单生成脚本",
    "code/audit_candidates.py": "候选可用性核查（每图候选数）",
    "code/m11_yolo/__init__.py": "适配器包入口",
    "code/m11_yolo/adapter.py": "YOLOAdapter：候选/native 导出（DetectorAdapter 契约）",
    "code/m11_yolo/config.py": "冻结的检测器/资产身份常量",
    "code/m11_yolo/features.py": "YOLO native → 90 维特征",
    "code/m11_yolo/nms.py": "规范 NMS 实现（05/06 共用）",
    # --- configs ---
    "configs/allocation_weights.json": "allocation_weight_id 类别权重（与上同源，按规范分列）",
    "configs/evaluation_weights.json": "evaluation_weight_id 类别权重",
    "configs/evaluator_config.json": "Coverage/QUALITY/COCO/bootstrap 评价口径",
    "configs/groups.csv": "DEV 50 组 × 40 图分组清单",
    "configs/roles.csv": "FIT/EARLY_STOP/CALIBRATION 角色清单",
    "configs/roles_raw.csv": "角色清单原始副本",
    "configs/yolo_adapter.json": "检测器冻结配置（预处理/候选导出点/分数语义）",
    # --- references ---
    "references/candidate_native_index.csv": "候选/native/GT 的包内路径·大小·SHA256·原始出处索引",
    "references/README_CN.md": "候选/native 文件说明",
    "references/data/DEV/candidates.parquet": "DEV 候选（重放 04–07 评价必需）",
    "references/data/DEV/native.npz": "DEV native 向量",
    "references/data/DEV/export_log.json": "DEV 导出日志",
    "references/data/TRAIN/candidates.parquet": "TRAIN 候选（重跑 01–03 训练）",
    "references/data/TRAIN/native.npz": "TRAIN native 向量",
    "references/data/TRAIN/export_log.json": "TRAIN 导出日志",
    "references/data/prep/DEV_gt.json": "DEV GT",
    "references/data/prep/TRAIN_gt.json": "TRAIN GT",
    # --- results ---
    "results/allocations.parquet": "每条件每图的 group_id/K_i/candidate_asset_id",
    "results/ap_ar_per_class.csv": "逐类 AP/AP50/AP75/AR100",
    "results/ap_ar_prefix.csv": "动作空间 A：冻结前缀（无 NMS）AP/AR",
    "results/ap_ar_reference_nms.csv": "动作空间 B：前缀 + NMS0.70 参考 AP/AR",
    "results/bootstrap_results.csv": "配对 bootstrap（逐种子 + 三种子汇总）",
    "results/bootstrap_core_summary.csv": "核心汇总：K10/15/20 等权平均的配对 bootstrap（逐种子 + 三种子汇总）",
    "results/class_results.csv": "200 行逐类结果（GT support/matched/recall/class AP-AR）",
    "results/main_results.csv": "25 条件主结果（Coverage/QUALITY/AP/AP50/AP75/AR100）",
    "results/paired_group_bootstrap.csv": "配对 bootstrap 原始输出（04 生成）",
    "results/per_group_results.csv": "1250 行逐组结果（供配对统计）",
    "results/per_image_results.csv": "50000 行逐图结果（K_i/coverage/quality）",
    "results/post_nms_counts.csv": "逐图 NMS 前后条数/删减数/存活 record IDs",
    "results/post_nms_group_summary.csv": "逐组 NMS 汇总",
    "results/post_nms_overall.csv": "全体 NMS 均值/最小/最大 + 空输出图数",
    "results/predicted_utility.parquet": "k=6..50 的 logits/prob/delta_hat/Uhat（3 种子）",
    "results/predicted_utility_manifest.json": "predicted_utility 溯源信息",
    "results/selected_records.parquet": "1150000 条选集 record IDs（可按 rank 严格重建）",
}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    rows, missing = [], []
    for p in sorted(DELIV.rglob("*")):
        if not p.is_file() or p.name == "manifest.csv":
            continue
        rel = p.relative_to(DELIV).as_posix()
        purpose = PURPOSE.get(rel)
        if purpose is None:
            missing.append(rel)
        rows.append({"relative_path": rel, "size_bytes": p.stat().st_size,
                     "sha256": sha256(p), "purpose": purpose or ""})
    df = pd.DataFrame(rows).sort_values("relative_path").reset_index(drop=True)
    df.to_csv(DELIV / "manifest.csv", index=False, encoding="utf-8")
    print(f"files: {len(df)}  without purpose: {missing}", flush=True)
    return len(df)


if __name__ == "__main__":
    main()
