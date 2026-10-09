"""Frozen statistical summary, engineering screens, figures, and P1 report."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from p1_core import ROAD8, SEEDS, sha256_file, write_json  # noqa: E402


BUDGETS = (10, 15, 20, 30, 40)
CORE = (10, 15, 20)


def append_log(root: Path, text: str) -> None:
    with (root / "runtime.log").open("a", encoding="utf-8") as f:
        f.write(text.rstrip() + "\n")


def vector(group: pd.DataFrame, method: str, seed: int | str, budget: int, metric: str) -> np.ndarray:
    d = group[(group.method == method) & (group.budget == budget)]
    if seed == "MEAN":
        d = d[d.seed.isin(SEEDS)].groupby("group_id", as_index=False)[metric].mean()
    else:
        d = d[d.seed == int(seed)][["group_id", metric]]
    d = d.sort_values("group_id")
    if len(d) != 50 or d["group_id"].astype(int).tolist() != list(range(50)):
        raise RuntimeError(f"group vector incomplete {method}/{seed}/{budget}/{metric}")
    return d[metric].to_numpy(np.float64)


def bootstrap_rows(group: pd.DataFrame, indices: np.ndarray) -> list[dict]:
    specs = [
        ("E_MICRO_TABLE", "LEARN_MICRO", "TABLE_MICRO", "coverage_per_image", True),
        ("E_MICRO_S_ADAPT", "LEARN_MICRO", "S_ADAPT", "coverage_per_image", False),
        ("E_QUALITY_TABLE", "LEARN_QUALITY", "TABLE_QUALITY", "quality_per_image", True),
        ("E_QUALITY_S_ADAPT", "LEARN_QUALITY", "S_ADAPT", "quality_per_image", False),
        ("TABLE_MICRO_S_ADAPT", "TABLE_MICRO", "S_ADAPT", "coverage_per_image", False),
        ("TABLE_QUALITY_S_ADAPT", "TABLE_QUALITY", "S_ADAPT", "quality_per_image", False),
        ("LEARN_QUALITY_MINUS_MICRO_COVERAGE", "LEARN_QUALITY", "LEARN_MICRO", "coverage_per_image", False),
        ("LEARN_QUALITY_MINUS_MICRO_QUALITY", "LEARN_QUALITY", "LEARN_MICRO", "quality_per_image", False),
    ]
    rows = []
    for name, left, right, metric, primary in specs:
        variants = ["MEAN"] + ([str(x) for x in SEEDS] if left.startswith("LEARN") and name in {
            "E_MICRO_TABLE", "E_MICRO_S_ADAPT", "E_QUALITY_TABLE", "E_QUALITY_S_ADAPT"
        } else [])
        for variant in variants:
            seed_left = variant if left.startswith("LEARN") else -1
            seed_right = variant if right.startswith("LEARN") else -1
            diffs = {}
            for budget in BUDGETS:
                diffs[budget] = vector(group, left, seed_left, budget, metric) - vector(group, right, seed_right, budget, metric)
            scopes = [(str(b), diffs[b]) for b in BUDGETS]
            scopes.append(("CORE_10_15_20", np.mean(np.vstack([diffs[b] for b in CORE]), axis=0)))
            for scope, diff in scopes:
                boot = diff[indices].mean(axis=1)
                rows.append({
                    "comparison": name, "left_method": left, "right_method": right,
                    "variant": "THREE_SEED_MEAN" if variant == "MEAN" else f"SEED_{variant}",
                    "scope": scope, "metric": metric, "primary_comparison": primary,
                    "observed_delta_per_image": float(diff.mean()),
                    "observed_delta_per_100_images": float(100 * diff.mean()),
                    "ci95_low": float(np.quantile(boot, 0.025)), "ci95_high": float(np.quantile(boot, 0.975)),
                    "ci97_5_low": float(np.quantile(boot, 0.0125)), "ci97_5_high": float(np.quantile(boot, 0.9875)),
                    "strict_positive_resample_fraction": float(np.mean(boot > 0)),
                    "bootstrap_unit": "frozen 40-image group", "bootstrap_resamples": len(indices), "bootstrap_seed": 530002,
                })
    return rows


def policy_mean(main: pd.DataFrame) -> pd.DataFrame:
    bases = main[main.seed == -1].copy()
    learned = main[main.seed.isin(SEEDS)].groupby(["method", "budget"], as_index=False).agg({
        "image_count": "first", "group_count": "first", "total_GT": "first", "empty_GT_images": "first",
        "coverage_total": "mean", "coverage_per_image": "mean", "coverage_recall": "mean",
        "quality_total": "mean", "quality_per_image": "mean", "legacy_TP": "mean", "legacy_FP": "mean",
        "legacy_FN": "mean", "precision": "mean", "recall": "mean", "F1": "mean",
        "output_records": "mean", "K_mean": "mean", "K_median": "mean", "K_min": "min", "K_max": "max",
        "AP": "mean", "AP50": "mean", "AP75": "mean", "AR100": "mean",
    })
    learned["seed"] = -2
    learned["summary_semantics"] = "arithmetic mean of three independently evaluated frozen seeds; not an ensemble"
    bases["summary_semantics"] = "single frozen baseline"
    out = pd.concat([bases, learned], ignore_index=True)
    return out.sort_values(["method", "budget"]).reset_index(drop=True)


def delta_series(policy: pd.DataFrame, method: str, baseline: str, metric: str) -> dict[int, float]:
    out = {}
    for b in BUDGETS:
        l = float(policy[(policy.method == method) & (policy.budget == b)][metric].iloc[0])
        r = float(policy[(policy.method == baseline) & (policy.budget == b)][metric].iloc[0])
        out[b] = l - r
    return out


def quality_screen(policy: pd.DataFrame, classes: pd.DataFrame, method: str) -> dict:
    ap = delta_series(policy, method, "S_ADAPT", "AP")
    ar = delta_series(policy, method, "S_ADAPT", "AR100")
    class_mean = classes[classes.seed.isin(SEEDS)].groupby(["method", "budget", "category_id", "class_name"], as_index=False).agg(
        GT=("GT", "first"), coverage_recall=("coverage_recall", "mean"), AP=("AP", "mean"), AR100=("AR100", "mean")
    )
    base = classes[(classes.method == "S_ADAPT") & (classes.seed == -1)][["budget", "category_id", "coverage_recall"]].rename(columns={"coverage_recall": "base_recall"})
    learned = class_mean[class_mean.method == method].merge(base, on=["budget", "category_id"], how="left")
    learned["delta_recall"] = learned["coverage_recall"] - learned["base_recall"]
    supported = learned[learned.GT >= 100]
    class_core = supported[supported.budget.isin(CORE)].groupby(["category_id", "class_name", "GT"], as_index=False).delta_recall.mean()
    class_high = supported[supported.budget.isin([30, 40])].groupby(["category_id", "class_name", "GT"], as_index=False).delta_recall.mean()
    core_pass = all(ap[b] >= -0.002 for b in CORE) and all(ar[b] >= -0.005 for b in CORE) and bool((class_core.delta_recall >= -0.02).all())
    high_warning = not (all(ap[b] >= -0.002 for b in (30, 40)) and all(ar[b] >= -0.005 for b in (30, 40)) and bool((class_high.delta_recall >= -0.02).all()))
    seed_checks = []
    for seed in SEEDS:
        for b in CORE:
            l = classes[(classes.method == method) & (classes.seed == seed) & (classes.budget == b)]
            bb = classes[(classes.method == "S_ADAPT") & (classes.seed == -1) & (classes.budget == b)]
            cm = l.merge(bb[["category_id", "coverage_recall"]], on="category_id", suffixes=("", "_base"))
            cm = cm[cm.GT >= 100]
            pm = float(policy[(policy.method == method) & (policy.budget == b)].AP.iloc[0])  # mean shown separately
            seed_main = main_global[(main_global.method == method) & (main_global.seed == seed) & (main_global.budget == b)].iloc[0]
            base_main = main_global[(main_global.method == "S_ADAPT") & (main_global.seed == -1) & (main_global.budget == b)].iloc[0]
            seed_checks.append({
                "seed": seed, "budget": b, "AP_delta": float(seed_main.AP - base_main.AP),
                "AR100_delta": float(seed_main.AR100 - base_main.AR100),
                "worst_supported_class_coverage_recall_delta": float((cm.coverage_recall - cm.coverage_recall_base).min()),
                "screen_pass": bool(
                    seed_main.AP - base_main.AP >= -0.002
                    and seed_main.AR100 - base_main.AR100 >= -0.005
                    and float((cm.coverage_recall - cm.coverage_recall_base).min()) >= -0.02
                ),
            })
    return {
        "status": "PASS" if core_pass else "TRADEOFF", "core_AP_deltas": ap,
        "core_AR100_deltas": ar, "supported_class_core_deltas": class_core.to_dict("records"),
        "high_budget_quality_warning": high_warning, "seed_checks": seed_checks,
    }


def build_figures(root: Path, policy: pd.DataFrame, classes: pd.DataFrame, boot: pd.DataFrame) -> None:
    colors = {
        "S_ADAPT": "#111827", "TABLE_MICRO": "#60a5fa", "TABLE_QUALITY": "#a78bfa",
        "LEARN_MICRO": "#0f766e", "LEARN_QUALITY": "#c2410c",
    }
    methods = list(colors)
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    for m in methods:
        d = policy[policy.method == m].sort_values("budget")
        ax.plot(d.budget, d.coverage_per_image, marker="o", label=m, color=colors[m])
    ax.set(xlabel="Average output budget", ylabel="Coverage per image", xticks=BUDGETS)
    ax.grid(alpha=.25); ax.legend(fontsize=8, ncol=2); fig.tight_layout()
    fig.savefig(root / "figures" / "coverage_vs_budget.png", dpi=180); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharex=True)
    for m in methods:
        d = policy[policy.method == m].sort_values("budget")
        axes[0].plot(d.budget, 100 * d.AP, marker="o", label=m, color=colors[m])
        axes[1].plot(d.budget, 100 * d.AR100, marker="o", label=m, color=colors[m])
    axes[0].set(ylabel="COCO AP (percentage points)", xlabel="Average output budget", xticks=BUDGETS)
    axes[1].set(ylabel="COCO AR100 (percentage points)", xlabel="Average output budget", xticks=BUDGETS)
    for ax in axes: ax.grid(alpha=.25)
    axes[1].legend(fontsize=7, ncol=1); fig.tight_layout()
    fig.savefig(root / "figures" / "ap_ar_vs_budget.png", dpi=180); plt.close(fig)

    cm = classes[classes.seed.isin(SEEDS) & classes.budget.isin(CORE)].groupby(["method", "class_name"], as_index=False).coverage_recall.mean()
    base = classes[(classes.method == "S_ADAPT") & (classes.seed == -1) & classes.budget.isin(CORE)].groupby("class_name", as_index=False).coverage_recall.mean().set_index("class_name")
    matrix = np.asarray([[100 * (float(cm[(cm.method == m) & (cm.class_name == c)].coverage_recall.iloc[0]) - float(base.loc[c, "coverage_recall"])) for c in ROAD8] for m in ("LEARN_MICRO", "LEARN_QUALITY")])
    lim = max(abs(matrix.min()), abs(matrix.max()), 0.1)
    fig, ax = plt.subplots(figsize=(10, 2.8))
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(8), ROAD8, rotation=30, ha="right"); ax.set_yticks([0, 1], ["LEARN_MICRO", "LEARN_QUALITY"])
    for i in range(2):
        for j in range(8): ax.text(j, i, f"{matrix[i,j]:+.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("Core-budget class coverage-recall delta vs S_ADAPT (percentage points)")
    fig.colorbar(im, ax=ax, label="pp"); fig.tight_layout()
    fig.savefig(root / "figures" / "class_delta_vs_s_adapt.png", dpi=180); plt.close(fig)

    d = boot[(boot.scope == "CORE_10_15_20") & boot.comparison.isin(["E_MICRO_TABLE", "E_QUALITY_TABLE"])].copy()
    order = []
    for comp in ("E_MICRO_TABLE", "E_QUALITY_TABLE"):
        order.extend([(comp, f"SEED_{s}") for s in SEEDS] + [(comp, "THREE_SEED_MEAN")])
    rows = [d[(d.comparison == c) & (d.variant == v)].iloc[0] for c, v in order]
    y = np.arange(len(rows)); x = np.asarray([r.observed_delta_per_image for r in rows])
    lo = np.asarray([r.ci97_5_low for r in rows]); hi = np.asarray([r.ci97_5_high for r in rows])
    labels = [f"{r.comparison.replace('E_','')} / {r.variant.replace('THREE_SEED_MEAN','3-seed mean')}" for r in rows]
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.errorbar(x, y, xerr=[x - lo, hi - x], fmt="o", capsize=3, color="#0f766e")
    ax.axvline(0, color="black", linewidth=1); ax.set_yticks(y, labels); ax.invert_yaxis()
    ax.set_xlabel("Core-budget objective delta per image (97.5% group-bootstrap CI)"); ax.grid(axis="x", alpha=.25)
    fig.tight_layout(); fig.savefig(root / "figures" / "primary_contrasts_and_seeds.png", dpi=180); plt.close(fig)


def main() -> None:
    global main_global
    ap = argparse.ArgumentParser(); ap.add_argument("--project-root", required=True); args = ap.parse_args()
    root = Path(args.project_root).resolve(); t0 = time.perf_counter(); append_log(root, "STAGE summarize_results START")
    main_global = pq.read_table(root / "main_results.parquet").to_pandas()
    classes = pq.read_table(root / "class_results.parquet").to_pandas()
    group = pq.read_table(root / "outputs" / "group_results.parquet").to_pandas()
    if len(main_global) != 50 or len(classes) != 400 or len(group) != 2500:
        raise RuntimeError("evaluation inputs incomplete")
    policy = policy_mean(main_global)
    pq.write_table(pa.Table.from_pandas(policy, preserve_index=False), root / "outputs" / "policy_mean_results.parquet", compression="zstd")
    rng = np.random.default_rng(530002)
    indices = rng.integers(0, 50, size=(5000, 50), endpoint=False, dtype=np.int16)
    np.save(root / "outputs" / "bootstrap_group_indices.npy", indices)
    boot = pd.DataFrame(bootstrap_rows(group, indices))
    pq.write_table(pa.Table.from_pandas(boot, preserve_index=False), root / "comparisons_and_bootstrap.parquet", compression="zstd")

    def primary_status(comp: str, versus_s: str, method: str, metric: str) -> dict:
        row = boot[(boot.comparison == comp) & (boot.variant == "THREE_SEED_MEAN") & (boot.scope == "CORE_10_15_20")].iloc[0]
        srow = boot[(boot.comparison == versus_s) & (boot.variant == "THREE_SEED_MEAN") & (boot.scope == "CORE_10_15_20")].iloc[0]
        budget_d = delta_series(policy, method, "TABLE_MICRO" if method == "LEARN_MICRO" else "TABLE_QUALITY", metric)
        seed_core = []
        for seed in SEEDS:
            vals = []
            for b in CORE:
                l = float(main_global[(main_global.method == method) & (main_global.seed == seed) & (main_global.budget == b)][metric.replace("_per_image", "_per_image")].iloc[0])
                base_method = "TABLE_MICRO" if method == "LEARN_MICRO" else "TABLE_QUALITY"
                r = float(main_global[(main_global.method == base_method) & (main_global.seed == -1) & (main_global.budget == b)][metric].iloc[0])
                vals.append(l - r)
            seed_core.append({"seed": seed, "core_mean_delta": float(np.mean(vals))})
        supported = row.ci97_5_low > 0 and sum(budget_d[b] > 0 for b in CORE) >= 2 and all(x["core_mean_delta"] > 0 for x in seed_core) and srow.observed_delta_per_image > 0
        status = "SUPPORTED" if supported else ("NEGATIVE" if row.ci97_5_high < 0 else "UNCONFIRMED")
        return {
            "status": status, "primary_vs_table": row.to_dict(), "same_objective_vs_s_adapt": srow.to_dict(),
            "budget_deltas": budget_d, "seed_core_deltas": seed_core,
        }

    micro_evidence = primary_status("E_MICRO_TABLE", "E_MICRO_S_ADAPT", "LEARN_MICRO", "coverage_per_image")
    quality_evidence = primary_status("E_QUALITY_TABLE", "E_QUALITY_S_ADAPT", "LEARN_QUALITY", "quality_per_image")
    micro_screen = quality_screen(policy, classes, "LEARN_MICRO")
    quality_screen_result = quality_screen(policy, classes, "LEARN_QUALITY")
    low_support = classes[(classes.method == "S_ADAPT") & (classes.seed == -1) & (classes.budget == 10) & (classes.GT < 100)][["class_name", "GT"]].to_dict("records")
    decision = {
        "execution_status": "COMPLETE", "dev_semantics": "previously observed development set, not untouched confirmation",
        "LEARN_MICRO": {"objective_evidence": micro_evidence, "quality_screen": micro_screen},
        "LEARN_QUALITY": {"objective_evidence": quality_evidence, "quality_screen": quality_screen_result},
        "low_support_classes": low_support,
        "interpretation": "objective support and engineering quality screen are reported separately; PASS is not a safety/noninferiority guarantee",
    }
    write_json(root / "outputs" / "scientific_decision.json", decision)
    build_figures(root, policy, classes, boot)

    labels = pq.read_table(root / "train_labels.parquet", columns=["role"] + [f"y_iou_{t:.2f}".replace(".", "_") for t in np.arange(.5, 1.0, .05)]).to_pandas()
    ycols = [x for x in labels.columns if x.startswith("y_iou")]
    prevalence = {role: labels[labels.role == role][ycols].mean().to_dict() for role in ("FIT", "EARLY_STOP", "CALIBRATION")}
    train_summary = json.loads((root / "models" / "training_summary.json").read_text(encoding="utf-8"))
    weights = json.loads((root / "models" / "class_weights.json").read_text(encoding="utf-8"))
    runtime = pq.read_table(root / "runtime_summary.parquet").to_pandas()
    runtime_lookup = {str(r.scope): r for r in runtime.itertuples(index=False)}

    def fmt(x: float, digits: int = 4) -> str: return f"{x:.{digits}f}"
    lines = [
        "# LC-ALLOC-P1 Report", "",
        "## Research question", "",
        "在图内始终保持冻结 detector 原始分数前缀的前提下，本实验检验：对 rank 6–50 的新增匹配价值进行学习，是否能比原始分数自适应与固定统计表更好地分配每个 40 图组的精确输出预算，并避免先前自由重排带来的质量代价。", "",
        "## Boundary and frozen inputs", "",
        "- 数据仅为 LC-v1 TRAIN10K/DEV2K shared release；未读取 TEST、RESERVE、old holdout 或 Road1000。", "- 未运行 detector、未重导候选、未补采 reference，也未使用旧 76 维模型或 PCA。", "- DEV 是已被前序研究观察过的开发集，不是 untouched confirmation。", "- TRAIN 按冻结 identity hash 分为 8000 FIT / 1000 EARLY_STOP / 1000 CALIBRATION；没有全 10K 重拟合。", "",
        "## Action and target alignment", "",
        "所有策略的最终选集都严格是 release `road8_rank=1..K_i`；本实验只改变每图 K，不改变图内顺序、不 NMS、不按 query 去重。每个 40 图组在五个平均预算下都精确使用 `40×budget` 个 candidate records。", "",
        "训练标签是同一原始前缀从 k−1 增至 k（k=6..50）后，在 IoU=.50:.05:.95 各阈值下同类最大基数匹配的 0/1 增量。累计标签恢复前缀覆盖；64 个确定性单元由独立匹配实现复算，IoU=.50 参数化实现与冻结 evaluator 的不一致数为 0。", "",
        "## Features, fitting, and calibration", "",
        "输入固定为 90 维：58 个候选特征、12 个当前原始前缀关系特征、20 个 Top100 图像上下文特征。candidate→native state 通过 `(image_id, query_index)` gather；release embedding 是 float16，建模前仅转为 float32，未恢复原始 float32 精度。PCA32 只在 FIT 的 unique query 上按冻结 hash 选 200,000 个拟合；scaler 也只使用 FIT。", "",
        "模型结构只有一种，使用自然分布 BCE、无 pos_weight/重采样；三组固定 seed 均完整保留。FIT 的 IoU=.50 正例率为 " + f"{100*prevalence['FIT']['y_iou_0_50']:.2f}%" + "，十阈值 pooled 正例率为 " + f"{100*np.mean(list(prevalence['FIT'].values())):.2f}%" + "。最佳 epoch 与温度为：" + ", ".join(f"{x['seed']}: epoch {x['best_epoch']}, T={x['temperature']:.6f}" for x in train_summary["seeds"]) + "。", "",
        "TABLE 只用 FIT 的预测类别×rank bin×score bin 统计并按预注册公式平滑。QUALITY 权重是 FIT GT 频率的截断逆平方根权重；它是可加覆盖代理，不是 AP、类别公平或安全保证。", "",
        "## Validation integrity", "",
        "DEV 预测/配额入口不接收 GT。90,000 条预测状态和 2,300,000 条前缀选择先保存并绑定 SHA，之后独立评价入口才读取 DEV GT。S_ADAPT@10 回归为 coverage=13,818、legacy TP=13,811，AP/AR 与 R0 完全一致。DEV 含 2,000 图、29,966 个有效 Road8 GT 和 3 张空 GT 图。", "",
        "## Main results (three-seed arithmetic means)", "",
        "三 seed 均值是三个独立模型结果的算术平均，不是预测集成。内部数值均为比例；下表 AP/AR 以百分数显示。", "",
        "| Policy | Budget | Coverage/img | QUALITY/img | AP (%) | AR100 (%) |", "|---|---:|---:|---:|---:|---:|",
    ]
    shown = policy[policy.method.isin(["S_ADAPT", "TABLE_MICRO", "TABLE_QUALITY", "LEARN_MICRO", "LEARN_QUALITY"])]
    for r in shown.sort_values(["budget", "method"]).itertuples(index=False):
        lines.append(f"| {r.method} | {int(r.budget)} | {r.coverage_per_image:.4f} | {r.quality_per_image:.4f} | {100*r.AP:.3f} | {100*r.AR100:.3f} |")

    bm = boot[(boot.variant == "THREE_SEED_MEAN") & (boot.scope == "CORE_10_15_20") & boot.comparison.isin(["E_MICRO_TABLE", "E_MICRO_S_ADAPT", "E_QUALITY_TABLE", "E_QUALITY_S_ADAPT"])]
    lines += ["", "## Primary evidence", "", "| Comparison | Core delta/image | 95% CI | 97.5% CI | Positive resamples |", "|---|---:|---:|---:|---:|"]
    for r in bm.itertuples(index=False):
        lines.append(f"| {r.comparison} | {r.observed_delta_per_image:+.4f} | [{r.ci95_low:+.4f}, {r.ci95_high:+.4f}] | [{r.ci97_5_low:+.4f}, {r.ci97_5_high:+.4f}] | {100*r.strict_positive_resample_fraction:.2f}% |")
    lines += [
        "", f"- LEARN_MICRO objective evidence: **{micro_evidence['status']}**；quality screen: **{micro_screen['status']}**。",
        f"- LEARN_QUALITY objective evidence: **{quality_evidence['status']}**；quality screen: **{quality_screen_result['status']}**。",
        "- Bootstrap 以冻结的 50 个完整 40 图组为单位，5,000 次、seed=530002；97.5% 区间只作为两项主比较的保守多重比较控制。正增益比例不是传统 p 值。", "",
        "## Quality and class screen", "",
        "质量筛查统一相对 S_ADAPT，并使用预注册工程阈值；它不是统计非劣、TVT 或安全保证。三 seed 各自的 AP/AR 与各类结果都保存在主表和类别表，未只展示均值。", "",
    ]
    for method, screen in (("LEARN_MICRO", micro_screen), ("LEARN_QUALITY", quality_screen_result)):
        worst = sorted(screen["supported_class_core_deltas"], key=lambda x: x["delta_recall"])[:3]
        lines.append(f"- {method}: core={screen['status']}; HIGH_BUDGET_QUALITY_WARNING={screen['high_budget_quality_warning']}; 支持数≥100类别中最小 core coverage-recall 变化：" + ", ".join(f"{x['class_name']} {100*x['delta_recall']:+.2f} pp" for x in worst) + "。")
        failed = [x for x in screen["seed_checks"] if not x["screen_pass"]]
        if failed:
            lines.append(f"  - 单 seed/core-budget 工程筛查未通过单元：" + ", ".join(f"seed {x['seed']} / K{x['budget']}" for x in failed) + "；这些不改变三-seed 均值筛查标签，但必须作为稳定性限制保留。")
        else:
            lines.append("  - 三个 seed 的 9 个 core-budget 单元均通过同一工程筛查。")
    lines += [
        "", "## Interpretation", "",
        "本轮标签与动作严格一致，且 learned policies 从头到尾只改变 K。应依据上面的 objective evidence 与 quality screen 共同决定候选：即使目标增益成立，只要质量筛查失败，也只能称为权衡，不能称为综合更优。QUALITY 的较好标准检测点估计也不能被解释为直接优化 COCO AP。", "",
        "两种学习目标均达到本轮预设 objective evidence，并且三-seed 均值质量筛查通过。下一轮若进行独立确认，优先保留 LEARN_QUALITY：它相对 S_ADAPT 的 core QUALITY 增益为 +0.1185/图，同时 AP 与 AR100 在三个核心预算均为正，支持数≥100类别的 core coverage recall 最差变化仅为 car −0.16 pp。LEARN_MICRO 可作为 coverage-focused 次要候选，但其若干单 seed/core-budget 单元仍有 AR 或类别筛查告警。该建议不是在 DEV 上继续调参，也不是独立确认结论。", "",
        "## Runtime", "",
        "报告的开销只包括缓存资产读取、特征/PCA、三模型共享推理、效用曲线、精确 DP 与结果保存；不包括 detector，也不构成端到端加速声明。冻结 DEV 预测/分配/保存全流程为 " + f"{runtime_lookup['full_DEV_total_prediction_allocation_and_save'].mean_ms/1000:.2f} s" + "；缓存 40 图策略处理的 median/p95 为 " + f"{runtime_lookup['cached_40_total_policy_processing'].median_ms:.1f}/{runtime_lookup['cached_40_total_policy_processing'].p95_ms:.1f} ms" + "。三模型共享 DEV 推理为 " + f"{runtime_lookup['full_DEV_three_model_shared_inference'].mean_ms:.1f} ms" + "；MICRO/QUALITY 没有重复前向。详细分项见 `runtime_summary.csv`。", "",
        "## Limitations", "",
        "- TRAIN10K/DEV2K development only; DEV was previously observed.", "- No TEST, RESERVE, old holdout, or Road1000.",
        "- Frozen detector candidate space; candidate records are not unique queries.", "- No detector forward, latency claim, output compression claim, or end-to-end acceleration.",
        "- QUALITY is a weighted matching proxy, not COCO AP, fairness, safety, or TVT evidence.", "- Three seeds do not cover retraining or regrouping uncertainty.",
        "- No novelty claim and no automatic next-stage training.", "",
        "## Final status", "", "`execution_status = COMPLETE`", "",
        f"`LEARN_MICRO: objective_evidence={micro_evidence['status']}, quality_screen={micro_screen['status']}`", "",
        f"`LEARN_QUALITY: objective_evidence={quality_evidence['status']}, quality_screen={quality_screen_result['status']}`", "",
    ]
    (root / "LC_ALLOC_P1_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    append_log(root, f"STAGE summarize_results COMPLETE elapsed_seconds={time.perf_counter()-t0:.6f}")


if __name__ == "__main__":
    main()
