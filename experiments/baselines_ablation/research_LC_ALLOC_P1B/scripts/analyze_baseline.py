"""Create P1B point contrasts, frozen-index bootstrap, screens and science figures."""

from __future__ import annotations

import argparse
import json
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
SEEDS = (530101, 530102, 530103)
BUDGETS = (10, 15, 20, 30, 40)
CORE = (10, 15, 20)
ROAD8 = ("person", "bicycle", "car", "motorcycle", "bus", "train", "truck", "traffic light")
METRICS = ("coverage_per_image", "quality_per_image", "AP", "AP50", "AP75", "AR100")
BOOT_METRICS = ("coverage_per_image", "quality_per_image")
COMPARISONS = (
    ("M11_MINUS_S_CLASS_ADAPT", "PRIMARY", "LEARN_QUALITY", "S_CLASS_ADAPT"),
    ("M10_MINUS_S_CLASS_ADAPT", "AUXILIARY", "LEARN_CLASS50", "S_CLASS_ADAPT"),
    ("S_CLASS_ADAPT_MINUS_S_ADAPT", "AUXILIARY", "S_CLASS_ADAPT", "S_ADAPT"),
)


def append_log(root: Path, value: str) -> None:
    with (root / "runtime.log").open("a", encoding="utf-8") as f:
        f.write(value.rstrip() + "\n")


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def value(main: pd.DataFrame, method: str, variant: str, budget: int, metric: str) -> float:
    d = main[(main["method"] == method) & (main["budget"] == budget)]
    if method in ("LEARN_CLASS50", "LEARN_QUALITY"):
        if variant == "THREE_SEED_MEAN":
            d = d[d["seed"].isin(SEEDS)]
            if len(d) != 3: raise RuntimeError(f"missing 3-seed row: {method}/{budget}")
            return float(d[metric].mean())
        d = d[d["seed"] == int(variant.removeprefix("SEED_"))]
    else:
        d = d[d["seed"] == -1]
    if len(d) != 1: raise RuntimeError(f"missing deterministic row: {method}/{budget}")
    return float(d.iloc[0][metric])


def variants(left: str, right: str) -> tuple[str, ...]:
    return tuple(f"SEED_{s}" for s in SEEDS) + ("THREE_SEED_MEAN",) if "LEARN_" in left or "LEARN_" in right else ("DETERMINISTIC",)


def group_vector(group: pd.DataFrame, method: str, budget: int, metric: str) -> np.ndarray:
    d = group[(group["method"] == method) & (group["budget"] == budget)]
    if method in ("LEARN_CLASS50", "LEARN_QUALITY"):
        d = d[d["seed"].isin(SEEDS)].groupby("group_id", as_index=False)[metric].mean()
    else:
        d = d[d["seed"] == -1][["group_id", metric]]
    d = d.sort_values("group_id")
    if len(d) != 50 or d["group_id"].astype(int).tolist() != list(range(50)):
        raise RuntimeError(f"group vector incomplete: {method}/{budget}/{metric}")
    return d[metric].to_numpy(np.float64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", required=True)
    ap.add_argument("--p1-root", required=True)
    args = ap.parse_args()
    root, p1 = Path(args.project_root).resolve(), Path(args.p1_root).resolve()
    t0 = time.perf_counter(); append_log(root, "STAGE analyze_baseline START")
    main_df = pq.read_table(root / "outputs" / "baseline_main.parquet").to_pandas()
    class_df = pq.read_table(root / "outputs" / "baseline_class.parquet").to_pandas()
    group_df = pq.read_table(root / "outputs" / "baseline_group.parquet").to_pandas()
    if len(main_df) != 40 or len(class_df) != 320 or len(group_df) != 2000:
        raise RuntimeError("combined science tables incomplete")

    scopes: tuple[int | str, ...] = BUDGETS + ("CORE_10_15_20",)
    rows = []
    for name, role, left, right in COMPARISONS:
        for variant in variants(left, right):
            for metric in METRICS:
                per_budget = {b: value(main_df, left, variant, b, metric) - value(main_df, right, variant, b, metric) for b in BUDGETS}
                for scope in scopes:
                    estimate = per_budget[scope] if isinstance(scope, int) else float(np.mean([per_budget[b] for b in CORE]))
                    rows.append({
                        "comparison_family": "SCORE_CLASS_BRIDGE", "comparison": name, "comparison_role": role,
                        "left_method": left, "right_method": right, "metric": metric, "scope": str(scope),
                        "variant": variant, "estimate": estimate,
                        "unit": "count_per_image" if metric.endswith("per_image") else "proportion",
                        "left_expression": left, "right_expression": right,
                    })
    contrasts = pd.DataFrame(rows)
    if len(contrasts) != 324:
        raise RuntimeError(f"contrast count {len(contrasts)} != 324")
    pq.write_table(pa.Table.from_pandas(contrasts, preserve_index=False), root / "outputs" / "baseline_comparison_points.parquet", compression="zstd")

    indices = np.load(p1 / "outputs" / "bootstrap_group_indices.npy", allow_pickle=False)
    if indices.shape != (5000, 50) or indices.min() != 0 or indices.max() != 49:
        raise RuntimeError("frozen bootstrap indices invalid")
    boot_rows = []
    for name, role, left, right in COMPARISONS:
        for metric in BOOT_METRICS:
            diffs = {b: group_vector(group_df, left, b, metric) - group_vector(group_df, right, b, metric) for b in BUDGETS}
            for scope in scopes:
                vec = diffs[scope] if isinstance(scope, int) else np.mean(np.vstack([diffs[b] for b in CORE]), axis=0)
                boot = vec[indices].mean(axis=1)
                boot_rows.append({
                    "comparison": name, "comparison_role": role, "left_method": left, "right_method": right,
                    "metric": metric, "scope": str(scope), "variant": "THREE_SEED_MEAN" if "LEARN_" in left else "DETERMINISTIC",
                    "observed_delta": float(vec.mean()), "ci95_low": float(np.quantile(boot, .025)),
                    "ci95_high": float(np.quantile(boot, .975)),
                    "strict_positive_resample_fraction": float(np.mean(boot > 0)),
                    "bootstrap_unit": "frozen 40-image group", "bootstrap_resamples": 5000,
                    "bootstrap_seed": 530002, "group_count": 50,
                    "seed_aggregation": "mean three complete frozen model results per group before contrast" if "LEARN_" in left else "deterministic single result",
                    "source": "P1 frozen bootstrap indices replayed in P1B",
                })
    boot_df = pd.DataFrame(boot_rows)
    if len(boot_df) != 36:
        raise RuntimeError(f"bootstrap count {len(boot_df)} != 36")
    pq.write_table(pa.Table.from_pandas(boot_df, preserve_index=False), root / "outputs" / "baseline_bootstrap.parquet", compression="zstd")

    # Descriptive replay of P1's engineering quality screen against S_ADAPT.
    screens = {}
    for method in ("S_CLASS_ADAPT", "LEARN_CLASS50", "LEARN_QUALITY"):
        ap_delta = {b: value(main_df, method, "THREE_SEED_MEAN", b, "AP") - value(main_df, "S_ADAPT", "THREE_SEED_MEAN", b, "AP") for b in BUDGETS}
        ar_delta = {b: value(main_df, method, "THREE_SEED_MEAN", b, "AR100") - value(main_df, "S_ADAPT", "THREE_SEED_MEAN", b, "AR100") for b in BUDGETS}
        class_deltas = []
        for ci in range(1, 9):
            m = class_df[(class_df["method"] == method) & (class_df["category_id"] == ci) & (class_df["budget"].isin(CORE))]
            s = class_df[(class_df["method"] == "S_ADAPT") & (class_df["category_id"] == ci) & (class_df["budget"].isin(CORE))]
            if method.startswith("LEARN_"):
                md = m[m["seed"].isin(SEEDS)].groupby("budget")["coverage_recall"].mean()
            else:
                md = m[m["seed"] == -1].set_index("budget")["coverage_recall"]
            sd = s[s["seed"] == -1].set_index("budget")["coverage_recall"]
            gt = int(s["GT"].iloc[0]); delta = float(np.mean([md.loc[b] - sd.loc[b] for b in CORE]))
            class_deltas.append({"category_id": ci, "class_name": ROAD8[ci - 1], "GT": gt, "core_coverage_recall_delta": delta})
        core_pass = all(ap_delta[b] >= -.002 and ar_delta[b] >= -.005 for b in CORE) and all(x["core_coverage_recall_delta"] >= -.02 for x in class_deltas if x["GT"] >= 100)
        high_warning = any(ap_delta[b] < -.002 or ar_delta[b] < -.005 for b in (30, 40))
        screens[method] = {"core_screen": "PASS" if core_pass else "TRADEOFF", "high_budget_quality_warning": high_warning,
                           "AP_delta_vs_S_ADAPT": ap_delta, "AR100_delta_vs_S_ADAPT": ar_delta,
                           "class_core_coverage_recall_delta_vs_S_ADAPT": class_deltas}
    write_json(root / "outputs" / "quality_screens.json", screens)

    # Figure 1: common coverage/QUALITY trade-off.
    mean_main = main_df[main_df["method"].isin(["LEARN_CLASS50", "LEARN_QUALITY"])].groupby(["method", "budget"], as_index=False).mean(numeric_only=True)
    mean_main = pd.concat([mean_main, main_df[main_df["method"].isin(["S_ADAPT", "S_CLASS_ADAPT"])]] , ignore_index=True, sort=False)
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for method, label in (("S_ADAPT", "S_ADAPT"), ("S_CLASS_ADAPT", "S_CLASS_ADAPT"), ("LEARN_CLASS50", "M10"), ("LEARN_QUALITY", "M11")):
        d = mean_main[mean_main["method"] == method].sort_values("budget")
        ax[0].plot(d["budget"], d["coverage_per_image"], marker="o", label=label)
        ax[1].plot(d["budget"], d["quality_per_image"], marker="o", label=label)
    ax[0].set(title="IoU .50 coverage", xlabel="Mean budget", ylabel="Coverage / image")
    ax[1].set(title="Frozen QUALITY proxy", xlabel="Mean budget", ylabel="QUALITY / image")
    for a in ax: a.grid(alpha=.25); a.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(root / "figures" / "baseline_coverage_quality_tradeoff.png", dpi=180); plt.close(fig)

    # Figure 2: primary QUALITY interval plus class recall deltas at the core mean.
    primary = boot_df[(boot_df["metric"] == "quality_per_image") & (boot_df["scope"] == "CORE_10_15_20")]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    labels = primary["comparison"].tolist(); y = np.arange(len(primary)); x = primary["observed_delta"].to_numpy()
    lo, hi = primary["ci95_low"].to_numpy(), primary["ci95_high"].to_numpy()
    ax[0].errorbar(x, y, xerr=np.vstack([x-lo, hi-x]), fmt="o", capsize=4); ax[0].axvline(0, color="black", lw=.8)
    ax[0].set_yticks(y, labels); ax[0].set_xlabel("Δ QUALITY / image (95% group bootstrap CI)"); ax[0].grid(axis="x", alpha=.25)
    cls = screens["LEARN_QUALITY"]["class_core_coverage_recall_delta_vs_S_ADAPT"]
    cls_base = screens["S_CLASS_ADAPT"]["class_core_coverage_recall_delta_vs_S_ADAPT"]
    xx = np.arange(8); ax[1].bar(xx-.18, [100*z["core_coverage_recall_delta"] for z in cls_base], .36, label="S_CLASS - S")
    ax[1].bar(xx+.18, [100*z["core_coverage_recall_delta"] for z in cls], .36, label="M11 - S")
    ax[1].axhline(0, color="black", lw=.8); ax[1].set_xticks(xx, ROAD8, rotation=35, ha="right")
    ax[1].set_ylabel("Core coverage-recall change (pp)"); ax[1].legend(fontsize=8); ax[1].grid(axis="y", alpha=.25)
    fig.tight_layout(); fig.savefig(root / "figures" / "quality_intervals_and_class_deltas.png", dpi=180); plt.close(fig)
    append_log(root, f"STAGE analyze_baseline COMPLETE elapsed_seconds={time.perf_counter()-t0:.6f}")


if __name__ == "__main__":
    main()
