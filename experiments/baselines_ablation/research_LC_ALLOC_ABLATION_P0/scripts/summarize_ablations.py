from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from common import BUDGETS, SEEDS, append_log, sha256_file, verify_input_binding, write_json, write_parquet  # noqa: E402

CORE = (10, 15, 20)
METRICS_GROUP = ("coverage_per_image", "quality_per_image")
METRICS_GLOBAL = ("AP", "AP50", "AP75", "AR100")
COMPARISONS = (
    ("FULL_MINUS_NO_EMBED", "PRIMARY", "FULL", "NO_EMBED"),
    ("FULL_MINUS_NO_PREFIX", "PRIMARY", "FULL", "NO_PREFIX"),
    ("FULL_MINUS_MATCHABILITY_TARGET", "PRIMARY", "FULL", "MATCHABILITY_TARGET"),
    ("FULL_DP_MINUS_NEXT_SLOT_GREEDY", "PRIMARY", "FULL", "NEXT_SLOT_GREEDY"),
    ("FULL_MINUS_S_ADAPT", "REFERENCE", "FULL", "S_ADAPT"),
    ("NO_EMBED_MINUS_S_ADAPT", "REFERENCE", "NO_EMBED", "S_ADAPT"),
    ("NO_PREFIX_MINUS_S_ADAPT", "REFERENCE", "NO_PREFIX", "S_ADAPT"),
    ("MATCHABILITY_TARGET_MINUS_S_ADAPT", "REFERENCE", "MATCHABILITY_TARGET", "S_ADAPT"),
    ("NEXT_SLOT_GREEDY_MINUS_S_ADAPT", "REFERENCE", "NEXT_SLOT_GREEDY", "S_ADAPT"),
)


def method_seed(method: str, seed: int) -> int:
    return -1 if method == "S_ADAPT" else int(seed)


def percentile_ci(values: np.ndarray) -> tuple[float, float, float]:
    x = np.asarray(values, dtype=np.float64)
    return float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5)), float(np.mean(x > 0.0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", required=True)
    ap.add_argument("--p1-root", required=True)
    args = ap.parse_args()
    root = Path(args.project_root).resolve()
    p1 = Path(args.p1_root).resolve()
    start = time.perf_counter()
    append_log(root, "STAGE summarize_ablations START")
    verify_input_binding(root)
    main_df = pq.read_table(root / "outputs" / "ablation_main_results.parquet").to_pandas()
    group_df = pq.read_table(root / "outputs" / "ablation_group_results.parquet").to_pandas()
    if len(main_df) != 80 or len(group_df) != 4000:
        raise RuntimeError(f"evaluation row cardinality mismatch main={len(main_df)} group={len(group_df)}")
    bootstrap_path = p1 / "outputs" / "bootstrap_group_indices.npy"
    ledger = pd.read_csv(p1 / "input_output_sha256.csv")
    ledger_row = ledger[ledger["path"].astype(str).str.replace("\\", "/", regex=False).str.lower().str.endswith("outputs/bootstrap_group_indices.npy")]
    if len(ledger_row) != 1 or sha256_file(bootstrap_path) != str(ledger_row.iloc[0]["sha256"]):
        raise RuntimeError("frozen bootstrap index hash mismatch")
    bootstrap_indices = np.load(bootstrap_path, allow_pickle=False)
    if bootstrap_indices.shape != (5000, 50) or bootstrap_indices.min() < 0 or bootstrap_indices.max() >= 50:
        raise RuntimeError("frozen bootstrap group index asset invalid")
    rows = []
    bootstrap_checks = 0
    for comparison, family, method_a, method_b in COMPARISONS:
        for metric in METRICS_GROUP + METRICS_GLOBAL:
            for scope in list(BUDGETS) + ["CORE_10_15_20"]:
                seed_points = []
                group_seed_deltas = []
                for seed in SEEDS:
                    sa, sb = method_seed(method_a, seed), method_seed(method_b, seed)
                    budgets = CORE if scope == "CORE_10_15_20" else (int(scope),)
                    if metric in METRICS_GROUP:
                        delta_by_budget = []
                        for budget in budgets:
                            a = group_df[(group_df.method == method_a) & (group_df.seed == sa) & (group_df.budget == budget)].sort_values("group_id")
                            b = group_df[(group_df.method == method_b) & (group_df.seed == sb) & (group_df.budget == budget)].sort_values("group_id")
                            if len(a) != 50 or len(b) != 50 or not np.array_equal(a.group_id.to_numpy(), np.arange(50)) or not np.array_equal(b.group_id.to_numpy(), np.arange(50)):
                                raise RuntimeError(f"group comparison rows incomplete {comparison} {metric} {scope} seed={seed}")
                            delta_by_budget.append(a[metric].to_numpy(np.float64) - b[metric].to_numpy(np.float64))
                        group_delta = np.mean(np.vstack(delta_by_budget), axis=0)
                        point = float(group_delta.mean())
                        group_seed_deltas.append(group_delta)
                    else:
                        deltas = []
                        for budget in budgets:
                            a = main_df[(main_df.method == method_a) & (main_df.seed == sa) & (main_df.budget == budget)]
                            b = main_df[(main_df.method == method_b) & (main_df.seed == sb) & (main_df.budget == budget)]
                            if len(a) != 1 or len(b) != 1:
                                raise RuntimeError(f"global comparison rows incomplete {comparison} {metric} {scope} seed={seed}")
                            deltas.append(float(a.iloc[0][metric] - b.iloc[0][metric]))
                        point = float(np.mean(deltas))
                    seed_points.append(point)
                    rows.append({
                        "comparison": comparison, "comparison_family": family, "method_a": method_a, "method_b": method_b,
                        "metric": metric, "budget_scope": str(scope), "seed_aggregation": str(seed),
                        "point_estimate_a_minus_b": point, "ci95_low": math.nan, "ci95_high": math.nan,
                        "bootstrap_strict_positive_fraction": math.nan, "bootstrap_resamples": 0,
                        "bootstrap_unit": "not computed for individual seed or full-DEV nonlinear metric",
                    })
                mean_point = float(np.mean(seed_points))
                ci_low = ci_high = positive = math.nan
                n_boot = 0
                unit = "full-DEV point estimate only; no COCO bootstrap"
                if metric in METRICS_GROUP:
                    mean_group_delta = np.mean(np.vstack(group_seed_deltas), axis=0)
                    if not np.isclose(mean_point, mean_group_delta.mean(), atol=1e-12, rtol=0.0):
                        raise RuntimeError("three-seed group-delta point reconciliation failed")
                    reps = mean_group_delta[bootstrap_indices].mean(axis=1)
                    ci_low, ci_high, positive = percentile_ci(reps)
                    n_boot = len(reps)
                    unit = "50 frozen 40-image groups, paired resampling"
                    bootstrap_checks += 1
                rows.append({
                    "comparison": comparison, "comparison_family": family, "method_a": method_a, "method_b": method_b,
                    "metric": metric, "budget_scope": str(scope), "seed_aggregation": "THREE_SEED_MEAN",
                    "point_estimate_a_minus_b": mean_point, "ci95_low": ci_low, "ci95_high": ci_high,
                    "bootstrap_strict_positive_fraction": positive, "bootstrap_resamples": n_boot,
                    "bootstrap_unit": unit,
                })
    comparisons = pd.DataFrame(rows)
    if len(comparisons) != 1296 or bootstrap_checks != 108:
        raise RuntimeError(f"comparison table cardinality mismatch rows={len(comparisons)} boot={bootstrap_checks}")
    write_parquet(comparisons, root / "outputs" / "ablation_comparisons.parquet")

    primary_core = comparisons[
        (comparisons["comparison_family"] == "PRIMARY")
        & (comparisons["budget_scope"] == "CORE_10_15_20")
        & (comparisons["seed_aggregation"] == "THREE_SEED_MEAN")
        & (comparisons["metric"].isin(METRICS_GROUP))
    ].copy()
    interpretations = []
    for r in primary_core.itertuples(index=False):
        if r.ci95_low > 0:
            status = "SUPPORTED_CONDITIONAL_CONTRIBUTION_FOR_FULL"
        elif r.ci95_high < 0:
            status = "ABLATION_OUTPERFORMS_FULL_ON_THIS_METRIC"
        else:
            status = "NOT_RESOLVED_BY_95CI"
        interpretations.append({"comparison": r.comparison, "metric": r.metric, "point": r.point_estimate_a_minus_b, "ci95": [r.ci95_low, r.ci95_high], "status": status})
    write_json(root / "outputs" / "analysis_summary.json", {
        "status": "PASS", "comparison_rows": len(comparisons), "bootstrap_resamples": 5000,
        "bootstrap_seed": 530002, "positive_fraction_is_not_p_value": True,
        "intervals_are_exploratory_unadjusted_95_percent": True,
        "multiplicity_controlled_confirmation": False,
        "ci_crossing_zero_does_not_establish_equivalence": True,
        "post_confirmation_dev_ablation": True, "primary_core_interpretations": interpretations,
        "elapsed_seconds": time.perf_counter() - start,
    })
    append_log(root, f"STAGE summarize_ablations COMPLETE elapsed_seconds={time.perf_counter()-start:.6f}")


if __name__ == "__main__":
    main()
