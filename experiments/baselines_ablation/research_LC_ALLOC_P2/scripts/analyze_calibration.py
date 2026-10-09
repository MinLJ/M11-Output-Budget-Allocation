"""Create LC-ALLOC-P2 contrasts and frozen-index group bootstrap results.

This stage consumes already evaluated parquet tables only.  It does not open
ground truth, candidate/native state, detector assets, or any restricted split.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.dont_write_bytecode = True

BUDGETS = (10, 15, 20, 30, 40)
CORE = (10, 15, 20)
SEEDS = (530101, 530102, 530103)
POINT_METRICS = ("coverage_per_image", "quality_per_image", "AP", "AP50", "AP75", "AR100")
BOOT_METRICS = ("coverage_per_image", "quality_per_image")
COMPARISONS = (
    ("M11_MINUS_CAL_TEMP", "LEARN_QUALITY", "CAL_TEMP_ALLOC"),
    ("M11_MINUS_CAL_ISO", "LEARN_QUALITY", "CAL_ISO_ALLOC"),
    ("CAL_TEMP_MINUS_S_ADAPT", "CAL_TEMP_ALLOC", "S_ADAPT"),
    ("CAL_ISO_MINUS_S_ADAPT", "CAL_ISO_ALLOC", "S_ADAPT"),
)


def append_log(root: Path, message: str) -> None:
    with (root / "runtime.log").open("a", encoding="utf-8") as f:
        f.write(message.rstrip() + "\n")


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), path, compression="zstd")


def method_value(main: pd.DataFrame, method: str, budget: int, metric: str) -> float:
    frame = main[(main["method"] == method) & (main["budget"] == budget)]
    if method == "LEARN_QUALITY":
        frame = frame[frame["seed"].isin(SEEDS)]
        if len(frame) != 3:
            raise RuntimeError(f"missing three frozen M11 rows: {budget}/{metric}")
        return float(frame[metric].mean())
    frame = frame[frame["seed"] == -1]
    if len(frame) != 1:
        raise RuntimeError(f"missing deterministic row: {method}/{budget}/{metric}")
    return float(frame.iloc[0][metric])


def method_group_vector(group: pd.DataFrame, method: str, budget: int, metric: str) -> np.ndarray:
    frame = group[(group["method"] == method) & (group["budget"] == budget)]
    if method == "LEARN_QUALITY":
        frame = frame[frame["seed"].isin(SEEDS)].groupby("group_id", as_index=False)[metric].mean()
    else:
        frame = frame[frame["seed"] == -1][["group_id", metric]]
    frame = frame.sort_values("group_id")
    if len(frame) != 50 or frame["group_id"].astype(int).tolist() != list(range(50)):
        raise RuntimeError(f"incomplete frozen group vector: {method}/{budget}/{metric}")
    return frame[metric].to_numpy(np.float64)


def point_unit(metric: str) -> str:
    if metric in ("coverage_per_image", "quality_per_image"):
        return "count_per_image"
    return "proportion"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--p1-root", required=True)
    parser.add_argument("--p1a-root", required=True)
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    p1 = Path(args.p1_root).resolve()
    p1a = Path(args.p1a_root).resolve()
    t0 = time.perf_counter()
    append_log(root, "STAGE analyze_calibration START")

    main_frame = pq.read_table(root / "outputs" / "calibration_combined_main.parquet").to_pandas()
    group_frame = pq.read_table(root / "outputs" / "calibration_combined_group.parquet").to_pandas()
    class_frame = pq.read_table(root / "outputs" / "calibration_combined_class.parquet").to_pandas()
    if len(main_frame) != 30 or len(group_frame) != 1500 or len(class_frame) != 240:
        raise RuntimeError("combined calibration tables are incomplete")
    expected_methods = {"S_ADAPT", "CAL_TEMP_ALLOC", "CAL_ISO_ALLOC", "LEARN_QUALITY"}
    if set(main_frame["method"]) != expected_methods:
        raise RuntimeError("combined calibration method set mismatch")

    scopes: tuple[int | str, ...] = BUDGETS + ("CORE_10_15_20",)
    point_rows: list[dict[str, Any]] = []
    for comparison, left, right in COMPARISONS:
        for metric in POINT_METRICS:
            per_budget = {
                budget: method_value(main_frame, left, budget, metric) - method_value(main_frame, right, budget, metric)
                for budget in BUDGETS
            }
            for scope in scopes:
                estimate = per_budget[scope] if isinstance(scope, int) else float(np.mean([per_budget[b] for b in CORE]))
                point_rows.append({
                    "comparison": comparison, "left_method": left, "right_method": right,
                    "metric": metric, "scope": str(scope), "estimate": estimate,
                    "unit": point_unit(metric),
                    "estimate_per_100_images": 100.0 * estimate if metric == "coverage_per_image" else math_nan(),
                    "estimate_percentage_points": 100.0 * estimate if metric in ("AP", "AP50", "AP75", "AR100") else math_nan(),
                    "m11_seed_aggregation": "arithmetic mean of three complete frozen seed results" if "LEARN_QUALITY" in (left, right) else "not_applicable",
                    "source": "P2 frozen output replay",
                })
    points = pd.DataFrame(point_rows)
    if len(points) != 144:
        raise RuntimeError(f"point contrast row count mismatch: {len(points)}")
    write_parquet(root / "outputs" / "calibration_comparison_points.parquet", points)

    indices = np.load(p1 / "outputs" / "bootstrap_group_indices.npy", allow_pickle=False)
    if indices.shape != (5000, 50) or int(indices.min()) != 0 or int(indices.max()) != 49:
        raise RuntimeError("frozen P1 bootstrap indices invalid")
    bootstrap_rows: list[dict[str, Any]] = []
    for comparison, left, right in COMPARISONS:
        for metric in BOOT_METRICS:
            differences = {
                budget: method_group_vector(group_frame, left, budget, metric)
                - method_group_vector(group_frame, right, budget, metric)
                for budget in BUDGETS
            }
            for scope in scopes:
                vector = differences[scope] if isinstance(scope, int) else np.mean(
                    np.vstack([differences[budget] for budget in CORE]), axis=0
                )
                resampled = vector[indices].mean(axis=1)
                bootstrap_rows.append({
                    "comparison": comparison, "left_method": left, "right_method": right,
                    "metric": metric, "scope": str(scope), "observed_delta": float(vector.mean()),
                    "ci95_low": float(np.quantile(resampled, 0.025)),
                    "ci95_high": float(np.quantile(resampled, 0.975)),
                    "strict_positive_resample_fraction": float(np.mean(resampled > 0)),
                    "bootstrap_unit": "frozen 40-image group", "bootstrap_resamples": 5000,
                    "bootstrap_seed": 530002, "group_count": 50,
                    "seed_aggregation": "mean three complete frozen M11 seed results per group before contrast"
                    if "LEARN_QUALITY" in (left, right) else "deterministic baseline contrast",
                    "interpretation_note": "strict-positive fraction is not a traditional p-value",
                    "source": "P1 frozen bootstrap indices replayed in P2",
                })
    bootstrap = pd.DataFrame(bootstrap_rows)
    if len(bootstrap) != 48:
        raise RuntimeError(f"bootstrap row count mismatch: {len(bootstrap)}")
    write_parquet(root / "outputs" / "calibration_bootstrap.parquet", bootstrap)

    # Audit the structural consequence of score-only calibration.  A strictly
    # increasing global transform should preserve slot order and therefore the
    # exact S_ADAPT allocation under the same tie-break.  Isotonic plateaus can
    # legitimately change the allocation via frozen tie handling.
    new_per = pq.read_table(root / "outputs" / "calibration_per_image.parquet").to_pandas()
    old_per = pq.read_table(p1a / "ablation_per_image_results.parquet").to_pandas()
    old_s = old_per[(old_per["method"] == "S_ADAPT") & (old_per["seed"] == -1)][["image_id", "budget", "K_i"]].copy()
    structural: dict[str, Any] = {
        "principle": "global strictly increasing score transforms preserve score-slot ordering; isotonic plateaus may expose tie-break differences",
        "comparisons": {},
    }
    for method in ("CAL_TEMP_ALLOC", "CAL_ISO_ALLOC"):
        method_per = new_per[new_per["method"] == method][["image_id", "budget", "K_i"]]
        merged = old_s.merge(method_per, on=["image_id", "budget"], suffixes=("_s", "_cal"), validate="one_to_one")
        by_budget = {}
        for budget in BUDGETS:
            subset = merged[merged["budget"] == budget]
            by_budget[str(budget)] = {
                "image_K_mismatch_count": int((subset["K_i_s"] != subset["K_i_cal"]).sum()),
                "total_absolute_K_difference": int(np.abs(subset["K_i_s"] - subset["K_i_cal"]).sum()),
            }
        structural["comparisons"][f"{method}_VS_S_ADAPT"] = by_budget
    write_json(root / "qa" / "score_calibration_structure.json", structural)

    # This rule must also appear in run_config before DEV evaluation.  It picks
    # the confirmation comparator, not a learned method or a tuned parameter.
    baseline_summary = []
    for method in ("CAL_TEMP_ALLOC", "CAL_ISO_ALLOC"):
        quality = float(np.mean([method_value(main_frame, method, budget, "quality_per_image") for budget in CORE]))
        coverage = float(np.mean([method_value(main_frame, method, budget, "coverage_per_image") for budget in CORE]))
        baseline_summary.append({"method": method, "core_quality_per_image": quality, "core_coverage_per_image": coverage})
    baseline_summary.sort(
        key=lambda row: (-row["core_quality_per_image"], -row["core_coverage_per_image"], 0 if row["method"] == "CAL_TEMP_ALLOC" else 1)
    )
    write_json(root / "outputs" / "confirmation_baseline_selection.json", {
        "status": "DEVELOPMENT_SET_BASELINE_SELECTION_ONLY",
        "predeclared_rule": "higher DEV core K10/K15/K20 mean QUALITY/image; tie-break higher coverage/image; final tie-break CAL_TEMP_ALLOC",
        "ranked_baselines": baseline_summary,
        "selected_baseline_for_confirmation_protocol": baseline_summary[0]["method"],
        "does_not_authorize_test_access": True,
    })

    write_json(root / "qa" / "analysis_validation.json", {
        "status": "PASS", "main_rows": len(main_frame), "group_rows": len(group_frame), "class_rows": len(class_frame),
        "point_contrast_rows": len(points), "bootstrap_rows": len(bootstrap),
        "bootstrap_index_shape": list(indices.shape), "bootstrap_seed": 530002,
        "elapsed_seconds": time.perf_counter() - t0,
    })
    append_log(root, f"STAGE analyze_calibration COMPLETE elapsed_seconds={time.perf_counter() - t0:.6f}")


def math_nan() -> float:
    return float("nan")


if __name__ == "__main__":
    main()
