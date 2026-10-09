"""Prepare compact JSON sources for Artifact Tool CSV export and reporting."""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(r"D:\AOP_DETR\research_LC_ALLOC_EFF_AUDIT")
P1B = Path(r"D:\AOP_DETR\research_LC_ALLOC_P1B")


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    samples = pq.read_table(ROOT / "outputs" / "runtime_pipeline_samples.parquet").to_pandas()
    summary = pq.read_table(ROOT / "outputs" / "runtime_pipeline_summary.parquet").to_pandas()
    if len(samples) != 450 or len(summary) != 9:
        raise RuntimeError(f"RUNTIME_ROW_COUNT_FAIL samples={len(samples)} summary={len(summary)}")
    grouped = samples.groupby(["images", "budget"], sort=True)
    if not (grouped.size() == 50).all():
        raise RuntimeError("RUNTIME_REPEAT_COUNT_FAIL")
    if not (grouped["selection_digest"].nunique() == 1).all():
        raise RuntimeError("RUNTIME_SELECTION_NONDETERMINISTIC")
    if not np.allclose(samples["total_ms"], samples["detector_ms"] + samples["allocator_ms"], atol=1e-9, rtol=0):
        raise RuntimeError("NESTED_TIMING_IDENTITY_FAIL")
    if not (samples["output_records"] == samples["images"] * samples["budget"]).all():
        raise RuntimeError("RUNTIME_OUTPUT_BUDGET_FAIL")

    science = pd.read_csv(P1B / "baseline_comparison_main.csv")
    m11 = science[science["method"] == "LEARN_QUALITY"].groupby("budget", as_index=True)[
        ["coverage_per_image", "quality_per_image", "AP", "AR100"]
    ].mean()
    score = science[science["method"] == "S_ADAPT"].groupby("budget", as_index=True)[
        ["coverage_per_image", "quality_per_image", "AP", "AR100"]
    ].mean()
    delta = m11 - score
    rows = []
    for row in summary.sort_values(["images", "K"]).to_dict("records"):
        k = int(row["K"])
        d = delta.loc[k]
        allocator_per_image = float(row["allocator_ms_per_image_median"])
        record = {
            "Images": int(row["images"]),
            "K": k,
            "TimingSeed": int(row["timing_seed"]),
            "Warmups": int(row["warmups"]),
            "Measurements": int(row["measurements"]),
            "Detector_ms_mean": float(row["detector_ms_mean"]),
            "Detector_ms_median": float(row["detector_ms_median"]),
            "Detector_ms_p95": float(row["detector_ms_p95"]),
            "Detector_ms_std": float(row["detector_ms_std"]),
            "Allocator_ms_mean": float(row["allocator_ms_mean"]),
            "Allocator_ms_median": float(row["allocator_ms_median"]),
            "Allocator_ms_p95": float(row["allocator_ms_p95"]),
            "Allocator_ms_std": float(row["allocator_ms_std"]),
            "Total_ms_mean": float(row["total_ms_mean"]),
            "Total_ms_median": float(row["total_ms_median"]),
            "Total_ms_p95": float(row["total_ms_p95"]),
            "Total_ms_std": float(row["total_ms_std"]),
            "Overhead_allocator_over_detector_pct": float(row["allocator_over_detector_pct_median"]),
            "Overhead_allocator_over_total_pct": float(row["allocator_over_total_pct_median"]),
            "Detector_ms_per_image_median": float(row["detector_ms_per_image_median"]),
            "Allocator_ms_per_image_median": allocator_per_image,
            "Total_ms_per_image_median": float(row["total_ms_per_image_median"]),
            "Delta_coverage_per_image_vs_S_ADAPT": float(d["coverage_per_image"]),
            "Delta_QUALITY_per_image_vs_S_ADAPT": float(d["quality_per_image"]),
            "Delta_AP_percentage_points_vs_S_ADAPT": 100.0 * float(d["AP"]),
            "Delta_AR100_percentage_points_vs_S_ADAPT": 100.0 * float(d["AR100"]),
            "Delta_coverage_per_additional_ms_per_image": float(d["coverage_per_image"]) / allocator_per_image,
            "Delta_QUALITY_per_additional_ms_per_image": float(d["quality_per_image"]) / allocator_per_image,
            "AllocationGrouping": str(row["allocation_grouping"]),
            "TimingBoundary": "host tensor -> canonical Top300 -> native state -> frozen M11 -> exact DP -> selected IDs",
            "QualityCostBasis": "DEV2000 three-seed M11 mean minus S_ADAPT; latency uses inherited representative timing seed 530101",
        }
        rows.append(record)

    columns = list(rows[0].keys())
    write_json(ROOT / "cache" / "runtime_pipeline_results.csv.json", {"columns": columns, "rows": [[r[c] for c in columns] for r in rows]})

    canonical = [r for r in rows if r["Images"] == 40]
    facts = {
        "sample_count": len(samples),
        "condition_count": len(summary),
        "selection_digest_unique_per_condition": True,
        "exact_budget_all_samples": True,
        "nested_timing_identity": True,
        "canonical_40_image_rows": canonical,
        "overall_detector_median_range_ms": [float(summary["detector_ms_median"].min()), float(summary["detector_ms_median"].max())],
        "overall_allocator_median_range_ms": [float(summary["allocator_ms_median"].min()), float(summary["allocator_ms_median"].max())],
        "overall_total_median_range_ms": [float(summary["total_ms_median"].min()), float(summary["total_ms_median"].max())],
        "all_finite": bool(np.isfinite(samples[["detector_ms", "allocator_ms", "total_ms"]].to_numpy()).all()),
    }
    write_json(ROOT / "outputs" / "runtime_facts.json", facts)


if __name__ == "__main__":
    main()
