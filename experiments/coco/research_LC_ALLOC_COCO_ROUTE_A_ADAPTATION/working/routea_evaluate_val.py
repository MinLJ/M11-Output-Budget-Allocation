"""Independent VAL2017 evaluation and frozen Route-B comparison for Route A.

Selection integrity is fully validated before this entry point opens the
official VAL annotations.  The existing Route-B evaluator is imported as a
read-only implementation reference and parameterized in memory; no Route-B
asset or source file is modified.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any

sys.dont_write_bytecode = True

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from routea_pipeline_common import (
    ASSET_ROOT_DEFAULT,
    BOOTSTRAP_INDICES_SHA256,
    BUDGETS,
    CANDIDATE_ASSET_ID,
    DATASET_VERSION,
    EXPECTED_GROUPS,
    EXPECTED_IMAGES,
    GROUP_MANIFEST_DEFAULT,
    METHOD_ORDER,
    P1_ROOT_DEFAULT,
    PYCOCO_SITE_DEFAULT,
    ROOT_DEFAULT,
    ROUTE_A_SEEDS,
    ROUTE_B_EVALUATION_ROOT_DEFAULT,
    ROUTE_B_EVALUATOR_DEFAULT,
    ROUTE_B_EVALUATOR_SHA256,
    ROUTE_B_LEDGER_SHA256,
    ROUTE_B_SELECTION_RUNNER_SHA256,
    ROUTE_B_SELECTION_ROOT_DEFAULT,
    ROUTE_B_SEEDS,
    SPLIT,
    VAL_GT_DEFAULT,
    VAL_GT_SHA256,
    expected_conditions,
    frozen_bootstrap_indices,
    load_class_weights,
    load_frozen_p1_modules,
    load_module,
    require,
    self_test_common,
    sha256_file,
    stable_condition_sort,
    utc_now,
    write_csv_atomic,
    write_json_atomic,
    write_text_atomic,
)


TASK = "LC-ALLOC-COCO-ROUTE-A-VAL-EVALUATION"
METRICS = ("AP", "AP50", "AP75", "AR100", "coverage_per_image", "coverage_recall", "quality_per_image")


def progress(stage: str, message: str) -> None:
    print(f"[{utc_now()}] {stage}: {message}", flush=True)


def append_log(root: Path, message: str) -> None:
    path = root / "working" / "routea_evaluation_runtime.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(f"{utc_now()}\t{message}\n")


def patch_routeb_evaluator(module: Any, selection_sha: str, output_root: Path) -> None:
    module.M11_SEEDS = ROUTE_A_SEEDS
    module.METHOD_ORDER = METHOD_ORDER
    module.EXPECTED_ASSET_ID = CANDIDATE_ASSET_ID
    module.EXPECTED_SELECTION_SHA = selection_sha
    module.expected_conditions = expected_conditions
    module.append_log = lambda _root, message: append_log(output_root, message)


def validate_selection_commit(output_root: Path) -> tuple[str, dict[str, Any]]:
    marker_path = output_root / "ROUTE_A_SELECTION_COMPLETE.json"
    require(marker_path.is_file(), "Route-A selection completion marker is missing")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    require(marker.get("status") == "ROUTE_A_SELECTION_COMPLETE" and marker.get("gt_accessed") is False, "invalid selection completion marker")
    required = ("selection_manifest.csv", "selected_records.parquet", "predicted_utility_curves.parquet", "routea_selection_config.json", "selection_QA.json")
    for name in required:
        path = output_root / name
        require(path.is_file(), f"missing committed selection output: {name}")
        expected = marker.get("files", {}).get(name)
        require(expected == sha256_file(path), f"selection completion SHA mismatch: {name}")
    selection_sha = sha256_file(output_root / "selected_records.parquet")
    manifest = pd.read_csv(output_root / "selection_manifest.csv")
    require(set(manifest["selected_records_sha256"].astype(str)) == {selection_sha}, "selection manifest selected-record SHA mismatch")
    require(set(manifest["status"].astype(str)) == {"FROZEN_BEFORE_GT_EVALUATION"}, "selection was not frozen before GT")
    require(set(manifest["gt_path_available_to_selection_entrypoint"].astype(str).str.lower()) == {"false"}, "selection entrypoint had GT available")
    return selection_sha, marker


def validate_selection_asset_bindings(output_root: Path, asset_root: Path, group_manifest: Path) -> dict[str, Any]:
    """Prove evaluation uses the exact canonical files frozen at selection time."""
    selection_config_path = output_root / "routea_selection_config.json"
    selection_config = json.loads(selection_config_path.read_text(encoding="utf-8"))
    require(selection_config.get("candidate_asset_id") == CANDIDATE_ASSET_ID, "selection config CandidateAssetID mismatch")
    bindings = selection_config.get("input_bindings", {})
    marker_path = asset_root / "CANONICAL_EXPORT_COMPLETE.json"
    marker_sha = sha256_file(marker_path)
    require(bindings.get("canonical_completion_marker", {}).get("sha256") == marker_sha, "selection-time canonical marker binding mismatch")
    require(bindings.get("group_manifest", {}).get("sha256") == sha256_file(group_manifest), "selection-time group manifest binding mismatch")
    require(bindings.get("routeb_selection_runner_readonly", {}).get("sha256") == ROUTE_B_SELECTION_RUNNER_SHA256, "selection-time Route-B runner binding mismatch")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    require(marker.get("status") == "COCO_CANONICAL_EXPORT_COMPLETE", "canonical asset completion status mismatch")
    require(marker.get("candidate_asset_id") == CANDIDATE_ASSET_ID, "canonical marker CandidateAssetID mismatch")
    for name in ("candidate_manifest.csv", "image_manifest.csv"):
        expected = marker.get("manifest_sha256", {}).get(name)
        require(expected == sha256_file(asset_root / name), f"canonical manifest SHA mismatch: {name}")
    return selection_config


def routeb_ledger_hashes(root: Path) -> dict[str, str]:
    ledger_path = root / "sha256_ledger.csv"
    require(ledger_path.is_file(), "Route-B evaluation ledger is missing")
    require(sha256_file(ledger_path) == ROUTE_B_LEDGER_SHA256, "frozen Route-B ledger identity mismatch")
    require(sha256_file(ROUTE_B_EVALUATOR_DEFAULT) == ROUTE_B_EVALUATOR_SHA256, "frozen Route-B evaluator identity mismatch")
    routeb_runner = ROUTE_B_SELECTION_ROOT_DEFAULT / "working" / "route_b_selection_runner.py"
    require(sha256_file(routeb_runner) == ROUTE_B_SELECTION_RUNNER_SHA256, "frozen Route-B selection runner identity mismatch")
    ledger = pd.read_csv(ledger_path)
    result: dict[str, str] = {}
    for name in ("main_results.csv", "m11_metrics_raw.csv", "evaluation_config.json"):
        rows = ledger[ledger["name"].astype(str) == name]
        require(len(rows) == 1, f"Route-B ledger entry missing/duplicate: {name}")
        path = root / name
        expected = str(rows.iloc[0]["sha256"])
        require(path.is_file() and sha256_file(path) == expected, f"Route-B output SHA mismatch: {name}")
        result[name] = expected
    return result


def build_adapted_bootstrap(rb: Any, raw: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    compatibility = raw.copy()
    compatibility.loc[compatibility["method"] == "M11-COCO", "method"] = "M11"
    result = rb.build_bootstrap(compatibility, config)
    result["comparison"] = result["comparison"].astype(str).str.replace("M11 - S_ADAPT", "M11-COCO - S_ADAPT", regex=False)
    return result


def build_domain_bootstrap(adapted_raw: pd.DataFrame, frozen_raw: pd.DataFrame) -> pd.DataFrame:
    indices, digest = frozen_bootstrap_indices()

    def group_vectors(frame: pd.DataFrame, method: str, seeds: tuple[int, ...], metric: str) -> dict[int, np.ndarray]:
        subset = frame[(frame["method"] == method) & frame["seed"].astype(int).isin(seeds)].copy()
        require(len(subset) == EXPECTED_IMAGES * len(BUDGETS) * len(seeds), f"domain bootstrap raw row count mismatch: {method}")
        grouped = subset.groupby(["seed", "budget", "group_id"], as_index=False).agg(value=(metric, "mean"), image_count=("coco_image_id", "size"))
        require(set(grouped["image_count"].astype(int)) == {40}, f"domain bootstrap group size mismatch: {method}")
        route_mean = grouped.groupby(["budget", "group_id"], as_index=False)["value"].mean()
        result: dict[int, np.ndarray] = {}
        for budget in BUDGETS:
            part = route_mean[route_mean["budget"] == budget].sort_values("group_id")
            require(len(part) == 125 and part["group_id"].astype(int).tolist() == list(range(125)), f"domain bootstrap group vector incomplete: {method}:{budget}")
            result[budget] = part["value"].to_numpy(np.float64)
        return result

    rows: list[dict[str, Any]] = []
    for metric in ("coverage", "quality"):
        adapted = group_vectors(adapted_raw, "M11-COCO", ROUTE_A_SEEDS, metric)
        frozen = group_vectors(frozen_raw, "M11", ROUTE_B_SEEDS, metric)
        unit = "coverage_per_image" if metric == "coverage" else "quality_per_image"
        for budget in BUDGETS:
            diff = adapted[budget] - frozen[budget]
            reps = diff[indices].mean(axis=1)
            rows.append({
                "comparison": "M11-COCO adapted - Frozen LC M11",
                "budget_scope": str(budget), "metric": unit, "point_estimate": float(diff.mean()),
                "ci95_lower": float(np.percentile(reps, 2.5)), "ci95_upper": float(np.percentile(reps, 97.5)),
                "positive_resample_fraction": float(np.mean(reps > 0)), "bootstrap_unit": "frozen 40-image group",
                "group_count": 125, "resamples": 5000, "bootstrap_seed": 530002,
                "bootstrap_indices_sha256": digest,
            })
        core = np.mean(np.stack([adapted[b] - frozen[b] for b in (10, 15, 20)]), axis=0)
        reps = core[indices].mean(axis=1)
        rows.append({
            "comparison": "M11-COCO adapted - Frozen LC M11",
            "budget_scope": "CORE_10_15_20", "metric": unit, "point_estimate": float(core.mean()),
            "ci95_lower": float(np.percentile(reps, 2.5)), "ci95_upper": float(np.percentile(reps, 97.5)),
            "positive_resample_fraction": float(np.mean(reps > 0)), "bootstrap_unit": "frozen 40-image group",
            "group_count": 125, "resamples": 5000, "bootstrap_seed": 530002,
            "bootstrap_indices_sha256": digest,
        })
    return pd.DataFrame(rows)


def build_evaluation_results(adapted_main: pd.DataFrame, frozen_main: pd.DataFrame) -> pd.DataFrame:
    columns = ["row_type", "route", "method", "seed", "budget", *METRICS, "reference"]
    rows: list[dict[str, Any]] = []

    for row in adapted_main.itertuples(index=False):
        rows.append({
            "row_type": "INDIVIDUAL", "route": "ROUTE_A_COCO_ADAPTED", "method": str(row.method),
            "seed": str(int(row.seed)), "budget": int(row.budget),
            **{metric: float(getattr(row, metric)) for metric in METRICS}, "reference": "",
        })
    frozen = frozen_main[frozen_main["method"] == "M11"].copy()
    require(set(frozen["seed"].astype(int)) == set(ROUTE_B_SEEDS) and len(frozen) == 15, "Frozen Route-B M11 matrix mismatch")
    for row in frozen.itertuples(index=False):
        rows.append({
            "row_type": "INDIVIDUAL", "route": "ROUTE_B_FROZEN_LC", "method": "M11-FROZEN-LC",
            "seed": str(int(row.seed)), "budget": int(row.budget),
            **{metric: float(getattr(row, metric)) for metric in METRICS}, "reference": "read-only Route-B evaluation",
        })

    adapted_m11 = adapted_main[adapted_main["method"] == "M11-COCO"]
    require(set(adapted_m11["seed"].astype(int)) == set(ROUTE_A_SEEDS) and len(adapted_m11) == 15, "adapted M11 matrix mismatch")
    for budget in BUDGETS:
        a = adapted_m11[adapted_m11["budget"] == budget]
        b = frozen[frozen["budget"] == budget]
        require(len(a) == len(b) == 3, f"three-seed budget matrix mismatch: {budget}")
        a_mean = {metric: float(a[metric].mean()) for metric in METRICS}
        b_mean = {metric: float(b[metric].mean()) for metric in METRICS}
        rows.append({
            "row_type": "THREE_SEED_MEAN", "route": "ROUTE_A_COCO_ADAPTED", "method": "M11-COCO",
            "seed": "THREE_SEED_MEAN", "budget": budget, **a_mean, "reference": "independent seed results averaged after evaluation",
        })
        rows.append({
            "row_type": "THREE_SEED_MEAN", "route": "ROUTE_B_FROZEN_LC", "method": "M11-FROZEN-LC",
            "seed": "THREE_SEED_MEAN", "budget": budget, **b_mean, "reference": "read-only Route-B evaluation",
        })
        rows.append({
            "row_type": "DOMAIN_ADAPTATION_GAIN", "route": "ROUTE_A_MINUS_ROUTE_B", "method": "M11-COCO - M11-FROZEN-LC",
            "seed": "THREE_SEED_MEAN", "budget": budget,
            **{metric: a_mean[metric] - b_mean[metric] for metric in METRICS},
            "reference": "positive means COCO adaptation improved the metric",
        })
    result = pd.DataFrame(rows, columns=columns)
    require(len(result) == 60, f"evaluation_results row count mismatch: {len(result)}")
    return result


def report_markdown(results: pd.DataFrame, domain_bootstrap: pd.DataFrame, config_sha: str) -> str:
    gains = results[results["row_type"] == "DOMAIN_ADAPTATION_GAIN"].sort_values("budget")
    adapted = results[(results["row_type"] == "THREE_SEED_MEAN") & (results["route"] == "ROUTE_A_COCO_ADAPTED")].set_index("budget")
    frozen = results[(results["row_type"] == "THREE_SEED_MEAN") & (results["route"] == "ROUTE_B_FROZEN_LC")].set_index("budget")
    lines = [
        "# COCO-adapted M11 allocator — Route A report",
        "",
        "```text",
        "routeA_status = COMPLETE",
        "detector_status = FROZEN_NO_RETRAINING",
        "routeB_status = READ_ONLY_COMPARATOR",
        "```",
        "",
        "## Domain adaptation gain",
        "",
        "正值表示 COCO-adapted M11 优于 frozen LC M11。三个 seed 始终独立预测、独立分配、独立评价，表中只在评价后取算术平均。",
        "",
        "| Budget | Coverage A | Coverage B | Gain | QUALITY A | QUALITY B | Gain | AP A | AP B | Gain | AR100 A | AR100 B | Gain |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in gains.itertuples(index=False):
        budget = int(row.budget)
        a = adapted.loc[budget]
        b = frozen.loc[budget]
        lines.append(
            f"| {budget} | {a.coverage_per_image:.6f} | {b.coverage_per_image:.6f} | {row.coverage_per_image:+.6f} "
            f"| {a.quality_per_image:.6f} | {b.quality_per_image:.6f} | {row.quality_per_image:+.6f} "
            f"| {a.AP:.6f} | {b.AP:.6f} | {row.AP:+.6f} | {a.AR100:.6f} | {b.AR100:.6f} | {row.AR100:+.6f} |"
        )
    lines.extend([
        "",
        "Coverage/QUALITY 的 Route-A − Route-B 置信区间使用同一冻结 125 个 40-image groups、5,000 次 paired bootstrap、seed 530002。AP/AP50/AP75/AR100 是完整 VAL2017 split 点估计，不平均 group AP。",
        "",
        "| Scope | Metric | Gain | 95% CI | Positive fraction |",
        "|---|---|---:|---:|---:|",
    ])
    for row in domain_bootstrap.itertuples(index=False):
        lines.append(f"| {row.budget_scope} | {row.metric} | {row.point_estimate:+.6f} | [{row.ci95_lower:+.6f}, {row.ci95_upper:+.6f}] | {row.positive_resample_fraction:.4f} |")
    lines.extend([
        "",
        "## Frozen protocol",
        "",
        "- Candidate pool: canonical original-score Road8 Top100; every output is the exact rank `1..K_i` prefix.",
        "- Groups/budgets: 125 × 40 images; mean budgets 10/15/20/30/40; every group uses exact budget.",
        "- Route A changes only target-domain PCA32, scaler, M11 weights and per-seed temperature. Architecture, loss, optimizer recipe, utility definition, DP solver and evaluation protocol stay fixed.",
        "- Common QUALITY weights retain the frozen LC FIT vector, so Route-A/Route-B gain is not confounded by a changed evaluation objective.",
        "- Route B source/results were read-only and were not edited or folded into Route-A training.",
        "- No detector training or second detector was started.",
        "",
        "## Curve artifact",
        "",
        "`predicted_utility_curves.parquet` stores every image and Route-A seed for k=1..50. Because original M11 supervises only prefix extensions k=6..50, k<5 is undefined, k=5 anchors relative utility at zero, and modeled deltas/U begin at k=6. This is exactly the additive-constant convention consumed by the frozen DP.",
        "",
        f"Evaluation config SHA256: `{config_sha}`.",
        "",
        "Detailed individual-seed/class/bootstrap/report-input tables are preserved under `working/evaluation_outputs/`.",
        "",
    ])
    return "\n".join(lines)


def publish(output_root: Path, staging: Path) -> None:
    root_names = ("evaluation_results.csv", "routeA_report.md")
    details = output_root / "working" / "evaluation_outputs"
    require(not details.exists(), f"evaluation output directory already exists: {details}")
    require(not any((output_root / name).exists() for name in root_names), "evaluation root output collision")
    for name in root_names:
        os.replace(staging / name, output_root / name)
    os.replace(staging, details)


def build_final_ledger(output_root: Path) -> pd.DataFrame:
    """Create the user-facing ledger only after all scientific outputs exist."""
    records: dict[str, dict[str, Any]] = {}
    training_ledger_path = output_root / "working" / "training_sha256_ledger.csv"
    require(training_ledger_path.is_file(), "training-stage ledger missing")
    training = pd.read_csv(training_ledger_path)
    for row in training.itertuples(index=False):
        relative = Path(str(row.path))
        path = (output_root / relative).resolve(strict=True)
        digest = sha256_file(path)
        require(digest == str(row.sha256), f"training artifact changed before final ledger: {relative.as_posix()}")
        records[relative.as_posix()] = {"path": relative.as_posix(), "sha256": digest, "bytes": path.stat().st_size, "stage": "TRAINING"}
    extra = [
        Path("working/training_sha256_ledger.csv"),
        Path("working/training_complete.json"),
        Path("selection_manifest.csv"), Path("selected_records.parquet"), Path("predicted_utility_curves.parquet"),
        Path("routea_selection_config.json"), Path("selection_QA.json"), Path("allocation_summary.csv"),
        Path("working/selection_allocations.parquet"), Path("ROUTE_A_SELECTION_COMPLETE.json"),
        Path("evaluation_results.csv"), Path("routeA_report.md"),
        Path("working/routea_pipeline_common.py"), Path("working/routea_select_val.py"), Path("working/routea_evaluate_val.py"),
    ]
    evaluation_dir = output_root / "working" / "evaluation_outputs"
    extra.extend(path.relative_to(output_root) for path in evaluation_dir.iterdir() if path.is_file())
    for relative in extra:
        key = relative.as_posix()
        path = (output_root / relative).resolve(strict=True)
        if key.startswith("working/training_"):
            stage = "TRAINING"
        elif "selection" in key.lower() or key in {"predicted_utility_curves.parquet", "allocation_summary.csv"}:
            stage = "SELECTION"
        else:
            stage = "EVALUATION"
        records[key] = {"path": key, "sha256": sha256_file(path), "bytes": path.stat().st_size, "stage": stage}
    frame = pd.DataFrame([records[key] for key in sorted(records)], columns=["path", "sha256", "bytes", "stage"])
    write_csv_atomic(output_root / "sha256_ledger.csv", frame)
    return frame


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    output_root = args.output_root.resolve()
    asset_root = args.asset_root.resolve(strict=True)
    group_manifest = args.group_manifest.resolve(strict=True)
    p1_root = args.p1_root.resolve(strict=True)
    routeb_root = args.route_b_evaluation_root.resolve(strict=True)
    evaluator_path = args.route_b_evaluator.resolve(strict=True)
    pycoco_site = args.pycocotools_site.resolve(strict=True)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "working").mkdir(parents=True, exist_ok=True)
    require(not (output_root / "ROUTE_A_EVALUATION_COMPLETE.json").exists(), "Route-A evaluation is already complete")

    selection_sha, selection_marker = validate_selection_commit(output_root)
    selection_config = validate_selection_asset_bindings(output_root, asset_root, group_manifest)
    if str(pycoco_site) not in sys.path:
        sys.path.insert(0, str(pycoco_site))
    importlib.import_module("pycocotools")
    require(sha256_file(evaluator_path) == ROUTE_B_EVALUATOR_SHA256, "Route-B evaluator SHA mismatch")
    rb = load_module("routea_readonly_routeb_evaluator", evaluator_path)
    patch_routeb_evaluator(rb, selection_sha, output_root)
    config = {
        "task": TASK,
        "created_at_utc": utc_now(),
        "execution_boundary": {"selection_regeneration": False, "evaluation_only": True, "gt_open_after_selection_validation": True},
        "dataset": {"dataset_version": DATASET_VERSION, "split": SPLIT, "image_count": EXPECTED_IMAGES, "candidate_asset_id": CANDIDATE_ASSET_ID},
        "methods": {"deterministic": ["S_FIXED", "S_ADAPT", "CAL_TEMP_ALLOC"], "m11": "M11-COCO", "m11_seeds": list(ROUTE_A_SEEDS), "seed_predictions_averaged": False},
        "budgets": list(BUDGETS),
        "bootstrap": {"unit": "frozen 40-image group", "group_count": 125, "resamples": 5000, "seed": 530002, "indices_sha256": BOOTSTRAP_INDICES_SHA256},
        "expected_hashes": {"selection_sha256": selection_sha, "gt_sha256": VAL_GT_SHA256},
        "paths": {
            "selected_records": str(output_root / "selected_records.parquet"),
            "selection_manifest": str(output_root / "selection_manifest.csv"),
            "candidate_asset_root": str(asset_root),
            "candidate_manifest": str(asset_root / "candidate_manifest.csv"),
            "image_manifest": str(asset_root / "image_manifest.csv"),
            "group_manifest": str(group_manifest),
            "gt": str(args.gt.resolve()),
            "pycocotools_site_packages": str(pycoco_site),
        },
        "input_bindings": {
            "selection_completion_marker": {"path": str(output_root / "ROUTE_A_SELECTION_COMPLETE.json"), "sha256": sha256_file(output_root / "ROUTE_A_SELECTION_COMPLETE.json")},
            "selected_records": {"path": str(output_root / "selected_records.parquet"), "sha256": selection_sha},
            "selection_manifest": {"path": str(output_root / "selection_manifest.csv"), "sha256": sha256_file(output_root / "selection_manifest.csv")},
            "selection_config": {"path": str(output_root / "routea_selection_config.json"), "sha256": sha256_file(output_root / "routea_selection_config.json")},
            "selection_time_canonical_marker_sha256": selection_config["input_bindings"]["canonical_completion_marker"]["sha256"],
            "candidate_manifest": {"path": str(asset_root / "candidate_manifest.csv"), "sha256": sha256_file(asset_root / "candidate_manifest.csv")},
            "image_manifest": {"path": str(asset_root / "image_manifest.csv"), "sha256": sha256_file(asset_root / "image_manifest.csv")},
            "group_manifest": {"path": str(group_manifest), "sha256": sha256_file(group_manifest)},
            "routea_evaluator": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__).resolve())},
            "routeb_evaluator_readonly": {"path": str(evaluator_path), "sha256": sha256_file(evaluator_path)},
            "gt_deferred": {"path": str(args.gt), "sha256": VAL_GT_SHA256, "defer_until_after_selection_validation": True},
        },
        "output_root": str(output_root),
    }
    staging = output_root / "working" / ("staging_routea_evaluation_" + selection_sha[:12])
    require(not staging.exists() or not any(staging.iterdir()), f"nonempty evaluation staging refused: {staging}")
    staging.mkdir(parents=True, exist_ok=True)
    write_json_atomic(staging / "routea_evaluation_config.json", config)
    config_sha = sha256_file(staging / "routea_evaluation_config.json")

    groups, images = rb.load_group_and_image_manifests(config)
    candidates, by_image = rb.load_candidates(config)
    allocation = rb.validate_frozen_selections(config, candidates, groups)
    progress("selection_validation", "PASS; selection is frozen and GT has not been opened")

    gt_path = args.gt.resolve(strict=True)
    require(sha256_file(gt_path) == VAL_GT_SHA256, "VAL GT SHA mismatch")
    coco_subset, gt_by_image, gt_summary = rb.load_official_gt(config, set(images["coco_image_id"].astype(int)))
    progress("gt_open", "official VAL GT opened only after selection validation")
    p1_core, _, _ = load_frozen_p1_modules(p1_root)
    weights, _ = load_class_weights(output_root / "class_weight_config.json")
    prefix_cache = rb.build_prefix_cache(p1_core, by_image, gt_by_image, output_root)
    adapted_raw, class_acc = rb.evaluate_m11_metrics(allocation, prefix_cache, weights, gt_summary, output_root)
    coco_results = rb.evaluate_all_coco(config, allocation, by_image, images, coco_subset, output_root)
    adapted_main, adapted_classes = rb.build_main_and_class_tables(adapted_raw, class_acc, coco_results, gt_summary, config_sha)
    adapted_bootstrap = build_adapted_bootstrap(rb, adapted_raw, config)
    compatibility_allocation = allocation.copy()
    compatibility_allocation.loc[compatibility_allocation["method"] == "M11-COCO", "method"] = "M11"
    selection_analysis = rb.build_selection_analysis(compatibility_allocation)
    selection_analysis = selection_analysis.rename(columns={"m11_seed": "m11_coco_seed", "K_m11": "K_m11_coco", "delta_K_m11_minus_s_adapt": "delta_K_m11_coco_minus_s_adapt"})

    routeb_hashes = routeb_ledger_hashes(routeb_root)
    frozen_main = pd.read_csv(routeb_root / "main_results.csv")
    frozen_raw = pd.read_csv(routeb_root / "m11_metrics_raw.csv")
    domain_bootstrap = build_domain_bootstrap(adapted_raw, frozen_raw)
    evaluation_results = build_evaluation_results(adapted_main, frozen_main)
    write_csv_atomic(staging / "evaluation_results.csv", evaluation_results)
    write_csv_atomic(staging / "adapted_main_results.csv", adapted_main)
    write_csv_atomic(staging / "adapted_class_results.csv", adapted_classes)
    write_csv_atomic(staging / "adapted_bootstrap_results.csv", adapted_bootstrap)
    write_csv_atomic(staging / "domain_adaptation_bootstrap.csv", domain_bootstrap)
    pq.write_table(pa.Table.from_pandas(adapted_raw, preserve_index=False), staging / "adapted_m11_metrics_raw.parquet", compression="zstd")
    pq.write_table(pa.Table.from_pandas(selection_analysis, preserve_index=False), staging / "adapted_selection_analysis.parquet", compression="zstd")
    write_json_atomic(staging / "adapted_coco_metrics_raw.json", {
        "conditions": {f"{method}|{seed}|{budget}": value for (method, seed, budget), value in coco_results.items()}
    })
    write_json_atomic(staging / "routeb_readonly_bindings.json", routeb_hashes)
    report = report_markdown(evaluation_results, domain_bootstrap, config_sha)
    write_text_atomic(staging / "routeA_report.md", report)

    completion_files = [
        "evaluation_results.csv", "routeA_report.md", "routea_evaluation_config.json", "adapted_main_results.csv",
        "adapted_class_results.csv", "adapted_bootstrap_results.csv", "domain_adaptation_bootstrap.csv",
        "adapted_m11_metrics_raw.parquet", "adapted_selection_analysis.parquet", "adapted_coco_metrics_raw.json",
        "routeb_readonly_bindings.json",
    ]
    completion = {
        "task": TASK,
        "status": "ROUTE_A_EVALUATION_COMPLETE",
        "routeA_status": "COMPLETE",
        "created_at_utc": utc_now(),
        "selection_sha256": selection_sha,
        "evaluation_config_sha256": config_sha,
        "routeb_inputs_read_only": True,
        "detector_forward": False,
        "detector_training": False,
        "files": {name: sha256_file(staging / name) for name in completion_files},
        "elapsed_seconds": time.perf_counter() - started,
    }
    publish(output_root, staging)
    ledger = build_final_ledger(output_root)
    completion["sha256_ledger_entries"] = int(len(ledger))
    completion["sha256_ledger_sha256"] = sha256_file(output_root / "sha256_ledger.csv")
    write_json_atomic(output_root / "ROUTE_A_EVALUATION_COMPLETE.json", completion)
    append_log(output_root, "ROUTE_A_EVALUATION_COMPLETE")
    return completion


def self_test() -> dict[str, Any]:
    common = self_test_common()
    rows = []
    for method, seeds in (("S_FIXED", (-1,)), ("S_ADAPT", (-1,)), ("CAL_TEMP_ALLOC", (-1,)), ("M11-COCO", ROUTE_A_SEEDS)):
        for seed in seeds:
            for budget in BUDGETS:
                rows.append({"method": method, "seed": seed, "budget": budget, **{metric: 1.0 + budget / 100.0 + seed / 1e9 for metric in METRICS}})
    adapted = pd.DataFrame(rows)
    frozen = pd.DataFrame([
        {"method": "M11", "seed": seed, "budget": budget, **{metric: 1.0 for metric in METRICS}}
        for seed in ROUTE_B_SEEDS for budget in BUDGETS
    ])
    result = build_evaluation_results(adapted, frozen)
    require(len(result[result.row_type == "DOMAIN_ADAPTATION_GAIN"]) == 5, "synthetic gain rows")
    return {"status": "PASS", "common": common, "evaluation_rows": len(result)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=ROOT_DEFAULT)
    parser.add_argument("--asset-root", type=Path, default=ASSET_ROOT_DEFAULT)
    parser.add_argument("--group-manifest", type=Path, default=GROUP_MANIFEST_DEFAULT)
    parser.add_argument("--p1-root", type=Path, default=P1_ROOT_DEFAULT)
    parser.add_argument("--route-b-evaluation-root", type=Path, default=ROUTE_B_EVALUATION_ROOT_DEFAULT)
    parser.add_argument("--route-b-evaluator", type=Path, default=ROUTE_B_EVALUATOR_DEFAULT)
    parser.add_argument("--pycocotools-site", type=Path, default=PYCOCO_SITE_DEFAULT)
    parser.add_argument("--gt", type=Path, default=VAL_GT_DEFAULT)
    parser.add_argument("--self-test", action="store_true", help="run synthetic tests only; opens no dataset or GT")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    result = self_test() if arguments.self_test else run(arguments)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
