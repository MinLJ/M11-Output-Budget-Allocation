"""Paired complete-group summaries for LC-ALLOC-BENCH-P0.

The script never opens candidate, prediction, or ground-truth assets.  It
consumes only the committed evaluation aggregates produced by
``evaluate_bench.py``.  Bootstrap samples are generated (or the frozen P1
sample is copied) before result tables are read, and every fixed
permutation/group-size configuration is resampled separately.
"""
from __future__ import annotations

import argparse
import shutil
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

from common import (  # noqa: E402
    BUDGETS,
    P1,
    P1B,
    ROOT,
    SEEDS,
    append_log,
    sha256_file,
    stable_config_seed,
    write_json,
    write_parquet,
)


CORE = (10, 15, 20)
PERMUTATIONS = ("R0", "R1", "R2")
GROUP_SIZES = (10, 20, 40, 80)
METRICS = ("coverage_per_image", "quality_per_image")


def _prepare_bootstrap_indices() -> tuple[dict[tuple[str, int], np.ndarray], dict[tuple[str, int], dict]]:
    """Freeze all sampling indices before any result value is loaded."""
    out_dir = ROOT / "working" / "bootstrap_indices"
    out_dir.mkdir(parents=True, exist_ok=True)
    frozen_p1 = P1 / "outputs" / "bootstrap_group_indices.npy"
    result: dict[tuple[str, int], np.ndarray] = {}
    meta: dict[tuple[str, int], dict] = {}
    for permutation in PERMUTATIONS:
        for group_size in GROUP_SIZES:
            group_count = 2000 // group_size
            target = out_dir / f"{permutation}_n{group_size}.npy"
            source = "BENCH_P0_DETERMINISTIC_CONFIG_SEED"
            seed = stable_config_seed(permutation, group_size)
            if permutation == "R0" and group_size == 40:
                if not frozen_p1.exists():
                    raise FileNotFoundError(frozen_p1)
                if not target.exists() or sha256_file(target) != sha256_file(frozen_p1):
                    shutil.copyfile(frozen_p1, target)
                source = "REUSED_FROZEN_P1_BOOTSTRAP_INDICES"
                seed = 530002
            elif not target.exists():
                rng = np.random.default_rng(seed)
                np.save(target, rng.integers(0, group_count, size=(5000, group_count), endpoint=False, dtype=np.int16))
            indices = np.load(target, allow_pickle=False)
            if indices.shape != (5000, group_count) or indices.min() < 0 or indices.max() >= group_count:
                raise RuntimeError(f"invalid bootstrap indices {target}: {indices.shape}")
            result[(permutation, group_size)] = indices
            meta[(permutation, group_size)] = {
                "permutation": permutation,
                "group_size": group_size,
                "group_count": group_count,
                "resamples": 5000,
                "seed": seed,
                "source": source,
                "path": str(target),
                "sha256": sha256_file(target),
            }
    write_json(ROOT / "working" / "bootstrap_indices_manifest.json", list(meta.values()))
    return result, meta


def _group_vector(frame: pd.DataFrame, method: str, seed: int | str, budget: int, metric: str, group_count: int) -> np.ndarray:
    d = frame[(frame["method"] == method) & (frame["budget"].astype(int) == int(budget))]
    if seed == "MEAN":
        d = d[d["seed"].astype(int).isin(SEEDS)].groupby("group_id", as_index=False)[metric].mean()
    else:
        d = d[d["seed"].astype(int) == int(seed)][["group_id", metric]]
    d = d.sort_values("group_id", kind="stable")
    if len(d) != group_count or d["group_id"].astype(int).tolist() != list(range(group_count)):
        raise RuntimeError(f"incomplete group vector method={method} seed={seed} budget={budget} metric={metric}")
    return d[metric].to_numpy(np.float64)


def _comparison_rows(
    frame: pd.DataFrame,
    indices: np.ndarray,
    *,
    permutation: str,
    group_size: int,
    index_meta: dict,
    comparison: str,
    left: str,
    right: str,
    left_learned: bool,
    right_learned: bool,
    primary: bool,
    metrics: tuple[str, ...] = METRICS,
) -> list[dict]:
    group_count = 2000 // group_size
    variants: list[int | str] = ["MEAN"] + list(SEEDS) if left_learned or right_learned else [-1]
    rows: list[dict] = []
    for variant in variants:
        left_seed: int | str = variant if left_learned else -1
        right_seed: int | str = variant if right_learned else -1
        variant_name = "THREE_SEED_MEAN" if variant == "MEAN" else (f"SEED_{variant}" if int(variant) != -1 else "DETERMINISTIC")
        for metric in metrics:
            differences = {
                budget: _group_vector(frame, left, left_seed, budget, metric, group_count)
                - _group_vector(frame, right, right_seed, budget, metric, group_count)
                for budget in BUDGETS
            }
            scopes: list[tuple[str, np.ndarray]] = [(str(b), differences[b]) for b in BUDGETS]
            scopes.append(("CORE_10_15_20", np.mean(np.vstack([differences[x] for x in CORE]), axis=0)))
            for scope, difference in scopes:
                boot = difference[indices].mean(axis=1)
                rows.append({
                    "row_type": "PAIRED_GROUP_BOOTSTRAP",
                    "permutation": permutation,
                    "group_size": group_size,
                    "group_count": group_count,
                    "comparison": comparison,
                    "left_method": left,
                    "right_method": right,
                    "variant": variant_name,
                    "scope": scope,
                    "metric": metric,
                    "primary_comparison": bool(primary),
                    "observed_delta_per_image": float(difference.mean()),
                    "observed_delta_per_100_images": float(100.0 * difference.mean()),
                    "ci95_low": float(np.quantile(boot, 0.025)),
                    "ci95_high": float(np.quantile(boot, 0.975)),
                    "strict_positive_resample_fraction": float(np.mean(boot > 0.0)),
                    "bootstrap_unit": f"complete frozen {group_size}-image allocation group",
                    "bootstrap_resamples": int(len(indices)),
                    "bootstrap_seed": int(index_meta["seed"]),
                    "bootstrap_index_sha256": str(index_meta["sha256"]),
                    "sensitivity_min": np.nan,
                    "sensitivity_max": np.nan,
                    "direction_positive_count": np.nan,
                    "direction_nonnegative_count": np.nan,
                    "notes": "paired within one fixed permutation and group size; no cross-permutation pooling",
                })
    return rows


def _add_permutation_sensitivity(rows: pd.DataFrame) -> pd.DataFrame:
    """Descriptive across-permutation range only; never a confidence interval."""
    base = rows[
        (rows["row_type"] == "PAIRED_GROUP_BOOTSTRAP")
        & rows["variant"].isin(["THREE_SEED_MEAN", "DETERMINISTIC"])
    ]
    sensitivity: list[dict] = []
    keys = ["group_size", "comparison", "left_method", "right_method", "variant", "scope", "metric", "primary_comparison"]
    for key, d in base.groupby(keys, dropna=False, sort=False):
        if set(d["permutation"].astype(str)) != set(PERMUTATIONS) or len(d) != 3:
            raise RuntimeError(f"permutation sensitivity incomplete: {key}")
        values = d["observed_delta_per_image"].to_numpy(np.float64)
        group_size = int(key[0])
        sensitivity.append({
            "row_type": "PERMUTATION_SENSITIVITY_DESCRIPTION",
            "permutation": "ALL_R0_R1_R2",
            "group_size": group_size,
            "group_count": 2000 // group_size,
            "comparison": key[1],
            "left_method": key[2],
            "right_method": key[3],
            "variant": key[4],
            "scope": key[5],
            "metric": key[6],
            "primary_comparison": bool(key[7]),
            "observed_delta_per_image": float(values.mean()),
            "observed_delta_per_100_images": float(100.0 * values.mean()),
            "ci95_low": np.nan,
            "ci95_high": np.nan,
            "strict_positive_resample_fraction": np.nan,
            "bootstrap_unit": "NOT_APPLICABLE",
            "bootstrap_resamples": 0,
            "bootstrap_seed": np.nan,
            "bootstrap_index_sha256": None,
            "sensitivity_min": float(values.min()),
            "sensitivity_max": float(values.max()),
            "direction_positive_count": int(np.sum(values > 0.0)),
            "direction_nonnegative_count": int(np.sum(values >= 0.0)),
            "notes": "same DEV2000 under three deterministic regroupings; descriptive mean/range, not CI or 6000 independent images",
        })
    return pd.concat([rows, pd.DataFrame(sensitivity)], ignore_index=True)


def summarize_a(indices_map: dict, meta_map: dict) -> pd.DataFrame:
    path = ROOT / "working" / "group_robustness_group_results.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    group = pq.read_table(path).to_pandas()
    expected_rows = sum((2000 // n) * 5 * 5 for n in GROUP_SIZES) * len(PERMUTATIONS)
    if len(group) != expected_rows:
        raise RuntimeError(f"A group aggregate rows {len(group)} != {expected_rows}")
    rows: list[dict] = []
    for permutation in PERMUTATIONS:
        for group_size in GROUP_SIZES:
            d = group[(group["permutation"] == permutation) & (group["group_size"].astype(int) == group_size)]
            indices, meta = indices_map[(permutation, group_size)], meta_map[(permutation, group_size)]
            rows.extend(_comparison_rows(
                d, indices, permutation=permutation, group_size=group_size, index_meta=meta,
                comparison="M11_MINUS_S_ADAPT", left="M11", right="S_ADAPT",
                left_learned=True, right_learned=False, primary=True,
            ))
            rows.extend(_comparison_rows(
                d, indices, permutation=permutation, group_size=group_size, index_meta=meta,
                comparison="S_ADAPT_MINUS_S_FIXED", left="S_ADAPT", right="S_FIXED",
                left_learned=False, right_learned=False, primary=True,
            ))
    result = _add_permutation_sensitivity(pd.DataFrame(rows))
    write_parquet(result, ROOT / "outputs" / "group_robustness_comparisons.parquet")
    return result


def _normalize_external_group() -> pd.DataFrame:
    b_path = ROOT / "working" / "external_baseline_group_results.parquet"
    if not b_path.exists():
        raise FileNotFoundError(b_path)
    external = pq.read_table(b_path).to_pandas()
    external["permutation"] = "R0"
    external["group_size"] = 40
    a = pq.read_table(ROOT / "working" / "group_robustness_group_results.parquet").to_pandas()
    a = a[(a["permutation"] == "R0") & (a["group_size"].astype(int) == 40)]
    a = a[a["method"].isin(["M11", "S_ADAPT", "S_FIXED"])]
    p1b = pq.read_table(P1B / "outputs" / "baseline_group.parquet").to_pandas()
    p1b = p1b[p1b["method"] == "S_CLASS_ADAPT"].copy()
    p1b["permutation"] = "R0"
    p1b["group_size"] = 40
    keep = ["permutation", "group_size", "method", "seed", "budget", "group_id", "image_count", "GT", "coverage", "quality", "legacy_TP", "output_records", "coverage_per_image", "quality_per_image"]
    for name, frame in (("external", external), ("A", a), ("P1B", p1b)):
        absent = sorted(set(keep) - set(frame.columns))
        if absent:
            raise RuntimeError(f"{name} group table missing columns {absent}")
    combined = pd.concat([external[keep], a[keep], p1b[keep]], ignore_index=True)
    if combined.duplicated(["method", "seed", "budget", "group_id"]).any():
        raise RuntimeError("external comparison group identities are duplicated")
    return combined


def summarize_b(indices: np.ndarray, meta: dict) -> pd.DataFrame:
    frame = _normalize_external_group()
    specs = [
        ("M11_MINUS_PS_CLASS_PREFIX", "M11", "PS_CLASS_PREFIX", True, False, True),
        ("M11_MINUS_PS_PREFIX", "M11", "PS_PREFIX", True, False, False),
        ("PS_PREFIX_MINUS_S_ADAPT", "PS_PREFIX", "S_ADAPT", False, False, False),
        ("PS_CLASS_PREFIX_MINUS_S_ADAPT", "PS_CLASS_PREFIX", "S_ADAPT", False, False, False),
        ("PS_CLASS_PREFIX_MINUS_S_CLASS_ADAPT", "PS_CLASS_PREFIX", "S_CLASS_ADAPT", False, False, False),
    ]
    rows: list[dict] = []
    for comparison, left, right, left_learned, right_learned, primary in specs:
        rows.extend(_comparison_rows(
            frame, indices, permutation="R0", group_size=40, index_meta=meta,
            comparison=comparison, left=left, right=right,
            left_learned=left_learned, right_learned=right_learned, primary=primary,
        ))
    result = pd.DataFrame(rows)
    write_parquet(result, ROOT / "outputs" / "external_baseline_comparisons.parquet")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=("A", "B", "ALL"), default="ALL")
    args = parser.parse_args()
    start = time.perf_counter()
    append_log(f"STAGE summarize START scope={args.scope}")
    # Required execution order: sampling identities precede reading scientific results.
    indices_map, meta_map = _prepare_bootstrap_indices()
    qa: dict[str, object] = {"status": "PASS", "scope": args.scope}
    prior_path = ROOT / "qa" / "summary_validation.json"
    if args.scope == "B" and prior_path.exists():
        import json
        prior = json.loads(prior_path.read_text(encoding="utf-8"))
        if prior.get("status") == "PASS" and "A_comparison_rows" in prior:
            qa["A_comparison_rows"] = prior["A_comparison_rows"]
    if args.scope in ("A", "ALL"):
        a = summarize_a(indices_map, meta_map)
        qa["A_comparison_rows"] = len(a)
    if args.scope in ("B", "ALL"):
        b_path = ROOT / "working" / "external_baseline_group_results.parquet"
        if not b_path.exists():
            if args.scope == "B":
                raise FileNotFoundError(b_path)
            append_log("SUMMARIZE B evaluation absent; continuing with A only")
        else:
            b = summarize_b(indices_map[("R0", 40)], meta_map[("R0", 40)])
            qa["B_comparison_rows"] = len(b)
    qa["elapsed_seconds"] = time.perf_counter() - start
    write_json(ROOT / "qa" / "summary_validation.json", qa)
    append_log(f"STAGE summarize COMPLETE scope={args.scope} elapsed_seconds={qa['elapsed_seconds']:.6f}")


if __name__ == "__main__":
    main()
