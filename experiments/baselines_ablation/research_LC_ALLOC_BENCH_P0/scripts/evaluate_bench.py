"""Evaluate committed LC-ALLOC-BENCH-P0 DEV prefix selections.

Selection/allocation files are byte-hash verified before this entrypoint opens
DEV ground truth.  No model or detector forward pass is performed here.
"""
from __future__ import annotations
import os

import argparse
import hashlib
import json
import math
import sys
import time
from collections import defaultdict
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
    R0,
    RELEASE,
    ROAD8,
    ROOT,
    SEEDS,
    append_log,
    import_file,
    load_candidate_frame,
    load_identity,
    sha256_file,
    write_json,
    write_parquet,
)


GROUP_KEYS = ["permutation", "group_size", "method", "seed", "budget"]
CONDITION_KEYS = GROUP_KEYS


def _read_commit(path: Path, *, expected_rows: int) -> tuple[Path, dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    commit = json.loads(path.read_text(encoding="utf-8"))
    raw_path = commit.get("path", commit.get("allocation_path"))
    if not raw_path:
        raise RuntimeError(f"allocation path missing from {path}")
    allocation_path = Path(raw_path).resolve()
    if not allocation_path.exists():
        raise RuntimeError(f"committed allocation is absent: {allocation_path}")
    actual = sha256_file(allocation_path)
    expected = str(commit.get("sha256", commit.get("allocation_sha256", ""))).lower()
    if actual != expected:
        raise RuntimeError(f"committed allocation hash mismatch: {allocation_path}")
    rows = pq.ParquetFile(allocation_path).metadata.num_rows
    declared_rows = commit.get("rows", commit.get("allocation_rows", rows))
    if rows != expected_rows or int(declared_rows) != expected_rows:
        raise RuntimeError(f"allocation row count mismatch: {rows} != {expected_rows}")
    return allocation_path, commit


def _verify_a_before_gt() -> Path:
    path, commit = _read_commit(ROOT / "outputs" / "group_selection_commit.json", expected_rows=600_000)
    qa_path = ROOT / "qa" / "allocation_validation.json"
    qa = json.loads(qa_path.read_text(encoding="utf-8"))
    if qa.get("status") != "PASS" or qa.get("selection_sha256") != commit["sha256"]:
        raise RuntimeError("group allocation validation is absent or does not bind the committed bytes")
    required = {
        "allocation_rows": 600_000,
        "r0_n40_k_mismatch": 0,
        "r0_n40_selected_id_mismatch": 0,
        "n1_mismatch": 0,
        "threshold_k_mismatch": 0,
        "threshold_objective_mismatch": 0,
    }
    for key, value in required.items():
        if int(qa.get(key, -1)) != value:
            raise RuntimeError(f"group allocation QA invariant failed: {key}")
    return path


def _verify_b_before_gt() -> Path | None:
    commit_path = ROOT / "qa" / "external_baseline_selection_freeze.json"
    if not commit_path.exists():
        return None
    path, commit = _read_commit(commit_path, expected_rows=20_000)
    qa = json.loads((ROOT / "qa" / "external_baseline_allocation_validation.json").read_text(encoding="utf-8"))
    if not qa.get("all_checks_pass") or int(qa.get("dev_gt_open_count", -1)) != 0:
        raise RuntimeError("external baseline pre-evaluation validation failed")
    if int(commit.get("dev_gt_open_count_at_commit", -1)) != 0:
        raise RuntimeError("external baseline selections were not committed before DEV GT")
    return path


def _load_candidates() -> tuple[pd.DataFrame, dict[str, list[dict]], dict[str, list[str]]]:
    frame = load_candidate_frame("DEV", max_rank=100)
    by_image: dict[str, list[dict]] = {}
    candidate_ids: dict[str, list[str]] = {}
    for image_id, rows in frame.groupby("image_id", sort=False):
        records = rows.sort_values("road8_rank", kind="stable").to_dict("records")
        by_image[str(image_id)] = records
        candidate_ids[str(image_id)] = [str(x["candidate_record_id"]) for x in records]
    return frame, by_image, candidate_ids


def _validate_group_membership(d: pd.DataFrame, *, expected_condition_rows: int) -> None:
    """Validate allocation identities against the frozen pre-GT group manifest."""
    manifest = pq.read_table(ROOT / "outputs" / "group_manifest.parquet").to_pandas()
    manifest["image_id"] = manifest["image_id"].astype(str)
    configs = d[["permutation", "group_size"]].drop_duplicates()
    manifest = manifest.merge(configs, on=["permutation", "group_size"], how="inner")
    membership_keys = ["permutation", "group_size", "group_id", "image_id"]
    actual_membership = d[membership_keys].drop_duplicates()
    frozen_membership = manifest[membership_keys].drop_duplicates()
    merged = actual_membership.merge(frozen_membership, on=membership_keys, how="outer", indicator=True)
    if len(actual_membership) != len(frozen_membership) or not (merged["_merge"] == "both").all():
        raise RuntimeError("allocation/group-manifest membership mismatch")
    condition = d.groupby(CONDITION_KEYS, sort=False).agg(rows=("image_id", "size"), images=("image_id", "nunique")).reset_index()
    if len(condition) != expected_condition_rows or not ((condition["rows"] == 2000) & (condition["images"] == 2000)).all():
        raise RuntimeError("allocation condition image coverage is not exactly DEV2000")
    grouped = d.groupby(CONDITION_KEYS + ["group_id"], sort=False).agg(rows=("image_id", "size"), images=("image_id", "nunique")).reset_index()
    if not ((grouped["rows"] == grouped["group_size"]) & (grouped["images"] == grouped["group_size"])).all():
        raise RuntimeError("allocation group cardinality does not match frozen group_size")


def _validate_committed_prefix_lists(path: Path, candidate_ids: dict[str, list[str]], *, expected_rows: int) -> int:
    """Stream the list column so evaluation is bound to committed selections."""
    checked = 0
    for batch in pq.ParquetFile(path).iter_batches(
        batch_size=4096, columns=["image_id", "K_i", "selected_record_ids"]
    ):
        frame = batch.to_pandas()
        for image_id, k, selected in zip(frame["image_id"], frame["K_i"], frame["selected_record_ids"]):
            image_id = str(image_id)
            k = int(k)
            if image_id not in candidate_ids:
                raise RuntimeError(f"foreign image identity in committed selection: {image_id}")
            selected_tuple = tuple(str(x) for x in selected)
            expected_tuple = tuple(candidate_ids[image_id][:k])
            if len(selected_tuple) != k or selected_tuple != expected_tuple or len(set(selected_tuple)) != k:
                raise RuntimeError(f"committed selection is not the canonical prefix: {image_id} K={k}")
            checked += 1
    if checked != expected_rows:
        raise RuntimeError(f"committed selection validation rows={checked} != {expected_rows}")
    return checked


def _load_a_numeric(path: Path, candidate_ids: dict[str, list[str]]) -> tuple[pd.DataFrame, dict]:
    columns = [
        "experiment", "permutation", "group_size", "group_id", "method", "seed",
        "budget", "image_id", "K_i", "group_predicted_objective", "source",
    ]
    d = pq.read_table(path, columns=columns).to_pandas()
    d["image_id"] = d["image_id"].astype(str)
    expected_methods = {("M11", x) for x in SEEDS} | {("S_ADAPT", -1), ("S_FIXED", -1)}
    got_methods = set(zip(d["method"].astype(str), d["seed"].astype(int)))
    if len(d) != 600_000 or got_methods != expected_methods:
        raise RuntimeError("A allocation method/row invariant failed")
    if set(d["permutation"]) != {"R0", "R1", "R2"} or set(d["group_size"].astype(int)) != {10, 20, 40, 80}:
        raise RuntimeError("A grouping configuration invariant failed")
    if set(d["budget"].astype(int)) != set(BUDGETS) or not d["K_i"].between(5, 50).all():
        raise RuntimeError("A budget/K invariant failed")
    condition_count = len(d[CONDITION_KEYS].drop_duplicates())
    if condition_count != 300:
        raise RuntimeError(f"A condition count {condition_count} != 300")
    _validate_group_membership(d, expected_condition_rows=300)
    sums = d.groupby(CONDITION_KEYS + ["group_id"], sort=False)["K_i"].sum().reset_index()
    if not np.array_equal(sums["K_i"].to_numpy(np.int64), (sums["group_size"] * sums["budget"]).to_numpy(np.int64)):
        raise RuntimeError("A exact group budget invariant failed")
    checked = _validate_committed_prefix_lists(path, candidate_ids, expected_rows=600_000)
    append_log(f"A_COMMITTED_PREFIX_VALIDATION_PASS rows={checked}")
    return d, {"committed_prefix_rows_checked": checked, "group_manifest_membership": "EXACT"}


def _load_b(path: Path, candidate_ids: dict[str, list[str]]) -> pd.DataFrame:
    d = pq.read_table(path).to_pandas()
    d["image_id"] = d["image_id"].astype(str)
    expected = {("PS_PREFIX", -1), ("PS_CLASS_PREFIX", -1)}
    if len(d) != 20_000 or set(zip(d["method"].astype(str), d["seed"].astype(int))) != expected:
        raise RuntimeError("B allocation method/row invariant failed")
    if set(d["budget"].astype(int)) != set(BUDGETS) or set(d["permutation"].astype(str)) != {"R0"} or set(d["group_size"].astype(int)) != {40}:
        raise RuntimeError("B protocol identity invariant failed")
    if not d["K_i"].between(5, 50).all():
        raise RuntimeError("B K bound invariant failed")
    prefix_bad = 0
    duplicate_bad = 0
    for row in d.itertuples(index=False):
        selected = list(row.selected_record_ids)
        expected_ids = candidate_ids[str(row.image_id)][: int(row.K_i)]
        prefix_bad += int(selected != expected_ids or len(selected) != int(row.K_i))
        duplicate_bad += int(len(selected) != len(set(selected)))
    if prefix_bad or duplicate_bad:
        raise RuntimeError(f"B selected prefixes invalid prefix={prefix_bad} duplicate={duplicate_bad}")
    _validate_group_membership(d, expected_condition_rows=10)
    sums = d.groupby(["method", "seed", "budget", "group_id"], sort=False)["K_i"].sum().reset_index()
    if not np.array_equal(sums["K_i"].to_numpy(np.int64), (40 * sums["budget"]).to_numpy(np.int64)):
        raise RuntimeError("B exact group budget invariant failed")
    rename = {"predicted_objective": "group_predicted_objective"}
    d = d.rename(columns={k: v for k, v in rename.items() if k in d.columns and v not in d.columns})
    if "source" not in d:
        d["source"] = "NEW_BENCH_P0_EXTERNAL_CALIBRATION"
    return d


def _cache_paths() -> tuple[Path, Path]:
    return ROOT / "working" / "dev_prefix_metrics.npz", ROOT / "working" / "dev_prefix_metrics_meta.json"


def _prepare_gt_and_prefix_cache(
    candidate_by_image: dict[str, list[dict]],
    r0,
    prefix_matching_counts,
) -> tuple[list[str], dict, dict, dict, dict[str, np.ndarray]]:
    """Open DEV GT only after all present selection assets were committed."""
    identity = load_identity()
    manifest = pq.read_table(RELEASE / identity["assets"]["DEV"]["split_manifest_path"]).to_pylist()
    ids = [str(x["image_id"]) for x in manifest]
    gt_path = RELEASE / "gt" / "DEV2K_ROAD8_GT.json"
    gt_sha = sha256_file(gt_path)
    evaluator_sha = sha256_file(RELEASE / "evaluator" / "shared_evaluator.py")
    with gt_path.open("r", encoding="utf-8") as handle:
        raw_gt = json.load(handle)
    shared = r0.load_shared(RELEASE)
    local_gt, coco_gt, remap = r0.prepare_gt(raw_gt, manifest, shared)
    del raw_gt
    if len(ids) != 2000 or sum(len(local_gt[x]) for x in ids) != 29_966 or sum(not local_gt[x] for x in ids) != 3:
        raise RuntimeError("frozen DEV GT identity/count invariant failed")

    cache_path, meta_path = _cache_paths()
    expected_meta = {
        "dev_gt_sha256": gt_sha,
        "evaluator_sha256": evaluator_sha,
        "candidate_asset_identity_sha256": sha256_file(RELEASE / "CANDIDATE_ASSET_IDENTITY.json"),
        "candidate_asset_id_dev": str(identity["assets"]["DEV"]["candidate_asset_id"]),
        "class_weights_sha256": sha256_file(P1 / "models" / "class_weights.json"),
        "r0_evaluation_wrapper_sha256": sha256_file(R0 / "scripts" / "evaluate_alloc.py"),
        "p1_core_sha256": sha256_file(P1 / "scripts" / "p1_core.py"),
        "image_count": 2000,
        "valid_road8_gt": 29_966,
        "max_k": 50,
    }
    cache: dict[str, np.ndarray]
    if cache_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if all(meta.get(k) == v for k, v in expected_meta.items()) and meta.get("cache_sha256") == sha256_file(cache_path):
            with np.load(cache_path, allow_pickle=False) as z:
                cache = {name: z[name].copy() for name in z.files}
            if cache["image_ids"].astype(str).tolist() != ids:
                raise RuntimeError("prefix cache image identity mismatch")
            append_log("EVALUATION prefix cache reused")
            return ids, local_gt, coco_gt, remap, cache

    weights = np.asarray(json.loads((P1 / "models" / "class_weights.json").read_text(encoding="utf-8"))["weights"], dtype=np.float64)
    n = len(ids)
    gt_count = np.zeros(n, dtype=np.int32)
    gt_class = np.zeros((n, 8), dtype=np.int32)
    pred_class_prefix = np.zeros((n, 51, 8), dtype=np.int16)
    coverage = np.zeros((n, 51), dtype=np.int16)
    legacy = np.zeros((n, 51), dtype=np.int16)
    coverage_class = np.zeros((n, 51, 8), dtype=np.int16)
    legacy_class = np.zeros((n, 51, 8), dtype=np.int16)
    quality = np.zeros((n, 51), dtype=np.float64)
    anchor_parity = 0
    for image_ix, image_id in enumerate(ids):
        rows = candidate_by_image[image_id]
        cand_obj = [shared.Candidate.from_mapping(x) for x in rows]
        graph = shared.build_image_graph(cand_obj, local_gt[image_id])
        cand_classes = np.asarray([int(x["predicted_road8_class_id"]) for x in rows], dtype=np.int16)
        cand_boxes = np.asarray([[x["bbox_x1"], x["bbox_y1"], x["bbox_x2"], x["bbox_y2"]] for x in rows], dtype=np.float64)
        gt_classes = np.asarray([int(x.category_id) for x in local_gt[image_id]], dtype=np.int16)
        gt_boxes = np.asarray([x.bbox_xyxy for x in local_gt[image_id]], dtype=np.float64).reshape(-1, 4)
        totals, by_class_t = prefix_matching_counts(cand_classes, cand_boxes, gt_classes, gt_boxes, max_k=50)
        coverage[image_ix] = totals[:, 0].astype(np.int16)
        coverage_class[image_ix] = by_class_t[:, :, 0].astype(np.int16)
        quality[image_ix] = np.sum(weights[None, :, None] * by_class_t.astype(np.float64), axis=1).mean(axis=1)
        gt_count[image_ix] = len(local_gt[image_id])
        gt_class[image_ix] = np.bincount(gt_classes, minlength=9)[1:9]
        for k in range(1, 51):
            pred_class_prefix[image_ix, k] = pred_class_prefix[image_ix, k - 1]
            pred_class_prefix[image_ix, k, cand_classes[k - 1] - 1] += 1

        # Frozen rank-order greedy semantics admit an exact incremental prefix cache.
        matched: set[int] = set()
        running_class = np.zeros(8, dtype=np.int16)
        for k in range(1, 51):
            ci = k - 1
            best_gt, best_iou = -1, -1.0
            for gi in graph.adjacency[ci]:
                if gi in matched:
                    continue
                overlap = float(graph.iou_matrix[ci, gi])
                if overlap > best_iou:
                    best_iou, best_gt = overlap, gi
            if best_gt >= 0:
                matched.add(best_gt)
                running_class[int(graph.ground_truth[best_gt].category_id) - 1] += 1
            legacy[image_ix, k] = len(matched)
            legacy_class[image_ix, k] = running_class

        # Six fixed anchors per image check both independent prefix caches against
        # the exact shared evaluator, without using any method outcome.
        for k in (5, 10, 20, 30, 40, 50):
            result = shared.evaluate_selected(list(range(1, k + 1)), graph)
            anchor_parity += int(result.coverage_matching.cardinality != int(coverage[image_ix, k]))
            anchor_parity += int(result.legacy_matching.cardinality != int(legacy[image_ix, k]))
        if (image_ix + 1) % 250 == 0:
            append_log(f"PREFIX_CACHE_PROGRESS images={image_ix + 1}")
    if anchor_parity:
        raise RuntimeError(f"prefix cache/shared evaluator mismatch={anchor_parity}")
    cache = {
        "image_ids": np.asarray(ids, dtype="U64"),
        "gt_count": gt_count,
        "gt_class": gt_class,
        "pred_class_prefix": pred_class_prefix,
        "coverage": coverage,
        "legacy": legacy,
        "coverage_class": coverage_class,
        "legacy_class": legacy_class,
        "quality": quality,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **cache)
    write_json(meta_path, {**expected_meta, "shared_anchor_checks": 2000 * 6 * 2, "shared_anchor_mismatch": 0, "cache_sha256": sha256_file(cache_path)})
    append_log(f"EVALUATION prefix cache created sha256={sha256_file(cache_path)}")
    return ids, local_gt, coco_gt, remap, cache


def _attach_prefix_metrics(alloc: pd.DataFrame, ids: list[str], cache: dict[str, np.ndarray]) -> pd.DataFrame:
    index = {image_id: ix for ix, image_id in enumerate(ids)}
    image_ix = alloc["image_id"].map(index)
    if image_ix.isna().any():
        raise RuntimeError("allocation contains a foreign DEV image identity")
    ii = image_ix.to_numpy(np.int64)
    kk = alloc["K_i"].to_numpy(np.int64)
    out = alloc.copy()
    out["GT"] = cache["gt_count"][ii]
    out["coverage"] = cache["coverage"][ii, kk]
    out["quality"] = cache["quality"][ii, kk]
    out["legacy_TP"] = cache["legacy"][ii, kk]
    out["output_records"] = kk
    out["empty_GT_image"] = out["GT"].to_numpy(np.int64) == 0
    out["hit_lower_bound"] = kk == 5
    out["hit_upper_bound"] = kk == 50
    return out


def _aggregate_group(per_image: pd.DataFrame, condition_keys: list[str]) -> pd.DataFrame:
    keys = condition_keys + ["group_id"]
    result = per_image.groupby(keys, as_index=False, dropna=False).agg(
        image_count=("image_id", "size"), GT=("GT", "sum"), coverage=("coverage", "sum"),
        quality=("quality", "sum"), legacy_TP=("legacy_TP", "sum"), output_records=("output_records", "sum"),
    )
    result["coverage_per_image"] = result["coverage"] / result["image_count"]
    result["quality_per_image"] = result["quality"] / result["image_count"]
    return result


def _selection_key(k_values: np.ndarray) -> str:
    values = np.asarray(k_values, dtype=np.uint8)
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


def _coco_cache_binding(site_packages: Path) -> dict:
    prefix_path, prefix_meta_path = _cache_paths()
    return {
        "dev_gt_sha256": sha256_file(RELEASE / "gt" / "DEV2K_ROAD8_GT.json"),
        "candidate_asset_identity_sha256": sha256_file(RELEASE / "CANDIDATE_ASSET_IDENTITY.json"),
        "candidate_asset_id_dev": str(load_identity()["assets"]["DEV"]["candidate_asset_id"]),
        "shared_evaluator_sha256": sha256_file(RELEASE / "evaluator" / "shared_evaluator.py"),
        "r0_evaluation_wrapper_sha256": sha256_file(R0 / "scripts" / "evaluate_alloc.py"),
        "prefix_cache_sha256": sha256_file(prefix_path),
        "prefix_cache_meta_sha256": sha256_file(prefix_meta_path),
        "pycocotools_coco_sha256": sha256_file(site_packages / "pycocotools" / "coco.py"),
        "pycocotools_cocoeval_sha256": sha256_file(site_packages / "pycocotools" / "cocoeval.py"),
    }


def _load_coco_cache(binding: dict) -> dict:
    path = ROOT / "working" / "coco_metric_cache.json"
    meta_path = ROOT / "working" / "coco_metric_cache_meta.json"
    if not path.exists():
        return {}
    cache = json.loads(path.read_text(encoding="utf-8"))
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if all(meta.get(k) == v for k, v in binding.items()) and meta.get("cache_sha256") == sha256_file(path):
            return cache
        append_log("COCO_CACHE_BINDING_MISMATCH discarded=true")
        return {}

    # Compatibility adoption is allowed only for the cache created by this
    # task's immediately preceding validated A run.  It prevents a costly
    # duplicate COCO pass while binding those exact bytes before future reuse.
    qa_path = ROOT / "qa" / "evaluation_validation.json"
    commit_path = ROOT / "outputs" / "group_selection_commit.json"
    if qa_path.exists() and commit_path.exists():
        qa = json.loads(qa_path.read_text(encoding="utf-8"))
        commit = json.loads(commit_path.read_text(encoding="utf-8"))
        if qa.get("status") == "PASS" and qa.get("A_selection_sha256") == commit.get("sha256"):
            if all(isinstance(v, dict) and {"metrics", "class_metrics"} <= set(v) for v in cache.values()):
                write_json(meta_path, {
                    **binding, "cache_sha256": sha256_file(path),
                    "entry_count": len(cache),
                    "origin": "same-task A run validated before cache-binding enhancement",
                })
                append_log(f"COCO_CACHE_ADOPTED_AND_BOUND entries={len(cache)}")
                return cache
    append_log("COCO_CACHE_UNBOUND discarded=true")
    return {}


def _save_coco_cache(cache: dict, binding: dict) -> None:
    path = ROOT / "working" / "coco_metric_cache.json"
    write_json(path, cache)
    write_json(ROOT / "working" / "coco_metric_cache_meta.json", {
        **binding, "cache_sha256": sha256_file(path), "entry_count": len(cache), "origin": "evaluate_bench.py",
    })


def _coco_for_conditions(
    per_image: pd.DataFrame,
    condition_keys: list[str],
    ids: list[str],
    coco_gt: dict,
    candidate_by_image: dict[str, list[dict]],
    remap: dict,
    r0,
    site_packages: Path,
) -> tuple[dict[tuple, tuple[dict, list[dict]]], dict]:
    result: dict[tuple, tuple[dict, list[dict]]] = {}
    binding = _coco_cache_binding(site_packages)
    persistent = _load_coco_cache(binding)
    conditions = per_image[condition_keys].drop_duplicates().sort_values(condition_keys, kind="stable")
    for ci, row in enumerate(conditions.itertuples(index=False), 1):
        key_tuple = tuple(getattr(row, x) for x in condition_keys)
        mask = np.ones(len(per_image), dtype=bool)
        for name, value in zip(condition_keys, key_tuple):
            mask &= per_image[name].to_numpy() == value
        d = per_image.loc[mask, ["image_id", "K_i"]]
        if len(d) != 2000 or d["image_id"].nunique() != 2000:
            raise RuntimeError(f"condition image coverage invalid: {key_tuple}")
        lookup = dict(zip(d["image_id"].astype(str), d["K_i"].astype(int)))
        kvals = np.asarray([lookup[x] for x in ids], dtype=np.uint8)
        selection_sha = _selection_key(kvals)
        cached = persistent.get(selection_sha)
        if cached is None:
            tag = "BENCH_" + hashlib.sha256(repr(key_tuple).encode("utf-8")).hexdigest()[:16]
            budget = int(key_tuple[condition_keys.index("budget")])
            selection = {(tag, budget, image_id): list(range(1, int(k) + 1)) for image_id, k in zip(ids, kvals)}
            metrics, class_metrics = r0.evaluate_coco(
                coco_gt, candidate_by_image, selection, tag, budget, remap, site_packages,
            )
            cached = {"metrics": metrics, "class_metrics": class_metrics}
            persistent[selection_sha] = cached
            _save_coco_cache(persistent, binding)
        result[key_tuple] = (cached["metrics"], cached["class_metrics"])
        if ci % 10 == 0 or ci == len(conditions):
            append_log(f"COCO_PROGRESS completed={ci}/{len(conditions)} unique_cache={len(persistent)}")
    return result, persistent


def _main_and_class(
    per_image: pd.DataFrame,
    condition_keys: list[str],
    ids: list[str],
    cache: dict[str, np.ndarray],
    coco_map: dict[tuple, tuple[dict, list[dict]]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    index = {image_id: ix for ix, image_id in enumerate(ids)}
    total_gt = int(cache["gt_count"].sum())
    main_rows: list[dict] = []
    class_rows: list[dict] = []
    conditions = per_image[condition_keys].drop_duplicates().sort_values(condition_keys, kind="stable")
    for row in conditions.itertuples(index=False):
        key_tuple = tuple(getattr(row, x) for x in condition_keys)
        mask = np.ones(len(per_image), dtype=bool)
        for name, value in zip(condition_keys, key_tuple):
            mask &= per_image[name].to_numpy() == value
        d = per_image.loc[mask]
        order = np.asarray([index[x] for x in d["image_id"].astype(str)], dtype=np.int64)
        kval = d["K_i"].to_numpy(np.int64)
        metrics, class_metrics = coco_map[key_tuple]
        coverage = int(d["coverage"].sum())
        legacy = int(d["legacy_TP"].sum())
        retained = int(d["output_records"].sum())
        precision = legacy / retained
        recall = legacy / total_gt
        base = dict(zip(condition_keys, key_tuple))
        main_rows.append({
            **base, "image_count": len(d), "group_count": int(d["group_id"].nunique()),
            "total_GT": total_gt, "empty_GT_images": int(d["empty_GT_image"].sum()),
            "coverage_total": coverage, "coverage_per_image": coverage / len(d), "coverage_recall": coverage / total_gt,
            "quality_total": float(d["quality"].sum()), "quality_per_image": float(d["quality"].mean()),
            "legacy_TP": legacy, "legacy_FP": retained - legacy, "legacy_FN": total_gt - legacy,
            "precision": precision, "recall": recall, "F1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "output_records": retained, "K_mean": float(d["K_i"].mean()), "K_median": float(d["K_i"].median()),
            "K_min": int(d["K_i"].min()), "K_max": int(d["K_i"].max()),
            "lower_bound_fraction": float(d["hit_lower_bound"].mean()), "upper_bound_fraction": float(d["hit_upper_bound"].mean()),
            "AP": metrics["AP"], "AP50": metrics["AP50"], "AP75": metrics["AP75"], "AR100": metrics["AR100"],
            "source": "NEW_BENCH_P0_EVALUATION",
        })
        gt_class = cache["gt_class"].sum(axis=0)
        images_with = (cache["gt_class"] > 0).sum(axis=0)
        selected = cache["pred_class_prefix"][order, kval].sum(axis=0)
        cov_class = cache["coverage_class"][order, kval].sum(axis=0)
        leg_class = cache["legacy_class"][order, kval].sum(axis=0)
        coco_by_id = {int(x["category_id"]): x for x in class_metrics}
        for category_id, class_name in enumerate(ROAD8, 1):
            gi = category_id - 1
            pp = float(leg_class[gi] / selected[gi]) if selected[gi] else 0.0
            rr = float(leg_class[gi] / gt_class[gi]) if gt_class[gi] else math.nan
            cm = coco_by_id[category_id]
            class_rows.append({
                **base, "category_id": category_id, "class_name": class_name,
                "GT": int(gt_class[gi]), "images_with_GT": int(images_with[gi]),
                "selected_records": int(selected[gi]), "coverage": int(cov_class[gi]),
                "coverage_recall": float(cov_class[gi] / gt_class[gi]) if gt_class[gi] else math.nan,
                "legacy_TP": int(leg_class[gi]), "precision": pp, "recall": rr,
                "F1": 2 * pp * rr / (pp + rr) if gt_class[gi] and pp + rr else 0.0,
                "AP": cm["AP"], "AP50": cm["AP50"], "AP75": cm["AP75"], "AR100": cm["AR100"],
                "source": "NEW_BENCH_P0_EVALUATION",
            })
    main = pd.DataFrame(main_rows)
    classes = pd.DataFrame(class_rows)
    for key, d in classes.groupby(condition_keys, sort=False):
        m = main
        for name, value in zip(condition_keys, key if isinstance(key, tuple) else (key,)):
            m = m[m[name] == value]
        if len(m) != 1 or int(d["coverage"].sum()) != int(m.iloc[0]["coverage_total"]):
            raise RuntimeError(f"class coverage reconciliation failed: {key}")
    return main, classes


def _metric_regression_a(main: pd.DataFrame, classes: pd.DataFrame) -> dict:
    p1_main = pq.read_table(P1 / "main_results.parquet").to_pandas()
    p1_class = pq.read_table(P1 / "class_results.parquet").to_pandas()
    observed = main[(main["permutation"] == "R0") & (main["group_size"] == 40)].copy()
    observed["p1_method"] = observed["method"].replace({"M11": "LEARN_QUALITY"})
    integer_cols = ["coverage_total", "legacy_TP", "output_records", "K_min", "K_max"]
    float_cols = ["coverage_per_image", "coverage_recall", "quality_per_image", "precision", "recall", "F1", "AP", "AP50", "AP75", "AR100"]
    mismatch = 0
    max_abs = 0.0
    for r in observed.itertuples(index=False):
        old = p1_main[(p1_main["method"] == r.p1_method) & (p1_main["seed"] == r.seed) & (p1_main["budget"] == r.budget)]
        if len(old) != 1:
            raise RuntimeError("P1 main regression row absent")
        old = old.iloc[0]
        for col in integer_cols:
            mismatch += int(int(getattr(r, col)) != int(old[col]))
        for col in float_cols:
            gap = abs(float(getattr(r, col)) - float(old[col]))
            max_abs = max(max_abs, gap)
            mismatch += int(not np.isclose(float(getattr(r, col)), float(old[col]), atol=1e-6, rtol=1e-5))
    class_mismatch = 0
    class_max_abs = 0.0
    observed_class = classes[(classes["permutation"] == "R0") & (classes["group_size"] == 40)].copy()
    observed_class["p1_method"] = observed_class["method"].replace({"M11": "LEARN_QUALITY"})
    for r in observed_class.itertuples(index=False):
        old = p1_class[(p1_class["method"] == r.p1_method) & (p1_class["seed"] == r.seed) & (p1_class["budget"] == r.budget) & (p1_class["category_id"] == r.category_id)]
        if len(old) != 1:
            raise RuntimeError("P1 class regression row absent")
        old = old.iloc[0]
        for col in ("GT", "selected_records", "coverage", "legacy_TP"):
            class_mismatch += int(int(getattr(r, col)) != int(old[col]))
        for col in ("coverage_recall", "AP", "AP50", "AP75", "AR100"):
            a, b = float(getattr(r, col)), float(old[col])
            if math.isnan(a) and math.isnan(b):
                continue
            gap = abs(a - b)
            class_max_abs = max(class_max_abs, gap)
            class_mismatch += int(not np.isclose(a, b, atol=1e-6, rtol=1e-5))
    if mismatch or class_mismatch:
        raise RuntimeError(f"R0/n40 metric regression failed main={mismatch} class={class_mismatch}")
    return {
        "r0_n40_main_rows": len(observed), "r0_n40_class_rows": len(observed_class),
        "main_metric_mismatch": mismatch, "class_metric_mismatch": class_mismatch,
        "main_max_abs_float_gap": max_abs, "class_max_abs_float_gap": class_max_abs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=("A", "B", "ALL"), default="ALL")
    parser.add_argument("--coco-site-packages", type=Path, default=(Path(os.environ.get('M11_ENVIRONMENT_ROOT', "external_assets")) / 'envs/aop_detr/Lib/site-packages'))
    args = parser.parse_args()
    start = time.perf_counter()
    append_log(f"STAGE evaluate_bench START scope={args.scope}")

    a_path = _verify_a_before_gt()
    b_path = _verify_b_before_gt()
    if args.scope == "B" and b_path is None:
        raise RuntimeError("B allocation commit is not available")
    if args.scope == "ALL" and b_path is None:
        append_log("EVALUATION B commit absent; continuing with A only")

    _, candidate_by_image, candidate_ids = _load_candidates()
    a_loaded = _load_a_numeric(a_path, candidate_ids) if args.scope in ("A", "ALL") else None
    a_alloc, a_pre_gt_validation = a_loaded if a_loaded is not None else (None, None)
    b_alloc = _load_b(b_path, candidate_ids) if b_path is not None and args.scope in ("B", "ALL") else None

    r0 = import_file(R0 / "scripts" / "evaluate_alloc.py", "lc_alloc_bench_r0_eval")
    sys.path.insert(0, str(P1 / "scripts"))
    from p1_core import prefix_matching_counts  # noqa: E402

    # This is the first DEV GT access in this process.
    ids, _, coco_gt, remap, prefix_cache = _prepare_gt_and_prefix_cache(candidate_by_image, r0, prefix_matching_counts)
    persistent_coco: dict = _load_coco_cache(_coco_cache_binding(args.coco_site_packages))

    summary: dict[str, object] = {
        "status": "PASS", "dev_gt_opened_after_all_present_selection_commits": True,
        "A_selection_sha256": sha256_file(a_path),
        "B_selection_sha256": sha256_file(b_path) if b_path is not None else None,
        "valid_road8_gt": int(prefix_cache["gt_count"].sum()), "empty_gt_images": int(np.sum(prefix_cache["gt_count"] == 0)),
    }
    prior_qa_path = ROOT / "qa" / "evaluation_validation.json"
    if args.scope == "B" and prior_qa_path.exists():
        prior_qa = json.loads(prior_qa_path.read_text(encoding="utf-8"))
        if prior_qa.get("status") == "PASS" and prior_qa.get("A_selection_sha256") == summary["A_selection_sha256"]:
            summary.update({k: v for k, v in prior_qa.items() if k.startswith("A_")})
    if a_pre_gt_validation is not None:
        summary["A_pre_gt_selection_validation"] = a_pre_gt_validation
    if a_alloc is not None:
        a_per = _attach_prefix_metrics(a_alloc, ids, prefix_cache)
        a_group = _aggregate_group(a_per, CONDITION_KEYS)
        # Gate the full 300-condition pass on one frozen P1 scientific anchor.
        anchor_per = a_per[
            (a_per["permutation"] == "R0") & (a_per["group_size"] == 40)
            & (a_per["method"] == "S_ADAPT") & (a_per["seed"] == -1)
            & (a_per["budget"] == 10)
        ]
        anchor_coco, persistent_coco = _coco_for_conditions(
            anchor_per, CONDITION_KEYS, ids, coco_gt, candidate_by_image, remap, r0, args.coco_site_packages,
        )
        anchor_main, anchor_class = _main_and_class(anchor_per, CONDITION_KEYS, ids, prefix_cache, anchor_coco)
        anchor_regression = _metric_regression_a(anchor_main, anchor_class)
        append_log("A_FROZEN_S_ADAPT_K10_REGRESSION_GATE_PASS")
        coco_map, persistent_coco = _coco_for_conditions(
            a_per, CONDITION_KEYS, ids, coco_gt, candidate_by_image, remap, r0, args.coco_site_packages,
        )
        a_main, a_class = _main_and_class(a_per, CONDITION_KEYS, ids, prefix_cache, coco_map)
        if len(a_main) != 300 or len(a_class) != 2400:
            raise RuntimeError(f"A result cardinality invalid main={len(a_main)} class={len(a_class)}")
        if not np.array_equal(a_main["output_records"].to_numpy(np.int64), (2000 * a_main["budget"]).to_numpy(np.int64)):
            raise RuntimeError("A full-DEV output budget mismatch")
        regression = _metric_regression_a(a_main, a_class)
        # Publish only after both the early anchor and complete frozen regression pass.
        write_parquet(a_per, ROOT / "working" / "group_robustness_per_image.parquet")
        write_parquet(a_group, ROOT / "working" / "group_robustness_group_results.parquet")
        write_parquet(a_main, ROOT / "outputs" / "group_robustness_main.parquet")
        write_parquet(a_class, ROOT / "outputs" / "group_robustness_class.parquet")
        summary.update({
            "A_main_rows": len(a_main), "A_class_rows": len(a_class),
            "A_anchor_metric_regression": anchor_regression, "A_metric_regression": regression,
        })

    if b_alloc is not None:
        # Normalize B to the same condition/evaluation frame without changing its selections.
        b_alloc = b_alloc.copy()
        if "experiment" not in b_alloc:
            b_alloc["experiment"] = "EXTERNAL_CALIBRATION_BASELINE"
        b_per = _attach_prefix_metrics(b_alloc, ids, prefix_cache)
        write_parquet(b_per, ROOT / "working" / "external_baseline_per_image.parquet")
        b_keys = ["method", "seed", "budget"]
        b_group = _aggregate_group(b_per, b_keys)
        write_parquet(b_group, ROOT / "working" / "external_baseline_group_results.parquet")
        coco_map, persistent_coco = _coco_for_conditions(
            b_per, b_keys, ids, coco_gt, candidate_by_image, remap, r0, args.coco_site_packages,
        )
        b_main, b_class = _main_and_class(b_per, b_keys, ids, prefix_cache, coco_map)
        if len(b_main) != 10 or len(b_class) != 80:
            raise RuntimeError(f"B result cardinality invalid main={len(b_main)} class={len(b_class)}")
        threshold_count_cols = [x for x in b_per.columns if x.startswith("selected_below_fit_") and x.endswith("_count")]
        if threshold_count_cols:
            threshold_summary = b_per.groupby(b_keys, as_index=False)[threshold_count_cols].sum()
            b_main = b_main.merge(threshold_summary, on=b_keys, how="left", validate="one_to_one")
            for name in threshold_count_cols:
                b_main[name.replace("_count", "_fraction_of_selected")] = b_main[name] / b_main["output_records"]
        write_parquet(b_main, ROOT / "outputs" / "external_baseline_results.parquet")
        write_parquet(b_class, ROOT / "working" / "external_baseline_class_results.parquet")
        summary.update({"B_main_rows": len(b_main), "B_class_rows": len(b_class)})

    summary["coco_unique_selection_cache_entries"] = len(persistent_coco)
    summary["elapsed_seconds"] = time.perf_counter() - start
    write_json(ROOT / "qa" / "evaluation_validation.json", summary)
    append_log(f"STAGE evaluate_bench COMPLETE scope={args.scope} elapsed_seconds={summary['elapsed_seconds']:.6f}")


if __name__ == "__main__":
    main()
