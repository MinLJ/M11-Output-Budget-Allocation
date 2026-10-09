"""Independent COCO2017 Route-B evaluation for frozen selections.

This entry point intentionally has no detector, feature, allocator, or model
imports.  It validates the committed selection artifact before opening GT,
then runs the frozen LC maximum-matching semantics and standard COCO bbox
evaluation on the official Road8 category IDs.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import importlib.util
import io
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
import sys
import time
from typing import Any

sys.dont_write_bytecode = True

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ROAD8 = ("person", "bicycle", "car", "motorcycle", "bus", "train", "truck", "traffic light")
COCO_CAT_IDS = (1, 2, 3, 4, 6, 7, 8, 10)
COCO_TO_ROAD8 = dict(zip(COCO_CAT_IDS, range(1, 9)))
ROAD8_TO_COCO = dict(zip(range(1, 9), COCO_CAT_IDS))
BUDGETS = (10, 15, 20, 30, 40)
M11_SEEDS = (530101, 530102, 530103)
METHOD_ORDER = {"S_FIXED": 0, "S_ADAPT": 1, "CAL_TEMP_ALLOC": 2, "M11": 3}
EXPECTED_ASSET_ID = "5ff6497669d5b02c888f90451882d3c8e594b74a3aedd136639a9c0c8621c707"
EXPECTED_SELECTION_SHA = "b1c75d96292f4c6a8aaac5208b6b50ad261eb9da6d419aa18e79f8e69c0b8639"


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def sha256_file(path: str | os.PathLike[str], chunk: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def append_log(root: Path, message: str) -> None:
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with (root / "working" / "runtime.log").open("a", encoding="utf-8") as handle:
        handle.write(f"{stamp}\t{message}\n")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def condition_label(method: str, seed: int, budget: int) -> str:
    seed_text = "DET" if seed < 0 else str(seed)
    return f"{method}|{seed_text}|K{budget}"


def stable_condition_sort(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["_method_order"] = out["method"].map(METHOD_ORDER)
    out = out.sort_values(["_method_order", "seed", "budget"], kind="mergesort").drop(columns="_method_order")
    return out.reset_index(drop=True)


def validate_hash_bindings(config: dict[str, Any]) -> None:
    for binding in config["input_bindings"].values():
        if binding.get("defer_until_after_selection_validation", False):
            continue
        path = Path(binding["path"])
        require(path.is_file(), f"missing bound input: {path}")
        actual = sha256_file(path)
        require(actual == binding["sha256"], f"hash mismatch: {path} expected={binding['sha256']} actual={actual}")


def load_group_and_image_manifests(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    groups = pd.read_csv(config["paths"]["group_manifest"], dtype={"canonical_image_id": "string", "group_key": "string"})
    images = pd.read_csv(config["paths"]["image_manifest"], dtype={"canonical_image_id": "string", "image_id": "string"})
    require(len(groups) == 5000 and groups["coco_image_id"].nunique() == 5000, "group manifest must contain 5000 unique images")
    require(groups["group_id"].nunique() == 125 and set(groups.groupby("group_id").size()) == {40}, "group manifest must be 125x40")
    require(len(images) == 5000 and images["coco_image_id"].nunique() == 5000, "image manifest must contain 5000 unique images")
    require(set(groups["coco_image_id"].astype(int)) == set(images["coco_image_id"].astype(int)), "image/group manifest set mismatch")
    require(set(images["candidate_asset_id"].astype(str)) == {EXPECTED_ASSET_ID}, "image manifest CandidateAssetID mismatch")
    group_ids = groups.set_index("coco_image_id")["group_id"].astype(int)
    image_group = images["coco_image_id"].map(group_ids)
    require(image_group.notna().all(), "image manifest group join failed")
    return groups, images


def load_candidates(config: dict[str, Any]) -> tuple[pd.DataFrame, dict[int, pd.DataFrame]]:
    root = Path(config["paths"]["candidate_asset_root"])
    manifest = pd.read_csv(config["paths"]["candidate_manifest"])
    require(len(manifest) == 10 and int(manifest["rows"].sum()) == 1_500_000, "candidate manifest shape mismatch")
    columns = [
        "dataset_version", "split", "coco_image_id", "image_id", "canonical_image_id", "candidate_asset_id",
        "candidate_record_id", "road8_rank", "query_index", "predicted_road8_class_id",
        "predicted_coco_category_id", "score", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
    ]
    frames = []
    for row in manifest.itertuples(index=False):
        path = root / str(row.path)
        require(sha256_file(path) == str(row.sha256), f"candidate shard hash mismatch: {path}")
        table = pq.read_table(path, columns=columns, filters=[("road8_rank", "<=", 50)])
        frame = table.to_pandas()
        require(len(frame) == int(row.image_count) * 50, f"Top50 shard row count mismatch: {path}")
        frames.append(frame)
    candidates = pd.concat(frames, ignore_index=True)
    candidates["coco_image_id"] = candidates["coco_image_id"].astype(np.int64)
    candidates["road8_rank"] = candidates["road8_rank"].astype(np.int16)
    candidates = candidates.sort_values(["coco_image_id", "road8_rank"], kind="mergesort").reset_index(drop=True)
    require(len(candidates) == 250_000, "candidate Top50 must contain 250000 rows")
    require(candidates["candidate_record_id"].nunique() == len(candidates), "candidate record IDs are not unique")
    require(set(candidates["dataset_version"].astype(str)) == {"COCO2017-Road8-v1"}, "candidate dataset version mismatch")
    require(set(candidates["split"].astype(str)) == {"VAL2017"}, "candidate split mismatch")
    require(set(candidates["candidate_asset_id"].astype(str)) == {EXPECTED_ASSET_ID}, "candidate asset ID mismatch")
    require(np.isfinite(candidates[["score", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]].to_numpy(np.float64)).all(), "candidate nonfinite numeric value")
    require(((candidates["bbox_x2"] - candidates["bbox_x1"]) >= 0).all() and ((candidates["bbox_y2"] - candidates["bbox_y1"]) >= 0).all(), "candidate reversed bbox")
    mapping_ok = candidates["predicted_road8_class_id"].map(ROAD8_TO_COCO).astype(int).to_numpy() == candidates["predicted_coco_category_id"].astype(int).to_numpy()
    require(bool(np.all(mapping_ok)), "candidate Road8/COCO category mapping mismatch")
    by_image: dict[int, pd.DataFrame] = {}
    for coco_id, frame in candidates.groupby("coco_image_id", sort=False):
        ranks = frame["road8_rank"].astype(int).tolist()
        require(ranks == list(range(1, 51)), f"candidate Top50 rank invariant failed: {coco_id}")
        by_image[int(coco_id)] = frame.reset_index(drop=True)
    require(len(by_image) == 5000, "candidate image coverage mismatch")
    return candidates, by_image


def expected_conditions() -> set[tuple[str, int, int]]:
    conditions = {(method, -1, budget) for method in ("S_FIXED", "S_ADAPT", "CAL_TEMP_ALLOC") for budget in BUDGETS}
    conditions.update({("M11", seed, budget) for seed in M11_SEEDS for budget in BUDGETS})
    return conditions


def validate_frozen_selections(
    config: dict[str, Any], candidates: pd.DataFrame, groups: pd.DataFrame
) -> pd.DataFrame:
    selection_path = Path(config["paths"]["selected_records"])
    require(sha256_file(selection_path) == EXPECTED_SELECTION_SHA, "selection SHA mismatch")
    selection_manifest = pd.read_csv(config["paths"]["selection_manifest"])
    require(len(selection_manifest) == 30, "selection manifest must contain 30 condition rows")
    require(set(selection_manifest["selected_records_sha256"].astype(str)) == {EXPECTED_SELECTION_SHA}, "selection manifest SHA binding mismatch")
    require(set(selection_manifest["status"].astype(str)) == {"FROZEN_BEFORE_GT_EVALUATION"}, "selection manifest is not frozen before GT")
    require(set(selection_manifest["gt_path_available_to_selection_entrypoint"].astype(str).str.lower()) == {"false"}, "selection entrypoint had GT path")

    candidate_lookup = candidates[[
        "coco_image_id", "road8_rank", "canonical_image_id", "candidate_record_id", "query_index",
        "predicted_road8_class_id", "score",
    ]].copy()
    group_map = groups.set_index("coco_image_id")["group_id"].astype(int)
    parquet = pq.ParquetFile(selection_path)
    require(parquet.metadata.num_rows == 3_450_000 and parquet.metadata.num_row_groups == 30, "selection parquet shape mismatch")
    seen_conditions: set[tuple[str, int, int]] = set()
    allocations = []
    total_rows = 0
    for rg in range(parquet.metadata.num_row_groups):
        frame = parquet.read_row_group(rg).to_pandas()
        keys = frame[["method", "seed", "budget"]].drop_duplicates()
        require(len(keys) == 1, f"selection row group {rg} mixes conditions")
        method, seed, budget = str(keys.iloc[0].method), int(keys.iloc[0].seed), int(keys.iloc[0].budget)
        condition = (method, seed, budget)
        require(condition in expected_conditions() and condition not in seen_conditions, f"unexpected/duplicate condition {condition}")
        seen_conditions.add(condition)
        require(len(frame) == 5000 * budget, f"selection capacity mismatch for {condition}")
        require(set(frame["dataset_version"].astype(str)) == {"COCO2017-Road8-v1"}, "selection dataset mismatch")
        require(set(frame["split"].astype(str)) == {"VAL2017"}, "selection split mismatch")
        require(set(frame["candidate_asset_id"].astype(str)) == {EXPECTED_ASSET_ID}, "selection CandidateAssetID mismatch")
        require(np.array_equal(frame["selection_rank"].to_numpy(), frame["road8_rank"].to_numpy()), "selection rank != road8 rank")
        joined = frame.merge(candidate_lookup, on=["coco_image_id", "road8_rank"], how="left", validate="many_to_one", suffixes=("_sel", "_cand"))
        require(joined["candidate_record_id_cand"].notna().all(), f"selection candidate join failed {condition}")
        for field in ("canonical_image_id", "candidate_record_id", "query_index", "predicted_road8_class_id"):
            require(np.array_equal(joined[f"{field}_sel"].astype(str).to_numpy(), joined[f"{field}_cand"].astype(str).to_numpy()), f"selection mutated {field}: {condition}")
        require(np.array_equal(joined["score_sel"].to_numpy(np.float64), joined["score_cand"].to_numpy(np.float64)), f"selection mutated score: {condition}")
        expected_groups = frame["coco_image_id"].map(group_map)
        require(expected_groups.notna().all() and np.array_equal(expected_groups.to_numpy(np.int64), frame["group_id"].to_numpy(np.int64)), f"selection group mismatch: {condition}")
        work = frame[["coco_image_id", "image_id", "canonical_image_id", "group_id", "K_i", "road8_rank"]].copy()
        work["rank_sq"] = work["road8_rank"].astype(np.int64) ** 2
        agg = work.groupby("coco_image_id", sort=False).agg(
            image_id=("image_id", "first"), canonical_image_id=("canonical_image_id", "first"), group_id=("group_id", "first"),
            K_i=("K_i", "first"), K_nunique=("K_i", "nunique"), selected_n=("road8_rank", "size"),
            rank_min=("road8_rank", "min"), rank_max=("road8_rank", "max"), rank_sum=("road8_rank", "sum"), rank_sq_sum=("rank_sq", "sum"),
        ).reset_index()
        require(len(agg) == 5000 and (agg["K_nunique"] == 1).all(), f"selection image cells incomplete: {condition}")
        k = agg["K_i"].astype(np.int64)
        require(((k >= 5) & (k <= 50)).all(), f"K bounds failed: {condition}")
        require((agg["selected_n"].astype(np.int64) == k).all(), f"selected count != K: {condition}")
        require((agg["rank_min"] == 1).all() and (agg["rank_max"].astype(np.int64) == k).all(), f"selection not prefix: {condition}")
        require((agg["rank_sum"].astype(np.int64) == k * (k + 1) // 2).all(), f"prefix rank sum failed: {condition}")
        require((agg["rank_sq_sum"].astype(np.int64) == k * (k + 1) * (2 * k + 1) // 6).all(), f"prefix rank squared sum failed: {condition}")
        if method == "S_FIXED":
            require((k == budget).all(), f"S_FIXED K mismatch: {condition}")
        budget_by_group = agg.groupby("group_id")["K_i"].sum()
        require(len(budget_by_group) == 125 and (budget_by_group == 40 * budget).all(), f"group exact budget failed: {condition}")
        agg.insert(0, "method", method)
        agg.insert(1, "seed", seed)
        agg.insert(2, "budget", budget)
        allocations.append(agg[["method", "seed", "budget", "coco_image_id", "image_id", "canonical_image_id", "group_id", "K_i"]])
        total_rows += len(frame)
        append_log(Path(config["output_root"]), f"SELECTION_ROW_GROUP_VALIDATED rg={rg} condition={condition_label(*condition)} rows={len(frame)}")
    require(seen_conditions == expected_conditions(), "selection condition matrix incomplete")
    require(total_rows == 3_450_000, "selection total rows mismatch")
    allocation = stable_condition_sort(pd.concat(allocations, ignore_index=True))
    require(len(allocation) == 150_000, "allocation cell count mismatch")
    s = allocation[allocation.method == "S_ADAPT"].sort_values(["budget", "coco_image_id"])["K_i"].to_numpy()
    c = allocation[allocation.method == "CAL_TEMP_ALLOC"].sort_values(["budget", "coco_image_id"])["K_i"].to_numpy()
    require(np.array_equal(s, c), "CAL_TEMP_ALLOC is not selection-equivalent to S_ADAPT")
    return allocation


def load_official_gt(config: dict[str, Any], image_ids: set[int]) -> tuple[dict[str, Any], dict[int, tuple[np.ndarray, np.ndarray]], dict[str, Any]]:
    gt_path = Path(config["paths"]["gt"])
    require(sha256_file(gt_path) == config["expected_hashes"]["gt_sha256"], "GT SHA mismatch")
    raw = json.loads(gt_path.read_text(encoding="utf-8"))
    require({int(x["id"]) for x in raw["images"]} == image_ids, "GT image set mismatch")
    categories = {int(x["id"]): str(x["name"]) for x in raw["categories"]}
    require({cid: categories.get(cid) for cid in COCO_CAT_IDS} == dict(zip(COCO_CAT_IDS, ROAD8)), "COCO Road8 category table mismatch")
    subset_annotations = [dict(x) for x in raw["annotations"] if int(x["category_id"]) in COCO_TO_ROAD8]
    coco_subset = {
        "info": dict(raw.get("info", {})), "licenses": list(raw.get("licenses", [])),
        "images": [dict(x) for x in raw["images"]], "annotations": subset_annotations,
        "categories": [dict(x) for x in raw["categories"] if int(x["id"]) in COCO_TO_ROAD8],
    }
    classes: dict[int, list[int]] = {image_id: [] for image_id in image_ids}
    boxes: dict[int, list[list[float]]] = {image_id: [] for image_id in image_ids}
    support = Counter()
    support_images: dict[int, set[int]] = defaultdict(set)
    official = Counter()
    crowd = Counter()
    ignored = Counter()
    invalid = Counter()
    for ann in subset_annotations:
        coco_class = int(ann["category_id"])
        road8_class = COCO_TO_ROAD8[coco_class]
        image_id = int(ann["image_id"])
        official[road8_class] += 1
        if int(ann.get("iscrowd", 0)):
            crowd[road8_class] += 1
            continue
        if int(ann.get("ignore", 0)):
            ignored[road8_class] += 1
            continue
        bbox = ann.get("bbox", [])
        if len(bbox) != 4:
            invalid[road8_class] += 1
            continue
        x, y, w, h = [float(v) for v in bbox]
        area = float(ann.get("area", w * h))
        if not all(math.isfinite(v) for v in (x, y, w, h, area)) or w <= 0 or h <= 0 or area <= 0:
            invalid[road8_class] += 1
            continue
        classes[image_id].append(road8_class)
        boxes[image_id].append([x, y, x + w, y + h])
        support[road8_class] += 1
        support_images[road8_class].add(image_id)
    gt_by_image = {
        image_id: (
            np.asarray(classes[image_id], dtype=np.int16),
            np.asarray(boxes[image_id], dtype=np.float64).reshape(-1, 4),
        )
        for image_id in image_ids
    }
    summary = {
        "official_road8_annotations": int(sum(official.values())),
        "valid_noncrowd_road8_gt": int(sum(support.values())),
        "crowd_excluded_for_m11": int(sum(crowd.values())),
        "ignore_excluded_for_m11": int(sum(ignored.values())),
        "invalid_excluded_for_m11": int(sum(invalid.values())),
        "images_with_valid_road8_gt": int(sum(bool(classes[x]) for x in image_ids)),
        "empty_valid_road8_gt_images": int(sum(not classes[x] for x in image_ids)),
        "by_class": [
            {
                "road8_class_id": ci, "class_name": ROAD8[ci - 1], "coco_category_id": ROAD8_TO_COCO[ci],
                "official_annotations": int(official[ci]), "crowd_annotations": int(crowd[ci]),
                "ignored_annotations": int(ignored[ci]), "invalid_annotations": int(invalid[ci]),
                "valid_noncrowd_gt": int(support[ci]), "images_with_valid_gt": len(support_images[ci]),
            }
            for ci in range(1, 9)
        ],
    }
    del raw
    return coco_subset, gt_by_image, summary


def build_prefix_cache(p1_core, by_image: dict[int, pd.DataFrame], gt_by_image: dict[int, tuple[np.ndarray, np.ndarray]], root: Path):
    cache = {}
    for ix, coco_id in enumerate(sorted(by_image)):
        frame = by_image[coco_id]
        cand_classes = frame["predicted_road8_class_id"].to_numpy(np.int16)
        cand_boxes = frame[["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]].to_numpy(np.float64)
        gt_classes, gt_boxes = gt_by_image[coco_id]
        totals, by_class = p1_core.prefix_matching_counts(cand_classes, cand_boxes, gt_classes, gt_boxes, max_k=50)
        prefix_selected = np.zeros((51, 8), dtype=np.int16)
        for k in range(1, 51):
            prefix_selected[k] = prefix_selected[k - 1]
            prefix_selected[k, int(cand_classes[k - 1]) - 1] += 1
        cache[coco_id] = {
            "totals": totals, "by_class": by_class, "selected_by_class": prefix_selected,
            "gt_by_class": np.bincount(gt_classes, minlength=9)[1:9].astype(np.int32),
        }
        if (ix + 1) % 500 == 0:
            append_log(root, f"PREFIX_MATCHING_PROGRESS images={ix+1}")
    return cache


def evaluate_m11_metrics(
    allocation: pd.DataFrame, prefix_cache: dict[int, dict[str, np.ndarray]], weights: np.ndarray,
    gt_summary: dict[str, Any], root: Path,
) -> tuple[pd.DataFrame, dict[tuple[str, int, int], dict[str, np.ndarray]]]:
    rows = []
    class_acc: dict[tuple[str, int, int], dict[str, np.ndarray]] = defaultdict(
        lambda: {
            "coverage": np.zeros(8, dtype=np.int64), "selected": np.zeros(8, dtype=np.int64),
            "quality": np.zeros(8, dtype=np.float64),
        }
    )
    total_valid_gt = int(gt_summary["valid_noncrowd_road8_gt"])
    for index, row in enumerate(allocation.itertuples(index=False)):
        key = (str(row.method), int(row.seed), int(row.budget))
        z = prefix_cache[int(row.coco_image_id)]
        k = int(row.K_i)
        coverage_by_class = z["by_class"][k, :, 0].astype(np.int64)
        quality_by_class = weights * z["by_class"][k].mean(axis=1).astype(np.float64)
        coverage = int(z["totals"][k, 0])
        gt_count = int(z["gt_by_class"].sum())
        quality = float(quality_by_class.sum())
        rows.append({
            "dataset": "COCO2017-Road8-v1", "split": "VAL2017", "method": row.method, "seed": int(row.seed),
            "budget": int(row.budget), "group_id": int(row.group_id), "image_id": row.image_id,
            "canonical_image_id": row.canonical_image_id, "coco_image_id": int(row.coco_image_id), "K_i": k,
            "valid_noncrowd_gt": gt_count, "coverage": coverage,
            "coverage_recall_image": coverage / gt_count if gt_count else math.nan,
            "quality": quality, "output_records": k, "empty_valid_gt_image": gt_count == 0,
        })
        a = class_acc[key]
        a["coverage"] += coverage_by_class
        a["selected"] += z["selected_by_class"][k].astype(np.int64)
        a["quality"] += quality_by_class
        if (index + 1) % 25_000 == 0:
            append_log(root, f"M11_METRIC_ROWS_PROGRESS rows={index+1}")
    frame = stable_condition_sort(pd.DataFrame(rows))
    require(len(frame) == 150_000, "M11 raw metric rows must be 150000")
    require(int(frame["output_records"].sum()) == 3_450_000, "M11 raw total outputs mismatch")
    require(total_valid_gt > 0, "valid GT denominator is zero")
    return frame, class_acc


def valid_mean(values: np.ndarray) -> float | None:
    values = np.asarray(values, dtype=np.float64)
    values = values[values > -1]
    return float(values.mean()) if values.size else None


def allocation_identity(frame: pd.DataFrame) -> str:
    ordered = frame.sort_values("coco_image_id", kind="mergesort")
    payload = "".join(f"{int(r.coco_image_id)}:{int(r.K_i)}\n" for r in ordered.itertuples(index=False)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def run_coco_eval(
    truth, coco_subset: dict[str, Any], alloc: pd.DataFrame, by_image: dict[int, pd.DataFrame],
    all_image_ids: list[int], pycoco_site: Path,
) -> dict[str, Any]:
    if str(pycoco_site) not in sys.path:
        sys.path.append(str(pycoco_site))
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    predictions = []
    k_map = dict(zip(alloc["coco_image_id"].astype(int), alloc["K_i"].astype(int)))
    require(set(k_map) == set(all_image_ids), "COCO allocation image set mismatch")
    for image_id in all_image_ids:
        frame = by_image[image_id].iloc[: k_map[image_id]]
        for row in frame.itertuples(index=False):
            x1, y1, x2, y2 = float(row.bbox_x1), float(row.bbox_y1), float(row.bbox_x2), float(row.bbox_y2)
            predictions.append({
                "image_id": image_id, "category_id": int(row.predicted_coco_category_id),
                "score": float(row.score), "bbox": [x1, y1, x2 - x1, y2 - y1],
            })
    with contextlib.redirect_stdout(io.StringIO()):
        detections = truth.loadRes(predictions)
        evaluator = COCOeval(truth, detections, "bbox")
        evaluator.params.imgIds = all_image_ids
        evaluator.params.catIds = list(COCO_CAT_IDS)
        evaluator.params.iouThrs = np.linspace(0.50, 0.95, 10)
        evaluator.params.recThrs = np.linspace(0.0, 1.0, 101)
        evaluator.params.maxDets = [1, 10, 100]
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    stats = evaluator.stats.astype(np.float64)
    require(len(stats) == 12, "COCOeval stats length mismatch")
    stat_names = ("AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large", "AR1", "AR10", "AR100", "AR_small", "AR_medium", "AR_large")
    summary = {name: (None if value < 0 else float(value)) for name, value in zip(stat_names, stats)}
    per_class = []
    for index, (road8_id, class_name, coco_id) in enumerate(zip(range(1, 9), ROAD8, COCO_CAT_IDS)):
        precision = evaluator.eval["precision"][:, :, index, 0, 2]
        recall = evaluator.eval["recall"][:, index, 0, 2]
        iou50 = int(np.flatnonzero(np.isclose(evaluator.params.iouThrs, 0.50))[0])
        iou75 = int(np.flatnonzero(np.isclose(evaluator.params.iouThrs, 0.75))[0])
        per_class.append({
            "road8_class_id": road8_id, "class_name": class_name, "coco_category_id": coco_id,
            "AP": valid_mean(precision), "AP50": valid_mean(precision[iou50]), "AP75": valid_mean(precision[iou75]),
            "AR100": valid_mean(recall),
        })
    return {"summary": summary, "per_class": per_class, "detections": len(predictions)}


def evaluate_all_coco(
    config: dict[str, Any], allocation: pd.DataFrame, by_image: dict[int, pd.DataFrame],
    images: pd.DataFrame, coco_subset: dict[str, Any], root: Path,
) -> dict[tuple[str, int, int], dict[str, Any]]:
    pycoco_site = Path(config["paths"]["pycocotools_site_packages"])
    if str(pycoco_site) not in sys.path:
        sys.path.append(str(pycoco_site))
    from pycocotools.coco import COCO
    with contextlib.redirect_stdout(io.StringIO()):
        truth = COCO()
        truth.dataset = coco_subset
        truth.createIndex()
    all_image_ids = images.sort_values("manifest_index", kind="mergesort")["coco_image_id"].astype(int).tolist()
    results: dict[tuple[str, int, int], dict[str, Any]] = {}
    cache: dict[str, tuple[tuple[str, int, int], dict[str, Any]]] = {}
    conditions = allocation[["method", "seed", "budget"]].drop_duplicates()
    conditions["_method_order"] = conditions["method"].map(METHOD_ORDER)
    conditions = conditions.sort_values(["_method_order", "seed", "budget"], kind="mergesort")
    for row in conditions.itertuples(index=False):
        key = (str(row.method), int(row.seed), int(row.budget))
        cell = allocation[(allocation.method == key[0]) & (allocation.seed == key[1]) & (allocation.budget == key[2])]
        identity = allocation_identity(cell)
        if identity in cache:
            source, result = cache[identity]
            copied = json.loads(json.dumps(result))
            copied["selection_identity_sha256"] = identity
            copied["reused_exact_selection_from"] = condition_label(*source)
            results[key] = copied
            append_log(root, f"COCO_EVAL_EXACT_REUSE condition={condition_label(*key)} source={condition_label(*source)}")
            continue
        result = run_coco_eval(truth, coco_subset, cell, by_image, all_image_ids, pycoco_site)
        result["selection_identity_sha256"] = identity
        result["reused_exact_selection_from"] = None
        results[key] = result
        cache[identity] = (key, result)
        append_log(root, f"COCO_EVAL_COMPLETE condition={condition_label(*key)} detections={result['detections']}")
    require(set(results) == expected_conditions(), "COCO result condition matrix incomplete")
    return results


def build_main_and_class_tables(
    raw: pd.DataFrame, class_acc: dict[tuple[str, int, int], dict[str, np.ndarray]], coco_results: dict,
    gt_summary: dict[str, Any], config_sha: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    total_gt = int(gt_summary["valid_noncrowd_road8_gt"])
    support_by_class = {int(x["road8_class_id"]): x for x in gt_summary["by_class"]}
    main_rows = []
    class_rows = []
    for key, frame in raw.groupby(["method", "seed", "budget"], sort=False):
        method, seed, budget = str(key[0]), int(key[1]), int(key[2])
        require(len(frame) == 5000, f"main result image count mismatch: {key}")
        output_records = int(frame["output_records"].sum())
        require(output_records == 5000 * budget, f"full split budget mismatch: {key}")
        group_totals = frame.groupby("group_id")["K_i"].sum()
        require(len(group_totals) == 125 and (group_totals == 40 * budget).all(), f"group budget mismatch during evaluation: {key}")
        coco = coco_results[key]
        main_rows.append({
            "dataset": "COCO2017-Road8-v1", "split": "VAL2017", "method": method, "seed": seed, "budget": budget,
            "image_count": 5000, "group_count": 125, "valid_noncrowd_road8_gt": total_gt,
            "images_with_valid_road8_gt": int(gt_summary["images_with_valid_road8_gt"]),
            "empty_valid_road8_gt_images": int(gt_summary["empty_valid_road8_gt_images"]),
            "coverage_total": int(frame["coverage"].sum()), "coverage_per_image": float(frame["coverage"].mean()),
            "coverage_recall": float(frame["coverage"].sum() / total_gt),
            "quality_total": float(frame["quality"].sum()), "quality_per_image": float(frame["quality"].mean()),
            "AP": coco["summary"]["AP"], "AP50": coco["summary"]["AP50"], "AP75": coco["summary"]["AP75"], "AR100": coco["summary"]["AR100"],
            "output_records": output_records, "K_mean": float(frame["K_i"].mean()), "K_std": float(frame["K_i"].std(ddof=0)),
            "K_median": float(frame["K_i"].median()), "K_min": int(frame["K_i"].min()), "K_max": int(frame["K_i"].max()),
            "images_at_K_min": int((frame["K_i"] == 5).sum()), "images_at_K_max": int((frame["K_i"] == 50).sum()),
            "exact_group_budget_pass": True, "selection_identity_sha256": coco["selection_identity_sha256"],
            "coco_eval_reused_from": coco["reused_exact_selection_from"], "evaluation_config_sha256": config_sha,
        })
        ap_class = {int(x["road8_class_id"]): x for x in coco["per_class"]}
        acc = class_acc[key]
        for ci in range(1, 9):
            support = support_by_class[ci]
            gt_count = int(support["valid_noncrowd_gt"])
            cm = ap_class[ci]
            class_rows.append({
                "dataset": "COCO2017-Road8-v1", "split": "VAL2017", "method": method, "seed": seed, "budget": budget,
                "road8_class_id": ci, "class_name": ROAD8[ci - 1], "coco_category_id": ROAD8_TO_COCO[ci],
                "valid_noncrowd_gt_count": gt_count, "valid_noncrowd_gt_image_count": int(support["images_with_valid_gt"]),
                "official_road8_annotation_count": int(support["official_annotations"]), "crowd_annotation_count": int(support["crowd_annotations"]),
                "ignored_annotation_count": int(support["ignored_annotations"]), "invalid_annotation_count": int(support["invalid_annotations"]),
                "selected_records": int(acc["selected"][ci - 1]), "coverage": int(acc["coverage"][ci - 1]),
                "coverage_recall": float(acc["coverage"][ci - 1] / gt_count) if gt_count else math.nan,
                "quality_contribution_total": float(acc["quality"][ci - 1]), "quality_contribution_per_image": float(acc["quality"][ci - 1] / 5000.0),
                "AP": cm["AP"], "AP50": cm["AP50"], "AP75": cm["AP75"], "AR100": cm["AR100"],
                "evaluation_config_sha256": config_sha,
            })
    main = stable_condition_sort(pd.DataFrame(main_rows))
    classes = stable_condition_sort(pd.DataFrame(class_rows))
    require(len(main) == 30 and len(classes) == 240, "final result row counts invalid")
    for key, frame in classes.groupby(["method", "seed", "budget"]):
        expected = int(main[(main.method == key[0]) & (main.seed == key[1]) & (main.budget == key[2])]["coverage_total"].iloc[0])
        require(int(frame["coverage"].sum()) == expected, f"class coverage sum mismatch: {key}")
    return main, classes


def frozen_bootstrap_indices(config: dict[str, Any]) -> tuple[np.ndarray, str]:
    rng = np.random.default_rng(530002)
    indices = rng.integers(0, 125, size=(5000, 125), endpoint=False, dtype=np.int16).astype("<i2", copy=False)
    header = (
        "schema=COCO2017_PAIRED_GROUP_BOOTSTRAP_V1"
        "\tgenerator=numpy.default_rng(PCG64)\tseed=530002"
        "\tshape=5000x125\tdtype=int16_little_endian\torder=C\tendpoint=false\n"
    ).encode("utf-8")
    digest = hashlib.sha256(header + indices.tobytes(order="C")).hexdigest()
    require(digest == config["bootstrap"]["indices_sha256"], f"bootstrap index identity mismatch: {digest}")
    return indices, digest


def build_bootstrap(raw: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    indices, digest = frozen_bootstrap_indices(config)
    grouped = raw.groupby(["method", "seed", "budget", "group_id"], as_index=False).agg(
        coverage=("coverage", "sum"), quality=("quality", "sum"), image_count=("coco_image_id", "size")
    )
    require(set(grouped["image_count"]) == {40}, "bootstrap group image count mismatch")
    grouped["coverage_per_image"] = grouped["coverage"] / 40.0
    grouped["quality_per_image"] = grouped["quality"] / 40.0
    rows = []

    def vector(method: str, seed: int, budget: int, metric: str) -> np.ndarray:
        part = grouped[(grouped.method == method) & (grouped.seed == seed) & (grouped.budget == budget)].sort_values("group_id")
        require(len(part) == 125 and part["group_id"].tolist() == list(range(125)), "bootstrap group vector incomplete")
        return part[metric].to_numpy(np.float64)

    def emit(scope: str, seed: int | None, budget_scope: str, metric: str, diff: np.ndarray) -> None:
        reps = diff[indices].mean(axis=1)
        rows.append({
            "comparison": "M11 - S_ADAPT", "m11_scope": scope, "m11_seed": seed,
            "budget_scope": budget_scope, "metric": metric, "unit": metric,
            "point_estimate": float(diff.mean()), "ci95_lower": float(np.percentile(reps, 2.5)),
            "ci95_upper": float(np.percentile(reps, 97.5)), "positive_resample_fraction": float(np.mean(reps > 0)),
            "bootstrap_unit": "frozen 40-image group", "group_count": 125, "resamples": 5000,
            "bootstrap_seed": 530002, "bootstrap_indices_sha256": digest,
        })

    for metric in ("coverage_per_image", "quality_per_image"):
        for budget in BUDGETS:
            baseline = vector("S_ADAPT", -1, budget, metric)
            seed_diffs = []
            for seed in M11_SEEDS:
                diff = vector("M11", seed, budget, metric) - baseline
                seed_diffs.append(diff)
                emit(f"SEED_{seed}", seed, str(budget), metric, diff)
            emit("THREE_SEED_MEAN", None, str(budget), metric, np.mean(np.stack(seed_diffs), axis=0))
        for seed in M11_SEEDS:
            core = np.mean(np.stack([vector("M11", seed, b, metric) - vector("S_ADAPT", -1, b, metric) for b in (10, 15, 20)]), axis=0)
            emit(f"SEED_{seed}", seed, "CORE_10_15_20", metric, core)
        mean_core = np.mean(
            np.stack([
                np.mean(np.stack([vector("M11", seed, b, metric) - vector("S_ADAPT", -1, b, metric) for seed in M11_SEEDS]), axis=0)
                for b in (10, 15, 20)
            ]), axis=0,
        )
        emit("THREE_SEED_MEAN", None, "CORE_10_15_20", metric, mean_core)
    out = pd.DataFrame(rows)
    require(len(out) == 48, "bootstrap result row count mismatch")
    return out


def build_selection_analysis(allocation: pd.DataFrame) -> pd.DataFrame:
    base = allocation[allocation.method == "S_ADAPT"][["budget", "coco_image_id", "canonical_image_id", "group_id", "K_i"]].rename(columns={"K_i": "K_s_adapt"})
    rows = []
    for seed in M11_SEEDS:
        m = allocation[(allocation.method == "M11") & (allocation.seed == seed)][["budget", "coco_image_id", "K_i"]].rename(columns={"K_i": "K_m11"})
        joined = base.merge(m, on=["budget", "coco_image_id"], validate="one_to_one")
        joined.insert(0, "m11_seed", seed)
        joined["delta_K_m11_minus_s_adapt"] = joined["K_m11"] - joined["K_s_adapt"]
        joined["abs_delta_K"] = joined["delta_K_m11_minus_s_adapt"].abs()
        joined["selected_record_overlap_count"] = joined[["K_m11", "K_s_adapt"]].min(axis=1)
        joined["selected_record_union_count"] = joined[["K_m11", "K_s_adapt"]].max(axis=1)
        joined["selected_record_jaccard"] = joined["selected_record_overlap_count"] / joined["selected_record_union_count"]
        joined["same_prefix_selection"] = joined["K_m11"] == joined["K_s_adapt"]
        rows.append(joined)
    out = pd.concat(rows, ignore_index=True).sort_values(["m11_seed", "budget", "group_id", "canonical_image_id"], kind="mergesort")
    require(len(out) == 75_000, "selection analysis row count mismatch")
    for (seed, budget, group_id), frame in out.groupby(["m11_seed", "budget", "group_id"]):
        require(int(frame["delta_K_m11_minus_s_adapt"].sum()) == 0, f"paired group K delta not zero: {(seed,budget,group_id)}")
    return out.reset_index(drop=True)


def write_report(
    root: Path, config: dict[str, Any], config_sha: str, gt_summary: dict[str, Any], main: pd.DataFrame,
    bootstrap: pd.DataFrame, selection_analysis: pd.DataFrame, elapsed: float,
) -> None:
    comparison_rows = []
    for budget in BUDGETS:
        base = main[(main.method == "S_ADAPT") & (main.budget == budget)].iloc[0]
        m11 = main[(main.method == "M11") & (main.budget == budget)]
        boot_cov = bootstrap[(bootstrap.m11_scope == "THREE_SEED_MEAN") & (bootstrap.budget_scope == str(budget)) & (bootstrap.metric == "coverage_per_image")].iloc[0]
        boot_quality = bootstrap[(bootstrap.m11_scope == "THREE_SEED_MEAN") & (bootstrap.budget_scope == str(budget)) & (bootstrap.metric == "quality_per_image")].iloc[0]
        comparison_rows.append(
            f"| {budget} | {base.coverage_per_image:.6f} | {m11.coverage_per_image.mean():.6f} | {m11.coverage_per_image.mean()-base.coverage_per_image:+.6f} "
            f"[{boot_cov.ci95_lower:+.6f}, {boot_cov.ci95_upper:+.6f}] | {base.quality_per_image:.6f} | {m11.quality_per_image.mean():.6f} | "
            f"{m11.quality_per_image.mean()-base.quality_per_image:+.6f} [{boot_quality.ci95_lower:+.6f}, {boot_quality.ci95_upper:+.6f}] | "
            f"{base.AP:.6f} | {m11.AP.mean():.6f} | {base.AR100:.6f} | {m11.AR100.mean():.6f} |"
        )
    core_cov = bootstrap[(bootstrap.m11_scope == "THREE_SEED_MEAN") & (bootstrap.budget_scope == "CORE_10_15_20") & (bootstrap.metric == "coverage_per_image")].iloc[0]
    core_quality = bootstrap[(bootstrap.m11_scope == "THREE_SEED_MEAN") & (bootstrap.budget_scope == "CORE_10_15_20") & (bootstrap.metric == "quality_per_image")].iloc[0]
    cal = main[main.method == "CAL_TEMP_ALLOC"].sort_values("budget")
    sadapt = main[main.method == "S_ADAPT"].sort_values("budget")
    cal_parity = bool(
        np.array_equal(cal["selection_identity_sha256"].to_numpy(), sadapt["selection_identity_sha256"].to_numpy())
        and np.allclose(cal[["coverage_per_image", "quality_per_image", "AP", "AR100"]].to_numpy(), sadapt[["coverage_per_image", "quality_per_image", "AP", "AR100"]].to_numpy(), rtol=0, atol=0)
    )
    shift_summary = selection_analysis.groupby(["m11_seed", "budget"], as_index=False).agg(
        mean_abs_K_shift=("abs_delta_K", "mean"), mean_prefix_jaccard=("selected_record_jaccard", "mean"),
        changed_images=("same_prefix_selection", lambda x: int((~x).sum())),
    )
    report = f"""# LC-ALLOC COCO Route-B Evaluation Report

## 状态

- `execution_status = COMPLETE`
- `gt_access_status = ACCESSED_FOR_AUTHORIZED_EVALUATION_ONLY`
- `evaluation_status = COMPLETE`
- detector forward：0；candidate export：0；feature/allocator inference：0；selection/K/group/baseline 修改：0。

本次 evaluator 在验证冻结 selection SHA、30 个条件、3,450,000 条记录、Top-prefix 身份和 125×40 精确预算后，才解析官方 COCO2017 VAL annotation。selection SHA 为 `{EXPECTED_SELECTION_SHA}`；GT SHA 为 `{config['expected_hashes']['gt_sha256']}`；evaluator config SHA 为 `{config_sha}`。

## 冻结评价口径

- 标准检测质量：官方 COCO bbox evaluator，完整 5,000 张 VAL2017，原生类别 ID `[1,2,3,4,6,7,8,10]`，IoU `.50:.05:.95`，`maxDets=[1,10,100]`，使用原 detector score。
- M11 Coverage：有效、非 crowd Road8 GT 上，同类、IoU≥.50、maximum-cardinality one-to-one matching。
- M11 QUALITY：冻结 LC 类别权重下，十个 IoU 阈值 maximum matching 的加权均值。未在 COCO 上重估权重。
- 标准 COCO crowd/ignore 语义与 M11 有效非 crowd 分母分开处理，不混称。

有效非 crowd Road8 GT 共 **{gt_summary['valid_noncrowd_road8_gt']:,}** 个，分布于 **{gt_summary['images_with_valid_road8_gt']:,}** 张图；**{gt_summary['empty_valid_road8_gt_images']:,}** 张图没有有效 Road8 GT，但仍参与预算和 COCO 全 split 评价。Road8 官方 annotations 共 {gt_summary['official_road8_annotations']:,}，其中 crowd {gt_summary['crowd_excluded_for_m11']:,}、ignore {gt_summary['ignore_excluded_for_m11']:,}、M11 sanitation 排除的 invalid {gt_summary['invalid_excluded_for_m11']:,}。

## 主结果

下表的 M11 是三个 seed 的**结果算术平均**，不是 prediction ensemble；区间是 125 个完整 40-image groups 的 paired bootstrap（5,000 次，seed 530002）。AP/AR 是完整 VAL2017 点估计，不做 group AP 平均。

| K | S_ADAPT Cov/img | M11 Cov/img | Δ Cov/img [95% CI] | S_ADAPT QUALITY/img | M11 QUALITY/img | Δ QUALITY/img [95% CI] | S_ADAPT AP | M11 AP | S_ADAPT AR100 | M11 AR100 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{os.linesep.join(comparison_rows)}

核心预算 K10/K15/K20 等权平均：

- Coverage/image：M11−S_ADAPT = **{core_cov.point_estimate:+.6f}**，95% CI **[{core_cov.ci95_lower:+.6f}, {core_cov.ci95_upper:+.6f}]**，正增益重采样比例 {core_cov.positive_resample_fraction:.4f}。
- QUALITY/image：M11−S_ADAPT = **{core_quality.point_estimate:+.6f}**，95% CI **[{core_quality.ci95_lower:+.6f}, {core_quality.ci95_upper:+.6f}]**，正增益重采样比例 {core_quality.positive_resample_fraction:.4f}。

正增益比例不是传统 p-value。各 seed、五个预算、各类别与 COCO AP/AP50/AP75/AR100 的未平均结果保存在 `main_results.csv`、`class_results.csv` 和 `bootstrap_results.csv`。

## Calibration sanity check

`CAL_TEMP_ALLOC` 与 `S_ADAPT` 在五个预算的冻结 K 向量和 record prefix 上 exact parity：**{str(cal_parity).upper()}**。因此本次 evaluator 对完全相同的 selection 复用同一 COCOeval 结果；这是一项无损计算复用，不是新增估计或参数调整。

## Selection 行为

M11 与 S_ADAPT 的逐图预算差、prefix overlap 和 Jaccard 已写入 `selection_analysis.csv`。跨三个 seed×五预算的汇总范围：mean absolute K shift = {shift_summary.mean_abs_K_shift.min():.4f}–{shift_summary.mean_abs_K_shift.max():.4f}，mean prefix Jaccard = {shift_summary.mean_prefix_jaccard.min():.4f}–{shift_summary.mean_prefix_jaccard.max():.4f}；每个 40-image group 的 K 差之和均为 0，说明差异仅来自组内预算重分配。

## 完整性与限制

- `selected_records.parquet` 的全部 3,450,000 行均按 `(coco_image_id, road8_rank)` 回连 canonical candidate，record ID/query/class/score 均精确一致；所有选集为 rank `1..K_i`。
- COCOeval 使用原生 COCO category ID；Coverage/QUALITY 使用映射后的 M11 Road8 ID。没有把 LC 派生 GT 的 1..8 category ID 假设带入官方 COCO。
- 结果属于 LC allocator 的冻结 Route-B 跨数据迁移评价；没有 Route A、第二 detector 或论文修改。
- QUALITY 是冻结的加权 matching proxy，不是 COCO AP，也不保证所有类别无损。

总评价耗时（不含 detector/allocator）为 {elapsed:.1f} s。详情与可复算原始表见同目录输出。
"""
    (root / "evaluation_report.md").write_text(report, encoding="utf-8", newline="\n")


def write_ledger(root: Path, config: dict[str, Any]) -> None:
    rows = []
    for name, binding in config["input_bindings"].items():
        path = Path(binding["path"])
        rows.append({"role": "frozen_input", "name": name, "path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path), "note": "read-only bound input"})
    runner = root / "working" / "evaluate_route_b.py"
    rows.append({"role": "evaluation_code", "name": "route_b_evaluator", "path": str(runner.relative_to(root)), "bytes": runner.stat().st_size, "sha256": sha256_file(runner), "note": "new adapter; no detector/allocator entrypoint"})
    for name in (
        "evaluation_config.json", "evaluation_report.md", "coco_metrics_raw.json", "m11_metrics_raw.csv",
        "main_results.csv", "class_results.csv", "bootstrap_results.csv", "selection_analysis.csv",
    ):
        path = root / name
        rows.append({"role": "evaluation_output", "name": name, "path": name, "bytes": path.stat().st_size, "sha256": sha256_file(path), "note": "frozen evaluation output"})
    with (root / "sha256_ledger.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["role", "name", "path", "bytes", "sha256", "note"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--expected-config-sha", required=True)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    require(sha256_file(config_path) == args.expected_config_sha.lower(), "evaluation config SHA mismatch")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    root = Path(config["output_root"]).resolve()
    require(config_path == root / "evaluation_config.json", "config path is not output-root evaluation_config.json")
    require(sha256_file(Path(__file__).resolve()) == config["evaluator_adapter_sha256"], "evaluator adapter SHA mismatch")
    root.mkdir(parents=True, exist_ok=True)
    (root / "working").mkdir(exist_ok=True)
    config_sha = args.expected_config_sha.lower()
    start = time.perf_counter()
    append_log(root, "EVALUATION_START")

    validate_hash_bindings(config)
    groups, images = load_group_and_image_manifests(config)
    candidates, by_image = load_candidates(config)
    allocation = validate_frozen_selections(config, candidates, groups)
    append_log(root, f"SELECTION_FREEZE_VALIDATED_BEFORE_GT_OPEN sha={EXPECTED_SELECTION_SHA}")

    # GT is first parsed only after the frozen selection artifact has passed all identity and budget checks.
    coco_subset, gt_by_image, gt_summary = load_official_gt(config, set(by_image))
    append_log(root, f"AUTHORIZED_GT_OPEN sha={config['expected_hashes']['gt_sha256']}")
    p1_core = load_module("lc_alloc_frozen_p1_core_eval", Path(config["paths"]["p1_core"]))
    require(tuple(p1_core.ROAD8) == ROAD8 and np.array_equal(p1_core.THRESHOLDS, np.asarray([0.50 + 0.05 * i for i in range(10)], dtype=np.float64)), "frozen matching constants mismatch")
    weights_payload = json.loads(Path(config["paths"]["class_weights"]).read_text(encoding="utf-8"))
    require(tuple(weights_payload["classes"]) == ROAD8, "class weight order mismatch")
    weights = np.asarray(weights_payload["weights"], dtype=np.float64)
    require(weights.shape == (8,) and np.isfinite(weights).all(), "class weights invalid")
    prefix_cache = build_prefix_cache(p1_core, by_image, gt_by_image, root)
    raw, class_acc = evaluate_m11_metrics(allocation, prefix_cache, weights, gt_summary, root)
    raw.to_csv(root / "m11_metrics_raw.csv", index=False, lineterminator="\n")

    coco_results = evaluate_all_coco(config, allocation, by_image, images, coco_subset, root)
    coco_payload = {
        "dataset": "COCO2017-Road8-v1", "split": "VAL2017", "image_count": 5000,
        "road8_coco_category_ids": list(COCO_CAT_IDS), "iou_thresholds": [0.50 + 0.05 * i for i in range(10)],
        "recall_threshold_count": 101, "max_dets": [1, 10, 100], "score_field": "original detector score",
        "gt_sha256": config["expected_hashes"]["gt_sha256"], "selection_sha256": EXPECTED_SELECTION_SHA,
        "evaluation_config_sha256": config_sha, "gt_summary": gt_summary,
        "conditions": [
            {"method": key[0], "seed": key[1], "budget": key[2], **coco_results[key]}
            for key in sorted(coco_results, key=lambda x: (METHOD_ORDER[x[0]], x[1], x[2]))
        ],
    }
    write_json(root / "coco_metrics_raw.json", coco_payload)

    main_table, class_table = build_main_and_class_tables(raw, class_acc, coco_results, gt_summary, config_sha)
    main_table.to_csv(root / "main_results.csv", index=False, lineterminator="\n")
    class_table.to_csv(root / "class_results.csv", index=False, lineterminator="\n")
    bootstrap = build_bootstrap(raw, config)
    bootstrap.to_csv(root / "bootstrap_results.csv", index=False, lineterminator="\n")
    selection_analysis = build_selection_analysis(allocation)
    selection_analysis.to_csv(root / "selection_analysis.csv", index=False, lineterminator="\n")

    elapsed = time.perf_counter() - start
    write_report(root, config, config_sha, gt_summary, main_table, bootstrap, selection_analysis, elapsed)
    write_ledger(root, config)
    append_log(root, f"EVALUATION_COMPLETE elapsed_seconds={elapsed:.6f}")
    print(json.dumps({
        "execution_status": "COMPLETE", "gt_access_status": "ACCESSED_FOR_AUTHORIZED_EVALUATION_ONLY",
        "evaluation_status": "COMPLETE", "selection_sha256": EXPECTED_SELECTION_SHA,
        "gt_sha256": config["expected_hashes"]["gt_sha256"], "evaluation_config_sha256": config_sha,
        "elapsed_seconds": elapsed,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
