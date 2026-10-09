"""Evaluate committed LC-ALLOC-P2 calibration allocations on frozen DEV2K.

The scientific ordering is deliberate: this program validates both committed
selection files, their hashes, exact group budgets, and exact score-prefix
identity before it opens DEV ground truth.  Only the two new score-calibration
baselines are evaluated.  Frozen S_ADAPT and three-seed LEARN_QUALITY rows are
referenced from P1A and never regenerated here.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.dont_write_bytecode = True

BUDGETS = (10, 15, 20, 30, 40)
METHODS = ("CAL_TEMP_ALLOC", "CAL_ISO_ALLOC")
SEEDS = (530101, 530102, 530103)
ROAD8 = ("person", "bicycle", "car", "motorcycle", "bus", "train", "truck", "traffic light")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def append_log(root: Path, message: str) -> None:
    with (root / "runtime.log").open("a", encoding="utf-8") as f:
        f.write(message.rstrip() + "\n")


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), path, compression="zstd")


def import_file(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def resolve_commit_entry(root: Path, commit: dict[str, Any], stem: str) -> tuple[Path, str]:
    """Accept the documented nested entry and a narrow legacy flat spelling."""
    entry = commit.get(stem)
    if isinstance(entry, dict):
        raw_path, digest = entry.get("path"), entry.get("sha256")
    else:
        raw_path = commit.get(f"{stem}_path") or commit.get(f"{stem[:-1]}_path")
        digest = commit.get(f"{stem}_sha256") or commit.get(f"{stem[:-1]}_sha256")
    if not raw_path or not digest:
        raise RuntimeError(f"selection commit lacks {stem} path/SHA256")
    path = Path(str(raw_path))
    if not path.is_absolute():
        path = root / path
    return path.resolve(), str(digest).lower()


def load_candidates(release_root: Path, identity: dict) -> tuple[dict[str, list[dict]], dict[str, dict]]:
    by_image: dict[str, list[dict]] = defaultdict(list)
    by_record: dict[str, dict] = {}
    for rel in identity["assets"]["DEV"]["candidate_shards"]:
        for batch in pq.ParquetFile(release_root / rel).iter_batches(batch_size=32768):
            for row in batch.to_pylist():
                if int(row["road8_rank"]) > 100:
                    continue
                image_id = str(row["image_id"])
                record_id = str(row["candidate_record_id"])
                if record_id in by_record:
                    raise RuntimeError(f"duplicate candidate_record_id: {record_id}")
                by_record[record_id] = row
                by_image[image_id].append(row)
    for image_id, rows in by_image.items():
        rows.sort(key=lambda x: int(x["road8_rank"]))
        if [int(x["road8_rank"]) for x in rows] != list(range(1, 101)):
            raise RuntimeError(f"Top100 rank invariant failed: {image_id}")
    if len(by_image) != 2000 or len(by_record) != 200_000:
        raise RuntimeError(f"DEV candidate invariant failed: images={len(by_image)} records={len(by_record)}")
    return dict(by_image), by_record


def same_scalar(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return left is None and right is None
    if isinstance(left, (float, np.floating)) or isinstance(right, (float, np.floating)):
        return bool(np.isclose(float(left), float(right), rtol=0.0, atol=0.0, equal_nan=True))
    return str(left) == str(right)


def validate_committed_selection(
    allocations: pd.DataFrame,
    selections: pd.DataFrame,
    candidates: dict[str, list[dict]],
) -> tuple[dict[tuple[str, int, str], int], dict[str, int], dict[str, Any]]:
    alloc_required = {"image_id", "group_id", "method", "seed", "budget", "K_i", "group_predicted_objective"}
    sel_required = {"image_id", "group_id", "method", "seed", "budget", "K_i", "candidate_record_id", "road8_rank"}
    if missing := sorted(alloc_required - set(allocations.columns)):
        raise RuntimeError(f"allocation columns missing: {missing}")
    if missing := sorted(sel_required - set(selections.columns)):
        raise RuntimeError(f"selection columns missing: {missing}")

    allocations = allocations.copy()
    selections = selections.copy()
    allocations["image_id"] = allocations["image_id"].astype(str)
    selections["image_id"] = selections["image_id"].astype(str)
    allocations["method"] = allocations["method"].astype(str)
    selections["method"] = selections["method"].astype(str)
    if len(allocations) != 20_000 or len(selections) != 460_000:
        raise RuntimeError(f"committed table size mismatch: allocations={len(allocations)} selections={len(selections)}")
    if set(allocations["method"]) != set(METHODS) or set(selections["method"]) != set(METHODS):
        raise RuntimeError("method set mismatch")
    if set(allocations["budget"].astype(int)) != set(BUDGETS) or set(selections["budget"].astype(int)) != set(BUDGETS):
        raise RuntimeError("budget set mismatch")
    if set(allocations["seed"].astype(int)) != {-1} or set(selections["seed"].astype(int)) != {-1}:
        raise RuntimeError("calibration baselines must be deterministic seed=-1")
    if allocations.duplicated(["method", "budget", "image_id"]).any():
        raise RuntimeError("duplicate allocation key")
    if not np.isfinite(allocations["group_predicted_objective"].to_numpy(np.float64)).all():
        raise RuntimeError("non-finite group predicted objective")
    if not allocations["K_i"].between(5, 50).all():
        raise RuntimeError("allocation outside K bounds")
    if set(allocations["image_id"]) != set(candidates):
        raise RuntimeError("allocation image set differs from frozen DEV candidates")

    image_groups = allocations.groupby("image_id")["group_id"].nunique()
    if int(image_groups.max()) != 1:
        raise RuntimeError("image assigned to multiple frozen groups")
    group_sizes = allocations.drop_duplicates("image_id").groupby("group_id")["image_id"].nunique().sort_index()
    if group_sizes.index.astype(int).tolist() != list(range(50)) or not (group_sizes == 40).all():
        raise RuntimeError("frozen group layout is not 50 x 40")
    budget_check = allocations.groupby(["method", "budget", "group_id"], sort=True)["K_i"].sum().reset_index()
    if not np.array_equal(budget_check["K_i"].to_numpy(np.int64), budget_check["budget"].to_numpy(np.int64) * 40):
        raise RuntimeError("exact group budget constraint failed")

    lookup: dict[tuple[str, int, str], int] = {}
    group_map: dict[str, int] = {}
    for row in allocations.itertuples(index=False):
        key = (str(row.method), int(row.budget), str(row.image_id))
        lookup[key] = int(row.K_i)
        prior = group_map.setdefault(str(row.image_id), int(row.group_id))
        if prior != int(row.group_id):
            raise RuntimeError("group map mismatch")

    optional_exact_fields = (
        "dataset_version", "split", "image_sha256", "detector_id", "checkpoint_sha256",
        "export_config_sha256", "candidate_asset_id", "source_order",
        "score", "predicted_road8_class_id", "predicted_road8_class_name", "query_index",
        "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "bbox_cx", "bbox_cy", "bbox_w", "bbox_h",
    )
    checked_records = 0
    for key, frame in selections.groupby(["method", "budget", "image_id"], sort=False):
        method, budget, image_id = str(key[0]), int(key[1]), str(key[2])
        expected_k = lookup.get((method, budget, image_id))
        if expected_k is None:
            raise RuntimeError(f"selection has no allocation row: {key}")
        frame = frame.sort_values("road8_rank", kind="stable")
        if len(frame) != expected_k or frame["K_i"].astype(int).nunique() != 1 or int(frame["K_i"].iloc[0]) != expected_k:
            raise RuntimeError(f"selection K mismatch: {key}")
        if int(frame["group_id"].nunique()) != 1 or int(frame["group_id"].iloc[0]) != group_map[image_id]:
            raise RuntimeError(f"selection group mismatch: {key}")
        if frame["candidate_record_id"].astype(str).duplicated().any():
            raise RuntimeError(f"duplicate candidate in one selected set: {key}")
        ranks = frame["road8_rank"].astype(int).tolist()
        if ranks != list(range(1, expected_k + 1)):
            raise RuntimeError(f"selected set is not exact prefix: {key}")
        expected_rows = candidates[image_id][:expected_k]
        actual_ids = frame["candidate_record_id"].astype(str).tolist()
        expected_ids = [str(x["candidate_record_id"]) for x in expected_rows]
        if actual_ids != expected_ids:
            raise RuntimeError(f"candidate identity mismatch: {key}")
        for actual, expected in zip(frame.to_dict("records"), expected_rows):
            for field in optional_exact_fields:
                if field in actual and field in expected and not same_scalar(actual[field], expected[field]):
                    raise RuntimeError(f"frozen candidate field changed: {key}/{field}/{actual['candidate_record_id']}")
        checked_records += len(frame)
    if checked_records != 460_000 or len(lookup) != 20_000:
        raise RuntimeError("selection/allocation key coverage incomplete")

    return lookup, group_map, {
        "allocation_rows": len(allocations), "selection_rows": checked_records,
        "images": len(group_map), "groups": 50, "methods": list(METHODS), "budgets": list(BUDGETS),
        "all_prefixes_exact": True, "all_group_budgets_exact": True,
    }


def prepare_gt(raw: dict, shared) -> tuple[dict[str, list], dict, dict[str, int]]:
    ids = [str(x["source_image_id"]) for x in raw["images"]]
    remap = {image_id: ix + 1 for ix, image_id in enumerate(ids)}
    local: dict[str, list] = {image_id: [] for image_id in ids}
    source_order: defaultdict[str, int] = defaultdict(int)
    coco_images = []
    for row in raw["images"]:
        image_id = str(row["source_image_id"])
        coco_images.append({"id": remap[image_id], "file_name": row["file_name"], "width": row["width"], "height": row["height"]})
    coco_annotations = []
    for ann in raw["annotations"]:
        image_id = str(ann["image_id"])
        x, y, w, h = map(float, ann["bbox"])
        order = int(source_order[image_id])
        source_order[image_id] += 1
        local[image_id].append(shared.GroundTruth(
            image_id=image_id, gt_id=str(ann["id"]), category_id=int(ann["category_id"]),
            bbox_xyxy=(x, y, x + w, y + h), source_order=order,
        ))
        coco_annotations.append({
            "id": int(ann["id"]), "image_id": remap[image_id], "category_id": int(ann["category_id"]),
            "bbox": [x, y, w, h], "area": float(ann["area"]), "iscrowd": int(ann.get("iscrowd", 0)),
        })
    dataset = {
        "info": raw.get("info", {}), "licenses": raw.get("licenses", []),
        "images": coco_images, "categories": raw["categories"], "annotations": coco_annotations,
    }
    return local, dataset, remap


def coco_eval(
    dataset: dict,
    candidates: dict[str, list[dict]],
    kvals: dict[str, int],
    remap: dict[str, int],
    coco_site: Path,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    if str(coco_site) not in sys.path:
        sys.path.insert(0, str(coco_site))
    from pycocotools.coco import COCO  # type: ignore
    from pycocotools.cocoeval import COCOeval  # type: ignore

    gt = COCO()
    gt.dataset = dataset
    with contextlib.redirect_stdout(io.StringIO()):
        gt.createIndex()
    detections = []
    for image_id, k in kvals.items():
        for row in candidates[image_id][: int(k)]:
            x1, y1, x2, y2 = map(float, (row["bbox_x1"], row["bbox_y1"], row["bbox_x2"], row["bbox_y2"]))
            detections.append({
                "image_id": remap[image_id], "category_id": int(row["predicted_road8_class_id"]),
                "bbox": [x1, y1, x2 - x1, y2 - y1], "score": float(row["score"]),
            })
    with contextlib.redirect_stdout(io.StringIO()):
        dt = gt.loadRes(detections)

    def run(cat_ids: list[int] | None) -> np.ndarray:
        evaluator = COCOeval(gt, dt, "bbox")
        evaluator.params.imgIds = sorted(remap.values())
        evaluator.params.catIds = list(range(1, 9)) if cat_ids is None else cat_ids
        evaluator.params.iouThrs = np.linspace(0.50, 0.95, 10)
        evaluator.params.maxDets = [1, 10, 100]
        with contextlib.redirect_stdout(io.StringIO()):
            evaluator.evaluate()
            evaluator.accumulate()
            evaluator.summarize()
        return np.asarray(evaluator.stats, np.float64)

    all_stats = run(None)
    metrics = {"AP": float(all_stats[0]), "AP50": float(all_stats[1]), "AP75": float(all_stats[2]), "AR100": float(all_stats[8])}
    per_class = []
    for category_id, class_name in enumerate(ROAD8, 1):
        stats = run([category_id])
        per_class.append({
            "category_id": category_id, "class_name": class_name,
            "AP": float(stats[0]), "AP50": float(stats[1]), "AP75": float(stats[2]), "AR100": float(stats[8]),
        })
    return metrics, per_class


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--p1-root", required=True)
    parser.add_argument("--p1a-root", required=True)
    parser.add_argument("--release-root", required=True)
    parser.add_argument("--selection-commit", required=True)
    parser.add_argument("--coco-site-packages", required=True)
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    p1 = Path(args.p1_root).resolve()
    p1a = Path(args.p1a_root).resolve()
    release_root = Path(args.release_root).resolve()
    commit_path = Path(args.selection_commit).resolve()
    coco_site = Path(args.coco_site_packages).resolve()
    t0 = time.perf_counter()
    append_log(root, "STAGE evaluate_calibration START")

    # This release is public TRAIN/DEV only.  Fail closed on a restricted path;
    # do not inspect any such directory to determine what it contains.
    forbidden_tokens = ("restricted_test", "test_manifest_private", "reserve", "road1000", "holdout")
    for label, path in (("release", release_root), ("commit", commit_path), ("p1", p1), ("p1a", p1a)):
        folded = str(path).casefold()
        if any(token in folded for token in forbidden_tokens):
            raise RuntimeError(f"restricted-data path refused ({label})")

    identity_path = release_root / "CANDIDATE_ASSET_IDENTITY.json"
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    candidates, _ = load_candidates(release_root, identity)

    commit = json.loads(commit_path.read_text(encoding="utf-8"))
    allocation_path, allocation_sha = resolve_commit_entry(root, commit, "allocations")
    selection_path, selection_sha = resolve_commit_entry(root, commit, "selections")
    actual_alloc_sha = sha256_file(allocation_path)
    actual_select_sha = sha256_file(selection_path)
    if actual_alloc_sha != allocation_sha:
        raise RuntimeError("committed allocation SHA256 mismatch")
    if actual_select_sha != selection_sha:
        raise RuntimeError("committed selection SHA256 mismatch")
    allocations = pq.read_table(allocation_path).to_pandas()
    selections = pq.read_table(selection_path).to_pandas()
    lookup, group_map, validation = validate_committed_selection(allocations, selections, candidates)
    # Bind the submitted group labels to P1's frozen DEV grouping without
    # loading any GT-derived result columns from that parquet.
    frozen_groups = pq.read_table(
        p1a / "ablation_per_image_results.parquet", columns=["image_id", "group_id"]
    ).to_pandas().drop_duplicates()
    if len(frozen_groups) != 2000 or frozen_groups["image_id"].astype(str).duplicated().any():
        raise RuntimeError("frozen P1A group map is incomplete")
    expected_group_map = dict(zip(frozen_groups["image_id"].astype(str), frozen_groups["group_id"].astype(int)))
    if group_map != expected_group_map:
        mismatch = sum(group_map.get(image_id) != group_id for image_id, group_id in expected_group_map.items())
        raise RuntimeError(f"submitted allocations do not use the frozen DEV grouping: mismatch={mismatch}")
    validation.update({
        "status": "PASS", "commit_path": str(commit_path), "commit_sha256": sha256_file(commit_path),
        "allocations_path": str(allocation_path), "allocations_sha256": actual_alloc_sha,
        "selections_path": str(selection_path), "selections_sha256": actual_select_sha,
        "frozen_group_map_match": True, "dev_gt_opened": False,
    })
    write_json(root / "qa" / "selection_pre_gt_validation.json", validation)
    append_log(root, f"SELECTIONS_VERIFIED_BEFORE_DEV_GT_OPEN allocation_sha={actual_alloc_sha} selection_sha={actual_select_sha}")

    # No DEV GT is read before this line.
    shared = import_file(release_root / "evaluator" / "shared_evaluator.py", "lc_p2_shared_eval")
    if str(p1 / "scripts") not in sys.path:
        sys.path.insert(0, str(p1 / "scripts"))
    from p1_core import legacy_prefix_counts, prefix_matching_counts  # type: ignore

    gt_path = release_root / "gt" / "DEV2K_ROAD8_GT.json"
    raw_gt = json.loads(gt_path.read_text(encoding="utf-8"))
    gt_sha = sha256_file(gt_path)
    local_gt, coco_dataset, remap = prepare_gt(raw_gt, shared)
    image_ids = [str(row["source_image_id"]) for row in raw_gt["images"]]
    del raw_gt
    total_gt = sum(len(local_gt[image_id]) for image_id in image_ids)
    empty_gt_images = sum(not local_gt[image_id] for image_id in image_ids)
    if len(image_ids) != 2000 or total_gt != 29_966 or empty_gt_images != 3:
        raise RuntimeError("frozen DEV GT invariant failed")
    if set(image_ids) != set(candidates) or set(image_ids) != set(group_map):
        raise RuntimeError("DEV GT/candidate/allocation image identity mismatch")
    weights_doc = json.loads((p1 / "models" / "class_weights.json").read_text(encoding="utf-8"))
    weights = np.asarray(weights_doc["weights"], np.float64)
    if weights.shape != (8,) or not np.isfinite(weights).all():
        raise RuntimeError("frozen class weights invalid")

    cache: dict[str, dict[str, Any]] = {}
    shared_parity_mismatch = 0
    for image_index, image_id in enumerate(image_ids):
        rows = candidates[image_id]
        pred_classes = np.asarray([int(row["predicted_road8_class_id"]) for row in rows], np.int16)
        pred_boxes = np.asarray([[row["bbox_x1"], row["bbox_y1"], row["bbox_x2"], row["bbox_y2"]] for row in rows], np.float64)
        gt_classes = np.asarray([int(row.category_id) for row in local_gt[image_id]], np.int16)
        gt_boxes = np.asarray([row.bbox_xyxy for row in local_gt[image_id]], np.float64).reshape(-1, 4)
        totals, by_class = prefix_matching_counts(pred_classes, pred_boxes, gt_classes, gt_boxes, max_k=50)
        legacy, legacy_class = legacy_prefix_counts(pred_classes, pred_boxes, gt_classes, gt_boxes, max_k=50)
        selected_by_class = np.zeros((51, 8), np.int16)
        for k in range(1, 51):
            selected_by_class[k] = selected_by_class[k - 1]
            selected_by_class[k, pred_classes[k - 1] - 1] += 1
        graph = shared.build_image_graph([shared.Candidate.from_mapping(row) for row in rows], local_gt[image_id])
        for method in METHODS:
            for budget in BUDGETS:
                k = lookup[(method, budget, image_id)]
                result = shared.evaluate_selected(list(range(1, k + 1)), graph)
                shared_parity_mismatch += int(result.coverage_matching.cardinality != int(totals[k, 0]))
                shared_parity_mismatch += int(result.legacy_matching.cardinality != int(legacy[k]))
        cache[image_id] = {
            "gt": len(local_gt[image_id]),
            "gt_class": np.bincount(gt_classes, minlength=9)[1:9],
            "coverage": totals[:, 0],
            "coverage_class": by_class[:, :, 0],
            "legacy": legacy,
            "legacy_class": legacy_class,
            "quality": np.sum(weights[None, :, None] * by_class.astype(np.float64), axis=1).mean(axis=1),
            "multi_iou": totals.astype(np.float64).mean(axis=1),
            "class50": np.sum(weights[None, :] * by_class[:, :, 0].astype(np.float64), axis=1),
            "selected_class": selected_by_class,
        }
        if (image_index + 1) % 250 == 0:
            append_log(root, f"PREFIX_EVALUATION_PROGRESS images={image_index + 1}")
    if shared_parity_mismatch:
        raise RuntimeError(f"shared/prefix evaluator mismatch={shared_parity_mismatch}")

    per_rows: list[dict[str, Any]] = []
    class_acc: defaultdict[tuple[str, int, int], dict[str, int]] = defaultdict(
        lambda: {"GT": 0, "images_with_GT": 0, "selected": 0, "coverage": 0, "legacy": 0}
    )
    for method in METHODS:
        variant_id = "CAL_TEMP_SCORE_ONLY" if method == "CAL_TEMP_ALLOC" else "CAL_ISO_SCORE_ONLY"
        for budget in BUDGETS:
            for image_id in image_ids:
                k = lookup[(method, budget, image_id)]
                z = cache[image_id]
                gt = int(z["gt"])
                coverage = int(z["coverage"][k])
                legacy = int(z["legacy"][k])
                precision = legacy / k
                recall = legacy / gt if gt else 0.0
                per_rows.append({
                    "image_id": image_id, "group_id": group_map[image_id], "method": method, "seed": -1,
                    "budget": budget, "K_i": k, "GT": gt, "coverage": coverage,
                    "coverage_recall": coverage / gt if gt else 0.0,
                    "quality": float(z["quality"][k]), "legacy_TP": legacy,
                    "precision": precision, "recall": recall,
                    "F1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
                    "output_records": k, "empty_GT_image": not gt, "variant_id": variant_id,
                    "class_weight_on": False, "multi_threshold_on": False, "source": "NEW_P2",
                    "multi_iou_coverage": float(z["multi_iou"][k]),
                    "class50_weighted_coverage": float(z["class50"][k]),
                })
                for class_index in range(8):
                    accumulator = class_acc[(method, budget, class_index + 1)]
                    accumulator["GT"] += int(z["gt_class"][class_index])
                    accumulator["images_with_GT"] += int(z["gt_class"][class_index] > 0)
                    accumulator["selected"] += int(z["selected_class"][k, class_index])
                    accumulator["coverage"] += int(z["coverage_class"][k, class_index])
                    accumulator["legacy"] += int(z["legacy_class"][k, class_index])
    new_per = pd.DataFrame(per_rows)
    if len(new_per) != 20_000 or int(new_per["output_records"].sum()) != 460_000:
        raise RuntimeError("new per-image result invariant failed")
    write_parquet(root / "outputs" / "calibration_per_image.parquet", new_per)

    new_group = new_per.groupby(
        ["method", "seed", "budget", "group_id", "variant_id", "source"], as_index=False
    ).agg(
        image_count=("image_id", "size"), GT=("GT", "sum"), coverage=("coverage", "sum"),
        quality=("quality", "sum"), multi_iou_coverage=("multi_iou_coverage", "sum"),
        class50_weighted_coverage=("class50_weighted_coverage", "sum"),
        legacy_TP=("legacy_TP", "sum"), output_records=("output_records", "sum"),
    )
    new_group["coverage_per_image"] = new_group["coverage"] / new_group["image_count"]
    new_group["quality_per_image"] = new_group["quality"] / new_group["image_count"]
    new_group["multi_iou_coverage_per_image"] = new_group["multi_iou_coverage"] / new_group["image_count"]
    new_group["class50_weighted_coverage_per_image"] = new_group["class50_weighted_coverage"] / new_group["image_count"]
    if len(new_group) != 500:
        raise RuntimeError("new group result invariant failed")
    write_parquet(root / "outputs" / "calibration_group_results.parquet", new_group)

    # Mandatory frozen evaluator regression before evaluating the new COCO selections.
    p1a_per = pq.read_table(p1a / "ablation_per_image_results.parquet").to_pandas()
    old_s10 = p1a_per[(p1a_per["method"] == "S_ADAPT") & (p1a_per["seed"] == -1) & (p1a_per["budget"] == 10)]
    if len(old_s10) != 2000:
        raise RuntimeError("frozen S_ADAPT@10 per-image reference missing")
    old_k10 = dict(zip(old_s10["image_id"].astype(str), old_s10["K_i"].astype(int)))
    regression_metrics, _ = coco_eval(coco_dataset, candidates, old_k10, remap, coco_site)
    old_main_all = pq.read_table(p1a / "outputs" / "ablation_main_results.parquet").to_pandas()
    old_s10_main = old_main_all[(old_main_all["method"] == "S_ADAPT") & (old_main_all["seed"] == -1) & (old_main_all["budget"] == 10)]
    if len(old_s10_main) != 1:
        raise RuntimeError("frozen S_ADAPT@10 main reference missing")
    regression_delta = {name: float(regression_metrics[name] - old_s10_main.iloc[0][name]) for name in ("AP", "AP50", "AP75", "AR100")}
    if max(map(abs, regression_delta.values())) > 1e-12:
        raise RuntimeError(f"S_ADAPT@10 COCO regression failed: {regression_delta}")
    old_s10_cov = int(old_s10["coverage"].sum())
    old_s10_legacy = int(old_s10["legacy_TP"].sum())
    if old_s10_cov != int(old_s10_main.iloc[0]["coverage_total"]) or old_s10_legacy != int(old_s10_main.iloc[0]["legacy_TP"]):
        raise RuntimeError("S_ADAPT@10 coverage/legacy regression failed")
    append_log(root, "S_ADAPT_AT_10_EVALUATOR_REGRESSION PASS")

    coco_metrics: dict[tuple[str, int], dict[str, float]] = {}
    coco_classes: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for method in METHODS:
        for budget in BUDGETS:
            kvals = {image_id: lookup[(method, budget, image_id)] for image_id in image_ids}
            coco_metrics[(method, budget)], coco_classes[(method, budget)] = coco_eval(
                coco_dataset, candidates, kvals, remap, coco_site
            )
            append_log(root, f"COCO_EVALUATION_COMPLETE method={method} budget={budget}")

    main_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    for method in METHODS:
        variant_id = "CAL_TEMP_SCORE_ONLY" if method == "CAL_TEMP_ALLOC" else "CAL_ISO_SCORE_ONLY"
        for budget in BUDGETS:
            subset = new_per[(new_per["method"] == method) & (new_per["budget"] == budget)]
            retained = int(subset["output_records"].sum())
            coverage = int(subset["coverage"].sum())
            legacy = int(subset["legacy_TP"].sum())
            precision = legacy / retained
            recall = legacy / total_gt
            main_rows.append({
                "method": method, "seed": -1, "budget": budget, "image_count": 2000, "group_count": 50,
                "total_GT": total_gt, "empty_GT_images": empty_gt_images,
                "coverage_total": coverage, "coverage_per_image": coverage / 2000,
                "coverage_recall": coverage / total_gt,
                "quality_total": float(subset["quality"].sum()), "quality_per_image": float(subset["quality"].mean()),
                "legacy_TP": legacy, "legacy_FP": retained - legacy, "legacy_FN": total_gt - legacy,
                "precision": precision, "recall": recall,
                "F1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
                "output_records": retained, "K_mean": float(subset["K_i"].mean()),
                "K_median": float(subset["K_i"].median()), "K_min": int(subset["K_i"].min()), "K_max": int(subset["K_i"].max()),
                **coco_metrics[(method, budget)], "variant_id": variant_id,
                "class_weight_on": False, "multi_threshold_on": False, "source": "NEW_P2",
                "multi_iou_coverage_total": float(subset["multi_iou_coverage"].sum()),
                "multi_iou_coverage_per_image": float(subset["multi_iou_coverage"].mean()),
                "class50_weighted_coverage_total": float(subset["class50_weighted_coverage"].sum()),
                "class50_weighted_coverage_per_image": float(subset["class50_weighted_coverage"].mean()),
            })
            class_coco = {int(row["category_id"]): row for row in coco_classes[(method, budget)]}
            for category_id, class_name in enumerate(ROAD8, 1):
                accumulator = class_acc[(method, budget, category_id)]
                class_precision = accumulator["legacy"] / accumulator["selected"] if accumulator["selected"] else 0.0
                class_recall = accumulator["legacy"] / accumulator["GT"] if accumulator["GT"] else math.nan
                cmetric = class_coco[category_id]
                class_rows.append({
                    "method": method, "seed": -1, "budget": budget, "category_id": category_id, "class_name": class_name,
                    "GT": accumulator["GT"], "images_with_GT": accumulator["images_with_GT"],
                    "selected_records": accumulator["selected"], "coverage": accumulator["coverage"],
                    "coverage_recall": accumulator["coverage"] / accumulator["GT"] if accumulator["GT"] else math.nan,
                    "legacy_TP": accumulator["legacy"], "precision": class_precision, "recall": class_recall,
                    "F1": 2 * class_precision * class_recall / (class_precision + class_recall)
                    if accumulator["GT"] and class_precision + class_recall else 0.0,
                    "AP": cmetric["AP"], "AP50": cmetric["AP50"], "AP75": cmetric["AP75"], "AR100": cmetric["AR100"],
                    "variant_id": variant_id, "class_weight_on": False, "multi_threshold_on": False, "source": "NEW_P2",
                })

    new_main = pd.DataFrame(main_rows)
    new_class = pd.DataFrame(class_rows)
    if len(new_main) != 10 or len(new_class) != 80:
        raise RuntimeError("new aggregate result invariant failed")
    for (method, budget), frame in new_class.groupby(["method", "budget"]):
        expected = int(new_main[(new_main["method"] == method) & (new_main["budget"] == budget)]["coverage_total"].iloc[0])
        if int(frame["coverage"].sum()) != expected:
            raise RuntimeError(f"class coverage does not reconcile: {method}/{budget}")
    write_parquet(root / "outputs" / "calibration_new_main.parquet", new_main)
    write_parquet(root / "outputs" / "calibration_new_class.parquet", new_class)

    old_main = old_main_all[
        ((old_main_all["method"] == "S_ADAPT") & (old_main_all["seed"] == -1))
        | ((old_main_all["method"] == "LEARN_QUALITY") & old_main_all["seed"].isin(SEEDS))
    ].copy()
    old_group_all = pq.read_table(p1a / "outputs" / "ablation_group_results.parquet").to_pandas()
    old_group = old_group_all[
        ((old_group_all["method"] == "S_ADAPT") & (old_group_all["seed"] == -1))
        | ((old_group_all["method"] == "LEARN_QUALITY") & old_group_all["seed"].isin(SEEDS))
    ].copy()
    old_class_all = pq.read_table(p1a / "outputs" / "ablation_class_results.parquet").to_pandas()
    old_class = old_class_all[
        ((old_class_all["method"] == "S_ADAPT") & (old_class_all["seed"] == -1))
        | ((old_class_all["method"] == "LEARN_QUALITY") & old_class_all["seed"].isin(SEEDS))
    ].copy()
    combined_main = pd.concat([old_main, new_main], ignore_index=True, sort=False)
    combined_group = pd.concat([old_group, new_group], ignore_index=True, sort=False)
    combined_class = pd.concat([old_class, new_class], ignore_index=True, sort=False)
    if len(combined_main) != 30 or len(combined_group) != 1500 or len(combined_class) != 240:
        raise RuntimeError(
            f"combined table size mismatch main={len(combined_main)} group={len(combined_group)} class={len(combined_class)}"
        )
    for key, frame in combined_class.groupby(["method", "seed", "budget"]):
        match = combined_main[
            (combined_main["method"] == key[0]) & (combined_main["seed"] == key[1]) & (combined_main["budget"] == key[2])
        ]
        if len(match) != 1 or int(frame["coverage"].sum()) != int(match.iloc[0]["coverage_total"]):
            raise RuntimeError(f"combined class reconciliation failed: {key}")
    write_parquet(root / "outputs" / "calibration_combined_main.parquet", combined_main)
    write_parquet(root / "outputs" / "calibration_combined_group.parquet", combined_group)
    write_parquet(root / "outputs" / "calibration_combined_class.parquet", combined_class)

    validation["dev_gt_opened"] = True
    validation["dev_gt_sha256"] = gt_sha
    validation["shared_evaluator_parity_mismatch"] = shared_parity_mismatch
    validation["s_adapt_at_10_regression"] = {
        "coverage_total": old_s10_cov, "legacy_TP": old_s10_legacy, "coco_metric_delta": regression_delta,
    }
    validation["new_result_rows"] = {"per_image": 20_000, "group": 500, "main": 10, "class": 80}
    validation["combined_result_rows"] = {"group": 1500, "main": 30, "class": 240}
    validation["elapsed_seconds"] = time.perf_counter() - t0
    write_json(root / "qa" / "calibration_evaluation_validation.json", validation)
    append_log(root, f"STAGE evaluate_calibration COMPLETE elapsed_seconds={time.perf_counter() - t0:.6f}")


if __name__ == "__main__":
    main()
