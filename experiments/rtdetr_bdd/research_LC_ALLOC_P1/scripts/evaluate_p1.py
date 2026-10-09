"""Independent DEV evaluation for already committed LC-ALLOC-P1 selections."""

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
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from p1_core import ROAD8, THRESHOLDS, prefix_matching_counts, sha256_file, write_json  # noqa: E402


BUDGETS = (10, 15, 20, 30, 40)


def append_log(root: Path, text: str) -> None:
    with (root / "runtime.log").open("a", encoding="utf-8") as f:
        f.write(text.rstrip() + "\n")


def import_r0(path: Path):
    spec = importlib.util.spec_from_file_location("lc_alloc_p1_r0_eval", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_candidates(release_root: Path, identity: dict) -> tuple[dict[str, list[dict]], dict[str, dict]]:
    by_image: dict[str, list[dict]] = defaultdict(list)
    by_record = {}
    for rel in identity["assets"]["DEV"]["candidate_shards"]:
        for batch in pq.ParquetFile(release_root / rel).iter_batches(batch_size=32768):
            for row in batch.to_pylist():
                if int(row["road8_rank"]) > 100:
                    continue
                image_id = str(row["image_id"])
                record = str(row["candidate_record_id"])
                if record in by_record:
                    raise RuntimeError("duplicate DEV candidate record identity")
                by_image[image_id].append(row)
                by_record[record] = row
    for image_id, rows in by_image.items():
        rows.sort(key=lambda x: int(x["road8_rank"]))
        if [int(x["road8_rank"]) for x in rows] != list(range(1, 101)):
            raise RuntimeError(f"candidate Top100 rank invariant failed {image_id}")
    if len(by_image) != 2000 or len(by_record) != 200000:
        raise RuntimeError("DEV candidate coverage invariant failed")
    return dict(by_image), by_record


def condition_key(method: str, seed: int) -> str:
    return method if seed < 0 else f"{method}__SEED_{seed}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", required=True)
    ap.add_argument("--release-root", required=True)
    ap.add_argument("--group-manifest", required=True)
    ap.add_argument("--r0-evaluator", required=True)
    ap.add_argument("--coco-site-packages", required=True)
    args = ap.parse_args()
    root = Path(args.project_root).resolve()
    release_root = Path(args.release_root).resolve()
    t0 = time.perf_counter()
    append_log(root, "STAGE evaluate_p1 START")

    commit = json.loads((root / "outputs" / "dev_selection_commit.json").read_text(encoding="utf-8"))
    selection_path = Path(commit["dev_selections_path"])
    prediction_path = Path(commit["dev_predictions_path"])
    allocation_path = Path(commit["allocation_path"])
    if sha256_file(selection_path) != commit["dev_selections_sha256"] or sha256_file(prediction_path) != commit["dev_predictions_sha256"] or sha256_file(allocation_path) != commit["allocation_sha256"]:
        raise RuntimeError("committed DEV prediction/allocation/selection hash mismatch")

    identity = json.loads((release_root / "CANDIDATE_ASSET_IDENTITY.json").read_text(encoding="utf-8"))
    candidates, by_record = load_candidates(release_root, identity)
    group = pq.read_table(args.group_manifest).to_pandas()
    group_map = dict(zip(group["image_id"].astype(str), group["group_id"].astype(int)))
    allocations = pq.read_table(allocation_path).to_pandas()
    allocations["image_id"] = allocations["image_id"].astype(str)
    if len(allocations) != 100000:
        raise RuntimeError("allocation cells incomplete")
    alloc_lookup = {
        (str(r.method), int(r.seed), int(r.budget), str(r.image_id)): int(r.K_i)
        for r in allocations.itertuples(index=False)
    }
    conditions = sorted({(str(r.method), int(r.seed)) for r in allocations.itertuples(index=False)}, key=lambda z: (z[0], z[1]))
    if len(conditions) != 10:
        raise RuntimeError(f"condition count {len(conditions)} != 10")

    # Full committed-selection validation precedes opening DEV ground truth.
    seen_rows = 0
    cell_counts = Counter()
    cell_rank_sums = Counter()
    cell_rank_sq_sums = Counter()
    for batch in pq.ParquetFile(selection_path).iter_batches(batch_size=32768):
        for row in batch.to_pylist():
            image_id, method = str(row["image_id"]), str(row["method"])
            seed, budget, rank, k = int(row["seed"]), int(row["budget"]), int(row["road8_rank"]), int(row["K_i"])
            if (method, seed) not in conditions or budget not in BUDGETS or image_id not in candidates:
                raise RuntimeError("selection protocol identity mismatch")
            if rank < 1 or rank > k or int(row["selection_rank"]) != rank:
                raise RuntimeError("selection is not an exact raw-score prefix")
            canonical = candidates[image_id][rank - 1]
            if str(row["candidate_record_id"]) != str(canonical["candidate_record_id"]):
                raise RuntimeError("selected candidate identity mutation")
            for field in ("query_index", "predicted_road8_class_id", "score", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"):
                if row[field] != canonical[field]:
                    raise RuntimeError(f"selected candidate field mutation: {field}")
            if k != alloc_lookup[(method, seed, budget, image_id)] or int(row["group_id"]) != group_map[image_id]:
                raise RuntimeError("selection/allocation/group mismatch")
            cell = (method, seed, budget, image_id)
            cell_counts[cell] += 1
            cell_rank_sums[cell] += rank
            cell_rank_sq_sums[cell] += rank * rank
            seen_rows += 1
    bad_cells = 0
    for cell, n in cell_counts.items():
        k = alloc_lookup[cell]
        bad_cells += int(n != k or cell_rank_sums[cell] != k * (k + 1) // 2 or cell_rank_sq_sums[cell] != k * (k + 1) * (2 * k + 1) // 6)
    if seen_rows != 2_300_000 or len(cell_counts) != 100000 or bad_cells:
        raise RuntimeError("committed selection capacity mismatch")
    append_log(root, f"SELECTIONS_VERIFIED_BEFORE_DEV_GT_OPEN selection_sha={commit['dev_selections_sha256']}")

    # Only now is DEV GT opened.
    r0 = import_r0(Path(args.r0_evaluator).resolve())
    shared = r0.load_shared(release_root)
    manifest = pq.read_table(release_root / identity["assets"]["DEV"]["split_manifest_path"]).to_pylist()
    ids = [str(x["image_id"]) for x in manifest]
    gt_path = release_root / "gt" / "DEV2K_ROAD8_GT.json"
    actual_gt_sha = sha256_file(gt_path)
    with gt_path.open("r", encoding="utf-8") as f:
        raw_gt = json.load(f)
    local_gt, coco_gt, remap = r0.prepare_gt(raw_gt, manifest, shared)
    del raw_gt
    if len(ids) != 2000 or sum(len(local_gt[x]) for x in ids) != 29966 or sum(not local_gt[x] for x in ids) != 3:
        raise RuntimeError("DEV GT frozen count mismatch")

    class_weights = np.asarray(json.loads((root / "models" / "class_weights.json").read_text(encoding="utf-8"))["weights"], dtype=np.float64)
    cache = {}
    prefix_parity_mismatch = 0
    for image_ix, image_id in enumerate(ids):
        rows = candidates[image_id]
        cand_obj = [shared.Candidate.from_mapping(x) for x in rows]
        graph = shared.build_image_graph(cand_obj, local_gt[image_id])
        cand_classes = np.asarray([int(x["predicted_road8_class_id"]) for x in rows], dtype=np.int16)
        cand_boxes = np.asarray([[x["bbox_x1"], x["bbox_y1"], x["bbox_x2"], x["bbox_y2"]] for x in rows], dtype=np.float64)
        gt_classes = np.asarray([int(x.category_id) for x in local_gt[image_id]], dtype=np.int16)
        gt_boxes = np.asarray([x.bbox_xyxy for x in local_gt[image_id]], dtype=np.float64).reshape(-1, 4)
        totals, by_class_t = prefix_matching_counts(cand_classes, cand_boxes, gt_classes, gt_boxes, max_k=50)
        pred_class_prefix = np.zeros((51, 8), dtype=np.int16)
        for k in range(1, 51):
            pred_class_prefix[k] = pred_class_prefix[k - 1]
            pred_class_prefix[k, cand_classes[k - 1] - 1] += 1
        unique_k = sorted({alloc_lookup[(m, s, b, image_id)] for m, s in conditions for b in BUDGETS})
        standard = {}
        for k in unique_k:
            result = shared.evaluate_selected(list(range(1, k + 1)), graph)
            coverage_class = np.zeros(8, dtype=np.int16)
            legacy_class = np.zeros(8, dtype=np.int16)
            for pair in result.coverage_matching.pairs:
                coverage_class[int(pair.category_id) - 1] += 1
            for pair in result.legacy_matching.pairs:
                legacy_class[int(pair.category_id) - 1] += 1
            prefix_parity_mismatch += int(int(totals[k, 0]) != result.coverage_matching.cardinality)
            standard[k] = {
                "coverage": result.coverage_matching.cardinality,
                "legacy": result.legacy_matching.cardinality,
                "coverage_class": coverage_class,
                "legacy_class": legacy_class,
            }
        quality = np.sum(class_weights[None, :, None] * by_class_t.astype(np.float64), axis=1).mean(axis=1)
        cache[image_id] = {
            "gt": len(local_gt[image_id]), "gt_class": np.bincount(gt_classes, minlength=9)[1:9],
            "standard": standard, "quality": quality, "pred_class_prefix": pred_class_prefix,
        }
        if (image_ix + 1) % 250 == 0:
            append_log(root, f"PREFIX_EVAL_PROGRESS images={image_ix+1}")
    if prefix_parity_mismatch:
        raise RuntimeError(f"parameterized IoU.50/shared evaluator parity mismatch={prefix_parity_mismatch}")

    def selection_for(condition: tuple[str, int], budget: int) -> dict:
        method, seed = condition
        tag = condition_key(method, seed)
        return {(tag, budget, image_id): list(range(1, alloc_lookup[(method, seed, budget, image_id)] + 1)) for image_id in ids}

    # Mandatory S_ADAPT@10 regression before evaluating new conditions.
    anchor_condition = ("S_ADAPT", -1)
    anchor_cov = sum(cache[x]["standard"][alloc_lookup[("S_ADAPT", -1, 10, x)]]["coverage"] for x in ids)
    anchor_legacy = sum(cache[x]["standard"][alloc_lookup[("S_ADAPT", -1, 10, x)]]["legacy"] for x in ids)
    anchor_tag = condition_key(*anchor_condition)
    anchor_coco, anchor_coco_class = r0.evaluate_coco(
        coco_gt, candidates, selection_for(anchor_condition, 10), anchor_tag, 10, remap, args.coco_site_packages,
    )
    frozen_anchor = {"coverage": 13818, "legacy_TP": 13811, "AP": 0.1536753879858031, "AR100": 0.2034422345381616}
    if anchor_cov != frozen_anchor["coverage"] or anchor_legacy != frozen_anchor["legacy_TP"] or not np.isclose(anchor_coco["AP"], frozen_anchor["AP"], atol=1e-6, rtol=1e-5) or not np.isclose(anchor_coco["AR100"], frozen_anchor["AR100"], atol=1e-6, rtol=1e-5):
        raise RuntimeError(f"S_ADAPT@10 evaluator regression failed cov={anchor_cov} legacy={anchor_legacy} coco={anchor_coco}")
    append_log(root, "S_ADAPT_AT_10_EVALUATOR_REGRESSION PASS")

    per_image_rows = []
    class_acc = defaultdict(lambda: {"GT": 0, "images_with_GT": 0, "selected": 0, "coverage": 0, "legacy": 0})
    for method, seed in conditions:
        for budget in BUDGETS:
            for image_id in ids:
                k = alloc_lookup[(method, seed, budget, image_id)]
                z = cache[image_id]
                s = z["standard"][k]
                precision = s["legacy"] / k
                recall = s["legacy"] / z["gt"] if z["gt"] else 0.0
                per_image_rows.append({
                    "image_id": image_id, "group_id": group_map[image_id], "method": method, "seed": seed,
                    "budget": budget, "K_i": k, "GT": z["gt"], "coverage": s["coverage"],
                    "coverage_recall": s["coverage"] / z["gt"] if z["gt"] else 0.0,
                    "quality": float(z["quality"][k]), "legacy_TP": s["legacy"],
                    "precision": precision, "recall": recall,
                    "F1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
                    "output_records": k, "empty_GT_image": not z["gt"],
                })
                for ci in range(8):
                    a = class_acc[(method, seed, budget, ci + 1)]
                    a["GT"] += int(z["gt_class"][ci]); a["images_with_GT"] += int(z["gt_class"][ci] > 0)
                    a["selected"] += int(z["pred_class_prefix"][k, ci])
                    a["coverage"] += int(s["coverage_class"][ci]); a["legacy"] += int(s["legacy_class"][ci])
    per_image = pd.DataFrame(per_image_rows)
    pq.write_table(pa.Table.from_pandas(per_image, preserve_index=False), root / "per_image_results.parquet", compression="zstd")
    group_results = per_image.groupby(["method", "seed", "budget", "group_id"], as_index=False).agg(
        image_count=("image_id", "size"), GT=("GT", "sum"), coverage=("coverage", "sum"),
        quality=("quality", "sum"), legacy_TP=("legacy_TP", "sum"), output_records=("output_records", "sum"),
    )
    group_results["coverage_per_image"] = group_results["coverage"] / group_results["image_count"]
    group_results["quality_per_image"] = group_results["quality"] / group_results["image_count"]
    pq.write_table(pa.Table.from_pandas(group_results, preserve_index=False), root / "outputs" / "group_results.parquet", compression="zstd")

    coco_map = {("S_ADAPT", -1, 10): (anchor_coco, anchor_coco_class)}
    for condition in conditions:
        method, seed = condition
        for budget in BUDGETS:
            if (method, seed, budget) in coco_map:
                continue
            tag = condition_key(method, seed)
            metrics, cls = r0.evaluate_coco(
                coco_gt, candidates, selection_for(condition, budget), tag, budget, remap, args.coco_site_packages,
            )
            coco_map[(method, seed, budget)] = (metrics, cls)
        append_log(root, f"COCO_EVAL_COMPLETE method={method} seed={seed}")

    main_rows, class_rows = [], []
    total_gt = sum(len(local_gt[x]) for x in ids)
    for method, seed in conditions:
        for budget in BUDGETS:
            d = per_image[(per_image.method == method) & (per_image.seed == seed) & (per_image.budget == budget)]
            coverage, legacy, retained = int(d.coverage.sum()), int(d.legacy_TP.sum()), int(d.output_records.sum())
            p, r = legacy / retained, legacy / total_gt
            metrics, clsmetrics = coco_map[(method, seed, budget)]
            main_rows.append({
                "method": method, "seed": seed, "budget": budget, "image_count": len(d), "group_count": 50,
                "total_GT": total_gt, "empty_GT_images": int(d.empty_GT_image.sum()),
                "coverage_total": coverage, "coverage_per_image": coverage / len(d), "coverage_recall": coverage / total_gt,
                "quality_total": float(d.quality.sum()), "quality_per_image": float(d.quality.mean()),
                "legacy_TP": legacy, "legacy_FP": retained - legacy, "legacy_FN": total_gt - legacy,
                "precision": p, "recall": r, "F1": 2 * p * r / (p + r) if p + r else 0.0,
                "output_records": retained, "K_mean": float(d.K_i.mean()), "K_median": float(d.K_i.median()),
                "K_min": int(d.K_i.min()), "K_max": int(d.K_i.max()), "AP": metrics["AP"],
                "AP50": metrics["AP50"], "AP75": metrics["AP75"], "AR100": metrics["AR100"],
            })
            ap_by_id = {int(x["category_id"]): x for x in clsmetrics}
            for ci, cname in enumerate(ROAD8, 1):
                a = class_acc[(method, seed, budget, ci)]
                pp = a["legacy"] / a["selected"] if a["selected"] else 0.0
                rr = a["legacy"] / a["GT"] if a["GT"] else math.nan
                class_rows.append({
                    "method": method, "seed": seed, "budget": budget, "category_id": ci, "class_name": cname,
                    "GT": a["GT"], "images_with_GT": a["images_with_GT"], "selected_records": a["selected"],
                    "coverage": a["coverage"], "coverage_recall": a["coverage"] / a["GT"] if a["GT"] else math.nan,
                    "legacy_TP": a["legacy"], "precision": pp, "recall": rr,
                    "F1": 2 * pp * rr / (pp + rr) if a["GT"] and pp + rr else 0.0,
                    "AP": ap_by_id[ci]["AP"], "AP50": ap_by_id[ci]["AP50"],
                    "AP75": ap_by_id[ci]["AP75"], "AR100": ap_by_id[ci]["AR100"],
                })
    main_df, class_df = pd.DataFrame(main_rows), pd.DataFrame(class_rows)
    if len(main_df) != 50 or len(class_df) != 400:
        raise RuntimeError("final evaluation table row counts invalid")
    for key, d in class_df.groupby(["method", "seed", "budget"]):
        expected_cov = int(main_df[(main_df.method == key[0]) & (main_df.seed == key[1]) & (main_df.budget == key[2])].coverage_total.iloc[0])
        if int(d.coverage.sum()) != expected_cov:
            raise RuntimeError("class coverage sum != overall coverage")
    pq.write_table(pa.Table.from_pandas(main_df, preserve_index=False), root / "main_results.parquet", compression="zstd")
    pq.write_table(pa.Table.from_pandas(class_df, preserve_index=False), root / "class_results.parquet", compression="zstd")
    write_json(root / "outputs" / "evaluation_summary.json", {
        "status": "PASS", "dev_gt_opened_after_selection_commit": True,
        "dev_gt_sha256": actual_gt_sha, "selection_sha256": commit["dev_selections_sha256"],
        "images": 2000, "valid_road8_gt": total_gt, "empty_gt_images": 3,
        "shared_iou050_parity_mismatch": prefix_parity_mismatch,
        "s_adapt_10_regression": {"coverage": anchor_cov, "legacy_TP": anchor_legacy, **anchor_coco},
        "main_rows": len(main_df), "class_rows": len(class_df), "elapsed_seconds": time.perf_counter() - t0,
    })
    append_log(root, f"STAGE evaluate_p1 COMPLETE elapsed_seconds={time.perf_counter()-t0:.6f}")


if __name__ == "__main__":
    main()
