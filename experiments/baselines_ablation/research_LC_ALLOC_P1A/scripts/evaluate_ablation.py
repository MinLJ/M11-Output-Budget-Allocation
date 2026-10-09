"""Independent DEV evaluation of committed LC-ALLOC-P1A allocations."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.dont_write_bytecode = True

SEEDS = (530101, 530102, 530103)
BUDGETS = (10, 15, 20, 30, 40)
NEW_POLICIES = ("LEARN_CLASS50", "LEARN_MULTI")
REUSED_POLICIES = ("LEARN_MICRO", "LEARN_QUALITY", "S_ADAPT")
ROAD8 = ("person", "bicycle", "car", "motorcycle", "bus", "train", "truck", "traffic light")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_log(root: Path, text: str) -> None:
    with (root / "runtime.log").open("a", encoding="utf-8") as f:
        f.write(text.rstrip() + "\n")


def import_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_candidates(release_root: Path, identity: dict) -> tuple[dict[str, list[dict]], dict[str, dict]]:
    by_image: dict[str, list[dict]] = defaultdict(list)
    by_record: dict[str, dict] = {}
    for rel in identity["assets"]["DEV"]["candidate_shards"]:
        for batch in pq.ParquetFile(release_root / rel).iter_batches(batch_size=32768):
            for row in batch.to_pylist():
                if int(row["road8_rank"]) > 100:
                    continue
                image_id, record = str(row["image_id"]), str(row["candidate_record_id"])
                if record in by_record:
                    raise RuntimeError("duplicate frozen candidate_record_id")
                by_image[image_id].append(row)
                by_record[record] = row
    for image_id, rows in by_image.items():
        rows.sort(key=lambda x: int(x["road8_rank"]))
        if [int(x["road8_rank"]) for x in rows] != list(range(1, 101)):
            raise RuntimeError(f"candidate Top100 invariant failed: {image_id}")
    if len(by_image) != 2000 or len(by_record) != 200000:
        raise RuntimeError("DEV candidate coverage invariant failed")
    return dict(by_image), by_record


def variant_metadata(method: str) -> tuple[str, bool, bool]:
    return {
        "LEARN_MICRO": ("M00", False, False),
        "LEARN_CLASS50": ("M10", True, False),
        "LEARN_MULTI": ("M01", False, True),
        "LEARN_QUALITY": ("M11", True, True),
        "S_ADAPT": ("BASELINE", False, False),
    }[method]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", required=True)
    ap.add_argument("--p1-root", required=True)
    ap.add_argument("--r0-evaluator", required=True)
    ap.add_argument("--coco-site-packages", required=True)
    args = ap.parse_args()
    root = Path(args.project_root).resolve()
    p1 = Path(args.p1_root).resolve()
    t0 = time.perf_counter()
    append_log(root, "STAGE evaluate_ablation START")

    p1_scripts = p1 / "scripts"
    sys.path.insert(0, str(p1_scripts))
    from p1_core import THRESHOLDS, prefix_matching_counts  # type: ignore

    config = json.loads((root / "run_config.json").read_text(encoding="utf-8"))
    release_root = Path(config["release_root"])
    commit = json.loads((root / "outputs" / "new_allocation_commit.json").read_text(encoding="utf-8"))
    allocation_path = Path(commit["new_allocations_path"])
    if sha256_file(allocation_path) != commit["new_allocations_sha256"]:
        raise RuntimeError("committed P1A allocation SHA mismatch")
    p1_commit = json.loads((p1 / "outputs" / "dev_selection_commit.json").read_text(encoding="utf-8"))
    for path_key, sha_key in (
        ("dev_predictions_path", "dev_predictions_sha256"),
        ("dev_selections_path", "dev_selections_sha256"),
        ("allocation_path", "allocation_sha256"),
    ):
        if sha256_file(Path(p1_commit[path_key])) != p1_commit[sha_key]:
            raise RuntimeError(f"frozen P1 hash mismatch: {path_key}")

    identity = json.loads((release_root / "CANDIDATE_ASSET_IDENTITY.json").read_text(encoding="utf-8"))
    candidates, by_record = load_candidates(release_root, identity)
    predictions = pq.read_table(p1 / "dev_predictions.parquet", columns=["image_id", "group_id"]).to_pandas().drop_duplicates()
    predictions["image_id"] = predictions["image_id"].astype(str)
    group_map = dict(zip(predictions["image_id"], predictions["group_id"].astype(int)))
    if len(group_map) != 2000:
        raise RuntimeError("frozen group map incomplete")

    allocations = pq.read_table(allocation_path).to_pandas()
    allocations["image_id"] = allocations["image_id"].astype(str)
    if len(allocations) != 60_000 or set(allocations["policy"]) != set(NEW_POLICIES):
        raise RuntimeError("new allocation cells incomplete")
    alloc_lookup: dict[tuple[str, int, int, str], int] = {}
    for r in allocations.itertuples(index=False):
        image_id, k = str(r.image_id), int(r.K_i)
        ids = list(r.selected_record_ids)
        expected = [str(x["candidate_record_id"]) for x in candidates[image_id][:k]]
        if ids != expected or len(ids) != k or int(r.group_id) != group_map[image_id]:
            raise RuntimeError("new selection is not the frozen raw-score prefix")
        alloc_lookup[(str(r.policy), int(r.seed), int(r.budget), image_id)] = k
    budget_check = allocations.groupby(["policy", "seed", "budget", "group_id"])["K_i"].sum().reset_index()
    if not np.array_equal(budget_check["K_i"].to_numpy(), budget_check["budget"].to_numpy() * 40):
        raise RuntimeError("new group budget mismatch")
    append_log(root, f"ALLOCATIONS_VERIFIED_BEFORE_DEV_GT_OPEN sha={commit['new_allocations_sha256']}")

    # Frozen P1 scientific outputs are read as explicitly authorized reused evidence.
    p1_per = pq.read_table(p1 / "per_image_results.parquet").to_pandas()
    p1_main = pq.read_table(p1 / "main_results.parquet").to_pandas()
    p1_class = pq.read_table(p1 / "class_results.parquet").to_pandas()
    reused_per = p1_per[p1_per["method"].isin(REUSED_POLICIES)].copy()
    reused_main = p1_main[p1_main["method"].isin(REUSED_POLICIES)].copy()
    reused_class = p1_class[p1_class["method"].isin(REUSED_POLICIES)].copy()
    if len(reused_per) != 70_000 or len(reused_main) != 35 or len(reused_class) != 280:
        raise RuntimeError("frozen P1 reused-result row counts changed")

    # Only after identity/selection verification is DEV GT opened.
    r0 = import_module(Path(args.r0_evaluator).resolve(), "lc_alloc_p1a_r0_eval")
    shared = r0.load_shared(release_root)
    manifest = pq.read_table(release_root / identity["assets"]["DEV"]["split_manifest_path"]).to_pylist()
    ids = [str(x["image_id"]) for x in manifest]
    gt_path = release_root / "gt" / "DEV2K_ROAD8_GT.json"
    actual_gt_sha = sha256_file(gt_path)
    with gt_path.open("r", encoding="utf-8") as f:
        raw_gt = json.load(f)
    local_gt, coco_gt, remap = r0.prepare_gt(raw_gt, manifest, shared)
    del raw_gt
    total_gt = sum(len(local_gt[x]) for x in ids)
    if len(ids) != 2000 or total_gt != 29966 or sum(not local_gt[x] for x in ids) != 3:
        raise RuntimeError("frozen DEV GT invariant failed")
    weights = np.asarray(json.loads((p1 / "models" / "class_weights.json").read_text(encoding="utf-8"))["weights"], np.float64)

    reused_k = {
        (str(r.method), int(r.seed), int(r.budget), str(r.image_id)): int(r.K_i)
        for r in reused_per.itertuples(index=False)
    }
    cache: dict[str, dict] = {}
    parity_mismatch = 0
    for ix, image_id in enumerate(ids):
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
        needed = {alloc_lookup[(m, s, b, image_id)] for m in NEW_POLICIES for s in SEEDS for b in BUDGETS}
        needed |= {reused_k[(m, s, b, image_id)] for m, s in [("S_ADAPT", -1)] + [(m, s) for m in ("LEARN_MICRO", "LEARN_QUALITY") for s in SEEDS] for b in BUDGETS}
        standard: dict[int, dict] = {}
        for k in sorted(needed):
            result = shared.evaluate_selected(list(range(1, k + 1)), graph)
            coverage_class = np.zeros(8, dtype=np.int16)
            legacy_class = np.zeros(8, dtype=np.int16)
            for pair in result.coverage_matching.pairs:
                coverage_class[int(pair.category_id) - 1] += 1
            for pair in result.legacy_matching.pairs:
                legacy_class[int(pair.category_id) - 1] += 1
            parity_mismatch += int(int(totals[k, 0]) != result.coverage_matching.cardinality)
            standard[k] = {
                "coverage": int(result.coverage_matching.cardinality),
                "legacy": int(result.legacy_matching.cardinality),
                "coverage_class": coverage_class, "legacy_class": legacy_class,
            }
        cache[image_id] = {
            "gt": len(local_gt[image_id]),
            "gt_class": np.bincount(gt_classes, minlength=9)[1:9],
            "standard": standard,
            "quality": np.sum(weights[None, :, None] * by_class_t.astype(np.float64), axis=1).mean(axis=1),
            "multi_iou_coverage": totals.astype(np.float64).mean(axis=1),
            "class50_weighted_coverage": np.sum(weights[None, :] * by_class_t[:, :, 0].astype(np.float64), axis=1),
            "pred_class_prefix": pred_class_prefix,
        }
        if (ix + 1) % 250 == 0:
            append_log(root, f"PREFIX_EVAL_PROGRESS images={ix+1}")
    if parity_mismatch:
        raise RuntimeError(f"IoU.50 evaluator parity mismatch={parity_mismatch}")

    per_rows: list[dict] = []
    class_acc = defaultdict(lambda: {"GT": 0, "images_with_GT": 0, "selected": 0, "coverage": 0, "legacy": 0})
    for policy in NEW_POLICIES:
        variant, class_on, multi_on = variant_metadata(policy)
        for seed in SEEDS:
            for budget in BUDGETS:
                for image_id in ids:
                    k = alloc_lookup[(policy, seed, budget, image_id)]
                    z, s = cache[image_id], cache[image_id]["standard"][k]
                    precision = s["legacy"] / k
                    recall = s["legacy"] / z["gt"] if z["gt"] else 0.0
                    per_rows.append({
                        "image_id": image_id, "group_id": group_map[image_id], "method": policy, "seed": seed,
                        "variant_id": variant, "class_weight_on": class_on, "multi_threshold_on": multi_on,
                        "source": "NEW_P1A", "budget": budget, "K_i": k, "GT": z["gt"],
                        "coverage": s["coverage"], "coverage_recall": s["coverage"] / z["gt"] if z["gt"] else 0.0,
                        "quality": float(z["quality"][k]),
                        "multi_iou_coverage": float(z["multi_iou_coverage"][k]),
                        "class50_weighted_coverage": float(z["class50_weighted_coverage"][k]),
                        "legacy_TP": s["legacy"], "precision": precision, "recall": recall,
                        "F1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
                        "output_records": k, "empty_GT_image": not z["gt"],
                    })
                    for ci in range(8):
                        a = class_acc[(policy, seed, budget, ci + 1)]
                        a["GT"] += int(z["gt_class"][ci]); a["images_with_GT"] += int(z["gt_class"][ci] > 0)
                        a["selected"] += int(z["pred_class_prefix"][k, ci])
                        a["coverage"] += int(s["coverage_class"][ci]); a["legacy"] += int(s["legacy_class"][ci])
    new_per = pd.DataFrame(per_rows)
    if len(new_per) != 60_000:
        raise RuntimeError("new per-image result row count mismatch")

    # Add the optional common-prefix measures to reused P1 rows without changing
    # any frozen P1 metric or selection.
    reused_per["variant_id"] = reused_per["method"].map(lambda x: variant_metadata(str(x))[0])
    reused_per["class_weight_on"] = reused_per["method"].map(lambda x: variant_metadata(str(x))[1])
    reused_per["multi_threshold_on"] = reused_per["method"].map(lambda x: variant_metadata(str(x))[2])
    reused_per["source"] = "REUSED_P1"
    reused_per["multi_iou_coverage"] = [float(cache[str(r.image_id)]["multi_iou_coverage"][int(r.K_i)]) for r in reused_per.itertuples(index=False)]
    reused_per["class50_weighted_coverage"] = [float(cache[str(r.image_id)]["class50_weighted_coverage"][int(r.K_i)]) for r in reused_per.itertuples(index=False)]
    per_image = pd.concat([reused_per, new_per], ignore_index=True, sort=False)
    if len(per_image) != 130_000:
        raise RuntimeError("combined per-image row count mismatch")
    pq.write_table(pa.Table.from_pandas(per_image, preserve_index=False), root / "ablation_per_image_results.parquet", compression="zstd")

    group_results = per_image.groupby(["method", "seed", "budget", "group_id", "variant_id", "source"], as_index=False).agg(
        image_count=("image_id", "size"), GT=("GT", "sum"), coverage=("coverage", "sum"), quality=("quality", "sum"),
        multi_iou_coverage=("multi_iou_coverage", "sum"), class50_weighted_coverage=("class50_weighted_coverage", "sum"),
        legacy_TP=("legacy_TP", "sum"), output_records=("output_records", "sum"),
    )
    group_results["coverage_per_image"] = group_results["coverage"] / group_results["image_count"]
    group_results["quality_per_image"] = group_results["quality"] / group_results["image_count"]
    group_results["multi_iou_coverage_per_image"] = group_results["multi_iou_coverage"] / group_results["image_count"]
    group_results["class50_weighted_coverage_per_image"] = group_results["class50_weighted_coverage"] / group_results["image_count"]
    pq.write_table(pa.Table.from_pandas(group_results, preserve_index=False), root / "outputs" / "ablation_group_results.parquet", compression="zstd")

    def selection_for(policy: str, seed: int, budget: int) -> dict:
        tag = f"{policy}__SEED_{seed}"
        return {(tag, budget, image_id): list(range(1, alloc_lookup[(policy, seed, budget, image_id)] + 1)) for image_id in ids}

    coco_map: dict[tuple[str, int, int], tuple[dict, list[dict]]] = {}
    for policy in NEW_POLICIES:
        for seed in SEEDS:
            tag = f"{policy}__SEED_{seed}"
            for budget in BUDGETS:
                coco_map[(policy, seed, budget)] = r0.evaluate_coco(
                    coco_gt, candidates, selection_for(policy, seed, budget), tag, budget, remap, args.coco_site_packages,
                )
            append_log(root, f"COCO_EVAL_COMPLETE policy={policy} seed={seed}")

    main_rows: list[dict] = []
    class_rows: list[dict] = []
    for policy in NEW_POLICIES:
        variant, class_on, multi_on = variant_metadata(policy)
        for seed in SEEDS:
            for budget in BUDGETS:
                d = new_per[(new_per["method"] == policy) & (new_per["seed"] == seed) & (new_per["budget"] == budget)]
                coverage, legacy, retained = int(d["coverage"].sum()), int(d["legacy_TP"].sum()), int(d["output_records"].sum())
                precision, recall = legacy / retained, legacy / total_gt
                metrics, class_metrics = coco_map[(policy, seed, budget)]
                main_rows.append({
                    "method": policy, "seed": seed, "budget": budget, "variant_id": variant,
                    "class_weight_on": class_on, "multi_threshold_on": multi_on, "source": "NEW_P1A",
                    "image_count": 2000, "group_count": 50, "total_GT": total_gt, "empty_GT_images": 3,
                    "coverage_total": coverage, "coverage_per_image": coverage / 2000, "coverage_recall": coverage / total_gt,
                    "quality_total": float(d["quality"].sum()), "quality_per_image": float(d["quality"].mean()),
                    "multi_iou_coverage_total": float(d["multi_iou_coverage"].sum()),
                    "multi_iou_coverage_per_image": float(d["multi_iou_coverage"].mean()),
                    "class50_weighted_coverage_total": float(d["class50_weighted_coverage"].sum()),
                    "class50_weighted_coverage_per_image": float(d["class50_weighted_coverage"].mean()),
                    "legacy_TP": legacy, "legacy_FP": retained - legacy, "legacy_FN": total_gt - legacy,
                    "precision": precision, "recall": recall, "F1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
                    "output_records": retained, "K_mean": float(d["K_i"].mean()), "K_median": float(d["K_i"].median()),
                    "K_min": int(d["K_i"].min()), "K_max": int(d["K_i"].max()),
                    "AP": metrics["AP"], "AP50": metrics["AP50"], "AP75": metrics["AP75"], "AR100": metrics["AR100"],
                })
                by_id = {int(x["category_id"]): x for x in class_metrics}
                for ci, cname in enumerate(ROAD8, 1):
                    a = class_acc[(policy, seed, budget, ci)]
                    pp = a["legacy"] / a["selected"] if a["selected"] else 0.0
                    rr = a["legacy"] / a["GT"] if a["GT"] else math.nan
                    cm = by_id[ci]
                    class_rows.append({
                        "method": policy, "seed": seed, "budget": budget, "variant_id": variant,
                        "class_weight_on": class_on, "multi_threshold_on": multi_on, "source": "NEW_P1A",
                        "category_id": ci, "class_name": cname, "GT": a["GT"], "images_with_GT": a["images_with_GT"],
                        "selected_records": a["selected"], "coverage": a["coverage"],
                        "coverage_recall": a["coverage"] / a["GT"] if a["GT"] else math.nan,
                        "legacy_TP": a["legacy"], "precision": pp, "recall": rr,
                        "F1": 2 * pp * rr / (pp + rr) if a["GT"] and pp + rr else 0.0,
                        "AP": cm["AP"], "AP50": cm["AP50"], "AP75": cm["AP75"], "AR100": cm["AR100"],
                    })

    new_main, new_class = pd.DataFrame(main_rows), pd.DataFrame(class_rows)
    if len(new_main) != 30 or len(new_class) != 240:
        raise RuntimeError("new main/class result row count mismatch")
    # Reused rows remain numerically untouched; only provenance/factor labels and
    # optional common-prefix summaries are attached.
    for df in (reused_main, reused_class):
        df["variant_id"] = df["method"].map(lambda x: variant_metadata(str(x))[0])
        df["class_weight_on"] = df["method"].map(lambda x: variant_metadata(str(x))[1])
        df["multi_threshold_on"] = df["method"].map(lambda x: variant_metadata(str(x))[2])
        df["source"] = "REUSED_P1"
    optional_main = per_image.groupby(["method", "seed", "budget"], as_index=False).agg(
        multi_iou_coverage_total=("multi_iou_coverage", "sum"), multi_iou_coverage_per_image=("multi_iou_coverage", "mean"),
        class50_weighted_coverage_total=("class50_weighted_coverage", "sum"), class50_weighted_coverage_per_image=("class50_weighted_coverage", "mean"),
    )
    reused_main = reused_main.merge(optional_main, on=["method", "seed", "budget"], how="left", validate="one_to_one")
    main_df = pd.concat([reused_main, new_main], ignore_index=True, sort=False)
    class_df = pd.concat([reused_class, new_class], ignore_index=True, sort=False)
    if len(main_df) != 65 or len(class_df) != 520:
        raise RuntimeError("combined main/class table row count mismatch")
    for key, d in class_df.groupby(["method", "seed", "budget"]):
        expected = int(main_df[(main_df["method"] == key[0]) & (main_df["seed"] == key[1]) & (main_df["budget"] == key[2])]["coverage_total"].iloc[0])
        if int(d["coverage"].sum()) != expected:
            raise RuntimeError(f"class coverage reconciliation failed: {key}")
    pq.write_table(pa.Table.from_pandas(main_df, preserve_index=False), root / "outputs" / "ablation_main_results.parquet", compression="zstd")
    pq.write_table(pa.Table.from_pandas(class_df, preserve_index=False), root / "outputs" / "ablation_class_results.parquet", compression="zstd")
    write_json(root / "outputs" / "evaluation_summary.json", {
        "status": "PASS", "dev_gt_opened_after_new_allocation_commit": True,
        "new_allocation_sha256": commit["new_allocations_sha256"], "dev_gt_sha256": actual_gt_sha,
        "images": 2000, "valid_road8_gt": total_gt, "empty_gt_images": 3,
        "shared_iou050_parity_mismatch": parity_mismatch,
        "new_main_rows": len(new_main), "combined_main_rows": len(main_df),
        "new_class_rows": len(new_class), "combined_class_rows": len(class_df),
        "elapsed_seconds": time.perf_counter() - t0,
    })
    append_log(root, f"STAGE evaluate_ablation COMPLETE elapsed_seconds={time.perf_counter()-t0:.6f}")


if __name__ == "__main__":
    main()
