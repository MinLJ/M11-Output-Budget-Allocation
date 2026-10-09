"""Independent DEV evaluation of the committed S_CLASS_ADAPT allocation.

The allocation file and all prefix identities are verified before this process
opens the frozen DEV ground truth.  Historical M10/M11/S_ADAPT metrics are read
from P1A and are not recomputed.
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

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.dont_write_bytecode = True
BUDGETS = (10, 15, 20, 30, 40)
ROAD8 = ("person", "bicycle", "car", "motorcycle", "bus", "train", "truck", "traffic light")
METHOD = "S_CLASS_ADAPT"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def append_log(root: Path, value: str) -> None:
    with (root / "runtime.log").open("a", encoding="utf-8") as f:
        f.write(value.rstrip() + "\n")


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def import_file(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_candidates(release_root: Path, identity: dict) -> dict[str, list[dict]]:
    by_image: dict[str, list[dict]] = defaultdict(list)
    seen: set[str] = set()
    for rel in identity["assets"]["DEV"]["candidate_shards"]:
        for batch in pq.ParquetFile(release_root / rel).iter_batches(batch_size=32768):
            for row in batch.to_pylist():
                if int(row["road8_rank"]) > 100:
                    continue
                rid = str(row["candidate_record_id"])
                if rid in seen:
                    raise RuntimeError("duplicate candidate_record_id")
                seen.add(rid)
                by_image[str(row["image_id"])].append(row)
    for image_id, rows in by_image.items():
        rows.sort(key=lambda x: int(x["road8_rank"]))
        if [int(x["road8_rank"]) for x in rows] != list(range(1, 101)):
            raise RuntimeError(f"Top100 rank invariant failed: {image_id}")
    if len(by_image) != 2000 or len(seen) != 200_000:
        raise RuntimeError("DEV candidate count invariant failed")
    return dict(by_image)


def prepare_gt(raw: dict, shared) -> tuple[dict[str, list], dict, dict[str, int]]:
    ids = [str(x["source_image_id"]) for x in raw["images"]]
    remap = {image_id: ix + 1 for ix, image_id in enumerate(ids)}
    local: dict[str, list] = {x: [] for x in ids}
    source_order = defaultdict(int)
    coco_images = []
    for row in raw["images"]:
        image_id = str(row["source_image_id"])
        coco_images.append({"id": remap[image_id], "file_name": row["file_name"], "width": row["width"], "height": row["height"]})
    coco_annotations = []
    for ann in raw["annotations"]:
        image_id = str(ann["image_id"])
        x, y, w, h = map(float, ann["bbox"])
        order = int(source_order[image_id]); source_order[image_id] += 1
        local[image_id].append(shared.GroundTruth(
            image_id=image_id, gt_id=str(ann["id"]), category_id=int(ann["category_id"]),
            bbox_xyxy=(x, y, x + w, y + h), source_order=order,
        ))
        coco_annotations.append({
            "id": int(ann["id"]), "image_id": remap[image_id], "category_id": int(ann["category_id"]),
            "bbox": [x, y, w, h], "area": float(ann["area"]), "iscrowd": int(ann.get("iscrowd", 0)),
        })
    dataset = {"info": raw.get("info", {}), "licenses": raw.get("licenses", []),
               "images": coco_images, "categories": raw["categories"], "annotations": coco_annotations}
    return local, dataset, remap


def coco_eval(dataset: dict, candidates: dict[str, list[dict]], kvals: dict[str, int], remap: dict[str, int], coco_site: Path) -> tuple[dict, list[dict]]:
    sys.path.insert(0, str(coco_site))
    from pycocotools.coco import COCO  # type: ignore
    from pycocotools.cocoeval import COCOeval  # type: ignore

    gt = COCO(); gt.dataset = dataset; gt.createIndex()
    det = []
    for image_id, k in kvals.items():
        for row in candidates[image_id][:int(k)]:
            x1, y1, x2, y2 = map(float, (row["bbox_x1"], row["bbox_y1"], row["bbox_x2"], row["bbox_y2"]))
            det.append({"image_id": remap[image_id], "category_id": int(row["predicted_road8_class_id"]),
                        "bbox": [x1, y1, x2 - x1, y2 - y1], "score": float(row["score"])})
    with contextlib.redirect_stdout(io.StringIO()):
        dt = gt.loadRes(det)

    def run(cat_ids: list[int] | None) -> np.ndarray:
        ev = COCOeval(gt, dt, "bbox")
        ev.params.imgIds = sorted(remap.values())
        ev.params.catIds = list(range(1, 9)) if cat_ids is None else cat_ids
        # Match the frozen COCO contract exactly.  ``arange`` produces a
        # 0.7500000000000002 cell on this NumPy build, causing COCOeval's exact
        # AP75 lookup to return -1 despite otherwise identical evaluation.
        ev.params.iouThrs = np.linspace(0.50, 0.95, 10)
        ev.params.maxDets = [1, 10, 100]
        with contextlib.redirect_stdout(io.StringIO()):
            ev.evaluate(); ev.accumulate(); ev.summarize()
        return np.asarray(ev.stats, np.float64)

    all_stats = run(None)
    metrics = {"AP": float(all_stats[0]), "AP50": float(all_stats[1]), "AP75": float(all_stats[2]), "AR100": float(all_stats[8])}
    class_metrics = []
    for ci, name in enumerate(ROAD8, 1):
        s = run([ci])
        class_metrics.append({"category_id": ci, "class_name": name, "AP": float(s[0]), "AP50": float(s[1]), "AP75": float(s[2]), "AR100": float(s[8])})
    return metrics, class_metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", required=True)
    ap.add_argument("--p1-root", required=True)
    ap.add_argument("--p1a-root", required=True)
    ap.add_argument("--coco-site-packages", required=True)
    args = ap.parse_args()
    root, p1, p1a = map(lambda x: Path(x).resolve(), (args.project_root, args.p1_root, args.p1a_root))
    t0 = time.perf_counter()
    append_log(root, "STAGE evaluate_weighted_score START")
    append_log(root, "IMPLEMENTATION_NOTE initial evaluation attempt stopped before aggregate scientific outputs because np.arange did not expose an exact 0.75 COCO threshold; the evaluator was corrected to the frozen 10-point linspace and rerun in full.")
    config = json.loads((root / "run_config.json").read_text(encoding="utf-8"))
    release_root = Path(config["release_root"])
    identity = json.loads((release_root / "CANDIDATE_ASSET_IDENTITY.json").read_text(encoding="utf-8"))
    candidates = load_candidates(release_root, identity)

    commit = json.loads((root / "outputs" / "weighted_score_allocation_commit.json").read_text(encoding="utf-8"))
    alloc_path = Path(commit["path"])
    if sha256_file(alloc_path) != commit["sha256"]:
        raise RuntimeError("committed S_CLASS allocation hash mismatch")
    alloc = pq.read_table(alloc_path).to_pandas(); alloc["image_id"] = alloc["image_id"].astype(str)
    if len(alloc) != 10_000 or set(alloc["method"]) != {METHOD}:
        raise RuntimeError("S_CLASS allocation table incomplete")
    group_map = dict(zip(alloc["image_id"], alloc["group_id"].astype(int)))
    lookup: dict[tuple[int, str], int] = {}
    for r in alloc.itertuples(index=False):
        image_id, k = str(r.image_id), int(r.K_i)
        expected = [str(x["candidate_record_id"]) for x in candidates[image_id][:k]]
        if list(r.selected_record_ids) != expected or len(expected) != k:
            raise RuntimeError("committed selection is not exact frozen prefix")
        lookup[(int(r.budget), image_id)] = k
    check = alloc.groupby(["budget", "group_id"], sort=True)["K_i"].sum().reset_index()
    if not np.array_equal(check["K_i"].to_numpy(), check["budget"].to_numpy() * 40):
        raise RuntimeError("committed S_CLASS budget mismatch")
    append_log(root, f"ALLOCATIONS_VERIFIED_BEFORE_DEV_GT_OPEN sha={commit['sha256']}")

    # The shared release evaluator and P1 prefix-matching implementation are the only evaluator sources used.
    shared = import_file(release_root / "evaluator" / "shared_evaluator.py", "lc_p1b_shared_eval")
    sys.path.insert(0, str(p1 / "scripts"))
    from p1_core import prefix_matching_counts, legacy_prefix_counts  # type: ignore

    gt_path = release_root / "gt" / "DEV2K_ROAD8_GT.json"
    actual_gt_sha = sha256_file(gt_path)
    raw_gt = json.loads(gt_path.read_text(encoding="utf-8"))
    local_gt, coco_dataset, remap = prepare_gt(raw_gt, shared)
    ids = [str(x["source_image_id"]) for x in raw_gt["images"]]
    del raw_gt
    total_gt = sum(len(local_gt[x]) for x in ids)
    if len(ids) != 2000 or total_gt != 29966 or sum(not local_gt[x] for x in ids) != 3:
        raise RuntimeError("frozen DEV GT invariant failed")
    weights = np.asarray(json.loads((p1 / "models" / "class_weights.json").read_text(encoding="utf-8"))["weights"], np.float64)

    cache: dict[str, dict] = {}
    parity_bad = 0
    for ix, image_id in enumerate(ids):
        rows = candidates[image_id]
        classes = np.asarray([int(x["predicted_road8_class_id"]) for x in rows], np.int16)
        boxes = np.asarray([[x["bbox_x1"], x["bbox_y1"], x["bbox_x2"], x["bbox_y2"]] for x in rows], np.float64)
        gt_cls = np.asarray([int(x.category_id) for x in local_gt[image_id]], np.int16)
        gt_box = np.asarray([x.bbox_xyxy for x in local_gt[image_id]], np.float64).reshape(-1, 4)
        totals, by_class = prefix_matching_counts(classes, boxes, gt_cls, gt_box, max_k=50)
        legacy, legacy_class = legacy_prefix_counts(classes, boxes, gt_cls, gt_box, max_k=50)
        pred_class = np.zeros((51, 8), np.int16)
        for k in range(1, 51):
            pred_class[k] = pred_class[k - 1]; pred_class[k, classes[k - 1] - 1] += 1
        graph = shared.build_image_graph([shared.Candidate.from_mapping(x) for x in rows], local_gt[image_id])
        for budget in BUDGETS:
            k = lookup[(budget, image_id)]
            result = shared.evaluate_selected(list(range(1, k + 1)), graph)
            parity_bad += int(result.coverage_matching.cardinality != int(totals[k, 0]))
            parity_bad += int(result.legacy_matching.cardinality != int(legacy[k]))
        cache[image_id] = {
            "gt": len(local_gt[image_id]), "gt_class": np.bincount(gt_cls, minlength=9)[1:9],
            "coverage": totals[:, 0], "coverage_class": by_class[:, :, 0],
            "legacy": legacy, "legacy_class": legacy_class,
            "quality": np.sum(weights[None, :, None] * by_class.astype(np.float64), axis=1).mean(axis=1),
            "pred_class": pred_class,
        }
        if (ix + 1) % 250 == 0:
            append_log(root, f"PREFIX_EVAL_PROGRESS images={ix+1}")
    if parity_bad:
        raise RuntimeError(f"shared/prefix evaluator mismatch={parity_bad}")

    per_rows = []
    class_acc = defaultdict(lambda: {"GT": 0, "images_with_GT": 0, "selected": 0, "coverage": 0, "legacy": 0})
    for budget in BUDGETS:
        for image_id in ids:
            k = lookup[(budget, image_id)]; z = cache[image_id]
            cov, leg, gt = int(z["coverage"][k]), int(z["legacy"][k]), int(z["gt"])
            precision, recall = leg / k, leg / gt if gt else 0.0
            per_rows.append({
                "image_id": image_id, "group_id": group_map[image_id], "method": METHOD, "seed": -1,
                "budget": budget, "K_i": k, "GT": gt, "coverage": cov,
                "coverage_recall": cov / gt if gt else 0.0, "quality": float(z["quality"][k]),
                "legacy_TP": leg, "precision": precision, "recall": recall,
                "F1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
                "output_records": k, "empty_GT_image": not gt, "variant_id": "S_CLASS",
                "class_weight_on": True, "multi_threshold_on": False, "source": "NEW_P1B",
            })
            for ci in range(8):
                a = class_acc[(budget, ci + 1)]
                a["GT"] += int(z["gt_class"][ci]); a["images_with_GT"] += int(z["gt_class"][ci] > 0)
                a["selected"] += int(z["pred_class"][k, ci]); a["coverage"] += int(z["coverage_class"][k, ci]); a["legacy"] += int(z["legacy_class"][k, ci])
    new_per = pd.DataFrame(per_rows)
    if len(new_per) != 10_000 or int(new_per["output_records"].sum()) != 230_000:
        raise RuntimeError("new per-image result size mismatch")
    pq.write_table(pa.Table.from_pandas(new_per, preserve_index=False), root / "outputs" / "weighted_score_per_image_results.parquet", compression="zstd")
    new_group = new_per.groupby(["method", "seed", "budget", "group_id", "variant_id", "source"], as_index=False).agg(
        image_count=("image_id", "size"), GT=("GT", "sum"), coverage=("coverage", "sum"), quality=("quality", "sum"),
        legacy_TP=("legacy_TP", "sum"), output_records=("output_records", "sum"),
    )
    new_group["coverage_per_image"] = new_group["coverage"] / new_group["image_count"]
    new_group["quality_per_image"] = new_group["quality"] / new_group["image_count"]

    coco = {}
    class_coco = {}
    # One old S_ADAPT@10 regression validates the independently implemented COCO path.
    p1a_per = pq.read_table(p1a / "ablation_per_image_results.parquet").to_pandas()
    old_s10 = p1a_per[(p1a_per["method"] == "S_ADAPT") & (p1a_per["seed"] == -1) & (p1a_per["budget"] == 10)]
    old_k10 = dict(zip(old_s10["image_id"].astype(str), old_s10["K_i"].astype(int)))
    old_metric, _ = coco_eval(coco_dataset, candidates, old_k10, remap, Path(args.coco_site_packages))
    old_main = pq.read_table(p1a / "outputs" / "ablation_main_results.parquet").to_pandas()
    old_row = old_main[(old_main["method"] == "S_ADAPT") & (old_main["seed"] == -1) & (old_main["budget"] == 10)].iloc[0]
    coco_regression = {m: float(old_metric[m] - old_row[m]) for m in ("AP", "AP50", "AP75", "AR100")}
    if max(map(abs, coco_regression.values())) > 1e-12:
        raise RuntimeError(f"S_ADAPT@10 COCO regression failed: {coco_regression}")
    append_log(root, "S_ADAPT_AT_10_COCO_REGRESSION PASS")
    for budget in BUDGETS:
        kvals = {image_id: lookup[(budget, image_id)] for image_id in ids}
        coco[budget], class_coco[budget] = coco_eval(coco_dataset, candidates, kvals, remap, Path(args.coco_site_packages))
        append_log(root, f"COCO_EVAL_COMPLETE method={METHOD} budget={budget}")

    main_rows, class_rows = [], []
    for budget in BUDGETS:
        d = new_per[new_per["budget"] == budget]
        retained, coverage, legacy = int(d["output_records"].sum()), int(d["coverage"].sum()), int(d["legacy_TP"].sum())
        precision, recall = legacy / retained, legacy / total_gt
        cm = {int(x["category_id"]): x for x in class_coco[budget]}
        main_rows.append({
            "method": METHOD, "seed": -1, "budget": budget, "image_count": 2000, "group_count": 50,
            "total_GT": total_gt, "empty_GT_images": 3, "coverage_total": coverage,
            "coverage_per_image": coverage / 2000, "coverage_recall": coverage / total_gt,
            "quality_total": float(d["quality"].sum()), "quality_per_image": float(d["quality"].mean()),
            "legacy_TP": legacy, "legacy_FP": retained - legacy, "legacy_FN": total_gt - legacy,
            "precision": precision, "recall": recall, "F1": 2 * precision * recall / (precision + recall),
            "output_records": retained, "K_mean": float(d["K_i"].mean()), "K_median": float(d["K_i"].median()),
            "K_min": int(d["K_i"].min()), "K_max": int(d["K_i"].max()), **coco[budget],
            "variant_id": "S_CLASS", "class_weight_on": True, "multi_threshold_on": False, "source": "NEW_P1B",
        })
        for ci, cname in enumerate(ROAD8, 1):
            a = class_acc[(budget, ci)]; pp = a["legacy"] / a["selected"] if a["selected"] else 0.0
            rr = a["legacy"] / a["GT"] if a["GT"] else math.nan
            class_rows.append({
                "method": METHOD, "seed": -1, "budget": budget, "category_id": ci, "class_name": cname,
                "GT": a["GT"], "images_with_GT": a["images_with_GT"], "selected_records": a["selected"],
                "coverage": a["coverage"], "coverage_recall": a["coverage"] / a["GT"] if a["GT"] else math.nan,
                "legacy_TP": a["legacy"], "precision": pp, "recall": rr,
                "F1": 2 * pp * rr / (pp + rr) if a["GT"] and pp + rr else 0.0,
                "AP": cm[ci]["AP"], "AP50": cm[ci]["AP50"], "AP75": cm[ci]["AP75"], "AR100": cm[ci]["AR100"],
                "variant_id": "S_CLASS", "class_weight_on": True, "multi_threshold_on": False, "source": "NEW_P1B",
            })
    new_main, new_class = pd.DataFrame(main_rows), pd.DataFrame(class_rows)
    if len(new_main) != 5 or len(new_class) != 40:
        raise RuntimeError("new aggregate result size mismatch")

    old_main = old_main[old_main["method"].isin(["S_ADAPT", "LEARN_CLASS50", "LEARN_QUALITY"])].copy()
    old_class = pq.read_table(p1a / "outputs" / "ablation_class_results.parquet").to_pandas()
    old_class = old_class[old_class["method"].isin(["S_ADAPT", "LEARN_CLASS50", "LEARN_QUALITY"])].copy()
    old_group = pq.read_table(p1a / "outputs" / "ablation_group_results.parquet").to_pandas()
    old_group = old_group[old_group["method"].isin(["S_ADAPT", "LEARN_CLASS50", "LEARN_QUALITY"])].copy()
    main_all = pd.concat([old_main, new_main], ignore_index=True, sort=False)
    class_all = pd.concat([old_class, new_class], ignore_index=True, sort=False)
    group_all = pd.concat([old_group, new_group], ignore_index=True, sort=False)
    if len(main_all) != 40 or len(class_all) != 320 or len(group_all) != 2000:
        raise RuntimeError("combined P1B result sizes mismatch")
    for key, d in class_all.groupby(["method", "seed", "budget"]):
        expected = int(main_all[(main_all["method"] == key[0]) & (main_all["seed"] == key[1]) & (main_all["budget"] == key[2])]["coverage_total"].iloc[0])
        if int(d["coverage"].sum()) != expected:
            raise RuntimeError(f"class coverage reconciliation failed: {key}")
    for name, df in (("baseline_main", main_all), ("baseline_class", class_all), ("baseline_group", group_all)):
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False), root / "outputs" / f"{name}.parquet", compression="zstd")
    write_json(root / "outputs" / "evaluation_summary.json", {
        "status": "PASS", "dev_gt_opened_after_allocation_commit": True,
        "allocation_sha256": commit["sha256"], "dev_gt_sha256": actual_gt_sha,
        "images": 2000, "valid_road8_gt": total_gt, "empty_gt_images": 3,
        "shared_evaluator_parity_mismatch": parity_bad, "s_adapt_at_10_coco_regression_delta": coco_regression,
        "new_main_rows": 5, "combined_main_rows": 40, "new_class_rows": 40, "combined_class_rows": 320,
        "elapsed_seconds": time.perf_counter() - t0,
    })
    append_log(root, f"STAGE evaluate_weighted_score COMPLETE elapsed_seconds={time.perf_counter()-t0:.6f}")


if __name__ == "__main__":
    main()
