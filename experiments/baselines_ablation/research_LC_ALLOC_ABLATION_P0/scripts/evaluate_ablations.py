from __future__ import annotations

import argparse
import importlib.util
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

from common import BUDGETS, SEEDS, VARIANTS, add_p1_scripts, append_log, sha256_file, verify_input_binding, write_json, write_parquet  # noqa: E402

ROAD8 = ("person", "bicycle", "car", "motorcycle", "bus", "train", "truck", "traffic light")
NEW_METHODS = VARIANTS + ("NEXT_SLOT_GREEDY",)


def import_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import module {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
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
                    raise RuntimeError("duplicate frozen DEV candidate_record_id")
                by_image[image_id].append(row)
                by_record[record] = row
    for image_id, rows in by_image.items():
        rows.sort(key=lambda x: int(x["road8_rank"]))
        if [int(x["road8_rank"]) for x in rows] != list(range(1, 101)):
            raise RuntimeError(f"DEV Top100 rank invariant failed {image_id}")
    if len(by_image) != 2000 or len(by_record) != 200000:
        raise RuntimeError("DEV candidate coverage invariant failed")
    return dict(by_image), by_record


def new_provenance(method: str) -> dict:
    if method == "NEXT_SLOT_GREEDY":
        return {"model_variant": "FULL", "solver": "NEXT_SLOT_GREEDY", "training_target": "PREFIX_MARGINAL", "mask": "NONE", "source": "NEW_ABLATION_P0_SOLVER_REPLAY", "source_method": "LEARN_QUALITY"}
    return {
        "model_variant": method, "solver": "EXACT_DP",
        "training_target": "MATCHABILITY" if method == "MATCHABILITY_TARGET" else "PREFIX_MARGINAL",
        "mask": method if method.startswith("NO_") else "NONE",
        "source": "NEW_ABLATION_P0_TRAIN", "source_method": "",
    }


def frozen_ledger_map(p1_root: Path) -> dict[str, str]:
    ledger = pd.read_csv(p1_root / "input_output_sha256.csv")
    out: dict[str, str] = {}
    for row in ledger.itertuples(index=False):
        raw = Path(str(row.path))
        path = raw if raw.is_absolute() else p1_root / raw
        out[str(path.resolve()).replace("\\", "/").lower()] = str(row.sha256).lower()
    return out


def verify_against_ledger(path: Path, ledger: dict[str, str], label: str) -> str:
    key = str(path.resolve()).replace("\\", "/").lower()
    expected = ledger.get(key)
    if expected is None:
        raise RuntimeError(f"{label} absent from frozen P1 ledger: {path}")
    actual = sha256_file(path)
    if actual.lower() != expected:
        raise RuntimeError(f"{label} hash mismatch: {path}")
    return actual


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", required=True)
    ap.add_argument("--p1-root", required=True)
    ap.add_argument("--release-root", required=True)
    ap.add_argument("--r0-evaluator", required=True)
    ap.add_argument("--coco-site-packages", required=True)
    args = ap.parse_args()
    root = Path(args.project_root).resolve()
    p1 = Path(args.p1_root).resolve()
    release = Path(args.release_root).resolve()
    t0 = time.perf_counter()
    append_log(root, "STAGE evaluate_ablations START")
    verify_input_binding(root)
    add_p1_scripts(p1)
    from p1_core import prefix_matching_counts  # noqa: E402

    commit = json.loads((root / "outputs" / "dev_selection_commit.json").read_text(encoding="utf-8"))
    allocation_path = Path(commit["allocation_path"])
    if sha256_file(allocation_path) != commit["allocation_sha256"]:
        raise RuntimeError("committed new DEV allocation hash mismatch")
    for asset in commit["prediction_assets"]:
        if sha256_file(Path(asset["path"])) != asset["sha256"]:
            raise RuntimeError(f"committed DEV prediction hash mismatch {asset['path']}")
    solver_gap_path = Path(commit["solver_gap_path"])
    if sha256_file(solver_gap_path) != commit["solver_gap_sha256"]:
        raise RuntimeError("committed solver-gap hash mismatch")
    solver_gap = pq.read_table(solver_gap_path).to_pandas()
    if len(solver_gap) != 750 or not solver_gap["dp_not_worse"].astype(bool).all():
        raise RuntimeError("committed solver-gap rows/invariant failed")

    p1_ledger = frozen_ledger_map(p1)
    p1_commit = json.loads((p1 / "outputs" / "dev_selection_commit.json").read_text(encoding="utf-8"))
    for path_key, sha_key in (("dev_predictions_path", "dev_predictions_sha256"), ("dev_selections_path", "dev_selections_sha256"), ("allocation_path", "allocation_sha256")):
        if sha256_file(Path(p1_commit[path_key])) != p1_commit[sha_key]:
            raise RuntimeError(f"frozen P1 asset hash mismatch {path_key}")
    for rel in ("per_image_results.parquet", "main_results.parquet", "class_results.parquet", "outputs/bootstrap_group_indices.npy"):
        verify_against_ledger(p1 / rel, p1_ledger, f"P1 {rel}")
    evaluator_path = Path(args.r0_evaluator).resolve()
    verify_against_ledger(evaluator_path, p1_ledger, "R0 evaluator")

    identity = json.loads((release / "CANDIDATE_ASSET_IDENTITY.json").read_text(encoding="utf-8"))
    candidates, by_record = load_candidates(release, identity)
    pred_identity = pq.read_table(p1 / "dev_predictions.parquet", columns=["image_id", "group_id"]).to_pandas().drop_duplicates()
    pred_identity["image_id"] = pred_identity["image_id"].astype(str)
    group_map = dict(zip(pred_identity["image_id"], pred_identity["group_id"].astype(int)))
    if len(group_map) != 2000:
        raise RuntimeError("frozen group map incomplete")
    group_sizes = pred_identity.groupby("group_id", sort=True)["image_id"].nunique()
    if list(group_sizes.index.astype(int)) != list(range(50)) or not (group_sizes.to_numpy(np.int64) == 40).all():
        raise RuntimeError("frozen group map is not exact IDs 0..49 with 40 images each")

    allocations = pq.read_table(allocation_path).to_pandas()
    allocations["image_id"] = allocations["image_id"].astype(str)
    if len(allocations) != 120000 or set(allocations["method"]) != set(NEW_METHODS):
        raise RuntimeError("new allocation cells incomplete")
    numeric_k = pd.to_numeric(allocations["K_i"], errors="coerce").to_numpy(np.float64)
    if not np.isfinite(numeric_k).all() or not np.equal(numeric_k, np.floor(numeric_k)).all():
        raise RuntimeError("new K_i values are not exact integers")
    allocations["K_i"] = numeric_k.astype(np.int64)
    alloc_lookup: dict[tuple[str, int, int, str], int] = {}
    for r in allocations.itertuples(index=False):
        image_id, k = str(r.image_id), int(r.K_i)
        expected = [str(x["candidate_record_id"]) for x in candidates[image_id][:k]]
        selected = list(r.selected_record_ids)
        if selected != expected or len(selected) != k or len(set(selected)) != k:
            raise RuntimeError("new selected IDs are not the canonical raw-score prefix")
        if int(r.group_id) != group_map[image_id] or str(r.solver) != new_provenance(str(r.method))["solver"]:
            raise RuntimeError("new allocation group/provenance mismatch")
        alloc_lookup[(str(r.method), int(r.seed), int(r.budget), image_id)] = k
    budget_check = allocations.groupby(["method", "seed", "budget", "group_id"], sort=False)["K_i"].sum().reset_index()
    if not np.array_equal(budget_check["K_i"].to_numpy(), budget_check["budget"].to_numpy(np.int64) * 40):
        raise RuntimeError("new exact group budget mismatch")
    append_log(root, f"ALLOCATIONS_VERIFIED_BEFORE_DEV_GT_OPEN sha={commit['allocation_sha256']}")

    # Reuse only FULL/M11 and S_ADAPT frozen scientific rows.
    p1_per = pq.read_table(p1 / "per_image_results.parquet").to_pandas()
    p1_main = pq.read_table(p1 / "main_results.parquet").to_pandas()
    p1_class = pq.read_table(p1 / "class_results.parquet").to_pandas()
    full_per = p1_per[p1_per["method"] == "LEARN_QUALITY"].copy()
    sadapt_per = p1_per[p1_per["method"] == "S_ADAPT"].copy()
    full_main = p1_main[p1_main["method"] == "LEARN_QUALITY"].copy()
    sadapt_main = p1_main[p1_main["method"] == "S_ADAPT"].copy()
    full_class = p1_class[p1_class["method"] == "LEARN_QUALITY"].copy()
    sadapt_class = p1_class[p1_class["method"] == "S_ADAPT"].copy()
    if (len(full_per), len(sadapt_per), len(full_main), len(sadapt_main), len(full_class), len(sadapt_class)) != (30000, 10000, 15, 5, 120, 40):
        raise RuntimeError("frozen FULL/S_ADAPT result row counts changed")
    reused_per = pd.concat([full_per, sadapt_per], ignore_index=True)
    reused_main = pd.concat([full_main, sadapt_main], ignore_index=True)
    reused_class = pd.concat([full_class, sadapt_class], ignore_index=True)
    reused_per["method"] = reused_per["method"].replace({"LEARN_QUALITY": "FULL"})
    reused_main["method"] = reused_main["method"].replace({"LEARN_QUALITY": "FULL"})
    reused_class["method"] = reused_class["method"].replace({"LEARN_QUALITY": "FULL"})
    for df in (reused_per, reused_main, reused_class):
        df["model_variant"] = np.where(df["method"] == "FULL", "FULL", "SCORE_ONLY")
        df["solver"] = np.where(df["method"] == "FULL", "EXACT_DP", "RAW_SCORE_EXACT_ALLOCATOR")
        df["training_target"] = np.where(df["method"] == "FULL", "PREFIX_MARGINAL", "NONE")
        df["mask"] = "NONE"
        df["source"] = "REUSED_P1"
        df["source_method"] = np.where(df["method"] == "FULL", "LEARN_QUALITY", "S_ADAPT")

    # GT is opened only after all new choices are hash-bound and identity-verified.
    r0 = import_module(evaluator_path, "lc_alloc_ablation_p0_r0_eval")
    shared = r0.load_shared(release)
    manifest = pq.read_table(release / identity["assets"]["DEV"]["split_manifest_path"]).to_pylist()
    ids = [str(x["image_id"]) for x in manifest]
    gt_path = release / "gt" / "DEV2K_ROAD8_GT.json"
    actual_gt_sha = sha256_file(gt_path)
    expected_gt_sha = verify_against_ledger(gt_path, p1_ledger, "DEV GT")
    if actual_gt_sha != expected_gt_sha:
        raise RuntimeError("DEV GT ledger identity mismatch")
    with gt_path.open("r", encoding="utf-8") as f:
        raw_gt = json.load(f)
    local_gt, coco_gt, remap = r0.prepare_gt(raw_gt, manifest, shared)
    del raw_gt
    total_gt = sum(len(local_gt[x]) for x in ids)
    if len(ids) != 2000 or total_gt != 29966 or sum(not local_gt[x] for x in ids) != 3:
        raise RuntimeError("frozen DEV GT count invariant failed")
    weights = np.asarray(json.loads((p1 / "models" / "class_weights.json").read_text(encoding="utf-8"))["weights"], dtype=np.float64)
    p1_alloc = pq.read_table(Path(p1_commit["allocation_path"])).to_pandas()
    p1_alloc = p1_alloc[p1_alloc["method"].isin(["LEARN_QUALITY", "S_ADAPT"])].copy()
    if len(p1_alloc) != 40000:
        raise RuntimeError("frozen P1 FULL/S_ADAPT allocation rows incomplete")
    reused_k = {
        ("FULL" if str(r.method) == "LEARN_QUALITY" else "S_ADAPT", int(r.seed), int(r.budget), str(r.image_id)): int(r.K_i)
        for r in p1_alloc.itertuples(index=False)
    }

    cache: dict[str, dict] = {}
    parity_mismatch = 0
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
        needed = {alloc_lookup[(m, s, b, image_id)] for m in NEW_METHODS for s in SEEDS for b in BUDGETS}
        needed |= {reused_k[("FULL", s, b, image_id)] for s in SEEDS for b in BUDGETS}
        needed |= {reused_k[("S_ADAPT", -1, b, image_id)] for b in BUDGETS}
        standard = {}
        for k in sorted(needed):
            result = shared.evaluate_selected(list(range(1, k + 1)), graph)
            coverage_class = np.zeros(8, dtype=np.int16)
            legacy_class = np.zeros(8, dtype=np.int16)
            for pair in result.coverage_matching.pairs:
                coverage_class[int(pair.category_id) - 1] += 1
            for pair in result.legacy_matching.pairs:
                legacy_class[int(pair.category_id) - 1] += 1
            parity_mismatch += int(int(totals[k, 0]) != result.coverage_matching.cardinality)
            standard[k] = {"coverage": int(result.coverage_matching.cardinality), "legacy": int(result.legacy_matching.cardinality), "coverage_class": coverage_class, "legacy_class": legacy_class}
        cache[image_id] = {
            "gt": len(local_gt[image_id]), "gt_class": np.bincount(gt_classes, minlength=9)[1:9],
            "standard": standard,
            "quality": np.sum(weights[None, :, None] * by_class_t.astype(np.float64), axis=1).mean(axis=1),
            "pred_class_prefix": pred_class_prefix,
        }
        if (image_ix + 1) % 250 == 0:
            append_log(root, f"PREFIX_EVAL_PROGRESS images={image_ix+1}")
    if parity_mismatch:
        raise RuntimeError(f"shared evaluator/parameterized IoU.50 mismatch={parity_mismatch}")

    # Frozen evaluator regression, not a new scientific condition.
    anchor_cov = sum(cache[x]["standard"][reused_k[("S_ADAPT", -1, 10, x)]]["coverage"] for x in ids)
    anchor_legacy = sum(cache[x]["standard"][reused_k[("S_ADAPT", -1, 10, x)]]["legacy"] for x in ids)
    anchor_selection = {("S_ADAPT", 10, x): list(range(1, reused_k[("S_ADAPT", -1, 10, x)] + 1)) for x in ids}
    anchor_coco, _ = r0.evaluate_coco(coco_gt, candidates, anchor_selection, "S_ADAPT", 10, remap, args.coco_site_packages)
    if anchor_cov != 13818 or anchor_legacy != 13811 or not np.isclose(anchor_coco["AP"], 0.1536753879858031, atol=1e-6, rtol=1e-5) or not np.isclose(anchor_coco["AR100"], 0.2034422345381616, atol=1e-6, rtol=1e-5):
        raise RuntimeError("S_ADAPT@10 evaluator regression failed")
    append_log(root, "S_ADAPT_AT_10_EVALUATOR_REGRESSION PASS")

    per_rows: list[dict] = []
    class_acc = defaultdict(lambda: {"GT": 0, "images_with_GT": 0, "selected": 0, "coverage": 0, "legacy": 0})
    for method in NEW_METHODS:
        prov = new_provenance(method)
        for seed in SEEDS:
            for budget in BUDGETS:
                for image_id in ids:
                    k = alloc_lookup[(method, seed, budget, image_id)]
                    z, s = cache[image_id], cache[image_id]["standard"][k]
                    precision = s["legacy"] / k
                    recall = s["legacy"] / z["gt"] if z["gt"] else 0.0
                    per_rows.append({
                        "image_id": image_id, "group_id": group_map[image_id], "method": method, "seed": seed,
                        **prov, "budget": budget, "K_i": k, "GT": z["gt"], "coverage": s["coverage"],
                        "coverage_recall": s["coverage"] / z["gt"] if z["gt"] else 0.0,
                        "quality": float(z["quality"][k]), "legacy_TP": s["legacy"], "precision": precision,
                        "recall": recall, "F1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
                        "output_records": k, "empty_GT_image": not z["gt"],
                    })
                    for ci in range(8):
                        a = class_acc[(method, seed, budget, ci + 1)]
                        a["GT"] += int(z["gt_class"][ci])
                        a["images_with_GT"] += int(z["gt_class"][ci] > 0)
                        a["selected"] += int(z["pred_class_prefix"][k, ci])
                        a["coverage"] += int(s["coverage_class"][ci])
                        a["legacy"] += int(s["legacy_class"][ci])
    new_per = pd.DataFrame(per_rows)
    if len(new_per) != 120000:
        raise RuntimeError("new per-image result row count mismatch")
    per_image = pd.concat([reused_per, new_per], ignore_index=True, sort=False)
    if len(per_image) != 160000:
        raise RuntimeError("combined per-image row count mismatch")
    write_parquet(per_image, root / "outputs" / "ablation_per_image_results.parquet")

    group_cols = ["method", "seed", "budget", "group_id", "model_variant", "solver", "training_target", "mask", "source", "source_method"]
    group_results = per_image.groupby(group_cols, as_index=False, dropna=False).agg(
        image_count=("image_id", "size"), GT=("GT", "sum"), coverage=("coverage", "sum"),
        quality=("quality", "sum"), legacy_TP=("legacy_TP", "sum"), output_records=("output_records", "sum"),
    )
    group_results["coverage_per_image"] = group_results["coverage"] / group_results["image_count"]
    group_results["quality_per_image"] = group_results["quality"] / group_results["image_count"]
    if len(group_results) != 4000 or not np.array_equal(group_results["output_records"].to_numpy(np.int64), group_results["budget"].to_numpy(np.int64) * 40):
        raise RuntimeError("combined group result invariant failed")
    write_parquet(group_results, root / "outputs" / "ablation_group_results.parquet")

    def selection_for(method: str, seed: int, budget: int) -> dict:
        tag = f"{method}__SEED_{seed}"
        return {(tag, budget, image_id): list(range(1, alloc_lookup[(method, seed, budget, image_id)] + 1)) for image_id in ids}

    coco_map: dict[tuple[str, int, int], tuple[dict, list[dict]]] = {}
    for method in NEW_METHODS:
        for seed in SEEDS:
            tag = f"{method}__SEED_{seed}"
            for budget in BUDGETS:
                coco_map[(method, seed, budget)] = r0.evaluate_coco(coco_gt, candidates, selection_for(method, seed, budget), tag, budget, remap, args.coco_site_packages)
            append_log(root, f"COCO_EVAL_COMPLETE method={method} seed={seed}")

    main_rows: list[dict] = []
    class_rows: list[dict] = []
    for method in NEW_METHODS:
        prov = new_provenance(method)
        for seed in SEEDS:
            for budget in BUDGETS:
                d = new_per[(new_per["method"] == method) & (new_per["seed"] == seed) & (new_per["budget"] == budget)]
                coverage, legacy, retained = int(d["coverage"].sum()), int(d["legacy_TP"].sum()), int(d["output_records"].sum())
                precision, recall = legacy / retained, legacy / total_gt
                metrics, class_metrics = coco_map[(method, seed, budget)]
                main_rows.append({
                    "method": method, "seed": seed, "budget": budget, **prov,
                    "image_count": 2000, "group_count": 50, "total_GT": total_gt, "empty_GT_images": 3,
                    "coverage_total": coverage, "coverage_per_image": coverage / 2000, "coverage_recall": coverage / total_gt,
                    "quality_total": float(d["quality"].sum()), "quality_per_image": float(d["quality"].mean()),
                    "legacy_TP": legacy, "legacy_FP": retained - legacy, "legacy_FN": total_gt - legacy,
                    "precision": precision, "recall": recall, "F1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
                    "output_records": retained, "K_mean": float(d["K_i"].mean()), "K_median": float(d["K_i"].median()),
                    "K_min": int(d["K_i"].min()), "K_max": int(d["K_i"].max()),
                    "AP": metrics["AP"], "AP50": metrics["AP50"], "AP75": metrics["AP75"], "AR100": metrics["AR100"],
                })
                by_id = {int(x["category_id"]): x for x in class_metrics}
                for ci, cname in enumerate(ROAD8, 1):
                    a = class_acc[(method, seed, budget, ci)]
                    pp = a["legacy"] / a["selected"] if a["selected"] else 0.0
                    rr = a["legacy"] / a["GT"] if a["GT"] else math.nan
                    cm = by_id[ci]
                    class_rows.append({
                        "method": method, "seed": seed, "budget": budget, **prov,
                        "category_id": ci, "class_name": cname, "GT": a["GT"], "images_with_GT": a["images_with_GT"],
                        "selected_records": a["selected"], "coverage": a["coverage"],
                        "coverage_recall": a["coverage"] / a["GT"] if a["GT"] else math.nan,
                        "legacy_TP": a["legacy"], "precision": pp, "recall": rr,
                        "F1": 2 * pp * rr / (pp + rr) if a["GT"] and pp + rr else 0.0,
                        "AP": cm["AP"], "AP50": cm["AP50"], "AP75": cm["AP75"], "AR100": cm["AR100"],
                    })
    new_main, new_class = pd.DataFrame(main_rows), pd.DataFrame(class_rows)
    if len(new_main) != 60 or len(new_class) != 480:
        raise RuntimeError("new main/class row count mismatch")
    main_df = pd.concat([reused_main, new_main], ignore_index=True, sort=False)
    class_df = pd.concat([reused_class, new_class], ignore_index=True, sort=False)
    if len(main_df) != 80 or len(class_df) != 640:
        raise RuntimeError("combined main/class row count mismatch")
    for key, d in class_df.groupby(["method", "seed", "budget"]):
        expected = int(main_df[(main_df["method"] == key[0]) & (main_df["seed"] == key[1]) & (main_df["budget"] == key[2])]["coverage_total"].iloc[0])
        if int(d["coverage"].sum()) != expected:
            raise RuntimeError(f"class coverage reconciliation failed {key}")
    if not np.array_equal(main_df["output_records"].to_numpy(np.int64), main_df["budget"].to_numpy(np.int64) * 2000):
        raise RuntimeError("main result exact total record budget mismatch")
    write_parquet(main_df, root / "outputs" / "ablation_main_results.parquet")
    write_parquet(class_df, root / "outputs" / "ablation_class_results.parquet")
    write_json(root / "outputs" / "evaluation_summary.json", {
        "status": "PASS", "dev_gt_opened_after_new_selection_commit": True,
        "new_allocation_sha256": commit["allocation_sha256"], "dev_gt_sha256": actual_gt_sha,
        "images": 2000, "valid_road8_gt": total_gt, "empty_gt_images": 3,
        "shared_iou050_parity_mismatch": parity_mismatch,
        "s_adapt_10_regression": {"coverage": anchor_cov, "legacy_TP": anchor_legacy, **anchor_coco},
        "new_main_rows": len(new_main), "combined_main_rows": len(main_df),
        "new_class_rows": len(new_class), "combined_class_rows": len(class_df),
        "elapsed_seconds": time.perf_counter() - t0,
    })
    append_log(root, f"STAGE evaluate_ablations COMPLETE elapsed_seconds={time.perf_counter()-t0:.6f}")


if __name__ == "__main__":
    main()
