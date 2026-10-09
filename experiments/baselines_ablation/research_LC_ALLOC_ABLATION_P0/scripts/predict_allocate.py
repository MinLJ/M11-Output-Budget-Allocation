from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from common import (BUDGETS, SEEDS, THRESHOLDS, VARIANTS, add_p1_scripts, append_log,
                    load_model, predict_probabilities, sha256_file, verify_input_binding, write_json, write_parquet)  # noqa: E402


def prediction_columns(seed: int) -> list[str]:
    return [f"learn_p_seed_{seed}_iou_{v:.2f}".replace(".", "_") for v in THRESHOLDS]


def new_prediction_columns() -> list[str]:
    return [f"p_iou_{v:.2f}".replace(".", "_") for v in THRESHOLDS]


def next_slot_greedy(margins: np.ndarray, budgets: tuple[int, ...]) -> dict[int, np.ndarray]:
    values = np.asarray(margins, dtype=np.float64)
    if values.shape != (40, 45) or not np.isfinite(values).all():
        raise ValueError("greedy expects finite [40,45] rank6..50 marginals")
    targets = {40 * int(b): int(b) for b in budgets}
    k = np.full(40, 5, dtype=np.int64)
    out: dict[int, np.ndarray] = {}
    used = int(k.sum())
    if used in targets:
        out[targets[used]] = k.copy()
    max_used = max(targets)
    while used < max_used:
        next_value = np.full(40, -np.inf, dtype=np.float64)
        feasible = np.flatnonzero(k < 50)
        next_value[feasible] = values[feasible, k[feasible] - 5]
        best = np.max(next_value)
        tied = np.flatnonzero(next_value == best)
        if tied.size == 0:
            raise RuntimeError("greedy has no feasible next slot")
        k[int(tied[0])] += 1
        used += 1
        if used in targets:
            out[targets[used]] = k.copy()
    if set(out) != set(budgets):
        raise RuntimeError("greedy did not produce every requested budget")
    return out


def objective_for_k(choice: np.ndarray, k: np.ndarray) -> float:
    total = np.float64(0.0)
    for i, selected in enumerate(np.asarray(k, dtype=np.int64)):
        total = np.float64(total + choice[i, int(selected) - 5])
    return float(total)


def load_dev_candidates(release: Path) -> pd.DataFrame:
    identity = json.loads((release / "CANDIDATE_ASSET_IDENTITY.json").read_text(encoding="utf-8"))
    parts = []
    for rel in identity["assets"]["DEV"]["candidate_shards"]:
        d = pq.read_table(release / rel).to_pandas()
        parts.append(d[d["road8_rank"] <= 100].copy())
    candidates = pd.concat(parts, ignore_index=True)
    candidates["image_id"] = candidates["image_id"].astype(str)
    candidates = candidates.sort_values(["image_id", "road8_rank"], kind="stable").reset_index(drop=True)
    if len(candidates) != 200000 or candidates["candidate_record_id"].duplicated().any():
        raise RuntimeError("DEV canonical Top100 candidate invariant failed")
    sizes = candidates.groupby("image_id", sort=False).size()
    if len(sizes) != 2000 or not (sizes == 100).all():
        raise RuntimeError("DEV candidate image coverage invariant failed")
    return candidates


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", required=True)
    ap.add_argument("--p1-root", required=True)
    ap.add_argument("--release-root", required=True)
    args = ap.parse_args()
    root = Path(args.project_root).resolve()
    p1 = Path(args.p1_root).resolve()
    release = Path(args.release_root).resolve()
    t0 = time.perf_counter()
    append_log(root, "STAGE predict_allocate START (DEV GT not opened by this entrypoint)")
    verify_input_binding(root)
    add_p1_scripts(p1)
    from dp_solver import group_choice_values_from_marginals, solve_group_allocations  # noqa: E402

    masks = json.loads((root / "feature_masks.json").read_text(encoding="utf-8"))
    class_weights = np.asarray(json.loads((p1 / "models" / "class_weights.json").read_text(encoding="utf-8"))["weights"], dtype=np.float64)
    meta = pq.read_table(p1 / "dev_predictions.parquet").to_pandas()
    meta["image_id"] = meta["image_id"].astype(str)
    meta["_row_index"] = np.arange(len(meta), dtype=np.int64)
    if len(meta) != 90000 or len(meta.groupby("image_id", sort=False)) != 2000:
        raise RuntimeError("frozen DEV prediction metadata invariant failed")
    if not (meta.groupby("image_id", sort=False).size() == 45).all():
        raise RuntimeError("DEV prediction rows must be ranks6..50")
    for _, d in meta.groupby("image_id", sort=False):
        if d["road8_rank"].astype(int).tolist() != list(range(6, 51)):
            raise RuntimeError("DEV prediction metadata rank order mismatch")
    x_dev = np.load(p1 / "cache" / "dev_X_scaled.npy", mmap_mode="r")
    if x_dev.shape != (90000, 90) or not np.isfinite(x_dev).all():
        raise RuntimeError("frozen DEV feature cache invalid")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Complete FULL adapter regression: the unledgered DEV feature cache must
    # reproduce every frozen post-temperature probability within the frozen tolerance.
    regression = []
    full_prob_by_seed: dict[int, np.ndarray] = {}
    for seed in SEEDS:
        model = load_model(p1 / "models" / f"marginal_mlp_seed_{seed}.pt", device)
        temperature = float(json.loads((p1 / "models" / f"temperature_seed_{seed}.json").read_text(encoding="utf-8"))["temperature"])
        prob = predict_probabilities(model, np.asarray(x_dev), temperature, device)
        expected = meta[prediction_columns(seed)].to_numpy(np.float64)
        max_abs = float(np.max(np.abs(prob - expected)))
        within = bool(np.allclose(prob, expected, atol=1e-6, rtol=1e-5))
        regression.append({"seed": seed, "probability_max_abs": max_abs, "within_tolerance": within})
        if not within:
            raise RuntimeError(f"FULL DEV probability regression failed seed={seed} max_abs={max_abs}")
        full_prob_by_seed[seed] = expected
        del model
    write_json(root / "qa" / "full_probability_regression.json", {"status": "PASS", "rows": 90000, "checks": regression, "atol": 1e-6, "rtol": 1e-5})

    pred_dir = root / "dev_predictions_and_selections" / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    prediction_assets = []
    new_prob: dict[tuple[str, int], np.ndarray] = {}
    meta_cols = ["image_id", "group_id", "road8_rank", "candidate_record_id", "query_index", "predicted_road8_class_id", "raw_score"]
    for variant in VARIANTS:
        indices = np.asarray(masks[variant]["indices"], dtype=np.int64)
        xv = np.asarray(x_dev, dtype=np.float32).copy()
        if len(indices):
            xv[:, indices] = 0.0
        for seed in SEEDS:
            model_path = root / "models" / variant / f"marginal_mlp_seed_{seed}.pt"
            temp_path = root / "models" / variant / f"temperature_seed_{seed}.json"
            snapshot = torch.load(model_path, map_location="cpu", weights_only=True)
            expected_target = "MATCHABILITY" if variant == "MATCHABILITY_TARGET" else "PREFIX_MARGINAL"
            if (str(snapshot.get("variant")) != variant or int(snapshot.get("seed", -1)) != seed
                    or str(snapshot.get("target")) != expected_target
                    or list(snapshot.get("mask_indices", [])) != indices.tolist()
                    or str(snapshot.get("mask_stage")) != "post_standardization"):
                raise RuntimeError(f"new model metadata mismatch {variant} seed={seed}")
            temp_meta = json.loads(temp_path.read_text(encoding="utf-8"))
            if str(temp_meta.get("variant")) != variant or int(temp_meta.get("seed", -1)) != seed:
                raise RuntimeError(f"new temperature metadata mismatch {variant} seed={seed}")
            model = load_model(model_path, device)
            temperature = float(temp_meta["temperature"])
            prob = predict_probabilities(model, xv, temperature, device)
            if prob.shape != (90000, 10) or not np.isfinite(prob).all():
                raise RuntimeError(f"invalid DEV prediction {variant} seed={seed}")
            new_prob[(variant, seed)] = prob
            out = meta[meta_cols].copy()
            for ti, column in enumerate(new_prediction_columns()):
                out[column] = prob[:, ti]
            path = pred_dir / f"{variant.lower()}_seed_{seed}.parquet"
            write_parquet(out, path)
            prediction_assets.append({
                "variant": variant, "seed": seed, "path": str(path), "sha256": sha256_file(path), "rows": len(out),
                "model_path": str(model_path), "model_sha256": sha256_file(model_path),
                "temperature_path": str(temp_path), "temperature_sha256": sha256_file(temp_path),
            })
            append_log(root, f"DEV_PREDICTION variant={variant} seed={seed} rows={len(out)}")
            del model
        del xv

    candidates = load_dev_candidates(release)
    candidate_by_image = {image_id: d.sort_values("road8_rank", kind="stable").reset_index(drop=True) for image_id, d in candidates.groupby("image_id", sort=False)}
    pred_by_image = {image_id: d.sort_values("road8_rank", kind="stable").reset_index(drop=True) for image_id, d in meta.groupby("image_id", sort=False)}
    if set(candidate_by_image) != set(pred_by_image):
        raise RuntimeError("P1 DEV metadata/release image identity mismatch")
    for image_id, pred_rows in pred_by_image.items():
        release_rows = candidate_by_image[image_id].iloc[5:50].reset_index(drop=True)
        for column in ("candidate_record_id", "query_index", "predicted_road8_class_id", "road8_rank"):
            left = pred_rows[column].astype(str if column == "candidate_record_id" else np.int64).to_numpy()
            right = release_rows[column].astype(str if column == "candidate_record_id" else np.int64).to_numpy()
            if not np.array_equal(left, right):
                raise RuntimeError(f"P1 DEV metadata/release mismatch image={image_id} column={column}")
        if not np.allclose(pred_rows["raw_score"].to_numpy(np.float64), release_rows["score"].to_numpy(np.float64), atol=0.0, rtol=0.0):
            raise RuntimeError(f"P1 DEV metadata/release raw score mismatch image={image_id}")
    group_images = {
        int(group_id): sorted(d["image_id"].astype(str).unique().tolist())
        for group_id, d in meta[["group_id", "image_id"]].drop_duplicates().groupby("group_id", sort=True)
    }
    if len(group_images) != 50 or any(len(x) != 40 for x in group_images.values()):
        raise RuntimeError("DEV groups are not the frozen 50x40 partition")
    if sorted(group_images) != list(range(50)):
        raise RuntimeError("DEV group IDs are not the frozen 0..49 range")
    p1_alloc = pq.read_table(p1 / "cache" / "dev_allocations.parquet").to_pandas()
    p1_full = p1_alloc[p1_alloc["method"] == "LEARN_QUALITY"].copy()
    p1_full_lookup = {(int(r.seed), int(r.budget), str(r.image_id)): int(r.K_i) for r in p1_full.itertuples(index=False)}
    if len(p1_full_lookup) != 30000:
        raise RuntimeError("frozen FULL allocation cells incomplete")

    allocation_rows = []
    gap_rows = []
    full_k_mismatch = 0
    total_selected_records = 0
    for group_id, ids in group_images.items():
        for seed in SEEDS:
            full_margins = np.vstack([
                meta.iloc[pred_by_image[x]["_row_index"].to_numpy(np.int64)][prediction_columns(seed)].to_numpy(np.float64).mean(axis=1)
                * class_weights[pred_by_image[x]["predicted_road8_class_id"].to_numpy(np.int16) - 1]
                for x in ids
            ])
            dp_k, dp_obj = solve_group_allocations(full_margins, BUDGETS)
            greedy_k = next_slot_greedy(full_margins, BUDGETS)
            choices = group_choice_values_from_marginals(full_margins)
            for bi, budget in enumerate(BUDGETS):
                frozen_k = np.asarray([p1_full_lookup[(seed, int(budget), image_id)] for image_id in ids], dtype=np.int64)
                full_k_mismatch += int(np.sum(dp_k[bi] != frozen_k))
                gk = greedy_k[int(budget)]
                dp_recomputed = objective_for_k(choices, dp_k[bi])
                greedy_objective = objective_for_k(choices, gk)
                if dp_recomputed != float(dp_obj[bi]):
                    raise RuntimeError("FULL DP objective reconstruction mismatch")
                gap = float(dp_recomputed - greedy_objective)
                tolerance = 1e-12 * max(1.0, abs(dp_recomputed), abs(greedy_objective))
                if gap < -tolerance:
                    raise RuntimeError(f"exact DP worse than next-slot greedy group={group_id} seed={seed} budget={budget} gap={gap}")
                l1 = int(np.abs(dp_k[bi] - gk).sum())
                gap_rows.append({
                    "seed": seed, "budget": int(budget), "group_id": group_id, "total_capacity": 40 * int(budget),
                    "dp_objective": dp_recomputed, "greedy_objective": greedy_objective,
                    "gap_dp_minus_greedy": gap,
                    "normalized_gap_by_abs_dp": gap / abs(dp_recomputed) if dp_recomputed != 0 else (0.0 if greedy_objective == 0 else math.nan),
                    "qa_tolerance": tolerance, "dp_not_worse": bool(gap >= -tolerance),
                    "images_K_changed": int(np.sum(dp_k[bi] != gk)), "l1_K_difference": l1,
                    "max_abs_K_difference": int(np.max(np.abs(dp_k[bi] - gk))),
                    "prefix_slots_reallocated": l1 // 2, "selection_symmetric_difference": l1,
                    "exact_selection_match": bool(np.array_equal(dp_k[bi], gk)),
                })
                for ii, image_id in enumerate(ids):
                    k = int(gk[ii])
                    selected = candidate_by_image[image_id].iloc[:k]["candidate_record_id"].astype(str).tolist()
                    allocation_rows.append({
                        "method": "NEXT_SLOT_GREEDY", "model_variant": "FULL", "solver": "NEXT_SLOT_GREEDY",
                        "training_target": "PREFIX_MARGINAL", "mask": "NONE", "seed": seed,
                        "budget": int(budget), "group_id": group_id, "image_id": image_id, "K_i": k,
                        "selected_record_ids": selected, "group_predicted_objective": greedy_objective,
                        "source": "NEW_ABLATION_P0_SOLVER_REPLAY", "source_method": "LEARN_QUALITY",
                    })
                    total_selected_records += k
        for variant in VARIANTS:
            for seed in SEEDS:
                prob = new_prob[(variant, seed)]
                margins = np.vstack([
                    prob[pred_by_image[x]["_row_index"].to_numpy(np.int64)].mean(axis=1)
                    * class_weights[pred_by_image[x]["predicted_road8_class_id"].to_numpy(np.int16) - 1]
                    for x in ids
                ])
                alloc, objectives = solve_group_allocations(margins, BUDGETS)
                for bi, budget in enumerate(BUDGETS):
                    for ii, image_id in enumerate(ids):
                        k = int(alloc[bi, ii])
                        selected = candidate_by_image[image_id].iloc[:k]["candidate_record_id"].astype(str).tolist()
                        allocation_rows.append({
                            "method": variant, "model_variant": variant, "solver": "EXACT_DP",
                            "training_target": "MATCHABILITY" if variant == "MATCHABILITY_TARGET" else "PREFIX_MARGINAL",
                            "mask": variant if variant.startswith("NO_") else "NONE", "seed": seed,
                            "budget": int(budget), "group_id": group_id, "image_id": image_id, "K_i": k,
                            "selected_record_ids": selected, "group_predicted_objective": float(objectives[bi]),
                            "source": "NEW_ABLATION_P0_TRAIN", "source_method": "",
                        })
                        total_selected_records += k
    if full_k_mismatch:
        raise RuntimeError(f"FULL exact-DP allocation regression mismatch images={full_k_mismatch}")
    allocations = pd.DataFrame(allocation_rows)
    if len(allocations) != 120000 or total_selected_records != 2_760_000:
        raise RuntimeError(f"new allocation/selection cardinality mismatch rows={len(allocations)} selected={total_selected_records}")
    if not allocations["K_i"].between(5, 50).all():
        raise RuntimeError("K bounds violated")
    budget_sum = allocations.groupby(["method", "seed", "budget", "group_id"], sort=False)["K_i"].sum()
    expected_sum = budget_sum.index.get_level_values("budget").to_numpy(np.int64) * 40
    if not np.array_equal(budget_sum.to_numpy(np.int64), expected_sum):
        raise RuntimeError("exact group budget violated")
    for r in allocations.itertuples(index=False):
        if len(r.selected_record_ids) != int(r.K_i):
            raise RuntimeError("selected record list length != K")
        canonical = candidate_by_image[str(r.image_id)].iloc[:int(r.K_i)]["candidate_record_id"].astype(str).tolist()
        if list(r.selected_record_ids) != canonical or len(set(r.selected_record_ids)) != int(r.K_i):
            raise RuntimeError("selected record list is not exact canonical prefix")

    allocation_path = root / "dev_predictions_and_selections" / "new_allocations_and_selections.parquet"
    gap_path = root / "outputs" / "solver_objective_gap.parquet"
    write_parquet(allocations, allocation_path)
    write_parquet(pd.DataFrame(gap_rows), gap_path)
    commit = {
        "status": "DEV_SELECTIONS_FROZEN_BEFORE_GT_EVALUATION",
        "prediction_assets": prediction_assets,
        "allocation_path": str(allocation_path), "allocation_sha256": sha256_file(allocation_path),
        "solver_gap_path": str(gap_path), "solver_gap_sha256": sha256_file(gap_path),
        "rows": {"prediction_shards": 9, "prediction_rows_total": 810000, "allocation_cells": len(allocations), "represented_selected_records": total_selected_records, "solver_gap_rows": len(gap_rows)},
        "full_probability_regression": "PASS", "full_exact_dp_K_regression_mismatch": full_k_mismatch,
        "group_budget_mismatch": 0, "prefix_identity_mismatch": 0,
        "dev_gt_opened_by_this_entrypoint": False,
        "dev_gt_path_derivable_from_release_root": True,
    }
    write_json(root / "outputs" / "dev_selection_commit.json", commit)
    append_log(root, f"DEV_SELECTIONS_FROZEN sha={commit['allocation_sha256']} rows={len(allocations)} represented_selected={total_selected_records}")
    append_log(root, f"STAGE predict_allocate COMPLETE elapsed_seconds={time.perf_counter()-t0:.6f}")


if __name__ == "__main__":
    main()
