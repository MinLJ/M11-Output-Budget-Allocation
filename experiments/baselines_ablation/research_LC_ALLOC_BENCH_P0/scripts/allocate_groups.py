"""Commit all A-chain allocations/selections before any DEV GT evaluation."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from common import (BUDGETS, P1, R0, ROOT, SEEDS, append_log, import_file,
                    load_candidate_id_map, load_dev_prediction_arrays,
                    selected_ids_for_k, sha256_file, solve_group, write_json,
                    write_parquet)


def ordered_slot_solution(image_ids: list[str], full_scores: np.ndarray, budgets: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Exact threshold/order-slot solution for non-increasing raw-score slots.

    Exact-value ties independently implement the frozen R0 rule: earlier slot,
    then lexicographically lower image_id.
    """
    values = np.asarray(full_scores, dtype=np.float64)
    n, slots = values.shape
    if slots != 50 or not np.all(values[:, :-1] >= values[:, 1:]):
        raise ValueError("ordered-slot solver requires 50 non-increasing raw scores")
    order = sorted(
        ((-float(values[i, j]), j, str(image_ids[i]), i) for i in range(n) for j in range(5, slots)),
        key=lambda z: (z[0], z[1], z[2]),
    )
    out, objectives = [], []
    for b in budgets:
        take = n * (int(b) - 5)
        k = np.full(n, 5, dtype=np.int64)
        objective = np.float64(0.0)
        for neg, j, _, i in order[:take]:
            if j != k[i]:
                raise AssertionError("non-prefix ordered slot encountered")
            k[i] += 1
            objective = np.float64(objective - neg)
        out.append(k)
        objectives.append(objective)
    return np.asarray(out), np.asarray(objectives, dtype=np.float64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=str(ROOT))
    args = ap.parse_args()
    root = Path(args.project_root).resolve()
    start = time.perf_counter()
    append_log("STAGE allocate_groups START (no DEV GT path in interface)")

    manifest_path = root / "outputs" / "group_manifest.parquet"
    manifest = pq.read_table(manifest_path).to_pandas()
    ids, _, _, m11_values, _ = load_dev_prediction_arrays()
    candidate_frame, candidate_ids = load_candidate_id_map("DEV", 50)
    if set(candidate_ids) != set(ids):
        raise RuntimeError("candidate/prediction identity mismatch")
    full_scores = {
        str(image_id): d.sort_values("road8_rank", kind="stable")["score"].to_numpy(np.float64)
        for image_id, d in candidate_frame.groupby("image_id", sort=False)
    }
    r0_solver = import_file(R0 / "scripts" / "alloc_solver.py", "lc_alloc_bench_r0_solver")

    rows: list[dict] = []
    threshold_cells = 0
    threshold_k_mismatch = 0
    threshold_objective_mismatch = 0
    for (permutation, group_size, group_id), group in manifest.groupby(["permutation", "group_size", "group_id"], sort=True):
        caller_ids = sorted(group["image_id"].astype(str).tolist())
        n = len(caller_ids)
        score_matrix = np.stack([full_scores[x] for x in caller_ids])
        score_rows, score_objectives = [], []
        for budget in BUDGETS:
            k_vec, objective = r0_solver.allocate_prefixes(caller_ids, score_matrix, n * budget, 5, 50)
            score_rows.append(k_vec)
            score_objectives.append(objective)
        score_k = np.asarray(score_rows, dtype=np.int64)
        score_obj = np.asarray(score_objectives, dtype=np.float64)
        ordered_k, ordered_optional_obj = ordered_slot_solution(caller_ids, score_matrix, BUDGETS)
        ordered_obj = np.asarray([
            sum(np.sum(score_matrix[i, :int(k)], dtype=np.float64) for i, k in enumerate(row))
            for row in ordered_k
        ], dtype=np.float64)
        threshold_cells += len(BUDGETS)
        threshold_k_mismatch += int(np.sum(np.any(score_k != ordered_k, axis=1)))
        threshold_objective_mismatch += int(np.sum(score_obj != ordered_obj))

        conditions = [("S_ADAPT", -1, score_k, score_obj)]
        fixed_k = np.repeat(np.asarray(BUDGETS, dtype=np.int64)[:, None], n, axis=1)
        fixed_obj = np.full(len(BUDGETS), np.nan, dtype=np.float64)
        conditions.append(("S_FIXED", -1, fixed_k, fixed_obj))
        for seed_index, seed in enumerate(SEEDS):
            matrix = np.stack([m11_values[x][seed_index] for x in caller_ids])
            learned_k, learned_obj = solve_group(matrix)
            conditions.append(("M11", seed, learned_k, learned_obj))

        for method, seed, all_k, objectives in conditions:
            for bi, budget in enumerate(BUDGETS):
                if int(all_k[bi].sum()) != n * budget or all_k[bi].min() < 5 or all_k[bi].max() > 50:
                    raise RuntimeError("group budget/bound invariant failed")
                for image_id, k in zip(caller_ids, all_k[bi]):
                    rows.append({
                        "experiment": "GROUP_ROBUSTNESS",
                        "permutation": str(permutation),
                        "group_size": int(group_size),
                        "group_id": int(group_id),
                        "method": method,
                        "seed": int(seed),
                        "budget": int(budget),
                        "image_id": image_id,
                        "K_i": int(k),
                        "selected_record_ids": selected_ids_for_k(candidate_ids, image_id, int(k)),
                        "group_predicted_objective": float(objectives[bi]),
                        "source": "NEW_BENCH_P0",
                    })
    allocations = pd.DataFrame(rows)
    if len(allocations) != 600000:
        raise RuntimeError(f"allocation rows={len(allocations)} != 600000")

    # Exact R0/n40 allocation and selected-ID parity with frozen P1 outputs.
    new_anchor = allocations[(allocations.permutation == "R0") & (allocations.group_size == 40)].copy()
    frozen_alloc = pq.read_table(P1 / "cache" / "dev_allocations.parquet").to_pandas()
    frozen_alloc = frozen_alloc[
        ((frozen_alloc.method == "LEARN_QUALITY") & frozen_alloc.seed.isin(SEEDS)) |
        ((frozen_alloc.method == "S_ADAPT") & (frozen_alloc.seed == -1)) |
        ((frozen_alloc.method == "S_FIXED") & (frozen_alloc.seed == -1))
    ].copy()
    frozen_alloc["method"] = frozen_alloc["method"].replace({"LEARN_QUALITY": "M11"})
    merged = new_anchor.merge(
        frozen_alloc[["method", "seed", "budget", "image_id", "K_i"]],
        on=["method", "seed", "budget", "image_id"], how="outer", suffixes=("_new", "_old"), indicator=True,
    )
    if len(merged) != 50000 or not (merged._merge == "both").all() or not (merged.K_i_new == merged.K_i_old).all():
        bad = merged[(merged._merge != "both") | (merged.K_i_new != merged.K_i_old)]
        raise RuntimeError(f"R0/n40 frozen allocation parity failed cells={len(bad)}")

    old_columns = ["method", "seed", "budget", "image_id", "road8_rank", "candidate_record_id"]
    old_selection = pq.read_table(P1 / "dev_allocations_and_selections.parquet", columns=old_columns).to_pandas()
    old_selection = old_selection[
        ((old_selection.method == "LEARN_QUALITY") & old_selection.seed.isin(SEEDS)) |
        ((old_selection.method == "S_ADAPT") & (old_selection.seed == -1)) |
        ((old_selection.method == "S_FIXED") & (old_selection.seed == -1))
    ].copy()
    old_selection["method"] = old_selection["method"].replace({"LEARN_QUALITY": "M11"})
    old_lists = old_selection.sort_values(["method", "seed", "budget", "image_id", "road8_rank"], kind="stable").groupby(
        ["method", "seed", "budget", "image_id"], sort=False
    )["candidate_record_id"].agg(list)
    new_lists = new_anchor.set_index(["method", "seed", "budget", "image_id"])["selected_record_ids"]
    if len(old_lists) != 50000 or set(old_lists.index) != set(new_lists.index):
        raise RuntimeError("R0/n40 frozen selection identity set mismatch")
    selection_mismatch = sum(tuple(old_lists.loc[k]) != tuple(new_lists.loc[k]) for k in old_lists.index)
    if selection_mismatch:
        raise RuntimeError(f"R0/n40 frozen selected-ID parity failed cells={selection_mismatch}")

    # Degenerate n=1 check for every method/budget on all images.
    n1_mismatch = 0
    for image_id in ids:
        score = full_scores[image_id][None, :]
        for budget in BUDGETS:
            k_score, _ = r0_solver.allocate_prefixes([image_id], score, budget, 5, 50)
            n1_mismatch += int(k_score[0] != budget)
        for si in range(len(SEEDS)):
            k_m11, _ = solve_group(m11_values[image_id][si][None, :])
            n1_mismatch += int(np.sum(k_m11[:, 0] != np.asarray(BUDGETS)))
    if n1_mismatch:
        raise RuntimeError(f"n=1 degeneracy mismatch={n1_mismatch}")

    out = root / "outputs" / "group_allocations.parquet"
    write_parquet(allocations, out)
    write_json(root / "qa" / "allocation_validation.json", {
        "status": "PASS",
        "allocation_rows": len(allocations),
        "condition_rows_expected_main": 300,
        "r0_n40_allocation_cells_compared": 50000,
        "r0_n40_k_mismatch": 0,
        "r0_n40_selected_id_mismatch": 0,
        "threshold_ordered_slot_cells": threshold_cells,
        "threshold_k_mismatch": threshold_k_mismatch,
        "threshold_objective_mismatch": threshold_objective_mismatch,
        "threshold_selection_parity": threshold_k_mismatch == 0,
        "threshold_objective_parity": threshold_objective_mismatch == 0,
        "n1_method_budget_image_checks": 2000 * 4 * len(BUDGETS),
        "n1_mismatch": n1_mismatch,
        "selection_sha256": sha256_file(out),
        "dev_gt_opened": False,
        "elapsed_seconds": time.perf_counter() - start,
    })
    write_json(root / "outputs" / "group_selection_commit.json", {
        "status": "COMMITTED_BEFORE_DEV_GT",
        "path": str(out),
        "sha256": sha256_file(out),
        "rows": len(allocations),
    })
    append_log(f"GROUP_SELECTIONS_COMMITTED_BEFORE_DEV_GT sha256={sha256_file(out)} rows={len(allocations)}")
    append_log(f"STAGE allocate_groups COMPLETE elapsed_seconds={time.perf_counter()-start:.6f}")


if __name__ == "__main__":
    main()
