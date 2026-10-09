"""GT-free DEV feature inference and exact prefix-budget allocation.

This command intentionally has no DEV-GT argument and never imports an
evaluation module.  It persists and hashes all choices before evaluation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from dp_solver import (  # noqa: E402
    BUDGET_MEANS, group_choice_values_from_marginals, self_test,
    solve_exact_multiple_choice, solve_group_allocations,
)
from p1_core import (  # noqa: E402
    ROAD8, SEEDS, THRESHOLDS, build_raw_features, predict_table,
    sha256_file, sigmoid, write_json,
)
from release_io import ReleaseReader  # noqa: E402
from train_models import MarginalMLP  # noqa: E402


def append_log(root: Path, text: str) -> None:
    with (root / "runtime.log").open("a", encoding="utf-8") as f:
        f.write(text.rstrip() + "\n")


def load_models(root: Path, device: torch.device) -> tuple[dict[int, MarginalMLP], dict[int, float]]:
    models, temperatures = {}, {}
    for seed in SEEDS:
        snapshot = torch.load(root / "models" / f"marginal_mlp_seed_{seed}.pt", map_location="cpu", weights_only=True)
        model = MarginalMLP(int(snapshot["input_dim"]))
        model.load_state_dict(snapshot["state_dict"])
        model.eval().to(device)
        models[seed] = model
        temperatures[seed] = float(json.loads((root / "models" / f"temperature_seed_{seed}.json").read_text(encoding="utf-8"))["temperature"])
    return models, temperatures


def model_predict(models: dict[int, MarginalMLP], temperatures: dict[int, float], x: np.ndarray, device: torch.device) -> dict[int, np.ndarray]:
    xt = torch.from_numpy(np.asarray(x, dtype=np.float32)).to(device)
    result = {}
    with torch.inference_mode():
        for seed, model in models.items():
            parts = []
            for start in range(0, len(xt), 16384):
                parts.append(model(xt[start:start + 16384]).float().cpu().numpy())
            logits = np.vstack(parts).astype(np.float64)
            result[seed] = sigmoid(logits / temperatures[seed])
    if device.type == "cuda":
        torch.cuda.synchronize()
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", required=True)
    ap.add_argument("--release-root", required=True)
    ap.add_argument("--group-manifest", required=True)
    ap.add_argument("--r0-selections", required=True)
    args = ap.parse_args()
    root = Path(args.project_root).resolve()
    release_root = Path(args.release_root).resolve()
    t0 = time.perf_counter()
    append_log(root, "STAGE predict_allocate START (DEV GT inaccessible by interface)")

    dp_test = self_test()
    if dp_test.get("status") != "PASS":
        raise RuntimeError(f"exact DP self-test failed: {dp_test}")
    write_json(root / "qa" / "dp_self_test.json", dp_test)

    reader = ReleaseReader(release_root)
    dev_manifest = reader.split_manifest("DEV")
    group = pq.read_table(args.group_manifest).to_pandas()
    if len(group) != 2000 or group["image_id"].nunique() != 2000 or group["group_id"].nunique() != 50:
        raise RuntimeError("frozen group manifest invariant failed")
    if set(group["image_id"].astype(str)) != set(dev_manifest["image_id"].astype(str)):
        raise RuntimeError("DEV/group identity mismatch")
    if not (group.groupby("group_id").size() == 40).all():
        raise RuntimeError("frozen groups are not 50x40")
    group_map = dict(zip(group["image_id"].astype(str), group["group_id"].astype(int)))

    pca = joblib.load(root / "models" / "pca32.joblib")
    scaler_bundle = joblib.load(root / "models" / "feature_scaler.joblib")
    scaler, standardize = scaler_bundle["scaler"], np.asarray(scaler_bundle["standardize_mask"], dtype=bool)
    with np.load(root / "models" / "table_model.npz", allow_pickle=False) as z:
        table_model = {k: z[k].copy() for k in z.files}
    class_weights = np.asarray(json.loads((root / "models" / "class_weights.json").read_text(encoding="utf-8"))["weights"], dtype=np.float64)

    dev_x = np.lib.format.open_memmap(root / "cache" / "dev_X_scaled.npy", mode="w+", dtype=np.float32, shape=(90000, 90))
    prediction_meta = []
    all_candidates = []
    image_order = []
    row0 = 0
    read_feature_start = time.perf_counter()
    for bundle in reader.iter_bundles("DEV"):
        x = build_raw_features(bundle.candidates, bundle.road8_logits, bundle.embeddings, pca, bundle.width, bundle.height)
        x[:, standardize] = scaler.transform(x[:, standardize])
        dev_x[row0:row0 + 45] = x.astype(np.float32)
        cur = bundle.candidates.iloc[5:50]
        for r in cur.itertuples(index=False):
            prediction_meta.append({
                "image_id": bundle.image_id, "group_id": group_map[bundle.image_id],
                "road8_rank": int(r.road8_rank), "candidate_record_id": str(r.candidate_record_id),
                "query_index": int(r.query_index), "predicted_road8_class_id": int(r.predicted_road8_class_id),
                "raw_score": float(r.score),
            })
        ccopy = bundle.candidates.copy()
        ccopy["group_id"] = group_map[bundle.image_id]
        all_candidates.append(ccopy)
        image_order.append(bundle.image_id)
        row0 += 45
    dev_x.flush()
    if row0 != 90000 or len(set(image_order)) != 2000:
        raise RuntimeError("DEV feature row/image count mismatch")
    feature_seconds = time.perf_counter() - read_feature_start
    candidates = pd.concat(all_candidates, ignore_index=True)
    candidates = candidates.sort_values(["image_id", "road8_rank"], kind="stable").reset_index(drop=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models, temperatures = load_models(root, device)
    infer_start = time.perf_counter()
    learned_prob = model_predict(models, temperatures, np.asarray(dev_x), device)
    inference_seconds = time.perf_counter() - infer_start
    meta = pd.DataFrame(prediction_meta)
    table_prob = predict_table(
        table_model, meta["predicted_road8_class_id"].to_numpy(),
        meta["road8_rank"].to_numpy(), meta["raw_score"].to_numpy(),
    )
    for ti, threshold in enumerate(THRESHOLDS):
        meta[f"table_p_iou_{threshold:.2f}".replace(".", "_")] = table_prob[:, ti]
    for seed in SEEDS:
        for ti, threshold in enumerate(THRESHOLDS):
            meta[f"learn_p_seed_{seed}_iou_{threshold:.2f}".replace(".", "_")] = learned_prob[seed][:, ti]
    prediction_path = root / "dev_predictions.parquet"
    pq.write_table(pa.Table.from_pandas(meta, preserve_index=False), prediction_path, compression="zstd")

    image_row = {image_id: ix for ix, image_id in enumerate(image_order)}
    candidate_by_image = {str(i): d.sort_values("road8_rank", kind="stable").reset_index(drop=True) for i, d in candidates.groupby("image_id", sort=False)}
    pred_by_image = {str(i): d.sort_values("road8_rank", kind="stable").reset_index(drop=True) for i, d in meta.groupby("image_id", sort=False)}

    policies: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    cls_all = meta["predicted_road8_class_id"].to_numpy(np.int16)
    policies[("TABLE_MICRO", -1)] = {x: pred_by_image[x]["table_p_iou_0_50"].to_numpy(np.float64) for x in image_order}
    table_quality = table_prob.mean(axis=1) * class_weights[cls_all - 1]
    tq_map = dict(zip(zip(meta["image_id"].astype(str), meta["road8_rank"].astype(int)), table_quality))
    policies[("TABLE_QUALITY", -1)] = {x: np.asarray([tq_map[(x, k)] for k in range(6, 51)], dtype=np.float64) for x in image_order}
    image_slices = {image_id: slice(ix * 45, (ix + 1) * 45) for ix, image_id in enumerate(image_order)}
    for seed in SEEDS:
        policies[("LEARN_MICRO", seed)] = {x: learned_prob[seed][image_slices[x], 0] for x in image_order}
        q = learned_prob[seed].mean(axis=1) * class_weights[cls_all - 1]
        policies[("LEARN_QUALITY", seed)] = {x: np.asarray(q[image_slices[x]], dtype=np.float64) for x in image_order}

    # Reuse frozen S_FIXED/S_ADAPT selections and K values exactly.
    r0 = pq.read_table(args.r0_selections, filters=[("method", "in", ["S_FIXED", "S_ADAPT"])]).to_pandas()
    r0["image_id"] = r0["image_id"].astype(str)
    k_frozen = r0.groupby(["method", "budget", "image_id"], sort=False).size().rename("K_i").reset_index()
    if len(k_frozen) != 2 * 5 * 2000:
        raise RuntimeError("R0 frozen S K cells incomplete")
    r0_prefix_bad = 0
    for _, d in r0.groupby(["method", "budget", "image_id"], sort=False):
        ranks = sorted(d["road8_rank"].astype(int).tolist())
        r0_prefix_bad += int(ranks != list(range(1, len(ranks) + 1)))
    if r0_prefix_bad:
        raise RuntimeError(f"R0 S prefix violations: {r0_prefix_bad}")
    frozen_k = {(str(r.method), int(r.budget), str(r.image_id)): int(r.K_i) for r in k_frozen.itertuples(index=False)}

    allocation_rows = []
    dp_start = time.perf_counter()
    raw_objective_gaps = []
    for group_id, gd in group.groupby("group_id", sort=True):
        ids = sorted(gd["image_id"].astype(str).tolist())
        capacities = [40 * b for b in BUDGET_MEANS]

        # Baselines: preserve exact frozen K and validate prefix-score optimum.
        for method in ("S_FIXED", "S_ADAPT"):
            for budget in BUDGET_MEANS:
                for image_id in ids:
                    k = frozen_k[(method, int(budget), image_id)]
                    allocation_rows.append({
                        "method": method, "seed": -1, "value_target": "RAW_SCORE",
                        "budget": int(budget), "group_id": int(group_id), "image_id": image_id, "K_i": k,
                        "allocation_source": "R0_FROZEN",
                    })
        raw_marginals = np.vstack([candidate_by_image[x].iloc[5:50]["score"].to_numpy(np.float64) for x in ids])
        raw_choices = group_choice_values_from_marginals(raw_marginals)
        _, opt_obj = solve_exact_multiple_choice(raw_choices, capacities, k_min=5)
        for bi, budget in enumerate(BUDGET_MEANS):
            kvals = np.asarray([frozen_k[("S_ADAPT", int(budget), x)] for x in ids], dtype=np.int64)
            frozen_obj = np.float64(0.0)
            for ii, k in enumerate(kvals):
                frozen_obj = np.float64(frozen_obj + raw_choices[ii, k - 5])
            raw_objective_gaps.append(float(opt_obj[bi] - frozen_obj))

        for (method, seed), per_image in policies.items():
            margins = np.vstack([per_image[x] for x in ids]).astype(np.float64)
            alloc, objectives = solve_group_allocations(margins, BUDGET_MEANS)
            for bi, budget in enumerate(BUDGET_MEANS):
                for ii, image_id in enumerate(ids):
                    allocation_rows.append({
                        "method": method, "seed": int(seed),
                        "value_target": "MICRO" if method.endswith("MICRO") else "QUALITY",
                        "budget": int(budget), "group_id": int(group_id), "image_id": image_id,
                        "K_i": int(alloc[bi, ii]), "allocation_source": "P1_EXACT_DP",
                        "group_predicted_objective": float(objectives[bi]),
                    })
    dp_seconds = time.perf_counter() - dp_start
    if max(np.abs(raw_objective_gaps), default=0.0) > 1e-9:
        raise RuntimeError(f"frozen S_ADAPT is not raw-score objective-optimal: maxgap={max(raw_objective_gaps)}")
    allocations = pd.DataFrame(allocation_rows)
    if len(allocations) != 10 * 5 * 2000:
        raise RuntimeError(f"allocation cell count {len(allocations)} != 100000")
    budget_check = allocations.groupby(["method", "seed", "budget", "group_id"])["K_i"].sum()
    expected = budget_check.index.get_level_values("budget").to_numpy() * 40
    if not np.array_equal(budget_check.to_numpy(), expected) or not allocations["K_i"].between(5, 50).all():
        raise RuntimeError("exact group budget invariant failed")

    selection_path = root / "dev_allocations_and_selections.parquet"
    selection_writer = None
    selection_rows = 0
    try:
        for (method, seed, budget), allocs in allocations.groupby(["method", "seed", "budget"], sort=True):
            alloc_small = allocs[["image_id", "K_i"]].copy()
            batch_df = candidates.merge(alloc_small, on="image_id", how="inner", validate="many_to_one")
            batch_df = batch_df[batch_df["road8_rank"].astype(int) <= batch_df["K_i"].astype(int)].copy()
            batch_df.insert(0, "selection_rank", batch_df["road8_rank"].to_numpy(np.int16))
            batch_df.insert(0, "budget", int(budget))
            batch_df.insert(0, "value_target", str(allocs["value_target"].iloc[0]))
            batch_df.insert(0, "seed", int(seed))
            batch_df.insert(0, "method", str(method))
            if len(batch_df) != int(allocs["K_i"].sum()):
                raise RuntimeError("vectorized prefix materialization count mismatch")
            if batch_df.duplicated(["method", "seed", "budget", "image_id", "candidate_record_id"]).any():
                raise RuntimeError("duplicate selected candidate within a condition/image")
            batch_table = pa.Table.from_pandas(batch_df, preserve_index=False)
            if selection_writer is None:
                selection_writer = pq.ParquetWriter(selection_path, batch_table.schema, compression="zstd")
            selection_writer.write_table(batch_table)
            selection_rows += len(batch_df)
    finally:
        if selection_writer is not None:
            selection_writer.close()
    if selection_rows != 2_300_000:
        raise RuntimeError(f"selection capacity invariant failed rows={selection_rows}")
    allocation_path = root / "cache" / "dev_allocations.parquet"
    pq.write_table(pa.Table.from_pandas(allocations, preserve_index=False), allocation_path, compression="zstd")

    commit = {
        "status": "DEV_SELECTIONS_FROZEN_BEFORE_GT_EVALUATION",
        "dev_predictions_path": str(prediction_path), "dev_predictions_sha256": sha256_file(prediction_path),
        "dev_selections_path": str(selection_path), "dev_selections_sha256": sha256_file(selection_path),
        "allocation_path": str(allocation_path), "allocation_sha256": sha256_file(allocation_path),
        "rows": {"predictions": len(meta), "allocation_cells": len(allocations), "selections": selection_rows},
        "group_budget_mismatch": 0, "prefix_mismatch": 0,
        "r0_s_adapt_raw_objective_max_gap": max(np.abs(raw_objective_gaps), default=0.0),
        "dev_gt_path_available_to_this_entrypoint": False,
    }
    write_json(root / "outputs" / "dev_selection_commit.json", commit)
    write_json(root / "outputs" / "prediction_runtime.json", {
        "device": str(device), "feature_and_release_read_seconds": feature_seconds,
        "three_model_shared_inference_seconds": inference_seconds, "exact_dp_seconds": dp_seconds,
        "total_seconds": time.perf_counter() - t0,
    })
    append_log(root, f"DEV_SELECTIONS_FROZEN sha={commit['dev_selections_sha256']} rows={selection_rows}")
    append_log(root, f"STAGE predict_allocate COMPLETE elapsed_seconds={time.perf_counter()-t0:.6f}")


if __name__ == "__main__":
    main()
