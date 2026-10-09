"""GT-free LC-ALLOC-P1A objective-ablation replay.

The script reads only frozen P1 probabilities and frozen candidate identities.
It first replays 24 MICRO/QUALITY regression units, then creates only the two
new CLASS50/MULTI policies. DEV ground truth is intentionally absent from the
interface and is not imported by this module.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.dont_write_bytecode = True

SEEDS = (530101, 530102, 530103)
BUDGETS = (10, 15, 20, 30, 40)
NEW_POLICIES = ("LEARN_CLASS50", "LEARN_MULTI")


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


def load_candidate_prefixes(release_root: Path) -> tuple[dict[str, list[str]], dict]:
    identity_path = release_root / "CANDIDATE_ASSET_IDENTITY.json"
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    rows: list[pd.DataFrame] = []
    for rel in identity["assets"]["DEV"]["candidate_shards"]:
        table = pq.read_table(
            release_root / rel,
            columns=["image_id", "road8_rank", "candidate_record_id"],
            filters=[("road8_rank", "<=", 50)],
        )
        rows.append(table.to_pandas())
    cand = pd.concat(rows, ignore_index=True)
    cand["image_id"] = cand["image_id"].astype(str)
    if len(cand) != 100_000 or cand["candidate_record_id"].duplicated().any():
        raise RuntimeError("frozen DEV Top50 candidate identity invariant failed")
    prefixes: dict[str, list[str]] = {}
    for image_id, d in cand.groupby("image_id", sort=False):
        d = d.sort_values("road8_rank", kind="stable")
        if d["road8_rank"].astype(int).tolist() != list(range(1, 51)):
            raise RuntimeError(f"noncanonical Top50 prefix: {image_id}")
        prefixes[str(image_id)] = d["candidate_record_id"].astype(str).tolist()
    if len(prefixes) != 2000:
        raise RuntimeError("DEV candidate image coverage mismatch")
    return prefixes, identity


def probability_columns(seed: int) -> list[str]:
    return [f"learn_p_seed_{seed}_iou_0_{x:02d}" for x in range(50, 100, 5)]


def margins_for(d: pd.DataFrame, seed: int, weights: np.ndarray) -> dict[str, np.ndarray]:
    # P1 materialized this matrix C-contiguously before its row-wise mean.
    # Preserve that exact floating-point reduction path; a pandas column slice
    # is otherwise commonly F-contiguous and can differ by 1--2 ULP.
    probs = np.ascontiguousarray(d[probability_columns(seed)].to_numpy(), dtype=np.float64)
    cls = d["predicted_road8_class_id"].to_numpy(np.int16)
    if probs.shape[1] != 10 or not np.isfinite(probs).all() or np.any((probs < 0) | (probs > 1)):
        raise RuntimeError("frozen learned probability invariant failed")
    return {
        "LEARN_MICRO": probs[:, 0],
        "LEARN_CLASS50": weights[cls - 1] * probs[:, 0],
        "LEARN_MULTI": probs.mean(axis=1),
        "LEARN_QUALITY": weights[cls - 1] * probs.mean(axis=1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", required=True)
    ap.add_argument("--p1-root", required=True)
    args = ap.parse_args()
    root = Path(args.project_root).resolve()
    p1 = Path(args.p1_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    for name in ("outputs", "qa", "cache", "figures"):
        (root / name).mkdir(exist_ok=True)
    t0 = time.perf_counter()
    append_log(root, "STAGE allocate_ablation START (DEV GT inaccessible by interface)")

    # Import the frozen P1 solver without copying or changing it.
    p1_scripts = p1 / "scripts"
    sys.path.insert(0, str(p1_scripts))
    from dp_solver import solve_group_allocations  # type: ignore

    p1_config_path = p1 / "run_config.json"
    p1_config = json.loads(p1_config_path.read_text(encoding="utf-8"))
    release_root = Path(p1_config["release_root"]).resolve()
    prediction_path = p1 / "dev_predictions.parquet"
    p1_alloc_path = p1 / "cache" / "dev_allocations.parquet"
    p1_selection_path = p1 / "dev_allocations_and_selections.parquet"
    weights_path = p1 / "models" / "class_weights.json"
    feature_schema_path = p1 / "feature_schema.json"
    for path in (prediction_path, p1_alloc_path, p1_selection_path, weights_path, feature_schema_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    predictions = pq.read_table(prediction_path).to_pandas()
    predictions["image_id"] = predictions["image_id"].astype(str)
    predictions = predictions.sort_values(["group_id", "image_id", "road8_rank"], kind="stable").reset_index(drop=True)
    if len(predictions) != 90_000 or predictions["image_id"].nunique() != 2000:
        raise RuntimeError("frozen DEV prediction row/image count mismatch")
    for image_id, d in predictions.groupby("image_id", sort=False):
        if d["road8_rank"].astype(int).tolist() != list(range(6, 51)):
            raise RuntimeError(f"frozen prediction prefix ranks invalid: {image_id}")
    group_sizes = predictions[["image_id", "group_id"]].drop_duplicates().groupby("group_id").size()
    if len(group_sizes) != 50 or not (group_sizes == 40).all():
        raise RuntimeError("frozen DEV groups are not 50x40")

    weight_obj = json.loads(weights_path.read_text(encoding="utf-8"))
    weights = np.asarray(weight_obj["weights"], dtype=np.float64)
    if weights.shape != (8,) or not np.isfinite(weights).all():
        raise RuntimeError("frozen class weights invalid")
    prefixes, identity = load_candidate_prefixes(release_root)

    p1_alloc = pq.read_table(p1_alloc_path).to_pandas()
    p1_alloc["image_id"] = p1_alloc["image_id"].astype(str)
    old_alloc_lookup = {
        (str(r.method), int(r.seed), int(r.budget), int(r.group_id), str(r.image_id)): (int(r.K_i), float(r.group_predicted_objective))
        for r in p1_alloc[p1_alloc["method"].isin(["LEARN_MICRO", "LEARN_QUALITY"])].itertuples(index=False)
    }

    # Only two groups, two old policies, three seeds, and two budgets are read
    # from the large frozen P1 selection table for the preregistered 24 units.
    old_sel = pq.read_table(
        p1_selection_path,
        columns=["method", "seed", "budget", "group_id", "image_id", "road8_rank", "candidate_record_id"],
        filters=[
            ("method", "in", ["LEARN_MICRO", "LEARN_QUALITY"]),
            ("seed", "in", list(SEEDS)),
            ("budget", "in", [10, 20]),
            ("group_id", "in", [0, 1]),
        ],
    ).to_pandas()
    old_sel["image_id"] = old_sel["image_id"].astype(str)
    frozen_selected = {
        (str(m), int(s), int(b), int(g), str(i)): tuple(d.sort_values("road8_rank")["candidate_record_id"].astype(str))
        for (m, s, b, g, i), d in old_sel.groupby(["method", "seed", "budget", "group_id", "image_id"], sort=False)
    }

    regression_rows = []
    for group_id in (0, 1):
        gd = predictions[predictions["group_id"] == group_id]
        ids = sorted(gd["image_id"].unique().tolist())
        for seed in SEEDS:
            per_image = {i: margins_for(gd[gd["image_id"] == i], seed, weights) for i in ids}
            for policy in ("LEARN_MICRO", "LEARN_QUALITY"):
                margins = np.vstack([per_image[i][policy] for i in ids]).astype(np.float64)
                # P1 solved all five capacities in one call.  Replay that same
                # numerical path, then inspect only the preregistered 10/20
                # units.  This avoids manufacturing tiny objective differences
                # by changing the solver's maximum-capacity workspace.
                alloc, objectives = solve_group_allocations(margins, BUDGETS)
                for budget in (10, 20):
                    bi = BUDGETS.index(budget)
                    unit_bad_k = 0
                    unit_bad_selection = 0
                    frozen_objective = None
                    for ii, image_id in enumerate(ids):
                        k, obj = old_alloc_lookup[(policy, seed, budget, group_id, image_id)]
                        frozen_objective = obj
                        unit_bad_k += int(int(alloc[bi, ii]) != k)
                        got = tuple(prefixes[image_id][: int(alloc[bi, ii])])
                        expected = frozen_selected[(policy, seed, budget, group_id, image_id)]
                        unit_bad_selection += int(got != expected)
                    objective_gap = float(objectives[bi] - float(frozen_objective))
                    regression_rows.append({
                        "group_id": group_id, "policy": policy, "seed": seed, "budget": budget,
                        "K_mismatch_images": unit_bad_k,
                        "selection_mismatch_images": unit_bad_selection,
                        "predicted_objective_gap": objective_gap,
                        "pass": unit_bad_k == 0 and unit_bad_selection == 0 and objective_gap == 0.0,
                    })
    if len(regression_rows) != 24 or not all(x["pass"] for x in regression_rows):
        raise RuntimeError(f"P1 old-condition regression failed: {regression_rows}")
    write_json(root / "qa" / "p1_regression_24_units.json", {"status": "PASS", "units": regression_rows})

    allocation_rows: list[dict] = []
    for group_id in sorted(predictions["group_id"].unique()):
        gd = predictions[predictions["group_id"] == group_id]
        ids = sorted(gd["image_id"].unique().tolist())
        if len(ids) != 40:
            raise RuntimeError("group image count changed")
        for seed in SEEDS:
            per_image = {i: margins_for(gd[gd["image_id"] == i], seed, weights) for i in ids}
            for policy in NEW_POLICIES:
                margins = np.vstack([per_image[i][policy] for i in ids]).astype(np.float64)
                alloc, objectives = solve_group_allocations(margins, BUDGETS)
                for bi, budget in enumerate(BUDGETS):
                    for ii, image_id in enumerate(ids):
                        k = int(alloc[bi, ii])
                        allocation_rows.append({
                            "image_id": image_id,
                            "group_id": int(group_id),
                            "seed": int(seed),
                            "policy": policy,
                            "variant_id": "M10" if policy == "LEARN_CLASS50" else "M01",
                            "class_weight_on": policy == "LEARN_CLASS50",
                            "multi_threshold_on": policy == "LEARN_MULTI",
                            "budget": int(budget),
                            "K_i": k,
                            "selected_record_ids": prefixes[image_id][:k],
                            "group_predicted_objective": float(objectives[bi]),
                            "source": "NEW_P1A",
                        })
    allocations = pd.DataFrame(allocation_rows)
    if len(allocations) != 60_000:
        raise RuntimeError(f"new allocation row count {len(allocations)} != 60000")
    if int(allocations["K_i"].sum()) != 1_380_000:
        raise RuntimeError("new selected-record theoretical total mismatch")
    if not allocations["K_i"].between(5, 50).all() or not (allocations["selected_record_ids"].map(len) == allocations["K_i"]).all():
        raise RuntimeError("new prefix/list length invariant failed")
    budget_check = allocations.groupby(["policy", "seed", "budget", "group_id"])["K_i"].sum().reset_index()
    if not np.array_equal(budget_check["K_i"].to_numpy(), budget_check["budget"].to_numpy() * 40):
        raise RuntimeError("new exact group budget invariant failed")
    output_path = root / "new_allocations.parquet"
    pq.write_table(pa.Table.from_pandas(allocations, preserve_index=False), output_path, compression="zstd")

    input_bindings = {
        "p1_run_config": {"path": str(p1_config_path), "sha256": sha256_file(p1_config_path)},
        "p1_feature_schema": {"path": str(feature_schema_path), "sha256": sha256_file(feature_schema_path)},
        "p1_dev_predictions": {"path": str(prediction_path), "sha256": sha256_file(prediction_path)},
        "p1_allocations": {"path": str(p1_alloc_path), "sha256": sha256_file(p1_alloc_path)},
        "p1_selections": {"path": str(p1_selection_path), "sha256": sha256_file(p1_selection_path)},
        "p1_class_weights": {"path": str(weights_path), "sha256": sha256_file(weights_path)},
        "release_manifest": {"path": str(release_root / "RELEASE_MANIFEST.json"), "sha256": sha256_file(release_root / "RELEASE_MANIFEST.json")},
        "candidate_asset_identity": {"path": str(release_root / "CANDIDATE_ASSET_IDENTITY.json"), "sha256": sha256_file(release_root / "CANDIDATE_ASSET_IDENTITY.json")},
    }
    model_bindings = []
    for seed in SEEDS:
        mp = p1 / "models" / f"marginal_mlp_seed_{seed}.pt"
        tp = p1 / "models" / f"temperature_seed_{seed}.json"
        model_bindings.append({
            "seed": seed, "model_path": str(mp), "model_sha256": sha256_file(mp),
            "temperature_path": str(tp), "temperature_sha256": sha256_file(tp),
            "temperature": float(json.loads(tp.read_text(encoding="utf-8"))["temperature"]),
        })
    run_config = {
        "task": "LC-ALLOC-P1A",
        "stage": "OBJECTIVE_ABLATION_AND_SINGLE_MODEL_PROFILE",
        "p1_root": str(p1),
        "release_root": str(release_root),
        "candidate_asset_ids": p1_config["candidate_asset_ids"],
        "input_bindings": input_bindings,
        "model_bindings": model_bindings,
        "pca": {"path": str(p1 / "models" / "pca32.joblib"), "sha256": sha256_file(p1 / "models" / "pca32.joblib")},
        "scaler": {"path": str(p1 / "models" / "feature_scaler.joblib"), "sha256": sha256_file(p1 / "models" / "feature_scaler.joblib")},
        "policies": {
            "M00": "LEARN_MICRO: p@0.50",
            "M10": "LEARN_CLASS50: FIT class weight * p@0.50",
            "M01": "LEARN_MULTI: mean p over IoU .50:.95",
            "M11": "LEARN_QUALITY: FIT class weight * mean p over IoU .50:.95",
        },
        "seeds": list(SEEDS), "budgets": list(BUDGETS), "k_bounds": [5, 50],
        "grouping": "reuse P1 frozen DEV 50 groups x 40 images; no cross-group borrowing",
        "solver": "P1 exact float64 multiple-choice DP with frozen lexicographic tie rule",
        "action": "raw road8_rank prefix 1..K_i only",
        "prediction_semantics": "read P1 post-temperature probabilities; no rescaling and no cross-seed allocation",
        "dev_boundary": "previously observed DEV; allocation entrypoint has no GT path",
        "restricted_scientific_splits_accessed": 0,
        "metadata_note": "P1 run_config dev_gt_declared_sha256 differs from the actual frozen release hash; P1 evaluation_summary and final ledger bind the actual release file. P1A uses the actual frozen release and does not alter P1.",
    }
    write_json(root / "run_config.json", run_config)
    commit = {
        "status": "NEW_P1A_ALLOCATIONS_FROZEN_BEFORE_GT_EVALUATION",
        "new_allocations_path": str(output_path),
        "new_allocations_sha256": sha256_file(output_path),
        "p1_dev_predictions_sha256": sha256_file(prediction_path),
        "rows": 60_000,
        "represented_selected_records": 1_380_000,
        "group_budget_mismatch": 0,
        "prefix_identity_mismatch": 0,
        "old_policy_regression_units": 24,
        "old_policy_regression_failures": 0,
        "dev_gt_path_available_to_this_entrypoint": False,
    }
    write_json(root / "outputs" / "new_allocation_commit.json", commit)
    append_log(root, f"NEW_ALLOCATIONS_FROZEN sha={commit['new_allocations_sha256']} rows=60000")
    append_log(root, f"STAGE allocate_ablation COMPLETE elapsed_seconds={time.perf_counter()-t0:.6f}")


if __name__ == "__main__":
    main()
