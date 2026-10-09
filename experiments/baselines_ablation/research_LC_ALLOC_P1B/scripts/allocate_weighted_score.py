"""GT-free construction of the frozen S_CLASS_ADAPT DEV allocations."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.dont_write_bytecode = True

BUDGETS = (10, 15, 20, 30, 40)
METHOD = "S_CLASS_ADAPT"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_log(root: Path, value: str) -> None:
    with (root / "runtime.log").open("a", encoding="utf-8") as f:
        f.write(value.rstrip() + "\n")


def load_dev_top100(release_root: Path, identity: dict) -> pd.DataFrame:
    cols = [
        "image_id", "candidate_record_id", "road8_rank", "score",
        "predicted_road8_class_id", "query_index", "bbox_x1", "bbox_y1",
        "bbox_x2", "bbox_y2", "bbox_cx", "bbox_cy", "bbox_w", "bbox_h",
    ]
    parts = []
    for rel in identity["assets"]["DEV"]["candidate_shards"]:
        parts.append(pq.read_table(release_root / rel, columns=cols, filters=[("road8_rank", "<=", 100)]).to_pandas())
    out = pd.concat(parts, ignore_index=True)
    out["image_id"] = out["image_id"].astype(str)
    out = out.sort_values(["image_id", "road8_rank"], kind="stable").reset_index(drop=True)
    if len(out) != 200_000 or out["image_id"].nunique() != 2000 or out["candidate_record_id"].duplicated().any():
        raise RuntimeError("DEV Top100 identity invariant failed")
    for image_id, d in out.groupby("image_id", sort=False):
        if d["road8_rank"].astype(int).tolist() != list(range(1, 101)):
            raise RuntimeError(f"noncanonical Top100: {image_id}")
        score = d["score"].to_numpy(np.float64)
        if not np.isfinite(score).all() or np.any(np.diff(score) > 0):
            raise RuntimeError(f"raw-score rank invariant failed: {image_id}")
    if not out["predicted_road8_class_id"].between(1, 8).all():
        raise RuntimeError("class id outside Road8")
    return out


def brute_force(choices: np.ndarray, capacity: int, k_min: int) -> tuple[float, tuple[int, ...]]:
    best, best_k = -np.inf, None
    k_max = k_min + choices.shape[1] - 1
    for ks in itertools.product(range(k_min, k_max + 1), repeat=choices.shape[0]):
        if sum(ks) != capacity:
            continue
        value = float(sum(choices[i, k - k_min] for i, k in enumerate(ks)))
        if value > best or (value == best and (best_k is None or ks < best_k)):
            best, best_k = value, ks
    if best_k is None:
        raise RuntimeError("synthetic capacity infeasible")
    return best, best_k


def synthetic_checks(dp) -> list[dict]:
    cases = [
        ("rising_weighted_margins", np.array([[0., 3., 1.], [2., 0., 2.]], np.float64), 4),
        ("all_zero", np.zeros((3, 3), np.float64), 5),
        ("ties", np.ones((3, 3), np.float64), 6),
        ("lower_bound", np.array([[1., 2.], [2., 1.]], np.float64), 2),
        ("upper_bound", np.array([[1., 2.], [2., 1.]], np.float64), 6),
    ]
    rows = []
    for name, optional, capacity in cases:
        choices = dp.choice_values_from_optional_marginals(optional)
        got, obj = dp.solve_exact_multiple_choice(choices, [capacity], k_min=1)
        expected_obj, expected_k = brute_force(choices, capacity, 1)
        passed = tuple(map(int, got[0])) == expected_k and float(obj[0]) == expected_obj
        rows.append({"case": name, "capacity": capacity, "allocation": list(map(int, got[0])),
                     "objective": float(obj[0]), "expected_allocation": list(expected_k),
                     "expected_objective": expected_obj, "pass": passed})
    if not all(x["pass"] for x in rows):
        raise RuntimeError(f"synthetic exact-DP check failed: {rows}")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", required=True)
    ap.add_argument("--p1-root", required=True)
    ap.add_argument("--p1a-root", required=True)
    args = ap.parse_args()
    root, p1, p1a = map(lambda x: Path(x).resolve(), (args.project_root, args.p1_root, args.p1a_root))
    for name in ("outputs", "qa", "figures", "scripts", "csv_sources"):
        (root / name).mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    append_log(root, "STAGE allocate_weighted_score START (DEV GT inaccessible by interface)")
    append_log(root, "DISCLOSURE read-only audit helper opened R0 group_manifest outside the requested R0 timing-only scope; no GT/model was read, no file was written, and P1B science does not use that file.")

    sys.path.insert(0, str(p1 / "scripts"))
    import dp_solver as dp  # type: ignore
    checks = synthetic_checks(dp)
    write_json(root / "qa" / "weighted_score_solver_checks.json", {"status": "PASS", "checks": checks})

    p1_config = json.loads((p1 / "run_config.json").read_text(encoding="utf-8"))
    release_root = Path(p1_config["release_root"]).resolve()
    identity_path = release_root / "CANDIDATE_ASSET_IDENTITY.json"
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    cand = load_dev_top100(release_root, identity)
    by_image = {str(i): d.reset_index(drop=True) for i, d in cand.groupby("image_id", sort=False)}

    pred_path = p1 / "dev_predictions.parquet"
    group = pq.read_table(pred_path, columns=["image_id", "group_id"]).to_pandas().drop_duplicates()
    group["image_id"] = group["image_id"].astype(str)
    if len(group) != 2000 or group["group_id"].nunique() != 50 or not (group.groupby("group_id").size() == 40).all():
        raise RuntimeError("frozen P1 DEV group map invariant failed")
    group_map = dict(zip(group["image_id"], group["group_id"].astype(int)))
    if set(group_map) != set(by_image):
        raise RuntimeError("candidate/group identity mismatch")

    weights_path = p1 / "models" / "class_weights.json"
    weight_obj = json.loads(weights_path.read_text(encoding="utf-8"))
    weights = np.asarray(weight_obj["weights"], np.float64)
    if weights.shape != (8,) or not np.isfinite(weights).all():
        raise RuntimeError("frozen FIT class weights invalid")

    rows: list[dict] = []
    for group_id in range(50):
        ids = sorted(group[group["group_id"] == group_id]["image_id"].tolist())
        margins = np.vstack([
            by_image[i].iloc[5:50]["score"].to_numpy(np.float64)
            * weights[by_image[i].iloc[5:50]["predicted_road8_class_id"].to_numpy(np.int16) - 1]
            for i in ids
        ])
        alloc, objectives = dp.solve_group_allocations(margins, BUDGETS)
        for bi, budget in enumerate(BUDGETS):
            for ii, image_id in enumerate(ids):
                k = int(alloc[bi, ii])
                prefix = by_image[image_id].iloc[:k]
                rows.append({
                    "image_id": image_id, "group_id": group_id, "method": METHOD, "seed": -1,
                    "variant_id": "S_CLASS", "base_signal": "RAW_DETECTOR_SCORE",
                    "class_weight_on": True, "multi_threshold_on": False,
                    "budget": int(budget), "K_i": k,
                    "selected_record_ids": prefix["candidate_record_id"].astype(str).tolist(),
                    "group_predicted_objective": float(objectives[bi]), "source": "NEW_P1B",
                })
    allocations = pd.DataFrame(rows)
    if len(allocations) != 10_000 or int(allocations["K_i"].sum()) != 230_000:
        raise RuntimeError("S_CLASS allocation size mismatch")
    if not allocations["K_i"].between(5, 50).all() or not (allocations["selected_record_ids"].map(len) == allocations["K_i"]).all():
        raise RuntimeError("S_CLASS prefix length invariant failed")
    budget_check = allocations.groupby(["budget", "group_id"], sort=True)["K_i"].sum().reset_index()
    if not np.array_equal(budget_check["K_i"].to_numpy(), budget_check["budget"].to_numpy() * 40):
        raise RuntimeError("S_CLASS group budget mismatch")
    for r in allocations.itertuples(index=False):
        expected = by_image[str(r.image_id)].iloc[:int(r.K_i)]["candidate_record_id"].astype(str).tolist()
        if list(r.selected_record_ids) != expected or int(r.group_id) != group_map[str(r.image_id)]:
            raise RuntimeError("S_CLASS candidate identity/prefix mismatch")

    # Reproduce the frozen raw-score DP through the same exact solver and compare every DEV K.
    p1_alloc_path = p1 / "cache" / "dev_allocations.parquet"
    old = pq.read_table(p1_alloc_path, filters=[("method", "=", "S_ADAPT")]).to_pandas()
    old["image_id"] = old["image_id"].astype(str)
    old_k = {(int(r.budget), str(r.image_id)): int(r.K_i) for r in old.itertuples(index=False)}
    replay_mismatch = 0
    for group_id in range(50):
        ids = sorted(group[group["group_id"] == group_id]["image_id"].tolist())
        raw = np.vstack([by_image[i].iloc[5:50]["score"].to_numpy(np.float64) for i in ids])
        got, _ = dp.solve_group_allocations(raw, BUDGETS)
        replay_mismatch += sum(int(int(got[bi, ii]) != old_k[(b, image_id)]) for bi, b in enumerate(BUDGETS) for ii, image_id in enumerate(ids))
    if replay_mismatch:
        raise RuntimeError(f"S_ADAPT exact-DP replay mismatch={replay_mismatch}")

    output = root / "weighted_score_allocations.parquet"
    pq.write_table(pa.Table.from_pandas(allocations, preserve_index=False), output, compression="zstd")
    config = {
        "task": "LC-ALLOC-P1B", "stage": "CLASS_WEIGHTED_SCORE_BASELINE_AND_EQUIVALENT_IMPLEMENTATION",
        "p1_root": str(p1), "p1a_root": str(p1a), "release_root": str(release_root),
        "candidate_asset_ids": p1_config["candidate_asset_ids"],
        "new_policy": {"name": METHOD, "slot_value": "frozen_FIT_class_weight[predicted_class] * original_detector_score",
                       "optional_ranks": [6, 50], "action": "original road8_rank prefix 1..K_i only"},
        "budgets": list(BUDGETS), "k_bounds": [5, 50], "groups": 50, "images_per_group": 40,
        "solver": "P1 exact float64 multiple-choice DP; fixed row order sorted(image_id); frozen lexicographic tie rule",
        "bootstrap": {"unit": "frozen 40-image group", "resamples": 5000, "seed": 530002,
                      "indices_path": str(p1 / "outputs" / "bootstrap_group_indices.npy")},
        "tolerances": {"atol": 1e-6, "rtol": 1e-5},
        "restricted_scientific_splits_accessed": 0,
        "dev_boundary": "previously observed DEV; allocation entrypoint has no GT path",
        "input_bindings": {
            "p1_run_config": {"path": str(p1 / "run_config.json"), "sha256": sha256_file(p1 / "run_config.json")},
            "p1_feature_schema": {"path": str(p1 / "feature_schema.json"), "sha256": sha256_file(p1 / "feature_schema.json")},
            "p1_dev_predictions": {"path": str(pred_path), "sha256": sha256_file(pred_path)},
            "p1_dev_allocations": {"path": str(p1_alloc_path), "sha256": sha256_file(p1_alloc_path)},
            "p1_class_weights": {"path": str(weights_path), "sha256": sha256_file(weights_path)},
            "p1_exact_dp": {"path": str(p1 / "scripts" / "dp_solver.py"), "sha256": sha256_file(p1 / "scripts" / "dp_solver.py")},
            "p1a_run_config": {"path": str(p1a / "run_config.json"), "sha256": sha256_file(p1a / "run_config.json")},
            "release_manifest": {"path": str(release_root / "RELEASE_MANIFEST.json"), "sha256": sha256_file(release_root / "RELEASE_MANIFEST.json")},
            "candidate_asset_identity": {"path": str(identity_path), "sha256": sha256_file(identity_path)},
        },
        "read_scope_deviation": "A delegated read-only audit opened R0 group_manifest; P1B implementation does not use it and no scientific data/model was read there.",
    }
    write_json(root / "run_config.json", config)
    commit = {
        "status": "S_CLASS_ADAPT_ALLOCATIONS_FROZEN_BEFORE_GT_EVALUATION",
        "path": str(output), "sha256": sha256_file(output), "rows": 10_000,
        "represented_selected_records": 230_000, "group_budget_mismatch": 0,
        "prefix_identity_mismatch": 0, "s_adapt_exact_dp_replay_k_mismatch": replay_mismatch,
        "dev_gt_path_available_to_this_entrypoint": False,
    }
    write_json(root / "outputs" / "weighted_score_allocation_commit.json", commit)
    append_log(root, f"S_CLASS_ALLOCATIONS_FROZEN sha={commit['sha256']} rows=10000 selected=230000")
    append_log(root, f"STAGE allocate_weighted_score COMPLETE elapsed_seconds={time.perf_counter()-t0:.6f}")


if __name__ == "__main__":
    main()
