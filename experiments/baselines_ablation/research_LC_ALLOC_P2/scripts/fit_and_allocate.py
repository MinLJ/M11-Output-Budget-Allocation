"""Fit FIT-only score calibrators and build DEV prefix allocations without GT access.

This entry point deliberately has no DEV-GT argument.  It freezes calibration
parameters, allocations, and concrete candidate selections before the separate
evaluation process is allowed to open DEV ground truth.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.isotonic import IsotonicRegression

sys.dont_write_bytecode = True

BUDGETS = (10, 15, 20, 30, 40)
METHODS = ("CAL_TEMP_ALLOC", "CAL_ISO_ALLOC")
TARGET_COLUMNS = tuple(f"y_iou_0_{x:02d}" for x in range(50, 100, 5))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_log(root: Path, msg: str) -> None:
    with (root / "runtime.log").open("a", encoding="utf-8") as f:
        f.write(msg.rstrip() + "\n")


def import_file(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_dev_top100(release_root: Path) -> pd.DataFrame:
    identity_path = release_root / "CANDIDATE_ASSET_IDENTITY.json"
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    columns = [
        "image_id", "candidate_record_id", "road8_rank", "score",
        "predicted_road8_class_id", "predicted_road8_class_name", "query_index",
        "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
        "bbox_cx", "bbox_cy", "bbox_w", "bbox_h",
    ]
    frames = []
    for rel in identity["assets"]["DEV"]["candidate_shards"]:
        frames.append(
            pq.read_table(
                release_root / rel,
                columns=columns,
                filters=[("road8_rank", "<=", 100)],
            ).to_pandas()
        )
    out = pd.concat(frames, ignore_index=True)
    out["image_id"] = out["image_id"].astype(str)
    out["candidate_record_id"] = out["candidate_record_id"].astype(str)
    out = out.sort_values(["image_id", "road8_rank"], kind="stable").reset_index(drop=True)
    if len(out) != 200_000 or out["image_id"].nunique() != 2000:
        raise RuntimeError("DEV Top100 count invariant failed")
    if out["candidate_record_id"].duplicated().any():
        raise RuntimeError("duplicate candidate_record_id")
    for image_id, d in out.groupby("image_id", sort=False):
        ranks = d["road8_rank"].astype(int).tolist()
        if ranks != list(range(1, 101)):
            raise RuntimeError(f"noncanonical Top100 ranks: {image_id}")
        scores = d["score"].to_numpy(np.float64)
        if not np.isfinite(scores).all() or np.any(np.diff(scores) > 0):
            raise RuntimeError(f"score ordering invariant failed: {image_id}")
    return out


def calibration_diagnostics(prob: np.ndarray, target: np.ndarray) -> dict:
    p = np.clip(np.asarray(prob, np.float64), 1e-12, 1 - 1e-12)
    y = np.asarray(target, np.float64)
    bce = float(np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p))))
    brier = float(np.mean((p - y) ** 2))
    bins = np.minimum((p * 10).astype(np.int64), 9)
    rows = []
    for b in range(10):
        mask = bins == b
        rows.append({
            "bin": b,
            "lower": b / 10,
            "upper": (b + 1) / 10,
            "rows": int(mask.sum()),
            "mean_probability": float(p[mask].mean()) if mask.any() else None,
            "mean_target": float(y[mask].mean()) if mask.any() else None,
        })
    return {"bce": bce, "brier": brier, "reliability_bins": rows}


def fit_calibrators(root: Path, p1_root: Path) -> tuple[float, IsotonicRegression, dict]:
    labels_path = p1_root / "train_labels.parquet"
    cols = ["role", "raw_score", *TARGET_COLUMNS]
    fit = pq.read_table(labels_path, columns=cols, filters=[("role", "=", "FIT")]).to_pandas()
    if len(fit) != 360_000:
        raise RuntimeError(f"FIT label row count {len(fit)} != 360000")
    scores = fit["raw_score"].to_numpy(np.float64)
    targets = fit[list(TARGET_COLUMNS)].to_numpy(np.float64)
    if not np.isfinite(scores).all() or not np.isfinite(targets).all():
        raise RuntimeError("nonfinite FIT calibration input")
    if not np.isin(targets, [0.0, 1.0]).all():
        raise RuntimeError("FIT targets are not binary")
    ybar = targets.mean(axis=1)
    clipped = np.clip(scores, 1e-6, 1 - 1e-6)
    logits = np.log(clipped / (1 - clipped))[:, None]

    p1_core = import_file(p1_root / "scripts" / "p1_core.py", "p2_p1_core")
    temperature, temp_fit = p1_core.fit_temperature(logits, targets)
    temp_prob = p1_core.sigmoid(logits[:, 0] / temperature)

    iso = IsotonicRegression(increasing=True, out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(scores, ybar)
    iso_prob = np.asarray(iso.predict(scores), np.float64)
    npz_path = root / "models" / "calibration_parameters.npz"
    np.savez_compressed(
        npz_path,
        isotonic_x_thresholds=np.asarray(iso.X_thresholds_, np.float64),
        isotonic_y_thresholds=np.asarray(iso.y_thresholds_, np.float64),
    )
    info = {
        "fit_role": "FIT",
        "fit_rows": int(len(fit)),
        "fit_images_implied": int(len(fit) // 45),
        "target": "mean of ten frozen maximum-matching prefix marginal labels",
        "target_columns": list(TARGET_COLUMNS),
        "target_positive_prevalence": float(targets.mean()),
        "score_min": float(scores.min()),
        "score_max": float(scores.max()),
        "score_clip": [1e-6, 1 - 1e-6],
        "temperature": float(temperature),
        "temperature_fit": temp_fit,
        "temperature_fit_diagnostics_on_ybar": calibration_diagnostics(temp_prob, ybar),
        "isotonic_fit_diagnostics_on_ybar": calibration_diagnostics(iso_prob, ybar),
        "isotonic_knot_count": int(len(iso.X_thresholds_)),
        "isotonic_unique_output_count": int(np.unique(iso.y_thresholds_).size),
        "isotonic_parameters_npz": str(npz_path),
        "isotonic_parameters_npz_sha256": sha256_file(npz_path),
        "class_weights_applied_to_baseline_slot_value": False,
        "restricted_split_access": 0,
    }
    write_json(root / "models" / "calibration_parameters.json", info)
    return float(temperature), iso, info


def synthetic_solver_checks(dp) -> list[dict]:
    cases = [
        ("all_zero", np.zeros((40, 45), np.float64)),
        ("strictly_decreasing", np.tile(np.linspace(1.0, 0.1, 45), (40, 1))),
        ("plateaus", np.tile(np.repeat(np.arange(9, dtype=np.float64)[::-1], 5), (40, 1))),
        ("nonconcave_guard", np.tile(np.linspace(0.1, 0.9, 45), (40, 1))),
    ]
    rows = []
    for name, margins in cases:
        alloc, objective = dp.solve_group_allocations(margins, BUDGETS)
        passed = (
            alloc.shape == (5, 40)
            and np.array_equal(alloc.sum(axis=1), np.asarray(BUDGETS) * 40)
            and np.isfinite(objective).all()
        )
        rows.append({"case": name, "pass": bool(passed), "objectives": objective.tolist()})
    if not all(x["pass"] for x in rows):
        raise RuntimeError("exact-DP synthetic checks failed")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", required=True)
    args = ap.parse_args()
    root = Path(args.project_root).resolve()
    config = json.loads((root / "run_config.json").read_text(encoding="utf-8"))
    p1_root = Path(config["inputs"]["p1_root"])
    release_root = Path(config["inputs"]["release_root"])
    t0 = time.perf_counter()
    append_log(root, "STAGE fit_and_allocate START (DEV GT is not accepted by this interface)")

    if sha256_file(p1_root / "train_labels.parquet") != config["inputs"]["train_labels_sha256"]:
        raise RuntimeError("train label freeze mismatch")
    if sha256_file(p1_root / "dev_predictions.parquet") != config["inputs"]["dev_predictions_sha256"]:
        raise RuntimeError("DEV prediction freeze mismatch")

    temperature, isotonic, calibration_info = fit_calibrators(root, p1_root)
    dp = import_file(p1_root / "scripts" / "dp_solver.py", "p2_dp_solver")
    checks = synthetic_solver_checks(dp)
    write_json(root / "qa" / "solver_checks.json", {"status": "PASS", "checks": checks})

    candidates = load_dev_top100(release_root)
    by_image = {str(k): d.reset_index(drop=True) for k, d in candidates.groupby("image_id", sort=False)}
    pred = pq.read_table(
        p1_root / "dev_predictions.parquet",
        columns=["image_id", "group_id", "road8_rank", "raw_score", "candidate_record_id"],
    ).to_pandas()
    pred["image_id"] = pred["image_id"].astype(str)
    pred["candidate_record_id"] = pred["candidate_record_id"].astype(str)
    if len(pred) != 90_000 or not pred["road8_rank"].between(6, 50).all():
        raise RuntimeError("frozen DEV optional-slot prediction table invalid")
    group_table = pred[["image_id", "group_id"]].drop_duplicates()
    if len(group_table) != 2000 or group_table["group_id"].nunique() != 50:
        raise RuntimeError("frozen group map invalid")
    if not (group_table.groupby("group_id").size() == 40).all():
        raise RuntimeError("frozen group size invariant failed")
    group_map = dict(zip(group_table["image_id"], group_table["group_id"].astype(int)))
    if set(group_map) != set(by_image):
        raise RuntimeError("candidate/group identity mismatch")

    # Verify the optional-slot score/identity table is exactly joined to release records.
    expected = candidates[candidates["road8_rank"].between(6, 50)][
        ["image_id", "road8_rank", "candidate_record_id", "score"]
    ].rename(columns={"score": "release_score"})
    joined = pred.merge(expected, on=["image_id", "road8_rank", "candidate_record_id"], how="outer", indicator=True)
    if len(joined) != 90_000 or not (joined["_merge"] == "both").all():
        raise RuntimeError("DEV optional-slot identity join failed")
    if not np.allclose(joined["raw_score"], joined["release_score"], atol=0.0, rtol=0.0):
        raise RuntimeError("DEV score mismatch between frozen prediction and release")

    old = pq.read_table(
        p1_root / "cache" / "dev_allocations.parquet",
        filters=[("method", "=", "S_ADAPT")],
    ).to_pandas()
    old["image_id"] = old["image_id"].astype(str)
    old_k = {(int(r.budget), str(r.image_id)): int(r.K_i) for r in old.itertuples(index=False)}

    allocation_rows: list[dict] = []
    selection_parts: list[pd.DataFrame] = []
    method_k: dict[str, dict[tuple[int, str], int]] = {m: {} for m in METHODS}
    iso_optional_values: list[np.ndarray] = []
    for group_id in range(50):
        ids = sorted(group_table[group_table["group_id"] == group_id]["image_id"].tolist())
        raw = np.vstack([by_image[i].iloc[5:50]["score"].to_numpy(np.float64) for i in ids])
        clipped = np.clip(raw, 1e-6, 1 - 1e-6)
        temp_values = 1.0 / (1.0 + np.exp(-np.log(clipped / (1 - clipped)) / temperature))
        iso_values = np.asarray(isotonic.predict(raw.reshape(-1)), np.float64).reshape(40, 45)
        iso_optional_values.append(iso_values.reshape(-1))
        for method, values in (("CAL_TEMP_ALLOC", temp_values), ("CAL_ISO_ALLOC", iso_values)):
            allocations, objectives = dp.solve_group_allocations(values, BUDGETS)
            for bi, budget in enumerate(BUDGETS):
                for ii, image_id in enumerate(ids):
                    k = int(allocations[bi, ii])
                    method_k[method][(budget, image_id)] = k
                    allocation_rows.append({
                        "image_id": image_id,
                        "group_id": group_id,
                        "method": method,
                        "seed": -1,
                        "budget": int(budget),
                        "K_i": k,
                        "group_predicted_objective": float(objectives[bi]),
                        "source": "NEW_P2",
                    })
                    prefix = by_image[image_id].iloc[:k].copy()
                    prefix.insert(1, "group_id", group_id)
                    prefix.insert(2, "method", method)
                    prefix.insert(3, "seed", -1)
                    prefix.insert(4, "budget", int(budget))
                    prefix.insert(5, "K_i", k)
                    prefix.insert(6, "source", "NEW_P2")
                    selection_parts.append(prefix)

    allocations = pd.DataFrame(allocation_rows).sort_values(
        ["method", "budget", "group_id", "image_id"], kind="stable"
    ).reset_index(drop=True)
    selections = pd.concat(selection_parts, ignore_index=True).sort_values(
        ["method", "budget", "group_id", "image_id", "road8_rank"], kind="stable"
    ).reset_index(drop=True)
    if len(allocations) != 20_000 or len(selections) != 460_000:
        raise RuntimeError(f"output row mismatch allocations={len(allocations)} selections={len(selections)}")
    if allocations["K_i"].sum() != len(selections):
        raise RuntimeError("selection count does not equal allocated count")
    if selections.duplicated(subset=["method", "budget", "candidate_record_id"]).any():
        raise RuntimeError("candidate duplicate within method/budget")
    budget_check = allocations.groupby(["method", "budget", "group_id"], as_index=False)["K_i"].sum()
    if not np.array_equal(budget_check["K_i"].to_numpy(), budget_check["budget"].to_numpy() * 40):
        raise RuntimeError("exact group budget invariant failed")
    prefix_summary = selections.groupby(
        ["method", "budget", "image_id"], as_index=False
    ).agg(
        selected_count=("candidate_record_id", "size"),
        distinct_records=("candidate_record_id", "nunique"),
        rank_min=("road8_rank", "min"),
        rank_max=("road8_rank", "max"),
        rank_sum=("road8_rank", "sum"),
    )
    prefix_summary = prefix_summary.merge(
        allocations[["method", "budget", "image_id", "K_i"]],
        on=["method", "budget", "image_id"], how="outer", validate="one_to_one",
    )
    expected_rank_sum = prefix_summary["K_i"] * (prefix_summary["K_i"] + 1) // 2
    prefix_bad = int((
        (prefix_summary["selected_count"] != prefix_summary["K_i"])
        | (prefix_summary["distinct_records"] != prefix_summary["K_i"])
        | (prefix_summary["rank_min"] != 1)
        | (prefix_summary["rank_max"] != prefix_summary["K_i"])
        | (prefix_summary["rank_sum"] != expected_rank_sum)
    ).sum())
    if prefix_bad:
        raise RuntimeError(f"prefix selection invariant mismatch={prefix_bad}")

    temp_k_mismatch = sum(
        int(k != old_k[(budget, image_id)])
        for (budget, image_id), k in method_k["CAL_TEMP_ALLOC"].items()
    )
    iso_k_mismatch = sum(
        int(k != old_k[(budget, image_id)])
        for (budget, image_id), k in method_k["CAL_ISO_ALLOC"].items()
    )
    temp_selection_mismatch = sum(
        abs(method_k["CAL_TEMP_ALLOC"][(b, i)] - old_k[(b, i)])
        for b in BUDGETS for i in by_image
    )
    iso_selection_mismatch = sum(
        abs(method_k["CAL_ISO_ALLOC"][(b, i)] - old_k[(b, i)])
        for b in BUDGETS for i in by_image
    )
    iso_values_all = np.concatenate(iso_optional_values)
    iso_unique = int(np.unique(iso_values_all).size)
    iso_plateau_rows = int(len(iso_values_all) - iso_unique)

    allocation_path = root / "outputs" / "calibration_allocations.parquet"
    selection_path = root / "outputs" / "calibration_selections.parquet"
    pq.write_table(pa.Table.from_pandas(allocations, preserve_index=False), allocation_path, compression="zstd")
    pq.write_table(pa.Table.from_pandas(selections, preserve_index=False), selection_path, compression="zstd")
    allocation_sha = sha256_file(allocation_path)
    selection_sha = sha256_file(selection_path)
    commit = {
        "status": "CALIBRATION_ALLOCATIONS_AND_SELECTIONS_FROZEN_BEFORE_DEV_GT",
        "allocations_path": str(allocation_path),
        "allocations_sha256": allocation_sha,
        "allocations_rows": int(len(allocations)),
        "selections_path": str(selection_path),
        "selections_sha256": selection_sha,
        "selections_rows": int(len(selections)),
        "represented_selected_records_per_method": 230000,
        "group_budget_mismatch": 0,
        "prefix_identity_mismatch": 0,
        "temperature_k_mismatch_vs_S_ADAPT": int(temp_k_mismatch),
        "temperature_selection_symmetric_difference_records_vs_S_ADAPT": int(temp_selection_mismatch * 2),
        "isotonic_k_mismatch_vs_S_ADAPT": int(iso_k_mismatch),
        "isotonic_selection_symmetric_difference_records_vs_S_ADAPT": int(iso_selection_mismatch * 2),
        "isotonic_dev_optional_unique_values": iso_unique,
        "isotonic_dev_optional_plateau_excess_rows": iso_plateau_rows,
        "dev_gt_path_available_to_this_entrypoint": False,
        "calibration_parameters_sha256": sha256_file(root / "models" / "calibration_parameters.json"),
        "calibration_parameters_npz_sha256": calibration_info["isotonic_parameters_npz_sha256"],
    }
    write_json(root / "outputs" / "calibration_selection_commit.json", commit)
    write_json(root / "qa" / "allocation_checks.json", {
        "status": "PASS",
        "allocation_rows": int(len(allocations)),
        "selection_rows": int(len(selections)),
        "budget_mismatch": 0,
        "prefix_mismatch": 0,
        "candidate_identity_mismatch": 0,
        "score_identity_mismatch": 0,
        "temperature_vs_s_adapt_k_mismatch": int(temp_k_mismatch),
        "isotonic_vs_s_adapt_k_mismatch": int(iso_k_mismatch),
    })
    append_log(root, f"CALIBRATION_FIT temperature={temperature:.12g} isotonic_knots={calibration_info['isotonic_knot_count']}")
    append_log(root, f"SELECTIONS_FROZEN allocations_sha={allocation_sha} selections_sha={selection_sha}")
    append_log(root, f"STRUCTURAL_CHECK temp_k_mismatch_vs_S_ADAPT={temp_k_mismatch} iso_k_mismatch_vs_S_ADAPT={iso_k_mismatch}")
    append_log(root, f"STAGE fit_and_allocate COMPLETE elapsed_seconds={time.perf_counter()-t0:.6f}")


if __name__ == "__main__":
    main()
