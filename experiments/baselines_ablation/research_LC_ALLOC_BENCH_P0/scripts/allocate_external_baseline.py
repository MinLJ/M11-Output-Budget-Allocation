"""Generate frozen DEV selections for the two external-calibration adapters.

This entry point intentionally has no GT argument and never opens DEV GT.  It
maps every frozen Top100 candidate through the FIT-only class-specific Platt
calibrator, solves the frozen exact prefix-budget problem on R0 40-image
groups, writes selections, and immediately records their byte SHA256.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.dont_write_bytecode = True

from common import BUDGETS, P1, ROAD8, ROOT, append_log, load_candidate_frame, sha256_file, sigmoid, solve_group, write_json, write_parquet


PARAM_PATH = ROOT / "calibrator_parameters.json"
GROUP_PATH = ROOT / "outputs" / "group_manifest.parquet"
VALUES_PATH = ROOT / "outputs" / "external_baseline_values.parquet"
ALLOC_PATH = ROOT / "outputs" / "external_baseline_allocations.parquet"
QA_PATH = ROOT / "qa" / "external_baseline_allocation_validation.json"
FREEZE_PATH = ROOT / "qa" / "external_baseline_selection_freeze.json"
WEIGHT_PATH = P1 / "models" / "class_weights.json"
EPS = np.finfo(np.float64).eps


def calibrated_probability(scores: np.ndarray, effective_scale: float, bias: float) -> np.ndarray:
    s = np.clip(np.asarray(scores, dtype=np.float64), EPS, 1.0 - EPS)
    z = np.log(s) - np.log1p(-s)
    return sigmoid(float(effective_scale) * z + float(bias))


def main() -> None:
    started = time.time()
    payload = json.loads(PARAM_PATH.read_text(encoding="utf-8"))
    classes = payload.get("classes")
    if not isinstance(classes, list) or len(classes) != 8:
        raise RuntimeError("calibrator_parameters.json must expose eight top-level classes")
    specs = {int(x["class_id"]): x for x in classes}
    if set(specs) != set(range(1, 9)):
        raise RuntimeError("calibrator class identity mismatch")
    for cid, spec in specs.items():
        if str(spec.get("class_name")) != ROAD8[cid - 1]:
            raise RuntimeError(f"calibrator class name mismatch: {cid}")
        if not np.isfinite([spec["effective_scale"], spec["bias"]]).all():
            raise RuntimeError(f"nonfinite calibrator parameters: class {cid}")

    weights_payload = json.loads(WEIGHT_PATH.read_text(encoding="utf-8"))
    weights = np.asarray(weights_payload["weights"], dtype=np.float64)
    if weights.shape != (8,) or not np.isfinite(weights).all():
        raise RuntimeError("frozen class weight vector invalid")

    dev = load_candidate_frame("DEV", max_rank=100)
    cls = dev["predicted_road8_class_id"].to_numpy(np.int16)
    raw = dev["score"].to_numpy(np.float64)
    pcal = np.empty(len(dev), dtype=np.float64)
    pre = np.empty(len(dev), dtype=np.float64)
    post = np.empty(len(dev), dtype=np.float64)
    for cid in range(1, 9):
        mask = cls == cid
        spec = specs[cid]
        pcal[mask] = calibrated_probability(raw[mask], spec["effective_scale"], spec["bias"])
        pre[mask] = np.nan if spec.get("pre_threshold_u") is None else float(spec["pre_threshold_u"])
        post[mask] = np.nan if spec.get("post_threshold_v") is None else float(spec["post_threshold_v"])
    if not np.isfinite(pcal).all() or np.any((pcal < 0.0) | (pcal > 1.0)):
        raise RuntimeError("invalid calibrated probabilities")
    dev = dev.copy()
    dev["calibrated_probability"] = pcal
    dev["fit_pre_threshold_u"] = pre
    dev["fit_post_threshold_v"] = post
    dev["below_fit_pre_threshold"] = np.isfinite(pre) & (raw < pre)
    dev["below_fit_post_threshold"] = np.isfinite(post) & (pcal < post)
    dev["ps_prefix_value"] = pcal
    dev["ps_class_prefix_value"] = pcal * weights[cls - 1]
    value_cols = [
        "image_id",
        "candidate_record_id",
        "road8_rank",
        "predicted_road8_class_id",
        "predicted_road8_class_name",
        "score",
        "calibrated_probability",
        "fit_pre_threshold_u",
        "fit_post_threshold_v",
        "below_fit_pre_threshold",
        "below_fit_post_threshold",
        "ps_prefix_value",
        "ps_class_prefix_value",
    ]
    write_parquet(dev[value_cols], VALUES_PATH)

    groups = pq.read_table(GROUP_PATH).to_pandas()
    groups["image_id"] = groups["image_id"].astype(str)
    r0 = groups.loc[(groups["permutation"] == "R0") & (groups["group_size"].astype(int) == 40)].copy()
    if len(r0) != 2000 or r0["image_id"].nunique() != 2000 or r0["group_id"].nunique() != 50:
        raise RuntimeError("R0 n40 group manifest invalid")
    if set(r0["image_id"]) != set(dev["image_id"]):
        raise RuntimeError("DEV candidate/group identity mismatch")

    # Per-image arrays are frozen in raw road8_rank order.
    by_image: dict[str, dict] = {}
    for image_id, d in dev.groupby("image_id", sort=False):
        ranks = d["road8_rank"].astype(int).tolist()
        if ranks != list(range(1, 101)):
            raise RuntimeError(f"Top100 rank invariant failed: {image_id}")
        by_image[str(image_id)] = {
            "candidate_ids": d["candidate_record_id"].astype(str).tolist(),
            "below_pre": d["below_fit_pre_threshold"].to_numpy(bool),
            "below_post": d["below_fit_post_threshold"].to_numpy(bool),
            "PS_PREFIX": d["ps_prefix_value"].to_numpy(np.float64),
            "PS_CLASS_PREFIX": d["ps_class_prefix_value"].to_numpy(np.float64),
        }

    rows: list[dict] = []
    for group_id, gm in r0.groupby("group_id", sort=True):
        image_ids = sorted(gm["image_id"].astype(str).tolist())
        if len(image_ids) != 40:
            raise RuntimeError(f"R0 group size invalid: {group_id}")
        for policy in ("PS_PREFIX", "PS_CLASS_PREFIX"):
            # Optional slots are precisely ranks 6..50.  Ranks 1..5 are the
            # common omitted constant but remain in every selected prefix.
            marginals = np.stack([by_image[x][policy][5:50] for x in image_ids], axis=0).astype(np.float64, copy=False)
            allocations, objectives = solve_group(marginals, BUDGETS)
            for bi, budget in enumerate(BUDGETS):
                ks = allocations[bi]
                if int(ks.sum()) != 40 * int(budget):
                    raise RuntimeError(f"exact group budget failed: {group_id}/{policy}/{budget}")
                for ii, image_id in enumerate(image_ids):
                    k = int(ks[ii])
                    ids = by_image[image_id]["candidate_ids"][:k]
                    below_pre = int(by_image[image_id]["below_pre"][:k].sum())
                    below_post = int(by_image[image_id]["below_post"][:k].sum())
                    rows.append(
                        {
                            "experiment": "EXTERNAL_CALIBRATION_BASELINE",
                            "permutation": "R0",
                            "group_size": 40,
                            "group_id": int(group_id),
                            "method": policy,
                            "seed": -1,
                            "budget": int(budget),
                            "image_id": image_id,
                            "K_i": k,
                            "selected_record_ids": ids,
                            "selected_below_fit_pre_threshold_count": below_pre,
                            "selected_below_fit_pre_threshold_fraction": float(below_pre / k),
                            "selected_below_fit_post_threshold_count": below_post,
                            "selected_below_fit_post_threshold_fraction": float(below_post / k),
                            "group_predicted_objective": float(objectives[bi]),
                            "source": "NEW_BENCH_P0",
                        }
                    )
    allocations_frame = pd.DataFrame(rows)
    write_parquet(allocations_frame, ALLOC_PATH)

    # Validate the committed output without any evaluation data.
    expected_rows = 2 * 5 * 2000
    total_selected = allocations_frame["K_i"].sum()
    expected_selected = 2 * 2000 * sum(BUDGETS)
    group_budget = allocations_frame.groupby(["method", "budget", "group_id"], sort=False)["K_i"].sum().reset_index()
    group_budget_ok = bool((group_budget["K_i"] == group_budget["budget"] * 40).all())
    per_policy_total = allocations_frame.groupby("method", sort=False)["K_i"].sum().astype(int).to_dict()
    prefix_ok = True
    duplicate_ok = True
    for row in allocations_frame.itertuples(index=False):
        ids = list(row.selected_record_ids)
        prefix_ok &= ids == by_image[str(row.image_id)]["candidate_ids"][: int(row.K_i)]
        duplicate_ok &= len(ids) == len(set(ids))
    qa = {
        "all_checks_pass": bool(
            len(allocations_frame) == expected_rows
            and int(total_selected) == expected_selected
            and group_budget_ok
            and prefix_ok
            and duplicate_ok
            and allocations_frame["K_i"].between(5, 50).all()
        ),
        "dev_gt_open_count": 0,
        "restricted_split_access_count": 0,
        "dev_image_count": int(dev["image_id"].nunique()),
        "candidate_value_row_count": int(len(dev)),
        "allocation_row_count": int(len(allocations_frame)),
        "selected_record_count": int(total_selected),
        "selected_record_count_expected": int(expected_selected),
        "selected_record_count_by_policy": per_policy_total,
        "exact_group_budget": group_budget_ok,
        "prefix_selection": bool(prefix_ok),
        "within_selection_candidate_id_unique": bool(duplicate_ok),
        "below_fit_pre_threshold_all_count": int(dev["below_fit_pre_threshold"].sum()),
        "below_fit_pre_threshold_all_fraction": float(dev["below_fit_pre_threshold"].mean()),
        "below_fit_post_threshold_all_count": int(dev["below_fit_post_threshold"].sum()),
        "below_fit_post_threshold_all_fraction": float(dev["below_fit_post_threshold"].mean()),
        "elapsed_seconds": time.time() - started,
    }
    write_json(QA_PATH, qa)
    if not qa["all_checks_pass"]:
        raise RuntimeError(f"external baseline allocation QA failed: {qa}")

    # This is the explicit prediction/allocation commit point.  A separate
    # evaluator may open DEV GT only after verifying this record.
    freeze = {
        "status": "EXTERNAL_BASELINE_SELECTIONS_FROZEN_BEFORE_DEV_GT",
        "dev_gt_open_count_at_commit": 0,
        "allocation_path": str(ALLOC_PATH),
        "allocation_sha256": sha256_file(ALLOC_PATH),
        "values_path": str(VALUES_PATH),
        "values_sha256": sha256_file(VALUES_PATH),
        "calibrator_parameters_path": str(PARAM_PATH),
        "calibrator_parameters_sha256": sha256_file(PARAM_PATH),
        "group_manifest_path": str(GROUP_PATH),
        "group_manifest_sha256": sha256_file(GROUP_PATH),
        "allocation_script_path": str(Path(__file__).resolve()),
        "allocation_script_sha256": sha256_file(Path(__file__).resolve()),
        "allocation_rows": int(len(allocations_frame)),
        "selected_records": int(total_selected),
        "policies": ["PS_PREFIX", "PS_CLASS_PREFIX"],
        "budgets": list(BUDGETS),
    }
    write_json(FREEZE_PATH, freeze)
    append_log(
        "B_EXTERNAL_SELECTION_FREEZE_COMPLETE "
        f"alloc_sha={freeze['allocation_sha256']} values_sha={freeze['values_sha256']} "
        f"rows={len(allocations_frame)} selected={int(total_selected)} dev_gt_open_count=0"
    )
    print(json.dumps({**qa, "selection_freeze_sha256": sha256_file(FREEZE_PATH), "allocation_sha256": freeze["allocation_sha256"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
