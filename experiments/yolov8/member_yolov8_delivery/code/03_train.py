# -*- coding: utf-8 -*-
"""03 — fit YOLO PCA/scaler, train 3 independent seed MLPs, calibrate, save asset.

Mirrors the package training smoke but full-scale and YOLO-featured.  Reuses the
shared M11 modules: prefix labels, PCA/scaler fitting, train_allocator,
fit_temperature, class-weight formula.

Run after 01_export + 02_prepare.  Outputs a self-contained asset dir loadable
by ``lc_alloc.models.assets.load_allocator_assets``.
"""
from __future__ import annotations
import os

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

BASE = Path(__file__).resolve().parent
M11 = Path(os.environ.get("M11_EXPERIMENT_ROOT", "external_assets")) / "material from memberA" / "LC_ALLOC_M11_HANDOFF_v1" / "handoff_LC_ALLOC_M11_v1"
if str(M11) not in sys.path:
    sys.path.insert(0, str(M11))
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from lc_alloc.calibration.temperature import fit_temperature  # noqa: E402
from lc_alloc.data.io import read_candidates, read_native  # noqa: E402
from lc_alloc.data.schema import NativeTable  # noqa: E402
from lc_alloc.features._executed_p1_core import compute_class_weights  # noqa: E402
from lc_alloc.features.preprocessing import fit_pca, fit_scaler  # noqa: E402
from lc_alloc.features.r18 import transform_features  # noqa: E402
from lc_alloc.labels.prefix import labels_for_image, load_normalized_gt  # noqa: E402
from lc_alloc.training.core import TrainRecipe, train_allocator  # noqa: E402
from lc_alloc.utils import sha256_file, write_json  # noqa: E402

from m11_yolo.config import SEEDS, YOLO_NATIVE_SCHEMA_ID  # noqa: E402
from m11_yolo.features import build_image_features  # noqa: E402


def _index_native(native: NativeTable):
    """Precompute per-image native row positions to avoid repeated full scans."""
    iid = np.asarray(native.image_ids).astype(str)
    sid = np.asarray(native.source_ids).astype(str)
    vec = np.asarray(native.native_vectors)
    sig = np.asarray(native.class_signals)
    groups = pd.DataFrame({"iid": iid}).groupby("iid", sort=False).indices
    return iid, sid, vec, sig, groups, native.native_schema_id


def _native_subset(ndx, image_id: str) -> NativeTable:
    iid, sid, vec, sig, groups, schema = ndx
    idx = groups[image_id]
    return NativeTable(iid[idx], sid[idx], vec[idx], sig[idx], schema)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-candidates", required=True)
    ap.add_argument("--train-native", required=True)
    ap.add_argument("--train-gt", required=True)
    ap.add_argument("--roles", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-images", type=int, default=0, help="cap TRAIN images for a smoke run")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = args.device

    candidates = read_candidates(args.train_candidates)
    candidates["image_id"] = candidates["image_id"].astype(str)
    native = read_native(args.train_native, expected_schema_id=YOLO_NATIVE_SCHEMA_ID)
    roles = pd.read_csv(args.roles)
    roles["image_id"] = roles["image_id"].astype(str)
    gt_map = load_normalized_gt(args.train_gt)

    if args.max_images:
        keep = sorted(set(candidates["image_id"]))[: args.max_images]
        candidates = candidates[candidates["image_id"].isin(keep)]
        roles = roles[roles["image_id"].isin(keep)]

    # Precompute candidate frames and native index once (avoids O(N^2) scans).
    frames = {iid: g for iid, g in candidates.groupby("image_id", sort=False)}
    image_ids = sorted(frames)
    role_map = dict(zip(roles.image_id, roles.role))
    fit_ids = {iid for iid in image_ids if role_map.get(iid) == "FIT"}
    ndx = _index_native(native)
    n_iid, n_sid, n_vec, _, _, _ = ndx

    # --- fit PCA on FIT native vectors ---
    fit_candidate_keys = {
        (iid, str(r.source_id))
        for iid in fit_ids
        for r in frames[iid].itertuples(index=False)
    }
    native_mask = np.asarray(
        [(iid, sid) in fit_candidate_keys for iid, sid in zip(n_iid, n_sid)],
        dtype=bool,
    )
    pca = fit_pca(
        n_iid[native_mask],
        n_sid[native_mask],
        n_vec[native_mask],
        namespace="YOLO_BDD_PCA|",
    )

    # --- build raw features (need pca) + labels ---
    raw_by_id, label_by_id = {}, {}
    for image_id in image_ids:
        frame = frames[image_id]
        raw_by_id[image_id] = build_image_features(frame, _native_subset(ndx, image_id), pca)
        label_by_id[image_id] = labels_for_image(frame, gt_map[image_id])

    # --- fit scaler on FIT raw features ---
    scaler_bundle = fit_scaler(np.vstack([raw_by_id[iid] for iid in sorted(fit_ids)]))

    # --- scale all + assemble role-split matrices ---
    xs, ys = {"FIT": [], "EARLY_STOP": [], "CALIBRATION": []}, {"FIT": [], "EARLY_STOP": [], "CALIBRATION": []}
    for image_id in image_ids:
        role = role_map[image_id]
        xs[role].append(transform_features(raw_by_id[image_id], scaler_bundle))
        ys[role].append(label_by_id[image_id].astype(np.float32))
    for role in xs:
        xs[role] = np.vstack(xs[role]) if xs[role] else np.empty((0, 90), dtype=np.float32)
        ys[role] = np.vstack(ys[role]) if ys[role] else np.empty((0, 10), dtype=np.float32)

    counts, class_weights = compute_class_weights(gt_map, fit_ids)

    # --- train 3 independent seeds ---
    models_dir = out / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    recipe = TrainRecipe(max_epochs=30, patience=5, batch_size=4096)
    seeds_meta = {}
    seed_summary = {}
    for seed in SEEDS:
        model, snapshot, history = train_allocator(
            xs["FIT"], ys["FIT"], xs["EARLY_STOP"], ys["EARLY_STOP"],
            seed=seed, recipe=recipe, device=device,
        )
        torch.save(snapshot, models_dir / f"marginal_mlp_seed_{seed}.pt")
        pd.DataFrame(history).to_csv(out / f"training_history_seed_{seed}.csv", index=False)
        with torch.inference_mode():
            logits = model(torch.from_numpy(xs["CALIBRATION"]).to(device)).cpu().numpy().astype(np.float64)
        temperature, tsum = fit_temperature(logits, ys["CALIBRATION"].astype(np.float64))
        write_json(models_dir / f"temperature_seed_{seed}.json", {"seed": int(seed), **tsum})
        seeds_meta[str(seed)] = {
            "path": f"models/marginal_mlp_seed_{seed}.pt",
            "sha256": sha256_file(models_dir / f"marginal_mlp_seed_{seed}.pt"),
            "temperature": float(temperature),
        }
        seed_summary[str(seed)] = {"best_epoch": snapshot["best_epoch"], "early_stop_bce": snapshot["early_stop_natural_bce"]}
        print(f"seed {seed}: best_epoch={snapshot['best_epoch']} "
              f"early_stop_bce={snapshot['early_stop_natural_bce']:.6f} temperature={temperature:.6f}", flush=True)

    # --- save PCA / scaler / class weights + asset manifest ---
    joblib.dump(pca, out / "pca32.joblib")
    joblib.dump(scaler_bundle, out / "scaler.joblib")
    write_json(out / "class_weights.json", {"weights": class_weights.tolist(), "counts": counts.tolist()})

    asset = {
        "asset_id": "yolo" + sha256_file(out / "class_weights.json")[:16],
        "asset_name": "YOLOv8n BDD100K M11 cross-detector (3 independent seeds)",
        "class_signal_dimension": 8,
        "feature_dimension": 90,
        "files": {
            "class_weights": {"path": "class_weights.json", "sha256": sha256_file(out / "class_weights.json")},
            "pca": {"path": "pca32.joblib", "sha256": sha256_file(out / "pca32.joblib")},
            "scaler": {"path": "scaler.joblib", "sha256": sha256_file(out / "scaler.joblib")},
        },
        "models": seeds_meta,
        "models_are_independent_not_ensemble": True,
        "native_schema_id": YOLO_NATIVE_SCHEMA_ID,
        "native_vector_dimension": 84,
        "output_dimension": 10,
        "reference_detector_only": False,
        "scientific_source": "YOLOv8n BDD100K cross-detector M11 validation",
    }
    write_json(out / "asset.json", asset)

    result = {
        "status": "PASS",
        "images_by_role": {r: int(len(xs[r]) // 45) for r in xs if len(xs[r])},
        "fit_native_rows": int(native_mask.sum()),
        "class_weights": class_weights.tolist(),
        "seeds": seed_summary,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
