# -*- coding: utf-8 -*-
"""07 — per-slot predicted utility for the frozen DEV selections.

Guideline §05: for every seed / image / slot k=6..50 persist
  logits (10), temperature-scaled probabilities (10),
  delta_hat  = w_class * mean_tau(prob),
  Uhat       = cumulative curve anchored at Uhat_i(5) = 0.

Slots k=1..5 are the mandatory constant prefix and are NOT fabricated as model
predictions — the curve simply starts at 0 at k=5.

This is a REPLAY of the already-trained frozen assets (no re-training, no
re-solving of allocations).  Provenance (asset_id, model SHA, weight id,
generation time) is recorded in the output file and in a companion JSON.
"""
from __future__ import annotations
import os

import argparse
import json
import sys
import time
from datetime import datetime, timezone
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

from lc_alloc.data.io import read_candidates, read_native  # noqa: E402
from lc_alloc.data.schema import NativeTable  # noqa: E402
from lc_alloc.features.r18 import transform_features  # noqa: E402
from lc_alloc.models.mlp import MarginalMLP  # noqa: E402
from lc_alloc.utils import read_json, sha256_file  # noqa: E402

from m11_yolo.config import SEEDS, YOLO_NATIVE_SCHEMA_ID  # noqa: E402
from m11_yolo.features import build_image_features  # noqa: E402

K_MIN, K_MAX = 5, 50


def _index_native(native: NativeTable):
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
    ap.add_argument("--dev-candidates", required=True)
    ap.add_argument("--dev-native", required=True)
    ap.add_argument("--assets", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    asset_dir = Path(args.assets)

    pca = joblib.load(asset_dir / "pca32.joblib")
    scaler_bundle = joblib.load(asset_dir / "scaler.joblib")
    weights = np.asarray(read_json(asset_dir / "class_weights.json")["weights"], dtype=np.float64)
    asset_manifest = read_json(asset_dir / "asset.json")

    candidates = read_candidates(args.dev_candidates)
    candidates["image_id"] = candidates["image_id"].astype(str)
    native = read_native(args.dev_native, expected_schema_id=YOLO_NATIVE_SCHEMA_ID)
    frames = {iid: g for iid, g in candidates.groupby("image_id", sort=False)}
    image_ids = sorted(frames)
    ndx = _index_native(native)
    print(f"DEV images={len(image_ids)}", flush=True)

    # features + slot class per image (positions 6..50 -> ranks index 5..49)
    scaled, slot_cls = {}, {}
    for iid in image_ids:
        raw = build_image_features(frames[iid], _native_subset(ndx, iid), pca)
        scaled[iid] = transform_features(raw, scaler_bundle)
        slot_cls[iid] = frames[iid].sort_values("rank", kind="mergesort").iloc[K_MIN:K_MAX]["predicted_road8_class_id"].to_numpy(np.int64)

    frames_out = []
    t0 = time.time()
    for seed in SEEDS:
        snap = torch.load(asset_dir / "models" / f"marginal_mlp_seed_{seed}.pt", map_location="cpu", weights_only=True)
        model = MarginalMLP(int(snap["input_dim"]))
        model.load_state_dict(snap["state_dict"], strict=True)
        model.eval().to(torch.device(args.device))
        for p in model.parameters():
            p.requires_grad_(False)
        temp = float(read_json(asset_dir / "models" / f"temperature_seed_{seed}.json")["temperature"])

        rows = []
        with torch.inference_mode():
            for iid in image_ids:
                x = torch.from_numpy(scaled[iid]).to(args.device)
                logits = model(x).cpu().numpy().astype(np.float64)
                probs = 1.0 / (1.0 + np.exp(-logits / temp))
                cls = slot_cls[iid]
                w_cls = weights[cls - 1]
                delta = probs.mean(axis=1) * w_cls
                uhat = np.concatenate([[0.0], np.cumsum(delta)[:-1]])  # Uhat(5)=0 anchor
                k_values = np.arange(K_MIN + 1, K_MAX + 1)
                n = logits.shape[0]
                rows.append(pd.DataFrame({
                    "image_id": iid,
                    "k": k_values[:n],
                    "predicted_class_id": cls[:n],
                    "weight_id": "yolo_class_weights_fit_v1",
                    "model_id": f"marginal_mlp_seed_{seed}",
                    "temperature": temp,
                    **{f"logit_t{t}": logits[:n, t] for t in range(logits.shape[1])},
                    **{f"prob_t{t}": probs[:n, t] for t in range(probs.shape[1])},
                    "delta_hat": delta[:n],
                    "uhat": uhat[:n],
                }))
        df = pd.concat(rows, ignore_index=True)
        df.insert(0, "seed", seed)
        frames_out.append(df)
        print(f"  seed {seed}: slots={len(df)} ({time.time()-t0:.1f}s)", flush=True)

    util = pd.concat(frames_out, ignore_index=True)
    util.to_parquet(out / "predicted_utility.parquet", index=False)

    meta = {
        "persisted": True,
        "note": "replayed from frozen assets; k=1..5 are the mandatory constant prefix and carry NO model prediction (curve anchors at Uhat(5)=0)",
        "asset_id": asset_manifest.get("asset_id"),
        "model_sha256": {k: v.get("sha256") for k, v in asset_manifest.get("models", {}).items()},
        "weight_id": "yolo_class_weights_fit_v1",
        "class_weights_sha256": sha256_file(asset_dir / "class_weights.json"),
        "rows": int(len(util)),
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (out / "predicted_utility_manifest.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "PASS", **{k: meta[k] for k in ("asset_id", "rows", "generated_utc")}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
