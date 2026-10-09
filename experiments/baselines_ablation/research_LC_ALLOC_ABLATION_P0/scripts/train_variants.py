from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from common import MarginalMLP, SEEDS, VARIANTS, add_p1_scripts, append_log, predict_probabilities, sha256_file, verify_input_binding, write_json, write_parquet  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", required=True)
    ap.add_argument("--p1-root", required=True)
    args = ap.parse_args()
    root = Path(args.project_root).resolve()
    p1 = Path(args.p1_root).resolve()
    start_all = time.perf_counter()
    append_log(root, "STAGE train_variants START")
    verify_input_binding(root)
    add_p1_scripts(p1)
    from p1_core import fit_temperature, reliability_rows  # noqa: E402

    masks = json.loads((root / "feature_masks.json").read_text(encoding="utf-8"))
    x_mem = np.load(p1 / "cache" / "train_X_scaled.npy", mmap_mode="r")
    y_marginal = np.load(p1 / "cache" / "train_y.npy", mmap_mode="r")
    y_match = np.load(root / "cache" / "train_y_matchability.npy", mmap_mode="r")
    role = np.load(p1 / "cache" / "train_role.npy", mmap_mode="r")
    fit_mask, stop_mask, cal_mask = role == 0, role == 1, role == 2
    if (int(fit_mask.sum()), int(stop_mask.sum()), int(cal_mask.sum())) != (360000, 45000, 45000):
        raise RuntimeError("TRAIN role row counts invalid")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.synchronize()
    histories = []
    reliability = []
    all_summaries = []
    loss_fn = nn.BCEWithLogitsLoss(reduction="mean")

    for variant in VARIANTS:
        variant_start = time.perf_counter()
        append_log(root, f"TRAIN_VARIANT START variant={variant}")
        indices = np.asarray(masks[variant]["indices"], dtype=np.int64)
        y_source = y_match if variant == "MATCHABILITY_TARGET" else y_marginal
        x_fit_np = np.asarray(x_mem[fit_mask], dtype=np.float32).copy()
        x_stop_np = np.asarray(x_mem[stop_mask], dtype=np.float32).copy()
        x_cal_np = np.asarray(x_mem[cal_mask], dtype=np.float32).copy()
        if len(indices):
            x_fit_np[:, indices] = 0.0
            x_stop_np[:, indices] = 0.0
            x_cal_np[:, indices] = 0.0
        if len(indices) and (not np.all(x_fit_np[:, indices] == 0.0) or not np.all(x_stop_np[:, indices] == 0.0) or not np.all(x_cal_np[:, indices] == 0.0)):
            raise RuntimeError(f"mask application failed {variant}")
        x_fit = torch.from_numpy(x_fit_np).to(device)
        x_stop = torch.from_numpy(x_stop_np).to(device)
        x_cal_np_for_predict = x_cal_np
        y_fit = torch.from_numpy(np.asarray(y_source[fit_mask], dtype=np.float32).copy()).to(device)
        y_stop = torch.from_numpy(np.asarray(y_source[stop_mask], dtype=np.float32).copy()).to(device)
        y_cal = np.asarray(y_source[cal_mask], dtype=np.float64).copy()
        variant_dir = root / "models" / variant
        variant_dir.mkdir(parents=True, exist_ok=True)
        summaries = []

        for seed in SEEDS:
            seed_start = time.perf_counter()
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            model = MarginalMLP(90).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            generator = torch.Generator(device=device.type).manual_seed(seed)
            best_loss = float("inf")
            best_epoch = 0
            best_state = None
            stale = 0
            epoch = 0
            for epoch in range(1, 31):
                model.train()
                order = torch.randperm(len(x_fit), generator=generator, device=device)
                fit_sum, fit_rows = 0.0, 0
                for batch_start in range(0, len(x_fit), 4096):
                    ix = order[batch_start:batch_start + 4096]
                    optimizer.zero_grad(set_to_none=True)
                    loss = loss_fn(model(x_fit[ix]), y_fit[ix])
                    loss.backward()
                    optimizer.step()
                    fit_sum += float(loss.detach().item()) * len(ix)
                    fit_rows += len(ix)
                model.eval()
                stop_sum = 0.0
                with torch.inference_mode():
                    for batch_start in range(0, len(x_stop), 16384):
                        z = model(x_stop[batch_start:batch_start + 16384])
                        n = len(z)
                        stop_sum += float(loss_fn(z, y_stop[batch_start:batch_start + n]).item()) * n
                stop_loss = stop_sum / len(x_stop)
                fit_loss = fit_sum / fit_rows
                improved = stop_loss < best_loss
                histories.append({
                    "variant": variant, "target": "MATCHABILITY" if variant == "MATCHABILITY_TARGET" else "PREFIX_MARGINAL",
                    "seed": seed, "epoch": epoch, "fit_natural_bce": fit_loss,
                    "early_stop_natural_bce": stop_loss, "strict_improvement": bool(improved),
                    "stale_epochs_after": 0 if improved else stale + 1,
                })
                if improved:
                    best_loss, best_epoch = stop_loss, epoch
                    best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
                    stale = 0
                else:
                    stale += 1
                if stale >= 5:
                    break
            if best_state is None:
                raise RuntimeError(f"no model checkpoint {variant} seed={seed}")
            model.load_state_dict(best_state)
            model.eval()
            model_path = variant_dir / f"marginal_mlp_seed_{seed}.pt"
            torch.save({
                "state_dict": best_state, "seed": seed, "input_dim": 90,
                "architecture": "Linear(90,128)-LayerNorm-GELU-Linear(128,64)-GELU-Linear(64,10)",
                "best_epoch": best_epoch, "early_stop_natural_bce": best_loss,
                "variant": variant, "target": "MATCHABILITY" if variant == "MATCHABILITY_TARGET" else "PREFIX_MARGINAL",
                "mask_indices": indices.tolist(), "mask_stage": "post_standardization",
            }, model_path)
            xt_cal = torch.from_numpy(np.asarray(x_cal_np_for_predict, dtype=np.float32)).to(device)
            logit_parts = []
            with torch.inference_mode():
                for batch_start in range(0, len(xt_cal), 16384):
                    logit_parts.append(model(xt_cal[batch_start:batch_start + 16384]).float().cpu().numpy())
            if device.type == "cuda":
                torch.cuda.synchronize()
            raw_logits = np.vstack(logit_parts).astype(np.float64)
            temperature, cal_summary = fit_temperature(raw_logits, y_cal)
            raw_prob = 1.0 / (1.0 + np.exp(-raw_logits))
            calibrated = 1.0 / (1.0 + np.exp(-raw_logits / temperature))
            reliability.extend({"variant": variant, **row} for row in reliability_rows(raw_prob, y_cal, seed, False))
            reliability.extend({"variant": variant, **row} for row in reliability_rows(calibrated, y_cal, seed, True))
            temp_path = variant_dir / f"temperature_seed_{seed}.json"
            write_json(temp_path, {"variant": variant, "seed": seed, **cal_summary})
            summary = {
                "variant": variant, "seed": seed, "best_epoch": best_epoch, "epochs_run": epoch,
                "early_stop_natural_bce": best_loss, "temperature": temperature, **cal_summary,
                "model_sha256": sha256_file(model_path), "temperature_sha256": sha256_file(temp_path),
                "elapsed_seconds": time.perf_counter() - seed_start,
            }
            summaries.append(summary)
            all_summaries.append(summary)
            append_log(root, f"TRAIN variant={variant} seed={seed} best_epoch={best_epoch} stop_bce={best_loss:.12g} T={temperature:.12g}")
        write_json(variant_dir / "training_summary.json", {
            "status": "PASS", "variant": variant, "device": str(device), "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda, "fit_rows": int(fit_mask.sum()), "early_stop_rows": int(stop_mask.sum()),
            "calibration_rows": int(cal_mask.sum()), "no_final_refit": True, "seeds": summaries,
            "elapsed_seconds": time.perf_counter() - variant_start,
        })
        del x_fit, x_stop, y_fit, y_stop, x_fit_np, x_stop_np, x_cal_np, x_cal_np_for_predict
        if device.type == "cuda":
            torch.cuda.empty_cache()
    write_parquet(pd.DataFrame(histories), root / "outputs" / "training_history.parquet")
    write_parquet(pd.DataFrame(reliability), root / "outputs" / "calibration_reliability.parquet")
    write_json(root / "models" / "training_summary.json", {
        "status": "PASS", "variants": list(VARIANTS), "models": all_summaries,
        "total_models": len(all_summaries), "elapsed_seconds": time.perf_counter() - start_all,
    })
    append_log(root, f"STAGE train_variants COMPLETE elapsed_seconds={time.perf_counter()-start_all:.6f}")


if __name__ == "__main__":
    main()
