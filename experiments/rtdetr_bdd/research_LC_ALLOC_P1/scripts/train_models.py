"""Train the single preregistered 90->128->64->10 MLP for three fixed seeds.

Only prepared TRAIN arrays are read.  No DEV asset or DEV ground truth is
accepted by this entrypoint.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch import nn

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from p1_core import SEEDS, bce_numpy, fit_temperature, reliability_rows, sha256_file, sigmoid, write_json  # noqa: E402


class MarginalMLP(nn.Module):
    def __init__(self, input_dim: int = 90):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128), nn.LayerNorm(128), nn.GELU(),
            nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def append_log(root: Path, text: str) -> None:
    with (root / "runtime.log").open("a", encoding="utf-8") as f:
        f.write(text.rstrip() + "\n")


@torch.inference_mode()
def batched_logits(model: nn.Module, x: torch.Tensor, batch_size: int = 16384) -> np.ndarray:
    model.eval()
    parts = []
    for start in range(0, len(x), batch_size):
        parts.append(model(x[start:start + batch_size]).float().cpu().numpy())
    return np.vstack(parts).astype(np.float64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", required=True)
    args = ap.parse_args()
    root = Path(args.project_root).resolve()
    t0 = time.perf_counter()
    append_log(root, "STAGE train_models START")

    x_np = np.load(root / "cache" / "train_X_scaled.npy", mmap_mode="r")
    y_np = np.load(root / "cache" / "train_y.npy", mmap_mode="r")
    role_np = np.load(root / "cache" / "train_role.npy", mmap_mode="r")
    if x_np.shape != (450000, 90) or y_np.shape != (450000, 10):
        raise RuntimeError(f"prepared array shape mismatch X={x_np.shape} y={y_np.shape}")
    fit_mask, stop_mask, cal_mask = role_np == 0, role_np == 1, role_np == 2
    if (int(fit_mask.sum()), int(stop_mask.sum()), int(cal_mask.sum())) != (360000, 45000, 45000):
        raise RuntimeError("TRAIN role row counts are invalid")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.synchronize()
    x_fit = torch.from_numpy(np.asarray(x_np[fit_mask]).copy()).to(device)
    y_fit = torch.from_numpy(np.asarray(y_np[fit_mask], dtype=np.float32).copy()).to(device)
    x_stop = torch.from_numpy(np.asarray(x_np[stop_mask]).copy()).to(device)
    y_stop = torch.from_numpy(np.asarray(y_np[stop_mask], dtype=np.float32).copy()).to(device)
    x_cal = torch.from_numpy(np.asarray(x_np[cal_mask]).copy()).to(device)
    y_cal_np = np.asarray(y_np[cal_mask], dtype=np.float64).copy()
    loss_fn = nn.BCEWithLogitsLoss(reduction="mean")
    histories = []
    reliability = []
    summaries = []

    for seed in SEEDS:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        model = MarginalMLP(90).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        generator = torch.Generator(device=device.type).manual_seed(seed)
        best_loss = float("inf")
        best_epoch = 0
        best_state = None
        stale = 0
        seed_start = time.perf_counter()
        for epoch in range(1, 31):
            model.train()
            order = torch.randperm(len(x_fit), generator=generator, device=device)
            total_loss = 0.0
            total_rows = 0
            for start in range(0, len(x_fit), 4096):
                ix = order[start:start + 4096]
                optimizer.zero_grad(set_to_none=True)
                logits = model(x_fit[ix])
                loss = loss_fn(logits, y_fit[ix])
                loss.backward()
                optimizer.step()
                n = len(ix)
                total_loss += float(loss.detach().item()) * n
                total_rows += n
            model.eval()
            stop_total = 0.0
            with torch.inference_mode():
                for start in range(0, len(x_stop), 16384):
                    z = model(x_stop[start:start + 16384])
                    n = len(z)
                    stop_total += float(loss_fn(z, y_stop[start:start + n]).item()) * n
            stop_loss = stop_total / len(x_stop)
            train_loss = total_loss / total_rows
            improved = stop_loss < best_loss
            histories.append({
                "seed": seed, "epoch": epoch, "fit_natural_bce": train_loss,
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
            raise RuntimeError(f"no best model captured for seed {seed}")
        model.load_state_dict(best_state)
        model.eval()
        snapshot = {
            "state_dict": best_state, "seed": seed, "input_dim": 90,
            "architecture": "Linear(90,128)-LayerNorm-GELU-Linear(128,64)-GELU-Linear(64,10)",
            "best_epoch": best_epoch, "early_stop_natural_bce": best_loss,
        }
        model_path = root / "models" / f"marginal_mlp_seed_{seed}.pt"
        torch.save(snapshot, model_path)
        cal_logits = batched_logits(model, x_cal)
        temperature, cal_summary = fit_temperature(cal_logits, y_cal_np)
        cal_prob_raw = sigmoid(cal_logits)
        cal_prob = sigmoid(cal_logits / temperature)
        reliability.extend(reliability_rows(cal_prob_raw, y_cal_np, seed, False))
        reliability.extend(reliability_rows(cal_prob, y_cal_np, seed, True))
        calibration_path = root / "models" / f"temperature_seed_{seed}.json"
        write_json(calibration_path, {"seed": seed, **cal_summary})
        summary = {
            "seed": seed, "best_epoch": best_epoch, "epochs_run": epoch,
            "early_stop_natural_bce": best_loss, "temperature": temperature,
            **cal_summary,
            "model_sha256": sha256_file(model_path), "calibration_sha256": sha256_file(calibration_path),
            "elapsed_seconds": time.perf_counter() - seed_start,
        }
        summaries.append(summary)
        append_log(root, f"TRAIN seed={seed} best_epoch={best_epoch} stop_bce={best_loss:.12g} T={temperature:.12g}")

    pq.write_table(pa.Table.from_pandas(pd.DataFrame(histories), preserve_index=False), root / "training_history.parquet", compression="zstd")
    pq.write_table(pa.Table.from_pandas(pd.DataFrame(reliability), preserve_index=False), root / "outputs" / "calibration_reliability.parquet", compression="zstd")
    write_json(root / "models" / "training_summary.json", {
        "status": "PASS", "device": str(device), "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda, "fit_rows": int(fit_mask.sum()),
        "early_stop_rows": int(stop_mask.sum()), "calibration_rows": int(cal_mask.sum()),
        "seeds": summaries, "no_final_refit": True,
    })
    if device.type == "cuda":
        torch.cuda.synchronize()
    append_log(root, f"STAGE train_models COMPLETE elapsed_seconds={time.perf_counter()-t0:.6f}")


if __name__ == "__main__":
    main()

