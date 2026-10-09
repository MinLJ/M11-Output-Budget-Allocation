"""Train and CALIBRATION-calibrate the three frozen COCO Route-A M11 seeds.

The entrypoint consumes only arrays committed by ``prepare_route_a.py``.  It
has no candidate, detector, VAL, selection, or evaluation argument.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch import nn

from route_a_common import (
    EXPECTED_TRAIN_ASSET_ID,
    PROJECT_ROOT,
    ROLE_TO_CODE,
    SEEDS,
    load_config,
    load_p1_core,
    sha256_file,
    write_dataframe_csv_atomic,
    write_json_atomic,
    write_sha256_ledger,
)


class MarginalMLP(nn.Module):
    """Exact P1 90->128->64->10 architecture."""

    def __init__(self, input_dim: int = 90):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 10),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value)


def progress(message: str) -> None:
    print(f"[Route-A train] {message}", flush=True)


def torch_save_atomic(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    os.close(handle)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def select_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def batched_logits(model: nn.Module, features: torch.Tensor, batch_size: int) -> np.ndarray:
    model.eval()
    pieces = []
    for start in range(0, len(features), batch_size):
        pieces.append(model(features[start:start + batch_size]).float().cpu().numpy())
    return np.vstack(pieces).astype(np.float64)


def verify_preparation(root: Path, config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, int]]:
    marker_path = root / "working" / "prepare_complete.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if marker.get("status") != "ROUTE_A_TRAIN_PREPARATION_COMPLETE":
        raise ValueError("Route-A preparation completion marker is missing/invalid")
    checks = {
        "route_a_config_sha256": root / "route_a_config.json",
        "split_manifest_sha256": root / "split_manifest.csv",
        "feature_config_sha256": root / "feature_config.json",
        "pca_sha256": root / "pca_model.joblib",
        "scaler_sha256": root / "scaler.joblib",
        "class_weight_config_sha256": root / "class_weight_config.json",
        "train_labels_sha256": root / "working" / "train_labels.parquet",
        "train_X_scaled_sha256": root / "working" / "cache" / "train_X_scaled.npy",
        "train_y_sha256": root / "working" / "cache" / "train_y.npy",
        "train_role_sha256": root / "working" / "cache" / "train_role.npy",
    }
    for key, path in checks.items():
        if sha256_file(path) != str(marker[key]):
            raise ValueError(f"prepared asset hash mismatch: {key}")
    counts = {key: int(value) for key, value in config["role_split"]["expected_counts"].items()}
    manifest = pd.read_csv(root / "split_manifest.csv", dtype={"image_id": str, "split": str, "hash": str})
    if manifest.columns.tolist() != ["image_id", "split", "hash"] or manifest["split"].value_counts().to_dict() != counts:
        raise ValueError("frozen split manifest schema/count mismatch")
    return marker, counts


def execute(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    root = args.output_root.resolve(strict=True)
    config = load_config(root / "route_a_config.json")
    marker, image_role_counts = verify_preparation(root, config)
    p1_core = load_p1_core(config)
    model_cfg = config["model"]
    seeds = tuple(int(value) for value in model_cfg["seeds"])
    if seeds != SEEDS:
        raise ValueError("Route-A seed identity changed")

    checkpoint_dir = root / "model_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    protected = [root / "training_history.csv", root / "temperature_params.json"] + [checkpoint_dir / f"marginal_mlp_seed_{seed}.pt" for seed in seeds]
    existing = [str(path) for path in protected if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite trained Route-A assets: {existing}")

    cache = root / "working" / "cache"
    x_np = np.load(cache / "train_X_scaled.npy", mmap_mode="r")
    y_np = np.load(cache / "train_y.npy", mmap_mode="r")
    role_np = np.load(cache / "train_role.npy", mmap_mode="r")
    total_images = sum(image_role_counts.values())
    total_rows = total_images * 45
    if x_np.shape != (total_rows, 90) or y_np.shape != (total_rows, 10) or role_np.shape != (total_rows,):
        raise ValueError(f"prepared cache shape mismatch: X={x_np.shape}, y={y_np.shape}, role={role_np.shape}")
    expected_role_rows = tuple(image_role_counts[role] * 45 for role in ("FIT", "EARLY_STOP", "CALIBRATION"))
    observed_role_rows = tuple(int(np.sum(role_np == ROLE_TO_CODE[role])) for role in ("FIT", "EARLY_STOP", "CALIBRATION"))
    if observed_role_rows != expected_role_rows:
        raise ValueError(f"prepared role-row counts changed: {observed_role_rows} != {expected_role_rows}")

    fit_mask = role_np == ROLE_TO_CODE["FIT"]
    stop_mask = role_np == ROLE_TO_CODE["EARLY_STOP"]
    calibration_mask = role_np == ROLE_TO_CODE["CALIBRATION"]
    device = select_device(args.device)
    synchronize(device)
    progress(f"loading prepared tensors on {device}; no VAL data is accessible")
    x_fit = torch.from_numpy(np.asarray(x_np[fit_mask]).copy()).to(device)
    y_fit = torch.from_numpy(np.asarray(y_np[fit_mask], dtype=np.float32).copy()).to(device)
    x_stop = torch.from_numpy(np.asarray(x_np[stop_mask]).copy()).to(device)
    y_stop = torch.from_numpy(np.asarray(y_np[stop_mask], dtype=np.float32).copy()).to(device)
    x_calibration = torch.from_numpy(np.asarray(x_np[calibration_mask]).copy()).to(device)
    y_calibration_np = np.asarray(y_np[calibration_mask], dtype=np.float64).copy()

    loss_fn = nn.BCEWithLogitsLoss(reduction="mean")
    histories: list[dict[str, Any]] = []
    reliability_rows: list[dict[str, Any]] = []
    temperature_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    config_sha = sha256_file(root / "route_a_config.json")
    feature_config_sha = sha256_file(root / "feature_config.json")
    class_weight_sha = sha256_file(root / "class_weight_config.json")
    pca_sha = sha256_file(root / "pca_model.joblib")
    scaler_sha = sha256_file(root / "scaler.joblib")

    for seed in seeds:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        model = MarginalMLP(90).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(model_cfg["learning_rate"]),
            weight_decay=float(model_cfg["weight_decay"]),
        )
        generator = torch.Generator(device=device.type).manual_seed(seed)
        best_loss = float("inf")
        best_epoch = 0
        best_state: dict[str, torch.Tensor] | None = None
        stale = 0
        seed_started = time.perf_counter()
        max_epochs = int(model_cfg["max_epochs"])
        batch_size = int(model_cfg["batch_size"])
        patience = int(model_cfg["early_stop_patience"])

        for epoch in range(1, max_epochs + 1):
            model.train()
            order = torch.randperm(len(x_fit), generator=generator, device=device)
            fit_total = 0.0
            fit_rows = 0
            for start in range(0, len(x_fit), batch_size):
                indices = order[start:start + batch_size]
                optimizer.zero_grad(set_to_none=True)
                logits = model(x_fit[indices])
                loss = loss_fn(logits, y_fit[indices])
                loss.backward()
                optimizer.step()
                count = len(indices)
                fit_total += float(loss.detach().item()) * count
                fit_rows += count
            model.eval()
            stop_total = 0.0
            with torch.inference_mode():
                for start in range(0, len(x_stop), int(model_cfg["inference_batch_size"])):
                    logits = model(x_stop[start:start + int(model_cfg["inference_batch_size"])])
                    count = len(logits)
                    stop_total += float(loss_fn(logits, y_stop[start:start + count]).item()) * count
            stop_loss = stop_total / len(x_stop)
            fit_loss = fit_total / fit_rows
            improved = stop_loss < best_loss
            next_stale = 0 if improved else stale + 1
            histories.append({
                "seed": seed,
                "epoch": epoch,
                "fit_natural_bce": fit_loss,
                "early_stop_natural_bce": stop_loss,
                "strict_improvement": bool(improved),
                "stale_epochs_after": next_stale,
            })
            if improved:
                best_loss = stop_loss
                best_epoch = epoch
                best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
                stale = 0
            else:
                stale = next_stale
            progress(f"seed={seed} epoch={epoch} fit_bce={fit_loss:.10g} stop_bce={stop_loss:.10g} stale={stale}")
            if stale >= patience:
                break

        if best_state is None:
            raise RuntimeError(f"no best model captured for seed {seed}")
        model.load_state_dict(best_state)
        model.eval()
        checkpoint = {
            "state_dict": best_state,
            "seed": seed,
            "input_dim": 90,
            "architecture": model_cfg["architecture"],
            "best_epoch": best_epoch,
            "early_stop_natural_bce": best_loss,
            "route": config["route"],
            "route_a_config_sha256": config_sha,
            "feature_config_sha256": feature_config_sha,
            "class_weight_config_sha256": class_weight_sha,
            "pca_sha256": pca_sha,
            "scaler_sha256": scaler_sha,
            "no_final_refit": True,
        }
        model_path = checkpoint_dir / f"marginal_mlp_seed_{seed}.pt"
        torch_save_atomic(checkpoint, model_path)
        calibration_logits = batched_logits(model, x_calibration, int(model_cfg["inference_batch_size"]))
        temperature, calibration_summary = p1_core.fit_temperature(calibration_logits, y_calibration_np)
        raw_probability = p1_core.sigmoid(calibration_logits)
        calibrated_probability = p1_core.sigmoid(calibration_logits / temperature)
        reliability_rows.extend(p1_core.reliability_rows(raw_probability, y_calibration_np, seed, False))
        reliability_rows.extend(p1_core.reliability_rows(calibrated_probability, y_calibration_np, seed, True))
        model_sha = sha256_file(model_path)
        temperature_row = {
            "seed": seed,
            **calibration_summary,
            "model_path": f"model_checkpoints/{model_path.name}",
            "model_sha256": model_sha,
            "best_epoch": best_epoch,
            "early_stop_natural_bce": best_loss,
        }
        temperature_rows.append(temperature_row)
        summaries.append({
            **temperature_row,
            "epochs_run": epoch,
            "elapsed_seconds": time.perf_counter() - seed_started,
        })
        progress(f"seed={seed} frozen best_epoch={best_epoch} T={temperature:.12g} model_sha256={model_sha}")

    history_frame = pd.DataFrame(histories)
    write_dataframe_csv_atomic(history_frame, root / "training_history.csv")
    pq.write_table(
        pa.Table.from_pandas(pd.DataFrame(reliability_rows), preserve_index=False),
        root / "working" / "calibration_reliability.parquet",
        compression="zstd",
    )
    temperature_payload = {
        "task": config["task"],
        "route": config["route"],
        "fit_role": "CALIBRATION",
        "shared_across_outputs": True,
        "one_parameter_per_seed": True,
        "bounds": config["calibration"]["bounds"],
        "fallback": config["calibration"]["fallback"],
        "objective": config["calibration"]["objective"],
        "val_accessed": False,
        "route_a_config_sha256": config_sha,
        "feature_config_sha256": feature_config_sha,
        "seeds": temperature_rows,
    }
    write_json_atomic(root / "temperature_params.json", temperature_payload)
    temperature_sha = sha256_file(root / "temperature_params.json")
    identity_bindings = {
        "route": config["route"],
        "train_candidate_asset_id": EXPECTED_TRAIN_ASSET_ID,
        "train_gt_sha256": config["train_ground_truth"]["sha256"],
        "route_a_config_sha256": config_sha,
        "feature_config_sha256": feature_config_sha,
        "pca_sha256": pca_sha,
        "scaler_sha256": scaler_sha,
        "class_weight_config_sha256": class_weight_sha,
        "temperature_params_sha256": temperature_sha,
        "model_sha256_by_seed": {str(row["seed"]): row["model_sha256"] for row in temperature_rows},
        "seeds": list(seeds),
    }
    allocator_asset_id = hashlib.sha256(
        json.dumps(identity_bindings, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()
    allocator_manifest = {
        "status": "COCO_ROUTE_A_ALLOCATOR_ASSET_COMPLETE",
        "allocator_asset_id": allocator_asset_id,
        "identity_hash": "SHA256(canonical compact UTF-8 JSON of identity_bindings)",
        "identity_bindings": identity_bindings,
        "models_are_independent_not_ensemble": True,
        "val_accessed": False,
        "detector_retrained": False,
    }
    write_json_atomic(root / "allocator_asset_manifest.json", allocator_manifest)
    allocator_manifest_sha = sha256_file(root / "allocator_asset_manifest.json")
    training_summary = {
        "status": "ROUTE_A_MODEL_TRAINING_COMPLETE",
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "fit_rows": observed_role_rows[0],
        "early_stop_rows": observed_role_rows[1],
        "calibration_rows": observed_role_rows[2],
        "seeds": summaries,
        "no_final_refit": True,
        "val_accessed": False,
        "detector_forward": False,
        "allocator_asset_id": allocator_asset_id,
        "allocator_asset_manifest_sha256": allocator_manifest_sha,
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json_atomic(root / "working" / "training_summary.json", training_summary)

    ledger_paths = [
        Path("route_a_config.json"), Path("split_manifest.csv"), Path("coco_allocator_split_manifest.csv"),
        Path("feature_config.json"), Path("pca_model.joblib"), Path("scaler.joblib"),
        Path("class_weight_config.json"), Path("training_history.csv"), Path("temperature_params.json"),
        Path("allocator_asset_manifest.json"),
        Path("working/prepare_complete.json"), Path("working/training_summary.json"),
        Path("working/label_audit.json"), Path("working/pca_sample_manifest.parquet"),
        Path("working/calibration_reliability.parquet"),
        Path("working/route_a_pipeline_config.json"),
        Path("scripts/route_a_common.py"), Path("scripts/prepare_route_a.py"), Path("scripts/train_route_a.py"),
        *[Path(f"model_checkpoints/marginal_mlp_seed_{seed}.pt") for seed in seeds],
    ]
    training_ledger_path = root / "working" / "training_sha256_ledger.csv"
    ledger = write_sha256_ledger(root, ledger_paths, ledger_path=training_ledger_path)
    training_summary["sha256_ledger_entries"] = len(ledger)
    training_summary["sha256_ledger_path"] = "working/training_sha256_ledger.csv"
    training_summary["sha256_ledger_sha256"] = sha256_file(training_ledger_path)
    write_json_atomic(root / "working" / "training_complete.json", training_summary)
    progress("training/calibration assets complete; no VAL data or detector was accessed")
    return training_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train three COCO-adapted M11 models and fit CALIBRATION-only temperatures")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    return parser.parse_args()


if __name__ == "__main__":
    result = execute(parse_args())
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
