from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch import nn

sys.dont_write_bytecode = True

SEEDS = (530101, 530102, 530103)
BUDGETS = (10, 15, 20, 30, 40)
VARIANTS = ("NO_EMBED", "NO_PREFIX", "MATCHABILITY_TARGET")
THRESHOLDS = np.asarray([0.50 + 0.05 * i for i in range(10)], dtype=np.float64)


class MarginalMLP(nn.Module):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def sha256_file(path: os.PathLike[str] | str, chunk: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            data = f.read(chunk)
            if not data:
                break
            h.update(data)
    return h.hexdigest()


def write_json(path: os.PathLike[str] | str, value: object) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, p)


def write_parquet(df: pd.DataFrame, path: os.PathLike[str] | str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), tmp, compression="zstd")
    os.replace(tmp, p)


def append_log(root: Path, text: str) -> None:
    with (root / "runtime.log").open("a", encoding="utf-8") as f:
        f.write(text.rstrip() + "\n")


def verify_input_binding(root: Path) -> dict:
    """Fail closed if any file frozen by prepare_inputs has changed."""
    binding_path = root / "outputs" / "input_binding.json"
    payload = json.loads(binding_path.read_text(encoding="utf-8"))
    mismatches = []
    for raw_path, expected in payload.get("files", {}).items():
        path = Path(raw_path)
        actual = sha256_file(path) if path.is_file() else None
        if actual != expected:
            mismatches.append({"path": str(path), "expected": expected, "actual": actual})
    if mismatches:
        raise RuntimeError(f"frozen input binding mismatch: {mismatches[:3]}")
    return payload


def add_p1_scripts(p1_root: Path) -> None:
    scripts = str((p1_root / "scripts").resolve())
    if scripts not in sys.path:
        sys.path.insert(0, scripts)


def stable_sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


@torch.inference_mode()
def predict_probabilities(model: nn.Module, x: np.ndarray, temperature: float, device: torch.device, batch_size: int = 16384) -> np.ndarray:
    model.eval()
    xt = torch.from_numpy(np.asarray(x, dtype=np.float32)).to(device)
    pieces = []
    for start in range(0, len(xt), batch_size):
        pieces.append(model(xt[start:start + batch_size]).float().cpu().numpy())
    if device.type == "cuda":
        torch.cuda.synchronize()
    logits = np.vstack(pieces).astype(np.float64)
    return stable_sigmoid(logits / float(temperature))


def load_model(model_path: Path, device: torch.device) -> MarginalMLP:
    snapshot = torch.load(model_path, map_location="cpu", weights_only=True)
    model = MarginalMLP(int(snapshot["input_dim"]))
    model.load_state_dict(snapshot["state_dict"])
    model.eval().to(device)
    return model


def condition_label(method: str, seed: int) -> str:
    return method if seed < 0 else f"{method}__SEED_{seed}"
