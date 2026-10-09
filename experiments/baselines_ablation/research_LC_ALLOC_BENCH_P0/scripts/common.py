"""Shared frozen helpers for LC-ALLOC-BENCH-P0.

This module has no top-level scientific I/O and disables bytecode writes so
imports do not mutate frozen projects.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[1]
P1 = (Path(os.environ.get('M11_EXPERIMENT_ROOT', "external_assets")) / 'research_LC_ALLOC_P1')
P1A = (Path(os.environ.get('M11_EXPERIMENT_ROOT', "external_assets")) / 'research_LC_ALLOC_P1A')
P1B = (Path(os.environ.get('M11_EXPERIMENT_ROOT', "external_assets")) / 'research_LC_ALLOC_P1B')
R0 = (Path(os.environ.get('M11_EXPERIMENT_ROOT', "external_assets")) / 'research_LC_ALLOC_R0')
RELEASE = (Path(os.environ.get('M11_EXPERIMENT_ROOT', "external_assets")) / 'shared_benchmark/AOP_ROAD8_LARGECLEAN_V1/P1_SHARED_EXPORT/release/AOP_Road8_LC_v1_RTDERTv2_R18VD_RELEASE_v1')
BUDGETS = (10, 15, 20, 30, 40)
SEEDS = (530101, 530102, 530103)
K_MIN, K_MAX = 5, 50
ROAD8 = ("person", "bicycle", "car", "motorcycle", "bus", "train", "truck", "traffic light")
THRESHOLDS = np.arange(0.50, 0.951, 0.05, dtype=np.float64)


def sha256_file(path: os.PathLike[str] | str, chunk: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_json(path: os.PathLike[str] | str, obj: object) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, p)


def write_parquet(frame: pd.DataFrame, path: os.PathLike[str] | str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), tmp, compression="zstd")
    os.replace(tmp, p)


def append_log(text: str) -> None:
    with (ROOT / "runtime.log").open("a", encoding="utf-8") as f:
        f.write(text.rstrip() + "\n")


def import_file(path: os.PathLike[str] | str, name: str):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_identity() -> dict:
    return json.loads((RELEASE / "CANDIDATE_ASSET_IDENTITY.json").read_text(encoding="utf-8"))


def load_candidate_frame(split: str, max_rank: int = 100) -> pd.DataFrame:
    identity = load_identity()
    rows: list[pd.DataFrame] = []
    for rel in identity["assets"][split]["candidate_shards"]:
        table = pq.read_table(RELEASE / rel)
        frame = table.to_pandas()
        rows.append(frame.loc[frame["road8_rank"].astype(int) <= max_rank])
    out = pd.concat(rows, ignore_index=True)
    out["image_id"] = out["image_id"].astype(str)
    out["candidate_record_id"] = out["candidate_record_id"].astype(str)
    out = out.sort_values(["image_id", "road8_rank"], kind="stable").reset_index(drop=True)
    expected_images = 2000 if split == "DEV" else 10000
    if out["image_id"].nunique() != expected_images or len(out) != expected_images * max_rank:
        raise RuntimeError(f"{split} candidate coverage mismatch: {len(out)} rows")
    rank_ok = out.groupby("image_id", sort=False)["road8_rank"].agg(list).map(lambda z: z == list(range(1, max_rank + 1))).all()
    if not rank_ok or out["candidate_record_id"].duplicated().any():
        raise RuntimeError(f"{split} rank or candidate identity invariant failed")
    return out


def load_dev_prediction_arrays() -> tuple[list[str], dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray], dict[str, list[str]]]:
    pred = pq.read_table(P1 / "dev_predictions.parquet").to_pandas()
    pred["image_id"] = pred["image_id"].astype(str)
    pred = pred.sort_values(["image_id", "road8_rank"], kind="stable")
    ids = sorted(pred["image_id"].unique().tolist())
    scores: dict[str, np.ndarray] = {}
    classes: dict[str, np.ndarray] = {}
    quality: dict[str, np.ndarray] = {}
    optional_ids: dict[str, list[str]] = {}
    weights = np.asarray(json.loads((P1 / "models" / "class_weights.json").read_text(encoding="utf-8"))["weights"], dtype=np.float64)
    for image_id, d in pred.groupby("image_id", sort=False):
        if d["road8_rank"].astype(int).tolist() != list(range(6, 51)):
            raise RuntimeError(f"prediction ranks invalid: {image_id}")
        cls = d["predicted_road8_class_id"].to_numpy(np.int16)
        scores[image_id] = d["raw_score"].to_numpy(np.float64)
        classes[image_id] = cls
        optional_ids[image_id] = d["candidate_record_id"].astype(str).tolist()
        q = np.empty((len(SEEDS), 45), dtype=np.float64)
        for si, seed in enumerate(SEEDS):
            cols = [f"learn_p_seed_{seed}_iou_{x:.2f}".replace(".", "_") for x in THRESHOLDS]
            q[si] = d[cols].to_numpy(np.float64).mean(axis=1) * weights[cls - 1]
        quality[image_id] = q
    if len(ids) != 2000:
        raise RuntimeError("DEV prediction image count mismatch")
    return ids, scores, classes, quality, optional_ids


_DP_MODULE = None


def solve_group(marginals: np.ndarray, budgets: Sequence[int] = BUDGETS) -> tuple[np.ndarray, np.ndarray]:
    global _DP_MODULE
    if _DP_MODULE is None:
        _DP_MODULE = import_file(P1 / "scripts" / "dp_solver.py", "lc_alloc_bench_dp")
    dp = _DP_MODULE
    values = dp.choice_values_from_optional_marginals(np.asarray(marginals, dtype=np.float64))
    n = values.shape[0]
    return dp.solve_exact_multiple_choice(values, [n * int(x) for x in budgets], k_min=K_MIN)


def selected_ids_for_k(candidate_ids: dict[str, list[str]], image_id: str, k: int) -> list[str]:
    ids = candidate_ids[image_id]
    if len(ids) < k:
        raise RuntimeError(f"candidate ID pool too short for {image_id}")
    return ids[:k]


def load_candidate_id_map(split: str = "DEV", max_rank: int = 50) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    frame = load_candidate_frame(split, max_rank=max_rank)
    mapping = {str(image_id): d["candidate_record_id"].astype(str).tolist() for image_id, d in frame.groupby("image_id", sort=False)}
    return frame, mapping


def stable_config_seed(permutation: str, group_size: int) -> int:
    digest = sha256_text(f"LC_ALLOC_BENCH_P0_BOOTSTRAP|{permutation}|{group_size}|530002")
    return int(digest[:16], 16) % (2**32)


def sigmoid(x: np.ndarray) -> np.ndarray:
    z = np.asarray(x, dtype=np.float64)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out
