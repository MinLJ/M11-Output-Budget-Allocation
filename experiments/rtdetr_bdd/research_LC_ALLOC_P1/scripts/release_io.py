"""Read-only adapters for the frozen LC-v1 shared release."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from p1_core import sigmoid


REQUIRED_CANDIDATE_COLUMNS = [
    "dataset_version", "split", "image_id", "image_sha256", "detector_id",
    "checkpoint_sha256", "export_config_sha256", "candidate_asset_id",
    "candidate_record_id", "road8_rank", "predicted_road8_class_id",
    "predicted_road8_class_name", "score", "bbox_x1", "bbox_y1", "bbox_x2",
    "bbox_y2", "bbox_cx", "bbox_cy", "bbox_w", "bbox_h", "query_index", "source_order",
]


@dataclass
class ImageBundle:
    image_id: str
    width: int
    height: int
    candidates: pd.DataFrame
    road8_logits: np.ndarray
    embeddings: np.ndarray


class ReleaseReader:
    def __init__(self, release_root: str | Path):
        self.root = Path(release_root).resolve()
        self.identity_path = self.root / "CANDIDATE_ASSET_IDENTITY.json"
        self.manifest_path = self.root / "RELEASE_MANIFEST.json"
        self.identity = json.loads(self.identity_path.read_text(encoding="utf-8"))
        self.release_manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if bool(self.release_manifest.get("TEST_INCLUDED", True)):
            raise RuntimeError("frozen release is not marked TEST_INCLUDED=false")
        self._manifest_cache: Dict[str, pd.DataFrame] = {}

    def asset(self, split: str) -> dict:
        s = split.upper()
        if s not in ("TRAIN", "DEV"):
            raise ValueError("LC-ALLOC-P1 only permits TRAIN and DEV")
        return self.identity["assets"][s]

    def split_manifest(self, split: str) -> pd.DataFrame:
        s = split.upper()
        if s not in self._manifest_cache:
            df = pq.read_table(self.root / self.asset(s)["split_manifest_path"]).to_pandas()
            expected = int(self.asset(s)["image_count"])
            if len(df) != expected or df["image_id"].nunique() != expected:
                raise RuntimeError(f"{s} manifest identity invariant failed")
            self._manifest_cache[s] = df
        return self._manifest_cache[s]

    def shard_paths(self, split: str) -> Iterator[Tuple[Path, Path]]:
        a = self.asset(split)
        cands = a["candidate_shards"]
        natives = a["native_state_shards"]
        if len(cands) != len(natives):
            raise RuntimeError("candidate/native shard count mismatch")
        for cp, npz in zip(cands, natives):
            yield self.root / cp, self.root / npz

    def iter_bundles(self, split: str) -> Iterator[ImageBundle]:
        s = split.upper()
        manifest = self.split_manifest(s).set_index("image_id", drop=False)
        expected_asset_id = self.asset(s)["candidate_asset_id"]
        seen: set[str] = set()
        for candidate_path, native_path in self.shard_paths(s):
            table = pq.read_table(
                candidate_path,
                columns=REQUIRED_CANDIDATE_COLUMNS,
                filters=[("road8_rank", "<=", 100)],
            )
            cdf = table.to_pandas()
            if len(cdf) == 0 or set(cdf["candidate_asset_id"].unique()) != {expected_asset_id}:
                raise RuntimeError(f"candidate asset identity mismatch in {candidate_path.name}")
            if cdf["candidate_record_id"].duplicated().any():
                raise RuntimeError(f"duplicate candidate identity in {candidate_path.name}")
            with np.load(native_path, allow_pickle=False) as state:
                image_ids = state["image_ids"].astype(str)
                query_index = state["query_index"].astype(np.int32)
                logits_all = state["l3_road8_logits"]
                embeddings_all = state["l3_query_embedding"]
                if logits_all.shape[1:] != (8,) or embeddings_all.shape[1:] != (256,):
                    raise RuntimeError(f"native tensor shape mismatch in {native_path.name}")
                starts: Dict[str, int] = {}
                for ix in range(0, len(image_ids), 300):
                    block_ids = image_ids[ix:ix + 300]
                    block_q = query_index[ix:ix + 300]
                    if len(block_ids) != 300 or len(set(block_ids)) != 1 or not np.array_equal(block_q, np.arange(300)):
                        raise RuntimeError(f"native 300-query block invariant failed at {native_path.name}:{ix}")
                    starts[str(block_ids[0])] = ix
                for image_id, rows in cdf.groupby("image_id", sort=False):
                    image_id = str(image_id)
                    if image_id in seen:
                        raise RuntimeError(f"duplicate image across shards: {image_id}")
                    seen.add(image_id)
                    rows = rows.sort_values("road8_rank", kind="stable").reset_index(drop=True)
                    if len(rows) != 100 or rows["road8_rank"].tolist() != list(range(1, 101)):
                        raise RuntimeError(f"Top100 rank invariant failed: {image_id}")
                    if image_id not in starts or image_id not in manifest.index:
                        raise RuntimeError(f"candidate/native/manifest join failed: {image_id}")
                    q = rows["query_index"].to_numpy(np.int32)
                    if np.any((q < 0) | (q >= 300)):
                        raise RuntimeError(f"query index out of range: {image_id}")
                    idx = starts[image_id] + q
                    if not np.all(image_ids[idx] == image_id) or not np.array_equal(query_index[idx], q):
                        raise RuntimeError(f"native gather identity failed: {image_id}")
                    logits = np.asarray(logits_all[idx], dtype=np.float32)
                    emb = np.asarray(embeddings_all[idx], dtype=np.float16)
                    cls0 = rows["predicted_road8_class_id"].to_numpy(np.int16) - 1
                    reconstructed = sigmoid(logits[np.arange(100), cls0])
                    max_diff = float(np.max(np.abs(reconstructed - rows["score"].to_numpy(np.float64))))
                    if max_diff > 1e-6:
                        raise RuntimeError(f"candidate/native score reconstruction failed {image_id}: {max_diff}")
                    m = manifest.loc[image_id]
                    yield ImageBundle(image_id, int(m["width"]), int(m["height"]), rows, logits, emb)
        expected = int(self.asset(s)["image_count"])
        if len(seen) != expected:
            raise RuntimeError(f"{s} image coverage {len(seen)} != {expected}")

