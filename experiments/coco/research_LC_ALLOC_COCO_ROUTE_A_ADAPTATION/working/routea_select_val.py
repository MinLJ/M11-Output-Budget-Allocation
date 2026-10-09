"""GT-free COCO2017 VAL selection for the COCO-adapted M11 allocator.

The script consumes the already-published VAL2017 canonical candidate/native
asset plus the newly trained Route-A allocator assets.  It has deliberately no
annotation or GT argument.  It writes all selections only after structural
checks pass and commits a completion marker last.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import platform
import shutil
import sys
import time
from typing import Any, Mapping

sys.dont_write_bytecode = True

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from routea_pipeline_common import (
    ASSET_ROOT_DEFAULT,
    BASELINE_SEED,
    BUDGETS,
    CAL_TEMP,
    CANDIDATE_ASSET_ID,
    DATASET_VERSION,
    EXPECTED_GROUPS,
    EXPECTED_IMAGES,
    GROUP_MANIFEST_DEFAULT,
    GROUP_SIZE,
    K_MAX,
    K_MIN,
    LC_CLASS_WEIGHT_SHA256,
    P1_ROOT_DEFAULT,
    ROOT_DEFAULT,
    ROUTE_A_SEEDS,
    ROUTE_B_RUNNER_DEFAULT,
    ROUTE_B_SELECTION_RUNNER_SHA256,
    SPLIT,
    TOP_N,
    condition_instances,
    condition_label,
    curve_frame,
    load_class_weights,
    load_frozen_p1_modules,
    load_module,
    load_temperatures,
    policy_id,
    read_json,
    require,
    self_test_common,
    sha256_array,
    sha256_file,
    stable_condition_sort,
    utc_now,
    validate_feature_config,
    write_csv_atomic,
    write_json_atomic,
)


TASK = "LC-ALLOC-COCO-ROUTE-A-VAL-SELECTION"
LC_PCA_SHA256 = "12d8e656d53bdad54e498125993f437f77d874e97be38ffe29a0c969e1ff41af"
LC_SCALER_SHA256 = "9d3da8c36b3b5a2cf8c8ab2deb963e3595c313b4d13f6d3e7fb8c27d923ced6d"


def progress(stage: str, message: str) -> None:
    print(f"[{utc_now()}] {stage}: {message}", flush=True)


def append_log(root: Path, message: str) -> None:
    path = root / "working" / "routea_selection_runtime.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(f"{utc_now()}\t{message}\n")


def _temperature_entries(payload: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    return {int(item["seed"]): item for item in payload["seeds"]}


def validate_training_completion(root: Path) -> dict[str, Any]:
    marker_path = root / "working" / "training_complete.json"
    ledger_path = root / "working" / "training_sha256_ledger.csv"
    require(marker_path.is_file() and ledger_path.is_file(), "Route-A training completion marker/ledger missing")
    marker = read_json(marker_path)
    require(marker.get("status") == "ROUTE_A_MODEL_TRAINING_COMPLETE", "Route-A model training is incomplete")
    require(marker.get("val_accessed") is False, "Route-A training accessed VAL")
    require(marker.get("detector_forward") is False, "Route-A training ran a detector forward pass")
    require(marker.get("no_final_refit") is True, "unexpected Route-A final refit")
    require({int(item["seed"]) for item in marker.get("seeds", [])} == set(ROUTE_A_SEEDS), "training completion seed set mismatch")
    require(marker.get("sha256_ledger_sha256") == sha256_file(ledger_path), "training ledger SHA mismatch")

    ledger = pd.read_csv(ledger_path)
    required = [
        "feature_config.json", "pca_model.joblib", "scaler.joblib", "class_weight_config.json",
        "training_history.csv", "temperature_params.json", "allocator_asset_manifest.json", "split_manifest.csv",
        "coco_allocator_split_manifest.csv", "working/prepare_complete.json",
        *[f"model_checkpoints/marginal_mlp_seed_{seed}.pt" for seed in ROUTE_A_SEEDS],
    ]
    for relative in required:
        rows = ledger[ledger["path"].astype(str).str.replace("\\\\", "/", regex=False) == relative]
        require(len(rows) == 1, f"training ledger entry missing/duplicate: {relative}")
        artifact = root / relative
        require(artifact.is_file(), f"training artifact missing: {relative}")
        require(sha256_file(artifact) == str(rows.iloc[0]["sha256"]), f"training artifact SHA mismatch: {relative}")
    allocator_manifest_path = root / "allocator_asset_manifest.json"
    allocator_manifest = read_json(allocator_manifest_path)
    require(allocator_manifest.get("status") == "COCO_ROUTE_A_ALLOCATOR_ASSET_COMPLETE", "allocator asset is incomplete")
    require(allocator_manifest.get("allocator_asset_id") == marker.get("allocator_asset_id"), "allocator asset ID mismatch")
    require(sha256_file(allocator_manifest_path) == marker.get("allocator_asset_manifest_sha256"), "allocator asset manifest SHA mismatch")
    require(allocator_manifest.get("models_are_independent_not_ensemble") is True, "allocator seed semantics changed")
    require(allocator_manifest.get("val_accessed") is False, "allocator asset creation accessed VAL")
    return marker


def load_routea_assets(root: Path, p1_root: Path, device: torch.device):
    training_marker = validate_training_completion(root)
    feature_config_path = root / "feature_config.json"
    class_config_path = root / "class_weight_config.json"
    temperature_path = root / "temperature_params.json"
    for path in (feature_config_path, class_config_path, temperature_path):
        require(path.is_file(), f"missing Route-A config: {path}")
    feature_config = validate_feature_config(feature_config_path, root)
    weights, class_config = load_class_weights(class_config_path)
    temperatures, temperature_payload = load_temperatures(temperature_path)

    pca_path = root / "pca_model.joblib"
    scaler_path = root / "scaler.joblib"
    pca_sha = sha256_file(pca_path)
    scaler_sha = sha256_file(scaler_path)
    require(pca_sha != LC_PCA_SHA256 and scaler_sha != LC_SCALER_SHA256, "Route-A must not use LC PCA/scaler")
    pca = joblib.load(pca_path)
    require(int(getattr(pca, "n_components_", getattr(pca, "n_components", -1))) == 32, "PCA component count mismatch")
    require(int(getattr(pca, "n_features_in_", -1)) == 256 and not bool(getattr(pca, "whiten", True)), "PCA metadata mismatch")
    scaler_bundle = joblib.load(scaler_path)
    require(isinstance(scaler_bundle, dict), "scaler.joblib must contain a dict")
    scaler = scaler_bundle.get("scaler")
    standardize = np.asarray(scaler_bundle.get("standardize_mask"), dtype=bool)
    feature_names = list(scaler_bundle.get("feature_names", []))
    require(scaler is not None and standardize.shape == (90,) and len(feature_names) == 90, "scaler bundle contract mismatch")
    require(int(standardize.sum()) == 81 and np.flatnonzero(~standardize).tolist() == [3, 4, 5, 6, 7, 8, 9, 10, 69], "standardization mask changed")
    require(int(getattr(scaler, "n_features_in_", -1)) == 81, "scaler input dimension mismatch")
    require(np.isfinite(np.asarray(scaler.mean_, dtype=np.float64)).all() and np.isfinite(np.asarray(scaler.scale_, dtype=np.float64)).all(), "scaler parameters nonfinite")

    p1_core, _, train_models = load_frozen_p1_modules(p1_root)
    expected_spec = p1_core.feature_spec()
    require(feature_config.get("columns") == expected_spec, "feature_config columns differ from frozen P1 feature spec")
    require(feature_names == [column["name"] for column in expected_spec], "scaler feature-name order mismatch")
    require(np.array_equal(standardize, np.asarray([bool(column["standardize_fit_only"]) for column in expected_spec], dtype=bool)), "scaler mask differs from frozen P1 feature spec")
    entries = _temperature_entries(temperature_payload)
    feature_config_sha = sha256_file(feature_config_path)
    class_config_sha = sha256_file(class_config_path)
    require(str(temperature_payload.get("feature_config_sha256", "")) == feature_config_sha, "temperature/feature-config SHA mismatch")
    models: dict[int, torch.nn.Module] = {}
    model_bindings: dict[str, Any] = {}
    for seed in ROUTE_A_SEEDS:
        path = root / "model_checkpoints" / f"marginal_mlp_seed_{seed}.pt"
        require(path.is_file(), f"missing model checkpoint: {path}")
        model_sha = sha256_file(path)
        require(str(entries[seed].get("model_sha256", "")) == model_sha, f"temperature/model SHA mismatch: {seed}")
        snapshot = torch.load(path, map_location="cpu", weights_only=True)
        require(int(snapshot.get("seed", -1)) == seed and int(snapshot.get("input_dim", -1)) == 90, f"model metadata mismatch: {seed}")
        require(str(snapshot.get("architecture", "")) == "Linear(90,128)-LayerNorm-GELU-Linear(128,64)-GELU-Linear(64,10)", f"model architecture mismatch: {seed}")
        require(snapshot.get("no_final_refit") is True, f"model has unexpected final refit: {seed}")
        require(str(snapshot.get("feature_config_sha256", "")) == feature_config_sha, f"model/feature-config SHA mismatch: {seed}")
        require(str(snapshot.get("class_weight_config_sha256", "")) == class_config_sha, f"model/class-weight-config SHA mismatch: {seed}")
        require(str(snapshot.get("pca_sha256", "")) == pca_sha and str(snapshot.get("scaler_sha256", "")) == scaler_sha, f"model feature-transform SHA mismatch: {seed}")
        model = train_models.MarginalMLP(90)
        model.load_state_dict(snapshot["state_dict"], strict=True)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model.eval().to(device)
        models[seed] = model
        model_bindings[str(seed)] = {"path": str(path.resolve()), "sha256": model_sha, "temperature": temperatures[seed]}

    bindings = {
        "feature_config": {"path": str(feature_config_path.resolve()), "sha256": sha256_file(feature_config_path)},
        "class_weight_config": {"path": str(class_config_path.resolve()), "sha256": sha256_file(class_config_path)},
        "temperature_params": {"path": str(temperature_path.resolve()), "sha256": sha256_file(temperature_path)},
        "pca_model": {"path": str(pca_path.resolve()), "sha256": pca_sha},
        "scaler": {"path": str(scaler_path.resolve()), "sha256": scaler_sha},
        "models": model_bindings,
        "training_complete": {"path": str((root / "working" / "training_complete.json").resolve()), "sha256": sha256_file(root / "working" / "training_complete.json")},
        "training_ledger": {"path": str((root / "working" / "training_sha256_ledger.csv").resolve()), "sha256": sha256_file(root / "working" / "training_sha256_ledger.csv")},
        "training_history": {"path": str((root / "training_history.csv").resolve()), "sha256": sha256_file(root / "training_history.csv")},
        "allocator_asset_manifest": {"path": str((root / "allocator_asset_manifest.json").resolve()), "sha256": sha256_file(root / "allocator_asset_manifest.json"), "allocator_asset_id": training_marker["allocator_asset_id"]},
    }
    return pca, scaler, standardize, weights, temperatures, models, feature_config, bindings


def read_val_features(
    rb: Any,
    asset_root: Path,
    candidate_manifest: list[dict[str, str]],
    native_manifest: list[dict[str, str]],
    image_manifest_rows: list[dict[str, str]],
    group_map: Mapping[str, int],
    p1_core: Any,
    pca: Any,
    scaler: Any,
    standardize: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, slice], pd.DataFrame]:
    image_manifest = pd.DataFrame(image_manifest_rows)
    image_manifest["canonical_image_id"] = image_manifest["canonical_image_id"].astype(str)
    image_manifest["coco_image_id"] = image_manifest["coco_image_id"].astype(np.int64)
    image_manifest["width"] = image_manifest["width"].astype(np.int64)
    image_manifest["height"] = image_manifest["height"].astype(np.int64)
    image_meta = image_manifest.set_index("canonical_image_id", drop=False)
    require(len(image_meta) == EXPECTED_IMAGES, "image manifest count mismatch")

    feature_count = EXPECTED_IMAGES * (K_MAX - K_MIN)
    features = np.empty((feature_count, 90), dtype=np.float32)
    feature_classes = np.empty(feature_count, dtype=np.int16)
    image_slices: dict[str, slice] = {}
    candidate_parts: list[pd.DataFrame] = []
    native_by_id = {row["shard_id"]: row for row in native_manifest}
    row0 = 0

    for shard_number, candidate_row in enumerate(candidate_manifest, start=1):
        shard_id = candidate_row["shard_id"]
        require(shard_id in native_by_id, f"candidate/native shard pair missing: {shard_id}")
        candidate_path = asset_root / candidate_row["path"]
        native_path = asset_root / native_by_id[shard_id]["path"]
        frame = pq.read_table(candidate_path, columns=rb.CANDIDATE_COLUMNS, filters=[("road8_rank", "<=", TOP_N)]).to_pandas()
        require(len(frame) == int(candidate_row["image_count"]) * TOP_N, f"Top100 shard row count mismatch: {shard_id}")
        require(set(frame["candidate_asset_id"].astype(str)) == {CANDIDATE_ASSET_ID}, f"candidate identity mismatch: {shard_id}")
        require(set(frame["dataset_version"].astype(str)) == {DATASET_VERSION} and set(frame["split"].astype(str)) == {SPLIT}, f"candidate dataset/split mismatch: {shard_id}")

        with np.load(native_path, allow_pickle=False) as native:
            native_image_ids = native["image_ids"].astype(str)
            native_query = native["query_index"].astype(np.int32)
            native_logits = native["l3_road8_logits"]
            native_embedding = native["l3_query_embedding"]
            require(native_logits.ndim == 2 and native_logits.shape[1] == 8, f"native logit shape mismatch: {shard_id}")
            require(native_embedding.ndim == 2 and native_embedding.shape[1] == 256, f"native embedding shape mismatch: {shard_id}")
            starts: dict[str, int] = {}
            for start in range(0, len(native_image_ids), 300):
                ids = native_image_ids[start : start + 300]
                query = native_query[start : start + 300]
                require(len(ids) == 300 and len(set(ids)) == 1 and np.array_equal(query, np.arange(300, dtype=np.int32)), f"native block mismatch: {shard_id}:{start}")
                starts[str(ids[0])] = start

            for image_id, rows in frame.groupby("canonical_image_id", sort=False):
                image_id = str(image_id)
                rows = rows.sort_values("road8_rank", kind="mergesort").reset_index(drop=True)
                require(len(rows) == TOP_N and rows["road8_rank"].astype(int).tolist() == list(range(1, TOP_N + 1)), f"Top100 rank mismatch: {image_id}")
                require(image_id in starts and image_id in image_meta.index and image_id in group_map, f"candidate/native/manifest join mismatch: {image_id}")
                queries = rows["query_index"].to_numpy(np.int32)
                native_rows = starts[image_id] + queries
                require(np.all(native_image_ids[native_rows] == image_id) and np.array_equal(native_query[native_rows], queries), f"native gather mismatch: {image_id}")
                logits = np.asarray(native_logits[native_rows], dtype=np.float32)
                embeddings = np.asarray(native_embedding[native_rows], dtype=np.float16)
                classes = rows["predicted_road8_class_id"].to_numpy(np.int16)
                require(np.all((classes >= 1) & (classes <= 8)), f"candidate class range mismatch: {image_id}")
                require(rows["predicted_road8_class_name"].astype(str).tolist() == [rb.ROAD8[value - 1] for value in classes], f"candidate class identity mismatch: {image_id}")
                require(np.all((queries >= 0) & (queries < 300)), f"query range mismatch: {image_id}")
                require((rows["image_id"].astype(str) == image_id).all(), f"image identity mismatch: {image_id}")
                meta = image_meta.loc[image_id]
                require((rows["coco_image_id"].to_numpy(np.int64) == int(meta["coco_image_id"])).all(), f"COCO image identity mismatch: {image_id}")
                reconstructed = p1_core.sigmoid(logits[np.arange(TOP_N), classes - 1])
                require(float(np.max(np.abs(reconstructed - rows["score"].to_numpy(np.float64)))) <= 1e-6, f"score reconstruction mismatch: {image_id}")
                raw = p1_core.build_raw_features(rows, logits, embeddings, pca, float(meta["width"]), float(meta["height"]))
                require(raw.shape == (45, 90) and np.isfinite(raw).all(), f"raw feature mismatch: {image_id}")
                scaled = raw.copy()
                scaled[:, standardize] = scaler.transform(scaled[:, standardize])
                require(np.isfinite(scaled).all(), f"scaled feature nonfinite: {image_id}")
                features[row0 : row0 + 45] = scaled.astype(np.float32)
                feature_classes[row0 : row0 + 45] = classes[5:50]
                image_slices[image_id] = slice(row0, row0 + 45)
                row0 += 45

        keep = frame[[
            "dataset_version", "split", "candidate_asset_id", "image_id", "canonical_image_id", "coco_image_id",
            "candidate_record_id", "road8_rank", "query_index", "predicted_road8_class_id", "score",
        ]].copy()
        keep["group_id"] = keep["canonical_image_id"].astype(str).map(group_map).astype(np.int16)
        candidate_parts.append(keep)
        progress("feature_construction", f"processed canonical shard {shard_number}/{len(candidate_manifest)}")

    require(row0 == feature_count and len(image_slices) == EXPECTED_IMAGES, "feature image coverage mismatch")
    candidates = pd.concat(candidate_parts, ignore_index=True)
    candidates["canonical_image_id"] = candidates["canonical_image_id"].astype(str)
    candidates["image_id"] = candidates["image_id"].astype(str)
    candidates = candidates.sort_values(["canonical_image_id", "road8_rank"], kind="mergesort").reset_index(drop=True)
    require(len(candidates) == EXPECTED_IMAGES * TOP_N and candidates["candidate_record_id"].nunique() == len(candidates), "candidate pool cardinality/identity mismatch")
    return features, feature_classes, image_slices, candidates


def infer_margins(
    features: np.ndarray,
    feature_classes: np.ndarray,
    image_slices: Mapping[str, slice],
    models: Mapping[int, torch.nn.Module],
    temperatures: Mapping[int, float],
    weights: np.ndarray,
    device: torch.device,
) -> tuple[dict[int, dict[str, np.ndarray]], dict[str, str], dict[str, bool]]:
    tensor = torch.from_numpy(features).to(device)
    margins: dict[int, dict[str, np.ndarray]] = {}
    prediction_hashes: dict[str, str] = {}
    repeat_exact: dict[str, bool] = {}
    for seed in ROUTE_A_SEEDS:
        parts: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(tensor), 16_384):
                parts.append(models[seed](tensor[start : start + 16_384]).float().cpu().numpy())
        logits = np.vstack(parts).astype(np.float64)
        repeat_count = min(16_384, len(tensor))
        with torch.inference_mode():
            repeated = models[seed](tensor[:repeat_count]).float().cpu().numpy().astype(np.float64)
        repeat_exact[str(seed)] = bool(np.array_equal(logits[:repeat_count], repeated))
        require(repeat_exact[str(seed)], f"repeat inference mismatch: {seed}")
        probabilities = 1.0 / (1.0 + np.exp(-(logits / float(temperatures[seed]))))
        require(probabilities.shape == (len(features), 10) and np.isfinite(probabilities).all(), f"probability shape/nonfinite: {seed}")
        values = probabilities.mean(axis=1) * weights[feature_classes - 1]
        require(np.isfinite(values).all() and np.all(values >= 0), f"M11 value invalid: {seed}")
        margins[seed] = {image_id: np.asarray(values[image_slices[image_id]], dtype=np.float64) for image_id in image_slices}
        prediction_hashes[str(seed)] = sha256_array(probabilities)
        progress("model_inference", f"seed {seed} complete")
    del tensor
    return margins, prediction_hashes, repeat_exact


def allocate(
    dp_solver: Any,
    groups: pd.DataFrame,
    candidates: pd.DataFrame,
    m11_margins: Mapping[int, Mapping[str, np.ndarray]],
) -> tuple[pd.DataFrame, dict[tuple[str, int, int, str], int]]:
    candidate_by_image = {
        str(image_id): frame.sort_values("road8_rank", kind="mergesort").reset_index(drop=True)
        for image_id, frame in candidates.groupby("canonical_image_id", sort=False)
    }
    score_margins = {image_id: frame.iloc[5:50]["score"].to_numpy(np.float64) for image_id, frame in candidate_by_image.items()}
    cal_margins: dict[str, np.ndarray] = {}
    for image_id, scores in score_margins.items():
        clipped = np.clip(scores, 1e-6, 1 - 1e-6)
        cal_margins[image_id] = 1.0 / (1.0 + np.exp(-(np.log(clipped / (1 - clipped)) / CAL_TEMP)))

    rows: list[dict[str, Any]] = []
    allocation_map: dict[tuple[str, int, int, str], int] = {}
    for group_id, frame in groups.groupby("group_id", sort=True):
        image_ids = sorted(frame["canonical_image_id"].astype(str).tolist())
        require(len(image_ids) == GROUP_SIZE, f"group size mismatch: {group_id}")
        for budget in BUDGETS:
            for image_id in image_ids:
                allocation_map[("S_FIXED", BASELINE_SEED, budget, image_id)] = budget
                rows.append({"method": "S_FIXED", "seed": BASELINE_SEED, "budget": budget, "group_id": int(group_id), "canonical_image_id": image_id, "K_i": budget, "group_predicted_objective": math.nan})
        policies: list[tuple[str, int, Mapping[str, np.ndarray]]] = [
            ("S_ADAPT", BASELINE_SEED, score_margins),
            ("CAL_TEMP_ALLOC", BASELINE_SEED, cal_margins),
            *[("M11-COCO", seed, m11_margins[seed]) for seed in ROUTE_A_SEEDS],
        ]
        for method, seed, per_image in policies:
            matrix = np.vstack([per_image[image_id] for image_id in image_ids]).astype(np.float64)
            solved, objectives = dp_solver.solve_group_allocations(matrix, BUDGETS)
            for budget_index, budget in enumerate(BUDGETS):
                for image_index, image_id in enumerate(image_ids):
                    k_value = int(solved[budget_index, image_index])
                    allocation_map[(method, seed, budget, image_id)] = k_value
                    rows.append({
                        "method": method, "seed": seed, "budget": budget, "group_id": int(group_id),
                        "canonical_image_id": image_id, "K_i": k_value,
                        "group_predicted_objective": float(objectives[budget_index]),
                    })
    allocation = stable_condition_sort(pd.DataFrame(rows))
    require(len(allocation) == len(condition_instances()) * len(BUDGETS) * EXPECTED_IMAGES, "allocation row count mismatch")
    require(allocation["K_i"].between(K_MIN, K_MAX).all(), "K bounds mismatch")
    sums = allocation.groupby(["method", "seed", "budget", "group_id"], sort=True)["K_i"].sum()
    require(np.array_equal(sums.to_numpy(np.int64), sums.index.get_level_values("budget").to_numpy(np.int64) * GROUP_SIZE), "exact group budget mismatch")
    return allocation, allocation_map


def materialize_selections(
    rb: Any,
    staging: Path,
    candidates: pd.DataFrame,
    allocation: pd.DataFrame,
    allocation_map: Mapping[tuple[str, int, int, str], int],
) -> tuple[Path, pd.DataFrame]:
    candidate_by_image = sorted(set(candidates["canonical_image_id"].astype(str)))
    selected_path = staging / "selected_records.parquet"
    writer: pq.ParquetWriter | None = None
    manifest_rows: list[dict[str, Any]] = []
    try:
        for method, seed, value_semantics in condition_instances():
            for budget in BUDGETS:
                k_by_image = {image_id: allocation_map[(method, seed, budget, image_id)] for image_id in candidate_by_image}
                k_values = candidates["canonical_image_id"].map(k_by_image).to_numpy(np.int16)
                selected = candidates[candidates["road8_rank"].to_numpy(np.int16) <= k_values].copy()
                selected["method"] = method
                selected["seed"] = seed
                selected["policy_instance_id"] = policy_id(method, seed)
                selected["value_semantics"] = value_semantics
                selected["budget"] = budget
                selected["K_i"] = selected["canonical_image_id"].map(k_by_image).astype(np.int16)
                selected["selection_rank"] = selected["road8_rank"].astype(np.int16)
                selected = selected.sort_values(["group_id", "canonical_image_id", "road8_rank"], kind="mergesort").reset_index(drop=True)
                selected = selected[[field.name for field in rb.SELECTED_SCHEMA]]
                require(len(selected) == EXPECTED_IMAGES * budget, f"selected row count mismatch: {(method,seed,budget)}")
                require(not selected.duplicated(["canonical_image_id", "candidate_record_id"]).any(), f"duplicate selected record: {(method,seed,budget)}")
                table = pa.Table.from_pandas(selected, schema=rb.SELECTED_SCHEMA, preserve_index=False, safe=True)
                if writer is None:
                    writer = pq.ParquetWriter(selected_path, rb.SELECTED_SCHEMA, compression="zstd")
                writer.write_table(table, row_group_size=len(table))
                manifest_rows.append({
                    "dataset": DATASET_VERSION, "split": SPLIT, "candidate_asset_id": CANDIDATE_ASSET_ID,
                    "method": method, "seed": seed, "policy_instance_id": policy_id(method, seed),
                    "value_semantics": value_semantics, "budget": budget, "group_count": EXPECTED_GROUPS,
                    "image_count": EXPECTED_IMAGES, "allocation_rows": EXPECTED_IMAGES,
                    "selected_rows": len(selected), "condition_content_sha256": rb.content_sequence_sha(selected),
                    "selected_records_path": "selected_records.parquet", "selected_records_sha256": "PENDING",
                    "status": "FROZEN_BEFORE_GT_EVALUATION", "gt_path_available_to_selection_entrypoint": False,
                })
                progress("selection_materialization", condition_label(method, seed, budget))
    finally:
        if writer is not None:
            writer.close()
    require(selected_path.is_file(), "selected records were not written")
    selected_sha = sha256_file(selected_path)
    for row in manifest_rows:
        row["selected_records_sha256"] = selected_sha
    manifest = stable_condition_sort(pd.DataFrame(manifest_rows))
    require(len(manifest) == 30, "selection manifest row count mismatch")
    write_csv_atomic(staging / "selection_manifest.csv", manifest)
    pq.write_table(pa.Table.from_pandas(allocation, preserve_index=False), staging / "selection_allocations.parquet", compression="zstd")
    return selected_path, manifest


def publish(output_root: Path, staging: Path, names: list[str], completion: dict[str, Any]) -> None:
    destinations = [output_root / name for name in names]
    destinations.append(output_root / "working" / "selection_allocations.parquet")
    require(not any(path.exists() for path in destinations), f"selection output collision: {[str(p) for p in destinations if p.exists()]}")
    for name in names:
        os.replace(staging / name, output_root / name)
    os.replace(staging / "selection_allocations.parquet", output_root / "working" / "selection_allocations.parquet")
    write_json_atomic(output_root / "ROUTE_A_SELECTION_COMPLETE.json", completion)
    shutil.rmtree(staging)


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    output_root = args.output_root.resolve()
    asset_root = args.asset_root.resolve(strict=True)
    group_manifest = args.group_manifest.resolve(strict=True)
    p1_root = args.p1_root.resolve(strict=True)
    rb_path = args.route_b_runner.resolve(strict=True)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "working").mkdir(parents=True, exist_ok=True)
    require(not (output_root / "ROUTE_A_SELECTION_COMPLETE.json").exists(), "Route-A selection is already complete")
    planned = (
        "selection_manifest.csv", "selected_records.parquet", "predicted_utility_curves.parquet",
        "routea_selection_config.json", "selection_QA.json", "allocation_summary.csv",
    )
    require(not any((output_root / name).exists() for name in planned), "partial Route-A selection publication refused")
    require(not (output_root / "working" / "selection_allocations.parquet").exists(), "selection allocation output already exists")
    staging = output_root / "working" / ("staging_routea_selection_" + CANDIDATE_ASSET_ID[:12])
    require(not staging.exists() or not any(staging.iterdir()), f"nonempty staging refused: {staging}")
    staging.mkdir(parents=True, exist_ok=True)

    require(sha256_file(rb_path) == ROUTE_B_SELECTION_RUNNER_SHA256, "frozen Route-B selection runner SHA mismatch")
    rb = load_module("routea_readonly_routeb_runner", rb_path)
    marker, candidate_manifest, native_manifest, image_manifest_rows = rb.verify_canonical_asset(asset_root)
    groups, group_map = rb.verify_groups(group_manifest, image_manifest_rows)
    p1_core, dp_solver, _ = load_frozen_p1_modules(p1_root)
    if str(args.device).startswith("cuda"):
        require(torch.cuda.is_available(), "requested CUDA device is unavailable")
    device = torch.device(args.device)

    pca, scaler, standardize, weights, temperatures, models, feature_config, routea_bindings = load_routea_assets(output_root, p1_root, device)
    selection_config = {
        "task": TASK,
        "created_at_utc": utc_now(),
        "dataset_version": DATASET_VERSION,
        "split": SPLIT,
        "candidate_asset_id": CANDIDATE_ASSET_ID,
        "route": "A_COCO_ADAPTED_M11",
        "methods": list(dict.fromkeys(method for method, _, _ in condition_instances())),
        "m11_seeds": list(ROUTE_A_SEEDS),
        "m11_seed_predictions_averaged": False,
        "budgets": list(BUDGETS),
        "k_bounds": [K_MIN, K_MAX],
        "group_size": GROUP_SIZE,
        "group_count": EXPECTED_GROUPS,
        "candidate_action": "canonical original Top100; original-score prefix rank 1..K only",
        "solver": "frozen exact float64 multiple-choice DP with lexicographically smallest K-vector tie break",
        "curve_semantics": "k<5 undefined; U(5)=0; modeled calibrated weighted deltas and cumulative relative U for k=6..50",
        "cal_temp": CAL_TEMP,
        "class_weights": weights.tolist(),
        "temperatures": {str(seed): temperatures[seed] for seed in ROUTE_A_SEEDS},
        "access_boundary": {"gt_or_annotation_argument": False, "gt_read": False, "matching": False, "scientific_evaluation": False},
        "input_bindings": {
            **routea_bindings,
            "canonical_completion_marker": {"path": str((asset_root / "CANONICAL_EXPORT_COMPLETE.json").resolve()), "sha256": sha256_file(asset_root / "CANONICAL_EXPORT_COMPLETE.json")},
            "group_manifest": {"path": str(group_manifest), "sha256": sha256_file(group_manifest)},
            "p1_core": {"path": str(Path(p1_core.__file__).resolve()), "sha256": sha256_file(Path(p1_core.__file__))},
            "dp_solver": {"path": str(Path(dp_solver.__file__).resolve()), "sha256": sha256_file(Path(dp_solver.__file__))},
            "routea_selection_runner": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__).resolve())},
            "routeb_selection_runner_readonly": {"path": str(rb_path), "sha256": sha256_file(rb_path)},
        },
        "canonical_marker_status": marker["status"],
        "feature_config_sha256": sha256_file(output_root / "feature_config.json"),
        "environment": {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__, "device": str(device), "numpy": np.__version__, "pyarrow": pa.__version__},
    }
    write_json_atomic(staging / "routea_selection_config.json", selection_config)
    progress("preflight", "Route-A assets and frozen VAL identity verified; GT remains unavailable")

    features, feature_classes, image_slices, candidates = read_val_features(
        rb, asset_root, candidate_manifest, native_manifest, image_manifest_rows, group_map,
        p1_core, pca, scaler, standardize,
    )
    margins, prediction_hashes, repeat_exact = infer_margins(
        features, feature_classes, image_slices, models, temperatures, weights, device,
    )
    image_ids = sorted(image_slices)
    curves = curve_frame(image_ids, margins)
    pq.write_table(pa.Table.from_pandas(curves, preserve_index=False), staging / "predicted_utility_curves.parquet", compression="zstd")
    allocation, allocation_map = allocate(dp_solver, groups, candidates, margins)
    selected_path, manifest = materialize_selections(rb, staging, candidates, allocation, allocation_map)

    s = allocation[allocation.method == "S_ADAPT"].sort_values(["budget", "canonical_image_id"])["K_i"].to_numpy(np.int16)
    c = allocation[allocation.method == "CAL_TEMP_ALLOC"].sort_values(["budget", "canonical_image_id"])["K_i"].to_numpy(np.int16)
    qa = {
        "status": "PASS",
        "images": EXPECTED_IMAGES,
        "groups": EXPECTED_GROUPS,
        "conditions": len(manifest),
        "allocation_rows": len(allocation),
        "selected_rows": int(manifest["selected_rows"].sum()),
        "selected_records_sha256": sha256_file(selected_path),
        "predicted_utility_curves_sha256": sha256_file(staging / "predicted_utility_curves.parquet"),
        "prediction_probability_sha256": prediction_hashes,
        "repeat_inference_exact_first_production_batch": repeat_exact,
        "cal_temp_s_adapt_k_exact_parity": bool(np.array_equal(s, c)),
        "gt_or_annotation_read": False,
        "detector_forward": False,
        "model_training": False,
    }
    require(qa["cal_temp_s_adapt_k_exact_parity"], "CAL_TEMP_ALLOC selection differs from S_ADAPT")
    write_json_atomic(staging / "selection_QA.json", qa)

    summary = allocation.groupby(["method", "seed", "budget"], as_index=False).agg(
        image_count=("canonical_image_id", "size"), K_mean=("K_i", "mean"), K_std=("K_i", "std"),
        K_min=("K_i", "min"), K_max=("K_i", "max"),
    )
    objective = (
        allocation[["method", "seed", "budget", "group_id", "group_predicted_objective"]]
        .drop_duplicates(["method", "seed", "budget", "group_id"])
        .groupby(["method", "seed", "budget"], as_index=False)["group_predicted_objective"]
        .sum(min_count=1)
        .rename(columns={"group_predicted_objective": "predicted_objective_group_sum"})
    )
    summary = summary.merge(objective, on=["method", "seed", "budget"], how="left", validate="one_to_one")
    write_csv_atomic(staging / "allocation_summary.csv", stable_condition_sort(summary))
    names = [
        "selection_manifest.csv", "selected_records.parquet", "predicted_utility_curves.parquet",
        "routea_selection_config.json", "selection_QA.json", "allocation_summary.csv",
    ]
    completion = {
        "task": TASK,
        "status": "ROUTE_A_SELECTION_COMPLETE",
        "created_at_utc": utc_now(),
        "candidate_asset_id": CANDIDATE_ASSET_ID,
        "gt_accessed": False,
        "files": {name: sha256_file(staging / name) for name in names},
        "elapsed_seconds": time.perf_counter() - started,
    }
    publish(output_root, staging, names, completion)
    append_log(output_root, "ROUTE_A_SELECTION_COMPLETE")
    return completion


def self_test(p1_root: Path) -> dict[str, Any]:
    common = self_test_common()
    _, dp_solver, _ = load_frozen_p1_modules(p1_root.resolve(strict=True))
    rng = np.random.default_rng(630100)
    margins = rng.random((40, 45), dtype=np.float64)
    solved, objectives = dp_solver.solve_group_allocations(margins, BUDGETS)
    require(solved.shape == (5, 40) and objectives.shape == (5,), "synthetic DP shape")
    require(np.array_equal(solved.sum(axis=1), np.asarray(BUDGETS) * 40), "synthetic exact budget")
    return {"status": "PASS", "common": common, "dp_k_sums": solved.sum(axis=1).tolist()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=ROOT_DEFAULT)
    parser.add_argument("--asset-root", type=Path, default=ASSET_ROOT_DEFAULT)
    parser.add_argument("--group-manifest", type=Path, default=GROUP_MANIFEST_DEFAULT)
    parser.add_argument("--p1-root", type=Path, default=P1_ROOT_DEFAULT)
    parser.add_argument("--route-b-runner", type=Path, default=ROUTE_B_RUNNER_DEFAULT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--self-test", action="store_true", help="run synthetic tests only; reads no dataset or GT")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    result = self_test(arguments.p1_root) if arguments.self_test else run(arguments)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
