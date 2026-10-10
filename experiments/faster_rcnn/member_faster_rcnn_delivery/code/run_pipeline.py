# -*- coding: utf-8 -*-
u"""在 Faster R-CNN 候选流上跑完整的 M11 流水线。

复用原则: 能调交接包原函数的一律原样调用 (标签、训练、温度校准、DP、评价、bootstrap),
只有**必须换检测器配方**的两处自己实现:
  1. M11 分配前的特征构造 (`feature_frame_frcnn`) —— 包内 `_feature_frame` 写死了
     256 维 L3 embedding 的 RT-DETR 配方;
  2. `allocate` 的 M11 分支 (DP 本身 `allocate_m11` 原样复用)。

冻结量一个不改: Road8 映射、前缀标签、十阈值、类别权重定义、`5<=K<=50`、
40 图精确预算、原分数前缀动作、DP 与评价口径。

用法:
  python -B lc_frcnn/run_pipeline.py --stage features
  python -B lc_frcnn/run_pipeline.py --stage train
  python -B lc_frcnn/run_pipeline.py --stage calibrate
  python -B lc_frcnn/run_pipeline.py --stage assets
  python -B lc_frcnn/run_pipeline.py --stage allocate
  python -B lc_frcnn/run_pipeline.py --stage evaluate
  python -B lc_frcnn/run_pipeline.py --stage all
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from argparse import Namespace
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
M11_ROOT = HERE.parent
EXPERIMENT_ROOT = Path(os.environ.get('M11_EXPERIMENT_ROOT', "external_assets"))
HANDOFF = EXPERIMENT_ROOT / 'LC_ALLOC_M11_HANDOFF_v1/handoff_LC_ALLOC_M11_v1'
sys.path.insert(0, str(M11_ROOT))
sys.path.insert(0, str(HANDOFF))

from lc_alloc.allocation.policies import allocate_m11, allocate_s_adapt, allocate_s_fixed, materialize_prefixes  # noqa: E402
from lc_alloc.cli import (  # noqa: E402
    _validate_groups, command_calibrate, command_evaluate, command_make_labels, command_train,
)
from lc_alloc.constants import FEATURE_DIM  # noqa: E402
from lc_alloc.data.io import read_candidates, read_native  # noqa: E402
from lc_alloc.data.schema import NativeTable  # noqa: E402
from lc_alloc.features._executed_p1_core import compute_class_weights  # noqa: E402
from lc_alloc.labels.prefix import load_normalized_gt  # noqa: E402
from lc_alloc.models.assets import load_allocator_assets, predict_probabilities  # noqa: E402
from lc_alloc.utils import sha256_file, write_json  # noqa: E402

from lc_frcnn import coco_eval  # noqa: E402
from lc_frcnn.features_frcnn import (  # noqa: E402
    NATIVE_SCHEMA_ID, PCA_COMPONENTS,
    build_image_features, feature_spec, fit_pca, fit_scaler, transform_features,
)

RELEASE = EXPERIMENT_ROOT / 'AOP_Road8_LC_v1_RTDERTv2_R18VD_RELEASE_v1'
SEEDS = (730101, 730102, 730103)   # FRCNN 臂自己的种子命名空间, 不与 53xxxx/63xxxx 混用
BUDGETS = (10, 15, 20, 30, 40)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _native_subset(native: NativeTable, image_id: str) -> NativeTable:
    mask = np.asarray(native.image_ids).astype(str) == str(image_id)
    return NativeTable(
        image_ids=np.asarray(native.image_ids)[mask],
        source_ids=np.asarray(native.source_ids)[mask],
        native_vectors=np.asarray(native.native_vectors)[mask],
        class_signals=np.asarray(native.class_signals)[mask],
        native_schema_id=native.native_schema_id,
    )


def feature_frame_frcnn(candidates, native, pca, scaler_bundle, roles: pd.DataFrame | None = None) -> pd.DataFrame:
    u"""包内 `_feature_frame` 的 Faster R-CNN 版本 —— 唯一改动是特征构造器。"""
    role_map = {} if roles is None else dict(zip(roles.image_id, roles.role))
    parts = []
    for image_id in sorted(set(candidates["image_id"].astype(str))):
        frame = candidates[candidates["image_id"].astype(str) == image_id]
        raw = build_image_features(frame, _native_subset(native, image_id), pca)
        scaled = transform_features(raw, scaler_bundle)
        part = pd.DataFrame(scaled, columns=[f"f{index:03d}" for index in range(FEATURE_DIM)])
        part.insert(0, "rank", np.arange(6, 51, dtype=np.int16))
        part.insert(0, "role", role_map.get(image_id, "UNSPECIFIED"))
        part.insert(0, "image_id", image_id)
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


# ── stages ────────────────────────────────────────────────────────────────
def stage_features(work: Path, prepared: Path) -> dict:
    u"""FIT-only PCA32 + scaler, 然后给 TRAIN 与 DEV 都算特征。"""
    out = work
    out.mkdir(parents=True, exist_ok=True)
    roles = pd.read_csv(prepared / "roles.csv")
    fit_ids = set(roles.loc[roles.role == "FIT", "image_id"].astype(str))

    # --- TRAIN: 拟合 PCA/scaler 并输出 features ---
    tr_cand = read_candidates(prepared / "candidates_train.parquet")
    tr_native = read_native(prepared / "native_train.npz", expected_schema_id=NATIVE_SCHEMA_ID)
    fit_keys = {
        (str(r.image_id), str(r.source_id))
        for r in tr_cand[tr_cand.image_id.astype(str).isin(fit_ids)].itertuples(index=False)
    }
    fit_mask = np.asarray(
        [(str(a), str(b)) in fit_keys for a, b in zip(tr_native.image_ids, tr_native.source_ids)], dtype=bool
    )
    pca = fit_pca(np.asarray(tr_native.image_ids)[fit_mask], np.asarray(tr_native.source_ids)[fit_mask],
                  np.asarray(tr_native.native_vectors)[fit_mask],
                  namespace="LC_ALLOC_FRCNN_PCA|", random_state=730100)
    log(f"PCA32 fit on {int(fit_mask.sum())} FIT native rows; "
        f"explained variance = {float(pca.explained_variance_ratio_.sum()):.4f}")

    raw_fit = []
    present = set(tr_cand.image_id.astype(str))
    for image_id in sorted(fit_ids):
        if image_id not in present:
            continue
        raw_fit.append(build_image_features(
            tr_cand[tr_cand.image_id.astype(str) == image_id], _native_subset(tr_native, image_id), pca))
    scaler_bundle = fit_scaler(np.vstack(raw_fit))
    joblib.dump(pca, out / "pca32.joblib")
    joblib.dump(scaler_bundle, out / "scaler.joblib")

    features = feature_frame_frcnn(tr_cand, tr_native, pca, scaler_bundle, roles)
    features.to_parquet(out / "features_train.parquet", index=False)
    log(f"features_train: rows={len(features)} images={features.image_id.nunique()}")

    # --- DEV: 用同一套 FIT-only PCA/scaler ---
    dv_cand = read_candidates(prepared / "candidates_dev.parquet")
    dv_native = read_native(prepared / "native_dev.npz", expected_schema_id=NATIVE_SCHEMA_ID)
    dv_features = feature_frame_frcnn(dv_cand, dv_native, pca, scaler_bundle, None)
    dv_features.to_parquet(out / "features_dev.parquet", index=False)
    log(f"features_dev: rows={len(dv_features)} images={dv_features.image_id.nunique()}")

    write_json(out / "feature_schema.json", {
        "dimension": FEATURE_DIM, "columns": feature_spec(),
        "native_schema_id": NATIVE_SCHEMA_ID, "pca_components": PCA_COMPONENTS,
        "pca_explained_variance": float(pca.explained_variance_ratio_.sum()),
    })
    return {"pca_explained_variance": float(pca.explained_variance_ratio_.sum()),
            "fit_native_rows": int(fit_mask.sum())}


def stage_train(work: Path) -> dict:
    results = {}
    for seed in SEEDS:
        out = work / f"train_seed_{seed}"
        if out.is_dir() and any(out.iterdir()):
            log(f"seed {seed}: already trained, skip")
            continue
        info = command_train(Namespace(
            features=str(work / "features_train.parquet"), labels=str(work / "labels.parquet"),
            seed=seed, output_root=str(out), device="cpu", max_epochs=30, patience=5, batch_size=4096))
        log(f"seed {seed}: best_epoch={info['best_epoch']} epochs_run={info['epochs_run']}")
        results[str(seed)] = info
    return results


def stage_calibrate(work: Path) -> dict:
    results = {}
    for seed in SEEDS:
        model = work / f"train_seed_{seed}" / f"marginal_mlp_seed_{seed}.pt"
        out = work / f"calib_seed_{seed}"
        if (out / "temperature.json").is_file():
            log(f"seed {seed}: already calibrated, skip")
            continue
        info = command_calibrate(Namespace(
            model=str(model), features=str(work / "features_train.parquet"),
            labels=str(work / "labels.parquet"), output_root=str(out), device="cpu"))
        log(f"seed {seed}: T={info['temperature']:.6f}  {info.get('status')}")
        results[str(seed)] = info
    return results


def stage_labels(work: Path, prepared: Path) -> dict:
    target = work / "labels.parquet"
    if target.is_file():
        return {"status": "SKIPPED_EXISTS", "rows": int(len(pd.read_parquet(target)))}
    out = work / "labels_out"
    if out.is_dir() and any(out.iterdir()):
        raise FileExistsError(f"stale non-empty {out}; remove it before re-running the labels stage")
    info = command_make_labels(Namespace(
        candidates=str(prepared / "candidates_train.parquet"),
        gt=str(prepared / "gt_train_normalized.json"), gt_format="normalized",
        roles=str(prepared / "roles.csv"), output_root=str(out)))
    target.write_bytes((out / "labels.parquet").read_bytes())
    log(f"labels: {info}")
    return info


def stage_assets(work: Path, prepared: Path) -> dict:
    u"""把 PCA/scaler/三个 seed 模型/温度/类别权重打包成一个**本检测器专属**的资产目录。"""
    assets = work / "assets"
    (assets / "models").mkdir(parents=True, exist_ok=True)
    gt = load_normalized_gt(prepared / "gt_train_normalized.json")
    roles = pd.read_csv(prepared / "roles.csv")
    fit_ids = sorted(roles.loc[roles.role == "FIT", "image_id"].astype(str))
    counts, weights = compute_class_weights(gt, fit_ids)
    write_json(assets / "class_weights.json", {"counts": counts.tolist(), "weights": weights.tolist()})

    # 温度单独成文件: §08 把"温度"与 PCA/scaler/3 个模型并列为资产项。
    # 只把它塞进 asset.json 的某个子键里, 接手的人得先知道去哪找才算得出来。
    models = {}
    temperatures = {}
    for seed in SEEDS:
        src = work / f"train_seed_{seed}" / f"marginal_mlp_seed_{seed}.pt"
        dst = assets / "models" / f"marginal_mlp_seed_{seed}.pt"
        dst.write_bytes(src.read_bytes())
        temp = json.loads((work / f"calib_seed_{seed}" / "temperature.json").read_text(encoding="utf-8"))
        value = float(temp["temperature"])
        models[str(seed)] = {"path": f"models/marginal_mlp_seed_{seed}.pt",
                             "sha256": sha256_file(dst), "temperature": value}
        temperatures[str(seed)] = {
            "temperature": value,
            "fitted_on": "CALIBRATION split (1000 images); DEV2K never used for tuning",
            "source": f"calib_seed_{seed}/temperature.json",
            # §05 的十输出 BCE: 温度作用在**逐输出 sigmoid** 上, 不是 softmax。
            # 此处公式必须与共享实现 predict_probabilities() 及 predicted_utility.parquet
            # 的 prob_* 列逐位一致 —— 旧文案写的 softmax 按之复算会得到完全不同的概率 (r1 #12)。
            "applied_to": "allocation-time probabilities: sigmoid(logits / T) applied independently "
                          "to each of the ten IoU-threshold outputs; see prob_* columns "
                          "in predicted_utility.parquet",
        }
    write_json(assets / "temperatures.json", temperatures)

    # 这两个已经在 assets/ 里写好了, 其余从 work/ 拷进来。
    already = {"class_weights.json", "temperatures.json"}
    files = {
        "pca": {"path": "pca32.joblib", "sha256": sha256_file(work / "pca32.joblib")},
        "scaler": {"path": "scaler.joblib", "sha256": sha256_file(work / "scaler.joblib")},
        "class_weights": {"path": "class_weights.json",
                          "sha256": sha256_file(assets / "class_weights.json")},
        "temperatures": {"path": "temperatures.json",
                         "sha256": sha256_file(assets / "temperatures.json")},
    }
    for entry in files.values():
        if entry["path"] not in already:
            (assets / entry["path"]).write_bytes((work / entry["path"]).read_bytes())

    schema = json.loads((work / "feature_schema.json").read_text(encoding="utf-8"))
    identity = {
        "asset_name": "Faster R-CNN R18-FPN Road8 M11",
        "native_schema_id": NATIVE_SCHEMA_ID,
        "native_vector_dimension": 1024,
        "class_signal_dimension": 9,
        "feature_dimension": FEATURE_DIM,
        "output_dimension": 10,
        "files": files,
        "models": models,
        "models_are_independent_not_ensemble": True,
        "detector_id": "FasterRCNN-resnet18-FPN-Road8",
        "class_weights_source": "FRCNN FIT effective GT counts, same truncated inverse-sqrt formula as the reference",
        "pca_explained_variance": schema["pca_explained_variance"],
        "reference_detector_assets_reusable": False,
        "scientific_source": "frcnn_m11 overlay",
    }
    asset_id = __import__("hashlib").sha256(
        json.dumps(identity, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
    payload = {"asset_id": asset_id, **identity}
    write_json(assets / "asset.json", payload)
    log(f"assets -> {assets}  asset_id={asset_id[:16]}…  class counts={counts.tolist()}")
    return {"asset_id": asset_id, "class_counts": counts.tolist(), "weights": weights.tolist()}


def stage_allocate(work: Path, prepared: Path) -> dict:
    u"""M11 / S_ADAPT / S_FIXED 三种方法各分配一遍 (M11 走本检测器的特征)。"""
    candidates = read_candidates(prepared / "candidates_dev.parquet")
    native = read_native(prepared / "native_dev.npz", expected_schema_id=NATIVE_SCHEMA_ID)
    groups = _validate_groups(pd.read_csv(prepared / "groups_dev.csv"), candidates, BUDGETS)
    assets = load_allocator_assets(work / "assets", device="cpu")
    pca, scaler_bundle = assets.pca, assets.scaler_bundle
    features = feature_frame_frcnn(candidates, native, pca, scaler_bundle, None)

    sorted_ids = sorted(set(candidates.image_id.astype(str)))
    class_by_row = []
    for image_id in sorted_ids:
        rows = candidates[candidates.image_id.astype(str) == image_id].sort_values("rank", kind="mergesort")
        class_by_row.extend(rows.iloc[5:50].predicted_road8_class_id.astype(int).tolist())

    allocation_parts, selection_parts = [], []
    methods = [("M11", seed) for seed in SEEDS] + [("S_ADAPT", -1), ("S_FIXED", -1)]
    for method, seed in methods:
        if method == "M11":
            probabilities = predict_probabilities(
                assets.models[seed],
                features[[f"f{index:03d}" for index in range(FEATURE_DIM)]].to_numpy(np.float32),
                assets.temperatures[seed], device="cpu")
            values = probabilities.mean(axis=1) * assets.class_weights[np.asarray(class_by_row) - 1]
            value_by_image = {image_id: values[i * 45:(i + 1) * 45] for i, image_id in enumerate(sorted_ids)}
        for group_id, group in groups.groupby("group_id", sort=True):
            image_ids = sorted(group.image_id.astype(str))
            if method == "M11":
                solved, objectives = allocate_m11(np.vstack([value_by_image[v] for v in image_ids]), BUDGETS)
            elif method == "S_ADAPT":
                solved, objectives = allocate_s_adapt(candidates, image_ids, BUDGETS)
            else:
                solved = allocate_s_fixed(len(image_ids), BUDGETS)
                objectives = np.full(len(BUDGETS), np.nan)
            for bi, budget in enumerate(BUDGETS):
                if int(solved[bi].sum()) != len(image_ids) * int(budget):
                    raise AssertionError(f"exact group budget failed group={group_id} budget={budget}")
                allocation_parts.append(pd.DataFrame({
                    "group_id": group_id, "image_id": image_ids, "K_i": solved[bi],
                    "method": method, "seed": seed, "budget": budget,
                    "group_objective": objectives[bi]}))
                selected = materialize_prefixes(candidates, image_ids, solved[bi])
                selected["group_id"] = group_id
                selected["method"] = method
                selected["seed"] = seed
                selected["budget"] = budget
                selection_parts.append(selected)
        log(f"allocated method={method} seed={seed}")

    allocations = pd.concat(allocation_parts, ignore_index=True)
    # §08 交付格式: allocations.parquet 每条件每图必须携带候选资产身份。取值与
    # merge_records 给 main_results / per_group_results / class_results 的完全相同
    # (同一份冻结候选表的第一行), 只是常量列, 不参与任何计算。
    allocations.insert(0, "candidate_asset_id", str(candidates["candidate_asset_id"].iloc[0]))
    allocations.to_parquet(work / "allocations.parquet", index=False)
    pd.concat(selection_parts, ignore_index=True).to_parquet(work / "selected_records.parquet", index=False)
    log(f"allocations -> {work/'allocations.parquet'}  rows={len(allocations)}")

    write_utility_table(work, features, sorted_ids, class_by_row, assets, candidates)
    return {"allocation_rows": int(len(allocations)), "methods": [m for m, _ in methods]}


def write_utility_table(work: Path, features: pd.DataFrame, sorted_ids: list[str],
                        class_by_row: list[int], assets, candidates: pd.DataFrame) -> None:
    u"""§05 要求: 每个 seed、每图、k=6…50 保存十维原始 logits、温度后概率、
    加权边际值 delta_hat 和以 Uhat(5)=0 为锚的累计曲线。

    这里保存的是**模型真正预测出来的东西**, 不是拿 detector score 或 K_i 反推的替身。
    """
    import torch

    x = features[[f"f{index:03d}" for index in range(FEATURE_DIM)]].to_numpy(np.float32)
    weight_of_row = assets.class_weights[np.asarray(class_by_row) - 1]
    frames = []
    for seed in SEEDS:
        model = assets.models[seed]
        temperature = float(assets.temperatures[seed])
        with torch.inference_mode():
            logits = model(torch.from_numpy(x)).float().cpu().numpy().astype(np.float64)
        scaled = logits / temperature
        probabilities = 1.0 / (1.0 + np.exp(-scaled))
        delta = probabilities.mean(axis=1) * weight_of_row
        rows = {
            "model_id": f"frcnn_marginal_mlp_seed_{seed}",
            "weight_id": "frcnn_road8_fit_effective_weights_v1",
            "seed": seed,
            "temperature": temperature,
            "image_id": np.repeat(np.asarray(sorted_ids), 45),
            "k": np.tile(np.arange(6, 51, dtype=np.int64), len(sorted_ids)),
        }
        for index in range(logits.shape[1]):
            rows[f"logit_{index:02d}"] = logits[:, index]
        for index in range(probabilities.shape[1]):
            rows[f"prob_{index:02d}"] = probabilities[:, index]
        rows["delta_hat"] = delta
        frame = pd.DataFrame(rows)
        # 累计曲线以 Uhat_i(5)=0 为锚 —— 前五条是必选常数项, 不为 k=1..5 造模型预测。
        frame["Uhat"] = frame.groupby("image_id", sort=False)["delta_hat"].cumsum()
        frames.append(frame)
    table = pd.concat(frames, ignore_index=True)
    table.to_parquet(work / "predicted_utility.parquet", index=False)
    log(f"predicted_utility -> rows={len(table)} seeds={len(SEEDS)} "
        f"k_range={int(table.k.min())}..{int(table.k.max())}")


def stage_evaluate(work: Path, prepared: Path) -> dict:
    u"""覆盖/质量 + 标准 COCO AP/AR, 全部读**既有选集**, 不重选 seed、不重解配额、不改分数。

    评价器的原始输出一律落在 `evaluation/raw/` 下, `merge_records` 只读它、**从不覆写**。
    早先两者共用同一个目录, 合并时把原始表原地改名 (road8_class_id -> class_id) 并覆盖,
    于是这个阶段只能跑一次, 第二次会在 merge 的 key 上直接 KeyError —— 交付件要求可重跑。
    """
    out = work / "evaluation"
    raw = out / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    if not (raw / "main_results.csv").is_file():
        info = command_evaluate(Namespace(
            candidates=str(prepared / "candidates_dev.parquet"),
            allocations=str(work / "allocations.parquet"),
            gt=str(prepared / "gt_dev_normalized.json"), gt_format="normalized",
            class_weights=str(work / "assets" / "class_weights.json"),
            coco_ap=False, output_root=str(raw)))
        log(f"evaluate (coverage/quality) -> {json.dumps(info, ensure_ascii=False)[:300]}")

    if not (raw / "coco_ap_results.csv").is_file():
        candidates = read_candidates(prepared / "candidates_dev.parquet")
        allocations = pd.read_parquet(work / "allocations.parquet")
        gt_json = RELEASE / "gt" / "DEV2K_ROAD8_GT.json"
        rows, class_rows = [], []
        conditions = allocations.groupby(["method", "seed", "budget"], sort=True)
        for (method, seed, budget), frame in conditions:
            payload = coco_eval.evaluate_selection(candidates, frame, gt_json, with_per_class=True)
            identity = {"method": method, "seed": int(seed), "budget": int(budget)}
            rows.append({**identity, **{k: v for k, v in payload.items()
                                        if k not in ("eval_params", "by_class")}})
            for row in payload["by_class"]:
                class_rows.append({**identity, **row})
            log(f"  coco AP {method} seed={seed} budget={budget}: "
                f"AP={payload['AP']} AP50={payload['AP50']} AR100={payload['AR100']}")
        pd.DataFrame(rows).to_csv(raw / "coco_ap_results.csv", index=False)
        pd.DataFrame(class_rows).to_csv(raw / "coco_class_results.csv", index=False)
        write_json(raw / "coco_eval_params.json", coco_eval.COCO_EVAL_PARAMS)

    merge_records(work, prepared)
    return {"status": "PASS", "main_results": str(out / "main_results.csv")}


def _resolve_overlap(left: pd.DataFrame, right: pd.DataFrame, key: list[str], label: str) -> pd.DataFrame:
    u"""合并前处理同名列: 逐列比对, 相同就丢掉 right 那份, 不同就报错。

    直接 merge 的话 pandas 会给冲突列加 _x/_y 后缀, §08 点名要求的裸列名 (image_count、
    gt_count、total_records) 就不存在了, 还会出现两个同名 total_records。两边的口径本来
    就该一致, 所以这里**先验证再丢弃**, 不静默挑一边。
    """
    shared = [column for column in right.columns if column in left.columns and column not in key]
    for column in shared:
        right_values = pd.to_numeric(right.set_index(key)[column], errors="coerce")
        left_values = pd.to_numeric(left.set_index(key)[column], errors="coerce").reindex(right_values.index)
        if not np.allclose(left_values.to_numpy(dtype=float), right_values.to_numpy(dtype=float),
                           equal_nan=True):
            raise ValueError(f"{label}: the two sources disagree on '{column}'; refusing to pick a side")
    if shared:
        log(f"merge[{label}]: dropped {len(shared)} overlapping column(s) after verifying equality: {shared}")
    return right.drop(columns=shared)


def merge_records(work: Path, prepared: Path) -> None:
    u"""按 §08 的字段要求组装 main_results / per_group_results / class_results。

    只读 `evaluation/raw/` 下的**评价器原始输出**, 结果写到 `evaluation/` 顶层。
    两者分开是为了可重跑: 早先合并直接覆写原始表 (还把 road8_class_id 改名成 class_id),
    于是同一个阶段跑第二次就会在 merge 的 key 上 KeyError —— 交付件要求能重复运行。
    """
    out = work / "evaluation"
    raw = out / "raw"
    main = pd.read_csv(raw / "main_results.csv")
    coco = pd.read_csv(raw / "coco_ap_results.csv")
    candidates = read_candidates(prepared / "candidates_dev.parquet")
    allocations = pd.read_parquet(work / "allocations.parquet")
    per_image = pd.read_csv(raw / "per_image_results.csv")
    asset_id = json.loads((work / "assets" / "asset.json").read_text(encoding="utf-8"))["asset_id"]
    candidate_asset = str(candidates["candidate_asset_id"].iloc[0])
    protocol = str(candidates["protocol_id"].iloc[0]) if "protocol_id" in candidates else "PRE_NMS_TOPN100"
    # §08 的"以上条件身份"以 main_results.csv 那一行为准: dataset_id / detector_id 是
    # 全部条件表共用的常量列, class_results.csv 也必须带上, 取值同源 (冻结候选表首行)。
    dataset_id = str(candidates["dataset_version"].iloc[0])
    detector_id = str(candidates["detector_id"].iloc[0])

    # §06 的 Coverage_i / QUALITY_i 都是**每图**量, 所以汇总列取每图均值那份,
    # 而不是把 coverage_total / quality_total 冒充成同一件事。
    # **先改名再合并**: 评价器管它叫 output_records, coco 侧叫 total_records —— 若先合并,
    # 两列各自带着不同的名字活到合并之后, 再改名就撞成了两个同名的 total_records
    # (写出去变成 total_records / total_records.1)。改成同名再合并, 冲突才会被显式比对。
    main = main.rename(columns={"coverage_per_image": "Coverage",
                                "quality_per_image": "QUALITY",
                                "output_records": "total_records"})

    key = ["method", "seed", "budget"]
    coco = _resolve_overlap(main, coco, key, "main")
    main = main.merge(coco, on=key, how="left", validate="one_to_one")
    main.insert(0, "protocol_id", protocol)
    main.insert(0, "candidate_asset_id", candidate_asset)
    main.insert(0, "allocator_asset_id", asset_id)
    main.insert(0, "detector_id", detector_id)
    main.insert(0, "dataset_id", dataset_id)
    main.to_csv(out / "main_results.csv", index=False)

    # 逐组: 供成对统计用, **不用组 AP 替代全体 AP**。
    # §08 最小内容除图数与 coverage/quality 合计外还点名"记录总数": 即该条件该组被选中的
    # 记录条数 = 组内每图 K_i 之和, 与 main_results 的 total_records 同口径 (评价器
    # output_records = Σ K_i), 也等于 allocations.parquet 中该组 sum(K_i)。
    grouped = (per_image.groupby(["method", "seed", "budget", "group_id"], dropna=False, sort=True)
               .agg(image_count=("image_id", "nunique"),
                    total_records=("K_i", "sum"),
                    coverage_total=("coverage", "sum"),
                    quality_total=("quality", "sum"))
               .reset_index())
    grouped.insert(0, "candidate_asset_id", candidate_asset)
    grouped.insert(0, "protocol_id", protocol)
    grouped.to_csv(out / "per_group_results.csv", index=False)

    cls = pd.read_csv(raw / "class_results.csv")
    coco_cls = pd.read_csv(raw / "coco_class_results.csv")
    class_key = ["method", "seed", "budget", "road8_class_id"]
    coco_cls = _resolve_overlap(cls, coco_cls, class_key, "class")
    cls = cls.merge(coco_cls, on=class_key, how="left", validate="one_to_one")
    # §08 的列名: matched_count 是覆盖到的 GT 数, gt_support 是该类的 GT 总数。
    cls = cls.rename(columns={"coverage": "matched_count", "gt_count": "gt_support",
                              "road8_class_id": "class_id"})
    # 条件身份按 §08 顺序: dataset_id / detector_id / candidate_asset_id / protocol_id,
    # 再是 method/seed/budget —— 与 main_results.csv 的身份块同值同源。
    cls.insert(0, "protocol_id", protocol)
    cls.insert(0, "candidate_asset_id", candidate_asset)
    cls.insert(0, "detector_id", detector_id)
    cls.insert(0, "dataset_id", dataset_id)
    for column in ("AP", "AP50", "AP75", "AR100"):
        cls = cls.rename(columns={column: f"class_{column}"})
    cls.to_csv(out / "class_results.csv", index=False)
    log(f"merged -> main_results({len(main)} rows) per_group({len(grouped)} rows) class({len(cls)} rows)")


def stage_bootstrap(work: Path) -> dict:
    u"""§06 的配对 bootstrap: 固定 40 图组为单位的配对差值, 5000 次重采样, seed=530002。

    同一 metric 的所有预算共用同一个 seed, 所以重采样索引矩阵逐位相同 (共享抽样索引)。
    三 seed 汇总时先对每组的三个 seed 差值取均值, 再重采样 —— 不把 seed 当新增图像。
    §06 还要求"保留每个 seed 的点估计和区间; 不得平均三个区间端点": 因此每个 seed 再用
    自己的 50 组配对差值单独重采样一次 (同 5000 次、同 seed=530002), 以 seed_summary=
    per_seed 的行 (training_seed 列标出是哪个训练 seed) 与三 seed 汇总行并列写入同一张表。

    §08 点名 bootstrap_results.csv 的最小内容含"预算汇总": 因此 K10/15/20 等权汇总行
    (budget_summary=K10_15_20_equal_weight) 与逐预算行写在同一张 bootstrap_results.csv;
    K10/15/20 等权行 (三 seed 汇总 + 逐 seed) 另存一份 bootstrap_core_summary.csv
    (两处同一批数字, 汇总不再为旁支文件独有)。
    """
    from lc_alloc.cli import command_bootstrap
    from lc_alloc.evaluation.bootstrap import paired_group_bootstrap

    out = work / "bootstrap"
    out.mkdir(parents=True, exist_ok=True)
    # per_image_results.csv 是评价阶段的原始输出, 位于 evaluation/raw/ (与 §08 交付表同源)
    per_image = work / "evaluation" / "raw" / "per_image_results.csv"

    # 逐预算行: 每 metric 的配对 bootstrap 落盘后即复用 (可重跑; 删掉该文件即强制重算)
    frames = []
    for metric in ("coverage", "quality"):
        paired_path = out / metric / "paired_group_bootstrap.csv"
        if not paired_path.is_file():
            command_bootstrap(Namespace(
                per_image_results=str(per_image),
                method_a="M11", method_b="S_ADAPT", metric=metric,
                budgets=list(BUDGETS), resamples=5000, seed=530002,
                output_root=str(out / metric)))
        frames.append(pd.read_csv(paired_path))
    per_budget = pd.concat(frames, ignore_index=True)
    per_budget["budget_summary"] = "single_budget"
    per_budget["seed_summary"] = "three_seeds_averaged_per_group"
    per_budget["resampling"] = "nonparametric bootstrap over 50 complete 40-image groups"

    # K10/15/20 等权平均 —— 先把三个预算在每张图上平均, 再按组配对
    rows = pd.read_csv(per_image)
    core = []
    for metric in ("coverage", "quality"):
        subset = rows[(rows["method"].isin(["M11", "S_ADAPT"])) & (rows["budget"].isin([10, 15, 20]))]
        per_seed = subset.groupby(["method", "seed", "group_id", "image_id"], dropna=False)[metric].mean()
        per_seed = per_seed.groupby(["method", "seed", "group_id"]).mean()
        per_group = per_seed.groupby(["method", "group_id"]).mean().unstack("method")
        if not {"M11", "S_ADAPT"}.issubset(per_group.columns):
            continue
        differences = (per_group["M11"] - per_group["S_ADAPT"]).dropna()
        result = paired_group_bootstrap(differences.to_numpy(np.float64), resamples=5000, seed=530002)
        core.append({
            "method_a": "M11", "method_b": "S_ADAPT", "metric": metric,
            "budget_summary": "K10_15_20_equal_weight", "group_count": int(len(differences)),
            "seed_summary": "three_seeds_averaged_per_group",
            "resampling": "nonparametric bootstrap over 50 complete 40-image groups",
            **result,
        })
    core_frame = pd.DataFrame(core)
    core_frame["training_seed"] = pd.NA

    # ── 逐 seed 行 (§06: 保留每个 seed 的点估计和区间; 不得平均三个区间端点) ──────
    # 每个 seed 只用自己的 50 组配对差值单独重采样, 不跨 seed 平均差值、也不平均区间端点;
    # 重采样参数与三 seed 汇总完全一致 (5000 次、seed=530002, 故各次共享同一组抽样索引)。
    def per_seed_row(metric: str, budgets: tuple, training_seed: int,
                     budget_summary: str, budget) -> dict:
        a = rows[(rows["method"] == "M11") & (rows["seed"] == training_seed)
                 & (rows["budget"].isin(budgets))]
        b = rows[(rows["method"] == "S_ADAPT") & (rows["budget"].isin(budgets))]
        a_group = a.groupby("group_id", dropna=False)[metric].mean().sort_index()
        b_group = b.groupby("group_id", dropna=False)[metric].mean().sort_index()
        common = a_group.index.intersection(b_group.index)
        if len(common) != len(a_group) or len(common) != len(b_group):
            raise ValueError(f"paired group identity mismatch: metric={metric}, "
                             f"budgets={budgets}, training_seed={training_seed}")
        result = paired_group_bootstrap(
            (a_group.loc[common] - b_group.loc[common]).to_numpy(np.float64),
            resamples=5000, seed=530002)
        return {
            "method_a": "M11", "method_b": "S_ADAPT", "metric": metric,
            "budget_summary": budget_summary, "budget": budget,
            "group_count": int(len(common)),
            "seed_summary": "per_seed", "training_seed": int(training_seed),
            "resampling": "nonparametric bootstrap over 50 complete 40-image groups",
            **result,
        }

    per_seed = [
        per_seed_row(metric, (int(budget),), seeding, "single_budget", int(budget))
        for seeding in SEEDS
        for metric in ("coverage", "quality")
        for budget in BUDGETS
    ] + [
        per_seed_row(metric, (10, 15, 20), seeding, "K10_15_20_equal_weight", pd.NA)
        for seeding in SEEDS
        for metric in ("coverage", "quality")
    ]
    per_seed_frame = pd.DataFrame(per_seed)

    columns = ["method_a", "method_b", "metric", "budget_summary", "budget",
               "group_count", "point_estimate", "ci95_low", "ci95_high",
               "positive_resample_fraction", "resamples", "seed",
               "seed_summary", "training_seed", "resampling"]
    # 核心汇总文件 = K10/15/20 等权行 (三 seed 汇总 + 逐 seed), 与主文件同一批数字
    core_all = pd.concat([core_frame, per_seed_frame[
        per_seed_frame["budget_summary"] == "K10_15_20_equal_weight"]], ignore_index=True)
    core_all[columns].to_csv(out / "bootstrap_core_summary.csv", index=False)

    # §08: 预算汇总行并入 bootstrap_results.csv; 汇总行不对应单一预算, budget 列留空
    merged = pd.concat([per_budget, core_frame, per_seed_frame], ignore_index=True)
    merged["budget"] = merged["budget"].astype("Int64")
    merged[columns].to_csv(out / "bootstrap_results.csv", index=False)
    log(f"bootstrap -> {out} (three-seed summary rows + {len(per_seed_frame)} per-seed rows)")
    return {"core": core, "per_seed_rows": int(len(per_seed_frame))}


def stage_pfx_nms(work: Path) -> dict:
    u"""§07 的 PFX_THEN_NMS 参考协议与计数表。主比较仍是 PFX_EXACT。"""
    from lc_frcnn import pfx_then_nms

    argv = ["pfx_then_nms", "--work", str(work), "--out", str(work / "pfx_then_nms")]
    saved, sys.argv = sys.argv, argv
    try:
        pfx_then_nms.main()
    finally:
        sys.argv = saved
    table = pd.read_csv(work / "pfx_then_nms/pfx_then_nms_by_condition.csv")
    return {"conditions": int(len(table)),
            "removed_total": int(table["removed_total"].sum()),
            "empty_images": int(table["empty_images"].sum())}


def stage_delivery(work: Path, prepared: Path, qa: Path) -> dict:
    u"""§08 的交付包。打包动作本身不产生任何科学量, 只做身份绑定与可复现清单。"""
    from lc_frcnn import make_delivery

    argv = ["make_delivery", "--work", str(work), "--prepared", str(prepared), "--qa", str(qa)]
    saved, sys.argv = sys.argv, argv
    try:
        make_delivery.main()
    finally:
        sys.argv = saved
    root = M11_ROOT / "delivery_out" / "member_faster_rcnn_delivery"
    rows = pd.read_csv(root / "manifest.csv")
    return {"files": int(len(rows)), "bytes": int(rows["bytes"].sum()), "root": str(root)}


ALL_STAGES = ("labels", "features", "train", "calibrate", "assets", "allocate",
              "evaluate", "pfx_nms", "bootstrap", "delivery")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=[*ALL_STAGES, "all"])
    ap.add_argument("--work", default=str(M11_ROOT / "work"))
    ap.add_argument("--prepared", default=str(M11_ROOT / "prepared"))
    ap.add_argument("--qa", default=str(M11_ROOT / "qa"))
    args = ap.parse_args()

    work = Path(args.work)
    prepared = Path(args.prepared)
    work.mkdir(parents=True, exist_ok=True)

    stages = ALL_STAGES if args.stage == "all" else (args.stage,)
    for stage in stages:
        log(f"===== {stage} =====")
        if stage == "labels":
            log(f"{stage}: {stage_labels(work, prepared)}")
        elif stage == "features":
            log(f"{stage}: {stage_features(work, prepared)}")
        elif stage == "train":
            log(f"{stage}: {stage_train(work)}")
        elif stage == "calibrate":
            log(f"{stage}: {stage_calibrate(work)}")
        elif stage == "assets":
            log(f"{stage}: {stage_assets(work, prepared)}")
        elif stage == "allocate":
            log(f"{stage}: {stage_allocate(work, prepared)}")
        elif stage == "evaluate":
            log(f"{stage}: {stage_evaluate(work, prepared)}")
        elif stage == "pfx_nms":
            log(f"{stage}: {stage_pfx_nms(work)}")
        elif stage == "bootstrap":
            log(f"{stage}: {stage_bootstrap(work)}")
        elif stage == "delivery":
            log(f"{stage}: {stage_delivery(work, prepared, Path(args.qa))}")
    log("pipeline done")


if __name__ == "__main__":
    main()
