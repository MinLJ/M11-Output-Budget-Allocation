"""Same-run group-size runtime audit for LC-ALLOC-BENCH-P0.

The measured boundary starts with policy-required candidate/native tensors
already resident in host memory and ends with exact K values and record IDs.
Detector execution, disk reads, model loading, evaluation, and result writes
are deliberately outside the main timing boundary.
"""
from __future__ import annotations
import os

import hashlib
import json
import platform
import random
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import psutil

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
P1 = (Path(os.environ.get('M11_EXPERIMENT_ROOT', "external_assets")) / 'research_LC_ALLOC_P1')
P1B = (Path(os.environ.get('M11_EXPERIMENT_ROOT', "external_assets")) / 'research_LC_ALLOC_P1B')
R0 = (Path(os.environ.get('M11_EXPERIMENT_ROOT', "external_assets")) / 'research_LC_ALLOC_R0')
RELEASE = (Path(os.environ.get('M11_EXPERIMENT_ROOT', "external_assets")) / 'shared_benchmark/AOP_ROAD8_LARGECLEAN_V1/P1_SHARED_EXPORT/release/AOP_Road8_LC_v1_RTDERTv2_R18VD_RELEASE_v1')
GROUP_SIZES = (10, 20, 40, 80)
BUDGET = 15
SEED = 530101
ORDER_SEED = 530003


def import_file(path: Path, name: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha_key(image_id: str) -> str:
    return hashlib.sha256(("LC_ALLOC_BENCH_P0_PROFILE_V1|" + str(image_id)).encode()).hexdigest()


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def qsummary(values: np.ndarray) -> dict[str, float | int]:
    a = np.asarray(values, dtype=np.float64)
    return {
        "samples": int(a.size), "mean_ms": float(a.mean()),
        "median_ms": float(np.median(a)), "p95_ms": float(np.quantile(a, .95)),
        "std_ms": float(a.std(ddof=1)) if a.size > 1 else 0.0,
        "min_ms": float(a.min()), "max_ms": float(a.max()),
    }


def effective_parameter_map(path: Path) -> dict[int, tuple[float, float]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    entries = obj.get("classes", obj.get("class_parameters", []))
    result: dict[int, tuple[float, float]] = {}
    if isinstance(entries, dict):
        entries = [dict(v, class_id=int(k)) for k, v in entries.items()]
    for row in entries:
        cid = int(row.get("class_id", row.get("road8_class_id")))
        scale = float(row.get("effective_scale", abs(float(row.get("raw_scale", row.get("scale", 1.0))))))
        bias = float(row.get("bias", row.get("shift", 0.0)))
        result[cid] = (scale, bias)
    if set(result) != set(range(1, 9)):
        raise RuntimeError(f"calibrator parameter coverage invalid: {sorted(result)}")
    return result


def stable_sigmoid(x: np.ndarray) -> np.ndarray:
    z = np.asarray(x, dtype=np.float64)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def main() -> None:
    start_all = time.perf_counter()
    with (ROOT / "runtime.log").open("a", encoding="utf-8") as f:
        f.write("STAGE profile_runtime START\n")

    p1b = import_file(P1B / "scripts" / "profile_optimizations.py", "bench_p1b_runtime")
    r0 = import_file(R0 / "scripts" / "alloc_solver.py", "bench_r0_solver")
    common = import_file(ROOT / "scripts" / "common.py", "bench_runtime_common")
    sys.path.insert(0, str(P1 / "scripts"))
    from p1_core import iou_xyxy  # type: ignore
    from train_models import MarginalMLP  # type: ignore

    role = pq.read_table(P1 / "train_role_split.parquet").to_pandas()
    role["image_id"] = role["image_id"].astype(str)
    fit_ids = role.loc[role["role"] == "FIT", "image_id"].astype(str).tolist()
    selected = sorted(fit_ids, key=lambda x: (sha_key(x), x))[:400]
    if len(fit_ids) != 8000 or len(selected) != 400 or len(set(selected)) != 400:
        raise RuntimeError("frozen FIT identity/profile selection invariant failed")
    profile_rows = []
    for pos, image_id in enumerate(selected):
        profile_rows.append({"image_id": image_id, "profile_order": pos,
                             "profile_block_80": pos // 80, "position_in_block": pos % 80,
                             "identity_key": sha_key(image_id)})
    pq.write_table(pa.Table.from_pandas(pd.DataFrame(profile_rows), preserve_index=False),
                   ROOT / "qa" / "runtime_profile_manifest.parquet", compression="zstd")

    identity = json.loads((RELEASE / "CANDIDATE_ASSET_IDENTITY.json").read_text(encoding="utf-8"))
    read_start = time.perf_counter()
    raw, _ = p1b.load_raw_images(RELEASE, identity, "TRAIN", selected)
    excluded_read_ms = (time.perf_counter() - read_start) * 1000.0

    # Five fixed 80-image blocks; smaller sizes subdivide each block without
    # changing its membership.  Caller order is frozen image_id order.
    group_ids: dict[tuple[int, int], list[str]] = {}
    for n in GROUP_SIZES:
        ordinal = 0
        for block in range(5):
            ids80 = selected[block * 80:(block + 1) * 80]
            if len(ids80) != 80:
                raise RuntimeError(f"profile block size failed block={block}")
            for offset in range(0, 80, n):
                member_ids = ids80[offset:offset + n]
                if len(member_ids) != n:
                    raise RuntimeError(f"profile subgroup size failed n={n} block={block} offset={offset}")
                group_ids[(n, ordinal)] = sorted(member_ids)
                ordinal += 1
        if ordinal != 400 // n:
            raise RuntimeError(f"profile grouping failed n={n}")

    load_start = time.perf_counter()
    pca = joblib.load(P1 / "models" / "pca32.joblib")
    scaler_bundle = joblib.load(P1 / "models" / "feature_scaler.joblib")
    scaler = scaler_bundle["scaler"]
    standardize = np.asarray(scaler_bundle["standardize_mask"], dtype=bool)
    weights = np.asarray(json.loads((P1 / "models" / "class_weights.json").read_text(encoding="utf-8"))["weights"], dtype=np.float64)
    snap = torch.load(P1 / "models" / f"marginal_mlp_seed_{SEED}.pt", map_location="cpu", weights_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MarginalMLP(int(snap["input_dim"]))
    model.load_state_dict(snap["state_dict"])
    model.eval().to(device)
    temperature = float(json.loads((P1 / "models" / f"temperature_seed_{SEED}.json").read_text(encoding="utf-8"))["temperature"])
    parameter_path = ROOT / "calibrator_parameters.json"
    b_freeze = json.loads((ROOT / "qa" / "external_baseline_selection_freeze.json").read_text(encoding="utf-8"))
    if common.sha256_file(parameter_path) != b_freeze.get("calibrator_parameters_sha256"):
        raise RuntimeError("runtime calibrator parameters do not match the frozen B selection identity")
    params = effective_parameter_map(parameter_path)
    sync(device)
    excluded_load_ms = (time.perf_counter() - load_start) * 1000.0

    @torch.inference_mode()
    def infer(x: np.ndarray) -> np.ndarray:
        xt = torch.from_numpy(np.asarray(x, dtype=np.float32)).to(device)
        logits = model(xt).float()
        prob = torch.sigmoid(logits / temperature).cpu().numpy().astype(np.float64)
        sync(device)
        return prob

    def views(ids: list[str]):
        out = []
        for image_id in ids:
            c = raw[image_id].candidates.sort_values("road8_rank", kind="stable").reset_index(drop=True)
            out.append((image_id, c["score"].to_numpy(np.float64),
                        c["predicted_road8_class_id"].to_numpy(np.int16),
                        c["candidate_record_id"].astype(str).tolist()))
        return out

    def s_adapt(ids: list[str]):
        v = views(ids)
        scores = np.vstack([x[1][:50] for x in v])
        k, objective = r0.allocate_prefixes([x[0] for x in v], scores, len(ids) * BUDGET, 5, 50)
        return k, [v[i][3][:int(k[i])] for i in range(len(v))], float(objective)

    def m11(ids: list[str]):
        group = [raw[x] for x in ids]
        x, classes, records = p1b.optimized_group_features(group, pca, scaler, standardize, iou_xyxy)
        prob = infer(x)
        margins = (prob.mean(axis=1) * weights[classes - 1]).reshape(len(ids), 45)
        k_all, objective = common.solve_group(margins, [BUDGET])
        k = k_all[0]
        return k, [records[i][:int(k[i])] for i in range(len(ids))], float(objective[0])

    def platt(ids: list[str], class_weighted: bool):
        v = views(ids)
        margins = []
        for _, scores, classes, _ in v:
            s = np.clip(scores[5:50], np.finfo(np.float64).eps, 1.0 - np.finfo(np.float64).eps)
            logits = np.log(s / (1.0 - s))
            scales = np.asarray([params[int(c)][0] for c in classes[5:50]], dtype=np.float64)
            biases = np.asarray([params[int(c)][1] for c in classes[5:50]], dtype=np.float64)
            value = stable_sigmoid(scales * logits + biases)
            if class_weighted:
                value *= weights[classes[5:50] - 1]
            margins.append(value)
        k_all, objective = common.solve_group(np.vstack(margins), [BUDGET])
        k = k_all[0]
        return k, [v[i][3][:int(k[i])] for i in range(len(v))], float(objective[0])

    def call(policy: str, n: int, gid: int):
        ids = group_ids[(n, gid)]
        if policy == "M11":
            return m11(ids)
        if policy == "S_ADAPT":
            return s_adapt(ids)
        if policy == "PS_PREFIX":
            return platt(ids, False)
        if policy == "PS_CLASS_PREFIX":
            return platt(ids, True)
        raise KeyError(policy)

    # Structural smoke before timing.
    for n in GROUP_SIZES:
        for policy in ("M11", "S_ADAPT"):
            k, chosen, _ = call(policy, n, 0)
            if int(np.sum(k)) != n * BUDGET or sum(map(len, chosen)) != n * BUDGET:
                raise RuntimeError(f"runtime smoke capacity failed {policy}/n{n}")
    for policy in ("PS_PREFIX", "PS_CLASS_PREFIX"):
        k, chosen, _ = call(policy, 40, 0)
        if int(np.sum(k)) != 40 * BUDGET or sum(map(len, chosen)) != 40 * BUDGET:
            raise RuntimeError(f"runtime smoke capacity failed {policy}")

    conditions: list[tuple[str, int, int]] = []
    for n in GROUP_SIZES:
        for gid in range(400 // n):
            conditions.extend([("M11", n, gid), ("S_ADAPT", n, gid)])
    for gid in range(10):
        conditions.extend([("PS_PREFIX", 40, gid), ("PS_CLASS_PREFIX", 40, gid)])
    rng = random.Random(ORDER_SEED)
    warm = [(p, n, g, rep) for p, n, g in conditions for rep in range(3)]
    measured = [(p, n, g, rep) for p, n, g in conditions for rep in range(10)]
    rng.shuffle(warm)
    rng.shuffle(measured)

    samples: list[dict] = []
    for phase, calls in (("WARMUP", warm), ("MEASURED", measured)):
        for order, (policy, n, gid, rep) in enumerate(calls):
            sync(device)
            t0 = time.perf_counter_ns()
            k, chosen, objective = call(policy, n, gid)
            sync(device)
            elapsed = (time.perf_counter_ns() - t0) / 1e6
            if int(np.sum(k)) != n * BUDGET or sum(map(len, chosen)) != n * BUDGET:
                raise RuntimeError(f"runtime exact-budget invariant failed {policy}/n{n}/g{gid}")
            if phase == "MEASURED":
                digest = hashlib.sha256("|".join(x for row in chosen for x in row).encode()).hexdigest()
                samples.append({
                    "phase": phase, "sample_order": order, "repeat_index": rep,
                    "policy": policy, "group_size": n, "profile_group_id": gid,
                    "budget": BUDGET, "elapsed_ms": elapsed,
                    "amortized_ms_per_image": elapsed / n,
                    "output_records": int(np.sum(k)), "predicted_objective": objective,
                    "selection_digest": digest,
                    "boundary": "host-resident policy-required candidate/native tensors through K and record IDs",
                })

    sample_df = pd.DataFrame(samples)
    pq.write_table(pa.Table.from_pandas(sample_df, preserve_index=False),
                   ROOT / "outputs" / "runtime_samples.parquet", compression="zstd")
    summaries: list[dict] = []
    for (policy, n), d in sample_df.groupby(["policy", "group_size"], sort=True):
        stats = qsummary(d["elapsed_ms"].to_numpy())
        summaries.append({"scope": "ACTUAL_GROUP", "policy": policy, "group_size": int(n),
                          "budget": BUDGET, "actual_groups": int(d["profile_group_id"].nunique()),
                          **stats, "amortized_median_ms_per_image": float(stats["median_ms"]) / int(n),
                          "full_400_summary_semantics": "not_applicable"})
        # Each repeat covers every actual group exactly once.  Summing those
        # measured group calls gives a directly observed 400-image processing total.
        totals = d.groupby("repeat_index", sort=True)["elapsed_ms"].sum().to_numpy(np.float64)
        full = qsummary(totals)
        summaries.append({"scope": "FULL_400_AGGREGATE", "policy": policy, "group_size": int(n),
                          "budget": BUDGET, "actual_groups": int(d["profile_group_id"].nunique()),
                          **full, "amortized_median_ms_per_image": float(full["median_ms"]) / 400.0,
                          "full_400_summary_semantics": "sum of all actual-group calls sharing repeat_index"})
    summary_df = pd.DataFrame(summaries)
    pq.write_table(pa.Table.from_pandas(summary_df, preserve_index=False),
                   ROOT / "outputs" / "runtime_summary.parquet", compression="zstd")
    environment = {
        "device": str(device), "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "cpu": platform.processor(), "ram_bytes": int(psutil.virtual_memory().total), "torch": torch.__version__,
        "cuda": torch.version.cuda, "numpy": np.__version__, "pandas": pd.__version__,
        "joblib": joblib.__version__, "python": sys.version, "os": platform.platform(),
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "profile_images": 400, "profile_role": "FIT", "group_sizes": list(GROUP_SIZES),
        "budget": BUDGET, "m11_seed": SEED, "warmups_per_actual_group": 3,
        "measurements_per_actual_group": 10, "order_seed": ORDER_SEED,
        "excluded_release_read_ms": excluded_read_ms, "excluded_model_preprocessor_load_ms": excluded_load_ms,
        "main_boundary": "host-resident policy-required candidate/native tensors through K and record IDs",
        "detector_included": False, "gt_read": False,
        "elapsed_seconds": time.perf_counter() - start_all,
    }
    (ROOT / "outputs" / "runtime_environment.json").write_text(
        json.dumps(environment, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (ROOT / "runtime.log").open("a", encoding="utf-8") as f:
        f.write(f"STAGE profile_runtime COMPLETE samples={len(sample_df)} elapsed_seconds={time.perf_counter()-start_all:.6f}\n")


if __name__ == "__main__":
    main()
