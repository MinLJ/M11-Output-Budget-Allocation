"""Profile only the incremental cached-asset policy overhead for LC-ALLOC-P1."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from dp_solver import solve_group_allocations  # noqa: E402
from p1_core import SEEDS, build_raw_features, predict_table, sigmoid  # noqa: E402
from release_io import ReleaseReader  # noqa: E402
from train_models import MarginalMLP  # noqa: E402


def sync(device: torch.device) -> None:
    if device.type == "cuda": torch.cuda.synchronize()


def summary(name: str, samples: list[float], unit: str, note: str) -> dict:
    a = np.asarray(samples, dtype=np.float64)
    return {
        "scope": name, "unit": unit, "repeats": len(a), "mean_ms": float(a.mean() * 1000),
        "median_ms": float(np.median(a) * 1000), "p95_ms": float(np.quantile(a, .95) * 1000),
        "min_ms": float(a.min() * 1000), "max_ms": float(a.max() * 1000), "notes": note,
    }


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--project-root", required=True); ap.add_argument("--release-root", required=True); ap.add_argument("--group-manifest", required=True); args = ap.parse_args()
    root, release_root = Path(args.project_root).resolve(), Path(args.release_root).resolve()
    reader = ReleaseReader(release_root)
    group = pq.read_table(args.group_manifest).to_pandas()
    ids = sorted(group[group.group_id == 0].image_id.astype(str).tolist())
    targets = set(ids)
    t = time.perf_counter(); bundles = {}
    for b in reader.iter_bundles("DEV"):
        if b.image_id in targets: bundles[b.image_id] = b
    read_seconds = time.perf_counter() - t
    if set(bundles) != targets: raise RuntimeError("cached runtime group incomplete")
    ordered = [bundles[x] for x in ids]

    pca = joblib.load(root / "models" / "pca32.joblib")
    sb = joblib.load(root / "models" / "feature_scaler.joblib"); scaler, mask = sb["scaler"], np.asarray(sb["standardize_mask"], bool)
    with np.load(root / "models" / "table_model.npz", allow_pickle=False) as z: table = {k: z[k].copy() for k in z.files}
    weights = np.asarray(json.loads((root / "models" / "class_weights.json").read_text(encoding="utf-8"))["weights"], np.float64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models, temps = {}, {}
    for seed in SEEDS:
        snap = torch.load(root / "models" / f"marginal_mlp_seed_{seed}.pt", map_location="cpu", weights_only=True)
        model = MarginalMLP(90); model.load_state_dict(snap["state_dict"]); model.eval().to(device); models[seed] = model
        temps[seed] = float(json.loads((root / "models" / f"temperature_seed_{seed}.json").read_text(encoding="utf-8"))["temperature"])

    def features() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        xs, cls, ranks, scores = [], [], [], []
        for b in ordered:
            x = build_raw_features(b.candidates, b.road8_logits, b.embeddings, pca, b.width, b.height)
            x[:, mask] = scaler.transform(x[:, mask]); xs.append(x.astype(np.float32))
            c = b.candidates.iloc[5:50]
            cls.append(c.predicted_road8_class_id.to_numpy(np.int16)); ranks.append(c.road8_rank.to_numpy(np.int16)); scores.append(c.score.to_numpy(np.float64))
        return np.vstack(xs), np.concatenate(cls), np.concatenate(ranks), np.concatenate(scores)

    @torch.inference_mode()
    def infer(x: np.ndarray) -> dict[int, np.ndarray]:
        xt = torch.from_numpy(x).to(device); out = {}
        for seed in SEEDS: out[seed] = sigmoid(models[seed](xt).float().cpu().numpy().astype(np.float64) / temps[seed])
        sync(device); return out

    def allocate(probs: dict[int, np.ndarray], cls: np.ndarray, ranks: np.ndarray, scores: np.ndarray) -> None:
        tp = predict_table(table, cls, ranks, scores)
        policies = [tp[:, 0].reshape(40, 45), (tp.mean(1) * weights[cls - 1]).reshape(40, 45)]
        for seed in SEEDS:
            policies += [probs[seed][:, 0].reshape(40, 45), (probs[seed].mean(1) * weights[cls - 1]).reshape(40, 45)]
        for p in policies: solve_group_allocations(p)

    base = features(); prob = infer(base[0]); allocate(prob, *base[1:])
    for _ in range(4):
        base = features(); prob = infer(base[0]); allocate(prob, *base[1:])
    feature_samples=[]; infer_samples=[]; dp_samples=[]; total_samples=[]
    for _ in range(20):
        sync(device); t=time.perf_counter(); x,cls,ranks,scores=features(); sync(device); feature_samples.append(time.perf_counter()-t)
        sync(device); t=time.perf_counter(); probs=infer(x); sync(device); infer_samples.append(time.perf_counter()-t)
        t=time.perf_counter(); allocate(probs,cls,ranks,scores); dp_samples.append(time.perf_counter()-t)
        sync(device); t=time.perf_counter(); x2,c2,r2,s2=features(); probs2=infer(x2); allocate(probs2,c2,r2,s2); sync(device); total_samples.append(time.perf_counter()-t)
    original = json.loads((root / "outputs" / "prediction_runtime.json").read_text(encoding="utf-8"))
    save_est = original["total_seconds"] - original["feature_and_release_read_seconds"] - original["three_model_shared_inference_seconds"] - original["exact_dp_seconds"]
    rows = [
        {"scope":"full_DEV_total_prediction_allocation_and_save","unit":"DEV2000","repeats":1,"mean_ms":original["total_seconds"]*1000,"median_ms":original["total_seconds"]*1000,"p95_ms":original["total_seconds"]*1000,"min_ms":original["total_seconds"]*1000,"max_ms":original["total_seconds"]*1000,"notes":"Actual frozen run; excludes detector and GT evaluation."},
        {"scope":"full_DEV_release_read_plus_feature","unit":"DEV2000","repeats":1,"mean_ms":original["feature_and_release_read_seconds"]*1000,"median_ms":original["feature_and_release_read_seconds"]*1000,"p95_ms":original["feature_and_release_read_seconds"]*1000,"min_ms":original["feature_and_release_read_seconds"]*1000,"max_ms":original["feature_and_release_read_seconds"]*1000,"notes":"Actual frozen run combined scope."},
        {"scope":"full_DEV_three_model_shared_inference","unit":"DEV2000","repeats":1,"mean_ms":original["three_model_shared_inference_seconds"]*1000,"median_ms":original["three_model_shared_inference_seconds"]*1000,"p95_ms":original["three_model_shared_inference_seconds"]*1000,"min_ms":original["three_model_shared_inference_seconds"]*1000,"max_ms":original["three_model_shared_inference_seconds"]*1000,"notes":"One forward per seed; MICRO/QUALITY share it."},
        {"scope":"full_DEV_exact_DP_all_policies","unit":"50 frozen groups","repeats":1,"mean_ms":original["exact_dp_seconds"]*1000,"median_ms":original["exact_dp_seconds"]*1000,"p95_ms":original["exact_dp_seconds"]*1000,"min_ms":original["exact_dp_seconds"]*1000,"max_ms":original["exact_dp_seconds"]*1000,"notes":"TABLE plus six learned target/seed policies."},
        {"scope":"full_DEV_selection_materialization_and_save_residual","unit":"DEV2000","repeats":1,"mean_ms":save_est*1000,"median_ms":save_est*1000,"p95_ms":save_est*1000,"min_ms":save_est*1000,"max_ms":save_est*1000,"notes":"Residual wall time; includes allocation dataframe/materialization and Parquet save, not a separately instrumented causal component."},
        {"scope":"DEV_release_read_adapter_postrun_warm_cache","unit":"DEV2000","repeats":1,"mean_ms":read_seconds*1000,"median_ms":read_seconds*1000,"p95_ms":read_seconds*1000,"min_ms":read_seconds*1000,"max_ms":read_seconds*1000,"notes":"Separate read/validation traversal measured after the scientific run; OS cache state differs, so do not subtract from the frozen combined scope."},
        summary("cached_40_feature_construction",feature_samples,"fixed 40-image group","In-memory candidate/native arrays; includes PCA/scaler, excludes disk."),
        summary("cached_40_three_model_shared_inference",infer_samples,"fixed 40-image group","CUDA synchronized; both value definitions reuse each seed forward."),
        summary("cached_40_utility_and_exact_DP",dp_samples,"fixed 40-image group","Eight TABLE/learned policies, five exact capacities each."),
        summary("cached_40_total_policy_processing",total_samples,"fixed 40-image group","Feature + three models + utilities + exact DP; CUDA synchronized."),
    ]
    pq.write_table(pa.Table.from_pandas(pd.DataFrame(rows), preserve_index=False), root / "runtime_summary.parquet", compression="zstd")
    with (root / "runtime.log").open("a", encoding="utf-8") as f: f.write("STAGE profile_runtime COMPLETE\n")


if __name__ == "__main__": main()

