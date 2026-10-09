"""Write the compact P1B report, runtime reconciliation and final status metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.dont_write_bytecode = True
SEEDS=(530101,530102,530103); BUDGETS=(10,15,20,30,40); CORE=(10,15,20)


def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(8<<20),b""): h.update(b)
    return h.hexdigest()


def append_log(root: Path,value: str) -> None:
    with (root/"runtime.log").open("a",encoding="utf-8") as f:f.write(value.rstrip()+"\n")


def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument("--project-root",required=True); ap.add_argument("--p1-root",required=True); ap.add_argument("--p1a-root",required=True); ap.add_argument("--r0-root",required=True)
    a=ap.parse_args(); root,p1,p1a,r0=map(lambda x:Path(x).resolve(),(a.project_root,a.p1_root,a.p1a_root,a.r0_root)); t0=time.perf_counter()
    append_log(root,"STAGE summarize_p1b START")
    main_df=pq.read_table(root/"outputs"/"baseline_main.parquet").to_pandas(); class_df=pq.read_table(root/"outputs"/"baseline_class.parquet").to_pandas()
    boot=pq.read_table(root/"outputs"/"baseline_bootstrap.parquet").to_pandas(); runtime=pq.read_table(root/"outputs"/"runtime_summary.parquet").to_pandas()
    parity=pq.read_table(root/"outputs"/"implementation_parity.parquet").to_pandas(); impl=json.loads((root/"outputs"/"implementation_status.json").read_text(encoding="utf-8"))
    screens=json.loads((root/"outputs"/"quality_screens.json").read_text(encoding="utf-8"))
    if (~parity["pass"]).any() or int(parity["K_mismatch_images"].sum()) or int(parity["selection_mismatch_images"].sum()): raise RuntimeError("final implementation parity is not clean")

    def meanrow(method,budget):
        d=main_df[(main_df.method==method)&(main_df.budget==budget)]
        return d[d.seed.isin(SEEDS)].mean(numeric_only=True) if method.startswith("LEARN_") else d[d.seed==-1].iloc[0]
    def core_delta(left,right,metric): return float(np.mean([meanrow(left,b)[metric]-meanrow(right,b)[metric] for b in CORE]))
    def bcore(comp,metric): return boot[(boot.comparison==comp)&(boot.metric==metric)&(boot.scope=="CORE_10_15_20")].iloc[0]
    def rt(policy,budget): return runtime[(runtime.timing_scope=="FULL_POLICY_PATH")&(runtime.policy==policy)&(runtime.budget==budget)].iloc[0]

    # Runtime figure (the third and final project figure).
    fig,ax=plt.subplots(figsize=(10.5,4.8)); policies=["S_ADAPT_REFERENCE","S_ADAPT_OPTIMIZED","S_CLASS_ADAPT","M11_REFERENCE","M11_OPTIMIZED"]
    labels=["S ref","S opt","S_CLASS","M11 ref","M11 opt"]; x=np.arange(3); width=.16
    for j,(p,label) in enumerate(zip(policies,labels)):
        vals=[rt(p,b).median_ms for b in CORE]; ax.bar(x+(j-2)*width,vals,width,label=label)
    ax.set_xticks(x,[f"K{b}" for b in CORE]); ax.set_ylabel("40-image policy latency (ms, median)"); ax.set_yscale("log"); ax.grid(axis="y",alpha=.25); ax.legend(ncol=3,fontsize=8)
    ax.set_title("Same-run host-resident policy processing (log scale)"); fig.tight_layout(); fig.savefig(root/"figures"/"runtime_reference_vs_optimized.png",dpi=180); plt.close(fig)

    reconciliation=[
        "# Runtime path reconciliation","",
        "`historical_timing_status = PARTIALLY_EXPLAINED`","",
        "## What the historical numbers measured","",
        "- R0 `PILOT_FIXED_CACHED_40` reported **6.598 ms median** for one cached 40-image list-of-dicts input, one ranking pass, all five budgets, CPU/BLAS one thread, and K allocations only; it did not materialize selected record IDs.",
        "- P1A reported roughly **64–65 ms median** per single budget for `S_ADAPT`. Its timed path rebuilt 4,000 full Python dictionaries (about 56,000 fields) from 40 DataFrames, ranked them, solved one budget, materialized record IDs, and was interleaved with GPU model paths under 12 torch threads.",
        "- R0's full-DEV score allocation was 0.3032 s for 50 groups and five budgets, or about 6.063 ms/group, consistent with its cached PILOT boundary.","",
        "These are different boundaries; `64/6.598` is not a valid speed-regression ratio. The near-constant P1A latency over K10/K15/K20 also points to adapter/object creation rather than capacity-dependent allocation.","",
        "## Current same-machine, same-input, same-boundary measurements","",
        "Every row below starts from its policy's required host-resident raw state and ends with K, ordered candidate record IDs, and objective. Each median/p95 uses 100 samples over the same five frozen FIT groups; budgets are solved separately.","",
        "| Path | K10 median / p95 ms | K15 median / p95 ms | K20 median / p95 ms |","|---|---:|---:|---:|",
    ]
    for p,label in zip(policies,labels):
        cells=[f"{rt(p,b).median_ms:.3f} / {rt(p,b).p95_ms:.3f}" for b in CORE]; reconciliation.append(f"| {label} | {' | '.join(cells)} |")
    reconciliation += ["","The accepted score fast path first proves that detector score order equals frozen `road8_rank`, then calls the frozen exact score allocator and materializes the same prefix IDs. Across all profile groups/budgets, K, ordered IDs, objective float64 bytes and capacity were identical.","",
                       "The accepted M11 implementation batches the unchanged PCA/scaler work. The 30,000 DEV K cells and corresponding candidate-ID prefixes had zero mismatch. Saved-probability differences were at most 2.452e-7, within atol=1e-6/rtol=1e-5; this is tolerance equivalence plus exact selection parity, not a claim of universal bitwise identity.","",
                       "P1B does not fully recreate the historical scheduling epoch or R0's one-thread/no-ID boundary. The historical gap is therefore structurally explained but not converted into a same-boundary numerical comparison; the honest status is `PARTIALLY_EXPLAINED`.",""]
    (root/"runtime_path_reconciliation.md").write_text("\n".join(reconciliation),encoding="utf-8")

    # Update the frozen run identity with direct scientific/runtime dependencies.
    cfg=json.loads((root/"run_config.json").read_text(encoding="utf-8")); cfg.update({
        "execution_status":"COMPLETE","baseline_analysis_status":"BASELINE_ANALYSIS_COMPLETE_M11_RETAINS_INCREMENT",
        "optimization_status":impl["optimization_status"],"historical_timing_status":"PARTIALLY_EXPLAINED","confirmation_status":"NOT_AUTHORIZED",
        "runtime_protocol":{"profile_manifest":str(p1a/"qa"/"profile_manifest.parquet"),"profile_images":200,"groups":5,"budgets":list(CORE),"warmups":5,"measurements_per_group_condition":20,"order_seed":530003,
                            "boundary":"host-resident policy-required raw tensors to K and ordered record IDs"},
        "model_preprocessor_bindings":{
            "pca32":{"path":str(p1/"models"/"pca32.joblib"),"sha256":sha(p1/"models"/"pca32.joblib")},
            "scaler":{"path":str(p1/"models"/"feature_scaler.joblib"),"sha256":sha(p1/"models"/"feature_scaler.joblib")},
            "models":[{"seed":s,"path":str(p1/"models"/f"marginal_mlp_seed_{s}.pt"),"sha256":sha(p1/"models"/f"marginal_mlp_seed_{s}.pt"),"temperature_path":str(p1/"models"/f"temperature_seed_{s}.json"),"temperature_sha256":sha(p1/"models"/f"temperature_seed_{s}.json")} for s in SEEDS],
        },
        "reused_science_bindings":{
            "p1a_main":{"path":str(p1a/"outputs"/"ablation_main_results.parquet"),"sha256":sha(p1a/"outputs"/"ablation_main_results.parquet")},
            "p1a_per_image":{"path":str(p1a/"ablation_per_image_results.parquet"),"sha256":sha(p1a/"ablation_per_image_results.parquet")},
            "p1a_group":{"path":str(p1a/"outputs"/"ablation_group_results.parquet"),"sha256":sha(p1a/"outputs"/"ablation_group_results.parquet")},
            "p1a_class":{"path":str(p1a/"outputs"/"ablation_class_results.parquet"),"sha256":sha(p1a/"outputs"/"ablation_class_results.parquet")},
            "bootstrap_indices":{"path":str(p1/"outputs"/"bootstrap_group_indices.npy"),"sha256":sha(p1/"outputs"/"bootstrap_group_indices.npy")},
            "r0_timing_code":{"path":str(r0/"scripts"/"alloc_solver.py"),"sha256":sha(r0/"scripts"/"alloc_solver.py")},
            "r0_runtime_table":{"path":str(r0/"runtime_summary.csv"),"sha256":sha(r0/"runtime_summary.csv")},
        },
    }); (root/"run_config.json").write_text(json.dumps(cfg,indent=2,sort_keys=True)+"\n",encoding="utf-8")

    report=["# LC-ALLOC-P1B Report","","## Scope and frozen action","",
            "This DEV-only analysis adds one non-neural baseline and audits numerically equivalent implementations. All policies retain the original detector-score prefix `road8_rank=1..K_i`; there is no reranking, NMS, query deduplication, detector forward, retraining, TEST access, or weight refit.","",
            "`S_CLASS_ADAPT` values optional slots 6..50 as frozen FIT class weight × original detector score, then uses the unchanged exact float64 multiple-choice DP for each frozen 40-image group. The allocation was SHA-frozen before DEV GT evaluation: 10,000 image-budget rows representing exactly 230,000 selected candidate records.","",
            "## Main DEV results","","M10/M11 entries are arithmetic means of three independently allocated and evaluated frozen models, not probability ensembles. COCO evaluation always uses the original detector score.","",
            "| Policy | K | coverage/img | QUALITY/img | AP | AR100 |","|---|---:|---:|---:|---:|---:|"]
    for b in BUDGETS:
        for method,label in (("S_ADAPT","S_ADAPT"),("S_CLASS_ADAPT","S_CLASS_ADAPT"),("LEARN_CLASS50","M10"),("LEARN_QUALITY","M11")):
            r=meanrow(method,b); report.append(f"| {label} | {b} | {r.coverage_per_image:.4f} | {r.quality_per_image:.4f} | {r.AP:.5f} | {r.AR100:.5f} |")
    report += ["","## Frozen comparisons","",
               "Core means average K10/K15/K20. Coverage and QUALITY intervals use 5,000 paired resamples of the same 50 frozen groups; positive fractions are descriptive, not p-values.","",
               "| Comparison | Δ coverage/img [95% CI] | Δ QUALITY/img [95% CI] | ΔAP pp | ΔAR100 pp |","|---|---:|---:|---:|---:|"]
    for comp,left,right in (("M11 − S_CLASS","LEARN_QUALITY","S_CLASS_ADAPT"),("M10 − S_CLASS","LEARN_CLASS50","S_CLASS_ADAPT"),("S_CLASS − S","S_CLASS_ADAPT","S_ADAPT")):
        key={"M11 − S_CLASS":"M11_MINUS_S_CLASS_ADAPT","M10 − S_CLASS":"M10_MINUS_S_CLASS_ADAPT","S_CLASS − S":"S_CLASS_ADAPT_MINUS_S_ADAPT"}[comp]
        c=bcore(key,"coverage_per_image");q=bcore(key,"quality_per_image")
        report.append(f"| {comp} | {c.observed_delta:+.4f} [{c.ci95_low:+.4f}, {c.ci95_high:+.4f}] | {q.observed_delta:+.4f} [{q.ci95_low:+.4f}, {q.ci95_high:+.4f}] | {100*core_delta(left,right,'AP'):+.3f} | {100*core_delta(left,right,'AR100'):+.3f} |")
    report += ["","The class-weighted score proxy does **not** explain the learned gains. Relative to S_ADAPT it trades away 0.5833 coverage/image and 0.1039 QUALITY/image, despite +0.225 AP points and +1.629 AR100 points. Its core car coverage recall falls 6.31 points, while person, bicycle, bus and truck improve; this is a real class/metric trade-off rather than a uniformly stronger baseline.","",
               "M10 and M11 both recover substantially more coverage and QUALITY than S_CLASS. M11 adds 0.6768 coverage/image and 0.2224 QUALITY/image with bootstrap intervals wholly above zero, and adds 0.245 AP points, but AR100 is 0.190 points lower than S_CLASS. Compared with S_CLASS, M11 restores car recall by 6.15 points while giving back some of S_CLASS's minority-class gains (for example truck −2.22 points and person −0.39 points).", "",
               "Against the original S_ADAPT screen, M10 and M11 remain `PASS`; S_CLASS is `TRADEOFF` because its supported car recall loss exceeds the frozen 2-point screen. This screen is descriptive engineering triage, not non-inferiority, fairness, safety or statistical proof.","",
               "## Numerical equivalence and implementation optimization","",
               f"The batched-PCA/scaler M11 implementation passed all {int((parity.scope=='FULL_DEV_750_UNITS').sum())} DEV group×seed×budget units: 30,000 K cells and all candidate-ID prefixes had zero mismatch. Maximum saved-probability deviation was {max(impl['probability_max_abs_by_seed'].values()):.3e}; continuous outputs are within atol=1e-6/rtol=1e-5, while selections are exact. Profile inputs were bitwise identical for features/probabilities/objectives in the recorded units, but no universal bitwise-equivalence claim is made.","",
               "The verified S_ADAPT fast path also has exact K, ordered-ID and float64-objective parity on every fixed profile group and core budget. It relies only on a checked identity between score sorting and frozen road8 rank; it does not touch native tensors.","",
               "## Same-run policy cost","",
               "Times below are medians for one 40-image group and one budget, starting from required raw state already in host memory and ending at K plus ordered record IDs. Each cell has 100 measured samples after five warm-ups per group/condition.","",
               "| Path | K10 ms | K15 ms | K20 ms |","|---|---:|---:|---:|"]
    for p,label in zip(policies,labels): report.append(f"| {label} | {rt(p,10).median_ms:.3f} | {rt(p,15).median_ms:.3f} | {rt(p,20).median_ms:.3f} |")
    comp=runtime.set_index("timing_scope"); report += ["",f"Component medians (cached-intermediate diagnostics, not summed into a fabricated total) were: reference feature/PCA/scaler {comp.loc['COMPONENT_FEATURE_REFERENCE','median_ms']:.3f} ms, optimized feature/PCA/scaler {comp.loc['COMPONENT_FEATURE_OPTIMIZED','median_ms']:.3f} ms, one model+temperature {comp.loc['COMPONENT_SINGLE_MODEL_TEMPERATURE','median_ms']:.3f} ms, and QUALITY+exact DP {comp.loc['COMPONENT_QUALITY_EXACT_DP','median_ms']:.3f} ms. Feature/preprocessing remains the largest cost; exact DP is second; model inference is small.","",
               f"M11 reference-to-optimized mean-of-budget medians improves by {impl['speed_ratio_reference_over_optimized']:.2f}× under this same-run boundary. This is policy-processing optimization only—not detector, end-to-end, single-frame or online-vehicle acceleration. Forty-image latency divided by 40 would only be amortized ms/image.","",
               "## Historical timing reconciliation","","R0's 6.598 ms and P1A's 64–65 ms used materially different inputs, outputs, budget batching and scheduling. The current same-run S reference is about 24.6–24.7 ms and the exact verified fast path about 1.24–1.29 ms. The boundary difference is now identified, but the historical scheduling epoch is not fully recreated; status remains `PARTIALLY_EXPLAINED`. See `runtime_path_reconciliation.md`.","",
               "## Answers","",
               "1. **How much does class-weighted score explain?** It explains the direction of minority-class/AP/AR rebalancing, but not the learned coverage/QUALITY gain; by itself it strongly suppresses car and loses micro coverage.",
               "2. **Do M10/M11 add value?** Yes on this observed DEV: both exceed S_CLASS for coverage and QUALITY with intervals above zero. M11 also raises AP but gives back a small amount of S_CLASS AR100.",
               "3. **Does standard quality justify complexity?** The learned policies offer a better balance than S_CLASS, not dominance on every metric/class. Neural necessity is strengthened relative to this particular strong baseline, but remains a DEV result rather than independent confirmation.",
               "4. **What dominates cost?** Frozen 90-D feature/PCA/scaler construction, then exact DP; the neural forward itself is small.",
               "5. **What happened to S_ADAPT timing?** Most of the historical disparity is boundary/adapter work, not the allocator. A residual cross-run difference remains, so it is only partially closed.","",
               "## Limitations","","- DEV2K was already observed; no independent confirmation or novelty claim.","- One detector, frozen Top100 candidate records, frozen 40-image groups and original-score prefixes only.","- S_CLASS uses an uncalibrated class-weighted detector score proxy; it is not a marginal-coverage probability.","- Three fixed models are reported separately and averaged only after evaluation; retraining variance is not covered.","- No TEST/RESERVE/old holdout/Road1000, detector forward, candidate re-export or Coverage model.","- A delegated input audit read the R0 group manifest outside the requested R0 timing-only scope. It was read-only, contained no GT/model, and was not used by the P1B implementation or science; P1 grouping came solely from frozen P1 predictions.","",
               "## Final status","","`execution_status = COMPLETE`","","`baseline_analysis_status = BASELINE_ANALYSIS_COMPLETE_M11_RETAINS_INCREMENT`","",f"`optimization_status = {impl['optimization_status']}`","","`historical_timing_status = PARTIALLY_EXPLAINED`","","`confirmation_status = NOT_AUTHORIZED`",""]
    (root/"LC_ALLOC_P1B_REPORT.md").write_text("\n".join(report),encoding="utf-8")
    append_log(root,f"STAGE summarize_p1b COMPLETE elapsed_seconds={time.perf_counter()-t0:.6f}")


if __name__=="__main__": main()
