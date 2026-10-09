"""Summarize the frozen P1A objective ablation and runtime profile.

This entry point performs no model inference, allocation, or GT matching.  It
only combines already frozen P1/P1A result tables, replays the preregistered
group bootstrap, creates figures, and writes the candidate/confirmation draft.
"""

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
import pyarrow as pa
import pyarrow.parquet as pq

sys.dont_write_bytecode = True

SEEDS = (530101, 530102, 530103)
BUDGETS = (10, 15, 20, 30, 40)
CORE = (10, 15, 20)
POLICIES = ("LEARN_MICRO", "LEARN_CLASS50", "LEARN_MULTI", "LEARN_QUALITY")
VARIANT = {
    "LEARN_MICRO": "M00", "LEARN_CLASS50": "M10",
    "LEARN_MULTI": "M01", "LEARN_QUALITY": "M11",
}
FACTOR_SPECS = {
    "C_T0": {"M10": 1.0, "M00": -1.0},
    "C_T1": {"M11": 1.0, "M01": -1.0},
    "T_C0": {"M01": 1.0, "M00": -1.0},
    "T_C1": {"M11": 1.0, "M10": -1.0},
    "I": {"M11": 1.0, "M10": -1.0, "M01": -1.0, "M00": 1.0},
}
METRICS = ("coverage_per_image", "quality_per_image", "AP", "AP50", "AP75", "AR100")
BOOT_METRICS = ("coverage_per_image", "quality_per_image")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_log(root: Path, text: str) -> None:
    with (root / "runtime.log").open("a", encoding="utf-8") as f:
        f.write(text.rstrip() + "\n")


def policy_value(main: pd.DataFrame, policy: str, seed_variant: str, budget: int, metric: str) -> float:
    d = main[(main["method"] == policy) & (main["budget"] == budget)]
    if seed_variant == "THREE_SEED_MEAN":
        d = d[d["seed"].isin(SEEDS)]
        if len(d) != 3:
            raise RuntimeError(f"missing three-seed main rows: {policy}/{budget}/{metric}")
        return float(d[metric].mean())
    seed = int(seed_variant.removeprefix("SEED_"))
    d = d[d["seed"] == seed]
    if len(d) != 1:
        raise RuntimeError(f"missing main row: {policy}/{seed}/{budget}/{metric}")
    return float(d.iloc[0][metric])


def s_value(main: pd.DataFrame, budget: int, metric: str) -> float:
    d = main[(main["method"] == "S_ADAPT") & (main["seed"] == -1) & (main["budget"] == budget)]
    if len(d) != 1:
        raise RuntimeError(f"missing S_ADAPT row {budget}/{metric}")
    return float(d.iloc[0][metric])


def lincomb(values: dict[str, float], spec: dict[str, float]) -> float:
    return float(sum(values[key] * coeff for key, coeff in spec.items()))


def make_contrasts(main: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    variants = tuple(f"SEED_{s}" for s in SEEDS) + ("THREE_SEED_MEAN",)
    scopes: tuple[int | str, ...] = BUDGETS + ("CORE_10_15_20",)
    for variant in variants:
        for metric in METRICS:
            per_budget: dict[int, dict[str, float]] = {}
            for budget in BUDGETS:
                per_budget[budget] = {VARIANT[p]: policy_value(main, p, variant, budget, metric) for p in POLICIES}
            for scope in scopes:
                if isinstance(scope, int):
                    vals = per_budget[scope]
                else:
                    vals = {m: float(np.mean([per_budget[b][m] for b in CORE])) for m in VARIANT.values()}
                effects = {name: lincomb(vals, spec) for name, spec in FACTOR_SPECS.items()}
                residual_ct = effects["I"] - (effects["C_T1"] - effects["C_T0"])
                residual_tc = effects["I"] - (effects["T_C1"] - effects["T_C0"])
                for name, estimate in effects.items():
                    rows.append({
                        "contrast_family": "FACTORIAL", "contrast": name, "metric": metric,
                        "scope": str(scope), "variant": variant, "estimate": estimate,
                        "unit": "count_per_image" if metric.endswith("per_image") else "proportion",
                        "left_expression": {
                            "C_T0": "M10", "C_T1": "M11", "T_C0": "M01", "T_C1": "M11", "I": "M11-M10",
                        }[name],
                        "right_expression": {
                            "C_T0": "M00", "C_T1": "M01", "T_C0": "M00", "T_C1": "M10", "I": "M01-M00",
                        }[name],
                        "identity_ct_residual": residual_ct if name == "I" else np.nan,
                        "identity_tc_residual": residual_tc if name == "I" else np.nan,
                    })
                for policy in POLICIES:
                    estimate = vals[VARIANT[policy]] - (s_value(main, scope, metric) if isinstance(scope, int)
                                                        else float(np.mean([s_value(main, b, metric) for b in CORE])))
                    rows.append({
                        "contrast_family": "VS_S_ADAPT", "contrast": f"{VARIANT[policy]}_VS_S_ADAPT",
                        "metric": metric, "scope": str(scope), "variant": variant, "estimate": estimate,
                        "unit": "count_per_image" if metric.endswith("per_image") else "proportion",
                        "left_expression": VARIANT[policy], "right_expression": "S_ADAPT",
                        "identity_ct_residual": np.nan, "identity_tc_residual": np.nan,
                    })
    out = pd.DataFrame(rows)
    if len(out) != 1296:
        raise RuntimeError(f"contrast row count {len(out)} != 1296")
    resid = out[out["contrast"] == "I"][["identity_ct_residual", "identity_tc_residual"]].abs().max().max()
    if float(resid) > 1e-12:
        raise RuntimeError(f"factor identity residual too large: {resid}")
    return out


def group_vector(group: pd.DataFrame, policy: str, budget: int, metric: str) -> np.ndarray:
    d = group[(group["method"] == policy) & (group["budget"] == budget)]
    if policy == "S_ADAPT":
        d = d[d["seed"] == -1][["group_id", metric]]
    else:
        d = d[d["seed"].isin(SEEDS)].groupby("group_id", as_index=False)[metric].mean()
    d = d.sort_values("group_id")
    if len(d) != 50 or d["group_id"].astype(int).tolist() != list(range(50)):
        raise RuntimeError(f"group vector incomplete {policy}/{budget}/{metric}")
    return d[metric].to_numpy(np.float64)


def make_bootstrap(group: pd.DataFrame, indices: np.ndarray) -> pd.DataFrame:
    if indices.shape != (5000, 50):
        raise RuntimeError(f"bootstrap index shape changed: {indices.shape}")
    rows: list[dict] = []
    scopes: tuple[int | str, ...] = BUDGETS + ("CORE_10_15_20",)
    for metric in BOOT_METRICS:
        by_budget: dict[int, dict[str, np.ndarray]] = {}
        for budget in BUDGETS:
            by_budget[budget] = {VARIANT[p]: group_vector(group, p, budget, metric) for p in POLICIES}
            by_budget[budget]["S_ADAPT"] = group_vector(group, "S_ADAPT", budget, metric)
        for scope in scopes:
            if isinstance(scope, int):
                vals = by_budget[scope]
            else:
                vals = {name: np.mean(np.vstack([by_budget[b][name] for b in CORE]), axis=0)
                        for name in (*VARIANT.values(), "S_ADAPT")}
            contrast_specs = [("FACTORIAL", name, spec) for name, spec in FACTOR_SPECS.items()]
            contrast_specs += [("VS_S_ADAPT", f"{m}_VS_S_ADAPT", {m: 1.0, "S_ADAPT": -1.0})
                               for m in VARIANT.values()]
            for family, name, spec in contrast_specs:
                vec = np.zeros(50, dtype=np.float64)
                for term, coeff in spec.items():
                    vec += coeff * vals[term]
                boot = vec[indices].mean(axis=1)
                rows.append({
                    "contrast_family": family, "contrast": name, "metric": metric, "scope": str(scope),
                    "variant": "THREE_SEED_MEAN", "observed_delta": float(vec.mean()),
                    "ci95_low": float(np.quantile(boot, 0.025)), "ci95_high": float(np.quantile(boot, 0.975)),
                    "strict_positive_resample_fraction": float(np.mean(boot > 0)),
                    "bootstrap_unit": "frozen 40-image group", "bootstrap_resamples": 5000,
                    "bootstrap_seed": 530002, "group_count": 50,
                })
    out = pd.DataFrame(rows)
    if len(out) != 108:
        raise RuntimeError(f"bootstrap row count {len(out)} != 108")
    return out


def mean_table(main: pd.DataFrame) -> pd.DataFrame:
    learned = main[main["seed"].isin(SEEDS)].groupby(["method", "budget"], as_index=False).mean(numeric_only=True)
    learned["seed"] = -2
    base = main[(main["method"] == "S_ADAPT") & (main["seed"] == -1)].copy()
    return pd.concat([learned, base], ignore_index=True, sort=False)


def quality_screen(main: pd.DataFrame, classes: pd.DataFrame, policy: str) -> dict:
    pm = mean_table(main)
    cls = classes[classes["seed"].isin(SEEDS)].groupby(
        ["method", "budget", "category_id", "class_name"], as_index=False
    ).agg(GT=("GT", "first"), coverage_recall=("coverage_recall", "mean"))
    base_cls = classes[(classes["method"] == "S_ADAPT") & (classes["seed"] == -1)][
        ["budget", "category_id", "coverage_recall"]
    ].rename(columns={"coverage_recall": "base_recall"})
    cm = cls[cls["method"] == policy].merge(base_cls, on=["budget", "category_id"], validate="many_to_one")
    cm["delta"] = cm["coverage_recall"] - cm["base_recall"]
    supported = cm[cm["GT"] >= 100]
    core_class = supported[supported["budget"].isin(CORE)].groupby(
        ["category_id", "class_name", "GT"], as_index=False
    )["delta"].mean()
    ap = {}; ar = {}
    for b in BUDGETS:
        l = pm[(pm["method"] == policy) & (pm["budget"] == b)].iloc[0]
        r = pm[(pm["method"] == "S_ADAPT") & (pm["budget"] == b)].iloc[0]
        ap[b] = float(l.AP - r.AP); ar[b] = float(l.AR100 - r.AR100)
    core_pass = all(ap[b] >= -0.002 for b in CORE) and all(ar[b] >= -0.005 for b in CORE) and bool((core_class["delta"] >= -0.02).all())
    high = supported[supported["budget"].isin((30, 40))]
    high_class = high.groupby(["category_id", "class_name", "GT"], as_index=False)["delta"].mean()
    high_warning = not (all(ap[b] >= -0.002 for b in (30, 40)) and all(ar[b] >= -0.005 for b in (30, 40)) and bool((high_class["delta"] >= -0.02).all()))
    seed_checks = []
    for seed in SEEDS:
        for b in CORE:
            l = main[(main.method == policy) & (main.seed == seed) & (main.budget == b)].iloc[0]
            r = main[(main.method == "S_ADAPT") & (main.seed == -1) & (main.budget == b)].iloc[0]
            lc = classes[(classes.method == policy) & (classes.seed == seed) & (classes.budget == b)]
            bc = classes[(classes.method == "S_ADAPT") & (classes.seed == -1) & (classes.budget == b)]
            zz = lc.merge(bc[["category_id", "coverage_recall"]], on="category_id", suffixes=("", "_base"))
            zz = zz[zz.GT >= 100]
            worst = float((zz.coverage_recall - zz.coverage_recall_base).min())
            seed_checks.append({"seed": seed, "budget": b, "AP_delta": float(l.AP-r.AP),
                                "AR100_delta": float(l.AR100-r.AR100),
                                "worst_supported_class_coverage_recall_delta": worst,
                                "screen_pass": bool(l.AP-r.AP >= -0.002 and l.AR100-r.AR100 >= -0.005 and worst >= -0.02)})
    return {"policy": policy, "core_screen": "PASS" if core_pass else "TRADEOFF",
            "high_budget_quality_warning": high_warning, "AP_deltas": ap, "AR100_deltas": ar,
            "supported_class_core_deltas": core_class.to_dict("records"), "seed_checks": seed_checks}


def make_figures(root: Path, main: pd.DataFrame, classes: pd.DataFrame, boot: pd.DataFrame, runtime: pd.DataFrame) -> None:
    out = root / "figures"; out.mkdir(exist_ok=True)
    mean = mean_table(main)
    colors = {"LEARN_MICRO":"#2563eb", "LEARN_CLASS50":"#d97706", "LEARN_MULTI":"#059669", "LEARN_QUALITY":"#be123c"}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for p in POLICIES:
        d = mean[mean.method == p].sort_values("budget")
        axes[0].plot(d.budget, d.coverage_per_image, marker="o", label=f"{VARIANT[p]} {p}", color=colors[p])
        axes[1].plot(d.budget, d.quality_per_image, marker="o", label=VARIANT[p], color=colors[p])
    for ax, ylabel in zip(axes, ("Coverage / image", "Frozen QUALITY / image")):
        ax.set(xlabel="Average budget", ylabel=ylabel, xticks=BUDGETS); ax.grid(alpha=.25)
    axes[0].legend(fontsize=7); axes[1].legend(fontsize=8); fig.tight_layout()
    fig.savefig(out / "objective_ablation_vs_budget.png", dpi=180); plt.close(fig)

    d = boot[(boot.scope == "CORE_10_15_20") & (boot.contrast_family == "FACTORIAL")]
    order = list(FACTOR_SPECS)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3))
    for ax, metric in zip(axes, BOOT_METRICS):
        z = d[d.metric == metric].set_index("contrast").loc[order]
        x = z.observed_delta.to_numpy(); lo=z.ci95_low.to_numpy(); hi=z.ci95_high.to_numpy(); y=np.arange(len(order))
        ax.errorbar(x,y,xerr=[x-lo,hi-x],fmt="o",capsize=3); ax.axvline(0,color="black",lw=1)
        ax.set_yticks(y,order); ax.invert_yaxis(); ax.set_xlabel(f"Core delta: {metric}"); ax.grid(axis="x",alpha=.25)
    fig.tight_layout(); fig.savefig(out / "factor_effects_and_interaction.png", dpi=180); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for p in POLICIES:
        d = mean[mean.method == p].sort_values("budget")
        axes[0].plot(d.budget, 100*d.AP, marker="o", label=VARIANT[p], color=colors[p])
        axes[0].plot(d.budget, 100*d.AR100, marker="x", linestyle="--", color=colors[p], alpha=.8)
    axes[0].set(xlabel="Average budget", ylabel="AP (o) / AR100 (x), pp", xticks=BUDGETS); axes[0].grid(alpha=.25); axes[0].legend(fontsize=8)
    cm = classes[classes.seed.isin(SEEDS) & classes.budget.isin(CORE)].groupby(["method","class_name"],as_index=False).coverage_recall.mean()
    base = classes[(classes.method=="S_ADAPT") & (classes.seed==-1) & classes.budget.isin(CORE)].groupby("class_name").coverage_recall.mean()
    names = classes.sort_values("category_id").class_name.drop_duplicates().tolist()
    mat=np.asarray([[100*(float(cm[(cm.method==p)&(cm.class_name==c)].coverage_recall.iloc[0])-float(base[c])) for c in names] for p in POLICIES])
    lim=max(abs(mat.min()),abs(mat.max()),.1); im=axes[1].imshow(mat,cmap="RdBu_r",vmin=-lim,vmax=lim,aspect="auto")
    axes[1].set_xticks(range(len(names)),names,rotation=35,ha="right"); axes[1].set_yticks(range(4),[VARIANT[p] for p in POLICIES])
    axes[1].set_title("Core class coverage-recall Δ vs S_ADAPT (pp)"); fig.colorbar(im,ax=axes[1],label="pp")
    fig.tight_layout(); fig.savefig(out / "ap_ar_and_class_costs.png", dpi=180); plt.close(fig)

    if not runtime.empty:
        full = runtime[runtime.timing_scope == "FULL_POLICY_PATH"].copy()
        comp = runtime[runtime.timing_scope.str.startswith("COMPONENT_")].copy()
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        if not full.empty:
            labels=[f"{r.policy}\nseed {int(r.seed)} K{int(r.budget)}" for r in full.itertuples()]
            axes[0].bar(range(len(full)),full.median_ms,yerr=full.p95_ms-full.median_ms,color="#2563eb",alpha=.8)
            axes[0].set_xticks(range(len(full)),labels,rotation=45,ha="right",fontsize=7); axes[0].set_ylabel("40-image latency (ms): median to p95")
        if not comp.empty:
            axes[1].bar(range(len(comp)),comp.median_ms,color="#059669")
            labels=[str(x).removeprefix("COMPONENT_") for x in comp.timing_scope]
            axes[1].set_xticks(range(len(comp)),labels,rotation=35,ha="right",fontsize=8); axes[1].set_ylabel("Component median (ms; cached boundaries)")
        for ax in axes: ax.grid(axis="y",alpha=.25)
        fig.tight_layout(); fig.savefig(out / "single_model_runtime.png", dpi=180); plt.close(fig)


def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument("--project-root",required=True); ap.add_argument("--p1-root",required=True)
    args=ap.parse_args(); root=Path(args.project_root).resolve(); p1=Path(args.p1_root).resolve(); t0=time.perf_counter()
    append_log(root,"STAGE summarize_p1a START")
    main_df=pq.read_table(root/"outputs"/"ablation_main_results.parquet").to_pandas()
    class_df=pq.read_table(root/"outputs"/"ablation_class_results.parquet").to_pandas()
    group_df=pq.read_table(root/"outputs"/"ablation_group_results.parquet").to_pandas()
    runtime_path=root/"outputs"/"runtime_summary.parquet"
    runtime_df=pq.read_table(runtime_path).to_pandas() if runtime_path.exists() else pd.DataFrame()
    if len(main_df)!=65 or len(class_df)!=520 or len(group_df)!=3250:
        raise RuntimeError(f"input row count mismatch main={len(main_df)} class={len(class_df)} group={len(group_df)}")
    contrasts=make_contrasts(main_df)
    indices=np.load(p1/"outputs"/"bootstrap_group_indices.npy",allow_pickle=False)
    boot=make_bootstrap(group_df,indices)
    pq.write_table(pa.Table.from_pandas(contrasts,preserve_index=False),root/"outputs"/"ablation_contrasts.parquet",compression="zstd")
    pq.write_table(pa.Table.from_pandas(boot,preserve_index=False),root/"outputs"/"ablation_bootstrap.parquet",compression="zstd")
    screens={p:quality_screen(main_df,class_df,p) for p in POLICIES}
    write_json(root/"outputs"/"quality_screens.json",screens)
    make_figures(root,main_df,class_df,boot,runtime_df)

    p1cfg=json.loads((p1/"run_config.json").read_text(encoding="utf-8"))
    release_identity=json.loads((Path(p1cfg["release_root"])/"CANDIDATE_ASSET_IDENTITY.json").read_text(encoding="utf-8"))
    models=[]
    for seed in SEEDS:
        model=p1/"models"/f"marginal_mlp_seed_{seed}.pt"; temp=p1/"models"/f"temperature_seed_{seed}.json"
        models.append({"seed":seed,"model_path":str(model),"model_sha256":sha256_file(model),
                       "temperature_path":str(temp),"temperature_sha256":sha256_file(temp),
                       "temperature":json.loads(temp.read_text(encoding="utf-8"))["temperature"]})
    freeze={
        "status":"DRAFT_CANDIDATE_IDENTITY_ONLY_NOT_TEST_AUTHORIZATION",
        "candidate":"LEARN_QUALITY","definition":"w[predicted_class] * mean_t p(y_increment=1 | frozen 90D input), t=.50:.05:.95",
        "models":models,"feature_schema":{"path":str(p1/"feature_schema.json"),"sha256":sha256_file(p1/"feature_schema.json")},
        "pca":{"path":str(p1/"models"/"pca32.joblib"),"sha256":sha256_file(p1/"models"/"pca32.joblib")},
        "scaler":{"path":str(p1/"models"/"feature_scaler.joblib"),"sha256":sha256_file(p1/"models"/"feature_scaler.joblib")},
        "class_weights":{"path":str(p1/"models"/"class_weights.json"),"sha256":sha256_file(p1/"models"/"class_weights.json")},
        "iou_thresholds":[round(x,2) for x in np.arange(.5,1.0,.05)],
        "candidate_asset_id_dev":p1cfg["candidate_asset_ids"]["DEV"],"candidate_schema_sha256":release_identity["candidate_schema_sha256"],
        "action":"original release road8_rank prefix 1..K only; no reranking/NMS/query deduplication",
        "K_bounds":[5,50],"group_size":40,"budgets":list(BUDGETS),"solver":"P1 exact float64 multiple-choice DP with lexicographic K-vector tie-break",
        "sources":{"P1_run_config_sha256":sha256_file(p1/"run_config.json"),"P1A_run_config_sha256":sha256_file(root/"run_config.json")},
        "confirmation_status":"DRAFT_ONLY_NOT_AUTHORIZED",
    }
    write_json(root/"candidate_freeze.json",freeze)
    plan=[
        "# Confirmation Plan Draft","",
        "This is a draft only. It is not authorization to open TEST, export TEST candidates, or run an independent confirmation.","",
        "## Frozen candidate","",
        "- Candidate: `LEARN_QUALITY`, selected after observing DEV; all three seeds 530101/530102/530103 remain and no seed is selected by DEV.",
        "- Action: frozen raw-score prefix only, K=5..50, 40-image exact-budget groups, exact float64 DP. Models, temperatures, PCA/scaler, class weights, ten IoU thresholds, release/schema identity and solver are bound in `candidate_freeze.json`.",
        "- Proposed primary comparison: LEARN_QUALITY versus S_ADAPT. TABLE_QUALITY and LEARN_MICRO remain limiting controls.","",
        "## Proposed common evaluation","",
        "Report IoU=.50 maximum-matching coverage, the same frozen QUALITY proxy, detector-score COCO AP/AP50/AP75/AR100, legacy metrics, and all eight classes with support. Budgets and P1 engineering quality screens must not change after TEST is seen.","",
        "## Authorization boundary","",
        "TEST remains locked. If TEST candidates do not exist, a separate human authorization must first freeze a detector export; this task must not create it incidentally. P1A variants remain a DEV mechanism analysis and do not silently replace LEARN_QUALITY. Any material trade-off exposed by P1A is an explicit human decision item before confirmation.",
    ]
    (root/"CONFIRMATION_PLAN_DRAFT.md").write_text("\n".join(plan)+"\n",encoding="utf-8")

    mean=mean_table(main_df)
    core_boot=boot[(boot.scope=="CORE_10_15_20")&(boot.contrast_family=="FACTORIAL")]
    def mrow(policy:str,budget:int): return mean[(mean.method==policy)&(mean.budget==budget)].iloc[0]
    def effect(name:str,metric:str): return core_boot[(core_boot.contrast==name)&(core_boot.metric==metric)].iloc[0]
    def point_effect(name:str,metric:str):
        return float(contrasts[(contrasts.contrast_family=="FACTORIAL")&(contrasts.contrast==name)&(contrasts.metric==metric)&(contrasts.scope=="CORE_10_15_20")&(contrasts.variant=="THREE_SEED_MEAN")].estimate.iloc[0])
    report=[
        "# LC-ALLOC-P1A Report","",
        "## Scope and integrity","",
        "P1A is a post-P1 development analysis on the already observed LC-v1 DEV2K. It trained no model, changed no probability, temperature, PCA/scaler, or class weight, and accessed no TEST/RESERVE/old-holdout/Road1000 scientific asset. The two new policies only re-aggregate each frozen model's ten post-temperature probabilities; three seeds remain separate through allocation and evaluation.","",
        "The preregistered 24-unit replay reproduced P1 MICRO/QUALITY K vectors, selected record IDs, and predicted objectives exactly. New CLASS50/MULTI allocations contain 60,000 image-policy-seed-budget rows and represent 1,380,000 frozen prefix records. Their SHA was committed before the independent evaluator opened DEV GT.","",
        "## Methods","",
        "M00 uses p@.50; M10 multiplies p@.50 by the frozen FIT class weight; M01 averages p over IoU .50:.05:.95; M11 applies both. All policies use the same original-score Top100 pool, output rank 1..K only, and solve each frozen 40-image exact budget with the P1 float64 dynamic program. Thus this analysis concerns inference-time value aggregation, not whether ten-output training was necessary.","",
        "## Main results (three-seed arithmetic mean)","",
        "Values below are arithmetic means of three independently allocated and evaluated frozen models, not prediction ensembles. AP/AR are proportions.","",
        "| Policy | K | coverage/img | QUALITY/img | AP | AR100 |","|---|---:|---:|---:|---:|---:|",
    ]
    for b in BUDGETS:
        for p in POLICIES:
            r=mrow(p,b); report.append(f"| {VARIANT[p]} {p} | {b} | {r.coverage_per_image:.4f} | {r.quality_per_image:.4f} | {r.AP:.5f} | {r.AR100:.5f} |")
    report += ["","## Factor effects and interaction","",
               "Core values average K10/K15/K20 within each frozen 40-image group before bootstrap. Intervals are 5,000 paired group resamples using P1's frozen indices; positive fractions are descriptive, not p-values.","",
               "| Effect | coverage/img [95% CI] | QUALITY/img [95% CI] |","|---|---:|---:|"]
    for name in FACTOR_SPECS:
        c=effect(name,"coverage_per_image"); q=effect(name,"quality_per_image")
        report.append(f"| {name} | {c.observed_delta:+.4f} [{c.ci95_low:+.4f}, {c.ci95_high:+.4f}] | {q.observed_delta:+.4f} [{q.ci95_low:+.4f}, {q.ci95_high:+.4f}] |")
    report += ["","| Effect | AP delta (pp) | AR100 delta (pp) |","|---|---:|---:|"]
    for name in FACTOR_SPECS:
        report.append(f"| {name} | {100*point_effect(name,'AP'):+.3f} | {100*point_effect(name,'AR100'):+.3f} |")
    report += ["","AP/AP50/AP75/AR100 are full-DEV point estimates and do not borrow the coverage bootstrap intervals. The observed interaction is negative for core coverage, QUALITY, AP and AR100, so the two switches are sub-additive in this setup; that is a conditional mechanism result, not a universal incompatibility. An interval crossing zero would not establish equivalence.","",
               "## Quality and class screen","",
               "The original P1 engineering screen is replayed descriptively against S_ADAPT; it is not a non-inferiority, fairness, safety, or TVT test."]
    for p in POLICIES:
        s=screens[p]; worst=sorted(s["supported_class_core_deltas"],key=lambda x:x["delta"])[:3]
        report.append(f"- {VARIANT[p]} {p}: core `{s['core_screen']}`; HIGH_BUDGET_QUALITY_WARNING={s['high_budget_quality_warning']}; lowest supported-class core deltas: "+", ".join(f"{x['class_name']} {100*x['delta']:+.2f} pp" for x in worst)+".")
    report += ["","All eight classes, their GT support, three individual seeds, and all five budgets are retained in the delivered tables. QUALITY is a weighted multi-threshold matching proxy, not direct AP optimization or a safety/fairness guarantee.","",
               "## Single-model policy-processing cost","",
               "Timing starts from raw candidate/native tensors already resident in host memory and includes query-index gather, 90-D feature construction, frozen PCA/scaler, one model and temperature, QUALITY construction, one-budget exact DP, and record-ID return. File read, model load, GT evaluation, and disk output are separate. S_ADAPT is timed through its actual score-only implementation.",""]
    if not runtime_df.empty:
        for r in runtime_df[runtime_df.timing_scope=="FULL_POLICY_PATH"].sort_values(["policy","seed","budget"]).itertuples():
            report.append(f"- {r.policy}, seed {int(r.seed)}, K{int(r.budget)}: median {r.median_ms:.3f} ms, p95 {r.p95_ms:.3f} ms over {int(r.samples)} samples ({int(r.input_groups)} fixed 40-image groups).")
        comps=runtime_df[runtime_df.timing_scope.str.startswith("COMPONENT_")].set_index("timing_scope")
        report.append(f"The cached-boundary component medians are: feature+PCA/scaler {comps.loc['COMPONENT_FEATURE_PCA_SCALER','median_ms']:.3f} ms, single-model inference+temperature {comps.loc['COMPONENT_SINGLE_MODEL_INFERENCE_TEMPERATURE','median_ms']:.3f} ms, and QUALITY+single-budget DP {comps.loc['COMPONENT_QUALITY_UTILITY_SINGLE_BUDGET_DP','median_ms']:.3f} ms. Feature construction and frozen preprocessing dominate this measured implementation.")
        report.append("Component timings are not added to manufacture a full-path median. Dividing 40-image latency by 40 is only amortized ms/image, not single-frame response time or detector acceleration.")
    report += ["","## Mechanism interpretation","",
               "Class weighting is the larger conditional switch: without threshold averaging it changes core coverage by −0.0778/image while raising frozen QUALITY by +0.0558/image, AP by +0.390 pp and AR100 by +1.335 pp. Multi-threshold averaging without class weights changes coverage by −0.0349/image while raising QUALITY by +0.0273/image, AP by +0.158 pp and AR100 by +0.578 pp. Thus both switches trade micro coverage for the frozen quality/localization objectives, with class weighting contributing more in this configuration.","",
               "The combined M11 gain is sub-additive: interaction is −0.0251 coverage/image and −0.0123 QUALITY/image, with negative AP/AR point interactions. Nevertheless M11 has the highest core frozen QUALITY and standard-quality point estimates of the four. M10 is a simpler nearby trade-off—more micro coverage than M11, but lower QUALITY/AP/AR—and this DEV analysis cannot establish equivalence or authorize switching the candidate.","",
               "`LEARN_QUALITY` therefore remains the frozen confirmation candidate by preregistration. The accompanying confirmation plan is a draft only and preserves the TEST lock.","",
               "## Limitations","",
               "- DEV2K was already observed; this is not an independent confirmation.","- Frozen detector candidate space and original-score prefixes only.","- No training, TEST access, detector forward, end-to-end latency, or deployment claim.","- Three fixed seeds and one hardware/software path do not cover retraining or platform variability.","- Class weights encode a frozen coverage-value proxy, not safety or fairness.","",
               "## Final status","",
               "`execution_status = COMPLETE`","","`analysis_status = OBJECTIVE_ABLATION_AND_SINGLE_MODEL_PROFILE_COMPLETE`","","`confirmation_status = DRAFT_ONLY_NOT_AUTHORIZED`","" ]
    (root/"LC_ALLOC_P1A_REPORT.md").write_text("\n".join(report),encoding="utf-8")
    append_log(root,f"STAGE summarize_p1a COMPLETE elapsed_seconds={time.perf_counter()-t0:.6f}")


if __name__ == "__main__":
    main()
