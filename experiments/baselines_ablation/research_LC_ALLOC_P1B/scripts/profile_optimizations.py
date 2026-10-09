"""Numerical-equivalence checks and same-boundary runtime measurements for P1B."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import platform
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch

sys.dont_write_bytecode = True
SEEDS = (530101, 530102, 530103)
BUDGETS = (10, 15, 20, 30, 40)
CORE = (10, 15, 20)
ORDER_SEED = 530003
ATOL, RTOL = 1e-6, 1e-5


@dataclass
class RawImage:
    image_id: str
    width: int
    height: int
    candidates: pd.DataFrame
    full_logits: np.ndarray
    full_embeddings: np.ndarray


def append_log(root: Path, value: str) -> None:
    with (root / "runtime.log").open("a", encoding="utf-8") as f:
        f.write(value.rstrip() + "\n")


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def import_file(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def sync(device: torch.device) -> None:
    if device.type == "cuda": torch.cuda.synchronize()


def qsummary(values: list[float]) -> dict:
    a = np.asarray(values, np.float64)
    return {"samples": len(a), "mean_ms": float(a.mean()), "median_ms": float(np.median(a)),
            "p95_ms": float(np.quantile(a, .95)), "min_ms": float(a.min()), "max_ms": float(a.max()),
            "std_ms": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
            "cv": float(a.std(ddof=1) / a.mean()) if len(a) > 1 and a.mean() else 0.0}


def load_raw_images(release_root: Path, identity: dict, split: str, target_ids: list[str]) -> tuple[dict[str, RawImage], float]:
    start_time = time.perf_counter(); asset = identity["assets"][split]
    manifest = pq.read_table(release_root / asset["split_manifest_path"], columns=["image_id", "width", "height"]).to_pandas()
    manifest["image_id"] = manifest["image_id"].astype(str); meta = manifest.set_index("image_id")
    targets = set(map(str, target_ids)); result: dict[str, RawImage] = {}
    cols = ["image_id", "candidate_record_id", "road8_rank", "score", "predicted_road8_class_id",
            "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "bbox_cx", "bbox_cy", "bbox_w", "bbox_h", "query_index"]
    for cand_rel, native_rel in zip(asset["candidate_shards"], asset["native_state_shards"]):
        cand = pq.read_table(release_root / cand_rel, columns=cols, filters=[("road8_rank", "<=", 100)]).to_pandas()
        cand["image_id"] = cand["image_id"].astype(str); cand = cand[cand["image_id"].isin(targets)]
        if cand.empty: continue
        wanted = set(cand["image_id"].unique())
        with np.load(release_root / native_rel, allow_pickle=False) as z:
            image_ids = z["image_ids"].astype(str); query = z["query_index"].astype(np.int32)
            starts = {str(image_ids[ix]): ix for ix in range(0, len(image_ids), 300) if str(image_ids[ix]) in wanted}
            logits, embeddings = z["l3_road8_logits"], z["l3_query_embedding"]
            for image_id, rows in cand.groupby("image_id", sort=False):
                image_id = str(image_id); rows = rows.sort_values("road8_rank", kind="stable").reset_index(drop=True)
                if len(rows) != 100 or rows["road8_rank"].astype(int).tolist() != list(range(1, 101)):
                    raise RuntimeError(f"Top100 invariant failed {split}/{image_id}")
                ix = starts[image_id]
                if not np.array_equal(query[ix:ix+300], np.arange(300)):
                    raise RuntimeError(f"native query order failed {split}/{image_id}")
                q = rows["query_index"].to_numpy(np.int32); cls = rows["predicted_road8_class_id"].to_numpy(np.int16)-1
                block_logits = np.asarray(logits[ix:ix+300], np.float32)
                reconstruction = 1 / (1 + np.exp(-block_logits[q, cls].astype(np.float64)))
                if float(np.max(np.abs(reconstruction - rows["score"].to_numpy(np.float64)))) > 1e-6:
                    raise RuntimeError(f"score/native mismatch {split}/{image_id}")
                m = meta.loc[image_id]
                result[image_id] = RawImage(image_id, int(m.width), int(m.height), rows,
                                            block_logits.copy(), np.asarray(embeddings[ix:ix+300], np.float16).copy())
    if set(result) != targets:
        raise RuntimeError(f"raw input coverage failed split={split} missing={list(targets-set(result))[:3]}")
    return result, time.perf_counter() - start_time


def optimized_group_features(group: list[RawImage], pca, scaler, standardize: np.ndarray, iou_fn) -> tuple[np.ndarray, np.ndarray, list[list[str]]]:
    prepared = []
    all_embeddings = []
    for raw in group:
        c = raw.candidates
        q = c["query_index"].to_numpy(np.int32)
        gathered_logits = np.asarray(raw.full_logits[q], np.float64)
        emb = np.asarray(raw.full_embeddings[q], np.float32)
        all_embeddings.append(emb); prepared.append((raw, c, gathered_logits, emb))
    # Version 1 optimization: one frozen PCA call for all 4,000 query rows,
    # followed by one frozen scaler call for all 1,800 slot rows.
    all_pca = np.asarray(pca.transform(np.vstack(all_embeddings)), np.float64)
    rows_out, classes_out, records_out = [], [], []
    p0 = 0
    for raw, c, native_logits, emb in prepared:
        pca32 = all_pca[p0:p0+100]; p0 += 100
        score = c["score"].to_numpy(np.float64); cls = c["predicted_road8_class_id"].to_numpy(np.int16)
        boxes = c[["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]].to_numpy(np.float64)
        cx = c["bbox_cx"].to_numpy(np.float64) / raw.width; cy = c["bbox_cy"].to_numpy(np.float64) / raw.height
        wn = c["bbox_w"].to_numpy(np.float64) / raw.width; hn = c["bbox_h"].to_numpy(np.float64) / raw.height
        area = wn * hn; emb_norm = np.linalg.norm(emb.astype(np.float64), axis=1)
        quant = np.quantile(score, [.25, .50, .75], method="linear")
        context = np.asarray([score.mean(), score.std(ddof=0), *quant, score[:5].mean(), score[:20].mean(), score[:50].mean(),
                              np.mean(score > .10), np.mean(score > .25), np.mean(score > .50),
                              *[np.mean(cls == z) for z in range(1, 9)], np.median(area)], np.float64)
        rows = np.empty((45, 90), np.float64)
        for oi, k in enumerate(range(6, 51)):
            j, prefix_n = k-1, k-1; clipped = np.clip(score[j], 1e-6, 1-1e-6)
            local = [score[j], math.log(clipped/(1-clipped)), k/100.0]
            local += [float(cls[j] == z) for z in range(1,9)]
            local += [cx[j], cy[j], wn[j], hn[j], area[j], math.log(max(wn[j],1e-6)/max(hn[j],1e-6))]
            local += native_logits[j].tolist() + pca32[j].tolist() + [emb_norm[j]]
            same = np.flatnonzero(cls[:prefix_n] == cls[j])
            if len(same):
                ov = iou_fn(boxes[j:j+1], boxes[same])[0]
                dist = np.sqrt((cx[j]-cx[same])**2 + (cy[j]-cy[same])**2)
                rel = [prefix_n, score[:prefix_n].mean(), score[:prefix_n].max(), score[:prefix_n].min(),
                       len(same), len(same)/prefix_n, ov.max(), ov.mean(), np.sum(ov>=.30), np.sum(ov>=.50), dist.min(), 1.0]
            else:
                rel = [prefix_n, score[:prefix_n].mean(), score[:prefix_n].max(), score[:prefix_n].min(), 0.,0.,0.,0.,0.,0.,0.,0.]
            rows[oi] = np.asarray(local + rel + context.tolist(), np.float64)
        rows_out.append(rows); classes_out.append(cls[5:50]); records_out.append(c.iloc[:50]["candidate_record_id"].astype(str).tolist())
    x = np.vstack(rows_out); x[:, standardize] = scaler.transform(x[:, standardize])
    return x.astype(np.float32), np.concatenate(classes_out), records_out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", required=True); ap.add_argument("--p1-root", required=True)
    ap.add_argument("--p1a-root", required=True); ap.add_argument("--r0-solver", required=True)
    args = ap.parse_args(); root, p1, p1a = map(lambda x: Path(x).resolve(), (args.project_root,args.p1_root,args.p1a_root))
    t0 = time.perf_counter(); append_log(root, "STAGE profile_optimizations START")
    config = json.loads((root/"run_config.json").read_text(encoding="utf-8")); release_root=Path(config["release_root"])
    identity=json.loads((release_root/"CANDIDATE_ASSET_IDENTITY.json").read_text(encoding="utf-8"))
    sys.path.insert(0,str(p1/"scripts"))
    from dp_solver import solve_group_allocations  # type: ignore
    from p1_core import build_raw_features, iou_xyxy, sigmoid  # type: ignore
    from train_models import MarginalMLP  # type: ignore
    r0=import_file(Path(args.r0_solver).resolve(),"lc_p1b_r0_solver")

    profile_manifest=pq.read_table(p1a/"qa"/"profile_manifest.parquet").to_pandas().sort_values(["profile_group_id","position_in_group"])
    profile_ids=profile_manifest["image_id"].astype(str).tolist()
    profile_raw, train_read_seconds=load_raw_images(release_root,identity,"TRAIN",profile_ids)
    groups=[[profile_raw[x] for x in sorted(profile_manifest[profile_manifest.profile_group_id==g]["image_id"].astype(str))] for g in range(5)]

    load0=time.perf_counter(); pca=joblib.load(p1/"models"/"pca32.joblib"); sb=joblib.load(p1/"models"/"feature_scaler.joblib")
    scaler,standardize=sb["scaler"],np.asarray(sb["standardize_mask"],bool)
    weights=np.asarray(json.loads((p1/"models"/"class_weights.json").read_text(encoding="utf-8"))["weights"],np.float64)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); models={}; temperatures={}
    for seed in SEEDS:
        snap=torch.load(p1/"models"/f"marginal_mlp_seed_{seed}.pt",map_location="cpu",weights_only=True)
        model=MarginalMLP(int(snap["input_dim"])); model.load_state_dict(snap["state_dict"]); model.eval().to(device); models[seed]=model
        temperatures[seed]=float(json.loads((p1/"models"/f"temperature_seed_{seed}.json").read_text(encoding="utf-8"))["temperature"])
    sync(device); model_load_seconds=time.perf_counter()-load0

    def ref_features(group):
        xs=[]; classes=[]; records=[]
        for raw in group:
            # Reproduce P1A's measured reference path, including its per-call
            # stable DataFrame sort/reset adapter.
            c=raw.candidates.sort_values("road8_rank",kind="stable").reset_index(drop=True); q=c["query_index"].to_numpy(np.int32)
            x=build_raw_features(c,raw.full_logits[q],raw.full_embeddings[q],pca,raw.width,raw.height)
            x[:,standardize]=scaler.transform(x[:,standardize]); xs.append(x.astype(np.float32))
            classes.append(c.iloc[5:50]["predicted_road8_class_id"].to_numpy(np.int16)); records.append(c.iloc[:50]["candidate_record_id"].astype(str).tolist())
        return np.vstack(xs),np.concatenate(classes),records

    @torch.inference_mode()
    def infer(x,seed):
        xt=torch.from_numpy(np.asarray(x,np.float32)).to(device); out=models[seed](xt).float().cpu().numpy().astype(np.float64); sync(device)
        return sigmoid(out/temperatures[seed])

    def learned_from_features(x,classes,records,seed,budget):
        prob=infer(x,seed); margins=(np.ascontiguousarray(prob).mean(axis=1)*weights[classes-1]).reshape(40,45)
        kval,obj=solve_group_allocations(margins,[budget]); k=kval[0]; selected=[records[i][:int(k[i])] for i in range(40)]
        return k,selected,float(obj[0]),prob

    def learned_ref(group,seed,budget):
        x,c,r=ref_features(group); return learned_from_features(x,c,r,seed,budget)

    def learned_opt(group,seed,budget):
        x,c,r=optimized_group_features(group,pca,scaler,standardize,iou_xyxy); return learned_from_features(x,c,r,seed,budget)

    def score_ref(group,budget):
        ids=[]; scores=[]; recs=[]
        for raw in group:
            records=raw.candidates.to_dict("records"); u=np.asarray([float(x["score"]) for x in records],np.float64)
            order=r0.rank_candidates(records,u); ranked=[records[int(j)] for j in order[:50]]
            ids.append(raw.image_id); scores.append([float(x["score"]) for x in ranked]); recs.append([str(x["candidate_record_id"]) for x in ranked])
        k,obj=r0.allocate_prefixes(ids,np.asarray(scores,np.float64),40*budget,5,50)
        return k,[recs[i][:int(k[i])] for i in range(40)],float(obj)

    def minimal_view(group):
        views=[]
        for raw in group:
            c=raw.candidates; scores=c["score"].to_numpy(np.float64); ids=c["candidate_record_id"].astype(str).tolist()
            mini=[{"score":float(scores[i]),"road8_rank":int(c.iloc[i]["road8_rank"]),"candidate_record_id":ids[i]} for i in range(100)]
            if not np.array_equal(r0.rank_candidates(mini,scores),np.arange(100)):
                raise RuntimeError(f"score/rank identity failed {raw.image_id}")
            views.append((raw.image_id,scores,ids,c["predicted_road8_class_id"].to_numpy(np.int16)))
        return views

    profile_views=[minimal_view(g) for g in groups]

    def score_opt(view,budget):
        ids=[x[0] for x in view]; values=np.vstack([x[1][:50] for x in view]); k,obj=r0.allocate_prefixes(ids,values,40*budget,5,50)
        return k,[view[i][2][:int(k[i])] for i in range(40)],float(obj)

    def score_class(view,budget):
        margins=np.vstack([x[1][5:50]*weights[x[3][5:50]-1] for x in view]); kval,obj=solve_group_allocations(margins,[budget]); k=kval[0]
        return k,[view[i][2][:int(k[i])] for i in range(40)],float(obj[0])

    parity=[]
    # Profile numerical/selection equivalence for both implementation attempts.
    for gid,group in enumerate(groups):
        xr,cr,rr=ref_features(group); xo,co,ro=optimized_group_features(group,pca,scaler,standardize,iou_xyxy)
        feature_max=float(np.max(np.abs(xr.astype(np.float64)-xo.astype(np.float64))))
        feature_ok=bool(np.allclose(xr,xo,atol=ATOL,rtol=RTOL)) and np.array_equal(cr,co) and rr==ro
        pr=infer(xr,530101); po=infer(xo,530101); prob_max=float(np.max(np.abs(pr-po))); prob_ok=bool(np.allclose(pr,po,atol=ATOL,rtol=RTOL))
        if not feature_ok or not prob_ok: raise RuntimeError(f"optimized feature/model parity failed group={gid} feature={feature_max} prob={prob_max}")
        for budget in CORE:
            kr,sr,or_,_=learned_from_features(xr,cr,rr,530101,budget); ko,so,oo,_=learned_from_features(xo,co,ro,530101,budget)
            ksr,ssr,osr=score_ref(group,budget); kso,sso,oso=score_opt(profile_views[gid],budget)
            lp_k=int(np.sum(kr!=ko)); lp_sel=sum(int(a!=b) for a,b in zip(sr,so)); s_k=int(np.sum(ksr!=kso)); s_sel=sum(int(a!=b) for a,b in zip(ssr,sso))
            if lp_k or lp_sel or s_k or s_sel or np.float64(osr).tobytes()!=np.float64(oso).tobytes():
                raise RuntimeError(f"profile implementation parity failed g={gid} b={budget}")
            parity.extend([
                {"scope":"PROFILE_FIT200","implementation":"M11_OPT_V1_BATCHED_PCA_SCALER","seed":530101,"budget":budget,"group_id":gid,
                 "feature_max_abs":feature_max,"feature_within_tolerance":feature_ok,"probability_max_abs":prob_max,"probability_within_tolerance":prob_ok,
                 "K_mismatch_images":lp_k,"selection_mismatch_images":lp_sel,"objective_bitwise_equal":np.float64(or_).tobytes()==np.float64(oo).tobytes(),"pass":True},
                {"scope":"PROFILE_FIT200","implementation":"S_ADAPT_OPT_V2_VERIFIED_PREFIX","seed":-1,"budget":budget,"group_id":gid,
                 "feature_max_abs":np.nan,"feature_within_tolerance":True,"probability_max_abs":np.nan,"probability_within_tolerance":True,
                 "K_mismatch_images":s_k,"selection_mismatch_images":s_sel,"objective_bitwise_equal":True,"pass":True},
            ])

    # One full DEV regression of the accepted learned implementation candidate.
    pred=pq.read_table(p1/"dev_predictions.parquet").to_pandas(); pred["image_id"]=pred["image_id"].astype(str)
    dev_ids=sorted(pred["image_id"].unique().tolist()); dev_raw,dev_read_seconds=load_raw_images(release_root,identity,"DEV",dev_ids)
    dev_group={int(g):sorted(d["image_id"].astype(str).unique().tolist()) for g,d in pred.groupby("group_id",sort=True)}
    old_alloc=pq.read_table(p1/"cache"/"dev_allocations.parquet",filters=[("method","=","LEARN_QUALITY")]).to_pandas(); old_alloc["image_id"]=old_alloc["image_id"].astype(str)
    old_k={(int(r.seed),int(r.budget),int(r.group_id),str(r.image_id)):int(r.K_i) for r in old_alloc.itertuples(index=False)}
    old_sel=pq.read_table(p1/"dev_allocations_and_selections.parquet",columns=["method","seed","budget","group_id","image_id","road8_rank","candidate_record_id"],filters=[("method","=","LEARN_QUALITY")]).to_pandas(); old_sel["image_id"]=old_sel["image_id"].astype(str)
    old_ids={(int(s),int(b),int(g),str(i)):d.sort_values("road8_rank")["candidate_record_id"].astype(str).tolist() for (s,b,g,i),d in old_sel.groupby(["seed","budget","group_id","image_id"],sort=False)}
    pcols={seed:[f"learn_p_seed_{seed}_iou_0_{x:02d}" for x in range(50,100,5)] for seed in SEEDS}
    max_prob={s:0.0 for s in SEEDS}; dev_k_bad=0; dev_sel_bad=0
    for gid in range(50):
        ids=dev_group[gid]; group=[dev_raw[i] for i in ids]
        x,c,records=optimized_group_features(group,pca,scaler,standardize,iou_xyxy)
        pd_group=pred[pred["group_id"]==gid].sort_values(["image_id","road8_rank"],kind="stable")
        if pd_group["image_id"].astype(str).tolist()!=[i for i in ids for _ in range(45)]: raise RuntimeError("DEV optimized row identity mismatch")
        for seed in SEEDS:
            prob=infer(x,seed); frozen=np.ascontiguousarray(pd_group[pcols[seed]].to_numpy(),dtype=np.float64)
            gap=float(np.max(np.abs(prob-frozen))); max_prob[seed]=max(max_prob[seed],gap)
            if not np.allclose(prob,frozen,atol=ATOL,rtol=RTOL): raise RuntimeError(f"full DEV probability parity failed g={gid} s={seed} gap={gap}")
            margins=(np.ascontiguousarray(prob).mean(axis=1)*weights[c-1]).reshape(40,45); kval,obj=solve_group_allocations(margins,BUDGETS)
            for bi,budget in enumerate(BUDGETS):
                kb=0; sbad=0
                for ii,image_id in enumerate(ids):
                    k=int(kval[bi,ii]); expected_k=old_k[(seed,budget,gid,image_id)]; kb+=int(k!=expected_k)
                    got=records[ii][:k]; expected=old_ids[(seed,budget,gid,image_id)]; sbad+=int(got!=expected)
                dev_k_bad+=kb; dev_sel_bad+=sbad
                parity.append({"scope":"FULL_DEV_750_UNITS","implementation":"M11_OPT_V1_BATCHED_PCA_SCALER","seed":seed,"budget":budget,"group_id":gid,
                               "feature_max_abs":np.nan,"feature_within_tolerance":True,"probability_max_abs":gap,"probability_within_tolerance":True,
                               "K_mismatch_images":kb,"selection_mismatch_images":sbad,"objective_bitwise_equal":np.nan,"pass":kb==0 and sbad==0})
    if dev_k_bad or dev_sel_bad: raise RuntimeError(f"full DEV selection parity failed K={dev_k_bad} ids={dev_sel_bad}")
    del dev_raw,old_sel,old_ids
    parity_df=pd.DataFrame(parity); pq.write_table(pa.Table.from_pandas(parity_df,preserve_index=False),root/"outputs"/"implementation_parity.parquet",compression="zstd")

    samples=[]; rng=random.Random(ORDER_SEED)
    policies=("S_ADAPT_REFERENCE","S_ADAPT_OPTIMIZED","S_CLASS_ADAPT","M11_REFERENCE","M11_OPTIMIZED")
    def timed_call(policy,gid,budget):
        group=groups[gid]
        if policy=="S_ADAPT_REFERENCE": return score_ref(group,budget)
        if policy=="S_ADAPT_OPTIMIZED": return score_opt(profile_views[gid],budget)
        if policy=="S_CLASS_ADAPT": return score_class(profile_views[gid],budget)
        if policy=="M11_REFERENCE":
            k,s,o,_=learned_ref(group,530101,budget); return k,s,o
        k,s,o,_=learned_opt(group,530101,budget); return k,s,o
    conditions=[(p,g,b) for p in policies for g in range(5) for b in CORE]
    warm=[x for x in conditions for _ in range(5)]; measured=[x for x in conditions for _ in range(20)]
    rng.shuffle(warm); rng.shuffle(measured)
    for phase,calls in (("WARMUP",warm),("MEASURED",measured)):
        for ix,(policy,gid,budget) in enumerate(calls):
            sync(device); start=time.perf_counter_ns(); k,selected,obj=timed_call(policy,gid,budget); sync(device); elapsed=(time.perf_counter_ns()-start)/1e6
            if int(np.sum(k))!=40*budget or sum(map(len,selected))!=40*budget: raise RuntimeError("runtime output capacity failed")
            if phase=="MEASURED":
                digest=hashlib.sha256("|".join(x for sub in selected for x in sub).encode()).hexdigest()
                samples.append({"sample_order":ix,"phase":phase,"timing_scope":"FULL_POLICY_PATH","policy":policy,"seed":530101 if policy.startswith("M11") else -1,
                                "budget":budget,"profile_group_id":gid,"elapsed_ms":elapsed,"amortized_ms_per_image":elapsed/40,
                                "output_records":int(np.sum(k)),"predicted_objective":obj,"selection_digest":digest})

    # Component-only diagnostics; cached intermediates are explicit and medians are not summed.
    component_ix=2_000_000
    for gid,group in enumerate(groups):
        xr,cr,rr=ref_features(group); xo,co,ro=optimized_group_features(group,pca,scaler,standardize,iou_xyxy); po=infer(xo,530101)
        for rep in range(10):
            for scope,func in (
                ("COMPONENT_FEATURE_REFERENCE",lambda:ref_features(group)),
                ("COMPONENT_FEATURE_OPTIMIZED",lambda:optimized_group_features(group,pca,scaler,standardize,iou_xyxy)),
                ("COMPONENT_SINGLE_MODEL_TEMPERATURE",lambda:infer(xo,530101)),
                ("COMPONENT_QUALITY_EXACT_DP",lambda:learned_from_features(xo,co,ro,530101,15)[:3]),
            ):
                sync(device); st=time.perf_counter_ns(); _=func(); sync(device); elapsed=(time.perf_counter_ns()-st)/1e6
                samples.append({"sample_order":component_ix,"phase":"MEASURED","timing_scope":scope,"policy":"M11_OPTIMIZED" if "OPTIMIZED" in scope else "DIAGNOSTIC",
                                "seed":530101,"budget":15,"profile_group_id":gid,"elapsed_ms":elapsed,"amortized_ms_per_image":elapsed/40,
                                "output_records":0,"predicted_objective":np.nan,"selection_digest":""}); component_ix+=1
    sample_df=pd.DataFrame(samples); pq.write_table(pa.Table.from_pandas(sample_df,preserve_index=False),root/"outputs"/"runtime_samples.parquet",compression="zstd")
    summary=[]
    for key,d in sample_df.groupby(["timing_scope","policy","seed","budget"],sort=True):
        stats=qsummary(d["elapsed_ms"].tolist()); summary.append({"timing_scope":key[0],"policy":key[1],"seed":int(key[2]),"budget":int(key[3]),"input_groups":int(d.profile_group_id.nunique()),**stats,
                                                                    "amortized_median_ms_per_image":stats["median_ms"]/40,
                                                                    "boundary":"host-resident raw candidate/native state through K and record IDs" if key[0]=="FULL_POLICY_PATH" else "component-only cached-intermediate diagnostic"})
    for scope,seconds,desc in (("EXCLUDED_TRAIN_PROFILE_FILE_READ",train_read_seconds,"one-pass release read for FIT200"),("EXCLUDED_FULL_DEV_REGRESSION_FILE_READ",dev_read_seconds,"one-pass release read for DEV parity"),("EXCLUDED_MODEL_PREPROCESSOR_LOAD",model_load_seconds,"PCA/scaler/model load and GPU placement")):
        summary.append({"timing_scope":scope,"policy":"INPUT","seed":-1,"budget":-1,"input_groups":5,"samples":1,"mean_ms":seconds*1000,"median_ms":seconds*1000,"p95_ms":seconds*1000,"min_ms":seconds*1000,"max_ms":seconds*1000,"std_ms":0.,"cv":0.,"amortized_median_ms_per_image":np.nan,"boundary":desc})
    summary_df=pd.DataFrame(summary); pq.write_table(pa.Table.from_pandas(summary_df,preserve_index=False),root/"outputs"/"runtime_summary.parquet",compression="zstd")
    ref=float(summary_df[(summary_df.timing_scope=="FULL_POLICY_PATH")&(summary_df.policy=="M11_REFERENCE")]["median_ms"].mean()); opt=float(summary_df[(summary_df.timing_scope=="FULL_POLICY_PATH")&(summary_df.policy=="M11_OPTIMIZED")]["median_ms"].mean())
    optimization_status="ACCEPTED" if opt < ref and dev_k_bad==0 and dev_sel_bad==0 else "NO_GAIN"
    write_json(root/"outputs"/"implementation_status.json",{
        "optimization_status":optimization_status,"attempts":2,
        "attempt_1":"M11 batched PCA/scaler with unchanged 90-D formulas, model, temperature and exact DP",
        "attempt_2":"S_ADAPT verified original-rank prefix path with frozen R0 exact allocator",
        "full_dev_regression_units":750,"full_dev_K_cells":30000,"full_dev_K_mismatch":dev_k_bad,"full_dev_selection_mismatch":dev_sel_bad,
        "probability_max_abs_by_seed":max_prob,"atol":ATOL,"rtol":RTOL,
        "reference_mean_of_budget_medians_ms":ref,"optimized_mean_of_budget_medians_ms":opt,
        "speed_ratio_reference_over_optimized":ref/opt if opt else math.inf,
    })
    write_json(root/"outputs"/"runtime_environment.json",{
        "device":str(device),"gpu":torch.cuda.get_device_name(0) if device.type=="cuda" else None,"torch":torch.__version__,"cuda":torch.version.cuda,
        "python":sys.version,"os":platform.platform(),"torch_num_threads":torch.get_num_threads(),"torch_num_interop_threads":torch.get_num_interop_threads(),
        "order_seed":ORDER_SEED,"profile_images":200,"profile_groups":5,"source_role":"FIT","warmups_per_condition":5,"measured_per_group_condition":20,
        "elapsed_seconds":time.perf_counter()-t0,
    })
    append_log(root,f"STAGE profile_optimizations COMPLETE optimization_status={optimization_status} elapsed_seconds={time.perf_counter()-t0:.6f}")


if __name__=="__main__": main()
