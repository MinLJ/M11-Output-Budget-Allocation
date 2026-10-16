#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Recompute seed spread (mean +- population std across the three independent models)
for the M11 manuscript tables, from frozen per-seed result CSVs.

Reviewer comment 13(b) asked whether tables report Mean +- STD.  This script builds
that evidence from already-published per-seed artifacts only; it trains nothing,
allocates nothing and evaluates nothing.  It is read-only with respect to the source
repositories.

Run:  python -B recompute_seed_spread.py            (from any directory)
Out:  seed_spread_report.md, seed_spread.csv         (next to this script)

Source artifacts (per-seed rows):
  1. repository  experiments/baselines_ablation/research_LC_ALLOC_ABLATION_P0/ablation_main_results.csv
  2. repository  experiments/baselines_ablation/research_LC_ALLOC_P1A/ablation_main_results.csv
  3. repository  experiments/coco/research_LC_ALLOC_COCO_ROUTE_B_EVALUATION/main_results.csv
  4. repository  experiments/coco/research_LC_ALLOC_COCO_ROUTE_A_ADAPTATION/working/evaluation_outputs/adapted_main_results.csv
  5. delivery    member_faster_rcnn_delivery/results/main_results.csv
"""
import csv
import os
import statistics as st

ROOT = r"f:/myuniversity/MFE204"
REPO = os.path.join(ROOT, "M11-Output-Budget-Allocation/experiments")
FRCNN = os.path.join(ROOT, "frcnn_m11/delivery_out/member_faster_rcnn_delivery/results")

CORE = {"10", "15", "20"}
DET = "deterministic"  # single-run baseline (no training seed)


def rows(path):
    with open(path, encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def pick(row, *names):
    for n in names:
        if n in row and row[n] not in ("", None):
            return float(row[n])
    raise KeyError(names)


def summarize(records, label, source, scale_ap=100.0):
    """records: list of (seed, row) covering the core budgets."""
    per_seed = {}
    for seed, r in records:
        acc = per_seed.setdefault(seed, {"c": [], "q": [], "ap": [], "ar": []})
        acc["c"].append(pick(r, "coverage_per_image", "Coverage"))
        acc["q"].append(pick(r, "quality_per_image", "QUALITY"))
        acc["ap"].append(pick(r, "AP") * scale_ap)
        acc["ar"].append(pick(r, "AR100") * scale_ap)
    seeds = sorted(per_seed)
    cs = [st.mean(per_seed[s]["c"]) for s in seeds]
    qs = [st.mean(per_seed[s]["q"]) for s in seeds]
    aps = [st.mean(per_seed[s]["ap"]) for s in seeds]
    ars = [st.mean(per_seed[s]["ar"]) for s in seeds]
    return {
        "row": label,
        "source": source,
        "n_models": len(seeds),
        "seeds": ",".join(seeds),
        "coverage": f"{st.mean(cs):.4f}",
        "coverage_sd": f"{st.pstdev(cs):.4f}" if len(seeds) > 1 else "",
        "quality": f"{st.mean(qs):.4f}",
        "quality_sd": f"{st.pstdev(qs):.4f}" if len(seeds) > 1 else "",
        "ap": f"{st.mean(aps):.2f}",
        "ap_sd": f"{st.pstdev(aps):.2f}" if len(seeds) > 1 else "",
        "ar100": f"{st.mean(ars):.2f}",
        "ar100_sd": f"{st.pstdev(ars):.2f}" if len(seeds) > 1 else "",
    }


def core_records(all_rows, method_key, methods, want_seeds=None, solver=None):
    out = []
    for r in all_rows:
        if r.get(method_key) not in methods:
            continue
        if solver is not None and r.get("solver") != solver:
            continue
        if r.get("budget") not in CORE:
            continue
        seed = r.get("seed", "-1")
        if want_seeds and seed not in want_seeds:
            continue
        out.append((seed, r))
    return out


def main():
    report = []

    # --- Table V + Table I(b): BDD DEV, RT-DETRv2-R18VD -----------------------
    abl = rows(os.path.join(REPO, "baselines_ablation/research_LC_ALLOC_ABLATION_P0/ablation_main_results.csv"))
    src_abl = "ABLATION_P0/ablation_main_results.csv"
    for variant, solver, label in [
        ("FULL", "EXACT_DP", "M11 (full)"),
        ("NO_EMBED", "EXACT_DP", "No embedding"),
        ("NO_PREFIX", "EXACT_DP", "No prefix statistics"),
        ("MATCHABILITY_TARGET", "EXACT_DP", "Matchability target"),
        ("FULL", "NEXT_SLOT_GREEDY", "Next-slot greedy"),
        ("SCORE_ONLY", None, "Score (S_ADAPT)"),
    ]:
        report.append(summarize(core_records(abl, "model_variant", {variant}, solver=solver), label, src_abl))

    p1a = rows(os.path.join(REPO, "baselines_ablation/research_LC_ALLOC_P1A/ablation_main_results.csv"))
    for method, label in [
        ("LEARN_QUALITY", "M11 (full)"),
        ("LEARN_MICRO", "M00: IoU 0.50"),
        ("LEARN_CLASS50", "M10: weighted 0.50"),
        ("LEARN_MULTI", "M01: multi-IoU"),
        ("S_ADAPT", "Score (S_ADAPT)"),
    ]:
        report.append(summarize(core_records(p1a, "method", {method}), label,
                                "P1A/ablation_main_results.csv"))

    # --- Table II: COCO2017 Road8 --------------------------------------------
    rb = rows(os.path.join(REPO, "coco/research_LC_ALLOC_COCO_ROUTE_B_EVALUATION/main_results.csv"))
    report.append(summarize(core_records(rb, "method", {"M11"}), "Frozen LC (COCO Route B)",
                            "COCO_ROUTE_B_EVALUATION/main_results.csv"))
    report.append(summarize(core_records(rb, "method", {"S_ADAPT"}), "Score (S_ADAPT)",
                            "COCO_ROUTE_B_EVALUATION/main_results.csv"))
    report.append(summarize(core_records(rb, "method", {"S_FIXED"}), "Fixed-K",
                            "COCO_ROUTE_B_EVALUATION/main_results.csv"))

    ra = rows(os.path.join(REPO, "coco/research_LC_ALLOC_COCO_ROUTE_A_ADAPTATION/working/evaluation_outputs/adapted_main_results.csv"))
    report.append(summarize(core_records(ra, "method", {"M11-COCO"}), "COCO-fitted (Route A)",
                            "COCO_ROUTE_A_ADAPTATION/.../adapted_main_results.csv"))

    # --- Table IV: Faster R-CNN arm (BDD DEV) --------------------------------
    fr = rows(os.path.join(FRCNN, "main_results.csv"))
    for method, label in [("M11", "FRCNN M11"), ("S_ADAPT", "FRCNN Score"), ("S_FIXED", "FRCNN Fixed-K")]:
        try:
            report.append(summarize(core_records(fr, "method", {method}), label,
                                    "member_faster_rcnn_delivery/results/main_results.csv"))
        except KeyError as exc:
            report.append({"row": label, "source": "frcnn main_results.csv",
                           "n_models": 0, "seeds": "", "coverage": f"n/a ({exc})"})

    # --- write ---------------------------------------------------------------
    out_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), "seed_spread.csv")
    cols = ["row", "n_models", "seeds", "coverage", "coverage_sd", "quality", "quality_sd",
            "ap", "ap_sd", "ar100", "ar100_sd", "source"]
    with open(out_csv, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(report)

    lines = ["# Seed spread of the manuscript tables (recomputed)", "",
             "Core budgets are 10/15/20 records per image, averaged equally.  For learned rows the",
             "value is the arithmetic mean over the three independently trained and separately",
             "evaluated models; the +/- figure is the population standard deviation across those",
             "three model evaluations.  Baselines are single deterministic runs, so no spread is",
             "defined.  Coverage and QUALITY are per-image counts; AP/AR100 are percentages.", "",
             "| Row | Models | Coverage | QUALITY | AP (%) | AR100 (%) | Source |",
             "|---|---:|---|---|---|---|---|"]
    for r in report:
        cov = r["coverage"] + (f" ± {r['coverage_sd']}" if r.get("coverage_sd") else "")
        q = r["quality"] + (f" ± {r['quality_sd']}" if r.get("quality_sd") else "")
        ap = r["ap"] + (f" ± {r['ap_sd']}" if r.get("ap_sd") else "")
        ar = r["ar100"] + (f" ± {r['ar100_sd']}" if r.get("ar100_sd") else "")
        lines.append(f"| {r['row']} | {r['n_models']} | {cov} | {q} | {ap} | {ar} | `{r['source']}` |")
    lines += ["", "Computed by `recompute_seed_spread.py` from frozen per-seed artifacts; no training,",
              "allocation or evaluation was re-run."]
    out_md = os.path.join(os.path.dirname(os.path.abspath(__file__)), "seed_spread_report.md")
    with open(out_md, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    print("\n".join(lines[4:]))


if __name__ == "__main__":
    main()
