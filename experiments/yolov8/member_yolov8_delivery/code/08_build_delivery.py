# -*- coding: utf-8 -*-
"""08 — assemble the machine-readable YOLO delivery package (guideline §08).

Reads the frozen artefacts produced by 01..07 and writes:

  yolo交付物/member_yolo_delivery/
    README_CN.md
    code/ configs/ assets/ results/ references/ manifest.csv

Nothing here re-trains, re-solves allocations, or alters any scientific value;
every table is rebuilt from the raw per-image / per-group / per-condition files.
"""
from __future__ import annotations
import os

import ast
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
M11 = Path(os.environ.get("M11_EXPERIMENT_ROOT", "external_assets")) / "material from memberA" / "LC_ALLOC_M11_HANDOFF_v1" / "handoff_LC_ALLOC_M11_v1"
if str(M11) not in sys.path:
    sys.path.insert(0, str(M11))
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from lc_alloc.evaluation.bootstrap import paired_group_bootstrap  # noqa: E402
from lc_alloc.labels.prefix import load_normalized_gt  # noqa: E402

ROAD8 = ["person", "bicycle", "car", "motorcycle", "bus", "train", "truck", "traffic light"]
ROAD8_CN = ["人", "自行车", "轿车", "摩托车", "公交车", "火车", "卡车", "交通灯"]
BUDGETS = (10, 15, 20, 30, 40)
SEEDS = (830101, 830102, 830103)

DATASET_ID = "bdd100k_road8_dev2k"
DETECTOR_ID = "yolov8n_ultralytics_8.4.77_f59b3d83"
CANDIDATE_ASSET_ID = "yolov8n_bdd100k_road8_top100_v1"
PROTOCOL_PFX = "PFX_EXACT"
PROTOCOL_NMS = "PFX_THEN_NMS"
RESAMPLES, BOOT_SEED = 5000, 530002

DELIV = BASE / "yolo交付物" / "member_yolo_delivery"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_list(s):
    return np.asarray(ast.literal_eval(s), dtype=np.float64)


def build_results(res_dir: Path):
    main = pd.read_csv(BASE / "work/eval/main_results.csv")
    ap = pd.read_csv(BASE / "work/eval/ap_ar_prefix.csv")
    apc = pd.read_csv(BASE / "work/eval/ap_ar_per_class.csv")
    per_img = pd.read_csv(BASE / "work/eval/per_image_results.csv")
    boot = pd.read_csv(BASE / "work/eval/paired_group_bootstrap.csv")
    gt_map = load_normalized_gt(BASE / "work/prep/DEV_gt.json")

    key = ["method", "seed", "budget"]

    # ---------- main_results.csv (prefix protocol) ----------
    m = main.merge(ap[ap.action_space == "frozen_prefix_no_nms"][
        key + ["AP", "AP50", "AP75", "AR100", "detections"]], on=key, how="left", validate="one_to_one")
    main_out = pd.DataFrame({
        "dataset_id": DATASET_ID,
        "detector_id": DETECTOR_ID,
        "candidate_asset_id": CANDIDATE_ASSET_ID,
        "protocol_id": PROTOCOL_PFX,
        "method": m["method"], "seed": m["seed"].astype(int), "budget": m["budget"].astype(int),
        "image_count": m["image_count"].astype(int),
        "total_records": m["output_records"].astype(int),
        "Coverage_per_image": m["coverage_per_image"],
        "QUALITY_per_image": m["quality_per_image"],
        "coverage_total": m["coverage_total"].astype(int),
        "quality_total": m["quality_total"],
        "coverage_recall": m["coverage_recall"],
        "valid_gt_total": m["valid_gt_total"].astype(int),
        "AP": m["AP"], "AP50": m["AP50"], "AP75": m["AP75"], "AR100": m["AR100"],
    })
    main_out = main_out.sort_values(["method", "seed", "budget"]).reset_index(drop=True)
    # inverse-budget sanity: total_records must equal group budget x images
    assert (main_out["total_records"] == main_out["image_count"] * main_out["budget"]).all(), "budget closure failed"
    main_out.to_csv(res_dir / "main_results.csv", index=False)

    # ---------- class_results.csv ----------
    gt_imgs_per_class = np.zeros(8, dtype=np.int64)
    gt_support = np.zeros(8, dtype=np.int64)
    for iid, g in gt_map.items():
        bc = np.bincount(g.classes, minlength=9)[1:9]
        gt_support += bc
        gt_imgs_per_class += (bc > 0).astype(np.int64)

    rows = []
    for r in m.itertuples(index=False):
        cc = parse_list(r.class_coverage)
        cg = parse_list(r.class_gt)
        for i in range(8):
            rows.append({
                "dataset_id": DATASET_ID, "detector_id": DETECTOR_ID,
                "candidate_asset_id": CANDIDATE_ASSET_ID, "protocol_id": PROTOCOL_PFX,
                "method": r.method, "seed": int(r.seed), "budget": int(r.budget),
                "class_id": i + 1, "class_name": ROAD8[i], "class_name_cn": ROAD8_CN[i],
                "gt_support": int(cg[i]), "gt_support_all_dev": int(gt_support[i]),
                "images_with_class": int(gt_imgs_per_class[i]),
                "matched_count": int(cc[i]),
                "coverage_recall": float(cc[i] / cg[i]) if cg[i] else np.nan,
                "class_gt_share_of_condition": float(cg[i] / cg.sum()) if cg.sum() else np.nan,
            })
    cls = pd.DataFrame(rows)
    # attach class AP/AR (IoU-0.50 coverage is the project metric; AP is the COCO one)
    apc_s = apc[apc.action_space == "frozen_prefix_no_nms"][
        key + ["category_id", "AP", "AP50", "AP75", "AR100"]].rename(
        columns={"category_id": "class_id", "AP": "class_AP", "AP50": "class_AP50",
                 "AP75": "class_AP75", "AR100": "class_AR100"})
    cls = cls.merge(apc_s, on=key + ["class_id"], how="left", validate="one_to_one")
    cls = cls.sort_values(["method", "seed", "budget", "class_id"]).reset_index(drop=True)
    cls.to_csv(res_dir / "class_results.csv", index=False)

    # ---------- per_group_results.csv ----------
    grp = (per_img.groupby(key + ["group_id"])
           .agg(images=("image_id", "size"), total_records=("K_i", "sum"),
                coverage_total=("coverage", "sum"), quality_total=("quality", "sum"))
           .reset_index())
    grp.insert(0, "protocol_id", PROTOCOL_PFX)
    grp.insert(0, "candidate_asset_id", CANDIDATE_ASSET_ID)
    grp.insert(0, "detector_id", DETECTOR_ID)
    grp.insert(0, "dataset_id", DATASET_ID)
    grp = grp.sort_values(["method", "seed", "budget", "group_id"]).reset_index(drop=True)
    grp.to_csv(res_dir / "per_group_results.csv", index=False)

    # ---------- bootstrap_results.csv ----------
    rows = []
    for r in boot.itertuples(index=False):
        for metric, pe, lo, hi in (("Coverage", r.coverage_diff, r.coverage_ci95_low, r.coverage_ci95_high),
                                   ("QUALITY", r.quality_diff, r.quality_ci95_low, r.quality_ci95_high)):
            rows.append({"comparison": "M11 - S_ADAPT", "budget": int(r.budget),
                         "seed_summary": f"seed_{int(r.seed)}", "metric": metric,
                         "point_estimate": pe, "ci95_low": lo, "ci95_high": hi,
                         "resamples": RESAMPLES, "bootstrap_seed": BOOT_SEED,
                         "resample_unit": "group_40_images", "paired": True,
                         "ci_note": "per-seed interval; endpoints are NOT averaged across seeds"})
    # three-seed summary: average the three per-group differences first, then bootstrap
    m11 = per_img[per_img.method == "M11"]
    sad = per_img[per_img.method == "S_ADAPT"]
    for budget in BUDGETS:
        for metric in ("coverage", "quality"):
            diffs = []
            for gid in sorted(m11.group_id.unique()):
                vals = []
                for seed in SEEDS:
                    a = m11[(m11.seed == seed) & (m11.budget == budget) & (m11.group_id == gid)][metric]
                    b = sad[(sad.budget == budget) & (sad.group_id == gid)][metric]
                    vals.append(a.mean() - b.mean())
                diffs.append(float(np.mean(vals)))
            d = paired_group_bootstrap(np.asarray(diffs, dtype=np.float64), resamples=RESAMPLES, seed=BOOT_SEED)
            rows.append({"comparison": "M11 - S_ADAPT", "budget": int(budget),
                         "seed_summary": "mean_of_3_seeds", "metric": "Coverage" if metric == "coverage" else "QUALITY",
                         "point_estimate": d["point_estimate"], "ci95_low": d["ci95_low"], "ci95_high": d["ci95_high"],
                         "resamples": RESAMPLES, "bootstrap_seed": BOOT_SEED,
                         "resample_unit": "group_40_images", "paired": True,
                         "ci_note": "3-seed summary: per-group 3-seed mean difference, then bootstrap"})
    bs = pd.DataFrame(rows).sort_values(["metric", "budget", "seed_summary"]).reset_index(drop=True)
    bs.to_csv(res_dir / "bootstrap_results.csv", index=False)

    # ---------- bootstrap_core_summary.csv (guideline §06: 核心汇总 = K10/15/20 等权平均) ----------
    # For each group, the M11-S_ADAPT difference is averaged over the three core
    # budgets (equal weight); the 3-seed summary averages the three per-seed diffs
    # BEFORE the budget pooling.  Then bootstrap over the 50 groups.
    CORE_BUDGETS = (10, 15, 20)
    core_rows = []
    for metric in ("coverage", "quality"):
        for seed in SEEDS:
            diffs = []
            for gid in sorted(m11.group_id.unique()):
                acc = [float(m11[(m11.seed == seed) & (m11.budget == b) & (m11.group_id == gid)][metric].mean()
                            - sad[(sad.budget == b) & (sad.group_id == gid)][metric].mean())
                       for b in CORE_BUDGETS]
                diffs.append(float(np.mean(acc)))
            d = paired_group_bootstrap(np.asarray(diffs, dtype=np.float64), resamples=RESAMPLES, seed=BOOT_SEED)
            core_rows.append({"comparison": "M11 - S_ADAPT", "budget_pool": "10_15_20_equal_weight",
                              "seed_summary": f"seed_{int(seed)}", "metric": "Coverage" if metric == "coverage" else "QUALITY",
                              "point_estimate": d["point_estimate"], "ci95_low": d["ci95_low"], "ci95_high": d["ci95_high"],
                              "resamples": RESAMPLES, "bootstrap_seed": BOOT_SEED,
                              "resample_unit": "group_40_images", "paired": True,
                              "ci_note": "core summary: equal-weight mean over budgets 10/15/20, per-seed interval"})
        # 3-seed summary: seed-averaged diffs, then budget pooling, then bootstrap
        diffs = []
        for gid in sorted(m11.group_id.unique()):
            acc = []
            for b in CORE_BUDGETS:
                seed_diffs = [float(m11[(m11.seed == seed) & (m11.budget == b) & (m11.group_id == gid)][metric].mean()
                                   - sad[(sad.budget == b) & (sad.group_id == gid)][metric].mean())
                              for seed in SEEDS]
                acc.append(float(np.mean(seed_diffs)))
            diffs.append(float(np.mean(acc)))
        d = paired_group_bootstrap(np.asarray(diffs, dtype=np.float64), resamples=RESAMPLES, seed=BOOT_SEED)
        core_rows.append({"comparison": "M11 - S_ADAPT", "budget_pool": "10_15_20_equal_weight",
                          "seed_summary": "mean_of_3_seeds", "metric": "Coverage" if metric == "coverage" else "QUALITY",
                          "point_estimate": d["point_estimate"], "ci95_low": d["ci95_low"], "ci95_high": d["ci95_high"],
                          "resamples": RESAMPLES, "bootstrap_seed": BOOT_SEED,
                          "resample_unit": "group_40_images", "paired": True,
                          "ci_note": "core summary: equal-weight mean over budgets 10/15/20, 3-seed summary"})
    core = pd.DataFrame(core_rows).sort_values(["metric", "seed_summary"]).reset_index(drop=True)
    core.to_csv(res_dir / "bootstrap_core_summary.csv", index=False)

    # ---------- allocations.parquet (with selected_record_ids) ----------
    alloc = pd.read_parquet(BASE / "work/eval/allocations.parquet")
    alloc["candidate_asset_id"] = CANDIDATE_ASSET_ID
    cand = pd.read_parquet(BASE / "work/DEV/candidates.parquet")
    cand["image_id"] = cand["image_id"].astype(str)
    cand["rank"] = cand["rank"].astype(int)
    idmap = {(iid, rk): rid for iid, rk, rid in
             zip(cand.image_id, cand["rank"], cand.candidate_record_id)}
    sel = alloc[["method", "seed", "budget", "image_id", "K_i"]].copy()
    recs = []
    for r in sel.itertuples(index=False):
        for rk in range(1, int(r.K_i) + 1):
            recs.append({"method": r.method, "seed": int(r.seed), "budget": int(r.budget),
                         "image_id": r.image_id, "rank": rk,
                         "candidate_record_id": idmap[(r.image_id, rk)]})
    selrec = pd.DataFrame(recs)
    assert len(selrec) == int(alloc.K_i.sum()), "selected record count != exact budget total"
    selrec.to_parquet(res_dir / "selected_records.parquet", index=False)
    alloc.sort_values(["method", "seed", "budget", "image_id"]).to_parquet(res_dir / "allocations.parquet", index=False)

    return dict(main=len(main_out), cls=len(cls), grp=len(grp), boot=len(bs),
                core=len(core), alloc=len(alloc), selected=len(selrec))


def main():
    if DELIV.exists():
        shutil.rmtree(DELIV)
    for sub in ("code", "configs", "assets/models", "results", "references"):
        (DELIV / sub).mkdir(parents=True, exist_ok=True)
    res = DELIV / "results"

    counts = build_results(res)
    print("built:", counts, flush=True)

    # ---------- copy code ----------
    for f in ["01_export.py", "02_prepare.py", "03_train.py", "04_allocate_eval.py",
              "05_ap_ar.py", "06_post_nms_counts.py", "07_predicted_utility.py",
              "08_build_delivery.py", "09_manifest.py", "audit_candidates.py"]:
        shutil.copy2(BASE / f, DELIV / "code" / f)
    (DELIV / "code" / "m11_yolo").mkdir(exist_ok=True)
    for f in ["__init__.py", "adapter.py", "config.py", "features.py", "nms.py"]:
        shutil.copy2(BASE / "m11_yolo" / f, DELIV / "code" / "m11_yolo" / f)

    # ---------- copy configs ----------
    shutil.copy2(BASE / "configs" / "yolo_adapter.json", DELIV / "configs" / "yolo_adapter.json")
    shutil.copy2(BASE / "work" / "prep" / "roles.csv", DELIV / "configs" / "roles.csv")
    shutil.copy2(BASE / "work" / "prep" / "dev_groups.csv", DELIV / "configs" / "groups.csv")
    shutil.copy2(BASE / "work" / "prep" / "roles.csv", DELIV / "configs" / "roles_raw.csv")
    shutil.copy2(BASE / "work" / "assets" / "class_weights.json", DELIV / "configs" / "evaluation_weights.json")
    shutil.copy2(BASE / "work" / "assets" / "class_weights.json", DELIV / "configs" / "allocation_weights.json")

    evaluator = {
        "protocol_id_prefix_exact": PROTOCOL_PFX,
        "protocol_id_prefix_then_nms": PROTOCOL_NMS,
        "project_metrics": {
            "Coverage_i(K)": "m_i^{0.50}(K)  -- max-cardinality same-class matching at IoU=0.50 ONLY (single threshold)",
            "QUALITY_i(K)": "mean_tau sum_c w_c * m_{i,c}^{tau}(K)  -- 10 thresholds 0.50:0.05:0.95, class-weighted",
            "delta_{i,k}^{tau}": "m_i^{tau}(k) - m_i^{tau}(k-1), k = 6..50  (label for training)",
            "note": "m = max-cardinality same-class matching, no candidate/GT reuse. Coverage uses IoU=0.50 only; QUALITY averages 10 thresholds. Empty-GT images are KEPT in the evaluation set.",
        },
        "thresholds": [round(0.50 + 0.05 * i, 2) for i in range(10)],
        "k_min": 5, "k_max": 50, "group_size": 40, "budgets": list(BUDGETS),
        "weight_ids": {"allocation_weight_id": "yolo_class_weights_fit_v1",
                       "evaluation_weight_id": "yolo_class_weights_fit_v1"},
        "weight_note": "allocation and evaluation weights are the SAME object here; they are stored under two names because the guideline requires the ids be separable",
        "coco_eval": {"iouType": "bbox", "useCats": 1, "iouThrs": "0.50:0.05:0.95 (10)",
                      "recThrs": "101 points", "maxDets": [1, 10, 100],
                      "areaRng": "all (default)", "library": "pycocotools COCOeval"},
        "coco_eval_note": "AP/AR are computed per seed on the FULL 2000-image DEV set; per-40-image-group AP is never averaged. No effective-GT category is silently treated as 0.",
        "bootstrap": {"unit": "group of 40 images", "resamples": RESAMPLES, "seed": BOOT_SEED,
                      "paired": True, "shared_draw_index": True,
                      "ci_note": "per-seed intervals are reported separately; endpoints are never averaged across seeds. Coverage/QUALITY intervals do not establish significance for AP/AR."},
        "units": "AP/AR/recall are stored as 0-1 in the raw CSVs; multiply by 100 for percent.",
    }
    (DELIV / "configs" / "evaluator_config.json").write_text(
        json.dumps(evaluator, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---------- copy assets ----------
    for f in ["pca32.joblib", "scaler.joblib", "asset.json", "class_weights.json",
              "training_history_seed_830101.csv", "training_history_seed_830102.csv",
              "training_history_seed_830103.csv"]:
        shutil.copy2(BASE / "work" / "assets" / f, DELIV / "assets" / f)
    for f in sorted((BASE / "work" / "assets" / "models").iterdir()):
        shutil.copy2(f, DELIV / "assets" / "models" / f.name)

    # ---------- copy remaining results ----------
    for f in ["ap_ar_prefix.csv", "ap_ar_reference_nms.csv", "ap_ar_per_class.csv",
              "post_nms_counts.csv", "post_nms_group_summary.csv", "post_nms_overall.csv",
              "predicted_utility.parquet", "predicted_utility_manifest.json",
              "paired_group_bootstrap.csv", "per_image_results.csv"]:
        shutil.copy2(BASE / "work" / "eval" / f, res / f)

    # ---------- references (candidate/native + GT delivered IN-PACKAGE) ----------
    (DELIV / "references" / "data").mkdir(parents=True, exist_ok=True)
    ref_rows = []
    for split in ("DEV", "TRAIN"):
        for name in ("candidates.parquet", "native.npz", "export_log.json"):
            src = BASE / "work" / split / name
            rel = Path("data") / split / name
            dst = DELIV / "references" / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            ref_rows.append({"split": split, "file": name, "rel_path": rel.as_posix(),
                             "size_bytes": src.stat().st_size, "sha256": sha256(src),
                             "source_path": str(src)})
    for name in ("DEV_gt.json", "TRAIN_gt.json"):
        src = BASE / "work" / "prep" / name
        rel = Path("data") / "prep" / name
        dst = DELIV / "references" / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        ref_rows.append({"split": "prep", "file": name, "rel_path": rel.as_posix(),
                         "size_bytes": src.stat().st_size, "sha256": sha256(src),
                         "source_path": str(src)})
    pd.DataFrame(ref_rows).to_csv(DELIV / "references" / "candidate_native_index.csv", index=False)
    (DELIV / "references" / "README_CN.md").write_text(
        "# references\n\n"
        "候选/native 与 GT 全量文件**已随包交付**在 `data/`（8 个文件，合计约 634 MB）。\n\n"
        "- `candidate_native_index.csv`：每个文件的**包内相对路径、大小、SHA256**，并保留 "
        "`source_path`（本机 D 盘原始出处）供溯源。\n"
        "- 复核时按索引的 `rel_path` 找到包内文件，校验 SHA 即可，无需依赖本机路径。\n\n"
        "重新生成方式：候选/native 由 `code/01_export.py` 从 BDD100K 原始图片重新生成"
        "（参数见 `configs/yolo_adapter.json`）；GT 由 `code/02_prepare.py` 从 RT-DETR release 的 "
        "`DEV2K_ROAD8_GT.json` / `TRAIN10K_ROAD8_GT.json` 转换而来。\n\n"
        "按规范 §08 不打包的仅限：原始 BDD100K 图片数据集、虚拟环境、令牌、未经授权的 "
        "TEST/RESERVE 资产（候选/native 是派生产物，不在此列）。\n",
        encoding="utf-8")

    # ---------- delivery README (guideline §08: 环境/入口/边界/复现顺序) ----------
    shutil.copy2(BASE / "delivery_README_CN.md", DELIV / "README_CN.md")

    print("assembly done", flush=True)

    # ---------- manifest (last: hashes every file placed above) ----------
    import runpy
    runpy.run_path(str(BASE / "09_manifest.py"), run_name="__main__")


if __name__ == "__main__":
    main()
