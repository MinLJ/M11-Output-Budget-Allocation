# -*- coding: utf-8 -*-
u"""把 Faster R-CNN 的导出分片转换成 M11 流水线能直接吃的输入。

产出 (默认 to `frcnn_m11/prepared/`):
  candidates_{split}.parquet   M11 candidate 表 (列名走 CANDIDATE_COLUMNS 直名)
  candidates_index_{split}.json  候选 parquet 的校验索引 (bytes+sha256, 合同 §08)
  native_{split}.npz           M11 native sidecar: image_ids/source_ids/native_vectors/
                               class_signals/native_schema_id
  gt_{split}_normalized.json   冻结 Road8 GT -> 包内 normalized 格式
  roles.csv                    TRAIN10K 的 FIT/EARLY_STOP/CALIBRATION 角色 (原文件副本)
  groups_dev.csv               DEV2K 的 50 个冻结 40 图组 (原文件副本)

转换本身**不重算任何科学量**: 候选/原生/GT 都是冻结文件之间的格式搬运。
唯一的改写是把冻结 Road8 GT 的 COCO xywh 换成包内 normalized 的 box_xyxy ——
数值不变, 不做有效性过滤(冻结 GT 已经保证 0<=x1<x2<=W)。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
EXPERIMENT_ROOT = Path(os.environ.get('M11_EXPERIMENT_ROOT', "external_assets"))
M11_ROOT = EXPERIMENT_ROOT / 'frcnn_m11'
HANDOFF = EXPERIMENT_ROOT / 'LC_ALLOC_M11_HANDOFF_v1/handoff_LC_ALLOC_M11_v1'
EXPORT_ROOT = EXPERIMENT_ROOT / 'frcnn_road8/outputs/frcnn_r18fpn_road8'

sys.path.insert(0, str(M11_ROOT))
from lc_frcnn.features_frcnn import NATIVE_SCHEMA_ID  # noqa: E402

CANDIDATE_COLS = [
    "image_id", "candidate_record_id", "rank", "predicted_road8_class_id", "predicted_road8_class_name",
    "original_score", "box_x1", "box_y1", "box_x2", "box_y2", "source_id", "image_width", "image_height",
]
IDENTITY_COLS = [
    "dataset_version", "split", "protocol_id", "image_sha256", "detector_id", "checkpoint_sha256",
    "export_config_sha256", "candidate_asset_id", "box_cx", "box_cy", "box_w", "box_h", "source_order",
]


def log(msg: str) -> None:
    print(f"[prepare] {msg}", flush=True)


def convert_candidates(export_dir: Path, out_path: Path, *, which: str, split_key: str) -> dict:
    u"""`which` = primary (M11 主用流) 或 post_nms (登记备查流)。

    文件名里的 split 记号由调用方显式给出: 导出目录叫 `export_dev`, 但导出文件叫
    `dev2k_candidates.parquet` —— 从目录名反推会找错文件。
    """
    prefix = split_key.lower()
    name = f"{prefix}_candidates.parquet" if which == "primary" else f"{prefix}_candidates_post_nms.parquet"
    src = export_dir / "candidates" / name
    if not src.is_file():
        raise FileNotFoundError(f"{src} missing; run export_frcnn.py first (it writes both streams)")
    frame = pd.read_parquet(src)
    keep = [c for c in CANDIDATE_COLS + IDENTITY_COLS if c in frame.columns]
    frame = frame[keep]
    frame = frame.sort_values(["image_id", "rank"], kind="mergesort").reset_index(drop=True)
    frame.to_parquet(out_path, index=False, compression="zstd")
    counts = frame.groupby("image_id", sort=False).size()
    protocols = sorted(set(frame["protocol_id"].astype(str))) if "protocol_id" in frame else []
    log(f"{which} -> {out_path.name}  protocol={protocols} images={len(counts)} rows={len(frame)} "
        f"per-image min/med/max={counts.min()}/{int(counts.median())}/{counts.max()}")
    return {"images": int(len(counts)), "rows": int(len(frame)), "protocol_id": protocols,
            "below_100": int((counts < 100).sum()), "below_50": int((counts < 50).sum())}


def index_candidates(out_dir: Path, stem: str) -> dict:
    u"""给候选 parquet 建校验清单。合同 §08 要求大候选文件即使共享受控路径也要有校验清单。

    两个流都登记: `candidates_{split}.parquet` (M11 主用流) 与
    `candidates_{split}_post_nms.parquet` (登记备查流)。bytes/sha256 按文件实算,
    供 references/README_CN.md 与读取端直接重算比对。
    """
    import hashlib

    rows = []
    for which, name in (
            ("primary", f"candidates_{stem}.parquet"),
            ("post_nms_reference", f"candidates_{stem}_post_nms.parquet")):
        path = out_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"{path} missing; run convert_candidates first")
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        rows.append({"stream": which, "path": str(path.resolve()), "file": name,
                     "bytes": int(path.stat().st_size), "sha256": digest.hexdigest()})
    payload = {
        "file_count": len(rows),
        "convention": (
            "candidate parquet tables delivered via the controlled path; each entry carries the "
            "bytes and sha256 of the file as delivered, so a reader can re-hash and compare"),
        # 这个键在候选表里**按设计非唯一**: 同一个 RoI 的多个类别记录共享一个 source_id
        # (合同 §04)。旧措辞直接抄了 native 索引的 "unique", 会让下游误以为可以按 source_id
        # 去重 —— 那正好违反 §04 的「不得合并」。所以这里改成面向 join 目标的说法。
        "join_key": (
            "(image_id, source_id) 用于把候选记录连到 native 行; 该键在 **native sidecar 内**唯一 "
            "(native 转换阶段对重复即报错), 在**候选表内按设计非唯一** —— 同一个 RoI 的不同类别"
            "记录共享同一 source_id, 不得去重、不得合并"),
        "files": rows,
    }
    out_path = out_dir / f"candidates_index_{stem}.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    total = int(sum(r["bytes"] for r in rows))
    log(f"candidate index -> {out_path.name}  files={len(rows)} bytes={total}")
    return {"index_file": out_path.name, "file_count": len(rows), "total_bytes": total}


def index_native(export_dir: Path, out_path: Path) -> dict:
    u"""给 native 分片建索引。合同 §08 要求大文件即使分卷也要有校验清单。"""
    import hashlib

    shards = sorted((export_dir / "native_state").glob("state_*.npz"))
    if not shards:
        raise FileNotFoundError(f"no native shards under {export_dir/'native_state'}")
    rows = []
    total = 0
    for shard in shards:
        digest = hashlib.sha256()
        with open(shard, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        with np.load(shard, allow_pickle=False) as raw:
            n = int(len(np.asarray(raw["source_id"])))
        rows.append({"path": str(shard.relative_to(export_dir.parent.parent)).replace("\\", "/"),
                     "file": shard.name, "bytes": shard.stat().st_size,
                     "sha256": digest.hexdigest(), "native_rows": n})
        total += n
    payload = {
        "shard_count": len(rows), "total_native_rows": total,
        "convention": (
            "one .npz per shard, holding MANY images; image_id / source_id / roi_index / native_vector / "
            "class_signal / proposal_xyxy_resized / native_schema_id are all stored PER ROW, so one row "
            "is not one image. Shard size is an export parameter, not part of the schema: readers must "
            "not assume a fixed number of images per shard."
        ),
        "row_key": "source_id = 'roi{index:05d}'; several class records of one RoI share it and must not be de-duplicated",
        "join_key": "(image_id, source_id), unique across all shards",
        "shards": rows,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"native index -> {out_path.name}  shards={len(rows)} rows={total}")
    return {"shard_count": len(rows), "total_native_rows": total}


def convert_native(export_dir: Path, out_path: Path) -> dict:
    shards = sorted((export_dir / "native_state").glob("state_*.npz"))
    if not shards:
        raise FileNotFoundError(f"no native shards under {export_dir/'native_state'}")
    image_ids, source_ids, vectors, signals, schemas = [], [], [], [], []
    for shard in shards:
        with np.load(shard, allow_pickle=False) as raw:
            img = np.asarray(raw["image_id"]).astype(str)
            src = np.asarray(raw["source_id"]).astype(str)
            vec = np.asarray(raw["native_vector"])
            sig = np.asarray(raw["class_signal"])
            sch = str(np.asarray(raw["native_schema_id"]).reshape(-1)[0])
        n = len(src)
        # image_id 必须是**逐行**长度: 分片里有几张图是导出时的分卷决定, 不是 schema 的一部分。
        # 早先这里断言 len(img) == 1 (每片一张图), 那是把分卷策略写进了读取端, 分卷一变就崩。
        if not (len(img) == n and vec.shape[0] == n and sig.shape[0] == n):
            raise ValueError(
                f"{shard.name}: shape mismatch img={img.shape} src={src.shape} "
                f"vec={vec.shape} sig={sig.shape}; image_id must be per-row")
        image_ids.append(img)
        source_ids.append(src)
        vectors.append(vec)
        signals.append(sig)
        schemas.append(sch)
    schema_set = set(schemas)
    if schema_set != {NATIVE_SCHEMA_ID}:
        raise ValueError(f"native_schema_id mismatch: {schema_set} != {{{NATIVE_SCHEMA_ID!r}}}")
    image_ids = np.concatenate(image_ids)
    source_ids = np.concatenate(source_ids)
    vectors = np.concatenate(vectors)
    signals = np.concatenate(signals)
    keys = pd.DataFrame({"image_id": image_ids, "source_id": source_ids})
    if keys.duplicated().any():
        raise ValueError("(image_id, source_id) is not unique across native shards")
    np.savez_compressed(
        out_path,
        image_ids=image_ids,
        source_ids=source_ids,
        native_vectors=vectors,
        class_signals=signals,
        native_schema_id=np.array(NATIVE_SCHEMA_ID),
    )
    log(f"native -> {out_path.name}  rows={len(image_ids)}  images={len(set(image_ids.tolist()))}  "
        f"vector={vectors.shape} signal={signals.shape}")
    return {"native_rows": int(len(image_ids)), "vector_dim": int(vectors.shape[1]),
            "signal_dim": int(signals.shape[1])}


def convert_gt(gt_json: Path, out_path: Path) -> dict:
    u"""冻结 Road8 COCO GT -> 包内 normalized 格式。xywh -> xyxy, 数值不变。"""
    raw = json.loads(gt_json.read_text(encoding="utf-8"))
    images = [{"image_id": str(im["id"]), "file_name": im["file_name"],
               "width": int(im["width"]), "height": int(im["height"])} for im in raw["images"]]
    anns = []
    for a in raw["annotations"]:
        if int(a.get("iscrowd", 0)) or int(a.get("ignore", 0)):
            continue
        x, y, w, h = [float(v) for v in a["bbox"]]
        anns.append({
            "image_id": str(a["image_id"]),
            "road8_class_id": int(a["category_id"]),
            "box_xyxy": [x, y, x + w, y + h],
            "area": float(a.get("area", w * h)),
            "source_object_index": int(a.get("source_object_index", -1)),
        })
    payload = {
        "info": {"format": "M11-package-normalized Road8 GT",
                 "source": str(gt_json),
                 "note": "converted from the frozen release Road8 COCO GT; xywh -> xyxy, no numeric change"},
        "images": images,
        "categories": raw["categories"],
        "annotations": anns,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    log(f"gt -> {out_path.name}  images={len(images)} anns={len(anns)}")
    return {"images": len(images), "annotations": len(anns)}


def copy_identity_files(roles_csv: Path, groups_parquet: Path, out_dir: Path) -> dict:
    roles = pd.read_csv(roles_csv)
    roles.to_csv(out_dir / "roles.csv", index=False)
    groups = pd.read_parquet(groups_parquet)
    groups = groups[["image_id", "group_id", "position_in_group"]].copy()
    groups.to_csv(out_dir / "groups_dev.csv", index=False)
    log(f"roles -> roles.csv  ({roles.role.value_counts().to_dict()})")
    log(f"groups -> groups_dev.csv  ({groups.group_id.nunique()} groups of "
        f"{sorted(groups.groupby('group_id').size().unique().tolist())})")
    return {"roles": roles.role.value_counts().to_dict(), "groups": int(groups.group_id.nunique())}


# 导出目录叫 export_dev, 但导出脚本写出的文件名用的是 GT 的 split 记号 (dev2k / train10k)。
# 两套记号各管一头: 目录/本地文件名用短记号, 导出文件用长记号 —— 显式写出来, 不靠字符串推断。
EXPORT_PREFIX = {"DEV": "dev2k", "TRAIN": "train10k"}


def convert_split(split_key: str, gt_name: str, out_dir: Path) -> dict:
    export_dir = EXPORT_ROOT / f"export_{split_key.lower()}"
    if not export_dir.is_dir():
        raise FileNotFoundError(export_dir)
    stem = split_key.lower()
    return {
        "split": split_key,
        "export_dir": str(export_dir),
        "candidates": convert_candidates(
            export_dir, out_dir / f"candidates_{stem}.parquet", which="primary",
            split_key=EXPORT_PREFIX[split_key]),
        "candidates_post_nms": convert_candidates(
            export_dir, out_dir / f"candidates_{stem}_post_nms.parquet", which="post_nms",
            split_key=EXPORT_PREFIX[split_key]),
        "candidates_index": index_candidates(out_dir, stem),
        "native": convert_native(export_dir, out_dir / f"native_{stem}.npz"),
        "native_index": index_native(export_dir, out_dir / f"native_index_{stem}.json"),
        "gt": convert_gt(EXPERIMENT_ROOT / 'AOP_Road8_LC_v1_RTDERTv2_R18VD_RELEASE_v1/gt' / gt_name,
                         out_dir / f"gt_{stem}_normalized.json"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(M11_ROOT / "prepared"))
    ap.add_argument("--splits", nargs="+", default=["DEV", "TRAIN"])
    ap.add_argument("--index-only", action="store_true",
                    help="只为 out 下已有的候选 parquet 重建 candidates_index_*.json, 不重跑转换")
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.index_only:
        for split_key in args.splits:
            index_candidates(out_dir, split_key.lower())
        log(f"DONE (index only) -> {out_dir}")
        return

    manifest = {"native_schema_id": NATIVE_SCHEMA_ID, "splits": {}}
    if "DEV" in args.splits:
        manifest["splits"]["DEV"] = convert_split(
            "DEV", "DEV2K_ROAD8_GT.json", out_dir)
    if "TRAIN" in args.splits:
        manifest["splits"]["TRAIN"] = convert_split(
            "TRAIN", "TRAIN10K_ROAD8_GT.json", out_dir)
    manifest["identity"] = copy_identity_files(
        HANDOFF / "configs/manifests/lc_train_role_split.csv",
        HANDOFF / "configs/manifests/lc_dev_groups.parquet",
        out_dir,
    )
    (out_dir / "prepare_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"DONE -> {out_dir}")


if __name__ == "__main__":
    main()
