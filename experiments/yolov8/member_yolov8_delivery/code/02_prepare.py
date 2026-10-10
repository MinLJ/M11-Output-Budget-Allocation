# -*- coding: utf-8 -*-
"""02 — convert BDD Road8 GT to M11 normalized GT + build role/group manifests.

Reuses the frozen LC Road8 GT JSON from the RT-DETR release (BDD100K, TRAIN10K
+ DEV2K).  Role and group assignment is deterministic and hash-decorrelated
(not sequential), mirroring the package's split_hash / group_images pattern.

Optional ``--max-train`` / ``--max-dev`` cap the manifests to their first N
rows (the same manifest order used by 01_export --max-images), so a smoke run
stays self-consistent: GT, roles, and groups all cover exactly the exported
image subset.

Produces (under --out):
  DEV_gt.json, TRAIN_gt.json      M11 normalized GT (image_id, road8_class_id, box_xyxy)
  roles.csv                       TRAIN -> FIT(8000)/EARLY_STOP(1000)/CALIBRATION(1000)
  dev_groups.csv                  DEV 2000 -> 50 groups x 40
"""
from __future__ import annotations
import os

import argparse
import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
M11 = Path(os.environ.get("M11_EXPERIMENT_ROOT", "external_assets")) / "material from memberA" / "LC_ALLOC_M11_HANDOFF_v1" / "handoff_LC_ALLOC_M11_v1"
RELEASE = Path(os.environ.get("M11_EXPERIMENT_ROOT", "external_assets")) / "material from memberA" / "AOP_Road8_LC_v1_RTDERTv2_R18VD_RELEASE_v1"

if str(M11) not in sys.path:
    sys.path.insert(0, str(M11))

from lc_alloc.data.identity import split_hash  # noqa: E402

GT_SRC = {
    "DEV": RELEASE / "gt" / "DEV2K_ROAD8_GT.json",
    "TRAIN": RELEASE / "gt" / "TRAIN10K_ROAD8_GT.json",
}
MANIFEST = {
    "DEV": RELEASE / "manifests" / "DEV2K.parquet",
    "TRAIN": RELEASE / "manifests" / "TRAIN10K.parquet",
}
GROUP_SIZE = 40
FIT, EARLY_STOP, CALIBRATION = 8000, 1000, 1000


def _manifest_image_ids(split: str, cap: int) -> list[str]:
    import pandas as pd

    ids = pd.read_parquet(MANIFEST[split])["image_id"].astype(str).tolist()
    return ids[:cap] if cap else ids


def convert_gt(split: str, out: Path, cap: int):
    src = json.loads(GT_SRC[split].read_text(encoding="utf-8"))
    image_ids = _manifest_image_ids(split, cap)
    image_set = set(image_ids)
    images = [{"image_id": iid} for iid in image_ids]
    annotations = []
    for a in src["annotations"]:
        if str(a["image_id"]) not in image_set:
            continue
        x, y, w, h = [float(v) for v in a["bbox"]]
        annotations.append({
            "image_id": str(a["image_id"]),
            "road8_class_id": int(a["category_id"]),
            "box_xyxy": [x, y, x + w, y + h],
            "iscrowd": 0,
            "ignore": 0,
        })
    out.write_text(json.dumps({"images": images, "annotations": annotations}, ensure_ascii=False), encoding="utf-8")
    return len(images), len(annotations)


def build_roles(out: Path, cap: int):
    import pandas as pd

    ids = _manifest_image_ids("TRAIN", cap)
    ordered = sorted(ids, key=lambda x: split_hash("YOLO_BDD_ROLE_V1|", x))
    fit = int(len(ids) * 0.80)
    stop = int(len(ids) * 0.90)
    rows = []
    for i, iid in enumerate(ordered):
        role = "FIT" if i < fit else "EARLY_STOP" if i < stop else "CALIBRATION"
        rows.append({"image_id": iid, "role": role})
    df = pd.DataFrame(rows).sort_values("image_id", kind="mergesort")
    df.to_csv(out, index=False)
    return df["role"].value_counts().to_dict()


def build_groups(out: Path, cap: int):
    import pandas as pd

    ids = _manifest_image_ids("DEV", cap)
    ordered = sorted(ids, key=lambda x: split_hash("YOLO_BDD_GROUP_V1|", x))
    rows = []
    for i, iid in enumerate(ordered):
        rows.append({"group_id": i // GROUP_SIZE, "image_id": iid, "position_in_group": i % GROUP_SIZE})
    df = pd.DataFrame(rows).sort_values(["group_id", "position_in_group"], kind="mergesort")
    df.to_csv(out, index=False)
    return len(ids) // GROUP_SIZE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-train", type=int, default=0, help="cap TRAIN images for a smoke run")
    ap.add_argument("--max-dev", type=int, default=0, help="cap DEV images for a smoke run")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    result = {}
    for split, cap in (("DEV", args.max_dev), ("TRAIN", args.max_train)):
        n_img, n_ann = convert_gt(split, out / f"{split}_gt.json", cap)
        result[f"{split}_gt"] = {"images": n_img, "annotations": n_ann}
    result["roles"] = build_roles(out / "roles.csv", args.max_train)
    result["dev_groups"] = build_groups(out / "dev_groups.csv", args.max_dev)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
