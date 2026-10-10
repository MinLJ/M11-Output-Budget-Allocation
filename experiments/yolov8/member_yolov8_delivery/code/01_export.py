# -*- coding: utf-8 -*-
"""01 — export YOLO candidates + native via YOLOAdapter (M11 contract).

Usage:
  python 01_export.py --split DEV  --max-images 40   --out work/smoke
  python 01_export.py --split DEV  --out work/DEV     # full split
  python 01_export.py --split TRAIN --out work/TRAIN  # full split
"""
from __future__ import annotations
import os

import argparse
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
M11 = Path(os.environ.get("M11_EXPERIMENT_ROOT", "external_assets")) / "material from memberA" / "LC_ALLOC_M11_HANDOFF_v1" / "handoff_LC_ALLOC_M11_v1"
RELEASE = Path(os.environ.get("M11_EXPERIMENT_ROOT", "external_assets")) / "material from memberA" / "AOP_Road8_LC_v1_RTDERTv2_R18VD_RELEASE_v1"
IMG_DIR = Path(os.environ.get("M11_EXPERIMENT_ROOT", "external_assets")) / "bdd100k_images" / "bdd100k" / "images" / "100k" / "train"

for p in (str(M11), str(BASE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from m11_yolo.adapter import YOLOAdapter  # noqa: E402

SPLIT_MANIFEST = {
    "DEV": RELEASE / "manifests" / "DEV2K.parquet",
    "TRAIN": RELEASE / "manifests" / "TRAIN10K.parquet",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="DEV", choices=["DEV", "TRAIN"])
    ap.add_argument("--max-images", type=int, default=0)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    adapter = YOLOAdapter()
    meta = adapter.describe()
    print("adapter metadata:", meta.to_dict(), flush=True)
    log = adapter.export(
        SPLIT_MANIFEST[args.split],
        IMG_DIR,
        args.out,
        split=args.split,
        max_images=args.max_images,
        batch=args.batch,
        device="cuda",
    )
    print("export done:", log, flush=True)


if __name__ == "__main__":
    main()
