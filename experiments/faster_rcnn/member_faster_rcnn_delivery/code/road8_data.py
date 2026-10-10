# -*- coding: utf-8 -*-
u"""Road8 检测数据集 —— 给 torchvision Faster R-CNN 用。

数据来源**只**用冻结 release 里的那份 ROAD8 GT, 不自己重解析 BDD100K 原始标注,
这样 Faster R-CNN 这条臂的监督信号和 RT-DETR 那条**逐条一致**。

release: AOP_Road8_LC_v1_RTDERTv2_R18VD_RELEASE_v1
  gt/TRAIN10K_ROAD8_GT.json   (COCO 格式, categories=Road8 的 8 类, bbox=xywh)
  gt/DEV2K_ROAD8_GT.json
图像像素不在 release 内, 按 file_name 在本地 BDD100K 目录里解析。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

# BDD100K 源类别 -> Road8, 与 ROAD8_MAPPING_FROZEN.json 一致 (这里只做自检, 不重新映射)
ROAD8_NAMES = ("person", "bicycle", "car", "motorcycle", "bus", "train", "truck", "traffic light")


class Road8Detection(Dataset):
    u"""COCO-格式 Road8 GT -> (image_tensor, target)。

    target 里的 boxes 是 float32 xyxy **原图坐标**, 不做任何裁剪 ——
    release 的 GT_boundary 声明 bbox 数值不变。torchvision 的
    GeneralizedRCNNTransform 会在内部同步缩放 image 与 boxes。
    """

    def __init__(self, gt_json: str | Path, image_root: str | Path, *, train: bool = False, hflip: float = 0.0):
        self.gt_json = Path(gt_json)
        self.image_root = Path(image_root)
        self.train = bool(train)
        self.hflip = float(hflip)
        with open(self.gt_json, "r", encoding="utf-8") as fh:
            coco = json.load(fh)
        self.images = coco["images"]
        self.categories = coco["categories"]
        names = tuple(c["name"] for c in sorted(self.categories, key=lambda c: c["id"]))
        if names != ROAD8_NAMES:
            raise ValueError(f"GT categories {names} != frozen Road8 order")
        by_image: dict[str, list] = {}
        for ann in coco["annotations"]:
            by_image.setdefault(ann["image_id"], []).append(ann)
        self.by_image = by_image
        self.info = coco.get("info", {})

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        rec = self.images[index]
        image_id = rec["id"]
        path = self.image_root / rec["file_name"]
        image = Image.open(path).convert("RGB")
        if image.size != (rec["width"], rec["height"]):
            raise ValueError(f"{image_id}: pixel size {image.size} != GT {(rec['width'], rec['height'])}")

        anns = self.by_image.get(image_id, [])
        if anns:
            xywh = np.asarray([a["bbox"] for a in anns], dtype=np.float64)
            boxes = np.stack(
                [xywh[:, 0], xywh[:, 1], xywh[:, 0] + xywh[:, 2], xywh[:, 1] + xywh[:, 3]], axis=1
            ).astype(np.float32)
            labels = np.asarray([a["category_id"] for a in anns], dtype=np.int64)
            area = np.asarray([a["area"] for a in anns], dtype=np.float32)
            crowd = np.asarray([a.get("iscrowd", 0) for a in anns], dtype=np.int64)
        else:
            boxes = np.zeros((0, 4), dtype=np.float32)
            labels = np.zeros((0,), dtype=np.int64)
            area = np.zeros((0,), dtype=np.float32)
            crowd = np.zeros((0,), dtype=np.int64)

        # torch.as_tensor 按需拷贝, 避开 PIL 只读缓冲区触发的 non-writable warning
        tensor = (
            torch.as_tensor(np.array(image, dtype=np.uint8, copy=True))
            .permute(2, 0, 1)
            .contiguous()
            .float()
            .div_(255.0)
        )

        # 唯一增强: 水平翻转。翻转后 box 仍在原图坐标系内, 数值语义不变。
        if self.train and self.hflip > 0.0 and torch.rand(()) < self.hflip:
            width = float(rec["width"])
            tensor = torch.flip(tensor, dims=[2])
            if len(boxes):
                x1 = width - boxes[:, 2]
                x2 = width - boxes[:, 0]
                boxes = np.stack([x1, boxes[:, 1], x2, boxes[:, 3]], axis=1).astype(np.float32)

        target = {
            "boxes": torch.from_numpy(boxes),
            "labels": torch.from_numpy(labels),
            "area": torch.from_numpy(area),
            "iscrowd": torch.from_numpy(crowd),
            "image_id": torch.tensor([index], dtype=torch.int64),
            "image_key": image_id,
        }
        return tensor, target


def collate(batch):
    return tuple(zip(*batch))


def build(gt_json, image_root, *, train: bool, hflip: float = 0.0) -> Road8Detection:
    return Road8Detection(gt_json, image_root, train=train, hflip=hflip)
