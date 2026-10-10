# -*- coding: utf-8 -*-
u"""Faster R-CNN (torchvision) 冻结声明式构造。

设计原则 (对应 M11 `docs/ADAPTER_CONTRACT.md` 的「每个新适配器必须冻结」):
  * detector family / version / 实现来源;
  * 骨干与容量 —— 选 ResNet-18 + FPN, 和参照臂 RT-DETRv2-**R18**VD 同深度,
    这样两条臂只差「检测范式」(两阶段 anchor-based vs 一阶段 query-based),
    不叠骨干容量这个混杂因子;
  * 预处理 (resize 策略 / 归一化 / dtype);
  * 后处理顺序 threshold -> NMS -> top-N 与本研究的**输出记录预算**的区别
    (RPN 的 proposal 预算是检测器内部管线, 不是输出记录预算)。
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torchvision
from torchvision.models import ResNet18_Weights, ResNet50_Weights
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# ── 冻结常量 ─────────────────────────────────────────────────────────────
ROAD8_NUM_CLASSES = 8
NUM_CLASSES = ROAD8_NUM_CLASSES + 1          # +1 = background (index 0)
INPUT_SIZE = 640                             # min_size == max_size -> 长边缩到 640, 保长宽比
IMAGE_MEAN = (0.485, 0.456, 0.406)           # GeneralizedRCNNTransform 默认 (ImageNet)
IMAGE_STD = (0.229, 0.224, 0.225)

# RPN 内部管线 —— **不是**本研究的输出记录预算, 只作为 detector 的一部分冻结
RPN_PRE_NMS_TRAIN = 2000
RPN_POST_NMS_TRAIN = 2000
RPN_PRE_NMS_TEST = 1000
RPN_POST_NMS_TEST = 1000
RPN_NMS_THRESH = 0.7

# RoI head 后处理 —— 本研究的**候选导出点**在这之后
BOX_SCORE_THRESH = 0.0        # 不设分数阈值: 记录一直保留到 score=0
BOX_NMS_THRESH = 0.5          # 标准 per-class NMS
BOX_DETECTIONS_PER_IMG = 300  # 输出记录上限, 与参照臂的 Top300 存储对齐

BACKBONES = {"resnet18": ResNet18_Weights.IMAGENET1K_V1, "resnet50": ResNet50_Weights.IMAGENET1K_V1}


def build(backbone: str = "resnet18", *, num_classes: int = NUM_CLASSES,
          min_size: int = INPUT_SIZE, max_size: int = INPUT_SIZE) -> FasterRCNN:
    if backbone not in BACKBONES:
        raise ValueError(f"backbone must be one of {sorted(BACKBONES)}, got {backbone!r}")
    bb = resnet_fpn_backbone(backbone_name=backbone, weights=BACKBONES[backbone], trainable_layers=5)
    model = FasterRCNN(
        backbone=bb,
        num_classes=num_classes,
        min_size=min_size,
        max_size=max_size,
        image_mean=list(IMAGE_MEAN),
        image_std=list(IMAGE_STD),
        rpn_pre_nms_top_n_train=RPN_PRE_NMS_TRAIN,
        rpn_post_nms_top_n_train=RPN_POST_NMS_TRAIN,
        rpn_pre_nms_top_n_test=RPN_PRE_NMS_TEST,
        rpn_post_nms_top_n_test=RPN_POST_NMS_TEST,
        rpn_nms_thresh=RPN_NMS_THRESH,
        box_score_thresh=BOX_SCORE_THRESH,
        box_nms_thresh=BOX_NMS_THRESH,
        box_detections_per_img=BOX_DETECTIONS_PER_IMG,
    )
    head = model.roi_heads.box_predictor
    if not isinstance(head, FastRCNNPredictor):
        raise TypeError("unexpected box predictor")
    return model


def declaration(backbone: str = "resnet18") -> dict:
    u"""导出点/预处理/分数语义/源身份的完整声明, 写进 config 与候选表。"""
    return {
        "detector_family": "Faster R-CNN",
        "detector_subtype": "two-stage, anchor-based RPN + RoI (Fast R-CNN) head, FPN",
        "implementation": f"torchvision.models.detection.faster_rcnn.FasterRCNN (torchvision {torchvision.__version__})",
        "torch": torch.__version__,
        "backbone": f"{backbone}-FPN, ImageNet-pretrained init (IMAGENET1K_V1), trainable_layers=5",
        "num_classes": NUM_CLASSES,
        "class_index_convention": "0 = background; 1..8 = Road8 顺序 person,bicycle,car,motorcycle,bus,train,truck,traffic light",
        "train_dataset": "AOP-Road8-LargeClean-v1 TRAIN10K (frozen release ROAD8 GT)",
        "preprocess": {
            "input": "RGB PIL image (1280x720)",
            "resize": f"GeneralizedRCNNTransform: 保持长宽比, min_size=max_size={INPUT_SIZE} -> 长边 {INPUT_SIZE}",
            "letterbox": False,
            "tensor_dtype": "float32",
            "scale": "divide by 255",
            "normalization_mean_std": [list(IMAGE_MEAN), list(IMAGE_STD)],
        },
        "rpn_internal_pipeline_not_output_budget": {
            "pre_nms_top_n_test": RPN_PRE_NMS_TEST,
            "post_nms_top_n_test": RPN_POST_NMS_TEST,
            "nms_thresh": RPN_NMS_THRESH,
            "note": "RPN proposal 预算是检测器内部管线, 不是本研究的最终输出记录预算。",
        },
        "candidate_export_point": {
            "primary_pre_nms_topn100": (
                "RoI head 完成分类与框回归、裁剪与去背景之后, **最终 RoI 分数筛选 / 退化框过滤 / "
                "NMS / top-N 之前**; 按原分数取前缀 top-100。这是 M11 主用候选流。"
            ),
            "secondary_post_nms_topn300": (
                "官方 postprocess_detections 的最终输出, 逐位等于 model(images); 只作登记备查, "
                "不参与 M11 比较。"
            ),
            "note": "两条流出自**同一次前向**, 导出点各自登记, 不是两次推理。",
        },
        "threshold_nms_topn_order": [
            f"1. 对每个 RoI 的 {NUM_CLASSES} 维 pre-softmax class logits 做 softmax",
            "2. 只保留 8 个 Road8 前景类 (丢掉 background 列)",
            f"3. score threshold = {BOX_SCORE_THRESH} (不设阈值)",
            f"4. per-class NMS (torchvision batched_nms), IoU 阈值 = {BOX_NMS_THRESH}",
            f"5. 各类拼接后按 score 取 top-{BOX_DETECTIONS_PER_IMG}",
        ],
        "threshold_nms_topn_scope": (
            "上面 5 步是**官方后处理管线**, 只在备查流上执行; 主用流在它之前导出。"
            "特别地, 主用流不做逐类 NMS, 所以同一目标可能留下多条近重复记录 —— "
            "它的 COCO AP 因此低于备查流。两条数字都要报, 不能只报高的那条。"
        ),
        "score_semantics": "softmax(class_logits[roi])[class] —— RoI 级类别后验概率; 不是 RPN objectness, 不是任何其它量",
        "source_id_definition": "source_id = 'roi{index:05d}', index 为该 RoI 在 box_head 输入批次中的稳定行号; 一条 RoI 可派生出多个不同类别候选记录, 它们共享同一 source_id, 不得去重",
        "record_to_native_association": (
            "one-to-many: 一个 native RoI 行最多对应 8 条 candidate records (每个前景类各一条)。"
            "主用流不做 NMS, 所以这个上界就是实际的 8 —— 不得按 source_id 去重或合并。"
        ),
        "native_layer": {
            "native_vector": "roi_heads.box_head 输出 (TwoMLPHead: 256*7*7 -> 1024 -> 1024), 每个 RoI 一行",
            "native_vector_dimension": 1024,
            "class_signal": "roi_heads.box_predictor.cls_score 的 **完整 pre-softmax** 输出, 9 列全存 (0=background, 1..8=Road8)",
            "class_signal_dimension": 9,
            "background_logit": (
                "第 0 列必须一起存: 少了它 softmax 的分母就不完整, "
                "score = softmax(logits)[class] 就无法**正向**重建并逐条校验 —— "
                "而用 score 的逆 sigmoid 冒充 logits 正是合同明令禁止的。"
                "特征构造时只用 1..8 列 (ROAD8_SIGNAL_SLICE), 与参照臂的 8 列 block 宽度对齐。"
            ),
        },
        "sorting_and_tie_break": [
            "score descending",
            "source_id ascending (RoI 行号小者在前)",
            "Road8 class id ascending",
            "原始 (RoI-major, class-minor) 次序 ascending",
        ],
        "bbox_policy": "box_regression 作用在 RoI 上得到的 xyxy, 映射回**原图**坐标系 (GeneralizedRCNNTransform.postprocess 逆变换), 不裁剪",
    }


def summarise(model: FasterRCNN) -> dict:
    n_param = sum(p.numel() for p in model.parameters())
    return {
        "parameters": int(n_param),
        "trainable_parameters": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "box_head": repr(model.roi_heads.box_head),
        "box_predictor": repr(model.roi_heads.box_predictor),
        "transform": repr(model.transform),
    }


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    d = declaration("resnet18")
    print(json.dumps(d, ensure_ascii=False, indent=2))
    m = build("resnet18")
    print(json.dumps(summarise(m), ensure_ascii=False, indent=2))
