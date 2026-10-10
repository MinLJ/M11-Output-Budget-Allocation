# -*- coding: utf-8 -*-
u"""r1 #11: 冻结声明的 bbox_policy 措辞更正 (「不裁剪」 → 官方 clip_boxes_to_image)。

缺陷 (tools/findings_r1.json findings[11]): 三份冻结配置的 `bbox_policy` 写
「…postprocess 逆变换, 不裁剪」, 与同一文件的导出点描述 («裁剪与去背景之后»)、导出实现
(code/export_frcnn.py 在 pre 导出点之前执行 box_ops.clip_boxes_to_image)、以及实测
(候选表存在贴边框、无越界框) 三重矛盾。实际策略 = 官方 postprocess_detections 的
clip_boxes_to_image: 在 resize 后的网络输入空间裁剪, 再经 transform.postprocess 逆变换
映射回原图坐标。

为什么在覆盖层更正、不回头改 frcnn_road8 源文件:
  * 本分支是覆盖层; 源声明的文本已随导出身份固化 —— export_config_sha256 (d5423501…)
    是在包含该措辞的原件上算出的, 并逐条盖在导出记录上, 不可变更;
  * 这是**措辞**错误, 不是导出参数错误: 实现、记录、数字一概不动。
因此更正只发生在覆盖层生成配置的环节, 且是**透明**的:
  * 每个被更正的声明副本都带 `bbox_policy_correction` 键 (更正前原文 + 理由 + 实测);
  * export_config 的 export_config_sha256_scope 写明原件身份哈希的重算路径
    (按 original_text 恢复该字段后即可逐字复算出原哈希)。

所有会写这些配置的覆盖层脚本 (make_adapter_config.py / make_delivery.py) 共用这里的
同一个函数, 三份交付配置的文本因此不可能互相漂移。
"""
from __future__ import annotations

import copy
import hashlib
import json

BBOX_POLICY_STALE = (
    "box_regression 作用在 RoI 上得到的 xyxy, 映射回**原图**坐标系 "
    "(GeneralizedRCNNTransform.postprocess 逆变换), 不裁剪")

BBOX_POLICY_CORRECTED = (
    "box_regression 作用在 RoI 上得到的 xyxy: 沿用官方 postprocess_detections 的 "
    "clip_boxes_to_image 在 resize 后的网络输入空间裁剪, 再经 "
    "GeneralizedRCNNTransform.postprocess 逆变换映射回**原图**坐标系 "
    "(框始终落在 [0, W] × [0, H] 内)")

BBOX_POLICY_CORRECTION_KEY = "bbox_policy_correction"

_CORRECTION_NOTE = {
    "corrected_by": "frcnn_m11 覆盖层打包更正 (findings_r1 #11); 只更正措辞, 不改动任何导出参数与记录",
    "original_text": BBOX_POLICY_STALE,
    "reason": ("原措辞 (原文见 original_text) 与导出实现矛盾: code/export_frcnn.py 在 pre 导出点之前"
               "先执行官方 box_ops.clip_boxes_to_image (postprocess_detections 链路的裁剪步骤), "
               "之后才做 transform.postprocess 逆变换; 实测候选框存在贴边 "
               "(x1=0 / x2=图像宽 / y1=0 / y2=图像高) 且无越界框。"),
    "evidence": ("prepared/candidates_dev.parquet 200000 条: x1==0.0 3014 / x2==图像宽 5151 / "
                 "y1==0.0 465 / y2==图像高 431; 越界 0 条; results/qa_closed_loop.md 亦报告贴边 4.9231%。"),
    "identity_note": ("export_config 原件 (含更正前措辞) 的身份哈希 export_config_sha256 不因本更正改变; "
                      "逐字重算路径见交付 configs/export_config.json 的 export_config_sha256_scope。"),
}


def bbox_policy_correction_note() -> dict:
    u"""更正说明的独立副本 (调用方改不动本模块的常量)。"""
    return copy.deepcopy(_CORRECTION_NOTE)


def corrected_detector_declaration(declaration: dict) -> dict:
    u"""返回更正后的声明副本; 源文本与预期不符 (含已被更正过) 时直接失败, 不静默放过。"""
    if BBOX_POLICY_CORRECTION_KEY in declaration:
        raise SystemExit("声明已带 bbox_policy_correction 键, 拒绝重复更正")
    original = declaration.get("bbox_policy")
    if original != BBOX_POLICY_STALE:
        raise SystemExit(f"bbox_policy 源文本与预期不符, 拒绝自动更正: {original!r}")
    corrected = copy.deepcopy(declaration)
    corrected["bbox_policy"] = BBOX_POLICY_CORRECTED
    corrected[BBOX_POLICY_CORRECTION_KEY] = bbox_policy_correction_note()
    return corrected


def canonical_sha256(declaration: dict) -> str:
    u"""与 make_adapter_config.py 原算法一致: 规范 JSON (sort_keys) 的 SHA256。"""
    return hashlib.sha256(
        json.dumps(declaration, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def stale_declaration_sha256(corrected: dict) -> str:
    u"""更正前源声明的同算法哈希 (由更正副本复原原文后计算), 供身份说明引用。"""
    stale = {key: value for key, value in corrected.items() if key != BBOX_POLICY_CORRECTION_KEY}
    stale["bbox_policy"] = BBOX_POLICY_STALE
    return canonical_sha256(stale)


def export_config_sha256_scope_note() -> str:
    u"""交付 export_config 的 export_config_sha256_scope 重写文本。

    原件哈希必须保持可复算: 按 bbox_policy_correction.original_text 恢复该字段、删掉
    bbox_policy_correction 键、再去掉本 scope 讲的三个键之后, 内容逐字复原原件。
    """
    return (
        "export_config_sha256 覆盖**导出运行时冻结原件**除 split / export_config_sha256 / "
        "export_config_sha256_scope 三键外的全部内容 (原件与源文件逐字节相同)。本交付副本是"
        "覆盖层打包更正版 (findings_r1 #11): 相对原件只有一处措辞更正 —— "
        "detector_declaration.bbox_policy 的源措辞按实际实现更正为官方 clip_boxes_to_image "
        "(resize 后空间裁剪, 再经 transform.postprocess 逆变换回原图坐标), 并新增同处的 "
        "bbox_policy_correction 说明键; 其余字段值与原件相同。重算路径: 用 "
        "bbox_policy_correction.original_text 覆盖 bbox_policy、删除 bbox_policy_correction 键、"
        "再去掉上述三个键后即逐字复原原件, 其 SHA256 就是 export_config_sha256。split 是本次"
        "运行的取图范围, 不计入配置身份, 因此 DEV2K 与 TRAIN10K 共享同一个 export_config_sha256; "
        "记录级 export_config_sha256 也因此可比。")
