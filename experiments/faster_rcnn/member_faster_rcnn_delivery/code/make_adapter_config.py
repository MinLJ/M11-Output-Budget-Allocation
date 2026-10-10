# -*- coding: utf-8 -*-
u"""从**真实产出物**生成填好的 `configs/faster_rcnn_adapter.json` 与交付版 `capabilities.json`。

为什么不用手填: 合同要求这个配置冻结住检测器族/版本/仓库提交/config 与 checkpoint 的 SHA、
预处理、候选导出点、阈值/NMS/top-N 顺序、score 语义、`source_id` 生成规则、
record↔native 基数、native 层/维度/dtype。手填的 SHA 一定会和实际跑的东西漂移;
这里全部从 `export_config.json` 和 `detector_declaration.json` 现读现算,
缺任何一项就**直接失败**, 不写 null 充数。

用法:
  python -B lc_frcnn/make_adapter_config.py \\
      --declaration ../frcnn_road8/outputs/frcnn_r18fpn_road8/detector_declaration.json \\
      --export-config ../frcnn_road8/outputs/frcnn_r18fpn_road8/export_dev/export_config.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
M11_ROOT = HERE.parent

# 按路径直接跑 `python lc_frcnn/make_adapter_config.py` 时, sys.path[0] 是 **lc_frcnn/ 自己**,
# 不是它的父目录 —— 于是 `import lc_frcnn` 找不到包。其余入口脚本都显式插了父目录, 这里照做。
sys.path.insert(0, str(M11_ROOT))

from lc_frcnn.adapter import DETECTOR_ID, DETECTOR_VERSION, NATIVE_SCHEMA_ID  # noqa: E402
from lc_frcnn.config_corrections import (  # noqa: E402
    canonical_sha256, corrected_detector_declaration, stale_declaration_sha256)
from lc_frcnn.features_frcnn import CLASS_SIGNAL_DIM, NATIVE_VECTOR_DIM  # noqa: E402

REQUIRED_DECLARATION_KEYS = (
    "detector_family", "detector_subtype", "implementation", "backbone", "num_classes",
    "class_index_convention", "preprocess", "candidate_export_point",
    "threshold_nms_topn_order", "score_semantics", "source_id_definition",
    "record_to_native_association", "native_layer", "sorting_and_tie_break", "bbox_policy",
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--declaration", required=True)
    ap.add_argument("--export-config", required=True)
    # 默认写到独立目录, 由 make_delivery 拷进交付包的 configs/ ——
    # 交付包每次重建都是整个目录删掉重来, 直接往里写会被下一次重建抹掉。
    ap.add_argument("--out-dir", default=str(M11_ROOT / "adapter_config"))
    args = ap.parse_args()

    declaration = json.loads(Path(args.declaration).read_text(encoding="utf-8"))
    export_config = json.loads(Path(args.export_config).read_text(encoding="utf-8"))

    missing = [key for key in REQUIRED_DECLARATION_KEYS if not declaration.get(key)]
    if missing:
        raise SystemExit(f"detector declaration is incomplete, refusing to write a config with gaps: {missing}")
    for key in ("checkpoint_sha256", "export_config_sha256"):
        if not export_config.get(key):
            raise SystemExit(f"export_config.json lacks {key}; run the export before writing the adapter config")

    # r1 #11: 源声明的 bbox_policy 写「不裁剪」, 与导出实现 (官方 clip_boxes_to_image) 矛盾。
    # 适配配置与两份交付配置共用同一个更正函数 (lc_frcnn.config_corrections), 措辞不可能漂移;
    # 更正只改文本, 导出参数、记录与哈希对象里的原件身份一律不动。
    declaration = corrected_detector_declaration(declaration)

    config = {
        "status": "ADAPTED",
        "detector_family": "Faster R-CNN",
        "detector_version": DETECTOR_VERSION,
        "detector_id": DETECTOR_ID,
        "implementation": declaration["implementation"],
        "backbone": declaration["backbone"],
        "num_classes": declaration["num_classes"],
        "class_index_convention": declaration["class_index_convention"],
        "config_identity": {
            # 对**更正后**声明算哈希 —— 与交付 configs/detector_declaration.json 的内容一致,
            # 两边可以逐字对账 (更正前源声明的同算法哈希保留在 note 里, 便于与更早引用对照)。
            "declaration_sha256": canonical_sha256(declaration),
            "declaration_sha256_note": (
                "对更正后声明 (交付 configs/detector_declaration.json; 与源声明只差 bbox_policy "
                "一处措辞, 见其 bbox_policy_correction 键) 的规范化 JSON (sort_keys) SHA256; "
                f"更正前源声明同算法哈希为 {stale_declaration_sha256(declaration)}。"),
            "export_config_sha256": export_config["export_config_sha256"],
        },
        "checkpoint_sha256": export_config["checkpoint_sha256"],
        "preprocess": declaration["preprocess"],
        "candidate_export_point": declaration["candidate_export_point"],
        "rpn_internal_pipeline_not_output_budget": declaration["rpn_internal_pipeline_not_output_budget"],
        "threshold_nms_topn_order": declaration["threshold_nms_topn_order"],
        "score_semantics": declaration["score_semantics"],
        "source_id_definition": declaration["source_id_definition"],
        "native_layer": declaration["native_layer"],
        "native_dimension": NATIVE_VECTOR_DIM,
        "class_signal_dimension": CLASS_SIGNAL_DIM,
        "native_dtype": "float16 stored on disk (RoI-head feature) / float32 stored on disk (pre-softmax class logits)",
        "record_to_native_association": declaration["record_to_native_association"],
        "record_to_native_cardinality": "one native row per RoI; several class records may share one source_id",
        "native_schema_id": NATIVE_SCHEMA_ID,
        "sorting_and_tie_break": declaration["sorting_and_tie_break"],
        "bbox_policy": declaration["bbox_policy"],
        "candidate_export_implemented": True,
        "native_export_implemented": True,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "faster_rcnn_adapter.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    capabilities = {
        "version": "1.0",
        "reference": {
            "detector": "RT-DETRv2-R18VD",
            "status": "REFERENCE_IMPLEMENTED",
            "native_schema_id": "rtdetrv2_r18vd_l3_logits8_embedding256_v1",
        },
        "faster_rcnn": {
            "status": "ADAPTED",
            "native_schema_id": NATIVE_SCHEMA_ID,
            "export_implemented": True,
            "candidate_export_point": config["candidate_export_point"],
            "required_next_step": "None for this detector. RT-DETR PCA/scaler/model assets must never be cross-loaded.",
        },
        "yolo": {
            "status": "SCAFFOLD_ONLY",
            "export_implemented": False,
            "required_next_step": (
                "Freeze a concrete YOLO version and declare pre/post-NMS records, score semantics, "
                "native layer, and stable source_id."
            ),
        },
    }
    (out_dir / "capabilities.json").write_text(
        json.dumps(capabilities, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # drop-in: 覆盖层里的 adapter.py 就是交接包里那份 faster_rcnn.py。
    # 拷过去而不是重写一份, 免得两边漂移; 拷完再核一遍字节相同。
    dropin = out_dir / "lc_alloc" / "adapters" / "faster_rcnn.py"
    dropin.parent.mkdir(parents=True, exist_ok=True)
    source_bytes = (HERE / "adapter.py").read_bytes()
    dropin.write_bytes(source_bytes)
    if dropin.read_bytes() != source_bytes:
        raise SystemExit("drop-in copy verification failed")
    stale_stub = "raise AdapterNotImplementedError" in source_bytes.decode("utf-8")
    if stale_stub:
        raise SystemExit("refusing to deliver an adapter that still raises AdapterNotImplementedError")

    print(f"[adapter-config] -> {out_dir/'faster_rcnn_adapter.json'}")
    print(f"[adapter-config] -> {out_dir/'capabilities.json'}")
    print(f"[adapter-config] -> {out_dir/'lc_alloc/adapters/faster_rcnn.py'} (drop-in, byte-identical to overlay adapter.py)")
    print(f"[adapter-config] checkpoint_sha256={config['checkpoint_sha256'][:16]}… "
          f"export_config_sha256={config['config_identity']['export_config_sha256'][:16]}…")


if __name__ == "__main__":
    main()
