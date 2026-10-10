# -*- coding: utf-8 -*-
u"""把训练好的 Faster R-CNN 冻结导出成 M11 的 candidate / native 表。

一次前向同时导出**两条**候选流, 导出点分别登记:

  主用流 `PRE_NMS_TOPN100`  —— 检测头完成分类与框回归之后、**最终 RoI 分数筛选与 NMS 之前**,
      按 (score 降序, source_id 升序, Road8 类升序) 取 Top100。
      这是指导文档 §03 建议的主支位置, 与 YOLO 的"固定 NMS 前 Top100 候选协议"对应。
  登记备查流 `POST_NMS_TOPN300` —— 官方 `postprocess_detections` 的最终输出, 逐位比对通过。

两条流来自**同一次前向**, 所以不额外花时间, 也不存在两次推理不一致的问题。

做法:
  1) 挂 hook 抓三样东西 —— 进 box_head 的 proposal、box_head 输出的 1024 维 RoI 特征、
     box_predictor 输出的 **pre-softmax** class logits 与 box regression;
  2) **逐条复刻** torchvision `RoIHeads.postprocess_detections` 的算法, 但额外把
     每条扁平化的 (roi, class) 二元组一路带下来 —— 官方实现不暴露这个对应关系;
     复刻结果与官方 `model(images)` 输出做**逐位 assert**, 对不上直接失败;
  3) 在复刻过程里, 顺手把"分数筛选/NMS 之前"的那一版记录下来, 就是主用流。

用法:
  python -B src/export_frcnn.py --checkpoint outputs/frcnn_r18fpn_road8/best.pth \
      --split DEV2K --out outputs/frcnn_r18fpn_road8/export_dev
  python -B src/export_frcnn.py --checkpoint ... --split DEV2K --limit 40   # 小样本 QA
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.ops import boxes as box_ops

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
EXPERIMENT_ROOT = Path(os.environ.get('M11_EXPERIMENT_ROOT', "external_assets"))
RELEASE = EXPERIMENT_ROOT / 'AOP_Road8_LC_v1_RTDERTv2_R18VD_RELEASE_v1'
IMAGE_ROOT = EXPERIMENT_ROOT / 'bdd100k_images/bdd100k/images/100k/train'

sys.path.insert(0, str(HERE))
import model as M  # noqa: E402
import road8_data as D  # noqa: E402

NATIVE_SCHEMA_ID = "frcnn_r18fpn_road8_roihead1024_logits9_v1"
ROAD8_NAMES = list(D.ROAD8_NAMES)
MIN_BOX_SIZE = 1e-2  # torchvision remove_small_boxes 的固定参数

PROTOCOL_PRIMARY = "PRE_NMS_TOPN100"
PROTOCOL_SECONDARY = "POST_NMS_TOPN300"
PRIMARY_TOP_N = 100


def log(msg: str, fp=None) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if fp is not None:
        fp.write(line + "\n")
        fp.flush()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class Capture:
    u"""抓 RoI head 的三个中间张量。batch 恒为 1, 所以每个都是单个张量。"""

    def __init__(self, model):
        self.model = model
        self.reset()
        self.h1 = model.roi_heads.box_roi_pool.register_forward_hook(self._roi_pool)
        self.h2 = model.roi_heads.box_head.register_forward_hook(self._head)
        self.h3 = model.roi_heads.box_predictor.register_forward_hook(self._pred)

    def reset(self):
        self.proposals = None
        self.box_features = None
        self.cls_logits = None
        self.box_regression = None

    def _roi_pool(self, module, inputs, output):
        # inputs = (features_dict, proposals_list, image_shapes)
        self.proposals = inputs[1][0].detach()

    def _head(self, module, inputs, output):
        self.box_features = output.detach()

    def _pred(self, module, inputs, output):
        self.cls_logits, self.box_regression = output[0].detach(), output[1].detach()

    def close(self):
        for h in (self.h1, self.h2, self.h3):
            h.remove()


def decode_records(model, cap: Capture, image_shape, original_size):
    u"""复刻 postprocess_detections, 同时留下**导出点之前**的那一版记录。

    返回 dict:
      pre  : (boxes_original_xyxy, scores, roi_index, class_index) —— 解码 + clip + 去背景之后,
             分数阈值 / 退化框过滤 / NMS / top-N **之前**。这是主用候选流的来源。
      post : 同上, 但走完官方全部后处理。**顺序与官方输出一致**, 已逐位验证。
    """
    roi_heads = model.roi_heads
    n_fg = model.roi_heads.box_predictor.cls_score.out_features - 1  # 8

    proposals = cap.proposals                       # (P,4) 在 resize 后坐标
    cls_logits = cap.cls_logits                     # (P, 9) pre-softmax
    box_regression = cap.box_regression             # (P, 36)
    P = int(proposals.shape[0])

    # BoxCoder.decode 只接受 list[Tensor] 的 boxes, 单图就是长度 1 的 list
    pred_boxes = roi_heads.box_coder.decode(box_regression, [proposals])   # (P,9,4)
    pred_scores = F.softmax(cls_logits, -1)                               # (P,9)

    boxes = box_ops.clip_boxes_to_image(pred_boxes, image_shape)          # 官方: 先 clip
    boxes = boxes[:, 1:]                                                  # 丢 background
    scores = pred_scores[:, 1:]

    # 建在与 scores 同一个设备上 —— 否则 `tensor[keep]` 会因 index 与数据不同设备而报错。
    device = box_regression.device
    roi_idx = torch.arange(P, device=device).view(P, 1).expand(P, n_fg).reshape(-1)
    cls_idx = torch.arange(n_fg, device=device).view(1, n_fg).expand(P, n_fg).reshape(-1)
    b = boxes.reshape(-1, 4)
    s = scores.reshape(-1)

    # ── 导出点 (主用流) ────────────────────────────────────────────────
    # 此刻每条 (roi, class) 都已经有了**该类别自己的 decoded box**和**完整分类空间 softmax 后
    # 取出的 Road8 分数**(没有重新归一化成八类)。尚未做分数阈值、退化框过滤、NMS、top-N。
    pre = (b, s, roi_idx, cls_idx)

    keep = torch.where(s > roi_heads.score_thresh)[0]
    b, s, roi_idx, cls_idx = b[keep], s[keep], roi_idx[keep], cls_idx[keep]

    keep = box_ops.remove_small_boxes(b, min_size=MIN_BOX_SIZE)
    b, s, roi_idx, cls_idx = b[keep], s[keep], roi_idx[keep], cls_idx[keep]

    # cls_idx + 1: **必须传 1 基类别号**, 不能传 0 基的类索引。
    # batched_nms 的按类隔离靠的是 offsets = idxs * (max_coordinate + 1) 再加到坐标上,
    # 所以 idxs 的绝对值会改变坐标的浮点舍入 —— IoU 恰好卡在阈值附近的成对样本会因此翻转。
    # 官方传的是 1 基 label (1..8), 复刻必须传同一个, 逐位一致才成立 (实测差异: 某图多留 1 条)。
    keep = box_ops.batched_nms(b, s, cls_idx + 1, roi_heads.nms_thresh)
    keep = keep[: roi_heads.detections_per_img]
    post = (b[keep], s[keep], roi_idx[keep], cls_idx[keep])

    # 两个流都走官方同一条 postprocess 逆变换 —— resize 的还原与裁剪保持完全一致,
    # 于是两条流处在同一个坐标空间里, 可以直接比。
    def to_original(entry):
        u"""逆变换到原图坐标。返回 (boxes, scores, labels, roi_index, class_index) ——
        labels 是 1 基的 Road8 id, 与官方输出的 labels 同一空间; roi/class 索引是官方不暴露的额外信息。"""
        bb, ss, rr, cc = entry
        if len(bb) == 0:
            return (bb.cpu().numpy().astype(np.float64), ss.cpu().numpy().astype(np.float64),
                    (cc + 1).cpu().numpy().astype(np.int64),
                    rr.cpu().numpy().astype(np.int64), cc.cpu().numpy().astype(np.int64))
        res = model.transform.postprocess(
            [{"boxes": bb, "labels": cc + 1, "scores": ss}], [image_shape], [original_size]
        )[0]
        return (res["boxes"].cpu().numpy().astype(np.float64), res["scores"].cpu().numpy().astype(np.float64),
                res["labels"].cpu().numpy().astype(np.int64),
                rr.cpu().numpy().astype(np.int64), cc.cpu().numpy().astype(np.int64))

    return {"pre": to_original(pre), "post": to_original(post)}


def canonical_order(boxes, scores, labels, roi_idx, cls_idx):
    u"""一次写定的排序: 分数降序 -> source_id 升序 -> Road8 ID 升序 -> 原始次序。"""
    order = np.lexsort((np.arange(len(scores)), cls_idx, roi_idx, -scores))
    return boxes[order], scores[order], labels[order], roi_idx[order], cls_idx[order]


def verify_against_official(official, mine, tol=0.0):
    u"""复刻结果 vs 官方输出逐位比对。tol=0 表示要求逐位相等。"""
    ob, os_, ol = official["boxes"], official["scores"], official["labels"]
    mb, ms, ml = mine[0], mine[1], mine[2]
    if not (len(ob) == len(mb) == len(os_) == len(ms) == len(ol) == len(ml)):
        raise AssertionError(f"record count mismatch: official={len(ob)} replica={len(mb)}")
    if tol == 0.0:
        if not torch.equal(ob.cpu(), torch.as_tensor(mb, dtype=ob.dtype)):
            raise AssertionError("boxes are not bit-identical to the official output")
        if not torch.equal(os_.cpu(), torch.as_tensor(ms, dtype=os_.dtype)):
            raise AssertionError("scores are not bit-identical to the official output")
    else:
        if not torch.allclose(ob.cpu(), torch.as_tensor(mb, dtype=ob.dtype), atol=tol, rtol=0):
            raise AssertionError("boxes differ beyond tolerance")
        if not torch.allclose(os_.cpu(), torch.as_tensor(ms, dtype=os_.dtype), atol=tol, rtol=0):
            raise AssertionError("scores differ beyond tolerance")
    if not torch.equal(ol.cpu().to(torch.int64), torch.as_tensor(ml, dtype=torch.int64)):
        raise AssertionError("labels differ from the official output")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="DEV2K", choices=["DEV2K", "TRAIN10K"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="只导出前 N 张 (小样本 QA)")
    ap.add_argument("--shard-images", type=int, default=500)
    ap.add_argument("--from-subset", default="", help="CSV/每行一个 image_id; 只导这些图")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "candidates").mkdir(exist_ok=True)
    (out / "native_state").mkdir(exist_ok=True)
    logf = open(out / "export_log.txt", "a", encoding="utf-8")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    backbone = ck.get("backbone", "resnet18")
    model = M.build(backbone)
    model.load_state_dict(ck["model"])
    model.eval().to(device)
    decl = M.declaration(backbone)
    ckpt_sha = sha256_file(Path(args.checkpoint))
    log(f"checkpoint {args.checkpoint}  sha256={ckpt_sha[:16]}…  epoch={ck.get('epoch')}  "
        f"val_f1={ck.get('val_f1_50')}", logf)

    cap = Capture(model)

    gt_name = "DEV2K_ROAD8_GT.json" if args.split == "DEV2K" else "TRAIN10K_ROAD8_GT.json"
    ds = D.build(RELEASE / "gt" / gt_name, IMAGE_ROOT, train=False)
    idx = list(range(len(ds)))
    if args.from_subset:
        wanted = {line.strip() for line in Path(args.from_subset).read_text().splitlines() if line.strip()}
        idx = [i for i, rec in enumerate(ds.images) if rec["id"] in wanted]
        log(f"restricted to {len(idx)} images from {args.from_subset}", logf)
    if args.limit:
        idx = idx[: args.limit]
    log(f"split={args.split}  images={len(idx)}", logf)

    detector_id = f"FasterRCNN-{backbone}-FPN-Road8"
    frozen_config = {
        "native_schema_id": NATIVE_SCHEMA_ID,
        "detector_id": detector_id,
        "detector_declaration": decl,
        "checkpoint_sha256": ckpt_sha,
        "checkpoint_epoch": ck.get("epoch"),
        "dataset_version": "AOP-Road8-LargeClean-v1",
        "source_id_definition": decl["source_id_definition"],
        "export_precision": {
            "inference": "float32, autocast disabled",
            "reason": "官方前向与逐条复刻必须同精度, 后处理的逐位一致性检查才成立",
            "training_precision": "float16 autocast with GradScaler (只影响训练, 不影响本冻结导出)",
        },
        "protocols": {
            PROTOCOL_PRIMARY: {
                "role": "primary candidate stream for the M11 comparison",
                "export_point": (
                    "after detection-head classification and box regression, after clipping and "
                    "background removal, BEFORE the final RoI score threshold / degenerate-box filter / "
                    "NMS / top-N"
                ),
                "upstream_rpn_pipeline_preserved": decl["rpn_internal_pipeline_not_output_budget"],
                "score_space": "softmax over the complete 9-way class space, then the 8 Road8 columns; not renormalised to eight classes",
                "box_for_each_record": "the decoded box of that record's own class (pred_boxes[:, class])",
                "box_space": "original image xyxy, via the same transform.postprocess inverse path as the official output",
                "order": ["score descending", "source_id ascending", "Road8 class id ascending", "original (roi, class) order"],
                "top_n": PRIMARY_TOP_N,
            },
            PROTOCOL_SECONDARY: {
                "role": "registered reference only; the M11 comparison does not use this stream",
                "export_point": "official torchvision postprocess_detections output (threshold -> remove_small_boxes -> per-class NMS -> top-N)",
                "bit_exact_against_official": True,
                "order": ["score descending", "source_id ascending", "Road8 class id ascending", "original (roi, class) order"],
                "top_n": int(model.roi_heads.detections_per_img),
            },
        },
        "native_policy": {
            "native_vector": "box_head 输出 (进 box_predictor 之前), 1024 维, float32 计算 -> float16 存储",
            "class_signal": "box_predictor.cls_score 的 pre-softmax 输出, 9 维 (0=background, 1..8=Road8), float32; 存完整 9 列而非只 8 个 Road8 列, 这样 softmax 分母不缺项, detector score 可正向重建并逐条校验",
            "stored_rows": "只存被两条候选流引用到的 RoI 行 (source_id 并集); 按 (image_id, source_id) 唯一, 不做任何去重",
            "shared_source_policy": "同一个 RoI 的不同类别记录共享 native 向量, 但其 decoded 框可能不同; 不得按 source_id 合并记录",
        },
    }
    # 冻结配置的身份只哈希上面那本字典 —— **不含 split**。
    # 早先把 "split" 放进被哈希的字典里, 结果 DEV2K 与 TRAIN10K 对同一份配置报出两个不同的
    # export_config_sha256, 每条导出记录上盖的章也跟着分裂。两份导出会看起来像用了两套配置,
    # 而实际上配置、checkpoint、协议逐字相同, 只是取图范围不同。split 是运行参数, 不是配置。
    cfg_sha = sha256_bytes(json.dumps(frozen_config, ensure_ascii=False, sort_keys=True).encode("utf-8"))
    export_config = dict(frozen_config)
    export_config["export_config_sha256"] = cfg_sha
    export_config["export_config_sha256_scope"] = (
        "本文件除 split / export_config_sha256 / export_config_sha256_scope 三键外的全部内容。"
        "split 是本次运行的取图范围, 不计入配置身份, 因此 DEV2K 与 TRAIN10K 共享同一个 "
        "export_config_sha256; 记录级 export_config_sha256 也因此可比。"
    )
    export_config["split"] = args.split
    (out / "export_config.json").write_text(json.dumps(export_config, ensure_ascii=False, indent=2), encoding="utf-8")

    primary_rows, secondary_rows = [], []
    pre_counts, primary_counts, secondary_counts = [], [], []
    index_rows = []

    # native 分片缓冲。**必须攒够一个分片再写**: 早先的写法是对同一个分片路径每图
    # np.savez_compressed 一次, 后一张图直接把前一张覆盖掉 —— 2000 图只留下 4 张。
    native_buffer: list[dict] = []
    shard_index = 0
    current_shard = ""

    def flush_shard(force: bool) -> None:
        u"""把缓冲里的图合成一个分片落盘。分片内 image_id 是**逐行**存的, 不是逐图。"""
        nonlocal native_buffer, shard_index, current_shard
        if not native_buffer:
            return
        current_shard = f"state_{shard_index:04d}.npz"
        np.savez_compressed(
            out / "native_state" / current_shard,
            image_id=np.concatenate([b["image_id"] for b in native_buffer]),
            source_id=np.concatenate([b["source_id"] for b in native_buffer]),
            roi_index=np.concatenate([b["roi_index"] for b in native_buffer]),
            native_vector=np.concatenate([b["native_vector"] for b in native_buffer]),
            class_signal=np.concatenate([b["class_signal"] for b in native_buffer]),
            proposal_xyxy_resized=np.concatenate([b["proposal_xyxy_resized"] for b in native_buffer]),
            native_schema_id=np.array([NATIVE_SCHEMA_ID]),
        )
        log(f"  shard {current_shard}: {len(native_buffer)} images, "
            f"{sum(len(b['source_id']) for b in native_buffer)} rows", logf)
        native_buffer = []
        shard_index += 1

    for n, i in enumerate(idx):
        tensor, _ = ds[i]
        rec = ds.images[i]
        image_id = rec["id"]
        original_size = (int(rec["height"]), int(rec["width"]))
        cap.reset()
        # 导出**不开 autocast**: 官方前向与我的复刻必须跑在同一精度上, 逐位 assert 才有意义。
        # 训练用 fp16 autocast 不影响这里 —— 冻结导出选 fp32 是为了可复现, 且写进 export_config。
        with torch.inference_mode():
            official = model([tensor.to(device)])[0]

        # transform 之后该图实际被 resize 成的尺寸 —— clip 在**这个**空间里做。
        # GeneralizedRCNNTransform 的 resize 是确定性的, 直接复算
        # (注意 max_size 是标量, 不是 tuple):
        h, w = original_size
        size = model.transform.min_size[0] / min(h, w)
        if max(h, w) * size > model.transform.max_size:
            size = model.transform.max_size / max(h, w)
        image_shape = (int(round(h * size)), int(round(w * size)))

        streams = decode_records(model, cap, image_shape, original_size)
        # 后处理复刻必须与官方逐位一致 —— 不成立就直接炸, 不静默降级。
        verify_against_official(official, streams["post"][:3], tol=0.0)

        pb, ps, pl, pri, pci = canonical_order(*streams["pre"])
        sb, ss, sl, sri, sci = canonical_order(*streams["post"])
        pre_counts.append(len(ps))
        primary = (pb[:PRIMARY_TOP_N], ps[:PRIMARY_TOP_N], pl[:PRIMARY_TOP_N],
                   pri[:PRIMARY_TOP_N], pci[:PRIMARY_TOP_N])
        primary_counts.append(len(primary[1]))
        secondary_counts.append(len(ss))
        if len(primary[1]) < PRIMARY_TOP_N:
            log(f"  ! {image_id}: pre-NMS stream has only {len(ps)} records (< {PRIMARY_TOP_N})", logf)

        # native: 两条流引用到的 RoI 并集, 按 box_head 输入次序排列
        used_rois = np.unique(np.concatenate([primary[3], sri]))
        roi_to_row = {int(r): k for k, r in enumerate(used_rois)}
        source_ids = np.array([f"roi{int(r):05d}" for r in used_rois])

        feat = cap.box_features.float().cpu().numpy()          # (P,1024) RoI 输入次序
        logits = cap.cls_logits.float().cpu().numpy()          # (P,9)
        proposals = cap.proposals.float().cpu().numpy()        # (P,4) resize 后坐标
        native_buffer.append({
            # 逐行重复的 image_id: 这样 join 只需要 (image_id, source_id) 两列, 不依赖
            # "分片里有几张图"这种隐含假设。
            "image_id": np.repeat(np.array([image_id]), len(source_ids)),
            "source_id": source_ids,
            "roi_index": used_rois.astype(np.int32),
            "native_vector": feat[used_rois].astype(np.float16),
            "class_signal": logits[used_rois].astype(np.float32),
            "proposal_xyxy_resized": proposals[used_rois].astype(np.float32),
        })
        if len(native_buffer) >= args.shard_images or n == len(idx) - 1:
            flush_shard(force=n == len(idx) - 1)

        img_sha = sha256_file(IMAGE_ROOT / rec["file_name"])
        payload = json.dumps({"split": args.split, "image_id": image_id, "n": len(primary[1]),
                              "protocol": PROTOCOL_PRIMARY, "det": ckpt_sha, "exp": cfg_sha},
                             sort_keys=True).encode()
        asset_id = sha256_bytes(payload)

        def emit(protocol, boxes, scores, labels, roi_idx, cls_idx, asset):
            frame = []
            for rank, (bx, sc, lb, ri, ci) in enumerate(zip(boxes, scores, labels, roi_idx, cls_idx), start=1):
                x1, y1, x2, y2 = [float(v) for v in bx]
                frame.append({
                    "dataset_version": "AOP-Road8-LargeClean-v1",
                    "split": args.split,
                    "protocol_id": protocol,
                    "image_id": image_id,
                    "image_sha256": img_sha,
                    "detector_id": detector_id,
                    "checkpoint_sha256": ckpt_sha,
                    "export_config_sha256": cfg_sha,
                    "candidate_asset_id": asset,
                    "candidate_record_id": f"frcnn:{args.split}:{image_id}:roi{int(ri):05d}:c{int(ci) + 1}",
                    "rank": rank,
                    "predicted_road8_class_id": int(lb),
                    "predicted_road8_class_name": ROAD8_NAMES[int(lb) - 1],
                    "original_score": float(sc),
                    "box_x1": x1, "box_y1": y1, "box_x2": x2, "box_y2": y2,
                    "box_cx": (x1 + x2) / 2.0, "box_cy": (y1 + y2) / 2.0,
                    "box_w": x2 - x1, "box_h": y2 - y1,
                    "source_id": f"roi{int(ri):05d}",
                    "source_order": int(ri) * model.roi_heads.box_predictor.cls_score.out_features + int(ci) + 1,
                    "image_width": int(rec["width"]),
                    "image_height": int(rec["height"]),
                })
            return frame

        primary_rows.extend(emit(PROTOCOL_PRIMARY, *primary, asset_id))
        secondary_rows.extend(emit(PROTOCOL_SECONDARY, sb, ss, sl, sri, sci, asset_id))
        index_rows.append({
            "image_id": image_id,
            "protocol_id": PROTOCOL_PRIMARY,
            "candidate_asset_id": asset_id,
            "checkpoint_sha256": ckpt_sha,
            "export_config_sha256": cfg_sha,
            "native_state_shard": current_shard,
            "native_rows": int(len(used_rois)),
            "pre_nms_records": int(len(ps)),
            "primary_records": int(len(primary[1])),
            "secondary_records": int(len(ss)),
        })

        if n % 100 == 0 or n == len(idx) - 1:
            log(f"  {n + 1}/{len(idx)}  {image_id}  pre_nms={len(ps)}  primary={len(primary[1])}  "
                f"post_nms={len(ss)}  rois_stored={len(used_rois)}", logf)

    flush_shard(force=True)
    cap.close()

    import pyarrow as pa
    import pyarrow.parquet as pq

    def write_table(rows, name):
        if not rows:
            return None
        table = pa.Table.from_pylist(rows)
        path = out / "candidates" / name
        pq.write_table(table, path, compression="zstd")
        return path

    primary_path = write_table(primary_rows, f"{args.split.lower()}_candidates.parquet")
    secondary_path = write_table(secondary_rows, f"{args.split.lower()}_candidates_post_nms.parquet")

    def describe(values):
        a = np.asarray(values, dtype=np.int64)
        if not len(a):
            return {}
        return {"min": int(a.min()), "median": float(np.median(a)), "mean": float(a.mean()),
                "max": int(a.max()), "p05": float(np.quantile(a, 0.05)),
                "p95": float(np.quantile(a, 0.95))}

    primary_arr = np.asarray(primary_counts, dtype=np.int64)
    summary = {
        "split": args.split,
        "detector_id": detector_id,
        "protocol_primary": PROTOCOL_PRIMARY,
        "protocol_secondary": PROTOCOL_SECONDARY,
        "images_exported": len(primary_counts),
        # §03 要求: 全体图像的**真实候选数**, 以及 N_i<100 / N_i<50 的图数。
        # 这里的 "真实候选数" 指导出点的候选流规模 (pre-NMS); Top100 是协议动作, 不是候选池大小。
        "pre_nms_records_per_image": describe(pre_counts),
        "pre_nms_images_below_100": int((np.asarray(pre_counts) < 100).sum()),
        "pre_nms_images_below_50": int((np.asarray(pre_counts) < 50).sum()),
        "primary_records_per_image": describe(primary_counts),
        "primary_images_below_100": int((primary_arr < 100).sum()),
        "primary_images_below_50": int((primary_arr < 50).sum()),
        "secondary_records_per_image": describe(secondary_counts),
        "total_primary_records": int(primary_arr.sum()),
        "candidates_file": primary_path.name if primary_path else None,
        "candidates_post_nms_file": secondary_path.name if secondary_path else None,
        "native_schema_id": NATIVE_SCHEMA_ID,
        "checkpoint_sha256": ckpt_sha,
        "export_config_sha256": cfg_sha,
    }
    (out / "export_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "candidate_index.json").write_text(
        json.dumps(index_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"pre-NMS records/image: {json.dumps(summary['pre_nms_records_per_image'], ensure_ascii=False)}", logf)
    log(f"pre-NMS images below 100: {summary['pre_nms_images_below_100']} / {len(pre_counts)}", logf)
    log(f"pre-NMS images below 50:  {summary['pre_nms_images_below_50']} / {len(pre_counts)}", logf)
    log(f"DONE -> {out}", logf)
    logf.close()


if __name__ == "__main__":
    main()
