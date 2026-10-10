# -*- coding: utf-8 -*-
u"""在 Road8 (BDD100K) 上从零训练 Faster R-CNN。

与参照臂 RT-DETRv2-R18VD 的对应关系:
  RT-DETR  : 官方 R18VD 骨干 + COCO 预训练权重 -> Road8 微调
  FRCNN    : R18+FPN 骨干 + ImageNet 预训练权重 -> Road8 训练
两者同深度、同输入 640、同 GT、同类别顺序 —— 只差检测范式。

用法:
  python -B src/train_frcnn.py --epochs 15 --batch-size 4
  python -B src/train_frcnn.py --epochs 1 --subset 200 --no-val   # 冒烟
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
EXPERIMENT_ROOT = Path(os.environ.get('M11_EXPERIMENT_ROOT', "external_assets"))
RELEASE = EXPERIMENT_ROOT / 'AOP_Road8_LC_v1_RTDERTv2_R18VD_RELEASE_v1'
IMAGE_ROOT = EXPERIMENT_ROOT / 'bdd100k_images/bdd100k/images/100k/train'
ROLECSV = EXPERIMENT_ROOT / 'LC_ALLOC_M11_HANDOFF_v1/handoff_LC_ALLOC_M11_v1/configs/manifests/lc_train_role_split.csv'

sys.path.insert(0, str(HERE))
import model as M  # noqa: E402
import road8_data as D  # noqa: E402


def log(msg: str, fp=None) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if fp is not None:
        fp.write(line + "\n")
        fp.flush()


def evaluate_ap(model, loader, device, max_images: int) -> float:
    u"""训练期用的轻量 mAP@0.5 —— 贪心同类一对一匹配, 不替代正式 COCO 评价。"""
    from torchvision.ops import box_iou

    model.eval()
    tp = fp_ = 0
    n_gt = 0
    seen = 0
    with torch.inference_mode():
        for images, targets in loader:
            images = [im.to(device) for im in images]
            with torch.autocast("cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
                preds = model(images)
            for pred, tgt in zip(preds, targets):
                gb = tgt["boxes"].to(device)
                gl = tgt["labels"].to(device)
                n_gt += int(len(gb))
                if len(gb) == 0 or len(pred["boxes"]) == 0:
                    continue
                iou = box_iou(pred["boxes"], gb)
                used = torch.zeros(len(gb), dtype=torch.bool, device=device)
                for i in range(len(pred["boxes"])):
                    cand = torch.nonzero((gl == pred["labels"][i]) & (~used) & (iou[i] >= 0.5)).flatten()
                    if len(cand) == 0:
                        fp_ += 1
                        continue
                    best = cand[torch.argmax(iou[i][cand])]
                    used[best] = True
                    tp += 1
            seen += len(images)
            if seen >= max_images:
                break
    model.train()
    if tp + fp_ == 0 or n_gt == 0:
        return 0.0
    precision = tp / (tp + fp_)
    recall = tp / n_gt
    return 2 * precision * recall / max(precision + recall, 1e-12)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="resnet18")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=None, help="默认按 batch 线性缩放: 0.02*(bs/16)")
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--warmup-iters", type=int, default=500)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--hflip", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=20260929)
    ap.add_argument("--subset", type=int, default=0, help="只用前 N 张 train (冒烟用)")
    ap.add_argument("--train-roles", default="FIT,CALIBRATION",
                    help="检测器训练角色; 默认 FIT+CALIBRATION=9000")
    ap.add_argument("--val-role", default="EARLY_STOP",
                    help="检测器选点角色; 默认 EARLY_STOP=1000。DEV2K 不参与选点, 留给 M11 评价")
    ap.add_argument("--val-every", type=int, default=2)
    ap.add_argument("--val-images", type=int, default=0, help="0 = 用整个 val 角色")
    ap.add_argument("--no-val", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "outputs" / "frcnn_r18fpn_road8"))
    ap.add_argument("--resume", default="")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "train_log.txt", "a", encoding="utf-8")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise SystemExit("CUDA 不可用 —— 这个训练脚本要求 GPU")
    log(f"device={torch.cuda.get_device_name(0)}  torch={torch.__version__}", logf)

    # 角色切分: 检测器只在训练角色上训, 在选点角色上挑 checkpoint。
    # DEV2K 完全不参与检测器训练/选点, 保证 M11 的评价集没被检测器选点污染。
    import csv

    role_of = {}
    with open(ROLECSV, "r", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            role_of[row["image_id"]] = row["role"]
    train_roles = {r.strip() for r in args.train_roles.split(",") if r.strip()}

    full_train = D.build(RELEASE / "gt" / "TRAIN10K_ROAD8_GT.json", IMAGE_ROOT, train=True, hflip=args.hflip)
    tr_idx = [i for i, rec in enumerate(full_train.images) if role_of.get(rec["id"]) in train_roles]
    tr_idx.sort(key=lambda i: full_train.images[i]["id"])
    train_ds = Subset(full_train, tr_idx)
    if args.subset:
        train_ds = Subset(full_train, tr_idx[: args.subset])

    val_ds = None
    val_ids: list[str] = []
    if not args.no_val:
        full_val = D.build(RELEASE / "gt" / "TRAIN10K_ROAD8_GT.json", IMAGE_ROOT, train=False)
        va_idx = [i for i, rec in enumerate(full_val.images) if role_of.get(rec["id"]) == args.val_role]
        va_idx.sort(key=lambda i: full_val.images[i]["id"])
        if args.val_images:
            va_idx = va_idx[: args.val_images]
        val_ids = [full_val.images[i]["id"] for i in va_idx]
        val_ds = Subset(full_val, va_idx)
    overlap = set(full_train.images[i]["id"] for i in tr_idx) & set(val_ids)
    if overlap:
        raise SystemExit(f"train/val role overlap: {len(overlap)} images")
    log(f"roles: train={sorted(train_roles)} n={len(train_ds)}  val={args.val_role} n={len(val_ids)}  "
        f"disjoint=True", logf)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
        collate_fn=D.collate, drop_last=True, persistent_workers=args.workers > 0, pin_memory=True,
    )
    log(f"train images={len(train_ds)}  val images={len(val_ds) if val_ds else 0}  "
        f"iters/epoch={len(train_loader)}", logf)

    model = M.build(args.backbone).to(device)
    lr = args.lr if args.lr is not None else 0.02 * args.batch_size / 16.0
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    total_iters = max(1, args.epochs * len(train_loader))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda it: min(1.0, (it + 1) / max(1, args.warmup_iters)) * 0.5 * (1 + math.cos(math.pi * min(1.0, it / total_iters)))
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    start_epoch = 0
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        start_epoch = ck["epoch"] + 1
        log(f"resumed from {args.resume} at epoch {start_epoch}", logf)

    decl = M.declaration(args.backbone)
    (out / "detector_declaration.json").write_text(
        json.dumps({**decl, **M.summarise(model), "train_args": vars(args)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    history = []
    best = -1.0
    global_it = start_epoch * len(train_loader)
    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0 = time.time()
        acc = {}
        for step, (images, targets) in enumerate(train_loader):
            images = [im.to(device, non_blocking=True) for im in images]
            targets = [{k: v.to(device) if torch.is_tensor(v) else v for k, v in t.items()} for t in targets]
            with torch.autocast("cuda", dtype=torch.float16, enabled=True):
                loss_dict = model(images, targets)
                loss = sum(loss_dict.values())
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 10.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            global_it += 1
            for k, v in loss_dict.items():
                acc[k] = acc.get(k, 0.0) + float(v.detach())
            if step % 100 == 0:
                msg = " ".join(f"{k}={v / (step + 1):.4f}" for k, v in sorted(acc.items()))
                log(f"ep{epoch} it{step}/{len(train_loader)} lr={opt.param_groups[0]['lr']:.5f} {msg}", logf)
            if args.subset and step >= 60:
                break

        n = max(1, (step + 1))
        means = {k: v / n for k, v in sorted(acc.items())}
        rec = {"epoch": epoch, "secs": round(time.time() - t0, 1), "lr": opt.param_groups[0]["lr"], **means}
        log(f"== epoch {epoch} done in {rec['secs']}s  " + " ".join(f"{k}={v:.4f}" for k, v in means.items()), logf)

        if val_ds is not None and ((epoch + 1) % args.val_every == 0 or epoch == args.epochs - 1):
            vl = DataLoader(val_ds, batch_size=2, shuffle=False, num_workers=2, collate_fn=D.collate)
            f1 = evaluate_ap(model, vl, device, args.val_images)
            rec["val_f1_50"] = f1
            log(f"   val mAP@0.5(贪心近似) = {f1:.4f}", logf)
            if f1 > best:
                best = f1
                torch.save({"model": model.state_dict(), "epoch": epoch, "backbone": args.backbone,
                            "declaration": decl, "val_f1_50": f1}, out / "best.pth")
                log(f"   -> saved best.pth ({f1:.4f})", logf)

        torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                    "epoch": epoch, "backbone": args.backbone, "declaration": decl, "val_f1_50": best},
                   out / "last.pth")
        history.append(rec)
        (out / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=1), encoding="utf-8")

    log(f"DONE. best val mAP@0.5 = {best:.4f}", logf)
    logf.close()


if __name__ == "__main__":
    main()
