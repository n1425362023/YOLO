#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""模型训练 (YOLOv8/YOLO11 迁移学习 + 类别平衡 + 实验归档)。

用法:
    python scripts/train.py --data data/processed/dior/data.yaml --model yolov8n \
        --epochs 5 --imgsz 320 --batch 8 --cls-pw 1.0
    python scripts/train.py --data data/processed/dior/data.yaml --model yolov8s --epochs 50 --device 0
    python scripts/train.py --data data/processed/dior/data.yaml --model yolov8n --resume
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rsdet import paths
from rsdet.config import get_config
from rsdet.train_service import train


def main() -> None:
    cfg = get_config()
    ap = argparse.ArgumentParser(description="训练 YOLOv8/YOLO11 目标检测模型")
    ap.add_argument("--data", required=True, help="data.yaml 路径")
    ap.add_argument("--model", default=None, help="yolov8n/yolov8s/yolo11n")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--device", default=None, help="auto/cpu/0")
    ap.add_argument("--cls-pw", type=float, default=None, help="类别权重幂次 (0=禁用)")
    ap.add_argument("--no-pretrained", action="store_true", help="不从 COCO 预训练, 从头训练")
    ap.add_argument("--resume", action="store_true", help="断点续训")
    ap.add_argument("--name", default=None, help="实验名")
    args = ap.parse_args()

    name = args.name or f"{args.model or cfg.get('model.name', 'yolov8n')}_{time.strftime('%Y%m%d_%H%M%S')}"
    out_run_dir = paths.RUNS_DIR / name

    pretrained = False if args.no_pretrained else None

    summary = train(cfg, data_yaml=args.data, out_run_dir=out_run_dir,
                    model_name=args.model, epochs=args.epochs, imgsz=args.imgsz,
                    batch=args.batch, device=args.device, cls_pw=args.cls_pw,
                    pretrained=pretrained, resume=args.resume)

    print("\n训练摘要:")
    print(f"  模型     : {summary['model']}")
    print(f"  mAP50    : {summary['mAP50']:.4f}")
    print(f"  mAP50-95 : {summary['mAP50_95']:.4f}")
    print(f"  精确率   : {summary['precision']:.4f}")
    print(f"  召回率   : {summary['recall']:.4f}")
    print(f"  权重     : {summary['best_weight']}")


if __name__ == "__main__":
    main()
