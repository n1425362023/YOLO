#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""目标检测推理 (自动大图切片 + 合并 NMS)。

用法:
    python scripts/infer.py --weights runs/xxx/weights/best.pt --image data/processed/dior/val/images/05863.jpg
    python scripts/infer.py --weights runs/xxx/weights/best.pt --dir data/processed/dior/val/images --force-slice
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2

from rsdet import paths
from rsdet.infer_service import detect_file, load_model


def main() -> None:
    ap = argparse.ArgumentParser(description="遥感影像目标检测推理")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--image", default=None, help="单张图片")
    ap.add_argument("--dir", default=None, help="图片目录 (批量)")
    ap.add_argument("--conf", type=float, default=0.1)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--tile-size", type=int, default=640)
    ap.add_argument("--imgsz", type=int, default=None, help="推理输入尺寸 (默认取模型训练 imgsz)")
    ap.add_argument("--overlap", type=float, default=0.25)
    ap.add_argument("--force-slice", action="store_true", help="强制切片推理")
    ap.add_argument("--out", default=None, help="输出目录 (默认 reports/infer)")
    args = ap.parse_args()

    if not args.image and not args.dir:
        ap.error("需要 --image 或 --dir")

    out = Path(args.out) if args.out else paths.REPORTS_DIR / "infer"
    out.mkdir(parents=True, exist_ok=True)

    model = load_model(args.weights)
    names = [model.names[i] for i in range(len(model.names))]

    files = [Path(args.image)] if args.image else sorted(Path(args.dir).iterdir())
    files = [f for f in files if f.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")]

    summary = []
    for f in files:
        r = detect_file(model, f, names, conf=args.conf, iou=args.iou,
                        tile_size=args.tile_size, overlap=args.overlap,
                        force_slice=args.force_slice, imgsz=args.imgsz)
        ann_path = out / (f.stem + "_annotated.jpg")
        cv2.imwrite(str(ann_path), r["annotated"])
        json_path = out / (f.stem + ".json")
        json_path.write_text(json.dumps({
            "image": str(f), "detections": r["detections"],
            "elapsed_ms": r["elapsed_ms"], "num_tiles": r["num_tiles"],
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"{f.name}: {len(r['detections'])} 个目标 | 切片 {r['num_tiles']} | "
              f"{r['elapsed_ms']:.0f}ms -> {ann_path.name}")
        summary.append({"image": str(f), "count": len(r["detections"])})

    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
    print(f"\n共处理 {len(files)} 张, 结果保存至 {out}")


if __name__ == "__main__":
    main()
