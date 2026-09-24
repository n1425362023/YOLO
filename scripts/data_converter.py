#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""标注格式转换: DOTA 旋转框 / COCO / VOC -> YOLO 水平框。

用法:
    # DOTA -> YOLO (需提供图片目录以读取尺寸)
    python scripts/data_converter.py --src dota --dota data/raw/DOTA/labelTxt \
        --images data/raw/DOTA/images --out data/processed/labels --classes plane ship vehicle storage-tank

    # COCO -> YOLO
    python scripts/data_converter.py --src coco --coco annotations.json --out data/processed/labels --classes ...

    # VOC -> YOLO
    python scripts/data_converter.py --src voc --xml data/raw/NWPU/Annotations \
        --out data/processed/labels --classes ...
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rsdet import paths
from rsdet.converter import (build_class_map, coco_to_yolo, convert_dota_dir,
                             voc_to_yolo, write_classes_txt)


def main() -> None:
    ap = argparse.ArgumentParser(description="标注格式转换 -> YOLO")
    ap.add_argument("--src", required=True, choices=["dota", "coco", "voc"])
    ap.add_argument("--classes", nargs="+", required=True, help="类别名 (顺序即类 id)")
    ap.add_argument("--out", required=True, help="YOLO 标注输出目录")
    # DOTA
    ap.add_argument("--dota", help="DOTA labelTxt 目录")
    ap.add_argument("--images", help="图片目录 (读取尺寸)")
    # COCO
    ap.add_argument("--coco", help="COCO JSON 文件")
    # VOC
    ap.add_argument("--xml", help="VOC XML 目录")
    ap.add_argument("--classes-out", default=None, help="额外输出 classes.txt 的路径")
    args = ap.parse_args()

    class_map = build_class_map(args.classes)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.src == "dota":
        if not args.dota or not args.images:
            ap.error("--src dota 需要 --dota 与 --images")
        stats = convert_dota_dir(args.dota, args.images, out, class_map)
    elif args.src == "coco":
        if not args.coco:
            ap.error("--src coco 需要 --coco")
        stats = coco_to_yolo(args.coco, Path(args.coco).parent, out, class_map)
    else:
        if not args.xml:
            ap.error("--src voc 需要 --xml")
        stats = voc_to_yolo(args.xml, out, class_map)

    print(f"转换完成: 图片 {stats.get('images')} | 框 {stats.get('boxes')}")
    if "skipped_no_label" in stats:
        print(f"  空标注图片: {stats['skipped_no_label']}")

    classes_out = Path(args.classes_out) if args.classes_out else out.parent / "classes.txt"
    write_classes_txt(args.classes, classes_out)
    print(f"类别文件: {classes_out}")


if __name__ == "__main__":
    main()
