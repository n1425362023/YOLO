#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""大图切片 + 标注坐标校正 (训练数据预处理)。

用法:
    python scripts/data_slicer.py --images data/processed/train/images --labels data/processed/train/labels \
        --out-images data/processed/tiles/images --out-labels data/processed/tiles/labels \
        --tile-size 640 --overlap 0.25
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rsdet.slicer import slice_dataset


def main() -> None:
    ap = argparse.ArgumentParser(description="大图切片 + 标注坐标校正")
    ap.add_argument("--images", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out-images", required=True)
    ap.add_argument("--out-labels", required=True)
    ap.add_argument("--tile-size", type=int, default=640)
    ap.add_argument("--overlap", type=float, default=0.25)
    args = ap.parse_args()

    stats = slice_dataset(args.images, args.labels, args.out_images, args.out_labels,
                          tile_size=args.tile_size, overlap=args.overlap)
    print(f"切片完成: 原图 {stats['images']} | 切片 {stats['tiles']} | 校正框 {stats['boxes']}")


if __name__ == "__main__":
    main()
