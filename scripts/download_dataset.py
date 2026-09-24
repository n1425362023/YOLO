#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""自动下载权威遥感数据集 (DOTA / DIOR / NWPU-VHR-10 / HRSC2016)。

用法:
    python scripts/download_dataset.py --list
    python scripts/download_dataset.py --dataset NWPU-VHR-10
    python scripts/download_dataset.py --dataset DOTA --out data/raw
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rsdet import paths
from rsdet.dataset_sources import DATASETS, download_dataset, list_datasets


def main() -> None:
    ap = argparse.ArgumentParser(description="下载权威遥感目标检测数据集")
    ap.add_argument("--dataset", help="数据集名 (DOTA/DIOR/NWPU-VHR-10/HRSC2016)")
    ap.add_argument("--list", action="store_true", help="列出可用数据集")
    ap.add_argument("--out", default=None, help="输出目录 (默认 data/raw)")
    args = ap.parse_args()

    if args.list:
        print("可用权威遥感数据集:")
        print(list_datasets())
        return

    if not args.dataset:
        ap.print_help()
        print("\n可用: " + ", ".join(DATASETS))
        return

    out = Path(args.out) if args.out else paths.DATA_RAW
    download_dataset(args.dataset, out)


if __name__ == "__main__":
    main()
