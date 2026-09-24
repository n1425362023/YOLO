#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从本机 DIOR 数据集构建 YOLO 检测格式数据 (官方划分 + 真实水平框 + 抽样子集)。

用法:
    python scripts/prepare_dior.py
    python scripts/prepare_dior.py --root <DIOR根> --out data/processed/dior \
        --sample-train-per-class 16 --sample-val-per-class 8 --seed 42
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rsdet import paths
from rsdet.dior_source import (discover_classes, prepare_dior_dataset,
                               resolve_dior_root)


def main() -> None:
    ap = argparse.ArgumentParser(description="从 DIOR 数据集生成 YOLO 检测格式数据")
    ap.add_argument("--root", default=None,
                    help="DIOR 根目录 (默认: config[dataset.dior.root], 再回退 paths.DIOR_ROOT)")
    ap.add_argument("--out", default=None, help="输出目录 (默认 data/processed/dior)")
    ap.add_argument("--classes", nargs="+", default=None,
                    help="类别白名单 (默认自动发现, 官方序)")
    ap.add_argument("--sample-train-per-class", type=int, default=None,
                    help="抽样子集: 每类从 train 取几张")
    ap.add_argument("--sample-val-per-class", type=int, default=None,
                    help="抽样子集: 每类从 val 取几张")
    ap.add_argument("--seed", type=int, default=None, help="抽样随机种子")
    ap.add_argument("--dry-run", action="store_true", help="仅打印计划, 不写文件")
    args = ap.parse_args()

    from rsdet.config import get_config
    cfg = get_config()
    out = Path(args.out) if args.out else paths.DATA_PROCESSED / "dior"
    st = args.sample_train_per_class
    if st is None:
        st = int(cfg.get("dataset.dior.sample_train_per_class", 16))
    sv = args.sample_val_per_class
    if sv is None:
        sv = int(cfg.get("dataset.dior.sample_val_per_class", 8))
    seed = args.seed if args.seed is not None else int(cfg.get("dataset.dior.seed", 42))

    root = resolve_dior_root(args.root)
    classes = args.classes or discover_classes(root)
    print(f"DIOR 数据源 : {root}")
    print(f"类别数     : {len(classes)}")
    print(f"输出目录   : {out}")
    print(f"sample train/val per class: {st}/{sv} | seed={seed}")
    if args.dry_run:
        print("(--dry-run) 未写入任何文件。")
        return

    stats = prepare_dior_dataset(root, out, classes=classes,
                                 sample_train_per_class=st,
                                 sample_val_per_class=sv, seed=seed)
    print("=" * 60)
    print("数据准备完成:")
    print(f"  train/val/test 图片: {stats['split_counts']}")
    print(f"  真实框总数         : {stats['boxes_total']}")
    print(f"  sample train 图片  : {sum(stats['sample_counts']['train'].values())}")
    print(f"  sample val 图片    : {sum(stats['sample_counts']['val'].values())}")
    print(f"  全量 data.yaml     : {stats['data_yaml_full']}")
    print(f"  抽样 data.yaml     : {stats['data_yaml_sample']}")


if __name__ == "__main__":
    main()