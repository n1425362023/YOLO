#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""模型评估: mAP50 / mAP50-95 / 精确率 / 召回率 / 每类 AP。

用法:
    python scripts/validate.py --weights runs/xxx/weights/best.pt --data data/processed/dior/data.yaml
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rsdet import paths
from rsdet.train_service import evaluate


def main() -> None:
    ap = argparse.ArgumentParser(description="评估目标检测模型")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--csv", default=None, help="输出 CSV (可选)")
    args = ap.parse_args()

    res = evaluate(args.weights, args.data, imgsz=args.imgsz, device=args.device)

    print("\n评估结果:")
    print(f"  mAP50    : {res['mAP50']:.4f}")
    print(f"  mAP50-95 : {res['mAP50_95']:.4f}")
    print(f"  精确率 P : {res['precision']:.4f}")
    print(f"  召回率 R : {res['recall']:.4f}")
    print("  每类 AP50:")
    for cls, ap in res["per_class_ap50"].items():
        print(f"    {cls:<16} : {ap:.4f}")

    if args.csv:
        out = Path(args.csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["metric", "value"])
            w.writerow(["mAP50", f"{res['mAP50']:.6f}"])
            w.writerow(["mAP50_95", f"{res['mAP50_95']:.6f}"])
            w.writerow(["precision", f"{res['precision']:.6f}"])
            w.writerow(["recall", f"{res['recall']:.6f}"])
            for cls, ap in res["per_class_ap50"].items():
                w.writerow([f"AP50/{cls}", f"{ap:.6f}"])
        print(f"\nCSV 已保存: {out}")


if __name__ == "__main__":
    main()
