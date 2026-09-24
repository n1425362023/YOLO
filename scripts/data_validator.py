#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""数据校验 + HTML 报告。

用法:
    python scripts/data_validator.py --images data/processed/train/images \
        --labels data/processed/train/labels --classes plane ship vehicle storage-tank \
        --report reports/validation.html
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rsdet import paths
from rsdet.report import class_dist_plot, write_html_report
from rsdet.validator import summarize, validate_dataset


def main() -> None:
    ap = argparse.ArgumentParser(description="数据集校验")
    ap.add_argument("--images", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--classes", nargs="+", required=True)
    ap.add_argument("--report", default=None, help="HTML 报告输出路径")
    args = ap.parse_args()

    report = validate_dataset(args.images, args.labels, args.classes)
    print(summarize(report, args.classes))

    if args.report:
        fig = class_dist_plot(dict(report["class_dist"]), args.classes)
        rows = ["<table><tr><th>指标</th><th>值</th></tr>",
                f"<tr><td>图片总数</td><td>{report['num_images']}</td></tr>",
                f"<tr><td>含标注图片</td><td>{report['images_with_labels']}</td></tr>",
                f"<tr><td>标注框总数</td><td>{report['num_boxes']}</td></tr>",
                f"<tr><td>校验结果</td><td>{'通过' if report['ok'] else '异常'}</td></tr>",
                "</table>"]
        if report["issues"]:
            rows.append("<h3>异常项</h3><ul>" + "".join(
                f"<li>{i}</li>" for i in report["issues"][:100]) + "</ul>")
        out = write_html_report(args.report, "数据集校验报告",
                                [("校验结果", "".join(rows))],
                                [("类别分布", fig)])
        print(f"报告已生成: {out}")


if __name__ == "__main__":
    main()
