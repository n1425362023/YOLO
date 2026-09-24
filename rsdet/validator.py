# -*- coding: utf-8 -*-
"""数据校验: 图像-标注成对、类别/越界/NaN 检查、类别分布统计。"""
from __future__ import annotations

import math
from collections import Counter
from pathlib import Path

import numpy as np

from .slicer import _parse_yolo_line

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def _read_label_boxes(label_file: Path) -> list[tuple[int, float, float, float, float]]:
    boxes = []
    for line in label_file.read_text(encoding="utf-8").splitlines():
        parsed = _parse_yolo_line(line)
        if parsed:
            boxes.append(parsed)
    return boxes


def validate_dataset(image_dir: str | Path, label_dir: str | Path,
                     classes: list[str]) -> dict:
    """校验一个 YOLO 数据集目录, 返回报告 dict。

    报告字段: num_images, num_boxes, images_with_labels, images_empty,
              class_dist, issues(列表), ok(布尔)
    """
    image_dir, label_dir = Path(image_dir), Path(label_dir)
    n_cls = len(classes)
    images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)

    report = {
        "num_images": len(images),
        "num_boxes": 0,
        "images_with_labels": 0,
        "images_empty": 0,
        "class_dist": Counter(),
        "issues": [],
        "ok": True,
    }

    for img in images:
        label_file = label_dir / (img.stem + ".txt")
        if not label_file.exists():
            report["issues"].append(f"{img.name}: 缺少标注文件")
            report["images_empty"] += 1
            report["ok"] = False
            continue
        boxes = _read_label_boxes(label_file)
        if not boxes:
            report["images_empty"] += 1
            continue
        report["images_with_labels"] += 1
        for (cls, cx, cy, w, h) in boxes:
            report["num_boxes"] += 1
            if not (0 <= cls < n_cls):
                report["issues"].append(f"{img.name}: 非法类别 id={cls}")
                report["ok"] = False
                continue
            report["class_dist"][cls] += 1
            for name, v in (("cx", cx), ("cy", cy), ("w", w), ("h", h)):
                if not math.isfinite(v):
                    report["issues"].append(f"{img.name}: {name}=NaN/Inf")
                    report["ok"] = False
            if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0 and 0.0 <= w <= 1.0 and 0.0 <= h <= 1.0):
                report["issues"].append(f"{img.name}: 越界框 ({cx:.2f},{cy:.2f},{w:.2f},{h:.2f})")
                report["ok"] = False
            if w <= 0 or h <= 0:
                report["issues"].append(f"{img.name}: 退化框 (w={w}, h={h})")
                report["ok"] = False

    # 归一化类别分布
    total = sum(report["class_dist"].values()) or 1
    report["class_dist_norm"] = {
        classes[c]: report["class_dist"].get(c, 0) / total for c in range(n_cls)
    }
    return report


def summarize(report: dict, classes: list[str]) -> str:
    """把校验报告格式化为可打印文本。"""
    lines = [
        f"图片总数        : {report['num_images']}",
        f"含标注图片      : {report['images_with_labels']}",
        f"空标注图片      : {report['images_empty']}",
        f"标注框总数      : {report['num_boxes']}",
        f"校验结果        : {'通过' if report['ok'] else '存在异常'}",
        "类别分布:",
    ]
    for c in range(len(classes)):
        n = report["class_dist"].get(c, 0)
        lines.append(f"  {c:>2} {classes[c]:<16} : {n}")
    if report["issues"]:
        lines.append(f"异常项 ({len(report['issues'])}):")
        for iss in report["issues"][:50]:
            lines.append(f"  - {iss}")
    return "\n".join(lines)


def class_distribution_array(report: dict, n_cls: int) -> np.ndarray:
    return np.array([report["class_dist"].get(c, 0) for c in range(n_cls)], dtype=int)
