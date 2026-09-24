# -*- coding: utf-8 -*-
"""标注格式转换核心: DOTA 旋转框 / COCO / VOC -> YOLO 水平框。

DOTA 标注行格式 (labelTxt):
    x1 y1 x2 y2 x3 y3 x4 y4 category difficult
即 8 个角点坐标 (定向包围盒 4 顶点, 像素) + 类别名 + 难例标志。
YOLO 格式 (归一化水平框):
    class_id cx cy w h          (cx/cy 为中心, w/h 为宽高, 均归一化到 [0,1])
"""
from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable

# DOTA 官方 15 类 (供权威数据集转换时作参考默认)
DOTA_CLASSES = [
    "plane", "ship", "storage-tank", "baseball-diamond", "tennis-court",
    "basketball-court", "ground-track-field", "harbor", "bridge",
    "large-vehicle", "small-vehicle", "helicopter", "roundabout",
    "soccer-ball-field", "swimming-pool",
]


def build_class_map(classes: Iterable[str]) -> dict[str, int]:
    """类别名 -> id 映射。"""
    return {name: i for i, name in enumerate(classes)}


def write_classes_txt(classes: Iterable[str], out_path: str | Path) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text("\n".join(classes) + "\n", encoding="utf-8")


def rotated_bbox_to_xyxy(points: list[float]) -> tuple[float, float, float, float]:
    """8 个坐标 (4 顶点) -> 轴对齐外接框 (xmin, ymin, xmax, ymax)。"""
    xs = points[0::2]
    ys = points[1::2]
    return min(xs), min(ys), max(xs), max(ys)


def _clip01(v: float) -> float:
    return max(0.0, min(1.0, v))


def xyxy_to_yolo(xmin: float, ymin: float, xmax: float, ymax: float,
                 img_w: int, img_h: int) -> tuple[float, float, float, float]:
    """像素框 -> 归一化 YOLO [cx, cy, w, h] (越界裁剪)。"""
    xmin, ymin = max(0.0, xmin), max(0.0, ymin)
    xmax, ymax = min(float(img_w), xmax), min(float(img_h), ymax)
    if xmax <= xmin or ymax <= ymin:
        return 0.0, 0.0, 0.0, 0.0  # 退化框
    cx = (xmin + xmax) / 2.0 / img_w
    cy = (ymin + ymax) / 2.0 / img_h
    w = (xmax - xmin) / img_w
    h = (ymax - ymin) / img_h
    return (_clip01(cx), _clip01(cy), _clip01(w), _clip01(h))


def dota_line_to_yolo(line: str, img_w: int, img_h: int,
                      class_map: dict[str, int]) -> str | None:
    """单行 DOTA 标注 -> YOLO 标签行字符串; 无效行返回 None。"""
    parts = line.split()
    if len(parts) < 9:
        return None
    try:
        coords = [float(x) for x in parts[:8]]
    except ValueError:
        return None  # 跳过 imagesource / gsd 等元信息行
    category = parts[8]
    if category not in class_map:
        return None
    cls_id = class_map[category]
    xmin, ymin, xmax, ymax = rotated_bbox_to_xyxy(coords)
    cx, cy, w, h = xyxy_to_yolo(xmin, ymin, xmax, ymax, img_w, img_h)
    if w <= 0 or h <= 0:
        return None
    return f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def dota_to_yolo(dota_file: str | Path, img_w: int, img_h: int,
                 class_map: dict[str, int], out_file: str | Path) -> int:
    """转换单个 DOTA labelTxt -> YOLO txt, 返回写入的框数量。"""
    dota_file = Path(dota_file)
    lines_out: list[str] = []
    if dota_file.exists():
        for line in dota_file.read_text(encoding="utf-8", errors="ignore").splitlines():
            yolo_line = dota_line_to_yolo(line, img_w, img_h, class_map)
            if yolo_line:
                lines_out.append(yolo_line)
    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    Path(out_file).write_text("\n".join(lines_out) + ("\n" if lines_out else ""),
                               encoding="utf-8")
    return len(lines_out)


def convert_dota_dir(dota_dir: str | Path, image_dir: str | Path,
                     label_out_dir: str | Path, class_map: dict[str, int],
                     image_ext: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"),
                     ) -> dict:
    """批量转换 DOTA 目录: 为每张图在同名 labelTxt 处生成 YOLO 标注。

    需要图片尺寸, 故用 PIL 读取每张图宽高。
    返回统计 dict。
    """
    from PIL import Image

    dota_dir, image_dir = Path(dota_dir), Path(image_dir)
    label_out_dir = Path(label_out_dir)
    label_out_dir.mkdir(parents=True, exist_ok=True)

    stats = {"images": 0, "boxes": 0, "skipped_no_label": 0}
    img_files = sorted(
        p for p in image_dir.iterdir() if p.suffix.lower() in image_ext
    )
    for img in img_files:
        dota_file = dota_dir / (img.stem + ".txt")
        try:
            with Image.open(img) as im:
                w, h = im.size
        except Exception:
            continue
        out_file = label_out_dir / (img.stem + ".txt")
        n = dota_to_yolo(dota_file, w, h, class_map, out_file)
        stats["images"] += 1
        stats["boxes"] += n
        if n == 0:
            stats["skipped_no_label"] += 1
    return stats


def coco_to_yolo(coco_json: str | Path, image_dir: str | Path,
                 label_out_dir: str | Path, class_map: dict[str, int]) -> dict:
    """COCO JSON 标注 -> YOLO txt (按 category_id 映射为 class_map 的值)。"""
    data = json.loads(Path(coco_json).read_text(encoding="utf-8"))
    id2name = {c["id"]: c["name"] for c in data["categories"]}
    # 类别名 -> 类 id (class_map 中值即 YOLO 类 id)
    id2cls = {cid: class_map[name] for cid, name in id2name.items() if name in class_map}

    img_meta = {im["id"]: im for im in data["images"]}
    ann_by_img: dict[int, list] = {}
    for ann in data["annotations"]:
        if ann["category_id"] not in id2cls:
            continue
        ann_by_img.setdefault(ann["image_id"], []).append(ann)

    label_out_dir = Path(label_out_dir)
    label_out_dir.mkdir(parents=True, exist_ok=True)
    stats = {"images": 0, "boxes": 0}
    for img_id, im in img_meta.items():
        w, h = im["width"], im["height"]
        stem = Path(im["file_name"]).stem
        lines = []
        for ann in ann_by_img.get(img_id, []):
            x, y, bw, bh = ann["bbox"]  # COCO: 左上角 + 宽高
            cx, cy, nw, nh = xyxy_to_yolo(x, y, x + bw, y + bh, w, h)
            if nw <= 0 or nh <= 0:
                continue
            lines.append(f"{id2cls[ann['category_id']]} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
        (label_out_dir / (stem + ".txt")).write_text("\n".join(lines) + ("\n" if lines else ""),
                                                      encoding="utf-8")
        stats["images"] += 1
        stats["boxes"] += len(lines)
    return stats


def voc_to_yolo(xml_dir: str | Path, label_out_dir: str | Path,
                class_map: dict[str, int]) -> dict:
    """Pascal VOC XML 目录 -> YOLO txt。"""
    xml_dir = Path(xml_dir)
    label_out_dir = Path(label_out_dir)
    label_out_dir.mkdir(parents=True, exist_ok=True)
    stats = {"images": 0, "boxes": 0}
    for xml_file in sorted(xml_dir.glob("*.xml")):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        size = root.find("size")
        w = int(float(size.find("width").text))
        h = int(float(size.find("height").text))
        lines = []
        for obj in root.findall("object"):
            name = obj.find("name").text
            if name not in class_map:
                continue
            bnd = obj.find("bndbox")
            xmin = float(bnd.find("xmin").text)
            ymin = float(bnd.find("ymin").text)
            xmax = float(bnd.find("xmax").text)
            ymax = float(bnd.find("ymax").text)
            cx, cy, nw, nh = xyxy_to_yolo(xmin, ymin, xmax, ymax, w, h)
            if nw <= 0 or nh <= 0:
                continue
            lines.append(f"{class_map[name]} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
        (label_out_dir / (xml_file.stem + ".txt")).write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        stats["images"] += 1
        stats["boxes"] += len(lines)
    return stats


def _safe_box(cx: float, cy: float, w: float, h: float) -> tuple[float, float, float, float]:
    """防止数值溢出 / NaN 的兜底。"""
    for v in (cx, cy, w, h):
        if not math.isfinite(v):
            return 0.0, 0.0, 0.0, 0.0
    return _clip01(cx), _clip01(cy), _clip01(w), _clip01(h)
