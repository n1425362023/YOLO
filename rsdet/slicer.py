# -*- coding: utf-8 -*-
"""大图切片 + 标注坐标校正 + 合并 NMS。

遥感影像普遍为超大分辨率 (数千×数千像素), 直接送入检测器会因下采样丢失
小目标。本模块提供:
  * compute_tiles         —— 计算重叠切片网格
  * slice_dataset         —— 训练数据切片 + 标注坐标校正
  * merge_detections      —— 切片推理结果回全局坐标 + NMS 去重
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np


# --------------------------------------------------------------------------
# 切片网格
# --------------------------------------------------------------------------
def compute_tiles(w: int, h: int, tile_size: int,
                  overlap: float = 0.25) -> list[tuple[int, int, int, int]]:
    """返回 (x0, y0, x1, y1) 像素坐标的切片列表 (边缘切片自动回缩以保证覆盖)。"""
    stride = max(1, int(tile_size * (1.0 - overlap)))

    def starts(length: int) -> list[int]:
        if length <= tile_size:
            return [0]
        xs = list(range(0, length - tile_size + 1, stride))
        if xs[-1] != length - tile_size:
            xs.append(length - tile_size)
        return xs

    tiles = []
    for x0 in starts(w):
        x1 = min(x0 + tile_size, w)
        for y0 in starts(h):
            y1 = min(y0 + tile_size, h)
            tiles.append((x0, y0, x1, y1))
    return tiles


# --------------------------------------------------------------------------
# 训练数据切片
# --------------------------------------------------------------------------
def _parse_yolo_line(line: str) -> tuple[int, float, float, float, float] | None:
    parts = line.split()
    if len(parts) < 5:
        return None
    try:
        cls = int(float(parts[0]))
        cx, cy, w, h = (float(x) for x in parts[1:5])
    except ValueError:
        return None
    return cls, cx, cy, w, h


def slice_yolo_labels(lines: Iterable[str], tile: tuple[int, int, int, int],
                      img_w: int, img_h: int) -> list[str]:
    """把整图 YOLO 标注投影到切片: 保留中心点落在切片内的框, 校正为切片局部坐标。"""
    x0, y0, x1, y1 = tile
    tw, th = x1 - x0, y1 - y0
    out: list[str] = []
    for line in lines:
        parsed = _parse_yolo_line(line)
        if parsed is None:
            continue
        cls, cx, cy, w, h = parsed
        # 归一化 -> 整图像素
        ax = cx * img_w
        ay = cy * img_h
        aw = w * img_w
        ah = h * img_h
        # 中心点是否落在切片内
        if not (x0 <= ax <= x1 and y0 <= ay <= y1):
            continue
        # 转为切片局部绝对坐标并裁剪
        lx0 = max(0.0, ax - aw / 2 - x0)
        ly0 = max(0.0, ay - ah / 2 - y0)
        lx1 = min(float(tw), ax + aw / 2 - x0)
        ly1 = min(float(th), ay + ah / 2 - y0)
        if lx1 <= lx0 or ly1 <= ly0:
            continue
        ncx = (lx0 + lx1) / 2.0 / tw
        ncy = (ly0 + ly1) / 2.0 / th
        nw = (lx1 - lx0) / tw
        nh = (ly1 - ly0) / th
        out.append(f"{cls} {ncx:.6f} {ncy:.6f} {nw:.6f} {nh:.6f}")
    return out


def slice_dataset(image_dir: str | Path, label_dir: str | Path,
                  out_image_dir: str | Path, out_label_dir: str | Path,
                  tile_size: int = 640, overlap: float = 0.25) -> dict:
    """切片整个数据集 (含标注坐标校正)。

    图像命名: {原文件名}__{x0}_{y0}.jpg ; 标注同步生成。
    """
    import cv2

    image_dir, label_dir = Path(image_dir), Path(label_dir)
    out_image_dir, out_label_dir = Path(out_image_dir), Path(out_label_dir)
    out_image_dir.mkdir(parents=True, exist_ok=True)
    out_label_dir.mkdir(parents=True, exist_ok=True)

    stats = {"images": 0, "tiles": 0, "boxes": 0}
    for img in sorted(image_dir.iterdir()):
        if img.suffix.lower() not in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
            continue
        image = cv2.imread(str(img))
        if image is None:
            continue
        h, w = image.shape[:2]
        label_file = label_dir / (img.stem + ".txt")
        lines = label_file.read_text(encoding="utf-8").splitlines() if label_file.exists() else []
        for (x0, y0, x1, y1) in compute_tiles(w, h, tile_size, overlap):
            tile_img = image[y0:y1, x0:x1]
            tile_labels = slice_yolo_labels(lines, (x0, y0, x1, y1), w, h)
            tile_name = f"{img.stem}__{x0}_{y0}.jpg"
            cv2.imwrite(str(out_image_dir / tile_name), tile_img)
            (out_label_dir / (tile_name[:-4] + ".txt")).write_text(
                "\n".join(tile_labels) + ("\n" if tile_labels else ""), encoding="utf-8")
            stats["tiles"] += 1
            stats["boxes"] += len(tile_labels)
        stats["images"] += 1
    return stats


# --------------------------------------------------------------------------
# NMS / 合并
# --------------------------------------------------------------------------
def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> np.ndarray:
    """纯 numpy 的按分数降序 NMS, 返回保留的索引。boxes 为 (N,4) xyxy。"""
    if len(boxes) == 0:
        return np.array([], dtype=int)
    order = scores.argsort()[::-1].astype(int)
    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        iw = np.maximum(0.0, xx2 - xx1)
        ih = np.maximum(0.0, yy2 - yy1)
        inter = iw * ih
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        union = area[i] + area[rest] - inter
        iou = inter / np.maximum(union, 1e-9)
        order = rest[np.where(iou <= iou_thr)[0]]
    return np.array(keep, dtype=int)


def merge_detections(tile_results: Iterable[dict], iou_thr: float = 0.45) -> list[dict]:
    """合并各切片检测结果: 偏移回全局坐标后按类别 NMS。

    每个 tile_result 元素为 dict:
        boxes_global: (N,4) xyxy 全局坐标, cls: (N,), conf: (N,)
    返回统一列表 [{cls, conf, bbox(x0,y0,x1,y1)}]。
    """
    all_boxes: list[np.ndarray] = []
    all_cls: list[int] = []
    all_conf: list[float] = []
    for r in tile_results:
        boxes = np.asarray(r["boxes_global"], dtype=float)
        cls = np.asarray(r["cls"], dtype=int)
        conf = np.asarray(r["conf"], dtype=float)
        if len(boxes) == 0:
            continue
        all_boxes.append(boxes)
        all_cls.append(cls)
        all_conf.append(conf)

    if not all_boxes:
        return []

    boxes = np.concatenate(all_boxes, axis=0)
    cls = np.concatenate(all_cls, axis=0)
    conf = np.concatenate(all_conf, axis=0)

    keep_all: list[int] = []
    for c in np.unique(cls):
        idx = np.where(cls == c)[0]
        keep = nms(boxes[idx], conf[idx], iou_thr)
        keep_all.extend(idx[keep].tolist())

    out = []
    for i in keep_all:
        x0, y0, x1, y1 = boxes[i]
        out.append({
            "cls": int(cls[i]),
            "conf": float(conf[i]),
            "bbox": [float(x0), float(y0), float(x1), float(y1)],
        })
    return out
