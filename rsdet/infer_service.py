# -*- coding: utf-8 -*-
"""推理服务: 单图 / 大图切片推理 (切片 + 合并 + NMS)。"""
from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from .slicer import compute_tiles, merge_detections

# 类别配色 (BGR)
_PALETTE = [
    (60, 180, 75), (255, 100, 100), (100, 160, 255), (255, 200, 40),
    (200, 100, 255), (255, 255, 80), (80, 220, 220), (220, 80, 180),
    (160, 160, 160), (40, 40, 220),
]


def _color_for(cls: int) -> tuple[int, int, int]:
    return _PALETTE[cls % len(_PALETTE)]


def load_model(weights: str | Path, device: str = "auto"):
    from ultralytics import YOLO
    return YOLO(str(weights))


def _model_imgsz(model) -> int:
    """读取模型训练时的输入尺寸; 训练/推理分辨率不一致会显著掉精度。"""
    try:
        return int(model.model.args.get("imgsz", 640) or 640)
    except Exception:
        return 640


def _run_tile(model, tile_img: np.ndarray, conf: float, iou: float,
              x0: int, y0: int, imgsz: int | None = None) -> dict:
    """对单一切片推理, 返回该切片检测 (全局坐标)。"""
    kwargs: dict = dict(source=tile_img, conf=conf, iou=iou, verbose=False)
    if imgsz:
        kwargs["imgsz"] = imgsz
    res = model.predict(**kwargs)[0]
    boxes = res.boxes
    if boxes is None or len(boxes) == 0:
        return {"boxes_global": np.empty((0, 4)), "cls": np.empty(0, int),
                "conf": np.empty(0, float)}
    xyxy = boxes.xyxy.cpu().numpy()
    xyxy[:, [0, 2]] += x0
    xyxy[:, [1, 3]] += y0
    cls = boxes.cls.cpu().numpy().astype(int)
    conf = boxes.conf.cpu().numpy().astype(float)
    return {"boxes_global": xyxy, "cls": cls, "conf": conf}


def detect(model, image_bgr: np.ndarray, names: list[str], conf: float = 0.25,
           iou: float = 0.45, tile_size: int = 640, overlap: float = 0.25,
           force_slice: bool = False, imgsz: int | None = None) -> dict:
    """目标检测 (自动决定是否切片)。

    imgsz: 推理输入尺寸; 默认取模型训练时的 imgsz, 保证与训练一致。

    返回 dict:
        detections: [{cls, conf, bbox:[x0,y0,x1,y1]}]
        annotated:  标注后的 BGR 图像
        elapsed_ms: 推理耗时
        num_tiles:  切片数量
    """
    if imgsz is None:
        imgsz = _model_imgsz(model)
    h, w = image_bgr.shape[:2]
    t0 = time.perf_counter()
    needs_slice = force_slice or (max(h, w) > tile_size * 1.25)

    if not needs_slice:
        res = model.predict(source=image_bgr, conf=conf, iou=iou, verbose=False,
                            imgsz=imgsz)[0]
        boxes = res.boxes
        detections = []
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            cls = boxes.cls.cpu().numpy().astype(int)
            confs = boxes.conf.cpu().numpy().astype(float)
            for b, c, cf in zip(xyxy, cls, confs):
                detections.append({"cls": int(c), "conf": float(cf),
                                   "bbox": [float(v) for v in b]})
        num_tiles = 1
    else:
        tile_results = []
        tiles = compute_tiles(w, h, tile_size, overlap)
        for (x0, y0, x1, y1) in tiles:
            tile_img = image_bgr[y0:y1, x0:x1]
            tile_results.append(_run_tile(model, tile_img, conf, iou, x0, y0,
                                          imgsz=imgsz))
        detections = merge_detections(tile_results, iou_thr=iou)
        num_tiles = len(tiles)

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    annotated = draw_detections(image_bgr, detections, names)
    return {"detections": detections, "annotated": annotated,
            "elapsed_ms": elapsed_ms, "num_tiles": num_tiles,
            "image_shape": [w, h]}


def detect_file(model, image_path: str | Path, names: list[str], **kwargs) -> dict:
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {image_path}")
    return detect(model, img, names, **kwargs)


def draw_detections(image_bgr: np.ndarray, detections: list[dict],
                    names: list[str]) -> np.ndarray:
    """在原图副本上绘制检测框 + 标签。"""
    out = image_bgr.copy()
    for d in detections:
        x0, y0, x1, y1 = (int(v) for v in d["bbox"])
        cls = int(d["cls"])
        conf = d["conf"]
        color = _color_for(cls)
        cv2.rectangle(out, (x0, y0), (x1, y1), color, 2)
        label = f"{names[cls]} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x0, max(0, y0 - th - 4)), (x0 + tw, y0), color, -1)
        cv2.putText(out, label, (x0, max(0, y0 - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return out
