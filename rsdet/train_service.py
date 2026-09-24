# -*- coding: utf-8 -*-
"""训练服务: 封装 ultralytics YOLO 训练 (迁移学习 + 类别平衡 + 实验归档)。"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

from . import system_utils
from .logging_utils import get_logger

log = get_logger("rsdet.train")


def _weights_cached(model_name: str) -> bool:
    """判断预训练权重是否已在本机缓存 (避免离线时反复尝试下载)。"""
    candidates = [Path(f"{model_name}.pt")]
    try:
        from ultralytics.utils import SETTINGS
        candidates.append(Path(SETTINGS.get("weights_dir", ".")) / f"{model_name}.pt")
    except Exception:
        pass
    return any(p.exists() for p in candidates)


def _host_reachable(host: str = "github.com", port: int = 443, timeout: float = 3.0) -> bool:
    import socket
    try:
        socket.create_connection((host, port), timeout=timeout)
        return True
    except Exception:
        return False


def evaluate(weights: str | Path, data_yaml: str | Path, imgsz: int = 320,
             device: str = "auto") -> dict:
    """在验证集上评估权重, 返回 mAP50 / mAP50-95 / P / R / 每类 AP。"""
    from ultralytics import YOLO
    model = YOLO(str(weights))
    device = system_utils.normalize_device(device)
    metrics = model.val(data=str(data_yaml), imgsz=imgsz, device=device, verbose=False)
    box = getattr(metrics, "box", metrics)
    per_class = {}
    ap = getattr(box, "ap", None)
    if ap is None:
        ap = []
    names = getattr(metrics, "names", {}) or {}
    for i, v in enumerate(ap):
        per_class[names.get(i, i)] = float(v)
    return {
        "mAP50": float(getattr(box, "map50", 0.0)),
        "mAP50_95": float(getattr(box, "map", 0.0)),
        "precision": float(getattr(box, "mp", 0.0)),
        "recall": float(getattr(box, "mr", 0.0)),
        "per_class_ap50": per_class,
    }


def write_data_yaml(out_path: str | Path, train_dir: str | Path,
                    val_dir: str | Path, classes: list[str]) -> Path:
    """生成 ultralytics 训练用 data.yaml。

    train/val 路径相对其共同父目录写出 (例如 path=processed, train=train/images,
    val=val/images), 避免两个划分都落到同一目录。
    """
    import os

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    train_dir = Path(train_dir).resolve()
    val_dir = Path(val_dir).resolve()
    common = Path(os.path.commonpath([str(train_dir), str(val_dir)]))
    content = {
        "path": str(common),
        "train": str(train_dir.relative_to(common)).replace("\\", "/"),
        "val": str(val_dir.relative_to(common)).replace("\\", "/"),
        "names": {i: c for i, c in enumerate(classes)},
    }
    import yaml
    out_path.write_text(yaml.safe_dump(content, allow_unicode=True, sort_keys=False),
                        encoding="utf-8")
    return out_path


def train(cfg: Any, data_yaml: str | Path, out_run_dir: str | Path,
          model_name: str | None = None, epochs: int | None = None,
          imgsz: int | None = None, batch: int | None = None,
          device: str | None = None, cls_pw: float | None = None,
          pretrained: bool | None = None,
          resume: bool = False, **extra) -> dict:
    """执行训练, 归档 best.pt / 配置 / 指标, 返回结果 dict。

    cls_pw: 类别权重幂次 (0.0=禁用, 1.0=按类别频率反比加权), 用于缓解类别不平衡
            (旧版 ultralytics 的 fl_gamma focal loss 已移除, 现由 cls_pw 承担)。
    """
    from ultralytics import YOLO

    model_name = model_name or cfg.get("model.name", "yolov8n")
    epochs = epochs if epochs is not None else cfg.get("model.epochs", 5)
    imgsz = imgsz if imgsz is not None else cfg.get("model.imgsz", 320)
    batch = batch if batch is not None else cfg.get("model.batch", 8)
    cls_pw = cls_pw if cls_pw is not None else cfg.get("model.cls_pw", 1.0)
    pretrained = pretrained if pretrained is not None else cfg.get("model.pretrained", True)
    device = device or cfg.get("model.device", "auto")
    device = system_utils.normalize_device(device)

    out_run_dir = Path(out_run_dir)
    out_run_dir.mkdir(parents=True, exist_ok=True)

    # 权重来源: 预训练迁移学习 或 从结构从头训练
    model_src = model_name if model_name.endswith(".pt") else f"{model_name}.pt"
    from_scratch = not pretrained
    if from_scratch and not model_name.endswith((".yaml", ".pt")):
        model_src = f"{model_name}.yaml"
    if resume:
        # 断点续训: 从已有权重继续
        last_pt = sorted(out_run_dir.glob("**/last.pt"))
        model_src = str(last_pt[-1]) if last_pt else model_src

    log.info("训练模型: %s | 数据: %s | epochs=%s imgsz=%s batch=%s device=%s",
             model_src, data_yaml, epochs, imgsz, batch, device)
    log.info("系统信息:\n%s", system_utils.format_gpu_summary())

    offline = False
    # 预检测: 权重未缓存且网络不可达 -> 直接从零训练, 避免缓慢的下载重试
    # 注意需同时探测 github.com (重定向源) 与 release-assets 实际下载主机。
    if pretrained and not resume and not model_name.endswith((".yaml", ".pt")) \
            and not _weights_cached(model_name) \
            and not (_host_reachable("github.com")
                     and _host_reachable("release-assets.githubusercontent.com")):
        log.warning("预训练权重未缓存且无法访问网络, 自动从零训练 %s.yaml", model_name)
        model_src = f"{model_name}.yaml"
        from_scratch = True
        offline = True

    try:
        model = YOLO(model_src)
    except Exception as e:
        # 预训练权重下载失败 (如离线/不可达 GitHub) -> 自动回退从零训练
        if pretrained and not resume and not model_name.endswith((".yaml", ".pt")):
            log.warning("预训练权重不可用 (%s), 回退到从零训练 %s.yaml", e, model_name)
            model_src = f"{model_name}.yaml"
            from_scratch = True
            offline = True
            model = YOLO(model_src)
        else:
            raise

    kwargs: dict = dict(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=str(out_run_dir.parent),
        name=out_run_dir.name,
        exist_ok=True,
        verbose=True,
        **extra,
    )
    if offline:
        # 离线环境: 关闭 AMP, 避免其 AMP 自检尝试下载参考权重而阻塞
        kwargs["amp"] = False
    if cls_pw:
        kwargs["cls_pw"] = cls_pw
    if resume:
        kwargs["resume"] = True

    model.train(**kwargs)

    # 收集指标
    metrics = dict(getattr(model.trainer, "metrics", {}) or {})
    best_pt = Path(getattr(model.trainer, "best", "")) if getattr(model.trainer, "best", "") else None
    if best_pt is None or not best_pt.exists():
        candidate = out_run_dir / "weights" / "best.pt"
        best_pt = candidate if candidate.exists() else None

    summary = {
        "model": model_name,
        "data": str(data_yaml),
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "device": device,
        "cls_pw": cls_pw,
        "pretrained": pretrained,
        "from_scratch": from_scratch,
        "resume": resume,
        "best_weight": str(best_pt) if best_pt else None,
        "mAP50": float(metrics.get("metrics/mAP50(B)", 0.0)),
        "mAP50_95": float(metrics.get("metrics/mAP50-95(B)", 0.0)),
        "precision": float(metrics.get("metrics/precision(B)", 0.0)),
        "recall": float(metrics.get("metrics/recall(B)", 0.0)),
        "metrics_raw": {k: float(v) for k, v in metrics.items()},
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # 归档 best.pt 到 weights/ 目录 (带时间戳)
    if best_pt and best_pt.exists():
        stamp = time.strftime("%Y%m%d_%H%M%S")
        archived = out_run_dir / f"best_{model_name}_{stamp}.pt"
        shutil.copy2(best_pt, archived)
        summary["archived_weight"] = str(archived)

    # 同步更新 weights/best.pt (固定名最佳权重入口, 供 app / infer 默认使用)
    try:
        from . import paths
        weights_dir = paths.WEIGHTS_DIR
        weights_dir.mkdir(parents=True, exist_ok=True)
        best_stable = weights_dir / "best.pt"
        shutil.copy2(best_pt, best_stable)
        summary["stable_weight"] = str(best_stable)
        log.info("已更新最佳权重 -> %s", best_stable)
    except Exception as e:  # 归档失败不影响训练结果
        log.warning("更新 weights/best.pt 失败: %s", e)

    (out_run_dir / "train_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("训练完成: mAP50=%.4f | 权重=%s", summary["mAP50"], summary["best_weight"])
    return summary
