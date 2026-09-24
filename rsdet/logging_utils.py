# -*- coding: utf-8 -*-
"""日志工具: 控制台 + 文件双输出, 分级。"""
from __future__ import annotations

import logging
import sys
from pathlib import Path


def setup_logger(name: str = "rsdet", log_file: str | Path | None = None,
                 level: int = logging.INFO) -> logging.Logger:
    """创建 (或复用) 一个同时输出到控制台与文件的 logger。"""
    logger = logging.getLogger(name)
    if logger.handlers:  # 已配置过则直接复用
        logger.setLevel(level)
        return logger

    logger.setLevel(level)
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s", "%H:%M:%S"
    )

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    logger.propagate = False
    return logger


def get_logger(name: str = "rsdet") -> logging.Logger:
    return logging.getLogger(name)
