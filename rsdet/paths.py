# -*- coding: utf-8 -*-
"""项目路径自动定位与目录创建。

项目根目录通过本文件位置向上两级定位 (rsdet/paths.py -> <root>)，
因此无论从何处启动脚本，均能正确解析 data / runs / weights / reports 等目录。
"""
from __future__ import annotations

from pathlib import Path

# 项目根目录 = 本文件 (rsdet/paths.py) 上两级
ROOT = Path(__file__).resolve().parent.parent

CONFIG_DIR = ROOT / "configs"
SCRIPTS_DIR = ROOT / "scripts"
APP_DIR = ROOT / "app"

# 数据 / 产物目录 (按需自动创建)
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
RUNS_DIR = ROOT / "runs"
WEIGHTS_DIR = ROOT / "weights"
REPORTS_DIR = ROOT / "reports"

# DIOR 真实遥感检测数据集根目录 (官方划分: ImageSets/Main/{train,val,test}.txt)
DIOR_ROOT = (ROOT / "DIOR" / "OpenDataLab___DIOR" / "raw"
             / "DIOR" / "DIOR")


def ensure_dirs() -> None:
    """确保所有关键目录存在。"""
    for d in (CONFIG_DIR, DATA_RAW, DATA_PROCESSED,
              RUNS_DIR, WEIGHTS_DIR, REPORTS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def resolve(path: str | Path) -> Path:
    """把相对路径 (相对项目根) 解析为绝对路径。"""
    p = Path(path)
    if p.is_absolute():
        return p
    return ROOT / p


ensure_dirs()
