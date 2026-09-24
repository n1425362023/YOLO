# -*- coding: utf-8 -*-
"""配置加载: YAML 配置文件 -> 可点号访问的配置对象 (带默认值)。

设计要点 (对齐 PDF「配置管理」要求):
  * DEFAULT 默认值内置在代码里, YAML 覆盖之, CLI 参数再覆盖之。
  * 所有取值经 .get 回退, 保证缺失字段不崩溃。
  * 路径字段统一解析为绝对路径 (相对项目根)。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from . import paths

DEFAULTS: dict[str, Any] = {
    "model": {
        "name": "yolov8n",
        "imgsz": 416,
        "epochs": 5,
        "batch": 8,
        "device": "auto",
        "cls_pw": 1.0,
        "pretrained": True,
    },
    "inference": {
        "tile_size": 640,
        "overlap": 0.25,
        "conf": 0.25,
        "iou": 0.45,
    },
    "dataset": {
        "dior": {
            "root": None,                  # None = 用 paths.DIOR_ROOT 默认路径; 可填绝对路径覆盖
            "sample_train_per_class": 16,  # 快速训练子集: 每类从 train 抽几张
            "sample_val_per_class": 8,     # 快速训练子集: 每类从 val 抽几张
            "seed": 42,                    # 抽样随机种子
        },
    },
    "selftest": {
        "epochs": 8,   # 快速训练轮数 (抽样训练, 不做达标阈值)
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    """递归合并字典, override 优先。"""
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


class Config:
    """配置对象: 支持 config.paths.data_processed 式访问, 缺失时回退默认值。"""

    def __init__(self, data: dict):
        self._data = data

    @classmethod
    def load(cls, yaml_path: str | Path | None = None) -> "Config":
        data: dict = _deep_merge(DEFAULTS, {})
        path = Path(yaml_path) if yaml_path else paths.CONFIG_DIR / "config.yaml"
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f) or {}
            data = _deep_merge(data, loaded)
        return cls(data)

    def get(self, dotted: str, default: Any = None) -> Any:
        cur: Any = self._data
        for part in dotted.split("."):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                return default
        return cur

    def __getattr__(self, name: str) -> Any:
        if name in self._data:
            v = self._data[name]
            return Config(v) if isinstance(v, dict) else v
        raise AttributeError(name)

    def as_dict(self) -> dict:
        return self._data


# 便捷单例 (懒加载)
_CONFIG: Config | None = None


def get_config(yaml_path: str | Path | None = None) -> Config:
    global _CONFIG
    if _CONFIG is None or yaml_path is not None:
        _CONFIG = Config.load(yaml_path)
    return _CONFIG
