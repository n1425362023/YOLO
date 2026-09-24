# -*- coding: utf-8 -*-
"""系统 / GPU 探测工具。"""
from __future__ import annotations

import platform


def gpu_info() -> dict:
    """返回 GPU / CUDA / 内存信息 (不 import torch 也能给出基本平台信息)。"""
    info: dict = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cuda_available": False,
        "gpu_name": None,
        "gpu_memory_mb": None,
        "torch_version": None,
        "device": "cpu",
    }
    try:
        import torch
        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if info["cuda_available"]:
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_memory_mb"] = int(
                torch.cuda.get_device_properties(0).total_memory / 1024 ** 2
            )
            info["device"] = "cuda:0"
    except Exception:
        pass
    return info


def get_device(preferred: str = "auto") -> str:
    """解析设备选择: auto -> 有 CUDA 用 cuda:0, 否则 cpu。"""
    if preferred not in ("auto", "cpu"):
        # 形如 "0" / "cuda:0"
        return preferred
    if preferred == "cpu":
        return "cpu"
    info = gpu_info()
    return info["device"]


def normalize_device(device: str | None) -> str:
    """把设备描述规范化为 ultralytics 可接受的形式。

    ultralytics 不接受 'auto' 字面量; 合法值: '0' / 'cpu' / 'cuda:0' / '' 等。
    规则: None/'auto'/'' -> '0'(有GPU) 或 'cpu'; 'cuda:0' -> '0'。
    """
    if device in (None, "auto", ""):
        return "0" if gpu_info()["cuda_available"] else "cpu"
    dev = str(device).strip()
    if dev.startswith("cuda:"):
        return dev.split(":")[-1] or "0"
    return dev


def format_gpu_summary() -> str:
    info = gpu_info()
    lines = [
        f"平台     : {info['platform']}",
        f"Python   : {info['python']}",
        f"PyTorch  : {info['torch_version'] or '未安装'}",
        f"CUDA 可用: {info['cuda_available']}",
    ]
    if info["cuda_available"]:
        lines.append(f"GPU      : {info['gpu_name']} ({info['gpu_memory_mb']} MiB)")
    lines.append(f"训练设备 : {info['device']}")
    return "\n".join(lines)
