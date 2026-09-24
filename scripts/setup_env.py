#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
遥感影像目标检测系统 (YOLO-OB) —— 一键自动搭建环境

功能:
  1. 安装 CUDA 版 PyTorch (torch / torchvision)   —— 来自 download.pytorch.org
  2. 修复 matplotlib / pandas 与 numpy 2.x 的二进制不兼容
  3. 安装 ultralytics / streamlit 等其余依赖        —— 来自已配置的 pip 镜像
  4. GPU 自检 (cuda.is_available + 一次前向)

用法:
    python scripts/setup_env.py            # 完整搭建 (默认安装 CUDA 版 torch)
    python scripts/setup_env.py --cpu      # 跳过 CUDA torch, 仅装其余依赖并修依赖
    python scripts/setup_env.py --skip-torch   # 完全跳过 torch 相关安装
"""
from __future__ import annotations

import argparse
import subprocess
import sys

TORCH_INDEX = "https://download.pytorch.org/whl/cu124"
TORCH_PKGS = ["torch==2.5.0", "torchvision==0.20.0"]


def run(cmd: list[str], desc: str) -> int:
    print("\n" + "=" * 70)
    print(f"[setup] {desc}")
    print("[setup] $ " + " ".join(cmd))
    print("=" * 70)
    rc = subprocess.call(cmd)
    if rc != 0:
        print(f"[setup] 警告: 步骤失败 (exit {rc}): {desc}", file=sys.stderr)
    return rc


def step_install_cuda_torch() -> int:
    """安装 CUDA 版 PyTorch (驱动 >= 525 即可向后兼容 CUDA 12.4)。"""
    cmd = [sys.executable, "-m", "pip", "install", "--upgrade", *TORCH_PKGS,
           "--index-url", TORCH_INDEX]
    return run(cmd, "安装 CUDA 版 PyTorch (约 2.5GB, 视网速需数分钟)")


def step_fix_broken_deps() -> int:
    """修复 matplotlib / pandas 与 numpy 2.x 的不兼容。"""
    cmd = [sys.executable, "-m", "pip", "install", "--upgrade",
           "matplotlib>=3.9", "pandas>=2.2.2"]
    return run(cmd, "修复 matplotlib / pandas (numpy 2.x 兼容)")


def step_install_rest() -> int:
    """安装 ultralytics / streamlit 等其余依赖 (走已配置的镜像)。"""
    cmd = [sys.executable, "-m", "pip", "install", "--upgrade",
           "ultralytics", "streamlit", "pyyaml", "pydantic", "tqdm",
           "opencv-python", "Pillow"]
    return run(cmd, "安装 ultralytics / streamlit 等依赖")


def step_verify_gpu() -> None:
    """GPU 自检。"""
    print("\n" + "=" * 70)
    print("[setup] 自检: 依赖导入 + GPU 探测")
    print("=" * 70)
    try:
        import torch  # noqa: F401
        import torchvision  # noqa: F401
        import ultralytics  # noqa: F401
        import matplotlib  # noqa: F401
        import pandas  # noqa: F401
        import streamlit  # noqa: F401
    except Exception as e:  # pragma: no cover
        print(f"[setup] 依赖导入失败: {e}", file=sys.stderr)
        return

    import torch
    print(f"[setup] torch        = {torch.__version__}")
    print(f"[setup] torchvision  = {torchvision.__version__}")
    print(f"[setup] ultralytics  = {ultralytics.__version__}")
    print(f"[setup] matplotlib   = {matplotlib.__version__}")
    print(f"[setup] pandas       = {pandas.__version__}")
    print(f"[setup] streamlit    = {streamlit.__version__}")

    cuda_ok = torch.cuda.is_available()
    print(f"[setup] CUDA 可用     = {cuda_ok}")
    if cuda_ok:
        dev = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / 1024 ** 2
        print(f"[setup] GPU           = {dev}  ({mem:.0f} MiB)")
        x = torch.zeros(1, 3, 64, 64, device="cuda")
        y = x.sum()
        print(f"[setup] CUDA 前向自检 = OK  (sum={float(y):.3f})")
    else:
        print("[setup] 未检测到可用 CUDA, 将回退 CPU 运行 (训练较慢)。")


def main() -> None:
    ap = argparse.ArgumentParser(description="一键搭建遥感目标检测系统环境")
    ap.add_argument("--cpu", action="store_true", help="跳过 CUDA 版 torch 安装")
    ap.add_argument("--skip-torch", action="store_true", help="完全跳过 torch 相关安装")
    args = ap.parse_args()

    print("遥感影像目标检测系统 (YOLO-OB) —— 环境自动搭建")
    print(f"Python: {sys.executable}  ({sys.version.split()[0]})")

    if not args.skip_torch and not args.cpu:
        step_install_cuda_torch()
    elif not args.skip_torch:
        print("[setup] --cpu 已指定, 跳过 CUDA 版 torch 安装。")

    step_fix_broken_deps()
    step_install_rest()
    step_verify_gpu()

    print("\n" + "=" * 70)
    print("[setup] 完成。下一步运行自测:  python scripts/self_test.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
