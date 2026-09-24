# 遥感影像目标检测系统 (YOLO-OB)

基于 YOLOv8 / YOLO11 的遥感影像目标检测系统，覆盖 **数据工程 → 模型训练 → 切片推理 → Web 平台 → 检验自测** 全链路。项目源自《2026 秋燕山大学实训方案 —— AI 目标检测》中的「遥感影像目标检测系统」选题。

## 功能特性

- **数据集环境自动搭建**：`scripts/setup_env.py` 一键安装 CUDA 版 PyTorch + 修复依赖 + ultralytics/streamlit，并做 GPU 自检。
- **权威数据集**：内置 DOTA v2.0 / DIOR / NWPU VHR-10 / HRSC2016 元数据与自动下载脚本；本地 DIOR 真实检测数据（20 类、约 2.3 万张 800×800、19 万真实水平框，官方 train/val/test 划分）可直接用于数据工程与快速训练验证。
- **数据工程**：DOTA 旋转框 → YOLO 水平框转换、大图切片 + 标注坐标校正、COCO/VOC 转换、数据校验 + HTML 报告。
- **模型训练**：YOLOv8/YOLO11 迁移学习（COCO 预训练，离线自动回退从零训练），支持类别权重 (cls_pw) 缓解类别不平衡，实验归档。
- **推理**：大图切片推理 + 合并 NMS，适配高分辨率遥感影像。
- **Web 平台**：Streamlit 上传大图 → 检测 + 密度热力图 + 报告下载。
- **检验自测**：`scripts/self_test.py` 端到端跑通并输出 PASS/FAIL。

## 目录结构

```
YOLO/
├── configs/config.yaml          # 全局配置
├── scripts/
│   ├── setup_env.py             # 一键自动搭建环境
│   ├── prepare_dior.py          # 从本机 DIOR 数据集生成 YOLO 检测格式数据
│   ├── download_dataset.py      # 下载权威数据集
│   ├── data_converter.py        # DOTA/COCO/VOC -> YOLO
│   ├── data_slicer.py           # 大图切片 + 标注校正
│   ├── data_validator.py        # 数据校验 + HTML 报告
│   ├── train.py                 # 训练 + 类别平衡 + 归档
│   ├── validate.py              # mAP/P/R/每类AP 评估
│   ├── infer.py                 # 切片推理 + 合并 NMS
│   └── self_test.py             # 一键端到端自测
├── rsdet/                       # 核心包
├── app/app.py                   # Streamlit Web 平台
└── data/ runs/ weights/ reports/  # 数据与产物 (自动创建)
```

## 快速开始

### 1. 自动搭建环境

```bash
python scripts/setup_env.py
```

该命令依次完成：安装 CUDA 版 PyTorch（约 2.5GB）→ 修复 matplotlib/pandas 与 numpy 2.x 的兼容 → 安装 ultralytics/streamlit → GPU 自检。

### 2. 一键检验自测

```bash
python scripts/self_test.py
```

自测会从本机 DIOR 真实检测数据按官方划分转换 train/val/test → 生成 data.yaml → 数据校验 → `yolov8n.yaml` 抽样快速训练（不下载权重）→ 训练权重真实图推理，最终输出 `PASS/FAIL`。

### 3. 训练真实数据集

```bash
# 下载权威数据集 (受网络限制时为手工获取指引)
python scripts/download_dataset.py --list
python scripts/download_dataset.py --dataset NWPU-VHR-10

# DOTA 旋转框 -> YOLO
python scripts/data_converter.py --src dota --dota <labelTxt目录> \
    --images <图片目录> --out data/processed/labels --classes plane ship ...

# 训练
python scripts/train.py --data data/processed/dior/data.yaml --model yolov8n --epochs 50 --cls-pw 1.0
# CPU/快速试跑可改用抽样子集: --data data/processed/dior/sample/data.yaml
```

### 4. 推理与评估

```bash
python scripts/validate.py --weights runs/xxx/weights/best.pt --data data/processed/dior/data.yaml
python scripts/infer.py --weights runs/xxx/weights/best.pt --image <图片> --force-slice
```

### 5. 启动 Web 平台

```bash
streamlit run app/app.py
```

## 权威数据集说明

| 数据集 | 类别 | 体积 | 标注格式 | 主页 |
|---|---|---|---|---|
| DOTA v2.0 | 18 | ~20 GB | DOTA OBB | [captain-whu.github.io/DOTA](https://captain-whu.github.io/DOTA/) |
| DIOR | 20 | ~3.7 GB | VOC XML | [gcheng-nwpu.github.io](https://gcheng-nwpu.github.io/) |
| NWPU VHR-10 | 10 | ~73 MB | VOC XML | [gcheng-nwpu.github.io](https://gcheng-nwpu.github.io/) |
| HRSC2016 | 1 | ~1.5 GB | VOC XML | [sites.google.com/site/hrsc2016](https://sites.google.com/site/hrsc2016/) |

> 这些数据集多托管于 Google Drive / Baidu，本机若无法直连，`download_dataset.py` 会打印官方主页与手工获取指引。离线环境下可使用本机 DIOR 真实检测数据：`python scripts/prepare_dior.py`（生成 YOLO 格式数据）→ `python scripts/self_test.py`（自测含快速训练）。

## 主要配置

`configs/config.yaml` 中可调整：模型名/输入尺寸/轮数、切片尺寸/重叠比例、DIOR 数据集（`dataset.dior`：root/sample_train_per_class/sample_val_per_class/seed）等。配置加载遵循 **DEFAULT → YAML → CLI 参数** 优先级。

## 环境要求

- Python 3.10+
- NVIDIA GPU（推荐，RTX 3050 4GB 即可跑 yolov8n；无 GPU 时自动回退 CPU）
- 依赖见 `requirements.txt`，由 `setup_env.py` 自动安装
