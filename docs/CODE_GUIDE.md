# 代码文件导航与训练指南

> 本文档回答两个问题：**每个代码文件是干什么的**、**怎么开始训练**。
> 架构与设计决策详见 `docs/TECHNICAL.md`。

---

## 一、训练代码在哪里（速答）

| 你要做什么 | 文件 | 命令 |
|---|---|---|
| **开始训练（入口）** | `scripts/train.py` | `python scripts/train.py --data ...` |
| 训练核心逻辑 | `rsdet/train_service.py` 的 `train()` | （被上面调用，不需要直接碰） |
| 训练配置 | `configs/config.yaml` + `rsdet/config.py` | 改配置即可，不用改代码 |
| 数据准备 | `scripts/prepare_dior.py` → `rsdet/dior_source.py` | 已就绪，无需重跑 |
| 训练数据清单 | `data/processed/dior/data.yaml`（全量）/ `sample/data.yaml`（抽样） | 已存在 |

一句话：**训练入口 = `scripts/train.py`，数据 = `data/processed/dior/*/data.yaml`，配置 = `configs/config.yaml`。**

---

## 二、代码文件逐个说明（23 个）

### 核心包 `rsdet/`（共 12 个文件）——所有业务逻辑都在这里

| 文件 | 职责 | 关键函数 |
|---|---|---|
| `paths.py` | 项目根目录定位 + 自动创建 data/runs/weights/reports 目录 | `ensure_dirs()` / `resolve()` |
| `config.py` | 配置加载：代码默认 → config.yaml → CLI 参数 三级覆盖 | `get_config()` / `Config.get()` |
| `converter.py` | 标注格式转换：DOTA 旋转框 / COCO / VOC → YOLO 水平框 | `convert_dota_dir()` / `coco_to_yolo()` / `voc_to_yolo()` |
| `dior_source.py` | **DIOR 数据集接入**：官方划分 → YOLO 格式 + 每类抽样子集 | `discover_classes()` / `prepare_dior_dataset()` / `resolve_dior_root()` |
| `train_service.py` | **训练服务**：封装 ultralytics 训练、离线自动从零回退、类别平衡、实验归档 | `train()` / `evaluate()` / `write_data_yaml()` |
| `slicer.py` | 大图切片 + 标注坐标校正 + 合并 NMS（纯 numpy） | `compute_tiles()` / `slice_dataset()` / `merge_detections()` |
| `infer_service.py` | **推理服务**：单图 / 大图自动切片推理 + 检测框可视化 | `detect()` / `detect_file()` / `load_model()` |
| `validator.py` | 数据校验：图像-标注成对、类别合法、越界/NaN、类别分布 | `validate_dataset()` / `summarize()` |
| `report.py` | 报告生成：类别分布图、密度热力图、自包含 HTML / Markdown 报告 | `write_html_report()` / `density_heatmap()` |
| `system_utils.py` | GPU / CUDA 探测、设备规范化 | `get_device()` / `normalize_device()` / `format_gpu_summary()` |
| `logging_utils.py` | 控制台 + 文件双输出日志 | `setup_logger()` |
| `dataset_sources.py` | 4 个权威数据集（DOTA/DIOR/NWPU/HRSC）元数据 + 尽力自动下载 | `download_dataset()` / `list_datasets()` |

### 命令行脚本 `scripts/`（共 10 个）——薄封装，参数解析后调 rsdet

| 文件 | 职责 | 常用命令 |
|---|---|---|
| `train.py` | **训练入口** | `python scripts/train.py --data X --model yolov8n --epochs N` |
| `prepare_dior.py` | DIOR → YOLO 数据生成（已跑过，数据就绪） | `python scripts/prepare_dior.py` |
| `self_test.py` | 端到端自测（6 步，约 13 分钟，输出 PASS/FAIL） | `python scripts/self_test.py` |
| `infer.py` | 推理（单图/目录批量，自动切片） | `python scripts/infer.py --weights .../best.pt --image f.jpg` |
| `validate.py` | 评估：mAP50 / mAP50-95 / P / R / 每类 AP | `python scripts/validate.py --weights ... --data ...` |
| `data_converter.py` | DOTA/COCO/VOC → YOLO（其他数据集用） | `python scripts/data_converter.py --src voc ...` |
| `data_slicer.py` | 大图切片 + 标注校正（训练数据预处理） | `python scripts/data_slicer.py --images ... --labels ...` |
| `data_validator.py` | 数据校验 + HTML 报告 | `python scripts/data_validator.py --images ... --labels ... --classes ...` |
| `download_dataset.py` | 权威数据集下载/手工获取指引 | `python scripts/download_dataset.py --list` |
| `setup_env.py` | 一键环境搭建（换机器时用） | `python scripts/setup_env.py` |

### 其他

| 文件/目录 | 职责 |
|---|---|
| `app/app.py` | Streamlit Web 平台（上传图片 → 检测 → 热力图 → 报告下载） |
| `configs/config.yaml` | 全局配置（模型、数据、推理、自测参数） |
| `requirements.txt` | Python 依赖清单 |
| `data/processed/dior/` | DIOR 转换产物（data.yaml、图片、标签、classes.txt） |
| `runs/` | 训练/自测产物（每个实验一个目录，含 weights/best.pt） |
| `weights/` | 手动放置的权重（app 会优先认这里） |
| `reports/` | 报告输出 |

---

## 三、怎么开始训练（三步）

### 前提确认（已满足）

```powershell
python -X utf8 -c "import sys; sys.path.insert(0,'.'); from rsdet.train_service import train; print('训练模块可用')"
```

### 第 1 步：快速验证训练链路（抽样数据，约 20-40 分钟，推荐先跑）

```powershell
python scripts/train.py --data data/processed/dior/sample/data.yaml --model yolov8n --epochs 30 --no-pretrained
```

- 数据：319 张 train（每类抽样 16 张），20 类
- `--no-pretrained`：因为本机无法访问权重下载主机（实测不可达），直接从头训练，避免卡在下载重试
- 预期：mAP50 从 0.003 升到 0.1+ → 链路正常
- 产物：`runs/yolov8n_<时间戳>/weights/best.pt`

### 第 2 步：用 DIOR 真实航拍图验证效果

```powershell
python scripts/infer.py --weights runs\yolov8n_<时间戳>\weights\best.pt --dir data\processed\dior\val\images --conf 0.25
```

> ⚠️ 必须用**俯视遥感图**测试。DIOR 的 `airplane`/`vehicle`/`ship` 都是俯视视角，平视/侧视飞机照片永远检测不到。

### 第 3 步：全量正式训练（几小时，出可用模型）

```powershell
python scripts/train.py --data data/processed/dior/data.yaml --model yolov8n --epochs 50 --imgsz 416 --batch 8 --no-pretrained
```

- 数据：5862 张 train 图；RTX 3050 4GB 预计数小时
- 中断后可续训：`--resume`（自动找 last.pt）
- 训练完供 app 使用：
  ```powershell
  Copy-Item runs\yolov8n_<时间戳>\weights\best.pt weights\model.pt
  streamlit run app/app.py   # 侧栏下拉选 model.pt
  ```

### 常用训练参数速查

| 参数 | 作用 | 默认 |
|---|---|---|
| `--model` | yolov8n / yolov8s / yolo11n | config: yolov8n |
| `--epochs` | 训练轮数 | config: 5 |
| `--imgsz` | 输入尺寸 | config: 416 |
| `--batch` | 批大小（4GB 显存建议 8） | config: 8 |
| `--cls-pw` | 类别权重幂次（缓解类别不平衡） | config: 1.0 |
| `--no-pretrained` | 不用 COCO 预训练，从头训练 | 默认用预训练 |
| `--resume` | 断点续训 | 关 |
| `--name` | 实验目录名 | 自动时间戳 |
| `--device` | auto / cpu / 0 | auto |

---

## 四、常见问题

| 问题 | 原因与解决 |
|---|---|
| app 检测不到目标 | 用的是自测权重（8 轮、每类 16 张，几乎无检测能力）；或图片不是俯视遥感图。按本文第 3 节训练即可 |
| `missing ScriptRunContext` 警告 | 用了 `python app/app.py`，应改用 `streamlit run app/app.py` |
| `AsyncFileLock` ImportError | filelock 过旧：`pip install "filelock>=3.16.1"` |
| 训练卡在下载 | 本机无法访问权重下载主机，加 `--no-pretrained` |
| 端口被占用 | `streamlit run app/app.py --server.port 8502` |