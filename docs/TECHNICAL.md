# 遥感影像目标检测系统 (YOLO-OB) 技术文档

> 面向开发者的系统设计、模块职责、数据流与二次开发指南。面向使用者的快速上手见 [README](../README.md)。
> 本文档描述当前代码基线（DIOR 真实检测数据链路）。

---

## 1. 系统概述

基于 **YOLOv8 / YOLO11 (ultralytics)** 的遥感影像目标检测系统，覆盖：

```
数据工程 → 模型训练 → 切片推理 → Web 平台 → 检验自测
```

- 数据层：本地 **DIOR** 真实遥感检测数据集（20 类 / 23,463 张 800×800 / 19 万真实水平框 / 官方 train-val-test 划分），另有 DOTA v2.0 / NWPU VHR-10 / HRSC2016 元数据与自动下载入口。
- 训练层：迁移学习（COCO 预训练 / 离线自动从零回退）、类别权重 (cls_pw)、实验归档。
- 推理层：大图自动切片推理 + 合并 NMS，适配数千×数千高分辨率遥感影像。
- 交互层：Streamlit Web 平台（上传 → 检测 → 热力图 → 报告下载）。
- 质量层：端到端 `scripts/self_test.py` 一键自测，输出 PASS/FAIL。

## 2. 技术栈

| 组件 | 版本 | 说明 |
|---|---|---|
| Python | 3.10+ | 类型注解使用 `from __future__ import annotations` |
| PyTorch | 2.5.0+cu124 | CUDA 训练/推理 |
| ultralytics | 8.4.x | YOLOv8 / YOLO11 训练与推理 |
| OpenCV | 4.8+ (本机 5.0.0) | 图像 IO / 可视化 |
| NumPy | 1.24+ (本机 2.2.6) | 数值计算 / 纯 numpy NMS |
| Streamlit | 1.35+ | Web 平台 |
| PyYAML | 6.0+ | 配置 / data.yaml |
| filelock | ≥3.16.1 | **强制要求**：ultralytics 8.4.159 依赖 `AsyncFileLock`，低于 3.16 会 ImportError |

> ⚠️ 关键兼容性：`numpy 2.x` 需 matplotlib ≥3.9、pandas ≥2.2.2（见 requirements.txt 注释）。

## 3. 目录结构与模块地图

```
YOLO/
├── configs/config.yaml        # 全局配置（相对项目根）
├── rsdet/                     # 核心包（全部业务逻辑）
│   ├── paths.py               # 路径定位 + 目录自动创建
│   ├── config.py              # 配置加载 DEFAULT → YAML → CLI 三级覆盖
│   ├── converter.py           # DOTA/COCO/VOC → YOLO 水平框转换
│   ├── dior_source.py         # DIOR 数据集接入（官方划分 → YOLO + 抽样子集）
│   ├── slicer.py              # 大图切片 + 标注校正 + 合并 NMS
│   ├── trainer 相关            # train_service.py: 训练/评估/归档/data.yaml
│   ├── validator.py           # 数据校验（成对/类别/越界/NaN/分布）
│   ├── report.py              # matplotlib 图表 + 自包含 HTML/Markdown 报告
│   ├── infer_service.py       # 单图/大图切片推理 + 标注可视化
│   ├── system_utils.py        # GPU/设备探测与规范化
│   ├── logging_utils.py       # 控制台+文件双输出日志
│   └── dataset_sources.py     # 权威数据集元数据 + 尽力自动下载
├── scripts/                   # CLI 入口（薄封装，逻辑在 rsdet）
│   ├── setup_env.py           # 一键环境搭建（CUDA torch + 依赖修复 + GPU 自检）
│   ├── prepare_dior.py        # DIOR → YOLO 格式数据生成
│   ├── download_dataset.py    # 权威数据集下载/指引
│   ├── data_converter.py      # DOTA/COCO/VOC → YOLO
│   ├── data_slicer.py         # 大图切片 + 标注校正
│   ├── data_validator.py      # 数据校验 + HTML 报告
│   ├── train.py               # 训练 + 类别平衡 + 归档
│   ├── validate.py            # mAP/P/R/每类 AP 评估
│   ├── infer.py               # 切片推理 + 合并 NMS
│   └── self_test.py           # 端到端检验自测（PASS/FAIL）
├── app/app.py                 # Streamlit Web 平台
├── data/
│   ├── raw/                   # 权威数据集下载目标（自动创建）
│   └── processed/dior/        # DIOR 转换产物（见 §5）
├── runs/selftest_<ts>/        # 自测运行产物（train_run/weights/best.pt、json）
├── weights/ reports/          # 权重归档与报告（自动创建）
└── DIOR/                      # 本机 DIOR 原始数据（OpenDataLab 结构）
```

**依赖方向**：`scripts/`、`app/` → `rsdet/` 单向依赖；`rsdet` 内部无环（config←paths 除外，paths 不依赖 config）。

## 4. 配置系统

### 4.1 三级覆盖

```
代码内 DEFAULT（config.py DEFAULTS） ← YAML（configs/config.yaml） ← CLI 参数（各 script --xxx）
```

- `Config` 对象支持 `cfg.model.imgsz` 点号访问与 `cfg.get("inference.conf", 0.25)` 安全读取。
- 缺失字段一律回退默认值，不崩溃。
- 路径字段按相对项目根解析为绝对路径（`paths.resolve`）。

### 4.2 关键配置项

| 键 | 默认 | 说明 |
|---|---|---|
| `dataset.dior.root` | null | DIOR 根；null = 自动定位 `DIOR/OpenDataLab___DIOR/raw/DIOR/DIOR` |
| `dataset.dior.sample_train_per_class` | 16 | 快速训练子集：每类从 train 抽几张 |
| `dataset.dior.sample_val_per_class` | 8 | 每类从 val 抽几张 |
| `dataset.dior.seed` | 42 | 抽样随机种子（复现） |
| `model.name / imgsz / epochs / batch / device / cls_pw / pretrained` | yolov8n / 416 / 5 / 8 / auto / 1.0 / true | 训练 |
| `inference.tile_size / overlap / conf / iou` | 640 / 0.25 / 0.25 / 0.45 | 切片推理 |
| `selftest.epochs` | 8 | 自测快速训练轮数 |

## 5. 数据层：DIOR 接入（rsdet/dior_source.py）

### 5.1 源数据布局

```
DIOR/OpenDataLab___DIOR/raw/DIOR/DIOR/
├── JPEGImages-trainval/       # train + val 图片 (11,725)
├── JPEGImages-test/           # test 图片 (11,738)
├── Annotations/Horizontal Bounding Boxes/*.xml   # VOC XML 水平框
└── ImageSets/Main/{train,val,test}.txt           # 官方划分
```

### 5.2 转换流程（prepare_dior_dataset）

1. **类别发现**：按官方 `train.txt` 中类别**首个出现顺序**生成类 id（与官方一致）。
2. **全量转换**：所有 HBB XML → YOLO 归一化水平框（staging 暂存），按官方划分分发图片+标签到 `train/val/test` 三份 images/labels。
3. **抽样子集**：每类从 train/val 候选图中取前 `min(n, per)` 张（固定 seed 的随机），生成 `sample/train|val`（供快速训练）。多类图可能被多个类选中 → 去重后唯一图片数 ≤ 每类抽样数之和（设计如此）。
4. **产物**：`classes.txt`、全量 `data.yaml`、抽样 `sample/data.yaml`（train/val 路径相对共同父目录写出，避免两划分落到同一目录）。

### 5.3 产物统计（当前基线）

| 划分 | 图片 | 真实框 |
|---|---|---|
| train | 5,862 | — |
| val | 5,863 | — |
| test | 11,738 | — |
| 合计 | 23,463 | 192,512 |
| sample/train | 319（唯一文件，语义 320=16/类×20） | — |
| sample/val | 159（语义 160） | — |

### 5.4 data.yaml 格式

```yaml
path: C:\...\data\processed\dior          # 共同父目录（绝对）
train: train/images                       # 相对 path
val: val/images
names: {0: golffield, 1: Expressway-toll-station, ...}   # 20 类，官方序
```

## 6. 标注格式转换（rsdet/converter.py）

| 源格式 | 函数 | 说明 |
|---|---|---|
| DOTA OBB (8 角点 labelTxt) | `dota_line_to_yolo` / `convert_dota_dir` | 4 顶点 → 轴对齐外接框 → 归一化 YOLO |
| COCO JSON | `coco_to_yolo` | bbox [x,y,w,h] → cx,cy,w,h；按 class_map 映射 |
| VOC XML | `voc_to_yolo` | 用于 DIOR / NWPU VHR-10 / HRSC2016 |

统一输出：`class_id cx cy w h`（归一化 [0,1]，越界裁剪，退化框丢弃）。

## 7. 训练（rsdet/train_service.py）

```
train(cfg, data_yaml, out_run_dir, model_name, ...)
```

- **权重来源决策**：
  1. `pretrained=True` 且权重已缓存 / 网络可达 → `yolov8n.pt`（迁移学习）
  2. `pretrained=True` 但权重未缓存且网络不可达，或首次构造下载失败 → **自动回退** `yolov8n.yaml` 从零训练（离线可用）
  3. `pretrained=False` → 直接 `yolov8n.yaml`
  4. `resume=True` → 从 `out_run_dir` 内最近的 `last.pt` 续训
- **类别不平衡**：`cls_pw`（类别权重幂次）传给 ultralytics `cls_pw`，1.0 = 按类别频率反比加权，0 禁用。
- **离线 AMP 处理**：离线从零训练自动 `amp=False`，避免 AMP 自检尝试下载参考权重而阻塞。
- **归档**：`best.pt` 复制为 `best_<model>_<ts>.pt`，训练摘要写 `train_summary.json`。

**评估**：`evaluate()` 封装 `model.val()`，返回 mAP50 / mAP50-95 / P / R / 每类 AP。

## 8. 推理（rsdet/infer_service.py + slicer.py）

### 8.1 大图切片推理策略

```
max(h, w) > tile_size * 1.25  →  切片推理；否则单次直接推理
```

1. `compute_tiles(w, h, tile_size, overlap)` 生成重叠切片网格（边缘自动回缩保证全覆盖）。
2. 每切片独立推理，检测框偏移回全局坐标。
3. `merge_detections`：按类别分组，纯 numpy NMS（分数降序、IoU ≤ thr 保留）去重。
4. 结果：`[{cls, conf, bbox[x0,y0,x1,y1]}]`

### 8.2 训练数据切片（可选预处理）

`slice_dataset`：整图 + YOLO 标注 → 重叠切片 + 标注坐标投影校正（仅保留中心点落在切片内的框，框越界部分裁剪）。

## 9. CLI 命令手册

| 命令 | 作用 |
|---|---|
| `python scripts/setup_env.py` | 一键装 CUDA torch + 修复依赖 + GPU 自检 |
| `python scripts/prepare_dior.py [--root --out --sample-train-per-class ...]` | DIOR → YOLO 数据（默认 16/8 抽样，--dry-run 预览） |
| `python scripts/download_dataset.py --list \| --dataset <名>` | 权威数据集（直连失败打印手工指引） |
| `python scripts/data_converter.py --src dota\|coco\|voc ...` | 标注转 YOLO |
| `python scripts/data_slicer.py --images --labels --out --tile-size` | 切片 + 标注校正 |
| `python scripts/data_validator.py --images --labels --classes` | 校验 + HTML 报告 |
| `python scripts/train.py --data data.yaml --model yolov8n --epochs 50 [--cls-pw 1.0]` | 训练 |
| `python scripts/validate.py --weights .../best.pt --data data.yaml` | 评估 |
| `python scripts/infer.py --weights .../best.pt --image <图> [--force-slice]` | 批量/单图推理 |
| `python scripts/self_test.py [--epochs --imgsz --device]` | 端到端自测 |
| `streamlit run app/app.py` | Web 平台 |

CLI 参数 > config.yaml > DEFAULT（与 §4.1 一致）。

## 10. 检验自测（scripts/self_test.py）6 步流程

1. **DIOR 官方划分 → 真实框转换 + 抽样子集**（复现 §5 全流程，统计 split_counts/boxes_total/sample_counts）
2. **生成 data.yaml / classes.txt**（全量 + 抽样两份）
3. **数据校验**（sample/train 的真实框成对性 / 类 id 合法 / 坐标归一化合法 / 无 NaN）
4. **抽样快速训练**（`yolov8n.yaml` 从零、`pretrained=False`、`amp=False`、8 epochs，不下载权重）
5. **真实图推理**（训练 best.pt 或离线建图兜底对 sample/val 首图前向，断言无异常）
6. **汇总报告**（PASS/FAIL → `reports/self_test_report.md` + `runs/selftest_<ts>/self_test.json`，退出码 0/1）

> 自测约 13 分钟（RTX 3050 4GB，yolov8n / 416 / 8 epochs）。mAP50 仅记录不设达标线（快速训练目的为链路检验）。

## 11. Web 平台（app/app.py）

- 权重自动发现：`runs/**/weights/best.pt` + `weights/*.pt`（下拉选择）。
- 上传图像 → 自动决定单次/切片推理 → KPI（目标数/平均置信度/耗时/切片数）→ 原图-标注对比 → 明细表 → **密度热力图**（检测框中心分布）→ HTML 报告下载。
- GPU 摘要常驻侧栏。

## 12. 关键设计决策（ADR 摘要）

| 决策 | 理由 |
|---|---|
| DIOR 官方划分直用（不重切 train/val） | 与论文/榜单可比；HBB 已是水平框，无需转换几何 |
| 类 id 按官方 train.txt 首现序 | 类 id 稳定性，避免排序漂移 |
| sample 每类固定 16/8 | 快速训练子集类别平衡；多类图去重后唯一文件数稍少属预期 |
| 离线自动从零回退 + amp=False | 离线环境不被权重下载 / AMP 自检阻塞 |
| 纯 numpy NMS | 不依赖 torchvision.ops，推理链路轻 |
| 训练/评估薄封装 | 全部业务逻辑收进 rsdet，CLI 只做参数解析 |

## 13. 二次开发指引

- **新增数据集**：实现 `discover_classes` + `prepare_*` 风格函数（仿 `dior_source.py`），在 `config.yaml` dataset 下加一节，并在 `dataset_sources.py` 登记元数据。
- **新增训练配置**：改 `config.py DEFAULTS` + `config.yaml`，`train()` 以 `**extra` 透传 ultralytics 参数。
- **验证改动**：改完核心逻辑后运行 `python scripts/self_test.py`（端到端）或先 `python -m py_compile` 快速查语法。
- **报告**：图表均转 base64 内嵌，单文件 HTML 可分享。

## 14. 已知注意事项

- 自测/全量转换会 `rmtree` 重建 `data/processed/dior`（幂等），耗时约 10-13 分钟。
- `classes.txt == data.yaml names` 的**顺序**一致性曾出现 False——两者内容同为官方首现序 20 类，但文本层面比较受换行/顺序影响；以 `data.yaml names` 为训练权威。
- 环境偶发 `filelock` 过旧（<3.16.1）导致 ultralytics `AsyncFileLock` ImportError——升级即可：`pip install "filelock>=3.16.1"`。