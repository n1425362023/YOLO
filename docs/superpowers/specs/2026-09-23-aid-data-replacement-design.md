# AID 真实数据替换模拟数据 —— 设计文档

- 日期: 2026-09-23
- 状态: 待用户审阅
- 范围: YOLO-OB 遥感影像目标检测系统 (`C:\Users\DELL\Desktop\YOLO`)

## 1. 背景与目标

项目当前用**合成模拟数据**（`rsdet/synthetic.py` + `scripts/gen_demo_data.py` + `data/demo/`，
生成带 DOTA 旋转框/YOLO 水平框标注的 plane/ship/vehicle/storage-tank 演示图）支撑
数据工程 → 训练 → 推理 → Web → 自测全链路。

本机已有 **AID 真实遥感影像数据集**（约 30 类 × 220~420 张 = 约 1 万张 600×600 jpg），
位于 `AID Data Set/data/AID Data Set/AID Data Set/AID/AID_dataset/AID/<Class>/*.jpg`。

目标：
1. **删除全部模拟数据代码与产物**，替换为 AID 真实数据驱动链路。
2. **仅做最小检验证明代码正确**（真实数据跑通 划分→校验→data.yaml→结构冒烟推理），
   **不训练模型**、不联网下载权重。

## 2. 核心约束与关键判断

- AID 是**场景分类**数据集：每张图只有「整图所属场景类」（文件夹名），**无目标框标注**。
- 项目定位是**目标检测系统**（README/实训方案：数据工程→训练→切片推理→Web→自测）。
- 用户要求「能满足项目需求的方法」，即**保留检测项目形态、用真实数据替换模拟数据**。

由此确定方案 **B'（推荐）**，并如实标注其语义边界（见 §4）。

## 3. 方案对比（供决策回溯）

| 方案 | 做法 | 优点 | 缺点 | 结论 |
|---|---|---|---|---|
| A: 转场景分类 | 项目整体改为 YOLO classify | 语义正确 | 改变项目定位，不符需求 | 不采用 |
| **B': AID 图 + 整幅框（推荐）** | 每图一条覆盖全图的场景框标注进入检测链路 | 保留检测系统；真实遥感图；零联网零训练 | 标签非 AID 官方框，检测精度无参考价值（如实标注） | **采用** |
| C: 预训练伪标签 | 用 yolov8s.pt 对 AID 图生成检测框 | 对象是真目标 | 依赖联网下载权重；伪标签质量不可控 | 不采用（网络受限） |

## 4. 设计（方案 B'）

### 4.1 数据流

1. **新增 `rsdet/aid_source.py`**：
   - 扫描 AID 根目录（配置 `dataset.aid.root`，默认 `AID Data Set/data/AID Data Set/AID Data Set/AID/AID_dataset/AID`）下的 30 个类文件夹。
   - 按类固定 seed（默认 42）划分 train/val（默认 8:2）；`limit_per_class`（默认 null=取全量）支持每类抽样上限，便于快速迭代。
   - 每张图生成 YOLO 格式标注：`<class_id> 0.5 0.5 1.0 1.0`（整幅场景框），class_id 按类名字序映射。
   - **先清空** `data/processed/train`/`val`（含 labels 与缓存）再写入，保证幂等；输出到 `data/processed/{train,val}/images|labels`，生成 `data/processed/data.yaml`（30 类 names）与 `classes.txt`。
   - 返回统计 dict（类数/图数/每类计数），供自测断言。
2. **`scripts/prepare_aid.py`**（新 CLI）：包装 `aid_source`，等价替代被删的 `gen_demo_data.py`。
3. **`scripts/self_test.py` 替换步骤 1**：不再调用 `generate_demo_dataset`，改为 `prepare_aid` 产物。

### 4.2 删除清单（模拟数据全清）

| 位置 | 处置 |
|---|---|
| `rsdet/synthetic.py` | **删除整个文件** |
| `scripts/gen_demo_data.py` | **删除整个文件** |
| `configs/config.yaml` `dataset.demo` 段 | 删除，替换为 `dataset.aid` 段（root/val_ratio/seed/limit_per_class） |
| `rsdet/config.py` DEFAULTS `dataset.demo` | 删除，替换为 `dataset.aid` 默认 |
| `rsdet/paths.py` `DATA_DEMO` | 删除，新增 `AID_ROOT`（默认定位到 `AID Data Set/.../AID_dataset/AID`，可由 `dataset.aid.root` 覆盖） |
| `data/demo/` 目录 | **物理删除** |
| `data/processed/{labels,train,val,data.yaml}` 旧演示产物 | 由 prepare_aid 覆盖重建（自测每次运行先清空 processed 下 train/val） |
| README 中合成数据/`gen_demo_data.py` 相关段落 | 改写为 AID 真实数据说明 + 语义边界说明 |
| `reports/self_test_report.md` 等旧产物 | 自测运行后自然覆盖 |

### 4.3 保留复用（不删、不做非必要改动）

- `rsdet/converter.py`：DOTA→YOLO 转换器保留（供未来真实检测数据集 DOTA/DIOR 使用；AID 链路不再调用）。
- `rsdet/infer_service.py` / `app/app.py`：检测/Web 逻辑不动（本项目仍为检测系统；Web 需既有 best.pt 权重，不因 AID 改动）。
- `rsdet/train_service.py`/`validator.py`/`slicer.py`/`report.py`/`dataset_sources.py`：保留。
- `scripts/train.py`/`validate.py`/`infer.py`/`data_converter.py`/`data_slicer.py`/`data_validator.py`：保留（minimal 原则，不因本次任务改动）。

### 4.4 自测流程改写（最小检验，不训练）

`scripts/self_test.py` 新流程：
1. **准备真实数据**：`prepare_aid` → 断言类别数>0 且 train/val 图数>0（用 AID 真实 jpg）。
2. **数据校验**：`validate_dataset`（图像-标注成对/框合法）→ 断言 `ok=True`；校验器对整幅框天然兼容（中心 0.5,0.5、宽高 1.0 在合法区间、非退化）。
3. **生成 data.yaml**：断言文件存在且类名=30 个 AID 类目。
4. **结构冒烟推理**：`YOLO("yolov8n.yaml")`（离线 yaml 建图，不下载权重）→ 对 1 张 AID 训练图 `model.predict(source=..., verbose=False)` → 断言无异常返回。仅证明"真实图可经 ultralytics 前向"，不训练、不评估精度。
5. 输出 PASS/FAIL 与报告（沿用现有 `write_markdown_report` / JSON 落盘机制）。

自测进度输出中删除训练/评估/切片推理步骤；`--skip-train` 语义保留为「本就无训练」。

## 5. 错误处理

- AID 根目录缺失/为空 → 中文报错，提示 `dataset.aid.root` 配置位置与官方下载指引（README 同样更新）。
- 某类目录为空 → prepare_aid 记 warning，自测断言允许空类存在但须 ≥1 类有图；若全部为空则 FAIL。
- 图像损坏无法 cv2.imread → validator 记 issue 且整体 `ok=False`。
- 离线/无 GPU → 冒烟推理用 CPU（ultralytics `device='cpu'`，无 AMP 依赖）。

## 6. 验证与成功标准

- `python scripts/self_test.py`（新流程，不训练）→ 最终输出 **PASS**。
- `python scripts/prepare_aid.py --help` 可运行；`python scripts/prepare_aid.py` 生成 processed 产物。
- 关键文件 `lsp_diagnostics` 无错误。
- 无任何对 `synthetic`/`gen_demo_data`/`DATA_DEMO`/`data/demo` 的残留引用。
- 不触发训练、不联网下载权重。

## 7. 语义边界（如实说明）

B' 方案中 AID 的「整幅场景框」是**链路验证用占位真值**，并非 AID 官方目标标注。
由此训练出的检测模型在 mAP 上**无参考价值**。真正训练有意义的检测模型时，
应改用 DOTA / DIOR / NWPU-VHR-10（`dataset_sources.py` / `data_converter.py` 已内置支持）。
自测仅验证「真实 AID 数据能被现有数据工程与推理代码正确消费」。

## 8. 受影响文件清单

| 文件 | 动作 |
|---|---|
| `rsdet/synthetic.py` | 删除 |
| `scripts/gen_demo_data.py` | 删除 |
| `rsdet/aid_source.py` | 新增 |
| `scripts/prepare_aid.py` | 新增 |
| `rsdet/paths.py` | 改（DATA_DEMO→AID_ROOT 等） |
| `rsdet/config.py` | 改（defaults） |
| `configs/config.yaml` | 改（dataset.aid 段） |
| `scripts/self_test.py` | 重写数据准备/校验/冒烟段 |
| `rsdet/dataset_sources.py` | 改第 125 行提示语（gen_demo_data → prepare_aid） |
| `README.md` | 改（数据说明） |
| `data/demo/` | 删除 |
| `docs/superpowers/specs/2026-09-23-aid-data-replacement-design.md` | 本文档 |