# DIOR 真实检测数据集替换 AID 占位数据 设计文档

- 日期: 2026-09-23
- 状态: 设计定稿（已获用户关键决策确认）
- 设计者: Sisyphus

## 1. 背景与目标

上一轮已将项目模拟数据替换为 AID 场景分类数据集。AID 仅有整图场景类别、无目标框，
为保留检测链路语义采用了「整幅场景框占位真值」方案——这是妥协，检测语义并不真正成立。

用户本次要求：**根据 DIOR 重新替换数据，并修改代码**。DIOR 是真实目标检测数据集，
每张图带真实水平框标注，替换后检测语义完全成立，可移除占位标签与 README 中的语义边界警示。

用户关键决策（已确认）：
1. **完全替换**：删除 AID 数据链路代码（aid_source.py / prepare_aid.py / dataset.aid / AID_ROOT），DIOR 为唯一数据源。
2. **本轮自测包含快速训练验证**（与上一轮「不训练」约束不同，本轮用户明确批准）。
3. **使用 DIOR 官方划分**（train/val/test），与基准可比。
4. **抽样快速训练**：自测用抽样子集训练，全量数据完整转换供外部使用。
5. **新建 dior_source 模块**（对称于 AID 方案，复用现有 converter.voc_to_yolo），不引入通用多源抽象（YAGNI）。

## 2. 数据源事实（本机已验证）

| 项 | 值 |
|---|---|
| 位置 | `DIOR/OpenDataLab___DIOR/raw/DIOR/DIOR/` |
| 标注 | `Annotations/Horizontal Bounding Boxes/*.xml` — 23463 个 VOC XML（VOC 格式，含 filename/size/object/bndbox） |
| 图像 | `JPEGImages-trainval/` 11725 张 + `JPEGImages-test/` 11738 张 = 23463 张，全部 800×800 |
| 官方划分 | `ImageSets/Main/{train,val,test}.txt` — train 5862 / val 5863 / test 11738 |
| 类别 | 20 类，192,518 个真实框；类名含 `Expressway-Service-area`、`Expressway-toll-station` 等连字符混合大小写 |
| 旋转框 | `Annotations/Oriented Bounding Boxes/` 另有 OBB 版（23463 个），本项目使用 HBB 水平框（VOC 标准），OBB 不接 |

类别分布前 5：ship 62537、vehicle 40365、storagetank 26403、tenniscourt 12241、airplane 10100。

## 3. 架构与数据流

### 3.1 新增模块：`rsdet/dior_source.py`

```
resolve_dior_root(root=None) -> Path
    CLI 参数 > config(dataset.dior.root) > paths.DIOR_ROOT；不存在则抛中文指引错误。

discover_classes(hbb_dir) -> list[str]
    扫描 HBB XML 的 object/name 去重，按官方 train.txt 类出现序排序（保证与官方类 id 一致）。

parse_split(split_file) -> list[str]
    读取 ImageSets/Main/*.txt 的行（不含扩展名），返回 stem 列表。

prepare_dior_dataset(dior_root, out_base, classes=None,
                     sample_train_per_class=16, sample_val_per_class=8,
                     seed=42) -> dict
    1. discover_classes + 三份官方划分
    2. 清空 out_base（幂等）
    3. 每张图: _locate_image(stem) 在 trainval/test 中找 jpg -> 复制到 <out_base>/train|val|test/images
       voc_to_yolo(hbb_xml_dir, <out_base>/split/labels, class_map) 直接转换
    4. 抽样子集: 每个类从 train 取 min(16, avail) 张进 sample/train, val 取 min(8, avail) 张进 sample/val
       (固定 seed 采样, 复制文件+标签, 用于快速训练)
    5. 写 classes.txt + data.yaml (经 write_data_yaml, path=sample 或根)
    6. 返回 stats

_locate_image(dior_root, stem) -> Path
    JPEGImages-trainval 或 -test 下定位 stem.*。图片均 800×800，无需读尺寸。

_subsample(class_stems, n, seed) -> list[str]
    每类内固定 seed 洗牌取前 n。
```

`data.yaml` 指向何处是两个可选口径：
- **全量 data.yaml**（`data/processed/dior/data.yaml`）：train=train/images, val=val/images —— 供外部完整训练/评估。
- **抽样 data.yaml**（`data/processed/dior/sample/data.yaml`）：train=sample/train/images, val=sample/val/images —— 供自测快速训练。
两个都生成，互不干扰。

### 3.2 输出布局

```
data/processed/dior/
├── classes.txt            # 20 类（官方序）
├── data.yaml              # 全量：train=train/images, val=val/images
├── train/images|labels    # 5862 张（官方 train）
├── val/images|labels      # 5863 张（官方 val）
├── test/images|labels     # 11738 张（官方 test，保留供评估）
└── sample/
    ├── train/images|labels  # ≤320 张（20 类 × ≤16）
    ├── val/images|labels    # ≤160 张（20 类 × ≤8）
    └── data.yaml            # 抽样：train=sample/train/images, val=sample/val/images
```

### 3.3 配置

`configs/config.yaml` + `rsdet/config.py` DEFAULTS（删除 `dataset.aid`）：

```yaml
dataset:
  dior:
    root: null                    # null=默认 DIOR_ROOT
    sample_train_per_class: 16
    sample_val_per_class: 8
    seed: 42
```

`rsdet/paths.py`：`AID_ROOT` → `DIOR_ROOT = ROOT / "DIOR" / "OpenDataLab___DIOR" / "raw" / "DIOR" / "DIOR"`。

`rsdet/config.py` DEFAULTS 的 `selftest` 段同步精简（新自测只消费 `selftest.epochs`；
`mAP50_threshold`/`min_detections` 已无消费者，删除以免误导）：

```python
"selftest": {
    "epochs": 8,   # 快速训练轮数 (抽样训练, 不做达标阈值)
},
```

## 4. 代码改动清单

### 删除
- `rsdet/aid_source.py`
- `scripts/prepare_aid.py`
- `data/processed/` 下旧 AID 产物（自测每次运行先清空再重建，无需手动管）

### 新增
- `rsdet/dior_source.py`
- `scripts/prepare_dior.py`（CLI：`--root/--out/--classes/--sample-train-per-class/--sample-val-per-class/--seed/--dry-run`，对称替代 prepare_aid）

### 修改
- `rsdet/paths.py`：AID_ROOT → DIOR_ROOT
- `rsdet/config.py`：dataset.aid → dataset.dior
- `configs/config.yaml`：dataset.aid → dataset.dior；`selftest` 段同步精简为 `epochs: 8`（删除 `mAP50_threshold`/`min_detections`，与新自测消费一致）
- `scripts/self_test.py`：重写（见 §5）
- `rsdet/dataset_sources.py:125`：prepare_aid → prepare_dior 提示语
- `README.md`：见 §7
- `scripts/data_converter.py / data_slicer.py / data_validator.py / infer.py`：不涉及（上轮已改示例路径，与 DIOR 无关）

### 复用不动
- `rsdet/converter.py`（voc_to_yolo / build_class_map / write_classes_txt 均可用）
- `rsdet/validator.py` / `report.py` / `train_service.py`（write_data_yaml/train/evaluate） / `system_utils.py` / `logging_utils.py`
- `scripts/train.py / validate.py / infer.py / data_converter.py / data_slicer.py / data_validator.py / download_dataset.py / setup_env.py`
- `app/app.py`

## 5. 自测流程（scripts/self_test.py 重写，6 步）

```
1. 数据准备: prepare_dior_dataset 全量 + 抽样子集
   [PASS] 20 类 / train 5862 val 5863 test 11738 / sample 20 类均有图
2. 生成 data.yaml (全量 + 抽样两张) + classes.txt 存在
   [PASS] 断言 names 20 项
3. 数据校验: validate_dataset(sample/train/images, sample/train/labels, classes)
   [PASS] ok=True 且 images_with_labels>0（真实框, 非占位）
4. 快速训练: YOLO("yolov8n.yaml") 从零训练 sample 子集
   epochs=cfg.selftest.epochs(默认 8) imgsz=model.imgsz(416) device=normalize_device(args.device)
   产出 best.pt; mAP50 仅记录不设达标线(避免随机种子偶发失败)
5. 训练后推理: best.pt predict 1~2 张 sample/val 真实图 -> 断言前向无异常 + 返回检测框
6. 报告: write_markdown_report + runs/selftest_*/self_test.json + PASS/FAIL + sys.exit(0/1)
```

快速训练失败兜底：步骤 4 异常 → 记录详情并该步 FAIL，但**不终止**；步骤 5 回退用
`YOLO("yolov8n.yaml")` 离线建图对 1 张真实图冒烟前向（同上一轮方案），保证结构链路仍可验证。
全程不下载权重（yolov8n.yaml 离线建图）。

## 6. 错误处理

| 场景 | 行为 |
|---|---|
| DIOR 根不存在 | `resolve_dior_root` 抛 `FileNotFoundError`，中文指引指向 config `dataset.dior.root` |
| HBB 标注目录缺失 | 显式错误，列出期望路径 |
| ImageSets/Main 缺文件 | 显式错误，列出三个期望 txt |
| 单 XML 损坏 | voc_to_yolo 跳过并计数；validate_dataset 记 issue |
| 某类抽样不足 | `min(available, n)` 兜底，20 类保底各 1 张 |
| 训练阶段异常 | 步骤 FAIL + 详情，冒烟兜底，不误伤结构链路结论 |

## 7. README 更新

- 功能特性·权威数据集：补「本地 DIOR 真实检测数据（20 类、约 2.3 万张、19 万真实框）」
- 目录结构：`prepare_aid.py` → `prepare_dior.py`
- 快速开始·自测：更新为 DIOR 流程描述（含快速训练）
- 离线提示与「主要配置」：dataset.aid → dataset.dior
- **删除整节「AID 真实数据集与语义边界」**（DIOR 为真实标注，占位语义不复存在）

## 8. 验证与成功标准

1. `python -m py_compile` 全部改动文件通过
2. `python scripts/prepare_dior.py --help` / `python scripts/prepare_dior.py --dry-run` 可用
3. 全量 `prepare_dior_dataset` → train 5862 / val 5863 / test 11738 / sample 20 类齐全
4. `python scripts/self_test.py --device 0` → 6 步 PASS（含快速训练真实框前向）
5. 残留扫描：`aid_source|prepare_aid|AID_ROOT|dataset\.aid|AID Data Set` 零命中（排除 docs/ 与 AID 数据目录）
6. 全程不联网下载