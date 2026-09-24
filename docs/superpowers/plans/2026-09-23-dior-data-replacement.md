# DIOR 真实检测数据替换 AID 占位数据 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 删除 AID 场景分类占位链路，将项目数据工程与自测切换为 DIOR 真实目标检测数据（官方 train/val/test 划分 + 真实水平框），自测含抽样快速训练验证（不下载权重）。

**Architecture:** 新增 `rsdet/dior_source.py`（DIOR 根解析、官方划分读取、20 类按官方 train.txt 首现序发现、全量 HBB VOC→YOLO 转换、每类抽样快速训练子集、双 data.yaml）与 `scripts/prepare_dior.py`（CLI）。重写 `scripts/self_test.py` 为 6 步：准备 → data.yaml → 真实框校验 → 抽样从零训练 → 真实图推理 → 报告。删除 AID 链路（aid_source/prepare_aid/dataset.aid/AID_ROOT）。复用 `converter.voc_to_yolo`、`train_service.write_data_yaml/train`、`validator.validate_dataset/summarize`。

**Tech Stack:** Python 3.10、ultralytics 8.4.159、torch 2.5.0+cu124、opencv、numpy。DIOR 本地数据（已核实）：`DIOR/OpenDataLab___DIOR/raw/DIOR/DIOR/`。

**执行环境事实（重要）：**
- 本项目**不是 git 仓库**，所有计划的 commit 步骤改为「无需提交」。
- 本项目**无 pytest 框架**，检验范式为 `python -m py_compile` + 即时 `python -c` 断言 + `scripts/self_test.py` 端到端自测，不引入新测试框架。
- DIOR 数据事实（已核实）：HBB 标注 `Annotations/Horizontal Bounding Boxes/*.xml` 23463 个；图片 `JPEGImages-trainval/` 11725 + `JPEGImages-test/` 11738 = 23463 张 800×800；官方划分 `ImageSets/Main/{train,val,test}.txt`（train 5862 / val 5863 / test 11738）；20 类 192,518 真实框，类名含 `Expressway-Service-area` 等连字符混合大小写。
- `converter.py::voc_to_yolo(xml_dir, label_out_dir, class_map)` 返回 `{images, boxes}`；`train_service.write_data_yaml(out, train_dir, val_dir, classes)` → Path；`train_service.train(cfg, data_yaml, out_run_dir, model_name, epochs, imgsz, device, pretrained, cls_pw, **extra)` → summary（含 best_weight/mAP50）；`validator.validate_dataset(image_dir, label_dir, classes)` 返回报告 dict。
- 配置优先级：DEFAULT → YAML → CLI。
- 设计文档：`docs/superpowers/specs/2026-09-23-dior-data-replacement-design.md`。

---

## 文件结构总览

| 文件 | 责任 | 动作 |
|---|---|---|
| `rsdet/dior_source.py` | DIOR 根解析、官方划分、类发现、全量转换+抽样、双 data.yaml | **新增** |
| `scripts/prepare_dior.py` | DIOR 数据准备 CLI（对称替代 prepare_aid） | **新增** |
| `rsdet/paths.py` | `AID_ROOT` → `DIOR_ROOT` | **修改** |
| `rsdet/config.py` | `dataset.aid` → `dataset.dior`；`selftest` 段精简为 `epochs: 8` | **修改** |
| `configs/config.yaml` | `dataset.aid` → `dataset.dior`；`selftest` 段精简为 `epochs: 8` | **修改** |
| `scripts/self_test.py` | 重写为 DIOR 6 步自测（含快速训练） | **修改** |
| `rsdet/dataset_sources.py` | 第 125 行提示语 prepare_aid → prepare_dior | **修改** |
| `README.md` | 数据集说明/目录/自测流程/配置；删除"AID 语义边界"整节 | **修改** |
| `rsdet/aid_source.py` | AID 占位链路 | **删除** |
| `scripts/prepare_aid.py` | AID CLI | **删除** |
| `data/processed/{train,val,data.yaml,classes.txt}` | 旧 AID 直接产物（新布局迁入 `data/processed/dior/`） | **删除** |

依赖顺序：Task 1（paths/config）→ Task 2（dior_source）→ Task 3（prepare_dior CLI）→ Task 4（self_test 重写，依赖 1-3 + 真实数据验证）→ Task 5（删 AID + 旧产物）→ Task 6（dataset_sources 提示语）→ Task 7（README）→ Task 8（全链路终验）。

---

### Task 1: `rsdet/paths.py` + `rsdet/config.py` + `configs/config.yaml`（AID → DIOR）

**Files:**
- Modify: `rsdet/paths.py:19-31`（AID_ROOT 段）
- Modify: `rsdet/config.py:34-47`（DEFAULTS dataset + selftest 段）
- Modify: `configs/config.yaml:14-43`（dataset + selftest 段）

- [ ] **Step 1: 修改 `rsdet/paths.py`**

将 `rsdet/paths.py` 中：

```python
# AID 真实遥感数据集根目录 (30 个场景类文件夹所在处)
AID_ROOT = (ROOT / "AID Data Set" / "data" / "AID Data Set"
            / "AID Data Set" / "AID" / "AID_dataset" / "AID")
```

替换为：

```python
# DIOR 真实遥感检测数据集根目录 (官方划分: ImageSets/Main/{train,val,test}.txt)
DIOR_ROOT = (ROOT / "DIOR" / "OpenDataLab___DIOR" / "raw"
             / "DIOR" / "DIOR")
```

- [ ] **Step 2: 修改 `rsdet/config.py` DEFAULTS**

将 `rsdet/config.py` DEFAULTS 中 `"dataset"` 段：

```python
    "dataset": {
        "aid": {
            "root": None,              # None = 用 paths.AID_ROOT 默认路径; 可填绝对路径覆盖
            "val_ratio": 0.2,          # 按类划分验证集比例
            "seed": 42,                # 固定划分随机种子 (可复现)
            "limit_per_class": None,   # 每类抽样上限 (None = 全量)
        },
    },
    "selftest": {
        "mAP50_threshold": 0.5,
        "min_detections": 1,
        "epochs": 150,
    },
```

替换为：

```python
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
```

- [ ] **Step 3: 修改 `configs/config.yaml`**

将 `configs/config.yaml` 中：

```yaml
# AID 真实遥感数据集 (场景分类 30 类; 整幅场景框仅作链路验证占位真值)
dataset:
  aid:
    root: null                     # null=默认 auto 定位到 "AID Data Set/.../AID_dataset/AID"; 可填绝对路径
    val_ratio: 0.2                 # 按类划分验证集比例
    seed: 42                       # 划分随机种子
    limit_per_class: null          # 每类抽样上限 (null=全量)
```

替换为：

```yaml
# DIOR 真实遥感检测数据集 (20 类 / 23463 张 / 19 万真实水平框; 官方 train/val/test 划分)
dataset:
  dior:
    root: null                     # null=默认 auto 定位到 "DIOR/OpenDataLab___DIOR/raw/DIOR/DIOR"; 可填绝对路径
    sample_train_per_class: 16     # 快速训练子集: 每类从 train 抽几张
    sample_val_per_class: 8        # 快速训练子集: 每类从 val 抽几张
    seed: 42                       # 抽样随机种子
```

并将 `configs/config.yaml` 的 `selftest` 段：

```yaml
# 检验自测阈值
selftest:
  mAP50_threshold: 0.5           # 合成数据 mAP50 达标线 (离线从零训练)
  min_detections: 1              # 切片推理至少需产出的检测框数
  epochs: 150                    # 自测训练轮数 (离线从零训练需较多轮次收敛)
```

替换为：

```yaml
# 检验自测
selftest:
  epochs: 8                      # 快速训练轮数 (抽样训练, 不做达标阈值)
```

- [ ] **Step 4: 验证配置加载**

运行：

```powershell
python -X utf8 -c "import sys; sys.path.insert(0,'.'); from rsdet.config import get_config; c=get_config(); print('dior.root:', c.get('dataset.dior.root')); print('sample_train:', c.get('dataset.dior.sample_train_per_class')); print('selftest.epochs:', c.get('selftest.epochs')); print('aid removed:', c.get('dataset.aid.root', 'GONE')); print('old thr removed:', c.get('selftest.mAP50_threshold', 'GONE'))"
```

预期输出：`dior.root: None`、`sample_train: 16`、`selftest.epochs: 8`、`aid removed: GONE`、`old thr removed: GONE`。

- [ ] **Step 5: 无需提交**

---

### Task 2: 新增 `rsdet/dior_source.py`

**Files:**
- Create: `rsdet/dior_source.py`

- [ ] **Step 1: 写入完整模块**

创建 `rsdet/dior_source.py`，内容如下（完整代码，可直接使用）：

```python
# -*- coding: utf-8 -*-
"""DIOR 真实遥感检测数据集接入: 官方 train/val/test 划分 -> YOLO 水平框格式。

DIOR (Detecting Objects in Remote sensing Images):
  - 20 类 / 23,463 张 800x800 / 192,518 个真实水平框
  - VOC XML 标注 (Annotations/Horizontal Bounding Boxes/*.xml)
  - 官方划分 (ImageSets/Main/{train,val,test}.txt)

设计文档: docs/superpowers/specs/2026-09-23-dior-data-replacement-design.md
"""
from __future__ import annotations

import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

from . import paths
from .config import get_config
from .converter import voc_to_yolo
from .train_service import write_data_yaml

HBB_REL = Path("Annotations") / "Horizontal Bounding Boxes"
IMAGESET_REL = Path("ImageSets") / "Main"
TRAINVAL_IMAGES = "JPEGImages-trainval"
TEST_IMAGES = "JPEGImages-test"
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def resolve_dior_root(root: str | Path | None = None) -> Path:
    """解析 DIOR 数据集根目录: CLI 参数 > config(dataset.dior.root) > paths.DIOR_ROOT。

    返回的目录必须存在, 否则抛错。
    """
    if root is not None:
        p = paths.resolve(str(root))
    else:
        cfg_root = get_config().get("dataset.dior.root", None)
        p = paths.resolve(str(cfg_root)) if cfg_root else paths.DIOR_ROOT
    if not p.is_dir():
        raise FileNotFoundError(
            f"DIOR 数据集目录不存在: {p}。请检查 configs/config.yaml 的 "
            f"dataset.dior.root, 或确认 'DIOR/OpenDataLab___DIOR/raw/DIOR/DIOR' 位于项目内。")
    return p


def _hbb_dir(dior_root: Path) -> Path:
    hbb = dior_root / HBB_REL
    if not hbb.is_dir():
        raise FileNotFoundError(f"DIOR HBB 标注目录缺失: {hbb}")
    return hbb


def _image_sets_dir(dior_root: Path) -> Path:
    d = dior_root / IMAGESET_REL
    if not d.is_dir():
        raise FileNotFoundError(f"DIOR ImageSets 目录缺失: {d}")
    return d


def parse_split(split_file: str | Path) -> list[str]:
    """读取官方划分 txt, 返回图像 stem 列表 (不含扩展名)。"""
    p = Path(split_file)
    if not p.exists():
        raise FileNotFoundError(f"DIOR 官方划分文件缺失: {p}")
    return [line.strip() for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def discover_classes(dior_root: str | Path) -> list[str]:
    """按官方 train.txt 首个出现序返回类别名 (保证类 id 与官方一致)。"""
    root = Path(dior_root)
    hbb = _hbb_dir(root)
    train_stems = parse_split(_image_sets_dir(root) / "train.txt")
    seen: list[str] = []
    in_seen: set[str] = set()
    for stem in train_stems:
        xml = hbb / f"{stem}.xml"
        if not xml.exists():
            continue
        try:
            root_el = ET.parse(xml).getroot()
        except Exception:
            continue
        for obj in root_el.findall("object"):
            name_el = obj.find("name")
            n = name_el.text if name_el is not None and name_el.text else ""
            if n and n not in in_seen:
                in_seen.add(n)
                seen.append(n)
    if not seen:
        raise ValueError(f"DIOR HBB 标注中未发现任何类别: {hbb}")
    return seen


def _locate_image(dior_root: Path, stem: str) -> Path:
    for sub in (TRAINVAL_IMAGES, TEST_IMAGES):
        d = dior_root / sub
        if not d.is_dir():
            continue
        for ext in IMAGE_EXTS:
            f = d / f"{stem}{ext}"
            if f.exists():
                return f
    raise FileNotFoundError(f"DIOR 图片缺失: {stem} (trainval/test 均未找到)")


def _stem_classes(hbb_dir: Path, stems: list[str], class_set: set[str]) -> dict[str, set[str]]:
    """stems -> 该图包含的类集合 (读 XML object/name 并过滤到 class_set)。"""
    result: dict[str, set[str]] = {}
    for stem in stems:
        xml = hbb_dir / f"{stem}.xml"
        names: set[str] = set()
        if xml.exists():
            try:
                root_el = ET.parse(xml).getroot()
                for obj in root_el.findall("object"):
                    name_el = obj.find("name")
                    n = name_el.text if name_el is not None and name_el.text else ""
                    if n in class_set:
                        names.add(n)
            except Exception:
                pass
        result[stem] = names
    return result


def _subsample(stems: list[str], n: int, rng: random.Random) -> list[str]:
    return stems if len(stems) <= n else rng.sample(stems, n)


def prepare_dior_dataset(dior_root: str | Path, out_base: str | Path,
                         classes: list[str] | None = None,
                         sample_train_per_class: int = 16,
                         sample_val_per_class: int = 8,
                         seed: int = 42) -> dict:
    """把 DIOR 官方划分转成 YOLO 水平框格式, 并生成快速训练抽样子集。

    输出 (out_base 即 data/processed/dior):
        train|val|test/images|labels   官方划分全量
        sample/train|val/images|labels 每类抽样 (供快速训练)
        classes.txt  data.yaml(全量) + sample/data.yaml(抽样)

    返回统计 dict: classes/split_counts/boxes_total/sample_counts/
                   data_yaml_full/data_yaml_sample
    """
    root = Path(dior_root)
    out = Path(out_base)
    hbb = _hbb_dir(root)
    image_sets = _image_sets_dir(root)
    rng = random.Random(seed)

    if classes is None:
        classes = discover_classes(root)
    class_set = set(classes)
    class_map = {c: i for i, c in enumerate(classes)}

    # 幂等清空
    shutil.rmtree(out, ignore_errors=True)
    for split in ("train", "val", "test"):
        (out / split / "images").mkdir(parents=True, exist_ok=True)
        (out / split / "labels").mkdir(parents=True, exist_ok=True)

    # 1) 一次性转换全部 HBB -> 暂存标签 (voc_to_yolo 输出按 xml stem 命名)
    staging = out / "_labels_all"
    conv = voc_to_yolo(hbb, staging, class_map)

    # 2) 按官方划分分发图片 + 标签
    split_counts: dict[str, int] = {}
    for split in ("train", "val", "test"):
        stems = parse_split(image_sets / f"{split}.txt")
        n = 0
        for stem in stems:
            img = _locate_image(root, stem)
            shutil.copy2(img, out / split / "images" / img.name)
            lab = staging / f"{stem}.txt"
            if lab.exists():
                shutil.copy2(lab, out / split / "labels" / f"{stem}.txt")
            n += 1
        split_counts[split] = n

    # 3) 抽样子集: 每类从对应划分取前 min(n, per) 张 (固定 seed)
    train_stems = parse_split(image_sets / "train.txt")
    val_stems = parse_split(image_sets / "val.txt")
    stem_classes = _stem_classes(hbb, train_stems + val_stems, class_set)
    samples = {"train": {}, "val": {}}
    for split, per in (("train", sample_train_per_class), ("val", sample_val_per_class)):
        stems = train_stems if split == "train" else val_stems
        (out / "sample" / split / "images").mkdir(parents=True, exist_ok=True)
        (out / "sample" / split / "labels").mkdir(parents=True, exist_ok=True)
        for c in classes:
            cand = [s for s in stems if c in stem_classes.get(s, set())]
            picked = _subsample(cand, per, rng)
            samples[split][c] = len(picked)
            for stem in picked:
                img = _locate_image(root, stem)
                shutil.copy2(img, out / "sample" / split / "images" / img.name)
                lab = staging / f"{stem}.txt"
                if lab.exists():
                    shutil.copy2(lab, out / "sample" / split / "labels" / f"{stem}.txt")

    # 4) 类别文件 + 两张 data.yaml (全量 + 抽样)
    (out / "classes.txt").write_text("\n".join(classes) + "\n", encoding="utf-8")
    data_yaml_full = write_data_yaml(out / "data.yaml",
                                     out / "train" / "images",
                                     out / "val" / "images", classes)
    data_yaml_sample = write_data_yaml(out / "sample" / "data.yaml",
                                       out / "sample" / "train" / "images",
                                       out / "sample" / "val" / "images", classes)
    shutil.rmtree(staging, ignore_errors=True)

    return {
        "classes": classes,
        "split_counts": split_counts,
        "boxes_total": conv["boxes"],
        "sample_counts": samples,
        "data_yaml_full": str(data_yaml_full),
        "data_yaml_sample": str(data_yaml_sample),
    }
```

注意点：
- `voc_to_yolo(hbb, staging, class_map)` 一次性转换全部 23463 个 XML，输出标签按 stem 命名（`00001.txt`），随后按官方划分分发——只做一次 XML 解析。
- 图片全部在 `JPEGImages-trainval/`（train+val）或 `JPEGImages-test/`（test）下，`_locate_image` 按序查找。
- `_subsample` 用 `rng.sample`，同一 `rng` 贯穿所有类，结果由 `seed` 完全确定。
- 多类图（一个 stem 属多类）会在多个类的候选中出现，`shutil.copy2` 幂等覆盖同名文件，无副作用。
- `write_data_yaml` 自动求 train/val 共同父目录：全量 → `path=dior`, `train=train/images`；抽样 → `path=dior/sample`, `train=train/images`。

- [ ] **Step 2: 冒烟验证（快速检查，不跑全量转换）**

运行：

```powershell
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
python -X utf8 -c "
import sys, tempfile, yaml; sys.path.insert(0,'.')
from pathlib import Path
import rsdet.dior_source as d
root = d.resolve_dior_root()
print('DIOR_ROOT:', root)
classes = d.discover_classes(root)
print('classes:', len(classes))
print(classes)
tr = d.parse_split(Path(root)/'ImageSets'/'Main'/'train.txt')
va = d.parse_split(Path(root)/'ImageSets'/'Main'/'val.txt')
te = d.parse_split(Path(root)/'ImageSets'/'Main'/'test.txt')
print('split counts:', len(tr), len(va), len(te))
img = d._locate_image(Path(root), tr[0])
print('first train image:', img)
"
```

预期输出：`DIOR_ROOT: ...raw/DIOR/DIOR`、`classes: 20` 及 20 个类名（含 `Expressway-Service-area`）、`split counts: 5862 5863 11738`、`first train image: ...JPEGImages-trainval/00001.jpg`。

- [ ] **Step 3: 无需提交**

---

### Task 3: 新增 `scripts/prepare_dior.py` CLI

**Files:**
- Create: `scripts/prepare_dior.py`

- [ ] **Step 1: 写入完整 CLI**

创建 `scripts/prepare_dior.py`，内容如下：

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从本机 DIOR 数据集构建 YOLO 检测格式数据 (官方划分 + 真实水平框 + 抽样子集)。

用法:
    python scripts/prepare_dior.py
    python scripts/prepare_dior.py --root <DIOR根> --out data/processed/dior \
        --sample-train-per-class 16 --sample-val-per-class 8 --seed 42
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rsdet import paths
from rsdet.dior_source import (discover_classes, prepare_dior_dataset,
                               resolve_dior_root)


def main() -> None:
    ap = argparse.ArgumentParser(description="从 DIOR 数据集生成 YOLO 检测格式数据")
    ap.add_argument("--root", default=None,
                    help="DIOR 根目录 (默认: config[dataset.dior.root], 再回退 paths.DIOR_ROOT)")
    ap.add_argument("--out", default=None, help="输出目录 (默认 data/processed/dior)")
    ap.add_argument("--classes", nargs="+", default=None,
                    help="类别白名单 (默认自动发现, 官方序)")
    ap.add_argument("--sample-train-per-class", type=int, default=None,
                    help="抽样子集: 每类从 train 取几张")
    ap.add_argument("--sample-val-per-class", type=int, default=None,
                    help="抽样子集: 每类从 val 取几张")
    ap.add_argument("--seed", type=int, default=None, help="抽样随机种子")
    ap.add_argument("--dry-run", action="store_true", help="仅打印计划, 不写文件")
    args = ap.parse_args()

    from rsdet.config import get_config
    cfg = get_config()
    out = Path(args.out) if args.out else paths.DATA_PROCESSED / "dior"
    st = args.sample_train_per_class
    if st is None:
        st = int(cfg.get("dataset.dior.sample_train_per_class", 16))
    sv = args.sample_val_per_class
    if sv is None:
        sv = int(cfg.get("dataset.dior.sample_val_per_class", 8))
    seed = args.seed if args.seed is not None else int(cfg.get("dataset.dior.seed", 42))

    root = resolve_dior_root(args.root)
    classes = args.classes or discover_classes(root)
    print(f"DIOR 数据源 : {root}")
    print(f"类别数     : {len(classes)}")
    print(f"输出目录   : {out}")
    print(f"sample train/val per class: {st}/{sv} | seed={seed}")
    if args.dry_run:
        print("(--dry-run) 未写入任何文件。")
        return

    stats = prepare_dior_dataset(root, out, classes=classes,
                                 sample_train_per_class=st,
                                 sample_val_per_class=sv, seed=seed)
    print("=" * 60)
    print("数据准备完成:")
    print(f"  train/val/test 图片: {stats['split_counts']}")
    print(f"  真实框总数         : {stats['boxes_total']}")
    print(f"  sample train 图片  : {sum(stats['sample_counts']['train'].values())}")
    print(f"  sample val 图片    : {sum(stats['sample_counts']['val'].values())}")
    print(f"  全量 data.yaml     : {stats['data_yaml_full']}")
    print(f"  抽样 data.yaml     : {stats['data_yaml_sample']}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 验证 `--help` 与 `--dry-run`**

运行：

```powershell
python scripts/prepare_dior.py --help
python scripts/prepare_dior.py --dry-run
```

预期：`--help` 打印全部参数；`--dry-run` 打印 DIOR 根路径、20 类、输出目录、sample 参数，并显示 `(--dry-run) 未写入任何文件。`。

- [ ] **Step 3: 全量真实转换（核心验证，复制 23463 张图，约 3-6 分钟）**

运行：

```powershell
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
python -X utf8 scripts/prepare_dior.py
```

预期输出：
- `train/val/test 图片: {'train': 5862, 'val': 5863, 'test': 11738}`
- `真实框总数: 192518`（与官方 192,472 微量差异属标注容差/解析差异，次数级一致即可）
- `sample train 图片: 320`（20 类 × 16）、`sample val 图片: 160`（20 类 × 8）
- 两张 data.yaml 路径打印

- [ ] **Step 4: 抽样与全量产物断言**

运行：

```powershell
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
python -X utf8 -c "
import sys, yaml; sys.path.insert(0,'.')
from pathlib import Path
out = Path('data/processed/dior')
print('train imgs:', len(list((out/'train'/'images').glob('*.jpg'))), 'labels:', len(list((out/'train'/'labels').glob('*.txt'))))
print('val imgs:', len(list((out/'val'/'images').glob('*.jpg'))))
print('test imgs:', len(list((out/'test'/'images').glob('*.jpg'))))
print('sample tr imgs:', len(list((out/'sample'/'train'/'images').glob('*.jpg'))))
print('sample va imgs:', len(list((out/'sample'/'val'/'images').glob('*.jpg'))))
d = yaml.safe_load((out/'data.yaml').read_text(encoding='utf-8'))
print('full data.yaml names:', len(d['names']), '| train:', d['train'], '| val:', d['val'])
s = yaml.safe_load((out/'sample'/'data.yaml').read_text(encoding='utf-8'))
print('sample data.yaml path:', s['path'].endswith('dior/sample'))
lbl = list((out/'train'/'labels').glob('00001.txt'))
print('sample label 00001:', lbl[0].read_text().strip() if lbl else 'MISSING')
"
```

预期输出：`train imgs: 5862 labels: 5862`、`val imgs: 5863`、`test imgs: 11738`、`sample tr imgs: 320`、`sample va imgs: 160`、`full data.yaml names: 20`、`sample data.yaml path: True`、`sample label 00001: 8 0.072500 0.532500 0.610000 0.507500`（golffield→类 id 与相同值即可，数字可能不同）。

- [ ] **Step 5: 无需提交**

---

### Task 4: 重写 `scripts/self_test.py`（DIOR 6 步自测，含快速训练）

**Files:**
- Modify: `scripts/self_test.py`（整文件重写）

- [ ] **Step 1: 整文件重写**

将 `scripts/self_test.py` 全部内容替换为：

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""端到端检验自测 (DIOR 真实检测数据, 含快速训练)。

流程:
  1. DIOR 官方划分 -> YOLO 水平框全量转换 + 每类抽样快速训练子集
  2. 生成 data.yaml (全量 + 抽样) 与 classes.txt
  3. 数据校验 (真实框: 图像-标注成对 / 框合法 / 类别合法)
  4. 抽样子集快速训练 (yolov8n.yaml 从零训练, 不下载权重)
  5. 训练后权重对真实验证图推理 (断言前向无异常)
  6. 报告 + PASS/FAIL

用法:
    python scripts/self_test.py
    python scripts/self_test.py --epochs 8 --device 0
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rsdet import paths
from rsdet.config import get_config
from rsdet.dior_source import (discover_classes, prepare_dior_dataset,
                               resolve_dior_root)
from rsdet.logging_utils import setup_logger
from rsdet.report import write_markdown_report
from rsdet.system_utils import format_gpu_summary, normalize_device
from rsdet.train_service import train
from rsdet.validator import summarize, validate_dataset

log = setup_logger("rsdet.selftest", paths.RUNS_DIR / "self_test.log")

CHECK, CROSS = "[PASS]", "[FAIL]"


def main() -> None:
    cfg = get_config()
    ap = argparse.ArgumentParser(description="端到端检验自测 (DIOR 真实检测数据)")
    ap.add_argument("--epochs", type=int, default=None, help="快速训练轮数")
    ap.add_argument("--imgsz", type=int, default=None, help="训练输入尺寸")
    ap.add_argument("--device", default=None, help="auto/cpu/0")
    args = ap.parse_args()

    epochs = args.epochs if args.epochs is not None else int(cfg.get("selftest.epochs", 8))
    imgsz = args.imgsz if args.imgsz is not None else int(cfg.get("model.imgsz", 416))
    device = args.device or cfg.get("model.device", "auto")
    device = normalize_device(device)

    dior_root = resolve_dior_root()
    classes = discover_classes(dior_root)
    n_cls = len(classes)
    dior_dir = paths.DATA_PROCESSED / "dior"
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = paths.RUNS_DIR / f"selftest_{stamp}"

    print("=" * 70)
    print("遥感影像目标检测系统 (YOLO-OB) —— 端到端自测 (DIOR 真实检测数据)")
    print("=" * 70)
    print(format_gpu_summary())
    print(f"DIOR 数据源: {dior_root}")
    print(f"类别: {n_cls} 类 | epochs={epochs} | imgsz={imgsz} | device={device}\n")

    steps: list[dict] = []
    t0 = time.time()

    # ---- 1. DIOR 数据准备 (全量 + 抽样) ----
    st_per = int(cfg.get("dataset.dior.sample_train_per_class", 16))
    sv_per = int(cfg.get("dataset.dior.sample_val_per_class", 8))
    seed = int(cfg.get("dataset.dior.seed", 42))
    stats = prepare_dior_dataset(dior_root, dior_dir, classes=classes,
                                 sample_train_per_class=st_per,
                                 sample_val_per_class=sv_per, seed=seed)
    all_splits_ok = all(v > 0 for v in stats["split_counts"].values())
    sample_ok = all(len(v) > 0 for v in stats["sample_counts"].values())
    steps.append({
        "name": "DIOR官方划分->真实框转换+抽样子集",
        "ok": all_splits_ok and sample_ok,
        "detail": f"{n_cls} 类 / train={stats['split_counts'].get('train')} "
                  f"val={stats['split_counts'].get('val')} "
                  f"test={stats['split_counts'].get('test')} / "
                  f"真实框={stats['boxes_total']} / sample "
                  f"tr={sum(stats['sample_counts']['train'].values())} "
                  f"va={sum(stats['sample_counts']['val'].values())}"})

    # ---- 2. data.yaml + classes.txt ----
    full_yaml = Path(stats["data_yaml_full"])
    sample_yaml = Path(stats["data_yaml_sample"])
    classes_txt = dior_dir / "classes.txt"
    steps.append({
        "name": "生成 data.yaml/classes.txt",
        "ok": full_yaml.exists() and sample_yaml.exists() and classes_txt.exists(),
        "detail": f"全量={full_yaml.name} 抽样={sample_yaml.name} 类数={n_cls}"})

    # ---- 3. 数据校验 (真实框) ----
    val_report = validate_dataset(dior_dir / "sample" / "train" / "images",
                                  dior_dir / "sample" / "train" / "labels", classes)
    val_ok = val_report["ok"] and val_report["images_with_labels"] > 0
    steps.append({"name": "数据校验 (真实框成对/合法)",
                  "ok": val_ok,
                  "detail": summarize(val_report, classes)})

    # ---- 4. 抽样快速训练 (从零, 不下载权重) ----
    train_ok = False
    best_weight = None
    mAP50 = 0.0
    train_detail = ""
    try:
        summary = train(cfg, sample_yaml, run_dir / "train_run",
                        model_name="yolov8n", epochs=epochs, imgsz=imgsz,
                        device=device, pretrained=False, cls_pw=0.0, amp=False)
        train_ok = True
        best_weight = summary["best_weight"]
        mAP50 = summary["mAP50"]
        train_detail = f"mAP50={mAP50:.4f} | best={best_weight}"
    except Exception as e:
        log.exception("快速训练失败")
        train_detail = f"{type(e).__name__}: {e}"
    steps.append({"name": "抽样快速训练 (yolov8n 从零)",
                  "ok": train_ok, "detail": train_detail})

    # ---- 5. 真实图推理 ----
    infer_ok = False
    infer_detail = ""
    val_imgs = sorted((dior_dir / "sample" / "val" / "images").glob("*.jpg"))
    try:
        from ultralytics import YOLO
        if train_ok and best_weight:
            model = YOLO(str(best_weight))
            src = str(val_imgs[0])
        else:
            model = YOLO("yolov8n.yaml")  # 兜底: 离线建图冒烟, 不下载
            src = str(val_imgs[0])
        res = model.predict(source=src, device=device, verbose=False)[0]
        n_det = len(res.boxes) if res.boxes is not None else 0
        infer_ok = True
        infer_detail = f"{Path(src).name} | 前向完成 | 检测框={n_det} | device={device}"
    except Exception as e:
        log.exception("真实图推理失败")
        infer_detail = f"{type(e).__name__}: {e}"
    steps.append({"name": "真实图推理 (训练权重/离线建图)",
                  "ok": infer_ok, "detail": infer_detail})

    # ---- 6. 汇总 ----
    all_ok = all(s["ok"] for s in steps)
    elapsed = time.time() - t0
    print("\n" + "=" * 70)
    print("自测结果")
    print("=" * 70)
    for s in steps:
        print(f"  {CHECK if s['ok'] else CROSS} {s['name']}")
        if s.get("detail"):
            print(f"      -> {s['detail']}")
    print("-" * 70)
    verdict = "PASS" if all_ok else "FAIL"
    print(f"  总耗时 {elapsed:.0f}s | 最终结论: {verdict}")

    report_lines = [f"- 结论: **{verdict}**", f"- 总耗时: {elapsed:.0f}s", ""]
    for s in steps:
        report_lines.append(f"- {'✅' if s['ok'] else '❌'} {s['name']}: {s.get('detail', '')}")
    report_path = write_markdown_report(paths.REPORTS_DIR / "self_test_report.md",
                                        "遥感目标检测系统自测报告 (DIOR 真实数据)",
                                        report_lines)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "self_test.json").write_text(json.dumps({
        "verdict": verdict, "elapsed_s": elapsed, "steps": steps,
        "classes": classes, "mAP50": mAP50, "dior_root": str(dior_root),
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n报告: {report_path}")
    print(f"详细: {run_dir / 'self_test.json'}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
```

要点：
- 步骤 4 训练用 `pretrained=False` → `train_service` 内部 `model_src="yolov8n.yaml"` 从零建图，**不下载权重**；`amp=False` 关闭 AMP 自检以避免离线时尝试下载参考权重；`cls_pw=0.0` 禁用类别加权（20 类数量极度不均，快速训练不做平衡）。
- 步骤 5 正常用训练权重；即使训练失败也回退 `YOLO("yolov8n.yaml")` 冒烟，保证结构链路可验证。

- [ ] **Step 2: 完整自测运行（核心验收，复制全量图 + 快速训练，约 8-15 分钟）**

运行：

```powershell
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
python -X utf8 scripts/self_test.py --device 0
```

预期输出：6 步全部 `[PASS]`、`最终结论: PASS`、`self_test_report.md` 更新、`runs/selftest_*/self_test.json` 落盘、mAP50 打印。全程无权重下载。

- [ ] **Step 3: 无需提交**

---

### Task 5: 删除 AID 链路代码与旧产物

**Files:**
- Delete: `rsdet/aid_source.py`
- Delete: `scripts/prepare_aid.py`
- Delete: `data/processed/train/`、`data/processed/val/`（旧 AID 直接产物）
- Delete: `data/processed/data.yaml`、`data/processed/classes.txt`（旧 AID 直接产物）

- [ ] **Step 1: 删除 AID 代码与旧产物**

运行：

```powershell
Remove-Item -LiteralPath "rsdet/aid_source.py" -Force
Remove-Item -LiteralPath "scripts/prepare_aid.py" -Force
Remove-Item -LiteralPath "data/processed/train" -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item -LiteralPath "data/processed/val" -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item -LiteralPath "data/processed/data.yaml" -Force -ErrorAction SilentlyContinue
Remove-Item -LiteralPath "data/processed/classes.txt" -Force -ErrorAction SilentlyContinue
Write-Output "aisrc: $(Test-Path 'rsdet/aid_source.py') | prepaid: $(Test-Path 'scripts/prepare_aid.py') | oldtrain: $(Test-Path 'data/processed/train')"
```

预期：`aisrc: False | prepaid: False | oldtrain: False`。

- [ ] **Step 2: 确认 DIOR 产物保留完好**

运行：`Test-Path "data/processed/dior/data.yaml"` → 预期 `True`（Task 3 生成）。

- [ ] **Step 3: 无需提交**

---

### Task 6: 更新 `rsdet/dataset_sources.py` 提示语

**Files:**
- Modify: `rsdet/dataset_sources.py:125`

- [ ] **Step 1: 更新提示行**

将 `rsdet/dataset_sources.py` 第 125 行：

```python
print("[dataset] 提示: 离线环境下可用本机 AID 真实数据: python scripts/prepare_aid.py")
```

替换为：

```python
print("[dataset] 提示: 离线环境下可用本机 DIOR 真实检测数据: python scripts/prepare_dior.py")
```

- [ ] **Step 2: 验证**

运行：`python -X utf8 -c "import sys; sys.path.insert(0,'.'); import rsdet.dataset_sources as d; assert 'prepare_aid' not in open(d.__file__, encoding='utf-8').read(); assert 'prepare_dior' in open(d.__file__, encoding='utf-8').read(); print('OK')"` → 预期 `OK`。

- [ ] **Step 3: 无需提交**

---

### Task 7: 更新 `README.md`

**Files:**
- Modify: `README.md`

- [ ] **Step 1: 功能特性·权威数据集**

将：

```markdown
- **权威数据集**：内置 DOTA v2.0 / DIOR / NWPU VHR-10 / HRSC2016 元数据与自动下载脚本；本地 AID 遥感影像数据集（30 类场景、约 1 万张 600×600）可直接用于数据工程链路验证。
```

替换为：

```markdown
- **权威数据集**：内置 DOTA v2.0 / DIOR / NWPU VHR-10 / HRSC2016 元数据与自动下载脚本；本地 DIOR 真实检测数据（20 类、约 2.3 万张 800×800、19 万真实水平框，官方 train/val/test 划分）可直接用于数据工程与快速训练验证。
```

- [ ] **Step 2: 目录结构**

将：

```markdown
│   ├── prepare_aid.py           # 从本机 AID 数据集生成 YOLO 检测格式数据
```

替换为：

```markdown
│   ├── prepare_dior.py          # 从本机 DIOR 数据集生成 YOLO 检测格式数据
```

- [ ] **Step 3: 快速开始·自测**

将：

```markdown
自测会从本机 AID 真实遥感数据集划分 train/val → 生成整幅场景框占位标注与 data.yaml → 数据校验 → `yolov8n.yaml` 结构冒烟推理（离线建图、不训练、不下载权重），最终输出 `PASS/FAIL`。
```

替换为：

```markdown
自测会从本机 DIOR 真实检测数据按官方划分转换 train/val/test → 生成 data.yaml → 数据校验 → `yolov8n.yaml` 抽样快速训练（不下载权重）→ 训练权重真实图推理，最终输出 `PASS/FAIL`。
```

- [ ] **Step 4: 离线提示**

将：

```markdown
> 这些数据集多托管于 Google Drive / Baidu，本机若无法直连，`download_dataset.py` 会打印官方主页与手工获取指引。离线环境下可使用本机 AID 真实遥感数据集验证数据工程链路：`python scripts/prepare_aid.py`（生成 YOLO 格式数据）→ `python scripts/self_test.py`（最小检验）。
```

替换为：

```markdown
> 这些数据集多托管于 Google Drive / Baidu，本机若无法直连，`download_dataset.py` 会打印官方主页与手工获取指引。离线环境下可使用本机 DIOR 真实检测数据：`python scripts/prepare_dior.py`（生成 YOLO 格式数据）→ `python scripts/self_test.py`（自测含快速训练）。
```

- [ ] **Step 5: 主要配置**

将：

```markdown
`configs/config.yaml` 中可调整：模型名/输入尺寸/轮数、切片尺寸/重叠比例、AID 数据集划分（`dataset.aid`：root/val_ratio/seed/limit_per_class）等。配置加载遵循 **DEFAULT → YAML → CLI 参数** 优先级。
```

替换为：

```markdown
`configs/config.yaml` 中可调整：模型名/输入尺寸/轮数、切片尺寸/重叠比例、DIOR 数据集（`dataset.dior`：root/sample_train_per_class/sample_val_per_class/seed）等。配置加载遵循 **DEFAULT → YAML → CLI 参数** 优先级。
```

- [ ] **Step 6: 删除「AID 真实数据集与语义边界」整节**

删除以下整节（含标题与三个要点）：

```markdown
## AID 真实数据集与语义边界

- AID（Aerial Image Dataset）为**场景分类**数据集：30 类、每类 220~420 张 600×600 遥感影像，仅提供整图场景类别，**无目标框标注**。
- 本系统在检测框架下使用 AID 时，为每张图生成一条「覆盖全图的场景框」占位真值（`cls 0.5 0.5 1.0 1.0`），目的是让数据工程/校验/推理链路在真实遥感影像上完整跑通。
- ⚠️ 该占位标签**不是 AID 官方检测标注**，据此训练的检测模型在 mAP 上无参考价值；需要有意义的检测模型请改用 DOTA / DIOR / NWPU-VHR-10（见 `scripts/data_converter.py`）。
```

替换为空（该节直接移除；`## 环境要求` 变为紧跟 `## 主要配置` 之后）。

- [ ] **Step 7: 验证 README 无残留**

运行：

```powershell
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
Select-String -Path README.md -Pattern 'AID|aid_source|prepare_aid|AID_ROOT|语义边界|全景|占位' | ForEach-Object { "$($_.LineNumber): $($_.Line.Trim())" }
```

预期：无匹配输出。

- [ ] **Step 8: 无需提交**

---

### Task 8: 全链路最终验证（关键验收）

**Files:**
- Verify: 全部改动文件
- 运行: `python -m py_compile` + `python scripts/self_test.py`

- [ ] **Step 1: 字节码编译检查**

运行：

```powershell
python -m py_compile rsdet/dior_source.py rsdet/paths.py rsdet/config.py rsdet/dataset_sources.py scripts/prepare_dior.py scripts/self_test.py
```

预期：`PY_COMPILE_OK`（用 `Write-Output "$(if ($?) {'PY_COMPILE_OK'})"` 确认）。

- [ ] **Step 2: 残留引用终检（AID 全清）**

运行：

```powershell
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$hits = Get-ChildItem -Recurse -File -Include *.py,*.md,*.yaml -Path . | Where-Object { $_.FullName -notmatch '\\(runs|weights|DIOR|AID Data Set|__pycache__|docs\\superpowers|\.venv|venv)\\' } | Select-String -Pattern 'aid_source|prepare_aid|AID_ROOT|dataset\.aid|AID Data Set' | ForEach-Object { "$($_.Path):$($_.LineNumber): $($_.Line.Trim())" }; if ($hits) { $hits } else { Write-Output "NO_AID_RESIDUAL" }
```

预期：`NO_AID_RESIDUAL`（docs/superpowers 下的设计与计划文档不参与实施残留判定）。

- [ ] **Step 3: 完整自测终跑（含快速训练）**

运行：

```powershell
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
python -X utf8 scripts/self_test.py --device 0
```

预期：6 步全 PASS、`最终结论: PASS`、报告落盘。全程不联网。

- [ ] **Step 4: 断言关键产物**

运行：

```powershell
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
python -X utf8 -c "
import sys, yaml; sys.path.insert(0,'.')
from pathlib import Path
out = Path('data/processed/dior')
d = yaml.safe_load((out/'data.yaml').read_text(encoding='utf-8'))
print('classes:', len(d['names']), '| first:', d['names'][0], '| last:', d['names'][19])
tr = len(list((out/'train'/'images').glob('*.jpg')))
va = len(list((out/'val'/'images').glob('*.jpg')))
te = len(list((out/'test'/'images').glob('*.jpg')))
print('splits:', tr, va, te)
assert tr == 5862 and va == 5863 and te == 11738
assert len(d['names']) == 20
print('ASSERT_OK')
"
```

预期：`classes: 20 | first: golffield | last: windmill`（首/末类名以实际为准）、`splits: 5862 5863 11738`、`ASSERT_OK`。

- [ ] **Step 5: 无需提交**

---

## 自审（对照设计文档）

**1. Spec 覆盖检查：**
- §3.1 模块函数签名（resolve_dior_root / discover_classes / parse_split / prepare_dior_dataset / _locate_image / _subsample）→ Task 2 全部实现 ✅
- §3.2 输出布局（train/val/test + sample + 双 data.yaml + classes.txt）→ Task 2 Step 1 + Task 3 Step 3/4 验证 ✅
- §3.3 配置（dataset.dior + selftest.epochs: 8；paths.DIOR_ROOT）→ Task 1 ✅
- §4 删除/新增/修改清单 → Task 1/2/3/4/5/6/7 全覆盖；`data/processed` 旧 AID 直接产物删除 → Task 5 ✅
- §5 自测 6 步（含快速训练 + mAP50 不设线 + 冒烟兜底）→ Task 4 ✅
- §6 错误处理（根/HBB/ImageSets 缺失、XML 损坏、抽样不足、训练异常）→ Task 2 `_hbb_dir`/`_image_sets_dir`/`parse_split` 抛错、`voc_to_yolo` 跳过损坏、`_subsample` min 兜底、Task 4 try/except 兜底 ✅
- §7 README 更新 + 删「AID 语义边界」节 → Task 7 ✅
- §8 成功标准（py_compile / prepare --help/--dry-run / 全量 5862-5863-11738 / self_test PASS / AID 残留零命中 / 不联网）→ Task 8 ✅

**2. 占位符扫描：** 全部步骤含完整代码/命令/预期输出，无 TBD/TODO/「相似于 Task N」表述 ✅

**3. 类型/命名一致性：**
- `resolve_dior_root(root=None)` 三处使用（Task 2/3/4）签名一致 ✅
- `prepare_dior_dataset(dior_root, out_base, classes=None, sample_train_per_class=16, sample_val_per_class=8, seed=42)` 在 Task 2 定义、Task 3/4 调用参数一致 ✅
- `discover_classes(dior_root)` 在 Task 2 定义、Task 3/4 调用一致 ✅
- `parse_split(split_file)` 在 Task 2 定义、内部调用一致 ✅
- 返回键（classes/split_counts/boxes_total/sample_counts/data_yaml_full/data_yaml_sample）在 Task 2 定义、Task 3/4 消费一致 ✅
- `paths.DIOR_ROOT` 在 Task 1 定义、Task 2 引用一致 ✅
- `cfg.get("dataset.dior.sample_train_per_class")` / `cfg.get("selftest.epochs")` 在 Task 1 配置定义、Task 3/4 消费一致 ✅
- Task 4 中 `train_service.train` 参数（cfg/data_yaml/run_dir/model_name/epochs/imgsz/device/pretrained/cls_pw/amp）与既有 `train_service.py:87-97` 签名完全对位 ✅
- `normalize_device`、`format_gpu_summary`、`write_markdown_report`、`validate_dataset`、`summarize`、`setup_logger` 均已核实现有签名 ✅