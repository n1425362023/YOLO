# AID 真实数据替换模拟数据 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 删除项目全部模拟数据代码与产物，改用本机真实 AID 遥感数据集（30 类场景分类，整幅场景框占位真值）驱动数据工程链路，并将 `self_test.py` 重写为「不训练、最小检验」的 AID 数据链路验证。

**Architecture:** 新增 `rsdet/aid_source.py`（AID 根目录扫描 + 按类固定 seed 划分 train/val + 每图生成整幅场景框 YOLO 标注 + 输出 data.yaml/classes.txt）与 `scripts/prepare_aid.py`（CLI 包装）。删除 `rsdet/synthetic.py`、`scripts/gen_demo_data.py`、`data/demo/`。将 `rsdet/paths.py`/`rsdet/config.py`/`configs/config.yaml` 的 `demo` 配置替换为 `aid`。重写 `scripts/self_test.py`：AID 数据准备 → 数据校验 → 结构冒烟推理（`YOLO("yolov8n.yaml")` 离线建图单张前向），全程不训练、不联网。检测链路（converter/infer/train/validator/app）全部保留不动。

**Tech Stack:** Python 3.10、ultralytics 8.4.159、torch 2.5.0+cu124、opencv、numpy。本机环境已验证可用（CUDA True）。

**执行环境事实（重要）：**
- 本项目**不是 git 仓库**，所有计划的 commit 步骤改为「无需提交」。
- 本项目**无 pytest 框架**（requirements.txt 无测试依赖），检验范式为 `scripts/self_test.py` 端到端自测 + 即时 `python -c` 断言。用户要求「仅最小检验、不训练」，因此计划的验证以「运行自测 + 命令断言」为主，不引入新测试框架。
- AID 数据根目录（本机已存在）：`AID Data Set/data/AID Data Set/AID Data Set/AID/AID_dataset/AID/`，含 30 个类文件夹（Airport…Viaduct），每类 220~420 张 600×600 jpg。
- 现有模拟数据产物：`data/demo/`（60 张合成图+标注）、`data/processed/`（旧演示数据集）、`runs/selftest_*`、`reports/self_test_report.md`。
- 配置文件优先级：DEFAULT → YAML → CLI（`rsdet/config.py` 已实现）。

---

## 文件结构总览

| 文件 | 责任 | 动作 |
|---|---|---|
| `rsdet/aid_source.py` | AID 根扫描、类发现、train/val 划分、整幅框标注、data.yaml/classes.txt 生成 | **新增** |
| `scripts/prepare_aid.py` | AID 数据准备 CLI（等价替代删掉的 gen_demo_data.py） | **新增** |
| `rsdet/paths.py` | `AID_ROOT`（默认 AID 路径）；删除 `DATA_DEMO` | **修改** |
| `rsdet/config.py` | DEFAULTS `dataset.demo` → `dataset.aid` | **修改** |
| `configs/config.yaml` | `dataset.aid` 段（root/val_ratio/seed/limit_per_class）；删除 `dataset.demo` | **修改** |
| `scripts/self_test.py` | 重写为 AID 数据链路最小检验（不训练） | **修改** |
| `rsdet/dataset_sources.py` | 第 125 行离线提示语改用 prepare_aid | **修改** |
| `README.md` | 数据说明、自测流程、语义边界 | **修改** |
| `rsdet/synthetic.py` | 模拟数据生成器 | **删除** |
| `scripts/gen_demo_data.py` | 模拟数据 CLI | **删除** |
| `data/demo/` | 模拟数据产物 | **删除**（物理） |
| `data/processed/` 旧产物 | 自测每次运行被覆盖 | 覆盖 |

依赖关系：`aid_source` → (`config`, `paths`, `train_service.write_data_yaml`)；`prepare_aid.py` → (`aid_source`, `config`, `paths`)；`self_test.py` → (`aid_source`, `validator`, `report`, `system_utils`, `config`, `paths`)。无循环依赖风险（train_service 不反向依赖 aid_source）。

---

### Task 1: `rsdet/paths.py` + `rsdet/config.py` + `configs/config.yaml` 配置替换（demo → aid）

**Files:**
- Modify: `rsdet/paths.py:20`（DATA_DEMO）、`rsdet/paths.py:29`（ensure_dirs）
- Modify: `rsdet/config.py:34-41`（DEFAULTS dataset.demo）
- Modify: `configs/config.yaml:14-20`（dataset.demo 段）

- [ ] **Step 1: 修改 `rsdet/paths.py` 删除 DATA_DEMO、新增 AID_ROOT**

将 `rsdet/paths.py` 中：

```python
# 数据 / 产物目录 (按需自动创建)
DATA_RAW = ROOT / "data" / "raw"
DATA_DEMO = ROOT / "data" / "demo"
DATA_PROCESSED = ROOT / "data" / "processed"
RUNS_DIR = ROOT / "runs"
WEIGHTS_DIR = ROOT / "weights"
REPORTS_DIR = ROOT / "reports"
```

替换为：

```python
# 数据 / 产物目录 (按需自动创建)
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
RUNS_DIR = ROOT / "runs"
WEIGHTS_DIR = ROOT / "weights"
REPORTS_DIR = ROOT / "reports"

# AID 真实遥感数据集根目录 (30 个场景类文件夹所在处)
AID_ROOT = (ROOT / "AID Data Set" / "data" / "AID Data Set"
            / "AID Data Set" / "AID" / "AID_dataset" / "AID")
```

并将 `ensure_dirs()` 中的目录元组：

```python
for d in (CONFIG_DIR, DATA_RAW, DATA_DEMO, DATA_PROCESSED,
          RUNS_DIR, WEIGHTS_DIR, REPORTS_DIR):
```

替换为：

```python
for d in (CONFIG_DIR, DATA_RAW, DATA_PROCESSED,
          RUNS_DIR, WEIGHTS_DIR, REPORTS_DIR):
```

- [ ] **Step 2: 修改 `rsdet/config.py` DEFAULTS**

将 `rsdet/config.py` DEFAULTS 中 `"dataset"` 段：

```python
    "dataset": {
        "demo": {
            "num_images": 60,
            "img_size": 640,
            "classes": ["plane", "ship", "vehicle", "storage-tank"],
            "objects_per_image": [3, 9],
        },
    },
```

替换为：

```python
    "dataset": {
        "aid": {
            "root": None,              # None = 用 paths.AID_ROOT 默认路径; 可填绝对路径覆盖
            "val_ratio": 0.2,          # 按类划分验证集比例
            "seed": 42,                # 固定划分随机种子 (可复现)
            "limit_per_class": None,   # 每类抽样上限 (None = 全量)
        },
    },
```

（`root: None` 时 `aid_source` 回退到 `paths.AID_ROOT`；若配置了字符串路径由 `paths.resolve()` 解析。此逻辑随 Task 2 实现。）

- [ ] **Step 3: 修改 `configs/config.yaml`**

将 `configs/config.yaml` 中：

```yaml
# 合成遥感演示数据 (离线自测用)
dataset:
  demo:
    num_images: 60               # 合成图片数量
    img_size: 640                # 图片边长 (像素)
    classes: [plane, ship, vehicle, storage-tank]   # 与 DOTA 类目对齐
    objects_per_image: [3, 9]    # 每图目标数量区间
```

替换为：

```yaml
# AID 真实遥感数据集 (场景分类 30 类; 整幅场景框仅作链路验证占位真值)
dataset:
  aid:
    root: null                     # None=默认 auto 定位到 "AID Data Set/.../AID_dataset/AID"; 可填绝对路径
    val_ratio: 0.2                 # 按类划分验证集比例
    seed: 42                       # 划分随机种子
    limit_per_class: null          # 每类抽样上限 (null=全量)
```

- [ ] **Step 4: 验证配置加载**

运行：

```powershell
python -c "import sys; sys.path.insert(0,'.'); from rsdet.config import get_config; c=get_config(); print('root:', c.get('dataset.aid.root')); print('val_ratio:', c.get('dataset.aid.val_ratio')); print('demo removed:', c.get('dataset.demo.num_images', 'GONE'))"
```

预期输出：`root: None`、`val_ratio: 0.2`、`demo removed: GONE`。

- [ ] **Step 5: 无需提交**

（项目非 git 仓库，下同，不再重复。）

---

### Task 2: 新增 `rsdet/aid_source.py`

**Files:**
- Create: `rsdet/aid_source.py`

- [ ] **Step 1: 写入完整模块**

创建 `rsdet/aid_source.py`，内容如下（完整代码，可直接使用）：

```python
# -*- coding: utf-8 -*-
"""AID 真实遥感数据集接入: 扫描类目录, 按类划分 train/val, 生成整幅场景框标注。

AID (Aerial Image Dataset) 是场景分类数据集: 30 类, 每类 220~420 张 600x600
遥感影像, 仅按 <Class>/ 文件夹区分整图场景类别, 官方无任何目标框标注。

本项目为检测系统, 为保留检测链路语义, 对每张 AID 图生成一条「覆盖全图的场景框」
占位真值 (cls_id 0.5 0.5 1.0 1.0), 输出标准 YOLO 检测格式 (labels/*.txt), 
使数据工程/校验/推理代码可在真实遥感影像上完整跑通。

设计文档: docs/superpowers/specs/2026-09-23-aid-data-replacement-design.md
"""
from __future__ import annotations

import random
import shutil
from pathlib import Path

from . import paths
from .config import get_config
from .train_service import write_data_yaml

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
FULL_FRAME_BBOX = "0.5 0.5 1.0 1.0"


def resolve_aid_root(root: str | Path | None = None) -> Path:
    """解析 AID 数据集根目录: CLI 参数 > config(dataset.aid.root) > paths.AID_ROOT。

    返回的目录必须存在且含至少一个类文件夹, 否则抛错。
    """
    if root is not None:
        p = paths.resolve(str(root))
    else:
        cfg_root = get_config().get("dataset.aid.root", None)
        p = paths.resolve(str(cfg_root)) if cfg_root else paths.AID_ROOT
    if not p.is_dir():
        raise FileNotFoundError(
            f"AID 数据集目录不存在: {p}。请检查 configs/config.yaml 的 "
            f"dataset.aid.root, 或确认 'AID Data Set/.../AID_dataset/AID' 位于项目内。")
    return p


def discover_classes(aid_root: Path) -> list[str]:
    """返回 AID 根目录下全部类文件夹名 (按字典序)。"""
    classes = sorted(d.name for d in Path(aid_root).iterdir() if d.is_dir())
    if not classes:
        raise ValueError(f"AID 根目录下未发现任何类文件夹: {aid_root}")
    return classes


def _list_images(class_dir: Path) -> list[Path]:
    return sorted(p for p in class_dir.iterdir()
                  if p.suffix.lower() in IMAGE_EXTS and p.is_file())


def prepare_aid_dataset(aid_root: str | Path, out_base: str | Path,
                        val_ratio: float = 0.2, seed: int = 42,
                        limit_per_class: int | None = None) -> dict:
    """把 AID 数据集整理为 YOLO 检测格式 (train/val 划分 + 整幅场景框标注)。

    输出:
        <out_base>/train/images/*.jpg   每类按 seed 洗牌后划入
        <out_base>/train/labels/*.txt   每行: <cls_id> 0.5 0.5 1.0 1.0
        <out_base>/val/images|labels    同上 (验证集)
        <out_base>/classes.txt          类名每行一个
        <out_base>/data.yaml            ultralytics 可读配置

    返回统计 dict: classes / train_images / val_images / per_class。
    """
    root = Path(aid_root)
    out = Path(out_base)
    classes = discover_classes(root)
    class_map = {c: i for i, c in enumerate(classes)}
    rng = random.Random(seed)

    # 幂等: 先清空旧划分与缓存
    for split in ("train", "val"):
        shutil.rmtree(out / split, ignore_errors=True)

    stats = {"classes": classes, "train_images": 0, "val_images": 0,
             "per_class": {c: 0 for c in classes}}

    for cls in classes:
        files = _list_images(root / cls)
        if limit_per_class:
            files = files[:limit_per_class]
        rng.shuffle(files)
        n_val = max(1, int(round(len(files) * val_ratio))) if files else 0
        n_val = min(n_val, max(0, len(files) - 1))  # 确保 train 至少 1 张
        val_files, train_files = files[:n_val], files[n_val:]

        for split, fs in (("train", train_files), ("val", val_files)):
            img_dir = out / split / "images"
            lab_dir = out / split / "labels"
            img_dir.mkdir(parents=True, exist_ok=True)
            lab_dir.mkdir(parents=True, exist_ok=True)
            for f in fs:
                dst = img_dir / f.name
                shutil.copy2(f, dst)
                (lab_dir / (f.stem + ".txt")).write_text(
                    f"{class_map[cls]} {FULL_FRAME_BBOX}\n", encoding="utf-8")
                stats["per_class"][cls] += 1
                stats[f"{split}_images"] += 1

    (out / "classes.txt").write_text("\n".join(classes) + "\n", encoding="utf-8")
    data_yaml = write_data_yaml(out / "data.yaml",
                                out / "train" / "images",
                                out / "val" / "images", classes)
    stats["data_yaml"] = str(data_yaml)
    return stats
```

注意点：
- `training_service.write_data_yaml` 生成的 `data.yaml` 中 `names: {0: Airport, 1: BareLand, ...}` 与 labels 的 cls_id 一致。
- jpg 文件名形如 `airport_1.jpg`，train/val 内图片名唯一（源于同一类目录），无需额外去重。

- [ ] **Step 2: 冒烟验证 aid_source（用最小抽样，不动全量）**

运行（取 5 类各 3 张，输出到临时目录，秒级完成）：

```powershell
python -c "
import sys, tempfile; sys.path.insert(0,'.')
import rsdet.aid_source as a
out = tempfile.mkdtemp(prefix='aid_smoke_')
root = a.resolve_aid_root()
print('AID_ROOT:', root)
print('classes:', a.discover_classes(root))
import os
os.environ['AID_SMOKE'] = '1'
"
```

预期输出：报错 `AttributeError: module 'rsdet.aid_source' has no attribute 'X'` 表示有问题；正常情况打印 AID_ROOT 与 30 个类名（Airport…Viaduct）。

**若想真正生成一份最小产物验证**，运行：

```powershell
python -c "
import sys, tempfile; sys.path.insert(0,'.')
import rsdet.aid_source as a
out = tempfile.mkdtemp(prefix='aid_smoke_')
s = a.prepare_aid_dataset(a.resolve_aid_root(), out, val_ratio=0.5, seed=42, limit_per_class=4)
print(s['classes'][:3], '...', len(s['classes']), 'classes')
print('train:', s['train_images'], 'val:', s['val_images'])
print('per_class sample:', dict(list(s['per_class'].items())[:2]))
import pathlib
print('data.yaml exists:', pathlib.Path(s['data_yaml']).exists())
"
```

预期输出：`['Airport', 'BareLand', 'BaseballField'] ... 30 classes`、`train: 60 val: 60`（30 类 × 每类 4 张 × 对半 = 60/60）、`data.yaml exists: True`。全部通过后临时目录可删除。

- [ ] **Step 3: 无需提交**

---

### Task 3: 新增 `scripts/prepare_aid.py` CLI

**Files:**
- Create: `scripts/prepare_aid.py`

- [ ] **Step 1: 写入完整 CLI**

创建 `scripts/prepare_aid.py`，内容如下：

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从本机 AID 真实遥感数据集构建 YOLO 检测格式数据 (整幅场景框占位真值)。

用法:
    python scripts/prepare_aid.py
    python scripts/prepare_aid.py --root <AID根目录> --out data/processed \
        --val-ratio 0.2 --seed 42 --limit-per-class 100
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rsdet import paths
from rsdet.aid_source import discover_classes, prepare_aid_dataset, resolve_aid_root


def main() -> None:
    ap = argparse.ArgumentParser(description="从 AID 数据集生成 YOLO 检测格式数据")
    ap.add_argument("--root", default=None,
                    help="AID 根目录 (默认: config[dataset.aid.root], 再回退 paths.AID_ROOT)")
    ap.add_argument("--out", default=None, help="输出目录 (默认 data/processed)")
    ap.add_argument("--val-ratio", type=float, default=None, help="验证集比例")
    ap.add_argument("--seed", type=int, default=None, help="划分随机种子")
    ap.add_argument("--limit-per-class", type=int, default=None, help="每类抽样上限")
    ap.add_argument("--dry-run", action="store_true", help="仅打印计划, 不写文件")
    args = ap.parse_args()

    from rsdet.config import get_config
    cfg = get_config()
    out = Path(args.out) if args.out else paths.DATA_PROCESSED
    val_ratio = args.val_ratio if args.val_ratio is not None else float(
        cfg.get("dataset.aid.val_ratio", 0.2))
    seed = args.seed if args.seed is not None else int(cfg.get("dataset.aid.seed", 42))
    limit = args.limit_per_class
    if limit is None:
        limit = cfg.get("dataset.aid.limit_per_class", None)

    root = resolve_aid_root(args.root)
    classes = discover_classes(root)
    print(f"AID 数据源     : {root}")
    print(f"类别数         : {len(classes)}")
    print(f"输出目录       : {out}")
    print(f"val_ratio={val_ratio} | seed={seed} | limit_per_class={limit}")
    if args.dry_run:
        print("(--dry-run) 未写入任何文件。")
        return

    stats = prepare_aid_dataset(root, out, val_ratio=val_ratio,
                                seed=seed, limit_per_class=limit)
    print("=" * 60)
    print("数据准备完成:")
    print(f"  train 图片 : {stats['train_images']}")
    print(f"  val   图片 : {stats['val_images']}")
    print(f"  data.yaml  : {stats['data_yaml']}")
    print("  每类计数:")
    for cls, n in stats["per_class"].items():
        print(f"    {cls:<20} : {n}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 验证 --help 与 --dry-run**

运行：

```powershell
python scripts/prepare_aid.py --help
python scripts/prepare_aid.py --dry-run --limit-per-class 10
```

预期：`--help` 打印全部参数；`--dry-run` 打印 AID 路径、30 类、val_ratio/seed/limit，并显示 `(--dry-run) 未写入任何文件。`。

- [ ] **Step 3: 验证真实产物生成（小抽样, 秒级）**

运行：

```powershell
python scripts/prepare_aid.py --out data/processed --limit-per-class 10 --seed 42
```

预期输出：`train 图片 : 240`（30 类 × 10 × 0.8）、`val 图片 : 60`、每类计数均为 10；`data/processed/data.yaml`、`classes.txt`、`train/val/images|labels` 均已生成。

- [ ] **Step 4: 无需提交**

---

### Task 4: 重写 `scripts/self_test.py`（AID 最小检验, 不训练）

**Files:**
- Modify: `scripts/self_test.py`（整文件重写）

- [ ] **Step 1: 整文件重写**

将 `scripts/self_test.py` 全部内容替换为：

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""端到端检验自测 (真实 AID 数据, 最小检验, 不训练模型)。

流程:
  1. AID 真实遥感数据 -> 按类固定 seed 划分 train/val + 整幅场景框占位真值
  2. 生成 data.yaml (30 类) 与 classes.txt
  3. 数据校验 (图像-标注成对 / 框合法 / 类别合法)
  4. 结构冒烟推理: YOLO("yolov8n.yaml") 离线建图, 对 1 张 AID 训练图 predict
     (不下载权重、不训练; 仅证明真实图可经 ultralytics 前向)

用法:
    python scripts/self_test.py
    python scripts/self_test.py --limit-per-class 10   # 快速抽样版
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rsdet import paths
from rsdet.aid_source import discover_classes, prepare_aid_dataset, resolve_aid_root
from rsdet.config import get_config
from rsdet.logging_utils import setup_logger
from rsdet.report import write_markdown_report
from rsdet.system_utils import format_gpu_summary
from rsdet.validator import summarize, validate_dataset

log = setup_logger("rsdet.selftest", paths.RUNS_DIR / "self_test.log")

CHECK, CROSS = "[PASS]", "[FAIL]"


def main() -> None:
    cfg = get_config()
    ap = argparse.ArgumentParser(description="端到端检验自测 (AID 真实数据, 不训练)")
    ap.add_argument("--limit-per-class", type=int, default=None,
                    help="每类抽样上限 (快速自测用, 默认全量)")
    ap.add_argument("--val-ratio", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--device", default=None, help="auto/cpu/0")
    args = ap.parse_args()

    limit = args.limit_per_class
    if limit is None:
        limit = cfg.get("dataset.aid.limit_per_class", None)
    val_ratio = args.val_ratio if args.val_ratio is not None else float(
        cfg.get("dataset.aid.val_ratio", 0.2))
    seed = args.seed if args.seed is not None else int(cfg.get("dataset.aid.seed", 42))
    device = args.device or cfg.get("model.device", "auto")

    aid_root = resolve_aid_root()
    classes = discover_classes(aid_root)
    proc_dir = paths.DATA_PROCESSED
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = paths.RUNS_DIR / f"selftest_{stamp}"

    print("=" * 70)
    print("遥感影像目标检测系统 (YOLO-OB) —— 端到端自测 (AID 真实数据)")
    print("=" * 70)
    print(format_gpu_summary())
    print(f"AID 数据源: {aid_root}")
    print(f"类别: {len(classes)} 类 | val_ratio={val_ratio} | seed={seed} | "
          f"limit_per_class={limit}\n")

    steps: list[dict] = []
    t0 = time.time()

    # ---- 1. AID 真实数据准备 ----
    stats = prepare_aid_dataset(aid_root, proc_dir, val_ratio=val_ratio,
                                seed=seed, limit_per_class=limit)
    steps.append({
        "name": "AID真实数据划分+整幅场景框标注",
        "ok": stats["train_images"] > 0 and stats["val_images"] > 0,
        "detail": f"{len(classes)} 类 / train={stats['train_images']} "
                  f"val={stats['val_images']}"})

    # ---- 2. data.yaml ----
    data_yaml = Path(stats["data_yaml"])
    steps.append({"name": "生成 data.yaml", "ok": data_yaml.exists(),
                  "detail": str(data_yaml)})

    # ---- 3. 数据校验 ----
    val_report = validate_dataset(proc_dir / "train" / "images",
                                  proc_dir / "train" / "labels", classes)
    val_ok = val_report["ok"] and val_report["images_with_labels"] > 0
    steps.append({"name": "数据校验 (图像-标注成对/框合法)",
                  "ok": val_ok,
                  "detail": summarize(val_report, classes)})

    # ---- 4. 结构冒烟推理 (不训练、不下载权重) ----
    smoke_ok = False
    smoke_detail = ""
    if val_ok:
        try:
            from ultralytics import YOLO
            model = YOLO("yolov8n.yaml")  # 离线从配置文件建图, 不下载任何权重
            sample = sorted((proc_dir / "train" / "images").glob("*"))[0]
            res = model.predict(source=str(sample), device=device, verbose=False)[0]
            n_det = len(res.boxes) if res.boxes is not None else 0
            smoke_ok = res is not None and n_det >= 0  # 只验证前向无异常
            smoke_detail = f"{sample.name} | 前向完成 (检测框数={n_det}) | device={device}"
        except Exception as e:
            log.exception("结构冒烟推理失败")
            smoke_detail = f"{type(e).__name__}: {e}"
    steps.append({"name": "结构冒烟推理 (yolov8n.yaml 离线建图)",
                  "ok": smoke_ok, "detail": smoke_detail})

    # ---- 汇总 ----
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
        report_lines.append(f"- {'✅' if s['ok'] else '❌'} {s['name']}: "
                            f"{s.get('detail', '')}")
    report_path = write_markdown_report(paths.REPORTS_DIR / "self_test_report.md",
                                        "遥感目标检测系统自测报告 (AID 真实数据)",
                                        report_lines)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "self_test.json").write_text(json.dumps({
        "verdict": verdict, "elapsed_s": elapsed, "steps": steps,
        "aid_root": str(aid_root), "classes": classes,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n报告: {report_path}")
    print(f"详细: {run_dir / 'self_test.json'}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 快速自测（抽样版, 不训练, 秒~分钟级）**

运行：

```powershell
python scripts/self_test.py --limit-per-class 10 --device cpu
```

预期输出：4 个步骤全部 `[PASS]`，`最终结论: PASS`，`self_test_report.md` 更新。全程无训练（无 epoch 打印）、无权重下载。

- [ ] **Step 3: 无需提交**

---

### Task 5: 删除模拟数据代码与产物

**Files:**
- Delete: `rsdet/synthetic.py`
- Delete: `scripts/gen_demo_data.py`
- Delete: `data/demo/`（整个目录）

- [ ] **Step 1: 删除代码文件**

运行：

```powershell
Remove-Item -LiteralPath "rsdet/synthetic.py" -Force
Remove-Item -LiteralPath "scripts/gen_demo_data.py" -Force
```

- [ ] **Step 2: 删除模拟数据产物目录**

运行：

```powershell
Remove-Item -LiteralPath "data/demo" -Recurse -Force
```

（`data/processed` 不手动删——自测每次运行会先清空 train/val 后重建；其中的 data.yaml 同样由自测重写。）

- [ ] **Step 3: 验证无残留引用**

运行（PowerShell，遍历所有 .py/.md/.yaml，排除 AID 数据目录）：

```powershell
Get-ChildItem -Recurse -File -Include *.py,*.md,*.yaml -Path .
  | Where-Object { $_.FullName -notmatch '\\(runs|weights|AID Data Set|__pycache__)\\' }
  | Select-String -Pattern 'synthetic|gen_demo_data|DATA_DEMO|data.demo|generate_demo_dataset|data/demo'
  | ForEach-Object { "$($_.Path):$($_.LineNumber): $($_.Line.Trim())" }
```

预期输出：**无匹配行**（除本计划与设计文档外不应有任何残留；若出现 README/dataset_sources 的提示语命中的 harmless 文案，由 Task 6/7 清理后复跑确认）。

- [ ] **Step 4: 确认 import 失败即删除成功**

运行：`python -c "import rsdet.synthetic"` → 预期 `ModuleNotFoundError: No module named 'rsdet.synthetic'`。

- [ ] **Step 5: 无需提交**

---

### Task 6: 更新 `rsdet/dataset_sources.py` 离线提示语

**Files:**
- Modify: `rsdet/dataset_sources.py:125`

- [ ] **Step 1: 更新提示行**

将 `rsdet/dataset_sources.py` 第 125 行：

```python
        print("[dataset] 提示: 离线环境下可用合成演示数据: python scripts/gen_demo_data.py")
```

替换为：

```python
        print("[dataset] 提示: 离线环境下可用本机 AID 真实数据: python scripts/prepare_aid.py")
```

- [ ] **Step 2: 验证**

运行：`python -c "import sys; sys.path.insert(0,'.'); import rsdet.dataset_sources as d; assert 'gen_demo_data' not in open(d.__file__, encoding='utf-8').read(); print('OK')"` → 预期 `OK`。

- [ ] **Step 3: 无需提交**

---

### Task 7: 更新 `README.md`

**Files:**
- Modify: `README.md`

- [ ] **Step 1: 更新功能特性中的数据集说明**

将：

```markdown
- **权威数据集**：内置 DOTA v2.0 / DIOR / NWPU VHR-10 / HRSC2016 元数据与自动下载脚本；受网络限制时可用合成遥感数据离线自测。
```

替换为：

```markdown
- **真实遥感数据**：内置 DOTA v2.0 / DIOR / NWPU VHR-10 / HRSC2016 元数据与自动下载脚本；本地 AID 遥感影像数据集（30 类场景、约 1 万张 600×600）可直接用于数据工程链路验证。
```

- [ ] **Step 2: 更新目录结构**

将：

```markdown
│   ├── gen_demo_data.py         # 合成遥感演示数据
```

替换为：

```markdown
│   ├── prepare_aid.py           # 从本机 AID 数据集生成 YOLO 检测格式数据
```

- [ ] **Step 3: 更新自测段落**

将：

```markdown
自测会生成合成遥感数据 → DOTA→YOLO 转换 → 划分 train/val → 数据校验 → YOLOv8n 迁移学习训练 → 验证集评估 → 大图切片推理，最终输出 `PASS/FAIL`。
```

替换为：

```markdown
自测会从本机 AID 真实遥感数据集划分 train/val → 生成整幅场景框占位标注与 data.yaml → 数据校验 → `yolov8n.yaml` 结构冒烟推理（离线建图、不训练、不下载权重），最终输出 `PASS/FAIL`。
```

- [ ] **Step 4: 更新「权威数据集说明」后的离线提示**

将：

```markdown
> 这些数据集多托管于 Google Drive / Baidu，本机若无法直连，`download_dataset.py` 会打印官方主页与手工获取指引。离线环境下使用 `gen_demo_data.py` 生成的合成数据即可完整自测（二者共用同一套转换/校验/训练/推理代码）。
```

替换为：

```markdown
> 这些数据集多托管于 Google Drive / Baidu，本机若无法直连，`download_dataset.py` 会打印官方主页与手工获取指引。离线环境下可使用本机 AID 真实遥感数据集验证数据工程链路：`python scripts/prepare_aid.py`（生成 YOLO 格式数据）→ `python scripts/self_test.py`（最小检验）。
```

- [ ] **Step 5: 在「主要配置」后补充 AID 与语义边界说明**

在 `## 主要配置` 段末尾追加：

```markdown
## AID 真实数据集与语义边界

- AID（Aerial Image Dataset）为**场景分类**数据集：30 类、每类 220~420 张 600×600 遥感影像，仅提供整图场景类别，**无目标框标注**。
- 本系统在检测框架下使用 AID 时，为每张图生成一条「覆盖全图的场景框」占位真值（`cls 0.5 0.5 1.0 1.0`），目的是让数据工程/校验/推理链路在真实遥感影像上完整跑通。
- ⚠️ 该占位标签**不是 AID 官方检测标注**，据此训练的检测模型在 mAP 上无参考价值；需要有意义的检测模型请改用 DOTA / DIOR / NWPU-VHR-10（见 `scripts/data_converter.py`）。
```

- [ ] **Step 6: 验证 README 无残留合成数据文案**

运行：

```powershell
Select-String -Path README.md -Pattern 'gen_demo_data|合成遥感|合成数据' | ForEach-Object { "$($_.LineNumber): $($_.Line.Trim())" }
```

预期：无输出（或仅剩「语义边界」中不涉及合成数据的说明）。

- [ ] **Step 7: 无需提交**

---

### Task 8: 全链路最终验证（关键验收）

**Files:**
- Verify: 全部改动文件
- 运行: `python scripts/self_test.py`（默认全量）

- [ ] **Step 1: LSP 诊断**

运行 `lsp_diagnostics` 于：`rsdet/aid_source.py`、`rsdet/paths.py`、`rsdet/config.py`、`scripts/prepare_aid.py`、`scripts/self_test.py`。
预期：无 error（warning 允许存在）。

- [ ] **Step 2: 全量真实 AID 自测（核心验收, 会复制全部 AID 图, 约需数分钟）**

运行：

```powershell
python scripts/self_test.py --device cpu
```

预期：
- Step 1 `[PASS]`：`30 类 / train≈8000 val≈2000`（30 类全量）
- Step 2 `[PASS]`：data.yaml 存在
- Step 3 `[PASS]`：数据校验通过（图像-标注成对、框合法）
- Step 4 `[PASS]`：结构冒烟推理完成（yolov8n.yaml 离线建图，前向 1 张真实 AID 图）
- `最终结论: PASS`，`runs/selftest_*/self_test.json` 落盘，`reports/self_test_report.md` 更新。
- **全程无训练打印、无权重下载。**

- [ ] **Step 3: 残留引用终检**

再次运行 Task 5 Step 3 的 PowerShell 扫描命令，预期 **无匹配行**。

- [ ] **Step 4: 断言关键文件存在与内容**

运行：

```powershell
python -c "
import sys, yaml; sys.path.insert(0,'.')
from pathlib import Path
p = Path('data/processed/data.yaml')
d = yaml.safe_load(p.read_text(encoding='utf-8'))
print('classes:', len(d['names']))
print('names[0],names[29]:', d['names'][0], d['names'][29])
print('train rel:', d['train'])
labels = list((Path('data/processed')/'train'/'labels').glob('*.txt'))
img = list((Path('data/processed')/'train'/'images').glob('*.jpg'))
print('train labels:', len(labels), 'images:', len(img))
print('sample label:', labels[0].read_text().strip())
"
```

预期：`classes: 30`、`names[0],names[29]: Airport Viaduct`、`train rel: train/images`、`train labels==train images`、`sample label: <0~29> 0.5 0.5 1.0 1.0`。

- [ ] **Step 5: 确认模拟数据全清**

运行：

```powershell
Test-Path "rsdet/synthetic.py"; Test-Path "scripts/gen_demo_data.py"; Test-Path "data/demo"
```

预期：三条均为 `False`。

- [ ] **Step 6: 无需提交**

---

## 自审（对照设计文档）

**1. Spec 覆盖检查：**
- §4.1 数据流（aid_source 扫描/划分/整幅框/data.yaml）→ Task 2/3 ✅
- §4.2 删除清单（synthetic.py、gen_demo_data.py、config demo 段、DATA_DEMO、data/demo、README 合成文案）→ Task 1/5/7 ✅
- §4.3 保留复用（converter/infer/train/validator/app 不动）→ 计划未触碰这些文件 ✅
- §4.4 自测新流程（准备→校验→data.yaml→结构冒烟，不训练）→ Task 4 ✅
- §5 错误处理（AID 缺失报错+指引=resolve_aid_root；空类=discover_classes 抛错；坏图=validator issues 且 FAIL；离线 CPU=--device cpu）→ Task 2/4 ✅
- §6 成功标准（self_test PASS、prepare_aid 可运行、无残留引用、lsp 无错）→ Task 8 ✅
- §7 语义边界标注 → Task 7 README 段 ✅
- §8 文件清单 → 与总览表一致 ✅
- 补充项：dataset_sources.py 提示语更新（设计文档已补充）→ Task 6 ✅

**2. 占位符扫描：** 全部步骤含完整代码/命令/预期输出，无 TBD/TODO/「相似于 Task N」表述 ✅

**3. 类型/命名一致性：**
- `resolve_aid_root(root=None)`：CLI 参数 > config > paths.AID_ROOT 优先级，Task 2 定义并贯穿 prepare_aid.py / self_test.py 使用 ✅
- `prepare_aid_dataset(...) -> dict{classes, train_images, val_images, per_class, data_yaml}` 签名在 Task 2/3/4 完全一致 ✅
- `discover_classes` 在 Task 2 定义，Task 3/4 调用一致 ✅
- `paths.AID_ROOT` 在 Task 1 定义，Task 2 引用一致 ✅
- `stats["train_images"]` 等键名在 Task 2/3/4 一致 ✅