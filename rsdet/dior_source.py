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