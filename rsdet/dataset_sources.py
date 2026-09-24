# -*- coding: utf-8 -*-
"""权威遥感数据集元数据 + 尽力自动下载。

文件中指定的 4 个权威数据集: DOTA v2.0 / DIOR / NWPU VHR-10 / HRSC2016。
这些数据集多托管于 Google Drive / GitHub / Baidu, 在受限网络下可能无法直接
下载, 故本模块: 直连 URL 可达时自动下载解压, 否则打印官方主页与手工获取指引。
"""
from __future__ import annotations

import shutil
import urllib.request
from pathlib import Path

DATASETS: dict[str, dict] = {
    "DOTA": {
        "name": "DOTA v2.0",
        "homepage": "https://captain-whu.github.io/DOTA/",
        "desc": "航空遥感目标检测最大基准之一, 大图(约4000x4000), 定向包围盒(OBB)标注。",
        "classes": "18 类 (plane/ship/storage-tank/harbor/bridge/vehicle/... )",
        "size": "约 20 GB",
        "format": "DOTA OBB (labelTxt, 8 角点)",
        "license": "仅限研究用途",
        "note": "官方经 Google Drive / Baidu 网盘发布。下载后解压, 用 data_converter.py 转 YOLO。",
        "direct_urls": [],
    },
    "DIOR": {
        "name": "DIOR",
        "homepage": "https://gcheng-nwpu.github.io/",
        "desc": "20 类遥感目标检测数据集, 23463 张 800x800 图像。",
        "classes": "20 类 (airplane/airport/vehicle/ship/storage-tank/... )",
        "size": "约 3.7 GB",
        "format": "Pascal VOC XML (水平框)",
        "license": "仅限研究用途",
        "note": "官方经 Google Drive 发布。用 data_converter.py --src voc 转 YOLO。",
        "direct_urls": [],
    },
    "NWPU-VHR-10": {
        "name": "NWPU VHR-10",
        "homepage": "https://gcheng-nwpu.github.io/",
        "desc": "10 类高分辨率遥感目标检测数据集, 800 张图像(650 正样本 + 150 负样本)。",
        "classes": "10 类 (airplane/ship/storage-tank/baseball-diamond/vehicle/... )",
        "size": "约 73 MB (体积最小, 适合快速验证)",
        "format": "Pascal VOC XML (水平框)",
        "license": "仅限研究用途",
        "note": "官方经 Google Drive / Baidu 发布。体积小, 推荐作为首个真实数据实验。",
        "direct_urls": [],
    },
    "HRSC2016": {
        "name": "HRSC2016",
        "homepage": "https://sites.google.com/site/hrsc2016/",
        "desc": "舰船检测数据集, 1061 张高分辨率图像。",
        "classes": "1 类 (ship)",
        "size": "约 1.5 GB",
        "format": "Pascal VOC XML + 定向框",
        "license": "仅限研究用途",
        "note": "官方经 Google Drive 发布。用 data_converter.py --src voc 转 YOLO。",
        "direct_urls": [],
    },
}


def list_datasets() -> str:
    lines = []
    for key, d in DATASETS.items():
        lines.append(f"{key:<14} {d['name']:<14} {d['size']:<12} {d['format']}")
    return "\n".join(lines)


def _download(url: str, dest: Path) -> bool:
    """下载单个 URL, 成功返回 True。"""
    try:
        print(f"[download] 下载 {url} ...")
        with urllib.request.urlopen(url, timeout=30) as resp, open(dest, "wb") as f:
            shutil.copyfileobj(resp, f)
        return True
    except Exception as e:
        print(f"[download] 失败: {e}")
        return False


def _extract_archive(archive: Path, dest: Path) -> bool:
    try:
        import zipfile
        if zipfile.is_zipfile(archive):
            with zipfile.ZipFile(archive) as z:
                z.extractall(dest)
            return True
    except Exception as e:
        print(f"[download] 解压失败: {e}")
        return False
    print("[download] 仅支持 zip 自动解压; 其它格式请手动解压。")
    return False


def download_dataset(name: str, target_dir: str | Path) -> Path:
    """尽力自动下载权威数据集, 返回 (或应放置数据的) 目标目录。"""
    key = name.upper()
    if key not in DATASETS:
        raise KeyError(f"未知数据集 {name}, 可选: {list(DATASETS)}")
    meta = DATASETS[key]
    target = Path(target_dir) / key
    target.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"[dataset] {meta['name']}")
    print(f"[dataset] 说明  : {meta['desc']}")
    print(f"[dataset] 类别  : {meta['classes']}")
    print(f"[dataset] 大小  : {meta['size']}")
    print(f"[dataset] 标注  : {meta['format']}")
    print(f"[dataset] 许可  : {meta['license']}")
    print("=" * 70)

    if meta["direct_urls"]:
        for url in meta["direct_urls"]:
            archive = target / Path(url).name
            if _download(url, archive):
                _extract_archive(archive, target)
                print(f"[dataset] 完成, 数据位于 {target}")
                return target
    else:
        print("[dataset] 该数据集未提供直连下载 URL (托管于 Google Drive / Baidu)。")
        print(f"[dataset] 请访问官方主页手工获取: {meta['homepage']}")
        print(f"[dataset] {meta['note']}")
        print(f"[dataset] 下载后请将数据放入: {target}")
        print("[dataset] 提示: 离线环境下可用本机 DIOR 真实检测数据: python scripts/prepare_dior.py")
    return target
