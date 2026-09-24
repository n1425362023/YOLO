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