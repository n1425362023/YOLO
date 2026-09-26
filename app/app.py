#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""遥感影像目标检测系统 (YOLO-OB) —— Streamlit Web 平台。

功能: 上传单张大图 -> 切片推理 -> 原图/标注图对比 + KPI + 检测明细 + 密度热力图 + 报告下载。

运行:
    streamlit run app/app.py
"""
from __future__ import annotations

import base64
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import streamlit as st

from rsdet import paths
from rsdet.config import get_config
from rsdet.infer_service import detect, load_model
from rsdet.report import density_heatmap, write_html_report
from rsdet.system_utils import format_gpu_summary

st.set_page_config(page_title="遥感影像目标检测系统", page_icon="🛰️", layout="wide")


def _discover_weights() -> list[Path]:
    """自动发现可用的 best.pt 权重 (按修改时间倒序, 最新权重优先)。"""
    found = list(paths.RUNS_DIR.glob("**/weights/best.pt")) + \
            list(paths.WEIGHTS_DIR.glob("*.pt"))
    found = list(dict.fromkeys(found))       # 去重保序
    found.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return found


@st.cache_resource(show_spinner=False)
def _load_model(weights: str):
    return load_model(weights)


def _img_to_b64_jpeg(img_bgr: np.ndarray, max_edge: int = 1600, quality: int = 90) -> str:
    """cv2 图像 (BGR) -> 缩放 -> JPEG base64, 用于内嵌进 HTML 报告 (控制文件体积)。"""
    h, w = img_bgr.shape[:2]
    if max(h, w) > max_edge:
        scale = max_edge / max(h, w)
        img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)),
                             interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("图像 JPEG 编码失败")
    return base64.b64encode(buf).decode("ascii")


def main() -> None:
    st.title("🛰️ 遥感影像目标检测系统 (YOLO-OB)")
    st.caption("大图切片推理 · 合并 NMS · 密度热力图 · 统计分析")

    cfg = get_config()

    with st.sidebar:
        st.header("⚙️ 参数")
        weights_files = _discover_weights()
        weights_label = {str(p): p.name for p in weights_files}
        if weights_files:
            default_w = str(weights_files[0])
            weights = st.selectbox("模型权重", list(weights_label.keys()),
                                   format_func=lambda k: weights_label[k],
                                   index=list(weights_label.keys()).index(default_w)
                                   if default_w in weights_label else 0)
        else:
            weights = st.text_input("模型权重路径 (.pt)", "")
        conf = st.slider("置信度阈值", 0.05, 0.9, float(cfg.get("inference.conf", 0.25)), 0.05)
        iou = st.slider("NMS IoU", 0.1, 0.9, float(cfg.get("inference.iou", 0.45)), 0.05)
        tile_size = st.slider("切片尺寸", 320, 1280, int(cfg.get("inference.tile_size", 640)), 64)
        overlap = st.slider("切片重叠比例", 0.0, 0.5, float(cfg.get("inference.overlap", 0.25)), 0.05)
        force_slice = st.toggle("强制切片推理", value=False)
        st.divider()
        st.markdown(format_gpu_summary().replace("\n", "  \n"))

    uploaded = st.file_uploader("上传遥感影像 (png/jpg/tif)", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"])

    if uploaded is None:
        st.info("请上传一张遥感影像。若无权重, 请先运行 `python scripts/self_test.py` 生成权重。")
        return

    if not weights or not Path(weights).exists():
        st.error("未找到可用权重。请先运行 `python scripts/self_test.py` 或指定 --weights 训练。")
        return

    file_bytes = np.frombuffer(uploaded.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        st.error("无法解析该图像。")
        return

    h, w = img.shape[:2]
    st.markdown(f"**图像尺寸**: {w} × {h} | 自动切片推理" if max(h, w) > tile_size * 1.25 else
                f"**图像尺寸**: {w} × {h} | 单次推理")

    model = _load_model(weights)
    names = [model.names[i] for i in range(len(model.names))]

    with st.spinner("推理中..."):
        t0 = time.perf_counter()
        r = detect(model, img, names, conf=conf, iou=iou, tile_size=tile_size,
                   overlap=overlap, force_slice=force_slice,
                   imgsz=int(cfg.get("model.imgsz", 416)))
        wall = (time.perf_counter() - t0) * 1000

    dets = r["detections"]

    # KPI
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("检测目标数", len(dets))
    c2.metric("平均置信度", f"{np.mean([d['conf'] for d in dets]):.2f}" if dets else "0.00")
    c3.metric("推理耗时", f"{r['elapsed_ms']:.0f} ms")
    c4.metric("切片数量", r["num_tiles"])

    # 原图 / 标注图对比
    col_a, col_b = st.columns(2)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ann_rgb = cv2.cvtColor(r["annotated"], cv2.COLOR_BGR2RGB)
    col_a.subheader("原图")
    col_a.image(img_rgb, use_container_width=True)
    col_b.subheader("检测结果")
    col_b.image(ann_rgb, use_container_width=True)

    # 检测明细表
    st.subheader("检测明细")
    if dets:
        import pandas as pd
        rows = [{"类别": names[d["cls"]], "置信度": round(d["conf"], 3),
                 "x0": round(d["bbox"][0]), "y0": round(d["bbox"][1]),
                 "x1": round(d["bbox"][2]), "y1": round(d["bbox"][3])} for d in dets]
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        st.write("未检测到目标。")

    # 密度热力图
    st.subheader("目标空间分布密度热力图")
    b64 = density_heatmap(dets, w, h, title="目标空间分布密度热力图")
    st.image(base64.b64decode(b64), use_container_width=True)

    # 报告下载
    if dets:
        import pandas as pd
        summary_rows = ["<table><tr><th>类别</th><th>数量</th></tr>"]
        from collections import Counter
        cnt = Counter(d["cls"] for d in dets)
        for c, n in cnt.most_common():
            summary_rows.append(f"<tr><td>{names[c]}</td><td>{n}</td></tr>")
        summary_rows.append("</table>")
        html = write_html_report(paths.REPORTS_DIR / "web_report.html", "遥感影像检测报告",
                                 [("统计", "".join(summary_rows)),
                                  ("参数", f"conf={conf}, iou={iou}, 切片={tile_size}, "
                                           f"重叠={overlap}, 耗时={r['elapsed_ms']:.0f}ms")],
                                 [("原图", _img_to_b64_jpeg(img)),
                                  ("检测结果", _img_to_b64_jpeg(r["annotated"])),
                                  ("密度热力图", b64)])
        st.download_button("下载 HTML 报告", html.read_bytes(),
                           file_name="检测报告.html", mime="text/html")


if __name__ == "__main__":
    main()
