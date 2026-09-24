# -*- coding: utf-8 -*-
"""报告生成: matplotlib 图表 + 自包含 HTML / Markdown 报告。

图表以 base64 内嵌进 HTML, 便于单文件分享。
"""
from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Iterable

import numpy as np


def _setup_plt_zh():
    """配置 matplotlib 渲染中文字体, 避免标题/轴标签/图例乱码 (方块)。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    plt.rcParams["axes.unicode_minus"] = False
    for font in ("Microsoft YaHei", "SimHei", "SimSun", "DengXian",
                 "Noto Sans CJK SC", "WenQuanYi Zen Hei"):
        try:
            font_manager.findfont(font, fallback_to_default=False)
            plt.rcParams["font.sans-serif"] = [font]
            break
        except Exception:
            continue
    return plt


def _fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def class_dist_plot(class_dist: dict[int, int], classes: list[str],
                    title: str = "类别分布") -> str:
    """类别分布直方图 -> base64 PNG。"""
    plt = _setup_plt_zh()

    n = len(classes)
    counts = [class_dist.get(c, 0) for c in range(n)]
    fig, ax = plt.subplots(figsize=(max(6, n * 0.8), 3.6))
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, n))
    ax.bar(range(n), counts, color=colors)
    ax.set_xticks(range(n))
    ax.set_xticklabels(classes, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("框数")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    b64 = _fig_to_base64(fig)
    plt.close(fig)
    return b64


def density_heatmap(detections: Iterable[dict], image_w: int, image_h: int,
                    grid: int = 32, title: str = "目标空间分布密度热力图") -> str:
    """检测框中心点密度热力图 -> base64 PNG。

    detections: [{cls, conf, bbox:[x0,y0,x1,y1]}] (像素坐标)。
    """
    plt = _setup_plt_zh()

    gx, gy = max(1, image_w // grid), max(1, image_h // grid)
    hist = np.zeros((gy, gx), dtype=float)
    for d in detections:
        x0, y0, x1, y1 = d["bbox"]
        cx = (x0 + x1) / 2
        cy = (y0 + y1) / 2
        xi = int(np.clip(cx / image_w * gx, 0, gx - 1))
        yi = int(np.clip(cy / image_h * gy, 0, gy - 1))
        hist[yi, xi] += 1.0

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(hist, cmap="inferno", origin="upper", aspect="auto")
    fig.colorbar(im, ax=ax, label="目标数")
    ax.set_title(title)
    ax.set_xlabel("图像宽度方向 (网格)")
    ax.set_ylabel("图像高度方向 (网格)")
    fig.tight_layout()
    b64 = _fig_to_base64(fig)
    plt.close(fig)
    return b64


def write_html_report(out_path: str | Path, title: str, sections: list[tuple[str, str]],
                      figures: list[tuple[str, str]] | None = None) -> Path:
    """写自包含 HTML 报告。

    sections: [(标题, html正文)] ; figures: [(图注, base64 png)]
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    body_parts = []
    for heading, content in sections:
        body_parts.append(f"<h2>{heading}</h2>\n{content}")
    if figures:
        body_parts.append("<h2>图表</h2>")
        for caption, b64 in figures:
            body_parts.append(
                f"<figure><img src='data:image/png;base64,{b64}' "
                f"style='max-width:100%'/><figcaption>{caption}</figcaption></figure>"
            )

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<title>{title}</title>
<style>
 body {{ font-family: "Microsoft YaHei", sans-serif; margin: 24px; color: #222; }}
 h1 {{ border-bottom: 2px solid #2c6fbb; padding-bottom: 8px; }}
 h2 {{ color: #2c6fbb; margin-top: 24px; }}
 table {{ border-collapse: collapse; margin: 12px 0; }}
 th, td {{ border: 1px solid #ccc; padding: 6px 12px; text-align: left; }}
 th {{ background: #f0f4f8; }}
 figure {{ margin: 16px 0; text-align: center; }}
 figcaption {{ color: #666; font-size: 13px; margin-top: 4px; }}
</style>
</head>
<body>
<h1>{title}</h1>
{''.join(body_parts)}
</body>
</html>"""
    out_path.write_text(html, encoding="utf-8")
    return out_path


def write_markdown_report(out_path: str | Path, title: str, lines: list[str]) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    content = f"# {title}\n\n" + "\n".join(lines) + "\n"
    out_path.write_text(content, encoding="utf-8")
    return out_path
