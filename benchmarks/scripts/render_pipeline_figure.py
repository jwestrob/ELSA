#!/usr/bin/env python3
"""Render a high-quality SVG pipeline figure and convert to PNG/PDF."""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
from pathlib import Path

SVG_TEMPLATE = """<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"1000\" height=\"1500\" viewBox=\"0 0 1000 1500\">
  <defs>
    <style>
      .title { font: 700 30px "DejaVu Sans", "Liberation Sans", sans-serif; fill: #0f172a; }
      .subtitle { font: 400 16px "DejaVu Sans", "Liberation Sans", sans-serif; fill: #475569; }
      .step-title { font: 700 18px "DejaVu Sans", "Liberation Sans", sans-serif; fill: #0f172a; }
      .step-body { font: 400 15px "DejaVu Sans", "Liberation Sans", sans-serif; fill: #475569; }
      .chip-text { font: 600 12px "DejaVu Sans", "Liberation Sans", sans-serif; fill: #0f172a; }
      .step-num { font: 700 14px "DejaVu Sans", "Liberation Sans", sans-serif; fill: #ffffff; }
      .label { font: 600 12px "DejaVu Sans", "Liberation Sans", sans-serif; fill: #ef4444; }
    </style>
    <filter id=\"shadow\" x=\"-20%\" y=\"-20%\" width=\"140%\" height=\"140%\">
      <feDropShadow dx=\"0\" dy=\"1.5\" stdDeviation=\"2\" flood-color=\"#0f172a\" flood-opacity=\"0.12\" />
    </filter>
    <marker id=\"arrow\" markerWidth=\"10\" markerHeight=\"10\" refX=\"5\" refY=\"5\" orient=\"auto\">
      <path d=\"M0,0 L10,5 L0,10 Z\" fill=\"#0f172a\"/>
    </marker>
  </defs>
  <rect x=\"0\" y=\"0\" width=\"1000\" height=\"1500\" fill=\"#ffffff\"/>

  <text x=\"70\" y=\"70\" class=\"title\">ELSA chaining workflow</text>
  <text x=\"70\" y=\"98\" class=\"subtitle\">Gene‑level pipeline (no macro path)</text>

  __TIMELINE__
  __BOXES__
  __ARROWS__
</svg>
"""


COLORS = {
    "inputs": "#2563eb",
    "embeddings": "#10b981",
    "projection": "#f59e0b",
    "anchors": "#0f766e",
    "chaining": "#ef4444",
    "blocks": "#64748b",
    "clustering": "#1d4ed8",
}


def chip(x, y, text, color):
    w = max(50, 8 * len(text) + 18)
    rect = f'<rect x="{x}" y="{y}" width="{w}" height="20" rx="10" ry="10" fill="#f1f5f9" stroke="{color}" stroke-width="1"/>'
    txt = f'<text x="{x + w / 2}" y="{y + 14}" text-anchor="middle" class="chip-text">{text}</text>'
    return rect + txt, w


def step_box(x, y, w, h, num, title, body, chips, icon, highlight=False):
    stroke = COLORS["chaining"] if highlight else "#0f172a"
    fill = "#fff5f5" if highlight else "#ffffff"
    rect = f'<rect x="{x}" y="{y}" rx="14" ry="14" width="{w}" height="{h}" fill="{fill}" stroke="{stroke}" stroke-width="1.6" filter="url(#shadow)"/>'
    num_circle = f'<circle cx="{x - 40}" cy="{y + h / 2}" r="16" fill="{stroke}"/>'
    num_text = f'<text x="{x - 40}" y="{y + h / 2 + 5}" text-anchor="middle" class="step-num">{num}</text>'

    title_text = f'<text x="{x + 24}" y="{y + 42}" class="step-title">{title}</text>'
    body_text = f'<text x="{x + 24}" y="{y + 72}" class="step-body">{body}</text>'

    chip_parts = []
    cx = x + 24
    cy = y + 94
    for text, color in chips:
        part, w_chip = chip(cx, cy, text, color)
        chip_parts.append(part)
        cx += w_chip + 8

    label = ""
    if highlight:
        label = f'<text x="{x + w - 150}" y="{y + 30}" class="label">core algorithm</text>'

    return f'<g>{rect}{num_circle}{num_text}{title_text}{body_text}{"".join(chip_parts)}{label}{icon}</g>'


def illustration_inputs(x, y):
    parts = []
    for row in range(2):
        yy = y + row * 24
        for i in range(8):
            xx = x + i * 22
            parts.append(f'<rect x="{xx}" y="{yy}" width="14" height="7" fill="#bfdbfe" stroke="#1f2937" stroke-width="0.6"/>')
            parts.append(f'<polygon points="{xx+14},{yy} {xx+20},{yy+3.5} {xx+14},{yy+7}" fill="#bfdbfe" stroke="#1f2937" stroke-width="0.6"/>')
    return "<g>" + "".join(parts) + "</g>"


def illustration_embeddings(x, y):
    coords = [(0,0),(20,10),(40,6),(12,26),(30,30),(52,22),(8,50),(28,50),(48,50)]
    dots = [f'<circle cx="{x+dx}" cy="{y+dy}" r="4.5" fill="#34d399" stroke="#0f172a" stroke-width="0.6"/>' for dx, dy in coords]
    return "<g>" + "".join(dots) + "</g>"


def illustration_projection(x, y):
    left = illustration_embeddings(x, y)
    right = illustration_embeddings(x + 90, y + 6)
    arrow = f'<line x1="{x+70}" y1="{y+26}" x2="{x+90}" y2="{y+34}" stroke="#0f172a" stroke-width="2" marker-end="url(#arrow)"/>'
    return f'<g>{left}{arrow}{right}</g>'


def illustration_knn(x, y):
    parts = []
    for i in range(5):
        parts.append(f'<circle cx="{x}" cy="{y+i*16}" r="4" fill="#60a5fa"/>')
        parts.append(f'<circle cx="{x+90}" cy="{y+i*16}" r="4" fill="#f59e0b"/>')
        if i in (1,2,3):
            parts.append(f'<line x1="{x}" y1="{y+i*16}" x2="{x+90}" y2="{y+(4-i)*16}" stroke="#9ca3af" stroke-width="1"/>')
    return "<g>" + "".join(parts) + "</g>"


def illustration_chaining(x, y):
    parts = []
    for i in range(7):
        for j in range(7):
            parts.append(f'<circle cx="{x+i*14}" cy="{y+j*14}" r="2.4" fill="#c7d2fe"/>')
    path_points = [(x+0, y+0), (x+14, y+14), (x+28, y+28), (x+42, y+42), (x+70, y+70)]
    path = "M " + " L ".join(f"{px},{py}" for px, py in path_points)
    parts.append(f'<path d="{path}" stroke="#ef4444" stroke-width="2.4" fill="none"/>')
    return "<g>" + "".join(parts) + "</g>"


def illustration_blocks(x, y):
    parts = [
        f'<rect x="{x}" y="{y+18}" width="90" height="26" fill="#e5e7eb" stroke="#6b7280" stroke-width="1"/>',
        f'<rect x="{x+20}" y="{y+8}" width="90" height="26" fill="#fde68a" stroke="#b45309" stroke-width="1.5"/>',
        f'<rect x="{x+40}" y="{y}" width="90" height="26" fill="#e5e7eb" stroke="#6b7280" stroke-width="1"/>',
    ]
    return "<g>" + "".join(parts) + "</g>"


def illustration_clusters(x, y):
    nodes = [(0,0),(24,10),(10,28),(40,30),(60,12),(72,32)]
    parts = []
    for a, b in [(0,1),(1,2),(2,3),(3,4),(4,5)]:
        x1,y1 = nodes[a]
        x2,y2 = nodes[b]
        parts.append(f'<line x1="{x+x1}" y1="{y+y1}" x2="{x+x2}" y2="{y+y2}" stroke="#94a3b8" stroke-width="1"/>')
    for dx, dy in nodes:
        parts.append(f'<circle cx="{x+dx}" cy="{y+dy}" r="6" fill="#93c5fd" stroke="#1f2937" stroke-width="0.8"/>')
    return "<g>" + "".join(parts) + "</g>"


def render_svg() -> str:
    x_card = 140
    w = 760
    h = 140
    y0 = 150
    gap = 28

    steps = [
        ("Inputs", "Genomes (FASTA + GFF)", [("FASTA", COLORS["inputs"]), ("GFF", COLORS["inputs"])], illustration_inputs, False),
        ("Protein embeddings", "ESM2‑t12, mean‑pooled", [("ESM2‑t12", COLORS["embeddings"])], illustration_embeddings, False),
        ("Projection", "PCA → 256D, L2 normalize", [("PCA 256D", COLORS["projection"]), ("L2", COLORS["projection"])], illustration_projection, False),
        ("Cross‑genome kNN anchors", "HNSW k=50, cosine ≥ 0.9", [("HNSW k=50", COLORS["anchors"]), ("cos ≥ 0.9", COLORS["anchors"])], illustration_knn, False),
        ("Anchor chaining", "DP/LIS, max_gap=2, orientation", [("DP/LIS", COLORS["chaining"]), ("max_gap=2", COLORS["chaining"])], illustration_chaining, True),
        ("Block selection", "Greedy non‑overlap scoring", [("non‑overlap", COLORS["blocks"])], illustration_blocks, False),
        ("Clustering → outputs", "Overlap Jaccard, mutual‑k → clusters + browser", [("Jaccard", COLORS["clustering"]), ("mutual‑k", COLORS["clustering"])], illustration_clusters, False),
    ]

    boxes = []
    arrows = []
    timeline = []

    total_height = y0 + len(steps) * h + (len(steps) - 1) * gap
    line_top = y0 + 10
    line_bottom = y0 + (len(steps) - 1) * (h + gap) + h - 10
    timeline.append(f'<line x1="80" y1="{line_top}" x2="80" y2="{line_bottom}" stroke="#e2e8f0" stroke-width="6"/>')

    for i, (title, body, chips, icon_fn, highlight) in enumerate(steps, start=1):
        y = y0 + (i - 1) * (h + gap)
        icon = icon_fn(x_card + w - 170, y + 46)
        boxes.append(step_box(x_card, y, w, h, i, title, body, chips, icon, highlight=highlight))
        if i < len(steps):
            arrows.append(
                f'<line x1="{x_card + w/2}" y1="{y + h}" x2="{x_card + w/2}" y2="{y + h + gap - 6}" stroke="#0f172a" stroke-width="1.6" marker-end="url(#arrow)"/>'
            )

    return (
        SVG_TEMPLATE.replace("__TIMELINE__", "".join(timeline))
        .replace("__BOXES__", "".join(boxes))
        .replace("__ARROWS__", "".join(arrows))
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    svg_path = output_dir / "fig1_pipeline_overview.svg"
    svg_path.write_text(render_svg(), encoding="utf-8")

    rsvg = shutil.which("rsvg-convert")
    if not rsvg:
        print(f"Saved SVG (no converter found): {svg_path}")
        return

    png_path = output_dir / "fig1_pipeline_overview.png"
    pdf_path = output_dir / "fig1_pipeline_overview.pdf"

    subprocess.run([rsvg, "-f", "png", "-o", str(png_path), str(svg_path)], check=True)
    subprocess.run([rsvg, "-f", "pdf", "-o", str(pdf_path), str(svg_path)], check=True)
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


if __name__ == "__main__":
    main()
