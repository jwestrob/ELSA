#!/usr/bin/env python3
"""Render a static, schematic genome-browser figure for the manuscript."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd

SCRIPT_DIR = Path(__file__).parent
BENCHMARKS_DIR = SCRIPT_DIR.parent


def pick_block(blocks: pd.DataFrame) -> pd.Series:
    candidates = blocks[(blocks["n_genes"] >= 8) & (blocks["n_genes"] <= 14)]
    if candidates.empty:
        return blocks.iloc[0]
    return candidates.iloc[0]


def draw_track(ax, y, n_genes, color, direction=1):
    spacing = 1.0
    width = 0.7
    height = 0.25
    for i in range(n_genes):
        x = i * spacing
        if direction == 1:
            rect = patches.FancyBboxPatch(
                (x, y - height / 2),
                width,
                height,
                boxstyle="round,pad=0.02,rounding_size=0.02",
                linewidth=0.5,
                edgecolor="#2c3e50",
                facecolor=color,
            )
            tri = patches.Polygon(
                [[x + width, y - height / 2], [x + width + 0.2, y], [x + width, y + height / 2]],
                closed=True,
                facecolor=color,
                edgecolor="#2c3e50",
                linewidth=0.5,
            )
        else:
            rect = patches.FancyBboxPatch(
                (x + 0.2, y - height / 2),
                width,
                height,
                boxstyle="round,pad=0.02,rounding_size=0.02",
                linewidth=0.5,
                edgecolor="#2c3e50",
                facecolor=color,
            )
            tri = patches.Polygon(
                [[x + 0.2, y - height / 2], [x, y], [x + 0.2, y + height / 2]],
                closed=True,
                facecolor=color,
                edgecolor="#2c3e50",
                linewidth=0.5,
            )
        ax.add_patch(rect)
        ax.add_patch(tri)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--blocks",
        type=Path,
        default=BENCHMARKS_DIR
        / "results"
        / "cross_species_chain"
        / "micro_chain"
        / "micro_chain_blocks.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=BENCHMARKS_DIR / "evaluation" / "figures" / "genome_browser.png",
    )
    args = parser.parse_args()

    blocks = pd.read_csv(args.blocks)
    block = pick_block(blocks)

    n_genes = int(block["n_genes"])
    orientation = int(block["orientation"])

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis("off")

    draw_track(ax, 1.0, n_genes, "#8ecae6", direction=1)
    draw_track(ax, 0.3, n_genes, "#ffb703", direction=orientation)

    for i in range(n_genes):
        x = i * 1.0 + 0.35
        y1 = 1.0
        y2 = 0.3
        ax.plot([x, x], [y2 + 0.1, y1 - 0.1], color="#95a5a6", linewidth=0.6, alpha=0.7)

    ax.text(0, 1.35, f"Query: {block['query_genome']} ({block['query_contig']})", fontsize=9)
    ax.text(0, 0.65, f"Target: {block['target_genome']} ({block['target_contig']})", fontsize=9)
    ax.text(
        0,
        -0.1,
        f"Representative block_id={block['block_id']} (n_genes={n_genes}, orientation={orientation})",
        fontsize=8,
        color="#7f8c8d",
    )

    ax.set_xlim(-0.5, n_genes * 1.0 + 1.0)
    ax.set_ylim(-0.3, 1.6)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output, dpi=300)
    plt.close(fig)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
