#!/usr/bin/env python3
"""Figure S3: Block Size Distributions.

Four panels showing block size histograms for each dataset:
a) S. pneumoniae, b) E. coli, c) Enterobacteriaceae, d) Borg genomes.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

setup_style()

# ── Data ────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent

datasets = {
    'S. pneumoniae\n(6 genomes)': BASE / 'syntenic_analysis/micro_chain/micro_chain_blocks.csv',
    'E. coli\n(20 genomes)': BASE / 'benchmarks/results/ecoli_chain/micro_chain_blocks.csv',
    'Enterobacteriaceae\n(30 genomes)': BASE / 'benchmarks/results/enterobacteriaceae_chain/micro_chain/micro_chain_blocks.csv',
    'Borg genomes\n(15 genomes)': BASE / 'syntenic_analysis_borg/micro_chain/micro_chain_blocks.csv',
}

# Use a different shade per dataset
dataset_colors = [
    '#4477AA',  # blue
    '#228833',  # green
    '#EE6677',  # red
    '#CCBB44',  # gold
]

# ── Figure ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL, DOUBLE_COL * 0.6))
axes = axes.flatten()

for i, ((name, csv_path), color) in enumerate(zip(datasets.items(), dataset_colors)):
    ax = axes[i]
    df = pd.read_csv(csv_path)

    # Use n_genes (or n_anchors if n_genes not present)
    size_col = 'n_genes' if 'n_genes' in df.columns else 'n_anchors'
    sizes = df[size_col].values

    # Determine bin range
    min_size = max(2, sizes.min())
    max_size = sizes.max()
    bins = np.logspace(np.log10(min_size), np.log10(max_size * 1.1), 35)

    ax.hist(sizes, bins=bins, color=color, edgecolor='white', linewidth=0.3,
            alpha=0.85)

    ax.set_xscale('log')
    ax.set_xlabel('Block size (genes)')
    ax.set_ylabel('Count')

    # Formatting for log ticks
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    # Pick sensible ticks based on max
    if max_size > 500:
        ax.set_xticks([2, 5, 10, 50, 100, 500, max_size])
    elif max_size > 50:
        ax.set_xticks([2, 5, 10, 50, max_size])
    else:
        ax.set_xticks([2, 5, 10, 20, max_size])

    # Annotations
    median = np.median(sizes)
    total = len(sizes)
    mean = np.mean(sizes)

    stats_text = (f'n = {total:,}\n'
                  f'median = {median:.0f}\n'
                  f'mean = {mean:.1f}\n'
                  f'max = {max_size:,}')
    ax.text(0.97, 0.95, stats_text, transform=ax.transAxes,
            fontsize=6, va='top', ha='right',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='lightgray', alpha=0.9))

    # Vertical line at median
    ax.axvline(median, color='black', linestyle='--', linewidth=0.6, alpha=0.5)

    ax.set_title(name, fontsize=7.5, pad=4)

    panel_labels = ['a', 'b', 'c', 'd']
    add_panel_label(ax, panel_labels[i])

plt.tight_layout(h_pad=1.5, w_pad=1.2)
save_figure(fig, 'figS3_block_sizes')
plt.close()
print('Done: Figure S3')
