#!/usr/bin/env python3
"""Figure S1: Gap Penalty Comparison.

Three panels comparing hard-cutoff vs concave gap penalty strategies
for collinear chaining in ELSA.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import *

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

setup_style()

# ── Data ────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent / 'benchmarks' / 'evaluation'

with open(BASE / 'gap_penalty_comparison.json') as f:
    data = json.load(f)

configs = ['hard_gap2', 'hard_gap5', 'concave_scale1']
labels = ['Hard gap=2', 'Hard gap=5', 'Concave\n(scale=1.0)']
short_labels = ['Hard\ngap=2', 'Hard\ngap=5', 'Concave\nscale=1']

# Load block CSVs for distributions
block_dfs = {}
for cfg in configs:
    csv_path = BASE / 'gap_penalty_runs' / cfg / 'micro_chain_blocks.csv'
    if csv_path.exists():
        block_dfs[cfg] = pd.read_csv(csv_path)

# ── Figure ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL, DOUBLE_COL * 0.32))

# ── Panel a: Operon recall grouped bar chart ────────────────────────
ax = axes[0]
recall_types = ['strict', 'independent', 'any_coverage']
recall_labels = ['Strict', 'Independent', 'Any']
colors = ['#4477AA', '#66CCEE', '#228833']

x = np.arange(len(configs))
width = 0.22

for i, (rtype, rlabel, color) in enumerate(zip(recall_types, recall_labels, colors)):
    values = [data[cfg]['operon_recall'][rtype] * 100 for cfg in configs]
    bars = ax.bar(x + (i - 1) * width, values, width, label=rlabel, color=color,
                  edgecolor='white', linewidth=0.5)
    # Add value labels on bars
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f'{val:.1f}', ha='center', va='bottom', fontsize=5.5)

ax.set_xticks(x)
ax.set_xticklabels(short_labels, fontsize=6.5)
ax.set_ylabel('Operon recall (%)')
ax.set_ylim(95, 101.5)
ax.legend(loc='lower left', fontsize=6)
add_panel_label(ax, 'a')

# ── Panel b: Block size distributions ───────────────────────────────
ax = axes[1]
colors_cfg = {'hard_gap2': '#4477AA', 'hard_gap5': '#EE6677', 'concave_scale1': '#228833'}
label_map = {'hard_gap2': 'Hard gap=2', 'hard_gap5': 'Hard gap=5', 'concave_scale1': 'Concave'}

bins = np.logspace(np.log10(2), np.log10(5000), 40)

for cfg in configs:
    if cfg in block_dfs:
        sizes = block_dfs[cfg]['n_genes'].values
        ax.hist(sizes, bins=bins, alpha=0.6, color=colors_cfg[cfg],
                label=f'{label_map[cfg]} (n={len(sizes):,})',
                edgecolor='none', density=True)

ax.set_xscale('log')
ax.set_xlabel('Block size (genes)')
ax.set_ylabel('Density')
ax.legend(fontsize=6, loc='upper right')
ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
ax.set_xticks([2, 5, 10, 50, 100, 500, 2000])
ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
add_panel_label(ax, 'b')

# ── Panel c: Blocks vs strict recall ────────────────────────────────
ax = axes[2]

total_blocks = [data[cfg]['block_stats']['total_blocks'] for cfg in configs]
strict_recall = [data[cfg]['operon_recall']['strict'] * 100 for cfg in configs]

for i, (cfg, tb, sr) in enumerate(zip(configs, total_blocks, strict_recall)):
    ax.scatter(tb, sr, s=80, color=colors_cfg[cfg], zorder=5,
               edgecolor='black', linewidth=0.5)
    # Offset labels to avoid overlap
    offsets = {'hard_gap2': (400, -0.08), 'hard_gap5': (400, 0.05),
               'concave_scale1': (500, 0.05)}
    dx, dy = offsets[cfg]
    ax.annotate(label_map[cfg],
                (tb, sr), (tb + dx, sr + dy),
                fontsize=6, ha='left', va='center')

ax.set_xlabel('Total blocks (thousands)')
ax.set_ylabel('Strict recall (%)')
ax.set_ylim(98.2, 99.05)
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x/1000:.0f}'))

# Arrow showing concave achieves similar recall with fewer blocks
arrow_y = 98.32
ax.annotate('', xy=(total_blocks[2] + 200, arrow_y),
            xytext=(total_blocks[0] - 200, arrow_y),
            arrowprops=dict(arrowstyle='<->', color='gray', lw=0.8))
ax.text((total_blocks[0] + total_blocks[2]) / 2, arrow_y + 0.04,
        '59% fewer blocks', ha='center', fontsize=5.5, color='gray', va='bottom')

add_panel_label(ax, 'c')

plt.tight_layout()
save_figure(fig, 'figS1_gap_penalty')
plt.close()
print('Done: Figure S1')
