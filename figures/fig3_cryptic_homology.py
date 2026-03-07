#!/usr/bin/env python3
"""
Figure 3: Cryptic homology detection by PLM embeddings.

Panel a — Embedding cosine similarity vs pairwise sequence identity (hexbin).
           Highlights the "twilight zone" where BLAST fails but ELSA succeeds.
Panel b — Cross-genus syntenic block counts: ELSA vs MCScanX.

Nature Methods double-column (183 mm).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import FancyArrowPatch

setup_style()

# ── Data ────────────────────────────────────────────────────────────────
DATA_CSV = str(
    Path(__file__).resolve().parent.parent
    / 'benchmarks' / 'evaluation' / 'cosine_vs_identity.csv'
)
df = pd.read_csv(DATA_CSV)
df['seq_id_pct'] = df['sequence_identity'] * 100  # convert to %

# Hardcoded benchmark numbers (verified)
ELSA_CROSS_GENUS = 55_898
MCSCANX_CROSS_GENUS = 14_186
ELSA_ONLY = 8_069   # blocks invisible to BLAST/MCScanX
ELSA_SHARED = ELSA_CROSS_GENUS - ELSA_ONLY  # blocks also found by MCScanX
RATIO = ELSA_CROSS_GENUS / MCSCANX_CROSS_GENUS

# ── Figure ──────────────────────────────────────────────────────────────
fig, (ax_a, ax_b) = plt.subplots(
    1, 2,
    figsize=(DOUBLE_COL, DOUBLE_COL * 0.40),
    gridspec_kw={'width_ratios': [1.4, 1], 'wspace': 0.35},
)

# ====================================================================
# Panel a: Cosine similarity vs sequence identity (hexbin)
# ====================================================================

hb = ax_a.hexbin(
    df['seq_id_pct'],
    df['cosine_similarity'],
    gridsize=35,
    cmap='viridis',
    mincnt=1,
    linewidths=0.2,
    edgecolors='face',
)

# Colorbar
cb = fig.colorbar(hb, ax=ax_a, shrink=0.75, pad=0.02, aspect=20)
cb.set_label('Gene pairs', fontsize=7)
cb.ax.tick_params(labelsize=6)

# Twilight zone line
ax_a.axvline(30, color='#CC3311', ls='--', lw=0.8, zorder=5)
ax_a.text(
    28, -0.55, 'BLAST\ntwilight\nzone',
    ha='right', va='bottom', fontsize=6.5, color='#CC3311',
    fontstyle='italic', linespacing=0.95,
)

# Annotate upper-left quadrant (cryptic homology region)
# Draw a subtle shaded rectangle
from matplotlib.patches import Rectangle
rect = Rectangle(
    (0, 0.8), 30, 0.25,
    linewidth=0.6, edgecolor='#CC3311', facecolor='#CC3311',
    alpha=0.07, zorder=1, linestyle='-',
)
ax_a.add_patch(rect)

# Count points in the cryptic region
n_cryptic = len(df[(df['seq_id_pct'] < 30) & (df['cosine_similarity'] > 0.8)])
ax_a.annotate(
    f'Cryptic homologs\n(n = {n_cryptic})',
    xy=(15, 0.87), xytext=(50, 0.4),
    fontsize=6.5, color='#CC3311',
    ha='center', va='center',
    arrowprops=dict(
        arrowstyle='->', color='#CC3311', lw=0.7,
        connectionstyle='arc3,rad=-0.2',
    ),
)

ax_a.set_xlabel('Pairwise sequence identity (%)')
ax_a.set_ylabel('Embedding cosine similarity')
ax_a.set_xlim(0, 105)
ax_a.set_ylim(-0.75, 1.05)

add_panel_label(ax_a, 'a')

# ====================================================================
# Panel b: ELSA vs MCScanX cross-genus blocks (stacked bar)
# ====================================================================

bar_width = 0.55
x_positions = [0, 1]

# MCScanX bar (single color)
ax_b.bar(
    x_positions[1], MCSCANX_CROSS_GENUS,
    width=bar_width, color=MCSCANX_COLOR,
    edgecolor='white', linewidth=0.5, zorder=3,
    label='MCScanX',
)

# ELSA bar — stacked: shared portion + ELSA-only portion
ax_b.bar(
    x_positions[0], ELSA_SHARED,
    width=bar_width, color=ELSA_COLOR,
    edgecolor='white', linewidth=0.5, zorder=3,
    label='ELSA (BLAST-visible)',
)
ax_b.bar(
    x_positions[0], ELSA_ONLY, bottom=ELSA_SHARED,
    width=bar_width, color='#882255',
    edgecolor='white', linewidth=0.5, zorder=3,
    label='ELSA-only (cryptic)',
)

# Annotate bar values — offset ELSA label to the left to avoid legend
ax_b.text(
    x_positions[0] - 0.05, ELSA_CROSS_GENUS + 800,
    f'{ELSA_CROSS_GENUS:,}',
    ha='center', va='bottom', fontsize=7.5, fontweight='bold',
    color='#333333',
)
ax_b.text(
    x_positions[1], MCSCANX_CROSS_GENUS + 800,
    f'{MCSCANX_CROSS_GENUS:,}',
    ha='center', va='bottom', fontsize=7.5, fontweight='bold',
    color='#333333',
)

# Annotate ELSA-only segment — place to the right, clear of legend
midpoint_elsa_only = ELSA_SHARED + ELSA_ONLY / 2
ax_b.annotate(
    f'{ELSA_ONLY:,}\ncryptic',
    xy=(x_positions[0] + bar_width / 2, midpoint_elsa_only),
    xytext=(x_positions[0] + 0.65, midpoint_elsa_only - 8000),
    fontsize=6.5, color='#882255', ha='left', va='center',
    arrowprops=dict(
        arrowstyle='->', color='#882255', lw=0.7,
        connectionstyle='arc3,rad=-0.15',
    ),
)

# Ratio annotation between bars
mid_x = (x_positions[0] + x_positions[1]) / 2
y_arrow = MCSCANX_CROSS_GENUS * 0.55
ax_b.annotate(
    '', xy=(x_positions[0] + bar_width / 2 + 0.02, y_arrow),
    xytext=(x_positions[1] - bar_width / 2 - 0.02, y_arrow),
    arrowprops=dict(
        arrowstyle='<->', color='#333333', lw=0.8,
    ),
)
ax_b.text(
    mid_x, y_arrow + 1200,
    f'{RATIO:.1f}\u00d7',
    ha='center', va='bottom', fontsize=9, fontweight='bold',
    color='#333333',
)

ax_b.set_xticks(x_positions)
ax_b.set_xticklabels(['ELSA', 'MCScanX'], fontsize=8)
ax_b.set_ylabel('Cross-genus syntenic blocks')
ax_b.set_ylim(0, ELSA_CROSS_GENUS * 1.32)
ax_b.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}'))

# Legend — place in upper-right, above the bars
ax_b.legend(
    loc='upper right', fontsize=6, frameon=False,
    handlelength=1.2, handleheight=0.8,
)

add_panel_label(ax_b, 'b')

# ── Save ────────────────────────────────────────────────────────────────
save_figure(fig, 'fig3_cryptic_homology')
plt.close(fig)
print('Done.')
