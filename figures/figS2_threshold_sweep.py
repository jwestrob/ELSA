#!/usr/bin/env python3
"""Figure S2: Similarity Threshold Sweep.

Two panels showing operon recall and block counts as a function
of the cosine similarity threshold (tau).
Panel a: E. coli (with operon recall).
Panel b: Borg genomes (blocks/anchors only, no ground truth).
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
BASE = Path(__file__).resolve().parent.parent / 'benchmarks' / 'evaluation'

ecoli = pd.read_csv(BASE / 'threshold_sweep_summary.csv')
borg = pd.read_csv(BASE / 'threshold_sweep_borg_summary.csv')

# ── Figure ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, DOUBLE_COL * 0.38))

# ── Panel a: E. coli ────────────────────────────────────────────────
ax1 = axes[0]
ax1_r = ax1.twinx()

# Recall lines (left y-axis)
recall_cols = [('strict_recall', 'Strict', '#4477AA', '-'),
               ('independent_recall', 'Independent', '#66CCEE', '--'),
               ('any_recall', 'Any', '#228833', '-.')]

for col, label, color, ls in recall_cols:
    ax1.plot(ecoli['tau'], ecoli[col] * 100, color=color, linestyle=ls,
             label=label, marker='o', markersize=3, zorder=3)

# Block count (right y-axis)
ax1_r.plot(ecoli['tau'], ecoli['n_blocks'] / 1000, color='#CCBB44',
           linestyle='-', marker='s', markersize=3, label='Blocks',
           alpha=0.7, zorder=2)
ax1_r.fill_between(ecoli['tau'], ecoli['n_blocks'] / 1000, alpha=0.08,
                    color='#CCBB44')

# Mark tau=0.85
ax1.axvline(x=0.85, color='gray', linestyle=':', linewidth=0.8, alpha=0.6)
ax1.text(0.852, 42, r'$\tau$=0.85', fontsize=6, color='gray', va='bottom')

ax1.set_xlabel(r'Similarity threshold ($\tau$)')
ax1.set_ylabel('Operon recall (%)')
ax1_r.set_ylabel('Blocks (thousands)', color='#CCBB44')
ax1_r.tick_params(axis='y', labelcolor='#CCBB44')
ax1_r.spines['right'].set_visible(True)
ax1_r.spines['right'].set_color('#CCBB44')
ax1_r.spines['right'].set_linewidth(0.5)

ax1.set_xlim(0.69, 0.97)
ax1.set_ylim(40, 102)

# Combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax1_r.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='center left',
           fontsize=6, frameon=False)

ax1.set_title('E. coli (20 genomes)', fontsize=8, pad=6)
add_panel_label(ax1, 'a')

# ── Panel b: Borg genomes ──────────────────────────────────────────
ax2 = axes[1]
ax2_r = ax2.twinx()

# Blocks (left y-axis)
ax2.plot(borg['tau'], borg['n_blocks'], color=ELSA_COLOR, marker='o',
         markersize=3, label='Blocks')
ax2.fill_between(borg['tau'], borg['n_blocks'], alpha=0.1, color=ELSA_COLOR)

# Anchors (right y-axis)
ax2_r.plot(borg['tau'], borg['n_anchors'] / 1000, color='#EE6677',
           marker='s', markersize=3, linestyle='--', label='Anchors')

# Mark tau=0.70
ax2.axvline(x=0.70, color='gray', linestyle=':', linewidth=0.8, alpha=0.6)
ax2.text(0.705, borg['n_blocks'].max() * 0.92, r'$\tau$=0.70',
         fontsize=6, color='gray', va='top')

ax2.set_xlabel(r'Similarity threshold ($\tau$)')
ax2.set_ylabel('Syntenic blocks', color=ELSA_COLOR)
ax2.tick_params(axis='y', labelcolor=ELSA_COLOR)
ax2_r.set_ylabel('Anchors (thousands)', color='#EE6677')
ax2_r.tick_params(axis='y', labelcolor='#EE6677')
ax2_r.spines['right'].set_visible(True)
ax2_r.spines['right'].set_color('#EE6677')
ax2_r.spines['right'].set_linewidth(0.5)

ax2.set_xlim(0.69, 0.97)

# Combined legend
lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax2_r.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right',
           fontsize=6, frameon=False)

ax2.set_title('Borg genomes (15 genomes)', fontsize=8, pad=6)
add_panel_label(ax2, 'b')

plt.tight_layout()
save_figure(fig, 'figS2_threshold_sweep')
plt.close()
print('Done: Figure S2')
