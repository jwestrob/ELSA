#!/usr/bin/env python3
"""
Create comparison figures for ELSA vs MCScanX.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from pathlib import Path

# Data from comparison
species_pairs = [
    'E.coli↔E.coli',
    'E.coli↔Salmonella',
    'E.coli↔Klebsiella',
    'Salmonella↔Salmonella',
    'Klebsiella↔Klebsiella',
    'Klebsiella↔Salmonella',
]

# Data from canonical no-PCA benchmark (March 2026)
elsa_counts = [19729, 22799, 27414, 678, 2283, 7322]
mcscanx_counts = [11627, 4430, 9037, 283, 1346, 1473]

# Figure 1: Side-by-side bar chart
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: Blocks by species pair
ax1 = axes[0]
x = np.arange(len(species_pairs))
width = 0.35

bars1 = ax1.bar(x - width/2, elsa_counts, width, label='ELSA', color='#2ecc71', edgecolor='black')
bars2 = ax1.bar(x + width/2, mcscanx_counts, width, label='MCScanX', color='#3498db', edgecolor='black')

ax1.set_xlabel('Species Comparison', fontsize=12)
ax1.set_ylabel('Number of Syntenic Blocks', fontsize=12)
ax1.set_title('Syntenic Blocks Detected by Species Pair', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(species_pairs, rotation=45, ha='right', fontsize=10)
ax1.legend(fontsize=11)
ax1.set_ylim(0, max(elsa_counts) * 1.15)

# Add count labels
for bar, count in zip(bars1, elsa_counts):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
             f'{count:,}', ha='center', va='bottom', fontsize=8, rotation=90)
for bar, count in zip(bars2, mcscanx_counts):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
             f'{count:,}', ha='center', va='bottom', fontsize=8, rotation=90)

# Right: Summary metrics
ax2 = axes[1]

metrics = ['Total\nBlocks', 'Cross-genus\nBlocks', 'Runtime\n(seconds)']
elsa_vals = [80225, 57535, 5]
mcscanx_vals = [28196, 14940, 508]

x = np.arange(len(metrics))
bars1 = ax2.bar(x - width/2, elsa_vals, width, label='ELSA', color='#2ecc71', edgecolor='black')
bars2 = ax2.bar(x + width/2, mcscanx_vals, width, label='MCScanX', color='#3498db', edgecolor='black')

ax2.set_xlabel('Metric', fontsize=12)
ax2.set_ylabel('Value', fontsize=12)
ax2.set_title('Overall Performance Comparison', fontsize=14, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(metrics, fontsize=11)
ax2.legend(fontsize=11)
ax2.set_yscale('log')

# Add ratio annotations
ratios = [f'{e/m:.1f}x' for e, m in zip(elsa_vals[:2], mcscanx_vals[:2])]
ratios.append(f'{mcscanx_vals[2]/elsa_vals[2]:.0f}x slower')
for i, (e, m, ratio) in enumerate(zip(elsa_vals, mcscanx_vals, ratios)):
    max_val = max(e, m)
    ax2.text(i, max_val * 1.5, ratio, ha='center', fontsize=10, fontweight='bold', color='#e74c3c')

plt.tight_layout()
plt.savefig('benchmarks/evaluation/elsa_vs_mcscanx_comparison.png', dpi=150, bbox_inches='tight')
plt.savefig('benchmarks/evaluation/elsa_vs_mcscanx_comparison.pdf', bbox_inches='tight')
print("Saved: benchmarks/evaluation/elsa_vs_mcscanx_comparison.png")

# Figure 2: Cross-genus fold improvement
fig2, ax = plt.subplots(figsize=(8, 5))

cross_pairs = ['E.coli↔Salmonella', 'E.coli↔Klebsiella', 'Klebsiella↔Salmonella']
elsa_cross = [22799, 27414, 7322]
mcscanx_cross = [4430, 9037, 1473]
ratios = [e/m for e, m in zip(elsa_cross, mcscanx_cross)]

colors = ['#e74c3c', '#9b59b6', '#f39c12']
bars = ax.bar(cross_pairs, ratios, color=colors, edgecolor='black', linewidth=1.5)

ax.axhline(y=1, color='gray', linestyle='--', linewidth=1, label='MCScanX baseline')
ax.set_ylabel('ELSA / MCScanX Block Count Ratio', fontsize=12)
ax.set_title('ELSA Detects More Cross-Genus Synteny', fontsize=14, fontweight='bold')
ax.set_ylim(0, max(ratios) * 1.2)

# Add ratio labels
for bar, ratio in zip(bars, ratios):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
            f'{ratio:.1f}x', ha='center', fontsize=14, fontweight='bold')

plt.xticks(rotation=15, ha='right', fontsize=11)
plt.tight_layout()
plt.savefig('benchmarks/evaluation/elsa_cross_genus_advantage.png', dpi=150, bbox_inches='tight')
plt.savefig('benchmarks/evaluation/elsa_cross_genus_advantage.pdf', bbox_inches='tight')
print("Saved: benchmarks/evaluation/elsa_cross_genus_advantage.png")

print("\nDone!")
