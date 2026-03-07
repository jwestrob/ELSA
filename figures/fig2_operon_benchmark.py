#!/usr/bin/env python3
"""Figure 2: Operon benchmark — ELSA vs MCScanX.

Four panels (2×2):
  (a) Grouped bar chart of operon recall (any, independent, strict)
  (b) Violin plots of region-level shared orthogroup rate
  (c) MCScanX block fragmentation — sub-block count histogram
  (d) Block size distributions — ELSA vs MCScanX overlaid
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import *
setup_style()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE = str(Path(__file__).resolve().parent.parent)

# ── Figure layout (2×2) ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL, DOUBLE_COL * 0.62))
ax_a, ax_b = axes[0]
ax_c, ax_d = axes[1]

# ══════════════════════════════════════════════════════════════════════
# Panel (a): Operon recall comparison — grouped bar chart
# ══════════════════════════════════════════════════════════════════════
categories = ["Any\ncoverage", "Independent", "Strict"]
elsa_vals  = [99.3, 99.0, 98.9]
mcscanx_vals = [100.0, 96.4, 80.3]

x = np.arange(len(categories))
width = 0.32

bars_e = ax_a.bar(x - width / 2, elsa_vals, width, color=ELSA_COLOR,
                  edgecolor="white", linewidth=0.3, label="ELSA", zorder=3)
bars_m = ax_a.bar(x + width / 2, mcscanx_vals, width, color=MCSCANX_COLOR,
                  edgecolor="white", linewidth=0.3, label="MCScanX", zorder=3)

for bar in bars_e:
    h = bar.get_height()
    ax_a.text(bar.get_x() + bar.get_width() / 2, h + 1.5,
              f"{h:.1f}%", ha="center", va="bottom", fontsize=6)
for bar in bars_m:
    h = bar.get_height()
    ax_a.text(bar.get_x() + bar.get_width() / 2, h + 1.5,
              f"{h:.1f}%", ha="center", va="bottom", fontsize=6)

ax_a.set_ylabel("Recall (%)")
ax_a.set_ylim(0, 115)
ax_a.set_xticks(x)
ax_a.set_xticklabels(categories)
ax_a.legend(loc="upper right", frameon=False)

# ══════════════════════════════════════════════════════════════════════
# Panel (b): Region-level shared orthogroup rate — violin plots
# ══════════════════════════════════════════════════════════════════════
df_elsa = pd.read_csv(f"{BASE}/benchmarks/evaluation/elsa_region_correspondence.csv")
df_mcs_corr = pd.read_csv(f"{BASE}/benchmarks/evaluation/mcscanx_region_correspondence.csv")
df_mcs_corr = df_mcs_corr[df_mcs_corr["genome_a"] != df_mcs_corr["genome_b"]].copy()

elsa_rate = df_elsa["min_shared_rate"].dropna().values
mcscanx_rate = df_mcs_corr["ortholog_rate"].dropna().values

positions = [1, 2]
parts = ax_b.violinplot(
    [elsa_rate, mcscanx_rate],
    positions=positions,
    showmeans=False, showmedians=False, showextrema=False,
)

colors = [ELSA_COLOR, MCSCANX_COLOR]
for pc, color in zip(parts["bodies"], colors):
    pc.set_facecolor(color)
    pc.set_edgecolor(color)
    pc.set_alpha(0.65)
    pc.set_linewidth(0.5)

bp = ax_b.boxplot(
    [elsa_rate, mcscanx_rate],
    positions=positions,
    widths=0.15,
    patch_artist=True,
    showfliers=False,
    zorder=5,
)
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_edgecolor("black")
    patch.set_linewidth(0.5)
    patch.set_alpha(0.9)
for element in ["whiskers", "caps"]:
    for line in bp[element]:
        line.set_color("black")
        line.set_linewidth(0.5)
for line in bp["medians"]:
    line.set_color("white")
    line.set_linewidth(1.0)

elsa_med = np.median(elsa_rate)
mcscanx_med = np.median(mcscanx_rate)
ax_b.text(1, 1.06, f"med = {elsa_med:.2f}",
          ha="center", va="bottom", fontsize=6, color=ELSA_COLOR,
          fontweight="bold")
ax_b.text(2, 1.06, f"med = {mcscanx_med:.2f}",
          ha="center", va="bottom", fontsize=6, color="#999933",
          fontweight="bold")

ax_b.set_ylabel("Shared orthogroup rate")
ax_b.set_xticks(positions)
ax_b.set_xticklabels(["ELSA", "MCScanX"])
ax_b.set_ylim(-0.05, 1.18)

# ══════════════════════════════════════════════════════════════════════
# Panel (c): MCScanX block fragmentation — sub-block histogram
# ══════════════════════════════════════════════════════════════════════
df_ovp = pd.read_csv(f"{BASE}/benchmarks/evaluation/mcscanx_overprediction_analysis.csv")
sub_blocks = df_ovp["n_sub_blocks"]

bins_sub = np.arange(0.5, sub_blocks.max() + 1.5, 1)
n_vals, _, patches = ax_c.hist(sub_blocks, bins=bins_sub, color=MCSCANX_COLOR,
                                edgecolor="white", linewidth=0.3, zorder=3, alpha=0.85)

# Color the 1-sub-block bars differently (unfragmented)
patches[0].set_facecolor("#DDCC77")  # lighter for "OK"
for p in patches[1:]:
    p.set_facecolor("#AA7744")  # darker for fragmented

pct_frag = 100 * (sub_blocks > 1).mean()
ax_c.text(0.97, 0.95,
          f"{pct_frag:.0f}% fragmented\nmean = {sub_blocks.mean():.1f} sub-blocks\nmax = {sub_blocks.max()}",
          transform=ax_c.transAxes, fontsize=6, va="top", ha="right",
          color="#666")

ax_c.set_xlabel("Sub-blocks per MCScanX block")
ax_c.set_ylabel("Count")
ax_c.set_xlim(0.5, 12.5)

# ══════════════════════════════════════════════════════════════════════
# Panel (d): Block size distributions — ELSA vs MCScanX
# ══════════════════════════════════════════════════════════════════════
# Canonical 30-genome datasets for fair comparison (anchor gene count for both)
eb = pd.read_csv(f"{BASE}/benchmarks/results/cross_species_chain/micro_chain/micro_chain_blocks.csv")
elsa_sizes = eb['n_genes'].values

# MCScanX v2 cross-genome blocks (n_genes = anchor pair count)
mc2 = pd.read_csv(f"{BASE}/benchmarks/results/mcscanx_comparison/mcscanx_blocks_v2.csv")
mc2 = mc2[mc2['query_genome'] != mc2['target_genome']]
mcs_sizes = mc2['n_genes'].values

# Hybrid bins: integer-aligned for small sizes (avoid empty-bin gaps),
# then log-spaced for larger sizes
small_bins = np.arange(1.5, 15.5, 1)  # 1.5, 2.5, ..., 14.5 — one bin per integer
large_bins = np.logspace(np.log10(15.5), np.log10(max(elsa_sizes.max(), mcs_sizes.max()) * 1.05), 28)
log_bins = np.concatenate([small_bins, large_bins])

ax_d.hist(elsa_sizes, bins=log_bins, color=ELSA_COLOR, alpha=0.7,
          edgecolor="white", linewidth=0.3, label=f"ELSA (n={len(elsa_sizes):,})", zorder=3)
ax_d.hist(mcs_sizes, bins=log_bins, color=MCSCANX_COLOR, alpha=0.7,
          edgecolor="white", linewidth=0.3, label=f"MCScanX (n={len(mcs_sizes):,})", zorder=3)

ax_d.set_xscale("log")
ax_d.set_xlim(0.7, None)
ax_d.set_xticks([1, 10, 100, 1000])
ax_d.set_xticklabels(["$10^0$", "$10^1$", "$10^2$", "$10^3$"])
ax_d.minorticks_off()
ax_d.set_xlabel("Block size (anchor genes)")
ax_d.set_ylabel("Count")
ax_d.legend(loc="upper right", fontsize=6, frameon=False)

# Annotate medians
elsa_med_size = np.median(elsa_sizes)
mcs_med_size = np.median(mcs_sizes)
ymax = ax_d.get_ylim()[1]
if elsa_med_size == mcs_med_size:
    # Both medians identical — single line + combined label
    ax_d.axvline(elsa_med_size, color="gray", ls="--", lw=0.8, alpha=0.7)
    ax_d.text(elsa_med_size, ymax * 0.85, f"  med = {elsa_med_size:.0f}\n  (both)",
              fontsize=6, color="#666", va="top")
else:
    ax_d.axvline(elsa_med_size, color=ELSA_COLOR, ls="--", lw=0.8, alpha=0.7)
    ax_d.axvline(mcs_med_size, color="#999933", ls="--", lw=0.8, alpha=0.7)
    ax_d.text(elsa_med_size, ymax * 0.85, f"  {elsa_med_size:.0f}", fontsize=6,
              color=ELSA_COLOR, va="top")
    ax_d.text(mcs_med_size, ymax * 0.72, f"  {mcs_med_size:.0f}", fontsize=6,
              color="#999933", va="top")

# ── Panel labels ──────────────────────────────────────────────────────
add_panel_label(ax_a, "a")
add_panel_label(ax_b, "b")
add_panel_label(ax_c, "c")
add_panel_label(ax_d, "d")

# ── Save ──────────────────────────────────────────────────────────────
fig.tight_layout(h_pad=2.5, w_pad=2.0)
save_figure(fig, "fig2_operon_benchmark")
plt.close(fig)
print("Done.")
