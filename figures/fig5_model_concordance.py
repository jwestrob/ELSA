#!/usr/bin/env python3
"""
Figure 5: Model concordance — ELSA produces consistent results across PLMs.

Panels:
  a) Operon recall comparison (ESM2 vs ProtT5)
  b) Block count comparison (total, cross-genus, E. coli)
  c) Block-level overlap between ESM2 and ProtT5 results

Usage:
  python figures/fig5_model_concordance.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import (
    DOUBLE_COL,
    ELSA_COLOR,
    add_panel_label,
    save_figure,
    setup_style,
)

setup_style()

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Colors ────────────────────────────────────────────────────────────
ESM2_COLOR = ELSA_COLOR        # "#4477AA" — consistent with other figures
PROTT5_COLOR = "#AA3377"       # magenta — Tol bright palette

# ── Data ──────────────────────────────────────────────────────────────
# Operon recall (strict / independent / any)
recall_categories = ["Strict", "Independent", "Any\ncoverage"]
esm2_recall = [98.9, 99.0, 99.3]
prott5_recall = [98.0, 98.1, 98.9]

# Block counts
block_labels = ["Total\nblocks", "Cross-genus\nblocks", "E. coli\nblocks"]
esm2_blocks = [80_225, 55_898, 20_117]
prott5_blocks = [76_947, 53_295, 19_212]

# ── Load blocks for overlap analysis ─────────────────────────────────
esm2_blocks_path = PROJECT_ROOT / "benchmarks/elsa_output/cross_species/syntenic_analysis/micro_chain/micro_chain_blocks.csv"
prott5_blocks_path = PROJECT_ROOT / "benchmarks/elsa_output/cross_species_prott5/micro_chain/micro_chain_blocks.csv"

def block_signatures(blocks_df):
    """Create a set of (query_genome, target_genome, query_start, query_end) tuples."""
    sigs = set()
    for _, b in blocks_df.iterrows():
        sigs.add((b["query_genome"], b["target_genome"],
                  int(b["query_start"]), int(b["query_end"]),
                  int(b["target_start"]), int(b["target_end"])))
    return sigs

def blocks_overlap(sig_a, sig_b, threshold=0.5):
    """Check if two blocks overlap on both sides by >= threshold."""
    qg_a, tg_a, qs_a, qe_a, ts_a, te_a = sig_a
    qg_b, tg_b, qs_b, qe_b, ts_b, te_b = sig_b

    if qg_a != qg_b or tg_a != tg_b:
        return False

    # Query-side overlap
    q_overlap = max(0, min(qe_a, qe_b) - max(qs_a, qs_b) + 1)
    q_size = min(qe_a - qs_a + 1, qe_b - qs_b + 1)
    if q_size == 0 or q_overlap / q_size < threshold:
        return False

    # Target-side overlap
    t_overlap = max(0, min(te_a, te_b) - max(ts_a, ts_b) + 1)
    t_size = min(te_a - ts_a + 1, te_b - ts_b + 1)
    if t_size == 0 or t_overlap / t_size < threshold:
        return False

    return True

# Compute anchor size distributions for both
esm2_df = pd.read_csv(esm2_blocks_path)
prott5_df = pd.read_csv(prott5_blocks_path)

esm2_sizes = esm2_df["n_anchors"].values
prott5_sizes = prott5_df["n_anchors"].values

# ── Create figure ─────────────────────────────────────────────────────
fig, axes = plt.subplots(
    1, 3,
    figsize=(DOUBLE_COL, DOUBLE_COL * 0.38),
    gridspec_kw={"width_ratios": [1, 1, 1.1], "wspace": 0.42},
)
ax_a, ax_b, ax_c = axes

# ======================================================================
# Panel a: Operon recall comparison
# ======================================================================
x = np.arange(len(recall_categories))
w = 0.32

bars_esm2 = ax_a.bar(x - w / 2, esm2_recall, w, color=ESM2_COLOR,
                      label="ESM2-t12 (480D)", edgecolor="white", linewidth=0.3)
bars_prott5 = ax_a.bar(x + w / 2, prott5_recall, w, color=PROTT5_COLOR,
                        label="ProtT5-XL (1024D)", edgecolor="white", linewidth=0.3)

# Value labels
for bar_group in [bars_esm2, bars_prott5]:
    for bar in bar_group:
        h = bar.get_height()
        ax_a.text(bar.get_x() + bar.get_width() / 2, h + 0.3,
                  f"{h:.1f}", ha="center", va="bottom", fontsize=5.5,
                  fontweight="semibold")

ax_a.set_xticks(x)
ax_a.set_xticklabels(recall_categories)
ax_a.set_ylabel("Operon recall (%)")
ax_a.set_ylim(95, 101)
ax_a.axhline(100, color="0.75", lw=0.4, ls="--", zorder=0)
ax_a.legend(loc="upper left", fontsize=6, handlelength=1.0,
            bbox_to_anchor=(0.0, 1.0))

# n annotation
ax_a.text(0.97, 0.03, "n = 10,182 instances\n58 operons, 20 genomes",
          transform=ax_a.transAxes, fontsize=5.5, ha="right", va="bottom",
          color="0.4")

add_panel_label(ax_a, "a")

# ======================================================================
# Panel b: Block count comparison
# ======================================================================
x = np.arange(len(block_labels))

bars_esm2_b = ax_b.bar(x - w / 2, [v / 1000 for v in esm2_blocks], w,
                         color=ESM2_COLOR, label="ESM2-t12",
                         edgecolor="white", linewidth=0.3)
bars_prott5_b = ax_b.bar(x + w / 2, [v / 1000 for v in prott5_blocks], w,
                          color=PROTT5_COLOR, label="ProtT5-XL",
                          edgecolor="white", linewidth=0.3)

# Value labels (in thousands)
for bar_group, vals in [(bars_esm2_b, esm2_blocks), (bars_prott5_b, prott5_blocks)]:
    for bar, v in zip(bar_group, vals):
        h = bar.get_height()
        ax_b.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                  f"{v / 1000:.1f}k", ha="center", va="bottom", fontsize=5.5,
                  fontweight="semibold")

ax_b.set_xticks(x)
ax_b.set_xticklabels(block_labels)
ax_b.set_ylabel("Blocks (thousands)")
ax_b.set_ylim(0, 95)
ax_b.legend(loc="upper right", fontsize=6, handlelength=1.0)

add_panel_label(ax_b, "b")

# ======================================================================
# Panel c: Block size distributions (both PLMs overlaid)
# ======================================================================
bins = np.logspace(np.log10(2), np.log10(max(esm2_sizes.max(), prott5_sizes.max()) + 1), 40)

ax_c.hist(esm2_sizes, bins=bins, alpha=0.55, color=ESM2_COLOR,
          label=f"ESM2 (med={int(np.median(esm2_sizes))})", edgecolor="white",
          linewidth=0.3)
ax_c.hist(prott5_sizes, bins=bins, alpha=0.55, color=PROTT5_COLOR,
          label=f"ProtT5 (med={int(np.median(prott5_sizes))})", edgecolor="white",
          linewidth=0.3)

ax_c.set_xscale("log")
ax_c.set_xlabel("Block size (anchor genes)")
ax_c.set_ylabel("Count")
ax_c.legend(loc="upper right", fontsize=6, handlelength=1.0)

# Median lines
ax_c.axvline(np.median(esm2_sizes), color=ESM2_COLOR, ls="--", lw=0.7, alpha=0.8)
ax_c.axvline(np.median(prott5_sizes), color=PROTT5_COLOR, ls="--", lw=0.7, alpha=0.8)

add_panel_label(ax_c, "c")

# ── Save ──────────────────────────────────────────────────────────────
fig.subplots_adjust(left=0.06, right=0.98, bottom=0.18, top=0.92, wspace=0.42)
save_figure(fig, "fig5_model_concordance")
plt.close(fig)
print("Done.")
