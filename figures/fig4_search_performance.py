#!/usr/bin/env python3
"""
Figure 4: ELSA search performance on E. coli operon benchmark.

Panels:
  a) Recall@k curve with shaded std region
  b) Recall by operon size category
  c) Cross-genus search hit composition (top 20 operons)

Usage:
  python figures/fig4_search_performance.py
"""

import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import (
    DOUBLE_COL,
    ELSA_COLOR,
    KLEBSIELLA,
    SALMONELLA,
    add_panel_label,
    save_figure,
    setup_style,
)

setup_style()

# ── Load data ─────────────────────────────────────────────────────────
EVAL_DIR = Path(__file__).resolve().parent.parent / "benchmarks" / "evaluation"

# Canonical no-PCA search recall JSON (k=1,3,5,10,20,25,50)
json_path = EVAL_DIR / "search_recall_at_k.json"
with open(json_path) as f:
    json_data = json.load(f)

per_operon = json_data["per_operon"]
k_values = sorted(int(k) for k in per_operon[0]["recall_at_k"].keys())

# Build operon size lookup
operon_sizes = {e["operon_id"]: e["n_genes"] for e in per_operon}

# ── Create figure ─────────────────────────────────────────────────────
fig, axes = plt.subplots(
    1, 3,
    figsize=(DOUBLE_COL, DOUBLE_COL * 0.38),
    gridspec_kw={"width_ratios": [1, 1, 1.15], "wspace": 0.38},
)
ax_a, ax_b, ax_c = axes

# ======================================================================
# Panel a: Recall@k curve
# ======================================================================
recalls_by_k = {k: [] for k in k_values}
for entry in per_operon:
    for k in k_values:
        recalls_by_k[k].append(entry["recall_at_k"][str(k)])

means = np.array([np.mean(recalls_by_k[k]) for k in k_values]) * 100
stds = np.array([np.std(recalls_by_k[k]) for k in k_values]) * 100

# Plot on log-ish x-axis positions
x_pos = np.arange(len(k_values))

ax_a.fill_between(x_pos, means - stds, np.minimum(means + stds, 100),
                   alpha=0.18, color=ELSA_COLOR, linewidth=0)
ax_a.plot(x_pos, means, "-o", color=ELSA_COLOR, markersize=4,
          markeredgecolor="white", markeredgewidth=0.4, zorder=3)

# Annotations for key values (k=25 and k=50)
idx_25 = k_values.index(25)
idx_50 = k_values.index(50)
ax_a.annotate(
    f"{means[idx_25]:.1f}%\n(k=25)",
    xy=(idx_25, means[idx_25]),
    xytext=(-38, -22),
    textcoords="offset points",
    fontsize=6,
    color=ELSA_COLOR,
    fontweight="semibold",
    ha="center",
    arrowprops=dict(arrowstyle="-", color=ELSA_COLOR, lw=0.5,
                    shrinkA=0, shrinkB=2),
)
ax_a.annotate(
    f"{means[idx_50]:.1f}%\n(k=50)",
    xy=(idx_50, means[idx_50]),
    xytext=(0, -24),
    textcoords="offset points",
    fontsize=6,
    color=ELSA_COLOR,
    fontweight="semibold",
    ha="center",
    arrowprops=dict(arrowstyle="-", color=ELSA_COLOR, lw=0.5,
                    shrinkA=0, shrinkB=2),
)

ax_a.set_xticks(x_pos)
ax_a.set_xticklabels([str(k) for k in k_values])
ax_a.set_xlabel("k (number of results)")
ax_a.set_ylabel("Mean recall (%)")
ax_a.set_ylim(0, 108)
ax_a.set_xlim(-0.3, len(k_values) - 0.7)

# Light reference line at 100%
ax_a.axhline(100, color="0.75", lw=0.4, ls="--", zorder=0)

# Sample size
n_operons = len(per_operon)
ax_a.text(0.97, 0.03, f"n = {n_operons} operons", transform=ax_a.transAxes,
          fontsize=6, ha="right", va="bottom", color="0.4")

add_panel_label(ax_a, "a")

# ======================================================================
# Panel b: Recall by operon size
# ======================================================================
# Split operons into size categories
categories = {
    "Short (2\u20134 genes)": lambda ng: ng <= 4,
    "Medium (5\u20137 genes)": lambda ng: 5 <= ng <= 7,
    "Long (8\u201314 genes)": lambda ng: ng >= 8,
}

cat_colors = {
    "Short (2\u20134 genes)": "#88CCEE",   # light blue
    "Medium (5\u20137 genes)": ELSA_COLOR,  # dark blue
    "Long (8\u201314 genes)": "#332288",    # indigo
}

cat_styles = {
    "Short (2\u20134 genes)": {"ls": "-", "marker": "o"},
    "Medium (5\u20137 genes)": {"ls": "-", "marker": "s"},
    "Long (8\u201314 genes)": {"ls": "-", "marker": "D"},
}

for cat_name, selector in categories.items():
    subset = [e for e in per_operon if selector(e["n_genes"])]
    n_operons_cat = len(subset)
    cat_means = []
    for k in k_values:
        vals = [e["recall_at_k"][str(k)] for e in subset]
        cat_means.append(np.mean(vals) * 100)

    ax_b.plot(
        x_pos, cat_means,
        color=cat_colors[cat_name],
        markersize=3.5,
        markeredgecolor="white",
        markeredgewidth=0.3,
        label=f"{cat_name} (n={n_operons_cat})",
        zorder=3,
        **cat_styles[cat_name],
    )

ax_b.axhline(100, color="0.75", lw=0.4, ls="--", zorder=0)
ax_b.set_xticks(x_pos)
ax_b.set_xticklabels([str(k) for k in k_values])
ax_b.set_xlabel("k (number of results)")
ax_b.set_ylabel("Mean recall (%)")
ax_b.set_ylim(0, 108)
ax_b.set_xlim(-0.3, len(k_values) - 0.7)
ax_b.legend(loc="lower right", fontsize=6, handlelength=1.5)

add_panel_label(ax_b, "b")

# ======================================================================
# Panel c: Cross-genus hits for top 20 operons (horizontal stacked bar)
# ======================================================================
# Compute per-operon species hit counts.
# JSON truncates cross_genus_hits to top 5 — scale species proportions
# to the true n_cross_genus_hits total.
operon_hits = []
for e in json_data["per_operon"]:
    sp_counts = Counter(h["species"] for h in e["cross_genus_hits"])
    detail_total = sum(sp_counts.values())
    true_total = e["n_cross_genus_hits"]
    if detail_total > 0 and true_total > detail_total:
        scale = true_total / detail_total
        sal = round(sp_counts.get("Salmonella", 0) * scale)
        kleb = true_total - sal
    else:
        sal = sp_counts.get("Salmonella", 0)
        kleb = sp_counts.get("Klebsiella", 0)
    operon_hits.append({
        "operon_id": e["operon_id"],
        "n_genes": e["n_genes"],
        "Salmonella": sal,
        "Klebsiella": kleb,
        "total": true_total,
    })

# Sort by total cross-genus hits descending, then by name
operon_hits.sort(key=lambda x: (-x["total"], x["operon_id"]))

# Take top 20
top = operon_hits[:20]
top.reverse()  # reverse so highest is at top of horizontal bar chart

y_pos = np.arange(len(top))
sal_vals = [o["Salmonella"] for o in top]
kleb_vals = [o["Klebsiella"] for o in top]
labels = [o["operon_id"] for o in top]

bars_sal = ax_c.barh(y_pos, sal_vals, height=0.65, color=SALMONELLA,
                      label="$\it{Salmonella}$", edgecolor="white",
                      linewidth=0.3)
bars_kleb = ax_c.barh(y_pos, kleb_vals, height=0.65, left=sal_vals,
                       color=KLEBSIELLA, label="$\it{Klebsiella}$",
                       edgecolor="white", linewidth=0.3)

ax_c.set_yticks(y_pos)
ax_c.set_yticklabels(labels, fontsize=5.5, family="monospace")
ax_c.set_xlabel("Cross-genus hits")
ax_c.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
max_total = max(o["total"] for o in top)
ax_c.set_xlim(0, max_total + 1)

# Horizontal legend above the bars
ax_c.legend(loc="lower right", bbox_to_anchor=(1.0, 1.02), fontsize=5.5,
            handlelength=0.8, frameon=False, ncol=2, borderpad=0,
            columnspacing=0.8, handletextpad=0.3)

add_panel_label(ax_c, "c", x=-0.28)

# ── Save ──────────────────────────────────────────────────────────────
fig.subplots_adjust(left=0.06, right=0.98, bottom=0.16, top=0.92, wspace=0.42)
save_figure(fig, "fig4_search_performance")
plt.close(fig)
print("Done.")
