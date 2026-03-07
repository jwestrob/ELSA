"""Shared matplotlib style for ELSA manuscript figures (Nature Methods)."""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# ── Colors ───────────────────────────────────────────────────────────
# Colorblind-safe palette (Tol bright)
ECOLI = "#4477AA"
SALMONELLA = "#EE6677"
KLEBSIELLA = "#228833"
ELSA_COLOR = "#4477AA"
MCSCANX_COLOR = "#CCBB44"

SPECIES_COLORS = {
    "E. coli": ECOLI,
    "Salmonella": SALMONELLA,
    "Klebsiella": KLEBSIELLA,
}

# ── Style setup ──────────────────────────────────────────────────────
def setup_style():
    """Apply Nature Methods-compatible matplotlib style."""
    mpl.rcParams.update({
        # Font
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,

        # Spines
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 3,
        "ytick.major.size": 3,

        # Layout
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "figure.facecolor": "white",
        "axes.facecolor": "white",

        # Lines
        "lines.linewidth": 1.0,
        "patch.linewidth": 0.5,

        # Legend
        "legend.frameon": False,
        "legend.borderpad": 0.3,
    })


def add_panel_label(ax, label, x=-0.15, y=1.05):
    """Add bold lowercase panel label (Nature Methods style)."""
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="bottom", ha="left")


def mm_to_inches(mm):
    """Convert mm to inches for figure sizing."""
    return mm / 25.4


# Nature Methods widths
SINGLE_COL = mm_to_inches(89)
DOUBLE_COL = mm_to_inches(183)


def save_figure(fig, name, output_dir="figures/output"):
    """Save as both PDF (vector) and PNG (300 dpi)."""
    from pathlib import Path
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf", format="pdf")
    fig.savefig(out / f"{name}.png", format="png", dpi=300)
    print(f"Saved: {out / name}.pdf and .png")
