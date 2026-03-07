"""
Browser integration for cluster architecture schema.

Loads precomputed architecture artifacts and provides display helpers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st


def find_schema_dir() -> Optional[Path]:
    """Return the schema output directory at the known analysis output path.

    Convention: ``<project_root>/syntenic_analysis/micro_chain/schema/``
    mirrors the hard-coded ``syntenic_analysis/`` paths used elsewhere in the
    browser for CSV artifacts.
    """
    project_root = Path(__file__).resolve().parent.parent
    schema_dir = project_root / "syntenic_analysis" / "micro_chain" / "schema"
    if schema_dir.is_dir() and (schema_dir / "cluster_architecture_summary.parquet").exists():
        return schema_dir
    return None


@st.cache_data(ttl=600)
def load_architecture_summary(_schema_dir: str) -> Optional[pd.DataFrame]:
    """Load cluster architecture summary from parquet."""
    path = Path(_schema_dir) / "cluster_architecture_summary.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return None


@st.cache_data(ttl=600)
def load_cluster_slots(_schema_dir: str, cluster_id: int) -> Optional[pd.DataFrame]:
    """Load slot summaries for a specific cluster."""
    path = Path(_schema_dir) / "cluster_slots.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    return df[df["cluster_id"] == cluster_id]


@st.cache_data(ttl=600)
def load_slot_assignments(_schema_dir: str, cluster_id: int) -> Optional[pd.DataFrame]:
    """Load slot assignments for a specific cluster."""
    path = Path(_schema_dir) / "cluster_slot_assignments.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    return df[df["cluster_id"] == cluster_id]


@st.cache_data(ttl=600)
def load_cluster_loci(_schema_dir: str, cluster_id: int) -> Optional[pd.DataFrame]:
    """Load locus instances for a specific cluster."""
    path = Path(_schema_dir) / "cluster_loci.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    return df[df["cluster_id"] == cluster_id]


def render_architecture_badge(arch_row: pd.Series) -> str:
    """Render a compact architecture badge for cluster explorer cards.

    Returns HTML string.
    """
    label = arch_row.get("arch_label", "unknown")
    n_slots = int(arch_row.get("n_slots", 0))
    n_core = int(arch_row.get("n_core_slots", 0))
    has_hotspot = bool(arch_row.get("has_replacement_hotspot", False))

    color_map = {
        "coherent": ("#e8f5e9", "#2e7d32"),
        "moderate": ("#fff3e0", "#ef6c00"),
        "fragmented": ("#fce4ec", "#c62828"),
        "variable (replacement)": ("#e3f2fd", "#1565c0"),
    }
    bg, fg = color_map.get(label, ("#f5f5f5", "#616161"))

    badge = (
        f'<span style="background:{bg};color:{fg};border-radius:6px;'
        f'padding:2px 8px;font-size:11px;font-weight:500;">'
        f'{n_slots} slots ({n_core} core) · {label}'
    )
    if has_hotspot:
        badge += ' · hotspot'
    badge += '</span>'
    return badge


def render_architecture_panel(
    schema_dir: str,
    cluster_id: int,
) -> None:
    """Render the architecture panel in the cluster detail view.

    Shows:
    1. Textual summary of cluster architecture
    2. Slot-by-locus alignment matrix
    """
    arch_df = load_architecture_summary(schema_dir)
    if arch_df is None or arch_df.empty:
        return

    arch_row = arch_df[arch_df["cluster_id"] == cluster_id]
    if arch_row.empty:
        return
    arch = arch_row.iloc[0]

    slots_df = load_cluster_slots(schema_dir, cluster_id)
    assignments_df = load_slot_assignments(schema_dir, cluster_id)
    loci_df = load_cluster_loci(schema_dir, cluster_id)

    # -- Textual summary --
    n_slots = int(arch["n_slots"])
    n_core = int(arch["n_core_slots"])
    n_var = int(arch["n_variable_slots"])
    n_loci = int(arch["n_loci"])
    coherence = float(arch["coherence"])
    label = arch["arch_label"]
    has_hotspot = bool(arch["has_replacement_hotspot"])

    summary_parts = [f"**{n_slots} inferred slots** ({n_core} core, {n_var} variable)"]
    summary_parts.append(f"**{n_loci} locus instances** across genomes")
    summary_parts.append(f"Coherence: **{coherence:.0%}** ({label})")
    if has_hotspot and slots_df is not None:
        hotspot_slots = slots_df[slots_df["has_alternate"]]
        if not hotspot_slots.empty:
            hs_indices = ", ".join(str(int(s)) for s in hotspot_slots["slot_idx"])
            summary_parts.append(f"Likely substitution at slot(s): **{hs_indices}**")

    st.markdown(" · ".join(summary_parts[:2]))
    st.markdown(" · ".join(summary_parts[2:]))

    # -- Slot summary table --
    if slots_df is not None and not slots_df.empty:
        st.markdown("**Slot occupancy:**")
        slot_display = slots_df[["slot_idx", "occupancy", "n_occupants", "dispersion", "is_core", "has_alternate"]].copy()
        slot_display["occupancy"] = slot_display["occupancy"].apply(lambda x: f"{x:.0%}")
        slot_display["dispersion"] = slot_display["dispersion"].apply(lambda x: f"{x:.3f}")
        slot_display = slot_display.rename(columns={
            "slot_idx": "Slot",
            "occupancy": "Occupancy",
            "n_occupants": "Loci",
            "dispersion": "Dispersion",
            "is_core": "Core",
            "has_alternate": "Alt. Type",
        })
        st.dataframe(slot_display, hide_index=True, use_container_width=True, height=min(400, 35 * (len(slot_display) + 1)))

    # -- Locus-by-slot alignment matrix --
    if assignments_df is not None and not assignments_df.empty and loci_df is not None:
        st.markdown("**Aligned locus-by-slot matrix:**")
        _render_slot_matrix(assignments_df, loci_df, slots_df)


def _render_slot_matrix(
    assignments_df: pd.DataFrame,
    loci_df: pd.DataFrame,
    slots_df: Optional[pd.DataFrame],
) -> None:
    """Render the locus-by-slot alignment matrix as an HTML table.

    Rows = locus instances, Columns = slots.
    Cells show: present (green), absent (gray), insertion (blue), alternate (orange).
    """
    # Filter out insertions for the main matrix
    main = assignments_df[~assignments_df["is_insertion"]].copy()

    if main.empty:
        st.caption("No slot assignments available.")
        return

    # Get unique loci and slots
    locus_ids = sorted(main["locus_id"].unique())
    slot_indices = sorted(main["slot_idx"].unique())

    if not slot_indices:
        return

    # Build a lookup of which loci are reference
    ref_loci = set(main[main["is_reference"]]["locus_id"].unique())

    # Build the matrix
    # For each (locus, slot): present/absent/alternate
    cell_data = {}
    for _, row in main.iterrows():
        key = (int(row["locus_id"]), int(row["slot_idx"]))
        cell_data[key] = {
            "sim": float(row["similarity"]),
            "gene_id": row["gene_id"],
            "is_ref": bool(row["is_reference"]),
        }

    # Detect alternate types per slot (from slots_df)
    alt_slots = set()
    if slots_df is not None:
        alt_slots = set(
            int(s) for s in slots_df[slots_df["has_alternate"]]["slot_idx"]
        )

    # Build HTML table
    max_loci = 20  # Limit display
    display_loci = locus_ids[:max_loci]
    n_slots = len(slot_indices)

    # Build locus labels from loci_df
    locus_labels = {}
    for _, row in loci_df.iterrows():
        lid = int(row["locus_id"])
        genome = str(row["genome_id"])
        # Shorten genome ID
        short = genome.split("_")[1] if "_" in genome else genome[:10]
        locus_labels[lid] = f"{short}:{row['contig_id']}:{row['start_idx']}-{row['end_idx']}"

    # Core slot markers
    core_slots = set()
    if slots_df is not None:
        core_slots = set(int(s) for s in slots_df[slots_df["is_core"]]["slot_idx"])

    html_parts = ['<div style="overflow-x:auto;"><table style="border-collapse:collapse;font-size:12px;white-space:nowrap;">']

    # Header row
    html_parts.append('<tr><th style="padding:3px 6px;border:1px solid #ddd;background:#f5f5f5;">Locus</th>')
    for s in slot_indices:
        bg = "#e8f5e9" if s in core_slots else ("#e3f2fd" if s in alt_slots else "#f5f5f5")
        html_parts.append(f'<th style="padding:3px 6px;border:1px solid #ddd;background:{bg};text-align:center;">S{s}</th>')
    html_parts.append('</tr>')

    # Data rows
    for lid in display_loci:
        is_ref = lid in ref_loci
        label = locus_labels.get(lid, str(lid))
        row_bg = "#fffde7" if is_ref else "white"
        html_parts.append(f'<tr><td style="padding:3px 6px;border:1px solid #ddd;background:{row_bg};font-weight:{"bold" if is_ref else "normal"};">{label}{"*" if is_ref else ""}</td>')

        for s in slot_indices:
            cell = cell_data.get((lid, s))
            if cell:
                sim = cell["sim"]
                if cell["is_ref"]:
                    bg = "#c8e6c9"
                    text = "ref"
                elif sim >= 0.8:
                    bg = "#a5d6a7"
                    text = f"{sim:.2f}"
                elif sim >= 0.5:
                    bg = "#fff9c4"
                    text = f"{sim:.2f}"
                else:
                    bg = "#ffccbc"
                    text = f"{sim:.2f}"
            else:
                bg = "#f5f5f5"
                text = "—"
            html_parts.append(f'<td style="padding:3px 6px;border:1px solid #ddd;background:{bg};text-align:center;">{text}</td>')

        html_parts.append('</tr>')

    if len(locus_ids) > max_loci:
        html_parts.append(f'<tr><td colspan="{n_slots + 1}" style="padding:3px 6px;border:1px solid #ddd;text-align:center;color:#999;">... and {len(locus_ids) - max_loci} more loci</td></tr>')

    html_parts.append('</table></div>')
    html_parts.append('<p style="font-size:11px;color:#666;">* = reference locus · Green = high similarity · Yellow = moderate · Red = low · Gray = absent</p>')

    st.markdown("".join(html_parts), unsafe_allow_html=True)
