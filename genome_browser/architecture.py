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
    2. Core backbone diagram (visual slot map)
    3. Detailed slot table (collapsed)
    4. Locus-by-slot alignment matrix (collapsed)
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

    # -- Core backbone diagram --
    if slots_df is not None and not slots_df.empty:
        _render_backbone_diagram(slots_df, assignments_df)

    # -- Detailed slot table (collapsed) --
    if slots_df is not None and not slots_df.empty:
        with st.expander("Slot occupancy table", expanded=False):
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

    # -- Locus-by-slot alignment matrix (collapsed) --
    if assignments_df is not None and not assignments_df.empty and loci_df is not None:
        with st.expander("Aligned locus-by-slot matrix", expanded=False):
            _render_slot_matrix(assignments_df, loci_df, slots_df)


def _get_pfam_for_genes(gene_ids: list) -> dict:
    """Look up PFAM domains for a list of gene_ids from the browser DB."""
    import sqlite3
    db_path = Path(__file__).resolve().parent / "genome_browser.db"
    if not db_path.exists():
        return {}
    try:
        conn = sqlite3.connect(str(db_path))
        placeholders = ",".join("?" for _ in gene_ids)
        rows = conn.execute(
            f"SELECT gene_id, pfam_domains FROM genes WHERE gene_id IN ({placeholders}) "
            f"AND pfam_domains IS NOT NULL AND pfam_domains != ''",
            gene_ids,
        ).fetchall()
        conn.close()
        return {r[0]: r[1] for r in rows}
    except Exception:
        return {}


def _render_backbone_diagram(
    slots_df: pd.DataFrame,
    assignments_df: Optional[pd.DataFrame],
) -> None:
    """Render a visual backbone diagram showing core vs variable slots.

    Core slots are tall (they define the backbone), variable slots are shorter,
    and replacement hotspots are marked orange with PFAM annotations shown.
    """
    slots = slots_df.sort_values("slot_idx")
    n_slots = len(slots)
    max_bar = 180  # max bar height in px
    min_bar = 60   # min bar height

    # Collect gene_ids at hotspot slots to look up PFAM
    hotspot_gene_ids = []
    hotspot_slot_genes = {}  # slot_idx -> [gene_ids]
    if assignments_df is not None and not assignments_df.empty:
        for _, slot in slots.iterrows():
            if bool(slot["has_alternate"]):
                sidx = int(slot["slot_idx"])
                slot_genes = assignments_df[
                    (assignments_df["slot_idx"] == sidx) & (~assignments_df["is_insertion"])
                ]["gene_id"].tolist()
                hotspot_slot_genes[sidx] = slot_genes
                hotspot_gene_ids.extend(slot_genes)

    # Batch PFAM lookup
    pfam_map = _get_pfam_for_genes(hotspot_gene_ids) if hotspot_gene_ids else {}

    # Build hotspot PFAM summary per slot: unique domain sets
    hotspot_pfam_summary = {}
    for sidx, gene_ids in hotspot_slot_genes.items():
        domain_sets = set()
        for gid in gene_ids:
            if gid in pfam_map:
                domain_sets.add(pfam_map[gid])
        if domain_sets:
            hotspot_pfam_summary[sidx] = sorted(domain_sets)

    html = ['<div style="padding:16px 0;width:100%;">']
    html.append(f'<div style="display:flex;align-items:flex-end;gap:6px;height:{max_bar + 40}px;width:100%;">')

    for _, slot in slots.iterrows():
        idx = int(slot["slot_idx"])
        occ = float(slot["occupancy"])
        is_core = bool(slot["is_core"])
        has_alt = bool(slot["has_alternate"])
        disp = float(slot["dispersion"])

        # Bar height = occupancy (the percentage shown on the bar)
        bar_h = max(int(min_bar + (max_bar - min_bar) * occ), min_bar)

        # Color based on slot type
        if has_alt:
            bg = "#e67e22"
            border = "#d35400"
            label_color = "#fff"
        elif is_core:
            if disp < 0.05:
                bg = "#27ae60"
            elif disp < 0.15:
                bg = "#2ecc71"
            else:
                bg = "#58d68d"
            border = "#1e8449"
            label_color = "#fff"
        else:
            bg = "#555"
            border = "#888"
            label_color = "#ccc"

        occ_pct = f"{occ:.0%}"
        tooltip = f"Slot {idx} | Occ: {occ_pct} | Disp: {disp:.3f}"
        if is_core:
            tooltip += " | CORE"
        if has_alt:
            tooltip += " | REPLACEMENT"
            if idx in hotspot_pfam_summary:
                tooltip += " | PFAM: " + " / ".join(hotspot_pfam_summary[idx])

        # Slot label: show "S0", "S1" etc.
        slot_text = f"S{idx}"

        html.append(
            f'<div title="{tooltip}" style="'
            f'flex:1;min-width:48px;height:{bar_h}px;'
            f'background:{bg};border:2px solid {border};border-radius:6px;'
            f'display:flex;flex-direction:column;align-items:center;justify-content:center;'
            f'font-size:15px;color:{label_color};cursor:default;">'
            f'<span style="font-weight:700;">{slot_text}</span>'
            f'<span style="font-size:13px;opacity:0.85;">{occ_pct}</span>'
            f'</div>'
        )

    html.append('</div>')

    # Legend
    html.append(
        '<div style="display:flex;gap:20px;margin-top:14px;font-size:15px;color:#ccc;">'
        '<span><span style="display:inline-block;width:16px;height:16px;'
        'background:#27ae60;border-radius:3px;vertical-align:middle;margin-right:6px;">'
        '</span>Core slot</span>'
        '<span><span style="display:inline-block;width:16px;height:16px;'
        'background:#555;border:1px solid #888;border-radius:3px;vertical-align:middle;margin-right:6px;">'
        '</span>Variable</span>'
        '<span><span style="display:inline-block;width:16px;height:16px;'
        'background:#e67e22;border-radius:3px;vertical-align:middle;margin-right:6px;">'
        '</span>Replacement hotspot</span>'
        '<span style="opacity:0.7;">Height = occupancy % · hover for details</span>'
        '</div>'
    )

    html.append('</div>')

    # Show PFAM details for replacement hotspots below the diagram
    if hotspot_pfam_summary:
        html2 = ['<div style="margin-top:8px;padding:10px 12px;background:#2a2a2a;'
                  'border-radius:6px;border:1px solid #444;">']
        html2.append('<div style="font-size:14px;color:#e67e22;font-weight:600;margin-bottom:6px;">'
                     'Replacement hotspot PFAM domains</div>')
        for sidx in sorted(hotspot_pfam_summary.keys()):
            domains = hotspot_pfam_summary[sidx]
            html2.append(f'<div style="font-size:13px;color:#ddd;margin-bottom:4px;">'
                         f'<span style="color:#e67e22;font-weight:600;">Slot {sidx}:</span> ')
            domain_spans = []
            for d in domains:
                domain_spans.append(
                    f'<span style="background:#3d2600;padding:2px 6px;border-radius:4px;'
                    f'margin-right:4px;font-size:12px;">{d}</span>'
                )
            html2.append(" ".join(domain_spans))
            html2.append('</div>')
        html2.append('</div>')
        st.markdown("".join(html), unsafe_allow_html=True)
        st.markdown("".join(html2), unsafe_allow_html=True)
    else:
        st.markdown("".join(html), unsafe_allow_html=True)


def _render_slot_matrix(
    assignments_df: pd.DataFrame,
    loci_df: pd.DataFrame,
    slots_df: Optional[pd.DataFrame],
) -> None:
    """Render the locus-by-slot alignment matrix as an HTML table.

    Dark-mode friendly colors.
    """
    main = assignments_df[~assignments_df["is_insertion"]].copy()

    if main.empty:
        st.markdown("No slot assignments available.")
        return

    locus_ids = sorted(main["locus_id"].unique())
    slot_indices = sorted(main["slot_idx"].unique())

    if not slot_indices:
        return

    ref_loci = set(main[main["is_reference"]]["locus_id"].unique())

    cell_data = {}
    for _, row in main.iterrows():
        key = (int(row["locus_id"]), int(row["slot_idx"]))
        cell_data[key] = {
            "sim": float(row["similarity"]),
            "gene_id": row["gene_id"],
            "is_ref": bool(row["is_reference"]),
        }

    alt_slots = set()
    if slots_df is not None:
        alt_slots = set(
            int(s) for s in slots_df[slots_df["has_alternate"]]["slot_idx"]
        )

    core_slots = set()
    if slots_df is not None:
        core_slots = set(int(s) for s in slots_df[slots_df["is_core"]]["slot_idx"])

    max_loci = 20
    display_loci = locus_ids[:max_loci]
    n_slots = len(slot_indices)

    locus_labels = {}
    for _, row in loci_df.iterrows():
        lid = int(row["locus_id"])
        genome = str(row["genome_id"])
        short = genome.split("_")[1] if "_" in genome else genome[:10]
        locus_labels[lid] = f"{short}:{row['contig_id']}:{row['start_idx']}-{row['end_idx']}"

    html_parts = [
        '<div style="overflow-x:auto;">'
        '<table style="border-collapse:collapse;font-size:13px;white-space:nowrap;">'
    ]

    # Header row
    html_parts.append(
        '<tr><th style="padding:4px 8px;border:1px solid #444;background:#2a2a2a;color:#ddd;">Locus</th>'
    )
    for s in slot_indices:
        if s in core_slots:
            bg = "#1e4d2b"
        elif s in alt_slots:
            bg = "#4a3000"
        else:
            bg = "#2a2a2a"
        html_parts.append(
            f'<th style="padding:4px 8px;border:1px solid #444;background:{bg};'
            f'color:#ddd;text-align:center;">S{s}</th>'
        )
    html_parts.append('</tr>')

    # Data rows
    for lid in display_loci:
        is_ref = lid in ref_loci
        label = locus_labels.get(lid, str(lid))
        row_bg = "#2d2d1a" if is_ref else "#1e1e1e"
        html_parts.append(
            f'<tr><td style="padding:4px 8px;border:1px solid #444;background:{row_bg};'
            f'color:#ddd;font-weight:{"bold" if is_ref else "normal"};">'
            f'{label}{"*" if is_ref else ""}</td>'
        )

        for s in slot_indices:
            cell = cell_data.get((lid, s))
            if cell:
                sim = cell["sim"]
                if cell["is_ref"]:
                    bg = "#1b5e20"
                    text_color = "#a5d6a7"
                    text = "ref"
                elif sim >= 0.8:
                    bg = "#2e7d32"
                    text_color = "#c8e6c9"
                    text = f"{sim:.2f}"
                elif sim >= 0.5:
                    bg = "#5d4037"
                    text_color = "#ffcc80"
                    text = f"{sim:.2f}"
                else:
                    bg = "#4a1a1a"
                    text_color = "#ef9a9a"
                    text = f"{sim:.2f}"
            else:
                bg = "#2a2a2a"
                text_color = "#666"
                text = "—"
            html_parts.append(
                f'<td style="padding:4px 8px;border:1px solid #444;background:{bg};'
                f'color:{text_color};text-align:center;">{text}</td>'
            )

        html_parts.append('</tr>')

    if len(locus_ids) > max_loci:
        html_parts.append(
            f'<tr><td colspan="{n_slots + 1}" style="padding:4px 8px;border:1px solid #444;'
            f'text-align:center;color:#999;background:#1e1e1e;">'
            f'... and {len(locus_ids) - max_loci} more loci</td></tr>'
        )

    html_parts.append('</table></div>')
    html_parts.append(
        '<p style="font-size:13px;color:#ccc;margin-top:6px;">'
        '* = reference locus &nbsp; '
        '<span style="color:#a5d6a7;">&#9632;</span> high similarity &nbsp; '
        '<span style="color:#ffcc80;">&#9632;</span> moderate &nbsp; '
        '<span style="color:#ef9a9a;">&#9632;</span> low &nbsp; '
        '<span style="color:#666;">&#9632;</span> absent'
        '</p>'
    )

    st.markdown("".join(html_parts), unsafe_allow_html=True)
