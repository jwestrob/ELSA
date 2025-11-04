"""Operon-style micro pipeline using embedding-first shingles and Sinkhorn."""

from __future__ import annotations

from dataclasses import dataclass
import ast
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import json
import sqlite3

import numpy as np
import pandas as pd

from operon_embed.preprocess import (
    fit_preprocessor,
    save_preprocessor,
    transform_preprocessor,
)
from operon_embed.shingle import build_shingles
from operon_embed.index_hnsw import build_hnsw_index, save_metadata
from operon_embed.sinkhorn import sinkhorn_distance


@dataclass
class OperonSummary:
    """Summary of the operon micro pipeline."""

    num_genes: int
    num_blocks: int
    num_pairs: int
    num_filtered_pairs: int
    num_clusters: int
    index_built: bool


def _log(console, message: str) -> None:
    if console is not None:
        try:
            console.print(message)
            return
        except Exception:
            pass
    print(message)


def _apply_macro_derep(
    pairs: pd.DataFrame,
    blocks_df: pd.DataFrame,
    db_path: Optional[Path],
    console=None,
) -> pd.DataFrame:
    """Remove operon pairs fully contained inside macro spans on both sides."""

    if db_path is None or pairs is None or pairs.empty or blocks_df.empty:
        return pairs

    db_path = Path(db_path)
    if not db_path.exists():
        return pairs

    try:
        conn = sqlite3.connect(str(db_path))
    except Exception as exc:  # pragma: no cover - defensive
        _log(console, f"[yellow]Operon dereplication skipped (DB open failed: {exc})[/yellow]")
        return pairs

    try:
        macro_df = pd.read_sql_query(
            """
            SELECT 
                sb.block_id AS macro_block_id,
                gb.block_role AS side,
                CAST(g.genome_id AS TEXT) AS genome_id,
                CAST(g.contig_id AS TEXT) AS contig_id,
                MIN(g.start_pos) AS span_start,
                MAX(g.end_pos) AS span_end
            FROM syntenic_blocks sb
            JOIN gene_block_mappings gb ON gb.block_id = sb.block_id
            JOIN genes g ON g.gene_id = gb.gene_id
            WHERE COALESCE(sb.block_type, 'macro') NOT IN ('micro', 'operon')
            GROUP BY sb.block_id, gb.block_role, g.genome_id, g.contig_id
            """,
            conn,
        )
    except Exception as exc:
        _log(console, f"[yellow]Operon dereplication skipped (macro span query failed: {exc})[/yellow]")
        conn.close()
        return pairs

    if macro_df.empty:
        conn.close()
        return pairs

    macro_q = macro_df[macro_df["side"].str.lower() == "query"].copy()
    macro_t = macro_df[macro_df["side"].str.lower() == "target"].copy()
    if macro_q.empty or macro_t.empty:
        conn.close()
        return pairs

    macro_q = macro_q.rename(
        columns={
            "genome_id": "macro_q_genome_id",
            "contig_id": "macro_q_contig_id",
            "span_start": "macro_q_start",
            "span_end": "macro_q_end",
        }
    )
    macro_t = macro_t.rename(
        columns={
            "genome_id": "macro_t_genome_id",
            "contig_id": "macro_t_contig_id",
            "span_start": "macro_t_start",
            "span_end": "macro_t_end",
        }
    )

    conn.close()

    block_info = (
        blocks_df[
            ["block_id", "sample_id", "contig_id", "start_bp", "end_bp"]
        ]
        .copy()
        .rename(
            columns={
                "sample_id": "genome_id",
                "contig_id": "contig_id",
                "start_bp": "start_bp",
                "end_bp": "end_bp",
            }
        )
    )
    if block_info.empty:
        return pairs

    block_info["genome_id"] = block_info["genome_id"].astype(str)
    block_info["contig_id"] = block_info["contig_id"].astype(str)
    for col in ("start_bp", "end_bp"):
        block_info[col] = pd.to_numeric(block_info[col], errors="coerce")
    block_info = block_info.dropna(subset=["start_bp", "end_bp"])
    mask_swap = block_info["start_bp"] > block_info["end_bp"]
    if mask_swap.any():
        tmp = block_info.loc[mask_swap, "start_bp"].copy()
        block_info.loc[mask_swap, "start_bp"] = block_info.loc[mask_swap, "end_bp"]
        block_info.loc[mask_swap, "end_bp"] = tmp

    if block_info.empty:
        return pairs

    pair_df = pairs.copy().reset_index(drop=True)
    pair_df["_pair_idx"] = np.arange(len(pair_df), dtype=np.int64)
    pair_df = pair_df.merge(
        block_info.add_prefix("q_"),
        left_on="block_id",
        right_on="q_block_id",
        how="left",
    )
    pair_df = pair_df.merge(
        block_info.add_prefix("t_"),
        left_on="neighbor_id",
        right_on="t_block_id",
        how="left",
    )

    # Guard against incomplete mappings
    required_cols = [
        "q_genome_id",
        "q_contig_id",
        "q_start_bp",
        "q_end_bp",
        "t_genome_id",
        "t_contig_id",
        "t_start_bp",
        "t_end_bp",
    ]
    pair_df = pair_df.dropna(subset=required_cols)
    if pair_df.empty:
        return pairs

    for col in required_cols:
        if col.endswith("_bp"):
            pair_df[col] = pd.to_numeric(pair_df[col], errors="coerce")
    mask_q = pair_df["q_start_bp"] > pair_df["q_end_bp"]
    if mask_q.any():
        tmp = pair_df.loc[mask_q, "q_start_bp"].copy()
        pair_df.loc[mask_q, "q_start_bp"] = pair_df.loc[mask_q, "q_end_bp"]
        pair_df.loc[mask_q, "q_end_bp"] = tmp
    mask_t = pair_df["t_start_bp"] > pair_df["t_end_bp"]
    if mask_t.any():
        tmp = pair_df.loc[mask_t, "t_start_bp"].copy()
        pair_df.loc[mask_t, "t_start_bp"] = pair_df.loc[mask_t, "t_end_bp"]
        pair_df.loc[mask_t, "t_end_bp"] = tmp

    def _contained_indices(
        pdf: pd.DataFrame,
        macro_q_df: pd.DataFrame,
        macro_t_df: pd.DataFrame,
        q_prefix: str,
        t_prefix: str,
    ) -> set[int]:
        if pdf.empty:
            return set()
        left = pdf.merge(
            macro_q_df,
            left_on=[f"{q_prefix}_genome_id", f"{q_prefix}_contig_id"],
            right_on=["macro_q_genome_id", "macro_q_contig_id"],
            how="inner",
        )
        if left.empty:
            return set()
        left = left[
            (left[f"{q_prefix}_start_bp"] >= left["macro_q_start"])
            & (left[f"{q_prefix}_end_bp"] <= left["macro_q_end"])
        ]
        if left.empty:
            return set()
        joined = left.merge(macro_t_df, on="macro_block_id", how="inner")
        if joined.empty:
            return set()
        joined = joined[
            (joined[f"{t_prefix}_genome_id"] == joined["macro_t_genome_id"])
            & (joined[f"{t_prefix}_contig_id"] == joined["macro_t_contig_id"])
            & (joined[f"{t_prefix}_start_bp"] >= joined["macro_t_start"])
            & (joined[f"{t_prefix}_end_bp"] <= joined["macro_t_end"])
        ]
        if joined.empty:
            return set()
        return set(joined["_pair_idx"].astype(int).tolist())

    drop_idx = _contained_indices(pair_df, macro_q, macro_t, "q", "t")
    drop_idx |= _contained_indices(pair_df, macro_q, macro_t, "t", "q")

    if not drop_idx:
        return pairs

    keep_mask = ~pair_df["_pair_idx"].isin(drop_idx)
    removed = (~keep_mask).sum()
    if removed <= 0:
        return pairs

    _log(
        console,
        f"[dim]Operon dereplication removed {removed} pair(s) fully contained in macro spans[/dim]",
    )

    columns = list(pairs.columns)
    result = pair_df.loc[keep_mask, columns].copy()
    result = result.reset_index(drop=True)
    return result


def _merge_shifted_clusters(
    blocks_df: pd.DataFrame,
    filtered_pairs: pd.DataFrame,
    min_genome_support: int,
    *,
    max_shift_genes: int = 2,
    console=None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Merge clusters that are identical up to small per-sample shifts.

    Sliding shingle windows often yield multiple connected components that differ
    only by ±1 gene offsets in every member genome. Consolidate such duplicates by
    keeping the earliest (smallest global_start) cluster and dropping the rest.
    """

    if blocks_df.empty:
        return blocks_df, filtered_pairs, pd.DataFrame(columns=["cluster_id", "size", "genome_support"])

    valid_blocks = blocks_df[blocks_df["cluster_id"] > 0].copy()
    if valid_blocks.empty:
        return blocks_df, filtered_pairs, pd.DataFrame(columns=["cluster_id", "size", "genome_support"])

    # Build per-cluster position maps keyed by (sample_id, contig_id)
    cluster_positions: Dict[int, Dict[Tuple[str, str], int]] = {}
    for cid, group in valid_blocks.groupby("cluster_id"):
        positions: Dict[Tuple[str, str], int] = {}
        for row in group.itertuples(index=False):
            positions[(str(row.sample_id), str(row.contig_id))] = int(row.global_start)
        cluster_positions[int(cid)] = positions

    # Group clusters that share the same set of genomes/contigs
    key_to_clusters: Dict[Tuple[Tuple[str, str], ...], List[int]] = {}
    for cid, positions in cluster_positions.items():
        key = tuple(sorted(positions.keys()))
        key_to_clusters.setdefault(key, []).append(cid)

    to_drop: set[int] = set()
    for key, cids in key_to_clusters.items():
        if len(cids) <= 1:
            continue
        cids_sorted = sorted(cids, key=lambda cid: min(cluster_positions[cid].values()))
        keeper = cids_sorted[0]
        base_positions = cluster_positions[keeper]
        for cid in cids_sorted[1:]:
            positions = cluster_positions[cid]
            shifts = [abs(positions[(sample, contig)] - base_positions[(sample, contig)]) for sample, contig in key]
            if all(shift <= max_shift_genes for shift in shifts):
                to_drop.add(cid)
            else:
                keeper = cid
                base_positions = positions

    if not to_drop:
        clusters_df = (
            blocks_df.groupby("cluster_id")
            .agg(size=("block_id", "size"), genome_support=("sample_id", "nunique"))
            .reset_index()
        )
        clusters_df = clusters_df[clusters_df["cluster_id"] > 0]
        clusters_df = clusters_df[clusters_df["genome_support"] >= int(min_genome_support)]
        clusters_df = clusters_df.sort_values("cluster_id").reset_index(drop=True)
        return blocks_df, filtered_pairs, clusters_df

    drop_block_ids = blocks_df[blocks_df["cluster_id"].isin(to_drop)]["block_id"].astype(int).tolist()
    if drop_block_ids:
        blocks_df = blocks_df[~blocks_df["block_id"].isin(drop_block_ids)].copy()
        if not filtered_pairs.empty:
            filtered_pairs = filtered_pairs[
                (~filtered_pairs["block_id"].isin(drop_block_ids))
                & (~filtered_pairs["neighbor_id"].isin(drop_block_ids))
            ].copy()

    # Recompute cluster statistics on surviving blocks
    if blocks_df.empty:
        clusters_df = pd.DataFrame(columns=["cluster_id", "size", "genome_support"])
        filtered_pairs = filtered_pairs.iloc[0:0]
        return blocks_df, filtered_pairs, clusters_df

    clusters_df = (
        blocks_df.groupby("cluster_id")
        .agg(size=("block_id", "size"), genome_support=("sample_id", "nunique"))
        .reset_index()
    )
    clusters_df = clusters_df[clusters_df["cluster_id"] > 0]
    clusters_df = clusters_df[clusters_df["genome_support"] >= int(min_genome_support)]

    # Drop clusters that fell below support threshold
    keep_cluster_ids = set(clusters_df["cluster_id"].astype(int).tolist())
    blocks_df = blocks_df[blocks_df["cluster_id"].isin(keep_cluster_ids)].copy()
    if not filtered_pairs.empty:
        filtered_pairs = filtered_pairs[
            filtered_pairs["block_id"].isin(blocks_df["block_id"])
            & filtered_pairs["neighbor_id"].isin(blocks_df["block_id"])
        ].copy()

    # Renumber clusters consecutively starting at 1
    sorted_ids = sorted(keep_cluster_ids)
    id_map = {old: idx + 1 for idx, old in enumerate(sorted_ids)}
    blocks_df["cluster_id"] = blocks_df["cluster_id"].map(id_map).astype(int)
    clusters_df["cluster_id"] = clusters_df["cluster_id"].map(id_map)
    clusters_df = clusters_df.sort_values("cluster_id").reset_index(drop=True)

    if not filtered_pairs.empty:
        block_to_cluster = blocks_df.set_index("block_id")["cluster_id"]
        filtered_pairs["cluster_id"] = filtered_pairs["block_id"].map(block_to_cluster)

    if console is not None and to_drop:
        console.print(
            f"[dim]Merged {len(to_drop)} operon clusters that differed only by small per-sample shifts[/dim]"
        )

    return blocks_df, filtered_pairs, clusters_df


class _UnionFind:
    def __init__(self, elements: Iterable[int]):
        self.parent = {int(x): int(x) for x in elements}

    def find(self, x: int) -> int:
        px = self.parent.get(x, x)
        if px != x:
            px = self.find(px)
            self.parent[x] = px
        return px

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if ra < rb:
            self.parent[rb] = ra
        else:
            self.parent[ra] = rb


def _merge_adjacent_clusters(
    blocks_df: pd.DataFrame,
    filtered_pairs: pd.DataFrame,
    min_genome_support: int,
    *,
    max_gap_bp: int = 50,
    support_ratio: float = 0.8,
    console=None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Merge clusters that form contiguous conserved loci across member genomes."""

    if blocks_df.empty:
        clusters_df = pd.DataFrame(columns=["cluster_id", "size", "genome_support"])
        return blocks_df, filtered_pairs, clusters_df

    if support_ratio <= 0.0 or support_ratio > 1.0:
        raise ValueError("support_ratio must be in (0, 1]")

    agg = (
        blocks_df.groupby(["cluster_id", "sample_id", "contig_id"])
        .agg(cluster_start_bp=("start_bp", "min"), cluster_end_bp=("end_bp", "max"))
        .reset_index()
    )

    if agg.empty:
        clusters_df = pd.DataFrame(columns=["cluster_id", "size", "genome_support"])
        return blocks_df, filtered_pairs, clusters_df

    membership_map: Dict[int, Dict[Tuple[str, str], Tuple[int, int]]] = {}
    for row in agg.itertuples(index=False):
        cid = int(row.cluster_id)
        membership_map.setdefault(cid, {})[(str(row.sample_id), str(row.contig_id))] = (
            int(row.cluster_start_bp),
            int(row.cluster_end_bp),
        )

    # Group clusters by identical membership (same genomes/contigs)
    key_to_clusters: Dict[Tuple[Tuple[str, str], ...], List[int]] = {}
    for cid, members in membership_map.items():
        key = tuple(sorted(members.keys()))
        key_to_clusters.setdefault(key, []).append(cid)

    uf = _UnionFind(membership_map.keys())

    for key, cluster_ids in key_to_clusters.items():
        if len(cluster_ids) <= 1:
            continue
        cluster_ids_sorted = sorted(
            cluster_ids,
            key=lambda cid: sum(membership_map[cid][member][0] for member in key),
        )
        for prev, curr in zip(cluster_ids_sorted, cluster_ids_sorted[1:]):
            prev_members = membership_map[prev]
            curr_members = membership_map[curr]
            contiguous = True
            for member in key:
                prev_end = prev_members[member][1]
                curr_start = curr_members[member][0]
                gap = curr_start - prev_end
                if gap > max_gap_bp:
                    contiguous = False
                    break
            if contiguous:
                uf.union(prev, curr)

    # Record adjacency support across genomes for clusters that may have
    # slightly different membership sets (e.g., missing genes in some genomes).
    genome_map: Dict[Tuple[str, str], List[Tuple[int, int, int]]] = {}
    for cid, members in membership_map.items():
        for (sample_id, contig_id), (start_bp, end_bp) in members.items():
            genome_map.setdefault((sample_id, contig_id), []).append((cid, start_bp, end_bp))

    pair_support: Dict[Tuple[int, int], Set[str]] = {}
    for (sample_id, contig_id), clusters in genome_map.items():
        clusters_sorted = sorted(clusters, key=lambda x: x[1])
        for (cid_a, start_a, end_a), (cid_b, start_b, end_b) in zip(clusters_sorted, clusters_sorted[1:]):
            gap = start_b - end_a
            if gap > max_gap_bp:
                continue
            parent_a = uf.find(cid_a)
            parent_b = uf.find(cid_b)
            if parent_a == parent_b:
                continue
            key = (parent_a, parent_b) if parent_a < parent_b else (parent_b, parent_a)
            support_set = pair_support.setdefault(key, set())
            support_set.add(sample_id)

    for (parent_a, parent_b), genomes in pair_support.items():
        if len(genomes) < int(min_genome_support):
            continue
        size_a = len(membership_map.get(parent_a, {}))
        size_b = len(membership_map.get(parent_b, {}))
        denom = max(1, min(size_a, size_b))
        ratio = len(genomes) / denom
        if ratio >= support_ratio:
            uf.union(parent_a, parent_b)

    parent_map = {cid: uf.find(cid) for cid in membership_map.keys()}
    if all(parent_map[cid] == cid for cid in parent_map):
        clusters_df = (
            blocks_df.groupby("cluster_id")
            .agg(size=("block_id", "size"), genome_support=("sample_id", "nunique"))
            .reset_index()
        )
        clusters_df = clusters_df[clusters_df["cluster_id"] > 0]
        clusters_df = clusters_df[clusters_df["genome_support"] >= int(min_genome_support)]
        clusters_df = clusters_df.sort_values("cluster_id").reset_index(drop=True)
        return blocks_df, filtered_pairs, clusters_df

    # Remap cluster IDs according to union-find parent
    blocks_df = blocks_df.copy()
    blocks_df["cluster_id"] = blocks_df["cluster_id"].map(lambda cid: parent_map.get(int(cid), int(cid)))

    # Recompute statistics and enforce genome support
    clusters_df = (
        blocks_df.groupby("cluster_id")
        .agg(size=("block_id", "size"), genome_support=("sample_id", "nunique"))
        .reset_index()
    )
    clusters_df = clusters_df[clusters_df["cluster_id"] > 0]
    clusters_df = clusters_df[clusters_df["genome_support"] >= int(min_genome_support)]

    keep_ids = set(clusters_df["cluster_id"].astype(int).tolist())
    blocks_df = blocks_df[blocks_df["cluster_id"].isin(keep_ids)].copy()

    # Renumber sequentially for readability
    sorted_ids = sorted(keep_ids)
    id_map = {old: idx + 1 for idx, old in enumerate(sorted_ids)}
    blocks_df["cluster_id"] = blocks_df["cluster_id"].map(id_map).astype(int)
    clusters_df["cluster_id"] = clusters_df["cluster_id"].map(id_map)
    clusters_df = clusters_df.sort_values("cluster_id").reset_index(drop=True)

    if not filtered_pairs.empty:
        block_to_cluster = blocks_df.set_index("block_id")["cluster_id"]
        filtered_pairs = filtered_pairs[
            filtered_pairs["block_id"].isin(block_to_cluster.index)
            & filtered_pairs["neighbor_id"].isin(block_to_cluster.index)
        ].copy()
        filtered_pairs["cluster_id"] = filtered_pairs["block_id"].map(block_to_cluster)

    merged_counts = len(set(parent_map.values())) - len(clusters_df)
    if console is not None and merged_counts > 0:
        console.print(
            f"[dim]Merged {merged_counts} operon cluster(s) that formed conserved loci (≤{max_gap_bp} bp gap, ≥{support_ratio:.2f} support ratio)[/dim]"
        )

    return blocks_df, filtered_pairs, clusters_df


def _project_operon_pairs_to_db(
    conn: sqlite3.Connection,
    filtered_pairs: pd.DataFrame,
    blocks_df: pd.DataFrame,
    clusters_df: pd.DataFrame,
    db_gene_ids: np.ndarray,
) -> Tuple[int, int, int]:
    """Project operon pairs into syntenic_blocks/clusters/gene_block_mappings."""

    if filtered_pairs.empty or clusters_df is None or clusters_df.empty:
        return 0, 0

    cur = conn.cursor()

    # Clear previous operon projections
    try:
        existing = cur.execute(
            "SELECT block_id FROM syntenic_blocks WHERE block_type='operon'"
        ).fetchall()
        existing_ids = [int(r[0]) for r in existing]
        if existing_ids:
            cur.executemany(
                "DELETE FROM gene_block_mappings WHERE block_id = ?",
                [(bid,) for bid in existing_ids],
            )
        cur.execute("DELETE FROM syntenic_blocks WHERE block_type='operon'")
        cur.execute("DELETE FROM clusters WHERE cluster_type='operon'")
        try:
            cur.execute(
                "DELETE FROM cluster_assignments WHERE cluster_id NOT IN (SELECT cluster_id FROM clusters)"
            )
        except Exception:
            pass
    except Exception:
        pass

    row = cur.execute("SELECT COALESCE(MAX(cluster_id), 0) FROM clusters").fetchone()
    start_cluster = int(row[0] or 0)
    row = cur.execute("SELECT COALESCE(MAX(block_id), 0) FROM syntenic_blocks").fetchone()
    start_block = int(row[0] or 0)

    cluster_map: Dict[int, int] = {}
    cluster_rows: List[Tuple] = []
    for rec in clusters_df.itertuples(index=False):
        raw = int(rec.cluster_id)
        mapped = start_cluster + raw
        cluster_map[raw] = mapped
        cluster_rows.append(
            (
                mapped,
                int(rec.size),
                None,
                None,
                None,
                None,
                None,
                'operon',
            )
        )
    if cluster_rows:
        cur.executemany(
            """
            INSERT OR REPLACE INTO clusters(
                cluster_id, size, consensus_length, consensus_score,
                diversity, representative_query, representative_target, cluster_type
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            cluster_rows,
        )

    block_lookup = blocks_df.set_index('block_id').to_dict('index')

    def _get_block_meta(block_id: int) -> Dict[str, Any]:
        meta = block_lookup.get(int(block_id))
        if not meta:
            raise KeyError(block_id)
        return meta

    def _normalize_indices(value: Any) -> List[int]:
        if isinstance(value, list):
            return [int(v) for v in value]
        if isinstance(value, np.ndarray):
            return [int(v) for v in value.tolist()]
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return []
            try:
                parsed = ast.literal_eval(value)
            except Exception:
                parsed = []
            return [int(v) for v in parsed]
        return []

    syntenic_rows: List[Tuple] = []
    gb_rows: List[Tuple[str, int, str, float]] = []

    def _genes_for(indices: List[int]) -> List[str]:
        if not indices:
            return []
        return [str(db_gene_ids[i]) for i in indices if 0 <= i < len(db_gene_ids)]

    for idx, rec in enumerate(filtered_pairs.itertuples(index=False)):
        q_meta = _get_block_meta(int(rec.block_id))
        t_meta = _get_block_meta(int(rec.neighbor_id))
        new_block_id = start_block + idx + 1

        cluster_id = int(q_meta.get('cluster_id', 0) or 0)
        mapped_cluster = cluster_map.get(cluster_id, 0)

        q_start = int(q_meta['start_bp'])
        q_end = int(q_meta['end_bp'])
        t_start = int(t_meta['start_bp'])
        t_end = int(t_meta['end_bp'])

        q_genome = str(q_meta['sample_id'])
        q_contig = str(q_meta['contig_id'])
        t_genome = str(t_meta['sample_id'])
        t_contig = str(t_meta['contig_id'])

        length = max(abs(q_end - q_start), abs(t_end - t_start))
        similarity = float(getattr(rec, 'similarity', 0.0) or 0.0)
        score = 1.0 - float(getattr(rec, 'transport_cost', 0.0) or 0.0)

        syntenic_rows.append(
            (
                new_block_id,
                mapped_cluster,
                f"{q_genome}:{q_contig}:{q_start}-{q_end}",
                f"{t_genome}:{t_contig}:{t_start}-{t_end}",
                q_genome,
                t_genome,
                q_contig,
                t_contig,
                int(length),
                similarity,
                score,
                q_meta.get('gene_count', 0),
                t_meta.get('gene_count', 0),
                None,
                None,
                None,
                None,
                None,
                None,
                'operon',
            )
        )

        q_indices = _normalize_indices(getattr(rec, 'query_gene_indices', []))
        t_indices = _normalize_indices(getattr(rec, 'target_gene_indices', []))
        q_genes = _genes_for(q_indices)
        t_genes = _genes_for(t_indices)

        def _append_mapping(genes: List[str], role: str) -> None:
            if not genes:
                return
            n = len(genes)
            denom = max(1, n - 1)
            for pos, gid in enumerate(genes):
                rel = pos / denom if denom > 0 else 0.0
                gb_rows.append((gid, new_block_id, role, float(rel)))

        _append_mapping(q_genes, 'query')
        _append_mapping(t_genes, 'target')

    if syntenic_rows:
        cur.executemany(
            """
            INSERT OR REPLACE INTO syntenic_blocks(
                block_id, cluster_id, query_locus, target_locus,
                query_genome_id, target_genome_id,
                query_contig_id, target_contig_id,
                length, identity, score,
                n_query_windows, n_target_windows,
                query_window_start, query_window_end,
                target_window_start, target_window_end,
                query_windows_json, target_windows_json,
                block_type
            ) VALUES (
                ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?
            )
            """,
            syntenic_rows,
        )

    if gb_rows:
        for start in range(0, len(gb_rows), 1000):
            cur.executemany(
                "INSERT OR REPLACE INTO gene_block_mappings(gene_id, block_id, block_role, relative_position) VALUES (?, ?, ?, ?)",
                gb_rows[start : start + 1000],
            )

    conn.commit()
    return len(cluster_rows), len(syntenic_rows), len(gb_rows)


def run_operon_pipeline(
    genes_parquet: Path,
    output_dir: Path,
    *,
    console=None,
    pca_dims: int = 96,
    eps: float = 1e-5,
    shingle_k: int = 3,
    shingle_stride: int = 1,
    hnsw_m: int = 32,
    hnsw_ef_construction: int = 200,
    hnsw_ef_search: int = 128,
    hnsw_top_k: int = 16,
    neighbors_per_block: int = 10,
    sinkhorn_epsilon: float = 0.05,
    sinkhorn_iters: int = 40,
    sinkhorn_topk: int = 8,
    similarity_tau: float = 0.55,
    min_genome_support: int = 2,
    merge_max_gap_bp: int = 50,
    merge_support_ratio: float = 0.8,
    db_path: Optional[Path] = None,
) -> OperonSummary:
    output_dir = output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    _log(console, f"[dim]Loading genes from {genes_parquet}[/dim]")
    df = pd.read_parquet(genes_parquet)
    df = df.sort_values(["sample_id", "contig_id", "start", "end"]).reset_index(drop=True)

    # Map pipeline gene IDs to browser DB gene IDs (contig name + local ordinal)
    def _to_db_gene_id(row: pd.Series) -> str:
        contig = str(row["contig_id"])
        gid = str(row["gene_id"])
        try:
            suffix = gid.rsplit("_", 1)[-1]
        except Exception:
            suffix = gid
        return f"{contig}_{suffix}"

    df["_db_gene_id"] = df.apply(_to_db_gene_id, axis=1)

    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    matrix = df[emb_cols].to_numpy(dtype=np.float64)
    _log(
        console,
        f"[dim]Fitting preprocessor on {matrix.shape[0]} genes ({matrix.shape[1]} dims)\n[/dim]",
    )
    target_dims = min(pca_dims, matrix.shape[1])
    if matrix.shape[0] > 1:
        target_dims = min(target_dims, matrix.shape[0] - 1)
    target_dims = max(2, target_dims)
    artifacts = fit_preprocessor(matrix, dims_out=target_dims, eps=eps)
    transformed = transform_preprocessor(matrix, artifacts)

    preproc_path = output_dir / "preprocessor.joblib"
    from joblib import dump as joblib_dump

    joblib_dump(save_preprocessor(artifacts), preproc_path)

    df["_global_index"] = np.arange(len(df), dtype=np.int64)
    gene_ids = df["gene_id"].astype(str).to_numpy()
    db_gene_ids = df["_db_gene_id"].astype(str).to_numpy()
    gene_samples = df["sample_id"].astype(str).to_numpy()
    gene_contigs = df["contig_id"].astype(str).to_numpy()
    gene_start_bp = df["start"].to_numpy(dtype=np.int64)
    gene_end_bp = df["end"].to_numpy(dtype=np.int64)

    contig_embeddings: List[np.ndarray] = []
    offsets: Dict[Tuple[str, str], int] = {}
    offset = 0
    for (sample, contig), group in df.groupby(["sample_id", "contig_id"], sort=False):
        idx = group["_global_index"].to_numpy()
        contig_embeddings.append(transformed[idx])
        offsets[(str(sample), str(contig))] = offset
        offset += len(group)

    shingle_result = build_shingles(contig_embeddings, k=shingle_k, stride=shingle_stride)
    vectors = shingle_result.vectors.astype(np.float32, copy=False)
    block_gene_indices = shingle_result.gene_indices

    blocks_meta: List[Dict[str, Any]] = []
    for block_id, (start_idx, end_idx) in enumerate(block_gene_indices):
        sample = gene_samples[start_idx]
        contig = gene_contigs[start_idx]
        offset = offsets.get((sample, contig), 0)
        blocks_meta.append(
            {
                "block_id": block_id,
                "sample_id": sample,
                "contig_id": contig,
                "start_gene": int(start_idx - offset),
                "end_gene": int(end_idx - offset),
                "start_bp": int(gene_start_bp[start_idx]),
                "end_bp": int(gene_end_bp[end_idx]),
                "global_start": int(start_idx),
                "global_end": int(end_idx),
            }
        )

    blocks_df = pd.DataFrame(blocks_meta)
    if not blocks_df.empty:
        blocks_df["gene_count"] = (blocks_df["end_gene"] - blocks_df["start_gene"] + 1).clip(lower=1)
    blocks_df.to_csv(output_dir / "operon_blocks.csv", index=False)

    np.savez_compressed(
        output_dir / "shingles.npz",
        vectors=vectors,
        gene_indices=np.asarray(block_gene_indices, dtype=np.int64),
        k=shingle_k,
        stride=shingle_stride,
    )

    index_built = False
    neighbors: List[Dict[str, Any]] = []
    try:
        index = build_hnsw_index(
            vectors,
            m=hnsw_m,
            ef_construction=hnsw_ef_construction,
            ef_search=hnsw_ef_search,
            space="cosine",
        )
        index_built = True
        index_path = output_dir / "hnsw_index.bin"
        meta_path = output_dir / "hnsw_index.json"
        index.save(index_path)
        save_metadata(
            {
                "dim": index.dim,
                "space": index.space,
                "M": hnsw_m,
                "ef_construction": hnsw_ef_construction,
                "ef_search": hnsw_ef_search,
                "num_vectors": int(vectors.shape[0]),
            },
            meta_path,
        )

        if vectors.shape[0] > 1:
            query_k = min(hnsw_top_k, vectors.shape[0])
            if query_k > 1:
                labels, _ = index.query(vectors, k=query_k)
                seen: set[Tuple[int, int]] = set()
                per_block_limit = max(1, min(neighbors_per_block, query_k - 1))
                for block_id, row in enumerate(labels):
                    taken = 0
                    for neighbor in row:
                        neighbor_id = int(neighbor)
                        if neighbor_id == block_id:
                            continue
                        pair_key = tuple(sorted((block_id, neighbor_id)))
                        if pair_key in seen:
                            continue
                        seen.add(pair_key)

                        q_start, q_end = block_gene_indices[block_id]
                        t_start, t_end = block_gene_indices[neighbor_id]
                        q_len = max(1, q_end - q_start + 1)
                        t_len = max(1, t_end - t_start + 1)
                        tk = max(1, min(sinkhorn_topk, q_len, t_len))
                        result = sinkhorn_distance(
                            transformed[q_start : q_end + 1],
                            transformed[t_start : t_end + 1],
                            epsilon=sinkhorn_epsilon,
                            n_iter=sinkhorn_iters,
                            top_k=tk,
                        )
                        neighbors.append(
                            {
                                "block_id": int(block_id),
                                "neighbor_id": int(neighbor_id),
                                "similarity": float(result.similarity),
                                "transport_cost": float(result.transport_cost),
                                "query_gene_indices": list(range(q_start, q_end + 1)),
                                "target_gene_indices": list(range(t_start, t_end + 1)),
                            }
                        )
                        taken += 1
                        if taken >= per_block_limit:
                            break
    except ImportError as exc:
        _log(console, f"[yellow]Skipping HNSW index build: {exc}[/yellow]")
    except Exception as exc:
        _log(console, f"[yellow]HNSW build failed: {exc}[/yellow]")

    neighbors_df = pd.DataFrame(neighbors)
    if not neighbors_df.empty:
        neighbors_df.to_csv(output_dir / "operon_neighbors.csv", index=False)
    else:
        neighbors_df = pd.DataFrame(
            columns=[
                "block_id",
                "neighbor_id",
                "similarity",
                "transport_cost",
                "query_gene_indices",
                "target_gene_indices",
            ]
        )

    filtered_pairs = (
        neighbors_df[neighbors_df["similarity"] >= float(similarity_tau)].copy()
        if not neighbors_df.empty
        else pd.DataFrame(columns=neighbors_df.columns)
    )

    filtered_pairs = _apply_macro_derep(filtered_pairs, blocks_df, db_path, console)

    clusters_df = pd.DataFrame(columns=["cluster_id", "size", "genome_support"])
    if not filtered_pairs.empty:
        adjacency: Dict[int, List[int]] = {}
        for row in filtered_pairs.itertuples(index=False):
            u = int(row.block_id)
            v = int(row.neighbor_id)
            adjacency.setdefault(u, []).append(v)
            adjacency.setdefault(v, []).append(u)

        visited: Dict[int, int] = {}
        cluster_id = 1
        block_cluster_map: Dict[int, int] = {}
        for block_id in range(len(block_gene_indices)):
            if block_id in visited:
                continue
            stack = [block_id]
            nodes: List[int] = []
            while stack:
                node = stack.pop()
                if node in visited:
                    continue
                visited[node] = cluster_id
                nodes.append(node)
                for nbr in adjacency.get(node, []):
                    if nbr not in visited:
                        stack.append(nbr)
            if len(nodes) <= 1:
                cluster_id += 1
                continue
            for n in nodes:
                block_cluster_map[n] = cluster_id
            cluster_id += 1

        if block_cluster_map:
            blocks_df["cluster_id"] = blocks_df["block_id"].map(block_cluster_map).fillna(0).astype(int)
            kept_clusters: Dict[int, int] = {}
            clusters: List[Dict[str, Any]] = []
            next_cluster = 1
            for cid, group in blocks_df.groupby("cluster_id"):
                if cid == 0:
                    continue
                genome_support = group["sample_id"].nunique()
                if genome_support < int(min_genome_support):
                    continue
                kept_clusters[cid] = next_cluster
                clusters.append(
                    {
                        "cluster_id": next_cluster,
                        "size": int(len(group)),
                        "genome_support": int(genome_support),
                    }
                )
                next_cluster += 1

            if kept_clusters:
                blocks_df = blocks_df[blocks_df["cluster_id"].isin(kept_clusters.keys())].copy()
                blocks_df["cluster_id"] = blocks_df["cluster_id"].map(kept_clusters)
                clusters_df = pd.DataFrame(clusters)
                filtered_pairs = filtered_pairs[
                    filtered_pairs["block_id"].isin(blocks_df["block_id"])
                    & filtered_pairs["neighbor_id"].isin(blocks_df["block_id"])
                ].copy()
                block_to_cluster = blocks_df.set_index("block_id")["cluster_id"]
                filtered_pairs["cluster_id"] = filtered_pairs["block_id"].map(block_to_cluster)
                blocks_df, filtered_pairs, clusters_df = _merge_shifted_clusters(
                    blocks_df,
                    filtered_pairs,
                    min_genome_support,
                    console=console,
                )
                blocks_df, filtered_pairs, clusters_df = _merge_adjacent_clusters(
                    blocks_df,
                    filtered_pairs,
                    min_genome_support,
                    max_gap_bp=merge_max_gap_bp,
                    support_ratio=merge_support_ratio,
                    console=console,
                )
            else:
                blocks_df = pd.DataFrame(columns=blocks_df.columns)
                filtered_pairs = pd.DataFrame(columns=filtered_pairs.columns)

    blocks_df.to_csv(output_dir / "operon_blocks.csv", index=False)
    filtered_pairs.to_csv(output_dir / "operon_pairs.csv", index=False)
    clusters_df.to_csv(output_dir / "operon_clusters.csv", index=False)

    if db_path and clusters_df is not None and not clusters_df.empty:
        try:
            conn = sqlite3.connect(str(db_path))
            cur = conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS operon_micro_clusters (
                    cluster_id INTEGER PRIMARY KEY,
                    size INTEGER,
                    genome_support INTEGER
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS operon_micro_blocks (
                    block_id INTEGER PRIMARY KEY,
                    cluster_id INTEGER,
                    sample_id TEXT,
                    contig_id TEXT,
                    start_gene INTEGER,
                    end_gene INTEGER,
                    start_bp INTEGER,
                    end_bp INTEGER
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS operon_micro_pairs (
                    block_id INTEGER,
                    neighbor_id INTEGER,
                    similarity REAL,
                    transport_cost REAL,
                    PRIMARY KEY (block_id, neighbor_id)
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS operon_micro_gene_mappings (
                    gene_id TEXT,
                    block_id INTEGER,
                    block_role TEXT,
                    relative_position REAL,
                    PRIMARY KEY (gene_id, block_id)
                )
                """
            )

            cur.executemany(
                "INSERT OR REPLACE INTO operon_micro_clusters(cluster_id, size, genome_support) VALUES (?, ?, ?)",
                clusters_df.to_records(index=False).tolist(),
            )
            cur.executemany(
                """
                INSERT OR REPLACE INTO operon_micro_blocks(
                    block_id, cluster_id, sample_id, contig_id, start_gene, end_gene, start_bp, end_bp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                blocks_df[[
                    "block_id",
                    "cluster_id",
                    "sample_id",
                    "contig_id",
                    "start_gene",
                    "end_gene",
                    "start_bp",
                    "end_bp",
                ]].to_records(index=False).tolist(),
            )
            if not filtered_pairs.empty:
                cur.executemany(
                    "INSERT OR REPLACE INTO operon_micro_pairs(block_id, neighbor_id, similarity, transport_cost) VALUES (?, ?, ?, ?)",
                    filtered_pairs[[
                        "block_id",
                        "neighbor_id",
                        "similarity",
                        "transport_cost",
                    ]].to_records(index=False).tolist(),
                )

            mapping_rows: List[Tuple[str, int, str, float]] = []
            for block_row in blocks_df.itertuples(index=False):
                start = int(block_row.global_start)
                end = int(block_row.global_end)
                span_gene_ids = db_gene_ids[start : end + 1]
                n = len(span_gene_ids)
                L = max(1, n - 1)
                for idx, gid in enumerate(span_gene_ids):
                    rel = idx / L if L > 0 else 0.0
                    mapping_rows.append((gid, int(block_row.block_id), 'operon', float(rel)))
            if mapping_rows:
                cur.executemany(
                    "INSERT OR REPLACE INTO operon_micro_gene_mappings(gene_id, block_id, block_role, relative_position) VALUES (?, ?, ?, ?)",
                    mapping_rows,
                )

            projected_clusters, projected_blocks, projected_mappings = _project_operon_pairs_to_db(
                conn,
                filtered_pairs,
                blocks_df,
                clusters_df,
                db_gene_ids,
            )

            conn.commit()
            if console is not None:
                console.print(
                    f"[green]Projected operon clusters into browser DB:[/green] {projected_clusters} clusters, {projected_blocks} blocks, {projected_mappings} gene mappings"
                )
        except Exception as exc:
            _log(console, f"[yellow]Operon DB integration skipped: {exc}[/yellow]")
        finally:
            if 'conn' in locals():
                conn.close()

    summary = OperonSummary(
        num_genes=len(df),
        num_blocks=len(block_gene_indices),
        num_pairs=len(neighbors),
        num_filtered_pairs=len(filtered_pairs),
        num_clusters=int(clusters_df["cluster_id"].nunique()) if not clusters_df.empty else 0,
        index_built=index_built,
    )

    summary_path = output_dir / "operon_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary.__dict__, handle, indent=2)

    return summary
