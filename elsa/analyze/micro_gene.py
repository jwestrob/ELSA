"""
Embedding-first micro-synteny clustering (2–3 gene cassettes).

Builds micro blocks per contig from per-gene embeddings, represents each
block by hashed k-gram shingles over gene-level codewords, and clusters
blocks via IDF-weighted Jaccard with mutual-top-k gating. Outputs sidecar
CSVs and optional SQLite tables without modifying macro results.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Optional, Set

import math
import sqlite3
import hashlib
from collections import defaultdict, Counter

import numpy as np
import pandas as pd


@dataclass
class MicroBlock:
    block_id: int
    cluster_id: int  # 0 until clustered
    genome_id: str
    contig_id: str
    start_index: int  # gene index within contig (0-based)
    end_index: int    # inclusive
    shingles: Set[int]


def _stable_gene_codeword(emb_vec: np.ndarray) -> bytes:
    """Compute a stable codeword bytes from a gene embedding.

    Uses float16 byte representation for determinism and compactness,
    then hashes with blake2b to an 8-byte digest (returned as bytes).
    """
    # Ensure float16 for consistent byte length; tolerate float32 inputs
    emb16 = emb_vec.astype(np.float16, copy=False)
    h = hashlib.blake2b(emb16.tobytes(order="C"), digest_size=8).digest()
    return h


def _hash_shingle(parts: List[bytes], strand_canonical: bool = True) -> int:
    """Hash a sequence of codeword bytes into a 64-bit integer.

    Optionally canonicalize by also hashing the reversed byte-order and
    taking the lexicographically smaller to be strand-insensitive.
    """
    shingle_bytes = b"".join(parts)
    if strand_canonical:
        rev = b"".join(reversed(parts))
        if rev < shingle_bytes:
            shingle_bytes = rev
    d = hashlib.blake2b(shingle_bytes, digest_size=8).digest()
    return int(np.frombuffer(d, dtype="<u8")[0])


def _build_blocks_for_contig(
    contig_genes: pd.DataFrame,
    k_values: Tuple[int, ...] = (2, 3),
    max_gap: int = 1,
) -> List[Tuple[Tuple[int, int], Set[int]]]:
    """Build micro blocks and their shingle sets for one contig.

    Returns a list of ((start_idx, end_idx), shingles) entries. Indices are
    0-based positions in the sorted contig gene list.
    """
    # Sort genes by genomic start position
    contig_genes = contig_genes.sort_values(["start", "end"], kind="mergesort").reset_index(drop=True)
    # Precompute per-gene codewords
    emb_cols = [c for c in contig_genes.columns if c.startswith("emb_")]
    gene_codes: List[bytes] = []
    for _, row in contig_genes.iterrows():
        emb = np.array([row[c] for c in emb_cols], dtype=np.float16)
        gene_codes.append(_stable_gene_codeword(emb))

    n = len(contig_genes)
    results: List[Tuple[Tuple[int, int], Set[int]]] = []

    # Build contiguous triad blocks by default; will include pair shingles too
    for i in range(0, max(0, n - 3 + 1)):
        start_i = i
        end_i = i + 2
        # Collect shingles for this locus
        parts = gene_codes[start_i:end_i + 1]
        shingles: Set[int] = set()

        # k-gram shingles (contiguous)
        for k in k_values:
            if k <= len(parts):
                for j in range(0, len(parts) - k + 1):
                    shingles.add(_hash_shingle(parts[j:j + k], strand_canonical=True))

        # Skip-1 variants within the 3-gene span if allowed
        if max_gap >= 1 and len(parts) >= 3:
            # pairs with one skip: (i, i+2)
            shingles.add(_hash_shingle([parts[0], parts[2]], strand_canonical=True))

        results.append(((start_i, end_i), shingles))

    # For very short contigs, allow 2-gene blocks
    if n >= 2 and n < 3:
        for i in range(0, n - 2 + 1):
            parts = gene_codes[i:i + 2]
            shingles = {_hash_shingle(parts, strand_canonical=True)}
            results.append(((i, i + 1), shingles))

    return results


def _idf_weighted_jaccard(a: Set[int], b: Set[int], idf: Dict[int, float]) -> float:
    if not a or not b:
        return 0.0
    inter = a & b
    if not inter:
        return 0.0
    w_inter = sum(idf[s] for s in inter)
    w_union = sum(idf[s] for s in (a | b))
    return float(w_inter / (w_union + 1e-9))


def _mutual_top_k(edges_by_u: Dict[int, List[Tuple[int, float]]], k: int) -> Set[Tuple[int, int]]:
    """Return set of undirected edges that are mutual top-k by weight."""
    topk: Dict[int, Set[int]] = {}
    for u, neigh in edges_by_u.items():
        neigh_sorted = sorted(neigh, key=lambda x: x[1], reverse=True)[:k]
        topk[u] = {v for v, _ in neigh_sorted}
    keep: Set[Tuple[int, int]] = set()
    for u, neigh in edges_by_u.items():
        for v, _w in neigh:
            if v in topk.get(u, set()) and u in topk.get(v, set()):
                a, b = (u, v) if u < v else (v, u)
                keep.add((a, b))
    return keep


def run_micro_clustering(
    genes_parquet_path: Path,
    output_dir: Path,
    db_path: Optional[Path] = None,
    *,
    k_values: Tuple[int, ...] = (2, 3),
    max_gap: int = 1,
    jaccard_tau: float = 0.7,
    mutual_k: int = 3,
    df_max: int = 50,
    min_genome_support: int = 3,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Execute micro-gene clustering pipeline.

    Returns (blocks_df, clusters_df). Also writes sidecar CSVs and optional DB tables.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load genes parquet
    df = pd.read_parquet(genes_parquet_path)
    required = {"sample_id", "contig_id", "gene_id", "start"}
    if not required.issubset(set(df.columns)):
        raise RuntimeError(f"genes.parquet missing required columns: {sorted(required - set(df.columns))}")

    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    if not emb_cols:
        raise RuntimeError("genes.parquet contains no embedding columns (emb_*)")

    # Build micro blocks for all contigs
    raw_blocks: List[MicroBlock] = []
    block_id = 0
    for (genome_id, contig_id), sub in df.groupby(["sample_id", "contig_id"], sort=False):
        contig_blocks = _build_blocks_for_contig(sub, k_values=k_values, max_gap=max_gap)
        for (start_idx, end_idx), shingles in contig_blocks:
            mb = MicroBlock(
                block_id=block_id,
                cluster_id=0,
                genome_id=str(genome_id),
                contig_id=str(contig_id),
                start_index=int(start_idx),
                end_index=int(end_idx),
                shingles=set(shingles),
            )
            if mb.shingles:
                raw_blocks.append(mb)
            block_id += 1

    # Optional pre-cluster dereplication against macro spans (DB required)
    blocks: List[MicroBlock] = raw_blocks
    if db_path is not None and Path(db_path).exists() and blocks:
        blocks = _precluster_deduplicate_blocks(blocks, Path(db_path))

    if not blocks:
        # Write empty CSVs and return
        blocks_df = pd.DataFrame(columns=["block_id", "cluster_id", "genome_id", "contig_id", "start_index", "end_index"])
        clusters_df = pd.DataFrame(columns=["cluster_id", "size", "genomes"])
        blocks_df.to_csv(output_dir / "micro_gene_blocks.csv", index=False)
        clusters_df.to_csv(output_dir / "micro_gene_clusters.csv", index=False)
        return blocks_df, clusters_df

    # Compute DF over remaining blocks, drop high-DF shingles and build postings
    df_counts: Counter = Counter()
    for mb in blocks:
        for s in mb.shingles:
            df_counts[s] += 1

    kept_blocks: List[MicroBlock] = []
    for mb in blocks:
        keep = {s for s in mb.shingles if df_counts[s] <= int(df_max)}
        if not keep:
            continue
        mb.shingles = keep
        kept_blocks.append(mb)

    blocks = kept_blocks
    if not blocks:
        # No blocks survive DF filter
        blocks_df = pd.DataFrame(columns=["block_id", "cluster_id", "genome_id", "contig_id", "start_index", "end_index"])
        clusters_df = pd.DataFrame(columns=["cluster_id", "size", "genomes"])
        blocks_df.to_csv(output_dir / "micro_gene_blocks.csv", index=False)
        clusters_df.to_csv(output_dir / "micro_gene_clusters.csv", index=False)
        return blocks_df, clusters_df

    # Recompute postings on filtered shingles and IDF
    postings: Dict[int, List[int]] = defaultdict(list)
    N = len(blocks)
    for mb in blocks:
        for s in mb.shingles:
            postings[s].append(mb.block_id)
    idf: Dict[int, float] = {}
    for s, dfc in df_counts.items():
        if dfc > 0:
            idf[s] = math.log1p(N / float(dfc))

    # Candidate edges via postings and weighted Jaccard
    edges_by_u: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
    block_map: Dict[int, MicroBlock] = {mb.block_id: mb for mb in blocks}

    for mb in blocks:
        candidates: Set[int] = set()
        for s in mb.shingles:
            candidates.update(postings.get(s, []))
        candidates.discard(mb.block_id)
        # Score
        for v in candidates:
            wj = _idf_weighted_jaccard(mb.shingles, block_map[v].shingles, idf)
            if wj >= float(jaccard_tau):
                edges_by_u[mb.block_id].append((v, wj))

    # Mutual-top-k gating
    mutual_edges = _mutual_top_k(edges_by_u, int(mutual_k)) if mutual_k and mutual_k > 0 else set()

    # Connected components over kept edges
    parent: Dict[int, int] = {}
    def find(x: int) -> int:
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    def union(a: int, b: int):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for u, v in mutual_edges:
        union(u, v)

    # Assign cluster IDs sequentially for components with min genome support
    comp_members: Dict[int, List[int]] = defaultdict(list)
    for mb in blocks:
        root = find(mb.block_id)
        comp_members[root].append(mb.block_id)

    clusters: Dict[int, int] = {}
    cluster_id = 1
    for root, members in comp_members.items():
        genomes = {block_map[bid].genome_id for bid in members}
        if len(genomes) >= int(min_genome_support) and len(members) >= 2:
            for bid in members:
                clusters[bid] = cluster_id
            cluster_id += 1

    # Prepare DataFrames
    out_blocks = []
    for mb in blocks:
        cid = int(clusters.get(mb.block_id, 0))
        if cid == 0:
            continue  # drop blocks not in passing clusters
        out_blocks.append({
            "block_id": int(mb.block_id),
            "cluster_id": cid,
            "genome_id": mb.genome_id,
            "contig_id": mb.contig_id,
            "start_index": int(mb.start_index),
            "end_index": int(mb.end_index),
        })

    blocks_df = pd.DataFrame(out_blocks)

    # Summarize clusters
    cluster_rows = []
    if not blocks_df.empty:
        for cid, g in blocks_df.groupby("cluster_id"):
            genomes = sorted(set(g["genome_id"].astype(str).tolist()))
            cluster_rows.append({
                "cluster_id": int(cid),
                "size": int(len(g)),
                "genomes": ";".join(genomes),
            })
    clusters_df = pd.DataFrame(cluster_rows)

    # Write CSVs
    blocks_df.to_csv(output_dir / "micro_gene_blocks.csv", index=False)
    clusters_df.to_csv(output_dir / "micro_gene_clusters.csv", index=False)

    # Optional: write DB tables
    if db_path is not None:
        try:
            _write_micro_tables_sqlite(db_path, blocks_df, clusters_df)
        except Exception:
            # Fail silently for DB writing; sidecar CSVs are authoritative
            pass

    return blocks_df, clusters_df


def _write_micro_tables_sqlite(db_path: Path, blocks_df: pd.DataFrame, clusters_df: pd.DataFrame) -> None:
    """Create and populate micro_gene_* tables in the genome browser DB."""
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS micro_gene_blocks (
                block_id INTEGER,
                cluster_id INTEGER,
                genome_id TEXT,
                contig_id TEXT,
                start_index INTEGER,
                end_index INTEGER,
                PRIMARY KEY (block_id)
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS micro_gene_clusters (
                cluster_id INTEGER PRIMARY KEY,
                size INTEGER,
                genomes TEXT
            )
            """
        )
        # Clear existing to avoid duplication on reruns
        cur.execute("DELETE FROM micro_gene_blocks")
        cur.execute("DELETE FROM micro_gene_clusters")

        if not blocks_df.empty:
            blocks_df.to_sql("micro_gene_blocks", conn, if_exists="append", index=False)
        if not clusters_df.empty:
            clusters_df.to_sql("micro_gene_clusters", conn, if_exists="append", index=False)
        conn.commit()
    finally:
        conn.close()


def _precluster_deduplicate_blocks(blocks: List[MicroBlock], db_path: Path) -> List[MicroBlock]:
    """Filter out micro blocks fully contained by macro blocks on both sides.

    Loads the provided micro blocks into a TEMP table and removes any block that
    participates in a contained pair for some macro block (one micro on query side,
    one on target side, both fully within that macro's spans). Returns filtered list.
    """
    if not blocks:
        return blocks

    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.cursor()
        # Basic table availability check
        for t in ("genes", "syntenic_blocks", "gene_block_mappings"):
            cur.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{t}'")
            if not cur.fetchone():
                return blocks

        # Temp table for micro blocks
        cur.execute("DROP TABLE IF EXISTS _micro_pre")
        cur.execute(
            """
            CREATE TEMP TABLE _micro_pre (
                block_id INTEGER,
                genome_id TEXT,
                contig_id TEXT,
                start_index INTEGER,
                end_index INTEGER
            )
            """
        )
        cur.executemany(
            "INSERT INTO _micro_pre(block_id, genome_id, contig_id, start_index, end_index) VALUES (?,?,?,?,?)",
            [(mb.block_id, mb.genome_id, mb.contig_id, mb.start_index, mb.end_index) for mb in blocks]
        )

        # Gene ordering
        cur.execute("DROP TABLE IF EXISTS _gene_order")
        cur.execute(
            """
            CREATE TEMP TABLE _gene_order AS
            SELECT 
                genome_id, contig_id, gene_id,
                start_pos, end_pos,
                ROW_NUMBER() OVER (PARTITION BY genome_id, contig_id ORDER BY start_pos, end_pos) - 1 AS idx
            FROM genes
            """
        )

        # Resolve absolute spans for micro blocks
        cur.execute("DROP TABLE IF EXISTS _micro_coords")
        cur.execute(
            """
            CREATE TEMP TABLE _micro_coords AS
            SELECT 
                m.block_id,
                m.genome_id,
                m.contig_id,
                s.start_pos AS start_pos,
                e.end_pos AS end_pos
            FROM _micro_pre m
            JOIN _gene_order s ON s.genome_id = m.genome_id AND s.contig_id = m.contig_id AND s.idx = m.start_index
            JOIN _gene_order e ON e.genome_id = m.genome_id AND e.contig_id = m.contig_id AND e.idx = m.end_index
            """
        )

        # Macro spans
        cur.execute("DROP TABLE IF EXISTS _macro_spans")
        cur.execute(
            """
            CREATE TEMP TABLE _macro_spans AS
            WITH per_side AS (
                SELECT 
                    sb.block_id,
                    gb.block_role AS side,
                    g.genome_id,
                    g.contig_id,
                    MIN(g.start_pos) AS span_start,
                    MAX(g.end_pos) AS span_end
                FROM syntenic_blocks sb
                JOIN gene_block_mappings gb ON gb.block_id = sb.block_id
                JOIN genes g ON g.gene_id = gb.gene_id
                GROUP BY sb.block_id, gb.block_role, g.genome_id, g.contig_id
            )
            SELECT * FROM per_side
            """
        )

        # Containment per side
        cur.execute("DROP TABLE IF EXISTS _q_contained")
        cur.execute(
            """
            CREATE TEMP TABLE _q_contained AS
            SELECT 
                m.block_id AS micro_block_id,
                ms.block_id AS macro_block_id
            FROM _micro_coords m
            JOIN _macro_spans ms
              ON ms.side = 'query'
             AND ms.genome_id = m.genome_id
             AND ms.contig_id = m.contig_id
             AND ms.span_start <= m.start_pos
             AND ms.span_end >= m.end_pos
            """
        )
        cur.execute("DROP TABLE IF EXISTS _t_contained")
        cur.execute(
            """
            CREATE TEMP TABLE _t_contained AS
            SELECT 
                m.block_id AS micro_block_id,
                ms.block_id AS macro_block_id
            FROM _micro_coords m
            JOIN _macro_spans ms
              ON ms.side = 'target'
             AND ms.genome_id = m.genome_id
             AND ms.contig_id = m.contig_id
             AND ms.span_start <= m.start_pos
             AND ms.span_end >= m.end_pos
            """
        )

        # Drop micro blocks that participate in both-side containment for some macro block
        cur.execute("DROP TABLE IF EXISTS _macro_pair_ids")
        cur.execute(
            """
            CREATE TEMP TABLE _macro_pair_ids AS
            SELECT DISTINCT q.macro_block_id
            FROM _q_contained q
            JOIN _t_contained t ON q.macro_block_id = t.macro_block_id
            """
        )

        cur.execute(
            """
            SELECT micro_block_id FROM _q_contained WHERE macro_block_id IN (SELECT macro_block_id FROM _macro_pair_ids)
            UNION
            SELECT micro_block_id FROM _t_contained WHERE macro_block_id IN (SELECT macro_block_id FROM _macro_pair_ids)
            """
        )
        drop_ids = {int(r[0]) for r in cur.fetchall()}

    finally:
        conn.close()

    if not drop_ids:
        return blocks

    kept = [mb for mb in blocks if mb.block_id not in drop_ids]
    return kept

def deduplicate_micro_against_macro(db_path: Path) -> Dict[str, int]:
    """Remove micro blocks fully contained within macro spans on both sides.

    Strategy:
      - Compute per-micro-block genomic span in absolute coordinates using gene order
        within each contig.
      - Compute per-macro-block spans for query and target sides via gene_block_mappings.
      - For any macro block M, find micro blocks contained within its query span (Q)
        and within its target span (T). If a micro pair (q in Q, t in T) shares the
        same micro cluster_id, mark both for deletion (per-side full containment).
      - Delete marked micro blocks and refresh micro_gene_clusters sizes; drop empties.

    Returns a summary dict with counts of deleted blocks and affected clusters.
    """
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.cursor()

        # Ensure required tables exist
        for t in ("genes", "syntenic_blocks", "gene_block_mappings", "micro_gene_blocks", "micro_gene_clusters"):
            cur.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{t}'")
            if not cur.fetchone():
                return {"deleted_blocks": 0, "affected_clusters": 0}

        # Build ordered genes with row numbers per genome+contig
        cur.execute("DROP TABLE IF EXISTS _gene_order")
        cur.execute(
            """
            CREATE TEMP TABLE _gene_order AS
            SELECT 
                genome_id, contig_id, gene_id,
                start_pos, end_pos,
                ROW_NUMBER() OVER (PARTITION BY genome_id, contig_id ORDER BY start_pos, end_pos) - 1 AS idx
            FROM genes
            """
        )

        # Resolve micro block absolute spans
        cur.execute("DROP TABLE IF EXISTS _micro_block_coords")
        cur.execute(
            """
            CREATE TEMP TABLE _micro_block_coords AS
            SELECT 
                m.block_id, m.cluster_id, m.genome_id, m.contig_id,
                s.start_pos AS start_pos,
                e.end_pos AS end_pos
            FROM micro_gene_blocks m
            JOIN _gene_order s ON s.genome_id = m.genome_id AND s.contig_id = m.contig_id AND s.idx = m.start_index
            JOIN _gene_order e ON e.genome_id = m.genome_id AND e.contig_id = m.contig_id AND e.idx = m.end_index
            """
        )

        # Compute macro per-side spans
        cur.execute("DROP TABLE IF EXISTS _macro_spans")
        cur.execute(
            """
            CREATE TEMP TABLE _macro_spans AS
            WITH per_side AS (
                SELECT 
                    sb.block_id,
                    gb.block_role AS side,
                    g.genome_id,
                    g.contig_id,
                    MIN(g.start_pos) AS span_start,
                    MAX(g.end_pos) AS span_end
                FROM syntenic_blocks sb
                JOIN gene_block_mappings gb ON gb.block_id = sb.block_id
                JOIN genes g ON g.gene_id = gb.gene_id
                GROUP BY sb.block_id, gb.block_role, g.genome_id, g.contig_id
            )
            SELECT * FROM per_side
            """
        )

        # Micro contained within query side
        cur.execute("DROP TABLE IF EXISTS _q_contained")
        cur.execute(
            """
            CREATE TEMP TABLE _q_contained AS
            SELECT 
                m.block_id AS micro_block_id,
                m.cluster_id AS micro_cluster_id,
                ms.block_id AS macro_block_id
            FROM _micro_block_coords m
            JOIN _macro_spans ms
              ON ms.side = 'query'
             AND ms.genome_id = m.genome_id
             AND ms.contig_id = m.contig_id
             AND ms.span_start <= m.start_pos
             AND ms.span_end >= m.end_pos
            """
        )

        # Micro contained within target side
        cur.execute("DROP TABLE IF EXISTS _t_contained")
        cur.execute(
            """
            CREATE TEMP TABLE _t_contained AS
            SELECT 
                m.block_id AS micro_block_id,
                m.cluster_id AS micro_cluster_id,
                ms.block_id AS macro_block_id
            FROM _micro_block_coords m
            JOIN _macro_spans ms
              ON ms.side = 'target'
             AND ms.genome_id = m.genome_id
             AND ms.contig_id = m.contig_id
             AND ms.span_start <= m.start_pos
             AND ms.span_end >= m.end_pos
            """
        )

        # Find micro pairs across the same macro block with the same micro cluster_id
        cur.execute("DROP TABLE IF EXISTS _to_delete")
        cur.execute(
            """
            CREATE TEMP TABLE _to_delete AS
            SELECT DISTINCT q.micro_block_id AS block_id
            FROM _q_contained q
            JOIN _t_contained t
              ON q.macro_block_id = t.macro_block_id
             AND q.micro_cluster_id = t.micro_cluster_id
            UNION
            SELECT DISTINCT t.micro_block_id AS block_id
            FROM _q_contained q
            JOIN _t_contained t
              ON q.macro_block_id = t.macro_block_id
             AND q.micro_cluster_id = t.micro_cluster_id
            """
        )

        # Count and delete
        cur.execute("SELECT COUNT(*) FROM _to_delete")
        n_delete = int(cur.fetchone()[0] or 0)
        if n_delete > 0:
            # Track affected clusters
            cur.execute(
                """
                CREATE TEMP TABLE _affected_clusters AS
                SELECT DISTINCT m.cluster_id AS cluster_id
                FROM micro_gene_blocks m
                JOIN _to_delete d ON d.block_id = m.block_id
                """
            )

            cur.execute("DELETE FROM micro_gene_blocks WHERE block_id IN (SELECT block_id FROM _to_delete)")

            # Recompute cluster sizes and drop empties
            cur.execute("DELETE FROM micro_gene_clusters WHERE cluster_id IN (SELECT cluster_id FROM _affected_clusters)")
            cur.execute(
                """
                INSERT INTO micro_gene_clusters(cluster_id, size, genomes)
                SELECT 
                    m.cluster_id,
                    COUNT(*) AS size,
                    GROUP_CONCAT(DISTINCT m.genome_id) AS genomes
                FROM micro_gene_blocks m
                WHERE m.cluster_id IN (SELECT cluster_id FROM _affected_clusters)
                GROUP BY m.cluster_id
                HAVING COUNT(*) > 0
                """
            )
            # Count affected clusters remaining
            cur.execute("SELECT COUNT(*) FROM _affected_clusters")
            affected = int(cur.fetchone()[0] or 0)
        else:
            affected = 0

        conn.commit()
        return {"deleted_blocks": n_delete, "affected_clusters": affected}
    finally:
        conn.close()
