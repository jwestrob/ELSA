"""
Batch pipeline orchestrator for gene-level anchor chaining.

Loads genes.parquet, runs index -> seed -> chain -> cluster,
and writes output CSVs + optional SQLite.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Optional, List
import sqlite3
import sys

import numpy as np
import pandas as pd

from ..index import build_gene_index
from ..seed import find_cross_genome_anchors, group_anchors_by_contig_pair
from ..chain import ChainedBlock, chain_anchors_lis, extract_nonoverlapping_chains, chain_groups_batched
from ..cluster import cluster_blocks_by_overlap, merge_contained_clusters


@dataclass
class ChainSummary:
    """Summary of chain pipeline results."""
    num_genes: int = 0
    num_anchors: int = 0
    num_blocks: int = 0
    num_clusters: int = 0
    num_singletons: int = 0
    genome_support_median: int = 0
    mean_block_size: float = 0.0


@dataclass
class ChainConfig:
    """Configuration for the chain pipeline (plain dataclass for CLI bridge)."""
    index_backend: str = "auto"
    faiss_nprobe: int = 32
    hnsw_k: int = 50
    hnsw_m: int = 32
    hnsw_ef_construction: int = 200
    hnsw_ef_search: int = 128
    similarity_threshold: float = 0.85
    max_gap_genes: int = 2
    min_chain_size: int = 2
    gap_penalty_scale: float = 0.0
    jaccard_tau: float = 0.3
    mutual_k: int = 5
    df_max: int = 500
    min_genome_support: int = 2


def run_chain_pipeline(
    genes_parquet: Optional[Path] = None,
    output_dir: Path = Path("syntenic_output"),
    config: Optional[ChainConfig] = None,
    db_path: Optional[Path] = None,
    genes_df: Optional[pd.DataFrame] = None,
    prebuilt_index: Optional[tuple] = None,
    embeddings: Optional[np.ndarray] = None,
) -> ChainSummary:
    """
    Run the gene-level anchor chaining pipeline.

    Args:
        genes_parquet: Path to genes.parquet with embeddings
        output_dir: Directory for output files
        config: Pipeline configuration (defaults used if None)
        db_path: Optional SQLite database path for genome browser
        genes_df: Pre-built DataFrame (alternative to genes_parquet).
                  May contain emb_* columns, or embeddings can be passed separately.
        prebuilt_index: Optional ("faiss", index) tuple from a SyntenyStore
        embeddings: Pre-extracted float32 embedding array (avoids DataFrame copy)

    Returns:
        ChainSummary with pipeline statistics
    """
    if config is None:
        config = ChainConfig()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load genes
    if genes_df is not None:
        print(f"[MicroChain] Using provided DataFrame ({len(genes_df)} genes)", file=sys.stderr, flush=True)
    elif genes_parquet is not None:
        print(f"[MicroChain] Loading genes from {genes_parquet}", file=sys.stderr, flush=True)
        genes_df = pd.read_parquet(genes_parquet)
    else:
        raise ValueError("Either genes_parquet or genes_df must be provided")

    required = {"sample_id", "contig_id", "gene_id", "start", "end"}
    if not required.issubset(set(genes_df.columns)):
        raise RuntimeError(f"genes.parquet missing required columns: {sorted(required - set(genes_df.columns))}")

    # Extract or use pre-provided embeddings
    import gc
    if embeddings is not None:
        emb_dim = embeddings.shape[1]
        emb_array = embeddings
        # Strip emb_ columns from genes_df if present (avoid carrying duplicates)
        emb_cols = [c for c in genes_df.columns if c.startswith("emb_")]
        if emb_cols:
            genes_df = genes_df.drop(columns=emb_cols)
    else:
        emb_cols = [c for c in genes_df.columns if c.startswith("emb_")]
        if not emb_cols:
            raise RuntimeError("No embeddings provided and genes_df has no emb_* columns")
        emb_dim = len(emb_cols)
        emb_array = genes_df[emb_cols].values.astype(np.float32)
        genes_df = genes_df.drop(columns=emb_cols)
        gc.collect()

    n_genes = len(genes_df)
    n_genomes = genes_df["sample_id"].nunique()
    print(f"[MicroChain] Loaded {n_genes} genes from {n_genomes} genomes (dim={emb_dim})",
          file=sys.stderr, flush=True)

    # Pre-sort genes + embeddings so _run_gene_chaining doesn't need to copy.
    # This lets us free emb_array here, avoiding a 4.3 GB duplicate during kNN.
    genes_df = genes_df.sort_values(['sample_id', 'contig_id', 'start', 'end'])
    genes_df['position_index'] = genes_df.groupby(['sample_id', 'contig_id']).cumcount()
    emb_array = emb_array[genes_df.index.values]  # reorder to match sort
    genes_df = genes_df.reset_index(drop=True)

    blocks = _run_gene_chaining(
        genes_df, emb_array,
        hnsw_k=config.hnsw_k,
        similarity_threshold=config.similarity_threshold,
        max_gap=config.max_gap_genes,
        min_chain_size=config.min_chain_size,
        gap_penalty_scale=config.gap_penalty_scale,
        hnsw_m=config.hnsw_m,
        hnsw_ef_construction=config.hnsw_ef_construction,
        hnsw_ef_search=config.hnsw_ef_search,
        index_backend=config.index_backend,
        prebuilt_index=prebuilt_index,
        faiss_nprobe=config.faiss_nprobe,
        _presorted=True,
    )

    if not blocks:
        print("[MicroChain] No blocks found", file=sys.stderr, flush=True)
        pd.DataFrame(columns=[
            "block_id", "cluster_id", "query_genome", "target_genome",
            "query_contig", "target_contig", "query_start", "query_end",
            "target_start", "target_end", "n_anchors", "chain_score", "orientation"
        ]).to_csv(output_dir / "micro_chain_blocks.csv", index=False)
        pd.DataFrame(columns=[
            "cluster_id", "size", "genome_support", "mean_chain_length", "genes_json"
        ]).to_csv(output_dir / "micro_chain_clusters.csv", index=False)
        return ChainSummary(num_genes=n_genes)

    print(f"[MicroChain] Found {len(blocks)} blocks, clustering by overlap...", file=sys.stderr, flush=True)

    block_to_cluster, clusters_df = cluster_blocks_by_overlap(
        blocks,
        jaccard_tau=config.jaccard_tau,
        mutual_k=config.mutual_k,
        min_genome_support=config.min_genome_support,
    )

    n_before = len(set(c for c in block_to_cluster.values() if c > 0))

    # Save pre-merge cluster assignments for sub-cluster reconstruction
    premerge_cluster = dict(block_to_cluster)

    # Merge clusters whose genomic footprints are contained within larger clusters
    block_to_cluster, raw_merge_map = merge_contained_clusters(block_to_cluster, blocks)

    n_real_clusters = len(set(c for c in block_to_cluster.values() if c > 0))
    n_singletons = sum(1 for c in block_to_cluster.values() if c == 0)
    n_merged = n_before - n_real_clusters

    print(f"[MicroChain] Formed {n_before} clusters, merged {n_merged} contained → {n_real_clusters} final ({n_singletons} singletons)",
          file=sys.stderr, flush=True)

    # Build output
    # Lookup table: gene_id → (start_bp, end_bp) for bp range output
    gene_bp = {}
    for _, row in genes_df[["gene_id", "start", "end"]].iterrows():
        gene_bp[row["gene_id"]] = (int(row["start"]), int(row["end"]))

    block_map = {b.block_id: b for b in blocks}
    block_rows = []
    for block in blocks:
        # Compute bp ranges from anchor genes
        q_ids = block.query_gene_ids()
        t_ids = block.target_gene_ids()
        q_bps = [gene_bp[g] for g in q_ids if g in gene_bp]
        t_bps = [gene_bp[g] for g in t_ids if g in gene_bp]
        block_rows.append({
            "block_id": block.block_id,
            "cluster_id": block_to_cluster.get(block.block_id, 0),
            "query_genome": block.query_genome,
            "target_genome": block.target_genome,
            "query_contig": block.query_contig,
            "target_contig": block.target_contig,
            "query_start": block.query_start,
            "query_end": block.query_end,
            "target_start": block.target_start,
            "target_end": block.target_end,
            "n_anchors": block.n_anchors,
            "chain_score": round(block.chain_score, 4),
            "orientation": block.orientation,
            "query_anchor_genes": json.dumps(q_ids),
            "target_anchor_genes": json.dumps(t_ids),
            "query_start_bp": min(s for s, e in q_bps) if q_bps else 0,
            "query_end_bp": max(e for s, e in q_bps) if q_bps else 0,
            "target_start_bp": min(s for s, e in t_bps) if t_bps else 0,
            "target_end_bp": max(e for s, e in t_bps) if t_bps else 0,
        })

    blocks_df = pd.DataFrame(block_rows)
    blocks_df["n_genes"] = blocks_df["n_anchors"]

    # Rebuild clusters_df from post-merge block_to_cluster
    cluster_rows = []
    clusters_by_id = defaultdict(list)
    for bid, cid in block_to_cluster.items():
        if cid > 0:
            clusters_by_id[cid].append(bid)

    for cid, member_bids in sorted(clusters_by_id.items()):
        genomes = set()
        total_genes = 0
        genes_by_genome = defaultdict(list)
        for bid in member_bids:
            block = block_map[bid]
            genomes.add(block.query_genome)
            genomes.add(block.target_genome)
            for idx in range(block.query_start, block.query_end + 1):
                genes_by_genome[block.query_genome].append(f"{block.query_contig}:{idx}")
            for idx in range(block.target_start, block.target_end + 1):
                genes_by_genome[block.target_genome].append(f"{block.target_contig}:{idx}")
            total_genes += block.n_anchors
        mean_chain_len = total_genes / len(member_bids) if member_bids else 0.0
        cluster_rows.append({
            "cluster_id": cid,
            "size": len(member_bids),
            "genome_support": len(genomes),
            "mean_chain_length": round(mean_chain_len, 2),
            "genes_json": json.dumps({g: list(set(ids)) for g, ids in genes_by_genome.items()}),
        })

    clusters_df = pd.DataFrame(cluster_rows) if cluster_rows else pd.DataFrame(
        columns=["cluster_id", "size", "genome_support", "mean_chain_length", "genes_json"]
    )

    blocks_path = output_dir / "micro_chain_blocks.csv"
    clusters_path = output_dir / "micro_chain_clusters.csv"

    blocks_df.to_csv(blocks_path, index=False)
    clusters_df.to_csv(clusters_path, index=False)

    # Save pre-merge cluster assignments and merge hierarchy
    if raw_merge_map:
        premerge_rows = [{"block_id": bid, "premerge_cluster_id": cid}
                         for bid, cid in premerge_cluster.items() if cid > 0]
        pd.DataFrame(premerge_rows).to_csv(
            output_dir / "micro_chain_blocks_premerge.csv", index=False)

        # Build resolved merge tree: child -> final parent
        def _resolve(c):
            while c in raw_merge_map:
                c = raw_merge_map[c]
            return c
        merge_rows = []
        for child, parent in raw_merge_map.items():
            merge_rows.append({
                "child_cluster": child,
                "parent_cluster": parent,
                "merged_into": _resolve(child),
            })
        pd.DataFrame(merge_rows).to_csv(
            output_dir / "micro_chain_merge_map.csv", index=False)
        print(f"[MicroChain] Saved merge hierarchy ({len(merge_rows)} merges) to {output_dir}",
              file=sys.stderr, flush=True)

    print(f"[MicroChain] Wrote {len(blocks_df)} blocks to {blocks_path}", file=sys.stderr, flush=True)
    print(f"[MicroChain] Wrote {len(clusters_df)} clusters to {clusters_path}", file=sys.stderr, flush=True)

    n_clusters = len(clusters_df)
    genome_support_median = int(clusters_df["genome_support"].median()) if not clusters_df.empty else 0
    mean_block_size = float(blocks_df["n_anchors"].mean()) if not blocks_df.empty else 0.0

    if db_path is not None and Path(db_path).exists():
        _write_to_database(blocks_df, clusters_df, db_path)

    # Run cluster architecture schema pipeline
    try:
        from ..schema import run_schema_pipeline
        schema_dir = output_dir / "schema"
        run_schema_pipeline(blocks_df, genes_df, schema_dir)
    except Exception as e:
        print(f"[Schema] Warning: architecture schema failed: {e}",
              file=sys.stderr, flush=True)

    return ChainSummary(
        num_genes=n_genes,
        num_anchors=sum(b.n_anchors for b in blocks),
        num_blocks=len(blocks),
        num_clusters=n_clusters,
        num_singletons=n_singletons,
        genome_support_median=genome_support_median,
        mean_block_size=mean_block_size,
    )


def _run_gene_chaining(
    genes_df: pd.DataFrame,
    embeddings: np.ndarray,
    hnsw_k: int = 50,
    similarity_threshold: float = 0.85,
    max_gap: int = 2,
    min_chain_size: int = 2,
    gap_penalty_scale: float = 0.0,
    hnsw_m: int = 32,
    hnsw_ef_construction: int = 200,
    hnsw_ef_search: int = 128,
    index_backend: str = "auto",
    faiss_nprobe: int = 16,
    prebuilt_index: Optional[tuple] = None,
    _presorted: bool = False,
) -> List[ChainedBlock]:
    """Complete gene-level chaining pipeline (internal).

    Args:
        genes_df: DataFrame with metadata columns (no emb_ columns needed)
        embeddings: Pre-extracted float32 embedding array, aligned to genes_df rows
        _presorted: If True, genes_df is already sorted with position_index assigned
                    and embeddings are aligned. Skips the sort+copy to save ~4.3 GB.
    """
    if _presorted:
        # Caller already sorted and assigned position_index — use as-is
        pass
    else:
        # Sort and assign position_index (creates a copy of embeddings)
        genes_df = genes_df.copy()
        genes_df = genes_df.sort_values(['sample_id', 'contig_id', 'start', 'end'])
        genes_df['position_index'] = genes_df.groupby(['sample_id', 'contig_id']).cumcount()
        embeddings = embeddings[genes_df.index.values]

    info_cols = ['gene_id', 'sample_id', 'contig_id', 'position_index']
    if 'strand' in genes_df.columns:
        info_cols.append('strand')
    gene_info = genes_df[info_cols].reset_index(drop=True)

    n_genes = len(genes_df)

    if prebuilt_index is not None:
        print(f"[GeneChain] Using pre-built index for {n_genes} genes...",
              file=sys.stderr, flush=True)
        index = prebuilt_index
    else:
        print(f"[GeneChain] Building index ({index_backend}) for {n_genes} genes...",
              file=sys.stderr, flush=True)
        index = build_gene_index(embeddings, m=hnsw_m,
                                 ef_construction=hnsw_ef_construction,
                                 ef_search=hnsw_ef_search,
                                 index_backend=index_backend,
                                 faiss_nprobe=faiss_nprobe)

    print(f"[GeneChain] Finding cross-genome anchors (k={hnsw_k}, threshold={similarity_threshold})...",
          file=sys.stderr, flush=True)
    anchors = find_cross_genome_anchors(index, embeddings, gene_info,
                                        k=hnsw_k,
                                        similarity_threshold=similarity_threshold)

    # Free large arrays no longer needed — critical for large datasets
    del embeddings, gene_info, index
    import gc; gc.collect()

    print(f"[GeneChain] Found {len(anchors)} cross-genome anchors",
          file=sys.stderr, flush=True)

    groups = group_anchors_by_contig_pair(anchors)
    del anchors; gc.collect()
    print(f"[GeneChain] {len(groups)} contig pairs to process",
          file=sys.stderr, flush=True)

    all_blocks = []
    block_id = 0
    n_chains = 0

    # Batched chaining: preprocess all groups, run a single Numba kernel
    chain_results = chain_groups_batched(
        groups,
        max_gap=max_gap,
        min_size=min_chain_size,
        gap_penalty_scale=gap_penalty_scale,
    )

    for key, chains in chain_results.items():
        n_chains += len(chains)
        blocks = extract_nonoverlapping_chains(chains, block_id_start=block_id)
        all_blocks.extend(blocks)
        block_id += len(blocks)

    print(f"[GeneChain] Extracted {len(all_blocks)} non-overlapping blocks from {n_chains} chains",
          file=sys.stderr, flush=True)

    return all_blocks


def _write_to_database(
    blocks_df: pd.DataFrame,
    clusters_df: pd.DataFrame,
    db_path: Path,
) -> None:
    """Write chain results to SQLite database."""
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.cursor()

        cur.execute("""
            CREATE TABLE IF NOT EXISTS micro_chain_blocks (
                block_id INTEGER PRIMARY KEY,
                cluster_id INTEGER,
                query_genome TEXT,
                target_genome TEXT,
                query_contig TEXT,
                target_contig TEXT,
                query_start INTEGER,
                query_end INTEGER,
                target_start INTEGER,
                target_end INTEGER,
                n_anchors INTEGER,
                chain_score REAL,
                orientation INTEGER
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS micro_chain_clusters (
                cluster_id INTEGER PRIMARY KEY,
                size INTEGER,
                genome_support INTEGER,
                mean_chain_length REAL,
                genes_json TEXT
            )
        """)

        cur.execute("DELETE FROM micro_chain_blocks")
        cur.execute("DELETE FROM micro_chain_clusters")

        if not blocks_df.empty:
            blocks_df[[
                "block_id", "cluster_id", "query_genome", "target_genome",
                "query_contig", "target_contig", "query_start", "query_end",
                "target_start", "target_end", "n_anchors", "chain_score", "orientation"
            ]].to_sql("micro_chain_blocks", conn, if_exists="append", index=False)

        if not clusters_df.empty:
            clusters_df.to_sql("micro_chain_clusters", conn, if_exists="append", index=False)

        conn.commit()
        print(f"[MicroChain] Wrote results to database: {db_path}", file=sys.stderr, flush=True)

    finally:
        conn.close()
