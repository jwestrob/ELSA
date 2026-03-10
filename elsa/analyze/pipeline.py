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
from ..chain import ChainedBlock, chain_anchors_lis, extract_nonoverlapping_chains, extract_nonoverlapping_chains_df, chain_groups_batched
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

    # --- Blocks checkpoint: skip kNN + chaining entirely if available ---
    blocks_ckpt = output_dir / "blocks_checkpoint.parquet"
    blocks_from_checkpoint = False
    if blocks_ckpt.exists():
        print(f"[MicroChain] Resuming from blocks checkpoint: {blocks_ckpt}",
              file=sys.stderr, flush=True)
        # Load as DataFrame directly — skip ChainedBlock materialization
        blocks_df_ckpt = pd.read_parquet(blocks_ckpt)
        blocks = blocks_df_ckpt  # DataFrame for clustering (no Python objects)
        print(f"[MicroChain] Loaded {len(blocks_df_ckpt):,} blocks from checkpoint "
              f"(delete {blocks_ckpt} to force re-chaining)",
              file=sys.stderr, flush=True)
        blocks_from_checkpoint = True
        del emb_array
        gc.collect()
    else:
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
            checkpoint_dir=output_dir,
        )

        # Free embeddings — no longer needed after chaining
        del emb_array
        gc.collect()

    n_blocks = len(blocks) if isinstance(blocks, (list, pd.DataFrame)) else 0
    if n_blocks == 0:
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

    # Save blocks checkpoint so clustering can resume without re-chaining
    blocks_ckpt = output_dir / "blocks_checkpoint.parquet"
    if not blocks_ckpt.exists():
        _save_blocks_checkpoint(blocks, blocks_ckpt)

    print(f"[MicroChain] Found {n_blocks:,} blocks, clustering by overlap...", file=sys.stderr, flush=True)

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

    # Build output — blocks is already a DataFrame from chaining or checkpoint
    blocks_df = blocks

    # Add cluster assignments
    print(f"[MicroChain] Building block output ({len(blocks_df):,} blocks)...",
          file=sys.stderr, flush=True)
    blocks_df['cluster_id'] = blocks_df['block_id'].map(block_to_cluster).fillna(0).astype(int)

    # Rename anchor gene columns for output compatibility
    if 'anchor_query_gene_ids' in blocks_df.columns:
        blocks_df = blocks_df.rename(columns={
            'anchor_query_gene_ids': 'query_anchor_genes',
            'anchor_target_gene_ids': 'target_anchor_genes',
        })

    # Drop position-index anchor IDs (not needed in CSV output)
    blocks_df = blocks_df.drop(
        columns=[c for c in ('anchor_query_ids', 'anchor_target_ids')
                 if c in blocks_df.columns])

    # Compute bp ranges via position index → genomic coordinate lookup
    pos_idx = pd.MultiIndex.from_arrays([
        genes_df['sample_id'], genes_df['contig_id'], genes_df['position_index'],
    ])
    start_bp_s = pd.Series(genes_df['start'].values.astype(int), index=pos_idx)
    end_bp_s = pd.Series(genes_df['end'].values.astype(int), index=pos_idx)

    for prefix in ('query', 'target'):
        s_keys = pd.MultiIndex.from_arrays([
            blocks_df[f'{prefix}_genome'], blocks_df[f'{prefix}_contig'],
            blocks_df[f'{prefix}_start'],
        ])
        e_keys = pd.MultiIndex.from_arrays([
            blocks_df[f'{prefix}_genome'], blocks_df[f'{prefix}_contig'],
            blocks_df[f'{prefix}_end'],
        ])
        blocks_df[f'{prefix}_start_bp'] = start_bp_s.reindex(s_keys).fillna(0).astype(int).values
        blocks_df[f'{prefix}_end_bp'] = end_bp_s.reindex(e_keys).fillna(0).astype(int).values

    del start_bp_s, end_bp_s
    blocks_df['n_genes'] = blocks_df['n_anchors']

    # Rebuild clusters_df from post-merge block_to_cluster (vectorized)
    clustered = blocks_df[blocks_df['cluster_id'] > 0]

    if clustered.empty:
        clusters_df = pd.DataFrame(
            columns=["cluster_id", "size", "genome_support", "mean_chain_length", "genes_json"]
        )
    else:
        # Numeric stats via groupby
        stats = clustered.groupby('cluster_id').agg(
            size=('block_id', 'count'),
            mean_chain_length=('n_anchors', 'mean'),
        ).reset_index()
        stats['mean_chain_length'] = stats['mean_chain_length'].round(2)

        # Genome support: unique genomes per cluster (union of query + target)
        qg = clustered[['cluster_id', 'query_genome']].rename(columns={'query_genome': 'genome'})
        tg = clustered[['cluster_id', 'target_genome']].rename(columns={'target_genome': 'genome'})
        gs = pd.concat([qg, tg]).drop_duplicates().groupby('cluster_id')['genome'].nunique()
        stats['genome_support'] = stats['cluster_id'].map(gs).fillna(0).astype(int)

        # genes_json: per-cluster, per-genome position labels via numpy array iteration
        _cids = clustered['cluster_id'].values
        _qg = clustered['query_genome'].values
        _qc = clustered['query_contig'].values
        _qs = clustered['query_start'].values.astype(int)
        _qe = clustered['query_end'].values.astype(int)
        _tg = clustered['target_genome'].values
        _tc = clustered['target_contig'].values
        _ts = clustered['target_start'].values.astype(int)
        _te = clustered['target_end'].values.astype(int)

        genes_by_cluster = defaultdict(lambda: defaultdict(set))
        for i in range(len(clustered)):
            cid = int(_cids[i])
            for idx in range(_qs[i], _qe[i] + 1):
                genes_by_cluster[cid][_qg[i]].add(f"{_qc[i]}:{idx}")
            for idx in range(_ts[i], _te[i] + 1):
                genes_by_cluster[cid][_tg[i]].add(f"{_tc[i]}:{idx}")

        stats['genes_json'] = stats['cluster_id'].map(
            lambda cid: json.dumps({g: sorted(ids) for g, ids in genes_by_cluster.get(cid, {}).items()})
        )
        clusters_df = stats

    blocks_path = output_dir / "micro_chain_blocks.csv"
    clusters_path = output_dir / "micro_chain_clusters.csv"

    # Separate meaningful clusters (size >= 2) from singletons
    if not clusters_df.empty:
        real_clusters = clusters_df[clusters_df["size"] >= 2]
        singleton_clusters = clusters_df[clusters_df["size"] < 2]
    else:
        real_clusters = clusters_df
        singleton_clusters = pd.DataFrame(
            columns=["cluster_id", "size", "genome_support", "mean_chain_length", "genes_json"]
        )

    blocks_df.to_csv(blocks_path, index=False)
    real_clusters.to_csv(clusters_path, index=False)

    if not singleton_clusters.empty:
        unclustered_path = output_dir / "micro_chain_unclustered.csv"
        singleton_clusters.to_csv(unclustered_path, index=False)
        print(f"[MicroChain] Wrote {len(singleton_clusters)} singleton clusters to {unclustered_path}",
              file=sys.stderr, flush=True)

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
    print(f"[MicroChain] Wrote {len(real_clusters)} clusters to {clusters_path}"
          f" ({len(singleton_clusters)} singletons written separately)", file=sys.stderr, flush=True)

    n_clusters = len(real_clusters)
    genome_support_median = int(real_clusters["genome_support"].median()) if not real_clusters.empty else 0
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
        num_anchors=int(blocks_df['n_anchors'].sum()),
        num_blocks=len(blocks_df),
        num_clusters=n_clusters,
        num_singletons=n_singletons,
        genome_support_median=genome_support_median,
        mean_block_size=mean_block_size,
    )


def _save_blocks_checkpoint(blocks, path: Path):
    """Save blocks (DataFrame or ChainedBlock list) to parquet checkpoint."""
    if isinstance(blocks, pd.DataFrame):
        blocks.to_parquet(path, index=False)
    else:
        import json as _json
        rows = []
        for b in blocks:
            rows.append({
                "block_id": b.block_id,
                "query_genome": b.query_genome, "target_genome": b.target_genome,
                "query_contig": b.query_contig, "target_contig": b.target_contig,
                "query_start": b.query_start, "query_end": b.query_end,
                "target_start": b.target_start, "target_end": b.target_end,
                "n_anchors": b.n_anchors, "chain_score": b.chain_score,
                "orientation": b.orientation,
                "anchor_query_ids": _json.dumps(b.anchor_query_ids),
                "anchor_target_ids": _json.dumps(b.anchor_target_ids),
                "anchor_query_gene_ids": _json.dumps(b.anchor_query_gene_ids),
                "anchor_target_gene_ids": _json.dumps(b.anchor_target_gene_ids),
            })
        pd.DataFrame(rows).to_parquet(path, index=False)
    print(f"[MicroChain] Saved blocks checkpoint ({len(blocks):,} blocks) to {path}",
          file=sys.stderr, flush=True)


def _load_blocks_checkpoint(path: Path):
    """Deserialize ChainedBlock list from parquet checkpoint."""
    import json as _json
    from ..chain import ChainedBlock
    df = pd.read_parquet(path)
    blocks = []
    for _, row in df.iterrows():
        blocks.append(ChainedBlock(
            block_id=int(row["block_id"]),
            query_genome=row["query_genome"], target_genome=row["target_genome"],
            query_contig=row["query_contig"], target_contig=row["target_contig"],
            query_start=int(row["query_start"]), query_end=int(row["query_end"]),
            target_start=int(row["target_start"]), target_end=int(row["target_end"]),
            n_anchors=int(row["n_anchors"]), chain_score=float(row["chain_score"]),
            orientation=int(row["orientation"]),
            anchor_query_ids=_json.loads(row["anchor_query_ids"]),
            anchor_target_ids=_json.loads(row["anchor_target_ids"]),
            anchor_query_gene_ids=_json.loads(row["anchor_query_gene_ids"]),
            anchor_target_gene_ids=_json.loads(row["anchor_target_gene_ids"]),
        ))
    return blocks


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
    checkpoint_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Complete gene-level chaining pipeline (internal).

    Args:
        genes_df: DataFrame with metadata columns (no emb_ columns needed)
        embeddings: Pre-extracted float32 embedding array, aligned to genes_df rows
        _presorted: If True, genes_df is already sorted with position_index assigned
                    and embeddings are aligned. Skips the sort+copy to save ~4.3 GB.
        checkpoint_dir: If set, save/load anchors.parquet checkpoint here.
                        Allows resuming after crashes without re-running kNN.

    Returns:
        DataFrame with block columns (block_id, query_genome, etc.)
    """
    import gc

    if _presorted:
        # Caller already sorted and assigned position_index — use as-is
        pass
    else:
        # Sort and assign position_index (creates a copy of embeddings)
        genes_df = genes_df.copy()
        genes_df = genes_df.sort_values(['sample_id', 'contig_id', 'start', 'end'])
        genes_df['position_index'] = genes_df.groupby(['sample_id', 'contig_id']).cumcount()
        embeddings = embeddings[genes_df.index.values]

    n_genes = len(genes_df)

    # --- Checkpoint: load anchors from parquet if available ---
    anchors_path = (Path(checkpoint_dir) / "anchors.parquet") if checkpoint_dir else None

    if anchors_path and anchors_path.exists():
        sz_gb = anchors_path.stat().st_size / (1024**3)
        print(f"[GeneChain] Loading anchors checkpoint ({sz_gb:.1f} GB): {anchors_path}",
              file=sys.stderr, flush=True)
        anchors = pd.read_parquet(anchors_path)
        print(f"[GeneChain] Loaded {len(anchors):,} anchors from checkpoint "
              f"(delete {anchors_path} to force recomputation)",
              file=sys.stderr, flush=True)
        # Free resources not needed when resuming from checkpoint
        del embeddings
        gc.collect()
    else:
        info_cols = ['gene_id', 'sample_id', 'contig_id', 'position_index']
        if 'strand' in genes_df.columns:
            info_cols.append('strand')
        gene_info = genes_df[info_cols].reset_index(drop=True)

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
        gc.collect()

        print(f"[GeneChain] Found {len(anchors)} cross-genome anchors",
              file=sys.stderr, flush=True)

        # Save checkpoint so kNN doesn't need to be re-run on crash
        if anchors_path and not anchors.empty:
            anchors_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"[GeneChain] Saving anchors checkpoint to {anchors_path}...",
                  file=sys.stderr, flush=True)
            anchors.to_parquet(anchors_path, index=False)
            sz_gb = anchors_path.stat().st_size / (1024**3)
            print(f"[GeneChain] Checkpoint saved ({sz_gb:.1f} GB)",
                  file=sys.stderr, flush=True)

    print(f"[GeneChain] Grouping {len(anchors):,} anchors by contig pair...",
          file=sys.stderr, flush=True)
    grouped = group_anchors_by_contig_pair(anchors)
    del anchors; gc.collect()
    n_contig_pairs = grouped.n_groups if hasattr(grouped, 'n_groups') else len(grouped)
    print(f"[GeneChain] {n_contig_pairs:,} contig pairs to process",
          file=sys.stderr, flush=True)

    all_chain_dfs = []
    block_id = 0
    n_chains = 0

    # Batched chaining: preprocess all groups, run a single Numba kernel
    chain_results = chain_groups_batched(
        grouped,
        max_gap=max_gap,
        min_size=min_chain_size,
        gap_penalty_scale=gap_penalty_scale,
    )

    for key, chains in chain_results.items():
        n_chains += len(chains)
        bdf = extract_nonoverlapping_chains_df(chains, block_id_start=block_id)
        if not bdf.empty:
            all_chain_dfs.append(bdf)
            block_id += len(bdf)

    if not all_chain_dfs:
        print("[GeneChain] No blocks extracted", file=sys.stderr, flush=True)
        return pd.DataFrame()

    all_blocks_df = pd.concat(all_chain_dfs, ignore_index=True)
    del all_chain_dfs

    print(f"[GeneChain] Extracted {len(all_blocks_df):,} non-overlapping blocks from {n_chains:,} chains",
          file=sys.stderr, flush=True)

    return all_blocks_df


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
