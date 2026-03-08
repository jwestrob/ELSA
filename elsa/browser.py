"""
Lightweight genome browser DB population from adapter metadata + pipeline output.

Populates the genome browser SQLite schema (genomes, contigs, genes,
syntenic_blocks, clusters, gene_block_mappings, cluster_assignments)
directly from a genes DataFrame and blocks/clusters CSVs — no FASTA
files or PFAM annotations required.

Optimized for bulk loading: creates tables without indexes, inserts all
data, then builds indexes at the end. Handles 400k+ genes and 800k+
blocks in under a minute.
"""

from __future__ import annotations

import json
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


# Schema DDL — tables only, no indexes (built post-load)
_SCHEMA_TABLES = """
CREATE TABLE IF NOT EXISTS genomes (
    genome_id TEXT PRIMARY KEY,
    organism_name TEXT,
    total_contigs INTEGER,
    total_genes INTEGER,
    genome_size INTEGER,
    file_path_fna TEXT,
    file_path_faa TEXT,
    file_path_gff TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS contigs (
    contig_id TEXT PRIMARY KEY,
    genome_id TEXT NOT NULL,
    contig_name TEXT,
    length INTEGER,
    gc_content REAL,
    gene_count INTEGER
);

CREATE TABLE IF NOT EXISTS genes (
    gene_id TEXT PRIMARY KEY,
    genome_id TEXT NOT NULL,
    contig_id TEXT NOT NULL,
    start_pos INTEGER NOT NULL,
    end_pos INTEGER NOT NULL,
    strand INTEGER NOT NULL,
    gene_length INTEGER,
    protein_id TEXT,
    protein_sequence TEXT,
    pfam_domains TEXT,
    pfam_count INTEGER DEFAULT 0,
    primary_pfam TEXT,
    gc_content REAL,
    partial_gene BOOLEAN DEFAULT FALSE,
    confidence_score REAL
);

CREATE TABLE IF NOT EXISTS syntenic_blocks (
    block_id INTEGER PRIMARY KEY,
    cluster_id INTEGER DEFAULT 0,
    query_locus TEXT NOT NULL,
    target_locus TEXT NOT NULL,
    query_genome_id TEXT,
    target_genome_id TEXT,
    query_contig_id TEXT,
    target_contig_id TEXT,
    length INTEGER NOT NULL,
    identity REAL NOT NULL,
    score REAL NOT NULL,
    n_query_windows INTEGER,
    n_target_windows INTEGER,
    query_window_start INTEGER,
    query_window_end INTEGER,
    target_window_start INTEGER,
    target_window_end INTEGER,
    query_windows_json TEXT,
    target_windows_json TEXT,
    block_type TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS clusters (
    cluster_id INTEGER PRIMARY KEY,
    size INTEGER NOT NULL,
    consensus_length INTEGER,
    consensus_score REAL,
    diversity REAL,
    representative_query TEXT,
    representative_target TEXT,
    cluster_type TEXT
);

CREATE TABLE IF NOT EXISTS cluster_assignments (
    block_id INTEGER NOT NULL,
    cluster_id INTEGER NOT NULL,
    PRIMARY KEY (block_id, cluster_id)
);

CREATE TABLE IF NOT EXISTS gene_block_mappings (
    gene_id TEXT NOT NULL,
    block_id INTEGER NOT NULL,
    block_role TEXT,
    relative_position REAL,
    PRIMARY KEY (gene_id, block_id)
);

CREATE TABLE IF NOT EXISTS block_consensus (
    block_id INTEGER PRIMARY KEY,
    consensus_len INTEGER,
    consensus_json TEXT
);

CREATE TABLE IF NOT EXISTS cluster_consensus (
    cluster_id INTEGER PRIMARY KEY,
    consensus_json TEXT,
    agree_frac REAL,
    core_tokens INTEGER
);

CREATE TABLE IF NOT EXISTS annotation_stats (
    genome_id TEXT PRIMARY KEY,
    total_genes INTEGER,
    annotated_genes INTEGER,
    total_pfam_domains INTEGER,
    unique_pfam_domains INTEGER,
    syntenic_genes INTEGER,
    non_syntenic_genes INTEGER,
    annotation_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS pfam_domains (
    pfam_id TEXT PRIMARY KEY,
    pfam_name TEXT,
    description TEXT,
    domain_type TEXT,
    protein_count INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS schema_meta (
    key TEXT PRIMARY KEY,
    value TEXT
);
"""

# Indexes built after all data is loaded
_POST_LOAD_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_genes_genome_contig_pos
    ON genes(genome_id, contig_id, start_pos, end_pos);
CREATE INDEX IF NOT EXISTS idx_genes_contig
    ON genes(contig_id);
CREATE INDEX IF NOT EXISTS idx_contigs_genome
    ON contigs(genome_id);
CREATE INDEX IF NOT EXISTS idx_blocks_cluster
    ON syntenic_blocks(cluster_id);
CREATE INDEX IF NOT EXISTS idx_blocks_query_genome
    ON syntenic_blocks(query_genome_id);
CREATE INDEX IF NOT EXISTS idx_blocks_target_genome
    ON syntenic_blocks(target_genome_id);
CREATE INDEX IF NOT EXISTS idx_gbm_block
    ON gene_block_mappings(block_id);
CREATE INDEX IF NOT EXISTS idx_gbm_gene
    ON gene_block_mappings(gene_id);
CREATE INDEX IF NOT EXISTS idx_ca_cluster
    ON cluster_assignments(cluster_id);
"""


def populate_browser_db(
    db_path: Path,
    genes_df: pd.DataFrame,
    blocks_csv: Path,
    clusters_csv: Path,
) -> None:
    """Populate a genome browser SQLite DB from adapter data + pipeline output.

    Optimized for large datasets (400k+ genes, 800k+ blocks):
    - journal_mode=OFF during bulk load (no WAL overhead)
    - Foreign keys disabled during load
    - Indexes built after all inserts complete
    - Vectorized gene-block mapping via pandas merge (no per-block SQL queries)
    """
    t0 = time.time()
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # Remove existing DB for clean bulk load (avoids DELETE overhead)
    if db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=OFF")
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute("PRAGMA foreign_keys=OFF")
    conn.execute("PRAGMA cache_size=-256000")  # 256 MB cache
    conn.execute("PRAGMA page_size=8192")
    conn.executescript(_SCHEMA_TABLES)

    try:
        _ingest_genes(conn, genes_df)
        blocks_df = _ingest_blocks(conn, blocks_csv)
        _ingest_clusters(conn, clusters_csv)
        n_mappings = _create_gene_block_mappings(conn, genes_df, blocks_df)

        # Build indexes after all data is loaded
        print("[Browser] Building indexes...", file=sys.stderr, flush=True)
        conn.executescript(_POST_LOAD_INDEXES)
        conn.commit()

        # Switch to WAL for runtime queries
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
    finally:
        conn.close()

    elapsed = time.time() - t0
    print(
        f"[Browser] Populated {db_path}: "
        f"{len(genes_df)} genes, "
        f"{genes_df['sample_id'].nunique()} genomes, "
        f"{n_mappings} gene-block mappings "
        f"({elapsed:.1f}s)",
        file=sys.stderr, flush=True,
    )


def _ingest_genes(conn: sqlite3.Connection, genes_df: pd.DataFrame) -> None:
    """Populate genomes, contigs, and genes tables."""
    cursor = conn.cursor()

    # Genomes
    genome_stats = genes_df.groupby("sample_id").agg(
        total_contigs=("contig_id", "nunique"),
        total_genes=("gene_id", "count"),
    ).reset_index()

    cursor.executemany(
        "INSERT INTO genomes (genome_id, organism_name, total_contigs, total_genes) "
        "VALUES (?, ?, ?, ?)",
        [
            (row.sample_id, f"Genome {row.sample_id}", int(row.total_contigs), int(row.total_genes))
            for row in genome_stats.itertuples()
        ],
    )

    # Contigs
    contig_stats = genes_df.groupby(["sample_id", "contig_id"]).agg(
        gene_count=("gene_id", "count"),
        max_bp=("end", "max"),
    ).reset_index()

    cursor.executemany(
        "INSERT INTO contigs (contig_id, genome_id, contig_name, length, gene_count) "
        "VALUES (?, ?, ?, ?, ?)",
        [
            (row.contig_id, row.sample_id, row.contig_id, int(row.max_bp), int(row.gene_count))
            for row in contig_stats.itertuples()
        ],
    )

    # Genes — deduplicate by gene_id (adapter merges can produce dupes)
    unique_genes = genes_df.drop_duplicates(subset="gene_id", keep="first")
    cursor.executemany(
        "INSERT OR IGNORE INTO genes (gene_id, genome_id, contig_id, start_pos, end_pos, strand, "
        "gene_length, protein_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (
                row.gene_id, row.sample_id, row.contig_id,
                int(row.start), int(row.end), int(row.strand),
                int(row.end - row.start), row.gene_id,
            )
            for row in unique_genes.itertuples()
        ],
    )

    n_genomes = len(genome_stats)
    n_contigs = len(contig_stats)
    n_genes = len(unique_genes)
    print(
        f"[Browser] Ingested {n_genomes} genomes, {n_contigs} contigs, {n_genes} genes",
        file=sys.stderr, flush=True,
    )
    conn.commit()


def _ingest_blocks(conn: sqlite3.Connection, blocks_csv: Path) -> pd.DataFrame:
    """Populate syntenic_blocks from pipeline CSV. Returns the blocks DataFrame."""
    df = pd.read_csv(blocks_csv)
    cursor = conn.cursor()

    # Build all rows in Python, then executemany
    rows = []
    for _, row in df.iterrows():
        block_id = int(row["block_id"])
        cluster_id = int(row.get("cluster_id", 0))
        n_anchors = int(row["n_anchors"])
        score = float(row.get("chain_score", 0.0))

        q_genome = row["query_genome"]
        t_genome = row["target_genome"]
        q_contig = row["query_contig"]
        t_contig = row["target_contig"]
        query_locus = f"{q_genome}:{q_contig}:{row['query_start']}-{row['query_end']}"
        target_locus = f"{t_genome}:{t_contig}:{row['target_start']}-{row['target_end']}"

        # bp-range metadata
        bp_meta = {}
        if "query_start_bp" in row and pd.notna(row.get("query_start_bp")):
            bp_meta = {
                "query_start_bp": int(row["query_start_bp"]),
                "query_end_bp": int(row["query_end_bp"]),
                "target_start_bp": int(row["target_start_bp"]),
                "target_end_bp": int(row["target_end_bp"]),
            }

        q_genes_json = row.get("query_anchor_genes", "[]")
        t_genes_json = row.get("target_anchor_genes", "[]")

        query_windows_json = json.dumps({
            "genes": q_genes_json if isinstance(q_genes_json, list) else q_genes_json,
            **({"start_bp": bp_meta["query_start_bp"], "end_bp": bp_meta["query_end_bp"]} if bp_meta else {}),
        })
        target_windows_json = json.dumps({
            "genes": t_genes_json if isinstance(t_genes_json, list) else t_genes_json,
            **({"start_bp": bp_meta["target_start_bp"], "end_bp": bp_meta["target_end_bp"]} if bp_meta else {}),
        })

        if n_anchors <= 3:
            block_type = "small"
        elif n_anchors <= 10:
            block_type = "medium"
        else:
            block_type = "large"

        identity = score / n_anchors if n_anchors > 0 else 0.0

        rows.append((
            block_id, cluster_id, query_locus, target_locus,
            q_genome, t_genome, q_contig, t_contig,
            n_anchors, round(identity, 4), round(score, 4),
            n_anchors, n_anchors,
            int(row["query_start"]), int(row["query_end"]),
            int(row["target_start"]), int(row["target_end"]),
            query_windows_json, target_windows_json,
            block_type,
        ))

    cursor.executemany(
        "INSERT INTO syntenic_blocks "
        "(block_id, cluster_id, query_locus, target_locus, "
        "query_genome_id, target_genome_id, query_contig_id, target_contig_id, "
        "length, identity, score, n_query_windows, n_target_windows, "
        "query_window_start, query_window_end, target_window_start, target_window_end, "
        "query_windows_json, target_windows_json, block_type) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()

    print(f"[Browser] Ingested {len(rows)} blocks", file=sys.stderr, flush=True)
    return df


def _ingest_clusters(conn: sqlite3.Connection, clusters_csv: Path) -> None:
    """Populate clusters and cluster_assignments."""
    df = pd.read_csv(clusters_csv)
    cursor = conn.cursor()

    rows = []
    for _, row in df.iterrows():
        cluster_id = int(row["cluster_id"])
        size = int(row["size"])
        mean_len = float(row.get("mean_chain_length", 0))

        if mean_len <= 3:
            ctype = "small"
        elif mean_len <= 10:
            ctype = "medium"
        else:
            ctype = "large"

        rows.append((cluster_id, size, int(mean_len), ctype))

    cursor.executemany(
        "INSERT INTO clusters (cluster_id, size, consensus_length, cluster_type) "
        "VALUES (?, ?, ?, ?)",
        rows,
    )

    # Cluster assignments from blocks table
    cursor.execute(
        "INSERT INTO cluster_assignments (block_id, cluster_id) "
        "SELECT block_id, cluster_id FROM syntenic_blocks WHERE cluster_id > 0"
    )
    conn.commit()
    print(f"[Browser] Ingested {len(rows)} clusters", file=sys.stderr, flush=True)


def _create_gene_block_mappings(
    conn: sqlite3.Connection,
    genes_df: pd.DataFrame,
    blocks_df: pd.DataFrame,
) -> int:
    """Map genes to blocks using vectorized pandas merge (no per-block SQL).

    For each block, uses bp-range overlap to find genes on the query and
    target contigs. Done entirely in pandas to avoid 800k+ SQL queries.
    """
    cursor = conn.cursor()

    # Build a gene lookup: (genome_id, contig_id) -> sorted DataFrame
    gene_cols = ["gene_id", "sample_id", "contig_id", "start", "end"]
    if not all(c in genes_df.columns for c in gene_cols):
        print("[Browser] Missing gene columns, skipping gene-block mappings",
              file=sys.stderr, flush=True)
        return 0

    genes = genes_df[gene_cols].drop_duplicates(subset="gene_id", keep="first").copy()
    genes = genes.sort_values(["sample_id", "contig_id", "start"])

    # Build per-contig gene arrays for fast interval overlap
    contig_genes = {}
    for (genome_id, contig_id), grp in genes.groupby(["sample_id", "contig_id"]):
        contig_genes[(genome_id, contig_id)] = (
            grp["gene_id"].values,
            grp["start"].values.astype(np.int64),
            grp["end"].values.astype(np.int64),
        )

    # Check which bp columns are available
    has_bp = all(c in blocks_df.columns for c in
                 ["query_start_bp", "query_end_bp", "target_start_bp", "target_end_bp"])

    if not has_bp:
        print("[Browser] No bp-range columns in blocks, skipping gene-block mappings",
              file=sys.stderr, flush=True)
        return 0

    mappings = []
    for _, row in blocks_df.iterrows():
        block_id = int(row["block_id"])

        for role, genome_col, contig_col, start_bp_col, end_bp_col in [
            ("query", "query_genome", "query_contig", "query_start_bp", "query_end_bp"),
            ("target", "target_genome", "target_contig", "target_start_bp", "target_end_bp"),
        ]:
            genome_id = row[genome_col]
            contig_id = row[contig_col]
            start_bp = row.get(start_bp_col)
            end_bp = row.get(end_bp_col)

            if pd.isna(start_bp) or pd.isna(end_bp):
                continue

            start_bp = int(start_bp)
            end_bp = int(end_bp)

            key = (genome_id, contig_id)
            if key not in contig_genes:
                continue

            gene_ids, starts, ends = contig_genes[key]

            # Vectorized interval overlap: end_pos >= start_bp AND start_pos <= end_bp
            mask = (ends >= start_bp) & (starts <= end_bp)
            hit_ids = gene_ids[mask]
            n = len(hit_ids)

            for i, gid in enumerate(hit_ids):
                rel_pos = i / max(n - 1, 1)
                mappings.append((gid, block_id, role, round(rel_pos, 4)))

    if mappings:
        cursor.executemany(
            "INSERT OR IGNORE INTO gene_block_mappings "
            "(gene_id, block_id, block_role, relative_position) "
            "VALUES (?, ?, ?, ?)",
            mappings,
        )
    conn.commit()

    print(
        f"[Browser] Created {len(mappings)} gene-block mappings",
        file=sys.stderr, flush=True,
    )
    return len(mappings)
