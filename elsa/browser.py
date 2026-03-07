"""
Lightweight genome browser DB population from adapter metadata + pipeline output.

Populates the genome browser SQLite schema (genomes, contigs, genes,
syntenic_blocks, clusters, gene_block_mappings, cluster_assignments)
directly from a genes DataFrame and blocks/clusters CSVs — no FASTA
files or PFAM annotations required.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path
from typing import Optional

import pandas as pd


# Full schema DDL — matches genome_browser/database/populate_db.py
_SCHEMA_DDL = """
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
    gene_count INTEGER,
    FOREIGN KEY (genome_id) REFERENCES genomes(genome_id)
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
    confidence_score REAL,
    FOREIGN KEY (genome_id) REFERENCES genomes(genome_id),
    FOREIGN KEY (contig_id) REFERENCES contigs(contig_id)
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
    PRIMARY KEY (block_id, cluster_id),
    FOREIGN KEY (block_id) REFERENCES syntenic_blocks(block_id),
    FOREIGN KEY (cluster_id) REFERENCES clusters(cluster_id)
);

CREATE TABLE IF NOT EXISTS gene_block_mappings (
    gene_id TEXT,
    block_id INTEGER,
    block_role TEXT,
    relative_position REAL,
    PRIMARY KEY (gene_id, block_id),
    FOREIGN KEY (gene_id) REFERENCES genes(gene_id),
    FOREIGN KEY (block_id) REFERENCES syntenic_blocks(block_id)
);

CREATE TABLE IF NOT EXISTS block_consensus (
    block_id INTEGER PRIMARY KEY,
    consensus_len INTEGER,
    consensus_json TEXT,
    FOREIGN KEY (block_id) REFERENCES syntenic_blocks(block_id)
);

CREATE TABLE IF NOT EXISTS cluster_consensus (
    cluster_id INTEGER PRIMARY KEY,
    consensus_json TEXT,
    agree_frac REAL,
    core_tokens INTEGER,
    FOREIGN KEY (cluster_id) REFERENCES clusters(cluster_id)
);

CREATE TABLE IF NOT EXISTS annotation_stats (
    genome_id TEXT PRIMARY KEY,
    total_genes INTEGER,
    annotated_genes INTEGER,
    total_pfam_domains INTEGER,
    unique_pfam_domains INTEGER,
    syntenic_genes INTEGER,
    non_syntenic_genes INTEGER,
    annotation_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (genome_id) REFERENCES genomes(genome_id)
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


def populate_browser_db(
    db_path: Path,
    genes_df: pd.DataFrame,
    blocks_csv: Path,
    clusters_csv: Path,
) -> None:
    """Populate a genome browser SQLite DB from adapter data + pipeline output.

    Args:
        db_path: Path to the SQLite database (created if missing)
        genes_df: DataFrame with sample_id, contig_id, gene_id, start, end, strand
        blocks_csv: Path to micro_chain_blocks.csv
        clusters_csv: Path to micro_chain_clusters.csv
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=OFF")
    conn.executescript(_SCHEMA_DDL)

    try:
        _ingest_genes(conn, genes_df)
        _ingest_blocks(conn, blocks_csv)
        _ingest_clusters(conn, clusters_csv)
        _create_gene_block_mappings(conn)
        conn.execute("PRAGMA foreign_keys=ON")
        conn.commit()
    finally:
        conn.close()

    print(
        f"[Browser] Populated {db_path}: "
        f"{len(genes_df)} genes, "
        f"{genes_df['sample_id'].nunique()} genomes",
        file=sys.stderr, flush=True,
    )


def _ingest_genes(conn: sqlite3.Connection, genes_df: pd.DataFrame) -> None:
    """Populate genomes, contigs, and genes tables from adapter DataFrame."""
    cursor = conn.cursor()

    # Clear existing data
    for table in ("genes", "contigs", "genomes"):
        cursor.execute(f"DELETE FROM {table}")

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

    # Genes
    cursor.executemany(
        "INSERT INTO genes (gene_id, genome_id, contig_id, start_pos, end_pos, strand, "
        "gene_length, protein_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (
                row.gene_id, row.sample_id, row.contig_id,
                int(row.start), int(row.end), int(row.strand),
                int(row.end - row.start), row.gene_id,
            )
            for row in genes_df.itertuples()
        ],
    )
    conn.commit()


def _ingest_blocks(conn: sqlite3.Connection, blocks_csv: Path) -> None:
    """Populate syntenic_blocks from pipeline CSV."""
    cursor = conn.cursor()
    cursor.execute("DELETE FROM cluster_assignments")
    cursor.execute("DELETE FROM gene_block_mappings")
    cursor.execute("DELETE FROM syntenic_blocks")

    df = pd.read_csv(blocks_csv)

    for _, row in df.iterrows():
        block_id = int(row["block_id"])
        cluster_id = int(row.get("cluster_id", 0))
        n_anchors = int(row["n_anchors"])
        score = float(row.get("chain_score", 0.0))
        orientation = row.get("orientation", "same")

        # Synthesize locus IDs for browser compatibility
        q_genome = row["query_genome"]
        t_genome = row["target_genome"]
        q_contig = row["query_contig"]
        t_contig = row["target_contig"]
        query_locus = f"{q_genome}:{q_contig}:{row['query_start']}-{row['query_end']}"
        target_locus = f"{t_genome}:{t_contig}:{row['target_start']}-{row['target_end']}"

        # bp-range metadata for gene_block_mappings
        bp_meta = {}
        if "query_start_bp" in row and pd.notna(row.get("query_start_bp")):
            bp_meta = {
                "query_start_bp": int(row["query_start_bp"]),
                "query_end_bp": int(row["query_end_bp"]),
                "target_start_bp": int(row["target_start_bp"]),
                "target_end_bp": int(row["target_end_bp"]),
            }

        # Anchor gene JSON
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

        # Block type
        if n_anchors <= 3:
            block_type = "small"
        elif n_anchors <= 10:
            block_type = "medium"
        else:
            block_type = "large"

        # Identity approximation from score/n_anchors
        identity = score / n_anchors if n_anchors > 0 else 0.0

        cursor.execute(
            "INSERT INTO syntenic_blocks "
            "(block_id, cluster_id, query_locus, target_locus, "
            "query_genome_id, target_genome_id, query_contig_id, target_contig_id, "
            "length, identity, score, n_query_windows, n_target_windows, "
            "query_window_start, query_window_end, target_window_start, target_window_end, "
            "query_windows_json, target_windows_json, block_type) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                block_id, cluster_id, query_locus, target_locus,
                q_genome, t_genome, q_contig, t_contig,
                n_anchors, round(identity, 4), round(score, 4),
                n_anchors, n_anchors,
                int(row["query_start"]), int(row["query_end"]),
                int(row["target_start"]), int(row["target_end"]),
                query_windows_json, target_windows_json,
                block_type,
            ),
        )

    conn.commit()


def _ingest_clusters(conn: sqlite3.Connection, clusters_csv: Path) -> None:
    """Populate clusters and cluster_assignments from pipeline CSV."""
    cursor = conn.cursor()
    cursor.execute("DELETE FROM clusters")

    df = pd.read_csv(clusters_csv)

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

        cursor.execute(
            "INSERT INTO clusters (cluster_id, size, consensus_length, cluster_type) "
            "VALUES (?, ?, ?, ?)",
            (cluster_id, size, int(mean_len), ctype),
        )

    # Cluster assignments from blocks table
    cursor.execute(
        "INSERT INTO cluster_assignments (block_id, cluster_id) "
        "SELECT block_id, cluster_id FROM syntenic_blocks WHERE cluster_id > 0"
    )
    conn.commit()


def _create_gene_block_mappings(conn: sqlite3.Connection) -> None:
    """Map genes to blocks using bp-range overlap queries."""
    cursor = conn.cursor()
    cursor.execute("DELETE FROM gene_block_mappings")

    # Load blocks with bp metadata
    blocks = cursor.execute(
        "SELECT block_id, query_genome_id, query_contig_id, "
        "target_genome_id, target_contig_id, "
        "query_windows_json, target_windows_json"
        " FROM syntenic_blocks"
    ).fetchall()

    mappings = []
    for block_id, q_genome, q_contig, t_genome, t_contig, q_json, t_json in blocks:
        for role, genome_id, contig_id, windows_json in [
            ("query", q_genome, q_contig, q_json),
            ("target", t_genome, t_contig, t_json),
        ]:
            if not windows_json:
                continue

            try:
                meta = json.loads(windows_json)
            except (json.JSONDecodeError, TypeError):
                continue

            start_bp = meta.get("start_bp")
            end_bp = meta.get("end_bp")

            if start_bp is not None and end_bp is not None:
                # bp-range overlap query
                genes = cursor.execute(
                    "SELECT gene_id, start_pos, end_pos FROM genes "
                    "WHERE genome_id = ? AND contig_id = ? "
                    "AND end_pos >= ? AND start_pos <= ? "
                    "ORDER BY start_pos",
                    (genome_id, contig_id, int(start_bp), int(end_bp)),
                ).fetchall()
            else:
                # Fallback: no bp metadata, skip
                continue

            n = len(genes)
            for i, (gid, _s, _e) in enumerate(genes):
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
