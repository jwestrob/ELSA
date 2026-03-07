#!/usr/bin/env python3
"""
Diagnostic script for micro-synteny visibility.

Checks that:
- micro_block_pairs and micro_gene_pair_mappings exist and have rows
- Combined block listing (macro + micro) would include micro pairs
- For a few sample micro pairs, both query and target loci have genes in the DB

Usage:
  python tools/diagnose_micro.py [--db genome_browser/genome_browser.db] [--limit 10]
"""

import argparse
import sqlite3
from pathlib import Path
import sys

import pandas as pd


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--db', type=Path, default=Path('genome_browser/genome_browser.db'))
    ap.add_argument('--limit', type=int, default=10)
    return ap.parse_args()


def get_macro_ceiling(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT COALESCE(MAX(cluster_id), 0) FROM clusters WHERE cluster_id > 0").fetchone()
    return int(row[0] or 0)


def check_tables(conn: sqlite3.Connection) -> dict:
    out = {}
    def count(name: str) -> int:
        try:
            return int(conn.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0] or 0)
        except Exception:
            return -1
    out['micro_block_pairs'] = count('micro_block_pairs')
    out['micro_gene_pair_mappings'] = count('micro_gene_pair_mappings')
    out['micro_gene_blocks'] = count('micro_gene_blocks')
    out['micro_gene_clusters'] = count('micro_gene_clusters')
    out['syntenic_blocks_micro'] = count("(SELECT 1 FROM syntenic_blocks WHERE block_type='micro')")
    return out


def load_micro_pairs(conn: sqlite3.Connection) -> pd.DataFrame:
    q = """
        SELECT 
            p.block_id,
            p.cluster_id,
            (p.query_genome_id || ':' || p.query_contig_id || ':' || CAST(p.query_start_bp AS TEXT) || '-' || CAST(p.query_end_bp AS TEXT)) AS query_locus,
            (p.target_genome_id || ':' || p.target_contig_id || ':' || CAST(p.target_start_bp AS TEXT) || '-' || CAST(p.target_end_bp AS TEXT)) AS target_locus,
            p.query_genome_id,
            p.target_genome_id,
            p.query_contig_id,
            p.target_contig_id,
            (CASE WHEN (p.query_end_bp - p.query_start_bp) > (p.target_end_bp - p.target_start_bp) THEN (p.query_end_bp - p.query_start_bp) ELSE (p.target_end_bp - p.target_start_bp) END) AS length,
            p.identity,
            p.score
        FROM micro_block_pairs p
        ORDER BY p.score DESC, p.block_id
    """
    try:
        return pd.read_sql_query(q, conn)
    except Exception:
        return pd.DataFrame()


def locus_has_genes(conn: sqlite3.Connection, locus: str) -> int:
    # locus format: genome:contig:start-end
    try:
        genome, rest = locus.split(':', 1)
        contig, span = rest.rsplit(':', 1)
        a, b = span.split('-', 1)
        s, e = int(a), int(b)
    except Exception:
        return -1
    row = conn.execute(
        "SELECT COUNT(*) FROM genes WHERE contig_id = ? AND end_pos >= ? AND start_pos <= ?",
        (contig, s, e)
    ).fetchone()
    return int(row[0] or 0)


def main():
    args = parse_args()
    if not args.db.exists():
        print(f"DB not found: {args.db}")
        sys.exit(2)
    conn = sqlite3.connect(str(args.db))
    try:
        # 1) Table sanity
        counts = check_tables(conn)
        print("Table counts:")
        for k, v in counts.items():
            print(f"  {k}: {v}")
        if counts['micro_block_pairs'] <= 0:
            print("FAIL: micro_block_pairs is empty or missing")
            sys.exit(1)

        # 2) Sample pairs
        pairs = load_micro_pairs(conn)
        print(f"\nLoaded micro pairs: {len(pairs)}")
        print(pairs.head(min(args.limit, len(pairs))).to_string(index=False))

        # 3) Verify loci have genes on both sides for a few samples
        print("\nVerifying genes present on both query and target loci...")
        ok = True
        for _, r in pairs.head(args.limit).iterrows():
            qg = locus_has_genes(conn, r['query_locus'])
            tg = locus_has_genes(conn, r['target_locus'])
            print(f"  block {int(r['block_id'])}: query_genes={qg}, target_genes={tg}")
            if qg <= 0 or tg <= 0:
                ok = False
        if not ok:
            print("FAIL: Some paired loci have no genes on one side")
            sys.exit(1)

        # 4) Cluster ID mapping sanity (display vs raw)
        ceil_id = get_macro_ceiling(conn)
        raw_ids = sorted(set(int(c) for c in pairs['cluster_id'].unique()))
        display_ids = [ceil_id + rid for rid in raw_ids]
        print("\nCluster ID ceiling:", ceil_id)
        print("Raw micro cluster IDs:", raw_ids[:10])
        print("Display micro cluster IDs (Explorer):", display_ids[:10])

        print("\nOK: Micro pairs and loci look consistent.")
    finally:
        conn.close()


if __name__ == '__main__':
    main()

