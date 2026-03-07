#!/usr/bin/env python3
"""
Sync genome_browser DB cluster assignments from syntenic_blocks.csv.

Actions:
- Load blocks CSV (default: syntenic_analysis/syntenic_blocks.csv)
- Update syntenic_blocks.cluster_id in DB to match CSV by block_id
- Rebuild clusters and cluster_assignments tables to reflect new assignments

Usage:
  python tools/update_db_from_blocks_csv.py \
    --db genome_browser/genome_browser.db \
    --blocks syntenic_analysis/syntenic_blocks.csv
"""
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--db', type=Path, default=Path('genome_browser/genome_browser.db'))
    ap.add_argument('--blocks', type=Path, default=Path('syntenic_analysis/syntenic_blocks.csv'))
    args = ap.parse_args()

    df = pd.read_csv(args.blocks, usecols=['block_id','cluster_id'])
    df['block_id'] = df['block_id'].astype(int)
    df['cluster_id'] = df['cluster_id'].astype(int)

    conn = sqlite3.connect(str(args.db))
    try:
        cur = conn.cursor()
        # Update cluster_id per block
        cur.execute('BEGIN')
        cur.execute('UPDATE syntenic_blocks SET cluster_id = 0')
        cur.executemany(
            'UPDATE syntenic_blocks SET cluster_id = ? WHERE block_id = ?',
            [(int(row.cluster_id), int(row.block_id)) for row in df.itertuples(index=False)]
        )
        conn.commit()

        # Rebuild clusters and cluster_assignments tables
        cur.execute('DELETE FROM cluster_assignments')
        cur.execute('DELETE FROM clusters')
        # Insert assignments where cluster_id > 0
        cur.execute('INSERT INTO cluster_assignments (block_id, cluster_id) SELECT block_id, cluster_id FROM syntenic_blocks WHERE cluster_id > 0')
        # Build clusters summary: size and representative loci (by highest score available)
        # Size
        cur.execute(
            """
            INSERT INTO clusters (cluster_id, size, consensus_length, consensus_score, diversity, representative_query, representative_target, cluster_type)
            SELECT cluster_id, COUNT(*) AS size, CAST(AVG(length) AS INT) AS consensus_length, AVG(score) AS consensus_score,
                   0.0 AS diversity, '' AS representative_query, '' AS representative_target, 'unknown' AS cluster_type
            FROM syntenic_blocks WHERE cluster_id > 0 GROUP BY cluster_id
            """
        )
        conn.commit()
        # Fill representative loci by picking top score per cluster if available
        cur.execute(
            """
            WITH rep AS (
              SELECT cluster_id, query_locus, target_locus,
                     ROW_NUMBER() OVER (PARTITION BY cluster_id ORDER BY score DESC) AS rn
              FROM syntenic_blocks WHERE cluster_id > 0 AND score IS NOT NULL
            )
            UPDATE clusters
            SET representative_query = (
                    SELECT query_locus FROM rep WHERE rep.cluster_id = clusters.cluster_id AND rn = 1
                ),
                representative_target = (
                    SELECT target_locus FROM rep WHERE rep.cluster_id = clusters.cluster_id AND rn = 1
                )
            WHERE EXISTS (SELECT 1 FROM rep WHERE rep.cluster_id = clusters.cluster_id)
            """
        )
        conn.commit()
        # Report
        n_clusters = cur.execute('SELECT COUNT(*) FROM clusters').fetchone()[0]
        print(f"DB updated from CSV. Clusters: {n_clusters}")
    finally:
        conn.close()


if __name__ == '__main__':
    main()

