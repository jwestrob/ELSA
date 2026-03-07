#!/usr/bin/env python3
"""
Evaluate clustering compactness and curated RP cluster purity.

Metrics:
- Total clusters (non-sink), size distribution
- Identify clusters that contain curated RP blocks (≥8 of curated PFAMs)
- For each RP-containing cluster: curated_count, total_size, purity, and whether total_size <= 20
- Report the main curated RP cluster (containing 1232/1263/1303 if present)
"""
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path
import pandas as pd

CURATED_RP = [
    "Ribosomal_S10","Ribosomal_L3","Ribosomal_L4","Ribosomal_L23","Ribosomal_L2","Ribosomal_L2_C",
    "Ribosomal_S19","Ribosomal_L22","Ribosomal_S3_C","Ribosomal_L16","Ribosomal_L29","Ribosomal_S17",
    "Ribosomal_L14","Ribosomal_L24","Ribosomal_L5","Ribosomal_L5_C","Ribosomal_S14","Ribosomal_S8",
    "Ribosomal_L6","Ribosomal_L18p","Ribosomal_S5","Ribosomal_S5_C","Ribosomal_L30","Ribosomal_L27A",
]


def load_blocks(blocks_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(blocks_csv)
    return df


def curated_rp_blocks(db_path: Path, min_markers: int = 8) -> set[int]:
    conn = sqlite3.connect(str(db_path))
    terms = [f"%{m.lower()}%" for m in CURATED_RP]
    like = " OR ".join(["LOWER(g.pfam_domains) LIKE ?" for _ in CURATED_RP])
    sql = f"""
    SELECT gbm.block_id, g.pfam_domains
    FROM gene_block_mappings gbm
    JOIN genes g ON gbm.gene_id = g.gene_id
    WHERE {like}
    """
    rows = conn.execute(sql, terms).fetchall()
    conn.close()
    from collections import defaultdict
    marks = defaultdict(set)
    for bid, pf in rows:
        pfl = str(pf or '').lower()
        for m in CURATED_RP:
            if m.lower() in pfl:
                marks[int(bid)].add(m)
    return {b for b, s in marks.items() if len(s) >= min_markers}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--blocks', type=Path, default=Path('syntenic_analysis/syntenic_blocks.csv'))
    ap.add_argument('--db', type=Path, default=Path('genome_browser/genome_browser.db'))
    ap.add_argument('--purity_size_cap', type=int, default=20)
    ap.add_argument('--min_rp_markers', type=int, default=8)
    args = ap.parse_args()

    df = load_blocks(args.blocks)
    rp_ids = curated_rp_blocks(args.db, args.min_rp_markers)

    non_sink = df[df['cluster_id'] != 0]
    total_clusters = non_sink['cluster_id'].nunique()
    print(f"Non-sink clusters: {total_clusters}")

    # Size distribution
    sizes = non_sink.groupby('cluster_id').size().sort_values(ascending=False)
    print("Top 10 cluster sizes:")
    print(sizes.head(10))

    # Curated RP metrics
    cur = df[df['block_id'].isin(rp_ids)]
    byc = cur.groupby('cluster_id').size().sort_values(ascending=False)
    print("\nCurated RP clusters by curated size (top 10):")
    print(byc.head(10))

    # Purity for clusters that contain curated RP
    print("\nCurated RP cluster purity (cluster_id, curated, total, purity, small<=cap):")
    for cid, curated_count in byc.items():
        total = int(sizes.get(cid, 0)) if cid != 0 else int(df[df['cluster_id']==0].shape[0])
        purity = (curated_count / total) if total > 0 else 0.0
        small = total <= args.purity_size_cap
        print(f"  {cid:>6}  curated={curated_count:<3} total={total:<4} purity={purity:.2f} small={small}")

    # Main RP cluster containing core blocks (if present)
    core = [1232, 1263, 1303]
    core_cids = cur[cur['block_id'].isin(core)]['cluster_id']
    if not core_cids.empty:
        main_cid = core_cids.mode().iloc[0]
        curated_in_main = int(byc.get(main_cid, 0))
        total_in_main = int(sizes.get(main_cid, 0))
        purity_main = (curated_in_main / total_in_main) if total_in_main > 0 else 0.0
        print(f"\nMain curated RP cluster: {main_cid}  curated={curated_in_main}  total={total_in_main}  purity={purity_main:.2f}")
    else:
        print("\nMain curated RP cluster: not found (core blocks missing)")


if __name__ == '__main__':
    main()

