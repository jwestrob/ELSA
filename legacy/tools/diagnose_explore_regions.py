#!/usr/bin/env python3
"""
Diagnose Explore Cluster region→block linkage for micro clusters.

For a given display cluster_id, this script:
- Resolves raw micro_id via MAX(clusters.cluster_id)
- Loads micro pairs from DB and (optionally) sidecar CSVs
- Builds display regions exactly like the app (per-side intervals merged by contig)
- Loads the block list used on the page (from DB syntenic_blocks where block_type='micro')
- Computes, per region, the intersection count between region.blocks and the block list
- Prints detailed stats and mismatches

Usage:
  python tools/diagnose_explore_regions.py --db genome_browser/genome_browser.db --display-id 77
"""

import argparse
import sqlite3
from pathlib import Path
from typing import List, Dict, Tuple
import pandas as pd


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--db', type=Path, default=Path('genome_browser/genome_browser.db'))
    ap.add_argument('--display-id', type=int, required=True)
    ap.add_argument('--sidecar-dir', type=Path, default=Path('syntenic_analysis/micro_gene'))
    ap.add_argument('--gap-bp', type=int, default=1000)
    return ap.parse_args()


def macro_ceiling(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT COALESCE(MAX(cluster_id),0) FROM clusters WHERE cluster_id > 0").fetchone()
    return int(row[0] or 0)


def merge_intervals(intervals: List[Dict], gap_bp: int = 1000) -> List[Dict]:
    if not intervals:
        return []
    ints = sorted(intervals, key=lambda x: (x['start_bp'], x['end_bp']))
    merged = []
    cur = {'start_bp': ints[0]['start_bp'], 'end_bp': ints[0]['end_bp'], 'blocks': {ints[0]['block_id']}}
    for iv in ints[1:]:
        if iv['start_bp'] <= cur['end_bp'] + gap_bp:
            cur['end_bp'] = max(cur['end_bp'], iv['end_bp'])
            cur['blocks'].add(iv['block_id'])
        else:
            merged.append(cur)
            cur = {'start_bp': iv['start_bp'], 'end_bp': iv['end_bp'], 'blocks': {iv['block_id']}}
    merged.append(cur)
    return merged


def main():
    args = parse_args()
    conn = sqlite3.connect(str(args.db))
    try:
        ceil = macro_ceiling(conn)
        raw_id = int(args.display_id) - ceil
        print(f"Display_id={args.display_id} → raw_micro_id={raw_id} (macro_ceiling={ceil})")

        # Load pairs from DB; if none, try sidecar
        q = (
            "SELECT block_id, query_genome_id, query_contig_id, query_start_bp, query_end_bp, "
            "target_genome_id, target_contig_id, target_start_bp, target_end_bp "
            "FROM micro_block_pairs WHERE cluster_id = ?"
        )
        pairs = pd.read_sql_query(q, conn, params=[raw_id])
        src = 'db'
        if pairs is None or pairs.empty:
            sc = args.sidecar_dir / 'micro_block_pairs.csv'
            if sc.exists():
                df = pd.read_csv(sc)
                pairs = df[df['cluster_id'] == raw_id].copy()
                src = 'sidecar'
        if pairs is None or pairs.empty:
            print("No pairs found in DB or sidecar; cannot diagnose regions.")
            return
        print(f"Pairs loaded from {src}: {len(pairs)} rows")

        # Build regions by side
        regions = []
        for role in ('query','target'):
            gcol = f"{role}_genome_id"; ccol=f"{role}_contig_id"; s=f"{role}_start_bp"; e=f"{role}_end_bp"
            side = pairs[['block_id',gcol,ccol,s,e]].dropna()
            side = side.rename(columns={gcol:'genome_id', ccol:'contig_id', s:'start_bp', e:'end_bp'})
            side['start_bp'] = side['start_bp'].astype(int)
            side['end_bp'] = side['end_bp'].astype(int)
            for (gid,cid), group in side.groupby(['genome_id','contig_id']):
                intervals = [
                    {'start_bp': int(r.start_bp), 'end_bp': int(r.end_bp), 'block_id': int(r.block_id)}
                    for _, r in group.iterrows()
                ]
                merged = merge_intervals(intervals, gap_bp=args.gap_bp)
                for iv in merged:
                    regions.append({'genome_id': gid, 'contig_id': cid, 'start_bp': iv['start_bp'], 'end_bp': iv['end_bp'], 'blocks': iv['blocks']})

        print(f"Built {len(regions)} merged regions from pairs")

        # Load cluster block list (micro) from DB syntenic_blocks
        bl = pd.read_sql_query(
            "SELECT block_id, query_locus, target_locus FROM syntenic_blocks WHERE cluster_id = ? AND block_type='micro'",
            conn,
            params=[int(args.display_id)]
        )
        print(f"Block list (DB syntenic_blocks, block_type='micro'): {len(bl)} rows")

        # If empty, try sidecar projection estimate by reading pairs directly
        if bl.empty:
            print("DB block list empty for this display cluster; consider projecting pairs into syntenic_blocks or using sidecar for the UI.")

        # Compute intersections per region
        bl_ids = set(int(b) for b in bl['block_id'].astype('int64').tolist()) if not bl.empty else set()
        for idx, r in enumerate(regions, 1):
            reg_blocks = set(int(b) for b in r['blocks'])
            inter = reg_blocks & bl_ids
            print(f"Region {idx}: {r['genome_id']}:{r['contig_id']} {r['start_bp']}-{r['end_bp']} | blocks_in_region={len(reg_blocks)} intersect_with_list={len(inter)}")
            if bl_ids and not inter:
                # Show example IDs
                ex = list(reg_blocks)[:5]
                print(f"  Example region block_ids (first 5): {ex}")
        print("\nDiagnosis complete.")
    finally:
        conn.close()


if __name__ == '__main__':
    main()

