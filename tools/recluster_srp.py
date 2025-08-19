#!/usr/bin/env python3
"""
Re-cluster syntenic blocks (SRP shingles, mutual-Jaccard) and persist to DB.
Uses syntenic_analysis/syntenic_blocks.csv and elsa_index/shingles/windows.parquet.
No PFAM is used. This mirrors the Cluster Tuner SRP path.
"""
from pathlib import Path
import sqlite3
import json
from types import SimpleNamespace
from typing import Dict, Any, List

import numpy as np
import pandas as pd

from elsa.analyze.cluster_mutual_jaccard import cluster_blocks_jaccard


class Block:
    def __init__(self, bid: int, qwins: List[str], twins: List[str], strand: int = 1, identity: float | None = None):
        self.id = int(bid)
        self.query_windows = list(qwins)
        self.target_windows = list(twins)
        self.strand = int(strand)
        self.identity = float(identity) if identity is not None else None
        # synthesize matches for robust gate
        self.matches = [SimpleNamespace(query_window_id=qw, target_window_id=tw) for qw, tw in zip(self.query_windows, self.target_windows)]


def _load_blocks_from_csv(csv_path: Path) -> List[Block]:
    df = pd.read_csv(csv_path)
    blocks = []
    for row in df.itertuples(index=False):
        bid = int(getattr(row, 'block_id'))
        qjson = str(getattr(row, 'query_windows_json') or '')
        tjson = str(getattr(row, 'target_windows_json') or '')
        qwins = [x for x in qjson.split(';') if x]
        twins = [x for x in tjson.split(';') if x]
        # ensure equal length by zipping
        n = min(len(qwins), len(twins))
        ident = getattr(row, 'identity', None)
        try:
            ident = float(ident) if ident is not None else None
        except Exception:
            ident = None
        blocks.append(Block(bid, qwins[:n], twins[:n], strand=1, identity=ident))
    return blocks


def _create_window_lookup(parquet_path: Path):
    df = pd.read_parquet(parquet_path)
    emb_cols = [c for c in df.columns if str(c).startswith('emb_')]
    df = df[['sample_id', 'locus_id', 'window_idx'] + emb_cols]
    df['wid'] = df['sample_id'].astype(str) + '_' + df['locus_id'].astype(str) + '_' + df['window_idx'].astype(str)
    mat = df[emb_cols].to_numpy(dtype=np.float32)
    # row index for quick lookup
    wid2i = {w: i for i, w in enumerate(df['wid'].tolist())}
    def lookup(wid: str):
        i = wid2i.get(str(wid))
        if i is None:
            return None
        return mat[i]
    return lookup


def _apply_assignments(assignments: Dict[int, int], db_path: Path):
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.cursor()
        cur.execute("DELETE FROM cluster_assignments")
        rows = [(int(bid), int(cid)) for bid, cid in assignments.items() if cid and cid > 0]
        if rows:
            cur.executemany("INSERT INTO cluster_assignments (block_id, cluster_id) VALUES (?, ?)", rows)
        # Set all to sink, then update assigned
        cur.execute("UPDATE syntenic_blocks SET cluster_id = 0")
        cur.execute(
            """
            UPDATE syntenic_blocks
            SET cluster_id = (
                SELECT ca.cluster_id FROM cluster_assignments ca WHERE ca.block_id = syntenic_blocks.block_id
            )
            WHERE block_id IN (SELECT block_id FROM cluster_assignments)
            """
        )
        conn.commit()
    finally:
        conn.close()


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--blocks', default='syntenic_analysis/syntenic_blocks.csv')
    p.add_argument('--windows', default='elsa_index/shingles/windows.parquet')
    p.add_argument('--db', default='genome_browser/genome_browser.db')
    p.add_argument('--tau', type=float, default=0.75)
    p.add_argument('--mutual_k', type=int, default=3)
    p.add_argument('--df_max', type=int, default=30)
    p.add_argument('--degree_cap', type=int, default=10)
    p.add_argument('--shingle_k', type=int, default=3)
    p.add_argument('--shingle_method', choices=['xor','subset','bandset'], default='xor')
    p.add_argument('--bands_per_window', type=int, default=4)
    p.add_argument('--band_stride', type=int, default=7)
    p.add_argument('--enable_hybrid_bandset', action='store_true')
    p.add_argument('--bandset_tau', type=float, default=0.25)
    p.add_argument('--bandset_df_max', type=int, default=2000)
    p.add_argument('--bandset_min_len', type=int, default=20)
    p.add_argument('--bandset_min_identity', type=float, default=0.98)
    p.add_argument('--enable_mutual_topk_filter', action='store_true')
    p.add_argument('--max_candidates_per_block', type=int, default=500)
    p.add_argument('--min_shared_shingles', type=int, default=2)
    p.add_argument('--bandset_topk_candidates', type=int, default=100)
    p.add_argument('--min_shared_band_tokens', type=int, default=2)
    args = p.parse_args()

    blocks_csv = Path(args.blocks)
    windows_parquet = Path(args.windows)
    db_path = Path(args.db)
    print(f"Loading blocks from {blocks_csv}")
    blocks = _load_blocks_from_csv(blocks_csv)
    print(f"Loaded {len(blocks)} blocks")

    print(f"Loading windows from {windows_parquet}")
    lookup = _create_window_lookup(windows_parquet)

    cfg = SimpleNamespace(
        jaccard_tau=args.tau,
        mutual_k=args.mutual_k,
        df_max=args.df_max,
        use_weighted_jaccard=True,
        min_low_df_anchors=3,
        idf_mean_min=1.0,
        max_df_percentile=None,
        v_mad_max_genes=0.5,
        min_anchors=4,
        min_span_genes=8,
        enable_cassette_mode=True,
        cassette_max_len=4,
        degree_cap=args.degree_cap,
        k_core_min_degree=3,
        triangle_support_min=1,
        use_community_detection=True,
        community_method='greedy',
        srp_bits=256, srp_bands=32, srp_band_bits=8, srp_seed=1337,
        shingle_k=args.shingle_k,
        shingle_method=args.shingle_method,
        bands_per_window=args.bands_per_window,
        band_stride=args.band_stride,
        keep_singletons=False, sink_label=0,
        size_ratio_min=0.5, size_ratio_max=2.0,
        enable_hybrid_bandset=args.enable_hybrid_bandset,
        bandset_tau=args.bandset_tau,
        bandset_df_max=args.bandset_df_max,
        bandset_min_len=args.bandset_min_len,
        bandset_min_identity=args.bandset_min_identity,
        enable_mutual_topk_filter=args.enable_mutual_topk_filter,
        max_candidates_per_block=args.max_candidates_per_block,
        min_shared_shingles=args.min_shared_shingles,
        bandset_topk_candidates=args.bandset_topk_candidates,
        min_shared_band_tokens=args.min_shared_band_tokens,
    )

    print("Clustering with SRP mutual-Jaccard…")
    assignments = cluster_blocks_jaccard(blocks, lookup, cfg)
    from collections import Counter
    ctr = Counter([c for c in assignments.values() if c and c > 0])
    print(f"Found {len(ctr)} clusters; top sizes: {sorted(ctr.values(), reverse=True)[:10]}")

    print(f"Writing assignments to {db_path}")
    _apply_assignments(assignments, db_path)
    print("Done.")


if __name__ == '__main__':
    main()
