#!/usr/bin/env python3
"""
Experiment with clustering parameter combinations and report RP coverage, clusters, and runtimes.
Uses tools/recluster_srp.py directly to avoid writing many YAMLs.
"""
from __future__ import annotations

import argparse
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

REQUIRED_GENOMES = {
    '1313.30775',
    'CAYEVI000000000',
    'JALJEL000000000',
    'JBBKAE000000000',
    'JBJIAH000000000',
    'JBLTKP000000000',
}


def run_once(params: Dict[str, str], blocks: Path, windows: Path, db: Path) -> Tuple[float, Dict]:
    cmd = [
        sys.executable, 'tools/recluster_srp.py',
        '--blocks', str(blocks),
        '--windows', str(windows),
        '--db', str(db),
        '--tau', str(params.get('tau', 0.5)),
        '--mutual_k', str(params.get('mutual_k', 3)),
        '--df_max', str(params.get('df_max', 200)),
        '--degree_cap', str(params.get('degree_cap', 10)),
        '--shingle_k', str(params.get('shingle_k', 3)),
        '--shingle_method', str(params.get('shingle_method', 'xor')),
        '--bands_per_window', str(params.get('bands_per_window', 4)),
        '--band_stride', str(params.get('band_stride', 7)),
        '--bandset_tau', str(params.get('bandset_tau', 0.25)),
        '--bandset_df_max', str(params.get('bandset_df_max', 3000)),
        '--bandset_min_len', str(params.get('bandset_min_len', 20)),
        '--bandset_min_identity', str(params.get('bandset_min_identity', 0.98)),
        '--max_candidates_per_block', str(params.get('max_candidates_per_block', 100000)),
        '--min_shared_shingles', str(params.get('min_shared_shingles', 1)),
        '--bandset_topk_candidates', str(params.get('bandset_topk_candidates', 100000)),
        '--min_shared_band_tokens', str(params.get('min_shared_band_tokens', 1)),
    ]
    if params.get('enable_hybrid_bandset', False):
        cmd.append('--enable_hybrid_bandset')
    if params.get('enable_mutual_topk_filter', False):
        cmd.append('--enable_mutual_topk_filter')

    print('\n$ ' + ' '.join(cmd))
    t0 = time.time()
    res = subprocess.run(cmd)
    dt = time.time() - t0
    if res.returncode != 0:
        return dt, {'error': f'returncode {res.returncode}'}

    # Inspect DB
    conn = sqlite3.connect(str(db))
    try:
        cur = conn.cursor()
        cur.execute('SELECT block_id, cluster_id, query_locus, target_locus, length FROM syntenic_blocks WHERE cluster_id > 0')
        rows = cur.fetchall()
    finally:
        conn.close()

    clusters: Dict[int, Dict] = {}
    for bid, cid, ql, tl, L in rows:
        info = clusters.setdefault(cid, {'genomes': set(), 'nblocks': 0, 'lens': []})
        info['nblocks'] += 1
        info['lens'].append(int(L or 0))
        qg = str(ql).split(':', 1)[0]
        tg = str(tl).split(':', 1)[0]
        info['genomes'].add(qg)
        info['genomes'].add(tg)

    # Summaries
    entries: List[Tuple[int, int, int]] = []  # (cid, coverage, nblocks)
    for cid, info in clusters.items():
        cov = len(info['genomes'] & REQUIRED_GENOMES)
        entries.append((cid, cov, info['nblocks']))
    entries.sort(key=lambda x: (-x[1], -x[2]))

    best_cov = entries[0][1] if entries else 0
    sixers = [cid for cid, cov, _ in entries if cov == 6]
    fivers = [cid for cid, cov, _ in entries if cov == 5]
    fours = [cid for cid, cov, _ in entries if cov == 4]

    summary = {
        'runtime_sec': dt,
        'n_clusters': len(clusters),
        'best_coverage': best_cov,
        'six_clusters': sixers,
        'five_clusters': fivers[:5],
        'four_clusters': fours[:5],
    }
    return dt, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--blocks', default='syntenic_analysis/syntenic_blocks.csv')
    ap.add_argument('--windows', default='elsa_index/shingles/windows.parquet')
    ap.add_argument('--db', default='genome_browser/genome_browser.db')
    args = ap.parse_args()

    blocks = Path(args.blocks)
    windows = Path(args.windows)
    db = Path(args.db)

    # Baseline and a small grid of promising hybrids
    configs: List[Tuple[str, Dict[str, str]]] = [
        ('baseline_control', {
            'enable_hybrid_bandset': False,
            'df_max': 200,
            'tau': 0.5,
            'min_shared_shingles': 1,
            'max_candidates_per_block': 100000,
        }),
        ('hybrid_loose', {
            'enable_hybrid_bandset': True,
            'df_max': 200,
            'tau': 0.5,
            'bandset_tau': 0.20,
            'bandset_df_max': 5000,
            'min_shared_shingles': 1,
            'max_candidates_per_block': 2000,
            'bandset_topk_candidates': 1000,
            'min_shared_band_tokens': 1,
        }),
        ('hybrid_mid', {
            'enable_hybrid_bandset': True,
            'df_max': 200,
            'tau': 0.5,
            'bandset_tau': 0.25,
            'bandset_df_max': 3000,
            'min_shared_shingles': 2,
            'max_candidates_per_block': 2000,
            'bandset_topk_candidates': 500,
            'min_shared_band_tokens': 2,
        }),
        ('hybrid_mutual', {
            'enable_hybrid_bandset': True,
            'enable_mutual_topk_filter': True,
            'df_max': 200,
            'tau': 0.5,
            'bandset_tau': 0.25,
            'bandset_df_max': 3000,
            'min_shared_shingles': 3,
            'max_candidates_per_block': 1000,
            'bandset_topk_candidates': 300,
            'min_shared_band_tokens': 2,
        }),
        ('hybrid_strict', {
            'enable_hybrid_bandset': True,
            'df_max': 200,
            'tau': 0.5,
            'bandset_tau': 0.30,
            'bandset_df_max': 3000,
            'min_shared_shingles': 3,
            'max_candidates_per_block': 1000,
            'bandset_topk_candidates': 200,
            'min_shared_band_tokens': 3,
        }),
    ]

    print("Trying", len(configs), "parameter combinations…")
    for label, params in configs:
        dt, summary = run_once(params, blocks, windows, db)
        print(f"\nResult [{label}]: time={summary.get('runtime_sec', dt):.1f}s, "
              f"clusters={summary.get('n_clusters', 0)}, best_cov={summary.get('best_coverage', 0)}, "
              f"six={summary.get('six_clusters', [])}, five={summary.get('five_clusters', [])}, four={summary.get('four_clusters', [])}")

    print("\nDone.")


if __name__ == '__main__':
    main()

