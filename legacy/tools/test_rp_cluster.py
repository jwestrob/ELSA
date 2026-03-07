#!/usr/bin/env python3
"""
Test harness to evaluate whether a 6-genome ribosomal protein (RP) cluster is recovered
under different clustering parameter profiles.

It runs tools/recluster_from_yaml.py with supplied configs, then inspects the
SQLite DB (genome_browser/genome_browser.db) to verify that at least one cluster
contains all six expected RP genomes.
"""
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple
import subprocess
import sys

REQUIRED_GENOMES = {
    '1313.30775',
    'CAYEVI000000000',
    'JALJEL000000000',
    'JBBKAE000000000',
    'JBJIAH000000000',
    'JBLTKP000000000',
}


def run_profile(config_path: Path, blocks: Path, windows: Path, db: Path, label: str) -> Tuple[bool, Dict]:
    cmd = [
        sys.executable, 'tools/recluster_from_yaml.py',
        '--config', str(config_path),
        '--blocks', str(blocks),
        '--windows', str(windows),
        '--db', str(db),
    ]
    print(f"\n=== Running profile: {label} ===\n$ {' '.join(cmd)}")
    res = subprocess.run(cmd)
    if res.returncode != 0:
        return False, {'error': f'profile {label} failed with code {res.returncode}'}

    # Inspect DB for 6-genome cluster
    conn = sqlite3.connect(str(db))
    try:
        cur = conn.cursor()
        cur.execute('SELECT block_id, cluster_id, query_locus, target_locus, length, identity FROM syntenic_blocks WHERE cluster_id > 0')
        rows = cur.fetchall()
    finally:
        conn.close()

    clusters: Dict[int, Dict] = {}
    for block_id, cluster_id, ql, tl, L, ident in rows:
        info = clusters.setdefault(cluster_id, {'genomes': set(), 'nblocks': 0, 'lens': []})
        info['nblocks'] += 1
        info['lens'].append(int(L or 0))
        qg = str(ql).split(':', 1)[0]
        tg = str(tl).split(':', 1)[0]
        info['genomes'].add(qg)
        info['genomes'].add(tg)

    passing = []
    for cid, info in clusters.items():
        if REQUIRED_GENOMES.issubset(info['genomes']):
            passing.append((cid, info))

    passing.sort(key=lambda x: (-len(x[1]['genomes']), -x[1]['nblocks']))
    ok = len(passing) > 0

    summary = {
        'label': label,
        'n_clusters': len(clusters),
        'passing_count': len(passing),
        'passing_clusters': [
            {
                'cluster_id': cid,
                'genomes': sorted(list(info['genomes'])),
                'nblocks': info['nblocks'],
                'mean_len': (sum(info['lens']) / max(1, len(info['lens']))),
            }
            for cid, info in passing
        ],
    }
    print(f"Profile '{label}': clusters={summary['n_clusters']}, passing={summary['passing_count']}")
    for pc in summary['passing_clusters'][:3]:
        print(f"  ✓ cluster {pc['cluster_id']}: genomes={pc['genomes']} blocks={pc['nblocks']} mean_len={pc['mean_len']:.2f}")
    if not ok:
        print("  ✗ No 6-genome RP cluster found")
    return ok, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--blocks', default='syntenic_analysis/syntenic_blocks.csv')
    ap.add_argument('--windows', default='elsa_index/shingles/windows.parquet')
    ap.add_argument('--db', default='genome_browser/genome_browser.db')
    ap.add_argument('--profiles', nargs='*', default=[
        'configs/elsa_cluster_control.yaml',
        'configs/elsa_cluster_hybrid.yaml',
    ])
    args = ap.parse_args()

    blocks = Path(args.blocks)
    windows = Path(args.windows)
    db = Path(args.db)

    overall_ok = True
    results: List[Dict] = []
    for p in args.profiles:
        ok, summary = run_profile(Path(p), blocks, windows, db, label=Path(p).stem)
        results.append(summary)
        overall_ok = overall_ok and ok

    # Exit non-zero if hybrid failed to find RP
    hybrid_res = next((r for r in results if r['label'] == 'elsa_cluster_hybrid'), None)
    if not hybrid_res or hybrid_res.get('passing_count', 0) == 0:
        print("\nHybrid profile failed to recover 6-genome RP cluster.")
        sys.exit(2)

    print("\nAll done. Hybrid profile recovered RP locus.")


if __name__ == '__main__':
    main()

