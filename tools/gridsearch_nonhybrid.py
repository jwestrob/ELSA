#!/usr/bin/env python3
"""
Grid search over non-hybrid clustering parameters.

For each parameter combination, runs tools/recluster_srp.py (hybrid disabled),
then inspects the DB to compute:
 - RP coverage (how many of the 6 genomes appear together in at least one cluster)
 - number of non-sink clusters
 - cluster size distribution (mean, median, P90, P95, max)
 - runtime seconds

Prints a ranked summary (best RP coverage first, then smaller clusters).
"""
from __future__ import annotations

import argparse
import itertools
import sqlite3
import statistics as stats
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

REQUIRED_GENOMES = {
    '1313.30775',
    'CAYEVI000000000',
    'JALJEL000000000',
    'JBBKAE000000000',
    'JBJIAH000000000',
    'JBLTKP000000000',
}


def recluster(params: Dict[str, object], blocks: Path, windows: Path, db: Path) -> float:
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
        '--shingle_method', 'xor',
        '--bands_per_window', '4',
        '--band_stride', '7',
        '--max_candidates_per_block', str(params.get('max_candidates_per_block', 2000)),
        '--min_shared_shingles', str(params.get('min_shared_shingles', 2)),
        '--bandset_topk_candidates', '0',  # ensure hybrid path contributes nothing even if toggled erroneously
        '--min_shared_band_tokens', '9'
    ]
    # Explicitly ensure hybrid is OFF
    # (We do not pass --enable_hybrid_bandset)
    if params.get('enable_mutual_topk_filter', False):
        cmd.append('--enable_mutual_topk_filter')

    print('$ ' + ' '.join(cmd))
    t0 = time.time()
    res = subprocess.run(cmd)
    dt = time.time() - t0
    if res.returncode != 0:
        raise RuntimeError(f"recluster failed with code {res.returncode}")
    return dt


def analyze_db(db: Path) -> Dict:
    conn = sqlite3.connect(str(db))
    try:
        cur = conn.cursor()
        cur.execute('SELECT cluster_id, query_locus, target_locus FROM syntenic_blocks WHERE cluster_id > 0')
        rows = cur.fetchall()
        # cluster sizes
        sizes = {}
        cover = {}
        for cid, ql, tl in rows:
            sizes[cid] = sizes.get(cid, 0) + 1
            gs = cover.setdefault(cid, set())
            gs.add(str(ql).split(':', 1)[0])
            gs.add(str(tl).split(':', 1)[0])
        if sizes:
            size_list = list(sizes.values())
            size_list.sort()
            mean_sz = sum(size_list) / len(size_list)
            med_sz = stats.median(size_list)
            def pct(p):
                k = int(round(p * (len(size_list) - 1)))
                return size_list[k]
            p90 = pct(0.90)
            p95 = pct(0.95)
            mx = size_list[-1]
        else:
            mean_sz = med_sz = p90 = p95 = mx = 0
        # coverage
        best_cov = 0
        sixers = []
        for cid, gs in cover.items():
            cov = len(gs & REQUIRED_GENOMES)
            if cov > best_cov:
                best_cov = cov
            if cov == 6:
                sixers.append(cid)
        # RP display-window consistency + ribosomal purity analysis
        def merge_intervals(ints: List[Tuple[int, int]], gap: int = 1000) -> List[Tuple[int, int]]:
            if not ints:
                return []
            ints = sorted(ints)
            m = [list(ints[0])]
            for a, b in ints[1:]:
                if a <= m[-1][1] + gap:
                    m[-1][1] = max(m[-1][1], b)
                else:
                    m.append([a, b])
            return [(a, b) for a, b in m]

        rp_best = {'cluster_id': None, 'cov': 0, 'sum_regions': 1e9, 'deviation': 1e9, 'size': 0, 'per_genome': {}, 'rib_frac': 0.0}
        # Only consider clusters that include >=4 RP genomes to save work
        candidate_cids = [cid for cid, gs in cover.items() if len(gs & REQUIRED_GENOMES) >= 4]
        # Preload PFAM domains per cluster for purity measurement
        cur.execute(
            """
            SELECT sb.cluster_id, g.pfam_domains
            FROM genes g
            JOIN gene_block_mappings gbm ON g.gene_id = gbm.gene_id
            JOIN syntenic_blocks sb ON gbm.block_id = sb.block_id
            WHERE sb.cluster_id > 0 AND g.pfam_domains IS NOT NULL AND g.pfam_domains != ''
            """
        )
        by_cluster_pf = defaultdict(list)
        for cid0, doms in cur.fetchall():
            by_cluster_pf[cid0].extend([d.strip() for d in str(doms).split(';') if d.strip()])

        import re
        rib_pat = re.compile(r"ribosomal|ribosome|^RL\d|^RS\d|Ribosomal_", re.I)

        for cid in candidate_cids:
            # Pull intervals per (genome_id, contig_id) for this cluster
            q = (
                "SELECT g.genome_id, g.contig_id, MIN(g.start_pos) AS s, MAX(g.end_pos) AS e, gbm.block_id "
                "FROM gene_block_mappings gbm "
                "JOIN genes g ON gbm.gene_id = g.gene_id "
                "JOIN syntenic_blocks sb ON gbm.block_id = sb.block_id "
                "WHERE sb.cluster_id = ? "
                "GROUP BY g.genome_id, g.contig_id, gbm.block_id"
            )
            cur.execute(q, (cid,))
            rows2 = cur.fetchall()
            per_gc: Dict[Tuple[str, str], List[Tuple[int, int]]] = {}
            for genome_id, contig_id, s, e, block_id in rows2:
                if genome_id is None or contig_id is None or s is None or e is None:
                    continue
                per_gc.setdefault((str(genome_id), str(contig_id)), []).append((int(s), int(e)))
            # Merge to display windows per genome
            per_genome_regions: Dict[str, int] = {}
            for (genome_id, contig_id), ints in per_gc.items():
                merged = merge_intervals(ints, gap=1000)
                per_genome_regions[genome_id] = per_genome_regions.get(genome_id, 0) + len(merged)
            # Evaluate RP genomes only
            rp_counts = [per_genome_regions.get(g, 0) for g in REQUIRED_GENOMES]
            cov = sum(1 for c in rp_counts if c > 0)
            if cov == 0:
                continue
            sum_regions = sum(rp_counts)
            deviation = sum(abs(c - 1) for c in rp_counts)  # prefer exactly one region per genome
            # Ribosomal purity: fraction of PFAM domains matching ribosomal pattern
            pf = by_cluster_pf.get(cid, [])
            total_pf = len(pf)
            rib_pf = sum(1 for d in pf if rib_pat.search(d))
            rib_frac = (rib_pf / total_pf) if total_pf > 0 else 0.0
            # Update best by (cov desc, deviation asc, sum_regions asc, cluster size asc)
            size = sizes.get(cid, 0)
            key = (cov, rib_frac, -deviation, -sum_regions, -size)
            best_key = (rp_best['cov'], rp_best['rib_frac'], -rp_best['deviation'], -rp_best['sum_regions'], -rp_best['size'])
            if key > best_key:
                rp_best = {
                    'cluster_id': cid,
                    'cov': cov,
                    'sum_regions': sum_regions,
                    'deviation': deviation,
                    'size': size,
                    'per_genome': {g: per_genome_regions.get(g, 0) for g in REQUIRED_GENOMES},
                    'rib_frac': rib_frac,
                }

        return {
            'n_clusters': len(sizes),
            'mean': mean_sz,
            'median': med_sz,
            'p90': p90,
            'p95': p95,
            'max': mx,
            'best_cov': best_cov,
            'six': sorted(sixers),
            'rp_best': rp_best,
        }
    finally:
        conn.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--blocks', default='syntenic_analysis/syntenic_blocks.csv')
    ap.add_argument('--windows', default='elsa_index/shingles/windows.parquet')
    ap.add_argument('--db', default='genome_browser/genome_browser.db')
    ap.add_argument('--limit', type=int, default=0, help='Limit number of combos (0 = all)')
    args = ap.parse_args()

    blocks = Path(args.blocks)
    windows = Path(args.windows)
    db = Path(args.db)

    # Grid definition (non-hybrid): keep small but informative
    grid = {
        'tau': [0.5],
        'df_max': [200, 300],
        'min_shared_shingles': [2, 3],
        'enable_mutual_topk_filter': [False, True],
        'max_candidates_per_block': [1000, 2000],
        'degree_cap': [8, 10],
    }

    keys = list(grid.keys())
    combos = [dict(zip(keys, vals)) for vals in itertools.product(*[grid[k] for k in keys])]
    if args.limit and args.limit > 0:
        combos = combos[:args.limit]

    print(f"Testing {len(combos)} non-hybrid parameter combinations…")
    results: List[Tuple[Dict, Dict, float]] = []
    for i, params in enumerate(combos, 1):
        print(f"\n=== {i}/{len(combos)} ===")
        try:
            dt = recluster(params, blocks, windows, db)
            summary = analyze_db(db)
            rp = summary.get('rp_best', {})
            print(
                "time={:.1f}s clusters={} best_cov={} six={} sizes(mean/med/p90/p95/max)={:.1f}/{:.1f}/{}/{}/{} | "
                "rp_best: cid={} cov={} deviation={} sum_regions={} rib_frac={:.2%} per_genome={}".format(
                    dt, summary['n_clusters'], summary['best_cov'], summary['six'],
                    summary['mean'], summary['median'], summary['p90'], summary['p95'], summary['max'],
                    rp.get('cluster_id'), rp.get('cov'), rp.get('deviation'), rp.get('sum_regions'), rp.get('rib_frac', 0.0), rp.get('per_genome')
                )
            )
            results.append((params, summary, dt))
        except Exception as e:
            print(f"✗ failed: {e}")

    # Rank: prioritize RP cluster with cov=6, high ribosomal purity, minimal deviation and sum_regions, then compact sizes
    def rank_key(r):
        s = r[1]
        rp = s.get('rp_best', {})
        return (
            -(rp.get('cov') or 0),
            -(rp.get('rib_frac') or 0.0),
            rp.get('deviation') or 1e9,
            rp.get('sum_regions') or 1e9,
            s['mean'],
            s['max'],
        )
    results.sort(key=rank_key)
    print("\n=== Top candidates (non-hybrid) ===")
    for params, summary, dt in results[:8]:
        rp = summary.get('rp_best', {})
        tag = 'WIN' if (rp.get('cov') == 6 and rp.get('deviation', 1) <= 6 and (rp.get('rib_frac') or 0) >= 0.5) else 'TRY'
        print(
            f"[{tag}] tau={params['tau']} df_max={params['df_max']} min_shared={params['min_shared_shingles']} mutual={params['enable_mutual_topk_filter']} "
            f"max_cand={params['max_candidates_per_block']} degree_cap={params['degree_cap']} | time={dt:.1f}s clusters={summary['n_clusters']} "
            f"sizes(mean/med/p90/p95/max)={summary['mean']:.1f}/{summary['median']:.1f}/{summary['p90']}/{summary['p95']}/{summary['max']} | "
            f"rp_best: cid={rp.get('cluster_id')} cov={rp.get('cov')} deviation={rp.get('deviation')} sum_regions={rp.get('sum_regions')} rib_frac={(rp.get('rib_frac') or 0):.2%} per_genome={rp.get('per_genome')}"
        )


if __name__ == '__main__':
    main()
