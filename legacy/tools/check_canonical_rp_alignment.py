#!/usr/bin/env python3
"""
Detect canonical RP blocks aligned to non-canonical loci.

For each block in syntenic_blocks:
- Count curated RP markers on the query side vs target side (by genome_id)
- Mark a side as canonical if it has >= min_markers distinct curated markers
- Report blocks where exactly one side is canonical (potential misaligned matches)

Outputs a CSV with per-block details and prints a summary.
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


def parse_side_ids(locus: str) -> tuple[str | None, str | None]:
    """Return (genome_id, contig_id) from a locus string if possible.

    Expected format: "<genome_id>:<contig_id>"; otherwise returns (None, locus).
    """
    if not locus:
        return None, None
    if ':' in locus:
        a, b = locus.split(':', 1)
        return a, b
    return None, locus


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--db', type=Path, default=Path('genome_browser/genome_browser.db'))
    ap.add_argument('--min_markers', type=int, default=8)
    ap.add_argument('--out', type=Path, default=Path('syntenic_analysis/canonical_rp_alignment_report.csv'))
    args = ap.parse_args()

    conn = sqlite3.connect(str(args.db))
    conn.row_factory = sqlite3.Row
    # Fetch blocks with loci + cluster
    blocks = pd.read_sql_query(
        "SELECT block_id, cluster_id, query_locus, target_locus FROM syntenic_blocks",
        conn,
    )

    # Fetch gene mappings with PFAMs
    rows = conn.execute(
        """
        SELECT gbm.block_id, g.gene_id, g.genome_id, g.contig_id, COALESCE(g.pfam_domains, '') AS pfam_domains
        FROM gene_block_mappings gbm
        JOIN genes g ON gbm.gene_id = g.gene_id
        """
    ).fetchall()
    conn.close()

    # Build per-block per-genome marker sets
    from collections import defaultdict
    block_genome_markers: dict[int, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    block_genome_counts: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in rows:
        bid = int(r['block_id'])
        gid = str(r['genome_id']) if r['genome_id'] is not None else ''
        pf = str(r['pfam_domains']).lower()
        block_genome_counts[bid][gid] += 1
        for m in CURATED_RP:
            if m.lower() in pf:
                block_genome_markers[bid][gid].add(m)

    # Analyze per block
    recs = []
    for row in blocks.itertuples(index=False):
        bid = int(getattr(row, 'block_id'))
        cl = int(getattr(row, 'cluster_id') or 0)
        ql = str(getattr(row, 'query_locus') or '')
        tl = str(getattr(row, 'target_locus') or '')
        qg, qc = parse_side_ids(ql)
        tg, tc = parse_side_ids(tl)

        gm = block_genome_markers.get(bid, {})
        gc = block_genome_counts.get(bid, {})

        # Choose query/target genomes: prefer parsed genome_ids, else top by count
        def pick_side(side_g: str | None, other_g: str | None) -> str | None:
            if side_g and side_g in gm:
                return side_g
            # fallback: pick a genome_id present that's not the other_g, with most genes
            if gc:
                candidates = sorted(gc.items(), key=lambda x: -x[1])
                for gid, _ in candidates:
                    if gid and gid != (other_g or ''):
                        return gid
            return side_g

        qgid = pick_side(qg, tg)
        tgid = pick_side(tg, qg)

        q_marks = gm.get(qgid or '', set())
        t_marks = gm.get(tgid or '', set())
        q_cnt = len(q_marks)
        t_cnt = len(t_marks)
        q_can = q_cnt >= args.min_markers
        t_can = t_cnt >= args.min_markers
        status = 'both_canonical' if (q_can and t_can) else ('one_side_canonical' if (q_can ^ t_can) else 'non_canonical')

        recs.append({
            'block_id': bid,
            'cluster_id': cl,
            'query_locus': ql,
            'target_locus': tl,
            'query_genome_id': qgid,
            'target_genome_id': tgid,
            'query_marker_count': q_cnt,
            'target_marker_count': t_cnt,
            'query_markers': ';'.join(sorted(q_marks)),
            'target_markers': ';'.join(sorted(t_marks)),
            'status': status,
        })

    df = pd.DataFrame(recs)
    df.to_csv(args.out, index=False)
    # Summary
    tot = len(df)
    mis = int((df['status'] == 'one_side_canonical').sum())
    both = int((df['status'] == 'both_canonical').sum())
    print(f"Analyzed {tot} blocks. Misaligned (one-side canonical): {mis}. Both-canonical: {both}.")
    # Show top 10 misaligned examples
    ex = df[df['status'] == 'one_side_canonical'].head(10)
    if not ex.empty:
        print("\nTop misaligned examples (block_id, cluster_id, q_cnt, t_cnt):")
        for r in ex.itertuples(index=False):
            print(f"  {r.block_id}  cid={r.cluster_id}  q={r.query_marker_count}  t={r.target_marker_count}")


if __name__ == '__main__':
    main()

