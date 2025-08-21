#!/usr/bin/env python3
"""
List contig IDs of canonical ribosomal protein loci not in a given cluster.

Definition:
- A (block, genome) is canonical if the block has >= min_markers distinct curated
  RP markers among genes from that genome.
- For each canonical (block, genome), choose the most frequent contig_id among its genes
  in that block as the contig for the locus.
- Output contig IDs for those canonical loci whose block.cluster_id != target_cluster.

Usage:
  python tools/list_rp_contigs_not_in_cluster.py \
    --db genome_browser/genome_browser.db \
    --cluster 1 \
    --min_markers 8 \
    --out syntenic_analysis/rp_contigs_not_in_cluster1.txt
"""
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path
from collections import defaultdict, Counter

CURATED_RP = [
    "Ribosomal_S10","Ribosomal_L3","Ribosomal_L4","Ribosomal_L23","Ribosomal_L2","Ribosomal_L2_C",
    "Ribosomal_S19","Ribosomal_L22","Ribosomal_S3_C","Ribosomal_L16","Ribosomal_L29","Ribosomal_S17",
    "Ribosomal_L14","Ribosomal_L24","Ribosomal_L5","Ribosomal_L5_C","Ribosomal_S14","Ribosomal_S8",
    "Ribosomal_L6","Ribosomal_L18p","Ribosomal_S5","Ribosomal_S5_C","Ribosomal_L30","Ribosomal_L27A",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--db', type=Path, default=Path('genome_browser/genome_browser.db'))
    ap.add_argument('--cluster', type=int, default=1, help='Target RP cluster id to compare against')
    ap.add_argument('--min_markers', type=int, default=8)
    ap.add_argument('--out', type=Path, default=Path('syntenic_analysis/rp_contigs_not_in_cluster.txt'))
    args = ap.parse_args()

    conn = sqlite3.connect(str(args.db))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # Load block cluster ids
    block2cluster = {int(r[0]): int(r[1]) for r in cur.execute('SELECT block_id, cluster_id FROM syntenic_blocks')}

    # Load gene->block mappings with genome and contig and PFAMs
    rows = cur.execute(
        """
        SELECT gbm.block_id, g.genome_id, g.contig_id, COALESCE(g.pfam_domains,'') AS pfam
        FROM gene_block_mappings gbm
        JOIN genes g ON gbm.gene_id = g.gene_id
        """
    ).fetchall()
    conn.close()

    # Count markers per (block, genome) and track contigs
    marks = defaultdict(lambda: defaultdict(set))  # (block)->(genome)->set(markers)
    contigs = defaultdict(lambda: defaultdict(Counter))  # (block)->(genome)->Counter(contig_id)
    for r in rows:
        bid = int(r['block_id'])
        gid = str(r['genome_id'])
        cid = str(r['contig_id'])
        pf = str(r['pfam']).lower()
        contigs[bid][gid][cid] += 1
        for m in CURATED_RP:
            if m.lower() in pf:
                marks[bid][gid].add(m)

    # For canonical pairs, pick dominant contig and check cluster
    out_contigs = []
    for bid, per_g in marks.items():
        for gid, mset in per_g.items():
            if len(mset) >= args.min_markers:
                dom_contig = None
                if contigs[bid][gid]:
                    dom_contig = contigs[bid][gid].most_common(1)[0][0]
                cl = int(block2cluster.get(bid, 0))
                if cl != int(args.cluster):
                    out_contigs.append((gid, dom_contig, bid, cl, len(mset)))

    # Write simple list and print summary
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w') as f:
        for gid, cid, bid, cl, k in sorted(out_contigs):
            f.write(f"{gid}\t{cid}\tblock={bid}\tcluster={cl}\tmarkers={k}\n")
    print(f"Found {len(out_contigs)} canonical RP loci not in cluster {args.cluster}. Written to {args.out}.")


if __name__ == '__main__':
    main()

