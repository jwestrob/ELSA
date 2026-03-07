#!/usr/bin/env python3
"""
Compare block precision between ELSA and MCScanX against operon ground truth.

For each operon GT instance (genome pair), find the best-overlapping block from
each method and compute:
  - Block size vs GT size (anchor/GT ratio)
  - GT gene recall (fraction of operon genes covered by block)
  - Block precision (fraction of block genes that are operon genes)

Usage:
    python benchmarks/scripts/compare_block_precision.py \
        --elsa-blocks benchmarks/results/enterobacteriaceae_chain/micro_chain/micro_chain_blocks.csv \
        --mcscanx-gff benchmarks/results/mcscanx_comparison/cross_species_v2.gff \
        --mcscanx-collinearity benchmarks/results/mcscanx_comparison/cross_species_v2.collinearity \
        --ground-truth benchmarks/ground_truth/ecoli_operon_gt_v2.tsv \
        --output benchmarks/evaluation/block_precision_comparison.md
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


# ── Data structures ──────────────────────────────────────────────────

@dataclass
class GTInstance:
    operon_id: str
    genome_a: str
    genome_b: str
    contig_a: str
    contig_b: str
    start_a: int
    end_a: int
    start_b: int
    end_b: int
    n_genes_a: int
    n_genes_b: int


@dataclass
class BlockMatch:
    """A block matched to a GT instance."""
    block_start: int
    block_end: int
    gt_start: int
    gt_end: int
    n_block_genes: int
    n_gt_genes: int

    @property
    def overlap_start(self) -> int:
        return max(self.block_start, self.gt_start)

    @property
    def overlap_end(self) -> int:
        return min(self.block_end, self.gt_end)

    @property
    def overlap_genes(self) -> int:
        return max(0, self.overlap_end - self.overlap_start + 1)

    @property
    def gt_recall(self) -> float:
        """Fraction of GT genes covered by block."""
        return self.overlap_genes / self.n_gt_genes if self.n_gt_genes > 0 else 0.0

    @property
    def block_precision(self) -> float:
        """Fraction of block genes that are GT genes."""
        return self.overlap_genes / self.n_block_genes if self.n_block_genes > 0 else 0.0

    @property
    def size_ratio(self) -> float:
        """Block size / GT size."""
        return self.n_block_genes / self.n_gt_genes if self.n_gt_genes > 0 else 0.0


@dataclass
class PrecisionResult:
    method: str
    n_gt_instances: int = 0
    n_matched: int = 0
    n_missed: int = 0
    matches: List[BlockMatch] = field(default_factory=list)
    per_operon: Dict[str, List[BlockMatch]] = field(default_factory=lambda: defaultdict(list))


# ── Ground truth loading ─────────────────────────────────────────────

def load_ground_truth(gt_path: Path) -> Tuple[List[GTInstance], set]:
    """Load operon ground truth TSV. Returns (instances, ecoli_genomes)."""
    df = pd.read_csv(gt_path, sep='\t')
    instances = []
    genomes = set()
    for _, row in df.iterrows():
        instances.append(GTInstance(
            operon_id=row['operon_id'],
            genome_a=row['genome_a'],
            genome_b=row['genome_b'],
            contig_a=row['contig_a'],
            contig_b=row['contig_b'],
            start_a=int(row['gene_idx_start_a']),
            end_a=int(row['gene_idx_end_a']),
            start_b=int(row['gene_idx_start_b']),
            end_b=int(row['gene_idx_end_b']),
            n_genes_a=int(row['n_genes_a']),
            n_genes_b=int(row['n_genes_b']),
        ))
        genomes.add(row['genome_a'])
        genomes.add(row['genome_b'])
    return instances, genomes


# ── ELSA block matching ──────────────────────────────────────────────

def match_elsa_blocks(
    blocks_csv: Path,
    gt_instances: List[GTInstance],
    ecoli_genomes: set,
) -> PrecisionResult:
    """Match ELSA blocks to GT instances and compute precision."""
    df = pd.read_csv(blocks_csv)
    result = PrecisionResult(method="ELSA")

    # Index blocks by (genome_a, genome_b, contig_a, contig_b)
    block_index = defaultdict(list)
    for _, row in df.iterrows():
        ga, gb = row['query_genome'], row['target_genome']
        if ga not in ecoli_genomes or gb not in ecoli_genomes:
            continue
        ca, cb = row['query_contig'], row['target_contig']
        entry = (int(row['query_start']), int(row['query_end']),
                 int(row['target_start']), int(row['target_end']),
                 int(row['n_anchors']))
        block_index[(ga, gb, ca, cb)].append(entry)
        # Also index reverse direction
        entry_rev = (entry[2], entry[3], entry[0], entry[1], entry[4])
        block_index[(gb, ga, cb, ca)].append(entry_rev)

    for gt in gt_instances:
        result.n_gt_instances += 1
        key = (gt.genome_a, gt.genome_b, gt.contig_a, gt.contig_b)
        candidates = block_index.get(key, [])

        best_match = _find_best_match(
            candidates, gt.start_a, gt.end_a, gt.start_b, gt.end_b,
            gt.n_genes_a, gt.n_genes_b,
        )

        if best_match:
            result.n_matched += 1
            result.matches.append(best_match)
            result.per_operon[gt.operon_id].append(best_match)
        else:
            result.n_missed += 1

    return result


def _find_best_match(
    candidates: list,
    gt_start_a: int, gt_end_a: int,
    gt_start_b: int, gt_end_b: int,
    n_gt_a: int, n_gt_b: int,
    min_overlap_frac: float = 0.5,
) -> Optional[BlockMatch]:
    """Find the SMALLEST block that covers the GT operon on both sides.

    Among all blocks with ≥50% overlap on both sides, pick the one with
    smallest span. This gives the tightest bound on the operon.
    """
    best = None
    best_span = float('inf')

    for (bs_a, be_a, bs_b, be_b, n_genes) in candidates:
        # Overlap on side A
        ov_a_start = max(bs_a, gt_start_a)
        ov_a_end = min(be_a, gt_end_a)
        ov_a = max(0, ov_a_end - ov_a_start + 1)
        frac_a = ov_a / n_gt_a if n_gt_a > 0 else 0.0

        # Overlap on side B
        ov_b_start = max(bs_b, gt_start_b)
        ov_b_end = min(be_b, gt_end_b)
        ov_b = max(0, ov_b_end - ov_b_start + 1)
        frac_b = ov_b / n_gt_b if n_gt_b > 0 else 0.0

        if frac_a >= min_overlap_frac and frac_b >= min_overlap_frac:
            # Use block span (all genes between start and end) as size
            span_a = be_a - bs_a + 1
            if span_a < best_span:
                best_span = span_a
                best = BlockMatch(
                    block_start=bs_a,
                    block_end=be_a,
                    gt_start=gt_start_a,
                    gt_end=gt_end_a,
                    n_block_genes=span_a,  # use span, not anchor count
                    n_gt_genes=n_gt_a,
                )

    return best


# ── MCScanX block matching ───────────────────────────────────────────

def parse_mcscanx_gff(gff_path: Path) -> Dict[str, Tuple[str, str, int]]:
    """Parse MCScanX GFF to map internal_id -> (genome, contig, gene_idx)."""
    chrom_genes = defaultdict(list)
    with open(gff_path) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 4:
                chrom = parts[0]
                internal_id = parts[1]
                start = int(parts[2])
                chrom_genes[chrom].append((start, internal_id))

    gene_to_idx = {}
    for chrom, genes in chrom_genes.items():
        genes.sort(key=lambda x: x[0])
        m = re.match(r'(GCF_\d+\.\d+)_(.+)', chrom)
        if not m:
            continue
        genome, contig = m.group(1), m.group(2)
        for gene_idx, (start, internal_id) in enumerate(genes):
            gene_to_idx[internal_id] = (genome, contig, gene_idx)

    return gene_to_idx


def parse_mcscanx_collinearity(
    coll_path: Path, gene_to_idx: dict
) -> List[dict]:
    """Parse collinearity file into blocks with gene index ranges."""
    blocks = []
    current = None

    with open(coll_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('## Alignment'):
                if current and current['idxs_a'] and current['idxs_b']:
                    current['start_a'] = min(current['idxs_a'])
                    current['end_a'] = max(current['idxs_a'])
                    current['start_b'] = min(current['idxs_b'])
                    current['end_b'] = max(current['idxs_b'])
                    current['n_genes'] = len(current['idxs_a'])
                    blocks.append(current)

                chrom_info = None
                for p in line.split():
                    if '&' in p:
                        chrom_info = p
                        break

                if chrom_info:
                    ca, cb = chrom_info.split('&')
                    m_a = re.match(r'(GCF_\d+\.\d+)_(.+)', ca)
                    m_b = re.match(r'(GCF_\d+\.\d+)_(.+)', cb)
                    if m_a and m_b:
                        current = {
                            'genome_a': m_a.group(1), 'contig_a': m_a.group(2),
                            'genome_b': m_b.group(1), 'contig_b': m_b.group(2),
                            'idxs_a': [], 'idxs_b': [],
                        }
                    else:
                        current = None
                else:
                    current = None

            elif current and line and not line.startswith('#'):
                parts = line.split('\t')
                if len(parts) >= 3:
                    gene_a = parts[1].strip()
                    gene_b = parts[2].strip()
                    if gene_a in gene_to_idx and gene_b in gene_to_idx:
                        _, _, idx_a = gene_to_idx[gene_a]
                        _, _, idx_b = gene_to_idx[gene_b]
                        current['idxs_a'].append(idx_a)
                        current['idxs_b'].append(idx_b)

    # Last block
    if current and current['idxs_a'] and current['idxs_b']:
        current['start_a'] = min(current['idxs_a'])
        current['end_a'] = max(current['idxs_a'])
        current['start_b'] = min(current['idxs_b'])
        current['end_b'] = max(current['idxs_b'])
        current['n_genes'] = len(current['idxs_a'])
        blocks.append(current)

    return blocks


def match_mcscanx_blocks(
    mcscanx_blocks: List[dict],
    gt_instances: List[GTInstance],
    ecoli_genomes: set,
) -> PrecisionResult:
    """Match MCScanX blocks to GT instances and compute precision."""
    result = PrecisionResult(method="MCScanX")

    # Index blocks
    block_index = defaultdict(list)
    for b in mcscanx_blocks:
        ga, gb = b['genome_a'], b['genome_b']
        if ga not in ecoli_genomes or gb not in ecoli_genomes:
            continue
        ca, cb = b['contig_a'], b['contig_b']
        entry = (b['start_a'], b['end_a'], b['start_b'], b['end_b'], b['n_genes'])
        block_index[(ga, gb, ca, cb)].append(entry)
        entry_rev = (entry[2], entry[3], entry[0], entry[1], entry[4])
        block_index[(gb, ga, cb, ca)].append(entry_rev)

    for gt in gt_instances:
        result.n_gt_instances += 1
        key = (gt.genome_a, gt.genome_b, gt.contig_a, gt.contig_b)
        candidates = block_index.get(key, [])

        best_match = _find_best_match(
            candidates, gt.start_a, gt.end_a, gt.start_b, gt.end_b,
            gt.n_genes_a, gt.n_genes_b,
        )

        if best_match:
            result.n_matched += 1
            result.matches.append(best_match)
            result.per_operon[gt.operon_id].append(best_match)
        else:
            result.n_missed += 1

    return result


# ── Reporting ────────────────────────────────────────────────────────

def compute_stats(matches: List[BlockMatch]) -> dict:
    """Compute aggregate precision statistics."""
    if not matches:
        return {}

    ratios = [m.size_ratio for m in matches]
    recalls = [m.gt_recall for m in matches]
    precisions = [m.block_precision for m in matches]

    return {
        'n': len(matches),
        'ratio_mean': np.mean(ratios),
        'ratio_median': np.median(ratios),
        'ratio_std': np.std(ratios),
        'exact_frac': np.mean([0.9 <= r <= 1.1 for r in ratios]),
        'over_frac': np.mean([r > 1.1 for r in ratios]),
        'under_frac': np.mean([r < 0.9 for r in ratios]),
        'recall_mean': np.mean(recalls),
        'recall_100pct': np.mean([r >= 1.0 for r in recalls]),
        'precision_mean': np.mean(precisions),
        'precision_ge90': np.mean([p >= 0.9 for p in precisions]),
        'precision_100pct': np.mean([p >= 1.0 for p in precisions]),
    }


def write_report(
    elsa_result: PrecisionResult,
    mcscanx_result: PrecisionResult,
    output_path: Path,
):
    """Write comparison report."""
    es = compute_stats(elsa_result.matches)
    ms = compute_stats(mcscanx_result.matches)

    with open(output_path, 'w') as f:
        f.write("# Block Precision vs Operon Ground Truth\n\n")
        f.write("How tightly do syntenic blocks match known operon boundaries?\n\n")
        f.write("- **GT recall**: fraction of operon genes covered by the matching block\n")
        f.write("- **Block precision**: fraction of block genes that are operon genes\n")
        f.write("- **Size ratio**: block size / operon size (1.0 = exact match)\n\n")

        # Head-to-head table
        f.write("## Head-to-Head Comparison\n\n")
        f.write("| Metric | ELSA | MCScanX |\n")
        f.write("|--------|------|--------|\n")
        f.write(f"| GT instances | {elsa_result.n_gt_instances:,} | {mcscanx_result.n_gt_instances:,} |\n")
        f.write(f"| Matched (≥50% overlap) | {elsa_result.n_matched:,} ({100*elsa_result.n_matched/max(1,elsa_result.n_gt_instances):.1f}%) | {mcscanx_result.n_matched:,} ({100*mcscanx_result.n_matched/max(1,mcscanx_result.n_gt_instances):.1f}%) |\n")
        f.write(f"| Missed | {elsa_result.n_missed:,} | {mcscanx_result.n_missed:,} |\n")

        if es and ms:
            f.write(f"| **Size ratio** (mean) | {es['ratio_mean']:.3f} | {ms['ratio_mean']:.3f} |\n")
            f.write(f"| Size ratio (median) | {es['ratio_median']:.3f} | {ms['ratio_median']:.3f} |\n")
            f.write(f"| Exact match (0.9-1.1×) | {es['exact_frac']:.1%} | {ms['exact_frac']:.1%} |\n")
            f.write(f"| Over-recovery (>1.1×) | {es['over_frac']:.1%} | {ms['over_frac']:.1%} |\n")
            f.write(f"| Under-recovery (<0.9×) | {es['under_frac']:.1%} | {ms['under_frac']:.1%} |\n")
            f.write(f"| **GT recall** (mean) | {es['recall_mean']:.1%} | {ms['recall_mean']:.1%} |\n")
            f.write(f"| GT recall = 100% | {es['recall_100pct']:.1%} | {ms['recall_100pct']:.1%} |\n")
            f.write(f"| **Block precision** (mean) | {es['precision_mean']:.1%} | {ms['precision_mean']:.1%} |\n")
            f.write(f"| Precision ≥ 90% | {es['precision_ge90']:.1%} | {ms['precision_ge90']:.1%} |\n")
            f.write(f"| Precision = 100% | {es['precision_100pct']:.1%} | {ms['precision_100pct']:.1%} |\n")

        # Per-operon comparison
        f.write("\n## Per-Operon Breakdown\n\n")
        all_operons = sorted(set(list(elsa_result.per_operon.keys()) +
                                 list(mcscanx_result.per_operon.keys())))

        f.write("| Operon | GT genes | ELSA matched | ELSA ratio | ELSA precision | MCScanX matched | MCScanX ratio | MCScanX precision |\n")
        f.write("|--------|----------|-------------|------------|----------------|----------------|--------------|------------------|\n")

        for op in all_operons:
            # Get GT gene count from first match
            e_matches = elsa_result.per_operon.get(op, [])
            m_matches = mcscanx_result.per_operon.get(op, [])

            gt_genes = e_matches[0].n_gt_genes if e_matches else (m_matches[0].n_gt_genes if m_matches else "?")

            if e_matches:
                e_n = len(e_matches)
                e_ratio = np.mean([m.size_ratio for m in e_matches])
                e_prec = np.mean([m.block_precision for m in e_matches])
                e_str = f"{e_n}"
                e_ratio_str = f"{e_ratio:.2f}×"
                e_prec_str = f"{e_prec:.1%}"
            else:
                e_str = "0"
                e_ratio_str = "—"
                e_prec_str = "—"

            if m_matches:
                m_n = len(m_matches)
                m_ratio = np.mean([m.size_ratio for m in m_matches])
                m_prec = np.mean([m.block_precision for m in m_matches])
                m_str = f"{m_n}"
                m_ratio_str = f"{m_ratio:.2f}×"
                m_prec_str = f"{m_prec:.1%}"
            else:
                m_str = "0"
                m_ratio_str = "—"
                m_prec_str = "—"

            f.write(f"| {op} | {gt_genes} | {e_str} | {e_ratio_str} | {e_prec_str} | {m_str} | {m_ratio_str} | {m_prec_str} |\n")

        # Size ratio distribution
        f.write("\n## Size Ratio Distribution\n\n")
        bins = [(0, 0.5, "<0.5×"), (0.5, 0.9, "0.5-0.9×"),
                (0.9, 1.1, "0.9-1.1× (exact)"), (1.1, 2.0, "1.1-2.0×"),
                (2.0, 5.0, "2.0-5.0×"), (5.0, 1000, ">5.0×")]

        f.write("| Range | ELSA | MCScanX |\n")
        f.write("|-------|------|--------|\n")

        for lo, hi, label in bins:
            e_count = sum(1 for m in elsa_result.matches if lo <= m.size_ratio < hi)
            m_count = sum(1 for m in mcscanx_result.matches if lo <= m.size_ratio < hi)
            e_pct = 100 * e_count / max(1, len(elsa_result.matches))
            m_pct = 100 * m_count / max(1, len(mcscanx_result.matches))
            f.write(f"| {label} | {e_count} ({e_pct:.1f}%) | {m_count} ({m_pct:.1f}%) |\n")

    print(f"Report written to: {output_path}")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Compare block precision: ELSA vs MCScanX")
    parser.add_argument('--elsa-blocks', type=Path, required=True)
    parser.add_argument('--mcscanx-gff', type=Path, required=True)
    parser.add_argument('--mcscanx-collinearity', type=Path, required=True)
    parser.add_argument('--ground-truth', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()

    # Load GT
    print("Loading ground truth...")
    gt_instances, ecoli_genomes = load_ground_truth(args.ground_truth)
    print(f"  {len(gt_instances):,} instances, {len(ecoli_genomes)} genomes")

    # ELSA
    print("\nMatching ELSA blocks...")
    elsa_result = match_elsa_blocks(args.elsa_blocks, gt_instances, ecoli_genomes)
    print(f"  Matched: {elsa_result.n_matched:,}/{elsa_result.n_gt_instances:,}")

    # MCScanX
    print("\nParsing MCScanX GFF...")
    gene_to_idx = parse_mcscanx_gff(args.mcscanx_gff)
    print(f"  {len(gene_to_idx):,} genes indexed")

    print("Parsing MCScanX collinearity...")
    mcscanx_blocks = parse_mcscanx_collinearity(args.mcscanx_collinearity, gene_to_idx)
    print(f"  {len(mcscanx_blocks):,} blocks parsed")

    print("\nMatching MCScanX blocks...")
    mcscanx_result = match_mcscanx_blocks(mcscanx_blocks, gt_instances, ecoli_genomes)
    print(f"  Matched: {mcscanx_result.n_matched:,}/{mcscanx_result.n_gt_instances:,}")

    # Report
    print("\nWriting report...")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_report(elsa_result, mcscanx_result, args.output)

    # Print summary to stdout
    es = compute_stats(elsa_result.matches)
    ms = compute_stats(mcscanx_result.matches)
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'':30s} {'ELSA':>10s} {'MCScanX':>10s}")
    print(f"{'Matched instances':30s} {elsa_result.n_matched:>10,d} {mcscanx_result.n_matched:>10,d}")
    if es and ms:
        print(f"{'Size ratio (mean)':30s} {es['ratio_mean']:>10.3f} {ms['ratio_mean']:>10.3f}")
        print(f"{'Size ratio (median)':30s} {es['ratio_median']:>10.3f} {ms['ratio_median']:>10.3f}")
        print(f"{'Exact match (0.9-1.1×)':30s} {es['exact_frac']:>10.1%} {ms['exact_frac']:>10.1%}")
        print(f"{'Over-recovery (>1.1×)':30s} {es['over_frac']:>10.1%} {ms['over_frac']:>10.1%}")
        print(f"{'GT recall (mean)':30s} {es['recall_mean']:>10.1%} {ms['recall_mean']:>10.1%}")
        print(f"{'Block precision (mean)':30s} {es['precision_mean']:>10.1%} {ms['precision_mean']:>10.1%}")
        print(f"{'Precision ≥90%':30s} {es['precision_ge90']:>10.1%} {ms['precision_ge90']:>10.1%}")


if __name__ == '__main__':
    main()
