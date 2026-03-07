#!/usr/bin/env python3
"""
Phase 5: Evaluate ELSA clusters against orthogroup-based ground truth.

Metrics:
1. Orthogroup recovery: Do ELSA clusters contain the same gene families?
2. Genome coverage agreement: Do they agree on which genomes share a block?
3. Adjusted Rand Index: Gene-pair co-clustering agreement
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd


def load_gt_blocks(gt_path: Path) -> list[dict]:
    """Load ground truth conserved blocks."""
    with open(gt_path) as f:
        return json.load(f)


def load_orthogroup_assignments(og_path: Path) -> dict[str, str]:
    """Load gene -> orthogroup mapping."""
    df = pd.read_csv(og_path, sep='\t')
    return dict(zip(df['gene_id'], df['orthogroup']))


def load_elsa_clusters(
    blocks_path: Path,
    clusters_path: Path,
    windows_path: Path,
) -> dict[int, dict]:
    """Load ELSA clusters with their gene content.

    Returns: cluster_id -> {genomes: set, genes: set, orthogroups: set}
    """
    # Load window -> genes mapping
    windows_df = pd.read_parquet(windows_path)
    window_to_genes = {}
    for _, row in windows_df.iterrows():
        wid = f"{row['sample_id']}_{row['locus_id']}_{row['window_idx']}"
        genes = row['gene_ids'].split(',') if pd.notna(row.get('gene_ids')) else []
        window_to_genes[wid] = set(genes)

    # Load blocks
    blocks_df = pd.read_csv(blocks_path)

    # Group by cluster
    clusters = defaultdict(lambda: {'genomes': set(), 'genes': set(), 'blocks': []})

    for _, row in blocks_df.iterrows():
        cluster_id = row['cluster_id']
        if cluster_id == 0:  # Skip sink cluster
            continue

        query_sample = row['query_locus'].split(':')[0]
        target_sample = row['target_locus'].split(':')[0]

        clusters[cluster_id]['genomes'].add(query_sample)
        clusters[cluster_id]['genomes'].add(target_sample)

        # Get genes from windows
        for wid in row['query_windows_json'].split(';'):
            if wid in window_to_genes:
                clusters[cluster_id]['genes'].update(window_to_genes[wid])
        for wid in row['target_windows_json'].split(';'):
            if wid in window_to_genes:
                clusters[cluster_id]['genes'].update(window_to_genes[wid])

        clusters[cluster_id]['blocks'].append(row['block_id'])

    return dict(clusters)


def compute_orthogroup_overlap(
    gt_blocks: list[dict],
    elsa_clusters: dict[int, dict],
    gene_to_og: dict[str, str],
) -> dict:
    """Compute orthogroup-based overlap between GT and ELSA."""

    # Add orthogroups to ELSA clusters
    for cluster_id, cluster in elsa_clusters.items():
        cluster['orthogroups'] = set(
            gene_to_og.get(g) for g in cluster['genes']
            if g in gene_to_og and gene_to_og.get(g) is not None
        )

    # For each GT block, find best matching ELSA cluster
    gt_matches = []

    for gt_block in gt_blocks:
        gt_ogs = set(gt_block['orthogroups'])
        gt_genomes = set(gt_block['genomes'])

        best_cluster = None
        best_og_jaccard = 0
        best_genome_jaccard = 0

        for cluster_id, cluster in elsa_clusters.items():
            elsa_ogs = cluster['orthogroups']
            elsa_genomes = cluster['genomes']

            # Orthogroup Jaccard
            if gt_ogs and elsa_ogs:
                og_inter = len(gt_ogs & elsa_ogs)
                og_union = len(gt_ogs | elsa_ogs)
                og_jaccard = og_inter / og_union if og_union > 0 else 0
            else:
                og_jaccard = 0

            # Genome Jaccard
            genome_inter = len(gt_genomes & elsa_genomes)
            genome_union = len(gt_genomes | elsa_genomes)
            genome_jaccard = genome_inter / genome_union if genome_union > 0 else 0

            # Combined score
            combined = og_jaccard * 0.7 + genome_jaccard * 0.3

            if combined > best_og_jaccard * 0.7 + best_genome_jaccard * 0.3:
                best_cluster = cluster_id
                best_og_jaccard = og_jaccard
                best_genome_jaccard = genome_jaccard

        gt_matches.append({
            'gt_block': gt_block['block_id'],
            'gt_n_ogs': len(gt_ogs),
            'gt_n_genomes': gt_block['n_genomes'],
            'best_cluster': best_cluster,
            'og_jaccard': best_og_jaccard,
            'genome_jaccard': best_genome_jaccard,
        })

    return gt_matches


def compute_gene_pair_ari(
    gt_blocks: list[dict],
    elsa_clusters: dict[int, dict],
) -> float:
    """Compute Adjusted Rand Index on gene-pair co-clustering."""

    # Build gene -> GT block mapping (gene can be in multiple blocks, use first)
    gene_to_gt = {}
    for block in gt_blocks:
        for genome, instance in block.get('instances', {}).items():
            for gene in instance.get('genes', []):
                if gene not in gene_to_gt:
                    gene_to_gt[gene] = block['block_id']

    # Build gene -> ELSA cluster mapping
    gene_to_elsa = {}
    for cluster_id, cluster in elsa_clusters.items():
        for gene in cluster['genes']:
            if gene not in gene_to_elsa:
                gene_to_elsa[gene] = cluster_id

    # Find genes in both
    common_genes = set(gene_to_gt.keys()) & set(gene_to_elsa.keys())

    if len(common_genes) < 2:
        return 0.0

    # Sample gene pairs (full enumeration is O(n^2))
    genes = list(common_genes)
    if len(genes) > 5000:
        np.random.seed(42)
        genes = list(np.random.choice(genes, 5000, replace=False))

    # Count contingency table entries
    a = 0  # Same in both
    b = 0  # Same in GT, different in ELSA
    c = 0  # Different in GT, same in ELSA
    d = 0  # Different in both

    for i, g1 in enumerate(genes):
        for g2 in genes[i+1:]:
            same_gt = (gene_to_gt[g1] == gene_to_gt[g2])
            same_elsa = (gene_to_elsa[g1] == gene_to_elsa[g2])

            if same_gt and same_elsa:
                a += 1
            elif same_gt and not same_elsa:
                b += 1
            elif not same_gt and same_elsa:
                c += 1
            else:
                d += 1

    # Compute ARI
    n = a + b + c + d
    if n == 0:
        return 0.0

    # ARI = (RI - Expected_RI) / (max_RI - Expected_RI)
    ri = (a + d) / n
    sum_gt_same = a + b
    sum_elsa_same = a + c
    expected_a = sum_gt_same * sum_elsa_same / n if n > 0 else 0
    expected_d = (n - sum_gt_same) * (n - sum_elsa_same) / n if n > 0 else 0
    expected_ri = (expected_a + expected_d) / n if n > 0 else 0

    if expected_ri >= 1.0:
        return 1.0

    ari = (ri - expected_ri) / (1.0 - expected_ri)
    return ari


def evaluate(
    gt_path: Path,
    orthogroups_path: Path,
    elsa_blocks_path: Path,
    elsa_clusters_path: Path,
    windows_path: Path,
    output_path: Path,
) -> dict:
    """Run full evaluation."""

    print("Loading data...")
    gt_blocks = load_gt_blocks(gt_path)
    print(f"  GT blocks: {len(gt_blocks)}")

    gene_to_og = load_orthogroup_assignments(orthogroups_path)
    print(f"  Gene-OG mappings: {len(gene_to_og)}")

    elsa_clusters = load_elsa_clusters(elsa_blocks_path, elsa_clusters_path, windows_path)
    print(f"  ELSA clusters: {len(elsa_clusters)}")

    # Compute matches
    print("\nComputing orthogroup overlap...")
    matches = compute_orthogroup_overlap(gt_blocks, elsa_clusters, gene_to_og)

    # Aggregate metrics
    og_jaccards = [m['og_jaccard'] for m in matches]
    genome_jaccards = [m['genome_jaccard'] for m in matches]

    matched_gt = sum(1 for m in matches if m['og_jaccard'] >= 0.3)
    high_match_gt = sum(1 for m in matches if m['og_jaccard'] >= 0.5)

    results = {
        'n_gt_blocks': len(gt_blocks),
        'n_elsa_clusters': len(elsa_clusters),
        'matched_gt_blocks': matched_gt,
        'high_match_gt_blocks': high_match_gt,
        'recall_at_0.3': matched_gt / len(gt_blocks) if gt_blocks else 0,
        'recall_at_0.5': high_match_gt / len(gt_blocks) if gt_blocks else 0,
        'mean_og_jaccard': np.mean(og_jaccards) if og_jaccards else 0,
        'mean_genome_jaccard': np.mean(genome_jaccards) if genome_jaccards else 0,
        'median_og_jaccard': np.median(og_jaccards) if og_jaccards else 0,
    }

    # Compute ARI
    print("Computing Adjusted Rand Index...")
    ari = compute_gene_pair_ari(gt_blocks, elsa_clusters)
    results['adjusted_rand_index'] = ari

    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS (Orthogroup-based GT)")
    print("=" * 60)
    print(f"GT Blocks:              {results['n_gt_blocks']}")
    print(f"ELSA Clusters:          {results['n_elsa_clusters']}")
    print("-" * 60)
    print(f"Matched GT (OG J≥0.3):  {results['matched_gt_blocks']} ({results['recall_at_0.3']:.1%})")
    print(f"High Match (OG J≥0.5):  {results['high_match_gt_blocks']} ({results['recall_at_0.5']:.1%})")
    print(f"Mean OG Jaccard:        {results['mean_og_jaccard']:.3f}")
    print(f"Median OG Jaccard:      {results['median_og_jaccard']:.3f}")
    print(f"Mean Genome Jaccard:    {results['mean_genome_jaccard']:.3f}")
    print("-" * 60)
    print(f"Adjusted Rand Index:    {results['adjusted_rand_index']:.3f}")
    print("=" * 60)

    # Save results
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_path}")

        # Save detailed matches
        matches_path = output_path.with_suffix('.matches.json')
        with open(matches_path, 'w') as f:
            json.dump(matches, f, indent=2)

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate ELSA against orthogroup GT")
    parser.add_argument("--gt", type=Path, required=True, help="GT conserved blocks JSON")
    parser.add_argument("--orthogroups", type=Path, required=True, help="Orthogroups TSV")
    parser.add_argument("--elsa-blocks", type=Path, required=True, help="ELSA syntenic_blocks.csv")
    parser.add_argument("--elsa-clusters", type=Path, required=True, help="ELSA syntenic_clusters.csv")
    parser.add_argument("--windows", type=Path, required=True, help="windows.parquet")
    parser.add_argument("-o", "--output", type=Path, help="Output JSON path")

    args = parser.parse_args()

    evaluate(
        args.gt,
        args.orthogroups,
        args.elsa_blocks,
        args.elsa_clusters,
        args.windows,
        args.output,
    )


if __name__ == "__main__":
    main()
