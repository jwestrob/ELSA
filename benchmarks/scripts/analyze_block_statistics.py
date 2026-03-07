#!/usr/bin/env python3
"""
Analyze ELSA block statistics: size distribution, cross-genus breakdown,
fragmentation metrics, and negative control (random genome pairs).
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def analyze_blocks(blocks_df: pd.DataFrame, genes_df: pd.DataFrame) -> dict:
    """Compute comprehensive block statistics."""
    n_blocks = len(blocks_df)
    n_clusters = blocks_df["cluster_id"].nunique() if "cluster_id" in blocks_df.columns else 0

    # Size distribution
    sizes = blocks_df["n_genes"].values if "n_genes" in blocks_df.columns else blocks_df["n_anchors"].values
    size_stats = {
        "mean": float(np.mean(sizes)),
        "median": float(np.median(sizes)),
        "std": float(np.std(sizes)),
        "min": int(np.min(sizes)),
        "max": int(np.max(sizes)),
        "q25": float(np.percentile(sizes, 25)),
        "q75": float(np.percentile(sizes, 75)),
    }

    # Size buckets
    buckets = {
        "2-5": int(((sizes >= 2) & (sizes <= 5)).sum()),
        "6-10": int(((sizes >= 6) & (sizes <= 10)).sum()),
        "11-25": int(((sizes >= 11) & (sizes <= 25)).sum()),
        "26-50": int(((sizes >= 26) & (sizes <= 50)).sum()),
        "51-100": int(((sizes >= 51) & (sizes <= 100)).sum()),
        "101-500": int(((sizes >= 101) & (sizes <= 500)).sum()),
        "500+": int((sizes > 500).sum()),
    }

    # Orientation breakdown
    if "orientation" in blocks_df.columns:
        orient_counts = blocks_df["orientation"].value_counts().to_dict()
        orientation = {
            "forward": int(orient_counts.get(1, 0)),
            "reverse": int(orient_counts.get(-1, 0)),
            "mixed": int(orient_counts.get(0, 0)),
        }
    else:
        orientation = {}

    # Cross-genome analysis
    genus_map = {}
    for sample in blocks_df["query_genome"].unique():
        genus_map[sample] = infer_genus(sample, genes_df)
    for sample in blocks_df["target_genome"].unique():
        if sample not in genus_map:
            genus_map[sample] = infer_genus(sample, genes_df)

    genus_pairs = Counter()
    for _, row in blocks_df.iterrows():
        g1 = genus_map.get(row["query_genome"], "Unknown")
        g2 = genus_map.get(row["target_genome"], "Unknown")
        pair = tuple(sorted([g1, g2]))
        genus_pairs[pair] += 1

    cross_genus = sum(c for (g1, g2), c in genus_pairs.items() if g1 != g2)
    within_genus = sum(c for (g1, g2), c in genus_pairs.items() if g1 == g2)

    return {
        "n_blocks": n_blocks,
        "n_clusters": n_clusters,
        "size_stats": size_stats,
        "size_buckets": buckets,
        "orientation": orientation,
        "genus_pairs": {f"{g1}-{g2}": c for (g1, g2), c in genus_pairs.most_common()},
        "cross_genus_blocks": cross_genus,
        "within_genus_blocks": within_genus,
        "cross_genus_fraction": cross_genus / n_blocks if n_blocks > 0 else 0.0,
    }


# Known GCF -> genus mappings for our benchmark genomes
GENUS_MAP = {
    # E. coli (20 genomes)
    "GCF_000597845.1": "Escherichia", "GCF_000599625.1": "Escherichia",
    "GCF_000599645.1": "Escherichia", "GCF_000599665.1": "Escherichia",
    "GCF_000599685.1": "Escherichia", "GCF_000599705.1": "Escherichia",
    "GCF_000784925.1": "Escherichia", "GCF_000801165.1": "Escherichia",
    "GCF_000801185.2": "Escherichia", "GCF_000814145.2": "Escherichia",
    "GCF_000819645.1": "Escherichia", "GCF_000830035.1": "Escherichia",
    "GCF_000833145.1": "Escherichia", "GCF_000833635.2": "Escherichia",
    "GCF_000931565.1": "Escherichia", "GCF_000952955.1": "Escherichia",
    "GCF_000953515.1": "Escherichia", "GCF_000967155.2": "Escherichia",
    "GCF_000971615.1": "Escherichia", "GCF_000987875.1": "Escherichia",
    # Salmonella (5 genomes)
    "GCF_000006945.2": "Salmonella",  # S. Typhimurium LT2
    "GCF_000022165.1": "Salmonella",  # S. Typhimurium 14028S
    "GCF_000007545.1": "Salmonella",  # S. Typhi CT18
    "GCF_000195995.1": "Salmonella",  # S. Typhi Ty2
    "GCF_000009505.1": "Salmonella",  # S. Enteritidis P125109
    # Klebsiella (5 genomes)
    "GCF_000240185.1": "Klebsiella",  # K. pneumoniae HS11286
    "GCF_000016305.1": "Klebsiella",  # K. pneumoniae MGH78578
    "GCF_000742755.1": "Klebsiella",  # K. pneumoniae KPNIH1
    "GCF_000733495.1": "Klebsiella",  # K. pneumoniae 1084
    "GCF_000714595.1": "Klebsiella",  # K. pneumoniae KP617
}


def infer_genus(sample_id: str, genes_df: pd.DataFrame) -> str:
    """Infer genus from sample ID."""
    return GENUS_MAP.get(sample_id, "Unknown")


def main():
    parser = argparse.ArgumentParser(description="Analyze ELSA block statistics")
    parser.add_argument("--blocks", type=Path, required=True)
    parser.add_argument("--genes", type=Path, required=True)
    parser.add_argument("-o", "--output", type=Path)
    args = parser.parse_args()

    blocks_df = pd.read_csv(args.blocks)
    genes_df = pd.read_parquet(args.genes, columns=["sample_id", "contig_id", "gene_id"])

    print(f"Loaded {len(blocks_df)} blocks, {len(genes_df)} genes")

    stats = analyze_blocks(blocks_df, genes_df)

    print(f"\n{'='*60}")
    print(f"BLOCK STATISTICS")
    print(f"{'='*60}")
    print(f"Total blocks: {stats['n_blocks']}")
    print(f"Total clusters: {stats['n_clusters']}")

    print(f"\nSize distribution:")
    for k, v in stats["size_stats"].items():
        print(f"  {k}: {v:.1f}" if isinstance(v, float) else f"  {k}: {v}")

    print(f"\nSize buckets:")
    for bucket, count in stats["size_buckets"].items():
        pct = count / stats["n_blocks"] * 100
        print(f"  {bucket:>8}: {count:>6} ({pct:.1f}%)")

    if stats["orientation"]:
        print(f"\nOrientation:")
        for k, v in stats["orientation"].items():
            print(f"  {k}: {v}")

    print(f"\nGenus pair breakdown:")
    for pair, count in stats["genus_pairs"].items():
        pct = count / stats["n_blocks"] * 100
        print(f"  {pair:>35}: {count:>6} ({pct:.1f}%)")

    print(f"\nCross-genus: {stats['cross_genus_blocks']} ({stats['cross_genus_fraction']:.1%})")
    print(f"Within-genus: {stats['within_genus_blocks']}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
