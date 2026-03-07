#!/usr/bin/env python3
"""
Search recall@k benchmark for ELSA.

For each operon instance in the ground truth, use one genome's copy as a
query locus and check whether the known syntenic partners in other genomes
are returned by `elsa search`.

This tests the end-to-end search pipeline: embed query -> kNN -> chain -> rank.
"""

import argparse
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from elsa.index import build_gene_index
from elsa.search import search_locus


def load_genes(parquet_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path)
    # Compute position_index if missing
    if "position_index" not in df.columns:
        df = df.sort_values(["sample_id", "contig_id", "start"])
        df["position_index"] = df.groupby(["sample_id", "contig_id"]).cumcount()
    return df


def load_operon_gt(gt_path: Path) -> pd.DataFrame:
    return pd.read_csv(gt_path, sep="\t")


def main():
    parser = argparse.ArgumentParser(description="Search recall@k benchmark")
    parser.add_argument("--genes", type=Path, required=True,
                        help="Path to genes.parquet")
    parser.add_argument("--operon-gt", type=Path, required=True,
                        help="Path to operon ground truth TSV")
    parser.add_argument("--k-values", type=int, nargs="+", default=[10, 25, 50],
                        help="k values for recall@k")
    parser.add_argument("--similarity-threshold", type=float, default=0.85)
    parser.add_argument("--max-gap", type=int, default=2)
    parser.add_argument("--min-chain-size", type=int, default=2)
    parser.add_argument("--index-backend", default="faiss_ivfflat",
                        choices=["faiss_ivfflat", "hnsw", "sklearn"])
    parser.add_argument("--n-queries", type=int, default=200,
                        help="Number of query loci to sample (0=all)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("-o", "--output", type=Path)
    args = parser.parse_args()

    print("=" * 60)
    print("ELSA Search Recall@k Benchmark")
    print("=" * 60)

    # Load data
    print("\n[1/4] Loading data...")
    genes_df = load_genes(args.genes)
    emb_cols = [c for c in genes_df.columns if c.startswith("emb_")]
    dim = len(emb_cols)
    n_genes = len(genes_df)
    print(f"  {n_genes} genes, {dim}D embeddings, "
          f"{genes_df['sample_id'].nunique()} genomes")

    gt_df = load_operon_gt(args.operon_gt)
    print(f"  {len(gt_df)} operon instances in ground truth")

    # Build index
    print("\n[2/4] Building search index...")
    embeddings = genes_df[emb_cols].values.astype(np.float32)
    index_tuple = build_gene_index(embeddings, index_backend=args.index_backend)
    print(f"  Index built ({args.index_backend})")

    # Build lookup: for each operon_id, which genomes have it and at what positions?
    # Group GT by operon_id to find multi-genome operon families
    operon_families = defaultdict(list)
    for _, row in gt_df.iterrows():
        operon_families[row["operon_id"]].append(row)

    # Build query loci: for each operon family, pick one genome as query,
    # check if the search returns blocks overlapping other genomes' copies
    query_loci = []
    for operon_id, instances in operon_families.items():
        # Get unique genome pairs (each instance is a pair)
        genomes_with_operon = set()
        for inst in instances:
            genomes_with_operon.add(inst["genome_a"])
            genomes_with_operon.add(inst["genome_b"])

        # Use the first instance's genome_a as the query
        first = instances[0]
        query_genome = first["genome_a"]
        query_contig = first["contig_a"]
        query_start_idx = first["gene_idx_start_a"]
        query_end_idx = first["gene_idx_end_a"]
        target_genomes = genomes_with_operon - {query_genome}

        if not target_genomes:
            continue

        query_loci.append({
            "operon_id": operon_id,
            "query_genome": query_genome,
            "query_contig": query_contig,
            "query_start": query_start_idx,
            "query_end": query_end_idx,
            "target_genomes": target_genomes,
            "n_targets": len(target_genomes),
        })

    print(f"  {len(query_loci)} query loci from {len(operon_families)} operons")

    # Sample if requested
    rng = np.random.RandomState(args.seed)
    if args.n_queries > 0 and args.n_queries < len(query_loci):
        indices = rng.choice(len(query_loci), args.n_queries, replace=False)
        query_loci = [query_loci[i] for i in sorted(indices)]
        print(f"  Sampled {len(query_loci)} queries")

    # Run searches
    print(f"\n[3/4] Running searches (k values: {args.k_values})...")
    max_k = max(args.k_values)

    results = []
    for qi, qlocus in enumerate(query_loci):
        if (qi + 1) % 50 == 0:
            print(f"  {qi + 1}/{len(query_loci)} queries...")

        # Get query genes from the parquet
        mask = (
            (genes_df["sample_id"] == qlocus["query_genome"]) &
            (genes_df["contig_id"] == qlocus["query_contig"])
        )
        contig_genes = genes_df[mask].sort_values("start").reset_index(drop=True)

        if qlocus["query_end"] >= len(contig_genes):
            continue

        # Extract the operon region with 2 gene flanks
        flank = 2
        start = max(0, qlocus["query_start"] - flank)
        end = min(len(contig_genes) - 1, qlocus["query_end"] + flank)
        query_genes = contig_genes.iloc[start:end + 1].copy()

        if "position_index" not in query_genes.columns:
            query_genes["position_index"] = range(len(query_genes))
        else:
            query_genes["position_index"] = range(len(query_genes))

        if len(query_genes) < 2:
            continue

        # Search
        blocks = search_locus(
            query_genes=query_genes,
            index_tuple=index_tuple,
            target_genes=genes_df,
            target_embeddings=embeddings,
            k=max_k,
            similarity_threshold=args.similarity_threshold,
            max_gap=args.max_gap,
            min_chain_size=args.min_chain_size,
            max_results=max_k * 2,
        )

        # Check which target genomes were found
        found_genomes = set()
        for block in blocks:
            found_genomes.add(block.target_genome)

        for k_val in args.k_values:
            # Only consider top-k blocks
            top_k_genomes = set()
            for block in blocks[:k_val]:
                top_k_genomes.add(block.target_genome)

            hits = len(qlocus["target_genomes"] & top_k_genomes)
            total = qlocus["n_targets"]

            results.append({
                "operon_id": qlocus["operon_id"],
                "k": k_val,
                "hits": hits,
                "total": total,
                "recall": hits / total if total > 0 else 0.0,
                "n_blocks_returned": min(len(blocks), k_val),
            })

    # Summarize
    print(f"\n[4/4] Computing summary...")
    results_df = pd.DataFrame(results)

    print("\n" + "=" * 60)
    print("SEARCH RECALL@k RESULTS")
    print("=" * 60)
    print(f"\nQueries evaluated: {results_df['operon_id'].nunique()}")

    for k_val in args.k_values:
        k_results = results_df[results_df["k"] == k_val]
        mean_recall = k_results["recall"].mean()
        perfect = (k_results["recall"] == 1.0).sum()
        total_q = len(k_results)
        mean_blocks = k_results["n_blocks_returned"].mean()

        print(f"\n  Recall@{k_val}:")
        print(f"    Mean recall:      {mean_recall:.1%}")
        print(f"    Perfect recall:   {perfect}/{total_q} ({perfect/total_q:.1%})")
        print(f"    Mean blocks ret:  {mean_blocks:.1f}")

    # Save
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(args.output, index=False)
        print(f"\nDetailed results saved to {args.output}")


if __name__ == "__main__":
    main()
