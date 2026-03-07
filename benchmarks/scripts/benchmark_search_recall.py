#!/usr/bin/env python3
"""
Benchmark locus search recall@k against RegulonDB operon ground truth.

For each of the 58 benchmark operons:
1. Extract the operon as a query locus from one reference genome
2. Run search against the 30-genome Enterobacteriaceae index
3. Score the ranked results against ground truth instances
4. Report recall@k for k in [1, 3, 5, 10, 20, 50]
5. Track cross-genus hits and their ranks

Usage:
    python benchmarks/scripts/benchmark_search_recall.py \
        --genes benchmarks/elsa_output/cross_species/elsa_index/ingest/genes.parquet \
        --ground-truth benchmarks/ground_truth/ecoli_operon_gt_v2.tsv \
        --output-dir benchmarks/evaluation \
        [--max-results 100] [--similarity-threshold 0.9]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from elsa.index import build_gene_index
from elsa.search import search_locus


# Species classification for cross-genus analysis
# Verified against FASTA headers (NCBI organism field)
SALMONELLA_ACCESSIONS = {
    "GCF_000006945.2",   # S. Typhimurium LT2
    "GCF_000007545.1",   # S. Typhi Ty2
    "GCF_000009505.1",   # S. Enteritidis P125109
    "GCF_000022165.1",   # S. Typhimurium 14028S
    "GCF_000195995.1",   # S. Typhi CT18
}
KLEBSIELLA_ACCESSIONS = {
    "GCF_000016305.1",   # K. pneumoniae MGH 78578
    "GCF_000240185.1",   # K. pneumoniae HS11286
    "GCF_000733495.1",   # K. michiganensis SA2
    "GCF_000742755.1",   # K. pneumoniae KPPR1
}


def classify_genome(genome_id: str) -> str:
    """Classify genome as E. coli, Salmonella, or Klebsiella."""
    if genome_id in SALMONELLA_ACCESSIONS:
        return "Salmonella"
    if genome_id in KLEBSIELLA_ACCESSIONS:
        return "Klebsiella"
    return "E_coli"


@dataclass
class OperonInstance:
    """A single operon instance in one genome."""
    operon_id: str
    genome: str
    contig: str
    gene_idx_start: int
    gene_idx_end: int
    n_genes: int
    gene_ids: List[str] = field(default_factory=list)  # Resolved from reference parquet


@dataclass
class SearchResult:
    """Result for one operon's search benchmark."""
    operon_id: str
    n_genes: int
    query_genome: str
    n_target_instances: int  # number of GT instances in other E. coli genomes
    recall_at_k: Dict[int, float] = field(default_factory=dict)
    hits_at_k: Dict[int, int] = field(default_factory=dict)
    n_blocks_returned: int = 0
    cross_genus_hits: List[Dict] = field(default_factory=list)
    query_time_s: float = 0.0
    # Per-instance detail: which genomes were found and at what rank
    genome_first_rank: Dict[str, int] = field(default_factory=dict)


def load_ground_truth(
    gt_path: Path,
    ref_genes: Optional[pd.DataFrame] = None,
) -> Dict[str, Dict[str, OperonInstance]]:
    """Load operon ground truth and build per-genome instance map.

    Args:
        gt_path: Path to ground truth TSV.
        ref_genes: Reference genes DataFrame (from the parquet that GT was
            built from). If provided, resolves gene_idx positions to gene_ids
            so that matching works across datasets with different gene-calling.

    Returns:
        Dict[operon_id -> Dict[genome_id -> OperonInstance]]
    """
    gt = pd.read_csv(gt_path, sep="\t")

    # Pre-compute position→gene_id lookup from reference parquet
    pos_to_gene: Dict[Tuple[str, str, int], str] = {}
    if ref_genes is not None:
        for _, row in ref_genes.iterrows():
            pos_to_gene[(row["sample_id"], row["contig_id"], row["position_index"])] = row["gene_id"]

    def _resolve_gene_ids(genome: str, contig: str, start: int, end: int) -> List[str]:
        if not pos_to_gene:
            return []
        return [
            pos_to_gene[(genome, contig, i)]
            for i in range(start, end + 1)
            if (genome, contig, i) in pos_to_gene
        ]

    # Build per-genome operon instances (deduplicate from pairwise format)
    instances: Dict[str, Dict[str, OperonInstance]] = defaultdict(dict)

    for _, row in gt.iterrows():
        op = row["operon_id"]
        # Side A
        ga = row["genome_a"]
        if ga not in instances[op]:
            instances[op][ga] = OperonInstance(
                operon_id=op, genome=ga, contig=row["contig_a"],
                gene_idx_start=int(row["gene_idx_start_a"]),
                gene_idx_end=int(row["gene_idx_end_a"]),
                n_genes=int(row["n_genes_a"]),
                gene_ids=_resolve_gene_ids(
                    ga, row["contig_a"],
                    int(row["gene_idx_start_a"]), int(row["gene_idx_end_a"]),
                ),
            )
        # Side B
        gb = row["genome_b"]
        if gb not in instances[op]:
            instances[op][gb] = OperonInstance(
                operon_id=op, genome=gb, contig=row["contig_b"],
                gene_idx_start=int(row["gene_idx_start_b"]),
                gene_idx_end=int(row["gene_idx_end_b"]),
                n_genes=int(row["n_genes_b"]),
                gene_ids=_resolve_gene_ids(
                    gb, row["contig_b"],
                    int(row["gene_idx_start_b"]), int(row["gene_idx_end_b"]),
                ),
            )

    return dict(instances)


def overlap_fraction(
    block_start: int, block_end: int,
    gt_start: int, gt_end: int,
) -> float:
    """Compute fraction of GT range covered by block range."""
    gt_len = gt_end - gt_start + 1
    if gt_len <= 0:
        return 0.0
    overlap_start = max(block_start, gt_start)
    overlap_end = min(block_end, gt_end)
    overlap = max(0, overlap_end - overlap_start + 1)
    return overlap / gt_len


def run_search_benchmark(
    genes_parquet: Path,
    gt_path: Path,
    output_dir: Path,
    ecoli_genes_parquet: Optional[Path] = None,
    max_results: int = 100,
    similarity_threshold: float = 0.85,
    max_gap: int = 2,
    min_chain_size: int = 2,
    gap_penalty_scale: float = 0.0,
    index_backend: str = "faiss_ivfflat",
    k_values: Optional[List[int]] = None,
    overlap_threshold: float = 0.5,
) -> List[SearchResult]:
    """Run the full search recall benchmark."""
    if k_values is None:
        k_values = [1, 3, 5, 10, 20, 25, 50]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load reference E. coli genes for GT position→gene_id mapping
    # (The GT was built from the E. coli 20-genome dataset, which may have
    # different gene-calling than the cross-species 30-genome dataset.)
    ref_genes = None
    if ecoli_genes_parquet and ecoli_genes_parquet.exists():
        print("[SearchBench] Loading reference E. coli genes for GT mapping...",
              file=sys.stderr, flush=True)
        ref_genes = pd.read_parquet(ecoli_genes_parquet)
        ref_genes = ref_genes.sort_values(["sample_id", "contig_id", "start", "end"])
        ref_genes["position_index"] = ref_genes.groupby(
            ["sample_id", "contig_id"]
        ).cumcount()
        print(f"[SearchBench] Reference: {len(ref_genes)} genes", file=sys.stderr, flush=True)

    # Load ground truth
    print("[SearchBench] Loading ground truth...", file=sys.stderr, flush=True)
    gt_instances = load_ground_truth(gt_path, ref_genes=ref_genes)
    n_resolved = sum(
        1 for ops in gt_instances.values()
        for inst in ops.values()
        if inst.gene_ids
    )
    total_inst = sum(len(ops) for ops in gt_instances.values())
    print(f"[SearchBench] {len(gt_instances)} operons loaded, "
          f"{n_resolved}/{total_inst} instances with gene_id mapping",
          file=sys.stderr, flush=True)

    # Load genes
    print("[SearchBench] Loading genes.parquet...", file=sys.stderr, flush=True)
    genes_df = pd.read_parquet(genes_parquet)
    genes_df = genes_df.sort_values(["sample_id", "contig_id", "start", "end"])
    genes_df["position_index"] = genes_df.groupby(
        ["sample_id", "contig_id"]
    ).cumcount()

    emb_cols = [c for c in genes_df.columns if c.startswith("emb_")]
    embeddings = genes_df[emb_cols].values.astype(np.float32)
    n_genes = len(genes_df)
    n_genomes = genes_df["sample_id"].nunique()
    print(
        f"[SearchBench] {n_genes} genes from {n_genomes} genomes (dim={len(emb_cols)})",
        file=sys.stderr, flush=True,
    )

    # Build gene_id to row index lookup for the search dataset
    gene_id_set = set(genes_df["gene_id"].values)

    # Build index
    print(f"[SearchBench] Building index ({index_backend})...", file=sys.stderr, flush=True)
    t0 = time.time()
    index_tuple = build_gene_index(
        embeddings,
        index_backend=index_backend,
        faiss_nprobe=32,
    )
    index_build_time = time.time() - t0
    print(f"[SearchBench] Index built in {index_build_time:.1f}s", file=sys.stderr, flush=True)

    # Run searches
    results: List[SearchResult] = []

    for op_idx, (operon_id, instances) in enumerate(sorted(gt_instances.items())):
        # Pick query genome: genome with the most complete copy (most genes)
        query_genome = max(instances.keys(), key=lambda g: instances[g].n_genes)
        query_instance = instances[query_genome]
        n_genes_op = query_instance.n_genes

        # Target instances: all OTHER E. coli genomes with this operon
        target_instances = {
            g: inst for g, inst in instances.items() if g != query_genome
        }
        n_targets = len(target_instances)

        # Extract query genes by gene_id (robust to position differences)
        if query_instance.gene_ids:
            query_mask = genes_df["gene_id"].isin(query_instance.gene_ids)
            query_genes_df = genes_df[query_mask].copy()
        else:
            # Fallback: position-based (only works if parquets match)
            query_mask = (
                (genes_df["sample_id"] == query_genome)
                & (genes_df["contig_id"] == query_instance.contig)
                & (genes_df["position_index"] >= query_instance.gene_idx_start)
                & (genes_df["position_index"] <= query_instance.gene_idx_end)
            )
            query_genes_df = genes_df[query_mask].copy()

        if len(query_genes_df) == 0:
            print(
                f"  [{op_idx+1}/58] {operon_id}: SKIP (no genes found for query "
                f"{query_genome}:{query_instance.contig}:"
                f"{query_instance.gene_idx_start}-{query_instance.gene_idx_end})",
                file=sys.stderr, flush=True,
            )
            continue

        # Run search
        t_start = time.time()
        blocks = search_locus(
            query_genes=query_genes_df,
            index_tuple=index_tuple,
            target_genes=genes_df,
            target_embeddings=embeddings,
            k=50,
            similarity_threshold=similarity_threshold,
            max_gap=max_gap,
            min_chain_size=min_chain_size,
            gap_penalty_scale=gap_penalty_scale,
            max_results=max_results,
        )
        query_time = time.time() - t_start

        # Score against ground truth
        result = SearchResult(
            operon_id=operon_id,
            n_genes=n_genes_op,
            query_genome=query_genome,
            n_target_instances=n_targets,
            n_blocks_returned=len(blocks),
            query_time_s=round(query_time, 4),
        )

        # Track which GT instances are found and at what rank
        found_genomes: Dict[str, int] = {}  # genome -> first rank found

        for rank, block in enumerate(blocks):
            tg = block.target_genome
            species = classify_genome(tg)

            # Check against GT instances (E. coli only)
            if tg in target_instances and tg not in found_genomes:
                gt_inst = target_instances[tg]

                # Match by gene_id overlap (robust to different gene-calling)
                if gt_inst.gene_ids:
                    gt_gene_set = set(gt_inst.gene_ids)
                    block_gene_ids = set(a.target_gene_id for a in block.anchors)
                    overlap_count = len(gt_gene_set & block_gene_ids)
                    frac = overlap_count / len(gt_gene_set) if gt_gene_set else 0.0
                else:
                    # Fallback: position-based overlap
                    frac = overlap_fraction(
                        block.target_start, block.target_end,
                        gt_inst.gene_idx_start, gt_inst.gene_idx_end,
                    )

                if frac >= overlap_threshold:
                    found_genomes[tg] = rank + 1  # 1-indexed rank

            # Track cross-genus hits
            if species != "E_coli":
                result.cross_genus_hits.append({
                    "rank": rank + 1,
                    "genome": tg,
                    "species": species,
                    "target_contig": block.target_contig,
                    "target_start": block.target_start,
                    "target_end": block.target_end,
                    "n_anchors": block.n_anchors,
                    "chain_score": round(block.chain_score, 4),
                })

        result.genome_first_rank = found_genomes

        # Compute recall@k
        for k in k_values:
            n_found = sum(1 for r in found_genomes.values() if r <= k)
            recall = n_found / max(1, n_targets)
            result.recall_at_k[k] = round(recall, 4)
            result.hits_at_k[k] = n_found

        print(
            f"  [{op_idx+1}/58] {operon_id} ({n_genes_op}g): "
            f"{len(blocks)} blocks, "
            f"R@5={result.recall_at_k.get(5, 0):.1%}, "
            f"R@20={result.recall_at_k.get(20, 0):.1%}, "
            f"R@50={result.recall_at_k.get(50, 0):.1%}, "
            f"cross-genus={len(result.cross_genus_hits)}, "
            f"time={query_time:.3f}s",
            file=sys.stderr, flush=True,
        )

        results.append(result)

    return results


def write_results(
    results: List[SearchResult],
    output_dir: Path,
    k_values: Optional[List[int]] = None,
):
    """Write JSON and summary markdown."""
    if k_values is None:
        k_values = [1, 3, 5, 10, 20, 25, 50]

    output_dir = Path(output_dir)

    # JSON output
    json_path = output_dir / "search_recall_at_k.json"
    json_data = {
        "metadata": {
            "n_operons": len(results),
            "k_values": k_values,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "per_operon": [
            {
                "operon_id": r.operon_id,
                "n_genes": r.n_genes,
                "query_genome": r.query_genome,
                "n_target_instances": r.n_target_instances,
                "n_blocks_returned": r.n_blocks_returned,
                "recall_at_k": r.recall_at_k,
                "hits_at_k": r.hits_at_k,
                "query_time_s": r.query_time_s,
                "n_cross_genus_hits": len(r.cross_genus_hits),
                "cross_genus_hits": r.cross_genus_hits[:5],  # top 5 only for JSON size
                "genome_first_rank": r.genome_first_rank,
            }
            for r in results
        ],
    }

    # Compute aggregates
    for k in k_values:
        recalls = [r.recall_at_k.get(k, 0) for r in results]
        json_data["metadata"][f"mean_recall_at_{k}"] = round(np.mean(recalls), 4)
        json_data["metadata"][f"std_recall_at_{k}"] = round(np.std(recalls), 4)
        json_data["metadata"][f"median_recall_at_{k}"] = round(np.median(recalls), 4)

    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"[SearchBench] Wrote {json_path}", file=sys.stderr, flush=True)

    # Summary markdown
    md_path = output_dir / "search_recall_at_k_summary.md"
    with open(md_path, "w") as f:
        f.write("# Search Recall@k Benchmark\n\n")
        f.write(f"**Operons evaluated**: {len(results)}\n")
        f.write(f"**Date**: {time.strftime('%Y-%m-%d')}\n\n")

        # Aggregate recall table
        f.write("## Aggregate Recall\n\n")
        f.write("| k | Mean Recall | Std | Median | Min | Max |\n")
        f.write("|---|------------|-----|--------|-----|-----|\n")
        for k in k_values:
            recalls = [r.recall_at_k.get(k, 0) for r in results]
            f.write(
                f"| {k} | {np.mean(recalls):.1%} | {np.std(recalls):.1%} | "
                f"{np.median(recalls):.1%} | {np.min(recalls):.1%} | "
                f"{np.max(recalls):.1%} |\n"
            )

        # Breakdown by operon size
        f.write("\n## Recall by Operon Size\n\n")
        short = [r for r in results if r.n_genes <= 4]
        long = [r for r in results if r.n_genes >= 8]
        mid = [r for r in results if 4 < r.n_genes < 8]

        f.write("| Size Category | N | R@1 | R@5 | R@10 | R@20 | R@50 |\n")
        f.write("|---------------|---|-----|-----|------|------|------|\n")
        for label, group in [("Short (2-4 genes)", short), ("Medium (5-7 genes)", mid),
                              ("Long (8-14 genes)", long), ("All", results)]:
            if not group:
                continue
            row = f"| {label} | {len(group)} |"
            for k in [1, 5, 10, 20, 50]:
                vals = [r.recall_at_k.get(k, 0) for r in group]
                row += f" {np.mean(vals):.1%} |"
            f.write(row + "\n")

        # Cross-genus hit summary
        f.write("\n## Cross-Genus Search Hits\n\n")
        n_with_cross = sum(1 for r in results if r.cross_genus_hits)
        total_cross = sum(len(r.cross_genus_hits) for r in results)
        sal_hits = sum(
            sum(1 for h in r.cross_genus_hits if h["species"] == "Salmonella")
            for r in results
        )
        kleb_hits = sum(
            sum(1 for h in r.cross_genus_hits if h["species"] == "Klebsiella")
            for r in results
        )
        f.write(f"- Operons with cross-genus hits: **{n_with_cross}/{len(results)}**\n")
        f.write(f"- Total cross-genus blocks returned: **{total_cross}**\n")
        f.write(f"  - Salmonella: {sal_hits}\n")
        f.write(f"  - Klebsiella: {kleb_hits}\n\n")

        # Cross-genus rank distribution
        if total_cross > 0:
            cross_ranks = []
            for r in results:
                for h in r.cross_genus_hits:
                    cross_ranks.append(h["rank"])
            f.write(f"- Cross-genus hit rank: median={np.median(cross_ranks):.0f}, "
                    f"mean={np.mean(cross_ranks):.1f}, "
                    f"min={np.min(cross_ranks)}, max={np.max(cross_ranks)}\n\n")

        # Per-operon detail table
        f.write("## Per-Operon Results\n\n")
        f.write("| Operon | Genes | Targets | Blocks | R@1 | R@5 | R@10 | R@20 | R@50 | Cross-Genus | Time(s) |\n")
        f.write("|--------|-------|---------|--------|-----|-----|------|------|------|-------------|--------|\n")
        for r in sorted(results, key=lambda x: x.operon_id):
            f.write(
                f"| {r.operon_id} | {r.n_genes} | {r.n_target_instances} | "
                f"{r.n_blocks_returned} | "
                f"{r.recall_at_k.get(1, 0):.1%} | "
                f"{r.recall_at_k.get(5, 0):.1%} | "
                f"{r.recall_at_k.get(10, 0):.1%} | "
                f"{r.recall_at_k.get(20, 0):.1%} | "
                f"{r.recall_at_k.get(50, 0):.1%} | "
                f"{len(r.cross_genus_hits)} | "
                f"{r.query_time_s:.3f} |\n"
            )

        # Query timing summary
        f.write("\n## Query Timing\n\n")
        times = [r.query_time_s for r in results]
        f.write(f"- Median: {np.median(times):.3f}s\n")
        f.write(f"- Mean: {np.mean(times):.3f}s\n")
        f.write(f"- 95th percentile: {np.percentile(times, 95):.3f}s\n")
        f.write(f"- Min: {np.min(times):.3f}s\n")
        f.write(f"- Max: {np.max(times):.3f}s\n")

    print(f"[SearchBench] Wrote {md_path}", file=sys.stderr, flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark ELSA locus search recall@k against operon ground truth"
    )
    parser.add_argument(
        "--genes",
        type=Path,
        default=PROJECT_ROOT / "benchmarks" / "elsa_output" / "cross_species" / "elsa_index" / "ingest" / "genes.parquet",
        help="Path to genes.parquet for the indexed dataset",
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=PROJECT_ROOT / "benchmarks" / "ground_truth" / "ecoli_operon_gt_v2.tsv",
        help="Path to operon ground truth TSV",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "benchmarks" / "evaluation",
        help="Directory for output files",
    )
    parser.add_argument("--max-results", type=int, default=100,
                        help="Maximum search results per query")
    parser.add_argument("--similarity-threshold", type=float, default=0.85)
    parser.add_argument("--max-gap", type=int, default=2)
    parser.add_argument("--gap-penalty-scale", type=float, default=0.0)
    parser.add_argument("--index-backend", type=str, default="faiss_ivfflat")
    parser.add_argument("--overlap-threshold", type=float, default=0.5,
                        help="Minimum overlap fraction for a GT match")
    parser.add_argument(
        "--ecoli-genes", type=Path,
        default=PROJECT_ROOT / "benchmarks" / "elsa_output" / "ecoli" / "elsa_index" / "ingest" / "genes.parquet",
        help="Path to E. coli genes.parquet (reference for GT position→gene_id mapping)",
    )

    args = parser.parse_args()

    if not args.genes.exists():
        # Try alternate paths
        for alt in [
            PROJECT_ROOT / "benchmarks" / "elsa_output" / "cross_species" / "ingest" / "genes.parquet",
            PROJECT_ROOT / "benchmarks" / "data" / "cross_species" / "cross_species_index" / "ingest" / "genes.parquet",
        ]:
            if alt.exists():
                args.genes = alt
                break
        else:
            print(f"ERROR: genes.parquet not found at {args.genes}", file=sys.stderr)
            sys.exit(1)

    if not args.ground_truth.exists():
        print(f"ERROR: Ground truth not found at {args.ground_truth}", file=sys.stderr)
        sys.exit(1)

    results = run_search_benchmark(
        genes_parquet=args.genes,
        gt_path=args.ground_truth,
        output_dir=args.output_dir,
        ecoli_genes_parquet=args.ecoli_genes if args.ecoli_genes.exists() else None,
        max_results=args.max_results,
        similarity_threshold=args.similarity_threshold,
        max_gap=args.max_gap,
        gap_penalty_scale=args.gap_penalty_scale,
        index_backend=args.index_backend,
        overlap_threshold=args.overlap_threshold,
    )

    write_results(results, args.output_dir)

    # Print summary to stdout
    k_values = [1, 3, 5, 10, 20, 25, 50]
    print("\n=== Search Recall@k Summary ===")
    for k in k_values:
        recalls = [r.recall_at_k.get(k, 0) for r in results]
        print(f"  Recall@{k:2d}: {np.mean(recalls):.1%} +/- {np.std(recalls):.1%}")

    n_cross = sum(1 for r in results if r.cross_genus_hits)
    print(f"  Cross-genus operons: {n_cross}/{len(results)}")
    print(f"  Median query time: {np.median([r.query_time_s for r in results]):.3f}s")


if __name__ == "__main__":
    main()
