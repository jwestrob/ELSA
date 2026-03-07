#!/usr/bin/env python3
"""
Benchmark gap penalty configurations on operon recall.

Compares three configurations:
1. Hard cutoff max_gap=2 (current default, strict)
2. Hard cutoff max_gap=5 (relaxed)
3. Concave gap penalty with log2 scaling (scale=1.0, max_gap=10)

Runs the full chain pipeline on 20-genome E. coli dataset and evaluates
operon recall for each configuration.

Usage:
    python benchmarks/scripts/benchmark_gap_penalty.py \
        --genes benchmarks/elsa_output/ecoli/elsa_index/ingest/genes.parquet \
        --ground-truth benchmarks/ground_truth/ecoli_operon_gt_v2.tsv \
        --output-dir benchmarks/evaluation
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from elsa.analyze.pipeline import run_chain_pipeline, ChainConfig, ChainSummary


@dataclass
class GapPenaltyConfig:
    """A single gap penalty configuration to benchmark."""
    name: str
    max_gap_genes: int
    gap_penalty_scale: float
    description: str


# The three configurations to compare
CONFIGS = [
    GapPenaltyConfig(
        name="hard_gap2",
        max_gap_genes=2,
        gap_penalty_scale=0.0,
        description="Hard cutoff max_gap=2 (current default)",
    ),
    GapPenaltyConfig(
        name="hard_gap5",
        max_gap_genes=5,
        gap_penalty_scale=0.0,
        description="Hard cutoff max_gap=5 (relaxed)",
    ),
    GapPenaltyConfig(
        name="concave_scale1",
        max_gap_genes=10,  # relaxed hard limit, let the penalty do the work
        gap_penalty_scale=1.0,
        description="Concave gap penalty scale=1.0 (log2), max_gap=10",
    ),
]


def load_operon_per_genome(gt_path: Path) -> Dict[str, Dict[str, Tuple[str, int, int]]]:
    """Load operon ground truth into per-genome format.

    Returns:
        Dict[operon_id -> Dict[genome_id -> (contig, start, end)]]
    """
    gt = pd.read_csv(gt_path, sep="\t")
    instances: Dict[str, Dict[str, Tuple[str, int, int]]] = defaultdict(dict)

    for _, row in gt.iterrows():
        op = row["operon_id"]
        ga = row["genome_a"]
        if ga not in instances[op]:
            instances[op][ga] = (row["contig_a"], int(row["gene_idx_start_a"]),
                                 int(row["gene_idx_end_a"]))
        gb = row["genome_b"]
        if gb not in instances[op]:
            instances[op][gb] = (row["contig_b"], int(row["gene_idx_start_b"]),
                                 int(row["gene_idx_end_b"]))

    return dict(instances)


def evaluate_operon_recall(
    blocks_df: pd.DataFrame,
    gt_instances: Dict[str, Dict[str, Tuple[str, int, int]]],
    overlap_threshold: float = 0.5,
) -> Dict:
    """Evaluate operon recall metrics against ground truth.

    Returns dict with strict, independent, any_coverage recall and per-operon detail.
    """
    results = {
        "strict_count": 0,
        "strict_total": 0,
        "independent_count": 0,
        "independent_total": 0,
        "any_count": 0,
        "any_total": 0,
        "per_operon": {},
    }

    # Build per-genome block lookup for fast overlap checking
    # genome -> contig -> list of (start, end)
    blocks_by_genome: Dict[str, Dict[str, List[Tuple[int, int]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for _, row in blocks_df.iterrows():
        qg = str(row["query_genome"])
        qc = str(row["query_contig"])
        blocks_by_genome[qg][qc].append((int(row["query_start"]), int(row["query_end"])))
        tg = str(row["target_genome"])
        tc = str(row["target_contig"])
        blocks_by_genome[tg][tc].append((int(row["target_start"]), int(row["target_end"])))

    # Also track which block covers which genome pair (for strict recall)
    block_pairs = []
    for _, row in blocks_df.iterrows():
        block_pairs.append({
            "query_genome": str(row["query_genome"]),
            "query_contig": str(row["query_contig"]),
            "query_start": int(row["query_start"]),
            "query_end": int(row["query_end"]),
            "target_genome": str(row["target_genome"]),
            "target_contig": str(row["target_contig"]),
            "target_start": int(row["target_start"]),
            "target_end": int(row["target_end"]),
        })

    def covers(genome, contig, gt_start, gt_end) -> bool:
        """Check if any block covers >=50% of the GT range in this genome."""
        gt_len = gt_end - gt_start + 1
        for bs, be in blocks_by_genome[genome].get(contig, []):
            ov_s = max(bs, gt_start)
            ov_e = min(be, gt_end)
            ov = max(0, ov_e - ov_s + 1)
            if ov / gt_len >= overlap_threshold:
                return True
        return False

    # Evaluate each operon pair
    all_genomes = set()
    for inst_map in gt_instances.values():
        all_genomes.update(inst_map.keys())
    genome_list = sorted(all_genomes)

    for operon_id, inst_map in gt_instances.items():
        genomes_with_operon = sorted(inst_map.keys())

        # For each pair of genomes
        pair_strict = 0
        pair_indep = 0
        pair_any = 0
        pair_total = 0

        for i, ga in enumerate(genomes_with_operon):
            for gb in genomes_with_operon[i + 1:]:
                pair_total += 1
                contig_a, start_a, end_a = inst_map[ga]
                contig_b, start_b, end_b = inst_map[gb]

                cov_a = covers(ga, contig_a, start_a, end_a)
                cov_b = covers(gb, contig_b, start_b, end_b)

                if cov_a or cov_b:
                    pair_any += 1
                if cov_a and cov_b:
                    pair_indep += 1

                # Strict: same block covers both
                strict_found = False
                for bp in block_pairs:
                    # Check if block covers ga->gb
                    if (bp["query_genome"] == ga and bp["target_genome"] == gb):
                        qa_len = end_a - start_a + 1
                        qb_len = end_b - start_b + 1
                        ov_a = max(0, min(bp["query_end"], end_a) - max(bp["query_start"], start_a) + 1)
                        ov_b = max(0, min(bp["target_end"], end_b) - max(bp["target_start"], start_b) + 1)
                        if ov_a / qa_len >= overlap_threshold and ov_b / qb_len >= overlap_threshold:
                            strict_found = True
                            break
                    # Check reverse direction
                    if (bp["query_genome"] == gb and bp["target_genome"] == ga):
                        qa_len = end_a - start_a + 1
                        qb_len = end_b - start_b + 1
                        ov_a = max(0, min(bp["target_end"], end_a) - max(bp["target_start"], start_a) + 1)
                        ov_b = max(0, min(bp["query_end"], end_b) - max(bp["query_start"], start_b) + 1)
                        if ov_a / qa_len >= overlap_threshold and ov_b / qb_len >= overlap_threshold:
                            strict_found = True
                            break
                if strict_found:
                    pair_strict += 1

        results["strict_count"] += pair_strict
        results["strict_total"] += pair_total
        results["independent_count"] += pair_indep
        results["independent_total"] += pair_total
        results["any_count"] += pair_any
        results["any_total"] += pair_total

        results["per_operon"][operon_id] = {
            "strict": pair_strict / max(1, pair_total),
            "independent": pair_indep / max(1, pair_total),
            "any": pair_any / max(1, pair_total),
            "n_pairs": pair_total,
        }

    results["strict_recall"] = results["strict_count"] / max(1, results["strict_total"])
    results["independent_recall"] = results["independent_count"] / max(1, results["independent_total"])
    results["any_recall"] = results["any_count"] / max(1, results["any_total"])

    return results


def run_gap_penalty_benchmark(
    genes_parquet: Path,
    gt_path: Path,
    output_dir: Path,
):
    """Run the full gap penalty comparison benchmark."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load ground truth
    gt_instances = load_operon_per_genome(gt_path)
    print(f"[GapBench] Loaded {len(gt_instances)} operons", file=sys.stderr, flush=True)

    all_results = {}

    for cfg in CONFIGS:
        print(f"\n{'='*60}", file=sys.stderr, flush=True)
        print(f"[GapBench] Running: {cfg.name} — {cfg.description}", file=sys.stderr, flush=True)
        print(f"{'='*60}", file=sys.stderr, flush=True)

        chain_config = ChainConfig(
            similarity_threshold=0.85,
            max_gap_genes=cfg.max_gap_genes,
            min_chain_size=2,
            gap_penalty_scale=cfg.gap_penalty_scale,
            hnsw_k=50,
            index_backend="faiss_ivfflat",
            faiss_nprobe=32,
            jaccard_tau=0.5,
            mutual_k=3,
            min_genome_support=2,
        )

        run_output = output_dir / "gap_penalty_runs" / cfg.name
        run_output.mkdir(parents=True, exist_ok=True)

        t0 = time.time()
        summary = run_chain_pipeline(
            genes_parquet=genes_parquet,
            output_dir=run_output,
            config=chain_config,
        )
        elapsed = time.time() - t0

        # Load resulting blocks
        blocks_path = run_output / "micro_chain_blocks.csv"
        blocks_df = pd.read_csv(blocks_path)

        # Evaluate operon recall
        recall_metrics = evaluate_operon_recall(blocks_df, gt_instances)

        # Block size distribution
        n_anchors = blocks_df["n_anchors"].values
        block_stats = {
            "total_blocks": len(blocks_df),
            "mean_size": float(np.mean(n_anchors)) if len(n_anchors) > 0 else 0,
            "median_size": float(np.median(n_anchors)) if len(n_anchors) > 0 else 0,
            "max_size": int(np.max(n_anchors)) if len(n_anchors) > 0 else 0,
            "blocks_gt20_genes": int(np.sum(n_anchors > 20)),
            "blocks_gt50_genes": int(np.sum(n_anchors > 50)),
            "blocks_gt100_genes": int(np.sum(n_anchors > 100)),
        }

        all_results[cfg.name] = {
            "config": {
                "name": cfg.name,
                "description": cfg.description,
                "max_gap_genes": cfg.max_gap_genes,
                "gap_penalty_scale": cfg.gap_penalty_scale,
            },
            "pipeline_summary": {
                "num_blocks": summary.num_blocks,
                "num_clusters": summary.num_clusters,
                "num_anchors": summary.num_anchors,
                "mean_block_size": summary.mean_block_size,
                "runtime_s": round(elapsed, 1),
            },
            "operon_recall": {
                "strict": round(recall_metrics["strict_recall"], 4),
                "independent": round(recall_metrics["independent_recall"], 4),
                "any_coverage": round(recall_metrics["any_recall"], 4),
            },
            "block_stats": block_stats,
        }

        print(
            f"\n[GapBench] {cfg.name}: "
            f"{summary.num_blocks} blocks, "
            f"strict={recall_metrics['strict_recall']:.1%}, "
            f"indep={recall_metrics['independent_recall']:.1%}, "
            f"any={recall_metrics['any_recall']:.1%}, "
            f"time={elapsed:.1f}s",
            file=sys.stderr, flush=True,
        )

    # Write JSON
    json_path = output_dir / "gap_penalty_comparison.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[GapBench] Wrote {json_path}", file=sys.stderr, flush=True)

    # Write summary markdown
    md_path = output_dir / "gap_penalty_comparison_summary.md"
    with open(md_path, "w") as f:
        f.write("# Gap Penalty Configuration Comparison\n\n")
        f.write(f"**Date**: {time.strftime('%Y-%m-%d')}\n")
        f.write("**Dataset**: 20-genome E. coli (90,566 genes)\n")
        f.write("**Ground truth**: 58 RegulonDB operons, 10,182 instances\n\n")

        f.write("## Configurations\n\n")
        f.write("| Config | max_gap | gap_penalty_scale | Description |\n")
        f.write("|--------|---------|-------------------|-------------|\n")
        for cfg in CONFIGS:
            f.write(f"| {cfg.name} | {cfg.max_gap_genes} | {cfg.gap_penalty_scale} | {cfg.description} |\n")

        f.write("\n## Operon Recall\n\n")
        f.write("| Config | Strict | Independent | Any Coverage |\n")
        f.write("|--------|--------|-------------|-------------|\n")
        for cfg in CONFIGS:
            r = all_results[cfg.name]["operon_recall"]
            f.write(f"| {cfg.name} | {r['strict']:.1%} | {r['independent']:.1%} | {r['any_coverage']:.1%} |\n")

        f.write("\n## Block Statistics\n\n")
        f.write("| Config | Total Blocks | Mean Size | Median Size | Max Size | Blocks >20g | Blocks >50g | Blocks >100g |\n")
        f.write("|--------|-------------|-----------|-------------|----------|-------------|-------------|-------------|\n")
        for cfg in CONFIGS:
            s = all_results[cfg.name]["block_stats"]
            f.write(
                f"| {cfg.name} | {s['total_blocks']:,} | {s['mean_size']:.1f} | "
                f"{s['median_size']:.0f} | {s['max_size']:,} | "
                f"{s['blocks_gt20_genes']:,} | {s['blocks_gt50_genes']:,} | "
                f"{s['blocks_gt100_genes']:,} |\n"
            )

        f.write("\n## Pipeline Runtime\n\n")
        f.write("| Config | Anchors | Blocks | Clusters | Runtime (s) |\n")
        f.write("|--------|---------|--------|----------|------------|\n")
        for cfg in CONFIGS:
            p = all_results[cfg.name]["pipeline_summary"]
            f.write(
                f"| {cfg.name} | {p['num_anchors']:,} | {p['num_blocks']:,} | "
                f"{p['num_clusters']:,} | {p['runtime_s']:.1f} |\n"
            )

        f.write("\n## Interpretation\n\n")
        f.write("The concave gap penalty with scale=1.0 should match or beat the BETTER of the two\n")
        f.write("hard cutoff configurations at each metric, demonstrating that users don't need to\n")
        f.write("manually select a gap cutoff. The log₂ penalty naturally balances between tight\n")
        f.write("chains (like max_gap=2) and allowing larger gaps (like max_gap=5) by penalizing\n")
        f.write("larger gaps diminishingly rather than applying a binary accept/reject.\n")

    print(f"[GapBench] Wrote {md_path}", file=sys.stderr, flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark gap penalty configurations on operon recall"
    )
    parser.add_argument(
        "--genes",
        type=Path,
        default=PROJECT_ROOT / "benchmarks" / "elsa_output" / "ecoli" / "elsa_index" / "ingest" / "genes.parquet",
        help="Path to genes.parquet (20-genome E. coli dataset)",
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

    args = parser.parse_args()

    if not args.genes.exists():
        print(f"ERROR: genes.parquet not found at {args.genes}", file=sys.stderr)
        sys.exit(1)

    run_gap_penalty_benchmark(args.genes, args.ground_truth, args.output_dir)


if __name__ == "__main__":
    main()
