#!/usr/bin/env python3
"""Sweep chaining similarity thresholds and report recall + block stats."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from elsa.analyze.gene_chain import (
    build_gene_index,
    find_cross_genome_anchors,
    group_anchors_by_contig_pair,
    chain_anchors_lis,
    extract_nonoverlapping_chains,
)


@dataclass(frozen=True)
class SweepResult:
    tau: float
    max_gap: int
    n_anchors: int
    n_blocks: int
    mean_block_size: float
    median_block_size: float
    strict_recall: float
    independent_recall: float
    any_recall: float
    runtime_sec: float


def parse_tau_list(values: str) -> list[float]:
    taus = []
    for raw in values.split(","):
        raw = raw.strip()
        if not raw:
            continue
        taus.append(float(raw))
    if not taus:
        raise ValueError("No tau values provided.")
    return sorted(set(taus))


def load_species_map(samples_path: Path) -> dict[str, str]:
    samples = pd.read_csv(samples_path, sep="\t")
    return dict(zip(samples["sample_id"], samples["species"]))


def filter_genes_by_species(
    genes_df: pd.DataFrame,
    species_map: dict[str, str] | None,
    species_filter: list[str] | None,
) -> pd.DataFrame:
    if not species_filter:
        return genes_df
    if species_map is None:
        raise ValueError("Species filter provided but no samples.tsv available.")
    keep = {sid for sid, sp in species_map.items() if sp in species_filter}
    filtered = genes_df[genes_df["sample_id"].isin(keep)].copy()
    return filtered


def check_overlap(block_start: int, block_end: int, operon_start: int, operon_end: int, threshold: float) -> tuple[bool, float]:
    overlap_start = max(block_start, operon_start)
    overlap_end = min(block_end, operon_end)
    if overlap_start > overlap_end:
        return False, 0.0
    overlap_size = overlap_end - overlap_start + 1
    operon_size = operon_end - operon_start + 1
    frac = overlap_size / operon_size
    return frac >= threshold, frac


def evaluate_operon_recall(blocks_df: pd.DataFrame, operon_gt: pd.DataFrame, threshold: float = 0.5) -> tuple[float, float, float]:
    blocks_by_pair: dict[tuple[str, str], list] = defaultdict(list)
    for row in blocks_df.itertuples(index=False):
        blocks_by_pair[(row.query_genome, row.target_genome)].append(row)

    strict_found = 0
    independent_found = 0
    any_found = 0

    total = len(operon_gt)
    for operon in operon_gt.itertuples(index=False):
        genome_a = operon.genome_a
        genome_b = operon.genome_b

        op_start_a = operon.gene_idx_start_a
        op_end_a = operon.gene_idx_end_a
        op_start_b = operon.gene_idx_start_b
        op_end_b = operon.gene_idx_end_b

        pair_blocks = blocks_by_pair.get((genome_a, genome_b), []) + blocks_by_pair.get((genome_b, genome_a), [])

        strict_hit = False
        best_a = 0.0
        best_b = 0.0

        for block in pair_blocks:
            if block.query_genome == genome_a:
                block_start_a = block.query_start
                block_end_a = block.query_end
                block_start_b = block.target_start
                block_end_b = block.target_end
            else:
                block_start_a = block.target_start
                block_end_a = block.target_end
                block_start_b = block.query_start
                block_end_b = block.query_end

            overlap_a, frac_a = check_overlap(block_start_a, block_end_a, op_start_a, op_end_a, threshold)
            overlap_b, frac_b = check_overlap(block_start_b, block_end_b, op_start_b, op_end_b, threshold)

            best_a = max(best_a, frac_a)
            best_b = max(best_b, frac_b)

            if overlap_a and overlap_b:
                strict_hit = True

        if strict_hit:
            strict_found += 1
        if best_a >= threshold and best_b >= threshold:
            independent_found += 1
        if best_a >= threshold or best_b >= threshold:
            any_found += 1

    if total == 0:
        return 0.0, 0.0, 0.0
    return strict_found / total, independent_found / total, any_found / total


def blocks_to_dataframe(blocks: list) -> pd.DataFrame:
    rows = []
    for block in blocks:
        rows.append(
            {
                "block_id": block.block_id,
                "cluster_id": 0,
                "query_genome": block.query_genome,
                "target_genome": block.target_genome,
                "query_contig": block.query_contig,
                "target_contig": block.target_contig,
                "query_start": block.query_start,
                "query_end": block.query_end,
                "target_start": block.target_start,
                "target_end": block.target_end,
                "n_anchors": block.n_anchors,
                "chain_score": round(block.chain_score, 4),
                "orientation": block.orientation,
                "n_genes": block.n_anchors,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--genes-parquet",
        type=Path,
        default=Path("benchmarks/elsa_output/cross_species/ingest/genes.parquet"),
    )
    parser.add_argument(
        "--samples",
        type=Path,
        default=Path("benchmarks/data/enterobacteriaceae/samples.tsv"),
        help="Samples TSV for species filtering (optional if --species all).",
    )
    parser.add_argument(
        "--operon-gt",
        type=Path,
        default=Path("benchmarks/ground_truth/ecoli_operon_gt_v2.tsv"),
        help="Operon ground truth TSV (optional if --no-operon).",
    )
    parser.add_argument(
        "--no-operon",
        action="store_true",
        help="Skip operon recall evaluation (for datasets without ground truth).",
    )
    parser.add_argument(
        "--species",
        default="ecoli",
        help="Comma-separated species filter (default: ecoli). Use 'all' for no filter.",
    )
    parser.add_argument(
        "--taus",
        default="0.70,0.75,0.80,0.82,0.85,0.88,0.90,0.92,0.94,0.96",
        help="Comma-separated cosine thresholds to sweep.",
    )
    parser.add_argument("--max-gap", type=int, default=2)
    parser.add_argument("--min-chain-size", type=int, default=2)
    parser.add_argument("--hnsw-k", type=int, default=50)
    parser.add_argument("--hnsw-m", type=int, default=32)
    parser.add_argument("--hnsw-ef-construction", type=int, default=200)
    parser.add_argument("--hnsw-ef-search", type=int, default=128)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/evaluation/threshold_sweep_summary.csv"),
    )
    args = parser.parse_args()

    taus = parse_tau_list(args.taus)
    tau_min = min(taus)

    genes_df = pd.read_parquet(args.genes_parquet)
    species_filter = None if args.species.strip().lower() == "all" else [s.strip() for s in args.species.split(",") if s.strip()]
    species_map = None
    if species_filter:
        if not args.samples.exists():
            raise FileNotFoundError(f"samples.tsv not found: {args.samples}")
        species_map = load_species_map(args.samples)
    genes_df = filter_genes_by_species(genes_df, species_map, species_filter)

    emb_cols = [c for c in genes_df.columns if c.startswith("emb_")]
    if not emb_cols:
        raise RuntimeError("genes.parquet contains no embedding columns (emb_*)")

    genes_df = genes_df.sort_values(["sample_id", "contig_id", "start", "end"]).copy()
    genes_df["position_index"] = genes_df.groupby(["sample_id", "contig_id"]).cumcount()

    embeddings = genes_df[emb_cols].values.astype(np.float32)
    gene_info = genes_df[["gene_id", "sample_id", "contig_id", "position_index"]].reset_index(drop=True)

    print(f"[Sweep] Genes: {len(genes_df):,} | Genomes: {genes_df['sample_id'].nunique()} | Dim: {len(emb_cols)}")
    print(f"[Sweep] Building index and anchors at tau >= {tau_min:.2f} (k={args.hnsw_k})...")

    index = build_gene_index(
        embeddings,
        m=args.hnsw_m,
        ef_construction=args.hnsw_ef_construction,
        ef_search=args.hnsw_ef_search,
    )
    anchors = find_cross_genome_anchors(
        index,
        embeddings,
        gene_info,
        k=args.hnsw_k,
        similarity_threshold=tau_min,
    )
    print(f"[Sweep] Anchors retained at tau >= {tau_min:.2f}: {len(anchors):,}")

    operon_gt = None
    gt_genomes: set[str] = set()
    if not args.no_operon:
        if args.operon_gt.exists():
            operon_gt = pd.read_csv(args.operon_gt, sep="\t")
            gt_genomes = set(operon_gt["genome_a"].unique()) | set(operon_gt["genome_b"].unique())
        else:
            print(f"[Sweep] Operon ground truth not found, skipping recall: {args.operon_gt}")

    results: list[SweepResult] = []

    for tau in taus:
        t0 = time.perf_counter()
        filtered = [a for a in anchors if a.similarity >= tau]
        groups = group_anchors_by_contig_pair(filtered)

        blocks = []
        block_id = 0
        n_chains = 0

        for _, group_anchors in groups.items():
            if len(group_anchors) < args.min_chain_size:
                continue
            chains = chain_anchors_lis(group_anchors, max_gap=args.max_gap, min_size=args.min_chain_size)
            if not chains:
                continue
            new_blocks = extract_nonoverlapping_chains(chains, block_id_start=block_id)
            blocks.extend(new_blocks)
            block_id += len(new_blocks)
            n_chains += len(chains)

        blocks_df = blocks_to_dataframe(blocks)
        if operon_gt is not None and not blocks_df.empty:
            blocks_df = blocks_df[
                (blocks_df["query_genome"].isin(gt_genomes)) & (blocks_df["target_genome"].isin(gt_genomes))
            ]

        if operon_gt is not None:
            strict_recall, independent_recall, any_recall = evaluate_operon_recall(blocks_df, operon_gt, threshold=0.5)
        else:
            strict_recall, independent_recall, any_recall = float("nan"), float("nan"), float("nan")

        sizes = blocks_df["n_anchors"].values if not blocks_df.empty else np.array([])
        mean_size = float(np.mean(sizes)) if sizes.size else 0.0
        median_size = float(np.median(sizes)) if sizes.size else 0.0

        runtime = time.perf_counter() - t0

        results.append(
            SweepResult(
                tau=tau,
                max_gap=args.max_gap,
                n_anchors=len(filtered),
                n_blocks=len(blocks_df),
                mean_block_size=mean_size,
                median_block_size=median_size,
                strict_recall=strict_recall,
                independent_recall=independent_recall,
                any_recall=any_recall,
                runtime_sec=runtime,
            )
        )

        recall_msg = "recall=NA"
        if operon_gt is not None:
            recall_msg = f"strict={strict_recall:.3f} indep={independent_recall:.3f} any={any_recall:.3f}"
        print(
            f"[Sweep] tau={tau:.2f} anchors={len(filtered):,} blocks={len(blocks_df):,} "
            f"{recall_msg} time={runtime:.1f}s"
        )

    out_df = pd.DataFrame([r.__dict__ for r in results])
    out_df.to_csv(args.output, index=False)
    print(f"[Sweep] Saved summary: {args.output}")


if __name__ == "__main__":
    main()
