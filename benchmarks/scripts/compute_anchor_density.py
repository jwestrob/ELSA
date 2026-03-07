#!/usr/bin/env python3
"""Compute anchor density distributions for ELSA and MCScanX blocks."""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import pandas as pd

SCRIPT_DIR = Path(__file__).parent
BENCHMARKS_DIR = SCRIPT_DIR.parent


def parse_chrom(chrom: str) -> tuple[str | None, str | None]:
    m = re.match(r"(GCF_\d+\.\d+)_(.+)", chrom)
    if not m:
        return None, None
    return m.group(1), m.group(2)


def parse_mcscanx_gff(gff_path: Path) -> pd.DataFrame:
    rows = []
    with open(gff_path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 4:
                continue
            chrom, gene_id, start, end = parts[0], parts[1], int(parts[2]), int(parts[3])
            genome, contig = parse_chrom(chrom)
            if genome is None:
                continue
            rows.append(
                {
                    "gene_id": gene_id,
                    "genome": genome,
                    "contig": contig,
                    "start": start,
                    "end": end,
                }
            )
    gff = pd.DataFrame(rows)
    gff = gff.sort_values(["genome", "contig", "start"])
    gff["gene_idx"] = gff.groupby(["genome", "contig"]).cumcount()
    return gff


def parse_mcscanx_collinearity(coll_path: Path) -> list[dict]:
    blocks = []
    current = None
    with open(coll_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("## Alignment"):
                if current and current["gene_pairs"]:
                    blocks.append(current)
                parts = line.split()
                block_id = int(parts[2].rstrip(":"))
                current = {"block_id": block_id, "gene_pairs": []}
            elif current and line and not line.startswith("#"):
                parts = line.split("\t")
                if len(parts) >= 3:
                    current["gene_pairs"].append((parts[1].strip(), parts[2].strip()))
    if current and current["gene_pairs"]:
        blocks.append(current)
    return blocks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--elsa-blocks",
        type=Path,
        default=BENCHMARKS_DIR
        / "results"
        / "cross_species_chain"
        / "micro_chain"
        / "micro_chain_blocks.csv",
    )
    parser.add_argument(
        "--mcscanx-gff",
        type=Path,
        default=BENCHMARKS_DIR / "results" / "mcscanx_comparison" / "cross_species_v2.gff",
    )
    parser.add_argument(
        "--mcscanx-collinearity",
        type=Path,
        default=BENCHMARKS_DIR
        / "results"
        / "mcscanx_comparison"
        / "cross_species_v2.collinearity",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=BENCHMARKS_DIR / "evaluation" / "anchor_density_summary.csv",
    )
    args = parser.parse_args()

    elsa_blocks = pd.read_csv(args.elsa_blocks)
    elsa_span = (elsa_blocks["query_end"] - elsa_blocks["query_start"] + 1).combine(
        (elsa_blocks["target_end"] - elsa_blocks["target_start"] + 1), max
    )
    elsa_density = elsa_blocks["n_anchors"] / elsa_span

    gff = parse_mcscanx_gff(args.mcscanx_gff)
    gene_idx = dict(zip(gff["gene_id"], gff["gene_idx"]))

    mc_blocks = parse_mcscanx_collinearity(args.mcscanx_collinearity)
    mc_density = []
    for block in mc_blocks:
        if not block["gene_pairs"]:
            continue
        idx_a = [gene_idx.get(a) for a, _ in block["gene_pairs"] if a in gene_idx]
        idx_b = [gene_idx.get(b) for _, b in block["gene_pairs"] if b in gene_idx]
        if not idx_a or not idx_b:
            continue
        span_a = max(idx_a) - min(idx_a) + 1
        span_b = max(idx_b) - min(idx_b) + 1
        span = max(span_a, span_b)
        mc_density.append(len(block["gene_pairs"]) / span)

    df = pd.DataFrame(
        {
            "elsa_anchor_density": pd.Series(elsa_density),
            "mcscanx_anchor_density": pd.Series(mc_density),
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
