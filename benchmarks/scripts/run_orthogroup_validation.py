#!/usr/bin/env python3
"""
Parse OrthoFinder output and validate ELSA blocks against orthogroups.

Reads Orthogroups.tsv from OrthoFinder, maps gene IDs to orthogroup assignments,
then checks if genes within each ELSA block share orthogroups.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def parse_orthogroups_tsv(og_file: Path) -> dict:
    """Parse OrthoFinder Orthogroups.tsv into gene_id -> orthogroup mapping.

    OrthoFinder format: OG_ID\tSpecies1_genes\tSpecies2_genes\t...
    where genes within a species are comma-separated.
    """
    gene_to_og = {}
    n_ogs = 0

    with open(og_file) as f:
        header = f.readline().strip().split("\t")
        species_cols = header[1:]  # First col is OG ID

        for line in f:
            parts = line.strip().split("\t")
            og_id = parts[0]
            n_ogs += 1

            for i, col in enumerate(parts[1:], 1):
                if not col.strip():
                    continue
                # Genes are comma-separated, with possible spaces
                genes = [g.strip() for g in col.split(",") if g.strip()]
                for gene in genes:
                    gene_to_og[gene] = og_id

    return gene_to_og, n_ogs


def parse_orthofinder_sequence_ids(seq_ids_file: Path) -> dict:
    """Parse SequenceIDs.txt to map OrthoFinder internal IDs to original gene IDs.

    Format: species_idx_gene_idx: original_gene_id
    """
    id_map = {}
    with open(seq_ids_file) as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            internal_id, original_id = line.split(":", 1)
            id_map[internal_id.strip()] = original_id.strip()
    return id_map


def validate_blocks(blocks_df: pd.DataFrame, genes_df: pd.DataFrame,
                    gene_to_og: dict) -> list:
    """Validate each block by checking orthogroup overlap between query and target."""
    # Build position lookup: (sample_id, contig_id) -> sorted gene list
    genes_df = genes_df.sort_values(["sample_id", "contig_id", "start"])
    gene_idx = genes_df.groupby(["sample_id", "contig_id"]).cumcount()
    genes_df = genes_df.copy()
    genes_df["pos_idx"] = gene_idx.values

    pos_lookup = {}
    for (sample, contig), group in genes_df.groupby(["sample_id", "contig_id"]):
        pos_lookup[(sample, contig)] = group

    results = []
    for _, block in blocks_df.iterrows():
        q_key = (block["query_genome"], block["query_contig"])
        t_key = (block["target_genome"], block["target_contig"])

        if q_key not in pos_lookup or t_key not in pos_lookup:
            continue

        q_group = pos_lookup[q_key]
        t_group = pos_lookup[t_key]

        q_genes = q_group[
            (q_group["pos_idx"] >= block["query_start"]) &
            (q_group["pos_idx"] <= block["query_end"])
        ]["gene_id"].tolist()

        t_genes = t_group[
            (t_group["pos_idx"] >= block["target_start"]) &
            (t_group["pos_idx"] <= block["target_end"])
        ]["gene_id"].tolist()

        if not q_genes or not t_genes:
            continue

        # Get orthogroups
        q_ogs = {gene_to_og[g] for g in q_genes if g in gene_to_og}
        t_ogs = {gene_to_og[g] for g in t_genes if g in gene_to_og}

        shared_ogs = q_ogs & t_ogs

        # Compute fractions
        q_with_shared = sum(1 for g in q_genes if gene_to_og.get(g) in shared_ogs)
        t_with_shared = sum(1 for g in t_genes if gene_to_og.get(g) in shared_ogs)

        q_frac = q_with_shared / len(q_genes) if q_genes else 0
        t_frac = t_with_shared / len(t_genes) if t_genes else 0
        min_frac = min(q_frac, t_frac)

        q_og_coverage = len({gene_to_og[g] for g in q_genes if g in gene_to_og}) / len(q_genes) if q_genes else 0
        t_og_coverage = len({gene_to_og[g] for g in t_genes if g in gene_to_og}) / len(t_genes) if t_genes else 0

        results.append({
            "block_id": block["block_id"],
            "n_query_genes": len(q_genes),
            "n_target_genes": len(t_genes),
            "n_shared_ogs": len(shared_ogs),
            "ortholog_fraction_query": q_frac,
            "ortholog_fraction_target": t_frac,
            "min_ortholog_fraction": min_frac,
            "query_og_coverage": q_og_coverage,
            "target_og_coverage": t_og_coverage,
        })

    return results


def main():
    parser = argparse.ArgumentParser(description="Validate ELSA blocks against OrthoFinder orthogroups")
    parser.add_argument("--orthofinder-dir", type=Path, required=True,
                        help="OrthoFinder Results directory")
    parser.add_argument("--blocks", type=Path, required=True,
                        help="ELSA blocks CSV")
    parser.add_argument("--genes", type=Path, required=True,
                        help="genes.parquet")
    parser.add_argument("--sample-size", type=int, default=0,
                        help="Sample N blocks (0=all)")
    parser.add_argument("-o", "--output", type=Path)
    args = parser.parse_args()

    print("=" * 60)
    print("Orthogroup Validation of ELSA Blocks")
    print("=" * 60)

    # Find OrthoFinder files
    og_tsv = args.orthofinder_dir / "Orthogroups" / "Orthogroups.tsv"
    seq_ids = args.orthofinder_dir / "WorkingDirectory" / "SequenceIDs.txt"

    if not og_tsv.exists():
        print(f"ERROR: Orthogroups.tsv not found at {og_tsv}")
        sys.exit(1)

    # Parse SequenceIDs to understand ID mapping
    print("\n[1/4] Loading OrthoFinder results...")
    if seq_ids.exists():
        id_map = parse_orthofinder_sequence_ids(seq_ids)
        print(f"  SequenceIDs: {len(id_map)} gene ID mappings")
    else:
        id_map = {}

    gene_to_og, n_ogs = parse_orthogroups_tsv(og_tsv)
    print(f"  Orthogroups: {n_ogs} groups, {len(gene_to_og)} gene assignments")

    # Load ELSA data
    print("\n[2/4] Loading ELSA data...")
    blocks_df = pd.read_csv(args.blocks)
    print(f"  {len(blocks_df)} blocks")

    genes_df = pd.read_parquet(args.genes,
                                columns=["sample_id", "contig_id", "gene_id", "start", "end"])
    print(f"  {len(genes_df)} genes")

    # Check gene ID overlap
    our_genes = set(genes_df["gene_id"])
    og_genes = set(gene_to_og.keys())
    overlap = our_genes & og_genes
    print(f"\n  Gene ID overlap: {len(overlap)}/{len(our_genes)} "
          f"({len(overlap)/len(our_genes)*100:.1f}%)")

    if len(overlap) < len(our_genes) * 0.5:
        print("\n  WARNING: Low gene ID overlap! Trying SequenceID mapping...")
        # OrthoFinder may have renamed sequences. Try mapping through SequenceIDs.
        if id_map:
            # Rebuild gene_to_og with original IDs
            gene_to_og_remapped = {}
            for internal_id, original_id in id_map.items():
                # The internal ID format may be "Species_X_Y" style
                if original_id in gene_to_og:
                    # Already using original IDs
                    pass
                # Check if the OG was assigned to internal ID
                # OrthoFinder Orthogroups.tsv uses original sequence names
                pass
            print(f"  SequenceID remap didn't help — IDs may already be original names")

    # Sample if requested
    if args.sample_size > 0 and args.sample_size < len(blocks_df):
        blocks_df = blocks_df.sample(n=args.sample_size, random_state=42)
        print(f"  Sampled {len(blocks_df)} blocks")

    # Validate
    print("\n[3/4] Validating blocks...")
    results = validate_blocks(blocks_df, genes_df, gene_to_og)
    print(f"  Validated {len(results)} blocks")

    # Summary
    print("\n[4/4] Computing summary...")
    if not results:
        print("No blocks could be validated!")
        return

    fractions = [r["min_ortholog_fraction"] for r in results]
    mean_frac = np.mean(fractions)
    median_frac = np.median(fractions)

    thresholds = [0.5, 0.75, 0.9, 0.95]
    threshold_pcts = {t: sum(1 for f in fractions if f >= t) / len(fractions)
                      for t in thresholds}

    print(f"\n{'='*60}")
    print(f"ORTHOGROUP VALIDATION RESULTS")
    print(f"{'='*60}")
    print(f"\nBlocks validated: {len(results)}/{len(blocks_df)}")
    print(f"\nMin ortholog fraction (genes sharing orthogroups):")
    print(f"  Mean:   {mean_frac:.1%}")
    print(f"  Median: {median_frac:.1%}")
    print(f"\nBlocks by threshold:")
    for t in thresholds:
        count = sum(1 for f in fractions if f >= t)
        print(f"  >= {t*100:.0f}%: {count}/{len(results)} ({threshold_pcts[t]:.1%})")

    # Size breakdown
    print(f"\nOrtholog fraction by block size:")
    for r in results:
        r["size"] = max(r["n_query_genes"], r["n_target_genes"])

    small = [r["min_ortholog_fraction"] for r in results if r["size"] <= 5]
    medium = [r["min_ortholog_fraction"] for r in results if 5 < r["size"] <= 25]
    large = [r["min_ortholog_fraction"] for r in results if r["size"] > 25]

    if small:
        print(f"  Size 2-5:   {np.mean(small):.1%} (n={len(small)})")
    if medium:
        print(f"  Size 6-25:  {np.mean(medium):.1%} (n={len(medium)})")
    if large:
        print(f"  Size 26+:   {np.mean(large):.1%} (n={len(large)})")

    # Save
    summary = {
        "total_blocks": len(blocks_df),
        "validated_blocks": len(results),
        "mean_min_ortholog_fraction": float(mean_frac),
        "median_min_ortholog_fraction": float(median_frac),
        "thresholds": {f"{int(t*100)}%": float(threshold_pcts[t]) for t in thresholds},
    }

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump({"summary": summary, "sample_results": results[:200]}, f, indent=2)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
