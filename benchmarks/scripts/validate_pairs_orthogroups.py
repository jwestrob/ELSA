#!/usr/bin/env python3
"""Validate ELSA/MCScanX gene pairs against OrthoFinder orthogroups."""

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


def load_orthogroups(orthogroups_tsv: Path) -> dict[str, str]:
    gene_to_og: dict[str, str] = {}
    og_df = pd.read_csv(orthogroups_tsv, sep="\t")
    for _, row in og_df.iterrows():
        og_id = row["Orthogroup"]
        for col in og_df.columns[1:]:
            cell = row[col]
            if pd.isna(cell):
                continue
            for gene in str(cell).replace(",", " ").split():
                gene_to_og[gene.strip()] = og_id
    return gene_to_og


def load_sequence_ids(og_results_dir: Path) -> dict[str, str]:
    """Load OrthoFinder internal ID -> protein accession mapping.

    MCScanX collinearity uses OrthoFinder internal IDs (e.g. '0_912').
    SequenceIDs.txt maps these to protein accessions (e.g. 'NP_459289.1').
    """
    seq_ids_file = og_results_dir / "WorkingDirectory" / "SequenceIDs.txt"
    if not seq_ids_file.exists():
        return {}

    mapping: dict[str, str] = {}
    with open(seq_ids_file) as f:
        for line in f:
            # Format: "0_64: NP_446529.1 putative viral protein [...]"
            parts = line.strip().split(": ", 1)
            if len(parts) == 2:
                internal_id = parts[0]
                protein_id = parts[1].split()[0]
                mapping[internal_id] = protein_id
    return mapping


def build_annotation_index(
    gff_dir: Path, samples: list[str], protein_dir: Path | None = None,
) -> tuple[dict, dict, dict]:
    """Build coordinate-based annotation index from GFF files and Prodigal FASTAs.

    Tries NCBI GFF first, then falls back to Prodigal FASTA headers for
    genomes without GFF annotations (e.g. E. coli processed with Prodigal).
    """
    exact: dict[tuple[str, str, int, int], str] = {}
    by_start: dict[tuple[str, str, int], str] = {}
    by_end: dict[tuple[str, str, int], str] = {}

    gff_indexed = set()

    for sample in samples:
        gff = gff_dir / f"{sample}.gff"
        if not gff.exists():
            continue
        # Check if it's a real GFF (not a broken macOS symlink)
        try:
            with open(gff) as f:
                first_line = f.readline()
                if first_line.startswith("XSym"):
                    # Broken macOS symlink file
                    continue
                f.seek(0)
                for line in f:
                    if line.startswith("#"):
                        continue
                    parts = line.strip().split("\t")
                    if len(parts) < 9 or parts[2] != "CDS":
                        continue
                    contig = parts[0]
                    start = int(parts[3])
                    end = int(parts[4])
                    attrs = parts[8]
                    protein_id = None
                    for attr in attrs.split(";"):
                        if attr.startswith("protein_id="):
                            protein_id = attr.split("=", 1)[1]
                            break
                    if not protein_id:
                        continue
                    exact[(sample, contig, start, end)] = protein_id
                    by_start.setdefault((sample, contig, start), protein_id)
                    by_end.setdefault((sample, contig, end), protein_id)
            gff_indexed.add(sample)
        except (OSError, UnicodeDecodeError):
            continue

    # Fall back to Prodigal FASTA headers for samples without GFF
    if protein_dir and protein_dir.exists():
        for sample in samples:
            if sample in gff_indexed:
                continue
            faa = protein_dir / f"{sample}.faa"
            if not faa.exists():
                continue
            try:
                with open(faa) as f:
                    for line in f:
                        if not line.startswith(">"):
                            continue
                        # Prodigal format: >contig_gene # start # end # strand # attrs
                        header_parts = line[1:].strip().split(" # ")
                        if len(header_parts) < 4:
                            continue
                        protein_id = header_parts[0]
                        start = int(header_parts[1])
                        end = int(header_parts[2])
                        # Parse contig from protein ID (format: contig_genenum)
                        contig_parts = protein_id.rsplit("_", 1)
                        contig = contig_parts[0] if len(contig_parts) > 1 else protein_id
                        exact[(sample, contig, start, end)] = protein_id
                        by_start.setdefault((sample, contig, start), protein_id)
                        by_end.setdefault((sample, contig, end), protein_id)
                gff_indexed.add(sample)
            except (OSError, UnicodeDecodeError):
                continue

    n_gff = len(gff_indexed)
    n_total = len(samples)
    print(f"  Indexed {n_gff}/{n_total} samples ({len(exact):,} CDS entries)")
    return exact, by_start, by_end


def map_to_protein(
    sample: str,
    contig: str,
    start: int,
    end: int,
    exact: dict,
    by_start: dict,
    by_end: dict,
) -> str | None:
    key = (sample, contig, start, end)
    if key in exact:
        return exact[key]
    key = (sample, contig, start)
    if key in by_start:
        return by_start[key]
    key = (sample, contig, end)
    return by_end.get(key)


def build_gene_lists(genes_df: pd.DataFrame) -> dict[tuple[str, str], list[str]]:
    genes_df = genes_df.sort_values(["sample_id", "contig_id", "start"])
    grouped = genes_df.groupby(["sample_id", "contig_id"])
    return {key: group["gene_id"].tolist() for key, group in grouped}


def parse_mcscanx_gff(gff_path: Path) -> dict[str, tuple[str, str, int, int]]:
    mapping: dict[str, tuple[str, str, int, int]] = {}
    with open(gff_path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 4:
                continue
            chrom, gene_id, start, end = parts[0], parts[1], int(parts[2]), int(parts[3])
            genome, contig = parse_chrom(chrom)
            if genome is None:
                continue
            mapping[gene_id] = (genome, contig, start, end)
    return mapping


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
                # MCScanX collinearity lines are tab-delimited:
                # "  0-  0:\t0_912\t0_1022\t 2e-200"
                tab_parts = line.split("\t")
                if len(tab_parts) >= 3:
                    current["gene_pairs"].append((tab_parts[1].strip(), tab_parts[2].strip()))
    if current and current["gene_pairs"]:
        blocks.append(current)
    return blocks


def evaluate_mcscanx(
    blocks: list[dict],
    mc_gene_to_og: dict[str, str],
) -> pd.DataFrame:
    rows = []
    for block in blocks:
        match = 0
        pairs_with_og = 0
        for gene_a, gene_b in block["gene_pairs"]:
            og_a = mc_gene_to_og.get(gene_a)
            og_b = mc_gene_to_og.get(gene_b)
            if og_a and og_b:
                pairs_with_og += 1
                if og_a == og_b:
                    match += 1
        n_pairs = len(block["gene_pairs"])
        rows.append(
            {
                "block_id": block["block_id"],
                "n_pairs": n_pairs,
                "pairs_with_og": pairs_with_og,
                "match_count": match,
                "match_rate": match / n_pairs if n_pairs else 0.0,
                "match_rate_with_og": match / pairs_with_og if pairs_with_og else 0.0,
                "og_coverage": pairs_with_og / n_pairs if n_pairs else 0.0,
            }
        )
    return pd.DataFrame(rows)


def evaluate_elsa(
    elsa_blocks: pd.DataFrame,
    gene_lists: dict[tuple[str, str], list[str]],
    gene_to_og: dict[str, str],
    sample_limit: int | None = None,
) -> pd.DataFrame:
    if sample_limit and len(elsa_blocks) > sample_limit:
        elsa_blocks = elsa_blocks.sample(n=sample_limit, random_state=42)

    rows = []
    for _, block in elsa_blocks.iterrows():
        q_key = (block["query_genome"], block["query_contig"])
        t_key = (block["target_genome"], block["target_contig"])
        q_list = gene_lists.get(q_key)
        t_list = gene_lists.get(t_key)
        if not q_list or not t_list:
            continue

        q_start, q_end = int(block["query_start"]), int(block["query_end"])
        t_start, t_end = int(block["target_start"]), int(block["target_end"])
        q_indices = list(range(q_start, q_end + 1))
        t_indices = list(range(t_start, t_end + 1))
        if block["orientation"] == -1:
            t_indices = list(reversed(t_indices))

        n_pairs = min(len(q_indices), len(t_indices))
        match = 0
        pairs_with_og = 0
        for q_idx, t_idx in zip(q_indices[:n_pairs], t_indices[:n_pairs]):
            if q_idx >= len(q_list) or t_idx >= len(t_list):
                continue
            og_q = gene_to_og.get(q_list[q_idx])
            og_t = gene_to_og.get(t_list[t_idx])
            if og_q and og_t:
                pairs_with_og += 1
                if og_q == og_t:
                    match += 1

        rows.append(
            {
                "block_id": block["block_id"],
                "n_pairs": n_pairs,
                "pairs_with_og": pairs_with_og,
                "match_count": match,
                "match_rate": match / n_pairs if n_pairs else 0.0,
                "match_rate_with_og": match / pairs_with_og if pairs_with_og else 0.0,
                "og_coverage": pairs_with_og / n_pairs if n_pairs else 0.0,
            }
        )
    return pd.DataFrame(rows)


def summarize(method: str, df: pd.DataFrame) -> dict:
    total_pairs = int(df["n_pairs"].sum())
    total_with_og = int(df["pairs_with_og"].sum())
    total_match = int(df["match_count"].sum())
    return {
        "method": method,
        "n_blocks": len(df),
        "n_pairs": total_pairs,
        "match_rate": total_match / total_pairs if total_pairs else 0.0,
        "match_rate_with_og": total_match / total_with_og if total_with_og else 0.0,
        "og_coverage": total_with_og / total_pairs if total_pairs else 0.0,
    }


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
        "--genes-parquet",
        type=Path,
        default=BENCHMARKS_DIR / "elsa_output" / "cross_species" / "ingest" / "genes.parquet",
    )
    parser.add_argument(
        "--orthogroups",
        type=Path,
        default=BENCHMARKS_DIR
        / "orthofinder"
        / "cross_species"
        / "Results_Jan31"
        / "Orthogroups"
        / "Orthogroups.tsv",
    )
    parser.add_argument(
        "--orthofinder-results",
        type=Path,
        default=BENCHMARKS_DIR / "orthofinder" / "cross_species" / "Results_Jan31",
        help="OrthoFinder results directory (contains WorkingDirectory/SequenceIDs.txt)",
    )
    parser.add_argument(
        "--samples",
        type=Path,
        default=BENCHMARKS_DIR / "data" / "enterobacteriaceae" / "samples.tsv",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=BENCHMARKS_DIR / "data" / "enterobacteriaceae" / "annotations",
    )
    parser.add_argument(
        "--protein-dir",
        type=Path,
        default=BENCHMARKS_DIR / "data" / "cross_species" / "proteins",
        help="Directory with Prodigal .faa files (fallback for genomes without GFF)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BENCHMARKS_DIR / "evaluation",
    )
    parser.add_argument("--elsa-sample-limit", type=int, default=None)
    args = parser.parse_args()

    samples = pd.read_csv(args.samples, sep="\t")
    sample_ids = samples["sample_id"].tolist()

    print("Loading orthogroups...")
    protein_to_og = load_orthogroups(args.orthogroups)
    print(f"  {len(protein_to_og):,} protein->orthogroup mappings")

    print("Building annotation index (GFF + Prodigal FASTA fallback)...")
    exact, by_start, by_end = build_annotation_index(
        args.annotations, sample_ids, protein_dir=args.protein_dir,
    )

    print("Mapping ELSA genes to orthogroups...")
    genes_df = pd.read_parquet(
        args.genes_parquet, columns=["sample_id", "contig_id", "gene_id", "start", "end"]
    )
    gene_to_og: dict[str, str] = {}
    mapped = 0
    for row in genes_df.itertuples(index=False):
        protein_id = map_to_protein(
            row.sample_id, row.contig_id, row.start, row.end, exact, by_start, by_end
        )
        if protein_id and protein_id in protein_to_og:
            gene_to_og[row.gene_id] = protein_to_og[protein_id]
            mapped += 1
    print(f"  Mapped {mapped:,}/{len(genes_df):,} genes to orthogroups")

    gene_lists = build_gene_lists(genes_df)

    print("Evaluating ELSA positional pairs...")
    elsa_blocks = pd.read_csv(args.elsa_blocks)
    elsa_results = evaluate_elsa(
        elsa_blocks, gene_lists, gene_to_og, sample_limit=args.elsa_sample_limit
    )
    print(f"  Evaluated {len(elsa_results):,} blocks")

    # MCScanX: Use SequenceIDs.txt for direct internal-ID-to-protein mapping.
    # This is more reliable than coordinate-based matching because MCScanX uses
    # OrthoFinder internal IDs (e.g. '0_912') which map directly to protein
    # accessions via SequenceIDs.txt.
    print("Loading OrthoFinder SequenceIDs for MCScanX ID translation...")
    internal_to_protein = load_sequence_ids(args.orthofinder_results)
    print(f"  Loaded {len(internal_to_protein):,} internal-to-protein mappings")

    mc_gene_to_og: dict[str, str] = {}
    for internal_id, protein_id in internal_to_protein.items():
        og = protein_to_og.get(protein_id)
        if og:
            mc_gene_to_og[internal_id] = og
    print(f"  Mapped {len(mc_gene_to_og):,}/{len(internal_to_protein):,} MCScanX genes to orthogroups")

    print("Evaluating MCScanX gene pairs...")
    mc_blocks = parse_mcscanx_collinearity(args.mcscanx_collinearity)
    mcscanx_results = evaluate_mcscanx(mc_blocks, mc_gene_to_og)
    print(f"  Evaluated {len(mcscanx_results):,} blocks")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    elsa_csv = output_dir / "elsa_correspondence.csv"
    mc_csv = output_dir / "mcscanx_correspondence.csv"
    elsa_results.to_csv(elsa_csv, index=False)
    mcscanx_results.to_csv(mc_csv, index=False)
    print(f"Saved: {elsa_csv}")
    print(f"Saved: {mc_csv}")

    summary = pd.DataFrame(
        [
            summarize("ELSA", elsa_results),
            summarize("MCScanX", mcscanx_results),
        ]
    )
    summary_path = output_dir / "anchor_orthogroup_precision.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
