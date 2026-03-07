#!/usr/bin/env python3
"""
Analyze gene correspondence precision for ELSA vs MCScanX.

For each tool, measure how precisely genes in syntenic blocks actually
correspond between genomes:
- ELSA: Uses embedding similarity for gene matching
- MCScanX: Uses BLAST hit relationships

Metrics:
1. For MCScanX: What fraction of gene pairs in blocks are direct orthologs?
2. For ELSA: How do gene positions align (since ELSA uses index ranges)?
3. Comparison using OrthoFinder orthogroup data
"""

import argparse
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).parent
BENCHMARKS_DIR = SCRIPT_DIR.parent


def parse_chrom(chrom: str) -> tuple:
    """Parse genome_contig into (genome, contig)."""
    m = re.match(r'(GCF_\d+\.\d+)_(.+)', chrom)
    if m:
        return m.group(1), m.group(2)
    return None, None


def build_gene_index(gff_path: Path) -> tuple:
    """
    Build two mappings:
    1. internal_id -> (genome, contig, gene_idx)
    2. (genome, contig, gene_idx) -> internal_id
    """
    chrom_genes = defaultdict(list)

    with open(gff_path) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 4:
                chrom = parts[0]
                internal_id = parts[1]
                start = int(parts[2])
                chrom_genes[chrom].append((start, internal_id))

    id_to_idx = {}
    idx_to_id = {}

    for chrom, genes in chrom_genes.items():
        genes.sort(key=lambda x: x[0])
        genome, contig = parse_chrom(chrom)
        if genome is None:
            continue
        for gene_idx, (start, internal_id) in enumerate(genes):
            id_to_idx[internal_id] = (genome, contig, gene_idx)
            idx_to_id[(genome, contig, gene_idx)] = internal_id

    return id_to_idx, idx_to_id


def load_sequence_ids(og_dir: Path) -> dict:
    """Load OrthoFinder internal ID -> protein accession mapping from SequenceIDs.txt.

    This is needed because MCScanX collinearity output uses OrthoFinder internal
    IDs (e.g. '0_912'), while Orthogroups.tsv uses protein accessions (e.g.
    'NP_459289.1' or 'NZ_CP007265.1_1114').
    """
    seq_ids_file = og_dir / 'WorkingDirectory' / 'SequenceIDs.txt'
    if not seq_ids_file.exists():
        print(f"  WARNING: SequenceIDs.txt not found at {seq_ids_file}")
        return {}

    mapping = {}
    with open(seq_ids_file) as f:
        for line in f:
            # Format: "0_64: NP_446529.1 putative viral protein [...]"
            parts = line.strip().split(': ', 1)
            if len(parts) == 2:
                internal_id = parts[0]
                protein_id = parts[1].split()[0]  # First token is the accession
                mapping[internal_id] = protein_id

    return mapping


def load_orthogroups(og_dir: Path) -> dict:
    """Load orthogroup assignments from OrthoFinder results."""
    og_file = og_dir / 'Orthogroups' / 'Orthogroups.tsv'
    if not og_file.exists():
        return {}

    gene_to_og = {}
    with open(og_file) as f:
        header = f.readline().strip().split('\t')
        for line in f:
            parts = line.strip().split('\t')
            og_id = parts[0]
            for i, cell in enumerate(parts[1:], 1):
                if cell:
                    # Each cell may contain multiple genes separated by comma/space
                    genes = [g.strip() for g in cell.replace(',', ' ').split() if g.strip()]
                    for gene in genes:
                        gene_to_og[gene] = og_id

    return gene_to_og


def parse_mcscanx_collinearity(coll_path: Path, id_to_idx: dict) -> list:
    """Parse MCScanX collinearity file to extract gene pair lists."""
    blocks = []
    current_block = None

    with open(coll_path) as f:
        for line in f:
            line = line.strip()

            if line.startswith('## Alignment'):
                if current_block and current_block['gene_pairs']:
                    blocks.append(current_block)

                parts = line.split()
                block_id = int(parts[2].rstrip(':'))

                chrom_info = None
                for p in parts:
                    if '&' in p:
                        chrom_info = p
                        break

                if chrom_info:
                    chrom_a, chrom_b = chrom_info.split('&')
                    genome_a, contig_a = parse_chrom(chrom_a)
                    genome_b, contig_b = parse_chrom(chrom_b)

                    current_block = {
                        'block_id': block_id,
                        'genome_a': genome_a,
                        'genome_b': genome_b,
                        'contig_a': contig_a,
                        'contig_b': contig_b,
                        'gene_pairs': [],
                    }
                else:
                    current_block = None

            elif current_block and line and not line.startswith('#'):
                # MCScanX collinearity lines are tab-delimited:
                # "  0-  0:\t0_912\t0_1022\t 2e-200"
                # Using tab-split gives: ['  0-  0:', '0_912', '0_1022', ' 2e-200']
                tab_parts = line.split('\t')
                if len(tab_parts) >= 3:
                    gene_a = tab_parts[1].strip()
                    gene_b = tab_parts[2].strip()
                    current_block['gene_pairs'].append((gene_a, gene_b))

    if current_block and current_block['gene_pairs']:
        blocks.append(current_block)

    return blocks


def analyze_mcscanx_correspondence(blocks: list, gene_to_og: dict,
                                    internal_to_protein: dict = None) -> pd.DataFrame:
    """
    For each MCScanX block, check if gene pairs are actual orthologs.

    MCScanX collinearity output uses OrthoFinder internal IDs (e.g. '0_912').
    If internal_to_protein is provided, these are translated to protein accessions
    before looking up orthogroups in gene_to_og.
    """
    results = []
    n_translated = 0
    n_failed_translate = 0

    for block in blocks:
        pairs = block['gene_pairs']
        if not pairs:
            continue

        # Check ortholog status of each pair
        ortholog_count = 0
        has_og_count = 0

        for gene_a, gene_b in pairs:
            # Translate internal IDs to protein accessions if mapping available
            if internal_to_protein:
                prot_a = internal_to_protein.get(gene_a)
                prot_b = internal_to_protein.get(gene_b)
                if prot_a is None or prot_b is None:
                    n_failed_translate += 1
                    continue
                n_translated += 1
                og_a = gene_to_og.get(prot_a)
                og_b = gene_to_og.get(prot_b)
            else:
                og_a = gene_to_og.get(gene_a)
                og_b = gene_to_og.get(gene_b)

            if og_a is not None and og_b is not None:
                has_og_count += 1
                if og_a == og_b:
                    ortholog_count += 1

        # Calculate metrics
        n_pairs = len(pairs)
        ortholog_rate = ortholog_count / n_pairs if n_pairs > 0 else 0
        og_coverage = has_og_count / n_pairs if n_pairs > 0 else 0
        ortholog_rate_with_og = ortholog_count / has_og_count if has_og_count > 0 else 0

        results.append({
            'block_id': block['block_id'],
            'genome_a': block['genome_a'],
            'genome_b': block['genome_b'],
            'n_pairs': n_pairs,
            'ortholog_count': ortholog_count,
            'has_og_count': has_og_count,
            'ortholog_rate': ortholog_rate,
            'og_coverage': og_coverage,
            'ortholog_rate_with_og': ortholog_rate_with_og,
        })

    if internal_to_protein:
        print(f"  ID translation: {n_translated:,} translated, {n_failed_translate:,} failed")

    return pd.DataFrame(results)


def build_elsa_gene_to_og(genes_df: pd.DataFrame, gene_to_og: dict,
                          annotation_dir: Path, protein_dir: Path) -> dict:
    """Map ELSA Prodigal gene_ids to orthogroups via coordinate matching.

    ELSA gene_ids are Prodigal-format (e.g. 'GCF_000006945.2_NC_003197.2_1'),
    while gene_to_og is keyed by protein accessions (e.g. 'WP_095033700.1').
    Bridge them using GFF/FASTA coordinate lookup.
    """
    # Build coordinate index from GFF + Prodigal FASTA
    exact: dict = {}
    by_start: dict = {}

    samples = genes_df['sample_id'].unique()
    gff_indexed = set()

    for sample in samples:
        gff = annotation_dir / f"{sample}.gff"
        if gff.exists():
            try:
                with open(gff) as f:
                    first_line = f.readline()
                    if first_line.startswith("XSym"):
                        pass  # Broken macOS symlink
                    else:
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
                            protein_id = None
                            for attr in parts[8].split(";"):
                                if attr.startswith("protein_id="):
                                    protein_id = attr.split("=", 1)[1]
                                    break
                            if protein_id:
                                exact[(sample, contig, start, end)] = protein_id
                                by_start.setdefault((sample, contig, start), protein_id)
                        gff_indexed.add(sample)
            except (OSError, UnicodeDecodeError):
                pass

        # Prodigal FASTA fallback
        if sample not in gff_indexed and protein_dir.exists():
            faa = protein_dir / f"{sample}.faa"
            if faa.exists():
                try:
                    with open(faa) as f:
                        for line in f:
                            if not line.startswith(">"):
                                continue
                            header_parts = line[1:].strip().split(" # ")
                            if len(header_parts) < 4:
                                continue
                            protein_id = header_parts[0]
                            start = int(header_parts[1])
                            end = int(header_parts[2])
                            contig_parts = protein_id.rsplit("_", 1)
                            contig = contig_parts[0] if len(contig_parts) > 1 else protein_id
                            exact[(sample, contig, start, end)] = protein_id
                            by_start.setdefault((sample, contig, start), protein_id)
                    gff_indexed.add(sample)
                except (OSError, UnicodeDecodeError):
                    pass

    # Map each ELSA gene to its orthogroup
    mapped_og: dict = {}
    mapped = 0
    for _, row in genes_df.iterrows():
        sid, cid, gid = row['sample_id'], row['contig_id'], row['gene_id']
        start, end = row['start'], row['end']
        protein_id = exact.get((sid, cid, start, end)) or by_start.get((sid, cid, start))
        if protein_id and protein_id in gene_to_og:
            mapped_og[gid] = gene_to_og[protein_id]
            mapped += 1

    print(f"    Mapped {mapped:,}/{len(genes_df):,} ELSA genes to orthogroups")
    return mapped_og


def analyze_elsa_correspondence(elsa_blocks: pd.DataFrame, genes_df: pd.DataFrame,
                                 gene_to_og: dict, sample_size: int = 5000) -> pd.DataFrame:
    """
    For each ELSA block, get genes in the index range and check ortholog overlap.

    Since ELSA blocks specify gene index ranges (not explicit pairs), we check
    if genes at corresponding positions share orthogroups.
    """
    results = []

    # Sample if dataset is too large
    if len(elsa_blocks) > sample_size:
        print(f"    Sampling {sample_size} blocks from {len(elsa_blocks)} for efficiency...")
        elsa_blocks = elsa_blocks.sample(n=sample_size, random_state=42)

    # Pre-compute gene indices for each sample/contig combination
    # (genes are sorted by start position to get their index)
    print("    Pre-computing gene indices...")
    genes_df = genes_df.copy()
    genes_df = genes_df.sort_values(['sample_id', 'contig_id', 'start'])
    genes_df['gene_idx'] = genes_df.groupby(['sample_id', 'contig_id']).cumcount()

    # Create lookup dictionaries for faster access
    genes_by_sample_contig = genes_df.groupby(['sample_id', 'contig_id'])

    n_blocks = len(elsa_blocks)
    print(f"    Analyzing {n_blocks} blocks...")

    for i, (_, block) in enumerate(elsa_blocks.iterrows()):
        query_genome = block['query_genome']
        target_genome = block['target_genome']
        query_contig = block['query_contig']
        target_contig = block['target_contig']

        query_start = int(block['query_start'])
        query_end = int(block['query_end'])
        target_start = int(block['target_start'])
        target_end = int(block['target_end'])

        # Get genes in query range
        query_mask = (
            (genes_df['sample_id'] == query_genome) &
            (genes_df['contig_id'] == query_contig)
        )
        query_genes = genes_df[query_mask].sort_values('gene_idx')

        target_mask = (
            (genes_df['sample_id'] == target_genome) &
            (genes_df['contig_id'] == target_contig)
        )
        target_genes = genes_df[target_mask].sort_values('gene_idx')

        # Filter to block range
        query_in_block = query_genes[
            (query_genes['gene_idx'] >= query_start) &
            (query_genes['gene_idx'] <= query_end)
        ]
        target_in_block = target_genes[
            (target_genes['gene_idx'] >= target_start) &
            (target_genes['gene_idx'] <= target_end)
        ]

        if len(query_in_block) == 0 or len(target_in_block) == 0:
            continue

        # Get orthogroups for each region
        query_ogs = set()
        target_ogs = set()
        query_with_og = 0
        target_with_og = 0

        for gene_id in query_in_block['gene_id']:
            og = gene_to_og.get(gene_id)
            if og:
                query_ogs.add(og)
                query_with_og += 1

        for gene_id in target_in_block['gene_id']:
            og = gene_to_og.get(gene_id)
            if og:
                target_ogs.add(og)
                target_with_og += 1

        # Shared orthogroups
        shared_ogs = query_ogs & target_ogs

        # Count genes in shared orthogroups
        query_in_shared = sum(1 for g in query_in_block['gene_id']
                              if gene_to_og.get(g) in shared_ogs)
        target_in_shared = sum(1 for g in target_in_block['gene_id']
                               if gene_to_og.get(g) in shared_ogs)

        n_query = len(query_in_block)
        n_target = len(target_in_block)

        if (i + 1) % 500 == 0:
            print(f"    Processed {i+1}/{n_blocks} blocks...")

        results.append({
            'block_id': block['block_id'],
            'genome_a': query_genome,
            'genome_b': target_genome,
            'n_query_genes': n_query,
            'n_target_genes': n_target,
            'n_shared_ogs': len(shared_ogs),
            'query_in_shared': query_in_shared,
            'target_in_shared': target_in_shared,
            'query_shared_rate': query_in_shared / n_query if n_query > 0 else 0,
            'target_shared_rate': target_in_shared / n_target if n_target > 0 else 0,
            'min_shared_rate': min(query_in_shared / n_query if n_query > 0 else 0,
                                   target_in_shared / n_target if n_target > 0 else 0),
            'query_og_coverage': query_with_og / n_query if n_query > 0 else 0,
            'target_og_coverage': target_with_og / n_target if n_target > 0 else 0,
        })

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mcscanx-gff',
                        default=BENCHMARKS_DIR / 'results' / 'mcscanx_comparison' / 'cross_species_v2.gff')
    parser.add_argument('--mcscanx-collinearity',
                        default=BENCHMARKS_DIR / 'results' / 'mcscanx_comparison' / 'cross_species_v2.collinearity')
    parser.add_argument('--elsa-blocks',
                        default=BENCHMARKS_DIR / 'results' / 'cross_species_chain' / 'micro_chain' / 'micro_chain_blocks.csv')
    parser.add_argument('--genes-parquet',
                        default=BENCHMARKS_DIR / 'elsa_output' / 'cross_species' / 'ingest' / 'genes.parquet')
    parser.add_argument('--orthofinder-results',
                        default=BENCHMARKS_DIR / 'orthofinder' / 'cross_species' / 'Results_Jan31')
    parser.add_argument('--output',
                        default=BENCHMARKS_DIR / 'evaluation' / 'gene_correspondence_precision.md')
    args = parser.parse_args()

    output_path = Path(args.output)

    print("=" * 70)
    print("Gene Correspondence Precision Analysis")
    print("=" * 70)

    # Load OrthoFinder results
    print("\n[1/6] Loading OrthoFinder orthogroups...")
    og_dir = Path(args.orthofinder_results)
    gene_to_og = load_orthogroups(og_dir)
    print(f"  Loaded {len(gene_to_og):,} gene-to-orthogroup mappings")

    if not gene_to_og:
        print("  ERROR: No orthogroup data found!")
        return

    # Load internal ID -> protein accession mapping for MCScanX ID translation
    print("\n[2/6] Loading OrthoFinder SequenceIDs (internal -> protein mapping)...")
    internal_to_protein = load_sequence_ids(og_dir)
    print(f"  Loaded {len(internal_to_protein):,} internal-to-protein mappings")

    # Diagnostic: check how many MCScanX-style IDs would match directly vs via translation
    sample_ids = list(internal_to_protein.keys())[:5]
    sample_proteins = [internal_to_protein[k] for k in sample_ids]
    print(f"  Sample mapping: {dict(zip(sample_ids, sample_proteins))}")
    n_direct_hits = sum(1 for k in list(internal_to_protein.keys())[:100]
                        if k in gene_to_og)
    n_translated_hits = sum(1 for k in list(internal_to_protein.keys())[:100]
                            if internal_to_protein[k] in gene_to_og)
    print(f"  Direct ID lookup (first 100): {n_direct_hits}/100 found in OG map")
    print(f"  Translated ID lookup (first 100): {n_translated_hits}/100 found in OG map")

    # Build gene index for MCScanX
    print("\n[3/6] Building gene index from MCScanX GFF...")
    id_to_idx, idx_to_id = build_gene_index(Path(args.mcscanx_gff))
    print(f"  Indexed {len(id_to_idx):,} genes")

    # Parse MCScanX collinearity
    print("\n[4/6] Parsing MCScanX collinearity...")
    mcscanx_blocks = parse_mcscanx_collinearity(Path(args.mcscanx_collinearity), id_to_idx)
    print(f"  Parsed {len(mcscanx_blocks):,} blocks")

    # Analyze MCScanX correspondence (with ID translation)
    print("\n[5/6] Analyzing MCScanX gene correspondence (with ID translation)...")
    mcscanx_results = analyze_mcscanx_correspondence(
        mcscanx_blocks, gene_to_og, internal_to_protein
    )
    print(f"  Analyzed {len(mcscanx_results):,} blocks")

    # Load and analyze ELSA blocks
    print("\n[6/6] Analyzing ELSA gene correspondence...")
    elsa_blocks = pd.read_csv(args.elsa_blocks)
    genes_df = pd.read_parquet(args.genes_parquet)

    # Map ELSA Prodigal gene_ids to orthogroups via coordinate matching
    annotation_dir = Path(args.mcscanx_gff).parent.parent.parent / 'data' / 'enterobacteriaceae' / 'annotations'
    protein_dir = Path(args.mcscanx_gff).parent.parent.parent / 'data' / 'cross_species' / 'proteins'
    elsa_gene_to_og = build_elsa_gene_to_og(genes_df, gene_to_og, annotation_dir, protein_dir)
    elsa_results = analyze_elsa_correspondence(elsa_blocks, genes_df, elsa_gene_to_og)
    print(f"  Analyzed {len(elsa_results):,} blocks")

    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)

    print("\n### MCScanX Gene Pair Correspondence ###")
    if len(mcscanx_results) > 0:
        print(f"Total blocks: {len(mcscanx_results):,}")
        print(f"Mean ortholog rate: {mcscanx_results['ortholog_rate'].mean():.1%}")
        print(f"Median ortholog rate: {mcscanx_results['ortholog_rate'].median():.1%}")
        print(f"Mean OG coverage: {mcscanx_results['og_coverage'].mean():.1%}")

        # By threshold
        for thresh in [0.5, 0.75, 0.9]:
            n_above = (mcscanx_results['ortholog_rate'] >= thresh).sum()
            print(f"Blocks with ≥{thresh*100:.0f}% ortholog rate: {n_above:,} ({n_above/len(mcscanx_results)*100:.1f}%)")

    print("\n### ELSA Region Correspondence ###")
    if len(elsa_results) > 0:
        print(f"Total blocks: {len(elsa_results):,}")
        print(f"Mean shared orthogroup rate: {elsa_results['min_shared_rate'].mean():.1%}")
        print(f"Median shared orthogroup rate: {elsa_results['min_shared_rate'].median():.1%}")

        # By threshold
        for thresh in [0.5, 0.75, 0.9]:
            n_above = (elsa_results['min_shared_rate'] >= thresh).sum()
            print(f"Blocks with ≥{thresh*100:.0f}% shared rate: {n_above:,} ({n_above/len(elsa_results)*100:.1f}%)")

    # Generate report
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        f.write("# Gene Correspondence Precision Analysis\n\n")
        f.write("## Overview\n\n")
        f.write("This analysis measures how precisely genes in syntenic blocks correspond\n")
        f.write("between genomes, using OrthoFinder orthogroup data as ground truth.\n\n")

        f.write("### Methodology\n\n")
        f.write("**MCScanX**: For each gene pair in a block, check if both genes belong\n")
        f.write("to the same orthogroup (i.e., are true orthologs).\n\n")
        f.write("**ELSA**: For genes within the block's index range in each genome, check\n")
        f.write("what fraction belong to shared orthogroups.\n\n")

        f.write("## Results\n\n")

        f.write("### MCScanX Gene Pair Precision\n\n")
        if len(mcscanx_results) > 0:
            f.write(f"| Metric | Value |\n")
            f.write(f"|--------|-------|\n")
            f.write(f"| Total blocks | {len(mcscanx_results):,} |\n")
            f.write(f"| Mean ortholog rate | {mcscanx_results['ortholog_rate'].mean():.1%} |\n")
            f.write(f"| Median ortholog rate | {mcscanx_results['ortholog_rate'].median():.1%} |\n")
            f.write(f"| Std dev | {mcscanx_results['ortholog_rate'].std():.1%} |\n")
            f.write(f"| Min | {mcscanx_results['ortholog_rate'].min():.1%} |\n")
            f.write(f"| Max | {mcscanx_results['ortholog_rate'].max():.1%} |\n\n")

            f.write("**Blocks by ortholog rate threshold:**\n\n")
            f.write("| Threshold | Count | Percentage |\n")
            f.write("|-----------|-------|------------|\n")
            for thresh in [0.5, 0.75, 0.9, 0.95]:
                n = (mcscanx_results['ortholog_rate'] >= thresh).sum()
                f.write(f"| ≥{thresh*100:.0f}% | {n:,} | {n/len(mcscanx_results)*100:.1f}% |\n")
            f.write("\n")

        f.write("### ELSA Orthogroup Overlap\n\n")
        if len(elsa_results) > 0:
            f.write(f"| Metric | Value |\n")
            f.write(f"|--------|-------|\n")
            f.write(f"| Total blocks | {len(elsa_results):,} |\n")
            f.write(f"| Mean shared OG rate | {elsa_results['min_shared_rate'].mean():.1%} |\n")
            f.write(f"| Median shared OG rate | {elsa_results['min_shared_rate'].median():.1%} |\n")
            f.write(f"| Std dev | {elsa_results['min_shared_rate'].std():.1%} |\n")
            f.write(f"| Mean shared OGs per block | {elsa_results['n_shared_ogs'].mean():.1f} |\n\n")

            f.write("**Blocks by shared OG rate threshold:**\n\n")
            f.write("| Threshold | Count | Percentage |\n")
            f.write("|-----------|-------|------------|\n")
            for thresh in [0.5, 0.75, 0.9, 0.95]:
                n = (elsa_results['min_shared_rate'] >= thresh).sum()
                f.write(f"| ≥{thresh*100:.0f}% | {n:,} | {n/len(elsa_results)*100:.1f}% |\n")
            f.write("\n")

        f.write("## Interpretation\n\n")

        f.write("### Key Difference in Metrics\n\n")
        f.write("**MCScanX ortholog rate** measures: Of the gene pairs explicitly linked\n")
        f.write("by BLAST hits in the collinearity file, what fraction are verified orthologs?\n\n")
        f.write("**ELSA shared OG rate** measures: Of all genes in the block's genomic span,\n")
        f.write("what fraction have orthologs in the corresponding region of the other genome?\n\n")

        f.write("### Why Rates May Differ\n\n")
        f.write("1. **MCScanX is more selective**: Only includes genes with significant BLAST hits.\n")
        f.write("   This tends to inflate ortholog rate (pre-filtered for homology).\n\n")
        f.write("2. **ELSA includes all genes**: Counts every gene in the index range,\n")
        f.write("   including lineage-specific insertions that don't have orthologs.\n\n")
        f.write("3. **Sparse vs dense blocks**: MCScanX blocks can span regions with\n")
        f.write("   intermittent BLAST hits; ELSA's index ranges include all genes.\n\n")

        f.write("### What Matters for Synteny Detection\n\n")
        f.write("For biological interpretation, we care about:\n")
        f.write("1. Do blocks identify regions of shared ancestry? (Both tools do this)\n")
        f.write("2. Are gene correspondences accurate? (MCScanX pairs are explicit; ELSA infers)\n")
        f.write("3. Are block boundaries meaningful? (ELSA may be more precise)\n")

    print(f"\nReport saved to: {output_path}")

    # Save detailed data (region-level analysis, distinct from pair-level CSVs)
    mcscanx_csv = output_path.parent / 'mcscanx_region_correspondence.csv'
    elsa_csv = output_path.parent / 'elsa_region_correspondence.csv'
    mcscanx_results.to_csv(mcscanx_csv, index=False)
    elsa_results.to_csv(elsa_csv, index=False)
    print(f"MCScanX data saved to: {mcscanx_csv}")
    print(f"ELSA data saved to: {elsa_csv}")


if __name__ == '__main__':
    main()
