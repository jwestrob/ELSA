#!/usr/bin/env python3
"""Sample anchor pairs and compare embedding cosine vs sequence identity."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
import re

import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio.Align import PairwiseAligner

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


def build_annotation_index(gff_dir: Path, samples: list[str]) -> tuple[dict, dict, dict]:
    exact: dict[tuple[str, str, int, int], str] = {}
    by_start: dict[tuple[str, str, int], str] = {}
    by_end: dict[tuple[str, str, int], str] = {}

    for sample in samples:
        gff = gff_dir / f"{sample}.gff"
        if not gff.exists():
            continue
        with open(gff) as f:
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


def load_sequences(proteins_dir: Path, sample_ids: list[str]) -> dict[str, str]:
    sequences: dict[str, str] = {}
    for sample in sample_ids:
        fasta = proteins_dir / f"{sample}.faa"
        if not fasta.exists():
            continue
        for record in SeqIO.parse(fasta, "fasta"):
            sequences[record.id] = str(record.seq)
    return sequences


def sample_pairs(
    blocks: pd.DataFrame,
    gene_lists: dict[tuple[str, str], list[str]],
    rng: random.Random,
    target_n: int,
) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    blocks = blocks.sample(frac=1, random_state=rng.randint(0, 1_000_000))
    for _, block in blocks.iterrows():
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
        if n_pairs <= 0:
            continue
        picks = min(3, n_pairs)
        for _ in range(picks):
            offset = rng.randrange(n_pairs)
            q_idx = q_indices[offset]
            t_idx = t_indices[offset]
            if q_idx >= len(q_list) or t_idx >= len(t_list):
                continue
            pairs.append((q_list[q_idx], t_list[t_idx]))
            if len(pairs) >= target_n:
                return pairs
    return pairs


def sequence_identity(seq_a: str, seq_b: str, aligner: PairwiseAligner) -> float:
    alignment = aligner.align(seq_a, seq_b)[0]
    matches = 0
    for (a_start, a_end), (b_start, b_end) in zip(alignment.aligned[0], alignment.aligned[1]):
        a_seg = seq_a[a_start:a_end]
        b_seg = seq_b[b_start:b_end]
        matches += sum(1 for a, b in zip(a_seg, b_seg) if a == b)
    aligned_length = max(alignment.shape[1], 1)
    return matches / aligned_length


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
        "--annotations",
        type=Path,
        default=BENCHMARKS_DIR / "data" / "enterobacteriaceae" / "annotations",
    )
    parser.add_argument(
        "--proteins",
        type=Path,
        default=BENCHMARKS_DIR / "data" / "enterobacteriaceae" / "proteins",
    )
    parser.add_argument(
        "--samples",
        type=Path,
        default=BENCHMARKS_DIR / "data" / "enterobacteriaceae" / "samples.tsv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=BENCHMARKS_DIR / "evaluation" / "cosine_vs_identity.csv",
    )
    parser.add_argument("--n-pairs", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    samples = pd.read_csv(args.samples, sep="\t")
    sample_ids = samples["sample_id"].tolist()
    species_map = dict(zip(samples["sample_id"], samples["species"]))

    print("Loading orthogroups...")
    protein_to_og = load_orthogroups(args.orthogroups)

    print("Building annotation index...")
    exact, by_start, by_end = build_annotation_index(args.annotations, sample_ids)

    print("Loading gene embeddings...")
    genes_df = pd.read_parquet(args.genes_parquet)
    emb_cols = [c for c in genes_df.columns if c.startswith("emb_")]
    embeddings = genes_df[emb_cols].to_numpy(dtype=np.float32)
    gene_ids = genes_df["gene_id"].tolist()
    gene_meta = genes_df[["sample_id", "contig_id", "start", "end"]]

    gene_to_vec = {gid: embeddings[i] for i, gid in enumerate(gene_ids)}
    gene_to_meta = {gid: tuple(gene_meta.iloc[i]) for i, gid in enumerate(gene_ids)}

    print("Mapping genes to protein IDs...")
    gene_to_protein: dict[str, str] = {}
    for gid, (sample, contig, start, end) in gene_to_meta.items():
        protein_id = map_to_protein(sample, contig, int(start), int(end), exact, by_start, by_end)
        if protein_id:
            gene_to_protein[gid] = protein_id

    gene_to_og = {
        gid: protein_to_og[prot]
        for gid, prot in gene_to_protein.items()
        if prot in protein_to_og
    }

    print("Loading protein sequences...")
    sequences = load_sequences(args.proteins, sample_ids)

    print("Sampling gene pairs...")
    blocks = pd.read_csv(args.elsa_blocks)
    gene_lists = build_gene_lists(genes_df[["sample_id", "contig_id", "gene_id", "start"]])
    rng = random.Random(args.seed)
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 1
    aligner.mismatch_score = 0
    aligner.open_gap_score = -1
    aligner.extend_gap_score = -0.5

    rows = []
    seen = set()
    target = args.n_pairs
    oversample = max(args.n_pairs * 20, args.n_pairs)
    attempts = 0

    while len(rows) < target and attempts < 5:
        candidate_pairs = sample_pairs(blocks, gene_lists, rng, oversample)
        for idx, (q_gene, t_gene) in enumerate(candidate_pairs):
            if len(rows) >= target:
                break
            key = (q_gene, t_gene)
            if key in seen:
                continue
            seen.add(key)
            if q_gene not in gene_to_vec or t_gene not in gene_to_vec:
                continue
            q_prot = gene_to_protein.get(q_gene)
            t_prot = gene_to_protein.get(t_gene)
            if not q_prot or not t_prot:
                continue
            q_seq = sequences.get(q_prot)
            t_seq = sequences.get(t_prot)
            if not q_seq or not t_seq:
                continue

            cosine = float(np.dot(gene_to_vec[q_gene], gene_to_vec[t_gene]))
            if cosine > 1.0:
                cosine = 1.0
            elif cosine < -1.0:
                cosine = -1.0
            identity = sequence_identity(q_seq, t_seq, aligner)
            q_sample = gene_to_meta[q_gene][0]
            t_sample = gene_to_meta[t_gene][0]
            rows.append(
                {
                    "query_gene_id": q_gene,
                    "target_gene_id": t_gene,
                    "query_protein_id": q_prot,
                    "target_protein_id": t_prot,
                    "query_genome": q_sample,
                    "target_genome": t_sample,
                    "species_pair": "-".join(
                        sorted([species_map[q_sample], species_map[t_sample]])
                    ),
                    "cosine_similarity": cosine,
                    "sequence_identity": identity,
                    "same_orthogroup": int(
                        gene_to_og.get(q_gene) == gene_to_og.get(t_gene)
                    ),
                }
            )
            if (idx + 1) % 500 == 0:
                print(f"  processed {idx+1} candidate pairs, kept {len(rows)}")
        attempts += 1
        oversample *= 2

    if len(rows) < target:
        print(f"Warning: only collected {len(rows)} pairs (target {target}).")

    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"Saved: {output} ({len(rows)} pairs)")


if __name__ == "__main__":
    main()
