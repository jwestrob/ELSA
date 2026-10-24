#!/usr/bin/env python3
"""Adversarial, read-only audit of the March 2026 ELSA benchmarks.

The script treats every canonical input as immutable and writes all derived
artifacts to a separate audit directory.  It reconstructs the operon-locus
cohort from GFF and ELSA coordinates, audits the exact protein sequences used
for embedding and OrthoFinder, recomputes positional block metrics, runs
deterministic null controls, and checks MCScanX identifier compatibility.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd
from Bio import SeqIO


AUDIT_SCHEMA_VERSION = "1.0.0"
STANDARD_AA = frozenset("ACDEFGHIKLMNPQRSTVWY")
EXPECTED_CANONICAL_SHA256 = {
    "operon_definitions": (
        "abc3bf821632e72b0ac34aba7b2de43d3130ce7aa37d83b5f22f49bc063d7397"
    ),
    "pairwise_ground_truth": (
        "1962072bd44b4d0f8a02876cc5d517bfc14e6e21d1ee59fe588e1298d79ba6a0"
    ),
    "ecoli_genes": (
        "7061b3e40ef1edb9c49e7f560bab3a06cb630f2352be6c017a3b07d1be7c67d8"
    ),
    "ecoli_blocks": (
        "1b225ea0419e4e8a1ff26f465ab0019e0cb2f9608a7de1841519a6f2e8f84873"
    ),
    "cross_genes": (
        "86556b741eeaa315599b1964c8259ca275d2e064e5a2764b417aba1a599850b6"
    ),
    "cross_blocks": (
        "d6c43b58050748cf8c53f4d1bd931b836c0c8009660a9dcacf15f0445cfe55b7"
    ),
    "taxonomy_table": (
        "584adb9194d98959e0634a4e6a36c364a82cac8d1981f5ddfa2631e442afde2c"
    ),
    "january_sequence_ids": (
        "f5260f558451945f1ea3162c23863b5a935c6e71ad81761388b37452a8a14d9d"
    ),
    "march_sequence_ids": (
        "add46127c7066bed5738faf13e627521cc11a05f2515c6f98ab1da261cff4b83"
    ),
    "january_mcscanx_gff": (
        "58f8a64be1c0e9c2698dab420bd24b6e1910ca50087ad640e97204dc74766774"
    ),
    "march_mcscanx_gff": (
        "6eeb8391bdb817ed5f442291fea8bca45e2756fec2e7104140b18f10fe8d996f"
    ),
    "mcscanx_collinearity": (
        "63191a8a1e5ae367b77e6cc8b4abe6266c30bc15ff369799b5f7b74730275658"
    ),
}
PRODIGAL_HEADER_RE = re.compile(
    r"^(?P<record>\S+) # (?P<start>\d+) # (?P<end>\d+) # "
    r"(?P<strand>-?\d+) # (?P<attributes>.*)$"
)


@dataclass(frozen=True)
class FastaRecord:
    """A minimal FASTA record."""

    identifier: str
    description: str
    sequence: str


@dataclass(frozen=True)
class ProdigalRecord:
    """A Prodigal translation keyed to its genomic coordinates."""

    sample_id: str
    contig_id: str
    start: int
    end: int
    strand: int
    sequence: str
    terminal_stop: bool
    partial: str | None
    start_type: str | None


def parse_fasta(path: Path) -> Iterator[FastaRecord]:
    """Parse FASTA without normalizing identifiers or sequences."""

    identifier: str | None = None
    description = ""
    sequence: list[str] = []
    with path.open() as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if identifier is not None:
                    yield FastaRecord(identifier, description, "".join(sequence))
                description = line[1:]
                identifier = description.split()[0]
                sequence = []
            else:
                if identifier is None:
                    raise ValueError(f"Sequence before FASTA header in {path}")
                sequence.append(line)
    if identifier is not None:
        yield FastaRecord(identifier, description, "".join(sequence))


def parse_semicolon_attributes(text: str) -> dict[str, str]:
    """Parse semicolon-delimited key-value attributes."""

    parsed: dict[str, str] = {}
    for field in text.split(";"):
        if "=" not in field:
            continue
        key, value = field.split("=", 1)
        parsed[key] = value
    return parsed


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the complete SHA-256 digest for a file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def json_scalar(value: Any) -> Any:
    """Convert NumPy/Pandas scalar values into JSON-native values."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if pd.isna(value):
        return None
    return value


def json_ready(value: Any) -> Any:
    """Recursively convert a result object to stable JSON-compatible values."""

    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return json_scalar(value)


def atomic_write_text(path: Path, text: str) -> None:
    """Write text atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        temp_path = Path(handle.name)
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_path, path)


def atomic_write_json(path: Path, value: Any) -> None:
    """Write deterministic JSON atomically."""

    atomic_write_text(
        path,
        json.dumps(json_ready(value), indent=2, sort_keys=True) + "\n",
    )


def atomic_write_tsv(path: Path, frame: pd.DataFrame) -> None:
    """Write a TSV atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        temp_path = Path(handle.name)
        frame.to_csv(handle, sep="\t", index=False, lineterminator="\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_path, path)


def atomic_write_tsv_gz(path: Path, frame: pd.DataFrame) -> None:
    """Write a reproducible gzip-compressed TSV atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "wb", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as raw_handle:
        temp_path = Path(raw_handle.name)
        with gzip.GzipFile(
            filename="", mode="wb", fileobj=raw_handle, mtime=0
        ) as gzip_handle:
            frame.to_csv(
                gzip_handle,
                sep="\t",
                index=False,
                lineterminator="\n",
            )
        raw_handle.flush()
        os.fsync(raw_handle.fileno())
    os.replace(temp_path, path)


def parse_operon_definitions(path: Path) -> dict[str, dict[str, Any]]:
    """Load the multi-gene operon definitions used by the benchmark."""

    definitions: dict[str, dict[str, Any]] = {}
    with path.open() as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            genes = [
                gene.strip().lower()
                for gene in row["genes"].split(",")
                if gene.strip()
            ]
            if len(genes) < 2:
                continue
            operon_id = row["operon_id"]
            if operon_id in definitions:
                raise ValueError(f"Duplicate operon definition: {operon_id}")
            definitions[operon_id] = {
                "operon_id": operon_id,
                "genes": genes,
                "evidence": row.get("evidence", "unknown"),
            }
    return definitions


def prepare_gene_coordinates(genes_path: Path) -> pd.DataFrame:
    """Load the coordinate contract and add both global and local ranks."""

    columns = [
        "sample_id",
        "contig_id",
        "gene_id",
        "start",
        "end",
        "strand",
        "original_length",
    ]
    genes = pd.read_parquet(genes_path, columns=columns).reset_index(drop=True)
    if genes["gene_id"].duplicated().any():
        examples = genes.loc[genes["gene_id"].duplicated(), "gene_id"].head().tolist()
        raise ValueError(f"Duplicate ELSA gene IDs: {examples}")
    genes["_source_row"] = np.arange(len(genes), dtype=np.int64)
    genes["sample_global_rank"] = genes.groupby(
        "sample_id", sort=False
    ).cumcount()

    sorted_rows = genes.sort_values(
        ["sample_id", "contig_id", "start", "end", "_source_row"],
        kind="mergesort",
    )
    local_ranks = sorted_rows.groupby(
        ["sample_id", "contig_id"], sort=False
    ).cumcount()
    genes["contig_local_rank"] = pd.Series(
        local_ranks.to_numpy(), index=sorted_rows.index
    ).sort_index()
    return genes


def load_prodigal_records(
    proteins_dir: Path, sample_ids: Sequence[str]
) -> tuple[dict[tuple[str, str, int, int, int], ProdigalRecord], dict[str, int]]:
    """Load exact Prodigal translations retained beside the canonical run."""

    records: dict[tuple[str, str, int, int, int], ProdigalRecord] = {}
    total_records = 0
    for sample_id in sample_ids:
        path = proteins_dir / f"{sample_id}.faa"
        if not path.exists():
            raise FileNotFoundError(f"Missing Prodigal FASTA: {path}")
        for record in parse_fasta(path):
            match = PRODIGAL_HEADER_RE.match(record.description)
            if match is None:
                raise ValueError(f"Malformed Prodigal header in {path}: {record.description}")
            record_name = match.group("record")
            if "_" not in record_name:
                raise ValueError(f"Prodigal record lacks numeric suffix: {record_name}")
            contig_id = record_name.rsplit("_", 1)[0]
            attributes = parse_semicolon_attributes(match.group("attributes"))
            sequence = record.sequence
            key = (
                sample_id,
                contig_id,
                int(match.group("start")),
                int(match.group("end")),
                int(match.group("strand")),
            )
            if key in records:
                raise ValueError(f"Duplicate Prodigal coordinate key: {key}")
            records[key] = ProdigalRecord(
                sample_id=sample_id,
                contig_id=contig_id,
                start=key[2],
                end=key[3],
                strand=key[4],
                sequence=sequence.removesuffix("*"),
                terminal_stop=sequence.endswith("*"),
                partial=attributes.get("partial"),
                start_type=attributes.get("start_type"),
            )
            total_records += 1
    return records, {"all_prodigal_records": total_records}


def load_named_sequences(
    directory: Path, sample_ids: Sequence[str]
) -> dict[str, str]:
    """Load per-sample FASTA files into one exact identifier map."""

    sequences: dict[str, str] = {}
    for sample_id in sample_ids:
        path = directory / f"{sample_id}.faa"
        if not path.exists():
            raise FileNotFoundError(f"Missing FASTA: {path}")
        for record in parse_fasta(path):
            if record.identifier in sequences:
                raise ValueError(f"Duplicate FASTA identifier: {record.identifier}")
            sequences[record.identifier] = record.sequence.removesuffix("*")
    return sequences


def load_genome_sequences(
    genomes_dir: Path, sample_ids: Sequence[str]
) -> dict[str, dict[str, Any]]:
    """Load nucleotide records for the supplied samples."""

    genomes: dict[str, dict[str, Any]] = {}
    for sample_id in sample_ids:
        path = genomes_dir / f"{sample_id}.fna"
        if not path.exists():
            raise FileNotFoundError(f"Missing genome FASTA: {path}")
        sample_records = {record.id: record.seq for record in SeqIO.parse(path, "fasta")}
        if not sample_records:
            raise ValueError(f"No nucleotide records in {path}")
        genomes[sample_id] = sample_records
    return genomes


def audit_sequence_integrity(
    genes: pd.DataFrame,
    prodigal_records: Mapping[tuple[str, str, int, int, int], ProdigalRecord],
    orthofinder_sequences: Mapping[str, str],
    genome_sequences: Mapping[str, Mapping[str, Any]],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compare embedded translations with the persisted OrthoFinder inputs.

    ``buggy_slice_match`` explicitly tests the historical coordinate error:
    slicing a one-based inclusive interval as ``sequence[start:end]``.
    """

    rows: list[dict[str, Any]] = []
    for gene in genes.itertuples(index=False):
        key = (
            gene.sample_id,
            gene.contig_id,
            int(gene.start),
            int(gene.end),
            int(gene.strand),
        )
        prodigal = prodigal_records.get(key)
        if prodigal is None:
            raise ValueError(f"ELSA gene has no exact Prodigal record: {key}")
        orthofinder = orthofinder_sequences.get(gene.gene_id)
        if orthofinder is None:
            raise ValueError(f"ELSA gene is absent from OrthoFinder input: {gene.gene_id}")
        contig_sequence = genome_sequences[gene.sample_id].get(gene.contig_id)
        if contig_sequence is None:
            raise ValueError(f"ELSA contig is absent from genome FASTA: {key[:2]}")

        # Deliberately reproduce the observed historical extraction error.
        buggy_nucleotides = contig_sequence[int(gene.start) : int(gene.end)]
        if int(gene.strand) == -1:
            buggy_nucleotides = buggy_nucleotides.reverse_complement()
        buggy_translation = str(buggy_nucleotides.translate()).replace("*", "X")

        actual = prodigal.sequence
        aligned_matches = sum(
            left == right for left, right in zip(actual, orthofinder)
        )
        aligned_length = max(len(actual), len(orthofinder))
        unexpected_actual = sorted(set(actual) - STANDARD_AA - {"X"})
        unexpected_orthofinder = sorted(set(orthofinder) - STANDARD_AA - {"X"})
        rows.append(
            {
                "sample_id": gene.sample_id,
                "contig_id": gene.contig_id,
                "gene_id": gene.gene_id,
                "start": int(gene.start),
                "end": int(gene.end),
                "strand": int(gene.strand),
                "prodigal_length": len(actual),
                "parquet_original_length": int(gene.original_length),
                "prodigal_partial": prodigal.partial,
                "prodigal_start_type": prodigal.start_type,
                "prodigal_terminal_stop": prodigal.terminal_stop,
                "prodigal_has_x": "X" in actual,
                "prodigal_x_count": actual.count("X"),
                "prodigal_other_nonstandard": ",".join(unexpected_actual),
                "orthofinder_length": len(orthofinder),
                "orthofinder_has_x": "X" in orthofinder,
                "orthofinder_x_count": orthofinder.count("X"),
                "orthofinder_other_nonstandard": ",".join(unexpected_orthofinder),
                "exact_sequence_match": actual == orthofinder,
                "match_after_first_residue": actual[1:] == orthofinder[1:],
                "positional_identity": (
                    aligned_matches / aligned_length if aligned_length else 1.0
                ),
                "buggy_slice_match": orthofinder == buggy_translation,
            }
        )

    detail = pd.DataFrame(rows).sort_values(
        ["sample_id", "contig_id", "start", "end", "gene_id"],
        kind="mergesort",
    )
    strand_summary = (
        detail.groupby("strand", sort=True)
        .agg(
            n=("gene_id", "size"),
            exact_sequence_matches=("exact_sequence_match", "sum"),
            orthofinder_x_sequences=("orthofinder_has_x", "sum"),
            prodigal_x_sequences=("prodigal_has_x", "sum"),
            match_after_first_residue=("match_after_first_residue", "sum"),
            mean_positional_identity=("positional_identity", "mean"),
            buggy_slice_matches=("buggy_slice_match", "sum"),
        )
        .reset_index()
    )
    sample_summary = (
        detail.groupby("sample_id", sort=True)
        .agg(
            n=("gene_id", "size"),
            exact_sequence_matches=("exact_sequence_match", "sum"),
            orthofinder_x_sequences=("orthofinder_has_x", "sum"),
            prodigal_x_sequences=("prodigal_has_x", "sum"),
            mean_positional_identity=("positional_identity", "mean"),
            buggy_slice_matches=("buggy_slice_match", "sum"),
        )
        .reset_index()
    )
    n_sequences = len(detail)
    summary = {
        "n_sequences": n_sequences,
        "exact_sequence_matches": int(detail["exact_sequence_match"].sum()),
        "exact_sequence_match_fraction": float(detail["exact_sequence_match"].mean()),
        "prodigal_x_sequences": int(detail["prodigal_has_x"].sum()),
        "prodigal_x_sequence_fraction": float(detail["prodigal_has_x"].mean()),
        "prodigal_x_residues": int(detail["prodigal_x_count"].sum()),
        "orthofinder_x_sequences": int(detail["orthofinder_has_x"].sum()),
        "orthofinder_x_sequence_fraction": float(detail["orthofinder_has_x"].mean()),
        "orthofinder_x_residues": int(detail["orthofinder_x_count"].sum()),
        "buggy_slice_matches": int(detail["buggy_slice_match"].sum()),
        "buggy_slice_match_fraction": float(detail["buggy_slice_match"].mean()),
        "partial_flag_counts": {
            str(key): int(value)
            for key, value in detail["prodigal_partial"]
            .fillna("missing")
            .value_counts()
            .sort_index()
            .items()
        },
        "strand_summary": strand_summary.to_dict(orient="records"),
        "sample_summary": sample_summary.to_dict(orient="records"),
    }
    return detail, summary


def parse_named_gff_genes(
    path: Path, expected_names: frozenset[str]
) -> list[dict[str, Any]]:
    """Parse GFF gene features carrying one of the expected gene names."""

    records: list[dict[str, Any]] = []
    with path.open() as handle:
        for raw_line in handle:
            if not raw_line or raw_line.startswith("#"):
                continue
            fields = raw_line.rstrip("\n").split("\t")
            if len(fields) < 9 or fields[2] != "gene":
                continue
            attributes = parse_semicolon_attributes(fields[8])
            gene_name = attributes.get("gene")
            if gene_name is None:
                continue
            normalized = gene_name.lower()
            if normalized not in expected_names:
                continue
            records.append(
                {
                    "gene_name": normalized,
                    "contig_id": fields[0],
                    "start": int(fields[3]),
                    "end": int(fields[4]),
                    "strand": fields[6],
                    "locus_tag": attributes.get("locus_tag", ""),
                    "pseudo": attributes.get("pseudo", "false").lower() == "true",
                }
            )
    return records


def map_named_genes_to_elsa(
    named_genes: Sequence[dict[str, Any]], sample_genes: pd.DataFrame
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Map named GFF genes to ELSA calls by at least 50% shorter-feature overlap."""

    by_contig = {
        contig_id: group.copy()
        for contig_id, group in sample_genes.groupby("contig_id", sort=False)
    }
    mapped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    mapping_events = 0
    ambiguous_features = 0
    unmapped_features = 0

    for feature_index, feature in enumerate(named_genes):
        contig_genes = by_contig.get(feature["contig_id"])
        if contig_genes is None:
            unmapped_features += 1
            continue
        starts = contig_genes["start"].to_numpy(dtype=np.int64)
        ends = contig_genes["end"].to_numpy(dtype=np.int64)
        overlap = np.minimum(feature["end"], ends) - np.maximum(
            feature["start"], starts
        ) + 1
        shorter = np.minimum(
            feature["end"] - feature["start"] + 1,
            ends - starts + 1,
        )
        good = (overlap > 0) & (overlap / shorter >= 0.5)
        candidates = contig_genes.loc[good]
        multiplicity = len(candidates)
        if multiplicity == 0:
            unmapped_features += 1
            continue
        if multiplicity > 1:
            ambiguous_features += 1
        for candidate in candidates.itertuples(index=False):
            mapped[feature["gene_name"]].append(
                {
                    "feature_index": feature_index,
                    "gene_name": feature["gene_name"],
                    "gff_start": feature["start"],
                    "gff_end": feature["end"],
                    "gff_strand": feature["strand"],
                    "gff_locus_tag": feature["locus_tag"],
                    "gff_pseudo": feature["pseudo"],
                    "mapping_multiplicity": multiplicity,
                    "gene_id": candidate.gene_id,
                    "contig_id": candidate.contig_id,
                    "elsa_start": int(candidate.start),
                    "elsa_end": int(candidate.end),
                    "elsa_strand": int(candidate.strand),
                    "sample_global_rank": int(candidate.sample_global_rank),
                    "contig_local_rank": int(candidate.contig_local_rank),
                }
            )
            mapping_events += 1
    return dict(mapped), {
        "named_gff_features": len(named_genes),
        "mapping_events": mapping_events,
        "ambiguous_gff_features": ambiguous_features,
        "unmapped_gff_features": unmapped_features,
    }


def is_subsequence(observed: Sequence[str], expected: Sequence[str]) -> bool:
    """Return whether ``observed`` occurs in order within ``expected``."""

    iterator = iter(expected)
    return all(any(item == candidate for candidate in iterator) for item in observed)


def order_status(observed: Sequence[str], expected: Sequence[str]) -> str:
    """Classify observed coordinate order relative to the supplied definition."""

    forward = is_subsequence(observed, expected)
    reverse = is_subsequence(observed, list(reversed(expected)))
    if forward and reverse:
        return "ambiguous"
    if forward:
        return "same"
    if reverse:
        return "reverse"
    return "scrambled"


def choose_operon_window(
    definition: Mapping[str, Any],
    mapped_by_name: Mapping[str, Sequence[dict[str, Any]]],
    max_gap: int = 3,
) -> list[dict[str, Any]] | None:
    """Reproduce the V2 builder's best-window rule using sample-global ranks."""

    available_names = [
        gene_name
        for gene_name in definition["genes"]
        if mapped_by_name.get(gene_name)
    ]
    if len(set(available_names)) < 2:
        return None

    by_contig: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for gene_name in available_names:
        for candidate in mapped_by_name[gene_name]:
            by_contig[candidate["contig_id"]].append(candidate)

    best: list[dict[str, Any]] | None = None
    best_unique_count = 0
    for positions in by_contig.values():
        positions = sorted(positions, key=lambda row: row["sample_global_rank"])
        if len(positions) < 2:
            continue
        for start_index in range(len(positions)):
            window: list[dict[str, Any]] = []
            seen: set[str] = set()
            for candidate in positions[start_index:]:
                if window:
                    gap = (
                        candidate["sample_global_rank"]
                        - window[-1]["sample_global_rank"]
                    )
                    if gap > max_gap + 1:
                        break
                window.append(candidate)
                seen.add(candidate["gene_name"])
            if len(seen) >= 2 and len(seen) > best_unique_count:
                best = window
                best_unique_count = len(seen)
    return best


def extract_unique_loci(pairwise_gt: pd.DataFrame) -> pd.DataFrame:
    """Collapse pairwise rows into exact unique operon-by-genome loci."""

    required = {
        "operon_id",
        "genome_a",
        "genome_b",
        "contig_a",
        "contig_b",
        "gene_idx_start_a",
        "gene_idx_end_a",
        "gene_idx_start_b",
        "gene_idx_end_b",
        "n_genes_a",
        "n_genes_b",
        "genes_a",
        "genes_b",
    }
    missing = required - set(pairwise_gt.columns)
    if missing:
        raise ValueError(f"Pairwise ground truth lacks columns: {sorted(missing)}")

    side_frames: list[pd.DataFrame] = []
    for side in ("a", "b"):
        frame = pairwise_gt[
            [
                "operon_id",
                f"genome_{side}",
                f"contig_{side}",
                f"gene_idx_start_{side}",
                f"gene_idx_end_{side}",
                f"n_genes_{side}",
                f"genes_{side}",
            ]
        ].copy()
        frame.columns = [
            "operon_id",
            "sample_id",
            "source_contig_id",
            "source_global_start",
            "source_global_end",
            "source_matched_count",
            "source_observed_names",
        ]
        side_frames.append(frame)
    stacked = pd.concat(side_frames, ignore_index=True)

    output: list[dict[str, Any]] = []
    source_columns = [
        "source_contig_id",
        "source_global_start",
        "source_global_end",
        "source_matched_count",
        "source_observed_names",
    ]
    for (operon_id, sample_id), group in stacked.groupby(
        ["operon_id", "sample_id"], sort=True
    ):
        variants = group[source_columns].drop_duplicates()
        first = variants.iloc[0]
        output.append(
            {
                "operon_id": operon_id,
                "sample_id": sample_id,
                **{column: first[column] for column in source_columns},
                "source_variant_count": len(variants),
                "pair_row_appearances": len(group),
            }
        )
    return pd.DataFrame(output)


def reconstruct_operon_loci(
    genes: pd.DataFrame,
    definitions: Mapping[str, Mapping[str, Any]],
    annotation_dir: Path,
    prodigal_records: Mapping[tuple[str, str, int, int, int], ProdigalRecord],
    max_gap: int = 3,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Reconstruct every detected locus from the GFF and coordinate contract."""

    expected_names = frozenset(
        gene_name
        for definition in definitions.values()
        for gene_name in definition["genes"]
    )
    rows: list[dict[str, Any]] = []
    mapping_summaries: list[dict[str, Any]] = []

    for sample_id, sample_genes in genes.groupby("sample_id", sort=True):
        gff_path = annotation_dir / f"{sample_id}.gff"
        if not gff_path.exists():
            raise FileNotFoundError(f"Missing annotation GFF: {gff_path}")
        named_features = parse_named_gff_genes(gff_path, expected_names)
        mapped_by_name, mapping_summary = map_named_genes_to_elsa(
            named_features, sample_genes
        )
        mapping_summaries.append({"sample_id": sample_id, **mapping_summary})

        for operon_id, definition in definitions.items():
            window = choose_operon_window(definition, mapped_by_name, max_gap=max_gap)
            if window is None:
                continue
            ordered = sorted(window, key=lambda row: row["sample_global_rank"])
            observed_names = [row["gene_name"] for row in ordered]
            observed_unique_names = list(dict.fromkeys(observed_names))
            selected_gene_ids = list(dict.fromkeys(row["gene_id"] for row in ordered))

            partial_flags: list[str] = []
            missing_prodigal = 0
            sequence_has_x = 0
            for selected in ordered:
                key = (
                    sample_id,
                    selected["contig_id"],
                    selected["elsa_start"],
                    selected["elsa_end"],
                    selected["elsa_strand"],
                )
                prodigal = prodigal_records.get(key)
                if prodigal is None:
                    missing_prodigal += 1
                    continue
                partial_flags.append(prodigal.partial or "missing")
                sequence_has_x += int("X" in prodigal.sequence)

            strands = {row["gff_strand"] for row in ordered}
            if len(strands) == 1:
                only_strand = next(iter(strands))
                strand_status = {
                    "+": "coherent_plus",
                    "-": "coherent_minus",
                }.get(only_strand, "coherent_unknown")
            else:
                strand_status = "mixed"

            expected = list(definition["genes"])
            observed_set = set(observed_unique_names)
            rows.append(
                {
                    "operon_id": operon_id,
                    "sample_id": sample_id,
                    "evidence_label": definition["evidence"],
                    "expected_definition_size": len(expected),
                    "expected_genes": ",".join(expected),
                    "observed_named_genes": ",".join(observed_names),
                    "observed_unique_genes": ",".join(observed_unique_names),
                    "observed_named_count": len(observed_names),
                    "observed_unique_count": len(observed_set),
                    "completeness_fraction": len(observed_set) / len(expected),
                    "complete_definition": observed_set == set(expected),
                    "order_status": order_status(observed_unique_names, expected),
                    "strand_status": strand_status,
                    "contig_id": ordered[0]["contig_id"],
                    "contig_local_start": min(
                        row["contig_local_rank"] for row in ordered
                    ),
                    "contig_local_end": max(
                        row["contig_local_rank"] for row in ordered
                    ),
                    "sample_global_start": min(
                        row["sample_global_rank"] for row in ordered
                    ),
                    "sample_global_end": max(
                        row["sample_global_rank"] for row in ordered
                    ),
                    "elsa_gene_ids": ",".join(selected_gene_ids),
                    "mapping_multiplicity_max": max(
                        row["mapping_multiplicity"] for row in ordered
                    ),
                    "mapping_multiplicity_sum": sum(
                        row["mapping_multiplicity"] for row in ordered
                    ),
                    "mapped_pseudogene_count": sum(
                        int(row["gff_pseudo"]) for row in ordered
                    ),
                    "prodigal_partial_flags": ",".join(sorted(set(partial_flags))),
                    "prodigal_partial_any": any(
                        flag not in {"00", "missing"} for flag in partial_flags
                    ),
                    "prodigal_x_sequence_count": sequence_has_x,
                    "missing_prodigal_records": missing_prodigal,
                }
            )

    loci = pd.DataFrame(rows).sort_values(
        ["operon_id", "sample_id"], kind="mergesort"
    )
    mapping_frame = pd.DataFrame(mapping_summaries)
    summary = {
        "reconstructed_loci": len(loci),
        "mapping_by_sample": mapping_frame.to_dict(orient="records"),
        "complete_loci": int(loci["complete_definition"].sum()),
        "completeness_mean": float(loci["completeness_fraction"].mean()),
        "completeness_median": float(loci["completeness_fraction"].median()),
        "order_status_counts": {
            str(key): int(value)
            for key, value in loci["order_status"].value_counts().sort_index().items()
        },
        "strand_status_counts": {
            str(key): int(value)
            for key, value in loci["strand_status"].value_counts().sort_index().items()
        },
        "partial_loci": int(loci["prodigal_partial_any"].sum()),
        "ambiguous_mapping_loci": int(
            (loci["mapping_multiplicity_max"] > 1).sum()
        ),
    }
    return loci, summary


def merge_source_and_reconstructed_loci(
    source_loci: pd.DataFrame, reconstructed: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Join the persisted cohort to its independently reconstructed loci."""

    merged = source_loci.merge(
        reconstructed,
        on=["operon_id", "sample_id"],
        how="outer",
        indicator=True,
        validate="one_to_one",
    )
    merged["source_present"] = merged["_merge"].isin(["both", "left_only"])
    merged["reconstructed_present"] = merged["_merge"].isin(["both", "right_only"])
    both = merged["_merge"] == "both"
    merged["source_contig_matches"] = (
        merged["source_contig_id"] == merged["contig_id"]
    ) & both
    merged["source_global_rank_matches"] = (
        (merged["source_global_start"] == merged["sample_global_start"])
        & (merged["source_global_end"] == merged["sample_global_end"])
        & both
    )
    merged["source_observed_names_match"] = (
        merged["source_observed_names"] == merged["observed_named_genes"]
    ) & both
    merged["global_local_start_delta"] = (
        merged["sample_global_start"] - merged["contig_local_start"]
    )
    merged["global_local_end_delta"] = (
        merged["sample_global_end"] - merged["contig_local_end"]
    )
    merged["source_uses_contig_local_ranks"] = (
        (merged["source_global_start"] == merged["contig_local_start"])
        & (merged["source_global_end"] == merged["contig_local_end"])
        & both
    )

    summary = {
        "source_unique_loci": len(source_loci),
        "reconstructed_loci": len(reconstructed),
        "joined_loci": int(both.sum()),
        "source_only_loci": int((merged["_merge"] == "left_only").sum()),
        "reconstructed_only_loci": int((merged["_merge"] == "right_only").sum()),
        "source_variant_loci": int((merged["source_variant_count"] > 1).sum()),
        "source_contig_matches": int(merged["source_contig_matches"].sum()),
        "source_global_rank_matches": int(
            merged["source_global_rank_matches"].sum()
        ),
        "source_observed_names_matches": int(
            merged["source_observed_names_match"].sum()
        ),
        "source_uses_contig_local_ranks": int(
            merged["source_uses_contig_local_ranks"].sum()
        ),
        "rank_shifted_loci": int(
            (
                both
                & (
                    (merged["sample_global_start"] != merged["contig_local_start"])
                    | (merged["sample_global_end"] != merged["contig_local_end"])
                )
            ).sum()
        ),
    }
    return (
        merged.drop(columns=["_merge"]).sort_values(
            ["operon_id", "sample_id"], kind="mergesort"
        ),
        summary,
    )


def translate_source_loci_to_local_coordinates(
    source_loci: pd.DataFrame,
    genes: pd.DataFrame,
    definitions: Mapping[str, Mapping[str, Any]],
    reconstructed: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Translate persisted sample-global boundaries through the gene table.

    This conversion retains every persisted source locus, including loci that
    the current annotation snapshot no longer reconstructs.
    """

    rank_lookup = genes.set_index(["sample_id", "sample_global_rank"])[
        ["contig_id", "contig_local_rank", "gene_id"]
    ]
    reconstructed_lookup = reconstructed.set_index(["operon_id", "sample_id"])
    rows: list[dict[str, Any]] = []
    boundary_contig_mismatches = 0
    reversed_boundaries = 0

    for source in source_loci.itertuples(index=False):
        start_key = (source.sample_id, int(source.source_global_start))
        end_key = (source.sample_id, int(source.source_global_end))
        try:
            start_gene = rank_lookup.loc[start_key]
            end_gene = rank_lookup.loc[end_key]
        except KeyError as error:
            raise ValueError(
                f"Persisted locus boundary has no gene-table row: {error}"
            ) from error
        start_contig = str(start_gene["contig_id"])
        end_contig = str(end_gene["contig_id"])
        contig_match = (
            start_contig == str(source.source_contig_id)
            and end_contig == str(source.source_contig_id)
        )
        boundary_contig_mismatches += int(not contig_match)
        if not contig_match:
            raise ValueError(
                "Persisted sample-global boundaries map outside their recorded "
                f"contig for {(source.operon_id, source.sample_id)}"
            )
        local_start = int(start_gene["contig_local_rank"])
        local_end = int(end_gene["contig_local_rank"])
        if local_end < local_start:
            reversed_boundaries += 1
            local_start, local_end = local_end, local_start

        definition = definitions[source.operon_id]
        expected = list(definition["genes"])
        observed = [
            name.strip().lower()
            for name in str(source.source_observed_names).split(",")
            if name.strip()
        ]
        observed_set = set(observed)
        reconstruction_key = (source.operon_id, source.sample_id)
        reconstruction_present = reconstruction_key in reconstructed_lookup.index
        if reconstruction_present:
            reconstruction_row = reconstructed_lookup.loc[reconstruction_key]
            strand_status = str(reconstruction_row["strand_status"])
        else:
            strand_status = "unresolved_current_annotation"
        rows.append(
            {
                "operon_id": source.operon_id,
                "sample_id": source.sample_id,
                "contig_id": source.source_contig_id,
                "contig_local_start": local_start,
                "contig_local_end": local_end,
                "complete_definition": observed_set == set(expected),
                "completeness_fraction": len(observed_set) / len(expected),
                "order_status": order_status(observed, expected),
                "strand_status": strand_status,
                "source_start_gene_id": str(start_gene["gene_id"]),
                "source_end_gene_id": str(end_gene["gene_id"]),
                "current_reconstruction_present": reconstruction_present,
            }
        )
    translated = pd.DataFrame(rows).sort_values(
        ["operon_id", "sample_id"], kind="mergesort"
    )
    return translated, {
        "translated_source_loci": len(translated),
        "boundary_contig_mismatches": boundary_contig_mismatches,
        "reversed_local_boundaries": reversed_boundaries,
        "current_annotation_unreconstructed_loci": int(
            (~translated["current_reconstruction_present"]).sum()
        ),
    }


def attach_corrected_loci_to_pairs(
    pairwise_gt: pd.DataFrame, loci: pd.DataFrame
) -> pd.DataFrame:
    """Replace stored sample-global ranks with audited contig-local ranks."""

    lookup_columns = [
        "operon_id",
        "sample_id",
        "contig_id",
        "contig_local_start",
        "contig_local_end",
        "complete_definition",
        "completeness_fraction",
        "order_status",
        "strand_status",
    ]
    lookup = loci[lookup_columns]
    corrected = pairwise_gt.copy()
    for side in ("a", "b"):
        side_lookup = lookup.rename(
            columns={
                "sample_id": f"genome_{side}",
                "contig_id": f"corrected_contig_{side}",
                "contig_local_start": f"corrected_start_{side}",
                "contig_local_end": f"corrected_end_{side}",
                "complete_definition": f"complete_definition_{side}",
                "completeness_fraction": f"completeness_fraction_{side}",
                "order_status": f"order_status_{side}",
                "strand_status": f"strand_status_{side}",
            }
        )
        corrected = corrected.merge(
            side_lookup,
            on=["operon_id", f"genome_{side}"],
            how="left",
            validate="many_to_one",
        )
    if corrected[
        ["corrected_start_a", "corrected_start_b"]
    ].isna().any(axis=None):
        missing = corrected.loc[
            corrected[["corrected_start_a", "corrected_start_b"]]
            .isna()
            .any(axis=1),
            ["operon_id", "genome_a", "genome_b"],
        ].head()
        raise ValueError(f"Pair rows lack reconstructed loci:\n{missing}")
    return corrected


def normalize_pair_rows(
    pairwise_gt: pd.DataFrame,
    *,
    corrected: bool,
) -> pd.DataFrame:
    """Normalize pair rows so genome A is lexicographically first."""

    rows: list[dict[str, Any]] = []
    for source_index, row in pairwise_gt.reset_index(drop=True).iterrows():
        swap = row["genome_a"] > row["genome_b"]
        left = "b" if swap else "a"
        right = "a" if swap else "b"
        if corrected:
            left_contig = row[f"corrected_contig_{left}"]
            right_contig = row[f"corrected_contig_{right}"]
            left_start = row[f"corrected_start_{left}"]
            left_end = row[f"corrected_end_{left}"]
            right_start = row[f"corrected_start_{right}"]
            right_end = row[f"corrected_end_{right}"]
        else:
            left_contig = row[f"contig_{left}"]
            right_contig = row[f"contig_{right}"]
            left_start = row[f"gene_idx_start_{left}"]
            left_end = row[f"gene_idx_end_{left}"]
            right_start = row[f"gene_idx_start_{right}"]
            right_end = row[f"gene_idx_end_{right}"]
        rows.append(
            {
                "pair_row_id": int(source_index),
                "operon_id": row["operon_id"],
                "genome_a": row[f"genome_{left}"],
                "genome_b": row[f"genome_{right}"],
                "contig_a": left_contig,
                "contig_b": right_contig,
                "start_a": int(left_start),
                "end_a": int(left_end),
                "start_b": int(right_start),
                "end_b": int(right_end),
                "complete_definition_a": bool(
                    row.get(f"complete_definition_{left}", False)
                ),
                "complete_definition_b": bool(
                    row.get(f"complete_definition_{right}", False)
                ),
                "completeness_fraction_a": float(
                    row.get(f"completeness_fraction_{left}", np.nan)
                ),
                "completeness_fraction_b": float(
                    row.get(f"completeness_fraction_{right}", np.nan)
                ),
            }
        )
    normalized = pd.DataFrame(rows)
    if (normalized["genome_a"] >= normalized["genome_b"]).any():
        raise ValueError("Ground truth contains self-comparisons or failed normalization")
    return normalized


def normalize_blocks(blocks: pd.DataFrame) -> pd.DataFrame:
    """Normalize ELSA blocks to the same lexical pair orientation."""

    required = {
        "block_id",
        "query_genome",
        "target_genome",
        "query_contig",
        "target_contig",
        "query_start",
        "query_end",
        "target_start",
        "target_end",
        "n_anchors",
    }
    missing = required - set(blocks.columns)
    if missing:
        raise ValueError(f"Block table lacks columns: {sorted(missing)}")

    rows: list[dict[str, Any]] = []
    for block in blocks.itertuples(index=False):
        swap = block.query_genome > block.target_genome
        rows.append(
            {
                "block_id": int(block.block_id),
                "genome_a": block.target_genome if swap else block.query_genome,
                "genome_b": block.query_genome if swap else block.target_genome,
                "contig_a": block.target_contig if swap else block.query_contig,
                "contig_b": block.query_contig if swap else block.target_contig,
                "start_a": int(block.target_start if swap else block.query_start),
                "end_a": int(block.target_end if swap else block.query_end),
                "start_b": int(block.query_start if swap else block.target_start),
                "end_b": int(block.query_end if swap else block.target_end),
                "n_anchors": int(block.n_anchors),
                "chain_score": float(getattr(block, "chain_score", np.nan)),
                "orientation": int(getattr(block, "orientation", 0)),
            }
        )
    normalized = pd.DataFrame(rows)
    if (normalized["genome_a"] >= normalized["genome_b"]).any():
        raise ValueError("ELSA block table contains self-comparisons")
    for start_column, end_column in (("start_a", "end_a"), ("start_b", "end_b")):
        if (
            (normalized[start_column] < 0)
            | (normalized[end_column] < normalized[start_column])
        ).any():
            raise ValueError(f"Malformed block intervals: {start_column}/{end_column}")
    return normalized


def build_block_index(
    blocks: pd.DataFrame,
) -> dict[tuple[str, str, str, str], dict[str, np.ndarray]]:
    """Index normalized blocks by exact genome and contig pair."""

    index: dict[tuple[str, str, str, str], dict[str, np.ndarray]] = {}
    group_columns = ["genome_a", "genome_b", "contig_a", "contig_b"]
    value_columns = [
        "block_id",
        "start_a",
        "end_a",
        "start_b",
        "end_b",
        "n_anchors",
        "chain_score",
        "orientation",
    ]
    for key, group in blocks.groupby(group_columns, sort=False):
        index[key] = {
            column: group[column].to_numpy()
            for column in value_columns
        }
    return index


def inclusive_overlap(
    starts: np.ndarray | int,
    ends: np.ndarray | int,
    interval_start: int,
    interval_end: int,
) -> np.ndarray:
    """Return inclusive overlap lengths."""

    return np.maximum(
        0,
        np.minimum(ends, interval_end) - np.maximum(starts, interval_start) + 1,
    )


def evaluate_pair_metrics(
    pair_rows: pd.DataFrame,
    block_index: Mapping[
        tuple[str, str, str, str], Mapping[str, np.ndarray]
    ],
) -> pd.DataFrame:
    """Persist boundary-aware metrics for every operon pair row."""

    output: list[dict[str, Any]] = []
    for row in pair_rows.itertuples(index=False):
        key = (row.genome_a, row.genome_b, row.contig_a, row.contig_b)
        candidate = block_index.get(key)
        locus_span_a = row.end_a - row.start_a + 1
        locus_span_b = row.end_b - row.start_b + 1
        base = {
            "pair_row_id": row.pair_row_id,
            "operon_id": row.operon_id,
            "genome_a": row.genome_a,
            "genome_b": row.genome_b,
            "contig_a": row.contig_a,
            "contig_b": row.contig_b,
            "start_a": row.start_a,
            "end_a": row.end_a,
            "start_b": row.start_b,
            "end_b": row.end_b,
            "locus_span_a": locus_span_a,
            "locus_span_b": locus_span_b,
            "complete_definition_a": row.complete_definition_a,
            "complete_definition_b": row.complete_definition_b,
            "completeness_fraction_a": row.completeness_fraction_a,
            "completeness_fraction_b": row.completeness_fraction_b,
        }
        if candidate is None:
            output.append(
                {
                    **base,
                    "candidate_blocks": 0,
                    "strict50": False,
                    "strict100": False,
                    "independent50": False,
                    "any50": False,
                    "best_block_id": -1,
                    "best_coverage_a": 0.0,
                    "best_coverage_b": 0.0,
                    "best_boundary_precision_a": 0.0,
                    "best_boundary_precision_b": 0.0,
                    "best_iou_a": 0.0,
                    "best_iou_b": 0.0,
                    "best_pair_iou": 0.0,
                    "best_block_span_a": 0,
                    "best_block_span_b": 0,
                    "best_n_anchors": 0,
                    "best_anchor_density": 0.0,
                    "smallest_containing_block_id": -1,
                    "smallest_containing_span_a": 0,
                    "smallest_containing_span_b": 0,
                    "smallest_containing_excess_genes": np.nan,
                }
            )
            continue

        block_span_a = candidate["end_a"] - candidate["start_a"] + 1
        block_span_b = candidate["end_b"] - candidate["start_b"] + 1
        overlap_a = inclusive_overlap(
            candidate["start_a"], candidate["end_a"], row.start_a, row.end_a
        )
        overlap_b = inclusive_overlap(
            candidate["start_b"], candidate["end_b"], row.start_b, row.end_b
        )
        coverage_a = overlap_a / locus_span_a
        coverage_b = overlap_b / locus_span_b
        precision_a = overlap_a / block_span_a
        precision_b = overlap_b / block_span_b
        iou_a = overlap_a / (locus_span_a + block_span_a - overlap_a)
        iou_b = overlap_b / (locus_span_b + block_span_b - overlap_b)
        pair_iou = np.sqrt(iou_a * iou_b)
        minimum_coverage = np.minimum(coverage_a, coverage_b)
        combined_span = block_span_a + block_span_b
        ordering = np.lexsort(
            (
                candidate["block_id"],
                combined_span,
                -pair_iou,
                -minimum_coverage,
            )
        )
        best_index = int(ordering[0])

        containing = (coverage_a >= 1.0) & (coverage_b >= 1.0)
        if containing.any():
            containing_indices = np.flatnonzero(containing)
            local_order = np.lexsort(
                (
                    candidate["block_id"][containing_indices],
                    combined_span[containing_indices],
                )
            )
            containing_index = int(containing_indices[local_order[0]])
            containing_id = int(candidate["block_id"][containing_index])
            containing_span_a = int(block_span_a[containing_index])
            containing_span_b = int(block_span_b[containing_index])
            containing_excess = (
                containing_span_a
                + containing_span_b
                - locus_span_a
                - locus_span_b
            )
        else:
            containing_id = -1
            containing_span_a = 0
            containing_span_b = 0
            containing_excess = np.nan

        output.append(
            {
                **base,
                "candidate_blocks": len(candidate["block_id"]),
                "strict50": bool(
                    ((coverage_a >= 0.5) & (coverage_b >= 0.5)).any()
                ),
                "strict100": bool(containing.any()),
                "independent50": bool(
                    coverage_a.max(initial=0.0) >= 0.5
                    and coverage_b.max(initial=0.0) >= 0.5
                ),
                "any50": bool(
                    coverage_a.max(initial=0.0) >= 0.5
                    or coverage_b.max(initial=0.0) >= 0.5
                ),
                "best_block_id": int(candidate["block_id"][best_index]),
                "best_coverage_a": float(coverage_a[best_index]),
                "best_coverage_b": float(coverage_b[best_index]),
                "best_boundary_precision_a": float(precision_a[best_index]),
                "best_boundary_precision_b": float(precision_b[best_index]),
                "best_iou_a": float(iou_a[best_index]),
                "best_iou_b": float(iou_b[best_index]),
                "best_pair_iou": float(pair_iou[best_index]),
                "best_block_span_a": int(block_span_a[best_index]),
                "best_block_span_b": int(block_span_b[best_index]),
                "best_n_anchors": int(candidate["n_anchors"][best_index]),
                "best_anchor_density": float(
                    candidate["n_anchors"][best_index]
                    / max(block_span_a[best_index], block_span_b[best_index])
                ),
                "smallest_containing_block_id": containing_id,
                "smallest_containing_span_a": containing_span_a,
                "smallest_containing_span_b": containing_span_b,
                "smallest_containing_excess_genes": containing_excess,
            }
        )
    return pd.DataFrame(output).sort_values("pair_row_id", kind="mergesort")


def recall_only(
    pair_rows: pd.DataFrame,
    block_index: Mapping[
        tuple[str, str, str, str], Mapping[str, np.ndarray]
    ],
) -> tuple[float, float]:
    """Compute paired 50% and 100% span recall with groupwise matrices."""

    strict50_hits = 0
    strict100_hits = 0
    group_columns = ["genome_a", "genome_b", "contig_a", "contig_b"]
    for key, loci in pair_rows.groupby(group_columns, sort=False):
        candidate = block_index.get(key)
        if candidate is None:
            continue
        locus_start_a = loci["start_a"].to_numpy(dtype=np.int64)[:, None]
        locus_end_a = loci["end_a"].to_numpy(dtype=np.int64)[:, None]
        locus_start_b = loci["start_b"].to_numpy(dtype=np.int64)[:, None]
        locus_end_b = loci["end_b"].to_numpy(dtype=np.int64)[:, None]
        overlap_a = np.maximum(
            0,
            np.minimum(locus_end_a, candidate["end_a"][None, :])
            - np.maximum(locus_start_a, candidate["start_a"][None, :])
            + 1,
        )
        overlap_b = np.maximum(
            0,
            np.minimum(locus_end_b, candidate["end_b"][None, :])
            - np.maximum(locus_start_b, candidate["start_b"][None, :])
            + 1,
        )
        span_a = locus_end_a - locus_start_a + 1
        span_b = locus_end_b - locus_start_b + 1
        coverage_a = overlap_a / span_a
        coverage_b = overlap_b / span_b
        strict50_hits += int(
            ((coverage_a >= 0.5) & (coverage_b >= 0.5)).any(axis=1).sum()
        )
        strict100_hits += int(
            ((coverage_a >= 1.0) & (coverage_b >= 1.0)).any(axis=1).sum()
        )
    denominator = len(pair_rows)
    return strict50_hits / denominator, strict100_hits / denominator


def quantile_interval(
    values: Sequence[float], alpha: float = 0.05
) -> tuple[float, float]:
    """Return the equal-tail empirical interval."""

    array = np.asarray(values, dtype=float)
    return (
        float(np.quantile(array, alpha / 2)),
        float(np.quantile(array, 1 - alpha / 2)),
    )


def clustered_bootstrap(
    pair_metrics: pd.DataFrame,
    metric: str,
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    """Bootstrap pair rows by operon, genome, and both dimensions."""

    rng = np.random.default_rng(seed)
    operons = np.array(sorted(pair_metrics["operon_id"].unique()), dtype=object)
    genomes = np.array(
        sorted(
            set(pair_metrics["genome_a"]) | set(pair_metrics["genome_b"])
        ),
        dtype=object,
    )
    values = pair_metrics[metric].astype(float).to_numpy()
    row_operons = pair_metrics["operon_id"].to_numpy()
    row_genome_a = pair_metrics["genome_a"].to_numpy()
    row_genome_b = pair_metrics["genome_b"].to_numpy()

    by_operon: list[float] = []
    by_genome: list[float] = []
    by_both: list[float] = []
    for _ in range(replicates):
        operon_counts = Counter(rng.choice(operons, size=len(operons), replace=True))
        genome_counts = Counter(rng.choice(genomes, size=len(genomes), replace=True))
        operon_weights = np.fromiter(
            (operon_counts[item] for item in row_operons),
            dtype=float,
            count=len(values),
        )
        genome_weights = np.fromiter(
            (
                genome_counts[left] * genome_counts[right]
                for left, right in zip(row_genome_a, row_genome_b)
            ),
            dtype=float,
            count=len(values),
        )
        if operon_weights.sum() > 0:
            by_operon.append(float(np.average(values, weights=operon_weights)))
        if genome_weights.sum() > 0:
            by_genome.append(float(np.average(values, weights=genome_weights)))
        combined_weights = operon_weights * genome_weights
        if combined_weights.sum() > 0:
            by_both.append(float(np.average(values, weights=combined_weights)))

    return {
        "replicates": replicates,
        "seed": seed,
        "operon_cluster_interval": quantile_interval(by_operon),
        "genome_cluster_interval": quantile_interval(by_genome),
        "operon_genome_interval": quantile_interval(by_both),
    }


def summarize_pair_metrics(
    pair_metrics: pd.DataFrame,
    *,
    bootstrap_replicates: int,
    seed: int,
) -> dict[str, Any]:
    """Aggregate pair, locus, operon, and genome-level endpoints."""

    metric_columns = ["strict50", "strict100", "independent50", "any50"]
    pair_micro = {
        metric: {
            "hits": int(pair_metrics[metric].sum()),
            "total": len(pair_metrics),
            "rate": float(pair_metrics[metric].mean()),
        }
        for metric in metric_columns
    }

    per_operon = (
        pair_metrics.groupby("operon_id", sort=True)[metric_columns]
        .mean()
        .reset_index()
    )
    genome_sides = pd.concat(
        [
            pair_metrics[["operon_id", "genome_a", *metric_columns]].rename(
                columns={"genome_a": "genome"}
            ),
            pair_metrics[["operon_id", "genome_b", *metric_columns]].rename(
                columns={"genome_b": "genome"}
            ),
        ],
        ignore_index=True,
    )
    per_genome = (
        genome_sides.groupby("genome", sort=True)[metric_columns]
        .mean()
        .reset_index()
    )
    locus_partner = (
        genome_sides.groupby(["operon_id", "genome"], sort=True)[metric_columns]
        .mean()
        .reset_index()
    )
    locus_any = (
        genome_sides.groupby(["operon_id", "genome"], sort=True)[metric_columns]
        .max()
        .reset_index()
    )
    strict50_rows = pair_metrics[pair_metrics["strict50"]]
    boundary_summary = {
        "strict50_rows": len(strict50_rows),
        "median_pair_iou": float(strict50_rows["best_pair_iou"].median()),
        "median_boundary_precision_a": float(
            strict50_rows["best_boundary_precision_a"].median()
        ),
        "median_boundary_precision_b": float(
            strict50_rows["best_boundary_precision_b"].median()
        ),
        "median_anchor_density": float(
            strict50_rows["best_anchor_density"].median()
        ),
        "median_smallest_containing_excess_genes": float(
            strict50_rows["smallest_containing_excess_genes"].median()
        ),
    }
    return {
        "pair_micro": pair_micro,
        "per_operon_macro": {
            metric: float(per_operon[metric].mean()) for metric in metric_columns
        },
        "per_genome_macro": {
            metric: float(per_genome[metric].mean()) for metric in metric_columns
        },
        "per_locus_partner_fraction_macro": {
            metric: float(locus_partner[metric].mean()) for metric in metric_columns
        },
        "per_locus_any_partner": {
            metric: {
                "hits": int(locus_any[metric].sum()),
                "total": len(locus_any),
                "rate": float(locus_any[metric].mean()),
            }
            for metric in metric_columns
        },
        "boundary_summary": boundary_summary,
        "bootstrap": {
            metric: clustered_bootstrap(
                pair_metrics,
                metric,
                replicates=bootstrap_replicates,
                seed=seed + offset,
            )
            for offset, metric in enumerate(("strict50", "strict100"))
        },
        "per_operon": per_operon.to_dict(orient="records"),
        "per_genome": per_genome.to_dict(orient="records"),
    }


def shuffle_target_loci(
    pair_rows: pd.DataFrame, rng: np.random.Generator
) -> pd.DataFrame:
    """Permute target-locus identities within each genome pair."""

    shuffled_groups: list[pd.DataFrame] = []
    target_columns = ["contig_b", "start_b", "end_b"]
    for _, group in pair_rows.groupby(["genome_a", "genome_b"], sort=False):
        shuffled = group.copy()
        permutation = rng.permutation(len(group))
        shuffled[target_columns] = group[target_columns].iloc[
            permutation
        ].to_numpy()
        shuffled_groups.append(shuffled)
    return pd.concat(shuffled_groups, ignore_index=True)


def shift_target_loci(
    pair_rows: pd.DataFrame,
    contig_sizes: Mapping[tuple[str, str], int],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Rotate target starts over every admissible placement on the same contig."""

    shifted = pair_rows.copy()
    new_starts: list[int] = []
    new_ends: list[int] = []
    for row in pair_rows.itertuples(index=False):
        contig_size = contig_sizes[(row.genome_b, row.contig_b)]
        span = row.end_b - row.start_b + 1
        maximum_start = contig_size - span
        if maximum_start < 0:
            raise ValueError(
                f"Locus span {span} exceeds {(row.genome_b, row.contig_b)} "
                f"length {contig_size}"
            )
        modulus = maximum_start + 1
        offset = int(rng.integers(0, modulus)) if modulus > 1 else 0
        start = (row.start_b + offset) % modulus
        new_starts.append(start)
        new_ends.append(start + span - 1)
    shifted["start_b"] = new_starts
    shifted["end_b"] = new_ends
    return shifted


def randomize_block_placements(
    blocks: pd.DataFrame,
    contig_sizes: Mapping[tuple[str, str], int],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Preserve block spans/pairs/contigs while randomizing both placements."""

    randomized = blocks.copy()
    for side in ("a", "b"):
        starts: list[int] = []
        ends: list[int] = []
        for row in blocks.itertuples(index=False):
            genome = getattr(row, f"genome_{side}")
            contig = getattr(row, f"contig_{side}")
            old_start = int(getattr(row, f"start_{side}"))
            old_end = int(getattr(row, f"end_{side}"))
            span = old_end - old_start + 1
            contig_size = contig_sizes[(genome, contig)]
            maximum_start = contig_size - span
            if maximum_start < 0:
                raise ValueError(
                    f"Block span {span} exceeds {(genome, contig)} "
                    f"length {contig_size}"
                )
            start = int(rng.integers(0, maximum_start + 1))
            starts.append(start)
            ends.append(start + span - 1)
        randomized[f"start_{side}"] = starts
        randomized[f"end_{side}"] = ends
    return randomized


def whole_contig_baseline(
    pair_rows: pd.DataFrame,
    contig_sizes: Mapping[tuple[str, str], int],
) -> pd.DataFrame:
    """Create one trivial whole-contig block for every evaluated contig pair."""

    rows: list[dict[str, Any]] = []
    keys = pair_rows[
        ["genome_a", "genome_b", "contig_a", "contig_b"]
    ].drop_duplicates()
    for block_id, row in enumerate(keys.itertuples(index=False)):
        rows.append(
            {
                "block_id": block_id,
                "genome_a": row.genome_a,
                "genome_b": row.genome_b,
                "contig_a": row.contig_a,
                "contig_b": row.contig_b,
                "start_a": 0,
                "end_a": contig_sizes[(row.genome_a, row.contig_a)] - 1,
                "start_b": 0,
                "end_b": contig_sizes[(row.genome_b, row.contig_b)] - 1,
                "n_anchors": 1,
                "chain_score": 0.0,
                "orientation": 1,
            }
        )
    return pd.DataFrame(rows)


def summarize_null_distribution(
    values: pd.DataFrame,
    observed_strict50: float,
    observed_strict100: float,
) -> dict[str, Any]:
    """Summarize null distributions and finite-sample empirical p-values."""

    output: dict[str, Any] = {}
    for control, group in values.groupby("control", sort=True):
        strict50 = group["strict50"].to_numpy(dtype=float)
        strict100 = group["strict100"].to_numpy(dtype=float)
        output[control] = {
            "replicates": len(group),
            "strict50_mean": float(strict50.mean()),
            "strict50_sd": float(strict50.std(ddof=1)) if len(group) > 1 else 0.0,
            "strict50_interval": quantile_interval(strict50),
            "strict50_observed_minus_null_mean": float(
                observed_strict50 - strict50.mean()
            ),
            "strict50_empirical_p_ge_observed": float(
                (1 + np.count_nonzero(strict50 >= observed_strict50))
                / (len(strict50) + 1)
            ),
            "strict100_mean": float(strict100.mean()),
            "strict100_sd": float(strict100.std(ddof=1)) if len(group) > 1 else 0.0,
            "strict100_interval": quantile_interval(strict100),
            "strict100_observed_minus_null_mean": float(
                observed_strict100 - strict100.mean()
            ),
            "strict100_empirical_p_ge_observed": float(
                (1 + np.count_nonzero(strict100 >= observed_strict100))
                / (len(strict100) + 1)
            ),
        }
    return output


def run_null_controls(
    pair_rows: pd.DataFrame,
    blocks: pd.DataFrame,
    contig_sizes: Mapping[tuple[str, str], int],
    *,
    permutations: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run three placement nulls plus the whole-contig ceiling control."""

    observed_index = build_block_index(blocks)
    observed_strict50, observed_strict100 = recall_only(
        pair_rows, observed_index
    )
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []

    for replicate in range(permutations):
        shuffled_loci = shuffle_target_loci(pair_rows, rng)
        strict50, strict100 = recall_only(shuffled_loci, observed_index)
        rows.append(
            {
                "control": "shuffle_target_loci_within_genome_pair",
                "replicate": replicate,
                "strict50": strict50,
                "strict100": strict100,
            }
        )

        shifted_loci = shift_target_loci(pair_rows, contig_sizes, rng)
        strict50, strict100 = recall_only(shifted_loci, observed_index)
        rows.append(
            {
                "control": "rotate_target_loci_within_contig",
                "replicate": replicate,
                "strict50": strict50,
                "strict100": strict100,
            }
        )

        randomized_blocks = randomize_block_placements(blocks, contig_sizes, rng)
        strict50, strict100 = recall_only(
            pair_rows, build_block_index(randomized_blocks)
        )
        rows.append(
            {
                "control": "randomize_block_placements_preserve_spans",
                "replicate": replicate,
                "strict50": strict50,
                "strict100": strict100,
            }
        )

    ceiling_blocks = whole_contig_baseline(pair_rows, contig_sizes)
    ceiling50, ceiling100 = recall_only(
        pair_rows, build_block_index(ceiling_blocks)
    )
    rows.append(
        {
            "control": "whole_contig_ceiling",
            "replicate": 0,
            "strict50": ceiling50,
            "strict100": ceiling100,
        }
    )
    detail = pd.DataFrame(rows)
    summary = {
        "permutations": permutations,
        "seed": seed,
        "observed_strict50": observed_strict50,
        "observed_strict100": observed_strict100,
        "distributions": summarize_null_distribution(
            detail[detail["control"] != "whole_contig_ceiling"],
            observed_strict50,
            observed_strict100,
        ),
        "whole_contig_ceiling": {
            "strict50": ceiling50,
            "strict100": ceiling100,
            "blocks": len(ceiling_blocks),
        },
    }
    return detail, summary


def legacy_span_metrics(
    pair_rows: pd.DataFrame, blocks: pd.DataFrame, threshold: float = 0.5
) -> dict[str, Any]:
    """Reproduce the published evaluator, including its contig omission."""

    by_pair: dict[tuple[str, str], pd.DataFrame] = {
        key: group
        for key, group in blocks.groupby(["genome_a", "genome_b"], sort=False)
    }
    strict = 0
    independent = 0
    any_coverage = 0
    for row in pair_rows.itertuples(index=False):
        candidates = by_pair.get((row.genome_a, row.genome_b))
        if candidates is None:
            continue
        overlap_a = inclusive_overlap(
            candidates["start_a"].to_numpy(),
            candidates["end_a"].to_numpy(),
            row.start_a,
            row.end_a,
        )
        overlap_b = inclusive_overlap(
            candidates["start_b"].to_numpy(),
            candidates["end_b"].to_numpy(),
            row.start_b,
            row.end_b,
        )
        coverage_a = overlap_a / (row.end_a - row.start_a + 1)
        coverage_b = overlap_b / (row.end_b - row.start_b + 1)
        strict += int(((coverage_a >= threshold) & (coverage_b >= threshold)).any())
        independent += int(
            coverage_a.max(initial=0.0) >= threshold
            and coverage_b.max(initial=0.0) >= threshold
        )
        any_coverage += int(
            coverage_a.max(initial=0.0) >= threshold
            or coverage_b.max(initial=0.0) >= threshold
        )
    denominator = len(pair_rows)
    return {
        "denominator": denominator,
        "strict_hits": strict,
        "strict_rate": strict / denominator,
        "independent_hits": independent,
        "independent_rate": independent / denominator,
        "any_hits": any_coverage,
        "any_rate": any_coverage / denominator,
        "contig_filter": False,
        "coordinate_contract": "stored sample-global ranks",
    }


def parse_mcscanx_gff(path: Path) -> dict[str, tuple[str, int, int]]:
    """Parse MCScanX's four-column coordinate file."""

    records: dict[str, tuple[str, int, int]] = {}
    with path.open() as handle:
        for line_number, raw_line in enumerate(handle, 1):
            fields = raw_line.rstrip("\n").split("\t")
            if len(fields) < 4:
                continue
            internal_id = fields[1]
            value = (fields[0], int(fields[2]), int(fields[3]))
            previous = records.get(internal_id)
            if previous is not None and previous != value:
                raise ValueError(
                    f"Conflicting MCScanX GFF ID {internal_id} at "
                    f"{path}:{line_number}"
                )
            records[internal_id] = value
    return records


def parse_sequence_ids(path: Path) -> dict[str, str]:
    """Parse OrthoFinder's internal-to-original identifier map."""

    records: dict[str, str] = {}
    with path.open() as handle:
        for line_number, raw_line in enumerate(handle, 1):
            line = raw_line.strip()
            if not line or ":" not in line:
                continue
            internal_id, original = line.split(":", 1)
            internal_id = internal_id.strip()
            original = original.strip()
            previous = records.get(internal_id)
            if previous is not None and previous != original:
                raise ValueError(
                    f"Conflicting SequenceIDs entry {internal_id} at "
                    f"{path}:{line_number}"
                )
            records[internal_id] = original
    return records


def parse_collinearity_references(
    path: Path,
) -> tuple[Counter[str], dict[str, int]]:
    """Count every internal ID referenced by an MCScanX collinearity file."""

    references: Counter[str] = Counter()
    pair_lines = 0
    block_headers = 0
    with path.open() as handle:
        for raw_line in handle:
            if raw_line.startswith("## Alignment"):
                block_headers += 1
                continue
            if raw_line.startswith("#"):
                continue
            fields = raw_line.rstrip("\n").split("\t")
            if len(fields) < 3:
                continue
            references[fields[1].strip()] += 1
            references[fields[2].strip()] += 1
            pair_lines += 1
    return references, {
        "block_headers": block_headers,
        "pair_lines": pair_lines,
    }


def audit_mcscanx_identity(
    *,
    january_sequence_ids: Path,
    march_sequence_ids: Path,
    january_gff: Path,
    march_gff: Path,
    collinearity: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Test whether January internal IDs retain meaning in March coordinates."""

    january_ids = parse_sequence_ids(january_sequence_ids)
    march_ids = parse_sequence_ids(march_sequence_ids)
    january_coordinates = parse_mcscanx_gff(january_gff)
    march_coordinates = parse_mcscanx_gff(march_gff)
    references, collinearity_summary = parse_collinearity_references(collinearity)

    rows: list[dict[str, Any]] = []
    for internal_id in sorted(references):
        old_coordinate = january_coordinates.get(internal_id)
        new_coordinate = march_coordinates.get(internal_id)
        old_name = january_ids.get(internal_id)
        new_name = march_ids.get(internal_id)
        both_coordinates = old_coordinate is not None and new_coordinate is not None
        if both_coordinates:
            same_chromosome = old_coordinate[0] == new_coordinate[0]
            overlap = (
                max(old_coordinate[1], new_coordinate[1])
                <= min(old_coordinate[2], new_coordinate[2])
                if same_chromosome
                else False
            )
            exact_coordinate = old_coordinate == new_coordinate
        else:
            same_chromosome = False
            overlap = False
            exact_coordinate = False
        rows.append(
            {
                "internal_id": internal_id,
                "collinearity_reference_count": int(references[internal_id]),
                "january_original_id": old_name or "",
                "march_original_id": new_name or "",
                "same_original_text": bool(
                    old_name is not None and old_name == new_name
                ),
                "same_original_first_token": bool(
                    old_name is not None
                    and new_name is not None
                    and old_name.split()[0] == new_name.split()[0]
                ),
                "january_gff_present": old_coordinate is not None,
                "march_gff_present": new_coordinate is not None,
                "january_chromosome": old_coordinate[0] if old_coordinate else "",
                "january_start": old_coordinate[1] if old_coordinate else np.nan,
                "january_end": old_coordinate[2] if old_coordinate else np.nan,
                "march_chromosome": new_coordinate[0] if new_coordinate else "",
                "march_start": new_coordinate[1] if new_coordinate else np.nan,
                "march_end": new_coordinate[2] if new_coordinate else np.nan,
                "same_chromosome": same_chromosome,
                "coordinate_overlap": overlap,
                "exact_coordinate": exact_coordinate,
            }
        )
    detail = pd.DataFrame(rows)
    both = detail["january_gff_present"] & detail["march_gff_present"]
    both_count = int(both.sum())
    summary = {
        **collinearity_summary,
        "unique_referenced_ids": len(detail),
        "january_sequence_ids": len(january_ids),
        "march_sequence_ids": len(march_ids),
        "january_gff_ids": len(january_coordinates),
        "march_gff_ids": len(march_coordinates),
        "referenced_ids_in_january_gff": int(detail["january_gff_present"].sum()),
        "referenced_ids_in_march_gff": int(detail["march_gff_present"].sum()),
        "referenced_ids_in_both_gffs": both_count,
        "same_original_text": int(detail["same_original_text"].sum()),
        "same_original_first_token": int(detail["same_original_first_token"].sum()),
        "same_chromosome": int((detail["same_chromosome"] & both).sum()),
        "coordinate_overlap": int((detail["coordinate_overlap"] & both).sum()),
        "coordinate_overlap_fraction": (
            float((detail["coordinate_overlap"] & both).sum() / both_count)
            if both_count
            else 0.0
        ),
        "exact_coordinate": int((detail["exact_coordinate"] & both).sum()),
        "exact_coordinate_fraction": (
            float((detail["exact_coordinate"] & both).sum() / both_count)
            if both_count
            else 0.0
        ),
    }
    return detail, summary


def contig_gene_counts(
    genes: pd.DataFrame,
) -> dict[tuple[str, str], int]:
    """Return the number of ELSA genes on every sample-contig pair."""

    counts = genes.groupby(["sample_id", "contig_id"], sort=False).size()
    return {
        (str(sample_id), str(contig_id)): int(count)
        for (sample_id, contig_id), count in counts.items()
    }


def load_taxonomy_table(
    path: Path, sample_ids: Iterable[str]
) -> tuple[dict[str, str], dict[str, Any]]:
    """Load and validate the sample-to-cohort-label mapping."""

    table = pd.read_csv(path, sep="\t", dtype=str)
    required = {"sample_id", "species"}
    missing_columns = required - set(table.columns)
    if missing_columns:
        raise ValueError(
            f"Taxonomy table lacks columns: {sorted(missing_columns)}"
        )
    if table["sample_id"].duplicated().any():
        duplicates = (
            table.loc[table["sample_id"].duplicated(), "sample_id"]
            .drop_duplicates()
            .head()
            .tolist()
        )
        raise ValueError(f"Taxonomy table has duplicate sample IDs: {duplicates}")
    mapping = dict(zip(table["sample_id"], table["species"]))
    requested = sorted(set(str(sample_id) for sample_id in sample_ids))
    missing_samples = sorted(set(requested) - set(mapping))
    if missing_samples:
        raise ValueError(
            f"Taxonomy table lacks {len(missing_samples)} samples: "
            f"{missing_samples[:5]}"
        )
    selected_counts = Counter(mapping[sample_id] for sample_id in requested)
    return mapping, {
        "table_rows": len(table),
        "selected_samples": len(requested),
        "selected_label_counts": {
            str(label): int(count)
            for label, count in sorted(selected_counts.items())
        },
        "unselected_table_rows": int(len(table) - len(requested)),
    }


def summarize_blocks(
    blocks: pd.DataFrame,
    taxonomy: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Summarize a normalized block table and its cohort composition."""

    span_a = blocks["end_a"] - blocks["start_a"] + 1
    span_b = blocks["end_b"] - blocks["start_b"] + 1
    samples = sorted(set(blocks["genome_a"]) | set(blocks["genome_b"]))
    genome_pairs = blocks[["genome_a", "genome_b"]].drop_duplicates()
    summary: dict[str, Any] = {
        "blocks": len(blocks),
        "samples": len(samples),
        "genome_pairs_with_blocks": len(genome_pairs),
        "possible_genome_pairs": len(samples) * (len(samples) - 1) // 2,
        "anchor_count": {
            "total": int(blocks["n_anchors"].sum()),
            "minimum": int(blocks["n_anchors"].min()),
            "median": float(blocks["n_anchors"].median()),
            "mean": float(blocks["n_anchors"].mean()),
            "maximum": int(blocks["n_anchors"].max()),
        },
        "span_a": {
            "minimum": int(span_a.min()),
            "median": float(span_a.median()),
            "mean": float(span_a.mean()),
            "maximum": int(span_a.max()),
        },
        "span_b": {
            "minimum": int(span_b.min()),
            "median": float(span_b.median()),
            "mean": float(span_b.mean()),
            "maximum": int(span_b.max()),
        },
    }
    if taxonomy is not None:
        missing = sorted(set(samples) - set(taxonomy))
        if missing:
            raise ValueError(
                f"Cohort mapping lacks {len(missing)} block samples: {missing[:5]}"
            )
        label_a = blocks["genome_a"].map(taxonomy)
        label_b = blocks["genome_b"].map(taxonomy)
        cross = label_a != label_b
        pair_labels = pd.DataFrame(
            {
                "left": np.minimum(label_a, label_b),
                "right": np.maximum(label_a, label_b),
            }
        )
        pair_label_counts = (
            pair_labels.value_counts(sort=False).sort_index()
        )
        summary["cohort"] = {
            "sample_label_counts": {
                str(label): int(count)
                for label, count in sorted(
                    Counter(taxonomy[sample] for sample in samples).items()
                )
            },
            "cross_label_blocks": int(cross.sum()),
            "cross_label_fraction": float(cross.mean()),
            "same_label_blocks": int((~cross).sum()),
            "same_label_fraction": float((~cross).mean()),
            "block_label_pair_counts": {
                f"{left}__{right}": int(count)
                for (left, right), count in pair_label_counts.items()
            },
        }
    return summary


def selected_directory_provenance(
    directory: Path,
    sample_ids: Sequence[str],
    suffix: str,
) -> dict[str, Any]:
    """Hash every selected per-sample file in a directory."""

    members: list[dict[str, Any]] = []
    aggregate = hashlib.sha256()
    for sample_id in sorted(set(sample_ids)):
        path = directory / f"{sample_id}{suffix}"
        if not path.is_file():
            raise FileNotFoundError(f"Missing selected input file: {path}")
        digest = sha256_file(path)
        size = path.stat().st_size
        members.append(
            {
                "sample_id": sample_id,
                "name": path.name,
                "bytes": size,
                "sha256": digest,
            }
        )
        aggregate.update(path.name.encode())
        aggregate.update(b"\0")
        aggregate.update(digest.encode())
        aggregate.update(b"\0")
    return {
        "directory": str(directory.resolve()),
        "member_count": len(members),
        "aggregate_sha256": aggregate.hexdigest(),
        "members": members,
    }


def command_output(command: Sequence[str], cwd: Path) -> str | None:
    """Run a read-only provenance command and return its stripped output."""

    try:
        completed = subprocess.run(
            list(command),
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def runtime_provenance(repository_root: Path) -> dict[str, Any]:
    """Record software and repository state for the audit run."""

    git_commit = command_output(
        ["git", "rev-parse", "HEAD"], cwd=repository_root
    )
    git_status = command_output(
        ["git", "status", "--porcelain"], cwd=repository_root
    )
    return {
        "audit_schema_version": AUDIT_SCHEMA_VERSION,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "biopython": getattr(sys.modules.get("Bio"), "__version__", "unknown"),
        "git_commit": git_commit,
        "git_dirty": bool(git_status),
        "git_status_entry_count": (
            len(git_status.splitlines()) if git_status else 0
        ),
    }


def hash_named_inputs(paths: Mapping[str, Path]) -> dict[str, Any]:
    """Hash required file inputs and fail closed on missing paths."""

    output: dict[str, Any] = {}
    for name, path in sorted(paths.items()):
        if not path.is_file():
            raise FileNotFoundError(f"Missing required input {name}: {path}")
        output[name] = {
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return output


def verify_canonical_hashes(input_hashes: Mapping[str, Mapping[str, Any]]) -> None:
    """Fail closed when an audited canonical input differs from the sealed hash."""

    missing_expectations = sorted(
        set(input_hashes) - set(EXPECTED_CANONICAL_SHA256)
    )
    if missing_expectations:
        raise ValueError(
            f"Canonical hash expectations absent for: {missing_expectations}"
        )
    mismatches = {
        name: {
            "expected": EXPECTED_CANONICAL_SHA256[name],
            "observed": str(metadata["sha256"]),
        }
        for name, metadata in input_hashes.items()
        if metadata["sha256"] != EXPECTED_CANONICAL_SHA256[name]
    }
    if mismatches:
        raise ValueError(
            "Canonical input hash mismatch: "
            + json.dumps(mismatches, sort_keys=True)
        )


def format_rate(rate: float | None, digits: int = 3) -> str:
    """Format a rate for compact tabular and narrative output."""

    if rate is None or not np.isfinite(rate):
        return ""
    return f"{100 * rate:.{digits}f}%"


def build_claim_ledger(results: Mapping[str, Any]) -> pd.DataFrame:
    """Classify every March headline claim against the audited evidence."""

    legacy_ecoli = results["operon_metrics"]["ecoli"]["legacy_published_evaluator"]
    legacy_cross = results["operon_metrics"]["cross_species"][
        "legacy_published_evaluator"
    ]
    corrected_ecoli = results["operon_metrics"]["ecoli"]["corrected"][
        "pair_micro"
    ]
    corrected_cross = results["operon_metrics"]["cross_species"]["corrected"][
        "pair_micro"
    ]
    ecoli_blocks = results["block_statistics"]["ecoli"]
    cross_blocks = results["block_statistics"]["cross_species"]
    mcscanx = results["mcscanx_identity"]
    sequence = results["sequence_integrity"]

    rows = [
        {
            "claim_id": "ecoli_operon_strict_recall",
            "published_claim": "98.9% strict E. coli operon recall",
            "reported_value": "10068/10182 (98.9%)",
            "audited_numeric_result": (
                f"{legacy_ecoli['strict_hits']}/{legacy_ecoli['denominator']} "
                f"({format_rate(legacy_ecoli['strict_rate'])})"
            ),
            "corrected_result": (
                f"{corrected_ecoli['strict50']['hits']}/"
                f"{corrected_ecoli['strict50']['total']} "
                f"({format_rate(corrected_ecoli['strict50']['rate'])})"
            ),
            "status": "INVALID_ENDPOINT",
            "basis": (
                "The published number reproduces under sample-global ranks and "
                "a genome-pair-only block filter. The denominator expands "
                "unique loci into pair rows, and span coverage rewards broad blocks."
            ),
            "evidence_artifact": "operon_pair_metrics_ecoli.tsv.gz",
        },
        {
            "claim_id": "cross_species_operon_strict_recall",
            "published_claim": "98.9% cross-species strict operon recall",
            "reported_value": "98.9%",
            "audited_numeric_result": format_rate(legacy_cross["strict_rate"]),
            "corrected_result": format_rate(
                corrected_cross["strict50"]["rate"]
            ),
            "status": "INVALID_ENDPOINT",
            "basis": (
                "The same pseudoreplicated ground truth and positional endpoint "
                "drive this result; exact contig-local evaluation is reported separately."
            ),
            "evidence_artifact": "operon_pair_metrics_cross_species.tsv.gz",
        },
        {
            "claim_id": "search_recall_at_50",
            "published_claim": "100% search recall@50",
            "reported_value": "100%",
            "audited_numeric_result": "",
            "corrected_result": "Pending exact-locus search regeneration",
            "status": "INVALID_ENDPOINT",
            "basis": (
                "The saved evaluator records target-genome presence. It lacks "
                "target-contig and target-interval correspondence."
            ),
            "evidence_artifact": "search_benchmark_status.json",
        },
        {
            "claim_id": "ortholog_anchor_precision",
            "published_claim": "97.1% chained-anchor orthogroup precision",
            "reported_value": "97.1%",
            "audited_numeric_result": (
                f"{sequence['exact_sequence_matches']}/"
                f"{sequence['n_sequences']} exact input matches"
            ),
            "corrected_result": "Pending clean-input anchor regeneration",
            "status": "INVALID_PROVENANCE",
            "basis": (
                "Persisted OrthoFinder inputs reproduce a one-base coordinate "
                "slice error, and canonical blocks omit row-level anchors."
            ),
            "evidence_artifact": "sequence_integrity_summary.tsv",
        },
        {
            "claim_id": "plm_distant_homology_advantage",
            "published_claim": "PLM anchors recover distant homologs missed by sequence search",
            "reported_value": "74.7% for one reported subset",
            "audited_numeric_result": "",
            "corrected_result": "Pending clean orthology and anchor regeneration",
            "status": "UNSUPPORTED",
            "basis": (
                "The reported subset lacks reproducible row-level provenance, "
                "and its orthology reference derives from corrupted inputs."
            ),
            "evidence_artifact": "audit_results.json",
        },
        {
            "claim_id": "ecoli_block_count",
            "published_claim": "20,117 E. coli syntenic blocks",
            "reported_value": "20,117",
            "audited_numeric_result": str(ecoli_blocks["blocks"]),
            "corrected_result": str(ecoli_blocks["blocks"]),
            "status": "REPRODUCED_ARTIFACT",
            "basis": "The immutable canonical block table contains this row count.",
            "evidence_artifact": "audit_results.json",
        },
        {
            "claim_id": "cross_species_block_count",
            "published_claim": "80,225 cross-species syntenic blocks",
            "reported_value": "80,225",
            "audited_numeric_result": str(cross_blocks["blocks"]),
            "corrected_result": str(cross_blocks["blocks"]),
            "status": "REPRODUCED_ARTIFACT",
            "basis": "The immutable canonical block table contains this row count.",
            "evidence_artifact": "audit_results.json",
        },
        {
            "claim_id": "cross_label_block_fraction",
            "published_claim": "71.7% of cross-species blocks are cross-genus",
            "reported_value": "71.7%",
            "audited_numeric_result": format_rate(
                cross_blocks["cohort"]["cross_label_fraction"]
            ),
            "corrected_result": format_rate(
                cross_blocks["cohort"]["cross_label_fraction"]
            ),
            "status": "REPRODUCED_COHORT_STATISTIC",
            "basis": (
                "The fraction reproduces from cohort labels and block endpoints; "
                "its magnitude depends on the selected cohort and pair opportunity structure."
            ),
            "evidence_artifact": "audit_results.json",
        },
        {
            "claim_id": "block_count_ratio_vs_mcscanx",
            "published_claim": "2.85x more blocks than MCScanX; 3.85x more cross-genus",
            "reported_value": "2.85x; 3.85x",
            "audited_numeric_result": (
                f"{cross_blocks['blocks']} ELSA blocks; "
                f"{mcscanx['block_headers']} MCScanX headers"
            ),
            "corrected_result": "A sealed matched-cohort rerun is required",
            "status": "INVALID_COMPARISON",
            "basis": (
                "The saved MCScanX output uses a different cohort and identifier "
                "mapping. Block count also measures segmentation, not accuracy."
            ),
            "evidence_artifact": "mcscanx_identity_audit.tsv.gz",
        },
        {
            "claim_id": "mcscanx_operon_recall",
            "published_claim": "80.3% MCScanX operon recall",
            "reported_value": "80.3%",
            "audited_numeric_result": (
                f"{mcscanx['exact_coordinate']} exact coordinates among "
                f"{mcscanx['referenced_ids_in_both_gffs']} shared referenced IDs"
            ),
            "corrected_result": "A sealed matched-cohort rerun is required",
            "status": "INVALID_PROVENANCE",
            "basis": (
                "The recall calculation paired January collinearity identifiers "
                "with March coordinates whose identifier meanings changed."
            ),
            "evidence_artifact": "mcscanx_identity_audit.tsv.gz",
        },
        {
            "claim_id": "concave_gap_segmentation",
            "published_claim": "Concave gap scoring yields 59.3% more blocks",
            "reported_value": "59.3%",
            "audited_numeric_result": "",
            "corrected_result": "Computational observation retained; endpoint value unresolved",
            "status": "QUALIFIED",
            "basis": (
                "The comparison supports a segmentation change. Operon span "
                "coverage supplies weak biological model selection."
            ),
            "evidence_artifact": "audit_results.json",
        },
    ]
    return pd.DataFrame(rows)


def render_report(results: Mapping[str, Any], ledger: pd.DataFrame) -> str:
    """Render the machine-readable audit into a concise Markdown report."""

    locus = results["ground_truth_integrity"]
    sequence = results["sequence_integrity"]
    mcscanx = results["mcscanx_identity"]
    ecoli_metrics = results["operon_metrics"]["ecoli"]
    cross_metrics = results["operon_metrics"]["cross_species"]
    nulls = results["null_controls"]["ecoli"]
    blocks = results["block_statistics"]

    lines = [
        "# ELSA Benchmark Audit — 2026-07-19",
        "",
        "## Verdict",
        "",
        (
            "The canonical ELSA block artifacts reproduce at their reported scale. "
            "The current benchmark artifacts establish an engineering result: ELSA "
            "constructs many long collinear chains across the selected genomes."
        ),
        "",
        (
            "The headline biological validation claims require new sealed runs. "
            "Three independent provenance failures drive that verdict: the operon "
            "endpoint uses pseudoreplicated positional spans, the persisted "
            "orthology inputs carry a coordinate-slice translation error, and the "
            "MCScanX recall calculation joins identifier spaces from different runs."
        ),
        "",
        "## Audited results",
        "",
        "| Result | Audit value | Interpretation |",
        "|---|---:|---|",
        (
            f"| E. coli blocks | {blocks['ecoli']['blocks']:,} | "
            "Canonical artifact row count reproduced |"
        ),
        (
            f"| Cross-species blocks | {blocks['cross_species']['blocks']:,} | "
            "Canonical artifact row count reproduced |"
        ),
        (
            "| Cross-label blocks | "
            f"{format_rate(blocks['cross_species']['cohort']['cross_label_fraction'], 2)} | "
            "Cohort statistic reproduced |"
        ),
        (
            "| Published-style E. coli strict span recall | "
            f"{format_rate(ecoli_metrics['legacy_published_evaluator']['strict_rate'], 3)} | "
            "Numeric result reproduced under the published evaluator |"
        ),
        (
            "| Exact-contig, contig-local E. coli 50% span recall | "
            f"{format_rate(ecoli_metrics['corrected']['pair_micro']['strict50']['rate'], 3)} | "
            "Corrected positional endpoint |"
        ),
        (
            "| Exact-contig, contig-local E. coli 100% span recall | "
            f"{format_rate(ecoli_metrics['corrected']['pair_micro']['strict100']['rate'], 3)} | "
            "Full-span containment |"
        ),
        (
            "| Exact-contig, contig-local cross-species 50% span recall | "
            f"{format_rate(cross_metrics['corrected']['pair_micro']['strict50']['rate'], 3)} | "
            "Corrected positional endpoint |"
        ),
        (
            f"| Unique operon loci | {locus['source_unique_loci']:,} | "
            f"Expanded to {locus['pairwise_rows']:,} pair rows |"
        ),
        (
            "| Exact embedded/orthology-input sequence matches | "
            f"{sequence['exact_sequence_matches']:,}/{sequence['n_sequences']:,} "
            f"({format_rate(sequence['exact_sequence_match_fraction'], 3)}) | "
            "Orthology input provenance failure |"
        ),
        (
            "| Historical coordinate-slice reproduction | "
            f"{sequence['buggy_slice_matches']:,}/{sequence['n_sequences']:,} "
            f"({format_rate(sequence['buggy_slice_match_fraction'], 3)}) | "
            "Root cause reproduced directly |"
        ),
        (
            "| MCScanX exact coordinate identity across joined runs | "
            f"{mcscanx['exact_coordinate']:,}/"
            f"{mcscanx['referenced_ids_in_both_gffs']:,} "
            f"({format_rate(mcscanx['exact_coordinate_fraction'], 3)}) | "
            "Identifier spaces are incompatible |"
        ),
        "",
        "## Ground-truth integrity",
        "",
        (
            f"The persisted denominator contains {locus['pairwise_rows']:,} pair "
            f"rows derived from {locus['source_unique_loci']:,} unique "
            "operon-by-genome loci. The integrity table records definition "
            "completeness, gene order, strand coherence, coordinate multiplicity, "
            "contig-local rank, sample-global rank, and retained caller flags."
        ),
        "",
        (
            f"Independent reconstruction joined {locus['joined_loci']:,} source "
            f"loci. {locus['rank_shifted_loci']:,} loci have different sample-global "
            "and contig-local coordinates."
        ),
        "",
        "## Boundary-aware metrics and controls",
        "",
        (
            "The corrected tables require exact contig identity and evaluate each "
            "locus pair against the same ELSA block. They persist 50% coverage, "
            "100% containment, boundary precision, IoU, smallest containing span, "
            "and anchor density for every pair row."
        ),
        "",
        (
            "For the E. coli block set, the target-locus shuffle null has mean "
            f"50% recall {format_rate(nulls['distributions']['shuffle_target_loci_within_genome_pair']['strict50_mean'], 3)} "
            "and observed-minus-null effect "
            f"{format_rate(nulls['distributions']['shuffle_target_loci_within_genome_pair']['strict50_observed_minus_null_mean'], 3)}. "
            "The whole-contig ceiling reaches "
            f"{format_rate(nulls['whole_contig_ceiling']['strict50'], 1)}."
        ),
        "",
        "## Claim ledger",
        "",
        "| Claim | Status | Audited result | Corrected disposition |",
        "|---|---|---|---|",
    ]
    for row in ledger.itertuples(index=False):
        lines.append(
            f"| {row.published_claim} | {row.status} | "
            f"{row.audited_numeric_result or '—'} | "
            f"{row.corrected_result or '—'} |"
        )
    lines.extend(
        [
            "",
            "## Required sealed follow-up",
            "",
            "1. Regenerate orthology from the exact retained caller translations.",
            "2. Persist every seed anchor and rerun anchor precision by reciprocal status, sequence support, identity bin, cohort pair, and locus integrity.",
            "3. Regenerate search results with exact target contig and interval fields, then score locus recall, AP, MRR, NDCG, and false loci.",
            "4. Rerun MCScanX from the same accession cohort and sealed protein/coordinate inputs used by ELSA.",
            "5. Add a sequence-only anchor baseline and a divergence-stratified benchmark before making a PLM-specific biological advantage claim.",
            "",
            "## Reproducibility artifacts",
            "",
            "- `audit_results.json`: hashes, versions, summaries, null distributions, and verdict inputs.",
            "- `claim_ledger.tsv`: one status row per headline claim.",
            "- `operon_locus_integrity.tsv`: one row per persisted unique locus.",
            "- `operon_pair_metrics_*.tsv.gz`: corrected row-level boundary metrics.",
            "- `null_controls_*.tsv`: deterministic permutation results.",
            "- `sequence_integrity_detail.tsv.gz`: exact per-gene provenance comparisons, expressed as metrics and flags.",
            "- `mcscanx_identity_audit.tsv.gz`: identifier and coordinate compatibility evidence.",
            "",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    """Construct the explicit-path command-line interface."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operon-definitions", required=True, type=Path)
    parser.add_argument("--pairwise-ground-truth", required=True, type=Path)
    parser.add_argument("--ecoli-genes", required=True, type=Path)
    parser.add_argument("--ecoli-blocks", required=True, type=Path)
    parser.add_argument("--cross-genes", required=True, type=Path)
    parser.add_argument("--cross-blocks", required=True, type=Path)
    parser.add_argument("--annotation-dir", required=True, type=Path)
    parser.add_argument("--prodigal-dir", required=True, type=Path)
    parser.add_argument("--genome-dir", required=True, type=Path)
    parser.add_argument("--orthofinder-input-dir", required=True, type=Path)
    parser.add_argument("--taxonomy-table", required=True, type=Path)
    parser.add_argument("--january-sequence-ids", required=True, type=Path)
    parser.add_argument("--march-sequence-ids", required=True, type=Path)
    parser.add_argument("--january-mcscanx-gff", required=True, type=Path)
    parser.add_argument("--march-mcscanx-gff", required=True, type=Path)
    parser.add_argument("--mcscanx-collinearity", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--permutations", type=int, default=100)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260719)
    return parser


def validate_arguments(args: argparse.Namespace) -> None:
    """Validate scalar and directory arguments before reading inputs."""

    if args.permutations < 100:
        raise ValueError("--permutations must be at least 100")
    if args.bootstrap_replicates < 100:
        raise ValueError("--bootstrap-replicates must be at least 100")
    for name in (
        "annotation_dir",
        "prodigal_dir",
        "genome_dir",
        "orthofinder_input_dir",
    ):
        path = getattr(args, name)
        if not path.is_dir():
            raise FileNotFoundError(f"Missing required directory {name}: {path}")
    output = args.output_dir.resolve()
    canonical_paths = [
        getattr(args, name).resolve()
        for name in (
            "annotation_dir",
            "prodigal_dir",
            "genome_dir",
            "orthofinder_input_dir",
        )
    ]
    if any(output == path or output.is_relative_to(path) for path in canonical_paths):
        raise ValueError("Output directory must be separate from canonical inputs")


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    """Execute the complete read-only audit and persist derived artifacts."""

    validate_arguments(args)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    file_inputs = {
        "operon_definitions": args.operon_definitions,
        "pairwise_ground_truth": args.pairwise_ground_truth,
        "ecoli_genes": args.ecoli_genes,
        "ecoli_blocks": args.ecoli_blocks,
        "cross_genes": args.cross_genes,
        "cross_blocks": args.cross_blocks,
        "taxonomy_table": args.taxonomy_table,
        "january_sequence_ids": args.january_sequence_ids,
        "march_sequence_ids": args.march_sequence_ids,
        "january_mcscanx_gff": args.january_mcscanx_gff,
        "march_mcscanx_gff": args.march_mcscanx_gff,
        "mcscanx_collinearity": args.mcscanx_collinearity,
    }
    input_hashes = hash_named_inputs(file_inputs)
    verify_canonical_hashes(input_hashes)

    definitions = parse_operon_definitions(args.operon_definitions)
    pairwise_gt = pd.read_csv(args.pairwise_ground_truth, sep="\t")
    ecoli_genes = prepare_gene_coordinates(args.ecoli_genes)
    cross_genes = prepare_gene_coordinates(args.cross_genes)
    sample_ids = sorted(ecoli_genes["sample_id"].unique())
    cross_sample_ids = sorted(cross_genes["sample_id"].unique())

    prodigal_records, prodigal_load_summary = load_prodigal_records(
        args.prodigal_dir, sample_ids
    )
    orthofinder_sequences = load_named_sequences(
        args.orthofinder_input_dir, sample_ids
    )
    genome_sequences = load_genome_sequences(args.genome_dir, sample_ids)
    sequence_detail, sequence_summary = audit_sequence_integrity(
        ecoli_genes,
        prodigal_records,
        orthofinder_sequences,
        genome_sequences,
    )
    del orthofinder_sequences
    del genome_sequences

    reconstructed_loci, reconstruction_summary = reconstruct_operon_loci(
        ecoli_genes,
        definitions,
        args.annotation_dir,
        prodigal_records,
    )
    source_loci = extract_unique_loci(pairwise_gt)
    translated_loci, translation_summary = (
        translate_source_loci_to_local_coordinates(
            source_loci,
            ecoli_genes,
            definitions,
            reconstructed_loci,
        )
    )
    locus_integrity_all, locus_merge_summary = merge_source_and_reconstructed_loci(
        source_loci, reconstructed_loci
    )
    translated_for_integrity = translated_loci.rename(
        columns={
            column: f"source_translated_{column}"
            for column in translated_loci.columns
            if column not in {"operon_id", "sample_id"}
        }
    )
    locus_integrity_all = locus_integrity_all.merge(
        translated_for_integrity,
        on=["operon_id", "sample_id"],
        how="left",
        validate="one_to_one",
    )
    locus_integrity = locus_integrity_all[
        locus_integrity_all["source_present"]
    ].copy()
    corrected_pairwise_gt = attach_corrected_loci_to_pairs(
        pairwise_gt, translated_loci
    )
    stored_pairs = normalize_pair_rows(corrected_pairwise_gt, corrected=False)
    corrected_pairs = normalize_pair_rows(corrected_pairwise_gt, corrected=True)

    ecoli_blocks = normalize_blocks(pd.read_csv(args.ecoli_blocks))
    cross_blocks = normalize_blocks(pd.read_csv(args.cross_blocks))
    taxonomy, taxonomy_summary = load_taxonomy_table(
        args.taxonomy_table, cross_sample_ids
    )
    ecoli_block_summary = summarize_blocks(ecoli_blocks)
    cross_block_summary = summarize_blocks(cross_blocks, taxonomy)

    ecoli_block_index = build_block_index(ecoli_blocks)
    cross_block_index = build_block_index(cross_blocks)
    ecoli_pair_metrics = evaluate_pair_metrics(
        corrected_pairs, ecoli_block_index
    )
    cross_pair_metrics = evaluate_pair_metrics(
        corrected_pairs, cross_block_index
    )
    ecoli_metric_summary = summarize_pair_metrics(
        ecoli_pair_metrics,
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.seed,
    )
    cross_metric_summary = summarize_pair_metrics(
        cross_pair_metrics,
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.seed + 10_000,
    )
    ecoli_legacy = legacy_span_metrics(stored_pairs, ecoli_blocks)
    cross_legacy = legacy_span_metrics(stored_pairs, cross_blocks)

    ecoli_sizes = contig_gene_counts(ecoli_genes)
    ecoli_null_detail, ecoli_null_summary = run_null_controls(
        corrected_pairs,
        ecoli_blocks,
        ecoli_sizes,
        permutations=args.permutations,
        seed=args.seed + 20_000,
    )
    cross_null_detail, cross_null_summary = run_null_controls(
        corrected_pairs,
        cross_blocks,
        contig_gene_counts(cross_genes),
        permutations=args.permutations,
        seed=args.seed + 30_000,
    )

    mcscanx_detail, mcscanx_summary = audit_mcscanx_identity(
        january_sequence_ids=args.january_sequence_ids,
        march_sequence_ids=args.march_sequence_ids,
        january_gff=args.january_mcscanx_gff,
        march_gff=args.march_mcscanx_gff,
        collinearity=args.mcscanx_collinearity,
    )

    repository_root = Path(__file__).resolve().parents[2]
    directory_hashes = {
        "annotations": selected_directory_provenance(
            args.annotation_dir, sample_ids, ".gff"
        ),
        "prodigal_translations": selected_directory_provenance(
            args.prodigal_dir, sample_ids, ".faa"
        ),
        "genomes": selected_directory_provenance(
            args.genome_dir, sample_ids, ".fna"
        ),
        "orthofinder_inputs": selected_directory_provenance(
            args.orthofinder_input_dir, sample_ids, ".faa"
        ),
    }
    search_status = {
        "status": "PENDING_EXACT_LOCUS_REGENERATION",
        "reason": (
            "The canonical saved search CSV retains genome-level counts and "
            "cannot support exact-locus ranking metrics."
        ),
        "required_columns": [
            "query_id",
            "target_genome",
            "target_contig",
            "target_start",
            "target_end",
            "rank",
            "score",
        ],
        "required_metrics": [
            "exact_locus_recall",
            "average_precision",
            "mean_reciprocal_rank",
            "ndcg",
            "false_loci_per_query",
        ],
    }
    results = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "parameters": {
            "permutations": args.permutations,
            "bootstrap_replicates": args.bootstrap_replicates,
            "seed": args.seed,
        },
        "runtime": runtime_provenance(repository_root),
        "canonical_hashes_verified": True,
        "input_files": input_hashes,
        "input_directories": directory_hashes,
        "cohort": {
            "ecoli_samples": len(sample_ids),
            "cross_species_samples": len(cross_sample_ids),
            "taxonomy": taxonomy_summary,
        },
        "sequence_integrity": {
            **prodigal_load_summary,
            **sequence_summary,
        },
        "ground_truth_integrity": {
            "operon_definitions": len(definitions),
            "pairwise_rows": len(pairwise_gt),
            **reconstruction_summary,
            **locus_merge_summary,
            **translation_summary,
        },
        "block_statistics": {
            "ecoli": ecoli_block_summary,
            "cross_species": cross_block_summary,
        },
        "operon_metrics": {
            "ecoli": {
                "legacy_published_evaluator": ecoli_legacy,
                "corrected": ecoli_metric_summary,
            },
            "cross_species": {
                "legacy_published_evaluator": cross_legacy,
                "corrected": cross_metric_summary,
            },
        },
        "null_controls": {
            "ecoli": ecoli_null_summary,
            "cross_species": cross_null_summary,
        },
        "mcscanx_identity": mcscanx_summary,
        "search_benchmark": search_status,
    }

    sequence_summary_frame = pd.DataFrame(
        [
            {
                key: value
                for key, value in results["sequence_integrity"].items()
                if not isinstance(value, (dict, list))
            }
        ]
    )
    atomic_write_tsv(
        output_dir / "sequence_integrity_summary.tsv",
        sequence_summary_frame,
    )
    atomic_write_tsv_gz(
        output_dir / "sequence_integrity_detail.tsv.gz",
        sequence_detail,
    )
    atomic_write_tsv(
        output_dir / "operon_locus_integrity.tsv",
        locus_integrity,
    )
    atomic_write_tsv(
        output_dir / "operon_loci_reconstructed_all.tsv",
        locus_integrity_all,
    )
    atomic_write_tsv_gz(
        output_dir / "operon_pair_metrics_ecoli.tsv.gz",
        ecoli_pair_metrics,
    )
    atomic_write_tsv_gz(
        output_dir / "operon_pair_metrics_cross_species.tsv.gz",
        cross_pair_metrics,
    )
    atomic_write_tsv(
        output_dir / "null_controls_ecoli.tsv",
        ecoli_null_detail,
    )
    atomic_write_tsv(
        output_dir / "null_controls_cross_species.tsv",
        cross_null_detail,
    )
    atomic_write_tsv_gz(
        output_dir / "mcscanx_identity_audit.tsv.gz",
        mcscanx_detail,
    )
    atomic_write_json(
        output_dir / "search_benchmark_status.json",
        search_status,
    )

    ledger = build_claim_ledger(results)
    atomic_write_tsv(output_dir / "claim_ledger.tsv", ledger)
    atomic_write_json(output_dir / "audit_results.json", results)
    atomic_write_text(
        output_dir.parent / "BENCHMARK_AUDIT_2026-07-19.md",
        render_report(results, ledger),
    )
    return results


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""

    parser = build_parser()
    args = parser.parse_args(argv)
    run_audit(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
