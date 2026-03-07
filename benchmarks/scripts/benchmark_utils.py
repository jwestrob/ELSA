"""Shared helpers for benchmark scripts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BENCHMARKS_DIR = ROOT / "benchmarks"
DEFAULT_SAMPLES = BENCHMARKS_DIR / "data" / "enterobacteriaceae" / "samples.tsv"


def load_species_map(samples_path: Path | None = None) -> dict[str, str]:
    """Load sample_id -> species mapping."""
    path = samples_path or DEFAULT_SAMPLES
    samples = pd.read_csv(path, sep="\t")
    return dict(zip(samples["sample_id"], samples["species"]))


def classify_species(genome_id: str, species_map: dict[str, str]) -> str:
    """Classify a genome using the provided species map."""
    if genome_id not in species_map:
        raise KeyError(f"Genome {genome_id} missing from species map.")
    return species_map[genome_id]


def attach_species(
    df: pd.DataFrame,
    species_map: dict[str, str],
    query_col: str = "query_genome",
    target_col: str = "target_genome",
) -> pd.DataFrame:
    """Attach query/target species columns using the species map."""
    df = df.copy()
    df["query_species"] = df[query_col].map(species_map)
    df["target_species"] = df[target_col].map(species_map)
    missing = df["query_species"].isna().sum() + df["target_species"].isna().sum()
    if missing:
        raise ValueError(f"Missing species labels for {missing} rows.")
    return df


def species_pair(a: str, b: str) -> str:
    """Stable, unordered species-pair label."""
    return "-".join(sorted([a, b]))
