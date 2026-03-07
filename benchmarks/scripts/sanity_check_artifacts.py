#!/usr/bin/env python3
"""Sanity-check canonical benchmark artifacts for the manuscript."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "benchmarks"

DEFAULT_SAMPLES = BENCH / "data" / "enterobacteriaceae" / "samples.tsv"
DEFAULT_ELSA_BLOCKS = (
    BENCH / "results" / "cross_species_chain" / "micro_chain" / "micro_chain_blocks.csv"
)
DEFAULT_MCSCANX_BLOCKS = (
    BENCH / "results" / "mcscanx_comparison" / "mcscanx_blocks_v2.csv"
)
DEFAULT_GLM2_BLOCKS = (
    BENCH / "results" / "cross_species_glm2" / "micro_chain" / "micro_chain_blocks.csv"
)

EXPECTED = {
    "elsa": {
        "total_blocks": 76954,
        "clusters": 76724,
        "species_pair_counts": {
            "ecoli-ecoli": 22042,
            "ecoli-klebsiella": 24576,
            "ecoli-salmonella": 22825,
            "klebsiella-klebsiella": 947,
            "klebsiella-salmonella": 5870,
            "salmonella-salmonella": 694,
        },
        "cross_genus_blocks": 53271,
    },
    "mcscanx": {
        "total_blocks": 28196,
        "species_pair_counts": {
            "ecoli-ecoli": 11627,
            "ecoli-klebsiella": 9037,
            "ecoli-salmonella": 4430,
            "klebsiella-klebsiella": 1346,
            "klebsiella-salmonella": 1473,
            "salmonella-salmonella": 283,
        },
        "cross_genus_blocks": 14940,
    },
    "glm2": {
        "total_blocks": 78519,
        "clusters": 31007,
    },
}


def load_species_map(samples_path: Path) -> dict[str, str]:
    samples = pd.read_csv(samples_path, sep="\t")
    return dict(zip(samples["sample_id"], samples["species"]))


def attach_species(df: pd.DataFrame, species_map: dict[str, str]) -> pd.DataFrame:
    df = df.copy()
    df["query_species"] = df["query_genome"].map(species_map)
    df["target_species"] = df["target_genome"].map(species_map)
    missing = df["query_species"].isna().sum() + df["target_species"].isna().sum()
    if missing:
        raise ValueError(f"Missing species labels for {missing} block rows.")
    df["species_pair"] = df.apply(
        lambda r: "-".join(sorted([r.query_species, r.target_species])), axis=1
    )
    df["is_cross_genus"] = df["query_species"] != df["target_species"]
    return df


def check_counts(label: str, df: pd.DataFrame, expected: dict[str, object]) -> list[str]:
    failures: list[str] = []
    if "total_blocks" in expected and len(df) != expected["total_blocks"]:
        failures.append(
            f"{label}: total_blocks {len(df)} != {expected['total_blocks']}"
        )
    if "clusters" in expected:
        clusters = df["cluster_id"].nunique()
        if clusters != expected["clusters"]:
            failures.append(
                f"{label}: clusters {clusters} != {expected['clusters']}"
            )
    if "species_pair_counts" in expected:
        counts = df["species_pair"].value_counts().to_dict()
        for pair, exp in expected["species_pair_counts"].items():
            if counts.get(pair, 0) != exp:
                failures.append(
                    f"{label}: species_pair {pair} {counts.get(pair, 0)} != {exp}"
                )
    if "cross_genus_blocks" in expected:
        cross = int(df["is_cross_genus"].sum())
        if cross != expected["cross_genus_blocks"]:
            failures.append(
                f"{label}: cross_genus_blocks {cross} != {expected['cross_genus_blocks']}"
            )
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sanity-check canonical benchmark artifacts."
    )
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument("--elsa-blocks", type=Path, default=DEFAULT_ELSA_BLOCKS)
    parser.add_argument("--mcscanx-blocks", type=Path, default=DEFAULT_MCSCANX_BLOCKS)
    parser.add_argument("--glm2-blocks", type=Path, default=DEFAULT_GLM2_BLOCKS)
    parser.add_argument("--skip-glm2", action="store_true")
    args = parser.parse_args()

    species_map = load_species_map(args.samples)

    elsa = pd.read_csv(args.elsa_blocks)
    elsa = attach_species(elsa, species_map)
    mcscanx = pd.read_csv(args.mcscanx_blocks)
    mcscanx = attach_species(mcscanx, species_map)

    failures = []
    failures.extend(check_counts("ELSA", elsa, EXPECTED["elsa"]))
    failures.extend(check_counts("MCScanX", mcscanx, EXPECTED["mcscanx"]))

    if not args.skip_glm2 and args.glm2_blocks.exists():
        glm2 = pd.read_csv(args.glm2_blocks)
        glm2 = attach_species(glm2, species_map)
        failures.extend(check_counts("gLM2", glm2, EXPECTED["glm2"]))

    print("Sanity-check summary:")
    print(f"  ELSA blocks: {len(elsa)} clusters: {elsa['cluster_id'].nunique()}")
    print(f"  MCScanX blocks: {len(mcscanx)}")
    if not args.skip_glm2 and args.glm2_blocks.exists():
        print(f"  gLM2 blocks: {len(glm2)} clusters: {glm2['cluster_id'].nunique()}")

    if failures:
        print("Failures:")
        for item in failures:
            print(f"  - {item}")
        return 1

    print("All checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
