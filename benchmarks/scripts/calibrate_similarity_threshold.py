#!/usr/bin/env python3
"""Calibrate `similarity_threshold` on the population it actually gates.

`similarity_threshold` filters **kNN neighbours**, not random pairs. Those are
two different distributions and they calibrate to two different numbers: a
threshold set at 5% FPR against random negatives is answering "is this pair
homologous at all", while the parameter is answering "of the fifty candidates
retrieval already chose, which survive to become anchors". Every candidate here
is by construction a near neighbour, so the negatives are hard.

The current default of 0.85 is inert in raw space — it retains essentially every
neighbour, so it is a threshold in name only. The question for a transformed
space is where a threshold starts doing work: keeping neighbours that share a
Pfam domain architecture and dropping those that do not.

Output is the whole retention/precision curve rather than a single number, since
the right operating point depends on whether a caller wants recall or purity,
and a default should be visibly a choice on a curve rather than a constant that
appeared from nowhere.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from elsa.transform import fit_transform, l2_normalize  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
SHARUR = Path("/Users/jacob/Documents/Sandbox/Sharur/Sharur/data/dpann")
STORE = SHARUR / "synteny/store_v2"


def clade_of(taxonomy: str, rank: str) -> Optional[str]:
    match = re.search(rf"{rank}__([^;]*)", str(taxonomy))
    value = match.group(1).strip() if match else ""
    return value or None


def knn_pairs(matrix: np.ndarray, queries: np.ndarray, labels: np.ndarray,
              k: int = 50):
    """Cosines and label agreement for every (query, retrieved neighbour) pair.

    This is exactly the set ``similarity_threshold`` sees.
    """
    cosines: List[np.ndarray] = []
    agrees: List[np.ndarray] = []
    keep = queries[labels[queries] >= 0]
    for start in range(0, len(keep), 512):
        block = keep[start:start + 512]
        similarity = matrix[block] @ matrix.T
        similarity[np.arange(len(block)), block] = -np.inf
        top = np.argpartition(-similarity, k, axis=1)[:, :k]
        values = np.take_along_axis(similarity, top, axis=1)
        labelled = labels[top] >= 0
        cosines.append(values[labelled])
        agrees.append((labels[top] == labels[block][:, None])[labelled])
    return np.concatenate(cosines), np.concatenate(agrees)


def curve(cosines: np.ndarray, agrees: np.ndarray,
          grid: np.ndarray) -> pd.DataFrame:
    rows = []
    base = float(agrees.mean())
    for threshold in grid:
        keep = cosines >= threshold
        retention = float(keep.mean())
        precision = float(agrees[keep].mean()) if keep.any() else float("nan")
        rows.append({
            "threshold": round(float(threshold), 4),
            "retention": round(retention, 4),
            "precision": round(precision, 4),
            "precision_lift": round(precision - base, 4),
            "true_anchors_kept": round(retention * precision / max(base, 1e-9), 4),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rank", default="c")
    parser.add_argument("--clades", type=int, default=4)
    parser.add_argument("--eval-sample", type=int, default=20_000)
    parser.add_argument("--fit-sample", type=int, default=60_000)
    parser.add_argument("--queries", type=int, default=2_000)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    metadata = pd.read_parquet(STORE / "metadata.parquet",
                               columns=["gene_id", "sample_id"])
    embeddings = np.load(STORE / "embeddings.npy", mmap_mode="r")

    connection = duckdb.connect(str(SHARUR / "sharur.duckdb"), read_only=True)
    try:
        bins = connection.execute("SELECT bin_id, taxonomy FROM bins").df()
        families = dict(connection.execute(
            "SELECT protein_id, string_agg(name, ';' ORDER BY start_aa) "
            "FROM annotations WHERE source = 'pfam' GROUP BY protein_id"
        ).fetchall())
    finally:
        connection.close()

    bins["clade"] = [clade_of(t, args.rank) for t in bins.taxonomy]
    metadata["clade"] = metadata.sample_id.map(dict(zip(bins.bin_id, bins.clade)))
    metadata = metadata[metadata.clade.notna()]
    metadata = metadata.assign(label=pd.factorize(pd.Series(
        [families.get(g) for g in metadata.gene_id]))[0])

    clades = list(metadata.clade.value_counts().index[: args.clades])
    rng = np.random.default_rng(args.seed)
    frames: List[pd.DataFrame] = []

    for clade in clades:
        rows = metadata[metadata.clade == clade]
        genomes = np.sort(rows.sample_id.unique()); rng.shuffle(genomes)
        fit_genomes = set(genomes[: max(1, len(genomes) // 2)])
        is_fit = rows.sample_id.isin(fit_genomes).to_numpy()
        fit_rows = rows.index.to_numpy()[is_fit]
        eval_rows = np.sort(rng.choice(rows.index.to_numpy()[~is_fit],
                                       min(args.eval_sample, int((~is_fit).sum())),
                                       replace=False))

        matrix = np.asarray(embeddings[eval_rows], dtype=np.float64)
        labels = metadata.label.loc[eval_rows].to_numpy()
        queries = rng.choice(len(matrix), min(args.queries, len(matrix)), replace=False)

        fit_matrix = np.asarray(
            embeddings[np.sort(rng.choice(fit_rows,
                                          min(args.fit_sample, len(fit_rows)),
                                          replace=False))], dtype=np.float64)
        transform = fit_transform("soft_zca", fit_matrix, alpha=args.alpha,
                                  seed=args.seed)

        for condition, projected, grid in (
            ("raw", l2_normalize(matrix), np.arange(0.80, 1.00, 0.01)),
            ("soft_zca_a10", l2_normalize(transform.apply(matrix)),
             np.arange(0.00, 0.85, 0.05)),
        ):
            cosines, agrees = knn_pairs(projected, queries, labels)
            part = curve(cosines, agrees, grid)
            part.insert(0, "condition", condition)
            part.insert(0, "clade", clade)
            frames.append(part)
            print(f"[{clade}] {condition}: {len(cosines):,} labelled kNN pairs, "
                  f"baseline precision {agrees.mean():.4f}", flush=True)
        del matrix, fit_matrix

    table = pd.concat(frames, ignore_index=True)
    out = REPO / "benchmarks/evaluation/transform_ablation"
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "threshold_calibration.tsv", sep="\t", index=False)

    print("\n=== MEAN ACROSS CLADES ===")
    summary = (table.groupby(["condition", "threshold"])
               [["retention", "precision", "precision_lift"]]
               .mean().round(4).reset_index())
    for condition in summary.condition.unique():
        print(f"\n{condition}:")
        print(summary[summary.condition == condition].to_string(index=False))
    print(f"\nwrote {out}/threshold_calibration.tsv")


if __name__ == "__main__":
    main()
