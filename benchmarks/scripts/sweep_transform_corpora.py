#!/usr/bin/env python3
"""Run the transform ladder across every available corpus and representation.

The DPANN result showed aggressive isotropization helping far more than the text
embedding literature predicts. Two things need establishing before that can be
called a property of protein language models rather than of one dataset:

1. **Does it generalise across corpora and taxa?** Raw ESM2 sits at ||mu|| ~ 0.92
   on archaea and bacteria alike, so the geometry is a model property — but the
   downstream benefit still has to be shown separately on each.
2. **Is it model-specific?** ProtT5 measures ||mu|| = 0.62 against ESM2's 0.92 on
   the *same genes*, which makes it a controlled test of whether benefit tracks
   anisotropy the way the literature's r = +0.72 claims.

Homology labels come from whatever each corpus has: Pfam families from a Sharur
database, or gene names parsed from GFF annotations. Both are external to ELSA,
which is what makes them usable to judge it.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from elsa.transform import (  # noqa: E402
    LADDER, fit_transform, isotropy_score, l2_normalize, mean_norm,
)

REPO = Path(__file__).resolve().parents[2]
SHARUR = Path("/Users/jacob/Documents/Sandbox/Sharur/Sharur/data")


# name, representation, embeddings, label source
CORPORA: List[Dict[str, object]] = [
    {"corpus": "DPANN", "representation": "ESM2-320",
     "embeddings": SHARUR / "dpann/synteny/store_v2",
     "labels": ("sharur", SHARUR / "dpann/sharur.duckdb")},
    {"corpus": "Omnitrophota", "representation": "ESM2-320",
     "embeddings": SHARUR / "omni_production/synteny/store",
     "labels": ("sharur", SHARUR / "omni_production/sharur.duckdb")},
    {"corpus": "E.coli", "representation": "ESM2-raw480",
     "embeddings": REPO / "benchmarks/elsa_output/ecoli/elsa_index/ingest/genes.parquet",
     "labels": ("gff", REPO / "benchmarks/data/ecoli/annotations")},
    {"corpus": "cross-species", "representation": "ESM2-raw480",
     "embeddings": REPO / "benchmarks/elsa_output/cross_species/ingest/genes.parquet",
     "labels": ("gff", REPO / "benchmarks/data/cross_species/annotations")},
    {"corpus": "cross-species", "representation": "ProtT5-1024",
     "embeddings": REPO / "benchmarks/elsa_output/cross_species_prott5/ingest/genes.parquet",
     "labels": ("gff", REPO / "benchmarks/data/cross_species/annotations")},
    {"corpus": "cross-species", "representation": "gLM2-pca256",
     "embeddings": REPO / "benchmarks/elsa_output/cross_species_glm2/ingest/genes.parquet",
     "labels": ("gff", REPO / "benchmarks/data/cross_species/annotations")},
    {"corpus": "E.coli", "representation": "ESM2-pca256",
     "embeddings": REPO / "benchmarks/elsa_output/ecoli/elsa_index_small/ingest/genes.parquet",
     "labels": ("gff", REPO / "benchmarks/data/ecoli/annotations")},
]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_embeddings(path: Path, sample: int, seed: int
                    ) -> Tuple[np.ndarray, np.ndarray, Optional[pd.DataFrame]]:
    """Return (matrix, gene_ids, coordinates) for a store dir or a genes.parquet.

    Coordinates come back when present, because gene identifiers alone cannot be
    matched to an NCBI annotation — ``GCF_000005845.1_CDS_1`` carries no gene
    name. The join key is (contig, CDS end), which is stable between a caller's
    output and the GFF it came from.
    """
    rng = np.random.default_rng(seed)
    if path.is_dir():
        meta = pd.read_parquet(path / "metadata.parquet", columns=["gene_id"])
        emb = np.load(path / "embeddings.npy", mmap_mode="r")
        rows = np.sort(rng.choice(len(meta), min(sample, len(meta)), replace=False))
        return np.asarray(emb[rows], dtype=np.float64), meta.gene_id.to_numpy()[rows], None

    names = pq.ParquetFile(path).schema_arrow.names
    emb_cols = [n for n in names if n.startswith("emb_")]
    id_col = "gene_id" if "gene_id" in names else names[0]
    coord_cols = [c for c in ("contig_id", "end") if c in names]
    table = pq.ParquetFile(path).read(columns=[id_col] + coord_cols + emb_cols).to_pandas()
    rows = np.sort(rng.choice(len(table), min(sample, len(table)), replace=False))
    subset = table.iloc[rows]
    coords = subset[coord_cols].reset_index(drop=True) if len(coord_cols) == 2 else None
    return (subset[emb_cols].to_numpy(dtype=np.float64),
            subset[id_col].to_numpy(), coords)


def labels_from_sharur(db: Path, gene_ids: np.ndarray) -> np.ndarray:
    import duckdb
    connection = duckdb.connect(str(db), read_only=True)
    try:
        rows = connection.execute(
            """
            SELECT protein_id, name FROM (
                SELECT protein_id, name,
                       row_number() OVER (PARTITION BY protein_id
                                          ORDER BY evalue NULLS LAST) AS rk
                FROM annotations WHERE source='pfam'
            ) WHERE rk = 1
            """
        ).fetchall()
    finally:
        connection.close()
    mapping = dict(rows)
    return pd.factorize(pd.Series([mapping.get(g) for g in gene_ids]))[0]


def labels_from_gff(directory: Path, coords: Optional[pd.DataFrame]) -> np.ndarray:
    """Gene name as a homology label: two genes named ``rplB`` are homologous.

    Coarser than Pfam and only covers named genes, but external to ELSA and
    available wherever NCBI annotations are. Matched on (contig, CDS end): the
    identifiers a gene caller emits carry no gene name, so a name-based match
    silently labels nothing.
    """
    if coords is None:
        return np.full(0, -1)
    by_end: Dict[Tuple[str, int], str] = {}
    for gff in sorted(directory.glob("*.gff")):
        with gff.open() as handle:
            for line in handle:
                if line.startswith("#"):
                    continue
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 9 or parts[2] != "CDS":
                    continue
                match = re.search(r"gene=([^;]+)", parts[8])
                if match:
                    by_end[(parts[0], int(parts[4]))] = match.group(1)
    names = [by_end.get((str(c), int(e)))
             for c, e in zip(coords.iloc[:, 0], coords.iloc[:, 1])]
    return pd.factorize(pd.Series(names))[0]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def auc_recall(pos: np.ndarray, neg: np.ndarray, fpr: float = 0.05
               ) -> Tuple[float, float]:
    labels = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    scores = np.r_[pos, neg]
    order = np.argsort(scores)
    ranks = np.empty(len(scores)); ranks[order] = np.arange(1, len(scores) + 1)
    n1 = labels.sum(); n0 = len(labels) - n1
    auc = float((ranks[labels == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
    return auc, float((pos >= np.quantile(neg, 1 - fpr)).mean())


def build_pairs(labels: np.ndarray, n: int, rng) -> Optional[tuple]:
    groups: Dict[int, List[int]] = {}
    for i, lab in enumerate(labels):
        if lab >= 0:
            groups.setdefault(int(lab), []).append(i)
    usable = [rows for rows in groups.values() if len(rows) >= 2]
    if len(usable) < 5:
        return None
    a, b = [], []
    while len(a) < n:
        rows = usable[rng.integers(len(usable))]
        i, j = rng.choice(len(rows), 2, replace=False)
        a.append(rows[i]); b.append(rows[j])
    total = len(labels)
    return (np.array(a), np.array(b),
            rng.integers(0, total, n), rng.integers(0, total, n))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path,
                        default=REPO / "benchmarks/evaluation/transform_ablation/corpora")
    parser.add_argument("--sample", type=int, default=30_000)
    parser.add_argument("--pairs", type=int, default=15_000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--conditions", nargs="*",
                        default=["raw", "random_direction", "r1_center",
                                 "abtt_k2", "soft_zca_a10", "pca_whiten"])
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    rows: List[Dict[str, object]] = []

    for entry in CORPORA:
        path = Path(entry["embeddings"])          # type: ignore[arg-type]
        tag = f"{entry['corpus']}/{entry['representation']}"
        if not (path.is_dir() or path.is_file()):
            print(f"[skip] {tag}: missing {path}", flush=True)
            continue
        try:
            matrix, gene_ids, coords = load_embeddings(path, args.sample, args.seed)
        except Exception as exc:
            print(f"[skip] {tag}: {exc}", flush=True)
            continue

        kind, source = entry["labels"]            # type: ignore[misc]
        labels = (labels_from_sharur(Path(source), gene_ids) if kind == "sharur"
                  else labels_from_gff(Path(source), coords))
        if len(labels) != len(matrix):
            print(f"[skip] {tag}: no coordinates available to match annotations",
                  flush=True)
            continue
        pairs = build_pairs(labels, args.pairs, rng)
        annotated = int((labels >= 0).sum())
        if pairs is None:
            print(f"[skip] {tag}: only {annotated:,} labelled, too few families",
                  flush=True)
            continue
        pos_a, pos_b, neg_a, neg_b = pairs

        unit = l2_normalize(matrix)
        print(f"\n{tag}  n={len(matrix):,} dim={matrix.shape[1]} "
              f"labelled={annotated:,}  ||mu||={mean_norm(unit):.4f}", flush=True)

        # split so no transform is fitted on the rows it is scored on
        half = len(matrix) // 2
        fit_part, eval_part = matrix[:half], matrix
        for name in args.conditions:
            spec = dict(LADDER[name]); tkind = spec.pop("kind")
            transform = fit_transform(tkind, fit_part, seed=args.seed, **spec)
            projected = l2_normalize(transform.apply(eval_part))
            pos = np.einsum("ij,ij->i", projected[pos_a], projected[pos_b])
            neg = np.einsum("ij,ij->i", projected[neg_a], projected[neg_b])
            auc, recall = auc_recall(pos, neg)
            rows.append({
                "corpus": entry["corpus"], "representation": entry["representation"],
                "n": len(matrix), "dim": matrix.shape[1], "labelled": annotated,
                "condition": name,
                "mean_norm": round(mean_norm(projected), 4),
                "isotropy": round(isotropy_score(projected), 4),
                "auc": round(auc, 4), "recall_at_5pct_fpr": round(recall, 4),
            })
            print(f"    {name:<18} AUC {auc:.4f}  recall {recall:.4f}", flush=True)
        pd.DataFrame(rows).to_csv(args.output.parent / "corpora_sweep.tsv",
                                  sep="\t", index=False) if rows else None

    args.output.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output / "corpora_sweep.tsv", sep="\t", index=False)
    print(f"\nwrote {args.output}/corpora_sweep.tsv")

    # Does benefit track baseline anisotropy, as the literature reports?
    pivot = frame.pivot_table(index=["corpus", "representation"],
                              columns="condition", values="auc")
    if {"raw", "soft_zca_a10"} <= set(pivot.columns):
        baseline = frame[frame.condition == "raw"].set_index(
            ["corpus", "representation"]).mean_norm
        gain = (pivot["soft_zca_a10"] - pivot["raw"]).rename("auc_gain")
        joined = pd.concat([baseline.rename("baseline_mean_norm"), gain], axis=1).dropna()
        print("\nbenefit vs baseline anisotropy:")
        print(joined.round(4).to_string())
        if len(joined) > 2:
            r = float(np.corrcoef(joined.baseline_mean_norm, joined.auc_gain)[0, 1])
            print(f"\nPearson r(||mu||, AUC gain) = {r:+.3f}  (n={len(joined)})")


if __name__ == "__main__":
    main()
