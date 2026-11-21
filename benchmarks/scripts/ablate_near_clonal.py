#!/usr/bin/env python3
"""Transform ablation for near-clonal cohorts, scored with hard negatives.

Against random negatives a near-clonal cohort saturates: raw cosine already
reaches AUC 0.998 separating "same gene name" from "a random other gene",
because in 20 near-identical genomes those are wildly different things. That
number says nothing useful, and reading it as "the transform does not help here"
would be wrong — it means the question was too easy.

The failure that actually afflicts near-clonal cohorts is *resolution*: chains
run a median of 142 genes through a 26-gene locus because too many candidates
look equally good. So the question is not "can a homolog be told from a random
protein" but "can the true ortholog be told from its closest competitor".

Three endpoints, all on the same corpus:

* **hard-negative AUC** — negatives are the highest-scoring genes that do *not*
  share a gene name, i.e. each query's most convincing impostor.
* **precision@1** — does a true ortholog outrank every impostor.
* **cross-genus positives** — orthologs between genera are remote homologs, so
  this restores a hard sensitivity test on a corpus whose within-genus pairs are
  saturated, and makes a same-genes comparison between PLMs informative.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from elsa.transform import LADDER, fit_transform, l2_normalize, mean_norm  # noqa: E402

REPO = Path(__file__).resolve().parents[2]

CORPORA = [
    {"tag": "cross-species/ESM2-raw480",
     "parquet": REPO / "benchmarks/elsa_output/cross_species/ingest/genes.parquet",
     "gff": REPO / "benchmarks/data/cross_species/annotations"},
    {"tag": "cross-species/ProtT5-1024",
     "parquet": REPO / "benchmarks/elsa_output/cross_species_prott5/ingest/genes.parquet",
     "gff": REPO / "benchmarks/data/cross_species/annotations"},
    {"tag": "E.coli/ESM2-raw480",
     "parquet": REPO / "benchmarks/elsa_output/ecoli/elsa_index/ingest/genes.parquet",
     "gff": REPO / "benchmarks/data/ecoli/annotations"},
]


def load(parquet: Path, gff_dir: Path, sample: int, seed: int):
    names = pq.ParquetFile(parquet).schema_arrow.names
    emb_cols = [n for n in names if n.startswith("emb_")]
    keep = [c for c in ("gene_id", "sample_id", "contig_id", "end") if c in names]
    table = pq.ParquetFile(parquet).read(columns=keep + emb_cols).to_pandas()

    by_end: Dict[Tuple[str, int], str] = {}
    for path in sorted(gff_dir.glob("*.gff")):
        for line in path.open():
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9 or parts[2] != "CDS":
                continue
            match = re.search(r"gene=([^;]+)", parts[8])
            if match:
                by_end[(parts[0], int(parts[4]))] = match.group(1)

    table["gene_name"] = [by_end.get((str(c), int(e)))
                          for c, e in zip(table.contig_id, table["end"])]
    table = table[table.gene_name.notna()].reset_index(drop=True)
    rng = np.random.default_rng(seed)
    if len(table) > sample:
        table = table.iloc[np.sort(rng.choice(len(table), sample, replace=False))]
        table = table.reset_index(drop=True)
    matrix = table[emb_cols].to_numpy(dtype=np.float64)
    return matrix, table


def genus_of(sample_id: str) -> str:
    """Coarse genus proxy from the assembly accession prefix.

    The cross-species cohort mixes Escherichia, Salmonella and Klebsiella, each
    occupying a distinct accession block, so the prefix separates them without a
    taxonomy lookup.
    """
    return str(sample_id)[:11]


def evaluate(matrix: np.ndarray, table: pd.DataFrame, rng, n_pairs: int
             ) -> Dict[str, float]:
    unit = l2_normalize(matrix)
    names = table.gene_name.to_numpy()
    genus = np.array([genus_of(s) for s in table.sample_id])

    by_name: Dict[str, List[int]] = {}
    for index, name in enumerate(names):
        by_name.setdefault(name, []).append(index)

    cross = []
    for rows in by_name.values():
        if len(rows) < 2:
            continue
        for _ in range(min(6, len(rows))):
            i, j = rng.choice(len(rows), 2, replace=False)
            a, b = rows[i], rows[j]
            if genus[a] != genus[b]:
                cross.append((a, b))
    rng.shuffle(cross)
    cross = cross[:n_pairs]

    # Hard negatives: for each query, the best-scoring gene that does NOT share
    # its name — the most convincing impostor available.
    queries = rng.choice(len(unit), min(1500, len(unit)), replace=False)
    sims = unit[queries] @ unit.T
    sims[np.arange(len(queries)), queries] = -np.inf
    same = names[None, :] == names[queries][:, None]
    hard_neg = np.where(same, -np.inf, sims).max(axis=1)
    best_pos = np.where(same, sims, -np.inf).max(axis=1)
    usable = np.isfinite(best_pos) & np.isfinite(hard_neg)

    top1 = sims.argmax(axis=1)
    p_at_1 = float((names[top1] == names[queries])[usable].mean())

    def auc(pos, neg):
        if len(pos) == 0 or len(neg) == 0:
            return float("nan")
        labels = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
        scores = np.r_[pos, neg]
        order = np.argsort(scores)
        ranks = np.empty(len(scores)); ranks[order] = np.arange(1, len(scores) + 1)
        n1 = labels.sum(); n0 = len(labels) - n1
        return float((ranks[labels == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

    pos_cross = (np.einsum("ij,ij->i",
                           unit[[p[0] for p in cross]],
                           unit[[p[1] for p in cross]])
                 if cross else np.array([]))

    return {
        "auc_hard_negative": round(auc(best_pos[usable], hard_neg[usable]), 4),
        "precision_at_1": round(p_at_1, 4),
        "auc_cross_genus_vs_hard": round(auc(pos_cross, hard_neg[usable]), 4),
        "margin_pos_minus_hardneg": round(
            float(np.median(best_pos[usable] - hard_neg[usable])), 4),
        "n_cross_genus_pairs": len(cross),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path,
                        default=REPO / "benchmarks/evaluation/transform_ablation/near_clonal.tsv")
    parser.add_argument("--sample", type=int, default=25_000)
    parser.add_argument("--pairs", type=int, default=8_000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--conditions", nargs="*",
                        default=["raw", "random_direction", "r1_center",
                                 "abtt_k2", "soft_zca_a10", "pca_whiten"])
    args = parser.parse_args()

    rows: List[Dict[str, object]] = []
    for entry in CORPORA:
        if not Path(entry["parquet"]).is_file():
            print(f"[skip] {entry['tag']}: missing parquet", flush=True)
            continue
        matrix, table = load(Path(entry["parquet"]), Path(entry["gff"]),
                             args.sample, args.seed)
        print(f"\n{entry['tag']}  n={len(matrix):,} dim={matrix.shape[1]} "
              f"named={len(table):,}  ||mu||={mean_norm(matrix):.4f}", flush=True)
        half = len(matrix) // 2
        for name in args.conditions:
            spec = dict(LADDER[name]); kind = spec.pop("kind")
            transform = fit_transform(kind, matrix[:half], seed=args.seed, **spec)
            projected = transform.apply(matrix)
            rng = np.random.default_rng(args.seed)   # identical pairs per condition
            metrics = evaluate(projected, table, rng, args.pairs)
            rows.append({"corpus": entry["tag"], "condition": name,
                         "mean_norm": round(mean_norm(projected), 4), **metrics})
            print(f"    {name:<18} hardneg-AUC {metrics['auc_hard_negative']:.4f}"
                  f"  P@1 {metrics['precision_at_1']:.4f}"
                  f"  cross-genus {metrics['auc_cross_genus_vs_hard']:.4f}"
                  f"  margin {metrics['margin_pos_minus_hardneg']:+.4f}", flush=True)

    frame = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, sep="\t", index=False)
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
