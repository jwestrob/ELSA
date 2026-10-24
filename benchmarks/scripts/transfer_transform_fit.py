#!/usr/bin/env python3
"""Does a transform fitted on one clade work on another, and does its threshold?

A transform is only deployable if it survives leaving its fit corpus. Two
questions decide that, and they are separate:

1. **Does the representation transfer?** Soft-ZCA suppresses the directions its
   fit corpus is spread along. On a corpus spread along other directions it can
   suppress signal instead of noise, and synthetic geometries show it coming back
   *worse than raw*. So the fit corpus has to be genuinely disjoint from the
   evaluation corpus, not a random row split of the same genomes — random rows
   share every genome, and a mean estimated from them is the evaluation set's own
   mean.

2. **Does the threshold transfer?** `similarity_threshold` is an absolute cosine.
   Quantile matching makes conditions comparable and deliberately refuses to
   answer this: it re-derives a threshold per corpus, which is exactly what a
   shipped default cannot do. A default is one number applied to a corpus nobody
   calibrated on, so it has to be scored that way — calibrate on clade A, apply
   the resulting *absolute* cosine to clade B, and report what it actually
   retains.

The external-fit arm the plan asked for cannot be run as specified: UniRef50 was
embedded at 480-d and the DPANN store is 320-d, so the frozen transform does not
apply to the only corpus with retrieval headroom. Clade-disjoint transfer within
DPANN measures the same generalisation gap on a corpus where it is visible.
`--corpus bacterial` runs the 480-d UniRef50 arm for the mechanics.

Read-only with respect to every input.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from elsa.transform import fit_transform, isotropy_score, l2_normalize, mean_norm  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
SHARUR = Path("/Users/jacob/Documents/Sandbox/Sharur/Sharur/data/dpann")
STORE = SHARUR / "synteny/store_v2"


# ---------------------------------------------------------------------------
# Metrics (shared shape with ablate_embedding_transforms.py)
# ---------------------------------------------------------------------------

def auc_of(positive: np.ndarray, negative: np.ndarray) -> float:
    labels = np.r_[np.ones(len(positive)), np.zeros(len(negative))]
    scores = np.r_[positive, negative]
    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    n_pos, n_neg = labels.sum(), len(labels) - labels.sum()
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def precision_at_k(matrix: np.ndarray, queries: np.ndarray,
                   labels: np.ndarray, k: int = 50) -> Optional[float]:
    """Fraction of each query's top-k sharing its label. The top-k is the anchor set."""
    keep = queries[labels[queries] >= 0]
    if not len(keep):
        return None
    hits = total = 0
    for start in range(0, len(keep), 512):                 # chunked: the corpus is wide
        block = keep[start:start + 512]
        similarity = matrix[block] @ matrix.T
        similarity[np.arange(len(block)), block] = -np.inf
        top = np.argpartition(-similarity, k, axis=1)[:, :k]
        valid = labels[top] >= 0
        hits += int(((labels[top] == labels[block][:, None]) & valid).sum())
        total += int(valid.sum())
    return float(hits / total) if total else None


def sample_pairs(labels: np.ndarray, n: int, rng: np.random.Generator
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    by_label: Dict[int, List[int]] = {}
    for index, value in enumerate(labels):
        if value >= 0:
            by_label.setdefault(int(value), []).append(index)
    groups = [rows for rows in by_label.values() if len(rows) >= 2]
    if not groups:
        raise SystemExit("no label has two members in this clade sample")
    a, b = [], []
    while len(a) < n:
        rows = groups[rng.integers(len(groups))]
        i, j = rng.choice(len(rows), 2, replace=False)
        a.append(rows[i]); b.append(rows[j])
    return (np.array(a), np.array(b),
            rng.integers(0, len(labels), n), rng.integers(0, len(labels), n))


def scored(matrix: np.ndarray, pairs, labels, queries, rng
           ) -> Dict[str, object]:
    """AUC, precision@k, and the absolute cosine at 5% FPR for this space."""
    unit = l2_normalize(matrix)
    pa, pb, na, nb = pairs
    positive = np.einsum("ij,ij->i", unit[pa], unit[pb])
    negative = np.einsum("ij,ij->i", unit[na], unit[nb])
    threshold = float(np.quantile(negative, 0.95))
    return {
        "auc": round(auc_of(positive, negative), 4),
        "recall_at_5pct_fpr": round(float((positive >= threshold).mean()), 4),
        "threshold_at_5pct_fpr": round(threshold, 4),
        "precision_at_50": round(precision_at_k(unit, queries, labels) or float("nan"), 4),
        "mean_norm": round(mean_norm(unit), 4),
        "isotropy": round(isotropy_score(unit), 4),
    }


# ---------------------------------------------------------------------------
# DPANN clade-disjoint arm
# ---------------------------------------------------------------------------

def clade_of(taxonomy: str, rank: str) -> Optional[str]:
    match = re.search(rf"{rank}__([^;]*)", str(taxonomy))
    value = match.group(1).strip() if match else ""
    return value or None


def run_dpann(args) -> pd.DataFrame:
    metadata = pd.read_parquet(STORE / "metadata.parquet",
                               columns=["gene_id", "sample_id"])
    embeddings = np.load(STORE / "embeddings.npy", mmap_mode="r")
    if len(metadata) != embeddings.shape[0]:
        raise SystemExit("store metadata and embeddings disagree on row count")

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
    clade_by_genome = dict(zip(bins.bin_id, bins.clade))
    metadata["clade"] = metadata.sample_id.map(clade_by_genome)
    metadata = metadata[metadata.clade.notna()]

    counts = metadata.clade.value_counts()
    clades = [c for c in counts.index[: args.clades]]
    print(f"clades at rank {args.rank}: " +
          ", ".join(f"{c} ({counts[c]:,} genes)" for c in clades), flush=True)

    rng = np.random.default_rng(args.seed)
    label_codes = pd.factorize(pd.Series(
        [families.get(g) for g in metadata.gene_id]))[0]
    metadata = metadata.assign(label=label_codes)

    # Per clade: disjoint genome halves, one to fit from and one to evaluate on.
    pools: Dict[str, Dict[str, np.ndarray]] = {}
    for clade in clades:
        rows = metadata[metadata.clade == clade]
        genomes = np.sort(rows.sample_id.unique())
        rng.shuffle(genomes)
        half = max(1, len(genomes) // 2)
        fit_genomes = set(genomes[:half])
        is_fit = rows.sample_id.isin(fit_genomes).to_numpy()
        fit_rows = rows.index.to_numpy()[is_fit]
        eval_rows = rows.index.to_numpy()[~is_fit]
        pools[clade] = {
            "fit": rng.choice(fit_rows, min(args.fit_sample, len(fit_rows)), replace=False),
            "eval": np.sort(rng.choice(eval_rows,
                                       min(args.eval_sample, len(eval_rows)), replace=False)),
            "n_genomes": len(genomes),
        }
        print(f"  {clade}: {len(genomes):,} genomes -> "
              f"{len(pools[clade]['fit']):,} fit rows, "
              f"{len(pools[clade]['eval']):,} eval rows", flush=True)

    # Materialise once per clade; the memmap is 4.3 GB and random gathers are slow.
    fit_matrices = {c: np.asarray(embeddings[np.sort(p["fit"])], dtype=np.float64)
                    for c, p in pools.items()}
    eval_matrices = {c: np.asarray(embeddings[p["eval"]], dtype=np.float64)
                     for c, p in pools.items()}
    # ``pools`` holds original embedding row numbers, so labels must be looked up
    # by index. A positional gather here would silently mislabel every gene.
    eval_labels = {c: metadata.label.loc[pools[c]["eval"]].to_numpy() for c in clades}

    records: List[Dict[str, object]] = []


    for eval_clade in clades:
        matrix = eval_matrices[eval_clade]
        labels = eval_labels[eval_clade]
        pair_rng = np.random.default_rng(args.seed)          # same pairs every arm
        pairs = sample_pairs(labels, args.pairs, pair_rng)
        queries = pair_rng.choice(len(matrix), min(args.queries, len(matrix)),
                                  replace=False)

        arms: List[Tuple[str, Optional[np.ndarray]]] = [("raw", None)]
        arms.append((f"fit:{eval_clade}(in-corpus)", fit_matrices[eval_clade]))
        for other in clades:
            if other != eval_clade:
                arms.append((f"fit:{other}", fit_matrices[other]))
        pooled = np.concatenate([fit_matrices[c] for c in clades if c != eval_clade])
        arms.append(("fit:pooled-others", pooled))

        for name, fit_matrix in arms:
            if fit_matrix is None:
                projected = matrix
            else:
                transform = fit_transform(args.transform, fit_matrix,
                                          alpha=args.alpha, seed=args.seed)
                projected = transform.apply(matrix)
            result = scored(projected, pairs, labels, queries, pair_rng)
            records.append({"eval_clade": eval_clade, "fit": name,
                            "n_eval_genomes": pools[eval_clade]["n_genomes"], **result})
            print(f"  [{eval_clade}] {name:<28} AUC {result['auc']:.4f}  "
                  f"recall {result['recall_at_5pct_fpr']:.4f}  "
                  f"P@50 {result['precision_at_50']:.4f}  "
                  f"thr {result['threshold_at_5pct_fpr']:.4f}", flush=True)
        del pooled

    frame = pd.DataFrame(records)

    # ---- does one absolute threshold transfer between clades? --------------
    print("\n=== THRESHOLD TRANSFER ===", flush=True)
    transfer: List[Dict[str, object]] = []
    for source in clades:
        calibrated = frame[(frame.eval_clade == source) &
                           (frame.fit == "fit:pooled-others")]
        if calibrated.empty:
            continue
        threshold = float(calibrated.threshold_at_5pct_fpr.iloc[0])
        for target in clades:
            matrix = eval_matrices[target]
            labels = eval_labels[target]
            pair_rng = np.random.default_rng(args.seed)
            pa, pb, na, nb = sample_pairs(labels, args.pairs, pair_rng)
            pooled = np.concatenate([fit_matrices[c] for c in clades if c != target])
            transform = fit_transform(args.transform, pooled, alpha=args.alpha,
                                      seed=args.seed)
            unit = l2_normalize(transform.apply(matrix))
            positive = np.einsum("ij,ij->i", unit[pa], unit[pb])
            negative = np.einsum("ij,ij->i", unit[na], unit[nb])
            transfer.append({
                "calibrated_on": source, "applied_to": target,
                "threshold": round(threshold, 4),
                "realized_fpr": round(float((negative >= threshold).mean()), 4),
                "realized_recall": round(float((positive >= threshold).mean()), 4),
            })
            del pooled
            print(f"  {source} -> {target}: thr {threshold:.4f}  "
                  f"FPR {transfer[-1]['realized_fpr']:.4f} (target 0.05)  "
                  f"recall {transfer[-1]['realized_recall']:.4f}", flush=True)

    out = REPO / "benchmarks/evaluation/transform_ablation"
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / "transfer_by_clade.tsv", sep="\t", index=False)
    pd.DataFrame(transfer).to_csv(out / "threshold_transfer.tsv", sep="\t", index=False)
    print(f"\nwrote {out}/transfer_by_clade.tsv and threshold_transfer.tsv")
    return frame


# ---------------------------------------------------------------------------
# 480-d external arm: fit on UniRef50, evaluate on bacterial cohorts
# ---------------------------------------------------------------------------

def run_bacterial(args) -> pd.DataFrame:
    import pyarrow.parquet as pq

    uniref = pq.ParquetFile(REPO / "data/frozen_pca/uniref50_raw.parquet")
    raw_cols = [c for c in uniref.schema_arrow.names if c.startswith("raw_")]
    external = uniref.read(columns=raw_cols).to_pandas().to_numpy(dtype=np.float64)
    print(f"external fit corpus: UniRef50 {external.shape}", flush=True)

    corpora = {
        "E.coli": (REPO / "benchmarks/elsa_output/ecoli/elsa_index/ingest/genes.parquet",
                   REPO / "benchmarks/data/ecoli/annotations"),
        "cross-species": (REPO / "benchmarks/elsa_output/cross_species/ingest/genes.parquet",
                          REPO / "benchmarks/data/cross_species/annotations"),
    }
    rng = np.random.default_rng(args.seed)
    records: List[Dict[str, object]] = []

    for tag, (parquet, gff_dir) in corpora.items():
        names = pq.ParquetFile(parquet).schema_arrow.names
        emb_cols = [n for n in names if n.startswith("emb_")]
        keep = [c for c in ("gene_id", "sample_id", "contig_id", "end") if c in names]
        table = pq.ParquetFile(parquet).read(columns=keep + emb_cols).to_pandas()

        by_end: Dict[Tuple[str, int], str] = {}
        for path in sorted(Path(gff_dir).glob("*.gff")):
            for line in path.open():
                if line.startswith("#"):
                    continue
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 9 or parts[2] != "CDS":
                    continue
                match = re.search(r"gene=([^;]+)", parts[8])
                if match:
                    by_end[(parts[0], int(parts[4]))] = match.group(1)
        table["name"] = [by_end.get((str(c), int(e)))
                         for c, e in zip(table.contig_id, table["end"])]
        table = table[table.name.notna()].reset_index(drop=True)
        if len(table) > args.eval_sample:
            table = table.iloc[np.sort(rng.choice(len(table), args.eval_sample,
                                                  replace=False))].reset_index(drop=True)
        matrix = table[emb_cols].to_numpy(dtype=np.float64)
        labels = pd.factorize(table.name)[0]

        pair_rng = np.random.default_rng(args.seed)
        pairs = sample_pairs(labels, args.pairs, pair_rng)
        queries = pair_rng.choice(len(matrix), min(args.queries, len(matrix)),
                                  replace=False)

        for name, fit_matrix in (("raw", None),
                                 ("fit:UniRef50(external)", external),
                                 ("fit:in-corpus", matrix[:len(matrix) // 2])):
            if fit_matrix is None:
                projected = matrix
            else:
                transform = fit_transform(args.transform, fit_matrix,
                                          alpha=args.alpha, seed=args.seed)
                projected = transform.apply(matrix)
            result = scored(projected, pairs, labels, queries, pair_rng)
            records.append({"corpus": tag, "fit": name, **result})
            print(f"  [{tag}] {name:<24} AUC {result['auc']:.4f}  "
                  f"P@50 {result['precision_at_50']:.4f}  "
                  f"thr {result['threshold_at_5pct_fpr']:.4f}", flush=True)

    frame = pd.DataFrame(records)
    out = REPO / "benchmarks/evaluation/transform_ablation"
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / "transfer_external_480d.tsv", sep="\t", index=False)
    print(f"\nwrote {out}/transfer_external_480d.tsv")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", choices=["dpann", "bacterial"], default="dpann")
    parser.add_argument("--transform", default="soft_zca")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--rank", default="c", help="GTDB rank letter for the clade split")
    parser.add_argument("--clades", type=int, default=4)
    parser.add_argument("--fit-sample", type=int, default=60_000)
    parser.add_argument("--eval-sample", type=int, default=20_000)
    parser.add_argument("--pairs", type=int, default=20_000)
    parser.add_argument("--queries", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    frame = run_dpann(args) if args.corpus == "dpann" else run_bacterial(args)
    print()
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
