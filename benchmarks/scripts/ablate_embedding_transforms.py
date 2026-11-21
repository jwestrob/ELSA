#!/usr/bin/env python3
"""Tier-1 ablation of embedding post-processing, scored against Pfam homology.

Anisotropy makes raw cosine nearly uninformative in this embedding space, and the
existing answer in ``benchmarks/PCA_NORMALIZATION_PLAN.md`` cannot settle the
question: it was scored on an endpoint the July 2026 audit voided, over a gene
set carrying the ``.faa`` coordinate defect (every row reports 12.4% strict
recall). This driver re-asks it with endpoints that discriminate.

Ground truth is Pfam family co-membership: two proteins carrying the same family
are homologous. That is independent of ELSA's own machinery, which is what makes
it usable to judge the machinery.

Design notes that matter for reading the output:

* **Quantile-matched thresholds.** ``similarity_threshold`` is an absolute
  cosine, and every transform shifts the cosine distribution, so a fixed 0.85
  compares different retention rates rather than different transforms. Each
  condition is scored at a threshold that retains the same *fraction* of pairs.
* **Fit / evaluate split.** Transforms are fitted on one genome split and scored
  on another. Fitting a mean on the evaluation set is mild; fitting a covariance
  on it is not.
The k^2/N adjacency statistic is deliberately absent here: it needs genomically
consecutive genes indexed against the whole corpus, which a random subsample
cannot provide. It belongs to the Tier-2 pipeline runs.

* **Negative control.** ``random_direction`` projects out an arbitrary direction.
  It should score like ``raw``. If it instead scores like the mean-removal arms,
  the effect is not anisotropy and the whole table is void.

Read-only with respect to every input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from elsa.transform import (  # noqa: E402
    LADDER,
    fit_transform,
    isotropy_score,
    l2_normalize,
    mean_norm,
)


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def sha256_head(path: Path, limit: int = 64 * 1024 * 1024) -> str:
    """Hash the first ``limit`` bytes — enough to detect substitution of a
    multi-gigabyte store without reading all of it."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        digest.update(handle.read(limit))
    return digest.hexdigest()


def load_corpus(store_dir: Path) -> Tuple[np.ndarray, pd.DataFrame]:
    metadata = pd.read_parquet(store_dir / "metadata.parquet",
                               columns=["gene_id", "sample_id"])
    embeddings = np.load(store_dir / "embeddings.npy", mmap_mode="r")
    if len(metadata) != embeddings.shape[0]:
        raise SystemExit(
            f"store is inconsistent: {len(metadata):,} metadata rows vs "
            f"{embeddings.shape[0]:,} embedding rows"
        )
    return embeddings, metadata


def load_families(annotations_db: Path, source: str = "pfam",
                  label: str = "architecture") -> Dict[str, str]:
    """Protein -> homology label derived from its Pfam domains.

    ``architecture`` uses the ordered domain string — the same domains in the
    same order along the protein. That is the protein-level identity signal:
    a single shared top domain would call an A+B protein the same thing as a
    bare A, and a Pfam *clan* is a collection of related families, too coarse to
    mean "the same protein".

    ``architecture_multi`` restricts to proteins carrying two or more domains,
    where order actually carries information.

    ``top_domain`` keeps the older single-domain behaviour for comparison.

    Two proteins with an identical architecture but low sequence identity are
    precisely the remote homologs a PLM is supposed to recover, which is why this
    label — HMM-derived, not alignment-derived — can judge that claim without
    the circularity of scoring against sequence search.
    """
    connection = duckdb.connect(str(annotations_db), read_only=True)
    try:
        if label == "top_domain":
            rows = connection.execute(
                f"""
                SELECT protein_id, name FROM (
                    SELECT protein_id, name,
                           row_number() OVER (PARTITION BY protein_id
                                              ORDER BY evalue NULLS LAST) AS rank
                    FROM annotations WHERE source = '{source}'
                ) WHERE rank = 1
                """
            ).fetchall()
        else:
            having = "HAVING COUNT(*) >= 2" if label == "architecture_multi" else ""
            rows = connection.execute(
                f"""
                SELECT protein_id, string_agg(name, ';' ORDER BY start_aa)
                FROM annotations WHERE source = '{source}'
                GROUP BY protein_id {having}
                """
            ).fetchall()
    finally:
        connection.close()
    return {protein: family for protein, family in rows}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def auc_and_recall(positive: np.ndarray, negative: np.ndarray,
                   fpr: float = 0.05) -> Tuple[float, float, float]:
    """AUC, plus recall at a false-positive rate matched across conditions.

    Recall at matched FPR is the honest comparison: a transform that shifts every
    score upward gains nothing real, and a fixed threshold would reward it.
    """
    labels = np.r_[np.ones(len(positive)), np.zeros(len(negative))]
    scores = np.r_[positive, negative]
    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    n_pos, n_neg = labels.sum(), len(labels) - labels.sum()
    auc = float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
    threshold = float(np.quantile(negative, 1.0 - fpr))
    recall = float((positive >= threshold).mean())
    return auc, recall, threshold


def brute_force_topk(matrix: np.ndarray, queries: np.ndarray, k: int) -> np.ndarray:
    """Exact top-k by inner product; the corpus is subsampled to keep this tractable."""
    similarity = matrix[queries] @ matrix.T
    similarity[np.arange(len(queries)), queries] = -np.inf
    top = np.argpartition(-similarity, k, axis=1)[:, :k]
    ordered = np.take_along_axis(
        top, np.argsort(-np.take_along_axis(similarity, top, axis=1), axis=1), axis=1
    )
    return ordered


def precision_at_k(neighbours: np.ndarray, queries: np.ndarray,
                   labels: np.ndarray) -> Optional[float]:
    """Fraction of retrieved neighbours sharing the query's family.

    The metric that matters most here: the top-k *is* ELSA's anchor set, so this
    measures the quality of what chaining actually consumes.
    """
    keep = labels[queries] >= 0
    if not keep.any():
        return None
    hits = labels[neighbours[keep]] == labels[queries[keep]][:, None]
    valid = labels[neighbours[keep]] >= 0
    denominator = valid.sum()
    return float(hits.sum() / denominator) if denominator else None


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def build_pairs(labels: np.ndarray, n: int, rng: np.random.Generator,
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Positive pairs share a family; negatives are drawn at random."""
    by_family: Dict[int, List[int]] = {}
    for index, label in enumerate(labels):
        if label >= 0:
            by_family.setdefault(int(label), []).append(index)
    families = [rows for rows in by_family.values() if len(rows) >= 2]
    if not families:
        raise SystemExit("no Pfam family has two members in the sampled corpus")

    pos_a, pos_b = [], []
    while len(pos_a) < n:
        rows = families[rng.integers(len(families))]
        i, j = rng.choice(len(rows), 2, replace=False)
        pos_a.append(rows[i]); pos_b.append(rows[j])
    total = len(labels)
    return (np.array(pos_a), np.array(pos_b),
            rng.integers(0, total, n), rng.integers(0, total, n))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--store", required=True, type=Path,
                        help="SyntenyStore directory (metadata.parquet + embeddings.npy)")
    parser.add_argument("--annotations", required=True, type=Path,
                        help="Sharur database supplying Pfam families")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--corpus-sample", type=int, default=60_000,
                        help="rows subsampled for brute-force retrieval metrics")
    parser.add_argument("--pairs", type=int, default=20_000)
    parser.add_argument("--queries", type=int, default=1_500)
    parser.add_argument("--k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--conditions", nargs="*", default=list(LADDER))
    parser.add_argument("--label", default="architecture",
                        choices=["architecture", "architecture_multi", "top_domain"],
                        help="Homology label: ordered Pfam domain architecture "
                             "(default), the multi-domain subset, or a single "
                             "top-scoring domain")
    args = parser.parse_args()

    started = time.time()
    rng = np.random.default_rng(args.seed)
    embeddings, metadata = load_corpus(args.store)
    families = load_families(args.annotations, label=args.label)
    print(f"corpus {embeddings.shape[0]:,} x {embeddings.shape[1]} | "
          f"{len(families):,} proteins labelled by {args.label}", flush=True)

    # ---- fit / evaluate split by genome, so no transform sees its test data ----
    genomes = metadata.sample_id.to_numpy()
    unique = np.unique(genomes)
    rng.shuffle(unique)
    fit_genomes = set(unique[: max(1, len(unique) // 2)].tolist())
    is_fit = np.fromiter((g in fit_genomes for g in genomes), bool, len(genomes))
    fit_rows = np.flatnonzero(is_fit)
    eval_rows = np.flatnonzero(~is_fit)
    print(f"split: {len(fit_genomes):,} genomes / {len(fit_rows):,} rows to fit; "
          f"{len(eval_rows):,} rows to evaluate", flush=True)

    take = np.sort(rng.choice(eval_rows, min(args.corpus_sample, len(eval_rows)),
                              replace=False))
    corpus = np.asarray(embeddings[take], dtype=np.float64)
    corpus_meta = metadata.iloc[take].reset_index(drop=True)
    gene_ids = corpus_meta.gene_id.to_numpy()
    genome_codes = pd.factorize(corpus_meta.sample_id)[0]

    family_codes, _ = pd.factorize(pd.Series([families.get(g) for g in gene_ids]))
    labels = np.asarray(family_codes)          # -1 where unannotated
    print(f"evaluation corpus {len(corpus):,} rows | "
          f"{(labels >= 0).sum():,} with a family", flush=True)

    pos_a, pos_b, neg_a, neg_b = build_pairs(labels, args.pairs, rng)
    queries = rng.choice(len(corpus), min(args.queries, len(corpus)), replace=False)

    # Sample ACROSS the fit split. Slicing its head would take a contiguous,
    # genome-ordered block — a handful of redundant genomes whose covariance is
    # rank-deficient, which both biases the fit and makes whitening divide by
    # near-zero variances.
    fit_sample = np.sort(rng.choice(fit_rows, min(200_000, len(fit_rows)),
                                    replace=False))
    fit_matrix = np.asarray(embeddings[fit_sample], dtype=np.float64)
    print(f"transform fitting matrix: {fit_matrix.shape[0]:,} rows sampled across "
          f"the fit genomes", flush=True)
    baseline_neighbours: Optional[np.ndarray] = None
    results: List[Dict[str, Any]] = []

    for name in args.conditions:
        spec = dict(LADDER[name])
        kind = spec.pop("kind")
        t0 = time.time()
        # Fitted on the fit-split rows only.
        transform = fit_transform(kind, fit_matrix, seed=args.seed, **spec)
        projected = l2_normalize(transform.apply(corpus))

        pos = np.einsum("ij,ij->i", projected[pos_a], projected[pos_b])
        neg = np.einsum("ij,ij->i", projected[neg_a], projected[neg_b])
        auc, recall, threshold = auc_and_recall(pos, neg)

        neighbours = brute_force_topk(projected, queries, args.k)
        if baseline_neighbours is None:
            baseline_neighbours = neighbours
            churn = 0.0
        else:
            churn = 1.0 - float(np.mean([
                len(set(a) & set(b)) / args.k
                for a, b in zip(baseline_neighbours, neighbours)
            ]))

        results.append({
            "condition": name,
            "transform": transform.describe(),
            "mean_norm": round(mean_norm(projected), 4),
            "isotropy": round(isotropy_score(projected), 4),
            "auc": round(auc, 4),
            "recall_at_5pct_fpr": round(recall, 4),
            "threshold_at_5pct_fpr": round(threshold, 4),
            "precision_at_k": (lambda v: round(v, 4) if v is not None else None)(
                precision_at_k(neighbours, queries, labels)),
            "neighbour_churn_vs_raw": round(churn, 4),
            "seconds": round(time.time() - t0, 1),
        })
        row = results[-1]
        print(f"  {name:<18} AUC {row['auc']:.4f}  recall {row['recall_at_5pct_fpr']:.3f}"
              f"  P@{args.k} {row['precision_at_k']}  churn "
              f"{row['neighbour_churn_vs_raw']:.2f}", flush=True)

    frame = pd.DataFrame(results)
    args.output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output / "tier1_metrics.tsv", sep="\t", index=False)
    (args.output / "tier1_provenance.json").write_text(json.dumps({
        "store": str(args.store),
        "store_embeddings_sha256_head": sha256_head(args.store / "embeddings.npy"),
        "annotations": str(args.annotations),
        "corpus_rows": int(len(corpus)),
        "fit_genomes": int(len(fit_genomes)),
        "pairs": int(args.pairs),
        "k": int(args.k),
        "seed": int(args.seed),
        "elapsed_seconds": round(time.time() - started, 1),
    }, indent=2))
    print(f"\nwrote {args.output}/tier1_metrics.tsv")

    control = frame[frame.condition == "random_direction"]
    raw = frame[frame.condition == "raw"]
    if len(control) and len(raw):
        gap = float(control.auc.iloc[0] - raw.auc.iloc[0])
        verdict = "behaves as a control" if abs(gap) < 0.01 else "DOES NOT behave as a control"
        print(f"negative control: AUC gap vs raw = {gap:+.4f} -> {verdict}")


if __name__ == "__main__":
    main()
