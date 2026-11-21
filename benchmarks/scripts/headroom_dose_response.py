#!/usr/bin/env python3
"""Does post-processing benefit scale with task headroom, and at what slope?

The corpus sweep established an ordering: across seven corpus-representation
combinations the gain tracked available headroom (r = +0.999) far better than the
mean norm (+0.477) or the covariance participation ratio (+0.055). Those seven
points form two clusters — two divergent corpora and five saturated ones — so
they support the ordering and leave the functional form open. A coefficient
computed across a gap is describing the gap.

This fills the gap by varying difficulty continuously on one corpus. Retrieval
within a genus is easy and within a phylum is hard, so restricting the evaluation
set to a taxon at a chosen rank sweeps difficulty while holding the embeddings,
the label definition, the transform, and the pipeline fixed. Everything that
differs between the original seven points — model, dimension, annotation source,
assembly provenance — is held constant here, which is what makes a slope
interpretable.

Two design choices matter.

**One pooled transform, fitted once.** Fitting per stratum would give small
strata less fit data, and stratum size correlates with taxonomic breadth, so the
fit quality would track the very axis under study. A single fit on held-out
genomes removes that confound and matches how a shipped transform is used;
clade-disjoint transfer costs under 0.001 AUC, so little is given up.

**Negatives are drawn inside the stratum.** A narrow stratum has similar proteins
throughout, so its negatives are correspondingly harder to reject. Sampling
negatives corpus-wide would make every stratum equally easy and flatten the axis
this experiment exists to create.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from elsa.transform import fit_transform, isotropy_score, l2_normalize, mean_norm  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
SHARUR = Path("/Users/jacob/Documents/Sandbox/Sharur/Sharur/data/dpann")
STORE = SHARUR / "synteny/store_v2"

RANKS = [("s", "species"), ("g", "genus"), ("f", "family"),
         ("o", "order"), ("c", "class"), ("p", "phylum")]


def auc_of(positive: np.ndarray, negative: np.ndarray) -> float:
    labels = np.r_[np.ones(len(positive)), np.zeros(len(negative))]
    scores = np.r_[positive, negative]
    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    n_pos, n_neg = labels.sum(), len(labels) - labels.sum()
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def sample_pairs(labels: np.ndarray, n: int, rng: np.random.Generator):
    """Positives share an ordered Pfam architecture; negatives are drawn at
    random from the same stratum, so stratum breadth sets their difficulty."""
    by_label: Dict[int, List[int]] = {}
    for index, value in enumerate(labels):
        if value >= 0:
            by_label.setdefault(int(value), []).append(index)
    groups = [rows for rows in by_label.values() if len(rows) >= 2]
    if not groups:
        return None
    a, b = [], []
    while len(a) < n:
        rows = groups[rng.integers(len(groups))]
        i, j = rng.choice(len(rows), 2, replace=False)
        a.append(rows[i]); b.append(rows[j])
    return (np.array(a), np.array(b),
            rng.integers(0, len(labels), n), rng.integers(0, len(labels), n))


def score(matrix: np.ndarray, pairs) -> float:
    unit = l2_normalize(matrix)
    pa, pb, na, nb = pairs
    return auc_of(np.einsum("ij,ij->i", unit[pa], unit[pb]),
                  np.einsum("ij,ij->i", unit[na], unit[nb]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-rank", type=int, default=12,
                        help="Largest taxa to take at each rank")
    parser.add_argument("--stratum-sample", type=int, default=15_000)
    parser.add_argument("--min-labelled", type=int, default=2_000)
    parser.add_argument("--pairs", type=int, default=20_000)
    parser.add_argument("--fit-genome-frac", type=float, default=0.3)
    parser.add_argument("--fit-sample", type=int, default=80_000)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", type=Path,
                        default=REPO / "benchmarks/evaluation/transform_ablation/headroom_dose_response.tsv")
    args = parser.parse_args()

    metadata = pd.read_parquet(STORE / "metadata.parquet",
                               columns=["gene_id", "sample_id"])
    print(f"loading embeddings ({(STORE / 'embeddings.npy').stat().st_size / 1e9:.1f} GB)...",
          flush=True)
    embeddings = np.asarray(np.load(STORE / "embeddings.npy", mmap_mode="r"),
                            dtype=np.float32)

    connection = duckdb.connect(str(SHARUR / "sharur.duckdb"), read_only=True)
    try:
        bins = connection.execute("SELECT bin_id, taxonomy FROM bins").df()
        families = dict(connection.execute(
            "SELECT protein_id, string_agg(name, ';' ORDER BY start_aa) "
            "FROM annotations WHERE source = 'pfam' GROUP BY protein_id"
        ).fetchall())
    finally:
        connection.close()

    metadata["label"] = pd.factorize(pd.Series(
        [families.get(g) for g in metadata.gene_id]))[0]
    taxonomy = dict(zip(bins.bin_id, bins.taxonomy))
    metadata["taxonomy"] = metadata.sample_id.map(taxonomy)
    metadata = metadata[metadata.taxonomy.notna()].copy()
    for letter, name in RANKS:
        metadata[name] = metadata.taxonomy.str.extract(rf"{letter}__([^;]*)")[0]
        metadata.loc[metadata[name] == "", name] = np.nan

    # Hold out genomes for the transform fit so no stratum is scored on data the
    # transform saw. One fit serves every stratum.
    rng = np.random.default_rng(args.seed)
    genomes = np.sort(metadata.sample_id.unique())
    rng.shuffle(genomes)
    n_fit = max(1, int(len(genomes) * args.fit_genome_frac))
    fit_genomes = set(genomes[:n_fit].tolist())
    is_fit = metadata.sample_id.isin(fit_genomes).to_numpy()
    fit_rows = metadata.index.to_numpy()[is_fit]
    evaluation = metadata.loc[~is_fit]
    print(f"{len(genomes):,} genomes -> {len(fit_genomes):,} to fit "
          f"({len(fit_rows):,} rows), {len(evaluation):,} rows to evaluate", flush=True)

    fit_pick = np.sort(rng.choice(fit_rows, min(args.fit_sample, len(fit_rows)),
                                  replace=False))
    transform = fit_transform("soft_zca", embeddings[fit_pick].astype(np.float64),
                              alpha=args.alpha, seed=args.seed)
    print(f"fitted {transform.describe()} on {len(fit_pick):,} held-out rows",
          flush=True)

    strata: List[Tuple[str, str, np.ndarray]] = [
        ("all", "DPANN", evaluation.index.to_numpy())
    ]
    for _, name in RANKS:
        counts = evaluation[name].value_counts()
        for taxon in counts.index[: args.per_rank]:
            rows = evaluation.index.to_numpy()[(evaluation[name] == taxon).to_numpy()]
            strata.append((name, taxon, rows))

    records: List[Dict[str, object]] = []
    for rank, taxon, rows in strata:
        if len(rows) < args.min_labelled:
            continue
        pick = np.sort(rng.choice(rows, min(args.stratum_sample, len(rows)),
                                  replace=False))
        labels = metadata.label.loc[pick].to_numpy()
        n_labelled = int((labels >= 0).sum())
        if n_labelled < args.min_labelled:
            continue
        matrix = embeddings[pick].astype(np.float64)
        pairs = sample_pairs(labels, args.pairs, np.random.default_rng(args.seed))
        if pairs is None:
            continue

        raw_unit = l2_normalize(matrix)
        auc_raw = score(matrix, pairs)
        auc_treated = score(transform.apply(matrix), pairs)
        records.append({
            "rank": rank, "taxon": taxon,
            "n_genomes": int(metadata.sample_id.loc[pick].nunique()),
            "n_proteins": len(pick), "n_labelled": n_labelled,
            "mean_norm": round(mean_norm(raw_unit), 4),
            "isotropy": round(isotropy_score(raw_unit), 4),
            "auc_raw": round(auc_raw, 5),
            "auc_treated": round(auc_treated, 5),
            "headroom": round(1.0 - auc_raw, 5),
            "gain": round(auc_treated - auc_raw, 5),
        })
        print(f"  {rank:<8} {taxon[:34]:<34} genomes {records[-1]['n_genomes']:>5}  "
              f"raw {auc_raw:.4f}  treated {auc_treated:.4f}  "
              f"gain {records[-1]['gain']:+.4f}", flush=True)
        del matrix, raw_unit

    frame = pd.DataFrame(records)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # ---- the confounded view, reported so it is visible rather than quoted ----
    #
    # gain = auc_treated - auc_raw and headroom = 1 - auc_raw share auc_raw, so
    # regressing one on the other is partly definitional: if the treatment landed
    # every stratum at one ceiling, the relation would hold by construction with
    # no empirical content. Both quantities below are printed to make the size of
    # that artifact explicit.
    print("\n=== SHARED-TERM CHECK (read before quoting any correlation) ===")
    tracking = float(np.corrcoef(frame.auc_raw, frame.auc_treated)[0, 1])
    ceiling_slope = float(np.polyfit(frame.auc_raw, frame.auc_treated, 1)[0])
    print(f"  corr(auc_raw, auc_treated) = {tracking:+.3f}   "
          f"slope of treated on raw = {ceiling_slope:+.3f}")
    print(f"  sd(auc_raw) {frame.auc_raw.std():.4f} -> sd(auc_treated) "
          f"{frame.auc_treated.std():.4f}  "
          f"({frame.auc_raw.std() / frame.auc_treated.std():.1f}x compression)")
    print("  slope 0 would mean a pure ceiling and an empty 'law'; slope 1 would")
    print("  mean a constant additive gain. Between the two, the informative")
    print("  quantity is the fraction of headroom recovered, below.")

    # ---- the quantity that survives: fraction of headroom recovered ----
    frame["recovered"] = frame.gain / frame.headroom
    rng_boot = np.random.default_rng(args.seed)
    draws = [np.median(rng_boot.choice(frame.recovered.to_numpy(), len(frame),
                                       replace=True)) for _ in range(2000)]
    low, high = np.percentile(draws, [2.5, 97.5])
    print("\n=== RECOVERED FRACTION (gain / headroom) ===")
    print(f"  median {frame.recovered.median():.3f}   "
          f"IQR [{frame.recovered.quantile(.25):.3f}, {frame.recovered.quantile(.75):.3f}]   "
          f"95% CI [{low:.3f}, {high:.3f}]")
    print("\n  by taxonomic breadth:")
    by_rank = (frame.groupby("rank").recovered.agg(["size", "median"])
               .reindex([name for _, name in RANKS][::-1] + ["all"]).dropna())
    for rank, row in by_rank.iterrows():
        print(f"    {rank:<8} n={int(row['size']):>3}  {row['median']:.3f}")
    print("\n  dependence on candidate predictors:")
    for predictor in ("headroom", "mean_norm", "isotropy", "n_genomes", "n_labelled"):
        r = float(np.corrcoef(frame[predictor], frame.recovered)[0, 1])
        print(f"    corr(recovered, {predictor:<11}) = {r:+.3f}")

    frame.to_csv(args.output, sep="\t", index=False)
    print(f"\nwrote {args.output}  ({len(frame)} strata)")


if __name__ == "__main__":
    main()
