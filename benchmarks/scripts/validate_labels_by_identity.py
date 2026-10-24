#!/usr/bin/env python3
"""Check the gene-name label against actual sequence identity.

Every bacterial metric so far rests on NCBI's ``gene=`` attribute, which carries
two liabilities: it inherits PGAP's naming decisions, including where a curator
declined to assign a name at all, and it exists only for genomes downloaded with
a GFF. Neither holds for a MAG or a self-called proteome, which is most of what
ELSA is pointed at.

Sequence identity has neither problem. It is model-free, computable for any pair
of proteins, and continuous rather than categorical. This script uses it to ask
whether the label is telling the truth:

* **Are "impostors" actually unrelated?** For each query, its best impostor is
  the highest-scoring gene with a *different* name. If those pairs turn out to
  share high identity, they are homologs the naming missed, the margin metric is
  partly measuring naming noise, and the near-clonal result needs qualifying.
* **Are "true orthologs" actually orthologous?** Same-name pairs should show high
  identity. Where they do not, the name is doing work the sequence does not
  support.
* **Does the transform track identity better than raw?** Correlation between
  cosine and identity is a label-free quality measure that works on any dataset,
  MAG or otherwise.
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
from Bio.Align import PairwiseAligner

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from elsa.transform import fit_transform, l2_normalize  # noqa: E402

REPO = Path(__file__).resolve().parents[2]


def build_aligner() -> PairwiseAligner:
    aligner = PairwiseAligner()
    aligner.mode = "local"
    aligner.open_gap_score = -11
    aligner.extend_gap_score = -1
    try:
        from Bio.Align import substitution_matrices
        aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    except Exception:
        aligner.match_score, aligner.mismatch_score = 2, -1
    return aligner


def identity(a: str, b: str, aligner: PairwiseAligner) -> float:
    """Local-alignment identity over the aligned region."""
    if not a or not b:
        return 0.0
    alignment = aligner.align(a, b)[0]
    matches = 0
    for (a0, a1), (b0, b1) in zip(alignment.aligned[0], alignment.aligned[1]):
        matches += sum(1 for x, y in zip(a[a0:a1], b[b0:b1]) if x == y)
    return matches / max(alignment.shape[1], 1)


def load_sequences(protein_dir: Path) -> Dict[Tuple[str, int], str]:
    """Protein sequences keyed by (contig, CDS end) to match the GFF join."""
    sequences: Dict[str, str] = {}
    for path in sorted(protein_dir.glob("*.faa")):
        header, chunks = None, []
        with path.open() as handle:
            for line in handle:
                if line.startswith(">"):
                    if header:
                        sequences[header] = "".join(chunks)
                    header, chunks = line[1:].split()[0], []
                else:
                    chunks.append(line.strip())
        if header:
            sequences[header] = "".join(chunks)
    return sequences


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--parquet", type=Path,
                        default=REPO / "benchmarks/elsa_output/ecoli/elsa_index/ingest/genes.parquet")
    parser.add_argument("--gff", type=Path, default=REPO / "benchmarks/data/ecoli/annotations")
    parser.add_argument("--proteins", type=Path, default=REPO / "benchmarks/data/ecoli/proteins")
    parser.add_argument("--sample", type=int, default=8000)
    parser.add_argument("--queries", type=int, default=250)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", type=Path,
                        default=REPO / "benchmarks/evaluation/transform_ablation/label_validation.tsv")
    args = parser.parse_args()

    names_in = pq.ParquetFile(args.parquet).schema_arrow.names
    emb_cols = [n for n in names_in if n.startswith("emb_")]
    keep = [c for c in ("gene_id", "sample_id", "contig_id", "start", "end") if c in names_in]
    table = pq.ParquetFile(args.parquet).read(columns=keep + emb_cols).to_pandas()

    by_end: Dict[Tuple[str, int], str] = {}
    for path in sorted(args.gff.glob("*.gff")):
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

    seqs = load_sequences(args.proteins)
    table["sequence"] = table.gene_id.map(seqs)
    have = table.sequence.notna()
    print(f"named genes: {len(table):,} | with a sequence: {int(have.sum()):,} "
          f"({100*have.mean():.1f}%)", flush=True)
    table = table[have].reset_index(drop=True)

    rng = np.random.default_rng(args.seed)
    if len(table) > args.sample:
        table = table.iloc[np.sort(rng.choice(len(table), args.sample, replace=False))]
        table = table.reset_index(drop=True)
    matrix = table[emb_cols].to_numpy(dtype=np.float64)
    labels = table.gene_name.to_numpy()
    seq_list = table.sequence.tolist()
    aligner = build_aligner()

    rows: List[Dict[str, object]] = []
    for condition in ("raw", "soft_zca_a10"):
        projected = (l2_normalize(matrix) if condition == "raw"
                     else l2_normalize(fit_transform("soft_zca", matrix[:len(matrix)//2],
                                                     alpha=0.1).apply(matrix)))
        queries = rng.choice(len(projected), min(args.queries, len(projected)),
                             replace=False)
        sims = projected[queries] @ projected.T
        sims[np.arange(len(queries)), queries] = -np.inf
        same = labels[None, :] == labels[queries][:, None]
        best_pos_idx = np.where(same, sims, -np.inf).argmax(axis=1)
        hard_neg_idx = np.where(same, -np.inf, sims).argmax(axis=1)

        for row, q in enumerate(queries):
            p, n = int(best_pos_idx[row]), int(hard_neg_idx[row])
            rows.append({
                "condition": condition, "query": labels[q],
                "pos_name": labels[p], "neg_name": labels[n],
                "pos_cosine": float(sims[row, p]), "neg_cosine": float(sims[row, n]),
                "pos_identity": identity(seq_list[q], seq_list[p], aligner),
                "neg_identity": identity(seq_list[q], seq_list[n], aligner),
            })
        print(f"  {condition}: scored {len(queries)} queries", flush=True)

    frame = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, sep="\t", index=False)

    print("\n=== IS THE LABEL TELLING THE TRUTH? ===")
    for condition, part in frame.groupby("condition"):
        print(f"\n{condition}:")
        print(f"  same-name pairs   identity: median {part.pos_identity.median():.3f}  "
              f"q10 {part.pos_identity.quantile(.1):.3f}")
        print(f"  'impostor' pairs  identity: median {part.neg_identity.median():.3f}  "
              f"q90 {part.neg_identity.quantile(.9):.3f}")
        high = (part.neg_identity > 0.30).mean()
        print(f"  impostors with >30% identity (likely real homologs): {100*high:.1f}%")
        r = float(np.corrcoef(part.neg_cosine, part.neg_identity)[0, 1])
        print(f"  corr(cosine, identity) among impostors: r = {r:+.3f}")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
