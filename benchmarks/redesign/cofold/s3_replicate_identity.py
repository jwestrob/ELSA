"""Pairwise identity between the two replicate instances of every screen-3 block, per chain (Biopython global
alignment, BLOSUM62, gap open -10 / extend -0.5; identity = identical aligned positions / shorter length).
Writes work/s3/replicate_identity_s3.tsv."""
import json
from pathlib import Path

import pandas as pd
from Bio import Align
from Bio.Align import substitution_matrices

HERE = Path(__file__).resolve().parent
W3 = HERE / "work" / "s3"
al = Align.PairwiseAligner(mode="global", substitution_matrix=substitution_matrices.load("BLOSUM62"),
                           open_gap_score=-10, extend_gap_score=-0.5)


def ident(a, b):
    if a == b:
        return 1.0
    aln = al.align(a, b)[0]
    n = sum(1 for i, j in zip(*aln.indices) if i >= 0 and j >= 0 and a[i] == b[j])
    return n / min(len(a), len(b))


seq = {}
for l in open(W3 / "jobs_s3.jsonl"):
    j = json.loads(l)
    seq[j["fold_id"]] = [c["seq"] for c in j["chains"]]
rows = []
for fid in sorted(f for f in seq if f.startswith("cog3_") and f.endswith("_r1")):
    f2 = fid[:-1] + "2"
    ids = [round(ident(a, b), 3) for a, b in zip(seq[fid], seq[f2])]
    rows.append(dict(s3_rank=int(fid[5:8]), chain_identity=";".join(map(str, ids)), min_identity=min(ids), max_identity=max(ids)))
out = pd.DataFrame(rows)
out.to_csv(W3 / "replicate_identity_s3.tsv", sep="\t", index=False)
print(out.min_identity.describe())
print("blocks with every chain identical between replicates:", int((out.min_identity == 1).sum()),
      "; with max identity >= 0.95:", int((out.max_identity >= 0.95).sum()), "of", len(out))
print(out[out.max_identity >= 0.95].to_string(index=False))
