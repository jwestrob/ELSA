#!/usr/bin/env python3
"""Check B sample order (predeclared_verify.json): strata over labeled proteins of 40-600 aa, one default_rng(20261121),
permutation of each stratum (gid order) drawn S1, S2, S3; the first 200 of each order go to partner extraction."""
from pathlib import Path
import numpy as np, pandas as pd
H = Path(__file__).resolve().parents[1]
d = pd.read_parquet(H.parent / "pass/dpann_slot_rescue_pass.parquet").sort_values("gid")
d = d[(d.length >= 40) & (d.length <= 600)]
S = {"S1": d[(d.votes_for >= 10) & (d.vote_share > 0.8)], "S2": d[(d.votes_for <= 9) & (d.vote_share > 0.8)], "S3": d[d.vote_share <= 0.8]}
rng = np.random.default_rng(20261121)
rows = []
for s in ("S1", "S2", "S3"):
    g = S[s]
    perm = rng.permutation(len(g))
    for o, i in enumerate(perm[:200]):
        r = g.iloc[i]
        rows.append(dict(stratum=s, order=o, gid=int(r.gid), protein_id=r.protein_id, label=r.proposed_label_positional_homology,
                         label_name=r.proposed_label_name, length=int(r.length), votes_for=int(r.votes_for), vote_share=float(r.vote_share),
                         confidence_bin=r.confidence_bin, stratum_size=len(g)))
out = pd.DataFrame(rows)
out.to_csv(H / "data/B_order.tsv", sep="\t", index=False)
print(out.groupby("stratum").agg(n=("gid", "size"), stratum_size=("stratum_size", "first")).to_string())
