#!/usr/bin/env python
"""EXPLORATORY (written after the final evaluation; not predeclared): does the critic's evaluation-1 advantage survive
when raw-cosine similarity of the two genes is held fixed? Tandem homologs split across families score highest
(evaluation 5), so this tests the self-similarity reading.
  binned  AUROC within deciles of raw cosine (deciles over the evaluated pairs), combined as the
          n_pos * n_neg weighted mean (a cosine-stratified AUROC), per method
  lowcos  AUROC on pairs with raw cosine below 0.839 (the median over all offset-1 adjacencies)
Writes $RD/grammar/eval/explore_cos.tsv."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

RD = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111")
EV = RD / "grammar/eval"
adj = pd.read_parquet(RD / "grammar/prep/adj1.parquet")
rows = []
for tag in ("cv", "phylum"):
    z = np.load(EV / f"assembled_{tag}.npz")
    sc = {k: z[k].astype(np.float64) for k in ("primary", "pmi_a1", "pmi_a01", "eng_nlogP", "raw_cos", "shuf", "w3", "glob")}
    dcf = EV / f"dc_{tag}.npz"
    if dcf.exists():
        d = np.load(dcf)
        for k in ("dc_primary", "dc_shuf"):
            sc[k] = d[k].astype(np.float64)
    cov = z["fold_of"] >= 0
    G3 = z["G3_train"]
    cos = adj.raw_cos.to_numpy()
    for caller in ("DefenseFinder", "TXSScan", "CRISPRCasTyper"):
        pos = adj[f"pos_{caller}"].to_numpy() & cov
        for negname, neg in (("nogold", adj.gold_none.to_numpy() & cov), ("straddle", adj[f"straddle_{caller}"].to_numpy() & cov)):
            for sname, sm in (("all", np.ones(len(adj), bool)), ("never_w3", G3 == 0)):
                p, q = pos & sm, neg & sm
                sel = p | q
                y = p[sel]
                c = cos[sel]
                edges = np.quantile(c, np.linspace(0, 1, 11))
                b = np.clip(np.searchsorted(edges, c, side="right") - 1, 0, 9)
                low = c < 0.839
                for m, s_all in sc.items():
                    s = s_all[sel]
                    num = den = 0.0
                    for k in range(10):
                        mk = b == k
                        npos, nneg = int(y[mk].sum()), int((~y[mk]).sum())
                        if npos and nneg:
                            w = npos * nneg
                            num += w * roc_auc_score(y[mk], s[mk])
                            den += w
                    a_low = roc_auc_score(y[low], s[low]) if y[low].any() and (~y[low]).any() else np.nan
                    rows.append(dict(split=tag, caller=caller, negset=negname, stratum=sname, method=m,
                                     n_pos=int(y.sum()), n_neg=int((~y).sum()), auc_all=roc_auc_score(y, s),
                                     auc_cos_binned=num / den if den else np.nan, n_pos_low=int(y[low].sum()),
                                     n_neg_low=int((~y[low]).sum()), auc_lowcos=a_low,
                                     pos_cos_median=float(np.median(c[y])), neg_cos_median=float(np.median(c[~y]))))
            print(tag, caller, flush=True)
out = pd.DataFrame(rows)
out.to_csv(EV / "explore_cos.tsv", sep="\t", index=False)
x = out[(out.negset == "nogold")]
print(x.pivot_table(index=["split", "caller", "stratum"], columns="method", values="auc_cos_binned").round(3).to_string())
print(x.pivot_table(index=["split", "caller", "stratum"], columns="method", values="auc_lowcos").round(3).to_string())
print(out.groupby(["split", "caller", "negset", "stratum"])[["n_pos", "n_pos_low", "pos_cos_median", "neg_cos_median"]].first().to_string())
