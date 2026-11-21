#!/usr/bin/env python
"""lut_loci e12 (Mac): T_context for the beta-roll family (rescue3 token 275376) with the n05 definitions: presence per genome
(data/br_members.tsv), lut-carrying (S1-S5) vs same-genus lut-negative (S0) genomes (data/context_presence.parquet),
Mantel-Haenszel OR over genera holding both statuses, permutation of lut status within genus (2,000 draws, seed 17);
K23995 share: family genes within +-8 genes of a K23995 gene (data/xoxf_window.parquet) over 585 K23995 genes.
Also the one-sided Mann-Whitney test of family score_cons against negatives B (tests_rule verdict clause)."""
import json

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

B = "/Users/jacob/Documents/Sandbox/elsa_test/ELSA/benchmarks/redesign/dpann_leads/lut_loci"
rng = np.random.default_rng(17)
M = pd.read_csv(B + "/data/br_members.tsv", sep="\t")
t = M[M.token == 275376]
P = pd.read_parquet(B + "/data/context_presence.parquet")
P["fam"] = P.genome.isin(set(t.genome))
d = P[P.lut | P.lut_neg].copy()
ok = d.groupby("genus").lut.agg(lambda s: s.any() and (~s).any())
d = d[d.genus.isin(ok[ok].index)]
idx = list(d.groupby("genus").indices.values())
f = d.fam.to_numpy()


def orr(l):
    num = den = 0.0
    for ix in idx:
        a = (l[ix] & f[ix]).sum(); b = (l[ix] & ~f[ix]).sum(); c = (~l[ix] & f[ix]).sum(); e = (~l[ix] & ~f[ix]).sum()
        n = len(ix); num += a * e / n; den += b * c / n
    return num / den if den > 0 else np.inf


obs = d.lut.to_numpy()
o = orr(obs)
null = []
for _ in range(2000):
    l2 = obs.copy()
    for ix in idx:
        l2[ix] = rng.permutation(l2[ix])
    null.append(orr(l2))
p = (1 + np.sum(np.array(null) >= o)) / 2001
X = pd.read_parquet(B + "/data/xoxf_window.parquet")
nx = X[X.off == 0].xoxf.nunique()
xs = X[X.gid.isin(t.gid) & (X.off != 0)].xoxf.nunique()
S = pd.read_csv(B + "/data/sites_br.tsv", sep="\t")
A = pd.read_csv(B + "/data/sites_apo.tsv", sep="\t")
mw = mannwhitneyu(S[S.role == "br"].score_cons, A[A.role == "negB"].score_cons, alternative="greater").pvalue
out = dict(all_lut=f"{int((P.lut & P.fam).sum())}/{int(P.lut.sum())}", all_neg=f"{int((P.lut_neg & P.fam).sum())}/{int(P.lut_neg.sum())}",
           matched_genera=int(d.genus.nunique()), matched_lut=f"{int((d.lut & d.fam).sum())}/{int(d.lut.sum())}",
           matched_neg=f"{int((~d.lut & d.fam).sum())}/{int((~d.lut).sum())}", mh_or=(None if not np.isfinite(o) else round(float(o), 2)),
           perm_p=round(float(p), 5), near_K23995=f"{xs}/{nx}", mw_p_score_cons_vs_negB=float(mw))
out["PASS_context"] = bool(((out["mh_or"] is None) or out["mh_or"] >= 5) and p <= 0.01 or xs / nx >= 0.2)
json.dump(out, open(B + "/data/br_context.json", "w"), indent=1)
print(json.dumps(out, indent=1))
