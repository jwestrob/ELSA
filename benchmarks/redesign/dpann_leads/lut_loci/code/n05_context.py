#!/usr/bin/env python
"""lut_loci n05 (biotite login node): genome-wide presence of the candidate and control families, and co-occurrence with
the lut cluster and with the lanthanide-dependent methanol dehydrogenase (K23995), against matched backgrounds.

Family genome-wide members: every gene in the rescue3 tokens that hold >= 5% of the family's window members (n02 families,
data/window_families.parquet joined to window_genes). A gene counts as 'at a lut locus' when it lies inside an n01 window.
Matched background: genomes of the same genera (genera that hold at least one lut-carrying and one lut-negative genome);
lut-carrying = lut_status S1-S5 (main locus present); lut-negative = S0. Mantel-Haenszel odds ratio across genera, with
the exact conditional test replaced by a permutation of lut status within genus (2,000 draws, seed 17).
K23995 genes: data/mdh_pkg.parquet (n04). Writes data/context_presence.parquet and data/context.json.
usage: n05_context.py   (run from the workstream remote folder)
"""
import json

import numpy as np
import pandas as pd

RD = "/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111"
PK = RD + "/slice_lut/testset/package_k8"
FAMS = {"DUF3280": 13, "lutC": 8, "lutG": 5, "lutI": 25, "lutD_LanD": 24, "lanM": 43, "fam55": 55, "fam57": 57,
        "fam28": 28, "fam42": 42, "fam19": 19, "lutA": 2, "lutB": 1, "lutE": 0, "lutF": 3, "lutH": 14, "PQQ_fam6": 6}
rng = np.random.default_rng(17)
gn = pd.read_parquet(PK + "/universe/genes.parquet", columns=["gid", "genome_code"])
gm = pd.read_parquet(PK + "/universe/genomes.parquet")
tok = pd.read_parquet(RD + "/long_loci/frozen/tok/mmseqs__seq__e3c50s75_rescue3__lut.parquet").token.to_numpy()
W = pd.read_parquet("data/window_genes.parquet")
F = pd.read_parquet("data/window_families.parquet")
W = W.merge(F, on=["instance_id", "gid"])
Wu = W.drop_duplicates("gid")
wset = set(W.gid)
gc = gn.genome_code.to_numpy()
mdh = pd.read_parquet("data/mdh_pkg.parquet")
gm["lut"] = gm.lut_status.isin(["S1_complete", "S2_receptor_transporter", "S3_receptor_module", "S4_split",
                                "S5_transporter_only"])
gm["lut_neg"] = gm.lut_status == "S0_negative"
pres = pd.DataFrame({"genome_code": gm.genome_code})
for ko in ["K23995", "K00114", "K14028"]:
    s = set(gc[mdh[mdh.ko == ko].gid.to_numpy()])
    pres[ko] = pres.genome_code.isin(s)
rep = {"family_tokens": {}}
for name, f in FAMS.items():
    x = Wu[Wu.fam == f]
    vc = x.token.value_counts()
    tks = vc[vc >= max(1, 0.05 * len(x))].index.to_numpy()
    genes = np.flatnonzero(np.isin(tok, tks))
    atl = np.array([g in wset for g in genes])
    rep["family_tokens"][name] = dict(window_members=int(len(x)), tokens=[int(t) for t in tks], genome_wide_genes=int(len(genes)),
                                      genes_in_lut_windows=int(atl.sum()))
    pres[name] = pres.genome_code.isin(set(gc[genes]))
    pres[name + "_offlocus"] = pres.genome_code.isin(set(gc[genes[~atl]]))
P = gm[["genome_code", "genome", "genus", "family", "ord", "class", "layer", "lut_status", "lut", "lut_neg",
        "completeness"]].merge(pres, on="genome_code")
P.to_parquet("data/context_presence.parquet")


def mh(df, col, grp="lut"):
    """Mantel-Haenszel OR of presence vs lut status, strata = genus with both statuses; permutation p (within genus)."""
    d = df[df.lut | df.lut_neg].copy()
    ok = d.groupby("genus").lut.agg(lambda s: s.any() and (~s).any())
    d = d[d.genus.isin(ok[ok].index)]
    if len(d) == 0:
        return None

    def orr(lutv):
        num = den = 0.0
        for g, idx in d.groupby("genus").indices.items():
            a = (lutv[idx] & d[col].to_numpy()[idx]).sum()
            b = (lutv[idx] & ~d[col].to_numpy()[idx]).sum()
            c = (~lutv[idx] & d[col].to_numpy()[idx]).sum()
            dd = (~lutv[idx] & ~d[col].to_numpy()[idx]).sum()
            n = len(idx)
            num += a * dd / n
            den += b * c / n
        return num / den if den > 0 else np.inf
    obs_l = d.lut.to_numpy()
    o = orr(obs_l)
    ind = list(d.groupby("genus").indices.values())
    null = []
    for _ in range(2000):
        l2 = obs_l.copy()
        for idx in ind:
            l2[idx] = rng.permutation(l2[idx])
        null.append(orr(l2))
    null = np.array(null)
    p = (1 + np.sum(null >= o)) / (1 + len(null))
    return dict(genera=int(d.genus.nunique()), lut_genomes=int(d.lut.sum()), neg_genomes=int((~d.lut).sum()),
                present_lut=int((d.lut & d[col]).sum()), present_neg=int((~d.lut & d[col]).sum()),
                mh_or=(None if not np.isfinite(o) else round(float(o), 2)), perm_p=round(float(p), 5))


out = {}
for name in list(FAMS) + ["K23995"]:
    r = dict(all_lut=f"{int((P.lut & P[name]).sum())}/{int(P.lut.sum())}",
             all_neg=f"{int((P.lut_neg & P[name]).sum())}/{int(P.lut_neg.sum())}",
             genera_present=int(P[P[name]].genus.nunique()))
    r["matched_genus"] = mh(P, name)
    if name in FAMS:
        r["offlocus_in_lut"] = f"{int((P.lut & P[name + '_offlocus']).sum())}/{int(P.lut.sum())}"
        r["with_K23995_lut"] = f"{int((P.lut & P[name] & P.K23995).sum())}/{int((P.lut & P[name]).sum())}"
        r["with_K23995_neg"] = f"{int((P.lut_neg & P[name] & P.K23995).sum())}/{int((P.lut_neg & P[name]).sum())}"
    out[name] = r
# K23995 association with each candidate, within lut-carrying genomes (does the candidate track xoxF beyond lut?)
for name in FAMS:
    d = P[P.lut]
    a = int((d[name] & d.K23995).sum()); b = int((d[name] & ~d.K23995).sum())
    c = int((~d[name] & d.K23995).sum()); dd = int((~d[name] & ~d.K23995).sum())
    out[name]["in_lut_genomes_cand_x_K23995"] = [a, b, c, dd]
rep["results"] = out
json.dump(rep, open("data/context.json", "w"), indent=1)
print(json.dumps(out, indent=1))
