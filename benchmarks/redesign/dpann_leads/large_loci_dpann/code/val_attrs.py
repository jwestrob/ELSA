#!/usr/bin/env python
"""Per-locus validation attributes for the 34,530 large loci (ribosomal-Pfam gene count, curated system types, the
ribosomal benchmark runs mapped to loci) and the validation of any partition column (predeclared.json bars).
usage: val_attrs.py SCRATCH  -> SCRATCH/loci_val.parquet; prints validation of fam_f33 / fam_f25 / fam_f50 (v1)
import: validate(L, col) with L holding locus, genome_code, n_ribo, cur_types, ribo_runs and the partition column
"""
import json
import sys
from collections import Counter

import numpy as np
import pandas as pd


def validate(L, col, mask=None):
    """L: locus, n_ribo, cur_types, ribo_runs (count of benchmark runs mapped to this locus), partition column col
    (-1 = singleton). mask: optional boolean restricting the loci evaluated (e.g. odd genome codes)."""
    X = L if mask is None else L[mask]
    unit = np.where(X[col] >= 0, X[col], -1 - X.locus)
    out = {}
    # ribosomal: runs mapped to loci (a locus can host more than one run)
    w = X.ribo_runs.to_numpy()
    cnt = Counter()
    for u, k in zip(unit, w):
        if k:
            cnt[u] += k
    tot = sum(cnt.values())
    top, ntop = cnt.most_common(1)[0]
    cum = np.cumsum(sorted(cnt.values(), reverse=True)) / tot
    memb = L[L[col] == top] if top >= 0 else L[L.locus == -1 - top]
    out["ribo"] = dict(runs=int(tot), units=len(cnt), largest_share=round(ntop / tot, 4),
                       families_for_90pct=int(np.searchsorted(cum, 0.9) + 1), largest_members=int(len(memb)),
                       largest_purity=round(float((memb.n_ribo >= 5).mean()), 4))
    # curated systems
    T = X[X.cur_types != ""].assign(t=X.cur_types.str.split(";"), unit=unit[(X.cur_types != "").to_numpy()]).explode("t")
    types = {}
    for t, sub in T.groupby("t"):
        if len(sub) < 10:
            continue
        c = Counter(sub.unit)
        cum = np.cumsum(sorted(c.values(), reverse=True)) / len(sub)
        types[t] = dict(loci=int(len(sub)), largest_share=round(c.most_common(1)[0][1] / len(sub), 3),
                        units_for_90pct=int(np.searchsorted(cum, 0.9) + 1),
                        in_families=round(float((sub[col] >= 0).mean()), 3))
    pn, pw, nf = 0, 0.0, 0
    called = X[(X.cur_types != "") & (X[col] >= 0)]
    for f, sub in called.groupby(col):
        if len(sub) < 3:
            continue
        ts = Counter(t for x in sub.cur_types for t in x.split(";"))
        dom = ts.most_common(1)[0][0]
        p = float(np.mean([dom in x.split(";") for x in sub.cur_types]))
        pn += len(sub)
        pw += p * len(sub)
        nf += 1
    out["systems"] = dict(purity=round(pw / max(pn, 1), 4), families_ge3_called=nf, types=types)
    t4p = types.get("TXSS:Archaeal-T4P", {}).get("largest_share", 0)
    out["bars"] = dict(ribo_largest_ge80=bool(out["ribo"]["largest_share"] >= 0.8),
                       ribo_le5_for90=bool(out["ribo"]["families_for_90pct"] <= 5),
                       ribo_purity_ge80=bool(out["ribo"]["largest_purity"] >= 0.8),
                       sys_purity_ge80=bool(out["systems"]["purity"] >= 0.8), t4p_largest_ge50=bool(t4p >= 0.5))
    out["pass"] = all(out["bars"].values())
    return out


def main(sp):
    L = pd.read_parquet(sp + "/out_v1/loci.parquet")
    U = pd.read_parquet(sp + "/genes.parquet", columns=["gid", "protein_id", "contig_code"]).sort_values("gid")
    A = pd.read_parquet(sp + "/annot_all.parquet", columns=["protein_id", "pfam", "cur_systems"])
    U = U.merge(A, on="protein_id", how="left").sort_values("gid").reset_index(drop=True)
    rib = U.pfam.fillna("").str.contains("(?:^|;)Ribosomal_", regex=True).to_numpy()
    loc_of = np.full(len(U), -1)
    for r in L.itertuples():
        loc_of[r.s:r.e + 1] = r.locus
    L["n_ribo"] = np.bincount(loc_of[rib & (loc_of >= 0)], minlength=len(L))
    cs = U[(loc_of >= 0) & U.cur_systems.notna()].assign(locus=loc_of[(loc_of >= 0) & U.cur_systems.notna().to_numpy()])
    ct = cs.assign(t=cs.cur_systems.str.split(";")).explode("t").groupby("locus").t.agg(lambda x: ";".join(sorted(set(x))))
    L["cur_types"] = L.locus.map(ct).fillna("")
    idx = np.flatnonzero(rib)
    cid = U.contig_code.to_numpy()
    brk = np.flatnonzero((np.diff(idx) > 3) | (cid[idx[1:]] != cid[idx[:-1]])) + 1
    runs = [r for r in np.split(idx, brk) if len(r) >= 16]
    rr = np.zeros(len(L), int)
    unm = 0
    for r in runs:
        lo = loc_of[r[0]:r[-1] + 1]
        lo = lo[lo >= 0]
        if len(lo) == 0:
            unm += 1
            continue
        rr[Counter(lo.tolist()).most_common(1)[0][0]] += 1
    L["ribo_runs"] = rr
    print("ribosomal runs", len(runs), "unmapped", unm, "distinct loci", int((rr > 0).sum()))
    L[["locus", "genome_code", "n_ribo", "cur_types", "ribo_runs"]].to_parquet(sp + "/loci_val.parquet", index=False)
    res = {c: validate(L, c) for c in ["fam_f33", "fam_f25", "fam_f50"]}
    for c, v in res.items():
        print(c, json.dumps(v["ribo"]), "sys purity", v["systems"]["purity"], v["bars"])
    json.dump(dict(runs=len(runs), unmapped=unm, v1=res), open(sp + "/validation_v1.json", "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
