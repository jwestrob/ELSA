"""Candidate 2 fold analysis (local, seconds) -> work/c2_interfaces.tsv, work/c2_summary.json.

For every chain pair of ranks 34 and 7 and both replicates:
  full-length complex (screen 2, s2_calls.tsv), truncated complex (single, MSA), two-chain pair full length,
  two-chain pair truncated; prepilin-like controls (two-chain, X kept with a control replacing Y and vice
  versa; full length 2 draws per side, truncated 1 draw per side).
Interface pass per replicate (screen-2 rule): pair ipTM >= 0.75, >= 10 contacts, interface PAE <= 5 A, interface
pLDDT >= 70. Control rule: cognate pair ipTM minus every control pair ipTM of the same variant >= 0.30.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dd_pairs import pair_metrics

HERE = Path(__file__).resolve().parent
W = HERE / "work"
F = W / "folds_c2"


def main():
    man = pd.read_csv(W / "c2_fold_manifest.tsv", sep="\t", dtype={"rank": str})
    s2 = pd.read_csv(HERE.parent / "screen2_tables/s2_calls.tsv", sep="\t")
    rows = []
    for rk in ("034", "007"):
        k = 4 if rk == "034" else 3
        ids = "ABCD"[:k]
        for rep in (1, 2):
            cs, _ = pair_metrics(F, f"c2_cplxT_{rk}_r{rep}")
            cm, _ = pair_metrics(F, f"c2_cplxTmsa_{rk}_r{rep}")
            for i in range(k):
                for j in range(i + 1, k):
                    pr = ids[i] + ids[j]
                    r = dict(rank=rk, rep=rep, pair=pr)
                    for mode in ("single", "msa"):
                        x = s2[(s2["mode"] == mode) & (s2.s2_rank == int(rk)) & (s2.i == i) & (s2.j == j)].iloc[0]
                        tag = "full_cplx" if mode == "single" else "full_cplx_msa"
                        r[f"{tag}_iptm"] = round(x[f"r{rep}_pair_iptm"], 3)
                        r[f"{tag}_pae"] = x[f"r{rep}_iface_pae"]
                        r[f"{tag}_contacts"] = x[f"r{rep}_contacts"]
                        r[f"{tag}_pass"] = bool(x[f"r{rep}_pass"])
                        r[f"{tag}_hit_s2"] = bool(x["interface_hit"])
                    for tag, d in (("trunc_cplx", cs), ("trunc_cplx_msa", cm)):
                        r[f"{tag}_iptm"] = round(d[pr]["pair_iptm"], 3)
                        r[f"{tag}_pae"] = d[pr]["iface_pae"] and round(d[pr]["iface_pae"], 2)
                        r[f"{tag}_contacts"] = d[pr]["contacts"]
                        r[f"{tag}_pass"] = d[pr]["pass"]
                    for tag, fid in (("pair", f"c2_pair_{rk}_r{rep}_{pr}"), ("pairT", f"c2_pairT_{rk}_r{rep}_{pr}")):
                        d, _ = pair_metrics(F, fid)
                        r[f"{tag}_iptm"] = round(d["AB"]["pair_iptm"], 3)
                        r[f"{tag}_pae"] = d["AB"]["iface_pae"] and round(d["AB"]["iface_pae"], 2)
                        r[f"{tag}_contacts"] = d["AB"]["contacts"]
                        r[f"{tag}_pass"] = d["AB"]["pass"]
                    for tag, pref in (("ctl", "c2_ctl_"), ("ctlT", "c2_ctlT_")):
                        ff = man[man.fold_id.str.startswith(f"{pref}{rk}_r{rep}_{pr}_")].fold_id
                        vals = []
                        for fid in ff:
                            d, _ = pair_metrics(F, fid)
                            vals.append((fid[-2:], round(d["AB"]["pair_iptm"], 3), d["AB"]["pass"]))
                        r[f"{tag}_iptms"] = ";".join(f"{a}:{b}" for a, b, _ in vals)
                        r[f"{tag}_max"] = max(v for _, v, _ in vals)
                        r[f"{tag}_n_pass"] = sum(p for _, _, p in vals)
                        r[f"{tag}_n"] = len(vals)
                    r["delta_full"] = round(r["pair_iptm"] - r["ctl_max"], 3)
                    r["delta_trunc"] = round(r["pairT_iptm"] - r["ctlT_max"], 3)
                    rows.append(r)
    T = pd.DataFrame(rows)
    T.to_csv(W / "c2_interfaces.tsv", sep="\t", index=False)
    pd.set_option("display.width", 300)
    cols = ["rank", "rep", "pair", "full_cplx_iptm", "full_cplx_pass", "full_cplx_hit_s2", "trunc_cplx_iptm",
            "trunc_cplx_pae", "trunc_cplx_pass", "trunc_cplx_msa_iptm", "trunc_cplx_msa_pass", "pair_iptm", "pair_pass",
            "pairT_iptm", "pairT_pae", "pairT_pass", "ctl_iptms", "ctl_n_pass", "ctlT_iptms", "delta_full", "delta_trunc"]
    print(T[cols].to_string())
    # interface-level summary: survives truncation in both replicates?
    S = {}
    for (rk, pr), x in T.groupby(["rank", "pair"]):
        S[f"{rk}_{pr}"] = dict(
            s2_hit=bool(x.full_cplx_hit_s2.iloc[0]),
            trunc_cplx_pass_both=bool(x.trunc_cplx_pass.all()), trunc_cplx_msa_pass_both=bool(x.trunc_cplx_msa_pass.all()),
            pair_pass_both=bool(x.pair_pass.all()), pairT_pass_both=bool(x.pairT_pass.all()),
            ctl_pass_any=int(x.ctl_n_pass.sum()), ctlT_pass_any=int(x.ctlT_n_pass.sum()),
            delta_full_min=float(x.delta_full.min()), delta_trunc_min=float(x.delta_trunc.min()),
            full_ctrl_rule=bool((x.delta_full >= 0.30).all()), trunc_ctrl_rule=bool((x.delta_trunc >= 0.30).all()))
    allctl = pd.concat([T[["ctl_iptms"]].assign(v=T.ctl_iptms.str.split(";")).explode("v")])
    cv = allctl.v.str.split(":").str[1].astype(float)
    S["controls_full"] = dict(n=int(len(cv)), median=float(cv.median()), q90=float(cv.quantile(0.9)),
                              ge075=int((cv >= 0.75).sum()), max=float(cv.max()))
    ct = T.ctlT_iptms.str.split(";").explode().str.split(":").str[1].astype(float)
    S["controls_trunc"] = dict(n=int(len(ct)), median=float(ct.median()), q90=float(ct.quantile(0.9)),
                               ge075=int((ct >= 0.75).sum()), max=float(ct.max()))
    S["cognate_pairs_full"] = dict(n=len(T), median=float(T.pair_iptm.median()), ge075=int((T.pair_iptm >= 0.75).sum()))
    S["cognate_pairs_trunc"] = dict(n=len(T), median=float(T.pairT_iptm.median()), ge075=int((T.pairT_iptm >= 0.75).sum()))
    json.dump(S, open(W / "c2_summary.json", "w"), indent=1)
    print(json.dumps(S, indent=1))
    for fid in ("c2_cplxT_034_r1", "c2_cplxT_034_r2", "c2_cplxT_007_r1", "c2_cplxT_007_r2", "c2_cplxTmsa_034_r1",
                "c2_cplxTmsa_034_r2", "c2_cplxTmsa_007_r1", "c2_cplxTmsa_007_r2"):
        _, m = pair_metrics(F, fid)
        print(fid, "complex ipTM", round(m["iptm"], 3), "chain pLDDT", m["chain_plddt"])


if __name__ == "__main__":
    main()
