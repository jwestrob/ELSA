#!/usr/bin/env python
"""Exploration on the TUNE half only (odd genome_code): reference readings and boundary features.

1. Landscape of the rescue3 arm recomputed and checked against boundary_support.npy.
2. Reference readings scored on odd-genome gold (min 2 members, 200 controls): the engine's full locus table
   (all kinds and levels; the source of every DPANN F1 so far), conserved@2 alone (the default reading), and
   each fixed level alone. Ribosomal check on the odd-genome runs.
3. Boundary features inside the level-2 locus that best covers each odd gold instance / ribosomal run:
   internal (inside the gold span) against extension (rest of the level-2 locus); AUROC per feature.
Writes $RD/boundary/results/explore.json and explore_boundaries.parquet.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import blib as B  # noqa: E402


def auroc(pos, neg):
    from scipy.stats import rankdata
    x = np.r_[pos, neg]
    r = rankdata(x)
    n1, n0 = len(pos), len(neg)
    if n1 == 0 or n0 == 0:
        return None
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, required=True)
    ap.add_argument("--arm", default="rescue3")
    a = ap.parse_args()
    import numba
    numba.set_num_threads(a.threads)
    t0 = time.time()
    out = {}
    L = B.Landscape(B.ARMS[a.arm])
    B.log("landscape", a.arm, "sig pairs", len(L.sp), "instances", len(L.inst_left), "L2 loci", len(L.s2),
          f"{time.time()-t0:.0f}s")
    out["landscape"] = dict(arm=a.arm, sig_pairs=int(len(L.sp)), sig_instances=int(len(L.inst_left)),
                            l2_loci=int(len(L.s2)), reproduced=True,
                            l2_span_quantiles=np.quantile(L.e2 - L.s2 + 1, [.1, .25, .5, .75, .9, .99, 1]).tolist())
    np.save(B.BD / "landscape" / f"{a.arm}_arg.npy", L.arg)
    np.save(B.BD / "landscape" / f"{a.arm}_bJ.npy", L.bJ)
    np.save(B.BD / "landscape" / f"{a.arm}_nspan.npy", L.nspan)

    pkg = B.make_split_pkg(1)
    code = B.odd_even_code()
    runs = B.ribo_runs()
    runs_odd = runs[runs.genome_code % 2 == 1].reset_index(drop=True)
    out["ribo_runs_odd"] = int(len(runs_odd))
    out["ribo_parv_odd"] = int(runs_odd[runs_odd.family == "Parvarchaeaceae"].genome_code.nunique())

    # ---- references --------------------------------------------------------------------
    full = pd.read_parquet(L.rdir / "loci_all_genomes.parquet")
    ref = {}
    per, _ = B.score_table(pkg, full, a.threads)
    ref["full_table"] = B.summarize_per(per)
    B.log("full table", ref["full_table"]["summed_DF_TXSS_excess"])
    for lv in B.LADDER[:11]:
        s, e = B.R.runs_at_least(L.b, L.cs, L.ce, float(lv), 0)
        df = L.table(s, e)
        per, _ = B.score_table(pkg, df, a.threads)
        r = B.summarize_per(per)
        sc = B.ribo_score(runs_odd, df.gid_s.to_numpy(), df.gid_e.to_numpy())
        r["ribo_odd"] = B.ribo_summary(runs_odd, sc)
        r["loci"] = B.locus_stats(df.gid_s.to_numpy(), df.gid_e.to_numpy(), L.u.n)
        ref[f"fixed@{lv}"] = r
        B.log(f"fixed@{lv}", {k: round(v["f1"], 3) for k, v in r.items() if isinstance(v, dict) and "f1" in v},
              "sum", round(r["summed_DF_TXSS_excess"], 4), "ribo", r["ribo_odd"]["resolved"], r["ribo_odd"]["recovered"],
              f"{time.time()-t0:.0f}s")
    out["references_odd"] = ref

    # ---- boundary features around odd gold instances and runs ---------------------------
    inst = pd.read_parquet(pkg / "gold/instances.parquet")
    inst = inst[inst.resolved.fillna(False) & inst.order_trusted.fillna(False) & (inst.n_members_resolved >= 2)]
    lab = L.boundary_labels()
    s2, e2 = L.s2, L.e2
    rows = []

    def add(kind, caller, iid, members):
        members = np.unique(np.asarray(members, dtype=np.int64))
        idx = np.searchsorted(s2, members, side="right") - 1
        ok = (idx >= 0) & (e2[np.maximum(idx, 0)] >= members)
        if not ok.any():
            return
        li = np.bincount(idx[ok]).argmax()
        s, e = s2[li], e2[li]
        a_, z = members.min(), members.max()
        ks = np.arange(s, e)
        internal = (ks >= a_) & (ks < z)
        lb = L.b[ks]
        med = np.median(lb)
        inner_lab = lab[ks[internal]]
        inner_lab = inner_lab[inner_lab >= 0]
        core_lab = np.bincount(inner_lab).argmax() if len(inner_lab) else -9
        gm = L.gene_module[np.arange(s, e + 1)]
        d = pd.DataFrame(dict(kind=kind, caller=caller, iid=iid, k=ks, internal=internal, b=lb, logb=np.log(lb),
                              rel=np.log(lb) - np.log(med), bJ=L.bJ[ks], nspan=L.nspan[ks],
                              same_lab=(lab[ks] == core_lab), locus_len=e - s + 1, span_len=z - a_ + 1,
                              n_members=len(members), ext_left=max(0, a_ - s), ext_right=max(0, e - z),
                              gm_left=gm[ks - s], gm_right=gm[ks - s + 1]))
        rows.append(d)

    for r in inst.itertuples():
        add("gold", r.caller, r.instance_id, r.member_gids)
    for r in runs_odd.itertuples():
        add("ribo", "ribosomal", r.run_id, np.arange(r.gid_start, r.gid_end + 1))
    bd = pd.concat(rows, ignore_index=True)
    bd["same_gm"] = (bd.gm_left == bd.gm_right) & (bd.gm_left >= 0)
    bd.to_parquet(B.BD / "results/explore_boundaries.parquet", index=False)
    feat = {}
    for c, g in bd.groupby("caller"):
        inst_g = g.drop_duplicates("iid")
        f = dict(instances=int(len(inst_g)), boundaries=int(len(g)), internal=int(g.internal.sum()),
                 extension=int((~g.internal).sum()),
                 locus_len_median=float(inst_g.locus_len.median()), span_len_median=float(inst_g.span_len.median()),
                 ext_left_median=float(inst_g.ext_left.median()), ext_right_median=float(inst_g.ext_right.median()),
                 frac_with_extension=float(((inst_g.ext_left + inst_g.ext_right) > 0).mean()))
        for k in ("logb", "rel", "bJ", "nspan", "same_lab", "same_gm"):
            f[f"auroc_{k}"] = auroc(g[k][g.internal].astype(float).to_numpy(), g[k][~g.internal].astype(float).to_numpy())
        for k in ("b", "bJ", "nspan"):
            f[f"{k}_internal_q"] = np.quantile(g[k][g.internal], [.1, .25, .5, .75, .9]).tolist() if g.internal.any() else []
            f[f"{k}_extension_q"] = np.quantile(g[k][~g.internal], [.1, .25, .5, .75, .9]).tolist() if (~g.internal).any() else []
        feat[c] = f
        B.log(c, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in f.items() if not k.endswith("_q")}))
    out["features_odd"] = feat
    out["seconds"] = round(time.time() - t0, 1)
    with open(B.BD / "results/explore.json", "w") as fh:
        json.dump(out, fh, indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x))
    B.log("done", out["seconds"])


if __name__ == "__main__":
    main()
