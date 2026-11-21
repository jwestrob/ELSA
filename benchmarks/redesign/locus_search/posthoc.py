#!/usr/bin/env python
"""Post-hoc analyses of stage-2 outputs (DefenseFinder primary).

1. Localization of correct hits at E <= 0.01: boundary precision (target members inside the window /
   window span), coverage, window span, per method; a gain built on wide windows that absorb
   neighbouring systems would show as lower boundary precision.
2. Per-type mean AP on even-genome queries (macro over types and the largest types), per method.
3. Targets found at E <= 0.01 (even queries): split by token reach (f_tok) and embedding reach
   (f_emb = best cosine above the random-pair 99.9th percentile or tail rank <= N_db/1000), and
   targets found by each method but not by M0.
usage: posthoc.py --out DIR"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench  # noqa: E402
import locus_search as ls  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ls.RD / "locus_search/out")
    ap.add_argument("--tag", default="", help="suffix of the stage-2 output files")
    a = ap.parse_args()
    db = ls.Database(vectors=None)
    callers = bench.load_gold(db)
    Q = pd.read_parquet(a.out / "queries.parquet")
    M = pd.read_parquet(a.out / f"final_metrics{a.tag}.parquet")
    M["mkey"] = M.method + "_r" + M["round"].astype(str)
    H = pd.read_parquet(a.out / f"final_hits_E10{a.tag}.parquet")
    H["mkey"] = H.method + "_r" + H["round"].astype(str)
    H = H.merge(Q[["qid", "caller", "split", "type", "n_members", "genome_code"]], on="qid")
    res = {}
    # ---- 1. localization of correct hits ------------------------------------------------------
    loc = {}
    for c in sorted(H.caller.unique()):
        C = callers[c]
        h = H[(H.caller == c) & (H.label == 1) & (H.evalue <= 0.01)]
        if not len(h):
            continue
        mem = C["mem"]
        mptr = np.r_[0, np.cumsum(C["nmem"])]
        mgid = np.concatenate(mem)
        ispan = np.array([m.max() - m.min() + 1 for m in mem])
        r = h.match_row.to_numpy()
        nm = C["nmem"][r]
        hid = np.repeat(np.arange(len(r)), nm)
        off = np.arange(len(hid)) - np.repeat(np.cumsum(nm) - nm, nm)
        g = mgid[mptr[r][hid] + off]
        s0, e0 = h.start_gid.to_numpy(), h.end_gid.to_numpy()
        ov = np.bincount(hid, weights=((g >= s0[hid]) & (g <= e0[hid])).astype(float), minlength=len(r))
        span = e0 - s0 + 1
        d = pd.DataFrame(dict(mkey=h.mkey.to_numpy(), split=h.split.to_numpy(), bp=ov / span, cov=ov / nm, span=span,
                              exact=(ov == nm) & (span == ispan[r])))
        loc[c] = {f"{k[0]}|{k[1]}": dict(hits=int(len(g)), median_boundary_precision=float(g.bp.median()),
                                         mean_boundary_precision=float(g.bp.mean()), mean_coverage=float(g["cov"].mean()),
                                         median_span=float(g.span.median()), mean_span=float(g.span.mean()),
                                         exact_span=float(g.exact.mean()))
                  for k, g in d.groupby(["mkey", "split"])}
    res["localization_correct_hits_E0.01"] = loc
    # ---- 2. per-type AP ----------------------------------------------------------------------------
    D = M[(M.caller == "DefenseFinder") & (M.split == "even") & (M.n_rel > 0)]
    pt = D.groupby(["type", "mkey"]).ap.mean().unstack()
    nq = D[D.mkey == "M0_r1"].groupby("type").size()
    res["defensefinder_even_per_type_ap"] = dict(
        macro_over_types=pt.mean().round(4).to_dict(), types=int(len(pt)),
        largest_types={t: dict(queries=int(nq[t]), **pt.loc[t].round(4).to_dict()) for t in nq.sort_values(ascending=False).index[:12]})
    # ---- 3. which targets are found ------------------------------------------------------------------
    P = pd.read_parquet(a.out / "pairs.parquet")
    found = {}
    for c in sorted(H.caller.unique()):
        qrow = Q[Q.caller == c].set_index("qid").row
        h = H[(H.caller == c) & (H.label == 1) & (H.evalue <= 0.01) & (H.split == "even")]
        Pc = P[P.caller == c].set_index(["q_row", "t_row"])
        per = {}
        sets = {}
        for mk, g in h.groupby("mkey"):
            key = pd.MultiIndex.from_arrays([qrow.loc[g.qid].to_numpy(), g.match_row.to_numpy()])
            key = key.unique()
            sets[mk] = set(key)
            f = Pc.loc[key]
            per[mk] = dict(found=int(len(f)), tok_visible=int((f.f_tok >= 0.5).sum()),
                           tok_invisible=int((f.f_tok < 0.5).sum()),
                           tok_invisible_emb_visible=int(((f.f_tok < 0.5) & (f.f_emb >= 0.5)).sum()),
                           tok_invisible_emb_invisible=int(((f.f_tok < 0.5) & (f.f_emb < 0.5)).sum()))
        even_q = Q[(Q.caller == c) & (Q.split == "even")].row
        Pe = P[(P.caller == c) & P.q_row.isin(even_q)]
        tot = dict(pairs=int(len(Pe)), tok_visible=int((Pe.f_tok >= 0.5).sum()), tok_invisible=int((Pe.f_tok < 0.5).sum()),
                   tok_invisible_emb_visible=int(((Pe.f_tok < 0.5) & (Pe.f_emb >= 0.5)).sum()))
        gained = {}
        if "M0_r1" in sets:
            for mk, s in sets.items():
                if mk == "M0_r1":
                    continue
                extra = list(s - sets["M0_r1"])
                lost = list(sets["M0_r1"] - s)
                fx = Pc.loc[pd.MultiIndex.from_tuples(extra)] if extra else Pc.iloc[:0]
                gained[mk] = dict(found_not_by_M0=len(extra), found_by_M0_not_this=len(lost),
                                  extra_tok_invisible=int((fx.f_tok < 0.5).sum()),
                                  extra_tok_invisible_emb_visible=int(((fx.f_tok < 0.5) & (fx.f_emb >= 0.5)).sum()),
                                  extra_types=fx.type.value_counts().head(8).to_dict() if len(fx) else {})
        found[c] = dict(all_even_pairs=tot, found=per, versus_M0=gained)
    res["targets_found_even_E0.01"] = found
    json.dump(res, open(a.out / f"posthoc{a.tag}.json", "w"), indent=1, default=float)
    print(json.dumps(res, indent=1, default=float)[:6000])


if __name__ == "__main__":
    main()
