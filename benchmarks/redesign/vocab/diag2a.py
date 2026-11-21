#!/usr/bin/env python
"""Round-2A diagnostics (gold used for evaluation only).

unitsize --src runs2a/neff_vec50 --tau 10
   Committed words of a context-N_eff vocabulary: N_eff/n (= f) and N_eff by merge round and by whether the word's
   instances lie in ribosomal runs; Spearman of f with word size. Does a growing span lift the context out of the unit?
casid --src <class vocabulary> --ref runs/main --tau 10 --tau-ref 10
   CRISPR-Cas gold instances (>= 2 members): the plurality top-level word (>= 50% of members) and its label; per label,
   the distinct CCTyper subtypes it covers, its groups against the best single exact type inside it, and members
   outside every word; class vocabulary against the reference vocabulary.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

RD = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111")
VD = RD / "vocab"
PKG = RD / "ws_dpann/package"
sys.path.insert(0, str(VD / "code"))
import vocab as V  # noqa: E402


def load(src, tau):
    d = VD / src / f"tau{tau:g}" / "obs"
    return pd.read_parquet(d / "words.parquet"), dict(np.load(d / "inst.npz")), dict(np.load(d / "live.npz"))


def ribo_mask(n):
    r = pd.read_parquet(RD / "ribosomal/gold/runs.parquet")
    r = r[(r.run_set == "g2") & (r.n_marked >= 16)]
    m = np.zeros(n + 1, dtype=np.int32)
    np.add.at(m, r.gid_start.to_numpy(), 1)
    np.add.at(m, r.gid_end.to_numpy() + 1, -1)
    return np.cumsum(m)[:n] > 0


def cmd_unitsize(a):
    cc, gc, _ = V.load_universe()
    w, I, _ = load(a.src, a.tau)
    rib = ribo_mask(len(cc))
    cs = np.r_[0, np.cumsum(rib)]
    frac = (cs[I["e"] + 1] - cs[I["s"]]) / (I["e"] - I["s"] + 1)
    span = I["e"] - I["s"] + 1
    t = pd.DataFrame(dict(sym=I["sym"], frac=frac, span=span)).groupby("sym").agg(frac=("frac", "mean"), span=("span", "median"))
    w = w.merge(t, left_on="sym", right_index=True, how="left")
    w["neff"] = w.f * w.n
    w["ribo"] = w.frac >= 0.5
    out = dict(src=a.src, words=int(len(w)), spearman_f_span=float(spearmanr(w.f, w.span).correlation))
    for lab, g in (("ribosomal", w[w.ribo]), ("other", w[~w.ribo])):
        out[lab] = dict(words=int(len(g)), f_q=g.f.quantile([.1, .5, .9]).round(4).tolist(),
                        neff_q=g.neff.quantile([.1, .5, .9]).round(2).tolist(), n_q=g.n.quantile([.1, .5, .9]).tolist(),
                        by_round={int(r): dict(words=int(len(x)), f_med=round(float(x.f.median()), 4),
                                              neff_med=round(float(x.neff.median()), 2), span_med=float(x.span.median()))
                                  for r, x in g.groupby("round")})
    print(json.dumps(out, indent=1))
    json.dump(out, open(VD / "results" / f"diag2a_unitsize_{a.src.replace('/', '_')}.json", "w"), indent=1)


def cas_view(src, tau, inst, mem, gcode, grp, n):
    w, I, L = load(src, tau)
    top = ~L["term"]
    ts, te, lab = L["ns"][top], L["ne"][top], L["sym"][top]
    exact = I["sym"][L["ninst"][top]]
    lens = te - ts + 1
    idx = np.repeat(np.arange(len(ts)), lens)
    off = np.r_[0, np.cumsum(lens)[:-1]]
    genes = ts[idx] + (np.arange(len(idx)) - off[idx])
    tog = np.full(n, -1, dtype=np.int64)
    tog[genes] = idx
    m = mem.copy()
    m["top"] = tog[m.gid.to_numpy()]
    tot = m.groupby("instance_id").size()
    mm = m[m.top >= 0].groupby(["instance_id", "top"]).size().rename("k").reset_index()
    mm = mm.sort_values(["instance_id", "k"], ascending=[True, False]).drop_duplicates("instance_id")
    mm = mm[mm.k / tot.loc[mm.instance_id].to_numpy() >= 0.5]
    mm["label"] = lab[mm.top.to_numpy()]
    mm["exact"] = exact[mm.top.to_numpy()]
    mm = mm.merge(inst[["instance_id", "system_subtype", "genome"]], on="instance_id")
    mm = mm[mm.genome.map(gcode) % 2 == 0]
    g_of_top = grp[gc_of_gene[ts]]
    lab_groups = pd.Series(g_of_top).groupby(lab).nunique()
    ex_groups = pd.Series(g_of_top).groupby(exact).nunique()
    per = []
    for L_, g in mm.groupby("label"):
        exs = np.unique(exact[lab == L_])
        per.append(dict(label=int(L_), cas_instances=int(len(g)), subtypes=int(g.system_subtype.nunique()),
                        class_groups=int(lab_groups.loc[L_]), best_exact_groups=int(ex_groups.loc[exs].max()),
                        exact_types=int(len(exs))))
    P = pd.DataFrame(per)
    outside = 1 - (m.top >= 0).mean()
    return dict(cas_instances_with_word=int(len(mm)), labels=int(len(P)),
                labels_multi_subtype=int((P.subtypes >= 2).sum()) if len(P) else 0,
                instances_in_multi_subtype_labels=int(P.cas_instances[P.subtypes >= 2].sum()) if len(P) else 0,
                median_groups_gain=float((P.class_groups / P.best_exact_groups).median()) if len(P) else None,
                labels_pooling_types=int((P.exact_types >= 2).sum()) if len(P) else 0,
                members_outside_words=float(outside)), P


gc_of_gene = None


def cmd_casid(a):
    global gc_of_gene
    cc, gc, _ = V.load_universe()
    gc_of_gene = gc
    grp = V.load_groups("main", int(gc.max()) + 1)
    inst = pd.read_parquet(PKG / "gold/instances.parquet", columns=["instance_id", "caller", "system_subtype", "genome"])
    inst = inst[inst.caller == "CRISPRCasTyper"]
    mem = pd.read_parquet(PKG / "gold/members.parquet")
    mem = mem[mem.caller == "CRISPRCasTyper"]
    k = mem.groupby("instance_id").gid.transform("size")
    mem = mem[k >= 2]
    gm = pd.read_parquet(PKG / "universe/genomes.parquet", columns=["genome", "genome_code"])
    gcode = dict(zip(gm.genome, gm.genome_code))
    out = {}
    for name, src, tau in (("class", a.src, a.tau), ("reference", a.ref, a.tau_ref)):
        s, P = cas_view(src, tau, inst, mem, gcode, grp, len(cc))
        out[name] = dict(src=src, **s)
        P.to_parquet(VD / "results" / f"diag2a_casid_{name}.parquet", index=False)
    print(json.dumps(out, indent=1))
    json.dump(out, open(VD / "results" / f"diag2a_casid_{a.src.replace('/', '_')}.json", "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["unitsize", "casid"])
    ap.add_argument("--src", required=True)
    ap.add_argument("--tau", type=float, default=10.0)
    ap.add_argument("--ref", default="runs/main")
    ap.add_argument("--tau-ref", type=float, default=10.0)
    ap.add_argument("--threads", type=int, default=1)
    a = ap.parse_args()
    {"unitsize": cmd_unitsize, "casid": cmd_casid}[a.cmd](a)


if __name__ == "__main__":
    main()
