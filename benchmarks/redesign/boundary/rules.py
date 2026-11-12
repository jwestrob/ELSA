#!/usr/bin/env python
"""Locus-reading rules on the engine's support landscape: tuning (odd genomes) and test (even genomes).

usage:
  rules.py tune --arm rescue3 --families F,A,E --threads N      -> results/tune_<arm>_<families>.json
  rules.py select --arm rescue3                                 -> results/selected_<arm>.json (from all tune files)
  rules.py test --arm rescue3 --configs results/selected_rescue3.json --threads N
                                                                -> results/test_<arm>.json, loci/<arm>/*.parquet,
                                                                   scores/<arm>/*.parquet
Rules (predeclared.json): F fixed level, A relative end trimming, B per-locus level (keep_low 0 core / 1 segmentation),
C module-consistent segmentation, D nested (level-2 locus + core), E consistency (runs of max J >= theta).
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import blib as B  # noqa: E402

LADDER_B = np.array([2, 3, 5, 8, 12, 20, 30, 50, 80, 120, 200, 300, 500, 800], dtype=np.float64)
GRID = {
    "F": [dict(level=x) for x in (3, 5, 8, 12, 20, 30, 50, 80, 120, 200)],
    "A": [dict(f=x) for x in (0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0)],
    "B": [dict(h=h, gain=g, keep_low=k) for k in (0, 1) for h in (1, 2, 3) for g in (0, 0.25, 0.5, 1.0, 1.5, 2.0)],
    "C": [dict(res=r) for r in (0.1, 0.3, 1, 3, 10, 30)],
    "E": [dict(theta=t) for t in (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)],
}
GRID["G"] = [dict(res=r, h=h, gain=g) for r in (0.1, 0.3, 1, 3, 10, 30) for h in (1, 2) for g in (0, 0.5, 1.0)]
GRID["H"] = [dict(res=r, f=f) for r in (0.1, 0.3, 1, 3, 10, 30) for f in (0.1, 0.3, 0.5)]
GRID["D"] = ([dict(core="A", **p) for p in GRID["A"]] + [dict(core="B", **p) for p in GRID["B"] if p["keep_low"] == 0]
             + [dict(core="F", **p) for p in GRID["F"]] + [dict(core="C", **p) for p in GRID["C"]]
             + [dict(core="E", **p) for p in GRID["E"]])
CALLERS = ("DefenseFinder", "TXSScan", "CRISPRCasTyper")
# odd-scaled ribosomal thresholds (predeclared): ceil(1100*848/1686), ceil(1650*848/1686), ceil(110*55/114)
ODD_RIBO = dict(resolved=554, recovered=830, parv=54)
ALL_RIBO = dict(resolved=1100, recovered=1650, parv=110)


def name_of(fam, p):
    return fam + "_" + "_".join(f"{k}{v}" for k, v in p.items())


class Builder:
    def __init__(self, L):
        self.L = L
        self.mod_cache = {}

    def modules(self, res, seed=17):
        if (res, seed) in self.mod_cache:
            return self.mod_cache[(res, seed)]
        import networkx as nx
        sp = self.L.sp
        H = nx.Graph()
        wt = sp.J.to_numpy() * np.log2(np.maximum(sp.L.to_numpy(), 2).astype(np.float64))
        H.add_weighted_edges_from(zip(sp.lo.tolist(), sp.hi.tolist(), wt.tolist()))
        tm = np.full(self.L.T, -1, dtype=np.int64)
        for i, c in enumerate(nx.community.louvain_communities(H, weight="weight", resolution=res, seed=seed)):
            if len(c) >= 2:
                tm[np.fromiter(c, dtype=np.int64)] = i
        lo, hi = sp.lo.to_numpy(), sp.hi.to_numpy()
        pm = tm[lo].copy()
        pm[tm[hi] != pm] = -1
        self.mod_cache[(res, seed)] = (tm, pm)
        B.log(f"modules res {res} seed {seed}: {len(np.unique(tm[tm >= 0]))} modules, {int((tm >= 0).sum())} tokens")
        return tm, pm

    def build(self, fam, p):
        L = self.L
        if fam == "R1":
            return L.s2, L.e2
        if fam == "F":
            return B.R.runs_at_least(L.b, L.cs, L.ce, float(p["level"]), 0)
        if fam == "A":
            return B.trim_ends(L.b, L.s2, L.e2, float(p["f"]))
        if fam == "B":
            s, e, _, _, _ = B.per_locus_level(L.b, L.s2, L.e2, LADDER_B, int(p["h"]), float(p["gain"]), int(p["keep_low"]))
            return s, e
        if fam == "C":
            _, pm = self.modules(p["res"], p.get("seed", 17))
            lab = L.boundary_labels(pm)
            s, e, _ = B.label_segments(L.b, lab, L.s2, L.e2)
            return s, e
        if fam == "E":
            return B.R.runs_at_least(L.bJ, L.cs, L.ce, float(p["theta"]), 0)
        if fam in ("G", "H"):
            cs_, ce_ = self.build("C", dict(res=p["res"], seed=p.get("seed", 17)))
            ok = ce_ > cs_
            cs_, ce_ = cs_[ok], ce_[ok]
            if fam == "G":
                s, e, _, _, _ = B.per_locus_level(L.b, cs_, ce_, LADDER_B, int(p["h"]), float(p["gain"]), 1)
                return s, e
            return B.trim_ends(L.b, cs_, ce_, float(p["f"]))
        if fam == "D":
            q = dict(p)
            core = q.pop("core")
            cs_, ce_ = self.build(core, q)
            s = np.r_[L.s2, cs_]
            e = np.r_[L.e2, ce_]
            se = np.unique(np.c_[s, e], axis=0)
            return se[:, 0], se[:, 1]
        raise ValueError(fam)


def evaluate(L, s, e, pkg, runs, threads):
    df = L.table(s, e)
    per, _ = B.score_table(pkg, df, threads)
    r = B.summarize_per(per)
    sc = B.ribo_score(runs, df.gid_s.to_numpy(), df.gid_e.to_numpy())
    r["ribo"] = B.ribo_summary(runs, sc)
    r["loci"] = B.locus_stats(df.gid_s.to_numpy(), df.gid_e.to_numpy(), L.u.n)
    return r, df, per, sc


def conditions(r, ref, thr):
    parv = int(str(r["ribo"]["parvarchaeaceae"]).split("/")[0])
    c1 = (r["ribo"]["resolved"] >= thr["resolved"]) and (r["ribo"]["recovered"] >= thr["recovered"]) and (parv >= thr["parv"])
    c2 = (r["summed_DF_TXSS_excess"] - ref["summed_DF_TXSS_excess"]) >= 0.02
    c3 = all(r[c]["f1"] - ref[c]["f1"] >= -0.01 for c in CALLERS)
    return [bool(c1), bool(c2), bool(c3)]


def cmd_tune(a):
    L = B.Landscape(B.ARMS[a.arm])
    bld = Builder(L)
    pkg = B.make_split_pkg(1)
    runs = B.ribo_runs()
    runs = runs[runs.genome_code % 2 == 1].reset_index(drop=True)
    t0 = time.time()
    ref, *_ = evaluate(L, L.s2, L.e2, pkg, runs, a.threads)
    out = dict(arm=a.arm, split="odd", R1=ref, configs=[])
    if "C" in a.families:
        tm, pm = bld.modules(10)
        out["module_res10_reproduces_engine"] = bool(np.array_equal(pm, L.pair_module))
        B.log("res-10 Louvain reproduces engine pair modules:", out["module_res10_reproduces_engine"])
    for fam in a.families.split(","):
        for p in GRID[fam]:
            s, e = bld.build(fam, p)
            r, *_ = evaluate(L, s, e, pkg, runs, a.threads)
            r.update(family=fam, params=p, name=name_of(fam, p), conditions_odd=conditions(r, ref, ODD_RIBO),
                     d_summed=r["summed_DF_TXSS_excess"] - ref["summed_DF_TXSS_excess"],
                     d_f1={c: r[c]["f1"] - ref[c]["f1"] for c in CALLERS})
            out["configs"].append(r)
            B.log(r["name"], "cond", r["conditions_odd"], "dsum", round(r["d_summed"], 4),
                  "dF1", {c[:4]: round(v, 3) for c, v in r["d_f1"].items()},
                  "ribo", r["ribo"]["resolved"], r["ribo"]["recovered"], r["ribo"]["parvarchaeaceae"],
                  "loci", r["loci"]["loci"], f"{time.time()-t0:.0f}s")
    fn = B.BD / "results" / f"tune_{a.arm}_{a.families.replace(',', '')}.json"
    with open(fn, "w") as fh:
        json.dump(out, fh, indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x))
    B.log("wrote", fn)


def cmd_select(a):
    rows = []
    for fn in sorted((B.BD / "results").glob(f"tune_{a.arm}_*.json")):
        d = json.load(open(fn))
        rows += d["configs"]
    sel = {}
    for fam in sorted({r["family"] for r in rows}):
        rr = [r for r in rows if r["family"] == fam]
        best = max(rr, key=lambda r: (sum(r["conditions_odd"]), r["d_summed"]))
        sel[fam] = dict(name=best["name"], params=best["params"], conditions_odd=best["conditions_odd"],
                        d_summed_odd=best["d_summed"], d_f1_odd=best["d_f1"], ribo_odd=best["ribo"])
    fn = B.BD / "results" / f"selected_{a.arm}.json"
    with open(fn, "w") as fh:
        json.dump(dict(arm=a.arm, written=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), selected=sel), fh, indent=1)
    print(json.dumps(sel, indent=1))


# --------------------------------------------------------------------------- test
def boot_mean(values, groups, rng, draws=2000):
    codes, inv = np.unique(groups, return_inverse=True)
    s = np.bincount(inv, weights=values, minlength=len(codes))
    n = np.bincount(inv, minlength=len(codes)).astype(float)
    pick = rng.integers(0, len(codes), size=(draws, len(codes)))
    m = s[pick].sum(1) / n[pick].sum(1)
    return [float(values.mean()), float(np.quantile(m, .025)), float(np.quantile(m, .975))]


def paired(A, Bf, gcode_of, split):
    """A, Bf: per-instance frames (instance_id index). Paired differences A - B per caller and summed DF+TXSS excess."""
    rng = np.random.default_rng(17)
    ids = A.index.intersection(Bf.index)
    x, y = A.loc[ids], Bf.loc[ids]
    g = x.genome.map(gcode_of).to_numpy()
    m = {"all": np.ones(len(ids), bool), "odd": g % 2 == 1, "even": g % 2 == 0}[split]
    out = {}
    tot = {}
    for c in CALLERS:
        k = m & (x.caller.to_numpy() == c)
        df1 = (x.f1.to_numpy() - y.f1.to_numpy())[k]
        dex = (x.excess.to_numpy() - y.excess.to_numpy())[k]
        dct = (x.control_mean_f1.to_numpy() - y.control_mean_f1.to_numpy())[k]
        out[c] = dict(n=int(k.sum()), f1_A=float(x.f1.to_numpy()[k].mean()), f1_B=float(y.f1.to_numpy()[k].mean()),
                      excess_A=float(x.excess.to_numpy()[k].mean()), excess_B=float(y.excess.to_numpy()[k].mean()),
                      d_f1=boot_mean(df1, g[k], rng), d_excess=boot_mean(dex, g[k], rng),
                      d_control=boot_mean(dct, g[k], rng))
        tot[c] = (dex, g[k])
    d1, g1 = tot["DefenseFinder"]
    d2, g2 = tot["TXSScan"]
    allg = np.unique(np.r_[g1, g2])
    idx = {c: i for i, c in enumerate(allg)}
    i1 = np.array([idx[c] for c in g1], dtype=np.int64)
    i2 = np.array([idx[c] for c in g2], dtype=np.int64)
    s1 = np.bincount(i1, weights=d1, minlength=len(allg))
    n1 = np.bincount(i1, minlength=len(allg))
    s2 = np.bincount(i2, weights=d2, minlength=len(allg))
    n2 = np.bincount(i2, minlength=len(allg))
    pick = rng.integers(0, len(allg), size=(2000, len(allg)))
    bs = s1[pick].sum(1) / np.maximum(n1[pick].sum(1), 1) + s2[pick].sum(1) / np.maximum(n2[pick].sum(1), 1)
    out["summed"] = [float(d1.mean() + d2.mean()), float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]
    out["summed_A"] = out["DefenseFinder"]["excess_A"] + out["TXSScan"]["excess_A"]
    out["summed_B"] = out["DefenseFinder"]["excess_B"] + out["TXSScan"]["excess_B"]
    return out


def ribo_paired(scA, scB):
    rng = np.random.default_rng(17)
    d = scA.resolved.to_numpy().astype(float) - scB.resolved.to_numpy().astype(float)
    pick = rng.integers(0, len(d), size=(2000, len(d)))
    bs = d[pick].sum(1)
    return [float(d.sum()), float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]


def test_conditions(rib, pr):
    parv = int(str(rib["parvarchaeaceae"]).split("/")[0])
    c1 = rib["resolved"] >= 1100 and rib["recovered"] >= 1650 and parv >= 110
    c2 = pr["summed"][0] >= 0.02 and pr["summed"][1] > 0
    c3 = all(pr[c]["d_f1"][0] >= -0.01 for c in CALLERS)
    return [bool(c1), bool(c2), bool(c3)]


def cmd_test(a):
    t0 = time.time()
    L = B.Landscape(B.ARMS[a.arm])
    bld = Builder(L)
    cfg = json.load(open(a.configs))["selected"]
    if a.only:
        cfg = {k: v for k, v in cfg.items() if k in a.only.split(",")}
    gcode = B.odd_even_code()
    runs = B.ribo_runs()
    even_runs = (runs.genome_code % 2 == 0).to_numpy()
    (B.BD / "loci" / a.arm).mkdir(parents=True, exist_ok=True)
    (B.BD / "scores" / a.arm).mkdir(parents=True, exist_ok=True)
    res = dict(arm=a.arm, rdir=str(L.rdir), configs={}, references={})

    def run_one(label, s, e, df=None):
        if df is None:
            df = L.table(s, e)
        df.to_parquet(B.BD / "loci" / a.arm / f"{label}.parquet", index=False)
        per, _ = B.score_table(B.PKG, df, a.threads)
        per.to_parquet(B.BD / "scores" / a.arm / f"{label}.parquet", index=False)
        per = per.set_index("instance_id")
        g = per.genome.map(gcode).to_numpy()
        summ = {sp: B.summarize_per(per[(g % 2 == (1 if sp == "odd" else 0))].reset_index()) for sp in ("odd", "even")}
        if "gid_s" in df:
            sc = B.ribo_score(runs, df.gid_s.to_numpy(), df.gid_e.to_numpy())
            rib = dict(all=B.ribo_summary(runs, sc), even=B.ribo_summary(runs[even_runs].reset_index(drop=True),
                                                                         sc[even_runs].reset_index(drop=True)))
            loc = B.locus_stats(df.gid_s.to_numpy(), df.gid_e.to_numpy(), L.u.n)
        else:
            sc, rib, loc = None, None, dict(loci=int(len(df)))
        return per, summ, sc, rib, loc

    r1 = run_one("R1_conserved2", L.s2, L.e2)
    res["references"]["R1"] = dict(summary=r1[1], ribo=r1[3], loci=r1[4])
    full = pd.read_parquet(L.rdir / "loci_all_genomes.parquet")
    contigs = pd.read_parquet(B.PKG / "universe/contigs.parquet", columns=["genome", "contig", "first_gid"])
    full = full.merge(contigs, on=["genome", "contig"], how="left")
    assert full.first_gid.notna().all()
    full["gid_s"] = full.first_gid.astype(np.int64) + full.start
    full["gid_e"] = full.first_gid.astype(np.int64) + full.end
    full = full.drop(columns="first_gid")
    r2 = run_one("R2_full_table", None, None, df=full)
    res["references"]["R2"] = dict(summary=r2[1], ribo=r2[3], loci=r2[4])
    B.log("references", f"{time.time()-t0:.0f}s")
    for fam, c in cfg.items():
        s, e = bld.build(fam, c["params"])
        per, summ, sc, rib, loc = run_one(c["name"], s, e)
        pr1 = paired(per, r1[0], gcode, "even")
        pr2 = paired(per, r2[0], gcode, "even")
        pr1_odd = paired(per, r1[0], gcode, "odd")
        item = dict(params=c["params"], summary=summ, ribo=rib, loci=loc, vs_R1_even=pr1, vs_R2_even=pr2,
                    vs_R1_odd=pr1_odd, ribo_vs_R1=ribo_paired(sc, r1[2]),
                    conditions_vs_R1=test_conditions(rib["all"], pr1), conditions_vs_R2=test_conditions(rib["all"], pr2))
        item["adopt_vs_R1"] = all(item["conditions_vs_R1"])
        res["configs"][fam] = item
        B.log(fam, c["name"], "cond R1", item["conditions_vs_R1"], "R2", item["conditions_vs_R2"],
              "dsum R1", [round(x, 4) for x in pr1["summed"]], "ribo", rib["all"]["resolved"], rib["all"]["recovered"],
              rib["all"]["parvarchaeaceae"], f"{time.time()-t0:.0f}s")
    fn = B.BD / "results" / f"test_{a.arm}{('_' + a.tag) if a.tag else ''}.json"
    with open(fn, "w") as fh:
        json.dump(res, fh, indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x))
    B.log("wrote", fn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["tune", "select", "test"])
    ap.add_argument("--arm", default="rescue3")
    ap.add_argument("--families", default="F,A,B,C,D,E")
    ap.add_argument("--configs")
    ap.add_argument("--only", default="")
    ap.add_argument("--tag", default="")
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    import numba
    numba.set_num_threads(a.threads)
    {"tune": cmd_tune, "select": cmd_select, "test": cmd_test}[a.cmd](a)


if __name__ == "__main__":
    main()
