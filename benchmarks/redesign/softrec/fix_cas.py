#!/usr/bin/env python
"""ws_cas Part 2: gated second tier (predeclared_cas.json, written 2026-11-13 01:36 UTC before any run).

Tier 1 = rw1000_e50 (W_e50 >= 1 and L >= 2, depth 1,000), unchanged. Tier 2 = instances with L >= 2 and Poisson
P(X >= G | E) <= 10^-e at depth 1,000, not tier 1, whose whole gene interval [left, left + d] contains no gene touched
by a tier-1 instance interval. Reading = round 4's post hoc configuration (instance-graph labels ic:k5o0.3_g1 on
tier 1, trimming f 0.8, DH); tier-2 instances are unlabelled (-1). The lineage unit is the default genome -> lineage
array already inside the stored statistics; no step reads MMseqs2 families, taxonomy or caller labels.

usage
  fix_cas.py tune --threads T            odd genomes: e in {none, 3, 4, 6, 8}; combined-rule FDR from the three nulls
                                         (gate recomputed in each null's order); round 4's odd score -> fix_odd.json
  fix_cas.py test --e E --threads T      all genomes (even = verdict) for e = E and the base (e = none) -> fix_all.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

RD = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111")
sys.path.insert(0, str(RD / "softrec/cas/code"))
import diag_cas as D  # noqa: E402

R4m, R2m, B, BR, SR = D.R4m, D.R2m, D.B, D.BR, D.SR
CAS, R2, R4 = D.CAS, D.R2, D.R4
LAND, W, LN10, CALLERS, CCT = D.LAND, D.W, D.LN10, D.CALLERS, D.CCT
F_TRIM = 0.8
GRID = (None, 3, 4, 6, 8)
log = D.log


def tiers(n, left, d, P, e):
    """Tier masks over instances of one order (slot space: left = left slot, d = distance)."""
    t1 = (P["Lt1000"] >= 2) & (P["We50"] >= 1)
    if e is None:
        return t1, np.zeros_like(t1)
    l1 = left[t1]
    r1 = left[t1] + d[t1]
    touch = np.bincount(l1, minlength=n + 1) - np.bincount(r1 + 1, minlength=n + 1)
    touched = np.cumsum(touch)[:n] > 0
    ct = np.r_[0, np.cumsum(touched)]
    free = (ct[left + d + 1] - ct[left]) == 0
    t2 = (P["Lt1000"] >= 2) & (P["logP1000"] <= -e * LN10) & ~t1 & free
    return t1, t2


def ename(e):
    return "base" if e is None else f"gate_p1e-{e}"


class Ctx2:
    def __init__(self, with_nulls):
        self.ctx = R2m.Ctx()
        su = SR.Universe()
        self.n = su.n
        nd, off, left, d = su.instances(W)
        self.left, self.d = left.astype(np.int64), d.astype(np.int64)
        self.P = D.load_pass(R2 / "stats/obs.npz")
        self.nulls = [D.load_pass(R2 / f"stats/null{r}.npz") for r in range(3)] if with_nulls else []
        cd = R2 / "cand" / LAND
        self.sidx = np.load(cd / "sidx.npy")
        self.val = np.load(cd / "value.npy").astype(np.float32)
        comm = np.load(R4 / "graph" / LAND / "ic_k5o0.3_g1.npy").astype(np.int64)
        sz = np.bincount(comm)
        self.cl = np.where(sz[comm] >= 2, comm, -1)
        t1, _ = tiers(self.n, self.left, self.d, self.P, None)
        assert int(t1.sum()) == len(self.sidx) and t1[self.sidx].all(), "tier 1 != stored rw1000_e50 candidate"

    def landscape(self, e):
        _, t2 = tiers(self.n, self.left, self.d, self.P, e)
        add = np.flatnonzero(t2)
        idx = np.r_[self.sidx, add].astype(np.int64)
        val = np.r_[self.val, self.P["Lt1000"][add].astype(np.float32)]
        labi = np.r_[self.cl, np.full(len(add), -1, np.int64)]
        b, arg = SR.boundary_support_argmax(self.left[idx], self.d[idx], val, np.arange(len(idx), dtype=np.int64), self.n)
        lab = np.where(arg >= 0, labi[np.maximum(arg, 0)], -2).astype(np.int64)
        return add, b, lab

    def fdr(self, e):
        t1, t2 = tiers(self.n, self.left, self.d, self.P, e)
        nn = [tiers(self.n, self.left, self.d, Q, e) for Q in self.nulls]
        n1 = float(np.mean([int(x[0].sum()) for x in nn]))
        n2 = float(np.mean([int(x[1].sum()) for x in nn]))
        o1, o2 = int(t1.sum()), int(t2.sum())
        return dict(tier1_observed=o1, tier2_observed=o2, tier1_null=n1, tier2_null=n2,
                    combined_fdr=(n1 + n2) / max(1, o1 + o2), tier2_fdr=(n2 / o2) if o2 else None)


def run(C, e, split, threads):
    ctx = C.ctx
    add, b, lab = C.landscape(e)
    gm = np.full(C.n, -1, np.int64)
    tabs, se = R4m.build_tables(ctx, b, lab, gm, F_TRIM, ["R1", "DH"])
    res, pers = R2m.evaluate(ctx, tabs, SimpleNamespace(name=ename(e)), split, threads)
    base_b = C.landscape(None)[1] if e is not None else b
    s2, e2 = SR.runs_at_least(b, ctx.cs, ctx.ce, 2.0, 0)
    s0, e0 = SR.runs_at_least(base_b, ctx.cs, ctx.ce, 2.0, 0)
    item = dict(tier2_instances=int(len(add)), R1_runs=int(len(s2)), R1_runs_base=int(len(s0)),
                island_runs=int(len(s2) - len(s0)),
                base_boundaries_unchanged=bool(np.array_equal(np.where(base_b > 0, b, 0), base_b)),
                genes_in_loci={rn: res[rn]["loci"]["genes_covered"] for rn in res},
                readings=res,
                member_support={sp: R2m.member_support(ctx, b, par) for sp, par in (("odd", 1),) + ((("even", 0),) if split != "odd" else ())})
    return item, pers, tabs


def cmd_tune(a):
    t0 = time.time()
    fn = CAS / "results/fix_odd.json"
    out = D.jload(fn)
    C = Ctx2(with_nulls=True)
    refs = R4m.Refs(C.ctx, "odd")
    out["_refs"] = dict(tok_DH_odd=refs.summary, tok_DH_ribo_odd=refs.ribo, bars=refs.bars)
    rows = {}
    for e in GRID:
        nm = ename(e)
        fd = C.fdr(e)
        log("fdr", nm, fd)
        item, pers, _ = run(C, e, "odd", a.threads)
        item["fdr"] = fd
        summ = item["readings"]["DH"]["summary"]["odd"]
        ribo = item["readings"]["DH"]["ribo"]["scored"]
        item["score"] = refs.margins(summ, ribo)
        item["eligible"] = bool(fd["combined_fdr"] <= 0.01)
        rows[nm] = item
        out[nm] = item
        D.jdump(fn, out)
        log("odd", nm, "shortfall", round(item["score"]["shortfall"], 3), "summed", round(summ["summed_DF_TXSS_excess"], 4),
            {c[:4]: round(summ[c]["f1"], 4) for c in CALLERS}, "ribo", ribo["recovered"], ribo["resolved"], ribo["parvarchaeaceae"],
            "tier2", item["tier2_instances"], "islands", item["island_runs"], "fdr", round(fd["combined_fdr"], 5),
            "base unchanged", item["base_boundaries_unchanged"])
    # base reproduction on odd: identical DH per-instance F1 to the stored post hoc odd scores
    cand = [(nm, it) for nm, it in rows.items() if nm != "base" and it["eligible"]]
    best = max(cand, key=lambda x: (-x[1]["score"]["shortfall"], x[1]["score"]["score"][1], int(x[0].split("-")[-1]))) if cand else None
    out["selection"] = dict(rule="max (-shortfall, odd summed excess); ties to larger e", selected=best[0] if best else None,
                            table={nm: dict(shortfall=it["score"]["shortfall"], summed=it["score"]["score"][1], eligible=it["eligible"],
                                            fdr=it["fdr"]["combined_fdr"], margins=it["score"]["margins"]) for nm, it in rows.items()})
    out["seconds"] = round(time.time() - t0, 1)
    D.jdump(fn, out)
    log("selection", json.dumps(out["selection"]))


def cmd_test(a):
    t0 = time.time()
    fn = CAS / "results/fix_all.json"
    out = D.jload(fn)
    C = Ctx2(with_nulls=False)
    ctx = C.ctx
    refs = R4m.Refs(ctx, "all")
    out["_refs"] = dict(tok_DH_even=refs.summary, tok_DH_ribo_all=refs.ribo)
    base_per = None
    for e in (None, a.e):
        nm = ename(e)
        item, pers, tabs = run(C, e, "all", a.threads)
        per = pers["DH"][0]
        pi = per.set_index("instance_id")
        if e is None:
            stored = pd.read_parquet(CAS / "scores/post__DH.parquet")
            item["check_vs_post"] = D.cmp_per(per, stored)
            base_per = pi
        per.to_parquet(CAS / f"scores/fix__{nm}__DH.parquet", index=False)
        tabs["DH"].to_parquet(CAS / f"loci/fix__{nm}__DH.parquet", index=False)
        item["paired"] = {}
        for sp in ("even", "odd"):
            item["paired"][f"{sp}:DH_vs_tok_DH"] = BR.paired(pi, refs.per["tok_DH"], ctx.gcode, sp)
            if e is not None:
                item["paired"][f"{sp}:DH_vs_base"] = BR.paired(pi, base_per, ctx.gcode, sp)
        pr = item["paired"]["even:DH_vs_tok_DH"]
        ribo = item["readings"]["DH"]["ribo"]["scored"]
        v = R4m.verdict(pr, ribo)
        v["beats_coordinator"] = bool(v["matches"] and pr["summed"][1] > 0)
        v["beats_round4"] = v.pop("beats")
        item["verdict_vs_tok_DH"] = v
        out[nm] = item
        D.jdump(fn, out)
        s = item["readings"]["DH"]["summary"]["even"]
        log("test", nm, "even", D.brief(s), "ribo", ribo["recovered"], ribo["resolved"], ribo["parvarchaeaceae"],
            ribo.get("parvarchaeaceae_resolved"), "verdict", json.dumps(v))
    out["seconds"] = round(time.time() - t0, 1)
    D.jdump(fn, out)
    log("test done", out["seconds"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("tune", "test"))
    ap.add_argument("--threads", type=int, required=True)
    ap.add_argument("--e", type=int, default=None)
    a = ap.parse_args()
    if a.cmd == "test":
        assert a.e is not None, "--e is required for test"
    {"tune": cmd_tune, "test": cmd_test}[a.cmd](a)


if __name__ == "__main__":
    main()
