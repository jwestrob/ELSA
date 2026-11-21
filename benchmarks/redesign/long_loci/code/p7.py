#!/usr/bin/env python
"""long_loci part 7 (predeclared7.json): S-add = nodes of the unmodified tree UNION the single-strand nodes of the
strand-first tree (part-6 S weights) that are not already nodes; S-add+B = the same over the B / SB trees (part-6 bridges).
Bonds, cohesion, bridges and node rules unchanged (part-6 code imported: p6.py).

usage (Slurm, 24 CPUs)
  p7.py mech                         mechanism check on part-6 S outputs (before any part-7 score)
  p7.py build --data dpann|sd --base P|Z [--thetas ...]
  p7.py score --base P|Z             DPANN scoring, odd selection, even report (part-6 metrics + loci, genes in loci)
  p7.py sd                           Spacedust S-add / S-add+B at the DPANN-selected theta, source207 (DEVELOPMENT)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import p6  # noqa: E402
import ll  # noqa: E402

RD, LL, P6D = p6.RD, p6.LL, p6.P6
P7 = LL / "p7"
log = ll.log
VARS = ("Sadd", "SaddB")
NM = {"P": "L4b", "Z": "L4bp"}
FROZEN_TAG = {"P": "dpann_WA", "Z": "dpann_Z"}


def jdump(obj, fn):
    p6.jdump(obj, fn)


# =========================================================================== mechanism check
def cmd_mech(a):
    import numba
    numba.set_num_threads(a.threads)
    import dscore as DS
    S1 = DS.S1
    t0 = time.time()
    P7.mkdir(parents=True, exist_ok=True)
    ctx = S1.Ctx()
    PKG = RD / "ws_dpann/package"
    strand = pd.read_parquet(PKG / "universe/genes.parquet", columns=["gid", "strand"]).sort_values("gid").strand.to_numpy()
    mem = pd.read_parquet(PKG / "gold/members.parquet")
    g = mem.groupby("instance_id").gid
    st = pd.DataFrame(dict(both=g.apply(lambda x: len(set(strand[x.to_numpy()])) > 1),
                           lo=g.min(), hi=g.max()))
    sc = np.r_[0, np.cumsum((strand[:-1] != strand[1:]).astype(np.int64))]
    st["change_inside"] = (sc[st.hi.to_numpy()] - sc[st.lo.to_numpy()]) > 0
    res = {}
    for base in ("P", "Z"):
        th = p6.FROZEN_TH[base]
        B0 = pd.read_parquet(LL / "scores" / FROZEN_TAG[base] / f"{NM[base]}_t{th:g}.parquet").set_index("instance_id")
        df = pd.read_parquet(LL / "loci" / f"p6_dpann_{base}S" / f"{NM[base]}_t{th:g}.parquet")
        per, summ, _, _, _ = S1.score(ctx, DS.table(ctx, df), a.threads)
        Sx = per.set_index("instance_id")
        j = B0[["f1", "caller", "genome"]].join(Sx[["f1"]], rsuffix="_S", how="inner").join(st, how="left")
        j["even"] = j.genome.map(ctx.gcode).to_numpy() % 2 == 0
        j["cls"] = np.where(j.f1_S < j.f1 - 1e-12, "worse", np.where(j.f1_S > j.f1 + 1e-12, "better", "same"))
        item = {}
        for half, sub in (("all", j), ("even", j[j.even])):
            item[half] = {c: dict(n=int((sub.cls == c).sum()), share_both_strands=float(sub.both[sub.cls == c].mean()) if (sub.cls == c).any() else None,
                                  share_change_inside=float(sub.change_inside[sub.cls == c].mean()) if (sub.cls == c).any() else None)
                          for c in ("worse", "same", "better")}
            item[half]["all_instances"] = dict(n=int(len(sub)), share_both_strands=float(sub.both.mean()),
                                               share_change_inside=float(sub.change_inside.mean()))
            item[half]["worse_by_caller"] = {k: dict(n=int(len(v)), share_both_strands=float(v.both.mean()), mean_dF1=float((v.f1_S - v.f1).mean()))
                                             for k, v in sub[sub.cls == "worse"].groupby("caller")}
        res[base] = item
        log(base, json.dumps(item["all"]))
    res["seconds"] = round(time.time() - t0, 1)
    jdump(res, P7 / "mech.json")


# =========================================================================== build
def single_strand(lo, hi, sc):
    """node [lo, hi] holds no strand change (sc = prefix sum of strand changes over boundaries)."""
    return (sc[hi] - sc[lo]) == 0


@njit(cache=True)
def crossings(add_a, add_b, ka, kb, maxlen):
    """added nodes that partially overlap some kept node (neither contains the other); ka sorted."""
    c = 0
    for x in range(len(add_a)):
        a, b = add_a[x], add_b[x]
        j = np.searchsorted(ka, a - maxlen)
        hit = False
        while j < len(ka) and ka[j] <= b:
            s, e = ka[j], kb[j]
            if e >= a and not ((s <= a and e >= b) or (s >= a and e <= b)):
                hit = True
                break
            j += 1
        c += int(hit)
    return c


def cmd_build(a):
    import numba
    numba.set_num_threads(a.threads)
    t0 = time.time()
    D = p6.Data(a.data)
    base = a.base
    s = D.s_of(base)
    kap = D.kappa(base)
    J = pd.read_parquet(P6D / f"bridges_{a.data}_{base}.parquet")
    br = J.k[J.bridge].to_numpy()
    sc = np.r_[0, np.cumsum(D.strand_change[:-1].astype(np.int64))]   # sc[i] = strand changes over boundaries < i
    thetas = [float(x) for x in a.thetas.split(",")] if a.thetas else list(p6.THETAS)
    spec = {"base": (None, None), "S": (D.strand_change, None), "B": (None, br), "SB": (D.strand_change, br)}
    trees = {(v, th): ll.tree(p6.weights(s, kap, th, *spec[v])) for v in spec for th in thetas}
    log("trees", len(trees), f"{time.time()-t0:.0f}s")
    keep = {}
    if base == "P":
        tok, gc, cc, grp = D.tok, D.gc, D.cc, D.grp
        T = int(tok.max()) + 1
        fsize = np.bincount(tok, minlength=T)
        mg = np.bincount(gc, minlength=int(gc.max()) + 1).astype(np.float64)
        alln = pd.concat([t[["lo", "hi"]] for t in trees.values()]).drop_duplicates().reset_index(drop=True)
        rec, _ = ll.recurrence(alln.lo.to_numpy(), alln.hi.to_numpy(), grp[gc[alln.lo.to_numpy()]], tok, fsize,
                               {-1: ll.fam_csr(tok, T)}, -1, cc, gc, grp, mg, a.blocks)
        rec = pd.concat([alln, rec], axis=1)
        rec["lam"] = ll.lam_of(rec.n.to_numpy(), rec.E_ana.to_numpy())
        node, key = json.load(open(p6.CFG[a.data]["tau_json"]["P"][0])), p6.CFG[a.data]["tau_json"]["P"][1]
        tau = float(node[key[0]][key[1]][key[2]])
        for kk, t in trees.items():
            t = t.merge(rec[["lo", "hi", "n", "nI", "lam"]], on=["lo", "hi"], how="left")
            km = (t.n.to_numpy() >= 2) & (t.lam.to_numpy() >= tau) & (t.nI.to_numpy() >= 2)
            keep[kk] = (t.lo.to_numpy()[km], t.hi.to_numpy()[km])
    else:
        for kk, t in trees.items():
            pm = t.hi.to_numpy() > t.lo.to_numpy()
            keep[kk] = (t.lo.to_numpy()[pm], t.hi.to_numpy()[pm])
    res = dict(data=a.data, base=base, thetas=thetas, outputs={}, gate={})
    for th in thetas:
        for v, (u, sx) in (("Sadd", ("base", "S")), ("SaddB", ("B", "SB"))):
            ua, ub = keep[(u, th)]
            sa, sb = keep[(sx, th)]
            m = single_strand(sa, sb, sc)
            known = set(zip(trees[(u, th)].lo.tolist(), trees[(u, th)].hi.tolist()))
            add = np.array([(x, y) for x, y in zip(sa[m].tolist(), sb[m].tolist()) if (x, y) not in known], np.int64).reshape(-1, 2)
            od = LL / "loci" / f"p7_{a.data}_{base}{v}"
            od.mkdir(parents=True, exist_ok=True)
            df = ll.save_loci(od, f"{NM[base]}_t{th:g}", np.r_[ua, add[:, 0]], np.r_[ub, add[:, 1]])
            o = np.lexsort((ub, ua))
            ka, kb = ua[o].astype(np.int64), ub[o].astype(np.int64)
            keep_add = add[add[:, 1] > add[:, 0]]
            ncross = crossings(keep_add[:, 0].copy(), keep_add[:, 1].copy(), ka, kb, int((kb - ka + 1).max()))
            info = ll.loci_info(df, D.n)
            info.update(unmodified_nodes=int(len(ua)), added_nodes=int(len(keep_add)), added_crossing=int(ncross))
            res["outputs"][f"{v}_t{th:g}"] = info
            log(base, v, th, info)
        if a.data == "dpann" or True:
            sp = Path(str(p6.CFG[a.data]["stored"][base]).format(th=f"{th:g}"))
            if sp.exists():
                ua, ub = keep[("base", th)]
                st_ = pd.read_parquet(sp)
                chk = pd.DataFrame(dict(s=ua, e=ub)).drop_duplicates()
                chk = chk[chk.e > chk.s].sort_values(["s", "e"]).reset_index(drop=True)
                res["gate"][f"t{th:g}"] = bool(st_[["s", "e"]].reset_index(drop=True).equals(chk))
    log("gate", res["gate"])
    assert all(res["gate"].values()) and len(res["gate"]) >= 1
    res["seconds"] = round(time.time() - t0, 1)
    res["peak_rss_gb"] = p6.rss_gb()
    jdump(res, P7 / f"build_{a.data}_{base}.json")


# =========================================================================== score (DPANN)
def cmd_score(a):
    import numba
    numba.set_num_threads(a.threads)
    import dscore as DS
    S1, RU, B = DS.S1, DS.RU, DS.B
    t0 = time.time()
    base = a.base
    ctx = S1.Ctx()
    ref = DS.ref_odd()
    dh_per = pd.read_parquet(S1.DH_SCORES).set_index("instance_id")
    nm, th0 = NM[base], p6.FROZEN_TH[base]
    frozen_scores = pd.read_parquet(LL / "scores" / FROZEN_TAG[base] / f"{nm}_t{th0:g}.parquet").set_index("instance_id")
    fz_s, fz_e = p6.read_loci(LL / "loci" / FROZEN_TAG[base] / f"{nm}_t{th0:g}.parquet")
    fz_ribo = B.ribo_score(ctx.runs, fz_s, fz_e)
    sys.path.insert(0, str(RD / "ribo_audit/code"))
    import bio
    import audit as AU
    targets = {}
    for rd, dd in (("strict", (150, 500)), ("loose", (500, 500))):
        bio.D_OP, bio.D_MEM = dd
        Bb = bio.Bio()
        targets[rd] = [Bb.run_units(i)[6] for i in range(len(Bb.D.runs))]
        del Bb

    def bio_count(path):
        L = AU.Loci(str(path))
        return {rd: int(sum(1 for t in tg if t and all(bio.score_target(L, ts, te)["ok"] for ts, te in t))) for rd, tg in targets.items()}

    PKG = RD / "ws_dpann/package"
    memb = pd.read_parquet(PKG / "gold/members.parquet")
    k_ = memb.groupby("instance_id").size()
    memb = memb[memb.instance_id.isin(k_[k_ >= 2].index)]
    sp = memb.groupby("instance_id").gid.agg(["min", "max"]).reset_index().sort_values("min").reset_index(drop=True)
    sp = sp[ctx.genes.contig.to_numpy()[sp["min"].to_numpy()] == ctx.genes.contig.to_numpy()[sp["max"].to_numpy()]]
    sp = sp[ctx.genome_arr[sp["min"].to_numpy()] == ctx.genome_arr[sp["max"].to_numpy()]].reset_index(drop=True)
    ist, ien = sp["min"].to_numpy(np.int64), sp["max"].to_numpy(np.int64)
    M = len(ist) + 1
    base_pairs = np.unique(p6.co_pairs(fz_s, fz_e, ist, ien, M))
    res = dict(base=base, frozen=dict(name=f"{FROZEN_TAG[base]}/{nm}_t{th0:g}", bio=bio_count(LL / "loci" / FROZEN_TAG[base] / f"{nm}_t{th0:g}.parquet"),
                                      any_fit=p6.any_fit_count(ctx.runs, fz_s, fz_e), curated_pairs_in_one_locus=int(len(base_pairs)),
                                      loci=int(len(fz_s))), outputs={}, selection={}, even={})
    log("frozen", res["frozen"])
    per_all = {}
    for v in VARS:
        for th in p6.THETAS:
            df = pd.read_parquet(LL / "loci" / f"p7_dpann_{base}{v}" / f"{nm}_t{th:g}.parquet")
            per, summ, sc, rib, loc = S1.score(ctx, DS.table(ctx, df), a.threads)
            score, cond = DS.sel_score(summ["odd"], rib["odd"], ref)
            per_all[(v, th)] = (per, summ, sc, rib, loc, df)
            res["outputs"][f"{v}_t{th:g}"] = dict(score_odd=score, conditions_odd=cond)
            log(base, v, th, "ODD", score, cond, f"{time.time()-t0:.0f}s")
    for v in VARS:
        c = {th: res["outputs"][f"{v}_t{th:g}"]["score_odd"] for th in p6.THETAS}
        res["selection"][v] = max(c, key=lambda th: tuple(c[th]))
    res["selection_written"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    jdump(res, P7 / f"score_dpann_{base}.json")
    log("SELECTION", res["selection"])
    C = ("DefenseFinder", "TXSScan", "CRISPRCasTyper")
    fz_summ_even = None
    for v in VARS:
        for th in sorted({res["selection"][v], th0}):
            per, summ, sc, rib, loc, df = per_all[(v, th)]
            P = per.set_index("instance_id")
            s_, e_ = df.s.to_numpy(np.int64), df.e.to_numpy(np.int64)
            item = dict(selected=bool(th == res["selection"][v]), summary_even=summ["even"], ribo_all=rib["all"], loci=loc,
                        genes_in_loci_frac=loc["genes_covered"] / len(ctx.gc))
            item["vs_DH_even"] = RU.paired(P, dh_per, ctx.gcode, "even")
            vd = S1.verdict(item["vs_DH_even"], rib["all"], ref["ribo_all"])
            bars = dict(summed=vd["summed_ok"], f1=vd["f1_ok"], recovered=bool(rib["all"]["recovered"] >= 1650),
                        resolved=bool(rib["all"]["resolved"] > 1512), parv=bool(S1.parv(rib["all"]["parvarchaeaceae"]) >= 110))
            bars["all"] = bool(bars["summed"] and all(bars["f1"].values()) and bars["recovered"] and bars["resolved"] and bars["parv"])
            item["seven_bars"] = bars
            item["vs_frozen_even"] = RU.paired(P, frozen_scores, ctx.gcode, "even")
            item["ribo_resolved_vs_frozen"] = RU.ribo_paired(sc, fz_ribo)
            item["any_fit_resolved"] = p6.any_fit_count(ctx.runs, s_, e_)
            item["bio"] = bio_count(LL / "loci" / f"p7_dpann_{base}{v}" / f"{nm}_t{th:g}.parquet")
            pr = np.unique(p6.co_pairs(s_, e_, ist, ien, M))
            item["new_curated_pairs_vs_frozen"] = int(len(np.setdiff1d(pr, base_pairs)))
            vf = item["vs_frozen_even"]
            item["justified"] = dict(
                strict_plus50=bool(item["bio"]["strict"] >= res["frozen"]["bio"]["strict"] + 50), seven_bars=bars["all"],
                no_new_pairs=bool(item["new_curated_pairs_vs_frozen"] == 0), summed=bool(vf["summed"][0] >= -0.01),
                f1={c: bool(vf[c]["d_f1"][0] >= -0.01) for c in C})
            item["justified"]["all"] = bool(item["justified"]["strict_plus50"] and bars["all"] and item["justified"]["no_new_pairs"]
                                            and item["justified"]["summed"] and all(item["justified"]["f1"].values()))
            res["even"][f"{v}_t{th:g}"] = item
            log(base, v, th, "EVEN summed", round(summ["even"]["summed_DF_TXSS_excess"], 4), "ribo", rib["all"]["resolved"],
                "bio", item["bio"], "pairs", item["new_curated_pairs_vs_frozen"], "bars", bars["all"], "JUSTIFIED", item["justified"]["all"])
    res["seconds"] = round(time.time() - t0, 1)
    res["peak_rss_gb"] = p6.rss_gb()
    jdump(res, P7 / f"score_dpann_{base}.json")


# =========================================================================== Spacedust
def cmd_sd(a):
    import numba
    numba.set_num_threads(a.threads)
    import s207
    t0 = time.time()
    sel = {b: json.load(open(P7 / f"score_dpann_{b}.json"))["selection"] for b in ("P", "Z")}
    g = s207.load_universe_genes()
    Bn = s207.Bench(a.threads)
    refs = {"P": pd.read_parquet(LL / "scores/sd_WA/pr_L4b_t0.2.parquet"), "Z": pd.read_parquet(LL / "scores/sd_Z/pr_L4bp_t0.02.parquet")}
    res = dict(label="DEVELOPMENT", selection_from_dpann=sel, outputs={})
    for base in ("P", "Z"):
        ths = sorted({p6.FROZEN_TH[base]} | set(sel[base].values()))
        cmd_build(argparse.Namespace(data="sd", base=base, thetas=",".join(f"{t:g}" for t in ths), blocks=a.blocks, threads=a.threads))
        for v in VARS:
            th = sel[base][v]
            df = pd.read_parquet(LL / "loci" / f"p7_sd_{base}{v}" / f"{NM[base]}_t{th:g}.parquet")
            s, e = df.s.to_numpy(np.int64), df.e.to_numpy(np.int64)
            r, pr = Bn.score(s207.gene_spans(g, s, e), f"p7 sd {base}{v} {th}", np.random.default_rng(17))
            od = LL / "scores" / f"p7_sd_{base}{v}"
            od.mkdir(parents=True, exist_ok=True)
            pr.to_parquet(od / f"pr_{NM[base]}_t{th:g}.parquet", index=False)
            pd_ = s207.paired(pr, refs[base])
            res["outputs"][f"{base}{v}_t{th:g}"] = dict(score=r, vs_frozen=pd_, verdict_vs_frozen=s207.verdict(pd_))
            log("SD", base, v, th, {k: r[k] for k in ("f1_observed", "f1_random", "f1_excess", "ci_lo", "ci_hi")}, pd_["excess"])
    res["seconds"] = round(time.time() - t0, 1)
    jdump(res, P7 / "sd_source207.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["mech", "build", "score", "sd"])
    ap.add_argument("--data", default="dpann")
    ap.add_argument("--base", default="Z")
    ap.add_argument("--thetas", default="")
    ap.add_argument("--blocks", type=int, default=4096)
    ap.add_argument("--threads", type=int, default=1)
    a = ap.parse_args()
    {"mech": cmd_mech, "build": cmd_build, "score": cmd_score, "sd": cmd_sd}[a.cmd](a)


if __name__ == "__main__":
    main()
