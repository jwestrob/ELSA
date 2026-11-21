#!/usr/bin/env python
"""long_loci part 8 (predeclared8.json): best-window bond test on both bases.

P-BW  family base: every token pair within 8 positions is tested at every window w from its own distance up to 8
      (lw.bonds_adaptive's count and expectation at window w) and keeps its largest Lambda; tau by the bond-FDR rule on
      the whole procedure; the rest of the frozen primary unchanged (lw.cmd_build with this bond function, arm 'WB').
Z-BW  list base: the frozen secondary's adjacencies (<= 2 intervening genes, stored K1000 masks) with list-partner
      support counted at every window 3..8 (one kernel pass; vl group counting and expectation per window), largest
      Lambda; tau by vl's rule on the whole procedure; kNN cohesion, L4b+ (every node kept).
usage (Slurm, 24 CPUs)
  p8.py pbw --data dpann|sd        gate (own-window mode == lw.bonds_adaptive), then lw.cmd_build arm WB (suffix _p8)
  p8.py zbw --data dpann|sd        gates (window 3 == vl.pass_count; window 8 == zwa.pass_count_w), ladder, bonds, L4b+
  p8.py score --base P|Z           DPANN scoring as part 7 + verdict-1 repairs, case 8233, window distribution
  p8.py sd                         Spacedust source207 rows at the DPANN-selected theta (DEVELOPMENT)
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit, prange

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ll  # noqa: E402
import lw  # noqa: E402
import words_ts as W  # noqa: E402
import p6  # noqa: E402
import zwa  # noqa: E402
from zwa import mark_min  # noqa: E402  (vl.mark copy with a window)

RD, LL = p6.RD, p6.LL
P8 = LL / "p8"
log = ll.log
WMAX = 8
ZW = (3, 4, 5, 6, 7, 8)
ZLADDER = (4.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0, 25.0, 30.0)
_WIN = {}


def jdump(obj, fn):
    p6.jdump(obj, fn)


# =========================================================================== P-BW
def bonds_bestwindow(tok, cc, gc, grp, tau, wmax=WMAX, own_only=False):
    st = lw.setup(tok, gc, grp)
    sym, ng = st["sym"], st["ng"]
    n = len(tok)
    k, d, key = W.occurrences(np.arange(n, dtype=np.int64), cc, sym, np.ones(n, dtype=np.bool_), wmax)
    ukey, kinv = np.unique(key, return_inverse=True)
    g_occ = grp[gc[k]]
    occ_lam = np.zeros(len(k))
    occ_win = np.zeros(len(k), np.int64)
    for w in range(1, wmax + 1):
        m = d <= w
        kg = np.unique(kinv[m] * ng + g_occ[m])
        nG = np.bincount(kg // ng, minlength=len(ukey))
        if own_only:
            has = np.zeros(len(ukey), bool)
            has[kinv[d == w]] = True
            cand = np.flatnonzero((nG >= 2) & has)
        else:
            cand = np.flatnonzero(nG >= 2)
        lo, hi = ukey[cand] >> ll.SHIFT, ukey[cand] & ll.MASK
        E = W.expected_groups(lo, hi, st["indptr"], st["r_of"], st["cnt"], st["m_rank"], st["grp_of_rank"], float(w))
        nc = nG[cand].astype(np.float64)
        lam = np.where(nc > E, nc * np.log(nc / np.maximum(E, 1e-300)) - nc + E, 0.0)
        key_lam = np.zeros(len(ukey))
        ok = lam >= tau
        key_lam[cand[ok]] = lam[ok]
        at = (d == w) if own_only else m
        cl = key_lam[kinv[at]]
        idx = np.flatnonzero(at)
        upd = cl > occ_lam[idx]
        occ_lam[idx[upd]] = cl[upd]
        occ_win[idx[upd]] = w
    si = occ_lam > 0
    return k[si], d[si], occ_lam[si], occ_win[si]


def cmd_pbw(a):
    import numba
    numba.set_num_threads(a.threads)
    t0 = time.time()
    P8.mkdir(parents=True, exist_ok=True)
    res = {}
    if a.data == "dpann":
        tok, cc, gc, grp, kappa = ll.load("dpann")
        k0, d0, l0 = lw.bonds_adaptive(tok, cc, gc, grp, 4.0)
        k1, d1, l1, w1 = bonds_bestwindow(tok, cc, gc, grp, 4.0, own_only=True)
        o0, o1 = np.lexsort((d0, k0)), np.lexsort((d1, k1))
        res["gate_own_window_equals_WA"] = bool(np.array_equal(k0[o0], k1[o1]) and np.array_equal(d0[o0], d1[o1])
                                                and np.allclose(l0[o0], l1[o1], rtol=0, atol=1e-9))
        log("GATE own-window == lw.bonds_adaptive", res["gate_own_window_equals_WA"], len(k0), len(k1))
        assert res["gate_own_window_equals_WA"]
        del k0, d0, l0, k1, d1, l1, w1
    orig = lw.arm_bonds

    def arm_bonds(arm, tok_, cc_, gc_, grp_, tau):
        if arm != "WB":
            return orig(arm, tok_, cc_, gc_, grp_, tau)
        k, d, lam, win = bonds_bestwindow(tok_, cc_, gc_, grp_, tau)
        if "obs" not in _WIN:
            _WIN["obs"] = (k, d, lam, win)
        return k, d, lam

    lw.arm_bonds = arm_bonds
    lw.cmd_build(argparse.Namespace(data=a.data, arms="WB", w3b=False, suffix="_p8", blocks=a.blocks, threads=a.threads))
    wb = json.load(open(LL / "results" / f"wbuild_{a.data}_p8.json"))["arms"]["WB"]
    tau = wb["tau_star"]
    k, d, lam, win = _WIN["obs"]
    keep = lam >= tau
    res["tau"] = tau
    res["windows"] = dict(bonds=int(keep.sum()), chosen_minus_own={str(x): int(c) for x, c in zip(*np.unique(win[keep] - d[keep], return_counts=True))},
                          chosen={str(x): int(c) for x, c in zip(*np.unique(win[keep], return_counts=True))},
                          share_chosen_gt_own=float((win[keep] > d[keep]).mean()))
    np.savez(LL / "nodes" / f"{a.data}_WB" / "bonds_windows.npz", k=k[keep], d=d[keep], lam=lam[keep], win=win[keep])
    res["seconds"] = round(time.time() - t0, 1)
    jdump(res, P8 / f"pbw_{a.data}.json")


# =========================================================================== Z-BW kernel
@njit(parallel=True, cache=True)
def pass_count_bw(m, aptr, ak, gene_at, lstart, tcnt, nbs, nbg, cid, grp, ngenome, ngroup, mg, W0, W1, out_ng, out_Eg, nchunks):
    """zwa.pass_count_w (vl.pass_count at round 0) evaluated for every window W0..W1 in one pass: support genomes keep
    the smallest partner distance; groups / expectation per window."""
    n = len(cid)
    NW = W1 - W0 + 1
    for ch in prange(nchunks):
        lo = ch * m // nchunks
        hi = (ch + 1) * m // nchunks
        Rst = np.full(n, -1, np.int64)
        Lst = np.full(n, -1, np.int64)
        Rd = np.zeros(n, np.int64)
        Ld = np.zeros(n, np.int64)
        ca = np.zeros(ngenome, np.int64)
        cast = np.full(ngenome, -1, np.int64)
        cb = np.zeros(ngenome, np.int64)
        cbst = np.full(ngenome, -1, np.int64)
        tb = np.zeros(ngenome, np.int64)
        gst = np.full(ngenome, -1, np.int64)
        gmd = np.zeros(ngenome, np.int64)
        grst = np.full(ngroup, -1, np.int64)
        grmin = np.zeros(ngroup, np.int64)
        grst2 = np.full(ngroup, -1, np.int64)
        grlp = np.zeros((ngroup, NW), np.float64)
        grt = np.zeros(ngroup, np.int64)
        grt2 = np.zeros(ngroup, np.int64)
        for j in range(lo, hi):
            if aptr[j + 1] == aptr[j]:
                continue
            g = gene_at[j]
            mark_min(j, lstart[g], np.int64(tcnt[g]), nbs, nbg, cid, n, Rst, Rd, Lst, Ld, ca, cast, W1)
            for r in range(aptr[j], aptr[j + 1]):
                k = ak[r]
                gk = gene_at[k]
                by = lstart[gk]
                cy = np.int64(tcnt[gk])
                ntb = 0
                nsg = 0
                for t in range(cy):
                    s = np.int64(nbs[by + t])
                    G = np.int64(nbg[by + t])
                    if cbst[G] != r:
                        cbst[G] = r
                        cb[G] = 0
                        tb[ntb] = G
                        ntb += 1
                    cb[G] += 1
                    dd = 1 << 30
                    if Rst[s] == j and Rd[s] < dd:
                        dd = Rd[s]
                    if Lst[s] == j and Ld[s] < dd:
                        dd = Ld[s]
                    if dd <= W1:
                        if gst[G] != r:
                            gst[G] = r
                            gmd[G] = dd
                        elif dd < gmd[G]:
                            gmd[G] = dd
                        gg = grp[G]
                        if grst2[gg] != r:
                            grst2[gg] = r
                            grmin[gg] = dd
                            grt2[nsg] = gg
                            nsg += 1
                        elif dd < grmin[gg]:
                            grmin[gg] = dd
                for wi in range(NW):
                    w = W0 + wi
                    c = 0
                    for q in range(nsg):
                        if grmin[grt2[q]] <= w:
                            c += 1
                    out_ng[r, wi] = c
                ngt = 0
                for q in range(ntb):
                    G = tb[q]
                    if cast[G] == j and ca[G] > 0:
                        gg = grp[G]
                        if grst[gg] != r:
                            grst[gg] = r
                            for wi in range(NW):
                                grlp[gg, wi] = 0.0
                            grt[ngt] = gg
                            ngt += 1
                        for wi in range(NW):
                            w = W0 + wi
                            x = 2.0 * w * ca[G] * cb[G] / mg[G]
                            if x >= 1.0:
                                grlp[gg, wi] = -1e300
                            else:
                                grlp[gg, wi] += math.log1p(-x)
                for wi in range(NW):
                    Eg = 0.0
                    for q in range(ngt):
                        Eg += 1.0 - math.exp(grlp[grt[q], wi])
                    out_Eg[r, wi] = Eg


def z_universe(data):
    import vl
    if data == "dpann":
        import lb
        U = vl.Univ(vl.GROUPS)
        LS, K = lb.make_lists(U, "z10.6")
        excl_dir = RD / "vocab_lists/excl"
        kappa = RD / "context_units/ctx/kappa_var_knn50.npy"
    else:
        import sdu
        U = sdu.SDUniv("group")
        LS, K = sdu.z_lists_sd(U, 10.6)
        excl_dir = LL / "sd_lists/excl"
        kappa = LL / "sd_lists/kappa_var_knn50.npy"
    return U, LS, K, excl_dir, kappa


def zbw_order(U, LS, K, r, excl, off3, threads):
    import softrec as SR
    import vl
    u = U.u
    n = u.n
    gene_at = np.arange(n, dtype=np.int64) if r < 0 else SR.null_order(u, r)[0]
    slot_of = np.empty_like(gene_at)
    slot_of[gene_at] = np.arange(n)
    nbs = np.empty_like(LS.nb)
    SR.map_slots(LS.nb, slot_of.astype(np.int32), nbs)
    tcnt = LS.tcnt[K]
    nlen = tcnt[gene_at]
    cid = u.cid.astype(np.int64)
    aptr, ak = zwa.adjacency_w(n, cid, nlen, excl, off3, 3)
    R = len(ak)
    ong = np.zeros((R, len(ZW)), np.int64)
    oEg = np.zeros((R, len(ZW)))
    t0 = time.time()
    pass_count_bw(n, aptr, ak, gene_at, LS.lstart, tcnt, nbs, LS.nbg, cid, U.grp, u.ng, U.ngroup, U.mg, ZW[0], ZW[-1],
                  ong, oEg, threads * 4)
    sec = time.time() - t0
    rj = np.repeat(np.arange(n, dtype=np.int64), np.diff(aptr))
    lam = np.zeros((R, len(ZW)))
    nu_all = np.zeros((R, len(ZW)))
    for wi, w in enumerate(ZW):
        own = np.minimum(1.0, 2.0 * w / U.mg[u.gc.astype(np.int64)[rj]])
        nu = ong[:, wi] + 1.0
        lam[:, wi] = np.where(nu >= 2, vl.lam_np(nu, oEg[:, wi] + own), 0.0)
        nu_all[:, wi] = nu
    best = lam.max(1)
    win = np.array(ZW)[lam.argmax(1)]
    return dict(rj=rj, ak=ak, d=ak - rj, ong=ong, oEg=oEg, lam=lam, best=best, win=win, seconds=round(sec, 1))


def gate_fixed(U, LS, K, excl, off3, Wf, threads):
    """zwa.pass_count_w at a fixed window Wf (marks up to Wf) over the frozen adjacencies (<= 3 apart, stored masks)."""
    u = U.u
    n = u.n
    gene_at = np.arange(n, dtype=np.int64)
    tcnt = LS.tcnt[K]
    cid = u.cid.astype(np.int64)
    aptr, ak = zwa.adjacency_w(n, cid, tcnt[gene_at], excl, off3, 3)
    R = len(ak)
    on, ong = np.zeros(R, np.int64), np.zeros(R, np.int64)
    oE, oEg = np.zeros(R), np.zeros(R)
    zwa.pass_count_w(n, aptr, ak, gene_at, LS.lstart, tcnt, LS.nb, LS.nbg, cid, U.grp, u.ng, U.ngroup, U.mg, Wf, Wf,
                     on, ong, oE, oEg, threads * 4)
    return dict(ak=ak, ong=ong, oEg=oEg)


def cmd_zbw(a):
    import numba
    numba.set_num_threads(a.threads)
    t0 = time.time()
    P8.mkdir(parents=True, exist_ok=True)
    U, LS, K, excl_dir, kappa_path = z_universe(a.data)
    u = U.u
    n = u.n
    nd3, off3, left3, d3 = u.instances(3)
    res = dict(data=a.data, lists=LS.info, timing={"lists": round(time.time() - t0, 1)})
    excl = {r: np.load(excl_dir / f"group_order{r}.npz")["K1000"] for r in (-1, 0, 1, 2)}
    lad, obs = {}, None
    for r in (-1, 0, 1, 2):
        x = zbw_order(U, LS, K, r, excl[r], off3, a.threads)
        if r < 0:
            # gates: window 3 == vl.pass_count (via zwa.run_order fixed window 3); window 8 == zwa.pass_count_w(fixedW 8)
            g3 = zwa.run_order(U, LS, K, -1, excl[-1], off3, 3, 3, a.threads)
            g8 = gate_fixed(U, LS, K, excl[-1], off3, 8, a.threads)
            res["gate"] = dict(rows=bool(np.array_equal(g3["ak"], x["ak"])),
                               ng3=bool(np.array_equal(g3["ong"], x["ong"][:, 0])), Eg3=bool(np.allclose(g3["oEg"], x["oEg"][:, 0], rtol=1e-12, atol=1e-12)),
                               lam3=bool(np.allclose(np.where(g3["nu"] >= 2, g3["lam"], 0.0), x["lam"][:, 0], rtol=1e-12, atol=1e-12)),
                               ng8=bool(np.array_equal(g8["ong"], x["ong"][:, -1])), Eg8=bool(np.allclose(g8["oEg"], x["oEg"][:, -1], rtol=1e-12, atol=1e-12)))
            log("GATE", res["gate"])
            assert all(res["gate"].values()), res["gate"]
            del g3, g8
        lad[str(r)] = dict(counts={str(t): int((x["best"] >= t).sum()) for t in ZLADDER}, adjacencies=int(len(x["ak"])),
                           seconds=x["seconds"])
        log("LADDER Z-BW", a.data, r, lad[str(r)], "rss", p6.rss_gb())
        if r < 0:
            obs = dict(k=x["rj"], d=x["d"], lam=x["best"], win=x["win"], lam3=x["lam"][:, 0])
        del x
    rows = []
    for t in ZLADDER:
        o = lad["-1"]["counts"][str(t)]
        nl = float(np.mean([lad[str(r)]["counts"][str(t)] for r in (0, 1, 2)]))
        rows.append(dict(tau=t, observed=o, null_mean=nl, fdr=nl / o if o else None))
    okt = [x["tau"] for x in rows if x["fdr"] is not None and x["fdr"] <= 0.01]
    tau = min(okt) if okt else None
    res.update(ladder=lad, tau_table=rows, tau_star=tau)
    log("TAU Z-BW", a.data, tau)
    keep = obs["lam"] >= tau
    k, d, lam, win = obs["k"][keep], obs["d"][keep], obs["lam"][keep], obs["win"][keep]
    res["windows"] = dict(bonds=int(len(k)), chosen={str(x): int(c) for x, c in zip(*np.unique(win, return_counts=True))},
                          bonds_where_window3_alone_fails=int((obs["lam3"][keep] < tau).sum()))
    s = ll.boundary_score(k, d, lam, n)
    tag = f"{a.data}_ZBW"
    od, nd = LL / "loci" / tag, LL / "nodes" / tag
    od.mkdir(parents=True, exist_ok=True)
    nd.mkdir(parents=True, exist_ok=True)
    np.savez(nd / "bonds_obs.npz", k=k, d=d, lam=lam, win=win)
    res["bonds"] = dict(tau=tau, bonds=int(len(k)), boundaries_bonded=int((s > 0).sum()))
    st_, en_ = ll.runs(s > 0)
    res["outputs"] = {"L1": ll.loci_info(ll.save_loci(od, "L1", st_, en_), n)}
    kv = np.load(kappa_path).astype(np.float64)
    for th in ll.THETAS:
        low = (kv < th) & (s > 0)
        w_ = np.where(low, 1e-6 + np.maximum(np.nan_to_num(kv, nan=0.0), 0.0), s)
        t = ll.tree(w_)
        pm = t.hi.to_numpy() > t.lo.to_numpy()
        res["outputs"][f"L4bp_t{th:g}"] = ll.loci_info(ll.save_loci(od, f"L4bp_t{th:g}", t.lo.to_numpy()[pm], t.hi.to_numpy()[pm]), n)
    res["timing"]["total"] = round(time.time() - t0, 1)
    res["peak_rss_gb"] = p6.rss_gb()
    jdump(res, P8 / f"zbw_{a.data}.json")


# =========================================================================== score (DPANN)
def cmd_score(a):
    import numba
    numba.set_num_threads(a.threads)
    import dscore as DS
    S1, RU, B = DS.S1, DS.RU, DS.B
    t0 = time.time()
    base = a.base
    nm, th0 = ("L4b", 0.2) if base == "P" else ("L4bp", 0.02)
    frozen_tag, var_tag, aname = ("dpann_WA", "dpann_WB", "WA_L4b") if base == "P" else ("dpann_Z", "dpann_ZBW", "Z_L4bp")
    ctx = S1.Ctx()
    ref = DS.ref_odd()
    dh_per = pd.read_parquet(S1.DH_SCORES).set_index("instance_id")
    frozen_scores = pd.read_parquet(LL / "scores" / frozen_tag / f"{nm}_t{th0:g}.parquet").set_index("instance_id")
    fz_s, fz_e = p6.read_loci(LL / "loci" / frozen_tag / f"{nm}_t{th0:g}.parquet")
    fz_ribo = B.ribo_score(ctx.runs, fz_s, fz_e)
    sys.path.insert(0, str(RD / "ribo_audit/code"))
    import bio
    import audit as AU
    targets, lit = {}, None
    for rd, dd in (("strict", (150, 500)), ("loose", (500, 500))):
        bio.D_OP, bio.D_MEM = dd
        Bb = bio.Bio()
        targets[rd] = [Bb.run_units(i)[6] for i in range(len(Bb.D.runs))]
        lit = Bb.lit
        del Bb

    def bio_count(path):
        L = AU.Loci(str(path))
        return {rd: int(sum(1 for t in tg if t and all(bio.score_target(L, ts, te)["ok"] for ts, te in t))) for rd, tg in targets.items()}

    cases = pd.read_csv(RD / "ribo_audit/work2/cases2.tsv", sep="\t")
    v1 = cases[(cases.output == aname) & (cases.verdict == 1)]
    rid = dict(zip(ctx.runs.run_id, range(len(ctx.runs))))
    v1i = np.array([rid[r] for r in v1.run_id if r in rid], np.int64)
    v1C = [(int(c.run_s) + int(c.C_s), int(c.run_s) + int(c.C_e)) for c in v1.itertuples()]

    def v1_fit(path):
        L = AU.Loci(str(path))
        return int(sum(1 for ts, te in v1C if bio.score_target(L, ts, te)["ok"]))

    # case 8233: S17 | L22 junction inside run 8233
    r8 = ctx.runs[ctx.runs.run_id == 8233]
    j8 = None
    if len(r8):
        s0, e0 = int(r8.gid_start.iat[0]), int(r8.gid_end.iat[0])
        for g in range(s0, e0):
            if {str(lit[g]), str(lit[g + 1])} == {"S17", "L22"}:
                j8 = g
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
    fz_path = LL / "loci" / frozen_tag / f"{nm}_t{th0:g}.parquet"
    res = dict(base=base, frozen=dict(name=f"{frozen_tag}/{nm}_t{th0:g}", bio=bio_count(fz_path), any_fit=p6.any_fit_count(ctx.runs, fz_s, fz_e),
                                      loci=int(len(fz_s)), verdict1_runs=int(len(v1i)), verdict1_resolved=int(fz_ribo.resolved.to_numpy()[v1i].sum()),
                                      verdict1_C_fit=v1_fit(fz_path)), outputs={}, selection=None, even={}, case8233={"junction_gid": j8})
    bfz = np.load(LL / "nodes" / frozen_tag / "bonds_obs.npz")
    log("frozen", res["frozen"])
    per_all = {}
    for th in p6.THETAS:
        df = pd.read_parquet(LL / "loci" / var_tag / f"{nm}_t{th:g}.parquet")
        per, summ, sc, rib, loc = S1.score(ctx, DS.table(ctx, df), a.threads)
        score, cond = DS.sel_score(summ["odd"], rib["odd"], ref)
        per_all[th] = (per, summ, sc, rib, loc, df)
        res["outputs"][f"t{th:g}"] = dict(score_odd=score, conditions_odd=cond)
        log(base, th, "ODD", score, cond)
    c = {th: res["outputs"][f"t{th:g}"]["score_odd"] for th in p6.THETAS}
    res["selection"] = max(c, key=lambda th: tuple(c[th]))
    res["selection_written"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    jdump(res, P8 / f"score_dpann_{base}.json")
    log("SELECTION", res["selection"])
    C = ("DefenseFinder", "TXSScan", "CRISPRCasTyper")
    # case 8233 bond
    bz = np.load(LL / "nodes" / var_tag / ("bonds_windows.npz" if base == "P" else "bonds_obs.npz"))
    if j8 is not None:
        n_ = len(ctx.gc)
        s_var = ll.boundary_score(bz["k"].astype(np.int64), bz["d"].astype(np.int64), bz["lam"].astype(np.float64), n_)
        s_fz = ll.boundary_score(bfz["k"].astype(np.int64), bfz["d"].astype(np.int64), bfz["lam"].astype(np.float64), n_)
        covering = np.flatnonzero((bz["k"] <= j8) & (bz["k"] + bz["d"] > j8))
        res["case8233"].update(boundary_score_frozen=float(s_fz[j8]), boundary_score_variant=float(s_var[j8]),
                               covering_bonds=int(len(covering)),
                               windows_of_covering=sorted(set(int(x) for x in bz["win"][covering])) if "win" in bz.files else None)
    for th in sorted({res["selection"], th0}):
        per, summ, sc, rib, loc, df = per_all[th]
        P = per.set_index("instance_id")
        s_, e_ = df.s.to_numpy(np.int64), df.e.to_numpy(np.int64)
        path = LL / "loci" / var_tag / f"{nm}_t{th:g}.parquet"
        item = dict(selected=bool(th == res["selection"]), summary_even=summ["even"], ribo_all=rib["all"], loci=loc,
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
        item["bio"] = bio_count(path)
        item["verdict1_resolved"] = int(sc.resolved.to_numpy()[v1i].sum())
        item["verdict1_C_fit"] = v1_fit(path)
        pr = np.unique(p6.co_pairs(s_, e_, ist, ien, M))
        item["new_curated_pairs_vs_frozen"] = int(len(np.setdiff1d(pr, base_pairs)))
        vf = item["vs_frozen_even"]
        item["justified"] = dict(strict_plus50=bool(item["bio"]["strict"] >= res["frozen"]["bio"]["strict"] + 50), seven_bars=bars["all"],
                                 no_new_pairs=bool(item["new_curated_pairs_vs_frozen"] == 0), summed=bool(vf["summed"][0] >= -0.01),
                                 f1={c_: bool(vf[c_]["d_f1"][0] >= -0.01) for c_ in C})
        item["justified"]["all"] = bool(item["justified"]["strict_plus50"] and bars["all"] and item["justified"]["no_new_pairs"]
                                        and item["justified"]["summed"] and all(item["justified"]["f1"].values()))
        res["even"][f"t{th:g}"] = item
        log(base, th, "EVEN summed", round(summ["even"]["summed_DF_TXSS_excess"], 4), "res", rib["all"]["resolved"], "bio", item["bio"],
            "v1", item["verdict1_resolved"], item["verdict1_C_fit"], "pairs", item["new_curated_pairs_vs_frozen"], "JUSTIFIED", item["justified"]["all"])
    res["seconds"] = round(time.time() - t0, 1)
    jdump(res, P8 / f"score_dpann_{base}.json")


# =========================================================================== Spacedust
def cmd_sd(a):
    import numba
    numba.set_num_threads(a.threads)
    import s207
    sel = {b: json.load(open(P8 / f"score_dpann_{b}.json"))["selection"] for b in ("P", "Z")}
    g = s207.load_universe_genes()
    Bn = s207.Bench(a.threads)
    refs = {"P": pd.read_parquet(LL / "scores/sd_WA/pr_L4b_t0.2.parquet"), "Z": pd.read_parquet(LL / "scores/sd_Z/pr_L4bp_t0.02.parquet")}
    res = dict(label="DEVELOPMENT", selection_from_dpann=sel, outputs={})
    for base, tag, nm in (("P", "sd_WB", "L4b"), ("Z", "sd_ZBW", "L4bp")):
        th = sel[base]
        df = pd.read_parquet(LL / "loci" / tag / f"{nm}_t{th:g}.parquet")
        s, e = df.s.to_numpy(np.int64), df.e.to_numpy(np.int64)
        r, pr = Bn.score(s207.gene_spans(g, s, e), f"p8 sd {base} {th}", np.random.default_rng(17))
        od = LL / "scores" / tag
        od.mkdir(parents=True, exist_ok=True)
        pr.to_parquet(od / f"pr_{nm}_t{th:g}.parquet", index=False)
        pd_ = s207.paired(pr, refs[base])
        res["outputs"][f"{base}_t{th:g}"] = dict(score=r, vs_frozen=pd_, verdict_vs_frozen=s207.verdict(pd_))
        log("SD", base, th, {k: r[k] for k in ("f1_observed", "f1_random", "f1_excess", "ci_lo", "ci_hi")}, pd_["excess"])
    jdump(res, P8 / "sd_source207.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["pbw", "zbw", "score", "sd"])
    ap.add_argument("--data", default="dpann")
    ap.add_argument("--base", default="P")
    ap.add_argument("--blocks", type=int, default=4096)
    ap.add_argument("--threads", type=int, default=1)
    a = ap.parse_args()
    {"pbw": cmd_pbw, "zbw": cmd_zbw, "score": cmd_score, "sd": cmd_sd}[a.cmd](a)


if __name__ == "__main__":
    main()
