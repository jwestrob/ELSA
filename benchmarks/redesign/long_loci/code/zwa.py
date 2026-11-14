#!/usr/bin/env python
"""long_loci part 5: Z-WA = ws_calib z >= 10.6 list bonds at the adaptive pair distance (predeclared5.json).

Round 0 of vl.py (lists outside the own group, group counting) with a per-adjacency window: every pair of genes within
8 positions (homolog pairs excluded) is tested at its own distance d: list partners must lie within d positions of each
other in the other genome; E per genome min(1, 2 d c_x c_y / m), own term min(1, 2 d / m). Kernels copied from vl.py
(mark, pass_count, adjacency) with the window as a parameter; in fixed-window mode at window 3 they must reproduce
vl.pass_count exactly (gate). Exclusion masks extended to distance 8 with softrec.exclusion (K1000 variant); their
d <= 3 part must equal the stored masks (gate).
usage: zwa.py run --data dpann|sd --threads T
"""
import argparse
import json
import math
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit, prange

RD = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111")
LL = RD / "long_loci"
sys.path.insert(0, str(RD / "softrec/code"))
sys.path.insert(0, str(RD / "vocab_lists/code"))
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import vl  # noqa: E402  (read-only import)
import ll  # noqa: E402
import lb  # noqa: E402

WMAX = 8
LADDER = (4.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0, 25.0, 30.0)
KAPPA = {"dpann": RD / "context_units/ctx/kappa_var_knn50.npy", "sd": LL / "sd_lists/kappa_var_knn50.npy"}
STORED_EXCL = {"dpann": RD / "vocab_lists/excl", "sd": LL / "sd_lists/excl"}
log = ll.log


def rss_gb():
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024, 1)


# =========================================================================== kernels (vl.py copies with a window)
@njit(cache=True)
def mark_min(stv, bx, cx, nbs, nbg, cid, n, Rst, Rd, Lst, Ld, ca, cast, wmark):
    """vl.mark for terminal nodes, marking up to wmark positions and keeping the minimum distance to any partner."""
    for t in range(cx):
        s = np.int64(nbs[bx + t])
        G = np.int64(nbg[bx + t])
        c = cid[s]
        u = s + 1
        dist = 1
        while dist <= wmark and u < n:
            if cid[u] != c:
                break
            if Rst[u] != stv:
                Rst[u] = stv
                Rd[u] = dist
            elif dist < Rd[u]:
                Rd[u] = dist
            u += 1
            dist += 1
        u = s - 1
        dist = 1
        while dist <= wmark and u >= 0:
            if cid[u] != c:
                break
            if Lst[u] != stv:
                Lst[u] = stv
                Ld[u] = dist
            elif dist < Ld[u]:
                Ld[u] = dist
            u -= 1
            dist += 1
        if cast[G] != stv:
            cast[G] = stv
            ca[G] = 0
        ca[G] += 1


@njit(parallel=True, cache=True)
def pass_count_w(m, aptr, ak, gene_at, lstart, tcnt, nbs, nbg, cid, grp, ngenome, ngroup, mg, wmark, fixedW,
                 out_n, out_ng, out_E, out_Eg, nchunks):
    """vl.pass_count at round 0 (terminal nodes; node j = slot j) with window = fixedW, or = the pair's distance d
    when fixedW == 0."""
    n = len(cid)
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
        grst = np.full(ngroup, -1, np.int64)
        grst2 = np.full(ngroup, -1, np.int64)
        grlp = np.zeros(ngroup, np.float64)
        grt = np.zeros(ngroup, np.int64)
        for j in range(lo, hi):
            if aptr[j + 1] == aptr[j]:
                continue
            g = gene_at[j]
            mark_min(j, lstart[g], np.int64(tcnt[g]), nbs, nbg, cid, n, Rst, Rd, Lst, Ld, ca, cast, wmark)
            for r in range(aptr[j], aptr[j + 1]):
                k = ak[r]
                win = fixedW if fixedW > 0 else k - j
                gk = gene_at[k]
                by = lstart[gk]
                cy = np.int64(tcnt[gk])
                ntb = 0
                nsup = 0
                ngs = 0
                for t in range(cy):
                    s = np.int64(nbs[by + t])
                    G = np.int64(nbg[by + t])
                    if cbst[G] != r:
                        cbst[G] = r
                        cb[G] = 0
                        tb[ntb] = G
                        ntb += 1
                    cb[G] += 1
                    if (Rst[s] == j and Rd[s] <= win) or (Lst[s] == j and Ld[s] <= win):
                        if gst[G] != r:
                            gst[G] = r
                            nsup += 1
                            gg = grp[G]
                            if grst2[gg] != r:
                                grst2[gg] = r
                                ngs += 1
                Es = 0.0
                ngt = 0
                for q in range(ntb):
                    G = tb[q]
                    if cast[G] == j and ca[G] > 0:
                        x = 2.0 * win * ca[G] * cb[G] / mg[G]
                        p = x if x < 1.0 else 1.0
                        Es += p
                        gg = grp[G]
                        if grst[gg] != r:
                            grst[gg] = r
                            grlp[gg] = 0.0
                            grt[ngt] = gg
                            ngt += 1
                        if p >= 1.0:
                            grlp[gg] = -1e300
                        else:
                            grlp[gg] += math.log1p(-p)
                Eg = 0.0
                for q in range(ngt):
                    Eg += 1.0 - math.exp(grlp[grt[q]])
                out_n[r] = nsup
                out_ng[r] = ngs
                out_E[r] = Es
                out_Eg[r] = Eg


@njit(cache=True)
def adjacency_w(m, cid, nlen, excl, off, wmax):
    """vl.adjacency at round 0 (all nodes terminal) up to distance wmax; excl indexed by instances(wmax) offsets."""
    cnt = 0
    aptr = np.zeros(m + 1, np.int64)
    ak = np.empty(m * wmax, np.int64)
    for j in range(m):
        aptr[j] = cnt
        if nlen[j] == 0:
            continue
        for d in range(1, wmax + 1):
            k = j + d
            if k >= m or cid[k] != cid[j]:
                break
            if nlen[k] == 0:
                continue
            if excl[off[j] + d - 1]:
                continue
            ak[cnt] = k
            cnt += 1
    aptr[m] = cnt
    return aptr, ak[:cnt].copy()


# =========================================================================== masks
def masks8(data, U, threads):
    """Exclusion masks over instances up to distance 8 (K1000 variant), observed order and 3 shuffles; gate: the
    d <= 3 part equals the stored masks."""
    import softrec as SR
    u = U.u
    n = u.n
    t0 = time.time()
    if data == "dpann":
        LSe, Lemb = vl.emb_lists(U)
        X, uq, capcos, tok, mode = Lemb.X, Lemb.uq, Lemb.capcos[Lemb.variants.index("K1000_fnone")][None, :], Lemb.tok, Lemb.excl_mode
        del LSe
    else:
        import sdu
        X = np.load(RD / "transfer_spacedust/emb/zca10_f32__uid.npy")
        uq = sdu.uid_of_gene()
        Ia = np.ascontiguousarray(np.load(LL / "sd_knn/zca10__k2000__I.npy"))
        Da = np.load(LL / "sd_knn/zca10__k2000__D.npy").astype(np.float32)
        ok = uq >= 0
        order = np.flatnonzero(ok)[np.argsort(uq[ok], kind="stable")]
        mptr = np.r_[0, np.cumsum(np.bincount(uq[ok], minlength=Ia.shape[0]))].astype(np.int64)
        nb = np.full((n, 1000), -1, np.int32)
        tail = np.zeros((n, 1000), np.uint16)
        llen = np.zeros(n, np.int64)
        ntie = np.zeros(n, np.int64)
        rf = np.zeros((n, 1), np.int64)
        cos_at = np.zeros((n, 2), np.float64)
        SR.expand_knn(uq, u.lin.astype(np.int64), Ia, Da, mptr, order.astype(np.int64), 1000, np.array([-2.0]),
                      np.array([200, 1000], np.int64), nb, tail, llen, ntie, rf, cos_at)
        del Ia, Da, nb, tail
        capcos = np.maximum(cos_at[:, 1], -2.0).astype(np.float32)[None, :]
        tok, mode = np.zeros(1, np.int64), 1
    nd8, off8, left8, d8 = u.instances(WMAX)
    nd3, off3, left3, d3 = u.instances(3)
    out, info = {}, {"seconds_lists": round(time.time() - t0, 1)}
    for r in (-1, 0, 1, 2):
        gene_at = np.arange(n, dtype=np.int64) if r < 0 else SR.null_order(u, r)[0]
        ex = np.zeros((1, len(left8)), np.bool_)
        SR.exclusion(n, 1, gene_at, nd8, off8, mode, tok, X, uq, capcos, ex)
        stored = np.load(STORED_EXCL[data] / f"group_order{r}.npz")["K1000"]
        same3 = bool(np.array_equal(ex[0][off8[left3] + d3 - 1], stored))
        info[str(r)] = dict(instances=int(len(left8)), excluded=int(ex[0].sum()),
                            excluded_by_d={str(dd): int(ex[0][d8 == dd].sum()) for dd in range(1, WMAX + 1)},
                            d_le_3_equals_stored=same3)
        log("mask", data, r, info[str(r)])
        assert same3, f"mask gate failed for order {r}"
        out[r] = ex[0]
    info["seconds"] = round(time.time() - t0, 1)
    return out, off8, off3, info


# =========================================================================== run
def run_order(U, LS, K, r, excl, off, wmax, fixedW, threads):
    import softrec as SR
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
    aptr, ak = adjacency_w(n, cid, nlen, excl, off, wmax)
    R = len(ak)
    on, ong = np.zeros(R, np.int64), np.zeros(R, np.int64)
    oE, oEg = np.zeros(R), np.zeros(R)
    t0 = time.time()
    pass_count_w(n, aptr, ak, gene_at, LS.lstart, tcnt, nbs, LS.nbg, cid, U.grp, u.ng, U.ngroup, U.mg, wmax, fixedW,
                 on, ong, oE, oEg, threads * 4)
    sec = time.time() - t0
    rj = np.repeat(np.arange(n, dtype=np.int64), np.diff(aptr))
    d = ak - rj
    win = d if fixedW == 0 else np.full(R, fixedW)
    own = np.minimum(1.0, 2.0 * win / U.mg[u.gc.astype(np.int64)[rj]])
    nu = ong + 1.0
    Eu = oEg + own
    lam = vl.lam_np(nu, Eu)
    return dict(rj=rj, ak=ak, d=d, on=on, ong=ong, oE=oE, oEg=oEg, nu=nu, Eu=Eu, lam=lam, seconds=round(sec, 1))


def cmd_run(a):
    import numba
    numba.set_num_threads(a.threads)
    t0 = time.time()
    if a.data == "dpann":
        U = vl.Univ(vl.GROUPS)
    else:
        import sdu
        U = sdu.SDUniv("group")
    u = U.u
    n = u.n
    res = dict(data=a.data, timing={})
    excl8, off8, off3, minfo = masks8(a.data, U, a.threads)
    res["masks"] = minfo
    res["timing"]["masks"] = round(time.time() - t0, 1)
    t1 = time.time()
    if a.data == "dpann":
        LS, K = lb.make_lists(U, "z10.6")
    else:
        import sdu
        LS, K = sdu.z_lists_sd(U, 10.6)
    res["lists"] = LS.info
    res["timing"]["lists"] = round(time.time() - t1, 1)
    # gate: fixed window 3 equals vl.pass_count on the observed order (stored masks, instances(3))
    t1 = time.time()
    stored = np.load(STORED_EXCL[a.data] / "group_order-1.npz")["K1000"]
    mine = run_order(U, LS, K, -1, stored, off3, 3, 3, a.threads)
    if a.data == "dpann":
        O = lb.LOrder(U, LS, -1, "group")
    else:
        import sdu
        O = sdu.LOrderSD(U, LS, -1, "group")
    Lr = lb.Lrn(U, LS, O, K, "grp", a.threads, excl_K=1000)
    st, aptr, ak, ev = Lr.round0(4.0)
    gate = dict(rows_equal=bool(np.array_equal(ak, mine["ak"])),
                n_equal=bool(np.array_equal(ev["n"] - 1.0, mine["on"].astype(np.float64))),
                ng_equal=bool(np.array_equal(ev["ng"] - 1.0, mine["ong"].astype(np.float64))),
                E_equal=bool(np.array_equal(ev["E"], mine["oE"] + np.minimum(1.0, 6.0 / U.mg[u.gc.astype(np.int64)[mine["rj"]]]))),
                Eg_equal=bool(np.array_equal(ev["Eg"], mine["Eu"])),
                lam_equal=bool(np.array_equal(ev["lam"], mine["lam"])), rows=int(len(ak)))
    res["gate_window3"] = gate
    res["timing"]["gate"] = round(time.time() - t1, 1)
    log("GATE window 3 vs vl.pass_count", gate)
    assert all(v for k, v in gate.items() if k != "rows"), "kernel gate failed"
    del O, Lr, ev, st, mine
    # adaptive round 0 in the observed order and the 3 shuffles
    lad, obs = {}, None
    for r in (-1, 0, 1, 2):
        t1 = time.time()
        x = run_order(U, LS, K, r, excl8[r], off8, WMAX, 0, a.threads)
        ok = (x["nu"] >= 2) & (x["lam"] >= min(LADDER))
        lad[str(r)] = dict(counts={str(t): int((ok & (x["lam"] >= t)).sum()) for t in LADDER}, adjacencies=int(len(x["ak"])),
                           pass_seconds=x["seconds"], seconds=round(time.time() - t1, 1))
        log("LADDER Z-WA", a.data, r, lad[str(r)], "rss", rss_gb())
        if r < 0:
            obs = dict(k=x["rj"][ok], d=x["d"][ok], lam=x["lam"][ok])
        del x
    rows = []
    for t in LADDER:
        o = lad["-1"]["counts"][str(t)]
        nl = float(np.mean([lad[str(r)]["counts"][str(t)] for r in (0, 1, 2)]))
        rows.append(dict(tau=t, observed=o, null_mean=nl, fdr=nl / o if o else None))
    okt = [x["tau"] for x in rows if x["fdr"] is not None and x["fdr"] <= 0.01]
    tau = min(okt) if okt else None
    res.update(ladder=lad, tau_table=rows, tau_star=tau)
    log("TAU Z-WA", a.data, tau, rows[:4])
    keep = obs["lam"] >= tau
    k, d, lam = obs["k"][keep], obs["d"][keep], obs["lam"][keep]
    s = ll.boundary_score(k, d, lam, n)
    near = d <= 3
    s_near = ll.boundary_score(k[near], d[near], lam[near], n)
    dist = np.bincount(d, minlength=WMAX + 1)[1:]
    res["bonds"] = dict(tau=tau, bonds=int(len(k)), by_distance={str(i + 1): int(v) for i, v in enumerate(dist)},
                        share_by_distance={str(i + 1): round(float(v / max(len(k), 1)), 4) for i, v in enumerate(dist)},
                        boundaries_bonded=int((s > 0).sum()), boundaries_only_by_d_ge4=int(((s > 0) & (s_near == 0)).sum()))
    tag = f"{a.data}_ZWA"
    od, nd = LL / "loci" / tag, LL / "nodes" / tag
    od.mkdir(parents=True, exist_ok=True)
    nd.mkdir(parents=True, exist_ok=True)
    np.savez(nd / "bonds_obs.npz", k=k, d=d, lam=lam)
    st_, en_ = ll.runs(s > 0)
    res["outputs"] = {"L1": ll.loci_info(ll.save_loci(od, "L1", st_, en_), n)}
    kv = np.load(KAPPA[a.data]).astype(np.float64)
    trees = {}
    for th in ll.THETAS:
        low = (kv < th) & (s > 0)
        w = np.where(low, 1e-6 + np.maximum(np.nan_to_num(kv, nan=0.0), 0.0), s)
        trees[th] = ll.tree(w)
    alln = pd.concat([t[["lo", "hi"]] for t in trees.values()]).drop_duplicates().reset_index(drop=True)
    lo, hi = alln.lo.to_numpy(np.int64), alln.hi.to_numpy(np.int64)
    grp = U.grp.astype(np.int64)
    gc = u.gc.astype(np.int64)
    t1 = time.time()
    nO, eO, nI = lb.count_lists(lo, hi, grp[gc[lo]], LS.lstart.astype(np.int64), LS.tcnt[K].astype(np.int64), LS.nb,
                                u.cid.astype(np.int64), gc, grp, int(grp.max()) + 1, U.mg.astype(np.float64), 8192)
    alln["n"], alln["E"], alln["nI"] = nO, np.maximum(eO, 0.0), nI
    alln["lam"] = ll.lam_of(alln.n.to_numpy(), alln.E.to_numpy())
    res["node_test"] = dict(unique_intervals=int(len(alln)), seconds=round(time.time() - t1, 1))
    for th, t in trees.items():
        t = t.merge(alln, on=["lo", "hi"], how="left")
        km = (t.n.to_numpy() >= 2) & (t.lam.to_numpy() >= tau) & (t.nI.to_numpy() >= 2)
        d1 = ll.save_loci(od, f"L4b_t{th:g}", t.lo.to_numpy()[km], t.hi.to_numpy()[km])
        pm = t.hi.to_numpy() > t.lo.to_numpy()
        d2 = ll.save_loci(od, f"L4bp_t{th:g}", t.lo.to_numpy()[pm], t.hi.to_numpy()[pm])
        res["outputs"][f"L4b_t{th:g}"] = ll.loci_info(d1, n)
        res["outputs"][f"L4bp_t{th:g}"] = ll.loci_info(d2, n)
    res["timing"]["total"] = round(time.time() - t0, 1)
    res["peak_rss_gb"] = rss_gb()
    log("DONE", {k_: v for k_, v in res.items() if k_ in ("tau_star", "bonds", "timing", "peak_rss_gb")})
    ll.jdump(res, LL / "results" / f"zwa_{a.data}.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run"])
    ap.add_argument("--data", default="dpann")
    ap.add_argument("--threads", type=int, default=1)
    a = ap.parse_args()
    cmd_run(a)


if __name__ == "__main__":
    main()
