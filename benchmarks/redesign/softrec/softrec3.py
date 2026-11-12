#!/usr/bin/env python
"""softrec round 3 (workstream softrec3): neighbour lists stratified by lineage.

A gene's list holds its single best match in each genome of every other genus (partner gene + cosine), kept when the
cosine clears an evidence floor calibrated for a per-genome maximum; per genus, entries are ranked by cosine (level
0 = the genus's best genome). The statistic counts genera with a supporting pair, as rounds 1-2.

usage
  softrec3.py prep --threads T
      zca10 vectors of the unique sequences (float16, $R3/vec/zca10_f16.npy) and the metadata the GPU job needs;
      null calibration on 10,000 random unique queries (histograms from the first 2,000) against all 3,585,303 genes (float32, exact): per-genome maximum,
      argmax and minimum, histograms of all pair cosines split by same / different MMseqs2 rescue3 family (diagnostic
      only); per-genome floors floor_G(alpha) = smallest cosine c with 1 - (1 - p(c))^n_G <= alpha, where p(c) is the
      mirrored lower tail P(cos <= -c) of all pairs (label-free chance model); checks of the model against the decoy
      maxima (-min) and against genomes lacking the query's MMseqs2 family.
  softrec3.py cheap --threads T
      Task 1a from the stored top-2,000 zca10 kNN: per gene, the genera (and genomes) reached by the list of genes
      outside its genus, by the first 1,000 such genes (the round-2 depth) and by the whole list (best per genus), with
      and without the calibrated floor; compared with the genera that hold a member of the gene's MMseqs2 family.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit, prange

RD = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111")
sys.path.insert(0, str(RD / "softrec/code"))
import softrec as SR  # noqa: E402
from embed_tokens import load_run, transform, stamp, rss_gb  # noqa: E402

R3 = RD / "softrec/r3"
EMB = SR.EMB
NQ = 10000
NH = 2000            # queries whose pair cosines enter the histograms
NBIN = 2000          # pair-cosine histogram bins on [-1, 1)
ALPHAS = (1e-2, 1e-3, 1e-4, 1e-5)
TOKENS = RD / "engine/dpann_tokens/mmseqs__seq__e3c50s75_rescue3__dpann.parquet"


def jdump(path, obj):
    tmp = Path(str(path) + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x)) + "\n")
    os.replace(tmp, path)


def rescue3_tokens(n):
    t = pd.read_parquet(TOKENS)
    tok = np.full(n, -1, np.int64)
    tok[t.gid.to_numpy()] = t.token.to_numpy()
    assert (tok >= 0).all()
    return tok


# ============================================================================== calibration kernel
@njit(parallel=True, cache=True)
def block_stats(S, seg_lo, seg_hi, seg_g, tq, tcol, nh, omax, oarg, omin, hsame, hdiff, nch):
    nq = S.shape[0]
    for ch in prange(nch):
        lo = ch * nq // nch
        hi = (ch + 1) * nq // nch
        for r in range(lo, hi):
            t = tq[r]
            for k in range(len(seg_lo)):
                best = -2.0
                bi = -1
                mn = 2.0
                for c in range(seg_lo[k], seg_hi[k]):
                    s = S[r, c]
                    if s > best:
                        best = s
                        bi = c - seg_lo[k]
                    if s < mn:
                        mn = s
                if r < nh:
                    for c in range(seg_lo[k], seg_hi[k]):
                        b = int((S[r, c] + 1.0) * 1000.0)
                        if b < 0:
                            b = 0
                        if b > 1999:
                            b = 1999
                        if tcol[c] == t:
                            hsame[ch, b] += 1
                        else:
                            hdiff[ch, b] += 1
                G = seg_g[k]
                omax[r, G] = best
                oarg[r, G] = bi
                omin[r, G] = mn


def floors_from_hist(h, n_g, alphas):
    """Mirrored lower tail p(c) = P(cos < -c) on the bin grid c = k / 1000; floor_G = smallest c with
    1 - (1 - p(c))^n_G <= alpha."""
    tot = h.sum()
    low = np.cumsum(h) / tot                      # low[b] = P(bin <= b) = P(cos < (b + 1) / 1000 - 1)
    ks = np.arange(0, 1000)
    # P(cos < -c) with c = k/1000: bins b with (b+1)/1000 - 1 <= -c  <=> b <= 999 - k
    p = np.array([low[999 - k] if 999 - k >= 0 else 0.0 for k in ks])
    out = np.zeros((len(alphas), len(n_g)), np.float32)
    for ai, al in enumerate(alphas):
        target = 1.0 - (1.0 - al) ** (1.0 / n_g)
        for G, tg in enumerate(target):
            ok = np.flatnonzero(p <= tg)
            out[ai, G] = ks[ok[0]] / 1000.0 if len(ok) else 1.0
    return out, p


def cmd_prep(a):
    import numba
    numba.set_num_threads(a.threads)
    t0 = time.time()
    (R3 / "vec").mkdir(parents=True, exist_ok=True)
    (R3 / "calib").mkdir(parents=True, exist_ok=True)
    u = SR.Universe()
    uq = np.load(EMB / "uid_of_row.npy").astype(np.int64)
    X, uu, valid, cnt = load_run(EMB)
    tinfo = transform(X, valid, cnt, "zca10")
    stamp("zca10", tinfo, f"{time.time() - t0:.0f}s rss {rss_gb()}")
    U, dim = X.shape
    fn = R3 / "vec/zca10_f16.npy"
    out = np.lib.format.open_memmap(fn, mode="w+", dtype=np.float16, shape=(U, dim))
    for s in range(0, U, 200_000):
        out[s:s + 200_000] = X[s:s + 200_000].astype(np.float16)
    out.flush()
    del out
    np.savez(R3 / "vec/meta.npz", uq=uq, g_start=u.g_start, g_end=u.g_end, n_g=u.n_g.astype(np.int64),
             lin_of_genome=u.lin_of_genome, valid=valid)
    stamp("vectors written", fn, f"{time.time() - t0:.0f}s")
    # ---------------------------------------------------------------- calibration
    tok = rescue3_tokens(u.n)
    _, first0 = np.unique(uq, return_index=True)
    tok_u = tok[first0]                                     # token of the sequence's first gene (diagnostic only)
    n_conflict = int((tok_u[uq] != tok).sum())              # genes whose token differs from their sequence's first gene
    rng = np.random.default_rng(17)
    qids = np.sort(rng.choice(np.flatnonzero(valid), NQ, replace=False))
    Xq = np.ascontiguousarray(X[qids])
    tq = tok_u[qids]
    NG = u.ng
    omax = np.full((NQ, NG), -2, np.float32)
    oarg = np.full((NQ, NG), -1, np.int16)
    omin = np.full((NQ, NG), 2, np.float32)
    nch = a.threads * 4
    hsame = np.zeros((nch, NBIN), np.int64)
    hdiff = np.zeros((nch, NBIN), np.int64)
    gs, ge = u.g_start, u.g_end
    G0 = 0
    t1 = time.time()
    while G0 < NG:
        G1 = G0
        while G1 < NG and ge[G1] - gs[G0] < 40000:
            G1 += 1
        G1 = max(G1, G0 + 1)
        c0, c1 = gs[G0], ge[G1 - 1]
        Y = X[uq[c0:c1]]
        S = Xq @ Y.T
        seg_lo = (gs[G0:G1] - c0).astype(np.int64)
        seg_hi = (ge[G0:G1] - c0).astype(np.int64)
        block_stats(S, seg_lo, seg_hi, np.arange(G0, G1, dtype=np.int64), tq, tok[c0:c1], NH, omax, oarg, omin, hsame, hdiff, nch)
        G0 = G1
    stamp("calibration pass", f"{time.time() - t1:.0f}s")
    hs, hd = hsame.sum(0), hdiff.sum(0)
    np.save(R3 / "calib/qids.npy", qids)
    np.save(R3 / "calib/omax.npy", omax)
    np.save(R3 / "calib/oarg.npy", oarg)
    np.save(R3 / "calib/omin.npy", omin)
    np.savez(R3 / "calib/hist.npz", same=hs, diff=hd)
    # family presence per (query, genome) (diagnostic null: genomes lacking the query's MMseqs2 family)
    key = np.unique(tok * NG + u.gc)
    famp = np.zeros((NQ, NG), bool)
    tset = {}
    kt, kg = key // NG, key % NG
    order = np.argsort(kt, kind="stable")
    kt, kg = kt[order], kg[order]
    ptr = np.searchsorted(kt, tq)
    ptr2 = np.searchsorted(kt, tq, side="right")
    for r in range(NQ):
        famp[r, kg[ptr[r]:ptr2[r]]] = True
    np.save(R3 / "calib/famp.npy", famp)
    # ---------------------------------------------------------------- floors
    hall = hs + hd
    fl, p = floors_from_hist(hall, u.n_g, ALPHAS)
    np.save(R3 / "calib/floors.npy", fl)
    res = dict(seconds=round(time.time() - t0, 1), queries=NQ, pairs=int(hall.sum()), same_family_pairs=int(hs.sum()),
               genes_with_rescue3_token_differing_from_identical_sequence=n_conflict)
    grid = [0.1, 0.15, 0.2, 0.25, 0.3, 0.32, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7]
    tail = {}
    for c in grid:
        k = int(round(c * 1000))
        up_all = hall[1000 + k:].sum() / hall.sum()
        lo_all = hall[:1000 - k].sum() / hall.sum()
        up_d = hd[1000 + k:].sum() / hd.sum()
        lo_d = hd[:1000 - k].sum() / hd.sum()
        tail[f"{c:g}"] = dict(upper_all=up_all, lower_all=lo_all, upper_difffam=up_d, lower_difffam=lo_d,
                              same_family_share_of_upper=(hs[1000 + k:].sum() / max(1, hall[1000 + k:].sum())))
    res["pair_tails"] = tail
    res["floors"] = {f"{al:g}": dict(q01_q50_q99=np.quantile(fl[i], [0.01, 0.5, 0.99]).tolist(), min=float(fl[i].min()),
                                     max=float(fl[i].max())) for i, al in enumerate(ALPHAS)}
    # model checks: decoy per-genome maxima (-min) and maxima in genomes lacking the query's family
    _, first_gene = np.unique(uq, return_index=True)       # first gene of each unique sequence (all U present)
    assert len(first_gene) == U
    qlin = u.lin[first_gene[qids]]
    other = u.lin_of_genome[None, :] != qlin[:, None]
    dec = -omin
    chk = {}
    for i, al in enumerate(ALPHAS):
        f = fl[i][None, :]
        d_rate = float((dec >= f)[other].mean())
        a_rate = float((omax >= f)[other & ~famp].mean())
        per_q = ((omax >= f) & other).sum(1)
        per_q_dec = ((dec >= f) & other).sum(1)
        lin_reach = np.zeros(NQ, np.int64)
        lin_fam = np.zeros(NQ, np.int64)
        lin_both = np.zeros(NQ, np.int64)
        for r in range(NQ):
            m = (omax[r] >= fl[i]) & other[r]
            lr = np.unique(u.lin_of_genome[m])
            lf = np.unique(u.lin_of_genome[famp[r] & other[r]])
            lin_reach[r], lin_fam[r] = len(lr), len(lf)
            lin_both[r] = len(np.intersect1d(lr, lf, assume_unique=True))
        chk[f"{al:g}"] = dict(decoy_exceed_rate=d_rate, famabsent_exceed_rate=a_rate,
                              entries_per_query_mean=float(per_q.mean()), entries_q50_q90_q99=np.quantile(per_q, [.5, .9, .99]).tolist(),
                              decoy_entries_per_query_mean=float(per_q_dec.mean()),
                              genera_reached_q50_mean=[float(np.median(lin_reach)), float(lin_reach.mean())],
                              family_genera_q50_mean=[float(np.median(lin_fam)), float(lin_fam.mean())],
                              family_genera_reached_share=float(lin_both.sum() / max(1, lin_fam.sum())),
                              reached_genera_without_family_member_share=float((lin_reach - lin_both).sum() / max(1, lin_reach.sum())))
    res["checks"] = chk
    jdump(R3 / "calib/calib.json", res)
    stamp("calib", json.dumps(res), f"rss {rss_gb()}")


# ============================================================================== Task 1a (cheap)
@njit(parallel=True, cache=True)
def reach_kernel(genes, uq, u0, lin, gc, Ic, Dc, mptr, members, floor_g, famlin_ptr, famlin, tokg,
                 n_lin, n_genome, depth, out, nch):
    """out columns: 0 genera (whole top-2000 list), 1 genera (first `depth` genes), 2 genera with best >= 0.32,
    3 genera with a genome best >= floor_g, 4 genomes >= floor_g, 5 family genera (other than own),
    6 family genera reached (whole list), 7 (first depth genes), 8 (>= floor), 9 genes in list."""
    m = len(genes)
    for ch in prange(nch):
        lo = ch * m // nch
        hi = (ch + 1) * m // nch
        gst = np.full(n_lin, -1, np.int64)
        gdep = np.full(n_lin, -1, np.int64)
        gfl = np.full(n_lin, -1, np.int64)
        mst = np.full(n_genome, -1, np.int64)
        for q in range(lo, hi):
            a = genes[q]
            U = uq[a]
            la = lin[a]
            n_all = 0
            n_dep = 0
            n_032 = 0
            n_fl = 0
            n_gen = 0
            k = 0
            # identical copies first (cosine 1)
            j = -1
            while True:
                if j == -1:
                    V = U
                    s = 1.0
                else:
                    if j >= Ic.shape[1]:
                        break
                    V = Ic[U - u0, j]
                    if V < 0:
                        break
                    s = Dc[U - u0, j]
                    if V == U:
                        j += 1
                        continue
                for t in range(mptr[V], mptr[V + 1]):
                    b = members[t]
                    lb = lin[b]
                    if lb == la:
                        continue
                    k += 1
                    if gst[lb] != q:
                        gst[lb] = q
                        n_all += 1
                        if s >= 0.32:
                            n_032 += 1
                    if k <= depth and gdep[lb] != q:
                        gdep[lb] = q
                        n_dep += 1
                    G = gc[b]
                    if mst[G] != q:
                        mst[G] = q
                        if s >= floor_g[G]:
                            n_gen += 1
                            if gfl[lb] != q:
                                gfl[lb] = q
                                n_fl += 1
                j += 1
            t0 = tokg[a]
            nf = 0
            r_all = 0
            r_dep = 0
            r_fl = 0
            for z in range(famlin_ptr[t0], famlin_ptr[t0 + 1]):
                lz = famlin[z]
                if lz == la:
                    continue
                nf += 1
                if gst[lz] == q:
                    r_all += 1
                if gdep[lz] == q:
                    r_dep += 1
                if gfl[lz] == q:
                    r_fl += 1
            out[q, 0] = n_all
            out[q, 1] = n_dep
            out[q, 2] = n_032
            out[q, 3] = n_fl
            out[q, 4] = n_gen
            out[q, 5] = nf
            out[q, 6] = r_all
            out[q, 7] = r_dep
            out[q, 8] = r_fl
            out[q, 9] = k


def reach_summary(R, famgen_bins=(1, 5, 20, 100, 300, 2000)):
    """R: per gene [reached, family genera, family genera reached] -> summary by family breadth."""
    out = {}
    reached, nf, inter = R[:, 0].astype(np.float64), R[:, 1].astype(np.float64), R[:, 2].astype(np.float64)
    out["all"] = dict(genes=int(len(R)), reached_q50_mean=[float(np.median(reached)), float(reached.mean())],
                      family_genera_q50_mean=[float(np.median(nf)), float(nf.mean())],
                      family_genera_reached_share=float(inter.sum() / max(1, nf.sum())),
                      reached_outside_family_share=float((reached - inter).sum() / max(1, reached.sum())),
                      genes_reaching_all_family_genera=float((inter >= nf).mean()))
    for lo_, hi_ in zip(famgen_bins[:-1], famgen_bins[1:]):
        m = (nf >= lo_) & (nf < hi_)
        if not m.any():
            continue
        out[f"famgen_{lo_}_{hi_ - 1}"] = dict(genes=int(m.sum()), family_genera_q50=float(np.median(nf[m])),
                                               reached_q50=float(np.median(reached[m])),
                                               family_genera_reached_share=float(inter[m].sum() / max(1, nf[m].sum())),
                                               median_gene_recall=float(np.median(inter[m] / np.maximum(nf[m], 1))))
    return out


def cmd_cheap(a):
    import numba
    numba.set_num_threads(a.threads)
    t0 = time.time()
    (R3 / "cheap").mkdir(parents=True, exist_ok=True)
    u = SR.Universe()
    uq = np.load(EMB / "uid_of_row.npy").astype(np.int64)
    U = int(uq.max()) + 1
    fl = np.load(R3 / "calib/floors.npy")
    floor_g = fl[ALPHAS.index(1e-3)].astype(np.float64)
    tok = rescue3_tokens(u.n)
    key = np.unique(tok * u.nl + u.lin)
    kt, kl = key // u.nl, key % u.nl
    T = int(tok.max()) + 1
    famlin_ptr = np.searchsorted(kt, np.arange(T + 1)).astype(np.int64)
    famlin = kl.astype(np.int64)
    order = np.argsort(uq, kind="stable")
    mptr = np.r_[0, np.cumsum(np.bincount(uq, minlength=U))].astype(np.int64)
    I = np.load(RD / "softrec/knn/zca10__k2000__I.npy", mmap_mode="r")
    D = np.load(RD / "softrec/knn/zca10__k2000__D.npy", mmap_mode="r")
    out = np.zeros((u.n, 10), np.int32)
    CH = 250_000
    for u0 in range(0, U, CH):
        u1 = min(U, u0 + CH)
        Ic = np.ascontiguousarray(I[u0:u1])
        Dc = np.ascontiguousarray(D[u0:u1]).astype(np.float32)
        genes = order[mptr[u0]:mptr[u1]]
        o = np.zeros((len(genes), 10), np.int32)
        reach_kernel(genes, uq, u0, u.lin.astype(np.int64), u.gc.astype(np.int64), Ic, Dc, mptr, order.astype(np.int64),
                     floor_g, famlin_ptr, famlin, tok, u.nl, u.ng, 1000, o, a.threads * 8)
        out[genes] = o
        stamp("cheap chunk", u0, f"{time.time() - t0:.0f}s")
    np.save(R3 / "cheap/reach_top2000.npy", out)
    res = dict(seconds=round(time.time() - t0, 1), floor="alpha 1e-3 per genome (calib/floors.npy row 1)",
               columns="0 genera whole list, 1 genera first 1000 genes, 2 genera best>=0.32, 3 genera floor, 4 genomes floor, "
                       "5 family genera, 6 fam reached whole, 7 fam reached depth1000, 8 fam reached floor, 9 genes in list")
    res["round2_depth1000"] = reach_summary(np.c_[out[:, 1], out[:, 5], out[:, 7]])
    res["top2000_best_per_genus_nofloor"] = reach_summary(np.c_[out[:, 0], out[:, 5], out[:, 6]])
    res["top2000_best_per_genus_floor1e-3"] = reach_summary(np.c_[out[:, 3], out[:, 5], out[:, 8]])
    res["top2000_genera_cos032_q50_mean"] = [float(np.median(out[:, 2])), float(out[:, 2].mean())]
    res["top2000_genomes_floor_q50_mean"] = [float(np.median(out[:, 4])), float(out[:, 4].mean())]
    res["list_genes_q50"] = float(np.median(out[:, 9]))
    jdump(R3 / "cheap/cheap.json", res)
    stamp("cheap", json.dumps(res))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("prep", "cheap"))
    ap.add_argument("--threads", type=int, required=True)
    a = ap.parse_args()
    {"prep": cmd_prep, "cheap": cmd_cheap}[a.cmd](a)


if __name__ == "__main__":
    main()
