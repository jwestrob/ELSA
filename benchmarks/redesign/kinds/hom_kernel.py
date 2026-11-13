"""Kind-recurrence pass with homology-aware withholding (coordinator-requested addition after ws_calib).

Same lists, support and realisation units as kinds.kind_pass with every other genome as reference; counts
0 A, 1 G, 2 F, 3 FX (4-7 left at 0). Withholding:
  excl_hom   drop a support pair if either member's MMseqs2 family is one of the query's two families or homologous
             to one of them (homology.hom: alignment CSR hp/hv, or shared Pfam family / clan / CDD superfamily codes
             cp/cv)
  mask       drop a support pair if either member's family is listed in mask[mask_id[q]]
"""
import numpy as np
from numba import njit, prange

from homology import hom


@njit(cache=True)
def _distinct(tmp, m, depths, row):
    if m == 0:
        return
    s = np.sort(tmp[:m])
    prev = -1
    for t in range(m):
        key = s[t] >> 10
        if key != prev:
            lev = s[t] & 1023
            for v in range(len(depths)):
                if lev <= depths[v]:
                    row[v] += 1
            prev = key


@njit(parallel=True, cache=True)
def kind_pass_hom(qa, qb, uq, I, Dmax, depths, mptr, members, gc, slot_of, gene_at, cid, tokM, tokF, TF,
                  excl_hom, mask, mask_id, hp, hv, cp, cv, U, out, overflow, nchunks, bufcap):
    nq = len(qa)
    n = len(gene_at)
    V = len(depths)
    for ch in prange(nchunks):
        lo = ch * nq // nchunks
        hi = (ch + 1) * nq // nchunks
        stamp = np.full(U, -1, np.int64)
        rk = np.zeros(U, np.int32)
        b_lev = np.empty(bufcap, np.int64)
        b_g = np.empty(bufcap, np.int64)
        b_kF = np.empty(bufcap, np.int64)
        b_fl = np.empty(bufcap, np.int8)
        tmp = np.empty(bufcap, np.int64)
        row = np.zeros(V, np.int64)
        for q in range(lo, hi):
            a = qa[q]
            b = qb[q]
            Ua = uq[a]
            Ub = uq[b]
            for v in range(V):
                for s_ in range(8):
                    out[q, v, s_] = 0
            if Ua < 0 or Ub < 0:
                continue
            stamp[Ub] = q
            rk[Ub] = 0
            for j in range(Dmax):
                x = I[Ub, j]
                if x < 0:
                    break
                if stamp[x] != q:
                    stamp[x] = q
                    rk[x] = j + 1
            ga = gc[a]
            fa = tokF[a]
            fb = tokF[b]
            ma = tokM[a]
            mb = tokM[b]
            mi = mask_id[q]
            k = 0
            for jj in range(Dmax + 1):
                if jj == 0:
                    Up = Ua
                    ra = 0
                else:
                    Up = I[Ua, jj - 1]
                    ra = jj
                    if Up < 0:
                        break
                for t in range(mptr[Up], mptr[Up + 1]):
                    a2 = members[t]
                    g2 = gc[a2]
                    if g2 == ga:
                        continue
                    s = slot_of[a2]
                    for dd in range(2):
                        s2 = s - 1 if dd == 0 else s + 1
                        if s2 < 0 or s2 >= n:
                            continue
                        if cid[s2] != cid[s]:
                            continue
                        b2 = gene_at[s2]
                        U2 = uq[b2]
                        if U2 < 0 or stamp[U2] != q:
                            continue
                        rb = rk[U2]
                        lev = ra if ra > rb else rb
                        m1 = tokM[a2]
                        m2 = tokM[b2]
                        if excl_hom:
                            if (hom(m1, ma, hp, hv, cp, cv) or hom(m1, mb, hp, hv, cp, cv)
                                    or hom(m2, ma, hp, hv, cp, cv) or hom(m2, mb, hp, hv, cp, cv)):
                                continue
                        if mi >= 0 and (mask[mi, m1] or mask[mi, m2]):
                            continue
                        if k >= bufcap:
                            overflow[q] = 1
                            continue
                        f1 = tokF[a2]
                        f2 = tokF[b2]
                        fl = 0
                        if f1 != fa and f1 != fb and f2 != fa and f2 != fb:
                            fl = 1
                        b_lev[k] = lev
                        b_g[k] = g2
                        b_kF[k] = f1 * TF + f2
                        b_fl[k] = fl
                        k += 1
            for t in range(k):
                for v in range(V):
                    if b_lev[t] <= depths[v]:
                        out[q, v, 0] += 1
            for st in range(1, 4):
                m = 0
                for t in range(k):
                    if st == 1:
                        key = b_g[t]
                    elif st == 2:
                        key = b_kF[t]
                    else:
                        if b_fl[t] == 0:
                            continue
                        key = b_kF[t]
                    tmp[m] = (key << 10) | b_lev[t]
                    m += 1
                for v in range(V):
                    row[v] = 0
                _distinct(tmp, m, depths, row)
                for v in range(V):
                    out[q, v, st] = row[v]
