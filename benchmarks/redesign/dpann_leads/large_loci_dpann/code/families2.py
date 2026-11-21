#!/usr/bin/env python
"""large_loci_dpann, rule v2 (predeclared.json amendment_2): reciprocal Z-list pairs, collinear chains between loci,
edge rule 1/3, top-30 sparsification, Louvain at six resolutions, selection by validation on odd-genome loci, gene
clusters, shuffled null. Reuses the list code of families.py (unchanged) and validate() of val_attrs.py.
Inputs: out/loci.parquet (job 120545), data/loci_val.parquet (val_attrs.py). Outputs out2/.
usage: families2.py --threads T
"""
import argparse
import json
import os
import time
from multiprocessing import Pool

import networkx as nx
import numba
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

import families as F1
from val_attrs import validate

W = F1.RD + "/dpann_leads/large_loci_dpann"
OUT = W + "/out2"
GAMMAS = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
TOPK, MINM, FRAC = 30, 5, 1 / 3
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:8.1f}s]", *a, flush=True)


@numba.njit(cache=False)
def lis_groups(starts, ends, pb):
    out = np.zeros(len(starts), np.int64)
    tails = np.empty(4096, np.int64)
    for gi in range(len(starts)):
        best = 0
        for sgn in (1, -1):
            n = 0
            for i in range(starts[gi], ends[gi]):
                x = sgn * pb[i]
                lo, hi = 0, n
                while lo < hi:
                    mid = (lo + hi) // 2
                    if tails[mid] < x:
                        lo = mid + 1
                    else:
                        hi = mid
                tails[lo] = x
                if lo == n:
                    n += 1
            if n > best:
                best = n
        out[gi] = best
    return out


def directed_pairs(qgenes, LP, LV, need, loc_of_gid):
    i_all = np.searchsorted(need, qgenes)
    assert (need[i_all] == qgenes).all()
    AA, BB = [], []
    for lo in range(0, len(qgenes), 100000):
        i = i_all[lo:lo + 100000]
        st, en = LP[i], LP[i + 1]
        c = en - st
        own = np.repeat(qgenes[lo:lo + len(i)], c)
        idx = np.repeat(st - np.r_[0, np.cumsum(c)[:-1]], c) + np.arange(int(c.sum()))
        part = LV[idx].astype(np.int64)
        ok = loc_of_gid[part] >= 0
        AA.append(own[ok])
        BB.append(part[ok])
    return np.concatenate(AA), np.concatenate(BB)


def reciprocal_edges(a, b, loc_of_gid, pos_of_gid, lsz, n):
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    key = lo * n + hi
    k, cnt = np.unique(key, return_counts=True)
    k = k[cnt == 2]
    u, v = k // n, k % n
    A, B = loc_of_gid[u], loc_of_gid[v]
    pu, pv = pos_of_gid[u], pos_of_gid[v]
    sw = A > B
    A2, B2 = np.where(sw, B, A), np.where(sw, A, B)
    pa, pb = np.where(sw, pv, pu), np.where(sw, pu, pv)
    gu, gv = np.where(sw, v, u), np.where(sw, u, v)
    nl = len(lsz)
    o = np.lexsort((pa, B2, A2))
    A2, B2, pa, pb, gu, gv = A2[o], B2[o], pa[o], pb[o], gu[o], gv[o]
    gk = A2 * nl + B2
    brk = np.flatnonzero(np.diff(gk)) + 1
    starts = np.r_[0, brk].astype(np.int64)
    ends = np.r_[brk, len(gk)].astype(np.int64)
    mcol = lis_groups(starts, ends, pb.astype(np.int64))
    E = pd.DataFrame(dict(A=A2[starts], B=B2[starts], m_rbh=ends - starts, m_col=mcol))
    E["sA"], E["sB"] = lsz[E.A.to_numpy()], lsz[E.B.to_numpy()]
    ok = (E.m_col >= MINM) & (E.m_col >= E.sA * FRAC) & (E.m_col >= E.sB * FRAC)
    E = E[ok].reset_index(drop=True)
    E["w"] = E.m_col / np.sqrt(E.sA * E.sB)
    return E, (gu, gv, A2, B2), int(len(k))


def sparsify(E, nl):
    d = pd.concat([E[["A", "B", "w"]].rename(columns={"A": "x", "B": "y"}).assign(i=np.arange(len(E))),
                   E[["A", "B", "w"]].rename(columns={"B": "x", "A": "y"}).assign(i=np.arange(len(E)))])
    d = d.sort_values(["x", "w"], ascending=[True, False])
    d["r"] = d.groupby("x").cumcount()
    keep = np.unique(d[d.r < TOPK].i.to_numpy())
    return E.iloc[keep].reset_index(drop=True)


def comps(E, nl):
    M = coo_matrix((np.ones(len(E)), (E.A.to_numpy(), E.B.to_numpy())), shape=(nl, nl))
    _, lab = connected_components(M, directed=False)
    return relabel(lab)


def relabel(lab):
    lab = np.asarray(lab)
    cs = np.bincount(lab[lab >= 0])
    fam = np.where((lab >= 0) & (cs[np.maximum(lab, 0)] >= 2), lab, -1)
    u, inv = np.unique(fam[fam >= 0], return_inverse=True)
    order = np.argsort(-np.bincount(inv), kind="stable")
    newid = np.empty(len(u), np.int64)
    newid[order] = np.arange(len(u))
    out = np.full(len(lab), -1, np.int64)
    out[fam >= 0] = newid[inv]
    return out


EG = {}


def louvain(g):
    Gx = EG["G"]
    cs = nx.community.louvain_communities(Gx, weight="weight", resolution=g, seed=17)
    lab = np.full(EG["nl"], -1, np.int64)
    for i, c in enumerate(cs):
        lab[list(c)] = i
    return g, relabel(lab)


def partitions(E, nl, threads, gammas):
    Gx = nx.Graph()
    Gx.add_weighted_edges_from(zip(E.A.tolist(), E.B.tolist(), E.w.tolist()))
    EG["G"], EG["nl"] = Gx, nl
    with Pool(min(threads, len(gammas))) as pool:
        res = pool.map(louvain, gammas)
    return dict(res)


def fam_stats(fam, lgroup):
    d = pd.DataFrame(dict(f=fam, g=lgroup))
    d = d[d.f >= 0]
    s = d.groupby("f").agg(n=("g", "size"), ng=("g", "nunique"))
    return dict(families=int(len(s)), loci_in_families=int(s.n.sum()), families_ge3_groups=int((s.ng >= 3).sum()),
                families_ge5_groups=int((s.ng >= 5).sum()), families_ge10_groups=int((s.ng >= 10).sum()),
                max_groups=int(s.ng.max()) if len(s) else 0, max_loci=int(s.n.max()) if len(s) else 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, required=True)
    a = ap.parse_args()
    th = a.threads
    os.makedirs(OUT, exist_ok=True)
    run = dict(threads=th, timing={})
    G = F1.G
    g = pd.read_parquet(F1.GENES, columns=["gid", "protein_id", "genome_code", "contig_code"])
    n = len(g)
    G["gc"] = gc = g.genome_code.to_numpy(np.int64)
    G["g_start"] = g_start = np.r_[0, np.flatnonzero(np.diff(gc)) + 1].astype(np.int64)
    g_end = np.r_[g_start[1:], n]
    lm = pd.read_parquet(F1.GROUPS, columns=["genome_code", "group"]).sort_values("genome_code")
    G["lin_of_genome"] = pd.factorize(lm.group.to_numpy())[0].astype(np.int64)
    G["lin"] = G["lin_of_genome"][gc]
    G["uq"] = np.load(F1.EMB + "/uid_of_row.npy").astype(np.int64)
    L = pd.read_parquet(W + "/out/loci.parquet").sort_values("locus").reset_index(drop=True)
    V = pd.read_parquet(W + "/data/loci_val.parquet")
    assert (V.locus.to_numpy() == L.locus.to_numpy()).all()
    nl = len(L)
    ls, le, lsz = L.s.to_numpy(np.int64), L.e.to_numpy(np.int64), L["size"].to_numpy(np.int64)
    lgen, lgroup = L.genome_code.to_numpy(), L.group.to_numpy()
    loc_of_gid = np.full(n, -1, np.int64)
    pos_of_gid = np.full(n, -1, np.int64)
    for i in range(nl):
        loc_of_gid[ls[i]:le[i] + 1] = i
        pos_of_gid[ls[i]:le[i] + 1] = np.arange(lsz[i])
    # null loci exactly as families.py (seed 17)
    rng = np.random.default_rng(17)
    null_genes = np.zeros(int(lsz.sum()), np.int64)
    null_loc = np.zeros(int(lsz.sum()), np.int64)
    null_pos = np.zeros(int(lsz.sum()), np.int64)
    pos = 0
    order = np.argsort(lgen, kind="stable")
    for gcode, idxs in pd.Series(order).groupby(lgen[order]):
        idxs = idxs.to_numpy()
        tot = int(lsz[idxs].sum())
        perm = rng.permutation(np.arange(g_start[gcode], g_end[gcode]))[:tot]
        o = 0
        for i in idxs:
            null_genes[pos:pos + lsz[i]] = perm[o:o + lsz[i]]
            null_loc[pos:pos + lsz[i]] = i
            null_pos[pos:pos + lsz[i]] = np.arange(lsz[i])
            pos += lsz[i]
            o += lsz[i]
    null_loc_of_gid = np.full(n, -1, np.int64)
    null_pos_of_gid = np.full(n, -1, np.int64)
    null_loc_of_gid[null_genes] = null_loc
    null_pos_of_gid[null_genes] = null_pos
    obs_genes = np.flatnonzero(loc_of_gid >= 0)
    need = np.unique(np.r_[obs_genes, null_genes])
    T_ = F1.RD + "/calib/table/"
    cnt = np.load(T_ + "counts.npy").astype(np.int64)
    G["zoff"] = np.r_[0, np.cumsum(cnt)]
    G["zG"] = np.memmap(T_ + "genome.bin", np.int16, "r")
    G["zL"] = np.memmap(T_ + "local.bin", np.int16, "r")
    G["zz"] = np.memmap(T_ + "z.bin", np.float16, "r")
    t = time.time()
    LP, LV = F1.build_lists(need, th)
    run["timing"]["lists"] = round(time.time() - t, 1)
    log("lists", len(need), len(LV))

    # observed
    t = time.time()
    da, db = directed_pairs(obs_genes, LP, LV, need, loc_of_gid)
    run["directed_pairs"] = int(len(da))
    E, (gu, gv, A2, B2), nrec = reciprocal_edges(da, db, loc_of_gid, pos_of_gid, lsz, n)
    del da, db
    run["reciprocal_pairs"] = nrec
    run["edges_v2"] = int(len(E))
    E.to_parquet(OUT + "/edges_v2.parquet", index=False)
    Es = sparsify(E, nl)
    run["edges_sparsified"] = int(len(Es))
    run["timing"]["edges"] = round(time.time() - t, 1)
    log("reciprocal pairs", nrec, "edges", len(E), "sparsified", len(Es))
    fams = {"comp": comps(E, nl), "comp_sparse": comps(Es, nl)}
    t = time.time()
    for gm, lab in partitions(Es, nl, th, GAMMAS).items():
        fams[f"g{gm:g}"] = lab
    run["timing"]["louvain"] = round(time.time() - t, 1)
    odd = (lgen % 2) == 1
    val, best = {}, None
    for nm, lab in fams.items():
        X = V.copy()
        X["p"] = lab
        val[nm] = dict(stats=fam_stats(lab, lgroup), all=validate(X, "p"), odd=validate(X, "p", odd),
                       even=validate(X, "p", ~odd))
        log(nm, val[nm]["stats"], "odd bars", val[nm]["odd"]["bars"])
    cands = [f"g{gm:g}" for gm in GAMMAS]
    passing = [c for c in cands if val[c]["odd"]["pass"]]
    if passing:
        sel, how = passing[0], "smallest gamma meeting all five bars on odd-genome loci"
    else:
        sel = max(cands, key=lambda c: (sum(val[c]["odd"]["bars"].values()), -cands.index(c)))
        how = "FAILURE: no gamma met all bars on odd loci; most bars met"
    run["selected"], run["selection"] = sel, how
    json.dump(val, open(OUT + "/validation_v2.json", "w"), indent=1, default=str)
    log("selected", sel, how)

    # gene clusters for the selected partition
    f = fams[sel]
    same = (f[A2] >= 0) & (f[A2] == f[B2])
    fam_genes = obs_genes[f[loc_of_gid[obs_genes]] >= 0]
    idx = pd.Series(np.arange(len(fam_genes)), index=fam_genes)
    ia, ib = idx.reindex(gu[same]).to_numpy().astype(np.int64), idx.reindex(gv[same]).to_numpy().astype(np.int64)
    M = coo_matrix((np.ones(len(ia)), (ia, ib)), shape=(len(fam_genes),) * 2)
    _, gcl = connected_components(M, directed=False)
    npart = np.bincount(np.r_[ia, ib], minlength=len(fam_genes))
    run["gene_links_selected"] = int(len(ia))
    gout = pd.DataFrame(dict(gid=obs_genes, locus=loc_of_gid[obs_genes]))
    gout["protein_id"] = g.protein_id.to_numpy()[obs_genes]
    gout = gout.merge(pd.DataFrame(dict(gid=fam_genes, gene_cluster=gcl, n_recip_partners=npart)), on="gid", how="left")
    gout.to_parquet(OUT + "/genes2.parquet", index=False)
    log("gene clusters", int(gcl.max()) + 1)

    # null
    t = time.time()
    na, nb = directed_pairs(null_genes, LP, LV, need, null_loc_of_gid)
    En, _, nrec_n = reciprocal_edges(na, nb, null_loc_of_gid, null_pos_of_gid, lsz, n)
    run["null_directed_pairs"] = int(len(na))
    run["null_reciprocal_pairs"] = nrec_n
    run["null_edges_v2"] = int(len(En))
    nf = comps(En, nl)
    run["null_stats_comp"] = fam_stats(nf, lgroup)
    run["timing"]["null"] = round(time.time() - t, 1)
    log("null", run["null_stats_comp"], "edges", len(En))

    Lo = L[["locus", "s", "e", "size", "genome_code", "group"]].copy()
    for nm, lab in fams.items():
        Lo["fam_" + nm] = lab
    Lo["family"] = fams[sel]
    deg = np.bincount(np.r_[E.A.to_numpy(), E.B.to_numpy()], minlength=nl)
    Lo["degree_v2"] = deg
    Lo.to_parquet(OUT + "/loci2.parquet", index=False)
    run["seconds"] = round(time.time() - T0, 1)
    json.dump(run, open(OUT + "/run2.json", "w"), indent=1, default=str)
    log("done", json.dumps(run, default=str)[:3000])


if __name__ == "__main__":
    main()
