#!/usr/bin/env python
"""large_loci_dpann: large outermost loci of the frozen DPANN Z L4b+ theta .02 tree, grouped into locus families by the
method's own Z-list gene correspondences (predeclared.json). Biotite, read-only on every input.

Steps
  1. outermost loci (= L1 runs), large = >= 10 genes; nested-node counts; whole-locus node test (root in L4b_t0.02)
  2. Z lists (z >= 10.6, own genome group excluded, best gene per genome; z_chunk copied from slot_rescue/code/score.py)
  3. m(A->B) = genes of A whose list holds a gene of B; edge rule at fraction 1/3 (primary), 1/4, 1/2; components
  4. null: gene identities shuffled within genomes (each locus keeps size + genome), same rule, same components
  5. gene clusters inside primary families (Z links whose partner is in another member locus of the same family)
Outputs in OUT: loci.parquet, genes.parquet, edges.parquet, null_loci.parquet, run.json

usage: families.py --threads T [--smoke N]
"""
import argparse
import json
import os
import resource
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

RD = "/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111"
EMB = "/groups/banfield/projects/environmental/sr/srvp2020/Jacob/embed_queue/runs/dpann__esmpp_small__20261111"
GROUPS = RD + "/lineagefree/groups/emb_hom_average_t0.75.parquet"
GENES = RD + "/ws_dpann/package/universe/genes.parquet"
LOCI = RD + "/long_loci/loci/dpann_Z/"
NODES = RD + "/long_loci/nodes/dpann_Z/obs4b_knn_t0.02.parquet"
OUT = RD + "/dpann_leads/large_loci_dpann/out"
ZCUT, MINSIZE, MINM = 10.6, 10, 5
FRACS = {"f33": 1 / 3, "f25": 1 / 4, "f50": 1 / 2}
G = {}
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:8.1f}s]", *a, flush=True)


def rss():
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024, 2)


def z_chunk(genes):
    """copied from slot_rescue/code/score.py (unchanged logic)"""
    cnt_off, zG, zL, zz = G["zoff"], G["zG"], G["zL"], G["zz"]
    uq, lin, lin_g, g_start = G["uq"], G["lin"], G["lin_of_genome"], G["g_start"]
    ptr, out = [0], []
    for g in genes:
        u = uq[g]
        if u >= 0:
            s, e = cnt_off[u], cnt_off[u + 1]
            z = np.asarray(zz[s:e]).astype(np.float32)
            gg = np.asarray(zG[s:e]).astype(np.int64)
            keep = np.flatnonzero((z >= ZCUT) & (lin_g[gg] != lin[g]))
            o = keep[np.argsort(-z[keep], kind="mergesort")]
            p = g_start[gg[o]] + np.asarray(zL[s:e]).astype(np.int64)[o]
            out.append(p.astype(np.int32))
            ptr.append(ptr[-1] + len(p))
        else:
            ptr.append(ptr[-1])
    return np.array(ptr[1:], np.int64), (np.concatenate(out) if out else np.zeros(0, np.int32))


def build_lists(need, threads):
    chunks = np.array_split(need, max(1, threads * 8))
    with Pool(threads) as pool:
        res = pool.map(z_chunk, chunks)
    ptr, vals, base = [np.zeros(1, np.int64)], [], 0
    for p, v in res:
        ptr.append(p + base)
        base += len(v)
        vals.append(v)
    return np.concatenate(ptr), np.concatenate(vals)


def pair_counts(qgenes, qloc, LP, LV, need, loc_of_gid, nl):
    """m(A->B) for every (A, B): genes a (locus qloc) whose list holds a gene of locus B. Returns keys A*nl+B, counts."""
    i_all = np.searchsorted(need, qgenes)
    assert (need[i_all] == qgenes).all()
    K, OWN, PART, AA, BB = [], [], [], [], []
    for lo in range(0, len(qgenes), 100000):
        i = i_all[lo:lo + 100000]
        st, en = LP[i], LP[i + 1]
        c = en - st
        own = np.repeat(np.arange(lo, lo + len(i)), c)
        idx = np.repeat(st - np.r_[0, np.cumsum(c)[:-1]], c) + np.arange(int(c.sum()))
        part = LV[idx].astype(np.int64)
        B = loc_of_gid[part]
        ok = B >= 0
        own, part, B = own[ok], part[ok], B[ok]
        A = qloc[own].astype(np.int64)
        # one partner per genome per gene: (gene, B) is unique by construction; dedupe anyway
        gk = own.astype(np.int64) * nl + B
        _, first = np.unique(gk, return_index=True)
        K.append(A[first] * nl + B[first])
        OWN.append(own)
        PART.append(part)
        AA.append(A)
        BB.append(B)
    k, cnt = np.unique(np.concatenate(K), return_counts=True)
    return k, cnt, (np.concatenate(OWN), np.concatenate(PART), np.concatenate(AA), np.concatenate(BB))


def edges_from_counts(k, cnt, size, nl):
    A, B = k // nl, k % nl
    rev = B * nl + A
    j = np.searchsorted(k, rev)
    j = np.minimum(j, len(k) - 1)
    has = k[j] == rev
    mBA = np.where(has, cnt[j], 0)
    keep = A < B
    E = pd.DataFrame(dict(A=A[keep], B=B[keep], mAB=cnt[keep], mBA=mBA[keep]))
    E["sA"], E["sB"] = size[E.A.to_numpy()], size[E.B.to_numpy()]
    # pairs where only B->A counts exist (A<B with no A->B entry) are covered by the reverse key with A>B: add them
    keep2 = (A > B) & ~has
    if keep2.any():
        E2 = pd.DataFrame(dict(A=B[keep2], B=A[keep2], mAB=np.zeros(int(keep2.sum()), np.int64), mBA=cnt[keep2]))
        E2["sA"], E2["sB"] = size[E2.A.to_numpy()], size[E2.B.to_numpy()]
        E = pd.concat([E, E2], ignore_index=True)
    return E


def components(E, nl, frac):
    ok = (E.mAB >= MINM) & (E.mBA >= MINM) & (E.mAB >= E.sA * frac) & (E.mBA >= E.sB * frac)
    e = E[ok]
    M = coo_matrix((np.ones(len(e)), (e.A.to_numpy(), e.B.to_numpy())), shape=(nl, nl))
    ncomp, lab = connected_components(M, directed=False)
    deg = np.bincount(np.r_[e.A.to_numpy(), e.B.to_numpy()], minlength=nl)
    csize = np.bincount(lab, minlength=ncomp)
    fam = np.where(csize[lab] >= 2, lab, -1)
    # renumber families by size (largest first)
    u, inv = np.unique(fam[fam >= 0], return_inverse=True)
    order = np.argsort(-np.bincount(inv), kind="stable")
    newid = np.empty(len(u), np.int64)
    newid[order] = np.arange(len(u))
    out = np.full(nl, -1, np.int64)
    out[fam >= 0] = newid[inv]
    return out, deg, int(ok.sum())


def fam_stats(fam, lgroup):
    d = pd.DataFrame(dict(f=fam, g=lgroup))
    d = d[d.f >= 0]
    s = d.groupby("f").agg(n=("g", "size"), ng=("g", "nunique"))
    return dict(families=int(len(s)), loci_in_families=int(s.n.sum()),
                families_ge3_groups=int((s.ng >= 3).sum()), families_ge5_groups=int((s.ng >= 5).sum()),
                families_ge10_groups=int((s.ng >= 10).sum()), max_groups=int(s.ng.max()) if len(s) else 0,
                max_loci=int(s.n.max()) if len(s) else 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, required=True)
    ap.add_argument("--smoke", type=int, default=0)
    a = ap.parse_args()
    th = a.threads
    os.makedirs(OUT, exist_ok=True)
    run = dict(threads=th, smoke=a.smoke, timing={})

    g = pd.read_parquet(GENES, columns=["gid", "protein_id", "genome_code", "contig_code", "position_index", "length"])
    assert (g.gid.to_numpy() == np.arange(len(g))).all()
    n = len(g)
    G["gc"] = gc = g.genome_code.to_numpy(np.int64)
    cid = g.contig_code.to_numpy(np.int64)
    assert (np.diff(gc) >= 0).all() and (np.diff(cid) >= 0).all()
    brk = np.flatnonzero(np.diff(cid)) + 1
    cstart = np.r_[0, brk].astype(np.int64)
    cend = np.r_[brk, n].astype(np.int64) - 1
    G["g_start"] = g_start = np.r_[0, np.flatnonzero(np.diff(gc)) + 1].astype(np.int64)
    g_end = np.r_[g_start[1:], n]
    lm = pd.read_parquet(GROUPS, columns=["genome_code", "group"]).sort_values("genome_code")
    assert np.array_equal(lm.genome_code.to_numpy(), np.arange(len(g_start)))
    G["lin_of_genome"] = pd.factorize(lm.group.to_numpy())[0].astype(np.int64)
    G["lin"] = G["lin_of_genome"][gc]
    uq = np.load(EMB + "/uid_of_row.npy").astype(np.int64)
    assert len(uq) == n
    G["uq"] = uq
    log("universe", n, "genomes", len(g_start), "groups", int(G["lin_of_genome"].max()) + 1)

    # ---- 1. loci
    L = pd.read_parquet(LOCI + "L4bp_t0.02.parquet").sort_values(["s", "e"], ascending=[True, False]).reset_index(drop=True)
    s_all, e_all = L.s.to_numpy(np.int64), L.e.to_numpy(np.int64)
    prev = np.r_[-1, np.maximum.accumulate(e_all)[:-1]]
    outer = e_all > prev
    O = L[outer].reset_index(drop=True)
    O["size"] = O.e - O.s + 1
    L1 = pd.read_parquet(LOCI + "L1.parquet")
    run["outermost"] = int(len(O))
    run["outermost_equal_L1"] = int(len(O.merge(L1, on=["s", "e"])))
    sizes = O["size"].to_numpy()
    run["size_bins"] = {f"{lo}-{hi}": int(((sizes >= lo) & (sizes <= hi)).sum()) for lo, hi in
                        [(2, 2), (3, 5), (6, 9), (10, 14), (15, 24), (25, 49), (50, 10 ** 6)]}
    Lg = O[O["size"] >= MINSIZE].reset_index(drop=True)
    nl = len(Lg)
    ls, le, lsz = Lg.s.to_numpy(np.int64), Lg.e.to_numpy(np.int64), Lg["size"].to_numpy(np.int64)
    assert (cid[ls] == cid[le]).all()
    lgen = gc[ls]
    lgroup = G["lin_of_genome"][lgen]
    loc_of_gid = np.full(n, -1, np.int64)
    for i in range(nl):
        loc_of_gid[ls[i]:le[i] + 1] = i
    # nested nodes >= 10 inside each large outermost locus
    szall = e_all - s_all + 1
    nest = np.flatnonzero((~outer) & (szall >= MINSIZE))
    host = loc_of_gid[s_all[nest]]
    assert (host >= 0).all() and (loc_of_gid[e_all[nest]] == host).all()
    n_nested = np.bincount(host, minlength=nl)
    max_nested = np.zeros(nl, np.int64)
    np.maximum.at(max_nested, host, szall[nest])
    # whole-locus node test: root interval present in L4b_t0.02 (n >= 2, Lambda >= tau)
    L4b = pd.read_parquet(LOCI + "L4b_t0.02.parquet")
    kept = set(zip(L4b.s.to_numpy().tolist(), L4b.e.to_numpy().tolist()))
    root_l4b = np.array([(int(x), int(y)) in kept for x, y in zip(ls, le)])
    N = pd.read_parquet(NODES, columns=["lo", "hi", "n", "E", "lam", "nI"])
    N = N.drop_duplicates(["lo", "hi"]).rename(columns={"lo": "s", "hi": "e", "n": "node_n", "E": "node_E", "lam": "node_lam"})
    Lg = Lg.merge(N[["s", "e", "node_n", "node_E", "node_lam"]], on=["s", "e"], how="left")
    assert len(Lg) == nl
    run["large_loci"] = nl
    run["large_genes"] = int(lsz.sum())
    run["large_with_nested_ge10"] = int((n_nested > 0).sum())
    run["nested_ge10_inside_large"] = int(len(nest))
    run["large_root_passes_L4b"] = int(root_l4b.sum())
    log("large loci", nl, "genes", int(lsz.sum()), "root in L4b", int(root_l4b.sum()))

    # ---- null loci: gene identities shuffled within genomes (seed 17)
    rng = np.random.default_rng(17)
    null_genes = np.zeros(int(lsz.sum()), np.int64)
    null_loc = np.zeros(int(lsz.sum()), np.int64)
    pos = 0
    order = np.argsort(lgen, kind="stable")
    for gcode, idxs in pd.Series(order).groupby(lgen[order]):
        idxs = idxs.to_numpy()
        tot = int(lsz[idxs].sum())
        perm = rng.permutation(np.arange(g_start[gcode], g_end[gcode]))[:tot]
        assert len(perm) == tot
        o = 0
        for i in idxs:
            null_genes[pos:pos + lsz[i]] = perm[o:o + lsz[i]]
            null_loc[pos:pos + lsz[i]] = i
            pos += lsz[i]
            o += lsz[i]
    null_loc_of_gid = np.full(n, -1, np.int64)
    null_loc_of_gid[null_genes] = null_loc
    assert (np.bincount(null_loc, minlength=nl) == lsz).all()

    obs_genes = np.flatnonzero(loc_of_gid >= 0)
    obs_loc = loc_of_gid[obs_genes]
    if a.smoke:
        keepL = np.zeros(nl, bool)
        keepL[: a.smoke] = True
        m1 = keepL[obs_loc]
        m2 = keepL[null_loc]
        obs_genes, obs_loc = obs_genes[m1], obs_loc[m1]
        null_genes, null_loc = null_genes[m2], null_loc[m2]
    need = np.unique(np.r_[obs_genes, null_genes])

    # ---- 2. Z lists
    t = time.time()
    T_ = RD + "/calib/table/"
    cnt = np.load(T_ + "counts.npy").astype(np.int64)
    G["zoff"] = np.r_[0, np.cumsum(cnt)]
    G["zG"] = np.memmap(T_ + "genome.bin", np.int16, "r")
    G["zL"] = np.memmap(T_ + "local.bin", np.int16, "r")
    G["zz"] = np.memmap(T_ + "z.bin", np.float16, "r")
    LP, LV = build_lists(need, th)
    run["timing"]["lists"] = round(time.time() - t, 1)
    ll = np.diff(LP)
    run["lists_built"] = int(len(need))
    run["list_len_q10_q50_q90"] = np.quantile(ll, [.1, .5, .9]).tolist()
    run["genes_without_uid"] = int((uq[need] < 0).sum())
    log("lists", len(need), "entries", len(LV), "rss", rss())

    # ---- 3. observed counts, edges, families
    t = time.time()
    k, c, (own, part, A, B) = pair_counts(obs_genes, obs_loc, LP, LV, need, loc_of_gid, nl)
    E = edges_from_counts(k, c, lsz, nl)
    fams = {}
    for nm, fr in FRACS.items():
        fams[nm], deg, ne = components(E, nl, fr)
        run[f"edges_{nm}"] = ne
        run[f"stats_{nm}"] = fam_stats(fams[nm], lgroup)
        if nm == "f33":
            deg33 = deg
    run["timing"]["observed"] = round(time.time() - t, 1)
    log("observed", run["stats_f33"])
    E25 = E[(E.mAB >= MINM) & (E.mBA >= MINM) & (E.mAB >= E.sA / 4) & (E.mBA >= E.sB / 4)]
    E25.to_parquet(OUT + "/edges.parquet", index=False)

    # ---- 5. gene clusters inside primary families
    t = time.time()
    f = fams["f33"]
    a_gid = obs_genes[own]
    same = (f[A] >= 0) & (f[A] == f[B])
    ga, gb = a_gid[same], part[same]
    fam_genes = obs_genes[f[obs_loc] >= 0]
    idx = pd.Series(np.arange(len(fam_genes)), index=fam_genes)
    ia, ib = idx.reindex(ga).to_numpy(), idx.reindex(gb).to_numpy()
    assert not np.isnan(ia).any() and not np.isnan(ib).any()
    M = coo_matrix((np.ones(len(ia)), (ia.astype(np.int64), ib.astype(np.int64))), shape=(len(fam_genes),) * 2)
    _, gcl = connected_components(M, directed=False)
    npart = np.bincount(ia.astype(np.int64), minlength=len(fam_genes))
    run["gene_links_in_families"] = int(len(ia))
    run["timing"]["gene_clusters"] = round(time.time() - t, 1)
    log("gene clusters", int(gcl.max()) + 1, "links", len(ia))

    genes_out = pd.DataFrame(dict(gid=obs_genes, locus=obs_loc))
    genes_out["protein_id"] = g.protein_id.to_numpy()[obs_genes]
    gi = pd.DataFrame(dict(gid=fam_genes, gene_cluster=gcl, n_family_partners=npart))
    genes_out = genes_out.merge(gi, on="gid", how="left")
    genes_out["list_len"] = ll[np.searchsorted(need, obs_genes)]
    genes_out.to_parquet(OUT + "/genes.parquet", index=False)

    # ---- 4. null
    t = time.time()
    kn, cn, _ = pair_counts(null_genes, null_loc, LP, LV, need, null_loc_of_gid, nl)
    En = edges_from_counts(kn, cn, lsz, nl)
    nullfam = {}
    for nm, fr in FRACS.items():
        nullfam[nm], _, ne = components(En, nl, fr)
        run[f"null_edges_{nm}"] = ne
        run[f"null_stats_{nm}"] = fam_stats(nullfam[nm], lgroup)
    run["timing"]["null"] = round(time.time() - t, 1)
    log("null", run["null_stats_f33"])
    pd.DataFrame(dict(locus=np.arange(nl), null_f33=nullfam["f33"], null_f25=nullfam["f25"],
                      null_f50=nullfam["f50"])).to_parquet(OUT + "/null_loci.parquet", index=False)

    # ---- loci table
    Lg["locus"] = np.arange(nl)
    Lg["genome_code"] = lgen
    Lg["group"] = lgroup
    Lg["contig_code"] = cid[ls]
    Lg["contig_genes"] = (cend[cid[ls]] - cstart[cid[ls]] + 1)
    Lg["flank_left"] = ls - cstart[cid[ls]]
    Lg["flank_right"] = cend[cid[ls]] - le
    Lg["n_nested_ge10"] = n_nested
    Lg["max_nested"] = max_nested
    Lg["root_in_L4b"] = root_l4b
    for nm in FRACS:
        Lg["fam_" + nm] = fams[nm]
    Lg["degree_f33"] = deg33
    Lg.to_parquet(OUT + "/loci.parquet", index=False)
    run["rss_gb"] = rss()
    run["seconds"] = round(time.time() - T0, 1)
    json.dump(run, open(OUT + f"/run{'_smoke' if a.smoke else ''}.json", "w"), indent=1)
    log("done", json.dumps(run))


if __name__ == "__main__":
    main()
