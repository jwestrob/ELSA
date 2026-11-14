#!/usr/bin/env python
"""context_units evaluation (reads GTDB taxonomy; evaluation only, never used by a rule).

  evaluate.py taxa --variants raw,gc,member --threads T
      Per observed token pair with G >= 2 and P <= 1e-2: genera, families, orders, phyla carrying it; Spearman of
      N_eff (each W) with each count, and the partial Spearman given G; by G stratum. Recently transferred blocks:
      pairs with two genomes of different GTDB orders whose member proteins both have zca10 cosine >= 0.95 (first
      instance per genome, at most 100 genomes per pair, seed 17): N_eff vs genus count and significance under the
      genus rule (P <= 1e-4 & L >= 2) and the context rule of each scored configuration.
  evaluate.py contigs --tags TAG,... --W W,...
      Gold instances by contig length (< 2W + 2 genes vs the rest): DH F1 of TAG minus genus DH F1, even genomes,
      paired genome bootstrap.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit, prange
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ctxrec as C  # noqa: E402

B, BR, RL = C.B, C.BR, C.RL
CU = C.CU
log = C.log


def spearman(x, y):
    rx, ry = rankdata(x), rankdata(y)
    return float(np.corrcoef(rx, ry)[0, 1])


def partial_spearman(x, y, z):
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    rxy = np.corrcoef(rx, ry)[0, 1]
    rxz = np.corrcoef(rx, rz)[0, 1]
    ryz = np.corrcoef(ry, rz)[0, 1]
    return float((rxy - rxz * ryz) / math.sqrt(max((1 - rxz ** 2) * (1 - ryz ** 2), 1e-12)))


@njit(parallel=True, cache=True)
def transfer_scan(ptr, ga_gene, gb_gene, ordr, X, uid, thr, out_any, out_pairs):
    """Per pair: is there a genome pair in different orders whose member proteins both have cosine >= thr."""
    d = X.shape[1]
    for p in prange(len(ptr) - 1):
        s, e = ptr[p], ptr[p + 1]
        cnt = 0
        for i in range(s, e):
            for j in range(i + 1, e):
                if ordr[i] == ordr[j]:
                    continue
                qa, qa2 = uid[ga_gene[i]], uid[ga_gene[j]]
                qb, qb2 = uid[gb_gene[i]], uid[gb_gene[j]]
                c1 = 0.0
                c2 = 0.0
                for k in range(d):
                    c1 += X[qa, k] * X[qa2, k]
                    c2 += X[qb, k] * X[qb2, k]
                if c1 >= thr and c2 >= thr:
                    cnt += 1
        out_any[p] = cnt > 0
        out_pairs[p] = cnt


def cmd_taxa(a):
    import numba
    numba.set_num_threads(a.threads)
    t0 = time.time()
    u = RL.Universe(C.U, lineage_genome=True)
    t = pd.read_parquet(C.TOK)
    tok = np.full(u.n, -1, dtype=np.int64)
    tok[t.gid.to_numpy()] = t.token.to_numpy()
    key, left, dist, T = RL.colocalization(u, tok, C.W_ENGINE)
    st = RL.pair_statistics(u, tok, key, left, C.W_ENGINE, T)
    gm = pd.read_parquet(C.U / "genomes.parquet").sort_values("genome_code").reset_index(drop=True)
    ranks = {}
    for col in ("genus", "family", "ord", "phylum"):
        v = gm[col].where(gm[col].notna() & (gm[col] != ""), "unk:" + gm.genome)
        ranks[col] = pd.factorize(v)[0].astype(np.int64)
    k_of, g_of = st["incidence"]
    selp = np.flatnonzero((st["P"] <= 1e-2) & (st["G"] >= 2))
    sk = st["key"][selp]
    pos = np.searchsorted(sk, k_of)
    pos = np.minimum(pos, len(sk) - 1)
    inn = sk[pos] == k_of
    pi, gi = pos[inn], g_of[inn]
    counts = {}
    for col, arr in ranks.items():
        z = np.unique(pi * (int(arr.max()) + 1) + arr[gi])
        counts[col] = np.bincount(z // (int(arr.max()) + 1), minlength=len(selp))
    tab = pd.DataFrame(dict(key=sk, G=st["G"][selp], P=st["P"][selp], L_genus=counts["genus"], n_family=counts["family"],
                            n_order=counts["ord"], n_phylum=counts["phylum"]))
    out = dict(pairs=int(len(tab)), variants={})
    # recently transferred blocks: first instance per (pair, genome), at most 100 genomes per pair
    X, uid, _ = C.load_vectors(type("b", (), {"u": u})(), a.threads)
    pos_i = np.minimum(np.searchsorted(sk, key), len(sk) - 1)
    found = sk[pos_i] == key
    ii = np.flatnonzero(found)
    pidx = pos_i[found]
    gen = u.gc[left[ii]].astype(np.int64)
    o = np.lexsort((gen, pidx))
    ii, pidx, gen = ii[o], pidx[o], gen[o]
    first = np.r_[True, (pidx[1:] != pidx[:-1]) | (gen[1:] != gen[:-1])]
    ii, pidx, gen = ii[first], pidx[first], gen[first]
    rng = np.random.default_rng(17)
    rnd = rng.random(len(ii))
    o = np.lexsort((rnd, pidx))
    ii, pidx, gen = ii[o], pidx[o], gen[o]
    rank_in = np.arange(len(pidx)) - np.searchsorted(pidx, pidx)
    keep = rank_in < 100
    ii, pidx, gen = ii[keep], pidx[keep], gen[keep]
    ptr = np.searchsorted(pidx, np.arange(len(selp) + 1)).astype(np.int64)
    anyx = np.zeros(len(selp), dtype=np.bool_)
    npair = np.zeros(len(selp), dtype=np.int64)
    transfer_scan(ptr, left[ii], left[ii] + dist[ii], ranks["ord"][gen], X, uid, 0.95, anyx, npair)
    tab["recent_transfer"] = anyx
    tab["transfer_genome_pairs"] = npair
    del X
    log("transfer scan", int(anyx.sum()), "pairs", f"{time.time()-t0:.0f}s")
    sig_genus = (tab.P <= 1e-4) & (tab.L_genus >= 2)
    out["recent_transfer_pairs"] = int(tab.recent_transfer.sum())
    out["genus_rule_significant"] = dict(all=int(sig_genus.sum()), recent_transfer=int((sig_genus & tab.recent_transfer).sum()))
    for v in a.variants.split(","):
        f = CU / "ctx" / f"pairs_{v}.parquet"
        if not f.exists():
            continue
        pv = pd.read_parquet(f)
        assert np.array_equal(pv.key.to_numpy(), tab.key.to_numpy())
        res = {}
        for c in [c for c in pv.columns if c.startswith("neff_W")]:
            ne = pv[c].to_numpy()
            m = (tab.P <= 1e-4).to_numpy()
            r = dict(spearman={}, partial_given_G={}, by_G={})
            for col in ("L_genus", "n_family", "n_order", "n_phylum", "G"):
                r["spearman"][col] = spearman(ne[m], tab[col].to_numpy()[m])
                if col != "G":
                    r["partial_given_G"][col] = partial_spearman(ne[m], tab[col].to_numpy()[m], tab.G.to_numpy()[m])
            for lo_, hi_ in ((2, 3), (4, 10), (11, 50), (51, 100000)):
                mm = m & (tab.G >= lo_).to_numpy() & (tab.G <= hi_).to_numpy()
                if mm.sum() > 100:
                    r["by_G"][f"{lo_}-{hi_}"] = dict(n=int(mm.sum()), spearman_L_genus=spearman(ne[mm], tab.L_genus.to_numpy()[mm]),
                                                    median_neff=float(np.median(ne[mm])),
                                                    median_L_genus=float(np.median(tab.L_genus.to_numpy()[mm])),
                                                    median_n_family=float(np.median(tab.n_family.to_numpy()[mm])))
            rt = tab.recent_transfer.to_numpy() & m
            r["recent_transfer"] = dict(
                n=int(rt.sum()), median_neff=float(np.median(ne[rt])) if rt.any() else None,
                median_L_genus=float(np.median(tab.L_genus.to_numpy()[rt])) if rt.any() else None,
                median_G=float(np.median(tab.G.to_numpy()[rt])) if rt.any() else None,
                frac_neff_ge2=float((ne[rt] >= 2).mean()) if rt.any() else None,
                frac_Lgenus_ge2=float((tab.L_genus.to_numpy()[rt] >= 2).mean()) if rt.any() else None,
                median_neff_over_L=float(np.median(ne[rt] / tab.L_genus.to_numpy()[rt])) if rt.any() else None)
            nr = m & ~tab.recent_transfer.to_numpy()
            r["other_pairs"] = dict(n=int(nr.sum()), frac_neff_ge2=float((ne[nr] >= 2).mean()),
                                    frac_Lgenus_ge2=float((tab.L_genus.to_numpy()[nr] >= 2).mean()),
                                    median_neff_over_L=float(np.median(ne[nr] / tab.L_genus.to_numpy()[nr])))
            res[c] = r
            log(v, c, "spearman L_genus", round(r["spearman"]["L_genus"], 3), "partial|G", round(r["partial_given_G"]["L_genus"], 3),
                "transfer", r["recent_transfer"])
        out["variants"][v] = res
    # gold-spanning pairs by caller: pairs of member genes within 3 positions on one contig (keys of the observed pairs)
    mem = pd.read_parquet(C.PKG / "gold/members.parquet", columns=["instance_id", "caller", "gid"])
    gcode_even = (u.gc[mem.gid.to_numpy()] % 2 == 0)
    mem = mem.assign(even=gcode_even).sort_values(["instance_id", "gid"])
    rows = []
    for inst, g in mem.groupby("instance_id", sort=False):
        gids = g.gid.to_numpy()
        for i in range(len(gids)):
            for j in range(i + 1, len(gids)):
                if 0 < gids[j] - gids[i] <= 3:
                    rows.append((g.caller.iloc[0], gids[i], gids[j]))
    gp = pd.DataFrame(rows, columns=["caller", "a", "b"])
    cidx = np.repeat(np.arange(len(u.c_start)), u.c_end - u.c_start)
    gp = gp[cidx[gp.a.to_numpy()] == cidx[gp.b.to_numpy()]]
    ta, tb = tok[gp.a.to_numpy()], tok[gp.b.to_numpy()]
    okt = (ta >= 0) & (tb >= 0) & (ta != tb)
    gp = gp[okt]
    gk = np.minimum(ta[okt], tb[okt]) * T + np.maximum(ta[okt], tb[okt])
    pos_g = np.minimum(np.searchsorted(sk, gk), len(sk) - 1)
    gp = gp.assign(row=np.where(sk[pos_g] == gk, pos_g, -1))
    gp = gp[gp.row >= 0]
    out["gold_pairs"] = {}
    for v in a.variants.split(","):
        f = CU / "ctx" / f"pairs_{v}.parquet"
        if not f.exists():
            continue
        pv = pd.read_parquet(f)
        for c in [c for c in pv.columns if c.startswith("neff_W")]:
            ne = pv[c].to_numpy()
            for t_ in (2.0,):
                res_c = {}
                for caller, g in gp.groupby("caller"):
                    rr = g.row.to_numpy()
                    sigG = (tab.P.to_numpy()[rr] <= 1e-4) & (tab.L_genus.to_numpy()[rr] >= 2)
                    sigN = (tab.P.to_numpy()[rr] <= 1e-4) & (ne[rr] >= t_)
                    lost = sigG & ~sigN
                    res_c[caller] = dict(gold_pairs=int(len(rr)), sig_genus=int(sigG.sum()), sig_neff=int(sigN.sum()),
                                         lost=int(lost.sum()), gained=int((sigN & ~sigG).sum()),
                                         lost_median_G=float(np.median(tab.G.to_numpy()[rr][lost])) if lost.any() else None,
                                         lost_median_L=float(np.median(tab.L_genus.to_numpy()[rr][lost])) if lost.any() else None,
                                         lost_median_neff=float(np.median(ne[rr][lost])) if lost.any() else None,
                                         lost_frac_G2=float((tab.G.to_numpy()[rr][lost] == 2).mean()) if lost.any() else None,
                                         lost_frac_recent_transfer=float(tab.recent_transfer.to_numpy()[rr][lost].mean()) if lost.any() else None,
                                         kept_frac_recent_transfer=float(tab.recent_transfer.to_numpy()[rr][sigG & sigN].mean()) if (sigG & sigN).any() else None)
                out["gold_pairs"][f"{v}:{c}:t{t_:g}"] = res_c
                log("gold pairs", v, c, {k[:4]: (x["sig_genus"], x["sig_neff"], x["lost"], x["lost_frac_G2"]) for k, x in res_c.items()})
    tab.to_parquet(CU / "ctx" / "pairs_taxa.parquet", index=False)
    C.jdump(CU / "results" / "evaluate_taxa.json", out)
    log("done", f"{time.time()-t0:.0f}s")


def cmd_contigs(a):
    gold = pd.read_parquet(C.PKG / "gold/instances.parquet", columns=["instance_id", "contig_code"])
    contigs = pd.read_parquet(C.U / "contigs.parquet", columns=["contig_code", "n_genes"])
    gold = gold.merge(contigs, on="contig_code", how="left").set_index("instance_id")
    gcode = B.odd_even_code()
    ref = pd.read_parquet(C.REF_SCORES / f"{C.REF_NAME['DH']}.parquet")
    ref["excess"] = ref.f1 - ref.control_mean_f1
    ref = ref.set_index("instance_id")
    out = {}
    for tag, W in zip(a.tags.split(","), [int(x) for x in a.W.split(",")]):
        per = pd.read_parquet(CU / "scores" / f"{tag}__DH.parquet").set_index("instance_id")
        ng = gold.n_genes.reindex(per.index).to_numpy()
        r = {}
        for lab, m in (("short", ng < 2 * W + 2), ("long", ng >= 2 * W + 2)):
            ids = per.index[m]
            r[lab] = dict(n_instances=int(m.sum()), paired=BR.paired(per.loc[ids], ref, gcode, "even"))
        out[tag] = r
        log(tag, {k: (v["n_instances"], [round(x, 4) for x in v["paired"]["summed"]],
                      {c[:4]: round(v["paired"][c]["d_f1"][0], 4) for c in C.CALLERS}) for k, v in r.items()})
    C.jdump(CU / "results" / "evaluate_contigs.json", out)


RELS = ("same_species", "same_genus", "same_family", "same_order", "same_class", "same_phylum", "different_phylum")


def cmd_decay(a):
    """Context similarity of two instances of the same token pair in different genomes, by their GTDB relation:
    raw and gc contexts at each W, family-centred member vectors, and the raw member cosine (mean of the two genes').
    Answers at which taxonomic distance the context similarity reaches the noise floor (evaluation only)."""
    import numba
    numba.set_num_threads(a.threads)
    t0 = time.time()
    u = RL.Universe(C.U, lineage_genome=True)
    t = pd.read_parquet(C.TOK)
    tok = np.full(u.n, -1, dtype=np.int64)
    tok[t.gid.to_numpy()] = t.token.to_numpy()
    key, left, dist, T = RL.colocalization(u, tok, C.W_ENGINE)
    cid = np.repeat(np.arange(len(u.c_start)), u.c_end - u.c_start).astype(np.int64)
    gen = u.gc[left].astype(np.int64)
    o = np.argsort(key, kind="stable")
    ks = key[o]
    starts = np.r_[0, np.flatnonzero(np.diff(ks)) + 1]
    ends = np.r_[starts[1:], len(ks)]
    size = ends - starts
    grp = np.repeat(np.arange(len(starts)), size)
    rng = np.random.default_rng(17)
    cand = np.flatnonzero(size[grp] >= 2)
    pick = rng.choice(cand, size=min(6_000_000, len(cand)), replace=False)
    other = starts[grp[pick]] + (rng.random(len(pick)) * size[grp[pick]]).astype(np.int64)
    ia, ib = o[pick], o[other]
    ok = gen[ia] != gen[ib]
    ia, ib = ia[ok], ib[ok]
    gm = pd.read_parquet(C.U / "genomes.parquet").sort_values("genome_code").reset_index(drop=True)
    rel = np.full(len(ia), len(RELS) - 1, dtype=np.int64)
    ga, gb = gen[ia], gen[ib]
    for r, col in reversed(list(enumerate(("species", "genus", "family", "ord", "class", "phylum")))):
        v = gm[col].where(gm[col].notna() & (gm[col] != ""), "unk:" + gm.genome).to_numpy()
        same = v[ga] == v[gb]
        rel[same] = np.minimum(rel[same], r)
    # cap 200,000 samples per relation
    keep = np.zeros(len(ia), bool)
    for r in range(len(RELS)):
        idx = np.flatnonzero(rel == r)
        keep[rng.permutation(idx)[:200_000]] = True
    ia, ib, rel = ia[keep], ib[keep], rel[keep]
    log("decay samples", {RELS[r]: int((rel == r).sum()) for r in range(len(RELS))})
    X, uid, _ = C.load_vectors(type("b", (), {"u": u})(), a.threads)
    nc = len(u.c_start)
    d = X.shape[1]
    P = np.empty((u.n + nc, d), dtype=np.float32)
    C.build_prefix(X, uid, u.c_start, u.c_end, P)
    mu_g = np.zeros((u.ng, d))
    cont_sum = P[u.c_end + np.arange(nc)].astype(np.float64) - P[u.c_start + np.arange(nc)].astype(np.float64)
    np.add.at(mu_g, u.gc[u.c_start], cont_sum)
    del cont_sum
    mu_g /= np.bincount(u.gc, minlength=u.ng)[:, None]
    mu_tok = C.token_means(tok, uid, X, T)
    lo, hi = left, left + dist
    Ws = np.array([10, 25, 50, 100], np.int64)
    z1 = np.zeros(1, np.int64)
    out = dict(samples={RELS[r]: int((rel == r).sum()) for r in range(len(RELS))}, variants={})
    for name, mode, center, Wv in (("raw", 0, False, Ws), ("gc", 0, True, Ws), ("member", 1, False, np.array([0], np.int64))):
        od = np.empty((len(ia), len(Wv)))
        on = np.empty((len(ia), len(Wv)), dtype=np.int64)
        C.floor_sample(ia, ib, mode, Wv, P, u.c_start, u.c_end, lo, hi, cid[lo], gen, mu_g, center, X, uid, lo, hi,
                       tok[lo] if mode == 1 else z1, tok[hi] if mode == 1 else z1, mu_tok, od, on)
        res = {}
        for wi, W in enumerate(Wv):
            res[int(W)] = {RELS[r]: dict(mean=float(np.nanmean(od[rel == r, wi])),
                                         q25_q50_q75=np.nanquantile(od[rel == r, wi], [.25, .5, .75]).tolist())
                           for r in range(len(RELS))}
        out["variants"][name] = res
        log(name, {int(W): [round(res[int(W)][x]["mean"], 3) for x in RELS] for W in Wv})
    # raw member cosine (mean of the two genes' cosines)
    xa = np.empty(len(ia))
    xb = np.empty(len(ia))
    same_orient = tok[lo[ia]] == tok[lo[ib]]          # align the two instances by token (pairs are unordered)
    pa = np.where(same_orient, lo[ib], hi[ib])
    pb = np.where(same_orient, hi[ib], lo[ib])
    C.gene_pair_cos(lo[ia], pa, X, uid, mu_g, gen, False, xa)
    C.gene_pair_cos(hi[ia], pb, X, uid, mu_g, gen, False, xb)
    mc = (xa + xb) / 2
    out["variants"]["member_raw_cosine"] = {RELS[r]: dict(mean=float(np.nanmean(mc[rel == r])),
                                                          q25_q50_q75=np.nanquantile(mc[rel == r], [.25, .5, .75]).tolist())
                                            for r in range(len(RELS))}
    log("member_raw_cosine", [round(out["variants"]["member_raw_cosine"][x]["mean"], 3) for x in RELS])
    C.jdump(CU / "results" / "evaluate_decay.json", out)
    log("done", f"{time.time()-t0:.0f}s")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("taxa", "contigs", "decay"))
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--variants", default="raw,gc,member")
    ap.add_argument("--tags", default="")
    ap.add_argument("--W", default="")
    a = ap.parse_args()
    {"taxa": cmd_taxa, "contigs": cmd_contigs, "decay": cmd_decay}[a.cmd](a)


if __name__ == "__main__":
    main()
