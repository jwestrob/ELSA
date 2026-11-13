#!/usr/bin/env python
"""long_loci part 4: family bonds at other pair distances (predeclared4.json).

  W5 / W8  learner round 0 with occurrences at distance 1..w and E with that w (ll.bonds with w)
  WA       adaptive: an occurrence at distance d is tested at window d (n_d over occurrences at distance <= d, E_d with
           w = d); bond if n_d >= 2 and Lambda_d >= tau
  W3       ladder only (check of the bond rule; part-1 outputs are the W3 reference rows)
tau = smallest ladder value with bond FDR (mean significant occurrences over 3 token shuffles / observed) <= 0.01.
Outputs per arm: L1, L4b_t{theta} (part-1 L4b: token cohesion orders splits, closed-form family node test at the arm's
tau), L4bp_t{theta}; accepted bonds by distance.
usage: lw.py build --data dpann|sd --threads T
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ll  # noqa: E402
import words_ts as W  # noqa: E402

LL = ll.LL
LADDER = (4.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0)
ARMS = ("W3", "W5", "W8", "WA")
log = ll.log


def setup(tok, gc, grp, min_n=2):
    """Labels and the per-genome symbol counts used by expected_groups (as ll.bonds / the learner's round 0)."""
    n_genomes = int(gc.max()) + 1
    ng = int(grp.max()) + 1
    order = np.lexsort((np.arange(n_genomes), grp))
    rank_of_genome = np.empty(n_genomes, dtype=np.int64)
    rank_of_genome[order] = np.arange(n_genomes)
    grp_of_rank = grp[order].astype(np.int64)
    T = int(tok.max()) + 1
    tg = np.unique(tok * ng + grp[gc])
    tok_groups = np.bincount(tg // ng, minlength=T)
    sym = np.where(tok_groups[tok] >= min_n, tok, -1).astype(np.int64)
    live = sym >= 0
    sr = np.unique(sym[live] * n_genomes + rank_of_genome[gc[live]], return_counts=True)
    s_of, r_of = sr[0] // n_genomes, sr[0] % n_genomes
    indptr = np.searchsorted(s_of, np.arange(T + 1)).astype(np.int64)
    m_rank = np.bincount(rank_of_genome[gc], minlength=n_genomes).astype(np.float64)
    return dict(sym=sym, ng=ng, indptr=indptr, r_of=r_of.astype(np.int64), cnt=sr[1].astype(np.float64), m_rank=m_rank,
                grp_of_rank=grp_of_rank)


def bonds_adaptive(tok, cc, gc, grp, tau, wmax=8):
    st = setup(tok, gc, grp)
    sym, ng = st["sym"], st["ng"]
    n = len(tok)
    k, d, key = W.occurrences(np.arange(n, dtype=np.int64), cc, sym, np.ones(n, dtype=np.bool_), wmax)
    ukey, kinv = np.unique(key, return_inverse=True)
    g_occ = grp[gc[k]]
    occ_lam = np.zeros(len(k))
    for dd in range(1, wmax + 1):
        m = d <= dd
        kg = np.unique(kinv[m] * ng + g_occ[m])
        nG = np.bincount(kg // ng, minlength=len(ukey))
        has = np.zeros(len(ukey), bool)
        has[kinv[d == dd]] = True
        cand = np.flatnonzero((nG >= 2) & has)
        lo, hi = ukey[cand] >> ll.SHIFT, ukey[cand] & ll.MASK
        E = W.expected_groups(lo, hi, st["indptr"], st["r_of"], st["cnt"], st["m_rank"], st["grp_of_rank"], float(dd))
        nc = nG[cand].astype(np.float64)
        lam = np.where(nc > E, nc * np.log(nc / np.maximum(E, 1e-300)) - nc + E, 0.0)
        key_lam = np.zeros(len(ukey))
        ok = lam >= tau
        key_lam[cand[ok]] = lam[ok]
        at = d == dd
        occ_lam[at] = key_lam[kinv[at]]
    si = occ_lam > 0
    return k[si], d[si], occ_lam[si]


def arm_bonds(arm, tok, cc, gc, grp, tau):
    if arm == "WA":
        return bonds_adaptive(tok, cc, gc, grp, tau)
    w = int(arm[1:])
    k, d, lam, info = ll.bonds(tok, cc, gc, grp, "group", tau, w=w)
    return k, d, lam


def cmd_build(a):
    import numba
    numba.set_num_threads(a.threads)
    t0 = time.time()
    tok, cc, gc, grp, kappa = ll.load(a.data)
    n = len(tok)
    ng = int(gc.max()) + 1
    toks = {r: ll.corpus_tokens(tok, gc, r) for r in [-1, 0, 1, 2]}
    T = int(tok.max()) + 1
    fsize = np.bincount(tok, minlength=T)
    mg = np.bincount(gc, minlength=ng).astype(np.float64)
    corpora = {-1: ll.fam_csr(tok, T)}
    res = dict(data=a.data, arms={}, timing={})
    for arm in (a.arms.split(",") if a.arms else ARMS):
        ta = time.time()
        lad, obs = {}, None
        for r in toks:
            tb = time.time()
            k, d, lam = arm_bonds(arm, toks[r], cc, gc, grp, min(LADDER))
            lad[str(r)] = {str(t): int((lam >= t).sum()) for t in LADDER}
            log(a.data, arm, "order", r, lad[str(r)], f"{time.time() - tb:.0f}s")
            if r < 0:
                obs = (k.astype(np.int64), d.astype(np.int64), lam.astype(np.float64))
        rows = []
        for t in LADDER:
            o = lad["-1"][str(t)]
            nl = float(np.mean([lad[str(r)][str(t)] for r in (0, 1, 2)]))
            rows.append(dict(tau=t, observed=o, null_mean=nl, fdr=nl / o if o else None))
        okt = [x["tau"] for x in rows if x["fdr"] is not None and x["fdr"] <= 0.01]
        tau = min(okt) if okt else None
        item = dict(tau_table=rows, tau_star=tau)
        log(a.data, arm, "tau*", tau, rows[:5])
        k, d, lam = obs
        keep = lam >= tau
        k, d, lam = k[keep], d[keep], lam[keep]
        dist = np.bincount(d, minlength=9)[1:]
        item["bonds"] = dict(bonds=int(len(k)), by_distance={str(i + 1): int(x) for i, x in enumerate(dist)},
                             share_by_distance={str(i + 1): round(float(x / max(len(k), 1)), 4) for i, x in enumerate(dist)})
        s = ll.boundary_score(k, d, lam, n)
        near = d <= 3
        s_near = ll.boundary_score(k[near], d[near], lam[near], n)
        item["bonds"]["boundaries_bonded"] = int((s > 0).sum())
        item["bonds"]["boundaries_only_by_d_ge4"] = int(((s > 0) & (s_near == 0)).sum())
        if arm == "W3" and not a.w3b:
            item["note"] = "ladder check only; part-1 dpann_group / sd_group are the W3 reference rows (tau 10)"
            res["arms"][arm] = item
            continue
        tag = f"{a.data}_{arm}" + ("b" if arm == "W3" else "")
        od, nd = LL / "loci" / tag, LL / "nodes" / tag
        od.mkdir(parents=True, exist_ok=True)
        nd.mkdir(parents=True, exist_ok=True)
        np.savez(nd / "bonds_obs.npz", k=k, d=d, lam=lam)
        st_, en_ = ll.runs(s > 0)
        item["outputs"] = {"L1": ll.loci_info(ll.save_loci(od, "L1", st_, en_), n)}
        trees = {}
        for th in ll.THETAS:
            low = (kappa < th) & (s > 0)
            wv = np.where(low, 1e-6 + np.maximum(np.nan_to_num(kappa, nan=0.0), 0.0), s)
            trees[th] = ll.tree(wv)
        alln = pd.concat([t[["lo", "hi"]] for t in trees.values()]).drop_duplicates().reset_index(drop=True)
        rec, kinfo = ll.recurrence(alln.lo.to_numpy(), alln.hi.to_numpy(), grp[gc[alln.lo.to_numpy()]], tok, fsize,
                                   corpora, -1, cc, gc, grp, mg, a.blocks)
        rec = pd.concat([alln, rec], axis=1)
        rec["lam"] = ll.lam_of(rec.n.to_numpy(), rec.E_ana.to_numpy())
        for th, t in trees.items():
            t = t.merge(rec[["lo", "hi", "n", "nI", "lam"]], on=["lo", "hi"], how="left")
            km = (t.n.to_numpy() >= 2) & (t.lam.to_numpy() >= tau) & (t.nI.to_numpy() >= 2)
            d1 = ll.save_loci(od, f"L4b_t{th:g}", t.lo.to_numpy()[km], t.hi.to_numpy()[km])
            pm = t.hi.to_numpy() > t.lo.to_numpy()
            d2 = ll.save_loci(od, f"L4bp_t{th:g}", t.lo.to_numpy()[pm], t.hi.to_numpy()[pm])
            item["outputs"][f"L4b_t{th:g}"] = ll.loci_info(d1, n)
            item["outputs"][f"L4bp_t{th:g}"] = ll.loci_info(d2, n)
        item["seconds"] = round(time.time() - ta, 1)
        log(a.data, arm, {k_: v for k_, v in item.items() if k_ != "tau_table"})
        res["arms"][arm] = item
        ll.jdump(res, LL / "results" / f"wbuild_{a.data}{a.suffix}.json")
    res["timing"]["total"] = round(time.time() - t0, 1)
    import resource
    res["peak_rss_gb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024, 1)
    ll.jdump(res, LL / "results" / f"wbuild_{a.data}{a.suffix}.json")


def cmd_smoke(a):
    """WA at wmax 3 restricted to d <= 3 must give, for d = 3 occurrences, the same Lambda as ll.bonds(w=3) for keys
    whose occurrences are all at distance 3; and W3 via arm_bonds equals ll.bonds(w=3). First 150 DPANN genomes."""
    tok, cc, gc, grp, kappa = ll.load("dpann", 150)
    k3, d3, l3, _ = ll.bonds(tok, cc, gc, grp, "group", 4.0, w=3)
    kw, dw, lw_ = arm_bonds("W3", tok, cc, gc, grp, 4.0)
    assert np.array_equal(k3, kw) and np.array_equal(l3, lw_)
    ka, da, la = bonds_adaptive(tok, cc, gc, grp, 4.0, wmax=8)
    k1, d1, l1, _ = ll.bonds(tok, cc, gc, grp, "group", 4.0, w=1)
    m = da == 1
    same1 = np.array_equal(np.sort(ka[m]), np.sort(k1)) and np.allclose(np.sort(la[m]), np.sort(l1))
    log("smoke: W3 equal", True, "| WA at d=1 equals w=1 bonds:", same1, "| WA bonds by d",
        np.bincount(da, minlength=9)[1:].tolist())
    assert same1
    log("smoke OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "smoke"])
    ap.add_argument("--data", default="dpann")
    ap.add_argument("--blocks", type=int, default=4096)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--arms", default="")
    ap.add_argument("--w3b", action="store_true")
    ap.add_argument("--suffix", default="")
    a = ap.parse_args()
    {"build": cmd_build, "smoke": cmd_smoke}[a.cmd](a)


if __name__ == "__main__":
    main()
