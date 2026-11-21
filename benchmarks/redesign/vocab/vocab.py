#!/usr/bin/env python
"""MDL vocabulary of gene-order words (workstream ws_vocab).

Genomes are unsegmented text over symbols (stage 1: MMseqs2 family tokens). A word is a node of a merge tree; its
instances are contiguous gene spans built from two adjacent live nodes (<= 2 intervening single genes, unordered).
The corpus that is compressed is the group-level occurrence record: per symbol pair, n = lineage groups with an
occurrence; E = expected groups under random order within each genome (live copy numbers). Merge gain
    Lambda = n ln(n/E) - n + E   (nats; Poisson log-likelihood ratio = code-length saving of naming the pair).
Rounds: every pair with n >= 2 and Lambda >= tau is applied in decreasing Lambda; occurrences claim nodes; a pair is
committed only if its claimed occurrences keep n >= 2 and Lambda >= tau. Until nothing commits.
tau is set from the identical procedure on genome-shuffled tokens (word-type FDR).

usage (biotite):
  vocab.py ladder --arm main --taus 4,6,8 --nulls 3 --threads N      obs + nulls per tau, in parallel processes
  vocab.py run --arm main --tau 20 [--null r] --threads 1             one run (writes $RD/vocab/runs/<arm>/tau<tau>/<obs|nullr>)
  vocab.py smoke --genomes 300                                        small correctness run (login node)
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

RD = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111")
PKG = RD / "ws_dpann/package"
VD = RD / "vocab"
TOK = RD / "engine/dpann_tokens/mmseqs__seq__e3c50s75_rescue3__dpann.parquet"
UNITS = {
    "main": RD / "lineagefree/groups/emb_hom_average_t0.75.parquet",
    "genus": RD / "lineagefree/groups/genus_map.parquet",
    "genome": RD / "lineagefree/groups/genome_map.parquet",
}
SHIFT = np.int64(31)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# --------------------------------------------------------------------------- data
def load_universe():
    g = pd.read_parquet(PKG / "universe/genes.parquet", columns=["gid", "genome_code", "contig_code", "position_index"])
    assert np.array_equal(g.gid.to_numpy(), np.arange(len(g)))
    cc = g.contig_code.to_numpy(np.int64)
    gc = g.genome_code.to_numpy(np.int64)
    assert np.all(np.diff(cc) >= 0) and np.all(np.diff(gc) >= 0)
    return cc, gc, g.position_index.to_numpy(np.int64)


def load_tokens(n):
    t = pd.read_parquet(TOK)
    tok = np.full(n, -1, dtype=np.int64)
    tok[t.gid.to_numpy()] = t.token.to_numpy()
    assert (tok >= 0).all()
    return tok


def load_groups(arm, n_genomes):
    m = pd.read_parquet(UNITS[arm], columns=["genome_code", "group"]).sort_values("genome_code")
    assert np.array_equal(m.genome_code.to_numpy(), np.arange(n_genomes))
    grp, _ = pd.factorize(m.group.to_numpy())
    return grp.astype(np.int64)


def genome_bounds(gc):
    ch = np.flatnonzero(np.diff(gc)) + 1
    return np.r_[0, ch].astype(np.int64), np.r_[ch, len(gc)].astype(np.int64)


def shuffle_tokens(tok, gc, r):
    """Engine genome null: tokens permuted within each genome (contig sizes kept), rng seed 1000 + r."""
    rng = np.random.default_rng(1000 + r)
    out = tok.copy()
    gs, ge = genome_bounds(gc)
    for s, e in zip(gs, ge):
        out[s:e] = rng.permutation(tok[s:e])
    return out


# --------------------------------------------------------------------------- kernels
@njit(cache=True)
def expected_groups(lo, hi, indptr, rank, cnt, m_rank, grp_of_rank, w):
    """E per pair: sum over groups of 1 - prod over the group's genomes of (1 - min(1, 2 w c_lo c_hi / m_g)).
    Each symbol's genome list is sorted by rank; ranks are ordered by group."""
    out = np.zeros(len(lo), dtype=np.float64)
    for p in range(len(lo)):
        i, ie = indptr[lo[p]], indptr[lo[p] + 1]
        j, je = indptr[hi[p]], indptr[hi[p] + 1]
        s = 0.0
        cur = -1
        lp = 0.0
        while i < ie and j < je:
            if rank[i] == rank[j]:
                g = grp_of_rank[rank[i]]
                if g != cur:
                    if cur >= 0:
                        s += 1.0 - np.exp(lp)
                    cur = g
                    lp = 0.0
                x = 2.0 * w * cnt[i] * cnt[j] / m_rank[rank[i]]
                if x >= 1.0:
                    lp = -1e300
                else:
                    lp += np.log1p(-x)
                i += 1
                j += 1
            elif rank[i] < rank[j]:
                i += 1
            else:
                j += 1
        if cur >= 0:
            s += 1.0 - np.exp(lp)
        out[p] = s
    return out


@njit(cache=True)
def gain(n, E):
    if n <= E:
        return 0.0
    return n * np.log(n / E) - n + E


@njit(cache=True)
def assign(occ_k, occ_d, occ_r, occ_g, key_E, tau, m, n_groups, min_n):
    """Greedy claim in key-rank order (occurrences pre-sorted by rank, distance, position)."""
    claimed = np.zeros(m, dtype=np.bool_)
    stamp = np.full(n_groups, -1, dtype=np.int64)
    nk = len(key_E)
    key_ok = np.zeros(nk, dtype=np.bool_)
    key_n = np.zeros(nk, dtype=np.int64)
    key_lam = np.zeros(nk, dtype=np.float64)
    key_occ = np.zeros(nk, dtype=np.int64)
    ck = np.empty(len(occ_k), dtype=np.int64)
    cd = np.empty(len(occ_k), dtype=np.int64)
    cr = np.empty(len(occ_k), dtype=np.int64)
    tmp = np.empty(len(occ_k), dtype=np.int64)
    nc = 0
    i = 0
    N = len(occ_k)
    while i < N:
        r = occ_r[i]
        j = i
        nt = 0
        ng = 0
        while j < N and occ_r[j] == r:
            k = occ_k[j]
            d = occ_d[j]
            free = True
            for t in range(k, k + d + 1):
                if claimed[t]:
                    free = False
                    break
            if free:
                for t in range(k, k + d + 1):
                    claimed[t] = True
                tmp[nt] = j
                nt += 1
                g = occ_g[j]
                if stamp[g] != r:
                    stamp[g] = r
                    ng += 1
            j += 1
        lam = gain(float(ng), key_E[r])
        if ng >= min_n and lam >= tau:
            key_ok[r] = True
            key_n[r] = ng
            key_lam[r] = lam
            key_occ[r] = nt
            for q in range(nt):
                ck[nc] = occ_k[tmp[q]]
                cd[nc] = occ_d[tmp[q]]
                cr[nc] = r
                nc += 1
        else:
            for q in range(nt):
                k = occ_k[tmp[q]]
                d = occ_d[tmp[q]]
                for t in range(k, k + d + 1):
                    claimed[t] = False
        i = j
    return ck[:nc], cd[:nc], cr[:nc], key_ok, key_n, key_lam, key_occ


@njit(cache=True)
def rebuild(ns, ne, sym, term, ninst, ck, cd, csym, inst_base):
    """Replace committed occurrences (ck sorted, non-overlapping) by one node each. Returns new live arrays and the
    new instance rows (sym, s, e, child_a, child_b); a child >= 0 is an instance id, < 0 is terminal gene -(gid+1)."""
    m = len(ns)
    nc = len(ck)
    drop = 0
    for c in range(nc):
        drop += cd[c]
    m2 = m - drop
    ns2 = np.empty(m2, dtype=np.int64)
    ne2 = np.empty(m2, dtype=np.int64)
    sym2 = np.empty(m2, dtype=np.int64)
    term2 = np.empty(m2, dtype=np.bool_)
    ninst2 = np.empty(m2, dtype=np.int64)
    i_sym = np.empty(nc, dtype=np.int64)
    i_s = np.empty(nc, dtype=np.int64)
    i_e = np.empty(nc, dtype=np.int64)
    i_a = np.empty(nc, dtype=np.int64)
    i_b = np.empty(nc, dtype=np.int64)
    i = 0
    c = 0
    o = 0
    while i < m:
        if c < nc and ck[c] == i:
            k = i
            d = cd[c]
            ns2[o] = ns[k]
            ne2[o] = ne[k + d]
            sym2[o] = csym[c]
            term2[o] = False
            ninst2[o] = inst_base + c
            i_sym[c] = csym[c]
            i_s[c] = ns[k]
            i_e[c] = ne[k + d]
            i_a[c] = ninst[k] if not term[k] else -(ns[k] + 1)
            i_b[c] = ninst[k + d] if not term[k + d] else -(ns[k + d] + 1)
            o += 1
            c += 1
            i = k + d + 1
        else:
            ns2[o] = ns[i]
            ne2[o] = ne[i]
            sym2[o] = sym[i]
            term2[o] = term[i]
            ninst2[o] = ninst[i]
            o += 1
            i += 1
    return ns2, ne2, sym2, term2, ninst2, i_sym, i_s, i_e, i_a, i_b


# --------------------------------------------------------------------------- one run
def occurrences(ns, ncont, sym, term, w):
    m = len(ns)
    ks, ds = [], []
    for d in range(1, w + 1):
        if m <= d:
            break
        ok = (ncont[:-d] == ncont[d:]) & (sym[:-d] >= 0) & (sym[d:] >= 0) & (sym[:-d] != sym[d:])
        for j in range(1, d):
            ok &= term[j:m - d + j]
        k = np.flatnonzero(ok)
        ks.append(k)
        ds.append(np.full(len(k), d, dtype=np.int64))
    k = np.concatenate(ks)
    d = np.concatenate(ds)
    a, b = sym[k], sym[k + d]
    key = (np.minimum(a, b) << SHIFT) | np.maximum(a, b)
    return k, d, key


def run_vocab(tok, cc, gc, grp, tau, w=3, max_rounds=500, min_n=2, verbose=True):
    """Returns dict(words=DataFrame, inst=dict of arrays, live=dict of arrays, rounds=list)."""
    t0 = time.time()
    n = len(tok)
    ng = int(grp.max()) + 1
    n_genomes = len(grp)
    # genome ranks ordered by group (so a symbol's genome list is grouped by group)
    order = np.lexsort((np.arange(n_genomes), grp))
    rank_of_genome = np.empty(n_genomes, dtype=np.int64)
    rank_of_genome[order] = np.arange(n_genomes)
    grp_of_rank = grp[order]
    # background: symbols present in < min_n groups
    T = int(tok.max()) + 1
    tg = np.unique(tok * ng + grp[gc])
    tok_groups = np.bincount(tg // ng, minlength=T)
    sym = np.where(tok_groups[tok] >= min_n, tok, -1).astype(np.int64)
    ns = np.arange(n, dtype=np.int64)
    ne = ns.copy()
    term = np.ones(n, dtype=np.bool_)
    ninst = np.full(n, -1, dtype=np.int64)
    ncont, ngen = cc.copy(), gc.copy()
    S = T
    words, inst_parts, rounds = [], [], []
    n_inst = 0
    for rnd in range(max_rounds):
        tr = time.time()
        k, d, key = occurrences(ns, ncont, sym, term, w)
        if len(k) == 0:
            break
        ukey, kinv = np.unique(key, return_inverse=True)
        g_occ = grp[ngen[k]]
        kg = np.unique(kinv * ng + g_occ)
        nG = np.bincount(kg // ng, minlength=len(ukey))
        cand = np.flatnonzero(nG >= min_n)
        lo, hi = ukey[cand] >> SHIFT, ukey[cand] & ((np.int64(1) << SHIFT) - 1)
        # live copy numbers per (symbol, genome rank)
        live = sym >= 0
        sr = np.unique(sym[live] * n_genomes + rank_of_genome[ngen[live]], return_counts=True)
        s_of, r_of = sr[0] // n_genomes, sr[0] % n_genomes
        indptr = np.searchsorted(s_of, np.arange(S + 1)).astype(np.int64)
        m_rank = np.bincount(rank_of_genome[ngen], minlength=n_genomes).astype(np.float64)
        E = expected_groups(lo, hi, indptr, r_of.astype(np.int64), sr[1].astype(np.float64), m_rank,
                            grp_of_rank.astype(np.int64), float(w))
        lam = nG[cand] * np.log(nG[cand] / np.maximum(E, 1e-300)) - nG[cand] + E
        lam = np.where(nG[cand] > E, lam, 0.0)
        pas = lam >= tau
        if not pas.any():
            rounds.append(dict(round=rnd, occurrences=int(len(k)), keys=int(len(ukey)), keys_n2=int(len(cand)),
                               candidates=0, committed=0, seconds=round(time.time() - tr, 1)))
            break
        ck_idx = cand[pas]
        ordk = np.lexsort((ck_idx, -lam[pas]))           # decreasing Lambda, ties by key
        ck_idx = ck_idx[ordk]
        cE, clam0, clo, chi, cn0 = E[pas][ordk], lam[pas][ordk], lo[pas][ordk], hi[pas][ordk], nG[ck_idx]
        rank_of_key = np.full(len(ukey), -1, dtype=np.int64)
        rank_of_key[ck_idx] = np.arange(len(ck_idx))
        orr = rank_of_key[kinv]
        sel = np.flatnonzero(orr >= 0)
        o = np.lexsort((k[sel], d[sel], orr[sel]))
        sel = sel[o]
        c_k, c_d, c_r, key_ok, key_n, key_lam, key_occ = assign(k[sel], d[sel], orr[sel], g_occ[sel], cE, float(tau),
                                                               len(ns), ng, min_n)
        n_commit = int(key_ok.sum())
        if n_commit == 0:
            rounds.append(dict(round=rnd, occurrences=int(len(k)), keys=int(len(ukey)), keys_n2=int(len(cand)),
                               candidates=int(len(ck_idx)), committed=0, seconds=round(time.time() - tr, 1)))
            break
        new_sym = np.full(len(ck_idx), -1, dtype=np.int64)
        okr = np.flatnonzero(key_ok)
        new_sym[okr] = S + np.arange(len(okr))
        words.append(pd.DataFrame(dict(sym=new_sym[okr], a=clo[okr], b=chi[okr], lam=key_lam[okr], n=key_n[okr],
                                       E=cE[okr], lam0=clam0[okr], n0=cn0[okr], occ=key_occ[okr], round=rnd)))
        S += len(okr)
        oc = np.argsort(c_k, kind="stable")
        c_k, c_d, c_s = c_k[oc], c_d[oc], new_sym[c_r[oc]]
        ns, ne, sym, term, ninst, i_sym, i_s, i_e, i_a, i_b = rebuild(ns, ne, sym, term, ninst, c_k, c_d, c_s, n_inst)
        # contig / genome of new nodes = those of their first gene
        ncont, ngen = cc[ns], gc[ns]
        inst_parts.append((i_sym, i_s, i_e, i_a, i_b, np.full(len(i_sym), rnd, dtype=np.int64)))
        n_inst += len(i_sym)
        rounds.append(dict(round=rnd, occurrences=int(len(k)), keys=int(len(ukey)), keys_n2=int(len(cand)),
                           candidates=int(len(ck_idx)), committed=n_commit, committed_occ=int(len(c_k)),
                           live_nodes=int(len(ns)), max_lam=float(clam0[0]), seconds=round(time.time() - tr, 1)))
        if verbose:
            log("round", rnd, {k_: v for k_, v in rounds[-1].items() if k_ != "round"}, f"{time.time()-t0:.0f}s")
    words = pd.concat(words, ignore_index=True) if words else pd.DataFrame(
        columns=["sym", "a", "b", "lam", "n", "E", "lam0", "n0", "occ", "round"])
    if inst_parts:
        inst = {nm: np.concatenate([p[i] for p in inst_parts]) for i, nm in enumerate(("sym", "s", "e", "a", "b", "round"))}
    else:
        inst = {nm: np.zeros(0, dtype=np.int64) for nm in ("sym", "s", "e", "a", "b", "round")}
    return dict(words=words, inst=inst, live=dict(ns=ns, ne=ne, sym=sym, term=term, ninst=ninst), rounds=rounds,
                T=T, seconds=round(time.time() - t0, 1), n_background_tokens=int((tok_groups < min_n).sum()))


def run_summary(res, n_genes):
    live = res["live"]
    top = ~live["term"]
    cov = int((live["ne"][top] - live["ns"][top] + 1).sum())
    return dict(word_types=int(len(res["words"])), instances=int(len(res["inst"]["sym"])),
                top_instances=int(top.sum()), top_types=int(len(np.unique(live["sym"][top]))),
                genes_in_top=cov, genes_in_top_frac=cov / n_genes, rounds=len(res["rounds"]), seconds=res["seconds"])


def save_run(res, out):
    out.mkdir(parents=True, exist_ok=True)
    res["words"].to_parquet(out / "words.parquet", index=False)
    np.savez(out / "inst.npz", **res["inst"])
    np.savez(out / "live.npz", **res["live"])
    with open(out / "run.json", "w") as fh:
        json.dump(dict(rounds=res["rounds"], T=res["T"], seconds=res["seconds"],
                       n_background_tokens=res["n_background_tokens"], summary=res.get("summary")), fh, indent=1)


def tau_tag(tau):
    return f"tau{tau:g}"


def one_run(args):
    arm, tau, null, save = args
    cc, gc, _ = load_universe()
    tok = load_tokens(len(cc))
    grp = load_groups(arm, int(gc.max()) + 1)
    if null >= 0:
        tok = shuffle_tokens(tok, gc, null)
    res = run_vocab(tok, cc, gc, grp, tau, verbose=False)
    res["summary"] = run_summary(res, len(cc))
    out = VD / "runs" / arm / tau_tag(tau) / ("obs" if null < 0 else f"null{null}")
    if save or null < 0:
        save_run(res, out)
    else:
        out.mkdir(parents=True, exist_ok=True)
        res["words"].to_parquet(out / "words.parquet", index=False)
        with open(out / "run.json", "w") as fh:
            json.dump(dict(rounds=res["rounds"], T=res["T"], seconds=res["seconds"], summary=res["summary"]), fh, indent=1)
    s = res["summary"]
    log(arm, tau, "obs" if null < 0 else f"null{null}", s)
    return dict(arm=arm, tau=tau, null=null, **s)


# --------------------------------------------------------------------------- commands
def cmd_ladder(a):
    import multiprocessing as mp
    taus = [float(x) for x in a.taus.split(",")]
    arms = a.arm.split("+")
    jobs = [(arm, t, r, a.save_nulls) for arm in arms for t in taus for r in [-1] + list(range(a.nulls))]
    jobs.sort(key=lambda j: (j[1], j[0] != "genome", j[2]))  # low tau first (slowest)
    log("ladder", arms, len(jobs), "runs on", a.procs, "processes")
    with mp.get_context("fork").Pool(a.procs) as pool:
        rows = pool.map(one_run, jobs, chunksize=1)
    df = pd.DataFrame(rows)
    for arm in arms:
        out = VD / "results" / f"ladder_{arm}{('_' + a.tag) if a.tag else ''}.json"
        summ = []
        for t, g in df[df.arm == arm].groupby("tau"):
            o = g[g.null < 0].iloc[0]
            nl = g[g.null >= 0]
            summ.append(dict(tau=t, obs_types=int(o.word_types), null_types_mean=float(nl.word_types.mean()),
                             fdr_types=float(nl.word_types.mean() / max(o.word_types, 1)),
                             obs_genes=int(o.genes_in_top), null_genes_mean=float(nl.genes_in_top.mean()),
                             fdr_genes=float(nl.genes_in_top.mean() / max(o.genes_in_top, 1)),
                             obs_rounds=int(o.rounds), obs_seconds=float(o.seconds),
                             null_seconds_mean=float(nl.seconds.mean())))
        sel = [s for s in summ if s["fdr_types"] <= 0.01]
        res = dict(arm=arm, runs=[r for r in rows if r["arm"] == arm], ladder=summ,
                   tau_star=(min(s["tau"] for s in sel) if sel else None),
                   written=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        with open(out, "w") as fh:
            json.dump(res, fh, indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x))
        for s in summ:
            log(arm, s)
        log(arm, "tau*", res["tau_star"], "wrote", out)


def cmd_run(a):
    r = one_run((a.arm, a.tau, a.null, True))
    print(json.dumps(r, indent=1))


def cmd_smoke(a):
    """Correctness checks on the first N genomes (gold-free)."""
    cc, gc, _ = load_universe()
    keep = gc < a.genomes
    cc, gc = cc[keep], gc[keep]
    tok = load_tokens(len(keep))[keep]
    grp = load_groups(a.arm, int(load_universe()[1].max()) + 1)[: a.genomes]
    grp = pd.factorize(grp)[0].astype(np.int64)
    res = run_vocab(tok, cc, gc, grp, a.tau, verbose=True)
    s = run_summary(res, len(tok))
    log("smoke", s)
    # invariants: live nodes tile every contig; instance spans = union of children spans + inserts; words n >= 2
    live = res["live"]
    assert live["ns"][0] == 0 and live["ne"][-1] == len(tok) - 1
    assert np.all(live["ns"][1:] == live["ne"][:-1] + 1), "live nodes do not tile the genes"
    assert np.all(cc[live["ns"]] == cc[live["ne"]]), "node crosses a contig"
    inst = res["inst"]
    for i in range(len(inst["sym"])):
        ea = inst["e"][inst["a"][i]] if inst["a"][i] >= 0 else -inst["a"][i] - 1
        sb = inst["s"][inst["b"][i]] if inst["b"][i] >= 0 else -inst["b"][i] - 1
        sa = inst["s"][inst["a"][i]] if inst["a"][i] >= 0 else -inst["a"][i] - 1
        eb = inst["e"][inst["b"][i]] if inst["b"][i] >= 0 else -inst["b"][i] - 1
        assert sa == inst["s"][i] and eb == inst["e"][i] and 0 <= sb - ea - 1 <= 2, i
    assert (res["words"].n >= 2).all()
    # every instance of a word type pairs the type's two symbols
    w = res["words"].set_index("sym")
    T = res["T"]

    def symof(c):
        return inst["sym"][c] if c >= 0 else tok[-c - 1]
    bad = 0
    for i in range(len(inst["sym"])):
        x, y = symof(inst["a"][i]), symof(inst["b"][i])
        row = w.loc[inst["sym"][i]]
        if {int(x), int(y)} != {int(row.a), int(row.b)}:
            bad += 1
    log("instances whose children do not match their type:", bad)
    assert bad == 0
    log("smoke OK", time.time())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["ladder", "run", "smoke"])
    ap.add_argument("--arm", default="main")
    ap.add_argument("--tau", type=float, default=20.0)
    ap.add_argument("--taus", default="4,6,8,10,12,15,20,25,30,40,50,70,100")
    ap.add_argument("--nulls", type=int, default=3)
    ap.add_argument("--null", type=int, default=-1)
    ap.add_argument("--procs", type=int, default=1)
    ap.add_argument("--save-nulls", action="store_true")
    ap.add_argument("--genomes", type=int, default=300)
    ap.add_argument("--tag", default="")
    ap.add_argument("--threads", type=int, default=1)
    a = ap.parse_args()
    if a.cmd == "ladder" and a.procs == 1:
        a.procs = a.threads
    {"ladder": cmd_ladder, "run": cmd_run, "smoke": cmd_smoke}[a.cmd](a)


if __name__ == "__main__":
    main()
