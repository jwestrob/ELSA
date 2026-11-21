#!/usr/bin/env python
"""Stage-2 MDL vocabulary over multi-resolution units (workstream ws_vocab).

Same objective and commit rule as vocab.py. Each live node carries L labels. Terminal labels = the gene's units at L
resolutions (vocabulary tree; or one level of MMseqs2 tokens for the reproduction check); a unit present in < 2
groups is background at that level. A word node's label at level l is its projection: the canonical id of the
unordered pair of its children's level-l labels (background if either is). Keys: an adjacency of two terminals
offers all L x L level pairs (each slot chooses its own radius); an adjacency involving a word offers the L same-level
pairs. All keys compete for nodes in decreasing Lambda; a committed key is a word type.

usage (biotite):
  vocab2.py repro --threads N                               L = 1 tokens at tau 10 vs the stored vocab.py run
  vocab2.py run --alphabet vt1k --tau 10 [--null r]         one run -> $RD/vocab/runs2/<alphabet>_<arm>/tau<t>/<obs|nullr>
  vocab2.py ladder --alphabet vt1k --taus 6,8,... --procs P  obs + 3 nulls per tau in parallel processes
"""
import argparse
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit, prange

RD = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111")
VD = RD / "vocab"
sys.path.insert(0, str(VD / "code"))
import vocab as V  # noqa: E402

SHIFT = np.int64(31)
WORKER_THREADS = 1
MASK = (np.int64(1) << SHIFT) - 1


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def rss_gb():
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024, 2)


@njit(parallel=True, cache=True)
def expected_groups_par(lo, hi, indptr, rank, cnt, m_rank, grp_of_rank, w):
    """Parallel copy of vocab.expected_groups (pairs are independent; identical results)."""
    out = np.zeros(len(lo), dtype=np.float64)
    for p in prange(len(lo)):
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


def load_labels(alphabet, n):
    """'tokens', '<tree tag>' (all levels) or '<tree tag>@<levels>' (e.g. vt1k@2, vt1k@12: columns 2 / 1,2)."""
    if alphabet == "tokens":
        return V.load_tokens(n)[:, None].copy()
    tag, _, cols = alphabet.partition("@")
    lab = np.load(VD / "units" / tag / "labels_gene.npy")
    assert lab.shape[0] == n
    if cols:
        lab = lab[:, [int(c) for c in cols]]
    return np.ascontiguousarray(lab.astype(np.int64))


@njit(cache=True)
def rebuild_multi(ns, ne, labs, term, ninst, ck, cd, newlab, inst_base):
    m, L = labs.shape
    nc = len(ck)
    drop = 0
    for c in range(nc):
        drop += cd[c]
    m2 = m - drop
    ns2 = np.empty(m2, dtype=np.int64)
    ne2 = np.empty(m2, dtype=np.int64)
    labs2 = np.empty((m2, L), dtype=np.int64)
    term2 = np.empty(m2, dtype=np.bool_)
    ninst2 = np.empty(m2, dtype=np.int64)
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
            for l in range(L):
                labs2[o, l] = newlab[c, l]
            term2[o] = False
            ninst2[o] = inst_base + c
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
            for l in range(L):
                labs2[o, l] = labs[i, l]
            term2[o] = term[i]
            ninst2[o] = ninst[i]
            o += 1
            i += 1
    return ns2, ne2, labs2, term2, ninst2, i_s, i_e, i_a, i_b


def occurrences_multi(ncont, labs, term, w):
    m, L = labs.shape
    ks, ds, keys, las, lbs = [], [], [], [], []
    for d in range(1, w + 1):
        if m <= d:
            break
        base = ncont[:-d] == ncont[d:]
        for j in range(1, d):
            base &= term[j:m - d + j]
        k0 = np.flatnonzero(base)
        tt = term[k0] & term[k0 + d]
        for la in range(L):
            for lb in range(L):
                k = k0 if la == lb else k0[tt]
                a, b = labs[k, la], labs[k + d, lb]
                ok = (a >= 0) & (b >= 0) & (a != b)
                k, a, b = k[ok], a[ok], b[ok]
                ks.append(k.astype(np.int64))
                ds.append(np.full(len(k), d, dtype=np.int8))
                keys.append((np.minimum(a, b) << SHIFT) | np.maximum(a, b))
                las.append(np.full(len(k), la, dtype=np.int8))
                lbs.append(np.full(len(k), lb, dtype=np.int8))
    return (np.concatenate(ks), np.concatenate(ds), np.concatenate(keys), np.concatenate(las), np.concatenate(lbs))


class Projector:
    """Canonical ids of unordered label pairs, per level, persistent across rounds."""

    def __init__(self, L, reuse=True):
        self.reuse = reuse
        self.keys = [np.zeros(0, dtype=np.int64) for _ in range(L)]
        self.ids = [np.zeros(0, dtype=np.int64) for _ in range(L)]

    def project(self, a, b, rank, S):
        """a, b: (nc, L) child labels; rank: commit rank of each occurrence. New ids are numbered by
        (level, first rank, key) from S. Returns (labels (nc, L), new S)."""
        nc, L = a.shape
        out = np.full((nc, L), -1, dtype=np.int64)
        for l in range(L):
            ok = (a[:, l] >= 0) & (b[:, l] >= 0)
            pk = (np.minimum(a[ok, l], b[ok, l]) << SHIFT) | np.maximum(a[ok, l], b[ok, l])
            known = self.keys[l] if self.reuse else np.zeros(0, dtype=np.int64)
            pos = np.searchsorted(known, pk)
            hit = (pos < len(known)) & (known[np.minimum(pos, max(len(known) - 1, 0))] == pk) if len(known) else \
                np.zeros(len(pk), dtype=bool)
            res = np.full(len(pk), -1, dtype=np.int64)
            res[hit] = self.ids[l][pos[hit]]
            if (~hit).any():
                nk, nr = pk[~hit], rank[ok][~hit]
                o = np.lexsort((nk, nr))
                nk_sorted = nk[o]
                first = np.unique(nk_sorted, return_index=True)[1]
                # order new keys by their first (smallest) rank, then key
                uk = nk_sorted[np.sort(first)]
                new_ids = S + np.arange(len(uk))
                S += len(uk)
                srt = np.argsort(uk)
                uks, ids_s = uk[srt], new_ids[srt]
                res[~hit] = ids_s[np.searchsorted(uks, pk[~hit])]
                if self.reuse:
                    allk = np.r_[known, uks]
                    alli = np.r_[self.ids[l], ids_s]
                    oo = np.argsort(allk, kind="stable")
                    self.keys[l], self.ids[l] = allk[oo], alli[oo]
            out[ok, l] = res
        return out, S


def run_vocab2(lab0, cc, gc, grp, tau, w=3, max_rounds=500, min_n=2, verbose=True, reuse=True):
    t0 = time.time()
    n, L = lab0.shape
    ng = int(grp.max()) + 1
    n_genomes = len(grp)
    order = np.lexsort((np.arange(n_genomes), grp))
    rank_of_genome = np.empty(n_genomes, dtype=np.int64)
    rank_of_genome[order] = np.arange(n_genomes)
    grp_of_rank = grp[order].astype(np.int64)
    labs = lab0.copy()
    S = int(lab0.max()) + 1
    n_bg = []
    for l in range(L):
        x = lab0[:, l]
        okx = x >= 0
        tg = np.unique(x[okx] * ng + grp[gc[okx]])
        cnt = np.bincount(tg // ng, minlength=S)
        bgl = okx & (cnt[np.maximum(x, 0)] < min_n)
        labs[bgl, l] = -1
        n_bg.append(int(bgl.sum()))
    T = S
    proj = Projector(L, reuse)
    ns = np.arange(n, dtype=np.int64)
    ne = ns.copy()
    term = np.ones(n, dtype=np.bool_)
    ninst = np.full(n, -1, dtype=np.int64)
    ncont, ngen = cc.copy(), gc.copy()
    words, inst_parts, rounds = [], [], []
    n_inst = 0
    n_types = 0
    for rnd in range(max_rounds):
        tr = time.time()
        k, d, key, la, lb = occurrences_multi(ncont, labs, term, w)
        if len(k) == 0:
            break
        ukey, kinv = np.unique(key, return_inverse=True)
        g_occ = grp[ngen[k]].astype(np.int32)
        kg = np.unique(kinv * ng + g_occ)
        nG = np.bincount(kg // ng, minlength=len(ukey))
        cand = np.flatnonzero(nG >= min_n)
        lo, hi = ukey[cand] >> SHIFT, ukey[cand] & MASK
        live_sym = labs.ravel()
        live_gen = np.repeat(rank_of_genome[ngen], L)
        okl = live_sym >= 0
        sr = np.unique(live_sym[okl] * n_genomes + live_gen[okl], return_counts=True)
        s_of, r_of = sr[0] // n_genomes, sr[0] % n_genomes
        indptr = np.searchsorted(s_of, np.arange(S + 1)).astype(np.int64)
        m_rank = np.bincount(rank_of_genome[ngen], minlength=n_genomes).astype(np.float64)
        E = expected_groups_par(lo, hi, indptr, r_of.astype(np.int64), sr[1].astype(np.float64), m_rank, grp_of_rank, float(w))
        nc_ = nG[cand]
        lam = np.where(nc_ > E, nc_ * np.log(nc_ / np.maximum(E, 1e-300)) - nc_ + E, 0.0)
        pas = lam >= tau
        if not pas.any():
            rounds.append(dict(round=rnd, occurrences=int(len(k)), keys=int(len(ukey)), keys_n2=int(len(cand)),
                               candidates=0, committed=0, seconds=round(time.time() - tr, 1)))
            break
        ck_idx = cand[pas]
        ordk = np.lexsort((ck_idx, -lam[pas]))
        ck_idx = ck_idx[ordk]
        cE, clam0, clo, chi, cn0 = E[pas][ordk], lam[pas][ordk], lo[pas][ordk], hi[pas][ordk], nG[ck_idx]
        rank_of_key = np.full(len(ukey), -1, dtype=np.int64)
        rank_of_key[ck_idx] = np.arange(len(ck_idx))
        orr = rank_of_key[kinv]
        sel = np.flatnonzero(orr >= 0)
        o = np.lexsort((k[sel], d[sel], orr[sel]))
        sel = sel[o]
        # level pair of each key: from its first occurrence (a key's symbols fix its levels)
        key_la = np.full(len(ck_idx), -1, dtype=np.int64)
        key_lb = np.full(len(ck_idx), -1, dtype=np.int64)
        key_la[orr[sel]] = la[sel]
        key_lb[orr[sel]] = lb[sel]
        del kinv, kg
        c_k, c_d, c_r, key_ok, key_n, key_lam, key_occ = V.assign(k[sel], d[sel], orr[sel], g_occ[sel], cE, float(tau),
                                                                 len(ns), ng, min_n)
        n_commit = int(key_ok.sum())
        if n_commit == 0:
            rounds.append(dict(round=rnd, occurrences=int(len(k)), keys=int(len(ukey)), keys_n2=int(len(cand)),
                               candidates=int(len(ck_idx)), committed=0, seconds=round(time.time() - tr, 1)))
            break
        okr = np.flatnonzero(key_ok)
        type_id = np.full(len(ck_idx), -1, dtype=np.int64)
        type_id[okr] = n_types + np.arange(len(okr))
        n_types += len(okr)
        oc = np.lexsort((c_k, c_r))      # by commit rank, then position: projection ids numbered by first rank
        c_k, c_d, c_r = c_k[oc], c_d[oc], c_r[oc]
        newlab, S2 = proj.project(labs[c_k], labs[c_k + c_d], c_r, S)
        # the committed key's own projection (for the word table): label at the key's level when la == lb
        S = S2
        op = np.argsort(c_k, kind="stable")
        c_k, c_d, c_r, newlab = c_k[op], c_d[op], c_r[op], newlab[op]
        first_lab = np.full(len(ck_idx), -1, dtype=np.int64)
        lvl = key_la[c_r]
        same = key_la[c_r] == key_lb[c_r]
        first_lab[c_r[same]] = newlab[np.flatnonzero(same), lvl[same]]
        words.append(pd.DataFrame(dict(type=type_id[okr], a=clo[okr], b=chi[okr], la=key_la[okr], lb=key_lb[okr],
                                       sym=first_lab[okr], lam=key_lam[okr], n=key_n[okr], E=cE[okr], lam0=clam0[okr],
                                       n0=cn0[okr], occ=key_occ[okr], round=rnd)))
        ns, ne, labs, term, ninst, i_s, i_e, i_a, i_b = rebuild_multi(ns, ne, labs, term, ninst, c_k, c_d, newlab, n_inst)
        ncont, ngen = cc[ns], gc[ns]
        inst_parts.append((type_id[c_r], i_s, i_e, i_a, i_b, np.full(len(i_s), rnd, dtype=np.int64)))
        n_inst += len(i_s)
        rounds.append(dict(round=rnd, occurrences=int(len(k)), keys=int(len(ukey)), keys_n2=int(len(cand)),
                           candidates=int(len(ck_idx)), committed=n_commit, committed_occ=int(len(c_k)),
                           live_nodes=int(len(ns)), max_lam=float(clam0[0]), symbols=int(S), rss_gb=rss_gb(),
                           seconds=round(time.time() - tr, 1)))
        if verbose:
            log("round", rnd, {k_: v for k_, v in rounds[-1].items() if k_ != "round"}, f"{time.time()-t0:.0f}s")
    words = pd.concat(words, ignore_index=True) if words else pd.DataFrame(
        columns=["type", "a", "b", "la", "lb", "sym", "lam", "n", "E", "lam0", "n0", "occ", "round"])
    if inst_parts:
        inst = {nm: np.concatenate([p[i] for p in inst_parts]) for i, nm in enumerate(("sym", "s", "e", "a", "b", "round"))}
    else:
        inst = {nm: np.zeros(0, dtype=np.int64) for nm in ("sym", "s", "e", "a", "b", "round")}
    top = ~term
    live = dict(ns=ns, ne=ne, labs=labs, term=term, ninst=ninst,
                sym=np.where(top, inst["sym"][np.maximum(ninst, 0)] if len(inst["sym"]) else -1, -1))
    return dict(words=words, inst=inst, live=live, rounds=rounds, T=T, L=L, seconds=round(time.time() - t0, 1),
                n_background=n_bg, rss_gb=rss_gb())


def summary(res, n):
    live = res["live"]
    top = ~live["term"]
    cov = int((live["ne"][top] - live["ns"][top] + 1).sum())
    w = res["words"]
    return dict(word_types=int(len(w)), instances=int(len(res["inst"]["sym"])), top_instances=int(top.sum()),
                top_types=int(len(np.unique(live["sym"][top]))), genes_in_top=cov, genes_in_top_frac=cov / n,
                rounds=len(res["rounds"]), seconds=res["seconds"], rss_gb=res["rss_gb"],
                types_by_level_pair={f"{a}{b}": int(c) for (a, b), c in w.groupby(["la", "lb"]).size().items()} if len(w) else {})


def save(res, out, light=False):
    out.mkdir(parents=True, exist_ok=True)
    res["words"].to_parquet(out / "words.parquet", index=False)
    if not light:
        np.savez(out / "inst.npz", **res["inst"])
        np.savez(out / "live.npz", **{k: v for k, v in res["live"].items()})
    with open(out / "run.json", "w") as fh:
        json.dump(dict(rounds=res["rounds"], T=res["T"], L=res["L"], seconds=res["seconds"], n_background=res["n_background"],
                       summary=res.get("summary")), fh, indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x))


def run_dir(alphabet, arm, tau, null):
    return VD / "runs2" / f"{alphabet}_{arm}" / f"tau{tau:g}" / ("obs" if null < 0 else f"null{null}")


def one(args):
    alphabet, arm, tau, null = args
    import numba
    numba.set_num_threads(WORKER_THREADS)
    cc, gc, _ = V.load_universe()
    lab = load_labels(alphabet, len(cc))
    grp = V.load_groups(arm, int(gc.max()) + 1)
    if null >= 0:
        rng = np.random.default_rng(1000 + null)
        gs, ge = V.genome_bounds(gc)
        perm = np.arange(len(cc))
        for s, e in zip(gs, ge):
            perm[s:e] = s + rng.permutation(e - s)
        lab = lab[perm]
    res = run_vocab2(lab, cc, gc, grp, tau, verbose=False)
    res["summary"] = summary(res, len(cc))
    save(res, run_dir(alphabet, arm, tau, null), light=null >= 0)
    log(alphabet, arm, tau, "obs" if null < 0 else f"null{null}", res["summary"])
    return dict(alphabet=alphabet, arm=arm, tau=tau, null=null, **{k: v for k, v in res["summary"].items()
                                                                    if k != "types_by_level_pair"})


def cmd_repro(a):
    """L = 1 tokens at tau 10 must equal the stored vocab.py run (main arm)."""
    cc, gc, _ = V.load_universe()
    lab = load_labels("tokens", len(cc))
    grp = V.load_groups("main", int(gc.max()) + 1)
    res = run_vocab2(lab, cc, gc, grp, 10.0, verbose=True, reuse=False)
    ref = VD / "runs" / "main" / "tau10" / "obs"
    w0 = pd.read_parquet(ref / "words.parquet").sort_values("sym").reset_index(drop=True)
    w1 = res["words"].sort_values("sym").reset_index(drop=True)
    i0 = dict(np.load(ref / "inst.npz"))
    checks = dict(
        n_types=(len(w0), len(w1)),
        word_sym_a_b_n_round=bool(len(w0) == len(w1) and all(np.array_equal(w0[c].to_numpy(), w1[c].to_numpy())
                                                             for c in ("sym", "a", "b", "n", "occ", "round"))),
        word_lam_maxabs=float(np.abs(w0.lam.to_numpy() - w1.lam.to_numpy()).max()) if len(w0) == len(w1) else None,
        inst_spans=bool(np.array_equal(i0["s"], res["inst"]["s"]) and np.array_equal(i0["e"], res["inst"]["e"])
                        and np.array_equal(i0["a"], res["inst"]["a"]) and np.array_equal(i0["b"], res["inst"]["b"])),
    )
    # instance symbols: stage 1 stores word symbols, stage 2 stores type ids; compare through the type -> sym map
    t2s = dict(zip(res["words"].type, res["words"].sym))
    checks["inst_sym"] = bool(np.array_equal(i0["sym"], np.array([t2s[t] for t in res["inst"]["sym"]])))
    checks["reproduced"] = bool(checks["word_sym_a_b_n_round"] and checks["inst_spans"] and checks["inst_sym"]
                                and (checks["word_lam_maxabs"] or 0) < 1e-9)
    log("repro", checks)
    with open(VD / "results" / "repro_vocab2.json", "w") as fh:
        json.dump(dict(checks=checks, seconds=res["seconds"], rss_gb=res["rss_gb"]), fh, indent=1, default=str)


def cmd_run(a):
    global WORKER_THREADS
    WORKER_THREADS = a.threads
    print(json.dumps(one((a.alphabet, a.arm, a.tau, a.null)), indent=1, default=str))


def cmd_ladder(a):
    import multiprocessing as mp
    taus = [float(x) for x in a.taus.split(",")]
    arms = a.arm.split("+")
    alphabets = a.alphabet.split("+")
    jobs = [(al, arm, t, r) for al in alphabets for arm in arms for t in taus for r in [-1] + list(range(a.nulls))]
    jobs.sort(key=lambda j: (j[2], j[3]))
    procs = a.procs or a.threads
    global WORKER_THREADS
    WORKER_THREADS = max(1, a.threads // procs)
    log("ladder2", alphabets, arms, len(jobs), "runs on", procs, "processes x", WORKER_THREADS, "numba threads")
    with mp.get_context("fork").Pool(procs, maxtasksperchild=1) as pool:
        rows = pool.map(one, jobs, chunksize=1)
    df = pd.DataFrame(rows)
    for al in alphabets:
        for arm in arms:
            summ = []
            for t, g in df[(df.arm == arm) & (df.alphabet == al)].groupby("tau"):
                o = g[g.null < 0].iloc[0]
                nl = g[g.null >= 0]
                summ.append(dict(tau=t, obs_types=int(o.word_types), null_types_mean=float(nl.word_types.mean()),
                                 fdr_types=float(nl.word_types.mean() / max(o.word_types, 1)), obs_genes=int(o.genes_in_top),
                                 null_genes_mean=float(nl.genes_in_top.mean()),
                                 fdr_genes=float(nl.genes_in_top.mean() / max(o.genes_in_top, 1)),
                                 obs_rounds=int(o.rounds), obs_seconds=float(o.seconds), obs_rss_gb=float(o.rss_gb)))
            sel = [s_ for s_ in summ if s_["fdr_types"] <= 0.01]
            res = dict(alphabet=al, arm=arm, ladder=summ, tau_star=(min(s_["tau"] for s_ in sel) if sel else None),
                       runs=[r for r in rows if r["arm"] == arm and r["alphabet"] == al],
                       written=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
            fn = VD / "results" / f"ladder2_{al}_{arm}.json"
            with open(fn, "w") as fh:
                json.dump(res, fh, indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x))
            for s_ in summ:
                log(al, arm, s_)
            log(al, arm, "tau*", res["tau_star"], "wrote", fn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["repro", "run", "ladder"])
    ap.add_argument("--alphabet", default="vt1k")
    ap.add_argument("--arm", default="main")
    ap.add_argument("--tau", type=float, default=10.0)
    ap.add_argument("--taus", default="6,8,10,12,15,20,25,30")
    ap.add_argument("--nulls", type=int, default=3)
    ap.add_argument("--null", type=int, default=-1)
    ap.add_argument("--procs", type=int, default=0)
    ap.add_argument("--threads", type=int, default=1)
    a = ap.parse_args()
    {"repro": cmd_repro, "run": cmd_run, "ladder": cmd_ladder}[a.cmd](a)


if __name__ == "__main__":
    main()
