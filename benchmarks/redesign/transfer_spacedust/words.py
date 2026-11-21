#!/usr/bin/env python
"""Stage-1 family words on the Spacedust collection (workstream transfer_spacedust).

Learner = vocab/vocab2a.py run_vocab2a without classes or context vectors, copied with its kernels (vocab.py
expected_groups / gain / rebuild / occurrences, vocab2a assign_f / syn_overlap / syn_neff_keys / synteny_matrix); logic
unchanged. Modes:
  group  n = genome groups (data-derived, emb_hom average linkage 0.75), E over groups          (P1)
  syn    n = genomes, Lambda_eff = Lambda_genome x N_eff / n, N_eff = n^2 / sum_{g,h} S(g,h)    (P2)
         (vocab2a mode neff, ctx syn), commit needs N_eff >= 2
  genome n = genomes, no correction                                                              (diagnostic)
Added for this workstream (no effect on the parse): the round-0 significant occurrences (gene k, distance d) are saved
(sig0.npz) for the connected-component baseline (amendment_0).

usage
  words.py ladder --specs syn,group --taus 4,6,... --nulls 3 --threads T [--tag X]
  words.py run --spec syn --tau 8 [--null r]
  words.py smoke                       small synthetic + first-genomes checks (login node)
Runs: $TS/runs/<spec>/tau<t>/<obs|nullr>/ (words.parquet, inst.npz, live.npz, b0.npy, sig0.npz, run.json).
"""
import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit, prange

RD = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111")
TS = RD / "transfer_spacedust"
UNIV = RD / "universe"
TOK = Path(os.environ.get("TS_TOK", str(TS / "tok/mmseqs__seq__e3c50s75_rescue3__spacedust.parquet")))
GROUPS = TS / "groups/emb_hom_average_t0.75.parquet"
SHIFT = np.int64(31)
MASK = (np.int64(1) << SHIFT) - 1
LADDER = (4, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 70, 100)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def rss_gb():
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024, 2)


# =========================================================================== kernels (vocab.py, copied)
@njit(cache=True)
def expected_groups(lo, hi, indptr, rank, cnt, m_rank, grp_of_rank, w):
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
def rebuild(ns, ne, sym, term, ninst, ck, cd, csym, inst_base):
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


# =========================================================================== kernels (vocab2a.py, copied)
@njit(cache=True)
def syn_overlap(kptr, kgen, n_genomes):
    O = np.zeros((n_genomes, n_genomes), dtype=np.int32)
    for r in range(len(kptr) - 1):
        a, b = kptr[r], kptr[r + 1]
        for i in range(a, b):
            gi = kgen[i]
            O[gi, gi] += 1
            for j in range(i + 1, b):
                O[gi, kgen[j]] += 1
    return O


@njit(parallel=True, cache=True)
def syn_neff_keys(ptr, gen, S):
    nk = len(ptr) - 1
    out = np.zeros(nk, dtype=np.float64)
    for r in prange(nk):
        m = 0
        last = -1
        for o in range(ptr[r], ptr[r + 1]):
            if gen[o] != last:
                m += 1
                last = gen[o]
        G = np.empty(m, dtype=np.int64)
        q = 0
        last = -1
        for o in range(ptr[r], ptr[r + 1]):
            if gen[o] != last:
                G[q] = gen[o]
                q += 1
                last = gen[o]
        tot = float(m)
        for i in range(m):
            for j in range(i + 1, m):
                tot += 2.0 * S[G[i], G[j]]
        out[r] = m * m / tot
    return out


@njit(cache=True)
def assign_f(occ_k, occ_d, occ_r, occ_g, key_E, key_f, tau, m, n_groups, min_n, min_neff):
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
        lam = gain(float(ng), key_E[r]) * key_f[r]
        if ng >= min_n and ng * key_f[r] >= min_neff and lam >= tau:
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


def synteny_matrix(tok, cc, gc, min_n=2, w=3):
    """S(g,h): share of g's round-0 adjacent token pairs also adjacent in h, symmetrised, diagonal 1 (vocab2a, copied)."""
    n_genomes = int(gc.max()) + 1
    T = int(tok.max()) + 1
    tg = np.unique(tok * n_genomes + gc)
    cnt = np.bincount(tg // n_genomes, minlength=T)
    sym = np.where(cnt[tok] >= min_n, tok, -1).astype(np.int64)
    n = len(tok)
    k, d, key = occurrences(np.arange(n, dtype=np.int64), cc, sym, np.ones(n, dtype=np.bool_), w)
    kg = np.unique(np.c_[key, gc[k]], axis=0)
    o = np.lexsort((kg[:, 1], kg[:, 0]))
    kg = kg[o]
    ch = np.flatnonzero(np.diff(kg[:, 0])) + 1
    kptr = np.r_[0, ch, len(kg)].astype(np.int64)
    O = syn_overlap(kptr, kg[:, 1].astype(np.int64), n_genomes).astype(np.float64)
    O = O + np.triu(O, 1).T
    diag = np.diag(O).copy()
    Sm = O / np.maximum(diag[:, None], 1.0)
    Sm = 0.5 * (Sm + Sm.T)
    np.fill_diagonal(Sm, 1.0)
    return Sm.astype(np.float32)


# =========================================================================== learner (vocab2a.run_vocab2a, copied)
def run_words(tok, cc, gc, grp_in, mode, tau, S=None, w=3, max_rounds=500, min_n=2, verbose=False):
    """mode: group | genome | syn. Logic of vocab2a.run_vocab2a for (group, none), (genome, none), (neff, syn)."""
    t0 = time.time()
    n = len(tok)
    n_genomes = int(gc.max()) + 1
    grp = grp_in if mode == "group" else np.arange(n_genomes, dtype=np.int64)
    ng = int(grp.max()) + 1
    order = np.lexsort((np.arange(n_genomes), grp))
    rank_of_genome = np.empty(n_genomes, dtype=np.int64)
    rank_of_genome[order] = np.arange(n_genomes)
    grp_of_rank = grp[order].astype(np.int64)
    T = int(tok.max()) + 1
    tg = np.unique(tok * ng + grp[gc])
    tok_groups = np.bincount(tg // ng, minlength=T)
    sym = np.where(tok_groups[tok] >= min_n, tok, -1).astype(np.int64)
    use_neff = mode == "syn"
    floor = None
    if use_neff:
        off = S[~np.eye(len(S), dtype=bool)]
        floor = dict(S_offdiag_q50_q90_q99=np.quantile(off, [.5, .9, .99]).tolist(), S_offdiag_mean=float(off.mean()))
    ns = np.arange(n, dtype=np.int64)
    ne = ns.copy()
    term = np.ones(n, dtype=np.bool_)
    ninst = np.full(n, -1, dtype=np.int64)
    ncont, ngen = cc.copy(), gc.copy()
    Ssym = T
    words, inst_parts, rounds = [], [], []
    n_inst = 0
    b0 = np.zeros(n, dtype=np.bool_)
    sig0_k = np.zeros(0, np.int64)
    sig0_d = np.zeros(0, np.int64)
    rnd = 0
    while rnd < max_rounds:
        tr = time.time()
        k, d, key = occurrences(ns, ncont, sym, term, w)
        committed_round = 0
        if len(k) == 0:
            break
        ukey, kinv = np.unique(key, return_inverse=True)
        g_occ = grp[ngen[k]]
        kg = np.unique(kinv * ng + g_occ)
        nG = np.bincount(kg // ng, minlength=len(ukey))
        cand = np.flatnonzero(nG >= min_n)
        lo_s, hi_s = ukey[cand] >> SHIFT, ukey[cand] & MASK
        live = sym >= 0
        sr = np.unique(sym[live] * n_genomes + rank_of_genome[ngen[live]], return_counts=True)
        s_of, r_of = sr[0] // n_genomes, sr[0] % n_genomes
        indptr = np.searchsorted(s_of, np.arange(Ssym + 1)).astype(np.int64)
        m_rank = np.bincount(rank_of_genome[ngen], minlength=n_genomes).astype(np.float64)
        E = expected_groups(lo_s, hi_s, indptr, r_of.astype(np.int64), sr[1].astype(np.float64), m_rank, grp_of_rank, float(w))
        nc_ = nG[cand]
        lam = np.where(nc_ > E, nc_ * np.log(nc_ / np.maximum(E, 1e-300)) - nc_ + E, 0.0)
        fct = np.ones(len(cand))
        neff_info = {}
        if use_neff:
            pre = np.flatnonzero(lam >= tau)
            rank_k = np.full(len(ukey), -1, dtype=np.int64)
            rank_k[cand[pre]] = np.arange(len(pre))
            orr = rank_k[kinv]
            sel = np.flatnonzero(orr >= 0)
            o = np.lexsort((ngen[k[sel]], orr[sel]))
            sel = sel[o]
            ptr = np.searchsorted(orr[sel], np.arange(len(pre) + 1)).astype(np.int64)
            gen_i = ngen[k[sel]]
            nn = nc_[pre].astype(np.float64)
            neff = np.minimum(syn_neff_keys(ptr, gen_i, S), nn)
            fct[:] = 0.0
            fct[pre] = neff / nn
            lam = lam * fct
            neff_info = dict(pre=int(len(pre)), neff_q=np.quantile(neff, [.1, .5, .9]).tolist() if len(neff) else [])
            pas = (lam >= tau) & (nc_ * fct >= 2)
        else:
            pas = lam >= tau
        if rnd == 0:
            sig_key = np.zeros(len(ukey), dtype=bool)
            sig_key[cand[pas]] = True
            si = sig_key[kinv]
            cov = np.zeros(n + 1, dtype=np.int32)
            np.add.at(cov, ns[k[si]], 1)
            np.add.at(cov, ne[k[si] + d[si]], -1)
            b0 = np.cumsum(cov)[:n] > 0
            sig0_k, sig0_d = ns[k[si]].copy(), d[si].copy()
        if pas.any():
            ck_idx = cand[pas]
            ordk = np.lexsort((ck_idx, -lam[pas]))
            ck_idx = ck_idx[ordk]
            cE, clam0, clo, chi = E[pas][ordk], lam[pas][ordk], lo_s[pas][ordk], hi_s[pas][ordk]
            cf, cn0 = fct[pas][ordk], nG[ck_idx]
            rank_of_key = np.full(len(ukey), -1, dtype=np.int64)
            rank_of_key[ck_idx] = np.arange(len(ck_idx))
            orr = rank_of_key[kinv]
            sel = np.flatnonzero(orr >= 0)
            o = np.lexsort((k[sel], d[sel], orr[sel]))
            sel = sel[o]
            c_k, c_d, c_r, key_ok, key_n, key_lam, key_occ = assign_f(
                k[sel], d[sel], orr[sel], g_occ[sel], cE, cf, float(tau), len(ns), ng, min_n, 2.0 if use_neff else 0.0)
            committed_round = int(key_ok.sum())
            if committed_round:
                new_sym = np.full(len(ck_idx), -1, dtype=np.int64)
                okr = np.flatnonzero(key_ok)
                new_sym[okr] = Ssym + np.arange(len(okr))
                words.append(pd.DataFrame(dict(sym=new_sym[okr], a=clo[okr], b=chi[okr], lam=key_lam[okr], n=key_n[okr],
                                               E=cE[okr], f=cf[okr], lam0=clam0[okr], n0=cn0[okr], occ=key_occ[okr],
                                               round=rnd)))
                Ssym += len(okr)
                oc = np.argsort(c_k, kind="stable")
                c_k, c_d, c_s = c_k[oc], c_d[oc], new_sym[c_r[oc]]
                ns, ne, sym, term, ninst, i_sym, i_s, i_e, i_a, i_b = rebuild(ns, ne, sym, term, ninst, c_k, c_d, c_s, n_inst)
                ncont, ngen = cc[ns], gc[ns]
                inst_parts.append((i_sym, i_s, i_e, i_a, i_b, np.full(len(i_sym), rnd, dtype=np.int64)))
                n_inst += len(i_sym)
        rounds.append(dict(round=rnd, occurrences=int(len(k)), keys=int(len(ukey)), keys_n2=int(len(cand)),
                           candidates=int(pas.sum()), committed=committed_round, live_nodes=int(len(ns)), rss_gb=rss_gb(),
                           seconds=round(time.time() - tr, 1), **neff_info))
        if verbose:
            log("round", rnd, {k_: v for k_, v in rounds[-1].items() if k_ != "round"}, f"{time.time()-t0:.0f}s")
        rnd += 1
        if not committed_round:
            break
    words = pd.concat(words, ignore_index=True) if words else pd.DataFrame(
        columns=["sym", "a", "b", "lam", "n", "E", "f", "lam0", "n0", "occ", "round"])
    if inst_parts:
        inst = {nm: np.concatenate([p[i] for p in inst_parts]) for i, nm in enumerate(("sym", "s", "e", "a", "b", "round"))}
    else:
        inst = {nm: np.zeros(0, dtype=np.int64) for nm in ("sym", "s", "e", "a", "b", "round")}
    inst["label"] = inst["sym"].copy()
    return dict(words=words, inst=inst, live=dict(ns=ns, ne=ne, sym=sym, term=term, ninst=ninst), rounds=rounds,
                b0=b0, sig0=dict(k=sig0_k, d=sig0_d), T=T, seconds=round(time.time() - t0, 1), floor=floor,
                rss_gb=rss_gb(), n_background_tokens=int((tok_groups < min_n).sum()))


def summary(res, n):
    live = res["live"]
    top = ~live["term"]
    cov = int((live["ne"][top] - live["ns"][top] + 1).sum())
    return dict(word_types=int(len(res["words"])), instances=int(len(res["inst"]["sym"])), top_instances=int(top.sum()),
                top_labels=int(len(np.unique(live["sym"][top]))), genes_in_top=cov, genes_in_top_frac=cov / n,
                rounds=len(res["rounds"]), seconds=res["seconds"], rss_gb=res["rss_gb"], floor=res["floor"],
                sig0_occurrences=int(len(res["sig0"]["k"])), b0_boundaries=int(res["b0"].sum()))


# =========================================================================== data
def load_base():
    g = pd.read_parquet(UNIV / "genes.parquet", columns=["gid", "genome_code", "contig_code"])
    assert np.array_equal(g.gid.to_numpy(), np.arange(len(g)))
    cc = g.contig_code.to_numpy(np.int64)
    gc = g.genome_code.to_numpy(np.int64)
    assert np.all(np.diff(cc) >= 0) and np.all(np.diff(gc) >= 0)
    t = pd.read_parquet(TOK)
    tok = np.full(len(g), -1, dtype=np.int64)
    tok[t.gid.to_numpy()] = t.token.to_numpy()
    assert (tok >= 0).all()
    return tok, cc, gc


def load_groups(path, n_genomes):
    m = pd.read_parquet(path, columns=["genome_code", "group"]).sort_values("genome_code")
    assert np.array_equal(m.genome_code.to_numpy(), np.arange(n_genomes))
    return pd.factorize(m.group.to_numpy())[0].astype(np.int64)


def genome_bounds(gc):
    ch = np.flatnonzero(np.diff(gc)) + 1
    return np.r_[0, ch].astype(np.int64), np.r_[ch, len(gc)].astype(np.int64)


def shuffled(tok, gc, r):
    """vocab2a.build_corpus null: tokens permuted within each genome (contig sizes kept), rng 1000 + r."""
    rng = np.random.default_rng(1000 + r)
    perm = np.arange(len(tok), dtype=np.int64)
    gs, ge = genome_bounds(gc)
    for s, e in zip(gs, ge):
        perm[s:e] = s + rng.permutation(e - s)
    return tok[perm]


def save(res, out, full):
    out.mkdir(parents=True, exist_ok=True)
    res["words"].to_parquet(out / "words.parquet", index=False)
    if full:
        np.savez(out / "inst.npz", **res["inst"])
        np.savez(out / "live.npz", **res["live"])
        np.save(out / "b0.npy", res["b0"])
        np.savez(out / "sig0.npz", **res["sig0"])


# =========================================================================== ladder
_BASE = {}


def _one(job):
    spec, tau, r, root = job
    import numba
    numba.set_num_threads(1)
    tok, cc, gc = _BASE["tok"], _BASE["cc"], _BASE["gc"]
    t = tok if r < 0 else _BASE["shuf"][r]
    S = _BASE["S"][r] if spec == "syn" else None
    res = run_words(t, cc, gc, _BASE.get("grp"), spec, tau, S=S, w=_BASE.get("w", 3))
    s = summary(res, len(t))
    out = Path(root) / spec / f"tau{tau:g}" / ("obs" if r < 0 else f"null{r}")
    save(res, out, full=(r < 0))
    (out / "run.json").write_text(json.dumps(dict(spec=spec, tau=tau, null=r, summary=s, rounds=res["rounds"], T=res["T"],
                                                  n_background_tokens=res["n_background_tokens"]),
                                             indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x)) + "\n")
    log(spec, tau, "obs" if r < 0 else f"null{r}", {k: v for k, v in s.items() if k != "floor"})
    return dict(spec=spec, tau=tau, null=r, **{k: v for k, v in s.items() if k != "floor"})


def prepare(specs, nulls, groups_path=GROUPS, base=None):
    tok, cc, gc = base if base is not None else load_base()
    _BASE.update(tok=tok, cc=cc, gc=gc, shuf={r: shuffled(tok, gc, r) for r in range(nulls)})
    if "group" in specs:
        _BASE["grp"] = load_groups(groups_path, int(gc.max()) + 1)
    if "syn" in specs:
        t0 = time.time()
        _BASE["S"] = {-1: synteny_matrix(tok, cc, gc)}
        for r in range(nulls):
            _BASE["S"][r] = synteny_matrix(_BASE["shuf"][r], cc, gc)
        log("synteny matrices", f"{time.time()-t0:.0f}s")


def cmd_ladder(a):
    import multiprocessing as mp
    specs = a.specs.split(",")
    taus = [float(x) for x in a.taus.split(",")]
    t0 = time.time()
    prepare(specs, a.nulls)
    _BASE["w"] = a.w           # occurrence window / E window (3 = DPANN; other values are development variants)
    root = TS / ("runs" if a.w == 3 else f"runs_w{a.w}")
    if "syn" in specs:
        np.save(TS / "results" / "synS_obs.npy", _BASE["S"][-1])
        off = _BASE["S"][-1][~np.eye(len(_BASE["S"][-1]), dtype=bool)]
        offn = _BASE["S"][0][~np.eye(len(_BASE["S"][0]), dtype=bool)]
        log("S offdiag obs q50/q90/q99/max", np.quantile(off, [.5, .9, .99, 1]).round(4).tolist(),
            "null0", np.quantile(offn, [.5, .9, .99, 1]).round(4).tolist())
    jobs = [(s, t, r, str(root)) for s in specs for t in taus for r in [-1] + list(range(a.nulls))]
    jobs.sort(key=lambda j: (j[1], j[2]))
    procs = min(a.threads, len(jobs))
    log("ladder", specs, taus, len(jobs), "runs on", procs, "processes")
    with mp.get_context("fork").Pool(procs, maxtasksperchild=1) as pool:
        rows = pool.map(_one, jobs, chunksize=1)
    df = pd.DataFrame(rows)
    out = {}
    for s in specs:
        summ = []
        for t, g in df[df.spec == s].groupby("tau"):
            o = g[g.null < 0].iloc[0]
            nl = g[g.null >= 0]
            nm = float(nl.word_types.mean())
            summ.append(dict(tau=float(t), obs_types=int(o.word_types), null_types_mean=nm,
                             fdr_types=nm / max(o.word_types, 1), obs_genes=int(o.genes_in_top),
                             genes_frac=float(o.genes_in_top_frac), null_genes_mean=float(nl.genes_in_top.mean()),
                             fdr_genes=float(nl.genes_in_top.mean() / max(o.genes_in_top, 1)), rounds=int(o.rounds),
                             obs_seconds=float(o.seconds), obs_rss_gb=float(o.rss_gb)))
        sel = [x for x in summ if x["fdr_types"] <= 0.01]
        out[s] = dict(spec=s, ladder=summ, tau_star=(min(x["tau"] for x in sel) if sel else None))
        for x in summ:
            log(s, x)
        log(s, "tau*", out[s]["tau_star"])
    fn = TS / "results" / f"ladder_{a.tag or '_'.join(specs)}.json"
    fn.write_text(json.dumps(dict(results=out, w=a.w, runs=rows, seconds=round(time.time() - t0, 1),
                                  written=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())), indent=1,
                             default=lambda x: x.item() if hasattr(x, "item") else str(x)) + "\n")
    log("wrote", fn)


def cmd_smoke(a):
    """Checks on the first genomes: runs complete, live nodes tile the genes, instances match their types; group mode with
    one genome per group equals genome mode; synteny N_eff with S = identity equals genome mode."""
    tok, cc, gc = load_base()
    keep = gc < a.genomes
    tok, cc, gc = tok[keep], cc[keep], gc[keep]
    ngn = int(gc.max()) + 1
    r1 = run_words(tok, cc, gc, np.arange(ngn, dtype=np.int64), "group", a.tau)
    r2 = run_words(tok, cc, gc, None, "genome", a.tau)
    same = len(r1["words"]) == len(r2["words"]) and np.array_equal(r1["inst"]["s"], r2["inst"]["s"])
    log("group (one genome per group) == genome:", same, len(r1["words"]))
    r3 = run_words(tok, cc, gc, None, "syn", a.tau, S=np.eye(ngn, dtype=np.float32))
    same3 = len(r3["words"]) == len(r2["words"]) and np.array_equal(r3["inst"]["s"], r2["inst"]["s"])
    log("syn with S = I == genome:", same3)
    S = synteny_matrix(tok, cc, gc)
    r4 = run_words(tok, cc, gc, None, "syn", a.tau, S=S)
    grp = np.arange(ngn) // 3
    r5 = run_words(tok, cc, gc, grp.astype(np.int64), "group", a.tau)
    for nm, r in (("genome", r2), ("syn", r4), ("group/3", r5)):
        lv = r["live"]
        assert lv["ns"][0] == 0 and lv["ne"][-1] == len(tok) - 1
        assert np.all(lv["ns"][1:] == lv["ne"][:-1] + 1) and np.all(cc[lv["ns"]] == cc[lv["ne"]])
        I = r["inst"]
        w = r["words"].set_index("sym")
        bad = 0
        for i in range(len(I["sym"])):
            x = I["sym"][I["a"][i]] if I["a"][i] >= 0 else tok[-I["a"][i] - 1]
            y = I["sym"][I["b"][i]] if I["b"][i] >= 0 else tok[-I["b"][i] - 1]
            row = w.loc[I["sym"][i]]
            bad += {int(x), int(y)} != {int(row.a), int(row.b)}
        assert bad == 0
        log("smoke", nm, summary(r, len(tok)))
    assert same and same3
    log("smoke OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["ladder", "smoke"])
    ap.add_argument("--specs", default="syn")
    ap.add_argument("--taus", default=",".join(str(x) for x in LADDER))
    ap.add_argument("--nulls", type=int, default=3)
    ap.add_argument("--tag", default="")
    ap.add_argument("--tau", type=float, default=8.0)
    ap.add_argument("--genomes", type=int, default=40)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--w", type=int, default=3)
    a = ap.parse_args()
    {"ladder": cmd_ladder, "smoke": cmd_smoke}[a.cmd](a)


if __name__ == "__main__":
    main()
