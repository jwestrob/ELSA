#!/usr/bin/env python
"""context_units Test 3: label-free core cuts in place of module-label cuts (token landscapes).

Reading LF(kind, theta): level-2 neighbourhood (any significant support) nested with cores; a core = a level-2 locus cut
where boundary cohesion kappa < theta, then end-trimmed (factor 0.5, as rule H). No module labels and no Louvain.
  var   successor variety. For gene i and the side facing boundary k (strand-oriented: downstream if i is on +), the
        neighbours on that side of the other members of i's token family; cohesion = their mean pairwise zca10 cosine,
        (||A - y||^2 - n) / (n (n - 1)) with A the family's side sum and y = i's own neighbour, minus the gene-pair floor.
        kappa(k) = min(cohesion of gene k toward k, cohesion of gene k+1 toward k); NaN (families < 3) = no cut.
        Cutting removes boundary k.
  cont  genome-level continuity: Jaccard of the genome sets of the best significant pairs spanning boundaries k and k+1;
        a low value starts a new segment at gene k+1 (junction gene to the side with the larger support).
Scored against the same landscape's DH (module cuts) and the genus DH. Logs print odd numbers only.

usage
  cuts.py run --threads T --landscapes TAG=RDIR[:REFTAG],...   REFTAG = scores/<REFTAG>__DH.parquet (default: stored genus DH)
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ctxrec as C  # noqa: E402

B, BR, RL = C.B, C.BR, C.RL
CU = C.CU
THETA = {"var": (0.05, 0.1, 0.2, 0.3, 0.4), "cont": (0.1, 0.2, 0.3, 0.5)}
log = C.log


@njit(cache=True)
def side_sums(tok, uid, X, strand, cid, T):
    """A[0, t] = sum of downstream neighbours' vectors over members of token t, A[1, t] upstream; counts."""
    d = X.shape[1]
    n = len(tok)
    A = np.zeros((2, T, d), dtype=np.float32)
    cnt = np.zeros((2, T), dtype=np.int64)
    for m in range(n):
        t = tok[m]
        if t < 0:
            continue
        for side in range(2):
            step = 1 if (strand[m] == 1) == (side == 0) else -1
            nm = m + step
            if nm < 0 or nm >= n or cid[nm] != cid[m]:
                continue
            q = uid[nm]
            cnt[side, t] += 1
            for j in range(d):
                A[side, t, j] += X[q, j]
    return A, cnt


@njit(parallel=True, cache=True)
def side_norms(A):
    out = np.zeros((A.shape[0], A.shape[1]))
    for t in prange(A.shape[1]):
        for s in range(A.shape[0]):
            x = 0.0
            for j in range(A.shape[2]):
                x += np.float64(A[s, t, j]) * A[s, t, j]
            out[s, t] = x
    return out


@njit(parallel=True, cache=True)
def gene_cohesion(tok, uid, X, strand, cid, A, cnt, normA2, c0, out_right, out_left):
    """Cohesion of gene i toward its right boundary (i, i+1) and its left boundary (i-1, i)."""
    n = len(tok)
    d = X.shape[1]
    for i in prange(n):
        out_right[i] = np.nan
        out_left[i] = np.nan
        t = tok[i]
        if t < 0:
            continue
        for which in range(2):          # 0: right boundary (neighbour i+1), 1: left boundary (neighbour i-1)
            nb = i + 1 if which == 0 else i - 1
            if nb < 0 or nb >= n or cid[nb] != cid[i]:
                continue
            # side of gene i facing the neighbour: downstream if (+ strand and right) or (- strand and left)
            side = 0 if (strand[i] == 1) == (which == 0) else 1
            nn = cnt[side, t] - 1
            if nn < 2:
                continue
            q = uid[nb]
            dot = 0.0
            for j in range(d):
                dot += A[side, t, j] * X[q, j]
            s2 = normA2[side, t] - 2.0 * dot + 1.0
            rho = (s2 - nn) / (nn * (nn - 1.0))
            rho = (rho - c0) / (1.0 - c0)
            if which == 0:
                out_right[i] = rho
            else:
                out_left[i] = rho


@njit(parallel=True, cache=True)
def gene_cohesion_knn(uq, I, Kh, mptr, members, uid, X, strand, cid, c0, cap_h, out_right, out_left):
    """Embedding-mode cohesion: the homolog set of gene i = other genes with i's sequence plus the genes of its top-Kh
    unique zca10 neighbours (at most cap_h genes); otherwise as gene_cohesion."""
    n = len(uq)
    d = X.shape[1]
    for i in prange(n):
        out_right[i] = np.nan
        out_left[i] = np.nan
        U = uq[i]
        if U < 0:
            continue
        for which in range(2):
            nb = i + 1 if which == 0 else i - 1
            if nb < 0 or nb >= n or cid[nb] != cid[i]:
                continue
            side = 0 if (strand[i] == 1) == (which == 0) else 1
            A = np.zeros(d)
            cnt = 0
            j = -1
            while j < Kh and cnt < cap_h:
                V = U if j < 0 else I[U, j]
                if V < 0:
                    break
                if j >= 0 and V == U:
                    j += 1
                    continue
                for t in range(mptr[V], mptr[V + 1]):
                    m = members[t]
                    if m == i or cnt >= cap_h:
                        continue
                    step = 1 if (strand[m] == 1) == (side == 0) else -1
                    nm = m + step
                    if nm < 0 or nm >= n or cid[nm] != cid[m]:
                        continue
                    q = uid[nm]
                    for k in range(d):
                        A[k] += X[q, k]
                    cnt += 1
                j += 1
            if cnt < 2:
                continue
            s2 = 0.0
            for k in range(d):
                s2 += A[k] * A[k]
            rho = (s2 - cnt) / (cnt * (cnt - 1.0))
            rho = (rho - c0) / (1.0 - c0)
            if which == 0:
                out_right[i] = rho
            else:
                out_left[i] = rho


@njit(cache=True)
def var_cores(b, kappa, s2, e2, theta):
    """Runs of boundaries inside each level-2 locus with kappa >= theta (NaN counts as >= theta)."""
    cap = 0
    for i in range(len(s2)):
        cap += e2[i] - s2[i] + 1
    os_ = np.empty(cap, np.int64)
    oe = np.empty(cap, np.int64)
    n = 0
    for i in range(len(s2)):
        s, e = s2[i], e2[i]
        k = s
        while k < e:
            if not (kappa[k] < theta):
                k0 = k
                while k < e and not (kappa[k] < theta):
                    k += 1
                os_[n] = k0
                oe[n] = k
                n += 1
            else:
                k += 1
    return os_[:n], oe[:n]


@njit(cache=True)
def cont_cores(b, kc, s2, e2, theta):
    """Split each level-2 locus where continuity between boundaries j and j+1 (kc[j]) < theta; junction gene j+1 goes
    to the side whose junction boundary has the larger support."""
    cap = 0
    for i in range(len(s2)):
        cap += e2[i] - s2[i] + 1
    os_ = np.empty(cap, np.int64)
    oe = np.empty(cap, np.int64)
    n = 0
    for i in range(len(s2)):
        s, e = s2[i], e2[i]
        seg = s
        for j in range(s, e - 1):
            if kc[j] < theta:
                if b[j] >= b[j + 1]:
                    le, rs = j + 1, j + 2
                else:
                    le, rs = j, j + 1
                os_[n] = seg
                oe[n] = le
                n += 1
                seg = rs
        os_[n] = seg
        oe[n] = e
        n += 1
    return os_[:n], oe[:n]


@njit(parallel=True, cache=True)
def continuity(arg, bits, n):
    out = np.full(n, np.nan)
    for k in prange(n - 1):
        a, c = arg[k], arg[k + 1]
        if a < 0 or c < 0:
            continue
        if a == c:
            out[k] = 1.0
            continue
        inter, union = 0, 0
        for w in range(bits.shape[1]):
            x, y = bits[a, w], bits[c, w]
            xi, xu = x & y, x | y
            while xi:
                xi &= xi - np.uint64(1)
                inter += 1
            while xu:
                xu &= xu - np.uint64(1)
                union += 1
        out[k] = inter / union if union else np.nan
    return out


def nest(L, cs, ce):
    ok = ce > cs
    cs, ce = cs[ok], ce[ok]
    ts, te = B.trim_ends(L.b, cs, ce, 0.5)
    se = np.unique(np.c_[np.r_[L.s2, ts], np.r_[L.e2, te]], axis=0)
    return se[:, 0], se[:, 1]


def cmd_run(a):
    import numba
    numba.set_num_threads(a.threads)
    t0 = time.time()
    base = object.__new__(C.Base)
    u = RL.Universe(C.U, lineage_genome=True)
    assert np.array_equal(u.gid, np.arange(u.n))
    base.u = u
    t = pd.read_parquet(C.TOK)
    tok = np.full(u.n, -1, dtype=np.int64)
    tok[t.gid.to_numpy()] = t.token.to_numpy()
    base.tok = tok
    base.cid = np.repeat(np.arange(len(u.c_start)), u.c_end - u.c_start).astype(np.int64)
    base.key, base.left, base.dist, base.T = RL.colocalization(u, tok, C.W_ENGINE)
    st = RL.pair_statistics(u, tok, base.key, base.left, C.W_ENGINE, base.T)
    base.st = st
    kfile = CU / "ctx" / "kappa_var_tokens.npy"
    if kfile.exists() and a.reuse_kappa:       # landscape-independent (token families + vectors): reuse
        kappa_var = np.load(kfile).astype(np.float64)
        desc = dict(reused=str(kfile))
    else:
        strand = pd.read_parquet(C.U / "genes.parquet", columns=["gid", "strand"]).sort_values("gid").strand.to_numpy().astype(np.int64)
        X, uid, _ = C.load_vectors(base, a.threads)
        # gene-pair floor (different genomes)
        rng = np.random.default_rng(17)
        ga = rng.integers(0, u.n, 1_000_000)
        gb = rng.integers(0, u.n, 1_000_000)
        ok = u.gc[ga] != u.gc[gb]
        o = np.empty(int(ok.sum()))
        C.gene_pair_cos(ga[ok], gb[ok], X, uid, np.zeros((u.ng, X.shape[1])), u.gc.astype(np.int64), False, o)
        c0 = float(np.nanmean(o))
        A, cnt = side_sums(tok, uid, X, strand, base.cid, base.T)
        normA2 = side_norms(A)
        kr = np.empty(u.n)
        kl = np.empty(u.n)
        gene_cohesion(tok, uid, X, strand, base.cid, A, cnt, normA2, c0, kr, kl)
        del A, X
        kappa_var = np.fmin(kr[:-1], kl[1:])           # boundary k = (k, k+1); fmin ignores one NaN
        same_contig = base.cid[:-1] == base.cid[1:]
        kappa_var = np.r_[np.where(same_contig, kappa_var, np.nan), np.nan]
        desc = dict(gene_pair_floor=c0, kappa_var_q=np.nanquantile(kappa_var, [.05, .25, .5, .75, .95]).tolist(),
                    kappa_var_nan=float(np.isnan(kappa_var[:-1][same_contig]).mean()))
        log("cohesion", desc, f"{time.time()-t0:.0f}s")
    if not desc.get("reused"):
        np.save(CU / "ctx" / "kappa_var_tokens.npy", kappa_var.astype(np.float32))
    sc = C.Scorer(base)
    gcode = sc.gcode
    out = dict(desc=desc, landscapes={})
    for spec in a.landscapes.split(","):
        ltag, rest = spec.split("=")
        rdir, _, reftag = rest.partition(":")
        L = sc.landscape(rdir)
        if reftag.startswith("lf_"):          # a lineagefree token run: its stored DH scores and loci
            ref = pd.read_parquet(C.LF / "scores" / f"tok_{reftag[3:]}__DH.parquet")
            rl = pd.read_parquet(C.LF / "loci" / f"tok_{reftag[3:]}__DH.parquet")
        elif reftag:
            ref = pd.read_parquet(CU / "scores" / f"{reftag}__DH.parquet")
            rl = pd.read_parquet(CU / "loci" / f"{reftag}__DH.parquet")
        else:
            ref = pd.read_parquet(C.REF_SCORES / f"{C.REF_NAME['DH']}.parquet")
            rl = pd.read_parquet(C.REF_LOCI / f"{C.REF_NAME['DH']}.parquet")
        if "excess" not in ref:
            ref["excess"] = ref.f1 - ref.control_mean_f1
        ref = ref.set_index("instance_id")
        rsc = B.ribo_score(sc.runs, rl.gid_s.to_numpy(), rl.gid_e.to_numpy())
        rib_ref = B.ribo_summary(sc.runs, rsc)
        # continuity from the significant pairs' genome sets
        sp = L.sp
        k_of, g_of = st["incidence"]
        sig = np.zeros(len(st["key"]), dtype=bool)
        sig[np.searchsorted(st["key"], sp.key.to_numpy())] = True
        inc_idx = np.searchsorted(st["key"], k_of)
        keep = sig[inc_idx]
        sig_rank = np.cumsum(sig) - 1
        bits = RL.fill_bitsets(sig_rank[inc_idx[keep]], g_of[keep], int(sig.sum()), (u.ng + 63) // 64)
        kc = continuity(L.arg, bits, u.n)
        del bits
        ld = dict(rdir=rdir, ref=reftag or "genus_DH", configs={},
                  kappa_cont_q=np.nanquantile(kc, [.05, .25, .5, .75, .95]).tolist())
        for kind, thetas in THETA.items():
            for th in thetas:
                t1 = time.time()
                if kind == "var":
                    cs, ce = var_cores(L.b, kappa_var, L.s2, L.e2, th)
                else:
                    cs, ce = cont_cores(L.b, kc, L.s2, L.e2, th)
                s, e = nest(L, cs, ce)
                df = L.table(s, e)
                name = f"{ltag}__LF{kind}{th:g}"
                df.to_parquet(CU / "loci" / f"{name}.parquet", index=False)
                per, _ = B.score_table(B.PKG, df, a.threads)
                per.to_parquet(CU / "scores" / f"{name}.parquet", index=False)
                pi = per.set_index("instance_id")
                g = pi.genome.map(gcode).to_numpy()
                summ = {sp_: B.summarize_per(pi[g % 2 == (1 if sp_ == "odd" else 0)].reset_index()) for sp_ in ("odd", "even")}
                scr = B.ribo_score(sc.runs, df.gid_s.to_numpy(), df.gid_e.to_numpy())
                er = sc.even_runs
                item = dict(kind=kind, theta=th, summary=summ, ribo=B.ribo_summary(sc.runs, scr),
                            ribo_odd=B.ribo_summary(sc.runs[~er].reset_index(drop=True), scr[~er].reset_index(drop=True)),
                            loci=B.locus_stats(df.gid_s.to_numpy(), df.gid_e.to_numpy(), u.n),
                            cores=int(len(cs)),
                            vs_same_even=BR.paired(pi, ref, gcode, "even"), vs_same_odd=BR.paired(pi, ref, gcode, "odd"),
                            ribo_vs_same=BR.ribo_paired(scr, rsc))
                gref = sc.refs["DH"]["genus"]
                item["vs_genus_even"] = BR.paired(pi, gref[0], gcode, "even")
                item["vs_genus_odd"] = BR.paired(pi, gref[0], gcode, "odd")
                item["conditions_odd"] = C.cond_odd(item["vs_same_odd"], item["ribo_odd"])
                item["conditions_even_vs_same"] = C.cond_even(item["vs_same_even"], item["ribo"], rib_ref)
                item["seconds"] = round(time.time() - t1, 1)
                ld["configs"][f"{kind}{th:g}"] = item
                od = summ["odd"]
                log(name, "ODD summed", round(od["summed_DF_TXSS_excess"], 4), {c[:4]: round(od[c]["f1"], 3) for c in C.CALLERS},
                    "d vs same DH", [round(x, 4) for x in item["vs_same_odd"]["summed"]],
                    {c[:4]: round(item["vs_same_odd"][c]["d_f1"][0], 4) for c in C.CALLERS},
                    "ribo odd", item["ribo_odd"]["recovered"], item["ribo_odd"]["resolved"], item["ribo_odd"]["parvarchaeaceae"],
                    "cond odd", item["conditions_odd"], f"{item['seconds']}s")
        out["landscapes"][ltag] = ld
        C.jdump(CU / "results" / "full" / f"cuts_{a.out_tag}.json", out)
        odd = dict(desc=desc, landscapes={k: dict(rdir=v["rdir"], ref=v["ref"], kappa_cont_q=v["kappa_cont_q"], configs={
            c: {kk: vv for kk, vv in it.items() if kk.endswith("_odd") or kk in ("kind", "theta", "loci", "cores")}
            | {"summary_odd": it["summary"]["odd"]} for c, it in v["configs"].items()}) for k, v in out["landscapes"].items()})
        C.jdump(CU / "results" / f"cuts_{a.out_tag}.odd.json", odd)
    log("done", f"{time.time()-t0:.0f}s")


def cmd_emb(a):
    """Label-free variety cuts on the embedding-only round-2 landscape (genus, stored), homologs from the zca10 kNN."""
    import numba
    numba.set_num_threads(a.threads)
    t0 = time.time()
    sys.path.insert(0, str(C.RD / "softrec/code"))
    import readings2 as RG
    ctx = RG.Ctx()
    u = ctx.u
    if a.cand_dir:                          # another embedding landscape (e.g. Test 2's), same reading machinery
        cdir = Path(a.cand_dir)
        sidx = np.load(cdir / "sidx.npy")
        land = RG.Land(a.name, ctx.left[sidx], ctx.d[sidx], np.load(cdir / "value.npy"), np.load(cdir / "b.npy"),
                       np.load(cdir / "arg.npy"))
    else:
        land = RG.land_candidate(ctx, "rw1000_e50")
    kfile = CU / "ctx" / f"kappa_var_knn{a.kh}.npy"
    cid = np.repeat(np.arange(len(u.c_start)), u.c_end - u.c_start).astype(np.int64)
    same = cid[:-1] == cid[1:]
    if kfile.exists() and a.cand_dir:          # cohesion is landscape-independent: reuse the stored array
        kappa = np.load(kfile).astype(np.float64)
        desc = dict(reused=str(kfile), kh=a.kh, kappa_q=np.nanquantile(kappa, [.05, .25, .5, .75, .95]).tolist())
    else:
        strand = pd.read_parquet(C.U / "genes.parquet", columns=["gid", "strand"]).sort_values("gid").strand.to_numpy().astype(np.int64)
        X, uid, _ = C.load_vectors(type("b", (), {"u": u})(), a.threads)
        I = np.ascontiguousarray(np.load(C.RD / "softrec/knn/zca10__k300__I.npy", mmap_mode="r")[:, :a.kh])
        ok = uid >= 0
        order = np.flatnonzero(ok)[np.argsort(uid[ok], kind="stable")]
        mptr = np.r_[0, np.cumsum(np.bincount(uid[ok], minlength=X.shape[0]))].astype(np.int64)
        cid = np.repeat(np.arange(len(u.c_start)), u.c_end - u.c_start).astype(np.int64)
        rng = np.random.default_rng(17)
        ga = rng.integers(0, u.n, 1_000_000)
        gb = rng.integers(0, u.n, 1_000_000)
        okp = u.gc[ga] != u.gc[gb]
        o = np.empty(int(okp.sum()))
        C.gene_pair_cos(ga[okp], gb[okp], X, uid, np.zeros((u.ng, X.shape[1])), u.gc.astype(np.int64), False, o)
        c0 = float(np.nanmean(o))
        kr = np.empty(u.n)
        kl = np.empty(u.n)
        gene_cohesion_knn(uid, I, a.kh, mptr, order.astype(np.int64), uid, X, strand, cid, c0, a.cap_h, kr, kl)
        del X, I
        kappa = np.fmin(kr[:-1], kl[1:])
        same = cid[:-1] == cid[1:]
        kappa = np.r_[np.where(same, kappa, np.nan), np.nan]
        np.save(CU / "ctx" / f"kappa_var_knn{a.kh}.npy", kappa.astype(np.float32))      # landscape-independent
        desc = dict(gene_pair_floor=c0, kh=a.kh, cap_h=a.cap_h, kappa_q=np.nanquantile(kappa, [.05, .25, .5, .75, .95]).tolist(),
                    kappa_nan=float(np.isnan(kappa[:-1][same]).mean()))
    log("cohesion (kNN homologs)", desc, f"{time.time()-t0:.0f}s")
    b = land.b
    s2, e2 = B.R.runs_at_least(b, u.c_start, u.c_end, 2.0, 0)
    ref = pd.read_parquet(a.ref_scores or (C.RD / "softrec/r2/scores/all/selected__rw1000_e50__l2000__0.3__DH.parquet"))
    ref["excess"] = ref.f1 - ref.control_mean_f1
    ref = ref.set_index("instance_id")
    runs = B.ribo_runs()
    er = (runs.genome_code % 2 == 0).to_numpy()
    out = dict(desc=desc, configs={})
    for th in THETA["var"]:
        t1 = time.time()
        cs, ce = var_cores(b, kappa, s2, e2, th)
        okc = ce > cs
        ts, te = B.trim_ends(b, cs[okc], ce[okc], 0.5)
        se = np.unique(np.c_[np.r_[s2, ts], np.r_[e2, te]], axis=0)
        df = ctx.table(se[:, 0], se[:, 1], np.full(u.n, -1, np.int64))
        name = f"{a.name}__LFvar{th:g}"
        df.to_parquet(CU / "loci" / f"{name}.parquet", index=False)
        per, _ = B.score_table(B.PKG, df, a.threads)
        per.to_parquet(CU / "scores" / f"{name}.parquet", index=False)
        pi = per.set_index("instance_id")
        g = pi.genome.map(ctx.gcode).to_numpy()
        summ = {sp_: B.summarize_per(pi[g % 2 == (1 if sp_ == "odd" else 0)].reset_index()) for sp_ in ("odd", "even")}
        scr = B.ribo_score(runs, df.gid_s.to_numpy(), df.gid_e.to_numpy())
        item = dict(theta=th, summary=summ, ribo=B.ribo_summary(runs, scr),
                    ribo_odd=B.ribo_summary(runs[~er].reset_index(drop=True), scr[~er].reset_index(drop=True)),
                    loci=B.locus_stats(df.gid_s.to_numpy(), df.gid_e.to_numpy(), u.n),
                    vs_same_even=BR.paired(pi, ref, ctx.gcode, "even"), vs_same_odd=BR.paired(pi, ref, ctx.gcode, "odd"))
        item["conditions_odd"] = C.cond_odd(item["vs_same_odd"], item["ribo_odd"])
        item["seconds"] = round(time.time() - t1, 1)
        out["configs"][f"var{th:g}"] = item
        log(name, "ODD summed", round(summ["odd"]["summed_DF_TXSS_excess"], 4),
            "d vs round-2 DH", [round(x, 4) for x in item["vs_same_odd"]["summed"]],
            {c[:4]: round(item["vs_same_odd"][c]["d_f1"][0], 4) for c in C.CALLERS},
            "ribo odd", item["ribo_odd"]["recovered"], item["ribo_odd"]["resolved"], item["ribo_odd"]["parvarchaeaceae"],
            "cond odd", item["conditions_odd"], f"{item['seconds']}s")
    C.jdump(CU / "results" / "full" / f"cuts_emb_{a.name}.json", out)
    odd = dict(desc=desc, configs={k: {kk: vv for kk, vv in v.items() if kk.endswith("_odd") or kk in ("theta", "loci")}
                                   | {"summary_odd": v["summary"]["odd"]} for k, v in out["configs"].items()})
    C.jdump(CU / "results" / f"cuts_emb_{a.name}.odd.json", odd)
    log("done", f"{time.time()-t0:.0f}s")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("run", "emb"))
    ap.add_argument("--threads", type=int, required=True)
    ap.add_argument("--landscapes", default="")
    ap.add_argument("--out-tag", default="")
    ap.add_argument("--kh", type=int, default=50)
    ap.add_argument("--cap-h", type=int, default=200)
    ap.add_argument("--cand-dir", default="")
    ap.add_argument("--ref-scores", default="")
    ap.add_argument("--name", default="sr_round2")
    ap.add_argument("--reuse-kappa", action="store_true")
    a = ap.parse_args()
    {"run": cmd_run, "emb": cmd_emb}[a.cmd](a)


if __name__ == "__main__":
    main()
