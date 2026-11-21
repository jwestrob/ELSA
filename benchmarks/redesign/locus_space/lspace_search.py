#!/usr/bin/env python
"""Locus space, part 4: search from the same space (workstream ws_locusspace).

  lspace_search.py run --threads T --rep REP
     1. reproduce the stored M1z AP (per-gene zca10 rank aggregation, frozen config W = span + 8, tau 3) by rerunning
        its scan on the even DefenseFinder queries with the stored lists (must equal 0.792 / stored per-query AP);
     2. locus-index search: one vector per nested-reading (DH) locus, query vector from the member genes, cosine,
        greedy non-overlapping hits (other genomes);
     3. window search: every window of <= span + 8 genes on every contig, scored by the cosine of its summed vector
        (linear representations only: computed from per-gene dot products and prefix sums of the local Gram);
     labels, AP and relevance from the round-1/3 benchmark code (bench.py, locus_search.py); matched wrong-type rates
     with thresholds on odd queries, recall on even queries; M1z's matched-rate recall recomputed from its stored hits.
Predeclaration: predeclared.json part_4_search.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lspace as LS  # noqa: E402
from lspace import log, dump, OUT, RD, SEED  # noqa: E402

sys.path.insert(1, str(RD / "locus_search/code"))
import bench  # noqa: E402
import locus_search as lsr  # noqa: E402
import round2  # noqa: E402

OUT1 = RD / "locus_search/out"
OUT3 = RD / "locus_search/out3"
SRCH = OUT / "search"
MAP = OUT / "map"
N_CAND = 50_000
N_KEEP = 10_000
WMAX = 60
LEVELS = (0.25, 0.5, 1.0)


class A:
    pass


# =========================================================================== benchmark context
def setup(threads):
    a = A()
    a.threads = threads
    db, callers, queries = bench.setup(a, vectors=None)
    bench.CTX["rel"] = bench.load_relevance(OUT1, queries, callers)
    g0 = np.load(OUT1 / "topk/gids.npy")
    assert np.array_equal(np.load(LS.TOPK / "gids.npy"), g0)
    round2.EV.update(row_of={int(g): k for k, g in enumerate(g0)}, e_idx=np.load(LS.TOPK / "idx.npy"),
                     e_s=np.load(LS.TOPK / "s.npy"))
    bench.warm_up(db)
    return db, callers, queries


def m1z_task(qid):
    q = bench.CTX["queries"].loc[qid]
    db = bench.CTX["db"]
    (eg, es_, ev), ns = round2.entries(q, False, 0.0, 1.0)
    ha, hb, sc, _ = lsr.scan(eg, es_, ev, ns, db.cc, db.pos, bench.W_of(q, 8), 3.0, 0)
    r, _, _ = bench.label_eval(q, ha, hb, sc, None)
    return int(qid), r["ap"], r["ap_strict"], r["n_rel"]


# =========================================================================== locus-index search
@njit(cache=True)
def greedy_nonoverlap(order, s, e, keep, used):
    """Accept intervals [s, e] (gid space; contigs are contiguous gid ranges) in the given order unless they overlap an
    accepted one. `used` is a zeroed bool buffer over gids, restored to zero before returning."""
    acc = np.empty(min(keep, order.shape[0]), np.int64)
    m = 0
    for t in range(order.shape[0]):
        i = order[t]
        ok = True
        for g in range(s[i], e[i] + 1):
            if used[g]:
                ok = False
                break
        if not ok:
            continue
        for g in range(s[i], e[i] + 1):
            used[g] = True
        acc[m] = i
        m += 1
        if m >= keep:
            break
    for t in range(m):
        i = acc[t]
        for g in range(s[i], e[i] + 1):
            used[g] = False
    return acc[:m]


def eval_hits(q, ha, hb, sc):
    C = bench.CTX["callers"][q.caller]
    lab, match, _ = lsr.label_hits(ha, hb, q.tcode, C["gptr"], C["ginst"], C["tcode"], C["nmem"], C["target"])
    rr, rv = bench.CTX["rel"][int(q.qid)]
    return lsr.evaluate(sc, lab, match, None, rr, rv), lab, match


# =========================================================================== window search
@njit(parallel=True, cache=True)
def local_gram_prefix(X, cc, W):
    """C[k, i] = sum_{t < i} x_t . x_{t+k} over same-contig pairs, k = 1..W-1 (float64 prefix sums)."""
    n = X.shape[0]
    Dk = np.zeros((W, n), np.float32)
    for i in prange(n):
        for k in range(1, W):
            j = i + k
            if j >= n or cc[j] != cc[i]:
                break
            acc = 0.0
            for d in range(X.shape[1]):
                acc += X[i, d] * X[j, d]
            Dk[k, i] = acc
    C = np.zeros((W, n + 1), np.float64)
    for k in prange(1, W):
        s = 0.0
        for i in range(n):
            C[k, i] = s
            s += Dk[k, i]
        C[k, n] = s
    return C


@njit(cache=True)
def window_scan(p, cand, cc, C, W, own_lo, own_hi, tau):
    """Windows [a, b] (same contig, <= W genes) whose first and last genes are candidates (p > tau); score = cosine of
    the query vector with the window's summed vector = sum p / ||sum x||. Returns arrays a, b, score."""
    n = cand.shape[0]
    P = np.zeros(1)
    outa = np.empty(64, np.int64)
    outb = np.empty(64, np.int64)
    outs = np.empty(64, np.float64)
    m = 0
    for ii in range(n):
        a = cand[ii]
        if a >= own_lo and a < own_hi:
            continue
        num = 0.0
        nrm2 = 0.0
        prev_b = a - 1
        for jj in range(ii, n):
            b = cand[jj]
            if cc[b] != cc[a] or b - a + 1 > W:
                break
            # extend window from prev_b to b
            for g in range(prev_b + 1, b + 1):
                num += p[g]
                # add x_g . x_g = 1 and 2 * sum_{t in [a, g-1]} x_t . x_g
                nrm2 += 1.0
                for k in range(1, g - a + 1):
                    nrm2 += 2.0 * (C[k, g - k + 1] - C[k, g - k])
            prev_b = b
            sc = num / math.sqrt(max(nrm2, 1e-12))
            if m == outa.shape[0]:
                na = np.empty(m * 2, np.int64)
                nb = np.empty(m * 2, np.int64)
                ns = np.empty(m * 2, np.float64)
                na[:m] = outa
                nb[:m] = outb
                ns[:m] = outs
                outa, outb, outs = na, nb, ns
            outa[m] = a
            outb[m] = b
            outs[m] = sc
            m += 1
    return outa[:m], outb[:m], outs[:m]


def greedy_windows(a, b, s, keep, used):
    o = np.argsort(-s, kind="stable").astype(np.int64)
    return greedy_nonoverlap(o, a, b, keep, used)


# =========================================================================== matched rates
def matched_recall(H, qs_odd, qs_even, rel, levels=LEVELS, score_col="score", higher_better=True):
    """Threshold on odd queries' wrong-type hits (L per query), recall on even queries (first correct hit of each
    relevant target counted when its score passes)."""
    out, per = {}, {}
    sgn = 1.0 if higher_better else -1.0
    Hc = H[H.label == 1].copy()
    Hc["v"] = sgn * Hc[score_col]
    best = Hc.groupby(["qid", "match_row"]).v.max().rename("v").reset_index()
    bq = {k: g.v.to_numpy() for k, g in best.groupby("qid")}
    wo = np.sort(sgn * H[(H.split == "odd") & (H.label == 2)][score_col].to_numpy())[::-1]
    for L in levels:
        allowed = int(math.floor(L * len(qs_odd) + 1e-9))
        thr = wo[allowed] if len(wo) > allowed else -np.inf
        rows = []
        for qid, gc in zip(qs_even.qid, qs_even.genome_code):
            v = bq.get(int(qid), np.empty(0))
            rows.append((int(qid), int(gc), float((v > thr).sum() / len(rel[int(qid)][0]))))
        R = pd.DataFrame(rows, columns=["qid", "genome_code", "recall"])
        He = H[(H.split == "even") & (sgn * H[score_col] > thr)]
        out[f"{L:g}"] = dict(threshold=float(sgn * thr) if np.isfinite(thr) else None,
                             recall=lsr.boot_mean(R.recall, R.genome_code),
                             even_wrong_per_query=float((He.label == 2).sum() / len(qs_even)),
                             even_unlabelled_per_query=float((He.label == 0).sum() / len(qs_even)))
        per[L] = R
    return out, per


# =========================================================================== stage
def stage_run(a):
    import faiss
    import numba
    numba.set_num_threads(a.threads)
    faiss.omp_set_num_threads(a.threads)
    SRCH.mkdir(parents=True, exist_ok=True)
    T0 = time.time()
    res = dict(stage="search", rep=a.rep)
    db, callers, queries = setup(a.threads)
    sel = json.load(open(OUT3 / "selection3.json"))
    c = sel["odd"]["M1z"]
    res["m1z_config"] = c
    assert abs(c["w"] - 1.0) < 1e-9 and c["wx"] == 8 and abs(c["tau"] - 3.0) < 1e-9, c
    Q = queries[queries.caller == "DefenseFinder"]
    # ---- 1. reproduce M1z
    t0 = time.time()
    rows = bench.pool_map(m1z_task, Q[Q.split == "even"].qid.tolist(), a.threads, chunksize=4)
    R1 = pd.DataFrame(rows, columns=["qid", "ap", "ap_strict", "n_rel"]).merge(Q[["qid", "genome_code"]], on="qid")
    R1 = R1[R1.n_rel > 0]
    M = pd.concat([pd.read_parquet(p) for p in sorted(OUT3.glob("final_df*_metrics.parquet"))], ignore_index=True)
    Ms = M[(M.fam == "M1z") & (M.w == c["w"]) & (M.wx == c["wx"]) & (M.tau == c["tau"]) & (M.split == "even")]
    Ms = Ms.drop_duplicates("qid").set_index("qid")
    d = np.abs(R1.set_index("qid").ap - Ms.loc[R1.qid, "ap"].to_numpy())
    res["reproduce_m1z"] = dict(queries=int(len(R1)), ap=float(R1.ap.mean()), stored_ap=float(Ms.loc[R1.qid, "ap"].mean()),
                                max_abs_diff=float(d.max()), seconds=time.time() - t0)
    log("reproduce M1z", res["reproduce_m1z"])
    dump(res, SRCH / f"search_{a.rep}.json")
    # ---- query vectors
    U = LS.Universe(vectors=False)
    X = np.load(LS.ZROW)
    log(f"vectors in RAM {X.shape}")
    Cdf = callers["DefenseFinder"]
    Qn = Q.copy()
    Qn = Qn[[len(bench.CTX["rel"][int(q)][0]) > 0 for q in Qn.qid]].reset_index(drop=True)
    qmem = [Cdf["mem"][int(r)] for r in Qn.row]
    qv = np.stack([X[m].sum(0) for m in qmem]).astype(np.float32)
    qv /= np.linalg.norm(qv, axis=1, keepdims=True)
    qv_mean = qv.copy()
    if a.rep in ("LZ", "LW"):                         # part-5 fitted maps on the pooled vector
        z = np.load(OUT / "fit" / "maps.npz")
        T = z["W1"] if a.rep == "LZ" else z["W2"] @ z["W1"]
        qv = ((qv - z["mu"]) @ T.T).astype(np.float32)
        qv /= np.linalg.norm(qv, axis=1, keepdims=True)
    # ---- 2. locus index (DH loci)
    D = pd.read_parquet(MAP / "dh_loci.parquet")
    V = np.load(MAP / f"V_{a.rep}.npy").astype(np.float32)
    s_, e_ = D.gid_s.to_numpy(np.int64), D.gid_e.to_numpy(np.int64)
    gcd = U.gc[s_]
    used = np.zeros(U.N + 1, np.bool_)
    hits, per = [], []
    t0 = time.time()
    tq = []
    for c0 in range(0, len(Qn), 64):
        tb = time.time()
        Sb = V @ qv[c0:c0 + 64].T                      # (n_loci, batch)
        for j in range(Sb.shape[1]):
            q = Qn.iloc[c0 + j]
            sc = Sb[:, j].copy()
            sc[gcd == int(q.genome_code)] = -np.inf
            top = np.argpartition(-sc, N_CAND)[:N_CAND]
            top = top[np.argsort(-sc[top], kind="stable")].astype(np.int64)
            top = top[np.isfinite(sc[top])]
            acc = greedy_nonoverlap(top, s_, e_, N_KEEP, used)
            ha, hb, hs = s_[acc], e_[acc], sc[acc].astype(np.float64)
            r, lab, match = eval_hits(q, ha, hb, hs)
            per.append(dict(qid=int(q.qid), genome_code=int(q.genome_code), split=q.split, ap=r["ap"],
                            ap_strict=r["ap_strict"], recall_any=r["recall_any"], n_rel=r["n_rel"]))
            hits.append(pd.DataFrame(dict(qid=int(q.qid), split=q.split, start_gid=ha, end_gid=hb, score=hs, label=lab,
                                          match_row=match, span=hb - ha + 1)))
        tq.append((time.time() - tb) / Sb.shape[1])
    PI = pd.DataFrame(per)
    HI = pd.concat(hits, ignore_index=True)
    res["locus_index"] = dict(seconds=time.time() - t0, per_query_seconds_median=float(np.median(tq)),
                              n_loci=int(len(D)))
    # single-query timing with a FAISS flat index (one query at a time)
    ix = faiss.IndexFlatIP(V.shape[1])
    ix.add(V)
    tt = []
    for j in range(20):
        t1 = time.time()
        ix.search(qv[j:j + 1], 2000)
        tt.append(time.time() - t1)
    res["locus_index"]["faiss_flat_single_query_seconds_median"] = float(np.median(tt))
    del ix
    # ceiling: relevant targets covered >= 50% by some DH locus (other genomes)
    HI.to_parquet(SRCH / f"hits_locusindex_{a.rep}.parquet", index=False)
    PI.to_parquet(SRCH / f"perquery_locusindex_{a.rep}.parquet", index=False)
    log("locus index done", time.time() - t0)
    del V
    # ---- 3. window search (MEAN only: linear in the gene vectors)
    if a.rep != "MEAN":
        return finish(a, res, (("locus_index", PI, HI),), Qn, M, c, T0)
    qv = qv_mean
    t0 = time.time()
    Cg = local_gram_prefix(X, U.cc, WMAX)
    res["window_gram_seconds"] = time.time() - t0
    log(f"local Gram prefix {time.time() - t0:.0f}s")
    hits, per, tq = [], [], []
    t0 = time.time()
    for c0 in range(0, len(Qn), 64):
        tb = time.time()
        Pb = X @ qv[c0:c0 + 64].T                      # (N, batch) per-gene dot products
        for j in range(Pb.shape[1]):
            q = Qn.iloc[c0 + j]
            p = np.ascontiguousarray(Pb[:, j])
            lo, hi = db.own_range(int(q.genome_code))
            cand = np.flatnonzero(p > a.tau).astype(np.int64)
            W = int(min(WMAX, q.span + 8))
            wa, wb, ws = window_scan(p, cand, U.cc, Cg, W, lo, hi, a.tau)
            acc = greedy_windows(wa, wb, ws, N_KEEP, used) if len(ws) else np.empty(0, np.int64)
            ha, hb, hs = wa[acc], wb[acc], ws[acc]
            r, lab, match = eval_hits(q, ha, hb, hs)
            per.append(dict(qid=int(q.qid), genome_code=int(q.genome_code), split=q.split, ap=r["ap"],
                            ap_strict=r["ap_strict"], recall_any=r["recall_any"], n_rel=r["n_rel"],
                            n_cand=int(len(cand)), n_windows=int(len(ws))))
            hits.append(pd.DataFrame(dict(qid=int(q.qid), split=q.split, start_gid=ha, end_gid=hb, score=hs, label=lab,
                                          match_row=match, span=hb - ha + 1)))
        tq.append((time.time() - tb) / Pb.shape[1])
        if (c0 // 64) % 8 == 0:
            log(f"  windows {c0 + 64:,}/{len(Qn):,} {time.time() - t0:.0f}s")
    PW = pd.DataFrame(per)
    HW = pd.concat(hits, ignore_index=True)
    res["window"] = dict(seconds=time.time() - t0, per_query_seconds_median=float(np.median(tq)), tau=a.tau)
    HW.to_parquet(SRCH / f"hits_window_{a.rep}.parquet", index=False)
    PW.to_parquet(SRCH / f"perquery_window_{a.rep}.parquet", index=False)
    return finish(a, res, (("locus_index", PI, HI), ("window", PW, HW)), Qn, M, c, T0)


def finish(a, res, arms, Qn, M, c, T0):
    """Summaries, paired against M1z (stored per-query AP at the frozen config), and matched wrong-type rates."""
    rel = bench.CTX["rel"]
    qs_odd = Qn[Qn.split == "odd"]
    qs_even = Qn[Qn.split == "even"]
    Mall = M[(M.fam == "M1z") & (M.w == c["w"]) & (M.wx == c["wx"]) & (M.tau == c["tau"])].drop_duplicates("qid")
    Mall = Mall.set_index("qid")
    for name, P_, H_ in arms:
        Pe = P_[P_.split == "even"].copy()
        Pe["ap_m1z"] = Mall.loc[Pe.qid, "ap"].to_numpy()
        Pe["ap_strict_m1z"] = Mall.loc[Pe.qid, "ap_strict"].to_numpy()
        d = res[name]
        d["even_queries"] = int(len(Pe))
        d["ap"] = lsr.boot_mean(Pe.ap, Pe.genome_code)
        d["ap_strict"] = lsr.boot_mean(Pe.ap_strict, Pe.genome_code)
        d["ap_m1z"] = lsr.boot_mean(Pe.ap_m1z, Pe.genome_code)
        d["ap_minus_m1z"] = lsr.boot_mean(Pe.ap - Pe.ap_m1z, Pe.genome_code)
        d["ap_strict_minus_m1z"] = lsr.boot_mean(Pe.ap_strict - Pe.ap_strict_m1z, Pe.genome_code)
        d["recall_any"] = lsr.boot_mean(Pe.recall_any, Pe.genome_code)
        d["odd_ap"] = float(P_[P_.split == "odd"].ap.mean())
        mr, mper = matched_recall(H_, qs_odd, qs_even, rel)
        d["matched_wrong"] = mr
        d["_per"] = {str(k): v for k, v in mper.items()}
        log(name, "AP", d["ap"], "minus M1z", d["ap_minus_m1z"])
    # M1z matched-rate recall from its stored hits (fast-E threshold), reproducing the post hoc table
    Hh = pd.read_parquet(OUT3 / "rates_hits.parquet", columns=["qid", "fam", "split", "label", "match_row", "E_fast"])
    Hm = Hh[Hh.fam == "M1z"].copy()
    mr, mper = matched_recall(Hm, qs_odd, qs_even, rel, score_col="E_fast", higher_better=False)
    res["m1z_matched_wrong"] = mr
    for name, _, _ in arms:
        per_ = res[name].pop("_per")
        res[name]["matched_wrong_minus_m1z"] = {
            k: lsr.boot_mean(per_[k].recall.to_numpy() - mper[float(k)].set_index("qid").loc[per_[k].qid, "recall"].to_numpy(),
                             per_[k].genome_code) for k in per_}
    res["seconds"] = time.time() - T0
    dump(res, SRCH / f"search_{a.rep}.json")
    log("search done", res["seconds"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["run"])
    ap.add_argument("--rep", default="MEAN")
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_ON_NODE", 4)))
    a = ap.parse_args()
    stage_run(a)


if __name__ == "__main__":
    main()
