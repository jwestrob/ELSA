#!/usr/bin/env python
"""Locus search, round 2 (workstream ws_locus_search2). Reuses bench.py and locus_search.py.

  round2.py cvgrid   --threads T             scan-only AP grid for S2 and S3 on every DefenseFinder query,
                                             5-fold genome cross-validation (results/criterion2.json, test A)
  round2.py final    --threads T --set S     200-replicate runs; hits (E <= 10), metrics and null tails saved
                     S = dfall  (DefenseFinder, all genomes: S2, S3; even genomes also M0, M2)
                     S = xeven  (TXSScan and CRISPRCasTyper, even genomes: S2, S3 at the frozen DF configs)
  round2.py rategrid --threads T --split P   candidate grid for the rate-constrained selection, P = odd | even:
                                             40-replicate null with a fitted exponential tail (fastnull.py)

Evidence modes per query slot (member gene g), each a list of (database gid, s = -log tail fraction):
  token t(g) (family size), sequence rank (MMseqs2 bitscore, round-1 topk_seq), embedding rank
  (ESM++ small exact cosine, round-1 topk); a config takes the per-gene maximum of
  token (if used), w_seq * sequence rank (if w_seq > 0) and w_emb * embedding rank (if w_emb > 0).
  M0 = token; M1 = embedding; M2 = max(token, w * embedding); S1 = sequence; S2 = max(token, w * sequence);
  S3 = max(token, sequence, w * embedding). These reproduce bench.py / seq_control.py exactly.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench  # noqa: E402
import fastnull  # noqa: E402
import locus_search as ls  # noqa: E402
from locus_search import log  # noqa: E402

OUT1 = Path(os.environ.get("LS_OUT1", ls.RD / "locus_search/out"))
OUT2 = Path(os.environ.get("LS_OUT2", ls.RD / "locus_search/out2"))
SEED = 17
EV = {}

# frozen round-1 configs (DefenseFinder, odd genomes): name -> (use_token, w_seq, w_emb, W extra, tau)
FROZEN = {
    "M0": (True, 0.0, 0.0, 8, 3.0),
    "M2": (True, 0.0, 0.5, 8, 0.0),
    "S2": (True, 1.0, 0.0, 8, 3.0),
    "S3": (True, 1.0, 0.5, 8, 0.0),
}


def family(use_tok, ws, we):
    if ws == 0 and we == 0:
        return "M0"
    if not use_tok:
        return "M1" if we > 0 else "S1"
    if ws == 0:
        return "M2"
    return "S2" if we == 0 else "S3"


def cfg_name(use_tok, ws, we, wx, tau):
    f = family(use_tok, ws, we)
    w = {"M0": "", "M1": "", "S1": "", "M2": f"w{we:g}", "S2": f"w{ws:g}", "S3": f"w{we:g}"}[f]
    return f"{f}{w}_x{wx}_t{tau:g}"


def load_ranks():
    g_e = np.load(OUT1 / "topk/gids.npy")
    g_s = np.load(OUT1 / "topk_seq/gids.npy")
    assert np.array_equal(g_e, g_s)
    EV.update(row_of={int(g): k for k, g in enumerate(g_e)},
              e_idx=np.load(OUT1 / "topk/idx.npy"), e_s=np.load(OUT1 / "topk/s.npy"),
              s_idx=np.load(OUT1 / "topk_seq/idx.npy"), s_s=np.load(OUT1 / "topk_seq/s.npy"))
    log(f"ranks loaded: {len(g_e):,} member genes, K = {EV['e_idx'].shape[1]:,}")


def entries(q, use_tok, ws, we):
    db = bench.CTX["db"]
    own = int(q.genome_code)
    N_db = db.n_db(own)
    lo, hi = db.own_range(own)
    lists = []
    for g in bench.q_slots(q):
        k = EV["row_of"][int(g)]
        G, S = [], []
        if use_tok:
            gt = db.token_members(int(db.tok[g]), own)
            G.append(gt)
            S.append(np.full(len(gt), math.log(N_db / len(gt)) if len(gt) else 0.0))
        if ws > 0:
            i = EV["s_idx"][k]
            ok = (i >= 0) & ((i < lo) | (i >= hi))
            G.append(i[ok].astype(np.int64))
            S.append(ws * EV["s_s"][k][ok].astype(np.float64))
        if we > 0:
            i = EV["e_idx"][k]
            ok = (i >= 0) & ((i < lo) | (i >= hi))
            G.append(i[ok].astype(np.int64))
            S.append(we * EV["e_s"][k][ok].astype(np.float64))
        gg = np.concatenate(G)
        ss = np.concatenate(S)
        if len(G) == 1:
            o = np.argsort(gg, kind="stable")
            lists.append((gg[o], ss[o]))
            continue
        o = np.lexsort((-ss, gg))
        gg, ss = gg[o], ss[o]
        first = np.r_[True, gg[1:] != gg[:-1]] if len(gg) else np.zeros(0, bool)
        lists.append((gg[first], ss[first]))
    return ls.assemble(lists), len(lists)


# =========================================================================== stage: CV grid
CV_GRID = ([(True, w, 0.0) for w in (0.5, 0.75, 1.0)] + [(True, 1.0, w) for w in (0.25, 0.5, 0.75, 1.0)])


def cv_task(qid):
    q = bench.CTX["queries"].loc[qid]
    db = bench.CTX["db"]
    out = []
    for use_tok, ws, we in CV_GRID:
        (eg, es_, ev), ns = entries(q, use_tok, ws, we)
        for wx, tau in product(bench.GRID_WX, bench.GRID_TAU):
            ha, hb, sc, _ = ls.scan(eg, es_, ev, ns, db.cc, db.pos, bench.W_of(q, wx), tau, 0)
            r, _, _ = bench.label_eval(q, ha, hb, sc, None)
            out.append(dict(qid=int(qid), fam=family(use_tok, ws, we), w=we if we > 0 else ws, wx=wx, tau=tau,
                            ap=r["ap"], ap_strict=r["ap_strict"], n_rel=r["n_rel"], n_hits=r["n_hits"]))
    return out


def setup(a):
    db, callers, queries = bench.setup(a, vectors=None)
    bench.CTX["rel"] = bench.load_relevance(OUT1, queries, callers)
    load_ranks()
    bench.warm_up(db)
    return db, callers, queries


def paired_boot(x, y, g, draws=2000, seed=SEED):
    return ls.boot_mean(np.asarray(x) - np.asarray(y), g, draws=draws, seed=seed)


def cvgrid(a):
    OUT2.mkdir(parents=True, exist_ok=True)
    db, callers, queries = setup(a)
    Q = queries[queries.caller == "DefenseFinder"]
    t0 = time.time()
    rows = [r for ch in bench.pool_map(cv_task, Q.qid.tolist(), a.threads) for r in ch]
    G = pd.DataFrame(rows).merge(queries[["qid", "genome_code", "split", "n_members", "type"]], on="qid")
    G.to_parquet(OUT2 / "cv_grid.parquet", index=False)
    log(f"CV grid: {len(Q):,} queries x {len(CV_GRID) * 16} configs, {time.time() - t0:.0f}s")
    cv_analyse(G)


def cv_analyse(G):
    G = G[G.n_rel > 0].copy()
    # reproducibility: round-1 S2 / S3 configs against round-1 metrics
    chk = {}
    for name, fam, w, wx, tau, f in (("S2", "S2", 1.0, 8, 3.0, "final_metrics_seqctl.parquet"),
                                     ("S3", "S3", 0.5, 8, 0.0, "final_metrics_seqctl3.parquet")):
        r1 = pd.read_parquet(OUT1 / f)
        r1 = r1[r1.method == name].set_index("qid").ap
        g = G[(G.fam == fam) & (G.w == w) & (G.wx == wx) & (G.tau == tau)].set_index("qid").ap
        ids = g.index.intersection(r1.index)
        chk[name] = dict(n=int(len(ids)), max_abs_diff=float((g.loc[ids] - r1.loc[ids]).abs().max()))
    log("reproduce round-1 AP", chk)
    gcodes = np.unique(G.genome_code)
    perm = np.random.default_rng(SEED).permutation(gcodes)
    fold_of = {int(c): i % 5 for i, c in enumerate(perm)}
    G["fold"] = G.genome_code.map(fold_of)
    held = []
    chosen = []
    for f in range(5):
        tr, te = G[G.fold != f], G[G.fold == f]
        for fam in ("S2", "S3"):
            agg = tr[tr.fam == fam].groupby(["w", "wx", "tau"]).ap.mean().reset_index()
            b = agg.sort_values(["ap", "wx", "tau", "w"], ascending=[False, True, False, True]).iloc[0]
            chosen.append(dict(fold=f, fam=fam, w=float(b.w), wx=int(b.wx), tau=float(b.tau), train_map=float(b.ap)))
            h = te[(te.fam == fam) & (te.w == b.w) & (te.wx == b.wx) & (te.tau == b.tau)]
            held.append(h.assign(fold=f))
    Hd = pd.concat(held, ignore_index=True)
    X = Hd.pivot_table(index=["qid", "genome_code", "split", "n_members", "fold"], columns="fam",
                       values=["ap", "ap_strict"]).reset_index()
    X.columns = ["_".join(c).strip("_") for c in X.columns]
    res = dict(chk_round1=chk, folds=chosen, n_queries=int(len(X)), n_genomes=int(X.genome_code.nunique()), strata={})
    for name, sel in (("all", np.ones(len(X), bool)), ("even", X.split == "even"), ("odd", X.split == "odd"),
                      ("min3", X.n_members >= 3)):
        S = X[sel]
        d = dict(queries=int(len(S)))
        for col in ("ap", "ap_strict"):
            d[f"S2_{col}"] = ls.boot_mean(S[f"{col}_S2"], S.genome_code)
            d[f"S3_{col}"] = ls.boot_mean(S[f"{col}_S3"], S.genome_code)
            d[f"S3-S2_{col}"] = paired_boot(S[f"{col}_S3"], S[f"{col}_S2"], S.genome_code)
        d["per_fold_S3-S2_ap"] = {int(f): float((g.ap_S3 - g.ap_S2).mean()) for f, g in S.groupby("fold")}
        res["strata"][name] = d
    A = res["strata"]["all"]["S3-S2_ap"]
    res["test_A"] = dict(mean=A[0], ci=[A[1], A[2]], passes=bool(A[0] >= 0.02 and A[1] > 0))
    X.to_parquet(OUT2 / "cv_heldout.parquet", index=False)
    json.dump(res, open(OUT2 / "cv_summary.json", "w"), indent=1, default=float)
    log("test A", json.dumps(res["test_A"]), "folds", json.dumps(chosen))


# =========================================================================== stage: shuffled-embedding control
SHUF_GRID = [(True, 1.0, w) for w in (0.25, 0.5, 0.75, 1.0)] + [(True, 1.0, 0.0)]


def shuf_task(qid):
    """S3 with each slot's embedding rank list taken from a random other member gene (permutation of
    rows, seed 17): the embedding's score distribution and list length are kept, its gene-specific
    information is destroyed. Scan only (AP)."""
    q = bench.CTX["queries"].loc[qid]
    db = bench.CTX["db"]
    out = []
    for use_tok, ws, we in SHUF_GRID:
        for shuffled in ((False, True) if we > 0 else (False,)):
            EV["e_idx"], EV["e_s"] = (EV["e_idx_shuf"], EV["e_s_shuf"]) if shuffled else (EV["e_idx_true"], EV["e_s_true"])
            (eg, es_, ev), ns = entries(q, use_tok, ws, we)
            for tau in bench.GRID_TAU:
                ha, hb, sc, _ = ls.scan(eg, es_, ev, ns, db.cc, db.pos, bench.W_of(q, 8), tau, 0)
                r, _, _ = bench.label_eval(q, ha, hb, sc, None)
                out.append(dict(qid=int(qid), fam=family(use_tok, ws, we) + ("shuf" if shuffled else ""), w=we, tau=tau,
                                ap=r["ap"], ap_strict=r["ap_strict"], n_rel=r["n_rel"]))
    return out


def shufctl(a):
    db, callers, queries = setup(a)
    perm = np.random.default_rng(SEED).permutation(len(EV["e_idx"]))
    EV.update(e_idx_true=EV["e_idx"], e_s_true=EV["e_s"], e_idx_shuf=EV["e_idx"][perm], e_s_shuf=EV["e_s"][perm])
    Q = queries[(queries.caller == "DefenseFinder") | (queries.split == "even")]
    t0 = time.time()
    rows = [r for ch in bench.pool_map(shuf_task, Q.qid.tolist(), a.threads, chunksize=2) for r in ch]
    G = pd.DataFrame(rows).merge(queries[["qid", "caller", "genome_code", "split", "n_members"]], on="qid")
    G.to_parquet(OUT2 / "shufctl_grid.parquet", index=False)
    log(f"shuffled-embedding control: {len(Q):,} queries, {time.time() - t0:.0f}s")
    G = G[G.n_rel > 0]
    res = {}
    for c, Gc in G.groupby("caller"):
        Gc = Gc if c == "DefenseFinder" else Gc[Gc.split == "even"]
        agg = Gc.groupby(["fam", "w", "tau"]).ap.mean().reset_index().sort_values("ap", ascending=False)
        res[c] = dict(grid=agg.round(4).to_dict("records"))
        piv = Gc.pivot_table(index=["qid", "genome_code"], columns=["fam", "w", "tau"], values="ap")
        s2 = piv[("S2", 0.0, 3.0)]
        for lab, col in (("S3_w0.5_t0", ("S3", 0.5, 0.0)), ("S3shuf_w0.5_t0", ("S3shuf", 0.5, 0.0))):
            res[c][f"{lab}-S2"] = paired_boot(piv[col], s2, piv.index.get_level_values("genome_code"))
        res[c]["S3-S3shuf_w0.5_t0"] = paired_boot(piv[("S3", 0.5, 0.0)], piv[("S3shuf", 0.5, 0.0)],
                                                  piv.index.get_level_values("genome_code"))
        best_shuf = agg[agg.fam == "S3shuf"].iloc[0]
        col = ("S3shuf", best_shuf.w, best_shuf.tau)
        res[c]["S3shuf_best"] = dict(w=float(best_shuf.w), tau=float(best_shuf.tau),
                                     minus_S2=paired_boot(piv[col], s2, piv.index.get_level_values("genome_code")))
        # matched tau: S3 (w 0.5) minus S2 (w 1) at each tau
        res[c]["matched_tau_S3w0.5-S2"] = {
            str(t): paired_boot(piv[("S3", 0.5, t)], piv[("S2", 0.0, t)], piv.index.get_level_values("genome_code"))
            for t in bench.GRID_TAU}
        log(c, json.dumps({k: v for k, v in res[c].items() if k != "grid"}, default=float))
    json.dump(res, open(OUT2 / "shufctl_summary.json", "w"), indent=1, default=float)


# =========================================================================== stage: final (R = 200)
def run_one(q, use_tok, ws, we, wx, tau, R, top_n_null, keep_top, seed, E_keep):
    db = bench.CTX["db"]
    (eg, es_, ev), ns = entries(q, use_tok, ws, we)
    W = bench.W_of(q, wx)
    t0 = time.time()
    ha, hb, sc, nf = ls.scan(eg, es_, ev, ns, db.cc, db.pos, W, tau, 0)
    t1 = time.time()
    lo, hi = db.own_range(int(q.genome_code))
    tail = ls.null_tail(eg, es_, ev, ns, db.cc, db.pos, db.N, lo, hi, W, tau, R, seed, top_n_null)
    t2 = time.time()
    return dict(ha=ha, hb=hb, sc=sc, nf=nf, tail=tail, W=W, n_entries=int(len(eg)), sec_scan=t1 - t0, sec_null=t2 - t1)


def final_task(args):
    qid, method, (use_tok, ws, we, wx, tau) = args
    q = bench.CTX["queries"].loc[qid]
    r = run_one(q, use_tok, ws, we, wx, tau, bench.R_NULL, bench.TOP_N, 32, SEED + int(qid), 10.0)
    E, _ = ls.evalues(r["sc"], r["tail"], bench.R_NULL)
    m, lab, match = bench.label_eval(q, r["ha"], r["hb"], r["sc"], E)
    m.update(qid=int(qid), method=method, W=r["W"], tau=tau, n_entries=r["n_entries"], sec_scan=r["sec_scan"],
             sec_null=r["sec_null"])
    keep = E <= 10
    hits = pd.DataFrame(dict(qid=int(qid), method=method, start_gid=r["ha"][keep], end_gid=r["hb"][keep],
                             score=r["sc"][keep], evalue=E[keep], label=lab[keep], match_row=match[keep],
                             n_slots_filled=r["nf"][keep]))
    return m, hits, r["tail"][:, :32].astype(np.float32)


def final(a):
    OUT2.mkdir(parents=True, exist_ok=True)
    db, callers, queries = setup(a)
    tasks = []
    if a.set == "dfall":
        Q = queries[queries.caller == "DefenseFinder"]
        for qid, split in zip(Q.qid, Q.split):
            for name in (("S2", "S3", "M0", "M2") if split == "even" else ("S2", "S3")):
                tasks.append((int(qid), name, FROZEN[name]))
    else:
        Q = queries[queries.caller.isin(["TXSScan", "CRISPRCasTyper"]) & (queries.split == "even")]
        for qid in Q.qid:
            for name in ("S2", "S3"):
                tasks.append((int(qid), name, FROZEN[name]))
    tasks.sort(key=lambda t: -queries.n_members[t[0]])           # large queries first
    t0 = time.time()
    res = bench.pool_map(final_task, tasks, a.threads, chunksize=1)
    log(f"final {a.set}: {len(tasks):,} runs, {time.time() - t0:.0f}s")
    M = pd.DataFrame([m for m, _, _ in res]).merge(queries, on="qid")
    M.to_parquet(OUT2 / f"final_{a.set}_metrics.parquet", index=False)
    pd.concat([h for _, h, _ in res], ignore_index=True).to_parquet(OUT2 / f"final_{a.set}_hits.parquet", index=False)
    np.savez(OUT2 / f"final_{a.set}_tails.npz", qid=np.array([m["qid"] for m, _, _ in res]),
             method=np.array([m["method"] for m, _, _ in res]), tails=np.stack([t for _, _, t in res]))
    summarize_final(a.set, M)


def summarize_final(tag, M):
    M = M[M.n_rel > 0]
    cols = ["ap", "ap_strict", "recall_E0.01", "recall_E1", "n_unlabelled_E0.01", "n_wrong_E0.01", "n_correct_E0.01"]
    res = {}
    if tag == "dfall":
        # reproducibility against round 1 (same seeds): S2, S3, M0, M2
        chk = {}
        r1 = pd.concat([pd.read_parquet(OUT1 / "final_metrics_seqctl.parquet"), pd.read_parquet(OUT1 / "final_metrics_seqctl3.parquet"),
                        pd.read_parquet(OUT1 / "final_metrics_dfr1.parquet")], ignore_index=True)
        for name in ("S2", "S3", "M0", "M2"):
            x = M[M.method == name].set_index("qid")
            y = r1[r1.method == name].drop_duplicates("qid").set_index("qid")
            ids = x.index.intersection(y.index)
            chk[name] = {c: float((x.loc[ids, c] - y.loc[ids, c]).abs().max()) for c in ("ap", "recall_E0.01", "n_unlabelled_E0.01")}
            chk[name]["n"] = int(len(ids))
        res["chk_round1"] = chk
        log("reproduce round-1", chk)
    for (c, split), S in M.groupby(["caller", "split"]):
        key = f"{c}|{split}"
        res[key] = {m: {col: ls.boot_mean(g[col], g.genome_code) for col in cols} | {"queries": int(len(g))}
                    for m, g in S.groupby("method")}
        a_, b_ = S[S.method == "S3"].set_index("qid"), S[S.method == "S2"].set_index("qid")
        ids = a_.index.intersection(b_.index)
        res[key]["S3-S2"] = {col: paired_boot(a_.loc[ids, col], b_.loc[ids, col], b_.loc[ids, "genome_code"])
                             for col in ("ap", "ap_strict", "recall_E0.01", "n_unlabelled_E0.01")}
        mm = S[S.n_members >= 3]
        a3, b3 = mm[mm.method == "S3"].set_index("qid"), mm[mm.method == "S2"].set_index("qid")
        i3 = a3.index.intersection(b3.index)
        if len(i3):
            res[key]["S3-S2_min3"] = {col: paired_boot(a3.loc[i3, col], b3.loc[i3, col], b3.loc[i3, "genome_code"])
                                      for col in ("ap", "ap_strict")} | {"queries": int(len(i3))}
        log(key, "S3-S2 ap", res[key]["S3-S2"]["ap"])
    tm = M.groupby("method")[["sec_scan", "sec_null"]].agg(["mean", "median"])
    res["timing"] = {m: {f"{a}_{b}": float(tm.loc[m, (a, b)]) for a, b in tm.columns} for m in tm.index}
    json.dump(res, open(OUT2 / f"final_{tag}_summary.json", "w"), indent=1, default=float)


# =========================================================================== stage: rate grid (R = 40)
RATE_FAMS = ([(True, 0.0, 0.0), (False, 0.0, 1.0)] + [(True, 0.0, w) for w in (0.5, 0.75, 1.0)] + [(False, 1.0, 0.0)]
             + [(True, w, 0.0) for w in (0.5, 0.75, 1.0)] + [(True, 1.0, w) for w in (0.25, 0.5, 0.75, 1.0)])
RATE_R = 40


def rate_task(qid):
    q = bench.CTX["queries"].loc[qid]
    mets, hits, tails = [], [], []
    for (use_tok, ws, we), tau in product(RATE_FAMS, bench.GRID_TAU):
        name = cfg_name(use_tok, ws, we, 8, tau)
        r = run_one(q, use_tok, ws, we, 8, tau, RATE_R, 64, 32, SEED + int(qid), 2.0)
        Ee = fastnull.evalue_emp(r["sc"], r["tail"], RATE_R)
        Ef = fastnull.evalue_fit(r["sc"], r["tail"], RATE_R, 1.0)
        m, lab, match = bench.label_eval(q, r["ha"], r["hb"], r["sc"], Ef)
        m.update(qid=int(qid), cfg=name, sec_scan=r["sec_scan"], sec_null=r["sec_null"], n_entries=r["n_entries"],
                 n_unlabelled_Eemp0025=int(((Ee <= 0.025) & (lab == 0)).sum()))
        keep = Ee <= 2.0
        hits.append(pd.DataFrame(dict(qid=int(qid), cfg=name, start_gid=r["ha"][keep], end_gid=r["hb"][keep],
                                      score=r["sc"][keep], E_emp=Ee[keep], E_fit=Ef[keep], label=lab[keep],
                                      match_row=match[keep], n_slots_filled=r["nf"][keep])))
        tails.append(r["tail"][:, :32].astype(np.float32))
        mets.append(m)
    return mets, pd.concat(hits, ignore_index=True), np.stack(tails)


def rategrid(a):
    OUT2.mkdir(parents=True, exist_ok=True)
    db, callers, queries = setup(a)
    Q = queries[(queries.caller == "DefenseFinder") & (queries.split == a.split)]
    qids = sorted(Q.qid.tolist(), key=lambda x: -queries.n_members[x])
    t0 = time.time()
    res = bench.pool_map(rate_task, qids, a.threads, chunksize=1)
    log(f"rate grid {a.split}: {len(qids):,} queries x {len(RATE_FAMS) * 4} configs, {time.time() - t0:.0f}s")
    M = pd.DataFrame([m for ms, _, _ in res for m in ms]).merge(queries, on="qid")
    M.to_parquet(OUT2 / f"rategrid_{a.split}_metrics.parquet", index=False)
    pd.concat([h for _, h, _ in res], ignore_index=True).to_parquet(OUT2 / f"rategrid_{a.split}_hits.parquet", index=False)
    np.savez(OUT2 / f"rategrid_{a.split}_tails.npz", qid=np.array([ms[0]["qid"] for ms, _, _ in res]),
             cfg=np.array([m["cfg"] for m in res[0][0]]), tails=np.stack([t for _, _, t in res]))
    S = M[M.n_rel > 0].groupby("cfg").agg(map=("ap", "mean"), unl=("n_unlabelled_E0.01", "mean"),
                                         unl_emp=("n_unlabelled_Eemp0025", "mean"), sec_null=("sec_null", "mean"),
                                         n=("ap", "size")).sort_values("map", ascending=False)
    S.to_csv(OUT2 / f"rategrid_{a.split}_summary.tsv", sep="\t")
    log("\n" + S.head(60).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["cvgrid", "cvanalyse", "final", "finalsum", "rategrid", "shufctl"])
    ap.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_ON_NODE", "2")))
    ap.add_argument("--set", default="dfall", choices=["dfall", "xeven"])
    ap.add_argument("--split", default="odd", choices=["odd", "even"])
    a = ap.parse_args()
    import numba
    numba.set_num_threads(a.threads)
    t0 = time.time()
    if a.stage == "cvgrid":
        cvgrid(a)
    elif a.stage == "cvanalyse":
        G = pd.read_parquet(OUT2 / "cv_grid.parquet")
        cv_analyse(G)
    elif a.stage == "final":
        final(a)
    elif a.stage == "shufctl":
        shufctl(a)
    elif a.stage == "finalsum":
        summarize_final(a.set, pd.read_parquet(OUT2 / f"final_{a.set}_metrics.parquet"))
    else:
        rategrid(a)
    log(f"done {a.stage} {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
