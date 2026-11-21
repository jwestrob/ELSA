#!/usr/bin/env python
"""softrec3 Task 5: locus search with stratified per-genome best matches as the only evidence, through the round-3 search
machinery (locus_search/code/search3.py, imported read-only; outputs under $RD/softrec/r3/search).

Evidence lists for the 18,414 DefenseFinder / benchmark member genes (out/topk/gids.npy), from the exact per-genome best
matches (r3/exact, cosine >= 0.32), every genome except the gene's own:
  st1  the best genome per genus (level 0)        st3  the best 3 genomes per genus        sta  every genome
s = log(N_db / t), t = 1 + N_db x P(cos >= c over all pairs) (the cosine's expected rank in a global list), as M1z's
s = log(N_db / rank) on its top-20,000 list.

  search3s.py lists                 build r3/search/lists/{st1,st3,sta}/{gids,idx,s}.npy
  search3s.py grid  --threads T     scan-only AP grid (W extra x tau) for Mst1, Mst3, Msta, all DefenseFinder queries
  search3s.py final --threads T     odd-genome configuration per family; 200-replicate null runs on every query
  search3s.py analyse --threads T   even genomes: AP vs M1z (round-3 final runs, frozen odd configuration W + 8, tau 3);
                                    fast-null curves (odd), recall at matched wrong-type / unlabelled / unsupported rates
"""
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

RD = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111")
R3 = RD / "softrec/r3"
OUT = R3 / "search"
ORIG = RD / "locus_search/out3"
os.environ["LS_OUT3"] = str(OUT)
sys.path.insert(0, str(RD / "locus_search/code"))
import search3 as S  # noqa: E402
import bench  # noqa: E402
import fastnull  # noqa: E402
import locus_search as ls  # noqa: E402
from locus_search import log  # noqa: E402

assert S.OUT3 == OUT
NEW = {"Mst1": "st1", "Mst3": "st3", "Msta": "sta", "Mzpg": "zpg"}   # Mzpg: control (M1z list, best gene per genome)
MCAP = {"st1": 1, "st3": 3, "sta": 10 ** 6}          # zpg is built by cmd_zpg
for f, src in NEW.items():
    S.SRC_DIR[src] = OUT / "lists" / src
    S.FAM[f] = [(False, 0.0, src, 1.0)]
S.SETS["strat"] = list(NEW)
S.RATE_FAMS = list(NEW)


def cmd_zpg(a):
    """Control: M1z's own zca10 top-20,000 list thinned to its first (best) gene per genome, M1z's own s."""
    t0 = time.time()
    meta = np.load(R3 / "vec/meta.npz")
    gs = meta["g_start"].astype(np.int64)
    src = S.SRC_DIR["zca10"]
    gids = np.load(src / "gids.npy")
    assert np.array_equal(gids, np.load(S.OUT1 / "topk/gids.npy"))
    I = np.load(src / "idx.npy", mmap_mode="r")
    Sv = np.load(src / "s.npy", mmap_mode="r")
    rows = []
    for r in range(len(gids)):
        i = np.asarray(I[r], np.int64)
        v = np.asarray(Sv[r], np.float32)
        ok = i >= 0
        i, v = i[ok], v[ok]
        G = np.searchsorted(gs, i, side="right") - 1
        _, first = np.unique(G, return_index=True)
        first = np.sort(first)
        rows.append((i[first].astype(np.int32), v[first]))
    K = max(len(x) for x, _ in rows)
    idx = np.full((len(rows), K), -1, np.int32)
    sv = np.zeros((len(rows), K), np.float32)
    for k, (x, y) in enumerate(rows):
        idx[k, :len(x)] = x
        sv[k, :len(y)] = y
    d = OUT / "lists" / "zpg"
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "gids.npy", gids)
    np.save(d / "idx.npy", idx)
    np.save(d / "s.npy", sv)
    ln = np.array([len(x) for x, _ in rows])
    log("zpg", dict(K=int(K), entries_per_gene_q10_q50_q90=np.quantile(ln, [.1, .5, .9]).tolist(), seconds=round(time.time() - t0, 1)))


def cmd_lists(a):
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    meta = np.load(R3 / "vec/meta.npz")
    uq = meta["uq"].astype(np.int64)
    gs, ge = meta["g_start"].astype(np.int64), meta["g_end"].astype(np.int64)
    lin_g = meta["lin_of_genome"].astype(np.int64)
    N = len(uq)
    counts = np.load(R3 / "exact/counts.npy").astype(np.int64)
    uptr = np.r_[0, np.cumsum(counts)]
    gen = np.memmap(R3 / "exact/genome.bin", dtype=np.int16, mode="r")
    loc = np.memmap(R3 / "exact/local.bin", dtype=np.int16, mode="r")
    val = np.memmap(R3 / "exact/value.bin", dtype=np.float16, mode="r")
    h = np.load(R3 / "calib/hist.npz")
    hall = (h["same"] + h["diff"]).astype(np.float64)
    up = np.cumsum(hall[::-1])[::-1] / hall.sum()
    p_ge = np.array([up[min(1000 + k, 1999)] for k in range(1001)])
    gids = np.load(S.OUT1 / "topk/gids.npy")
    gc = np.searchsorted(gs, gids, side="right") - 1
    rows = {k: [] for k in MCAP}
    for g, own in zip(gids, gc):
        U = uq[g]
        G = np.asarray(gen[uptr[U]:uptr[U + 1]], np.int64)
        c = np.asarray(val[uptr[U]:uptr[U + 1]], np.float32)
        p = gs[G] + np.asarray(loc[uptr[U]:uptr[U + 1]], np.int64)
        keep = G != own
        G, c, p = G[keep], c[keep], p[keep]
        o = np.argsort(-c, kind="stable")
        G, c, p = G[o], c[o], p[o]
        lev = pd.Series(lin_g[G]).groupby(lin_g[G]).cumcount().to_numpy()
        ndb = N - (ge[own] - gs[own])
        t = 1.0 + ndb * p_ge[np.clip(np.floor(c * 1000 + 1e-4).astype(np.int64), 0, 1000)]
        s = np.log(ndb / t)
        for k, m in MCAP.items():
            sel = lev < m
            rows[k].append((p[sel].astype(np.int32), s[sel].astype(np.float32)))
    info = {}
    for k, rr in rows.items():
        K = max(1, max(len(x) for x, _ in rr))
        idx = np.full((len(rr), K), -1, np.int32)
        sv = np.zeros((len(rr), K), np.float32)
        for i, (x, y) in enumerate(rr):
            idx[i, :len(x)] = x
            sv[i, :len(y)] = y
        d = OUT / "lists" / k
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "gids.npy", gids)
        np.save(d / "idx.npy", idx)
        np.save(d / "s.npy", sv)
        ln = np.array([len(x) for x, _ in rr])
        info[k] = dict(K=int(K), entries_per_gene_q10_q50_q90=np.quantile(ln, [.1, .5, .9]).tolist(),
                       s_q10_q50_q90=np.quantile(np.concatenate([y for _, y in rr]), [.1, .5, .9]).tolist())
    info["seconds"] = round(time.time() - t0, 1)
    S.dump(info, OUT / "lists/lists.json")
    log("lists", json.dumps(info))


def cmd_grid(a):
    a.set = "strat"
    S.grid(a)


def odd_configs():
    G = pd.read_parquet(OUT / "grid_strat.parquet")
    G = G[G.n_rel > 0]
    return {f: S.choose(G[(G.fam == f) & (G.split == "odd")]) for f in NEW}


def cmd_final(a):
    odd = odd_configs()
    S.dump(dict(odd=odd, note="odd-genome DefenseFinder grid; same choose() as round 3"), OUT / "selection_strat.json")
    db, callers, queries = S.setup(a, S.fam_sources(list(NEW)))
    Q = queries[queries.caller == "DefenseFinder"]
    tasks = sorted({(int(q), f, c["w"], c["wx"], c["tau"]) for q in Q.qid for f, c in odd.items()},
                   key=lambda t: (-int(queries.n_members[t[0]]), t))
    log(f"final strat: {len(tasks):,} runs; configs {odd}")
    t0 = time.time()
    res = bench.pool_map(S.final_task, tasks, a.threads, chunksize=1)
    log(f"final strat: {time.time() - t0:.0f}s")
    M = pd.DataFrame([m for m, _, _ in res]).merge(queries, on="qid")
    M.to_parquet(OUT / "final_strat_metrics.parquet", index=False)
    pd.concat([h for _, h, _ in res], ignore_index=True).to_parquet(OUT / "final_strat_hits.parquet", index=False)
    np.savez(OUT / "final_strat_tails.npz", qid=np.array([m["qid"] for m, _, _ in res]), fam=np.array([m["fam"] for m, _, _ in res]),
             tails=np.stack([t for _, _, t in res]))


def cmd_analyse(a):
    import evidence as evm
    t0 = time.time()
    rng = np.random.default_rng(S.SEED)
    sel_o = json.load(open(ORIG / "selection3.json"))
    cz = sel_o["odd"]["M1z"]
    odd = json.load(open(OUT / "selection_strat.json"))["odd"]
    queries = pd.read_parquet(S.OUT1 / "queries.parquet")
    Mo = pd.read_parquet(ORIG / "final_df_metrics.parquet")
    Mz = Mo[(Mo.fam == "M1z") & (Mo.w == cz["w"]) & (Mo.wx == cz["wx"]) & (Mo.tau == cz["tau"])].drop_duplicates("qid")
    Mn = pd.read_parquet(OUT / "final_strat_metrics.parquet")
    ok = set(Mo[Mo.n_rel > 0].qid)
    res = dict(configs=dict(M1z=cz, **odd), ap_even={}, paired_ap_even={}, curves={}, calibration_even={}, at_E0_01={},
               matched={}, paired_matched={})
    ev_q = lambda M: M[(M.split == "even") & M.qid.isin(ok)].set_index("qid")  # noqa: E731
    z = ev_q(Mz)
    res["ap_even"]["M1z"] = dict(ap=S.boot(z.ap, z.genome_code), ap_strict=S.boot(z.ap_strict, z.genome_code),
                                 recall_E200_0p01=S.boot(z["recall_E0.01"], z.genome_code), queries=int(len(z)))
    for f in NEW:
        x = ev_q(Mn[Mn.fam == f])
        ids = x.index.intersection(z.index)
        res["ap_even"][f] = dict(ap=S.boot(x.ap, x.genome_code), ap_strict=S.boot(x.ap_strict, x.genome_code),
                                 recall_E200_0p01=S.boot(x["recall_E0.01"], x.genome_code), queries=int(len(x)),
                                 wrong_E200_0p01=S.boot(x["n_wrong_E0.01"], x.genome_code),
                                 unlabelled_E200_0p01=S.boot(x["n_unlabelled_E0.01"], x.genome_code))
        res["paired_ap_even"][f"{f}-M1z"] = {c: S.boot(x.loc[ids, c] - z.loc[ids, c], z.loc[ids, "genome_code"])
                                             for c in ("ap", "ap_strict", "recall_E0.01", "n_wrong_E0.01", "n_unlabelled_E0.01")}
    log("AP even", json.dumps(res["paired_ap_even"], default=S.jd))
    # fast null curves for the new families (odd runs), E_fast for their hits
    zt = np.load(OUT / "final_strat_tails.npz", allow_pickle=True)
    T = pd.DataFrame(dict(qid=zt["qid"], fam=zt["fam"]))
    T["split"] = T.qid.map(dict(zip(queries.qid, queries.split)))
    Hn = pd.read_parquet(OUT / "final_strat_hits.parquet")
    Hs = []
    for f in NEW:
        oi = np.flatnonzero(((T.fam == f) & (T.split == "odd")).to_numpy())
        ei = np.flatnonzero(((T.fam == f) & (T.split == "even")).to_numpy())
        curve = fastnull.calibration_curve(np.array([S.null_counts(zt["tails"][i]) for i in oi]))
        XG = S.XG
        res["curves"][f] = {str(x): float(curve[int(np.argmin(np.abs(np.log10(XG) - np.log10(x))))] / x) for x in (0.001, 0.01, 0.1, 1.0)}
        cnt_even = np.array([S.null_counts(zt["tails"][i]) for i in ei])
        xprime = 10.0 ** np.interp(np.log10(XG), np.log10(curve), np.log10(XG))
        cal = np.array([np.interp(np.log10(xprime), np.log10(XG), row) for row in cnt_even])
        j = int(np.argmin(np.abs(np.log10(XG) - np.log10(0.01))))
        res["calibration_even"][f] = float(cal[:, j].mean() / 0.01)
        h = Hn[(Hn.fam == f) & Hn.qid.isin(ok)].copy()
        h["E_fast"] = S.apply_curve_ext(h.Egum40.to_numpy(), curve)
        Hs.append(h)
    Hn = pd.concat(Hs, ignore_index=True).merge(queries[["qid", "split", "genome_code", "tcode", "row"]], on="qid")
    # unsupported classification of unlabelled hits with E_fast <= 10 (round-3 evidence rule)
    try:
        db = evm.attach_db()
        callers = bench.load_gold(db)
        U = Hn[(Hn.label == 0) & (Hn.E_fast <= 10)].reset_index(drop=True)
        F, _ = evm.classify(evm.Evidence(db), U, queries, callers, rng, controls=False)
        U["supported"] = F.supported.to_numpy()
        Hn = Hn.merge(U[["qid", "fam", "start_gid", "end_gid", "supported"]], on=["qid", "fam", "start_gid", "end_gid"], how="left")
        Hn["unsupported"] = (Hn.label == 0) & (Hn.supported == False)  # noqa: E712
        res["unsupported_rule"] = "round-3 evidence rule (evidence.classify)"
    except Exception as e:  # report, do not hide
        log("unsupported classification failed:", repr(e))
        Hn["unsupported"] = False
        res["unsupported_rule"] = f"FAILED: {e!r}"
        db = ls.Database(vectors=None)
        callers = bench.load_gold(db)
    Hn.to_parquet(OUT / "rates_hits_strat.parquet", index=False)
    Hz = pd.read_parquet(ORIG / "rates_hits.parquet")
    Hz = Hz[Hz.fam == "M1z"]
    rel = bench.load_relevance(S.OUT1, queries, callers)
    Hall = pd.concat([Hz[["qid", "fam", "split", "genome_code", "E_fast", "label", "match_row", "unsupported", "score"]],
                      Hn[["qid", "fam", "split", "genome_code", "E_fast", "label", "match_row", "unsupported", "score"]]],
                     ignore_index=True)
    Q = queries[(queries.caller == "DefenseFinder") & queries.qid.isin(ok)]
    qs = {s: Q[Q.split == s] for s in ("odd", "even")}
    nq = {s: len(v) for s, v in qs.items()}
    C = Hall[Hall.label == 1].groupby(["fam", "qid", "match_row"]).E_fast.min().rename("Emin").reset_index()
    kinds = {"wrong": (lambda h: h.label == 2, (0.25, 0.5, 1.0)), "unlabelled": (lambda h: h.label == 0, (1.0, 2.0, 4.0)),
             "unsupported": (lambda h: h.unsupported.fillna(False).astype(bool), (0.01, 0.03, 0.1))}
    rec = {}
    for fam in ["M1z"] + list(NEW):
        h = Hall[Hall.fam == fam]
        cq = {k: g.Emin.to_numpy() for k, g in C[C.fam == fam].groupby("qid")}
        for kind, (fn, levels) in kinds.items():
            xo = np.sort(h[(h.split == "odd") & fn(h)].E_fast.to_numpy())
            for L in levels:
                allowed = int(math.floor(L * nq["odd"] + 1e-9))
                strict = len(xo) > allowed
                thr = float(xo[allowed]) if strict else 10.0
                he = h[(h.split == "even") & ((h.E_fast < thr) if strict else (h.E_fast <= thr))]
                rows = []
                for qid, g_ in zip(qs["even"].qid, qs["even"].genome_code):
                    e = cq.get(int(qid), np.empty(0))
                    rows.append((int(qid), int(g_), int(((e < thr) if strict else (e <= thr)).sum()) / len(rel[int(qid)][0])))
                R = pd.DataFrame(rows, columns=["qid", "genome_code", "recall"])
                rec[(fam, kind, L)] = R
                res["matched"][f"{fam}|{kind}|{L:g}"] = dict(threshold=thr, reached=strict, recall=S.boot(R.recall, R.genome_code),
                                                             even_rate=float(fn(he).sum() / nq["even"]))
        e001 = h[(h.split == "even") & (h.E_fast <= 0.01)]
        res["at_E0_01"][fam] = dict(wrong_per_query=float((e001.label == 2).sum() / nq["even"]),
                                    unlabelled_per_query=float((e001.label == 0).sum() / nq["even"]),
                                    unsupported_per_query=float(e001.unsupported.fillna(False).astype(bool).sum() / nq["even"]))
    for kind, (_, levels) in kinds.items():
        for L in levels:
            for f in NEW:
                A, B = rec[(f, kind, L)], rec[("M1z", kind, L)]
                assert np.array_equal(A.qid.to_numpy(), B.qid.to_numpy())
                res["paired_matched"][f"{f}-M1z|{kind}|{L:g}"] = S.boot(A.recall.to_numpy() - B.recall.to_numpy(), B.genome_code)
    res["queries"] = nq
    res["seconds"] = round(time.time() - t0, 1)
    S.dump(res, OUT / "summary_strat.json")
    log("matched", json.dumps({k: v[0] for k, v in res["paired_matched"].items()}, default=S.jd))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["lists", "zpg", "grid", "final", "analyse"])
    ap.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_ON_NODE", "2")))
    a = ap.parse_args()
    import numba
    numba.set_num_threads(a.threads)
    {"lists": cmd_lists, "zpg": cmd_zpg, "grid": cmd_grid, "final": cmd_final, "analyse": cmd_analyse}[a.stage](a)


if __name__ == "__main__":
    main()
