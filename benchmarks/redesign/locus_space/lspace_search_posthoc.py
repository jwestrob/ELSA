#!/usr/bin/env python
"""POST HOC (written after the part-4 result: at matched wrong-type rates the locus-vector arms return 57-131
unlabelled hits per query against 0.5-3.4 for M1z). Recall at matched rates of unlabelled hits (1 / 2 / 4 per query)
and of unsupported unlabelled hits (0.01 / 0.03 / 0.1 per query; round-2 evidence rule: a raw DefenseFinder profile of
the query's type in the window, or a shared engine module), thresholds on odd queries, recall on even queries, for
the locus-index and window arms and M1z (its stored hits, rates_hits.parquet, same procedure as round 3's post hoc).
Unlabelled hits are classified within the top 300 hits per query (by score).
  lspace_search_posthoc.py --threads T --rep MEAN
"""
from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lspace import log, dump, OUT, RD  # noqa: E402

sys.path.insert(1, str(RD / "locus_search/code"))
import bench  # noqa: E402
import evidence as evm  # noqa: E402
import locus_search as lsr  # noqa: E402

OUT1 = RD / "locus_search/out"
OUT3 = RD / "locus_search/out3"
SRCH = OUT / "search"
TOPN = 300
LEVELS = {"unlabelled": (1.0, 2.0, 4.0), "unsupported": (0.01, 0.03, 0.1), "wrong": (0.25, 0.5, 1.0)}
CTX = {}


def _classify(chunk):
    ev, queries, callers = CTX["ev"], CTX["queries"], CTX["callers"]
    F, _ = evm.classify(ev, chunk, queries, callers, None, controls=False)
    return chunk.index.to_numpy(), F.supported.to_numpy(), F.df_any.to_numpy()


def matched(H, kind, qs_odd, qs_even, rel, col, higher):
    sgn = 1.0 if higher else -1.0
    v = sgn * H[col].to_numpy()
    flag = {"wrong": H.label.to_numpy() == 2, "unlabelled": H.label.to_numpy() == 0,
            "unsupported": H.unsupported.to_numpy()}[kind]
    odd = H.split.to_numpy() == "odd"
    xo = np.sort(v[odd & flag])[::-1]
    Hc = H[H.label == 1].assign(v=sgn * H[H.label == 1][col])
    best = Hc.groupby(["qid", "match_row"]).v.max().reset_index()
    bq = {k: g.v.to_numpy() for k, g in best.groupby("qid")}
    out, per = {}, {}
    for L in LEVELS[kind]:
        allowed = int(math.floor(L * len(qs_odd) + 1e-9))
        thr = xo[allowed] if len(xo) > allowed else -np.inf
        R = pd.DataFrame([(int(q), int(g), float((bq.get(int(q), np.empty(0)) > thr).sum() / len(rel[int(q)][0])))
                          for q, g in zip(qs_even.qid, qs_even.genome_code)], columns=["qid", "genome_code", "recall"])
        ev = (H.split.to_numpy() == "even") & (v > thr)
        out[f"{L:g}"] = dict(recall=lsr.boot_mean(R.recall, R.genome_code),
                             even_rate=float((flag & ev).sum() / len(qs_even)),
                             even_wrong=float(((H.label.to_numpy() == 2) & ev).sum() / len(qs_even)),
                             even_unlabelled=float(((H.label.to_numpy() == 0) & ev).sum() / len(qs_even)))
        per[L] = R.set_index("qid")
    return out, per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rep", default="MEAN")
    ap.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_ON_NODE", 4)))
    a = ap.parse_args()
    T0 = time.time()
    db = evm.attach_db()
    callers = bench.load_gold(db)
    queries = pd.read_parquet(OUT1 / "queries.parquet")
    rel = bench.load_relevance(OUT1, queries, callers)
    ev = evm.Evidence(db)
    CTX.update(ev=ev, queries=queries, callers=callers)
    Q = queries[(queries.caller == "DefenseFinder")]
    Q = Q[[len(rel[int(q)][0]) > 0 for q in Q.qid]]
    qs_odd, qs_even = Q[Q.split == "odd"], Q[Q.split == "even"]
    res = dict(note="POST HOC; not predeclared", top_n_classified=TOPN, arms={})
    per_all = {}
    arms = [arm for arm in ("locusindex", "window") if (SRCH / f"hits_{arm}_{a.rep}.parquet").exists()]
    for arm in arms:
        H = pd.read_parquet(SRCH / f"hits_{arm}_{a.rep}.parquet")
        H = H.sort_values(["qid", "score"], ascending=[True, False])
        H["rank"] = H.groupby("qid").cumcount()
        H = H[H["rank"] < TOPN].reset_index(drop=True)
        U = H[H.label == 0]
        chunks = [U.iloc[i:i + 2000] for i in range(0, len(U), 2000)]
        sup = np.zeros(len(H), bool)
        dfa = np.zeros(len(H), bool)
        t0 = time.time()
        with mp.get_context("fork").Pool(a.threads) as pool:
            for idx, s, d in pool.imap_unordered(_classify, chunks):
                sup[idx] = s
                dfa[idx] = d
        H["supported"] = sup
        H["df_any"] = dfa
        H["unsupported"] = (H.label == 0) & ~H.supported
        log(f"{arm}: {len(U):,} unlabelled hits classified in {time.time() - t0:.0f}s; supported "
            f"{H.supported[H.label == 0].mean():.3f}")
        d = dict(unlabelled_top300=int(len(U)), supported_frac=float(H.supported[H.label == 0].mean()),
                 df_any_frac=float(H.df_any[H.label == 0].mean()))
        for kind in LEVELS:
            d[kind], per_all[(arm, kind)] = matched(H, kind, qs_odd, qs_even, rel, "score", True)
        res["arms"][arm] = d
        H.to_parquet(SRCH / f"hits_{arm}_{a.rep}_classified.parquet", index=False)
    Hm = pd.read_parquet(OUT3 / "rates_hits.parquet")
    Hm = Hm[Hm.fam == "M1z"].reset_index(drop=True)
    Hm["unsupported"] = Hm.unsupported.fillna(False).astype(bool)
    d = {}
    for kind in LEVELS:
        d[kind], per_all[("m1z", kind)] = matched(Hm, kind, qs_odd, qs_even, rel, "E_fast", False)
    res["arms"]["m1z"] = d
    res["paired_minus_m1z"] = {}
    for arm in arms:
        for kind, lv in LEVELS.items():
            for L in lv:
                A, B = per_all[(arm, kind)][L], per_all[("m1z", kind)][L]
                res["paired_minus_m1z"][f"{arm}|{kind}|{L:g}"] = lsr.boot_mean(
                    A.recall.to_numpy() - B.loc[A.index, "recall"].to_numpy(), A.genome_code.to_numpy())
    res["seconds"] = time.time() - T0
    dump(res, SRCH / f"posthoc_{a.rep}.json")
    log("posthoc done", res["seconds"])


if __name__ == "__main__":
    main()
