#!/usr/bin/env python
"""Sequence-rank control for the embedding gain (DefenseFinder).

  seq_control.py query  --out DIR            query FASTA of the member genes (same order as out/topk/gids.npy)
  (sbatch: mmseqs2 search of those genes against all 3,585,303 proteins, -s 7.5, -e 10, --max-seqs 20000)
  seq_control.py topk   --out DIR            per member gene: hits outside its genome ranked by bitscore,
                                             s = log(N_db / tail rank) -> out/topk_seq/{gids,idx,s}.npy
  seq_control.py bench  --out DIR --threads  S1 (sequence rank only) and S2 (max(token, w * sequence rank)),
                                             tuned on odd-genome DefenseFinder queries, tested on even,
                                             paired against M0, M1 and M2 from final_metrics_dfr1.parquet

Question: is the M2 gain specific to the language-model embedding, or does any graded per-gene
similarity rank (here MMseqs2 bitscore) give it?
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench  # noqa: E402
import locus_search as ls  # noqa: E402
from locus_search import log  # noqa: E402

FASTA = ls.RD / "ws_dpann/work/proteins_gid.faa"


def write_query(out):
    gids = set(np.load(out / "topk/gids.npy").tolist())
    d = out / "seqsearch"
    d.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(FASTA) as f, open(d / "query.faa", "w") as w:
        keep = False
        for line in f:
            if line.startswith(">"):
                keep = int(line[1:].split()[0]) in gids
                n += keep
            if keep:
                w.write(line)
    log(f"query FASTA: {n:,} of {len(gids):,} member genes")


def build_topk(out, K=20000):
    import pyarrow.csv as pc
    db = ls.Database(vectors=None)
    gids = np.load(out / "topk/gids.npy")
    t0 = time.time()
    tab = pc.read_csv(out / "seqsearch/hits.tsv", read_options=pc.ReadOptions(column_names=["q", "t", "bits", "evalue"]),
                      parse_options=pc.ParseOptions(delimiter="\t"),
                      convert_options=pc.ConvertOptions(column_types={"q": "int64", "t": "int64", "bits": "float64", "evalue": "float64"}))
    q, t, b = (tab.column(k).to_numpy() for k in ("q", "t", "bits"))
    log(f"read {len(q):,} hits in {time.time() - t0:.0f}s")
    ok = db.gc[q] != db.gc[t]                                # database = other genomes
    q, t, b = q[ok], t[ok], b[ok]
    o = np.lexsort((t, -b, q))
    q, t, b = q[o], t[o], b[o]
    row_of = {int(g): k for k, g in enumerate(gids)}
    idx = np.full((len(gids), K), -1, np.int32)
    s = np.zeros((len(gids), K), np.float32)
    starts = np.flatnonzero(np.r_[True, q[1:] != q[:-1]])
    ends = np.r_[starts[1:], len(q)]
    nh = []
    for a, e in zip(starts, ends):
        g = int(q[a])
        k = row_of[g]
        bb = b[a:e][:K]
        n = len(bb)
        # tail rank with ties: number of hits with bitscore >= this one
        rank = np.searchsorted(-bb, -bb, side="right")
        idx[k, :n] = t[a:a + n]
        s[k, :n] = np.log(db.n_db(int(db.gc[g])) / rank)
        nh.append(e - a)
    d = out / "topk_seq"
    d.mkdir(exist_ok=True)
    np.save(d / "gids.npy", gids)
    np.save(d / "idx.npy", idx)
    np.save(d / "s.npy", s)
    nh = np.array(nh)
    info = dict(genes=int(len(gids)), genes_with_hits=int(len(nh)), hits_per_gene_quantiles=np.quantile(nh, [.1, .5, .9, .99]).tolist(),
                K=K, seconds=round(time.time() - t0, 1))
    json.dump(info, open(d / "topk_seq.json", "w"), indent=1)
    log("sequence top-K", info)


def run_bench(a):
    out = a.out
    db, callers, queries = bench.setup(a, vectors=None)
    gids = np.load(out / "topk_seq/gids.npy")
    bench.CTX.update(row_of={int(g): k for k, g in enumerate(gids)}, tk_idx=np.load(out / "topk_seq/idx.npy"),
                     tk_s=np.load(out / "topk_seq/s.npy"))
    bench.CTX["rel"] = bench.load_relevance(out, queries, callers)
    bench.warm_up(db)
    # 'emb' / 'comb' modes now read the sequence ranks: S1 = rank only, S2w = max(token, w * rank)
    bench.MODES.clear()
    bench.MODES.update({"S1": ("emb", 1.0), **{f"S2w{w:g}": ("comb", w) for w in bench.GRID_W}})
    df = queries[queries.caller == "DefenseFinder"]
    tq = df[df.split == "odd"].qid.tolist()
    t0 = time.time()
    rows = [r for ch in bench.pool_map(bench.tune_task, tq, a.threads) for r in ch]
    tune = pd.DataFrame(rows)
    tune.to_parquet(out / "seqctl_tune_odd.parquet", index=False)
    agg = tune[tune.n_rel > 0].groupby(["mode", "wx", "tau"]).agg(map=("ap", "mean")).reset_index()
    best = {}
    for method, sub in (("S1", agg[agg["mode"] == "S1"]), ("S2", agg[agg["mode"].str.startswith("S2")])):
        b = sub.sort_values(["map", "wx", "tau"], ascending=[False, True, False]).iloc[0]
        mode, w = bench.MODES[b["mode"]]
        best[method] = dict(mode=mode, w=w, wx=int(b.wx), tau=float(b.tau), odd_map=float(b["map"]), grid_name=b["mode"])
    log("tuning", f"{time.time() - t0:.0f}s", json.dumps(best))
    tasks = [(int(q), m, best[m]["mode"], best[m]["w"], best[m]["wx"], best[m]["tau"]) for q in df.qid for m in ("S1", "S2")]
    res = bench.pool_map(bench.final_task, tasks, a.threads)
    M = pd.DataFrame([r for r, _, _ in res]).merge(queries, on="qid")
    M.to_parquet(out / "final_metrics_seqctl.parquet", index=False)
    H = bench.hits_frame([h for _, h, _ in res])
    H.to_parquet(out / "final_hits_E10_seqctl.parquet", index=False)
    # paired comparison against the round-1 embedding runs
    R = pd.read_parquet(out / "final_metrics_dfr1.parquet")
    A = pd.concat([R[R.method.isin(["M0", "M1", "M2"])], M], ignore_index=True)
    A = A[A.n_rel > 0]
    summ = dict(best=best, tables={}, paired={})
    cols = ["ap", "ap_strict", "recall_E0.01", "recall_E1", "recall_inv_E0.01", "recall_inv_E1", "n_unlabelled_E0.01"]
    for split in ("odd", "even"):
        for mm in (2, 3):
            S = A[(A.split == split) & (A.n_members >= mm)]
            key = f"DefenseFinder|{split}|min{mm}"
            summ["tables"][key] = {m: {c: ls.boot_mean(g[c].to_numpy(), g.genome_code.to_numpy()) for c in cols}
                                   for m, g in S.groupby("method")}
            pr = {}
            for x, y in (("S1", "M0"), ("S2", "M0"), ("M2", "S2"), ("M1", "S1"), ("M2", "M0")):
                gx, gy = S[S.method == x].set_index("qid"), S[S.method == y].set_index("qid")
                ids = gx.index.intersection(gy.index)
                gc = gy.loc[ids, "genome_code"].to_numpy()
                pr[f"{x}-{y}"] = {c: ls.boot_mean(gx.loc[ids, c].to_numpy() - gy.loc[ids, c].to_numpy(), gc) for c in cols}
            summ["paired"][key] = pr
    json.dump(summ, open(out / "summary_seqctl.json", "w"), indent=1, default=float)
    log("paired even min2", json.dumps(summ["paired"]["DefenseFinder|even|min2"], default=float)[:1500])


def entries3(q, mode, w):
    """mode 'comb3': per slot s = max(s_token, s_sequence_rank, w * s_embedding_rank); else delegate."""
    if mode != "comb3":
        return ENTRIES_ORIG(q, mode, w)
    C = bench.CTX
    db = C["db"]
    own = q.genome_code
    lists = []
    for g in bench.q_slots(q):
        k = C["row_of"][int(g)]
        gt = db.token_members(int(db.tok[g]), own)
        st = np.full(len(gt), math.log(db.n_db(own) / len(gt)) if len(gt) else 0.0)
        si, ss = C["tk_idx"][k], C["tk_s"][k]
        ei, es = C["ek_idx"][k], C["ek_s"][k]
        gg = np.concatenate([gt, si[si >= 0].astype(np.int64), ei[ei >= 0].astype(np.int64)])
        sv = np.concatenate([st, ss[si >= 0].astype(np.float64), w * es[ei >= 0].astype(np.float64)])
        o = np.lexsort((-sv, gg))
        gg, sv = gg[o], sv[o]
        first = np.r_[True, gg[1:] != gg[:-1]] if len(gg) else np.zeros(0, bool)
        lists.append((gg[first], sv[first]))
    return ls.assemble(lists), len(lists)


ENTRIES_ORIG = bench.entries_for


def run_bench3(a):
    """S3 = token + sequence rank + w * embedding rank; tuned on odd DefenseFinder, paired against S2 on even."""
    out = a.out
    db, callers, queries = bench.setup(a, vectors=None)
    gids = np.load(out / "topk_seq/gids.npy")
    assert np.array_equal(gids, np.load(out / "topk/gids.npy"))
    bench.CTX.update(row_of={int(g): k for k, g in enumerate(gids)}, tk_idx=np.load(out / "topk_seq/idx.npy"),
                     tk_s=np.load(out / "topk_seq/s.npy"), ek_idx=np.load(out / "topk/idx.npy"), ek_s=np.load(out / "topk/s.npy"))
    bench.CTX["rel"] = bench.load_relevance(out, queries, callers)
    bench.entries_for = entries3
    bench.warm_up(db)
    bench.MODES.clear()
    bench.MODES.update({f"S3w{w:g}": ("comb3", w) for w in (0.25, 0.5, 0.75, 1.0)})
    df = queries[queries.caller == "DefenseFinder"]
    rows = [r for ch in bench.pool_map(bench.tune_task, df[df.split == "odd"].qid.tolist(), a.threads) for r in ch]
    tune = pd.DataFrame(rows)
    tune.to_parquet(out / "seqctl3_tune_odd.parquet", index=False)
    agg = tune[tune.n_rel > 0].groupby(["mode", "wx", "tau"]).agg(map=("ap", "mean")).reset_index()
    agg.to_csv(out / "seqctl3_tune_odd_summary.tsv", sep="\t", index=False)
    b = agg.sort_values(["map", "wx", "tau"], ascending=[False, True, False]).iloc[0]
    mode, w = bench.MODES[b["mode"]]
    best = dict(mode=mode, w=w, wx=int(b.wx), tau=float(b.tau), odd_map=float(b["map"]), grid_name=b["mode"])
    log("S3 tuning", json.dumps(best))
    tasks = [(int(q), "S3", mode, w, best["wx"], best["tau"]) for q in df.qid]
    res = bench.pool_map(bench.final_task, tasks, a.threads)
    M = pd.DataFrame([r for r, _, _ in res]).merge(queries, on="qid")
    M.to_parquet(out / "final_metrics_seqctl3.parquet", index=False)
    S = pd.read_parquet(out / "final_metrics_seqctl.parquet")
    R = pd.read_parquet(out / "final_metrics_dfr1.parquet")
    A = pd.concat([R[R.method.isin(["M0", "M2"])], S, M], ignore_index=True)
    A = A[A.n_rel > 0]
    cols = ["ap", "ap_strict", "recall_E0.01", "recall_E1", "recall_inv_E0.01", "recall_inv_E1", "n_unlabelled_E0.01"]
    summ = dict(best=best, tables={}, paired={})
    for split in ("odd", "even"):
        for mm in (2, 3):
            X = A[(A.split == split) & (A.n_members >= mm)]
            key = f"DefenseFinder|{split}|min{mm}"
            summ["tables"][key] = {m: {c: ls.boot_mean(g[c].to_numpy(), g.genome_code.to_numpy()) for c in cols}
                                   for m, g in X.groupby("method") if m == "S3"}
            pr = {}
            for x, y in (("S3", "S2"), ("S3", "M2"), ("S3", "M0")):
                gx, gy = X[X.method == x].set_index("qid"), X[X.method == y].set_index("qid")
                ids = gx.index.intersection(gy.index)
                gc = gy.loc[ids, "genome_code"].to_numpy()
                pr[f"{x}-{y}"] = {c: ls.boot_mean(gx.loc[ids, c].to_numpy() - gy.loc[ids, c].to_numpy(), gc) for c in cols}
            summ["paired"][key] = pr
    json.dump(summ, open(out / "summary_seqctl3.json", "w"), indent=1, default=float)
    log("S3 paired even min2", json.dumps(summ["paired"]["DefenseFinder|even|min2"], default=float)[:1500])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["query", "topk", "bench", "bench3"])
    ap.add_argument("--out", type=Path, default=ls.RD / "locus_search/out")
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    import numba
    numba.set_num_threads(a.threads)
    if a.step == "query":
        write_query(a.out)
    elif a.step == "topk":
        build_topk(a.out)
    elif a.step == "bench3":
        run_bench3(a)
    else:
        run_bench(a)


if __name__ == "__main__":
    main()
