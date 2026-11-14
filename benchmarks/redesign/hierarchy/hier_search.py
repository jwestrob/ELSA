#!/usr/bin/env python
"""Task 3: hierarchy evidence in locus search, no per-query similarity search (workstream hierarchy).

Evidence for query gene a and database gene b: s(a, b) = max over levels l with tok_l(a) = tok_l(b) of
log(N_db / n_l(a)), where n_l(a) = genes of a's level-l cluster outside the query genome and N_db = genes outside
the query genome. One level = M0 (round-1 token evidence) exactly. Everything else is round-1 locus search
unchanged (locus_search.py scan / null / labels, bench.py queries, relevance, windows, tie-breaking).

  hier_search.py tune  --out DIR   odd-genome DefenseFinder queries, scan only, grid wx x tau per variant
  hier_search.py final --out DIR   even-genome queries at each variant's chosen configuration, 200-replicate null
  hier_search.py summary --out DIR paired comparisons with the frozen M0 / M2 / S2 / S3 per-query metrics
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

RD = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111")
sys.path.insert(0, str(RD / "locus_search/code"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench  # noqa: E402
import locus_search as ls  # noqa: E402
from levels import LEVEL_SETS, by_tokens, load_tokens  # noqa: E402

FROZEN = RD / "locus_search/out2/final_dfall_metrics.parquet"
R1 = RD / "locus_search/out"
CALLER = "DefenseFinder"
GRID_WX = (0, 2, 4, 8)
GRID_TAU = (0.0, 1.0, 2.0, 3.0)

VARIANTS = {
    "H_emb_L8": ("emb", LEVEL_SETS["L8"]),
    "H_emb_L13": ("emb", LEVEL_SETS["L13"]),
    "H_seq_S3": ("seq", LEVEL_SETS["S3"]),
    "H_both_L8S3": ("both", by_tokens(LEVEL_SETS["L8"] + LEVEL_SETS["S3"])),
    "H_both_L13S3": ("both", by_tokens(LEVEL_SETS["L13"] + LEVEL_SETS["S3"])),
    "chk_e3c50s75": ("check", ["e3c50s75"]),
    "ref_r19000": ("ref", ["r19000"]),
}
NESTED = {
    "N_emb_L8": ("emb_nested", [f"nested__L8__{x}" for x in LEVEL_SETS["L8"]]),
    "N_emb_L13": ("emb_nested", [f"nested__L13__{x}" for x in LEVEL_SETS["L13"]]),
}
LV = {}          # level name -> (tok, order, ptr)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def build_levels(names, N):
    for x in names:
        if x in LV:
            continue
        tok = load_tokens(x, N)
        assert len(tok) == N
        order = np.argsort(tok, kind="stable")
        order = order[tok[order] >= 0].astype(np.int64)
        ptr = np.searchsorted(tok[order], np.arange(int(tok.max()) + 2)).astype(np.int64)
        LV[x] = (tok, order, ptr)


def slot_list(db, own, g, names):
    """(sorted gids, evidence) of one query gene under hierarchy evidence."""
    N_db = db.n_db(own)
    lo, hi = db.own_range(own)
    gs, ss = [], []
    for x in names:
        tok, order, ptr = LV[x]
        t = tok[g]
        if t < 0:
            continue
        m = order[ptr[t]:ptr[t + 1]]
        a, b = np.searchsorted(m, lo), np.searchsorted(m, hi)
        m = np.concatenate([m[:a], m[b:]])
        if len(m):
            gs.append(m)
            ss.append(np.full(len(m), math.log(N_db / len(m))))
    if not gs:
        return np.empty(0, np.int64), np.empty(0, np.float64)
    g_ = np.concatenate(gs)
    s_ = np.concatenate(ss)
    o = np.lexsort((-s_, g_))
    g_, s_ = g_[o], s_[o]
    first = np.r_[True, g_[1:] != g_[:-1]]
    return g_[first], s_[first]


def entries(q, names):
    db = bench.CTX["db"]
    lists = [slot_list(db, q.genome_code, int(g), names) for g in bench.q_slots(q)]
    return ls.assemble(lists), len(lists)


def tune_task(qid):
    q = bench.CTX["queries"].loc[qid]
    db = bench.CTX["db"]
    out = []
    for var, (_, names) in bench.CTX["variants"].items():
        t0 = time.time()
        (eg, es_, ev), ns = entries(q, names)
        t_ev = time.time() - t0
        for wx, tau in product(GRID_WX, GRID_TAU):
            t1 = time.time()
            ha, hb, sc, _ = ls.scan(eg, es_, ev, ns, db.cc, db.pos, bench.W_of(q, wx), tau, 0)
            r, _, _ = bench.label_eval(q, ha, hb, sc, None)
            r.update(qid=int(qid), variant=var, wx=wx, tau=tau, sec_scan=time.time() - t1, sec_evidence=t_ev,
                     n_entries=int(len(eg)))
            out.append(r)
    return out


def final_task(args):
    qid, var, wx, tau = args
    q = bench.CTX["queries"].loc[qid]
    db = bench.CTX["db"]
    names = bench.CTX["variants"][var][1]
    t0 = time.time()
    (eg, es_, ev), ns = entries(q, names)
    t1 = time.time()
    W = bench.W_of(q, wx)
    ha, hb, sc, nf = ls.scan(eg, es_, ev, ns, db.cc, db.pos, W, tau, 0)
    t2 = time.time()
    lo, hi = db.own_range(q.genome_code)
    tail = ls.null_tail(eg, es_, ev, ns, db.cc, db.pos, db.N, lo, hi, W, tau, bench.R_NULL, bench.SEED + int(qid), bench.TOP_N)
    E, _ = ls.evalues(sc, tail, bench.R_NULL)
    t3 = time.time()
    r, lab, match = bench.label_eval(q, ha, hb, sc, E)
    r.update(qid=int(qid), method=var, W=W, tau=tau, wx=wx, n_entries=int(len(eg)), sec_evidence=t1 - t0,
             sec_scan=t2 - t1, sec_null=t3 - t2)
    keep = E <= 10
    hits = pd.DataFrame(dict(qid=int(qid), method=var, start_gid=ha[keep], end_gid=hb[keep], score=sc[keep], evalue=E[keep],
                             label=lab[keep], match_row=match[keep], n_slots_filled=nf[keep]))
    return r, hits


def setup(a, variants):
    db, callers, queries = bench.setup(a, vectors="none")
    names = sorted({x for _, v in variants.values() for x in v})
    t0 = time.time()
    build_levels(names, db.N)
    log(f"levels {len(names)} built {time.time() - t0:.0f}s")
    bench.CTX["rel"] = bench.load_relevance(R1, queries, callers)
    bench.CTX["variants"] = variants
    bench.warm_up(db)
    return db, queries


def pick_variants(a):
    v = dict(VARIANTS)
    if a.nested:
        v.update(NESTED)
    if a.only:
        v = {k: v[k] for k in a.only.split(",")}
    return v


def stage_tune(a):
    variants = pick_variants(a)
    db, queries = setup(a, variants)
    tq = queries[(queries.caller == CALLER) & (queries.split == "odd")].qid.tolist()
    t0 = time.time()
    rows = [r for chunk in bench.pool_map(tune_task, tq, a.threads) for r in chunk]
    tune = pd.DataFrame(rows).merge(queries[["qid", "n_members", "split", "genome_code"]], on="qid")
    tag = "_nested" if a.nested and a.only else ""
    tune.to_parquet(a.out / f"tune_odd{tag}.parquet", index=False)
    log(f"tune: {len(tq):,} odd queries x {len(variants)} variants x 16 configs, {time.time() - t0:.0f}s")
    agg = tune[tune.n_rel > 0].groupby(["variant", "wx", "tau"]).agg(
        map=("ap", "mean"), map_strict=("ap_strict", "mean"), hits=("n_hits", "mean"), n=("ap", "size"),
        sec_scan=("sec_scan", "mean"), sec_evidence=("sec_evidence", "mean"), entries=("n_entries", "mean")).reset_index()
    agg.to_csv(a.out / f"tune_odd_summary{tag}.tsv", sep="\t", index=False)
    best = {}
    for var, sub in agg.groupby("variant"):
        b = sub.sort_values(["map", "wx", "tau"], ascending=[False, True, False]).iloc[0]
        best[var] = dict(wx=int(b.wx), tau=float(b.tau), odd_map=float(b["map"]), odd_map_strict=float(b.map_strict),
                         levels=variants[var][1], family=variants[var][0])
        log(var, json.dumps({k: v for k, v in best[var].items() if k != "levels"}))
    f = a.out / "best_configs.json"
    allb = json.load(open(f)) if f.exists() else {}
    allb.update(best)
    json.dump(allb, open(f, "w"), indent=1)


def stage_final(a):
    variants = pick_variants(a)
    best = json.load(open(a.out / "best_configs.json"))
    db, queries = setup(a, variants)
    eq = queries[(queries.caller == CALLER) & (queries.split == "even")].qid.tolist()
    tasks = [(int(q), v, best[v]["wx"], best[v]["tau"]) for v in variants for q in eq]
    t0 = time.time()
    res = bench.pool_map(final_task, tasks, a.threads, chunksize=1)
    log(f"final: {len(tasks):,} query-variant runs, {time.time() - t0:.0f}s")
    M = pd.DataFrame([r for r, _ in res]).merge(queries, on="qid")
    H = pd.concat([h for _, h in res if len(h)], ignore_index=True)
    tag = "_nested" if a.nested and a.only else ""
    M.to_parquet(a.out / f"final_even_metrics{tag}.parquet", index=False)
    H.to_parquet(a.out / f"final_even_hits_E10{tag}.parquet", index=False)
    summary(a)


def summary(a):
    parts = [pd.read_parquet(f) for f in sorted(a.out.glob("final_even_metrics*.parquet"))]
    M = pd.concat(parts, ignore_index=True)
    M = M[M.n_rel > 0]
    F = pd.read_parquet(FROZEN)
    F = F[(F.split == "even") & (F.n_rel > 0)]
    best = json.load(open(a.out / "best_configs.json"))
    m0_rate = float(F[F.method == "M0"]["n_unlabelled_E0.01"].mean())
    out = dict(m0_unlabelled_rate=m0_rate, rate_bound=1.5 * m0_rate, variants={})
    ref = {m: F[F.method == m].set_index("qid") for m in ("M0", "M2", "S2", "S3")}
    for var, g in M.groupby("method"):
        g = g.set_index("qid")
        d = dict(queries=int(len(g)), config=best.get(var), ap=float(g.ap.mean()), ap_strict=float(g.ap_strict.mean()),
                 recall_E001=float(g["recall_E0.01"].mean()), recall_E1=float(g["recall_E1"].mean()),
                 unlabelled_E001=float(g["n_unlabelled_E0.01"].mean()), wrong_E001=float(g["n_wrong_E0.01"].mean()),
                 unlabelled_x_M0=float(g["n_unlabelled_E0.01"].mean() / m0_rate),
                 cost=dict(evidence_s=float(g.sec_evidence.mean()), scan_s=float(g.sec_scan.mean()),
                           null_s=float(g.sec_null.mean()), total_mean_s=float((g.sec_evidence + g.sec_scan + g.sec_null).mean()),
                           total_median_s=float((g.sec_evidence + g.sec_scan + g.sec_null).median()),
                           entries_mean=float(g.n_entries.mean())),
                 paired={})
        for m, R in ref.items():
            ids = g.index.intersection(R.index)
            gc = R.loc[ids, "genome_code"].to_numpy()
            d["paired"][m] = {c: ls.boot_mean(g.loc[ids, c].to_numpy() - R.loc[ids, c].to_numpy(), gc)
                              for c in ("ap", "ap_strict", "recall_E0.01", "n_unlabelled_E0.01")}
            if m == "M0":
                d["identical_ap_to_M0"] = bool(np.array_equal(g.loc[ids, "ap"].to_numpy(), R.loc[ids, "ap"].to_numpy()))
        S = g[g.n_members >= 3]
        d["ap_min3"] = float(S.ap.mean())
        out["variants"][var] = d
        log(var, f"AP {d['ap']:.4f} strict {d['ap_strict']:.4f} unl/q {d['unlabelled_E001']:.3f} "
                 f"({d['unlabelled_x_M0']:.2f}x M0) dAP vs M0 {d['paired']['M0']['ap'][0]:+.4f} "
                 f"cost {d['cost']['total_mean_s']:.3f}s")
    # predeclared verdicts: single-source variants chosen on odd genomes
    odd = {v: b["odd_map"] for v, b in best.items()}
    verdict = {}
    fams = []
    for suf in ("", "_rc"):        # "" = predeclared; "_rc" = exploratory rate-constrained selection (post hoc)
        for pre in ("H_emb", "H_seq", "H_both", "N_emb"):
            fams.append((pre + suf, [v for v in odd if v.startswith(pre) and v.endswith("_rc") == (suf == "_rc")]))
    for fam, members in fams:
        members = [v for v in members if v in out["variants"]]
        if not members:
            continue
        pick = max(members, key=lambda v: odd[v])
        d = out["variants"][pick]
        rate_ok = d["unlabelled_E001"] <= 1.5 * m0_rate
        verdict[fam] = dict(selected=pick, odd_map={v: odd[v] for v in members}, even_ap=d["ap"], unlabelled=d["unlabelled_E001"],
                            rate_ok=bool(rate_ok), graded_without_search=bool(d["ap"] >= 0.576 and rate_ok),
                            matches_sequence_search=bool(d["ap"] >= 0.620 and rate_ok))
    out["verdict"] = verdict
    json.dump(out, open(a.out / "summary.json", "w"), indent=1, default=float)
    log("verdict", json.dumps(verdict, default=float))


# =========================================================================== exploratory (post hoc, not predeclared)
RATE_WX = (8, 12, 16)
RATE_TAU = (3.0, 4.0, 5.0, 6.0, 7.0)


def rate_task(args):
    """Odd-genome query x variant over the extended grid, with the 200-replicate null (as the finals)."""
    qid, var = args
    q = bench.CTX["queries"].loc[qid]
    db = bench.CTX["db"]
    names = bench.CTX["variants"][var][1]
    (eg, es_, ev), ns = entries(q, names)
    lo, hi = db.own_range(q.genome_code)
    out = []
    for wx, tau in product(RATE_WX, RATE_TAU):
        W = bench.W_of(q, wx)
        t0 = time.time()
        ha, hb, sc, _ = ls.scan(eg, es_, ev, ns, db.cc, db.pos, W, tau, 0)
        tail = ls.null_tail(eg, es_, ev, ns, db.cc, db.pos, db.N, lo, hi, W, tau, bench.R_NULL, bench.SEED + int(qid), bench.TOP_N)
        E, _ = ls.evalues(sc, tail, bench.R_NULL)
        r, _, _ = bench.label_eval(q, ha, hb, sc, E)
        r.update(qid=int(qid), variant=var, wx=wx, tau=tau, seconds=time.time() - t0)
        out.append(r)
    return out


def stage_rate(a):
    """Rate-constrained selection on odd genomes: highest mean AP among configurations whose unlabelled hits per
    query at E <= 0.01 are <= 1.5 x that of M0 (chk_e3c50s75 at its round-1 configuration) on the same queries."""
    variants = {k: v for k, v in pick_variants(a).items()}
    db, queries = setup(a, variants)
    tq = queries[(queries.caller == CALLER) & (queries.split == "odd")].qid.tolist()
    tasks = [(int(q), v) for v in variants for q in tq]
    t0 = time.time()
    rows = [r for chunk in bench.pool_map(rate_task, tasks, a.threads, chunksize=1) for r in chunk]
    R = pd.DataFrame(rows)
    R.to_parquet(a.out / "rate_odd.parquet", index=False)
    log(f"rate grid: {len(tasks):,} query-variants x {len(RATE_WX) * len(RATE_TAU)} configs, {time.time() - t0:.0f}s")
    R = R[R.n_rel > 0]
    agg = R.groupby(["variant", "wx", "tau"]).agg(map=("ap", "mean"), unl=("n_unlabelled_E0.01", "mean"),
                                                  rec=("recall_E0.01", "mean"), n=("ap", "size")).reset_index()
    agg.to_csv(a.out / "rate_odd_summary.tsv", sep="\t", index=False)
    m0 = agg[(agg.variant == "chk_e3c50s75") & (agg.wx == 8) & (agg.tau == 3.0)].unl.iloc[0]
    best = {}
    for var, sub in agg.groupby("variant"):
        ok = sub[sub.unl <= 1.5 * m0]
        if ok.empty:
            continue
        b = ok.sort_values(["map", "wx", "tau"], ascending=[False, True, False]).iloc[0]
        best[var + "_rc"] = dict(wx=int(b.wx), tau=float(b.tau), odd_map=float(b["map"]), odd_unl=float(b.unl),
                                 odd_m0_unl=float(m0), levels=variants[var][1], family=variants[var][0] + "_rc")
        log(var, json.dumps({k: v for k, v in best[var + "_rc"].items() if k != "levels"}))
    f = a.out / "best_configs.json"
    allb = json.load(open(f))
    allb.update(best)
    json.dump(allb, open(f, "w"), indent=1)


def stage_rfinal(a):
    best = json.load(open(a.out / "best_configs.json"))
    base = pick_variants(a)
    variants = {k + "_rc": base[k] for k in base if k + "_rc" in best}
    db, queries = setup(a, variants)
    eq = queries[(queries.caller == CALLER) & (queries.split == "even")].qid.tolist()
    tasks = [(int(q), v, best[v]["wx"], best[v]["tau"]) for v in variants for q in eq]
    t0 = time.time()
    res = bench.pool_map(final_task, tasks, a.threads, chunksize=1)
    log(f"rate-constrained finals: {len(tasks):,} runs, {time.time() - t0:.0f}s")
    M = pd.DataFrame([r for r, _ in res]).merge(queries, on="qid")
    H = pd.concat([h for _, h in res if len(h)], ignore_index=True)
    M.to_parquet(a.out / "final_even_metrics_rc.parquet", index=False)
    H.to_parquet(a.out / "final_even_hits_E10_rc.parquet", index=False)
    summary(a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["tune", "final", "summary", "rate", "rfinal"])
    ap.add_argument("--out", type=Path, default=RD / "hierarchy/search")
    ap.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_ON_NODE", "2")))
    ap.add_argument("--nested", action="store_true")
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    import numba
    numba.set_num_threads(a.threads)
    t0 = time.time()
    {"tune": stage_tune, "final": stage_final, "summary": summary, "rate": stage_rate, "rfinal": stage_rfinal}[a.stage](a)
    log(f"done {a.stage} {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
