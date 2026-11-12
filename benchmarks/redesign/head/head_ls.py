#!/usr/bin/env python
"""Workstream head, step 3: the chosen student's ranks in the locus-search benchmark (round-1 / round-2 code).

  head_ls.py topk     exact top-20,000 student cosine per gold member gene (gene level, own genome excluded),
                      s = log(N_db / tail rank) as locus_search.topk_cosine -> $RD/head/ls/topk_student/
  head_ls.py tune     scan-only AP grid on odd-genome DefenseFinder queries: M1' (student), M2' (token + w * student),
                      S3' (token + MMseqs2 rank + w * student); W extra {0,2,4,8} x tau {0..3}
  head_ls.py final    200-replicate null runs (seeds 17 + qid, as rounds 1-2): DefenseFinder even genomes M1', M2', S3';
                      TXSScan and CRISPRCasTyper even genomes M1', M2', S3' and raw M1, M2 at the DefenseFinder configs
  head_ls.py evidence unlabelled hits at E <= 0.01 against the round-2 'supported' rule (evidence.py)
  head_ls.py summary  tables and paired genome-bootstrap comparisons with M0, M1, M2, S1, S2, S3
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
import bench  # noqa: E402
import locus_search as ls  # noqa: E402
import round2  # noqa: E402
from locus_search import log  # noqa: E402

OUT1 = RD / "locus_search/out"
OUT2 = RD / "locus_search/out2"
H = RD / "head"
LS = H / os.environ.get("HEAD_LS", "ls")
VEC = os.environ.get("HEAD_VEC", "student")          # student | zca10 (amendment 2 control)
TAG = "p" if VEC == "student" else "z"
EMB = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/embed_queue/runs/dpann__esmpp_small__20261111")
SEED = 17
EV = round2.EV
FAMS = {"M1p": [(False, 0.0, 1.0)], "M2p": [(True, 0.0, w) for w in (0.5, 0.75, 1.0)],
        "S3p": [(True, 1.0, w) for w in (0.25, 0.5, 0.75, 1.0)]}
RAW_FROZEN = {"M1": (False, 0.0, 1.0, 8, 2.0), "M2": (True, 0.0, 0.5, 8, 0.0)}


def pick_student():
    if VEC == "zca10":
        return "raw_zca10", H / "data/zca10_f16__uid.npy"
    sel = json.loads((H / "student/selection.json").read_text())
    return sel["selected"], H / "student" / sel["selected"] / "student_f32__uid.npy"


# =========================================================================== topk
def topk(a):
    name, path = pick_student()
    t0 = time.time()
    db = ls.Database(vectors=None)
    Z = np.load(path).astype(np.float32)
    u = np.load(EMB / "uid_of_row.npy").astype(np.int64)
    db.X = np.ascontiguousarray(Z[u])
    del Z
    gids = np.load(OUT1 / "topk/gids.npy")
    log(f"student {name}: gene matrix {db.X.shape}, {time.time() - t0:.0f}s")
    t1 = time.time()
    idx, cos, s = ls.topk_cosine(db, db.X[gids], db.gc[gids], K=20000)
    secs = time.time() - t1
    d = LS / "topk_student"
    d.mkdir(parents=True, exist_ok=True)
    for k, v in (("gids", gids), ("idx", idx), ("cos", cos), ("s", s)):
        np.save(d / f"{k}.npy", v)
    info = dict(student=name, n=int(len(gids)), K=20000, dim=int(db.X.shape[1]), seconds=round(secs, 1),
                vectors_per_second=len(gids) / secs, threads=a.threads,
                raw_reference_vectors_per_second=json.load(open(OUT1 / "topk/topk.json"))["vectors_per_second"])
    json.dump(info, open(d / "topk.json", "w"), indent=1)
    log("student top-K", info)


# =========================================================================== shared setup
def setup(a, student=True):
    db, callers, queries = round2.setup(a)               # raw embedding + sequence ranks (round 1)
    EV.update(raw_idx=EV["e_idx"], raw_s=EV["e_s"], st_idx=EV["e_idx"], st_s=EV["e_s"])
    if student:
        d = LS / "topk_student"
        assert np.array_equal(np.load(d / "gids.npy"), np.load(OUT1 / "topk/gids.npy"))
        EV.update(st_idx=np.load(d / "idx.npy"), st_s=np.load(d / "s.npy"))
    return db, callers, queries


def use(src):
    EV["e_idx"], EV["e_s"] = (EV["st_idx"], EV["st_s"]) if src == "student" else (EV["raw_idx"], EV["raw_s"])


def fam_name(use_tok, ws, we):
    return {"M1": "M1p", "M2": "M2p", "S3": "S3p"}[round2.family(use_tok, ws, we)]


def selfcheck(a):
    """With the raw embedding lists in the student slot, M1' and S3' at the round-1 configurations must reproduce
    round-1 M1 and round-2 S3 exactly (AP, recall at E <= 0.01, unlabelled hits) on 24 even DefenseFinder queries."""
    LS.mkdir(parents=True, exist_ok=True)
    db, callers, queries = setup(a, student=False)
    r1 = pd.read_parquet(OUT1 / "final_metrics_dfr1.parquet").query("method == 'M1'").drop_duplicates("qid").set_index("qid")
    s3 = pd.read_parquet(OUT2 / "final_dfall_metrics.parquet").query("method == 'S3'").set_index("qid")
    Q = queries[(queries.caller == "DefenseFinder") & (queries.split == "even")]
    qids = [int(q) for q in Q.qid if q in r1.index and q in s3.index][:400:17][:24]
    tasks = [(q, "M1p", "student", (False, 0.0, 1.0, 8, 2.0)) for q in qids] + \
            [(q, "S3p", "student", (True, 1.0, 0.5, 8, 0.0)) for q in qids]
    res = bench.pool_map(final_task, tasks, a.threads, chunksize=1)
    out = {}
    for m, ref in (("M1p", r1), ("S3p", s3)):
        x = pd.DataFrame([r for r, _ in res if r["method"] == m]).set_index("qid")
        out[m] = {c: float((x[c] - ref.loc[x.index, c]).abs().max()) for c in ("ap", "recall_E0.01", "n_unlabelled_E0.01")}
    json.dump(dict(queries=qids, max_abs_diff=out), open(LS / "selfcheck.json", "w"), indent=1)
    log("selfcheck", json.dumps(out))
    assert all(v < 1e-9 for d in out.values() for v in d.values()), "student slot does not reproduce round 1/2"


# =========================================================================== tune
def tune_task(qid):
    q = bench.CTX["queries"].loc[qid]
    db = bench.CTX["db"]
    use("student")
    out = []
    for fam, cfgs in FAMS.items():
        for use_tok, ws, we in cfgs:
            (eg, es_, ev), ns = round2.entries(q, use_tok, ws, we)
            for wx, tau in product(bench.GRID_WX, bench.GRID_TAU):
                ha, hb, sc, _ = ls.scan(eg, es_, ev, ns, db.cc, db.pos, bench.W_of(q, wx), tau, 0)
                r, _, _ = bench.label_eval(q, ha, hb, sc, None)
                out.append(dict(qid=int(qid), fam=fam, use_tok=use_tok, ws=ws, we=we, wx=wx, tau=tau, ap=r["ap"],
                                ap_strict=r["ap_strict"], n_rel=r["n_rel"]))
    return out


def tune(a):
    LS.mkdir(parents=True, exist_ok=True)
    db, callers, queries = setup(a)
    Q = queries[(queries.caller == "DefenseFinder") & (queries.split == "odd")]
    t0 = time.time()
    rows = [r for ch in bench.pool_map(tune_task, Q.qid.tolist(), a.threads) for r in ch]
    G = pd.DataFrame(rows)
    G.to_parquet(LS / "tune_odd.parquet", index=False)
    G = G[G.n_rel > 0]
    best = {}
    for fam, g in G.groupby("fam"):
        agg = g.groupby(["use_tok", "ws", "we", "wx", "tau"]).ap.mean().reset_index()
        b = agg.sort_values(["ap", "wx", "tau", "we"], ascending=[False, True, False, True]).iloc[0]
        best[fam] = dict(use_tok=bool(b.use_tok), ws=float(b.ws), we=float(b.we), wx=int(b.wx), tau=float(b.tau),
                         odd_ap=float(b.ap), queries=int(g.qid.nunique()))
    json.dump(best, open(LS / "best_configs.json", "w"), indent=1)
    log(f"tune: {len(Q):,} odd queries, {time.time() - t0:.0f}s", json.dumps(best))


# =========================================================================== final
def final_task(args):
    qid, method, src, (use_tok, ws, we, wx, tau) = args
    use(src)
    q = bench.CTX["queries"].loc[qid]
    r = round2.run_one(q, use_tok, ws, we, wx, tau, bench.R_NULL, bench.TOP_N, 32, SEED + int(qid), 10.0)
    E, _ = ls.evalues(r["sc"], r["tail"], bench.R_NULL)
    m, lab, match = bench.label_eval(q, r["ha"], r["hb"], r["sc"], E)
    m.update(qid=int(qid), method=method, W=r["W"], tau=tau, n_entries=r["n_entries"], sec_scan=r["sec_scan"],
             sec_null=r["sec_null"])
    keep = E <= 10
    hits = pd.DataFrame(dict(qid=int(qid), method=method, start_gid=r["ha"][keep], end_gid=r["hb"][keep],
                             score=r["sc"][keep], evalue=E[keep], label=lab[keep], match_row=match[keep],
                             n_slots_filled=r["nf"][keep]))
    return m, hits


def final(a):
    db, callers, queries = setup(a)
    best = json.load(open(LS / "best_configs.json"))
    cfg = {f: (b["use_tok"], b["ws"], b["we"], b["wx"], b["tau"]) for f, b in best.items()}
    tasks = []
    ev = queries[queries.split == "even"]
    for qid, c in zip(ev.qid, ev.caller):
        for f in ("M1p", "M2p", "S3p"):
            tasks.append((int(qid), f[:-1] + TAG, "student", cfg[f]))
        if c != "DefenseFinder" and VEC == "student":
            for f, c_ in RAW_FROZEN.items():
                tasks.append((int(qid), f + "raw", "raw", c_))
    tasks.sort(key=lambda t: -queries.n_members[t[0]])
    t0 = time.time()
    res = bench.pool_map(final_task, tasks, a.threads, chunksize=1)
    log(f"final: {len(tasks):,} runs, {time.time() - t0:.0f}s")
    M = pd.DataFrame([m for m, _ in res]).merge(queries, on="qid")
    M.to_parquet(LS / "final_metrics.parquet", index=False)
    pd.concat([h for _, h in res], ignore_index=True).to_parquet(LS / "final_hits_E10.parquet", index=False)


# =========================================================================== evidence (round-2 rule)
def evidence_stage(a):
    import evidence as evm
    db = evm.attach_db()
    callers = bench.load_gold(db)
    queries = pd.read_parquet(OUT1 / "queries.parquet")
    ev = evm.Evidence(db)
    rng = np.random.default_rng(SEED)
    Hs = pd.read_parquet(LS / "final_hits_E10.parquet", filters=[("evalue", "<=", 0.01), ("label", "==", 0)])
    Hs = Hs[Hs.qid.isin(queries[queries.caller == "DefenseFinder"].qid)]
    r1 = pd.read_parquet(OUT1 / "final_hits_E10.parquet", filters=[("evalue", "<=", 0.01), ("label", "==", 0)])
    r1 = r1[(r1.method == "M1") & (r1["round"] == 1)]
    r1 = r1[r1.qid.isin(queries[(queries.caller == "DefenseFinder") & (queries.split == "even")].qid)]
    H = pd.concat([Hs[["qid", "method", "start_gid", "end_gid", "evalue"]],
                   r1[["qid", "method", "start_gid", "end_gid", "evalue"]].assign(method="M1")], ignore_index=True)
    F, _ = evm.classify(ev, H.reset_index(drop=True), queries, callers, rng, controls=False)
    H = pd.concat([H.reset_index(drop=True), F.reset_index(drop=True)], axis=1)
    H.to_parquet(LS / "unlabelled_DF_E0.01.parquet", index=False)
    log("evidence", H.groupby("method").supported.agg(["size", "mean"]).to_dict())


# =========================================================================== summary
def summary(a):
    queries = pd.read_parquet(OUT1 / "queries.parquet")
    M = pd.read_parquet(LS / "final_metrics.parquet")
    M = M[M.n_rel > 0]
    U = pd.read_parquet(LS / "unlabelled_DF_E0.01.parquet")
    unsup = U[~U.supported].groupby(["method", "qid"]).size().rename("n_unsupported_E0.01").reset_index()
    ref = pd.concat([pd.read_parquet(OUT1 / "final_metrics_dfr1.parquet").query("method in ['M0','M1','M2']"),
                     pd.read_parquet(OUT1 / "final_metrics_seqctl.parquet"),
                     pd.read_parquet(OUT2 / "final_dfall_metrics.parquet").query("method == 'S3'")], ignore_index=True)
    ref = ref[(ref.caller == "DefenseFinder") & (ref.split == "even") & (ref.n_rel > 0)].drop_duplicates(["qid", "method"])
    xref = pd.read_parquet(OUT2 / "final_xeven_metrics.parquet")
    xref = xref[xref.n_rel > 0]
    cols = ["ap", "ap_strict", "recall_E0.01", "recall_E1", "n_unlabelled_E0.01", "n_wrong_E0.01", "n_unsupported_E0.01"]
    A = pd.concat([M, ref, xref], ignore_index=True)
    A = A.merge(unsup, on=["method", "qid"], how="left")
    dfq = A.caller.eq("DefenseFinder")
    A.loc[dfq, "n_unsupported_E0.01"] = A.loc[dfq, "n_unsupported_E0.01"].fillna(0)
    A["sec_query"] = A.sec_scan + A.sec_null
    out = dict(best_configs=json.load(open(LS / "best_configs.json")),
               topk=json.load(open(LS / "topk_student/topk.json")), tables={}, paired={})
    for c, g in A[A.split == "even"].groupby("caller"):
        out["tables"][c] = {}
        for m, gm in g.groupby("method"):
            d = {"queries": int(len(gm))}
            for col in cols + ["sec_query"]:
                if col in gm and gm[col].notna().any():
                    d[col] = ls.boot_mean(gm[col].to_numpy(), gm.genome_code.to_numpy())
            out["tables"][c][m] = d
        p_, q_, r_ = "M1" + TAG, "M2" + TAG, "S3" + TAG
        pairs = [(p_, "M1"), (p_, "S1"), (p_, "M0"), (q_, "M2"), (q_, "M0"), (r_, "S3"), (r_, "S2"),
                 ("S3", "S2"), (p_, "M1raw"), (q_, "M2raw")]
        out["paired"][c] = {}
        for x, y in pairs:
            gx, gy = g[g.method == x].set_index("qid"), g[g.method == y].set_index("qid")
            ids = gx.index.intersection(gy.index)
            if not len(ids):
                continue
            gc = gy.loc[ids, "genome_code"].to_numpy()
            out["paired"][c][f"{x}-{y}"] = {col: ls.boot_mean(gx.loc[ids, col].to_numpy() - gy.loc[ids, col].to_numpy(), gc)
                                            for col in ("ap", "ap_strict", "recall_E0.01", "n_unlabelled_E0.01")}
            out["paired"][c][f"{x}-{y}"]["queries"] = int(len(ids))
    # >= 3-member stratum, DefenseFinder
    g3 = A[(A.split == "even") & dfq & (A.n_members >= 3)]
    out["tables"]["DefenseFinder_min3"] = {m: dict(queries=int(len(gm)), ap=ls.boot_mean(gm.ap, gm.genome_code))
                                           for m, gm in g3.groupby("method")}
    U2 = U.merge(queries[["qid", "split"]], on="qid")
    U2 = U2[U2.split == "even"]
    nq = int(A[(A.split == "even") & dfq & (A.method == "M1" + TAG)].qid.nunique())
    out["unlabelled_DF_even"] = {m: dict(unlabelled=int(len(g)), unsupported=int((~g.supported).sum()),
                                         unsupported_per_query=float((~g.supported).sum() / nq),
                                         supported_share=float(g.supported.mean()), df_own_share=float(g.df_own.mean()))
                                 for m, g in U2.groupby("method")}
    out["unlabelled_DF_even_queries"] = nq
    m1p = out["tables"]["DefenseFinder"]["M1" + TAG]
    uq_ = out["unlabelled_DF_even"].get("M1" + TAG, {}).get("unsupported_per_query", 0.0)
    out["H1"] = dict(M1p_ap=m1p["ap"], unsupported_per_query=uq_,
                     met=bool(m1p["ap"][0] >= 0.600 and uq_ <= 0.05), partial=bool(m1p["ap"][0] >= 0.52 and uq_ <= 0.05))
    s3 = out["paired"]["DefenseFinder"].get(f"S3{TAG}-S3", {}).get("ap")
    out["S3prime_vs_S3"] = dict(diff=s3, student_adds_more_than_raw=bool(s3 is not None and s3[1] > 0))
    json.dump(out, open(LS / "summary.json", "w"), indent=1, default=float)
    log("H1", json.dumps(out["H1"], default=float), "S3'", json.dumps(out["S3prime_vs_S3"], default=float))


def combine(a):
    """Student against the zca10-rank control (amendment 2), paired per query, even genomes."""
    P = pd.read_parquet(H / "ls/final_metrics.parquet")
    Z = pd.read_parquet(H / "ls_zca10/final_metrics.parquet")
    A = pd.concat([P, Z], ignore_index=True)
    A = A[(A.split == "even") & (A.n_rel > 0)]
    Up = pd.read_parquet(H / "ls/unlabelled_DF_E0.01.parquet")
    Uz = pd.read_parquet(H / "ls_zca10/unlabelled_DF_E0.01.parquet")
    U = pd.concat([Up[Up.method != "M1"], Uz], ignore_index=True)
    uns = U[~U.supported].groupby(["method", "qid"]).size().rename("n_unsupported_E0.01").reset_index()
    A = A.merge(uns, on=["method", "qid"], how="left")
    A.loc[A.caller == "DefenseFinder", "n_unsupported_E0.01"] = A.loc[A.caller == "DefenseFinder", "n_unsupported_E0.01"].fillna(0)
    out = {}
    for c, g in A.groupby("caller"):
        out[c] = {"tables": {m: {col: ls.boot_mean(gm[col].to_numpy(), gm.genome_code.to_numpy())
                                 for col in ("ap", "ap_strict", "recall_E0.01", "n_unlabelled_E0.01", "n_unsupported_E0.01")
                                 if gm[col].notna().any()} | {"queries": int(len(gm))}
                             for m, gm in g.groupby("method")}}
        for x, y in (("M1p", "M1z"), ("M2p", "M2z"), ("S3p", "S3z")):
            gx, gy = g[g.method == x].set_index("qid"), g[g.method == y].set_index("qid")
            ids = gx.index.intersection(gy.index)
            out[c][f"{x}-{y}"] = {col: ls.boot_mean(gx.loc[ids, col].to_numpy() - gy.loc[ids, col].to_numpy(),
                                                    gy.loc[ids, "genome_code"].to_numpy())
                                  for col in ("ap", "ap_strict", "recall_E0.01", "n_unlabelled_E0.01")} | {"queries": int(len(ids))}
    json.dump(out, open(H / "ls/combine_vs_zca10.json", "w"), indent=1, default=float)
    log("combine", json.dumps({c: {k: v["ap"] for k, v in d.items() if k != "tables"} for c, d in out.items()}, default=float))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stages", help="comma-separated: topk,selfcheck,tune,final,evidence,summary")
    ap.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_ON_NODE", "2")))
    a = ap.parse_args()
    import numba
    numba.set_num_threads(a.threads)
    for st in a.stages.split(","):
        t0 = time.time()
        {"topk": topk, "selfcheck": selfcheck, "tune": tune, "final": final, "evidence": evidence_stage,
         "summary": summary, "combine": combine}[st](a)
        log(f"stage {st} done {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
