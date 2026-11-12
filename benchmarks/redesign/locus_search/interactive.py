#!/usr/bin/env python
"""Task 5 of round 2: an interactive locus-search path (approximate per-gene ranks, fast null).

  interactive.py prep  --threads T     FAISS HNSW index (M 32, efSearch 128, top 2,000) on the ESM++ small
                                       vectors; HNSW ranks for every member gene; MMseqs2 createindex of the
                                       target database; indexed searches of the DefenseFinder even-genome query
                                       genes (exact parameters and fast parameters); AP of S3 with each
                                       substitution against the exact version (scan only; AP needs no null)
  interactive.py search --gids a,b,c [--method S3] [--threads T] [--top 30] [--emb exact|hnsw40|hnsw200] [--fast]
                                       the one command: search one locus (gids of one genome) and print ranked
                                       hits with E-values (40-replicate null, Gumbel tail + per-method calibration)
  interactive.py bench --threads T     wall time of `search` for 20 DefenseFinder queries in one process
                                       (first query includes every load) and one cold one-shot command

S3 = per slot max(token, sequence rank, 0.5 * embedding rank), W = query span + 8, tau 0 (round-1 config).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fastnull  # noqa: E402
import locus_search as ls  # noqa: E402
from locus_search import log  # noqa: E402

OUT1 = Path(os.environ.get("LS_OUT1", ls.RD / "locus_search/out"))
OUT2 = Path(os.environ.get("LS_OUT2", ls.RD / "locus_search/out2"))
IDIR = OUT2 / "interactive"
MM = "/groups/banfield/users/jwestrob/bin/mmseqs_18/mmseqs/bin/mmseqs"
HNSW_FILE = IDIR / "hnsw_m32_esmpp_small.faiss"
HNSW_FILE_200 = IDIR / "hnsw_m32_efc200_esmpp_small.faiss"
TDB = IDIR / "tdb"
KEY_OF_GID = IDIR / "tdb_key_of_gid.npy"
KEY = {}
K_FAST = 2000
METHODS = {"S3": (True, 1.0, 0.5, 8, 0.0), "S2": (True, 1.0, 0.0, 8, 3.0), "M2": (True, 0.0, 0.5, 8, 0.0),
           "M0": (True, 0.0, 0.0, 8, 3.0)}


def sh(cmd):
    log("$", cmd)
    subprocess.run(cmd, shell=True, check=True)


# =========================================================================== rank helpers
def hnsw_ranks(index, X, gids, own_lo, own_hi, n_db, k=K_FAST, margin=200):
    D, I = index.search(np.ascontiguousarray(X, dtype=np.float32), k + margin)
    idx = np.full((len(gids), k), -1, np.int32)
    s = np.zeros((len(gids), k), np.float32)
    for r in range(len(gids)):
        row = I[r]
        row = row[(row >= 0) & ~((row >= own_lo[r]) & (row < own_hi[r]))][:k]
        idx[r, :len(row)] = row
        s[r, :len(row)] = np.log(n_db[r] / np.arange(1, len(row) + 1))
    return idx, s


def seq_ranks(hits_tsv, gids, db, K):
    """Per query gene: hits outside its genome ranked by bitscore (ties share the tail rank)."""
    idx = np.full((len(gids), K), -1, np.int32)
    s = np.zeros((len(gids), K), np.float32)
    if Path(hits_tsv).stat().st_size == 0:
        return idx, s
    t = pd.read_csv(hits_tsv, sep="\t", header=None, names=["q", "t", "bits", "evalue"])
    t = t[db.gc[t.q.to_numpy()] != db.gc[t.t.to_numpy()]]
    t = t.sort_values(["q", "bits", "t"], ascending=[True, False, True])
    row_of = {int(g): k for k, g in enumerate(gids)}
    for q, g in t.groupby("q", sort=False):
        k = row_of.get(int(q))
        if k is None:
            continue
        bb = g.bits.to_numpy()[:K]
        rank = np.searchsorted(-bb, -bb, side="right")
        idx[k, :len(bb)] = g.t.to_numpy()[:len(bb)]
        s[k, :len(bb)] = np.log(db.n_db(int(db.gc[int(q)])) / rank)
    return idx, s


def make_qdb(gids, path, tmp):
    """Query database = sub-database of the target database (MMseqs2 key looked up from the gid)."""
    if "k" not in KEY:
        KEY["k"] = np.load(KEY_OF_GID)
    lst = Path(str(path) + ".keys")
    np.savetxt(lst, KEY["k"][np.asarray(gids, np.int64)], fmt="%d")
    sh(f"{MM} createsubdb {lst} {TDB} {path} -v 1")
    sh(f"{MM} createsubdb {lst} {TDB}_h {path}_h -v 1")


# =========================================================================== prep (64 CPUs)
def prep(a):
    import faiss
    import numba
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import bench
    import round2
    IDIR.mkdir(parents=True, exist_ok=True)
    faiss.omp_set_num_threads(a.threads)
    numba.set_num_threads(a.threads)
    res = json.load(open(IDIR / "prep.json")) if (IDIR / "prep.json").exists() else {}
    t0 = time.time()
    db = ls.Database(vectors="ram")
    log(f"vectors {db.X.shape} {time.time() - t0:.0f}s")
    gids = np.load(OUT1 / "topk/gids.npy")
    own = db.gc[gids]
    lo, hi = db.g_lo[own], db.g_hi[own]
    n_db = np.array([db.n_db(int(g)) for g in own], np.float64)
    # ---- HNSW (efConstruction 40 as built first; 200 as a graph-quality check) ---------------------
    ex = np.load(OUT1 / "topk/idx.npy", mmap_mode="r")
    ranks = {}
    for tag, efc, path in (("efc40", 40, HNSW_FILE), ("efc200", 200, HNSW_FILE_200)):
        if (IDIR / f"hnsw_{tag}_idx.npy").exists():
            ranks[tag] = (np.load(IDIR / f"hnsw_{tag}_idx.npy"), np.load(IDIR / f"hnsw_{tag}_s.npy"))
            continue
        if tag == "efc40" and (IDIR / "hnsw_idx.npy").exists():
            ranks[tag] = (np.load(IDIR / "hnsw_idx.npy"), np.load(IDIR / "hnsw_s.npy"))
            np.save(IDIR / f"hnsw_{tag}_idx.npy", ranks[tag][0])
            np.save(IDIR / f"hnsw_{tag}_s.npy", ranks[tag][1])
            continue
        if path.exists():
            t1 = time.time()
            index = faiss.read_index(str(path))
            res[f"hnsw_{tag}_load_seconds"] = round(time.time() - t1, 1)
        else:
            t1 = time.time()
            index = faiss.IndexHNSWFlat(db.X.shape[1], 32, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efConstruction = efc
            index.add(db.X)
            res[f"hnsw_{tag}_build_seconds_64cpu"] = round(time.time() - t1, 1)
            faiss.write_index(index, str(path))
            res[f"hnsw_{tag}_file_bytes"] = path.stat().st_size
            log("HNSW built", tag, res[f"hnsw_{tag}_build_seconds_64cpu"], "s")
        index.hnsw.efSearch = 128
        t1 = time.time()
        ranks[tag] = hnsw_ranks(index, db.X[gids], gids, lo, hi, n_db)
        secs = time.time() - t1
        res[f"hnsw_{tag}_search"] = dict(genes=int(len(gids)), k=K_FAST, k_requested=K_FAST + 200, efSearch=128,
                                         seconds_64cpu=round(secs, 1),
                                         genes_per_second=len(gids) / secs)
        np.save(IDIR / f"hnsw_{tag}_idx.npy", ranks[tag][0])
        np.save(IDIR / f"hnsw_{tag}_s.npy", ranks[tag][1])
        del index
    for tag, (hidx, _) in ranks.items():
        rec = []
        for r in range(len(gids)):
            e = ex[r, :K_FAST]
            e = e[e >= 0]
            if len(e):
                rec.append(len(np.intersect1d(e, hidx[r][hidx[r] >= 0])) / len(e))
        rec = np.asarray(rec)
        rec50 = []
        for r in range(len(gids)):
            e = ex[r, :50]
            e = e[e >= 0]
            if len(e):
                rec50.append(len(np.intersect1d(e, hidx[r][:50])) / len(e))
        res[f"hnsw_{tag}_recall_vs_exact"] = dict(at2000_mean=float(rec.mean()), at2000_q10_50_90=np.quantile(rec, [.1, .5, .9]).tolist(),
                                                 at50_mean=float(np.mean(rec50)))
        log("HNSW", tag, res[f"hnsw_{tag}_recall_vs_exact"])
    json.dump(res, open(IDIR / "prep.json", "w"), indent=1, default=float)
    # ---- exact brute force for a few query genes (what an interactive query needs) -----------------
    rng = np.random.default_rng(17)
    t_ex = []
    for _ in range(5):
        pick = rng.choice(len(gids), 4, replace=False)
        t1 = time.time()
        ls.topk_cosine(db, db.X[gids[pick]], own[pick], K=K_FAST, log_every=0)
        t_ex.append(time.time() - t1)
    res["exact_topk_4genes_seconds_64cpu"] = t_ex
    # ---- MMseqs2 index and searches ---------------------------------------------------------------
    src = OUT1 / "seqsearch"
    for f in src.glob("tdb*"):
        dst = IDIR / f.name
        if not dst.exists():
            dst.symlink_to(f)
    lk = pd.read_csv(IDIR / "tdb.lookup", sep="\t", header=None, names=["key", "acc", "file"])
    res["tdb_key_equals_gid"] = bool((lk.key.to_numpy() == lk.acc.to_numpy().astype(np.int64)).all())
    kog = np.full(db.N, -1, np.int64)
    kog[lk.acc.to_numpy().astype(np.int64)] = lk.key.to_numpy()
    assert (kog >= 0).all()
    np.save(KEY_OF_GID, kog)
    KEY.clear()
    if not (IDIR / "tdb.idx").exists():
        t1 = time.time()
        sh(f"{MM} createindex {TDB} {IDIR / 'tmp_idx'} --threads {a.threads}")
        res["mmseqs_createindex_seconds_64cpu"] = round(time.time() - t1, 1)
        res["mmseqs_index_bytes"] = sum(p.stat().st_size for p in IDIR.glob("tdb.idx*"))
    json.dump(res, open(IDIR / "prep.json", "w"), indent=1, default=float)
    db_, callers, queries = bench.setup(a, vectors=None)
    Q = queries[(queries.caller == "DefenseFinder") & (queries.split == "even")]
    qg = np.unique(np.concatenate([callers["DefenseFinder"]["mem"][r] for r in Q.row]))
    make_qdb(qg, IDIR / "qdb_dfeven", IDIR)
    for tag, opts in (("exactparams", "-s 7.5 -e 10 --max-seqs 20000"), ("fast", f"-s 5.7 -e 10 --max-seqs {K_FAST}")):
        out = IDIR / f"res_{tag}"
        if not (IDIR / f"hits_{tag}.tsv").exists():
            t1 = time.time()
            sh(f"{MM} search {IDIR / 'qdb_dfeven'} {TDB} {out} {IDIR / ('tmp_' + tag)} {opts} --threads {a.threads} "
               f"--db-load-mode 2")
            sh(f"{MM} convertalis {IDIR / 'qdb_dfeven'} {TDB} {out} {IDIR / f'hits_{tag}.tsv'} "
               f"--format-output query,target,bits,evalue --threads {a.threads}")
            res[f"mmseqs_{tag}_seconds_64cpu"] = round(time.time() - t1, 1)
            res[f"mmseqs_{tag}_genes"] = int(len(qg))
        json.dump(res, open(IDIR / "prep.json", "w"), indent=1, default=float)
    # ---- AP of the substitutions (DefenseFinder even queries, scan only) ---------------------------
    round2.load_ranks()
    E0 = {k: round2.EV[k] for k in ("e_idx", "e_s", "s_idx", "s_s")}
    sq_ex, ss_ex = seq_ranks(IDIR / "hits_exactparams.tsv", gids, db, 20000)
    sq_f, ss_f = seq_ranks(IDIR / "hits_fast.tsv", gids, db, K_FAST)
    rows_q = np.isin(gids, qg)
    same = []
    for r in np.flatnonzero(rows_q):
        a1, a2 = E0["s_idx"][r], sq_ex[r]
        same.append(np.array_equal(a1[a1 >= 0], a2[a2 >= 0]))
    res["indexed_exactparams_identical_rank_lists"] = float(np.mean(same))
    trunc_e = E0["e_idx"].copy()
    trunc_e[:, K_FAST:] = -1
    trunc_s = E0["s_idx"].copy()
    trunc_s[:, K_FAST:] = -1
    variants = {
        "exact": (E0["e_idx"], E0["e_s"], E0["s_idx"], E0["s_s"]),
        "emb_exact_top2000": (trunc_e, E0["e_s"], E0["s_idx"], E0["s_s"]),
        "emb_hnsw_efc40_top2000": (*ranks["efc40"], E0["s_idx"], E0["s_s"]),
        "emb_hnsw_efc200_top2000": (*ranks["efc200"], E0["s_idx"], E0["s_s"]),
        "seq_exact_top2000": (E0["e_idx"], E0["e_s"], trunc_s, E0["s_s"]),
        "seq_indexed_fast": (E0["e_idx"], E0["e_s"], sq_f, ss_f),
        "both_hnsw_efc40_seqfast": (*ranks["efc40"], sq_f, ss_f),
        "both_hnsw_efc200_seqfast": (*ranks["efc200"], sq_f, ss_f),
        "both_exact2000_seqfast": (trunc_e, E0["e_s"], sq_f, ss_f),
    }
    bench.CTX["rel"] = bench.load_relevance(OUT1, queries, callers)
    bench.warm_up(db_)
    qids = Q.qid.tolist()
    rows = []
    for name, (ei, es, si, ss) in variants.items():
        round2.EV.update(e_idx=ei, e_s=es, s_idx=si, s_s=ss)
        for meth in ("S3", "M2", "S2"):
            use_tok, ws, we, wx, tau = METHODS[meth]
            if (meth == "M2" and name.startswith("seq")) or (meth == "S2" and name.startswith("emb")):
                continue
            t1 = time.time()
            out = [r for r in bench.pool_map(_ap_task, [(q, use_tok, ws, we, wx, tau) for q in qids], a.threads, chunksize=4)]
            for qid, ap, aps, nrel in out:
                rows.append(dict(variant=name, method=meth, qid=qid, ap=ap, ap_strict=aps, n_rel=nrel))
            log(name, meth, f"{time.time() - t1:.0f}s")
    A = pd.DataFrame(rows).merge(queries[["qid", "genome_code"]], on="qid")
    A = A[A.n_rel > 0]
    A.to_parquet(IDIR / "ap_variants.parquet", index=False)
    cmp = {}
    for meth, g in A.groupby("method"):
        base = g[g.variant == "exact"].set_index("qid")
        for name, h in g.groupby("variant"):
            h = h.set_index("qid")
            ids = h.index.intersection(base.index)
            cmp[f"{meth}|{name}"] = dict(map=float(h.loc[ids, "ap"].mean()),
                                         delta=ls.boot_mean(h.loc[ids, "ap"] - base.loc[ids, "ap"], base.loc[ids, "genome_code"]))
    res["ap_variants_even_defensefinder"] = cmp
    json.dump(res, open(IDIR / "prep.json", "w"), indent=1, default=float)
    log("AP variants", json.dumps(cmp, default=float))


def _ap_task(args):
    import bench
    import round2
    qid, use_tok, ws, we, wx, tau = args
    q = bench.CTX["queries"].loc[qid]
    db = bench.CTX["db"]
    (eg, es_, ev), ns = round2.entries(q, use_tok, ws, we)
    ha, hb, sc, _ = ls.scan(eg, es_, ev, ns, db.cc, db.pos, bench.W_of(q, wx), tau, 0)
    r, _, _ = bench.label_eval(q, ha, hb, sc, None)
    return int(qid), r["ap"], r["ap_strict"], r["n_rel"]


# =========================================================================== the search command
class Engine:
    """Everything a search needs, loaded once (a persistent session amortizes this)."""

    def __init__(self, threads, emb="exact", fast=False):
        """emb = 'exact' (brute-force top K over the vectors held in RAM) or 'hnsw40' / 'hnsw200' (HNSW, top 2,000).
        fast = False: K = 20,000 and MMseqs2 -s 7.5 --max-seqs 20000 (the benchmark's rank lists, identical results);
        fast = True: K = 2,000 and MMseqs2 -s 5.7 --max-seqs 2000."""
        import faiss
        t0 = time.time()
        self.threads = threads
        self.emb = emb
        self.K = K_FAST if (fast or emb != "exact") else 20000
        self.seq_opts = f"-s 5.7 -e 10 --max-seqs {K_FAST}" if fast else "-s 7.5 -e 10 --max-seqs 20000"
        self.Kseq = K_FAST if fast else 20000
        faiss.omp_set_num_threads(threads)
        self.db = ls.Database(vectors="ram" if emb == "exact" else "mmap")
        t1 = time.time()
        self.index = None
        if emb.startswith("hnsw"):
            self.index = faiss.read_index(str(HNSW_FILE if emb == "hnsw40" else HNSW_FILE_200))
            self.index.hnsw.efSearch = 128
        KEY["k"] = np.load(KEY_OF_GID)
        cf = IDIR / "fastnull_curves_gum40.npz"
        if not cf.exists():
            cv = fastnull.learn_curves(OUT2, "gum", 40, 1.0)
            np.savez(cf, **cv)
        z = np.load(cf)
        self.curves = {k: z[k] for k in z.files}
        self.t_load = dict(database_and_vectors=t1 - t0, hnsw_index_and_curves=time.time() - t1)

    def search(self, gids, method="S3", R=40, max_E=10.0, workdir=None):
        db = self.db
        use_tok, ws, we, wx, tau = METHODS[method]
        q = np.unique(np.asarray(gids, np.int64))
        own = int(db.gc[q[0]])
        assert np.all(db.gc[q] == own), "query genes must come from one genome"
        N_db = db.n_db(own)
        lo, hi = db.own_range(own)
        tm = {}
        t0 = time.time()
        if we > 0 and self.index is not None:
            ei, es = hnsw_ranks(self.index, db.X[q], q, np.full(len(q), lo), np.full(len(q), hi), np.full(len(q), float(N_db)))
        elif we > 0:
            ei, _, es = ls.topk_cosine(db, db.X[q], np.full(len(q), own), K=self.K, log_every=0)
        tm["embedding_ranks"] = time.time() - t0
        t0 = time.time()
        if ws > 0:
            wd = Path(workdir or (IDIR / "queries" / f"q_{os.getpid()}_{int(time.time() * 1000)}"))
            wd.mkdir(parents=True, exist_ok=True)
            make_qdb(q, wd / "qdb", wd)
            sh(f"{MM} search {wd / 'qdb'} {TDB} {wd / 'res'} {wd / 'tmp'} {self.seq_opts} "
               f"--threads {self.threads} --db-load-mode 2 -v 1")
            sh(f"{MM} convertalis {wd / 'qdb'} {TDB} {wd / 'res'} {wd / 'hits.tsv'} --format-output query,target,bits,evalue "
               f"--threads {self.threads} -v 1")
            si, ss = seq_ranks(wd / "hits.tsv", q, db, self.Kseq)
        tm["sequence_ranks"] = time.time() - t0
        t0 = time.time()
        lists = []
        for k, g in enumerate(q):
            G, S = [], []
            if use_tok:
                gt = db.token_members(int(db.tok[g]), own)
                G.append(gt)
                S.append(np.full(len(gt), math.log(N_db / len(gt)) if len(gt) else 0.0))
            if ws > 0:
                ok = si[k] >= 0
                G.append(si[k][ok].astype(np.int64))
                S.append(ws * ss[k][ok].astype(np.float64))
            if we > 0:
                ok = ei[k] >= 0
                G.append(ei[k][ok].astype(np.int64))
                S.append(we * es[k][ok].astype(np.float64))
            gg, sv = np.concatenate(G), np.concatenate(S)
            o = np.lexsort((-sv, gg))
            gg, sv = gg[o], sv[o]
            first = np.r_[True, gg[1:] != gg[:-1]] if len(gg) else np.zeros(0, bool)
            lists.append((gg[first], sv[first]))
        eg, es_, ev = ls.assemble(lists)
        W = int(min(60, db.pos[q].max() - db.pos[q].min() + 1 + wx))
        ha, hb, sc, nf = ls.scan(eg, es_, ev, len(q), db.cc, db.pos, W, tau, 0)
        tm["scan"] = time.time() - t0
        t0 = time.time()
        tail = ls.null_tail(eg, es_, ev, len(q), db.cc, db.pos, db.N, lo, hi, W, tau, R, 17, 16)
        E = fastnull.apply_curve(fastnull.evalue_gumbel(sc, tail, R), self.curves[method])
        tm["null"] = time.time() - t0
        keep = E <= max_E
        df = db.window_rows(ha[keep], hb[keep])
        df["score"] = sc[keep]
        df["evalue"] = E[keep]
        df["n_slots_filled"] = nf[keep]
        return df.reset_index(drop=True), tm


def cmd_search(a):
    import numba
    numba.set_num_threads(a.threads)
    t0 = time.time()
    eng = Engine(a.threads, a.emb, a.fast)
    gids = [int(x) for x in a.gids.split(",")]
    df, tm = eng.search(gids, a.method)
    wall = time.time() - t0
    pd.set_option("display.width", 200)
    print(df.head(a.top).to_string())
    print(f"\n{len(df):,} hits with E <= 10 ({(df.evalue <= 0.01).sum():,} at E <= 0.01); method {a.method}; "
          f"load {({k: round(v, 1) for k, v in eng.t_load.items()})}; per-query {({k: round(v, 2) for k, v in tm.items()})}; "
          f"wall {wall:.1f}s on {a.threads} CPUs")


def hnsw_ef(a):
    """efSearch sweep on the efConstruction-40 index (k = 2,200): deep-list recall, cost, and S3 / M2 AP."""
    import faiss
    import numba
    import bench
    import round2
    faiss.omp_set_num_threads(a.threads)
    numba.set_num_threads(a.threads)
    res = {}
    db = ls.Database(vectors="ram")
    gids = np.load(OUT1 / "topk/gids.npy")
    own = db.gc[gids]
    lo, hi = db.g_lo[own], db.g_hi[own]
    n_db = np.array([db.n_db(int(g)) for g in own], np.float64)
    ex = np.load(OUT1 / "topk/idx.npy", mmap_mode="r")
    t1 = time.time()
    index = faiss.read_index(str(HNSW_FILE))
    res["load_seconds"] = round(time.time() - t1, 1)
    db_, callers, queries = bench.setup(a, vectors=None)
    Q = queries[(queries.caller == "DefenseFinder") & (queries.split == "even")]
    round2.load_ranks()
    E0 = {k: round2.EV[k] for k in ("e_idx", "e_s", "s_idx", "s_s")}
    bench.CTX["rel"] = bench.load_relevance(OUT1, queries, callers)
    bench.warm_up(db_)
    rows = []
    for ef in (128, 512, 2200, 4400):
        index.hnsw.efSearch = ef
        t1 = time.time()
        hidx, hs = hnsw_ranks(index, db.X[gids], gids, lo, hi, n_db)
        secs = time.time() - t1
        rec = np.array([len(np.intersect1d(ex[r, :K_FAST][ex[r, :K_FAST] >= 0], hidx[r][hidx[r] >= 0])) / max((ex[r, :K_FAST] >= 0).sum(), 1)
                        for r in range(len(gids))])
        res[f"ef{ef}"] = dict(seconds=round(secs, 1), genes_per_second=len(gids) / secs, recall2000_mean=float(rec.mean()),
                              recall2000_q10_50_90=np.quantile(rec, [.1, .5, .9]).tolist())
        round2.EV.update(e_idx=hidx, e_s=hs, s_idx=E0["s_idx"], s_s=E0["s_s"])
        for meth in ("S3", "M2"):
            use_tok, ws, we, wx, tau = METHODS[meth]
            for qid, ap, aps, nrel in bench.pool_map(_ap_task, [(q, use_tok, ws, we, wx, tau) for q in Q.qid], a.threads, chunksize=4):
                rows.append(dict(ef=ef, method=meth, qid=qid, ap=ap, n_rel=nrel))
        log("efSearch", ef, res[f"ef{ef}"])
    A = pd.DataFrame(rows).merge(queries[["qid", "genome_code"]], on="qid")
    A = A[A.n_rel > 0]
    base = pd.read_parquet(IDIR / "ap_variants.parquet")
    base = base[base.variant == "exact"]
    for (ef, meth), g in A.groupby(["ef", "method"]):
        b = base[base.method == meth].set_index("qid")
        g = g.set_index("qid")
        ids = g.index.intersection(b.index)
        res[f"ef{ef}"][f"{meth}_map"] = float(g.loc[ids, "ap"].mean())
        res[f"ef{ef}"][f"{meth}_delta_vs_exact"] = ls.boot_mean(g.loc[ids, "ap"] - b.loc[ids, "ap"], b.loc[ids, "genome_code"])
    json.dump(res, open(IDIR / "hnsw_ef.json", "w"), indent=1, default=float)
    log("hnsw ef", json.dumps(res, default=float))


def verify(a):
    """The accurate command against the benchmark: every S3 benchmark hit (E200 <= 10) of 20 even-genome DefenseFinder
    queries must appear among the command's windows with the same start, end and score."""
    import numba
    numba.set_num_threads(a.threads)
    import bench as bn
    q = pd.read_parquet(OUT1 / "queries.parquet")
    db0 = ls.Database(vectors=None)
    callers = bn.load_gold(db0)
    Q = q[(q.caller == "DefenseFinder") & (q.split == "even")].sample(20, random_state=17)
    ref = pd.read_parquet(OUT2 / "final_dfall_hits.parquet", filters=[("method", "==", "S3")])
    eng = Engine(a.threads, "exact", False)
    rows, out = [], []
    for qid, r in zip(Q.qid, Q.row):
        df, _ = eng.search(callers["DefenseFinder"]["mem"][r], "S3", max_E=np.inf)
        df["qid"] = int(qid)
        out.append(df)
        b = ref[ref.qid == qid][["start_gid", "end_gid", "score", "evalue"]]
        m = b.merge(df[["start_gid", "end_gid", "score", "evalue"]], on=["start_gid", "end_gid"], how="left", suffixes=("", "_cli"))
        rows.append(dict(qid=int(qid), bench_hits=int(len(b)), found=int(m.score_cli.notna().sum()),
                         same_score=int(np.isclose(m.score, m.score_cli).sum()),
                         bench_E001=int((b.evalue <= 0.01).sum()), cli_E001_on_those=int((m.evalue_cli <= 0.01).sum()),
                         cli_E001_all=int((df.evalue <= 0.01).sum())))
    T = pd.DataFrame(rows)
    pd.concat(out).to_parquet(IDIR / "verify_cli_hits.parquet", index=False)
    T.to_csv(IDIR / "verify.tsv", sep="\t", index=False)
    res = dict(queries=int(len(T)), bench_hits=int(T.bench_hits.sum()), found=int(T.found.sum()), same_score=int(T.same_score.sum()),
               bench_E001=int(T.bench_E001.sum()), cli_E001_on_bench_hits=int(T.cli_E001_on_those.sum()),
               cli_E001_all=int(T.cli_E001_all.sum()))
    json.dump(res, open(IDIR / "verify.json", "w"), indent=1)
    log("verify", res)


def bench_cli(a):
    import numba
    numba.set_num_threads(a.threads)
    import bench as bn
    res = {}
    q = pd.read_parquet(OUT1 / "queries.parquet")
    db0 = ls.Database(vectors=None)
    callers = bn.load_gold(db0)
    Q = q[(q.caller == "DefenseFinder") & (q.split == "even")].sample(20, random_state=17)
    mem = [callers["DefenseFinder"]["mem"][r] for r in Q.row]
    ref = pd.read_parquet(OUT2 / "final_dfall_hits.parquet", filters=[("method", "==", "S3")])
    ref = ref[ref.qid.isin(Q.qid)]
    # one-shot commands (separate processes: every load included), accurate settings
    for i in range(2):
        t0 = time.time()
        sh(f"{sys.executable} -u {Path(__file__).resolve()} search --gids {','.join(map(str, mem[i]))} --threads {a.threads} "
           f"> {IDIR / f'oneshot_example_{i}.txt'}")
        res[f"oneshot_{i}_wall_seconds"] = round(time.time() - t0, 1)
    for label, emb, fast in (("accurate_exact", "exact", False), ("fast_exact", "exact", True), ("fast_hnsw200", "hnsw200", True)):
        eng = Engine(a.threads, emb, fast)
        res[f"session_{label}_load_seconds"] = {k: round(v, 1) for k, v in eng.t_load.items()}
        rows = []
        for qid, m in zip(Q.qid, mem):
            t1 = time.time()
            df, tm = eng.search(m, "S3")
            wall = time.time() - t1
            r = ref[ref.qid == qid]
            # the benchmark keeps hits with E200 <= 10; compare the top 10 windows and scores
            top = df.sort_values("score", ascending=False).head(10)
            rt = r.sort_values("score", ascending=False).head(10)
            same = (len(top) == len(rt) and np.array_equal(top.start_gid.to_numpy(), rt.start_gid.to_numpy())
                    and np.allclose(top.score.to_numpy(), rt.score.to_numpy()))
            rows.append(dict(qid=int(qid), n_genes=len(m), wall=wall, hits=len(df), hits_E001=int((df.evalue <= 0.01).sum()),
                             bench_hits_E001=int((r.evalue <= 0.01).sum()), top10_identical_to_benchmark=bool(same),
                             **{k: round(v, 3) for k, v in tm.items()}))
        T = pd.DataFrame(rows)
        T.to_csv(IDIR / f"bench_cli_{label}.tsv", sep="\t", index=False)
        res[f"per_query_session_{label}"] = dict(median_wall=float(T.wall.median()), mean_wall=float(T.wall.mean()),
                                                 first_wall=float(T.wall.iloc[0]), max_wall=float(T.wall.max()),
                                                 top10_identical_to_benchmark=float(T.top10_identical_to_benchmark.mean()),
                                                 hits_E001=int(T.hits_E001.sum()), bench_hits_E001=int(T.bench_hits_E001.sum()),
                                                 median_parts={k: float(T[k].median()) for k in
                                                               ("embedding_ranks", "sequence_ranks", "scan", "null")})
        log(label, json.dumps(res[f"per_query_session_{label}"], default=float))
        del eng
    res["threads"] = a.threads
    json.dump(res, open(IDIR / "bench_cli.json", "w"), indent=1, default=float)
    log("bench", json.dumps(res, default=float))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["prep", "search", "bench", "hnswef", "verify"])
    ap.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_ON_NODE", "2")))
    ap.add_argument("--gids", default="")
    ap.add_argument("--method", default="S3", choices=list(METHODS))
    ap.add_argument("--fast", action="store_true", help="K = 2,000 and MMseqs2 -s 5.7 --max-seqs 2000")
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--emb", default="exact", choices=["exact", "hnsw40", "hnsw200"])
    a = ap.parse_args()
    {"prep": prep, "search": cmd_search, "bench": bench_cli, "hnswef": hnsw_ef, "verify": verify}[a.step](a)


if __name__ == "__main__":
    main()
