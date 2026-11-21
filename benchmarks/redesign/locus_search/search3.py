#!/usr/bin/env python
"""Locus search, round 3 (workstream ws_search3): zca10 embedding ranks as the single input.
Predeclaration: results/criterion3.json ($RD/locus_search/out3/criterion3.json, 13:10 UTC 12 Nov 2026).
Reuses bench.py, round2.py (entries / run_one), locus_search.py, fastnull.py, evidence.py, phase0_control.py.

  search3.py lists   --threads T   control rank lists for the 18,414 member genes (task 2b/2c) and the gene-level
                                   zca10 matrix: gc, wo, genc, zodd, lc -> out3/lists/<name>/{gids,idx,s}.npy
  search3.py grid    --set main|ctl  scan-only AP grid (W extra x tau x weights), all DefenseFinder queries
  search3.py select                cross-validated and odd-genome configuration choice; task 1 AP verdict, tasks 2, 6
  search3.py final   --set df|x [--part i/n]   200-replicate null runs (seed 17 + qid); metrics, hits
                                   (E200 <= 10 or Gumbel-40 E <= 20) and null tails (top 32 per replicate)
  search3.py analyse               task 1 recall / wrong-type hits, generality (TXSScan, CRISPRCasTyper), task 4b
  search3.py rates                 task 3: fast-null curves (odd genomes), calibration on even, evidence, matched rate
  search3.py phase0                task 4a (zca10 headroom and different-type control)
  search3.py m3                    task 4c (iterative profile seeded with M2z, zca10 centroids)
  search3.py hnsw                  task 5: HNSW index on zca10 vectors; k / efSearch sweep against exact top 20,000

Families (per slot evidence = max over the listed inputs; s = log(N_db / tail rank)):
  M0 tokens | S1 sequence rank | S3 tokens + sequence + w * L2 | M1 L2 | M1z zca10 | M1p student | M2z tokens + w * zca10
  S3z tokens + sequence + w * zca10 | SM1z sequence + w * zca10 | controls C*: one list each (shuffled zca10, length and
  composition, gc, wo, genc, zca10 fitted on odd genomes, zca10 truncated to 200 / 2,000).
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
import round2  # noqa: E402
from locus_search import log  # noqa: E402

RD = ls.RD
OUT1 = RD / "locus_search/out"
OUT2 = RD / "locus_search/out2"
OUT3 = Path(os.environ.get("LS_OUT3", RD / "locus_search/out3"))
H = RD / "head"
EMBRUN = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/embed_queue/runs/dpann__esmpp_small__20261111")
FASTA = RD / "ws_dpann/work/proteins_gid.faa"
ZROW = OUT3 / "vec/zca10_f32__row.npy"
IDIR3 = OUT3 / "interactive"
HNSW_Z = IDIR3 / "hnsw_m32_efc40_zca10.faiss"
SEED = 17
K = 20000
R_NULL = 200
R_FAST = 40
EV = round2.EV

SRC_DIR = {"l2": OUT1 / "topk", "seq": OUT1 / "topk_seq", "zca10": H / "ls_zca10/topk_student",
           "student": H / "ls/topk_student", **{n: OUT3 / "lists" / n for n in ("gc", "wo", "genc", "zodd", "lc")}}
DERIVED = {"zshuf": "zca10", "z200": "zca10", "z2000": "zca10", "hnsw": "zca10"}
W_E = (0.25, 0.5, 0.75, 1.0)
W_M2 = (0.5, 0.75, 1.0)
FAM = {
    "M0": [(True, 0.0, None, 0.0)],
    "S1": [(False, 1.0, None, 0.0)],
    "S3": [(True, 1.0, "l2", w) for w in W_E],
    "M1": [(False, 0.0, "l2", 1.0)],
    "M1z": [(False, 0.0, "zca10", 1.0)],
    "M1p": [(False, 0.0, "student", 1.0)],
    "M2z": [(True, 0.0, "zca10", w) for w in W_M2],
    "S3z": [(True, 1.0, "zca10", w) for w in W_E],
    "SM1z": [(False, 1.0, "zca10", w) for w in W_M2],
    "Cshuf": [(False, 0.0, "zshuf", 1.0)],
    "Clc": [(False, 0.0, "lc", 1.0)],
    "Cgc": [(False, 0.0, "gc", 1.0)],
    "Cwo": [(False, 0.0, "wo", 1.0)],
    "Cgenc": [(False, 0.0, "genc", 1.0)],
    "Czodd": [(False, 0.0, "zodd", 1.0)],
    "Cz200": [(False, 0.0, "z200", 1.0)],
    "Cz2000": [(False, 0.0, "z2000", 1.0)],
    "Hnsw": [(False, 0.0, "hnsw", 1.0)],
}
SETS = {"main": ["M0", "S1", "S3", "M1", "M1z", "M1p", "M2z", "S3z", "SM1z"],
        "ctl": ["Cshuf", "Clc", "Cgc", "Cwo", "Cgenc", "Czodd", "Cz200", "Cz2000"]}
RATE_FAMS = ["M0", "S1", "S3", "M1z", "M1p", "M2z", "SM1z", "S3z"]
CV_FAMS = ["M1z", "S1", "S3", "M1p", "SM1z"]
X_FAMS = ["M0", "S1", "S3", "M1z"]
SRCS = {}


def jd(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def dump(obj, path):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=jd) + "\n")
    tmp.replace(path)


def wkey(cfg):
    use_tok, ws, src, we = cfg
    return float(we if src is not None else ws)


def cfg_of(fam, w):
    for c in FAM[fam]:
        if abs(wkey(c) - w) < 1e-9:
            return c
    raise KeyError((fam, w))


def boot(v, g):
    return ls.boot_mean(np.asarray(v, float), np.asarray(g))


# =========================================================================== sources
def load_sources(names):
    g0 = np.load(OUT1 / "topk/gids.npy")
    need = {"seq"} | {DERIVED.get(n, n) for n in names if n}
    for n in sorted(need):
        if n in SRCS:
            continue
        d = SRC_DIR[n]
        assert np.array_equal(np.load(d / "gids.npy"), g0), f"gids of {n} differ"
        SRCS[n] = (np.load(d / "idx.npy"), np.load(d / "s.npy"))
    z = SRCS.get("zca10")
    if "zshuf" in names:
        perm = np.random.default_rng(SEED).permutation(len(g0))       # as round 2's shuffled control
        SRCS["zshuf"] = (z[0][perm], z[1][perm])
    if "z200" in names:
        SRCS["z200"] = (z[0][:, :200], z[1][:, :200])
    if "z2000" in names:
        SRCS["z2000"] = (z[0][:, :2000], z[1][:, :2000])
    EV["row_of"] = {int(g): k for k, g in enumerate(g0)}
    EV["s_idx"], EV["s_s"] = SRCS["seq"]
    log("sources", sorted(SRCS))
    return g0


def use(src):
    if src is not None:
        EV["e_idx"], EV["e_s"] = SRCS[src]


def setup(a, sources):
    db, callers, queries = bench.setup(a, vectors=None)
    bench.CTX["rel"] = bench.load_relevance(OUT1, queries, callers)
    load_sources(sources)
    bench.warm_up(db)
    return db, callers, queries


def fam_sources(fams):
    return sorted({c[2] for f in fams for c in FAM[f] if c[2]})


# =========================================================================== lists (task 2b, 2c)
def lists(a):
    sys.path.insert(0, str(RD / "lab/code"))
    import embed_tokens as et
    (OUT3 / "lists").mkdir(parents=True, exist_ok=True)
    ZROW.parent.mkdir(parents=True, exist_ok=True)
    db = ls.Database(vectors=None)
    gids = np.load(OUT1 / "topk/gids.npy")
    u = np.load(EMBRUN / "uid_of_row.npy").astype(np.int64)
    assert len(u) == db.N and u.min() >= 0
    info = json.load(open(OUT3 / "lists/lists.json")) if (OUT3 / "lists/lists.json").exists() else {}
    t0 = time.time()
    if not ZROW.exists():
        Z = np.load(H / "data/zca10_f16__uid.npy").astype(np.float32)
        np.save(ZROW, Z[u])
        del Z
        info["zca10_rows_seconds"] = round(time.time() - t0, 1)

    def topk_rows(name, R):
        d = OUT3 / "lists" / name
        if (d / "s.npy").exists():
            log("exists", name)
            return
        d.mkdir(parents=True, exist_ok=True)
        db.X = R
        t1 = time.time()
        idx, _, s = ls.topk_cosine(db, R[gids], db.gc[gids], K=K)
        np.save(d / "gids.npy", gids)
        np.save(d / "idx.npy", idx)
        np.save(d / "s.npy", s)
        info[name] = dict(info.get(name, {}), topk_seconds=round(time.time() - t1, 1), dim=int(R.shape[1]))
        db.X = None
        dump(info, OUT3 / "lists/lists.json")
        log("lists", name, info[name])

    def rows16(V):
        return V.astype(np.float16).astype(np.float32)[u]              # as zca10 (stored float16)

    X, uu, valid, cnt = et.load_run(EMBRUN)
    assert (uu == u).all() and valid.all()
    gc_row = db.gc
    # zca10 recomputation check against the stored vectors
    if "zca10_check" not in info:
        V = X.copy()
        tz = et.transform(V, valid, cnt, "zca10")
        Zs = np.load(H / "data/zca10_f16__uid.npy", mmap_mode="r")
        pick = np.sort(np.random.default_rng(SEED).choice(len(V), 100000, replace=False))
        info["zca10_check"] = dict(max_abs_diff_vs_stored_f16=float(np.abs(V[pick] - Zs[pick].astype(np.float32)).max()),
                                   **{k: v for k, v in tz.items() if k != "transform"})
        del V, Zs
        dump(info, OUT3 / "lists/lists.json")
        log("zca10 check", info["zca10_check"])
    # gc: mean removal alone
    if not (OUT3 / "lists/gc/s.npy").exists():
        V = X.copy()
        info["gc"] = et.transform(V, valid, cnt, "gc")
        R = rows16(V)
        del V
        topk_rows("gc", R)
        del R
    # wo: whitening alone (soft ZCA of the uncentred second-moment matrix)
    if not (OUT3 / "lists/wo/s.npy").exists():
        V = X.copy()
        et.unit(V)
        W, ev = et.soft_zca(et.weighted_cov(V, cnt))
        et.apply_linear(V, W)
        et.unit(V)
        info["wo"] = dict(eigenvalue_q=np.quantile(ev, [0, 0.5, 0.9, 1]).tolist(),
                          participation_ratio_before=float(ev.sum() ** 2 / (ev ** 2).sum()))
        R = rows16(V)
        del V
        topk_rows("wo", R)
        del R
    # zodd: zca10 with mean and covariance from odd-genome sequences only
    if not (OUT3 / "lists/zodd/s.npy").exists():
        cnt_odd = np.bincount(u[gc_row % 2 == 1], minlength=len(X)).astype(np.float64)
        V = X.copy()
        et.unit(V)
        mu = et.weighted_mean(V, cnt_odd)
        for s in range(0, len(V), et.BLK):
            V[s:s + et.BLK] -= mu.astype(np.float32)
        W, ev = et.soft_zca(et.weighted_cov(V, cnt_odd))
        et.apply_linear(V, W)
        et.unit(V)
        info["zodd"] = dict(odd_rows=int(cnt_odd.sum()), uids_with_odd_rows=int((cnt_odd > 0).sum()),
                            mean_norm=float(np.linalg.norm(mu)), eigenvalue_q=np.quantile(ev, [0, 0.5, 0.9, 1]).tolist())
        R = rows16(V)
        del V
        topk_rows("zodd", R)
        del R
    # genc: per-genome mean removal (gene level)
    if not (OUT3 / "lists/genc/s.npy").exists():
        V = X.copy()
        et.unit(V)
        R = V[u]
        del V
        mn = []
        for g in range(db.n_genomes):
            lo, hi = int(db.g_lo[g]), int(db.g_hi[g])
            if hi > lo:
                m = R[lo:hi].mean(0)
                mn.append(float(np.linalg.norm(m)))
                R[lo:hi] -= m
        et.unit(R)
        info["genc"] = dict(genome_mean_norm_q=np.quantile(mn, [0, 0.5, 1]).tolist())
        R = R.astype(np.float16).astype(np.float32)
        topk_rows("genc", R)
        del R
    del X
    # lc: length and amino-acid composition
    if not (OUT3 / "lists/lc/s.npy").exists():
        t1 = time.time()
        raw = open(FASTA, "rb").read()
        lines = raw.split(b"\n")
        del raw
        if lines and lines[-1] == b"":
            lines.pop()
        assert len(lines) == 2 * db.N, "FASTA is not one header + one sequence line per gene"
        hdr = np.fromiter((int(h[1:].split()[0]) for h in lines[0::2]), np.int64, db.N)
        assert np.array_equal(hdr, np.arange(db.N))
        seqs = lines[1::2]
        del lines
        lens = np.fromiter((len(s) for s in seqs), np.int64, db.N)
        arr = np.frombuffer(b"".join(seqs), np.uint8)
        del seqs
        lut = np.full(256, 20, np.uint8)
        for i, c in enumerate(b"ACDEFGHIKLMNPQRSTVWY"):
            lut[c] = i
            lut[c + 32] = i
        codes = lut[arr]
        off = np.r_[0, np.cumsum(lens)]
        cnt21 = np.zeros((db.N, 21), np.int64)
        for a0 in range(0, db.N, 200000):
            b0 = min(db.N, a0 + 200000)
            seg = codes[off[a0]:off[b0]].astype(np.int64)
            gi = np.repeat(np.arange(b0 - a0), lens[a0:b0])
            cnt21[a0:b0] = np.bincount(gi * 21 + seg, minlength=(b0 - a0) * 21).reshape(b0 - a0, 21)
        std = cnt21[:, :20].sum(1)
        F = np.empty((db.N, 21), np.float64)
        F[:, :20] = cnt21[:, :20] / np.maximum(std, 1)[:, None]
        F[:, 20] = np.log10(np.maximum(std, 1))
        F = (F - F.mean(0)) / F.std(0)
        ev, U = np.linalg.eigh(np.cov(F, rowvar=False))
        keep = ev > 1e-10 * ev.max()                                   # composition sums to 1: one null direction
        Wz = (U[:, keep] / np.sqrt(ev[keep])) @ U[:, keep].T
        Fw = (F @ Wz).astype(np.float32)
        B = np.empty((db.N, 22), np.float32)                          # inner product = f_a.f_b - |f_b|^2 / 2
        B[:, :21] = Fw
        B[:, 21] = -0.5 * (Fw.astype(np.float64) ** 2).sum(1)
        info["lc"] = dict(features="20 standard amino-acid frequencies + log10(standard residues); z-scored; ZCA "
                                   "(pseudo-inverse: null eigen-directions dropped)", eigen_kept=int(keep.sum()),
                          nonstandard_residue_share=float(cnt21[:, 20].sum() / cnt21.sum()),
                          length_q=np.quantile(std, [0.1, 0.5, 0.9]).tolist(), feature_seconds=round(time.time() - t1, 1))
        d = OUT3 / "lists/lc"
        d.mkdir(parents=True, exist_ok=True)
        db.X = B
        Aq = np.empty((len(gids), 22), np.float32)
        Aq[:, :21] = Fw[gids]
        Aq[:, 21] = 1.0
        t1 = time.time()
        idx, _, s = ls.topk_cosine(db, Aq, db.gc[gids], K=K)
        np.save(d / "gids.npy", gids)
        np.save(d / "idx.npy", idx)
        np.save(d / "s.npy", s)
        info["lc"]["topk_seconds"] = round(time.time() - t1, 1)
        # rank sanity: the first neighbour must be the database gene at the smallest Euclidean distance
        r = 0
        dist = ((Fw - Fw[gids[r]]) ** 2).sum(1)
        lo, hi = db.own_range(int(db.gc[gids[r]]))
        dist[lo:hi] = np.inf
        info["lc"]["check_first_neighbour"] = bool(dist[idx[r, 0]] <= dist.min() + 1e-4)
        db.X = None
    info["seconds"] = round(time.time() - t0, 1)
    dump(info, OUT3 / "lists/lists.json")
    log("lists done", info)


# =========================================================================== grid (scan-only AP)
def grid_task(qid):
    q = bench.CTX["queries"].loc[qid]
    db = bench.CTX["db"]
    out = []
    for fam in bench.CTX["grid_fams"]:
        for cfg in FAM[fam]:
            use_tok, ws, src, we = cfg
            use(src)
            (eg, es_, ev), ns = round2.entries(q, use_tok, ws, we)
            for wx, tau in product(bench.GRID_WX, bench.GRID_TAU):
                ha, hb, sc, _ = ls.scan(eg, es_, ev, ns, db.cc, db.pos, bench.W_of(q, wx), tau, 0)
                r, _, _ = bench.label_eval(q, ha, hb, sc, None)
                out.append((int(qid), fam, wkey(cfg), wx, tau, r["ap"], r["ap_strict"], r["n_rel"], r["n_hits"]))
    return out


def grid(a):
    OUT3.mkdir(parents=True, exist_ok=True)
    fams = SETS[a.set]
    db, callers, queries = setup(a, fam_sources(fams))
    bench.CTX["grid_fams"] = fams
    Q = queries[queries.caller == "DefenseFinder"]
    qids = sorted(Q.qid.tolist(), key=lambda x: -queries.n_members[x])
    t0 = time.time()
    rows = [r for ch in bench.pool_map(grid_task, qids, a.threads, chunksize=2) for r in ch]
    G = pd.DataFrame(rows, columns=["qid", "fam", "w", "wx", "tau", "ap", "ap_strict", "n_rel", "n_hits"])
    G = G.merge(queries[["qid", "genome_code", "split", "n_members", "type"]], on="qid")
    G.to_parquet(OUT3 / f"grid_{a.set}.parquet", index=False)
    log(f"grid {a.set}: {len(qids):,} queries x {sum(len(FAM[f]) for f in fams) * 16} configs, {time.time() - t0:.0f}s")
    if a.set == "main":
        selfcheck_grid(G)


def _cmp(x, ycol="ap_ref"):
    d = (x["ap"].fillna(-1) - x[ycol].fillna(-1)).abs()
    return dict(n=int(len(x)), max_abs_diff=float(d.max()) if len(x) else None)


def selfcheck_grid(G):
    chk = {}
    r2 = pd.read_parquet(OUT2 / "cv_grid.parquet")
    r2 = r2[r2.fam == "S3"][["qid", "w", "wx", "tau", "ap"]].rename(columns={"ap": "ap_ref"})
    chk["S3_vs_round2_cv_grid"] = _cmp(G[G.fam == "S3"].merge(r2, on=["qid", "w", "wx", "tau"]))
    for fam, path in (("M1z", H / "ls_zca10/tune_odd.parquet"), ("M1p", H / "ls/tune_odd.parquet")):
        h = pd.read_parquet(path)
        h = h[h.fam == "M1p"][["qid", "wx", "tau", "ap"]].rename(columns={"ap": "ap_ref"})
        chk[f"{fam}_vs_head_tune_odd"] = _cmp(G[G.fam == fam].merge(h, on=["qid", "wx", "tau"]))
    t1 = pd.read_parquet(OUT1 / "tune_odd.parquet")
    for fam, mode in (("M0", "M0"), ("M1", "M1")):
        h = t1[t1["mode"] == mode][["qid", "wx", "tau", "ap"]].rename(columns={"ap": "ap_ref"})
        chk[f"{fam}_vs_round1_tune_odd"] = _cmp(G[G.fam == fam].merge(h, on=["qid", "wx", "tau"]))
    ts = pd.read_parquet(OUT1 / "seqctl_tune_odd.parquet")
    h = ts[ts["mode"] == "S1"][["qid", "wx", "tau", "ap"]].rename(columns={"ap": "ap_ref"})
    chk["S1_vs_round1_seqctl_tune_odd"] = _cmp(G[G.fam == "S1"].merge(h, on=["qid", "wx", "tau"]))
    dump(chk, OUT3 / "grid_selfcheck.json")
    log("grid self-check", json.dumps(chk))
    bad = {k: v for k, v in chk.items() if not v["n"] or v["max_abs_diff"] > 1e-9}
    assert not bad, f"grid does not reproduce earlier rounds: {bad}"


# =========================================================================== select (task 1 AP, 2, 6)
def folds_of(G):
    gcodes = np.unique(G.genome_code)
    perm = np.random.default_rng(SEED).permutation(gcodes)
    return {int(c): i % 5 for i, c in enumerate(perm)}


def choose(tr):
    agg = tr.groupby(["w", "wx", "tau"]).agg(ap=("ap", "mean")).reset_index()
    b = agg.sort_values(["ap", "wx", "tau", "w"], ascending=[False, True, False, True]).iloc[0]
    return dict(w=float(b.w), wx=int(b.wx), tau=float(b.tau), train_ap=float(b.ap), queries=int(tr.qid.nunique()))


def select(a):
    G = pd.concat([pd.read_parquet(p) for p in sorted(OUT3.glob("grid_*.parquet"))], ignore_index=True)
    G = G[G.n_rel > 0].copy()
    fold_of = folds_of(G[G.fam == "M0"])
    G["fold"] = G.genome_code.map(fold_of)
    r2h = pd.read_parquet(OUT2 / "cv_heldout.parquet")[["qid", "fold", "ap_S3"]]
    qf = G.drop_duplicates("qid").set_index("qid").fold
    folds_same = bool((qf.loc[r2h.qid].to_numpy() == r2h.fold.to_numpy()).all() and len(qf) == len(r2h))
    fams = [f for f in FAM if f in set(G.fam)]
    chosen = {f: [choose(G[(G.fam == f) & (G.fold != k)]) for k in range(5)] for f in fams}
    odd = {f: choose(G[(G.fam == f) & (G.split == "odd")]) for f in fams}
    held = []
    for f in fams:
        for k in range(5):
            c = chosen[f][k]
            held.append(G[(G.fam == f) & (G.fold == k) & (G.w == c["w"]) & (G.wx == c["wx"]) & (G.tau == c["tau"])])
    Hd = pd.concat(held, ignore_index=True)
    Hd.to_parquet(OUT3 / "cv_heldout.parquet", index=False)
    X = Hd.pivot_table(index=["qid", "genome_code", "split", "n_members", "fold"], columns="fam",
                       values=["ap", "ap_strict"]).reset_index()
    X.columns = ["_".join(c).strip("_") for c in X.columns]
    s3chk = X.set_index("qid").ap_S3.loc[r2h.qid] - r2h.set_index("qid").ap_S3
    res = dict(folds_reproduce_round2=folds_same, n_queries=int(len(X)), n_genomes=int(X.genome_code.nunique()),
               S3_heldout_vs_round2_max_abs_diff=float(s3chk.abs().max()), chosen=chosen, odd=odd, tables={}, paired={})
    strata = {"all": np.ones(len(X), bool), "even": (X.split == "even").to_numpy(), "odd": (X.split == "odd").to_numpy(),
              "min3": (X.n_members >= 3).to_numpy()}
    pairs = [("M1z", "S1"), ("M1z", "S3"), ("M1z", "M1"), ("M1z", "M0"), ("M1p", "M1z"), ("M1p", "S1"), ("M2z", "M1z"),
             ("SM1z", "M1z"), ("S3z", "M1z"), ("S3z", "S3"), ("S1", "M0"), ("S3", "S1"),
             ("M1z", "Cshuf"), ("M1z", "Clc"), ("Clc", "S1"), ("Clc", "M0"), ("Cshuf", "M0"), ("M1z", "Cgc"), ("M1z", "Cwo"),
             ("M1z", "Cgenc"), ("Cgc", "M1"), ("Cwo", "M1"), ("Cgenc", "M1"), ("Czodd", "M1z"), ("Cz200", "M1z"),
             ("Cz2000", "M1z")]
    for name, m in strata.items():
        S = X[m]
        res["tables"][name] = {f: dict(queries=int(S[f"ap_{f}"].notna().sum()), ap=boot(S[f"ap_{f}"], S.genome_code),
                                       ap_strict=boot(S[f"ap_strict_{f}"], S.genome_code))
                               for f in fams if f"ap_{f}" in S}
        pr = {}
        for x, y in pairs:
            if f"ap_{x}" in S and f"ap_{y}" in S:
                pr[f"{x}-{y}"] = {c: boot(S[f"{c}_{x}"] - S[f"{c}_{y}"], S.genome_code) for c in ("ap", "ap_strict")}
        res["paired"][name] = pr
    d1, d2 = res["paired"]["all"]["M1z-S1"]["ap"], res["paired"]["all"]["M1z-S3"]["ap"]
    res["task1"] = dict(M1z_minus_S1=d1, M1z_minus_S3=d2, cond_i=bool(d1[0] >= 0.10 and d1[1] > 0),
                        cond_ii=bool(d2[0] > 0 and d2[1] > 0),
                        per_fold={f"{x}-{y}": {int(k): float((g[f"ap_{x}"] - g[f"ap_{y}"]).mean()) for k, g in X.groupby("fold")}
                                  for x, y in (("M1z", "S1"), ("M1z", "S3"))})
    res["task1"]["confirmed"] = bool(res["task1"]["cond_i"] and res["task1"]["cond_ii"])
    if "ap_M1p" in X:
        res["task6"] = dict(M1p_minus_M1z_even=res["paired"]["even"]["M1p-M1z"],
                            survives=bool(res["paired"]["even"]["M1p-M1z"]["ap"][1] > 0))
    dump(res, OUT3 / "cv_summary3.json")
    dump(dict(fold_of={str(k): v for k, v in fold_of.items()}, chosen=chosen, odd=odd), OUT3 / "selection3.json")
    log("task1", json.dumps(res["task1"], default=jd), "folds same", folds_same, "S3 vs round 2", res["S3_heldout_vs_round2_max_abs_diff"])
    log("odd configs", json.dumps({f: (c["w"], c["wx"], c["tau"]) for f, c in odd.items()}))


# =========================================================================== final (200-replicate null)
def final_task(args):
    qid, fam, w, wx, tau = args
    use_tok, ws, src, we = cfg_of(fam, w)
    use(src)
    q = bench.CTX["queries"].loc[qid]
    r = round2.run_one(q, use_tok, ws, we, wx, tau, R_NULL, bench.TOP_N, 32, SEED + int(qid), 10.0)
    E200, _ = ls.evalues(r["sc"], r["tail"], R_NULL)
    Eg = fastnull.evalue_gumbel(r["sc"], r["tail"][:R_FAST], R_FAST)
    m, lab, match = bench.label_eval(q, r["ha"], r["hb"], r["sc"], E200)
    m.update(qid=int(qid), fam=fam, w=float(w), wx=int(wx), tau=float(tau), W=r["W"], n_entries=r["n_entries"],
             sec_scan=r["sec_scan"], sec_null=r["sec_null"])
    keep = (E200 <= 10) | (Eg <= 20)
    hits = pd.DataFrame(dict(qid=int(qid), fam=fam, w=float(w), wx=int(wx), tau=float(tau), start_gid=r["ha"][keep],
                             end_gid=r["hb"][keep], score=r["sc"][keep], E200=E200[keep], Egum40=Eg[keep],
                             label=lab[keep], match_row=match[keep], n_slots_filled=r["nf"][keep]))
    return m, hits, r["tail"][:, :32].astype(np.float32)


def final_tasks(queries, sel, which):
    fold_of = {int(k): v for k, v in sel["fold_of"].items()}
    tasks = set()
    if which == "df":
        Q = queries[queries.caller == "DefenseFinder"]
        for qid, gc in zip(Q.qid, Q.genome_code):
            for f in RATE_FAMS:
                c = sel["odd"][f]
                tasks.add((int(qid), f, c["w"], c["wx"], c["tau"]))
            k = fold_of.get(int(gc))
            if k is not None:
                for f in CV_FAMS:
                    c = sel["chosen"][f][k]
                    tasks.add((int(qid), f, c["w"], c["wx"], c["tau"]))
    elif which == "tau":                         # POST HOC: the other tau values of the grid for M1z and S1 (W + 8)
        Q = queries[queries.caller == "DefenseFinder"]
        for qid in Q.qid:
            for f in ("M1z", "S1"):
                for tau in (0.0, 1.0, 2.0):
                    tasks.add((int(qid), f, 1.0, 8, tau))
    else:
        Q = queries[queries.caller.isin(["TXSScan", "CRISPRCasTyper"]) & (queries.split == "even")]
        for qid in Q.qid:
            for f in X_FAMS:
                c = sel["odd"][f]
                tasks.add((int(qid), f, c["w"], c["wx"], c["tau"]))
    return sorted(tasks, key=lambda t: (-int(queries.n_members[t[0]]), t))


def final(a):
    sel = json.load(open(OUT3 / "selection3.json"))
    fams = {"df": RATE_FAMS, "x": X_FAMS, "tau": ["M1z", "S1"]}[a.set]
    db, callers, queries = setup(a, fam_sources(fams))
    tasks = final_tasks(queries, sel, a.set)
    tag = a.set
    if a.part:
        i, n = (int(x) for x in a.part.split("/"))
        tasks = tasks[i::n]
        tag = f"{a.set}{i}"
    log(f"final {tag}: {len(tasks):,} runs")
    t0 = time.time()
    res = bench.pool_map(final_task, tasks, a.threads, chunksize=1)
    log(f"final {tag}: {len(tasks):,} runs, {time.time() - t0:.0f}s")
    M = pd.DataFrame([m for m, _, _ in res]).merge(queries, on="qid")
    M.to_parquet(OUT3 / f"final_{tag}_metrics.parquet", index=False)
    pd.concat([h for _, h, _ in res], ignore_index=True).to_parquet(OUT3 / f"final_{tag}_hits.parquet", index=False)
    np.savez(OUT3 / f"final_{tag}_tails.npz", qid=np.array([m["qid"] for m, _, _ in res]),
             fam=np.array([m["fam"] for m, _, _ in res]), w=np.array([m["w"] for m, _, _ in res]),
             wx=np.array([m["wx"] for m, _, _ in res]), tau=np.array([m["tau"] for m, _, _ in res]),
             tails=np.stack([t for _, _, t in res]))


# =========================================================================== analyse (task 1 recall, generality, 4b)
def load_final(which):
    ps = sorted(OUT3.glob(f"final_{which}*_metrics.parquet"))
    return pd.concat([pd.read_parquet(p) for p in ps], ignore_index=True) if ps else pd.DataFrame()


def pick_rows(M, sel, fam, role):
    """Rows of one family at its odd-genome configuration (role 'odd') or its fold configuration (role 'cv')."""
    g = M[M.fam == fam]
    if role == "odd":
        c = sel["odd"][fam]
        return g[(g.w == c["w"]) & (g.wx == c["wx"]) & (g.tau == c["tau"])]
    fold_of = {int(k): v for k, v in sel["fold_of"].items()}
    f = g.genome_code.map(fold_of)
    ok = np.zeros(len(g), bool)
    for k in range(5):
        c = sel["chosen"][fam][k]
        ok |= ((f == k) & (g.w == c["w"]) & (g.wx == c["wx"]) & (g.tau == c["tau"])).to_numpy()
    return g[ok]


COLS = ["ap", "ap_strict", "recall_E0.01", "recall_E1", "n_wrong_E0.01", "n_unlabelled_E0.01", "n_correct_E0.01",
        "precision_E0.01"]


def table(rows, cols=COLS):
    return {c: boot(rows[c], rows.genome_code) for c in cols if c in rows} | {"queries": int(len(rows))}


def paired(x, y, cols):
    x, y = x.set_index("qid"), y.set_index("qid")
    ids = x.index.intersection(y.index)
    g = y.loc[ids, "genome_code"].to_numpy()
    return {c: boot(x.loc[ids, c].to_numpy() - y.loc[ids, c].to_numpy(), g) for c in cols} | {"queries": int(len(ids))}


def invisible(rows):
    d = {}
    for t in ("E0.01", "E1"):
        d[f"micro_recall_inv_{t}"] = float(rows[f"found_inv_{t}"].sum() / max(rows.n_rel_inv.sum(), 1))
        d[f"micro_recall_vis_{t}"] = float(rows[f"found_vis_{t}"].sum() / max(rows.n_rel_vis.sum(), 1))
        d[f"found_inv_{t}"] = int(rows[f"found_inv_{t}"].sum())
        d[f"recall_inv_{t}"] = boot(rows[f"recall_inv_{t}"], rows.genome_code)
    d["invisible_targets"] = int(rows.n_rel_inv.sum())
    d["targets"] = int(rows.n_rel.sum())
    return d


def analyse(a):
    sel = json.load(open(OUT3 / "selection3.json"))
    M = load_final("df")
    M = M[M.n_rel > 0]
    X = load_final("x")
    X = X[X.n_rel > 0] if len(X) else X
    res = dict(task1_cv={}, task1_cv_even={}, generality={}, task4b={}, selfcheck={})
    # ---- self-checks against rounds 1-2 and the head workstream (same seeds, same configurations)
    r2 = pd.read_parquet(OUT2 / "final_dfall_metrics.parquet")
    hz = pd.read_parquet(H / "ls_zca10/final_metrics.parquet")
    hp = pd.read_parquet(H / "ls/final_metrics.parquet")
    for fam, ref, cfg in (("S3", r2[r2.method == "S3"], (0.5, 8, 0.0)), ("M0", r2[r2.method == "M0"], (0.0, 8, 3.0)),
                          ("M1z", hz[hz.method == "M1z"], (1.0, 8, 3.0)), ("M1p", hp[hp.method == "M1p"], (1.0, 8, 3.0))):
        x = M[(M.fam == fam) & (M.w == cfg[0]) & (M.wx == cfg[1]) & (M.tau == cfg[2])].drop_duplicates("qid").set_index("qid")
        y = ref.drop_duplicates("qid").set_index("qid")
        ids = x.index.intersection(y.index)
        res["selfcheck"][fam] = {c: float((x.loc[ids, c] - y.loc[ids, c]).abs().max()) for c in ("ap", "recall_E0.01", "n_unlabelled_E0.01")} \
            | {"n": int(len(ids)), "config": cfg}
    log("self-check", json.dumps(res["selfcheck"]))
    # ---- task 1: fold configurations, all held-out queries (and even only)
    cv = {f: pick_rows(M, sel, f, "cv") for f in CV_FAMS}
    for key, flt in (("task1_cv", None), ("task1_cv_even", "even")):
        T = {f: (r if flt is None else r[r.split == flt]) for f, r in cv.items()}
        res[key]["tables"] = {f: table(r) for f, r in T.items()}
        res[key]["paired"] = {f"{x}-{y}": paired(T[x], T[y], ["ap", "ap_strict", "recall_E0.01", "recall_E1", "n_wrong_E0.01",
                                                              "n_unlabelled_E0.01"])
                              for x, y in (("M1z", "S1"), ("M1z", "S3"), ("M1p", "M1z"), ("SM1z", "M1z"), ("SM1z", "S1"))}
    # ---- generality: frozen (odd-genome DefenseFinder) configurations, even genomes, three callers
    for caller in ("DefenseFinder", "TXSScan", "CRISPRCasTyper"):
        src = M[M.split == "even"] if caller == "DefenseFinder" else X[X.caller == caller]
        if not len(src):
            continue
        T = {f: pick_rows(src, sel, f, "odd") for f in X_FAMS}
        d = dict(tables={f: table(r) for f, r in T.items()},
                 paired={f"{x}-{y}": paired(T[x], T[y], ["ap", "ap_strict", "recall_E0.01", "n_wrong_E0.01"])
                         for x, y in (("M1z", "S1"), ("M1z", "S3"), ("M1z", "M0"))})
        d["same_sign"] = bool(d["paired"]["M1z-S1"]["ap"][0] > 0 and d["paired"]["M1z-S3"]["ap"][0] > 0)
        mm = src[src.n_members >= 3]
        d["min3_ap"] = {f: boot(pick_rows(mm, sel, f, "odd").ap, pick_rows(mm, sel, f, "odd").genome_code) for f in X_FAMS}
        res["generality"][caller] = d
    res["general"] = bool(all(res["generality"].get(c, {}).get("same_sign", False) for c in ("TXSScan", "CRISPRCasTyper")))
    # ---- task 4b: token-invisible recall (odd-genome configurations, even genomes; and fold configurations, all)
    ev = M[M.split == "even"]
    T = {f: pick_rows(ev, sel, f, "odd") for f in RATE_FAMS}
    res["task4b"]["even_odd_configs"] = {f: invisible(r) for f, r in T.items()}
    inv = {f: r[r.n_rel_inv > 0] for f, r in T.items()}
    res["task4b"]["paired_even"] = {f"{x}-{y}": paired(inv[x], inv[y], ["recall_inv_E0.01", "recall_inv_E1"])
                                    for x, y in (("M1z", "S1"), ("SM1z", "S1"), ("SM1z", "M1z"), ("M1z", "M0"), ("S3z", "S3"))}
    res["task4b"]["cv_all"] = {f: invisible(r) for f, r in cv.items()}
    res["timing_cpu_seconds_per_query"] = {f: dict(scan=float(r.sec_scan.mean()), null200=float(r.sec_null.mean()))
                                           for f, r in T.items()}
    dump(res, OUT3 / "summary_final3.json")
    log("generality", json.dumps({c: v["paired"]["M1z-S1"]["ap"] for c, v in res["generality"].items()}, default=jd))


# =========================================================================== rates (task 3)
XG = fastnull.XGRID


def apply_curve_ext(E, r):
    lx, lr = np.log10(XG), np.log10(r)
    le = np.log10(np.maximum(np.asarray(E, float), 1e-12))
    out = np.interp(le, lx, lr)
    lo, hi = le < lx[0], le > lx[-1]
    out[lo] = le[lo] + (lr[0] - lx[0])
    out[hi] = le[hi] + (lr[-1] - lx[-1])
    return 10.0 ** out


def null_counts(tail):
    fit, hold = tail[:R_FAST].astype(np.float64), tail[R_FAST:].astype(np.float64)
    e = np.sort(fastnull.evalue_gumbel(hold[np.isfinite(hold)], fit, R_FAST))
    return np.searchsorted(e, XG, side="right") / hold.shape[0]


def rates(a):
    import evidence as evm
    t0 = time.time()
    sel = json.load(open(OUT3 / "selection3.json"))
    db = evm.attach_db()
    callers = bench.load_gold(db)
    queries = pd.read_parquet(OUT1 / "queries.parquet")
    rel = bench.load_relevance(OUT1, queries, callers)
    M = load_final("df")
    zs = [np.load(p, allow_pickle=True) for p in sorted(OUT3.glob("final_df*_tails.npz"))]
    T = pd.DataFrame(dict(qid=np.concatenate([z["qid"] for z in zs]), fam=np.concatenate([z["fam"] for z in zs]),
                          w=np.concatenate([z["w"] for z in zs]), wx=np.concatenate([z["wx"] for z in zs]),
                          tau=np.concatenate([z["tau"] for z in zs])))
    tails = np.concatenate([z["tails"] for z in zs])
    T["split"] = T.qid.map(dict(zip(queries.qid, queries.split)))
    Hh = pd.concat([pd.read_parquet(p) for p in sorted(OUT3.glob("final_df*_hits.parquet"))], ignore_index=True)
    nrel_ok = set(M[M.n_rel > 0].qid)
    res = dict(curves_ratio_at_E={}, calibration_even={}, at_E0_01={}, matched={}, paired_matched={}, paired_E0_01={})
    curves, Hs = {}, []
    rng = np.random.default_rng(SEED)
    for fam in RATE_FAMS:
        c = sel["odd"][fam]
        is_f = ((T.fam == fam) & (T.w == c["w"]) & (T.wx == c["wx"]) & (T.tau == c["tau"])).to_numpy()
        odd_i = np.flatnonzero(is_f & (T.split == "odd").to_numpy())
        even_i = np.flatnonzero(is_f & (T.split == "even").to_numpy())
        cnt_odd = np.array([null_counts(tails[i]) for i in odd_i])
        curve = fastnull.calibration_curve(cnt_odd)
        curves[fam] = curve
        res["curves_ratio_at_E"][fam] = {str(x): float(curve[int(np.argmin(np.abs(np.log10(XG) - np.log10(x))))] / x)
                                         for x in (0.001, 0.01, 0.1, 1.0)} | {"odd_runs": int(len(odd_i))}
        # calibration on even genomes: held-out replicates 41-200 counted through the curve
        cnt_even = np.array([null_counts(tails[i]) for i in even_i])
        xprime = 10.0 ** np.interp(np.log10(XG), np.log10(curve), np.log10(XG))
        cal = np.array([np.interp(np.log10(xprime), np.log10(XG), row) for row in cnt_even])
        d = {}
        for x in (0.001, 0.01, 0.1, 1.0):
            j = int(np.argmin(np.abs(np.log10(XG) - np.log10(x))))
            v = cal[:, j]
            bm = v[rng.integers(0, len(v), size=(2000, len(v)))].mean(1)
            d[str(x)] = dict(ratio=float(v.mean() / x), ci=[float(np.quantile(bm, .025) / x), float(np.quantile(bm, .975) / x)])
        res["calibration_even"][fam] = d | {"even_runs": int(len(even_i))}
        h = Hh[(Hh.fam == fam) & (Hh.w == c["w"]) & (Hh.wx == c["wx"]) & (Hh.tau == c["tau"]) & Hh.qid.isin(nrel_ok)].copy()
        h["E_fast"] = apply_curve_ext(h.Egum40.to_numpy(), curve)
        Hs.append(h)
    np.savez(OUT3 / "fastnull_curves_search3.npz", **curves)
    Hh = pd.concat(Hs, ignore_index=True)
    Hh = Hh.merge(queries[["qid", "split", "genome_code", "tcode", "row"]], on="qid")
    # evidence for unlabelled hits with E_fast <= 10
    U = Hh[(Hh.label == 0) & (Hh.E_fast <= 10)].reset_index(drop=True)
    ev = evm.Evidence(db)
    F, _ = evm.classify(ev, U, queries, callers, rng, controls=False)
    U["supported"] = F.supported.to_numpy()
    U["df_own"] = F.df_own.to_numpy()
    Hh = Hh.merge(U[["qid", "fam", "start_gid", "end_gid", "supported", "df_own"]], on=["qid", "fam", "start_gid", "end_gid"],
                  how="left")
    Hh["unsupported"] = (Hh.label == 0) & (Hh.supported == False)  # noqa: E712
    Hh.to_parquet(OUT3 / "rates_hits.parquet", index=False)
    log(f"rates: {len(Hh):,} hits, {len(U):,} unlabelled with E_fast <= 10 classified, {time.time() - t0:.0f}s")
    Q = queries[(queries.caller == "DefenseFinder") & queries.qid.isin(nrel_ok)]
    q_even, q_odd = Q[Q.split == "even"], Q[Q.split == "odd"]

    def per_query(h, qs, thr, strict_lt=False):
        """Per query: AP over hits passing the threshold, recall, unlabelled / unsupported / wrong counts."""
        rows = []
        hq = {k: g for k, g in h.groupby("qid")}
        for qid, gc in zip(qs.qid, qs.genome_code):
            g = hq.get(int(qid))
            rr, rv = rel[int(qid)]
            if g is None or not len(g):
                rows.append(dict(qid=int(qid), genome_code=int(gc), ap=0.0, recall=0.0, unl=0, unsup=0, wrong=0))
                continue
            m = (g.E_fast < thr) if strict_lt else (g.E_fast <= thr)
            g = g[m.to_numpy()]
            r = ls.evaluate(g.score.to_numpy(), g.label.to_numpy().astype(np.int8), g.match_row.to_numpy().astype(np.int64),
                            None, rr, rv)
            rows.append(dict(qid=int(qid), genome_code=int(gc), ap=r["ap"], recall=r["recall_any"],
                             unl=int((g.label == 0).sum()), unsup=int(g.unsupported.sum()), wrong=int((g.label == 2).sum())))
        return pd.DataFrame(rows)

    PQ = {}
    for fam in RATE_FAMS:
        h = Hh[Hh.fam == fam]
        base = pick_rows(M[M.split == "even"], sel, fam, "odd")
        e001 = per_query(h, q_even, 0.01)
        e1 = per_query(h, q_even, 1.0)
        res["at_E0_01"][fam] = dict(ap_full=boot(base.ap, base.genome_code), ap_strict_full=boot(base.ap_strict, base.genome_code),
                                    recall_E0_01=boot(e001.recall, e001.genome_code), recall_E1=boot(e1.recall, e1.genome_code),
                                    unlabelled_per_query=boot(e001.unl, e001.genome_code),
                                    unsupported_per_query=boot(e001.unsup, e001.genome_code),
                                    wrong_per_query=boot(e001.wrong, e001.genome_code), queries=int(len(e001)))
        # matched unsupported rate: threshold from odd-genome queries
        uo = np.sort(h[(h.split == "odd") & h.unsupported & (h.E_fast <= 10)].E_fast.to_numpy())
        allowed = int(math.floor(0.03 * len(q_odd) + 1e-9))
        if len(uo) <= allowed:
            thr, lt, reached = 10.0, False, False
        else:
            thr, lt, reached = float(uo[allowed]), True, True
        odd_rate = float((uo < thr).sum() / len(q_odd)) if lt else float(len(uo) / len(q_odd))
        pm = per_query(h, q_even, thr, lt)
        PQ[fam] = pm
        res["matched"][fam] = dict(threshold_E_fast=thr, threshold_is_strict_upper=lt, rate_reached_below_E10=reached,
                                   odd_unsupported_per_query=odd_rate, recall=boot(pm.recall, pm.genome_code),
                                   ap=boot(pm.ap, pm.genome_code), unsupported_per_query=boot(pm.unsup, pm.genome_code),
                                   unlabelled_per_query=boot(pm.unl, pm.genome_code), wrong_per_query=boot(pm.wrong, pm.genome_code))
        PQ[fam + "@0.01"] = e001
        log(fam, "E<=0.01", {k: (round(v[0], 3) if isinstance(v, tuple) else v) for k, v in res["at_E0_01"][fam].items()},
            "matched", round(thr, 4), round(res["matched"][fam]["recall"][0], 3), round(res["matched"][fam]["ap"][0], 3))
    for x, y in (("M1z", "S1"), ("M1z", "S3"), ("M1z", "M0"), ("M1p", "M1z"), ("M2z", "M1z"), ("SM1z", "M1z"), ("S3z", "M1z"),
                 ("S3z", "S3")):
        a_, b_ = PQ[x].set_index("qid"), PQ[y].set_index("qid")
        ids = a_.index.intersection(b_.index)
        g = b_.loc[ids, "genome_code"].to_numpy()
        res["paired_matched"][f"{x}-{y}"] = {c: boot(a_.loc[ids, c] - b_.loc[ids, c], g) for c in ("recall", "ap", "unsup", "wrong")}
        a_, b_ = PQ[x + "@0.01"].set_index("qid"), PQ[y + "@0.01"].set_index("qid")
        res["paired_E0_01"][f"{x}-{y}"] = {c: boot(a_.loc[ids, c] - b_.loc[ids, c], g) for c in ("recall", "unl", "unsup", "wrong")}
    res["seconds"] = round(time.time() - t0, 1)
    dump(res, OUT3 / "rates3.json")
    log("rates done", res["seconds"])


# =========================================================================== POST HOC (not predeclared)
PH = {}
PH_KIND = {"wrong": lambda h: h.label == 2, "unlabelled": lambda h: h.label == 0, "unsupported": lambda h: h.unsupported}


def _posthoc_fam(task):
    fam, kind = task
    Hh, C, qs, nq, rel, levels = (PH[k] for k in ("Hh", "C", "qs", "nq", "rel", "levels"))
    h = Hh[Hh.fam == fam]
    cq = {k: g.Emin.to_numpy() for k, g in C[C.fam == fam].groupby("qid")}
    rec, out = {}, {}
    for lv in (levels[kind],):
        xo = np.sort(h[(h.split == "odd") & PH_KIND[kind](h)].E_fast.to_numpy())
        for L in lv:
            allowed = int(math.floor(L * nq["odd"] + 1e-9))
            strict = len(xo) > allowed
            thr = float(xo[allowed]) if strict else 10.0
            he = h[(h.split == "even") & ((h.E_fast < thr) if strict else (h.E_fast <= thr))]
            rows = []
            for qid, gc in zip(qs["even"].qid, qs["even"].genome_code):
                e = cq.get(int(qid), np.empty(0))
                found = int(((e < thr) if strict else (e <= thr)).sum())
                rows.append((int(qid), int(gc), found / len(rel[int(qid)][0])))
            R = pd.DataFrame(rows, columns=["qid", "genome_code", "recall"])
            rec[(fam, kind, L)] = R
            out[f"{fam}|{kind}|{L:g}"] = dict(threshold=thr, reached=strict, recall=boot(R.recall, R.genome_code),
                                              even_rate=float(PH_KIND[kind](he).sum() / nq["even"]),
                                              even_wrong=float((he.label == 2).sum() / nq["even"]),
                                              even_unl=float((he.label == 0).sum() / nq["even"]),
                                              even_unsup=float(he.unsupported.sum() / nq["even"]))
    return fam, rec, out


def posthoc(a):
    """POST HOC, written after the task-3 matched-rate result (M1z below S1 / S3 at 0.03 unsupported hits per query).
    (i) recall at matched rates of three false-positive kinds, thresholds on odd genomes, recall on even genomes:
        wrong-type hits (curated negatives), all unlabelled hits, unsupported unlabelled hits (as task 3);
    (ii) what the unlabelled hits at fast E <= 0.01 are: concentration over queries, raw DefenseFinder profiles of any
        type, and sequence support (a window gene in the MMseqs2 list, E <= 10, of a query member gene)."""
    t0 = time.time()
    queries = pd.read_parquet(OUT1 / "queries.parquet")
    db = ls.Database(vectors=None)
    callers = bench.load_gold(db)
    rel = bench.load_relevance(OUT1, queries, callers)
    Hh = pd.read_parquet(OUT3 / "rates_hits.parquet")
    M = load_final("df")
    ok = set(M[M.n_rel > 0].qid)
    Q = queries[(queries.caller == "DefenseFinder") & queries.qid.isin(ok)]
    qs = {s: Q[Q.split == s] for s in ("odd", "even")}
    nq = {s: len(v) for s, v in qs.items()}
    res = dict(note="POST HOC; not predeclared", queries=nq, matched={}, paired={}, unlabelled_E001={})
    levels = {"wrong": (0.25, 0.5, 1.0), "unlabelled": (1.0, 2.0, 4.0), "unsupported": (0.01, 0.03, 0.1)}
    # first E at which each relevant target is found (correct hits only; E is monotone in score within a run)
    C = Hh[Hh.label == 1].groupby(["fam", "qid", "match_row"]).E_fast.min().rename("Emin").reset_index()
    PH.update(Hh=Hh, C=C, qs=qs, nq=nq, rel=rel, levels=levels)
    rec = {}
    import multiprocessing as mp
    tasks = [(f, k) for f in RATE_FAMS for k in levels]          # 24 tasks: one worker each on 24 CPUs
    with mp.get_context("fork").Pool(a.threads) as pool:
        for fam, r_, m_ in pool.imap_unordered(_posthoc_fam, tasks):
            rec.update(r_)
            res["matched"].update(m_)
    for kind, lv in levels.items():
        for L in lv:
            for x, y in (("M1z", "S1"), ("M1z", "S3"), ("M1z", "M0"), ("M2z", "M1z"), ("SM1z", "M1z"), ("S3z", "M1z"), ("M1p", "M1z")):
                A, B = rec[(x, kind, L)], rec[(y, kind, L)]
                res["paired"][f"{x}-{y}|{kind}|{L:g}"] = boot(A.recall.to_numpy() - B.recall.to_numpy(), B.genome_code)
    # (ii) unlabelled hits at fast E <= 0.01
    import evidence as evm
    z = np.load(evm.EVD / "evidence.npz")
    df_any = z["df_any"]
    g0 = np.load(OUT1 / "topk/gids.npy")
    row_of = {int(g): k for k, g in enumerate(g0)}
    si = np.load(SRC_DIR["seq"] / "idx.npy", mmap_mode="r")
    zi = np.load(SRC_DIR["zca10"] / "idx.npy", mmap_mode="r")
    zs_ = np.load(SRC_DIR["zca10"] / "s.npy", mmap_mode="r")
    Cg = callers["DefenseFinder"]
    qrow = queries.set_index("qid")
    seqset, zrank = {}, {}
    U = Hh[(Hh.label == 0) & (Hh.E_fast <= 0.01)].copy()
    for qid in U.qid.unique():
        mem = Cg["mem"][int(qrow.row[qid])]
        rr = [row_of[int(g)] for g in mem]
        s = np.concatenate([si[r][si[r] >= 0] for r in rr]).astype(np.int64)
        seqset[int(qid)] = np.unique(s)
        zz = np.concatenate([zi[r] for r in rr]).astype(np.int64)
        sv = np.concatenate([zs_[r] for r in rr])
        o = np.lexsort((-sv, zz))
        zz, sv = zz[o], sv[o]
        first = np.r_[True, zz[1:] != zz[:-1]]
        zrank[int(qid)] = (zz[first], sv[first])
    seq_sup, best_s = [], []
    for qid, x, y in zip(U.qid, U.start_gid, U.end_gid):
        s = seqset[int(qid)]
        seq_sup.append(bool(np.searchsorted(s, y, "right") - np.searchsorted(s, x, "left") > 0))
        zz, sv = zrank[int(qid)]
        i0, i1 = np.searchsorted(zz, x, "left"), np.searchsorted(zz, y, "right")
        best_s.append(float(sv[i0:i1].max()) if i1 > i0 else 0.0)
    U["seq_support"] = seq_sup
    U["zca10_best_s"] = best_s
    U["df_any"] = np.searchsorted(df_any, U.end_gid.to_numpy(), "right") - np.searchsorted(df_any, U.start_gid.to_numpy(), "left") > 0
    U = U.merge(queries[["qid", "type"]], on="qid")
    U.to_parquet(OUT3 / "posthoc_unlabelled_E001.parquet", index=False)
    for fam, g in U.groupby("fam"):
        d = {}
        for sp in ("even", "all"):
            gg = g if sp == "all" else g[g.split == "even"]
            un = gg[gg.unsupported]
            d[sp] = dict(unlabelled=int(len(gg)), unsupported=int(len(un)),
                         unsupported_queries=int(un.qid.nunique()),
                         unsupported_top5_query_share=float(un.qid.value_counts().head(5).sum() / max(len(un), 1)),
                         unsupported_types=un.type.value_counts().head(6).to_dict(),
                         unsupported_seq_support=float(un.seq_support.mean()) if len(un) else None,
                         supported_seq_support=float(gg[~gg.unsupported].seq_support.mean()) if len(gg) else None,
                         unsupported_df_any=float(un.df_any.mean()) if len(un) else None,
                         unsupported_slots_q25_50_75=np.quantile(un.n_slots_filled, [.25, .5, .75]).tolist() if len(un) else None,
                         supported_slots_median=float(gg[~gg.unsupported].n_slots_filled.median()) if len(gg) else None,
                         unsupported_zca10_best_s_median=float(un.zca10_best_s.median()) if len(un) else None,
                         supported_zca10_best_s_median=float(gg[~gg.unsupported].zca10_best_s.median()) if len(gg) else None)
        res["unlabelled_E001"][fam] = d
    res["seconds"] = round(time.time() - t0, 1)
    dump(res, OUT3 / "posthoc3.json")
    log("posthoc", json.dumps({k: v["recall"][0] for k, v in res["matched"].items() if k.startswith(("M1z", "S1", "S3|"))}))


def _ph2_task(task):
    key, kind = task
    G, ok_odd, ok_even, rel, levels = (PH2[k] for k in ("G", "ok_odd", "ok_even", "rel", "levels"))
    h = G[key]
    cmin = h[h.label == 1].groupby(["qid", "match_row"]).E_fast.min().rename("Emin").reset_index()
    cq = {k: g.Emin.to_numpy() for k, g in cmin.groupby("qid")}
    out = {}
    xo = np.sort(h[(h.split == "odd") & PH_KIND[kind](h)].E_fast.to_numpy())
    for L in levels[kind]:
        allowed = int(math.floor(L * len(ok_odd) + 1e-9))
        strict = len(xo) > allowed
        thr = float(xo[allowed]) if strict else 10.0
        for sp, qq in (("odd", ok_odd), ("even", ok_even)):
            rows = []
            for qid, gc in zip(qq.qid, qq.genome_code):
                e = cq.get(int(qid), np.empty(0))
                rows.append((int(qid), int(gc), int(((e < thr) if strict else (e <= thr)).sum()) / len(rel[int(qid)][0])))
            out[(key, kind, L, sp)] = (thr, pd.DataFrame(rows, columns=["qid", "genome_code", "recall"]))
    return out


PH2 = {}


def posthoc2(a):
    """POST HOC: matched-rate recall of M1z and S1 at every tau of the grid (W + 8); per method the tau with the best
    odd-genome recall at each rate level is carried to even genomes (both methods get the same freedom)."""
    import evidence as evm
    import multiprocessing as mp
    t0 = time.time()
    sel = json.load(open(OUT3 / "selection3.json"))
    queries = pd.read_parquet(OUT1 / "queries.parquet")
    db = evm.attach_db()
    callers = bench.load_gold(db)
    rel = bench.load_relevance(OUT1, queries, callers)
    M = load_final("df")
    ok = set(M[M.n_rel > 0].qid)
    Q = queries[(queries.caller == "DefenseFinder") & queries.qid.isin(ok)]
    ok_odd, ok_even = Q[Q.split == "odd"], Q[Q.split == "even"]
    zs = [np.load(p) for p in sorted(OUT3.glob("final_df*_tails.npz")) + sorted(OUT3.glob("final_tau*_tails.npz"))]
    T = pd.DataFrame({k: np.concatenate([z[k] for z in zs]) for k in ("qid", "fam", "w", "wx", "tau")})
    tails = np.concatenate([z["tails"] for z in zs])
    T["split"] = T.qid.map(dict(zip(queries.qid, queries.split)))
    Hall = pd.concat([pd.read_parquet(p, filters=[("fam", "in", ["M1z", "S1"])])
                      for p in sorted(OUT3.glob("final_df*_hits.parquet")) + sorted(OUT3.glob("final_tau*_hits.parquet"))],
                     ignore_index=True)
    Hall = Hall[(Hall.wx == 8) & Hall.qid.isin(ok)].merge(queries[["qid", "split", "genome_code", "tcode", "row"]], on="qid")
    G, res = {}, dict(note="POST HOC; not predeclared", curves_ratio_at_E0_01={}, per_tau={}, chosen={}, paired={})
    for fam in ("M1z", "S1"):
        for tau in (0.0, 1.0, 2.0, 3.0):
            key = f"{fam}_t{tau:g}"
            isf = ((T.fam == fam) & (T.wx == 8) & (T.tau == tau)).to_numpy()
            odd_i = np.flatnonzero(isf & (T.split == "odd").to_numpy())
            curve = fastnull.calibration_curve(np.array([null_counts(tails[i]) for i in odd_i]))
            res["curves_ratio_at_E0_01"][key] = float(curve[fastnull.J001] / XG[fastnull.J001])
            h = Hall[(Hall.fam == fam) & (Hall.tau == tau)].copy()
            h["E_fast"] = apply_curve_ext(h.Egum40.to_numpy(), curve)
            G[key] = h
    allh = pd.concat([g.assign(key=k) for k, g in G.items()], ignore_index=True)
    U = allh[(allh.label == 0) & (allh.E_fast <= 10)].reset_index(drop=True)
    F, _ = evm.classify(evm.Evidence(db), U, queries, callers, np.random.default_rng(SEED), controls=False)
    U["supported"] = F.supported.to_numpy()
    allh = allh.merge(U[["key", "qid", "start_gid", "end_gid", "supported"]], on=["key", "qid", "start_gid", "end_gid"], how="left")
    allh["unsupported"] = (allh.label == 0) & (allh.supported == False)  # noqa: E712
    G = {k: g for k, g in allh.groupby("key")}
    levels = {"wrong": (0.25, 0.5, 1.0), "unlabelled": (1.0, 2.0, 4.0), "unsupported": (0.01, 0.03, 0.1)}
    PH2.update(G=G, ok_odd=ok_odd, ok_even=ok_even, rel=rel, levels=levels)
    out = {}
    with mp.get_context("fork").Pool(a.threads) as pool:
        for o in pool.imap_unordered(_ph2_task, [(k, kind) for k in G for kind in levels]):
            out.update(o)
    for kind, lv in levels.items():
        for L in lv:
            best = {}
            for fam in ("M1z", "S1"):
                cand = {}
                for tau in (0.0, 1.0, 2.0, 3.0):
                    key = f"{fam}_t{tau:g}"
                    thr, Ro = out[(key, kind, L, "odd")]
                    _, Re = out[(key, kind, L, "even")]
                    cand[tau] = (Ro.recall.mean(), Re)
                    res["per_tau"][f"{key}|{kind}|{L:g}"] = dict(threshold=thr, odd_recall=float(Ro.recall.mean()),
                                                                 even_recall=float(Re.recall.mean()))
                tb = max(cand, key=lambda t: (cand[t][0], t))
                best[fam] = (tb, cand[tb][1])
                res["chosen"][f"{fam}|{kind}|{L:g}"] = dict(tau=tb, even_recall=boot(cand[tb][1].recall, cand[tb][1].genome_code))
            A, B = best["M1z"][1], best["S1"][1]
            res["paired"][f"M1z-S1|{kind}|{L:g}"] = boot(A.recall.to_numpy() - B.recall.to_numpy(), B.genome_code)
    res["seconds"] = round(time.time() - t0, 1)
    dump(res, OUT3 / "posthoc3_tau.json")
    log("posthoc2", json.dumps({k: v[0] for k, v in res["paired"].items()}))


# =========================================================================== phase 0 (task 4a)
def phase0(a):
    import multiprocessing as mp
    import phase0_control as p0c
    t0 = time.time()
    db, callers, queries = bench.setup(a, vectors=None)
    db.X = np.load(ZROW)
    g0 = np.load(OUT1 / "topk/gids.npy")
    d = SRC_DIR["zca10"]
    zi, zs = np.load(d / "idx.npy"), np.load(d / "s.npy")
    qs, npairs = bench.random_pair_quantiles(db)
    cstar = qs["0.999"]
    rank_s = math.log(1000.0)
    log("zca10 random-pair cosine quantiles", qs)
    pairs = bench.relevance_and_headroom(db, callers, (g0, zi, zs), cstar, rank_s)
    pairs.to_parquet(OUT3 / "pairs_zca10.parquet", index=False)
    old = pd.read_parquet(OUT1 / "pairs.parquet", columns=["caller", "q_row", "t_row", "f_tok"])
    same_rel = bool(len(old) == len(pairs) and (old.q_row.to_numpy() == pairs.q_row.to_numpy()).all()
                    and np.allclose(old.f_tok.to_numpy(), pairs.f_tok.to_numpy()))
    hs = bench.headroom_summary(pairs, queries)
    hs.update(random_pair_cosine_quantiles=qs, random_pairs=npairs, cos_threshold_0999=cstar, same_pairs_and_f_tok_as_round1=same_rel)
    # different-type control (phase0_control.py logic with zca10 vectors and lists)
    top = [np.sort(zi[k][zs[k] >= rank_s].astype(np.int64)) for k in range(len(g0))]
    p0c.G.update(tok=db.tok, top=top, Xm=np.asarray(db.X[g0], dtype=np.float32), row={int(g): k for k, g in enumerate(g0)},
                 cstar=cstar)
    rng = np.random.default_rng(SEED)
    ctl = {}
    for c in bench.CALLERS:
        C = callers[c]
        p0c.G["C"] = C
        tr = np.flatnonzero(C["target"])
        q = rng.choice(tr, 200000 * 3)
        t = rng.choice(tr, 200000 * 3)
        ok = (C["tcode"][q] != C["tcode"][t]) & (C["gc"][q] != C["gc"][t])
        pl = list(zip(q[ok][:200000].tolist(), t[ok][:200000].tolist()))
        chunks = [pl[i:i + 2000] for i in range(0, len(pl), 2000)]
        if not pl:
            ctl[c] = dict(pairs=0)
            continue
        with mp.get_context("fork").Pool(a.threads) as pool:
            v = np.array([x for ch in pool.map(p0c.work, chunks) for x in ch])
        ft, fc, fr, fe = v[:, 0], v[:, 1], v[:, 2], v[:, 3]
        ctl[c] = dict(pairs=int(len(v)), tok_visible=float((ft >= 0.5).mean()), f_cos_visible=float((fc >= 0.5).mean()),
                      f_rank_visible=float((fr >= 0.5).mean()), emb_visible=float((fe >= 0.5).mean()),
                      tok_invisible_and_emb_visible=float(((ft < 0.5) & (fe >= 0.5)).mean()),
                      union_visible=float(((ft >= 0.5) | (fe >= 0.5)).mean()))
        log("control", c, ctl[c])
    hs["control_different_type"] = ctl
    hs["round1_l2"] = dict(headroom=json.load(open(OUT1 / "phase0_headroom.json")), control=json.load(open(OUT1 / "phase0_control.json")))
    hs["seconds"] = round(time.time() - t0, 1)
    dump(hs, OUT3 / "phase0_zca10.json")
    for c in bench.CALLERS:
        if c in hs:
            log(f"phase0 zca10 {c}: tok {hs[c]['f_tok']['frac_ge_half']:.3f} emb {hs[c]['f_emb']['frac_ge_half']:.3f} "
                f"inv&emb {hs[c]['token_invisible_and_f_emb_ge_half']:.3f}")


# =========================================================================== M3 with zca10 centroids (task 4c)
def m3(a):
    import evidence as evm
    t0 = time.time()
    sel = json.load(open(OUT3 / "selection3.json"))
    db, callers, queries = bench.setup(a, vectors=None)
    bench.CTX["rel"] = bench.load_relevance(OUT1, queries, callers)
    db.X = np.load(ZROW)
    g0 = np.load(OUT1 / "topk/gids.npy")
    d = SRC_DIR["zca10"]
    bench.CTX.update(row_of={int(g): k for k, g in enumerate(g0)}, tk_idx=np.load(d / "idx.npy"), tk_s=np.load(d / "s.npy"))
    bench.warm_up(db)
    c = sel["odd"]["M2z"]
    Q = queries[(queries.caller == "DefenseFinder") & (queries.split == "even")]
    tasks = [(int(q), "M2", "comb", c["w"], c["wx"], c["tau"]) for q in Q.qid]
    tasks.sort(key=lambda t: -queries.n_members[t[0]])
    res = bench.pool_map(bench.final_task, tasks, a.threads, chunksize=1)
    log(f"M3z seed (M2z): {len(tasks):,} runs, {time.time() - t0:.0f}s")
    metrics = [r for r, _, _ in res]
    hits = [h for _, h, _ in res]
    profiles = {}
    for r, _, p in res:
        for v, pp in p.items():
            profiles[(r["qid"], v)] = pp
    del res
    info = {}
    for rnd in (2, 3):
        t1 = time.time()
        cent_row, cidx, cs = bench.centroid_topk(db, profiles, queries)
        bench.CTX.update(cent_row=cent_row, cent_idx=cidx, cent_s=cs)
        info[f"round{rnd}_centroid_vectors"] = int(len(cidx))
        rt = [(int(qid), v, rnd, p, c["w"], c["tau"]) for (qid, v), p in profiles.items()]
        rr = bench.pool_map(bench.round_task, rt, a.threads)
        profiles = {}
        for key, r, h, nxt in rr:
            metrics.append(r)
            hits.append(h)
            profiles[key] = nxt
        del rr
        info[f"round{rnd}_seconds"] = round(time.time() - t1, 1)
        log(f"M3z round {rnd}: {len(rt):,} runs, {time.time() - t1:.0f}s")
    M = pd.DataFrame(metrics).merge(queries, on="qid")
    M["method"] = M.method.replace({"M2": "M2z"})
    M.to_parquet(OUT3 / "m3z_metrics.parquet", index=False)
    Hf = bench.hits_frame(hits)
    Hf["method"] = Hf.method.replace({"M2": "M2z"})
    Hf.to_parquet(OUT3 / "m3z_hits_E10.parquet", index=False)
    # unsupported unlabelled hits at E200 <= 0.01 (round-2 rule)
    U = Hf[(Hf.evalue <= 0.01) & (Hf.label == 0)].reset_index(drop=True)
    ev = evm.Evidence(evm.attach_db())
    F, _ = evm.classify(ev, U, queries, callers, np.random.default_rng(SEED), controls=False)
    U["supported"] = F.supported.to_numpy()
    uns = U[~U.supported].groupby(["qid", "method", "round"]).size().rename("n_unsupported_E0.01").reset_index()
    M = M.merge(uns, on=["qid", "method", "round"], how="left").fillna({"n_unsupported_E0.01": 0})
    M = M[M.n_rel > 0]
    M["mkey"] = M.method + "_r" + M["round"].astype(str)
    out = dict(seed_config=c, info=info, tables={}, paired_vs_seed={})
    cols = ["ap", "ap_strict", "recall_E0.01", "recall_E1", "n_wrong_E0.01", "n_unlabelled_E0.01", "n_unsupported_E0.01",
            "precision_E0.01", "precision_strict_E0.01"]
    seed = M[M.mkey == "M2z_r1"]
    for mk, g in M.groupby("mkey"):
        out["tables"][mk] = {k: boot(g[k], g.genome_code) for k in cols} | {"queries": int(len(g))} | \
            {"micro_recall_inv_E0.01": float(g["found_inv_E0.01"].sum() / max(g.n_rel_inv.sum(), 1))}
        if mk != "M2z_r1":
            out["paired_vs_seed"][mk] = paired(g, seed, ["ap", "recall_E0.01", "n_wrong_E0.01", "n_unlabelled_E0.01"])
    w1 = out["tables"]["M2z_r1"]["n_wrong_E0.01"][0]
    out["drift"] = {mk: dict(wrong_ratio_vs_seed=float(v["n_wrong_E0.01"][0] / w1) if w1 > 0 else None,
                             drifts=bool(w1 > 0 and v["n_wrong_E0.01"][0] >= 2 * w1))
                    for mk, v in out["tables"].items() if mk.endswith("_r3")}
    # round-1 raw-vector M3 for reference (same queries, 200-replicate null)
    r1 = pd.read_parquet(OUT1 / "final_metrics_dfr1.parquet")
    r1 = r1[(r1.caller == "DefenseFinder") & (r1.split == "even") & (r1.n_rel > 0)]
    r1["mkey"] = r1.method + "_r" + r1["round"].astype(str)
    out["round1_raw"] = {mk: {k: float(g[k].mean()) for k in ("ap", "recall_E0.01", "n_wrong_E0.01", "n_unlabelled_E0.01")}
                         for mk, g in r1.groupby("mkey")}
    out["seconds"] = round(time.time() - t0, 1)
    dump(out, OUT3 / "m3z_summary.json")
    log("M3z", json.dumps({k: {c_: round(v[c_][0], 3) for c_ in ("ap", "recall_E0.01", "n_wrong_E0.01")} for k, v in out["tables"].items()}))


# =========================================================================== HNSW on zca10 (task 5)
SWEEP = [(2000, 2200), (5000, 5200), (10000, 10200), (20000, 20200), (20000, 40000)]


def ap_task(args):
    qid, fam, w, wx, tau = args
    use_tok, ws, src, we = cfg_of(fam, w)
    use(src)
    q = bench.CTX["queries"].loc[qid]
    db = bench.CTX["db"]
    (eg, es_, ev), ns = round2.entries(q, use_tok, ws, we)
    ha, hb, sc, _ = ls.scan(eg, es_, ev, ns, db.cc, db.pos, bench.W_of(q, wx), tau, 0)
    r, _, _ = bench.label_eval(q, ha, hb, sc, None)
    return int(qid), r["ap"], r["ap_strict"], r["n_rel"]


def hnsw(a):
    import faiss
    import interactive as it
    faiss.omp_set_num_threads(a.threads)
    IDIR3.mkdir(parents=True, exist_ok=True)
    sel = json.load(open(OUT3 / "selection3.json"))
    c = sel["odd"]["M1z"]
    db, callers, queries = setup(a, ["zca10"])
    res = json.load(open(IDIR3 / "hnsw_sweep.json")) if (IDIR3 / "hnsw_sweep.json").exists() else {}
    t0 = time.time()
    Xz = np.load(ZROW)
    res["vector_load_seconds"] = round(time.time() - t0, 1)
    if HNSW_Z.exists():
        t0 = time.time()
        index = faiss.read_index(str(HNSW_Z))
        res["index_load_seconds"] = round(time.time() - t0, 1)
    else:
        t0 = time.time()
        index = faiss.IndexHNSWFlat(Xz.shape[1], 32, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = 40
        index.add(Xz)
        res["build_seconds_64cpu"] = round(time.time() - t0, 1)
        faiss.write_index(index, str(HNSW_Z))
        res["index_file_bytes"] = HNSW_Z.stat().st_size
        log("HNSW built", res["build_seconds_64cpu"], "s", res["index_file_bytes"] / 1e9, "GB")
    dump(res, IDIR3 / "hnsw_sweep.json")
    Q = queries[(queries.caller == "DefenseFinder") & (queries.split == "even")]
    qg = np.unique(np.concatenate([callers["DefenseFinder"]["mem"][r] for r in Q.row]))
    own = db.gc[qg]
    lo, hi = db.g_lo[own], db.g_hi[own]
    n_db = np.array([db.n_db(int(g)) for g in own], np.float64)
    zi, zsc = SRCS["zca10"]
    rows = np.array([EV["row_of"][int(g)] for g in qg])
    task_args = [(int(q), "Hnsw", 1.0, c["wx"], c["tau"]) for q in Q.qid]
    # exact reference: M1z at the odd-genome configuration with exact lists truncated to k
    ap_rows = []
    for kk in sorted({k for k, _ in SWEEP}):
        SRCS["hnsw"] = (zi[:, :kk], zsc[:, :kk])
        for qid, ap, aps, nrel in bench.pool_map(ap_task, task_args, a.threads, chunksize=4):
            ap_rows.append(dict(variant=f"exact_k{kk}", qid=qid, ap=ap, ap_strict=aps, n_rel=nrel))
    for k, ef in SWEEP:
        index.hnsw.efSearch = ef
        t1 = time.time()
        hidx, hs = it.hnsw_ranks(index, Xz[qg], qg, lo, hi, n_db, k=k)
        secs = time.time() - t1
        ex = zi[rows, :k]
        rec = np.array([len(np.intersect1d(ex[r][ex[r] >= 0], hidx[r][hidx[r] >= 0])) / max((ex[r] >= 0).sum(), 1)
                        for r in range(len(qg))])
        full_i = np.full((len(zi), k), -1, np.int32)
        full_s = np.zeros((len(zi), k), np.float32)
        full_i[rows], full_s[rows] = hidx, hs
        SRCS["hnsw"] = (full_i, full_s)
        for qid, ap, aps, nrel in bench.pool_map(ap_task, task_args, a.threads, chunksize=4):
            ap_rows.append(dict(variant=f"hnsw_k{k}_ef{ef}", qid=qid, ap=ap, ap_strict=aps, n_rel=nrel))
        res[f"hnsw_k{k}_ef{ef}"] = dict(genes=int(len(qg)), seconds_64cpu=round(secs, 1), genes_per_second_64cpu=len(qg) / secs,
                                        recall_vs_exact_mean=float(rec.mean()),
                                        recall_vs_exact_q10_50_90=np.quantile(rec, [.1, .5, .9]).tolist())
        log("HNSW", k, ef, res[f"hnsw_k{k}_ef{ef}"])
        del full_i, full_s
        dump(res, IDIR3 / "hnsw_sweep.json")
    A = pd.DataFrame(ap_rows).merge(queries[["qid", "genome_code"]], on="qid")
    A = A[A.n_rel > 0]
    A.to_parquet(IDIR3 / "hnsw_ap.parquet", index=False)
    base = A[A.variant == "exact_k20000"].set_index("qid")
    cmp = {}
    for v, g in A.groupby("variant"):
        g = g.set_index("qid")
        ids = g.index.intersection(base.index)
        cmp[v] = dict(map=float(g.loc[ids, "ap"].mean()),
                      delta_vs_exact20000=boot(g.loc[ids, "ap"] - base.loc[ids, "ap"], base.loc[ids, "genome_code"]))
    res["ap"] = cmp
    ok = [(k, ef) for k, ef in SWEEP if cmp[f"hnsw_k{k}_ef{ef}"]["delta_vs_exact20000"][0] >= -0.01]
    res["chosen"] = dict(k=ok[0][0], efSearch=ok[0][1]) if ok else None
    res["M1z_config"] = c
    dump(res, IDIR3 / "hnsw_sweep.json")
    log("HNSW AP", json.dumps({k: round(v["delta_vs_exact20000"][0], 4) for k, v in cmp.items()}), "chosen", res["chosen"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["lists", "grid", "select", "final", "analyse", "rates", "phase0", "m3", "hnsw", "posthoc",
                                      "posthoc2"])
    ap.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_ON_NODE", "2")))
    ap.add_argument("--set", default="main")
    ap.add_argument("--part", default="")
    a = ap.parse_args()
    import numba
    numba.set_num_threads(a.threads)
    t0 = time.time()
    {"lists": lists, "grid": grid, "select": select, "final": final, "analyse": analyse, "rates": rates, "phase0": phase0,
     "m3": m3, "hnsw": hnsw, "posthoc": posthoc, "posthoc2": posthoc2}[a.stage](a)
    log(f"done {a.stage} {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
