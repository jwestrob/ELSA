#!/usr/bin/env python
"""Locus-search benchmark on the DPANN gold loci.

  bench.py stage1 --out DIR --threads T   top-K for member genes, phase 0 headroom, tuning (odd genomes)
  bench.py stage2 --out DIR --threads T   final M0/M1/M2 with null E-values, M3 rounds, summaries

Queries: gold instances with >= 2 members (DefenseFinder primary; TXSScan and CRISPRCasTyper
secondary). Type label: system_type (DefenseFinder, TXSScan), system_subtype (CRISPRCasTyper,
whose system_type is mostly the string "False"). Targets: instances of the same caller and type with
>= 2 members in other genomes. Database: every gene of every other genome.

Hit labels: correct = covers >= 50% of the members of a target instance; wrong = otherwise overlaps
a member of a different-type instance of the caller; same-type other = overlaps a same-type instance
below 50% or a single-member one; unlabelled = no overlap with the caller's instances.
Primary AP ignores same-type-other, unlabelled and duplicate hits; strict AP counts unlabelled as wrong.
Split: tune = odd genome_code, test = even genome_code.
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import locus_search as ls  # noqa: E402
from locus_search import log  # noqa: E402

PREDECLARED = dict(
    written="2026-11-11, before any locus-search number was computed",
    criterion=("Embeddings count as a gain for locus search if, on DefenseFinder queries with >= 2 members from "
               "even genome_code genomes, the best of M1/M2/M3 exceeds M0 in mean primary AP by >= 0.02, with a "
               "paired genome-bootstrap 95% interval (query genomes resampled, 2,000 draws, seed 17) excluding 0, "
               "at an unlabelled-hit rate (unlabelled hits per query at E <= 0.01) at most 1.5x that of M0."),
    details=["Best of M1/M2/M3 = highest mean primary AP on odd-genome DefenseFinder queries; that method carries "
             "the criterion on the even half. All three are reported; if picking the best on the even half would "
             "change the verdict, the report says so.",
             "All settings (window, per-slot floor tau, combination weight, M3 variant and round) are chosen by mean "
             "primary AP on odd-genome queries of the same caller.",
             "If M0's unlabelled rate at E <= 0.01 is below 0.1 per query, the rate condition becomes an absolute "
             "bound of 0.15 per query.",
             "Reported separately: recall at E <= 0.01 and E <= 1 on token-invisible targets (< 50% of the target's "
             "members share an e3c50 token with a query member), paired against M0."],
    refutation=("(i) criterion fails and (ii) token-invisible recall of the selected embedding method is not above "
                "M0's (paired interval includes or lies below 0): embeddings add nothing to locus search here."),
)

CALLERS = ("DefenseFinder", "TXSScan", "CRISPRCasTyper")
K = int(os.environ.get("LS_K", 20000))          # override for synthetic tests only
W_CAP = 60
GRID_WX = (0, 2, 4, 8)
GRID_TAU = (0.0, 1.0, 2.0, 3.0)
GRID_W = (0.5, 0.75, 1.0)
R_NULL = int(os.environ.get("LS_R", 200))       # override for synthetic tests only
TOP_N = 128
SEED = 17
M3_ECONF = 0.01
M3_MAXHITS = 200
M3_VARIANTS = {"refine": 0, "grow2": 2}       # flank (genes) searched for recurring new slots; 0 = none
M3_NEW_MIN_FRAC = 0.5
M3_NEW_MAX = 10

CTX = {}


# =========================================================================== data
def load_gold(db):
    inst = pd.read_parquet(ls.PKG / "gold/instances.parquet")
    gm = pd.read_parquet(ls.PKG / "universe/genomes.parquet", columns=["genome", "genome_code"])
    inst["genome_code"] = inst.genome.map(dict(zip(gm.genome, gm.genome_code))).astype(np.int64)
    inst["label"] = np.where(inst.caller.eq("CRISPRCasTyper"), inst.system_subtype, inst.system_type).astype(str)
    callers = {}
    for c in CALLERS:
        I = inst[inst.caller == c].reset_index(drop=True)
        tcode, tlabels = pd.factorize(I.label)
        mem = [np.unique(np.asarray(m, np.int64)) for m in I.member_gids]
        nmem = np.array([len(m) for m in mem], np.int64)
        gl = np.concatenate(mem)
        tl = np.repeat(np.arange(len(I)), nmem)
        o = np.lexsort((tl, gl))
        gl, tl = gl[o], tl[o]
        callers[c] = dict(I=I, tcode=tcode.astype(np.int64), tlabels=np.asarray(tlabels), mem=mem, nmem=nmem,
                          target=nmem >= 2, gptr=np.searchsorted(gl, np.arange(db.N + 1)).astype(np.int64),
                          ginst=tl.astype(np.int64), gc=I.genome_code.to_numpy(np.int64))
    return callers


def make_queries(db, callers):
    rows = []
    for c in CALLERS:
        C = callers[c]
        for r in np.flatnonzero(C["target"]):
            m = C["mem"][r]
            span = int(db.pos[m].max() - db.pos[m].min() + 1)
            rows.append(dict(qid=len(rows), caller=c, row=int(r), instance_id=int(C["I"].instance_id[r]),
                             genome_code=int(C["gc"][r]), type=str(C["tlabels"][C["tcode"][r]]),
                             tcode=int(C["tcode"][r]), n_members=len(m), span=span,
                             split="odd" if C["gc"][r] % 2 else "even"))
    return pd.DataFrame(rows)


def member_topk(db, callers, out):
    """Top-K neighbours of every member gene of every >= 2-member instance (computed once, cached)."""
    gids = np.unique(np.concatenate([np.concatenate([callers[c]["mem"][r] for r in np.flatnonzero(callers[c]["target"])])
                                     for c in CALLERS]))
    d = out / "topk"
    if (d / "s.npy").exists():
        g0 = np.load(d / "gids.npy")
        assert np.array_equal(g0, gids)
        log("top-K cache", d)
        return gids, np.load(d / "idx.npy"), np.load(d / "cos.npy"), np.load(d / "s.npy")
    d.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    idx, cos, s = ls.topk_cosine(db, db.X[gids], db.gc[gids], K=K)
    secs = time.time() - t0
    log(f"top-K for {len(gids):,} member genes: {secs:.0f}s ({len(gids) / secs:.1f} vectors/s)")
    for name, arr in (("gids", gids), ("idx", idx), ("cos", cos), ("s", s)):
        np.save(d / f"{name}.npy", arr)
    json.dump(dict(n=int(len(gids)), K=K, seconds=round(secs, 1), vectors_per_second=len(gids) / secs),
              open(d / "topk.json", "w"), indent=1)
    return gids, idx, cos, s


def random_pair_quantiles(db, n=2_000_000, seed=17):
    rng = np.random.default_rng(seed)
    i = rng.integers(0, db.N, n)
    j = rng.integers(0, db.N, n)
    ok = db.gc[i] != db.gc[j]
    i, j = i[ok], j[ok]
    c = np.concatenate([np.einsum("ij,ij->i", db.X[i[a:a + 200000]], db.X[j[a:a + 200000]])
                        for a in range(0, len(i), 200000)])
    qs = (0.5, 0.9, 0.99, 0.999, 0.9999)
    return {str(q): float(np.quantile(c, q)) for q in qs}, len(c)


def relevance_and_headroom(db, callers, tk, cstar, rank_s):
    """Per (query, target) of one caller and type: f_tok, f_cos, f_rank, f_union (fraction of the
    target's members reachable from the query's members). Returns the pair table."""
    gids, idx, s = tk
    row_of = {int(g): k for k, g in enumerate(gids)}
    frames = []
    for c in CALLERS:
        C = callers[c]
        tr = np.flatnonzero(C["target"])
        for ty in np.unique(C["tcode"][tr]):
            rows = tr[C["tcode"][tr] == ty]
            mem = [C["mem"][r] for r in rows]
            lens = np.array([len(m) for m in mem])
            starts = np.r_[0, np.cumsum(lens)[:-1]]
            G = np.concatenate(mem)
            T = db.tok[G]
            n, m = len(rows), len(G)
            Vt = np.zeros((n, m), bool)
            for i in range(n):
                Vt[i] = np.isin(T, db.tok[mem[i]])
            Xg = db.X[G]
            Cm = Xg @ Xg.T
            Vc = np.maximum.reduceat(Cm, starts, axis=0) >= cstar
            Rg = np.zeros((m, m), bool)
            for a in range(m):
                k = row_of[int(G[a])]
                Rg[a] = np.isin(G, idx[k][s[k] >= rank_s])
            Vr = np.logical_or.reduceat(Rg, starts, axis=0)
            f = {}
            for name, V in (("f_tok", Vt), ("f_cos", Vc), ("f_rank", Vr), ("f_union", Vt | Vc | Vr),
                            ("f_emb", Vc | Vr)):
                f[name] = np.add.reduceat(V.astype(np.float64), starts, axis=1) / lens[None, :]
            qi, ti = np.nonzero(C["gc"][rows][:, None] != C["gc"][rows][None, :])
            frames.append(pd.DataFrame(dict(caller=c, type=str(C["tlabels"][ty]), q_row=rows[qi], t_row=rows[ti],
                                            **{k: v[qi, ti].astype(np.float32) for k, v in f.items()})))
    return pd.concat(frames, ignore_index=True)


def headroom_summary(pairs, queries):
    out = {}
    gq = queries.set_index(["caller", "row"]).genome_code
    for c, P in pairs.groupby("caller"):
        d = dict(ordered_pairs=int(len(P)), types=int(P.type.nunique()))
        for k in ("f_tok", "f_cos", "f_rank", "f_emb", "f_union"):
            v = P[k].to_numpy()
            d[k] = dict(mean=float(v.mean()), frac_ge_half=float((v >= 0.5).mean()), frac_zero=float((v == 0).mean()),
                        quantiles=np.quantile(v, [0.1, 0.25, 0.5, 0.75, 0.9]).round(4).tolist())
        inv = P.f_tok < 0.5
        d["token_invisible"] = float(inv.mean())
        for k in ("f_cos", "f_rank", "f_emb"):
            d[f"token_invisible_and_{k}_ge_half"] = float((inv & (P[k] >= 0.5)).mean())
        d["union_ge_half"] = float((P.f_union >= 0.5).mean())
        per_type = P.groupby("type").agg(pairs=("f_tok", "size"), f_tok=("f_tok", "mean"),
                                         vis_tok=("f_tok", lambda v: (v >= 0.5).mean()),
                                         f_emb=("f_emb", "mean"), vis_emb=("f_emb", lambda v: (v >= 0.5).mean()),
                                         vis_union=("f_union", lambda v: (v >= 0.5).mean()))
        d["macro_over_types"] = {k: float(per_type[k].mean()) for k in ("f_tok", "vis_tok", "f_emb", "vis_emb", "vis_union")}
        d["per_type_top"] = per_type.sort_values("pairs", ascending=False).head(15).round(4).reset_index().to_dict("records")
        out[c] = d
    return out


# =========================================================================== per-query machinery
def q_slots(q):
    C = CTX["callers"][q.caller]
    return C["mem"][q.row]


def rel_for(q):
    return CTX["rel"][q.qid]


def entries_for(q, mode, w, slot_nbrs=None, slot_tokens=None):
    """Evidence entries for a plain query (member genes) or a profile (slot tokens + neighbour rows)."""
    db = CTX["db"]
    own = q.genome_code
    lists = []
    if slot_tokens is None:
        for g in q_slots(q):
            k = CTX["row_of"][int(g)]
            lists.append(ls.slot_evidence(db, own, [db.tok[g]], CTX["tk_idx"][k], CTX["tk_s"][k], mode, w))
    else:
        for toks, (ni, ns) in zip(slot_tokens, slot_nbrs):
            lists.append(ls.slot_evidence(db, own, toks, ni, ns, mode, w))
    return ls.assemble(lists), len(lists)


def label_eval(q, ha, hb, score, E):
    C = CTX["callers"][q.caller]
    lab, match, _ = ls.label_hits(ha, hb, q.tcode, C["gptr"], C["ginst"], C["tcode"], C["nmem"], C["target"])
    rr, rv = rel_for(q)
    return ls.evaluate(score, lab, match, E, rr, rv), lab, match


def W_of(q, wx):
    return int(min(W_CAP, q.span + wx))


MODES = {"M0": ("tok", 1.0), "M1": ("emb", 1.0)}
for _w in GRID_W:
    MODES[f"M2w{_w:g}"] = ("comb", _w)


def tune_task(qid):
    q = CTX["queries"].loc[qid]
    db = CTX["db"]
    out = []
    for name, (mode, w) in MODES.items():
        (eg, es_, ev), ns = entries_for(q, mode, w)
        for wx, tau in product(GRID_WX, GRID_TAU):
            t0 = time.time()
            ha, hb, sc, _ = ls.scan(eg, es_, ev, ns, db.cc, db.pos, W_of(q, wx), tau, 0)
            r, _, _ = label_eval(q, ha, hb, sc, None)
            r.update(qid=int(qid), mode=name, wx=wx, tau=tau, seconds=time.time() - t0)
            out.append(r)
    return out


def final_task(args):
    """One query, one method config: scan + null + evaluation; also round-2 profile info for M2."""
    qid, method, mode, w, wx, tau = args
    q = CTX["queries"].loc[qid]
    db = CTX["db"]
    t0 = time.time()
    (eg, es_, ev), ns = entries_for(q, mode, w)
    W = W_of(q, wx)
    res = ls.run_entries(db, q.genome_code, (eg, es_, ev), ns, W, tau, R_NULL, SEED + qid, TOP_N)
    secs = time.time() - t0
    r, lab, match = label_eval(q, res["ha"], res["hb"], res["score"], res["E"])
    r.update(qid=int(qid), method=method, round=1, W=W, tau=tau, w=w, seconds=secs, n_entries=int(len(eg)))
    keep = res["E"] <= 10
    hits = dict(qid=int(qid), method=method, round=1, ha=res["ha"][keep], hb=res["hb"][keep],
                score=res["score"][keep], E=res["E"][keep], lab=lab[keep], match=match[keep], nfill=res["nfill"][keep])
    prof = None
    if method == "M2":
        base = [[int(g)] for g in q_slots(q)]
        prof = {v: profile_info(q, (eg, es_, ev), ns, res, W, tau, flank, [[db.tok[g]] for g in q_slots(q)],
                                base, base) for v, flank in M3_VARIANTS.items()}
    return r, hits, prof


def profile_info(q, entries, n_slots, res, W, tau, flank, slot_tokens, slot_genes, base):
    """Next-round profile from confident hits: per slot the genes for its centroid (previous genes plus
    the genes assigned to it in confident hits) and its token set, plus new slots for tokens recurring
    inside hit windows (+- flank). base = the query gene of each original slot."""
    db = CTX["db"]
    eg, es_, ev = entries
    sel = np.flatnonzero(res["E"] <= M3_ECONF)[:M3_MAXHITS]
    ha, hb = res["ha"][sel], res["hb"][sel]
    h, sl, g = ls.assign_windows(eg, es_, ev, n_slots, db.pos, ha, hb, tau)
    genes = [list(dict.fromkeys(list(slot_genes[i]) + g[sl == i].tolist())) for i in range(n_slots)]
    presence = [(1 + len(np.unique(h[sl == i]))) / (1 + len(sel)) for i in range(n_slots)]
    toks = [sorted(set(int(t) for t in slot_tokens[i]) | set(int(db.tok[x]) for x in genes[i])) for i in range(n_slots)]
    n_new = 0
    if flank > 0 and len(sel) >= 3:
        known = set(t for ts in toks for t in ts)
        assigned = set(g.tolist())
        first = db.c_first[db.cc[ha]]
        last = first + db.c_n[db.cc[ha]] - 1
        count, where = {}, {}
        for k in range(len(sel)):
            gg = np.arange(max(ha[k] - flank, first[k]), min(hb[k] + flank, last[k]) + 1)
            seen = set()
            for x in gg:
                t = int(db.tok[x])
                if x in assigned or t < 0 or t in known or t in seen:
                    continue
                seen.add(t)
                count[t] = count.get(t, 0) + 1
                where.setdefault(t, []).append(int(x))
        need = max(3, M3_NEW_MIN_FRAC * len(sel))
        cand = sorted((t for t, n in count.items() if n >= need), key=lambda t: -count[t])[:M3_NEW_MAX]
        for t in cand:
            genes.append(where[t])
            toks.append([t])
            presence.append(count[t] / (1 + len(sel)))
            n_new += 1
    return dict(genes=genes, toks=toks, presence=presence, n_conf=int(len(sel)), n_new=n_new,
                W=int(W + (2 * flank if n_new else 0)), base=[list(x) for x in base])


def round_task(args):
    """M3 round r >= 2 for one query and variant, from a profile whose neighbour rows are in CTX."""
    qid, variant, rnd, prof, w, tau = args
    q = CTX["queries"].loc[qid]
    db = CTX["db"]
    t0 = time.time()
    nbrs = []
    for i in range(len(prof["genes"])):
        key = (qid, variant, i)
        if key in CTX["cent_row"]:
            k = CTX["cent_row"][key]
            nbrs.append((CTX["cent_idx"][k], CTX["cent_s"][k]))
        else:                                   # profile slot equals its query gene: reuse its neighbours
            k = CTX["row_of"][int(prof["genes"][i][0])]
            nbrs.append((CTX["tk_idx"][k], CTX["tk_s"][k]))
    (eg, es_, ev), ns = entries_for(q, "comb", w, slot_nbrs=nbrs, slot_tokens=prof["toks"])
    res = ls.run_entries(db, q.genome_code, (eg, es_, ev), ns, prof["W"], tau, R_NULL, SEED + qid + 1000 * rnd, TOP_N)
    secs = time.time() - t0
    r, lab, match = label_eval(q, res["ha"], res["hb"], res["score"], res["E"])
    method = f"M3_{variant}"
    r.update(qid=int(qid), method=method, round=rnd, W=prof["W"], tau=tau, w=w, seconds=secs, n_entries=int(len(eg)),
             n_slots=ns, n_new_slots=prof["n_new"], n_conf_prev=prof["n_conf"])
    keep = res["E"] <= 10
    hits = dict(qid=int(qid), method=method, round=rnd, ha=res["ha"][keep], hb=res["hb"][keep], score=res["score"][keep],
                E=res["E"][keep], lab=lab[keep], match=match[keep], nfill=res["nfill"][keep])
    flank = M3_VARIANTS[variant]
    base_W = prof["W"] - (2 * flank if prof["n_new"] else 0)
    nxt = profile_info(q, (eg, es_, ev), ns, res, base_W, tau, flank if prof["n_new"] == 0 else 0,
                       prof["toks"], prof["genes"], prof["base"])
    if prof["n_new"]:                           # keep the slots added earlier; carry their window
        nxt["W"] = prof["W"]
        nxt["n_new"] = prof["n_new"]
    return (int(qid), variant), r, hits, nxt


def pool_map(fn, tasks, threads, chunksize=2):
    ctx = mp.get_context("fork")
    out = []
    t0 = time.time()
    with ctx.Pool(threads) as pool:
        for i, x in enumerate(pool.imap_unordered(fn, tasks, chunksize=chunksize)):
            out.append(x)
            if (i + 1) % 1000 == 0:
                log(f"  {fn.__name__}: {i + 1:,}/{len(tasks):,} {time.time() - t0:.0f}s")
    return out


def warm_up(db):
    z = np.array([0, 1], np.int64)
    ls.scan(z, np.zeros(2, np.int64), np.ones(2), 1, db.cc, db.pos, 3, 0.0, 0)
    ls.null_tail(z, np.zeros(2, np.int64), np.ones(2), 1, db.cc, db.pos, db.N, 0, 0, 3, 0.0, 2, 1, 4)
    ls.assign_windows(z, np.zeros(2, np.int64), np.ones(2), 1, db.pos, z[:1], z[1:], 0.0)
    C = CTX["callers"]["DefenseFinder"]
    ls.label_hits(z[:1], z[1:], 0, C["gptr"], C["ginst"], C["tcode"], C["nmem"], C["target"])


# =========================================================================== stages
def setup(a, vectors="ram"):
    t0 = time.time()
    db = ls.Database(vectors=vectors)
    cf = db.contigs.sort_index()
    db.c_first = np.zeros(int(cf.index.max()) + 1, np.int64)
    db.c_n = np.zeros_like(db.c_first)
    db.c_first[cf.index.to_numpy()] = cf.first_gid.to_numpy()
    db.c_n[cf.index.to_numpy()] = cf.n_genes.to_numpy()
    log(f"database: {db.N:,} genes, {db.n_genomes:,} genomes, {db.n_tokens:,} tokens, vectors={vectors}, {time.time() - t0:.0f}s")
    callers = load_gold(db)
    queries = make_queries(db, callers)
    log("queries", queries.groupby(["caller", "split"]).size().to_dict())
    CTX.update(db=db, callers=callers, queries=queries)
    return db, callers, queries


def load_relevance(out, queries, callers):
    pairs = pd.read_parquet(out / "pairs.parquet", columns=["caller", "q_row", "t_row", "f_tok"])
    rel = {}
    key = {(c, r): qid for qid, c, r in zip(queries.qid, queries.caller, queries.row)}
    for (c, qr), P in pairs.groupby(["caller", "q_row"], sort=False):
        rel[key[(c, int(qr))]] = (P.t_row.to_numpy(np.int64), (P.f_tok.to_numpy() >= 0.5))
    for qid in queries.qid:
        rel.setdefault(int(qid), (np.empty(0, np.int64), np.empty(0, bool)))
    return rel


def stage1(a):
    out = a.out
    out.mkdir(parents=True, exist_ok=True)
    json.dump(PREDECLARED, open(out / "criterion.json", "w"), indent=1)
    db, callers, queries = setup(a)
    queries.to_parquet(out / "queries.parquet", index=False)
    tk = member_topk(db, callers, out)
    gids, idx, cos, s = tk
    CTX.update(row_of={int(g): k for k, g in enumerate(gids)}, tk_idx=idx, tk_s=s)

    # ---- phase 0: headroom -------------------------------------------------------------------
    t0 = time.time()
    qs, npairs = random_pair_quantiles(db)
    cstar = qs["0.999"]
    rank_s = math.log(1000.0)              # tail rank <= N_db / 1000, i.e. the top 0.1% of the gene's ranking
    log("random-pair cosine quantiles", qs, "pairs", npairs)
    pairs = relevance_and_headroom(db, callers, (gids, idx, s), cstar, rank_s)
    pairs.to_parquet(out / "pairs.parquet", index=False)
    hs = headroom_summary(pairs, queries)
    hs["random_pair_cosine_quantiles"] = qs
    hs["random_pairs"] = npairs
    hs["cos_threshold_0.999"] = cstar
    hs["rank_threshold"] = "tail rank <= N_db/1000 (s >= log 1000)"
    hs["seconds"] = round(time.time() - t0, 1)
    json.dump(hs, open(out / "phase0_headroom.json", "w"), indent=1)
    for c in CALLERS:
        d = hs[c]
        log(f"phase0 {c}: pairs {d['ordered_pairs']:,} f_tok>=.5 {d['f_tok']['frac_ge_half']:.3f} "
            f"f_emb>=.5 {d['f_emb']['frac_ge_half']:.3f} union {d['union_ge_half']:.3f} "
            f"tok-invisible&emb-visible {d['token_invisible_and_f_emb_ge_half']:.3f}")

    # ---- tuning on odd genomes ---------------------------------------------------------------
    CTX["rel"] = load_relevance(out, queries, callers)
    warm_up(db)
    tq = queries[(queries.split == "odd")].qid.tolist()
    t0 = time.time()
    rows = [r for chunk in pool_map(tune_task, tq, a.threads) for r in chunk]
    tune = pd.DataFrame(rows).merge(queries[["qid", "caller", "n_members", "split"]], on="qid")
    tune.to_parquet(out / "tune_odd.parquet", index=False)
    log(f"tuning: {len(tq):,} odd queries x {len(MODES) * len(GRID_WX) * len(GRID_TAU)} configs, {time.time() - t0:.0f}s")
    best = {}
    agg = tune[tune.n_rel > 0].groupby(["caller", "mode", "wx", "tau"]).agg(map=("ap", "mean"), map_strict=("ap_strict", "mean"),
                                                                          hits=("n_hits", "mean"), n=("ap", "size")).reset_index()
    agg.to_csv(out / "tune_odd_summary.tsv", sep="\t", index=False)
    for c in CALLERS:
        A = agg[agg.caller == c]
        best[c] = {}
        for method, sub in (("M0", A[A["mode"] == "M0"]), ("M1", A[A["mode"] == "M1"]), ("M2", A[A["mode"].str.startswith("M2")])):
            b = sub.sort_values(["map", "wx", "tau"], ascending=[False, True, False]).iloc[0]
            mode, w = MODES[b["mode"]]
            best[c][method] = dict(mode=mode, w=w, wx=int(b.wx), tau=float(b.tau), odd_map=float(b["map"]),
                                   odd_map_strict=float(b.map_strict), grid_name=b["mode"])
        log(c, json.dumps(best[c]))
    json.dump(best, open(out / "best_configs.json", "w"), indent=1)


def hits_frame(hl):
    parts = []
    for h in hl:
        n = len(h["ha"])
        if n:
            parts.append(pd.DataFrame(dict(qid=h["qid"], method=h["method"], round=h["round"], start_gid=h["ha"],
                                           end_gid=h["hb"], score=h["score"], evalue=h["E"], label=h["lab"],
                                           match_row=h["match"], n_slots_filled=h["nfill"])))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def centroid_topk(db, profiles, queries):
    """Batch top-K for every profile slot whose gene set differs from its single query gene."""
    rows, vecs, owns, uniq = {}, [], [], {}
    for (qid, variant), p in profiles.items():
        own = int(queries.genome_code[qid])
        for i, genes in enumerate(p["genes"]):
            if len(genes) == 1 and i < len(p["base"]) and p["base"][i] == [genes[0]]:
                continue
            key = (own, tuple(sorted(int(g) for g in genes)))     # identical gene sets share one search
            if key not in uniq:
                v = db.X[np.asarray(key[1], np.int64)].mean(0)
                uniq[key] = len(vecs)
                vecs.append(v / np.linalg.norm(v))
                owns.append(own)
            rows[(qid, variant, i)] = uniq[key]
    if not vecs:
        return {}, np.empty((0, K), np.int32), np.empty((0, K), np.float32)
    t0 = time.time()
    idx, _, s = ls.topk_cosine(db, np.vstack(vecs).astype(np.float32), np.asarray(owns), K=K)
    log(f"centroid top-K: {len(vecs):,} distinct vectors for {len(rows):,} profile slots, {time.time() - t0:.0f}s")
    return rows, idx, s


def stage2(a):
    out = a.out
    db, callers, queries = setup(a)
    best = json.load(open(out / "best_configs.json"))
    gids, idx, _, s = member_topk(db, callers, out)
    CTX.update(row_of={int(g): k for k, g in enumerate(gids)}, tk_idx=idx, tk_s=s)
    CTX["rel"] = load_relevance(out, queries, callers)
    warm_up(db)
    tasks = []
    keep_callers = set(a.callers.split(",")) if a.callers else set(CALLERS)
    for qid, c in zip(queries.qid, queries.caller):
        if c not in keep_callers:
            continue
        for method in ("M0", "M1", "M2"):
            b = best[c][method]
            tasks.append((int(qid), method, b["mode"], b["w"], b["wx"], b["tau"]))
    t0 = time.time()
    res = pool_map(final_task, tasks, a.threads)
    log(f"final M0/M1/M2: {len(tasks):,} query-method runs, {time.time() - t0:.0f}s")
    metrics = [r for r, _, _ in res]
    hits = [h for _, h, _ in res]
    profiles = {}
    for r, _, p in res:
        if p is not None:
            for v, pp in p.items():
                profiles[(r["qid"], v)] = pp
    del res
    pd.DataFrame(metrics).merge(queries, on="qid").to_parquet(out / f"checkpoint_metrics_r1{a.tag}.parquet", index=False)
    # ---- M3 rounds 2 and 3 ------------------------------------------------------------------
    for rnd in range(2, a.rounds + 1):
        cent_row, cidx, cs = centroid_topk(db, profiles, queries)
        CTX.update(cent_row=cent_row, cent_idx=cidx, cent_s=cs)
        rt = []
        for (qid, variant), p in profiles.items():
            b = best[queries.caller[qid]]["M2"]
            rt.append((int(qid), variant, rnd, p, b["w"], b["tau"]))
        t0 = time.time()
        rr = pool_map(round_task, rt, a.threads)
        log(f"M3 round {rnd}: {len(rt):,} runs, {time.time() - t0:.0f}s")
        profiles = {}
        for key, r, h, nxt in rr:
            metrics.append(r)
            hits.append(h)
            profiles[key] = nxt
        del rr
        pd.DataFrame(metrics).merge(queries, on="qid").to_parquet(out / f"checkpoint_metrics_r{rnd}{a.tag}.parquet", index=False)
    M = pd.DataFrame(metrics).merge(queries, on="qid")
    M.to_parquet(out / f"final_metrics{a.tag}.parquet", index=False)
    H = hits_frame(hits)
    H.to_parquet(out / f"final_hits_E10{a.tag}.parquet", index=False)
    log(f"saved {len(M):,} metric rows, {len(H):,} hits (E <= 10)")
    summarize(out, a.tag)


# =========================================================================== summaries
def summarize(out, tag=""):
    M = pd.read_parquet(out / f"final_metrics{tag}.parquet")
    M = M[M.n_rel > 0].copy()
    M["mkey"] = M.method + "_r" + M["round"].astype(str)
    summ = dict(criterion=PREDECLARED, timing={}, tables={}, paired={}, selection={})
    for mk, g in M.groupby("mkey"):
        summ["timing"][mk] = dict(queries=int(len(g)), mean_seconds=float(g.seconds.mean()),
                                  median_seconds=float(g.seconds.median()), queries_per_cpu_second=float(1 / g.seconds.mean()))
    cols = ["ap", "ap_strict", "recall_any", "recall_E0.01", "recall_E1", "recall_vis_E0.01", "recall_inv_E0.01",
            "recall_vis_E1", "recall_inv_E1", "n_unlabelled_E0.01", "n_unlabelled_E1", "n_wrong_E0.01", "n_correct_E0.01",
            "precision_E0.01", "precision_strict_E0.01"]
    for c, Mc in M.groupby("caller"):
        for split in ("odd", "even"):
            for mm in (2, 3):
                S = Mc[(Mc.split == split) & (Mc.n_members >= mm)]
                tab = {}
                for mk, g in S.groupby("mkey"):
                    d = {"queries": int(len(g))}
                    for col in cols:
                        d[col] = ls.boot_mean(g[col].to_numpy(), g.genome_code.to_numpy())
                    d["micro_recall_inv_E0.01"] = float(g["found_inv_E0.01"].sum() / max(g.n_rel_inv.sum(), 1))
                    d["micro_recall_vis_E0.01"] = float(g["found_vis_E0.01"].sum() / max(g.n_rel_vis.sum(), 1))
                    d["micro_recall_inv_E1"] = float(g["found_inv_E1"].sum() / max(g.n_rel_inv.sum(), 1))
                    d["targets_inv_share"] = float(g.n_rel_inv.sum() / max(g.n_rel.sum(), 1))
                    tab[mk] = d
                summ["tables"][f"{c}|{split}|min{mm}"] = tab
                base = S[S.mkey == "M0_r1"].set_index("qid")
                pr = {}
                for mk, g in S.groupby("mkey"):
                    if mk == "M0_r1":
                        continue
                    g = g.set_index("qid")
                    ids = g.index.intersection(base.index)
                    gc = base.loc[ids, "genome_code"].to_numpy()
                    pr[mk] = {col: ls.boot_mean(g.loc[ids, col].to_numpy() - base.loc[ids, col].to_numpy(), gc)
                              for col in ("ap", "ap_strict", "recall_E0.01", "recall_E1", "recall_inv_E0.01", "recall_inv_E1",
                                          "n_unlabelled_E0.01")}
                summ["paired"][f"{c}|{split}|min{mm}"] = pr
    # ---- criterion -----------------------------------------------------------------------------
    if "DefenseFinder|odd|min2" not in summ["tables"]:
        json.dump(summ, open(out / f"summary{tag}.json", "w"), indent=1, default=float)
        return
    odd = summ["tables"]["DefenseFinder|odd|min2"]
    cands = {k: v["ap"][0] for k, v in odd.items() if not k.startswith("M0")}
    pick = max(cands, key=cands.get)
    ev = summ["tables"]["DefenseFinder|even|min2"]
    pe = summ["paired"]["DefenseFinder|even|min2"]
    m0_unl = ev["M0_r1"]["n_unlabelled_E0.01"][0]

    def verdict(k):
        d = pe[k]["ap"]
        unl = ev[k]["n_unlabelled_E0.01"][0]
        rate_ok = unl <= 0.15 if m0_unl < 0.1 else unl <= 1.5 * m0_unl
        return dict(delta_ap=d, unlabelled=unl, m0_unlabelled=m0_unl, rate_ok=bool(rate_ok),
                    gain=bool(d[0] >= 0.02 and d[1] > 0 and rate_ok),
                    inv_recall_delta_E0_01=pe[k]["recall_inv_E0.01"], inv_recall_delta_E1=pe[k]["recall_inv_E1"])
    summ["selection"] = dict(odd_map=cands, selected=pick, verdict=verdict(pick),
                             all_on_even={k: verdict(k) for k in pe})
    json.dump(summ, open(out / f"summary{tag}.json", "w"), indent=1, default=float)
    log("criterion", json.dumps(summ["selection"]["verdict"], default=float))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["stage1", "stage2", "summary"])
    ap.add_argument("--out", type=Path, default=ls.RD / "locus_search/out")
    ap.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_ON_NODE", "2")))
    ap.add_argument("--callers", default="", help="comma-separated callers for stage2 (default all)")
    ap.add_argument("--rounds", type=int, default=3, help="stage2: 1 = M0/M1/M2 only, 3 = with M3 rounds 2-3")
    ap.add_argument("--tag", default="", help="suffix of stage2 output files")
    a = ap.parse_args()
    import numba
    numba.set_num_threads(a.threads)
    t0 = time.time()
    if a.stage == "stage1":
        stage1(a)
    elif a.stage == "stage2":
        stage2(a)
    else:
        summarize(a.out, a.tag)
    log(f"done {a.stage} {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
