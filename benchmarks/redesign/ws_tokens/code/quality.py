#!/usr/bin/env python
"""Token quality for one or more arms.

prep       per arm: size/genome statistics, homogeneity/completeness/V-measure against the
           reference arm, pair recall on every label set, and two samples of 200,000
           same-token cross-genome pairs (pair-uniform and token-uniform). Writes the
           union of sampled pairs as a fake MMseqs prefilter TSV (gid keys).
precision  join the MMseqs alignment of the sampled pairs; sequence precision per sampler.
tmprep     per arm: up to --tm-n sequence-invisible pair-uniform pairs with both AFDB
           structures -> fake prefilter TSVs for foldseek, split by side combination.
tmcollect  join TM-scores; structural confirmation of the invisible same-token pairs.
collect    assemble quality.tsv (one row per arm) from the per-arm JSON files.

Samplers (exact, with replacement, seed 17):
  pair-uniform:  gene i drawn with probability proportional to w_i = n_t - n_{t,genome(i)}
                 (its number of cross-genome co-members), then j uniform among those
                 co-members. Each ordered cross-genome same-token pair has probability
                 1 / sum_i w_i, so the sample is uniform over pairs; large tokens dominate.
  token-uniform: token drawn uniformly among tokens with >= 1 cross-genome pair, then
                 i proportional to w_i inside it and j as above; every token weighs the same.
"""
import argparse
import json
import math
import os
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.metrics import homogeneity_completeness_v_measure

from common import NGENES, NREF, TOK, UNIVERSE, WS, peak_rss_gb, stamp

Q = WS / "quality"
REF_ARM = "mmseqs__seq__id30c80"
SETS = ["sd_vis", "sd_invis", "blk650_vis", "blk650_invis", "tm_hi", "tm_lo", "random"]


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(c - h, 5), round(c + h, 5))


def load_tokens(arm):
    t = duckdb.connect().execute(f"SELECT gid, token FROM read_parquet('{TOK}/{arm}.parquet') ORDER BY gid").fetchnumpy()
    assert len(t["gid"]) == NGENES and np.array_equal(t["gid"], np.arange(NGENES))
    return t["token"].astype(np.int64)


def universe():
    u = duckdb.connect().execute(f"""SELECT gid, genome_code, side, has_struct, mmseqs_key, length
                                      FROM read_parquet('{UNIVERSE}') ORDER BY gid""").fetchnumpy()
    assert np.array_equal(u["gid"], np.arange(NGENES))
    return u


def token_tables(tok, genome):
    """Dense token ids, members sorted by (token, genome), and per-token/per-member counts."""
    a = np.flatnonzero(tok >= 0)
    uniq, dense = np.unique(tok[a], return_inverse=True)
    T = len(uniq)
    order = np.lexsort((genome[a], dense))
    mem = a[order]                      # gids sorted by (token, genome)
    mt = dense[order]
    mg = genome[mem]
    n_t = np.bincount(mt, minlength=T)
    t_start = np.concatenate([[0], np.cumsum(n_t)[:-1]])
    # (token, genome) blocks
    key_change = np.ones(len(mem), dtype=bool)
    key_change[1:] = (mt[1:] != mt[:-1]) | (mg[1:] != mg[:-1])
    blk_id = np.cumsum(key_change) - 1
    blk_start = np.flatnonzero(key_change)
    blk_n = np.diff(np.concatenate([blk_start, [len(mem)]]))
    G_t = np.bincount(mt[blk_start], minlength=T)          # genomes carrying token
    w = n_t[mt] - blk_n[blk_id]                            # cross-genome co-members of each member
    return dict(T=T, mem=mem, mt=mt, n_t=n_t, t_start=t_start, G_t=G_t, w=w.astype(np.int64),
                blk_id=blk_id, blk_start=blk_start, blk_n=blk_n)


def draw_partner(tt, i_pos, rng):
    """For member positions i_pos, draw j uniformly among cross-genome co-members."""
    t = tt["mt"][i_pos]
    b = tt["blk_id"][i_pos]
    a0 = tt["blk_start"][b] - tt["t_start"][t]             # genome block offset within token
    bl = tt["blk_n"][b]
    m = tt["n_t"][t] - bl
    r = np.floor(rng.random(len(i_pos)) * m).astype(np.int64)
    r = np.where(r >= a0, r + bl, r)
    return tt["t_start"][t] + r


def sample_pairs(tt, n, rng, mode):
    # Integer cumulative weights: with giant tokens the float version rounded u up to the
    # next token's first weight (graph arms, 2026-11-11 04:43 PST).
    W = np.cumsum(tt["w"], dtype=np.int64)
    if mode == "pair":
        u = rng.integers(0, W[-1], n)
        i_pos = np.searchsorted(W, u, side="right")
    else:
        P = np.zeros(tt["T"], dtype=np.int64)
        np.add.at(P, tt["mt"], tt["w"])
        cross = np.flatnonzero(P > 0)
        t = cross[rng.integers(0, len(cross), n)]
        lo = np.where(tt["t_start"][t] > 0, W[tt["t_start"][t] - 1], 0)
        hi = W[tt["t_start"][t] + tt["n_t"][t] - 1]
        u = rng.integers(lo, hi)
        i_pos = np.searchsorted(W, u, side="right")
        assert (tt["mt"][i_pos] == t).all()
    assert (tt["w"][i_pos] > 0).all()
    j_pos = draw_partner(tt, i_pos, rng)
    g1, g2 = tt["mem"][i_pos], tt["mem"][j_pos]
    assert (tt["mt"][i_pos] == tt["mt"][j_pos]).all()
    return np.minimum(g1, g2), np.maximum(g1, g2)


def pair_recall(arm, tok, L, st, out):
    rec = []
    for s in SETS:
        d = L[L.lset == s]
        if len(d) == 0:
            continue
        t1, t2 = tok[d.g1.to_numpy()], tok[d.g2.to_numpy()]
        hit = (t1 == t2) & (t1 >= 0)
        both = (t1 >= 0) & (t2 >= 0)
        k, n = int(hit.sum()), len(d)
        rec.append(dict(arm=arm, lset=s, n=n, n_both_tokened=int(both.sum()), shared=k,
                        recall=k / n, ci_lo=wilson(k, n)[0], ci_hi=wilson(k, n)[1],
                        recall_given_tokens=k / max(int(both.sum()), 1)))
        st[f"recall_{s}"] = round(k / n, 5)
        st[f"recall_{s}_ci"] = wilson(k, n)
        st[f"n_{s}"] = n
    pd.DataFrame(rec).to_csv(out / "recall.tsv", sep="\t", index=False)


def recall_only(a):
    L = duckdb.connect().execute(f"SELECT lset, g1, g2 FROM read_parquet('{WS}/labels/label_pairs.parquet')").df()
    arms = a.arms.split(",") if a.arms else [d.name for d in sorted(Q.iterdir()) if (d / "stats.json").exists()]
    for arm in arms:
        out = Q / arm
        st = json.load(open(out / "stats.json"))
        pair_recall(arm, load_tokens(arm), L, st, out)
        with open(out / "stats.json", "w") as fh:
            json.dump(st, fh, indent=1, default=float)
        stamp(arm, {k: v for k, v in st.items() if k.startswith("recall_") and not k.endswith("_ci")})


def prep(a):
    u = universe()
    genome = u["genome_code"].astype(np.int64)
    ref = load_tokens(REF_ARM) if (TOK / f"{REF_ARM}.parquet").exists() else None
    L = duckdb.connect().execute(f"SELECT lset, g1, g2 FROM read_parquet('{WS}/labels/label_pairs.parquet')").df()
    allpairs = []
    for arm in a.arms.split(","):
        t0 = time.time()
        out = Q / arm
        out.mkdir(parents=True, exist_ok=True)
        tok = load_tokens(arm)
        man = json.load(open(TOK / f"{arm}.json"))
        tt = token_tables(tok, genome)
        n_t, G_t = tt["n_t"], tt["G_t"]
        cpg = n_t / G_t
        P = np.zeros(tt["T"], dtype=np.float64)
        np.add.at(P, tt["mt"], tt["w"])
        P /= 2
        st = dict(arm=arm, source=man.get("source"), model=man.get("model"), transform=man.get("transform"),
                  detail=man.get("detail"), n_tokens=int(tt["T"]), n_assigned=int((tok >= 0).sum()),
                  n_unassigned=int((tok < 0).sum()),
                  size_q50=float(np.quantile(n_t, 0.5)), size_q90=float(np.quantile(n_t, 0.9)),
                  size_q99=float(np.quantile(n_t, 0.99)), size_max=int(n_t.max()), size_mean=float(n_t.mean()),
                  singleton_tokens=int((n_t == 1).sum()),
                  genes_in_largest_1pct_tokens=float(np.sort(n_t)[::-1][:max(1, tt["T"] // 100)].sum() / n_t.sum()),
                  tokens_ge5_genomes=int((G_t >= 5).sum()), tokens_ge50_genomes=int((G_t >= 50).sum()),
                  genes_in_tokens_ge5_genomes=int(n_t[G_t >= 5].sum()),
                  copies_per_genome_mean=float(cpg.mean()),
                  copies_per_genome_mean_ge5=float(cpg[G_t >= 5].mean()) if (G_t >= 5).any() else None,
                  cross_genome_same_token_pairs=float(P.sum()),
                  token_seconds=man.get("seconds"), token_peak_rss_gb=man.get("peak_rss_gb"),
                  token_cpus=man.get("cpus"))
        if ref is not None:
            m = (tok >= 0) & (ref >= 0)
            h, c, v = homogeneity_completeness_v_measure(ref[m], tok[m])
            st.update(homogeneity=round(h, 5), completeness=round(c, 5), v_measure=round(v, 5),
                      v_measure_genes=int(m.sum()), v_measure_sample="all genes with a token in both arms")
        pair_recall(arm, tok, L, st, out)
        rng = np.random.default_rng(a.seed)
        samp = []
        for mode in ("pair", "token"):
            g1, g2 = sample_pairs(tt, a.n_sample, rng, mode)
            samp.append(pd.DataFrame({"sampler": mode, "draw": np.arange(a.n_sample), "g1": g1, "g2": g2}))
        samp = pd.concat(samp, ignore_index=True)
        assert (genome[samp.g1.to_numpy()] != genome[samp.g2.to_numpy()]).all()
        assert (tok[samp.g1.to_numpy()] == tok[samp.g2.to_numpy()]).all()
        samp.to_parquet(out / "sample.parquet", index=False)
        st["sample_unique_pairs"] = {m: int(samp[samp.sampler == m][["g1", "g2"]].drop_duplicates().shape[0])
                                     for m in ("pair", "token")}
        st["prep_seconds"] = round(time.time() - t0, 1)
        st["prep_peak_rss_gb"] = peak_rss_gb()
        with open(out / "stats.json", "w") as fh:
            json.dump(st, fh, indent=1, default=float)
        allpairs.append(samp[["g1", "g2"]])
        stamp(arm, {k: v for k, v in st.items() if not k.endswith("_ci")})
    P = pd.concat(allpairs).drop_duplicates().sort_values(["g1", "g2"])
    P["s"] = 0
    P["d"] = 0
    P.to_csv(a.pref, sep="\t", header=False, index=False)
    stamp("pref pairs", len(P), a.pref)


def precision(a):
    con = duckdb.connect()
    con.execute(f"PRAGMA threads={a.threads}")
    # A few MMseqs rows are degenerate (qstart -1, seqid printed as "0.-2147483648", E 1e-323);
    # they are treated as "no alignment".
    raw = f"""SELECT q, t, TRY_CAST(bits AS DOUBLE) AS bits, TRY_CAST(seqid AS DOUBLE) AS seqid,
                     TRY_CAST(evalue AS DOUBLE) AS evalue, TRY_CAST(qs AS BIGINT) AS qs
              FROM read_csv('{a.aln}', delim='\t', header=false, all_varchar=true,
                   names=['q','t','bits','seqid','evalue','qs','qe','ql','ts','te','tl'])"""
    n_bad = con.execute(f"SELECT count(*) FROM ({raw}) WHERE seqid IS NULL OR evalue IS NULL OR qs < 0").fetchone()[0]
    stamp("degenerate alignment rows treated as no alignment:", n_bad)
    aln = con.execute(f"""
        SELECT CAST(q AS BIGINT) AS g1, CAST(t AS BIGINT) AS g2, min(evalue) AS evalue, max(seqid) AS seqid,
               max(bits) AS bits
        FROM ({raw}) WHERE seqid IS NOT NULL AND evalue IS NOT NULL AND qs >= 0
        GROUP BY ALL""").df()
    for arm in a.arms.split(","):
        out = Q / arm
        samp = pd.read_parquet(out / "sample.parquet").merge(aln, on=["g1", "g2"], how="left")
        samp["vis"] = samp.evalue.fillna(np.inf) <= 1e-3
        samp.to_parquet(out / "sample_aln.parquet", index=False)
        st = json.load(open(out / "stats.json"))
        for m in ("pair", "token"):
            d = samp[samp.sampler == m]
            k, n = int(d.vis.sum()), len(d)
            st[f"seqprec_{m}"] = round(k / n, 5)
            st[f"seqprec_{m}_ci"] = wilson(k, n)
            st[f"seqprec_{m}_n"] = n
            st[f"seqprec_{m}_no_alignment"] = int(d.evalue.isna().sum())
            st[f"seqid_median_vis_{m}"] = float(d[d.vis].seqid.median()) if k else None
        with open(out / "stats.json", "w") as fh:
            json.dump(st, fh, indent=1, default=float)
        stamp(arm, "seqprec pair", st["seqprec_pair"], "token", st["seqprec_token"])


def tmprep(a):
    u = universe()
    rng = np.random.default_rng(a.seed)
    rows = []
    for arm in a.arms.split(","):
        out = Q / arm
        s = pd.read_parquet(out / "sample_aln.parquet")
        s = s[(s.sampler == a.tm_sampler) & ~s.vis][["g1", "g2"]].drop_duplicates()
        s = s[u["has_struct"][s.g1.to_numpy()] & u["has_struct"][s.g2.to_numpy()]]
        n_avail = len(s)
        if len(s) > a.tm_n:
            s = s.iloc[np.sort(rng.choice(len(s), a.tm_n, replace=False))]
        s = s.assign(arm=arm)
        rows.append(s)
        st = json.load(open(out / "stats.json"))
        sfx = "" if a.tm_sampler == "pair" else "_token"
        st["tm_invisible_with_struct_available" + sfx] = n_avail
        st["tm_requested" + sfx] = len(s)
        with open(out / "stats.json", "w") as fh:
            json.dump(st, fh, indent=1, default=float)
    R = pd.concat(rows)
    # per-arm requests, so tmcollect scores each arm only on the pairs sampled for it
    R[["arm", "g1", "g2"]].to_parquet(Path(a.tmdir) / f"tm_requested_{a.tm_sampler}.parquet", index=False)
    # orient: query-side protein first; keys are MMseqs/foldseek keys of each side's database
    g1, g2 = R.g1.to_numpy(), R.g2.to_numpy()
    q1, q2 = g1 >= NREF, g2 >= NREF
    swap = ~q1 & q2
    A = np.where(swap, g2, g1)
    B = np.where(swap, g1, g2)
    combo = np.where(A >= NREF, np.where(B >= NREF, "qq", "qr"), "rr")
    key = u["mmseqs_key"]
    T = pd.DataFrame({"A": A, "B": B, "ka": key[A], "kb": key[B], "combo": combo}).drop_duplicates(["A", "B"])
    T.to_parquet(Path(a.tmdir) / "tm_pairs.parquet", index=False)
    for c in ("rr", "qr", "qq"):
        d = T[T.combo == c].sort_values(["ka", "kb"])
        d[["ka", "kb"]].assign(s=0, x=0).to_csv(Path(a.tmdir) / f"pref_{c}.tsv", sep="\t", header=False, index=False)
        stamp("tm combo", c, len(d))


def tmcollect(a):
    T = pd.read_parquet(Path(a.tmdir) / "tm_pairs.parquet")
    res = []
    for c in ("rr", "qr", "qq"):
        f = Path(a.tmdir) / f"aln_{c}.tsv"
        if not f.exists() or f.stat().st_size == 0:
            continue
        r = pd.read_csv(f, sep="\t", header=None, usecols=[0, 1, 2, 3, 5], names=["qn", "tn", "qtm", "ttm", "lddt"],
                        dtype={"qn": str, "tn": str})
        r["ka"] = r.qn.str[1:].astype(np.int64)
        r["kb"] = r.tn.str[1:].astype(np.int64)
        r["combo"] = c
        res.append(r)
    R = pd.concat(res)
    R["tm_max"] = R[["qtm", "ttm"]].max(axis=1)
    R["tm_min"] = R[["qtm", "ttm"]].min(axis=1)
    R = R.sort_values("tm_max", ascending=False).drop_duplicates(["combo", "ka", "kb"])
    T = T.merge(R[["combo", "ka", "kb", "tm_max", "tm_min", "lddt"]], on=["combo", "ka", "kb"], how="left")
    T.to_parquet(Path(a.tmdir) / "tm_scores.parquet", index=False)
    u = universe()
    for arm in a.arms.split(","):
        out = Q / arm
        s = pd.read_parquet(out / "sample_aln.parquet")
        s = s[(s.sampler == a.tm_sampler) & ~s.vis][["g1", "g2"]].drop_duplicates()
        s = s[u["has_struct"][s.g1.to_numpy()] & u["has_struct"][s.g2.to_numpy()]]
        req_f = Path(a.tmdir) / f"tm_requested_{a.tm_sampler}.parquet"
        if req_f.exists():
            req = pd.read_parquet(req_f)
            s = s.merge(req[req.arm == arm][["g1", "g2"]], on=["g1", "g2"], how="inner")
        s["A"] = np.where((s.g1 < NREF) & (s.g2 >= NREF), s.g2, s.g1)
        s["B"] = np.where((s.g1 < NREF) & (s.g2 >= NREF), s.g1, s.g2)
        d = s.merge(T, on=["A", "B"], how="inner")
        st = json.load(open(out / "stats.json"))
        n = int(d.tm_max.notna().sum())
        k = int((d.tm_max >= 0.5).sum())
        sfx = "" if a.tm_sampler == "pair" else "_token"
        st.update({"tm_n" + sfx: n, "tm_missing" + sfx: int(d.tm_max.isna().sum()),
                   "tm_conf" + sfx: round(k / n, 5) if n else None, "tm_conf_ci" + sfx: wilson(k, n),
                   "tm_conf_strict" + sfx: round(int((d.tm_min >= 0.5).sum()) / n, 5) if n else None,
                   "tm_median" + sfx: float(d.tm_max.median()) if n else None})
        sp_key = "seqprec_pair" if a.tm_sampler == "pair" else "seqprec_token"
        if n and sp_key in st:
            sp = st[sp_key]
            st["homology_prec_" + a.tm_sampler] = round(sp + (1 - sp) * k / n, 5)
        d.to_parquet(out / f"tm_invisible{sfx}.parquet", index=False)
        with open(out / "stats.json", "w") as fh:
            json.dump(st, fh, indent=1, default=float)
        stamp(arm, "tm_conf" + sfx, st["tm_conf" + sfx], "n", n, "homology_prec_" + a.tm_sampler,
              st.get("homology_prec_" + a.tm_sampler))


def collect(a):
    rows = []
    for d in sorted(Q.iterdir()):
        f = d / "stats.json"
        if f.exists():
            st = json.load(open(f))
            rows.append({k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in st.items()})
    df = pd.DataFrame(rows)
    front = ["arm", "source", "model", "transform", "detail", "n_tokens", "n_assigned", "size_q50", "size_q90",
             "size_q99", "size_max", "tokens_ge5_genomes", "tokens_ge50_genomes", "copies_per_genome_mean",
             "homogeneity", "completeness", "v_measure"] + [f"recall_{s}" for s in SETS] + \
            ["seqprec_pair", "seqprec_token", "tm_conf", "homology_prec_pair", "tm_conf_token", "homology_prec_token",
             "token_seconds", "token_peak_rss_gb", "token_cpus"]
    cols = [c for c in front if c in df.columns] + [c for c in df.columns if c not in front]
    df = df[cols]
    tmp = WS / f".quality.tsv.tmp{os.getpid()}"
    df.to_csv(tmp, sep="\t", index=False)
    os.replace(tmp, WS / "quality.tsv")
    stamp("quality.tsv rows", len(df))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["prep", "recall", "precision", "tmprep", "tmcollect", "collect"])
    ap.add_argument("--arms", default="")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--n-sample", type=int, default=200_000)
    ap.add_argument("--tm-n", type=int, default=20_000)
    ap.add_argument("--pref", default="")
    ap.add_argument("--aln", default="")
    ap.add_argument("--tmdir", default="")
    ap.add_argument("--tm-sampler", default="pair", choices=["pair", "token"])
    a = ap.parse_args()
    {"prep": prep, "recall": recall_only, "precision": precision, "tmprep": tmprep, "tmcollect": tmcollect, "collect": collect}[a.mode](a)
