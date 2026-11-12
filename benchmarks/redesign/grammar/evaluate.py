#!/usr/bin/env python
"""Neighbourhood-grammar workstream, step 3: evaluations on held-out orders.

  e123   evaluation 1 (curated systems), 2 (co-folding), 3 (geometry) for the 5-fold order CV (pooled) and
         for the phylum split; writes $RD/grammar/eval/*.tsv and eval_summary.json
Methods: critic scores (primary, w3, glob, shuf, sel_c*), baseline (i) training-side PMI (pmi_a1; variants
pmi_a01, pmi_a10, G3_train, nlogP_train), (ii) engine all-genome -log P (eng_nlogP), (iii) raw cosine.
AUROC intervals: genome-level bootstrap (1,000 draws, seed 17); differences are paired on the same draws.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit, prange

RD = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111")
PREP = RD / "grammar/prep"
RUNS = Path(os.environ.get("GR_RUNS", RD / "grammar/runs"))
EV = Path(os.environ.get("GR_EVAL", RD / "grammar/eval"))
CALLERS = ("DefenseFinder", "TXSScan", "CRISPRCasTyper")
T0 = time.time()


def log(*a):
    print(time.strftime("%H:%M:%S"), f"{time.time() - T0:7.0f}s", *a, flush=True)


# ------------------------------------------------------------------------------------- AUROC with bootstrap
@njit(cache=True)
def _auc_w(yy, gg, starts, w):
    n = len(yy)
    cum_n, num, wp_tot, wn_tot = 0.0, 0.0, 0.0, 0.0
    for k in range(len(starts)):
        s = starts[k]
        e = starts[k + 1] if k + 1 < len(starts) else n
        wp, wn = 0.0, 0.0
        for i in range(s, e):
            x = w[gg[i]]
            if yy[i]:
                wp += x
            else:
                wn += x
        num += wp * (cum_n + 0.5 * wn)
        cum_n += wn
        wp_tot += wp
        wn_tot += wn
    if wp_tot == 0.0 or wn_tot == 0.0:
        return np.nan
    return num / (wp_tot * wn_tot)


@njit(parallel=True, cache=True)
def _auc_draws(yy, gg, starts, W):
    out = np.empty(W.shape[0])
    for d in prange(W.shape[0]):
        out[d] = _auc_w(yy, gg, starts, W[d])
    return out


class Boot:
    def __init__(self, groups, draws=1000, seed=17):
        codes, self.inv = np.unique(groups, return_inverse=True)
        rng = np.random.default_rng(seed)
        G = len(codes)
        self.W = rng.multinomial(G, np.full(G, 1.0 / G), size=draws).astype(np.float64)
        self.one = np.ones(G)

    def auc(self, s, y):
        o = np.argsort(s, kind="mergesort")
        ss, yy, gg = s[o], y[o], self.inv[o]
        starts = np.r_[0, np.flatnonzero(np.diff(ss)) + 1].astype(np.int64)
        return _auc_w(yy, gg, starts, self.one), _auc_draws(yy, gg, starts, self.W)


def ap_score(s, y):
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(y, s)) if y.any() and (~y).any() else np.nan


# ------------------------------------------------------------------------------------- assembling scores
def assemble(split_names, adj, base_prefix):
    """Per adjacency (row of adj1): the score of each model from the split whose test set holds it,
    and the training-side baselines of that split."""
    n = len(adj)
    sc, fold_of = {}, np.full(n, -1, dtype=np.int8)
    for k, s in enumerate(split_names):
        z = np.load(RUNS / s / "scores_adj1.npz")
        idx = z["idx"]
        fold_of[idx] = k
        for m in z.files:
            if m == "idx":
                continue
            if m not in sc:
                sc[m] = np.full(n, np.nan, dtype=np.float64)
            sc[m][idx] = z[m]
        b = np.load(PREP / f"base_{s}.npz")
        for m in b.files:
            if m not in sc:
                sc[m] = np.full(n, np.nan, dtype=np.float64)
            sc[m][idx] = b[m][idx]
        # symmetric primary score (mean of both orders for every pair) from the saved towers
        uv = np.load(RUNS / s / "uv_primary.npz")
        row = np.full(n_genes_global, -1, dtype=np.int64)
        row[uv["genes"]] = np.arange(len(uv["genes"]))
        U, V = uv["U"].astype(np.float32), uv["V"].astype(np.float32)
        a, b_ = adj.a.to_numpy()[idx], adj.b.to_numpy()[idx]
        s_ab = np.einsum("ij,ij->i", U[row[a]], V[row[b_]]) * float(uv["scale"]) + float(uv["bias"])
        s_ba = np.einsum("ij,ij->i", U[row[b_]], V[row[a]]) * float(uv["scale"]) + float(uv["bias"])
        if "primary_sym" not in sc:
            sc["primary_sym"] = np.full(n, np.nan)
            sc["primary_ab"] = np.full(n, np.nan)
        sc["primary_sym"][idx] = 0.5 * (s_ab + s_ba)
        sc["primary_ab"][idx] = s_ab
    sc["eng_nlogP"] = adj.eng_nlogP.to_numpy(np.float64)
    sc["raw_cos"] = adj.raw_cos.to_numpy(np.float64)
    return sc, fold_of


METHODS_MAIN = ["primary", "w3", "glob", "shuf", "pmi_a1", "pmi_a01", "pmi_a10", "G3_train", "nlogP_train",
                "eng_nlogP", "raw_cos"]


def eval1(tag, adj, sc, fold_of, out_rows, draws=1000, methods=None, strata_keep=None, ref="primary"):
    covered = fold_of >= 0
    g = adj.genome_code.to_numpy()
    G3 = sc["G3_train"]
    G1 = sc["G1_train"]
    single = adj.singleton_either.to_numpy()
    distinct = ~adj.same_token.to_numpy()
    strata = {
        "all": np.ones(len(adj), bool),
        "never_w3": G3 == 0, "1-4_w3": (G3 >= 1) & (G3 <= 4), ">=5_w3": G3 >= 5,
        "never_w1": G1 == 0, "1-4_w1": (G1 >= 1) & (G1 <= 4), ">=5_w1": G1 >= 5,
        "singleton_either": single, "no_singleton": ~single,
        "distinct_tokens": distinct,
    }
    if methods is None:
        methods = [m for m in METHODS_MAIN + sorted(k for k in sc if k.startswith("sel_")) + ["primary_sym"] if m in sc]
    if strata_keep is not None:
        strata = {k: v for k, v in strata.items() if k in strata_keep}
    for caller in CALLERS:
        pos = adj[f"pos_{caller}"].to_numpy() & covered
        negsets = {"nogold": adj.gold_none.to_numpy() & covered,
                   "straddle": adj[f"straddle_{caller}"].to_numpy() & covered}
        for negname, neg in negsets.items():
            for sname, smask in strata.items():
                if caller != "DefenseFinder" and sname.endswith("_w1"):
                    continue
                p, q = pos & smask, neg & smask
                if p.sum() < 5 or q.sum() < 5:
                    out_rows.append(dict(split=tag, caller=caller, negset=negname, stratum=sname,
                                         n_pos=int(p.sum()), n_neg=int(q.sum())))
                    continue
                sel = p | q
                y = p[sel]
                boot = Boot(g[sel], draws=draws)
                aucs, draws_of = {}, {}
                for m in methods:
                    s = sc[m][sel]
                    if np.isnan(s).any():
                        continue
                    aucs[m], draws_of[m] = boot.auc(s, y)
                    aucs[m + "__ap"] = ap_score(s, y)
                for m in [k for k in aucs if not k.endswith("__ap")]:
                    r = dict(split=tag, caller=caller, negset=negname, stratum=sname, n_pos=int(p.sum()),
                             n_neg=int(q.sum()), n_genomes=int(len(np.unique(g[sel]))), method=m,
                             auc=aucs[m], auc_lo=float(np.nanquantile(draws_of[m], .025)),
                             auc_hi=float(np.nanquantile(draws_of[m], .975)), ap=aucs[m + "__ap"],
                             prevalence=float(y.mean()))
                    if m != ref and ref in draws_of:
                        d = draws_of[ref] - draws_of[m]
                        r.update(ref=ref, ref_minus=aucs[ref] - aucs[m], rm_lo=float(np.nanquantile(d, .025)),
                                 rm_hi=float(np.nanquantile(d, .975)))
                    out_rows.append(r)
                log(tag, caller, negname, sname, int(p.sum()), int(q.sum()),
                    {m: round(aucs[m], 3) for m in methods if m in aucs})


def eval2(tag, splits, out_rows, uncalled_hits):
    fs = []
    for s in splits:
        f = pd.read_parquet(RUNS / s / "scores_folded.parquet")
        f["split"] = s
        fs.append(f)
    f = pd.concat(fs, ignore_index=True)
    if "GR_RUNS" in os.environ:          # pipeline test on one run linked as every fold
        f = f.drop_duplicates("fold_id").reset_index(drop=True)
    assert f.fold_id.is_unique
    f["hit"] = f.iptm >= 0.75
    f["neg_len_sum"] = -f.len_sum.astype(float)
    methods = [c for c in f.columns if c.startswith("s_")] + ["eng_nlogP", "neg_len_sum"]
    res = {}
    for cls_name, m_cls in (("cognate+uncalled", np.ones(len(f), bool)), ("uncalled", (f.cls == "uncalled").to_numpy()),
                            ("cognate", (f.cls == "cognate").to_numpy())):
        sub = f[m_cls]
        y = sub.hit.to_numpy()
        boot = Boot(np.arange(len(sub)), draws=2000, seed=17)
        dr = {}
        for m in methods:
            a, dr[m] = boot.auc(sub[m].to_numpy(np.float64), y)
            r = dict(split=tag, set=cls_name, n=int(len(sub)), n_hit=int(y.sum()), method=m, auc=a,
                     auc_lo=float(np.nanquantile(dr[m], .025)), auc_hi=float(np.nanquantile(dr[m], .975)),
                     ap=ap_score(sub[m].to_numpy(np.float64), y))
            out_rows.append(r)
        if "s_primary" in dr:
            for m in methods:
                if m != "s_primary":
                    d = dr["s_primary"] - dr[m]
                    out_rows.append(dict(split=tag, set=cls_name, method=f"s_primary_minus_{m}",
                                         auc=float(np.nanmean(d)), auc_lo=float(np.nanquantile(d, .025)),
                                         auc_hi=float(np.nanquantile(d, .975))))
        res[cls_name] = {m: float(boot.auc(sub[m].to_numpy(np.float64), y)[0]) for m in methods}
    unc = f[f.cls == "uncalled"].copy()
    unc["pct_primary"] = unc.s_primary.rank(pct=True)
    unc["rank_primary"] = unc.s_primary.rank(ascending=False)
    unc["pct_engine"] = unc.eng_nlogP.rank(pct=True)
    hits = unc[unc.fold_id.isin(uncalled_hits)][["fold_id", "iptm", "s_primary", "rank_primary", "pct_primary", "eng_nlogP",
                                                  "pct_engine", "len_sum", "split"]]
    hi = unc[unc.hit][["fold_id", "iptm", "s_primary", "rank_primary", "pct_primary", "eng_nlogP", "pct_engine", "len_sum", "split"]]
    f.to_parquet(EV / f"folded_scored_{tag}.parquet", index=False)
    log(tag, "eval2", json.dumps(res, default=float))
    return res, hits.to_dict("records"), hi.to_dict("records"), int(len(unc))


def eval3(tag, adj, sc, fold_of, out_rows):
    covered = fold_of >= 0
    cfg = adj.config.to_numpy()
    tight = covered & (cfg == 0) & (adj.intergenic.to_numpy() <= 50)
    opp = covered & (cfg != 0)
    sel = tight | opp
    y = tight[sel]
    boot = Boot(adj.genome_code.to_numpy()[sel], draws=200)
    res = {}
    for m in ["primary", "primary_sym", "primary_ab", "w3", "glob", "shuf", "pmi_a1", "eng_nlogP", "raw_cos"]:
        if m not in sc:
            continue
        a, d = boot.auc(sc[m][sel], y)
        res[m] = a
        out_rows.append(dict(split=tag, method=m, n_pos=int(tight.sum()), n_neg=int(opp.sum()), auc=a,
                             auc_lo=float(np.quantile(d, .025)), auc_hi=float(np.quantile(d, .975))))
    # divergent and convergent separately against the tight co-directional pairs
    for nm, cmask in (("divergent", cfg == 1), ("convergent", cfg == 2)):
        s2 = tight | (covered & cmask)
        a = Boot(adj.genome_code.to_numpy()[s2], draws=1).auc(sc["primary"][s2], tight[s2])[0]
        out_rows.append(dict(split=tag, method=f"primary_vs_{nm}", n_pos=int(tight.sum()), n_neg=int((covered & cmask).sum()), auc=a))
    log(tag, "eval3", {k: round(v, 4) for k, v in res.items()})
    return res


n_genes_global = 0


def cmd_e123():
    global n_genes_global
    EV.mkdir(parents=True, exist_ok=True)
    adj = pd.read_parquet(PREP / "adj1.parquet")
    meta = pd.read_parquet(PREP / "genes_meta.parquet", columns=["gid"])
    n_genes_global = len(meta)
    hits4 = ["unc_002_r1", "unc_071_r2", "unc_069_r2", "unc_087_r1"]
    summary = {}
    rows1, rows2, rows3 = [], [], []
    cv = os.environ.get("GR_CV", "cv0,cv1,cv2,cv3,cv4").split(",")       # interim runs may name a subset
    for tag, splits in (("cv", cv), ("phylum", ["phylum"])):
        if not all((RUNS / s / "summary.json").exists() for s in splits):
            log("skip", tag, "(runs missing)")
            continue
        sc, fold_of = assemble(splits, adj, "base")
        log(tag, "assembled", int((fold_of >= 0).sum()), "adjacencies")
        # per-fold AUROC of primary and pmi_a1 on DefenseFinder vs no-gold (scale check for pooling)
        pf = {}
        for k in range(len(splits)):
            mk = fold_of == k
            p = adj.pos_DefenseFinder.to_numpy() & mk
            q = adj.gold_none.to_numpy() & mk
            sel = p | q
            if p.sum() >= 5:
                b = Boot(adj.genome_code.to_numpy()[sel], draws=1)
                pf[splits[k]] = dict(n_pos=int(p.sum()), primary=b.auc(sc["primary"][sel], p[sel])[0],
                                     pmi_a1=b.auc(sc["pmi_a1"][sel], p[sel])[0])
        summary[f"{tag}_per_fold_DF_nogold"] = pf
        eval1(tag, adj, sc, fold_of, rows1)
        res2, hits, hi, n_unc = eval2(tag, splits, rows2, hits4)
        summary[f"{tag}_eval2"] = dict(auc=res2, uncalled_n=n_unc, four_cofolders=hits, uncalled_iptm_ge_075=hi)
        summary[f"{tag}_eval3"] = eval3(tag, adj, sc, fold_of, rows3)
        np.savez(EV / f"assembled_{tag}.npz", fold_of=fold_of, **{k: v.astype(np.float32) for k, v in sc.items()})
    pd.DataFrame(rows1).to_csv(EV / "eval1.tsv", sep="\t", index=False)
    pd.DataFrame(rows2).to_csv(EV / "eval2.tsv", sep="\t", index=False)
    pd.DataFrame(rows3).to_csv(EV / "eval3.tsv", sep="\t", index=False)
    # run summaries (selection, epochs)
    for s in [f"cv{k}" for k in range(5)] + ["phylum"]:
        f = RUNS / s / "summary.json"
        if f.exists():
            j = json.load(open(f))
            summary[f"run_{s}"] = dict(selected=j.get("selected_config"), selection=j.get("selection"),
                                       genes=j.get("genes"), pairs=j.get("pairs_offset1"),
                                       best={m: (r["best_val_auc"], r["best_epoch"]) for m, r in j["training"].items()},
                                       seconds=j.get("seconds"))
    json.dump(summary, open(EV / "eval_summary.json", "w"), indent=1, default=float)
    log("done e123")


# ------------------------------------------------------------------------------------- evaluation 4: loci
ENGINE = RD / "engine/results_dpann_repair/dpann__mmseqs__seq__e3c50s75_rescue3__w3"


@njit(cache=True)
def runs_on(on, starts, ends, gap):
    """Maximal runs of boundaries that are on inside each contig [start, end), bridging up to `gap`
    consecutive off boundaries (the engine's run rule at one level). Inclusive gene index ranges."""
    out_s, out_e = [], []
    for c in range(len(starts)):
        s, e = starts[c], ends[c]
        k = s
        while k < e - 1:
            if on[k]:
                k0 = k
                last = k
                k += 1
                while k < e - 1:
                    if on[k]:
                        last = k
                        k += 1
                    elif k - last <= gap:
                        k += 1
                    else:
                        break
                out_s.append(k0)
                out_e.append(last + 1)
                k = last + 1
            else:
                k += 1
    return np.array(out_s, dtype=np.int64), np.array(out_e, dtype=np.int64)


def cmd_loci():
    meta = pd.read_parquet(PREP / "genes_meta.parquet", columns=["gid", "genome_code", "contig_code", "position_index"])
    n = len(meta)
    cc = meta.contig_code.to_numpy()
    cs = np.r_[0, np.flatnonzero(np.diff(cc)) + 1].astype(np.int64)
    ce = np.r_[cs[1:], n].astype(np.int64)
    contigs = pd.read_parquet(RD / "ws_dpann/package/universe/contigs.parquet").sort_values("first_gid").reset_index(drop=True)
    assert np.array_equal(contigs.first_gid.to_numpy(), cs)
    genome_of = np.repeat(contigs.genome.to_numpy(), contigs.n_genes.to_numpy())
    contig_of = np.repeat(contigs.contig.to_numpy(), contigs.n_genes.to_numpy())
    pos = meta.position_index.to_numpy()
    adj = pd.read_parquet(PREP / "adj1.parquet", columns=["i", "genome_code", "pos_DefenseFinder", "gold_none"])
    z = np.load(EV / "assembled_cv.npz")
    fold_of = z["fold_of"]
    info = {}
    on = {}
    for model in ("primary", "w3"):
        thr = {}
        for k in range(5):
            ns = np.load(RUNS / f"cv{k}" / "null_scores.npz")[model]
            thr[k] = float(np.quantile(ns, 0.99))
        t_adj = np.array([thr[k] for k in range(5)])[np.maximum(fold_of, 0)]
        s = z[model]
        m = (fold_of >= 0) & (s > t_adj)
        o = np.zeros(n, dtype=bool)
        o[adj.i.to_numpy()[m]] = True
        on[f"critic_{model}"] = o
        info[f"critic_{model}"] = dict(thresholds=thr, adjacencies_on=int(m.sum()), adjacencies=int((fold_of >= 0).sum()),
                                       frac_on=float(m.mean()),
                                       frac_on_DF_within=float(m[adj.pos_DefenseFinder.to_numpy()].mean()),
                                       frac_on_nogold=float(m[adj.gold_none.to_numpy()].mean()))
        log(model, info[f"critic_{model}"])
    b = np.load(ENGINE / "boundary_support.npy")
    assert len(b) == n
    on["engine_c2"] = b >= 2
    on["union"] = on["critic_primary"] | on["engine_c2"]
    on["inter"] = on["critic_primary"] & on["engine_c2"]
    info["engine_c2"] = dict(adjacencies_on=int(on["engine_c2"].sum()))
    LO = Path(os.environ.get("GR_LOCI", RD / "grammar/loci"))
    for name, o in on.items():
        parts = []
        for gap, kind in ((0, "conserved"), (2, "conserved_g2")):
            s_idx, e_idx = runs_on(o, cs, ce, gap)
            parts.append(pd.DataFrame(dict(genome=genome_of[s_idx], contig=contig_of[s_idx], start=pos[s_idx], end=pos[e_idx],
                                           kind=kind, level=2.0, module_id=-1)))
        loci = pd.concat(parts, ignore_index=True)
        # the ribosomal check reads a unit@0.5 set; a copy of the gap-0 runs stands in and is not reported
        loci = pd.concat([loci, parts[0].assign(kind="unit", level=0.5)], ignore_index=True)
        d = LO / name
        d.mkdir(parents=True, exist_ok=True)
        loci.to_parquet(d / "loci_all_genomes.parquet", index=False)
        g0 = parts[0]
        cov = np.zeros(n, dtype=bool)
        s_idx, e_idx = runs_on(o, cs, ce, 0)
        for s_, e_ in zip(s_idx, e_idx):
            cov[s_:e_ + 1] = True
        info[name] = dict(info.get(name, {}), loci_gap0=int(len(g0)), loci_gap2=int(len(parts[1])),
                          genes_covered_gap0=int(cov.sum()), genes_covered_frac=float(cov.mean()),
                          locus_len_q50_q90_max=[float(x) for x in np.quantile(g0.end - g0.start + 1, [.5, .9, 1])] if len(g0) else [])
        json.dump(dict(arm=f"grammar_{name}", w=1), open(d / "summary.json", "w"))
        log(name, info[name])
    # engine reading restricted to the same two kinds at level 2 (like for like)
    el = pd.read_parquet(ENGINE / "loci_all_genomes.parquet")
    e2 = el[el.kind.isin(["conserved", "conserved_g2"]) & (el.level == 2.0)]
    d = LO / "engine_c2_table"
    d.mkdir(parents=True, exist_ok=True)
    pd.concat([e2, e2[e2.kind == "conserved"].assign(kind="unit", level=0.5)]).to_parquet(d / "loci_all_genomes.parquet", index=False)
    json.dump(dict(arm="engine_c2_table", w=3), open(d / "summary.json", "w"))
    json.dump(info, open(LO / "loci_info.json", "w"), indent=1, default=float)
    log("done loci")


# ------------------------------------------------------------------------------------- evaluation 5: what it learned
def cmd_learned():
    import duckdb
    import faiss
    import torch
    sys.path.insert(0, str(Path(__file__).parent))
    import train as TR
    threads = numba.get_num_threads()
    torch.set_num_threads(threads)
    faiss.omp_set_num_threads(threads)
    EMBF = TR.L2
    meta = pd.read_parquet(PREP / "genes_meta.parquet")
    genes = pd.read_parquet(RD / "ws_dpann/package/universe/genes.parquet", columns=["gid", "protein_id"]).sort_values("gid")
    flags = pd.read_parquet(RD / "ws_dpann/package/annotation_flags.parquet", columns=["gid", "has_pfam"]).sort_values("gid")
    gm = pd.read_parquet(PREP / "genomes_meta.parquet")
    adj = pd.read_parquet(PREP / "adj1.parquet")
    z = np.load(EV / "assembled_cv.npz")
    fold_of, prim = z["fold_of"], z["primary"].astype(np.float64)
    m = ((fold_of >= 0) & adj.unannotated_both.to_numpy() & adj.gold_none.to_numpy() & ~adj.eng_called.to_numpy()
         & ~adj.same_token.to_numpy())
    maxcos = float(os.environ.get("GR_LEARN_MAXCOS", "2"))      # exploratory: drop paralog-like pairs
    sfx = os.environ.get("GR_LEARN_SUFFIX", "")
    m &= adj.raw_cos.to_numpy() < maxcos
    d = adj[m].copy()
    d["score"] = prim[m]
    d["fold"] = fold_of[m]
    d["lo"] = np.minimum(d.ta, d.tb)
    d["hi"] = np.maximum(d.ta, d.tb)
    d["ord"] = gm.ord.to_numpy()[d.genome_code.to_numpy()]
    d["phylum"] = gm.phylum.to_numpy()[d.genome_code.to_numpy()]
    log("eligible adjacencies (unannotated both, no gold, engine-uncalled, distinct tokens)", len(d), "of", int((fold_of >= 0).sum()))
    agg = d.groupby(["lo", "hi"]).agg(n_inst=("score", "size"), n_genomes=("genome_code", "nunique"), n_orders=("ord", "nunique"),
                                       n_phyla=("phylum", "nunique"), mean_score=("score", "mean"), min_score=("score", "min"),
                                       folds=("fold", "nunique")).reset_index()
    nulls = {k: float(np.quantile(np.load(RUNS / f"cv{k}" / "null_scores.npz")["primary"], 0.99)) for k in range(5)}
    log("pair types", len(agg), "in >= 2 genomes", int((agg.n_genomes >= 2).sum()), "null q99 by fold", nulls)
    single = agg[agg.n_genomes == 1].sort_values("mean_score", ascending=False).head(10)
    top = agg[agg.n_genomes >= 2].sort_values("mean_score", ascending=False).head(30).reset_index(drop=True)
    top["type"] = [f"T{k + 1:02d}" for k in range(len(top))]
    inst = d.merge(top[["lo", "hi", "type"]], on=["lo", "hi"])
    L = meta.length.to_numpy()
    rows = []
    for t, g in inst.groupby("type", sort=True):
        rows.append(dict(type=t, len_a_med=float(np.median(L[g.a])), len_b_med=float(np.median(L[g.b])),
                         config_co=float((g.config == 0).mean()), intergenic_med=float(g.intergenic.median()),
                         raw_cos_med=float(g.raw_cos.median()), eng_G=int(g.eng_G.max()), eng_L=int(g.eng_L.max()),
                         eng_nlogP=float(g.eng_nlogP.max()), token_size_a=int(meta.token_size.to_numpy()[g.a].max()),
                         token_size_b=int(meta.token_size.to_numpy()[g.b].max()),
                         phyla=";".join(sorted(g.phylum.unique())), orders=";".join(sorted(g.ord.unique())),
                         example=f"{int(g.a.iloc[0])}->{int(g.b.iloc[0])}"))
    top = top.merge(pd.DataFrame(rows), on="type")
    # (u, v) space: one common critic (cv0 primary) embeds every instance, so the spaces are comparable
    s0 = json.load(open(RUNS / "cv0" / "summary.json"))
    role = meta.role_cv0.to_numpy()
    Xg, _ = TR.features(meta.genome_code.to_numpy(np.int64), role <= 1, "genome")
    model = TR.Critic(Xg.shape[1], TR.CONFIGS[s0["selected_config"]])
    model.load_state_dict(torch.load(RUNS / "cv0" / "models" / f"sel_{s0['selected_config']}.pt"))
    model.eval()
    with torch.no_grad():
        ua = model.u(torch.from_numpy(Xg[inst.a.to_numpy()].astype(np.float32))).numpy()
        vb = model.v(torch.from_numpy(Xg[inst.b.to_numpy()].astype(np.float32))).numpy()
    del Xg
    ua /= np.linalg.norm(ua, axis=1, keepdims=True)
    vb /= np.linalg.norm(vb, axis=1, keepdims=True)
    emb = np.hstack([ua, vb])
    tv = np.vstack([emb[(inst.type == t).to_numpy()].mean(0) for t in top.type])
    tv /= np.linalg.norm(tv, axis=1, keepdims=True)
    from scipy.cluster.hierarchy import fcluster, linkage
    Z = linkage(tv, method="average", metric="cosine")
    top["uv_cluster"] = fcluster(Z, t=0.5, criterion="distance")
    sim = tv @ tv.T
    top["uv_nearest_type"] = [top.type[np.argsort(-sim[k])[1]] for k in range(len(top))]
    top["uv_nearest_cos"] = [float(np.sort(sim[k])[-2]) for k in range(len(top))]
    # nearest annotated neighbours (raw L2 vectors, cosine) of each member gene; observed Pfam names
    X = np.load(EMBF, mmap_mode="r")
    ann = np.flatnonzero(flags.has_pfam.to_numpy())
    index = faiss.IndexFlatIP(X.shape[1])
    for s_ in range(0, len(ann), 500_000):
        index.add(np.ascontiguousarray(X[ann[s_:s_ + 500_000]]))
    q_genes = np.unique(np.r_[inst.a.to_numpy(), inst.b.to_numpy()])
    D, I = index.search(np.ascontiguousarray(X[q_genes]), 10)
    nb = ann[I]
    pid = genes.protein_id.to_numpy()
    con = duckdb.connect(str(RD / "ws_dpann/input/sharur.duckdb"), read_only=True)
    con.execute(f"PRAGMA threads={threads}")
    ids = pd.DataFrame({"protein_id": np.unique(pid[nb.ravel()])})
    con.register("ids", ids)
    pf = con.execute("SELECT a.protein_id, a.name FROM annotations a JOIN ids USING (protein_id) WHERE a.source = 'pfam'").df()
    names = pf.groupby("protein_id").name.apply(lambda x: "+".join(sorted(set(x)))).to_dict()
    nbr = {int(g): (nb[k], D[k]) for k, g in enumerate(q_genes)}

    def side(genes_list):
        from collections import Counter
        c, cos = Counter(), []
        for g in genes_list:
            nn_, dd = nbr[int(g)]
            cos.append(float(np.median(dd)))
            for x in nn_:
                for nm in names.get(pid[x], "").split("+"):
                    if nm:
                        c[nm] += 1
        tot = 10 * len(genes_list)
        return "; ".join(f"{k} {v}/{tot}" for k, v in c.most_common(3)), float(np.median(cos))

    da, ca, db, cb = [], [], [], []
    for t in top.type:
        g = inst[inst.type == t]
        x, y = side(g.a.unique())
        da.append(x)
        ca.append(y)
        x, y = side(g.b.unique())
        db.append(x)
        cb.append(y)
    top["nbr_domains_a"], top["nbr_cos_a"], top["nbr_domains_b"], top["nbr_cos_b"] = da, ca, db, cb
    top.to_csv(EV / f"learned_top30{sfx}.tsv", sep="\t", index=False)
    single.to_csv(EV / f"learned_top_single_genome{sfx}.tsv", sep="\t", index=False)
    inst[["type", "a", "b", "genome_code", "config", "intergenic", "score", "fold"]].assign(
        protein_a=pid[inst.a.to_numpy()], protein_b=pid[inst.b.to_numpy()]).to_csv(EV / f"learned_top30_instances{sfx}.tsv", sep="\t", index=False)
    # score distribution of the eligible set against the null threshold
    thr = np.array([nulls[k] for k in range(5)])[d.fold.to_numpy()]
    json.dump(dict(max_raw_cos=maxcos, eligible=int(len(d)), eligible_above_null_q99=int((d.score.to_numpy() > thr).sum()),
                   pair_types=int(len(agg)), pair_types_ge2_genomes=int((agg.n_genomes >= 2).sum()),
                   clusters=int(top.uv_cluster.nunique()), null_q99=nulls),
              open(EV / f"learned_summary{sfx}.json", "w"), indent=1)
    log("done learned", top[["type", "n_genomes", "mean_score", "uv_cluster", "nbr_domains_a", "nbr_domains_b"]].to_string())


# ------------------------------------------------------------------------------------- exploratory: interaction-only score
def cmd_dc():
    """Exploratory (added after the evaluation-pipeline test, before any real result): the critic's
    interaction term with per-gene marginal terms removed, s_dc(a, b) = scale * (U_a - mean_g U) . (V_b - mean_g V),
    tower outputs centred over the genes of each held-out genome (exactly the score minus its row and column means
    against the genome's genes). Evaluations 1-3 are recomputed for these scores."""
    import torch
    sys.path.insert(0, str(Path(__file__).parent))
    import train as TR
    torch.set_num_threads(numba.get_num_threads())
    meta = pd.read_parquet(PREP / "genes_meta.parquet")
    gc = meta.genome_code.to_numpy(np.int64)
    strand = meta.strand.to_numpy(np.int64)
    adj = pd.read_parquet(PREP / "adj1.parquet")
    n = len(meta)
    rows1, rows2, rows3 = [], [], []
    cv = os.environ.get("GR_CV", "cv0,cv1,cv2,cv3,cv4").split(",")
    for tag, splits in (("cv", cv), ("phylum", ["phylum"])):
        if not (EV / f"assembled_{tag}.npz").exists():
            continue
        z = np.load(EV / f"assembled_{tag}.npz")
        sc = {k: z[k].astype(np.float64) for k in z.files if k != "fold_of"}
        fold_of = z["fold_of"]
        for m in ("primary", "w3", "shuf", "glob"):
            sc[f"dc_{m}"] = np.full(len(adj), np.nan)
        fdf = pd.read_parquet(EV / f"folded_scored_{tag}.parquet")
        for m in ("primary", "w3", "shuf", "glob"):
            fdf[f"dc_{m}"] = np.nan
        for s in splits:
            summ = json.load(open(RUNS / s / "summary.json"))
            cfg = TR.CONFIGS[summ["selected_config"]]
            role = meta[f"role_{s}"].to_numpy()
            uv = np.load(RUNS / s / "uv_primary.npz")
            test = uv["genes"]
            if "GR_RUNS" not in os.environ:
                assert np.array_equal(test, np.flatnonzero(role == 2))
            row = np.full(n, -1, dtype=np.int64)
            row[test] = np.arange(len(test))
            tw = {}
            tw["primary"] = (uv["U"].astype(np.float32), uv["V"].astype(np.float32), float(uv["scale"]))
            for mode, models in (("genome", ("w3", "shuf")), ("global", ("glob",))):
                X, _ = TR.features(gc, role <= 1, mode)
                for m in models:
                    model = TR.Critic(X.shape[1], cfg)
                    model.load_state_dict(torch.load(RUNS / s / "models" / f"{m}.pt"))
                    model.eval()
                    U, V = TR.towers(model, X, test)
                    tw[m] = (U, V, model.scale)
                del X
            gt = gc[test]
            st = np.r_[0, np.flatnonzero(np.diff(gt)) + 1]
            cnt = np.diff(np.r_[st, len(gt)])[:, None]
            ai = np.flatnonzero((row[adj.a.to_numpy()] >= 0) & (row[adj.b.to_numpy()] >= 0))
            a, b, sym = adj.a.to_numpy()[ai], adj.b.to_numpy()[ai], adj.config.to_numpy()[ai] != 0
            fm = (row[fdf.g0.to_numpy()] >= 0) & (row[fdf.g1.to_numpy()] >= 0)
            fa, fb, fs = TR.orient(fdf.g0.to_numpy()[fm], fdf.g1.to_numpy()[fm], strand)
            for m, (U, V, scale) in tw.items():
                gi = np.repeat(np.arange(len(st)), cnt.ravel())
                Uc = U - (np.add.reduceat(U, st, axis=0) / cnt)[gi]
                Vc = V - (np.add.reduceat(V, st, axis=0) / cnt)[gi]

                def sc_pairs(x, y, sy):
                    s_ab = np.einsum("ij,ij->i", Uc[row[x]], Vc[row[y]]) * scale
                    s_ba = np.einsum("ij,ij->i", Uc[row[y]], Vc[row[x]]) * scale
                    return np.where(sy, 0.5 * (s_ab + s_ba), s_ab)
                sc[f"dc_{m}"][ai] = sc_pairs(a, b, sym)
                fdf.loc[fm, f"dc_{m}"] = sc_pairs(fa, fb, fs)
            log(tag, s, "dc scores done")
        for m in ("primary", "shuf"):
            sc[f"marg_{m}"] = sc[m] - sc[f"dc_{m}"]       # the per-gene (marginal) part of the score
        mlist = ["dc_primary", "dc_w3", "dc_shuf", "dc_glob", "marg_primary", "marg_shuf", "primary", "shuf", "pmi_a1",
                 "eng_nlogP", "raw_cos"]
        eval1(tag, adj, sc, fold_of, rows1, methods=mlist,
              strata_keep=("all", "never_w3", "1-4_w3", ">=5_w3", "singleton_either", "distinct_tokens"), ref="dc_primary")
        fdf["hit"] = fdf.iptm >= 0.75
        for cls_name, mk in (("cognate+uncalled", np.ones(len(fdf), bool)), ("uncalled", (fdf.cls == "uncalled").to_numpy())):
            sub = fdf[mk]
            y = sub.hit.to_numpy()
            boot = Boot(np.arange(len(sub)), draws=2000)
            for m in ["dc_primary", "dc_w3", "dc_shuf", "dc_glob", "s_primary", "eng_nlogP"]:
                a_, d_ = boot.auc(sub[m].to_numpy(np.float64), y)
                rows2.append(dict(split=tag, set=cls_name, method=m, n=int(len(sub)), n_hit=int(y.sum()), auc=a_,
                                  auc_lo=float(np.nanquantile(d_, .025)), auc_hi=float(np.nanquantile(d_, .975))))
        cfg_ = adj.config.to_numpy()
        cov = fold_of >= 0
        tight = cov & (cfg_ == 0) & (adj.intergenic.to_numpy() <= 50)
        opp = cov & (cfg_ != 0)
        sel = tight | opp
        bt = Boot(adj.genome_code.to_numpy()[sel], draws=200)
        for m in ("dc_primary", "dc_w3", "dc_shuf", "dc_glob"):
            a_, d_ = bt.auc(sc[m][sel], tight[sel])
            rows3.append(dict(split=tag, method=m, auc=a_, auc_lo=float(np.quantile(d_, .025)), auc_hi=float(np.quantile(d_, .975))))
        fdf.to_parquet(EV / f"folded_scored_dc_{tag}.parquet", index=False)
        np.savez(EV / f"dc_{tag}.npz", **{k: sc[k].astype(np.float32) for k in sc if k.startswith("dc_")})
    pd.DataFrame(rows1).to_csv(EV / "eval1_dc.tsv", sep="\t", index=False)
    pd.DataFrame(rows2).to_csv(EV / "eval2_dc.tsv", sep="\t", index=False)
    pd.DataFrame(rows3).to_csv(EV / "eval3_dc.tsv", sep="\t", index=False)
    log("done dc")


if __name__ == "__main__":
    import numba
    numba.set_num_threads(int(sys.argv[2]))
    {"e123": cmd_e123, "loci": cmd_loci, "learned": cmd_learned, "dc": cmd_dc}[sys.argv[1]]()
