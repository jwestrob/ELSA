#!/usr/bin/env python
"""Task 1 of round 2: what are the unlabelled locus-search hits? Also the rate-constrained selection.

  evidence.py prep      evidence tables -> out2/evidence/: raw DefenseFinder profile hits per gene (the
                        per-gene `defensefinder` annotation rows of the Sharur DPANN copy), the map
                        profile -> system types (profile_names of curated `defense_systems` rows), and
                        the engine's conserved loci (dpann__mmseqs__seq__e3c50s75__w3) in gid space
  evidence.py analyse   unlabelled hits at E <= 0.01 of M0, M2, S2, S3 and M3 grow round 3 on even-genome
                        DefenseFinder queries, against span-matched random windows on the same contigs
  evidence.py rate      rate-constrained selection (results/criterion2.json) from the round-2 rate grid

Everything reported for an unlabelled hit is an observation (profile hits present, conserved-locus
overlap, slots filled); no system name is assigned to a hit.
Supported (predeclared in criterion2.json): a_own (a gene of the window carries a raw DefenseFinder
profile hit whose profile occurs in a curated system of the query's system_type) or c_module (the
window overlaps an engine conserved locus with a module of a conserved locus overlapping the query's
members).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench  # noqa: E402
import locus_search as ls  # noqa: E402
from locus_search import log  # noqa: E402

OUT1 = Path(os.environ.get("LS_OUT1", ls.RD / "locus_search/out"))
OUT2 = Path(os.environ.get("LS_OUT2", ls.RD / "locus_search/out2"))
SHARUR = ls.RD / "ws_dpann/input/sharur.duckdb"
ENGINE = ls.RD / "engine/results_dpann_v5/dpann__mmseqs__seq__e3c50s75__w3"
EVD = OUT2 / "evidence"
N_CTRL = 10
RATE_R = 40


# =========================================================================== prep
def prep(a):
    import duckdb
    EVD.mkdir(parents=True, exist_ok=True)
    db = ls.Database(vectors=None)
    callers = bench.load_gold(db)
    C = callers["DefenseFinder"]
    genes = pd.read_parquet(ls.PKG / "universe/genes.parquet", columns=["gid", "protein_id"])
    con = duckdb.connect(str(SHARUR), read_only=True)
    con.execute(f"SET threads={a.threads}")
    A = con.execute("select protein_id, accession, min(evalue) as evalue, max(score) as score from annotations "
                    "where source = 'defensefinder' group by 1, 2").df()
    D = con.execute("select system_type, profile_names from defense_systems").df()
    n_rows = con.execute("select count(*) from annotations where source = 'defensefinder'").fetchone()[0]
    A["gid"] = A.protein_id.map(dict(zip(genes.protein_id, genes.gid)))
    info = dict(defensefinder_rows=int(n_rows), protein_profile_pairs=int(len(A)),
                proteins=int(A.protein_id.nunique()), unmapped_proteins=int(A[A.gid.isna()].protein_id.nunique()),
                profiles=int(A.accession.nunique()), curated_systems=int(len(D)))
    A = A[A.gid.notna()].copy()
    A["gid"] = A.gid.astype(np.int64)
    P = D.assign(p=D.profile_names.str.split(",")).explode("p")
    P["p"] = P.p.str.strip()
    ptypes = P.groupby("p").system_type.apply(lambda s: sorted(set(s))).to_dict()
    info["profiles_in_curated_systems"] = int(len(ptypes))
    info["raw_profiles_in_curated_systems"] = int(A.accession.isin(list(ptypes)).groupby(A.accession).first().sum())
    A.to_parquet(EVD / "df_raw_hits.parquet", index=False)
    json.dump(ptypes, open(EVD / "profile_types.json", "w"), indent=0)
    # per type: sorted gids carrying a raw hit to a profile of that type
    arr = {"df_any": np.unique(A.gid.to_numpy()),
           "df_any_e5": np.unique(A.gid[A.evalue <= 1e-5].to_numpy())}
    types = list(C["tlabels"])
    for code, t in enumerate(types):
        prof = [p for p, ts in ptypes.items() if t in ts]
        arr[f"type_{code}"] = np.unique(A.gid[A.accession.isin(prof)].to_numpy())
    # gold DefenseFinder members (any member count) and single-member instances per type
    arr["df_member"] = np.unique(np.concatenate(C["mem"]))
    single = C["nmem"] == 1
    for code in range(len(types)):
        rows = np.flatnonzero(single & (C["tcode"] == code))
        arr[f"single_{code}"] = np.unique(np.concatenate([C["mem"][r] for r in rows])) if len(rows) else np.empty(0, np.int64)
    # engine conserved loci in gid space, one non-overlapping set per level
    L = pd.read_parquet(ENGINE / "loci_all_genomes.parquet")
    L = L[L.kind == "conserved"]
    cf = db.contigs.reset_index()[["genome", "contig", "first_gid"]]
    L = L.merge(cf, on=["genome", "contig"], how="left")
    assert L.first_gid.notna().all()
    L["sg"] = (L.first_gid + L.start).astype(np.int64)
    L["eg"] = (L.first_gid + L.end).astype(np.int64)
    for lv, g in L.groupby("level"):
        g = g.sort_values("sg")
        assert np.all(g.sg.to_numpy()[1:] > g.eg.to_numpy()[:-1]), "conserved loci of one level overlap"
        arr[f"cons_s_{lv:g}"] = g.sg.to_numpy()
        arr[f"cons_e_{lv:g}"] = g.eg.to_numpy()
        arr[f"cons_m_{lv:g}"] = g.module_id.to_numpy(np.int64)
    cov = np.zeros(db.N, np.int64)
    g2 = L[L.level == 2.0]
    d = np.zeros(db.N + 1, np.int64)
    np.add.at(d, g2.sg.to_numpy(), 1)
    np.add.at(d, g2.eg.to_numpy() + 1, -1)
    cov = np.cumsum(d)[:db.N] > 0
    arr["cons_cov"] = cov
    info["conserved_loci"] = {f"{k:g}": int(v) for k, v in L.groupby("level").size().items()}
    info["genes_in_conserved_loci"] = float(cov.mean())
    info["genes_with_raw_df_hit"] = float(len(arr["df_any"]) / db.N)
    np.savez(EVD / "evidence.npz", **arr, types=np.array(types))
    json.dump(info, open(EVD / "prep.json", "w"), indent=1)
    log("evidence prep", info)


class Evidence:
    def __init__(self, db):
        z = np.load(EVD / "evidence.npz")
        self.z = {k: z[k] for k in z.files}
        self.levels = sorted(float(k[7:]) for k in self.z if k.startswith("cons_s_"))
        self.cum_cov = np.r_[0, np.cumsum(self.z["cons_cov"])]
        mem = np.zeros(db.N, np.int64)
        mem[self.z["df_member"]] = 1
        self.cum_mem = np.r_[0, np.cumsum(mem)]
        self.db = db

    @staticmethod
    def any_in(sorted_gids, a, b):
        return (np.searchsorted(sorted_gids, b, "right") - np.searchsorted(sorted_gids, a, "left")) > 0

    def modules(self, a, b):
        out = set()
        for lv in self.levels:
            S, E, M = self.z[f"cons_s_{lv:g}"], self.z[f"cons_e_{lv:g}"], self.z[f"cons_m_{lv:g}"]
            i0 = np.searchsorted(E, a, "left")
            i1 = np.searchsorted(S, b, "right")
            out.update(M[i0:i1].tolist())
        out.discard(-1)
        return out

    def query_modules(self, members):
        out = set()
        for g in members:
            out |= self.modules(int(g), int(g))
        return out

    def windows(self, a, b, tcode, qmods):
        """Evidence flags for windows [a, b] (gid, inclusive) of queries of type tcode with modules qmods."""
        a = np.asarray(a, np.int64)
        b = np.asarray(b, np.int64)
        L = b - a + 1
        df_any = self.any_in(self.z["df_any"], a, b)
        df_any_e5 = self.any_in(self.z["df_any_e5"], a, b)
        n_own = np.array([np.searchsorted(self.z[f"type_{t}"], y, "right") - np.searchsorted(self.z[f"type_{t}"], x, "left")
                          for t, x, y in zip(tcode, a, b)], np.int64)
        df_own = n_own > 0
        cov = (self.cum_cov[b + 1] - self.cum_cov[a]) / L
        cmod = np.array([len(self.modules(int(x), int(y)) & m) > 0 for x, y, m in zip(a, b, qmods)], bool)
        return pd.DataFrame(dict(df_any=df_any, df_any_e5=df_any_e5, df_own=df_own, own_ge2=n_own >= 2, cons_any=cov > 0,
                                 cons_half=cov >= 0.5, c_module=cmod, supported=df_own | cmod))

    def controls(self, a, b, rng, n=N_CTRL, tries=400):
        """Span-matched windows on the same contig that contain no gold member of the caller."""
        db = self.db
        rows = []
        for i, (x, y) in enumerate(zip(a, b)):
            L = y - x + 1
            c = db.cc[x]
            f = int(db.c_first[c])
            m = int(db.c_n[c])
            if m < L:
                continue
            st = rng.integers(f, f + m - L + 1, size=tries)
            ok = (self.cum_mem[st + L] - self.cum_mem[st]) == 0
            st = st[ok][:n]
            rows.extend((i, int(s), int(s + L - 1)) for s in st)
        return pd.DataFrame(rows, columns=["hit", "a", "b"])


def attach_db():
    db = ls.Database(vectors=None)
    cf = db.contigs.sort_index()
    db.c_first = np.zeros(int(cf.index.max()) + 1, np.int64)
    db.c_n = np.zeros_like(db.c_first)
    db.c_first[cf.index.to_numpy()] = cf.first_gid.to_numpy()
    db.c_n[cf.index.to_numpy()] = cf.n_genes.to_numpy()
    return db


def classify(ev, H, queries, callers, rng, controls=True):
    """H: hits with qid, start_gid, end_gid. Returns per-hit flags and (optionally) per-control flags."""
    C = callers["DefenseFinder"]
    qrow = queries.set_index("qid")
    qm = {}
    for qid in np.unique(H.qid):
        qm[int(qid)] = ev.query_modules(C["mem"][int(qrow.row[qid])])
    tc = qrow.tcode.loc[H.qid].to_numpy()
    mods = [qm[int(q)] for q in H.qid]
    F = ev.windows(H.start_gid.to_numpy(), H.end_gid.to_numpy(), tc, mods)
    F.index = H.index
    if not controls:
        return F, None
    K = ev.controls(H.start_gid.to_numpy(), H.end_gid.to_numpy(), rng)
    G = ev.windows(K.a.to_numpy(), K.b.to_numpy(), tc[K.hit.to_numpy()], [mods[i] for i in K.hit])
    G["hit"] = K.hit.to_numpy()
    G["a"], G["b"] = K.a.to_numpy(), K.b.to_numpy()
    ha, hb = H.start_gid.to_numpy()[K.hit.to_numpy()], H.end_gid.to_numpy()[K.hit.to_numpy()]
    G["overlaps_hit"] = (G.a.to_numpy() <= hb) & (G.b.to_numpy() >= ha)
    return F, G


# =========================================================================== analyse
def analyse(a):
    t0 = time.time()
    db = attach_db()
    callers = bench.load_gold(db)
    queries = pd.read_parquet(OUT1 / "queries.parquet")
    M1 = pd.read_parquet(OUT1 / "final_metrics.parquet", columns=["qid", "method", "round", "n_rel"])
    qset = set(queries[(queries.caller == "DefenseFinder") & (queries.split == "even")].qid) & \
        set(M1[(M1.method == "M0") & (M1.n_rel > 0)].qid)
    log(f"{len(qset):,} even-genome DefenseFinder queries with n_rel > 0")
    cols = ["qid", "start_gid", "end_gid", "score", "evalue", "label", "match_row", "n_slots_filled"]
    parts = []
    r1 = pd.read_parquet(OUT1 / "final_hits_E10.parquet", filters=[("evalue", "<=", 0.01)])
    for meth, rnd, name in (("M0", 1, "M0"), ("M2", 1, "M2"), ("M3_grow2", 3, "M3grow_r3"), ("M3_refine", 3, "M3refine_r3")):
        h = r1[(r1.method == meth) & (r1["round"] == rnd)][cols]
        parts.append(h.assign(method=name))
    s = pd.read_parquet(OUT1 / "final_hits_E10_seqctl.parquet", filters=[("evalue", "<=", 0.01)])
    parts.append(s[s.method == "S2"][cols].assign(method="S2"))
    parts.append(s[s.method == "S1"][cols].assign(method="S1"))
    f = pd.read_parquet(OUT2 / "final_dfall_hits.parquet", filters=[("evalue", "<=", 0.01)])
    parts.append(f[f.method == "S3"][cols].assign(method="S3"))
    H = pd.concat(parts, ignore_index=True)
    H = H[H.qid.isin(qset)].reset_index(drop=True)
    H = H.merge(queries[["qid", "n_members", "tcode", "type", "genome_code"]], on="qid")
    H["completeness"] = H.n_slots_filled / H.n_members
    U = H[H.label == 0].reset_index(drop=True)
    log(f"hits at E <= 0.01: {len(H):,}; unlabelled {len(U):,}; {time.time() - t0:.0f}s")
    ev = Evidence(db)
    rng = np.random.default_rng(17)
    F, G = classify(ev, U, queries, callers, rng)
    U = pd.concat([U, F], axis=1)
    G["method"] = U.method.to_numpy()[G.hit.to_numpy()]
    U.to_parquet(OUT2 / "unlabelled_hits_E0.01.parquet", index=False)
    G.to_parquet(OUT2 / "unlabelled_controls_E0.01.parquet", index=False)
    flags = ["df_any", "df_any_e5", "df_own", "own_ge2", "cons_any", "cons_half", "c_module", "supported"]
    nq = len(qset)
    res = dict(queries=nq, prep=json.load(open(EVD / "prep.json")), methods={})
    for m, g in U.groupby("method"):
        gc = G[G.method == m]
        gn = gc[~gc.overlaps_hit]
        d = dict(unlabelled_hits=int(len(g)), per_query=len(g) / nq, controls=int(len(gc)),
                 hits_with_control=int(gc.hit.nunique()), controls_overlapping_their_hit=float(gc.overlaps_hit.mean()),
                 single_slot_share=float((g.n_slots_filled == 1).mean()),
                 queries_with_unlabelled=int(g.qid.nunique()),
                 top10_query_share=float(g.qid.value_counts().head(10).sum() / len(g)),
                 type_share_AbiE=float((g.type == "AbiE").mean()), type_share_RM=float(g.type.str.startswith("RM_").mean()))
        # bootstrap over query genomes for the supported fraction
        for fl in flags:
            d[fl] = dict(hits=float(g[fl].mean()), control=float(gc[fl].mean()) if len(gc) else None,
                         excess=float(g[fl].mean() - gc[fl].mean()) if len(gc) else None,
                         control_not_overlapping_hit=float(gn[fl].mean()) if len(gn) else None)
        d["supported_share_ci"] = ls.boot_mean(g.supported.astype(float), g.genome_code)
        d["unsupported_per_query"] = float((~g.supported).sum() / nq)
        d["supported_per_query"] = float(g.supported.sum() / nq)
        d["completeness_unlabelled"] = dict(mean=float(g.completeness.mean()),
                                            frac_full=float((g.completeness >= 1).mean()),
                                            q25_50_75=np.quantile(g.completeness, [.25, .5, .75]).tolist())
        c = H[(H.method == m) & (H.label == 1)]
        d["completeness_correct"] = dict(n=int(len(c)), mean=float(c.completeness.mean()),
                                         frac_full=float((c.completeness >= 1).mean()))
        d["span_genes_median"] = float((g.end_gid - g.start_gid + 1).median())
        d["completeness_by_support"] = {str(k): float(v) for k, v in g.groupby("supported").completeness.mean().items()}
        d["types_top"] = g.type.value_counts().head(8).to_dict()
        res["methods"][m] = d
        log(m, json.dumps({k: (v if not isinstance(v, dict) else {kk: (round(vv, 3) if isinstance(vv, float) else vv)
                                                                   for kk, vv in v.items()}) for k, v in d.items()
                           if k in ("unlabelled_hits", "per_query", "df_own", "c_module", "supported")}))
    # same-type-other hits (label 3): single-member same-type instance overlap
    L3 = H[H.label == 3]
    s3 = {}
    for m, g in L3.groupby("method"):
        ov = np.array([Evidence.any_in(ev.z[f"single_{t}"], x, y) for t, x, y in zip(g.tcode, g.start_gid, g.end_gid)], bool)
        s3[m] = dict(hits=int(len(g)), per_query=len(g) / nq, overlap_single_member_same_type=float(ov.mean()))
    res["sametype_other_E0.01"] = s3
    # observed raw profiles on unlabelled hits (top profiles per method; observations only)
    raw = pd.read_parquet(EVD / "df_raw_hits.parquet")
    rg = raw.groupby("gid").accession.apply(list)
    for m, g in U.groupby("method"):
        cnt = {}
        for x, y in zip(g.start_gid, g.end_gid):
            seen = set()
            for gg in range(x, y + 1):
                seen.update(rg.get(gg, []))
            for p in seen:
                cnt[p] = cnt.get(p, 0) + 1
        res["methods"][m]["raw_profiles_top"] = dict(sorted(cnt.items(), key=lambda t: -t[1])[:10])
    json.dump(res, open(OUT2 / "unlabelled_summary.json", "w"), indent=1, default=float)
    log(f"done analyse {time.time() - t0:.0f}s")


# =========================================================================== rate-constrained selection
def rate(a):
    t0 = time.time()
    db = attach_db()
    callers = bench.load_gold(db)
    queries = pd.read_parquet(OUT1 / "queries.parquet")
    ev = Evidence(db)
    rng = np.random.default_rng(17)
    import fastnull
    curves = fastnull.learn_curves(OUT2, "gum", RATE_R, 1.0)
    res = {"E_model": "Gumbel on 40 per-replicate null maxima + per-method calibration curve (M1 -> M2 curve, S1 -> S2 curve); "
                      "M3 candidates: round-1 200-replicate empirical E",
           "curve_ratio_at_E0.01": {m: float(c[fastnull.J001] / fastnull.XGRID[fastnull.J001]) for m, c in curves.items()}}
    tabs = {}
    for split in ("odd", "even"):
        f = OUT2 / f"rategrid_{split}_metrics.parquet"
        if not f.exists():
            continue
        M = pd.read_parquet(f)
        M = M[M.n_rel > 0].drop(columns=["n_unlabelled_E0.01"])
        # E-values: Gumbel fitted to the 40 per-replicate null maxima, then the per-method calibration curve
        # (fastnull.learn_curves; M1 uses the M2 curve, S1 the S2 curve)
        z = np.load(OUT2 / f"rategrid_{split}_tails.npz", allow_pickle=True)
        qi = {int(q): i for i, q in enumerate(z["qid"])}
        ci = {str(c): i for i, c in enumerate(z["cfg"])}
        tails = z["tails"]
        H = pd.read_parquet(OUT2 / f"rategrid_{split}_hits.parquet", filters=[("label", "==", 0)])
        H = H[H.qid.isin(M.qid)].reset_index(drop=True)
        Ecal = np.empty(len(H))
        for (q, c), g in H.groupby(["qid", "cfg"]).groups.items():
            fam = c[:2]
            curve = curves[{"M1": "M2", "S1": "S2"}.get(fam, fam)]
            e = fastnull.evalue_gumbel(H.score.to_numpy()[g], tails[qi[int(q)], ci[c]].astype(np.float64), RATE_R)
            Ecal[g] = fastnull.apply_curve(e, curve)
        H["E_cal"] = Ecal
        H = H[H.E_cal <= 0.01].reset_index(drop=True)
        n_unl = H.groupby(["qid", "cfg"]).size().rename("n_unlabelled_E0.01")
        M = M.merge(n_unl.reset_index(), on=["qid", "cfg"], how="left").fillna({"n_unlabelled_E0.01": 0})
        F, _ = classify(ev, H, queries, callers, rng, controls=False)
        H = pd.concat([H, F], axis=1)
        sup = H.groupby(["qid", "cfg"]).agg(n_unl_supported=("supported", "sum"), n_unl_strict=("own_ge2", "sum"))
        M = M.merge(sup.reset_index(), on=["qid", "cfg"], how="left").fillna({"n_unl_supported": 0, "n_unl_strict": 0})
        M["n_unl_unsupported"] = M["n_unlabelled_E0.01"] - M.n_unl_supported
        M["n_unl_unsup_strict"] = M["n_unlabelled_E0.01"] - M.n_unl_strict
        # M3 candidates from round 1 (200-replicate empirical E)
        R1 = pd.read_parquet(OUT1 / "final_metrics.parquet")
        R1 = R1[(R1.caller == "DefenseFinder") & (R1.split == split) & R1.method.str.startswith("M3") & (R1.n_rel > 0)]
        R1["cfg"] = R1.method + "_r" + R1["round"].astype(str)
        h1 = pd.read_parquet(OUT1 / "final_hits_E10.parquet", filters=[("evalue", "<=", 0.01), ("label", "==", 0)])
        h1 = h1[h1.method.str.startswith("M3") & h1.qid.isin(R1.qid)].reset_index(drop=True)
        h1["cfg"] = h1.method + "_r" + h1["round"].astype(str)
        F1, _ = classify(ev, h1, queries, callers, rng, controls=False)
        h1 = pd.concat([h1, F1], axis=1)
        sup1 = h1.groupby(["qid", "cfg"]).agg(n_unl_supported=("supported", "sum"), n_unl_strict=("own_ge2", "sum"))
        R1 = R1.merge(sup1.reset_index(), on=["qid", "cfg"], how="left").fillna({"n_unl_supported": 0, "n_unl_strict": 0})
        R1["n_unl_unsupported"] = R1["n_unlabelled_E0.01"] - R1.n_unl_supported
        R1["n_unl_unsup_strict"] = R1["n_unlabelled_E0.01"] - R1.n_unl_strict
        keep = ["qid", "cfg", "genome_code", "ap", "ap_strict", "n_unlabelled_E0.01", "n_unl_supported", "n_unl_unsupported",
                "n_unl_unsup_strict"]
        A = pd.concat([M[keep], R1[keep]], ignore_index=True)
        tabs[split] = A
        S = A.groupby("cfg").agg(map=("ap", "mean"), unl=("n_unlabelled_E0.01", "mean"), unl_unsup=("n_unl_unsupported", "mean"),
                                 unl_unsup_strict=("n_unl_unsup_strict", "mean"), n=("ap", "size"))
        S.to_csv(OUT2 / f"rate_{split}_table.tsv", sep="\t")
        res[f"{split}_table_top"] = S.sort_values("map", ascending=False).head(25).round(4).reset_index().to_dict("records")
    if "odd" not in tabs:
        json.dump(res, open(OUT2 / "rate_selection.json", "w"), indent=1, default=float)
        return
    S = tabs["odd"].groupby("cfg").agg(map=("ap", "mean"), unl=("n_unlabelled_E0.01", "mean"),
                                       unl_unsup=("n_unl_unsupported", "mean"),
                                       unl_unsup_strict=("n_unl_unsup_strict", "mean"))
    ref = "M0_x8_t3"
    sel = {}
    for variant, col in (("no_exemption", "unl"), ("exemption", "unl_unsup"), ("exemption_strict_own_ge2", "unl_unsup_strict")):
        m0 = float(S.loc[ref, col])
        bound = 0.15 if m0 < 0.1 else 1.5 * m0
        ok = S[S[col] <= bound]
        pick = ok["map"].idxmax()
        d = dict(m0_rate_train=m0, bound=bound, candidates_meeting=int(len(ok)), selected=pick,
                 train_map=float(ok.loc[pick, "map"]), train_rate=float(ok.loc[pick, col]),
                 unconstrained_best=S["map"].idxmax(), unconstrained_best_map=float(S["map"].max()))
        if "even" in tabs:
            E = tabs["even"]
            x, y = E[E.cfg == pick].set_index("qid"), E[E.cfg == ref].set_index("qid")
            ids = x.index.intersection(y.index)
            d["test"] = dict(queries=int(len(ids)), map=float(x.loc[ids, "ap"].mean()), m0_map=float(y.loc[ids, "ap"].mean()),
                             delta_ap=ls.boot_mean(x.loc[ids, "ap"] - y.loc[ids, "ap"], y.loc[ids, "genome_code"]),
                             rate=float(x.loc[ids, "n_unlabelled_E0.01"].mean()),
                             rate_unsupported=float(x.loc[ids, "n_unl_unsupported"].mean()),
                             m0_rate=float(y.loc[ids, "n_unlabelled_E0.01"].mean()),
                             m0_rate_unsupported=float(y.loc[ids, "n_unl_unsupported"].mean()),
                             rate_unsup_strict=float(x.loc[ids, "n_unl_unsup_strict"].mean()),
                             m0_rate_unsup_strict=float(y.loc[ids, "n_unl_unsup_strict"].mean()))
        sel[variant] = d
        log(variant, json.dumps(d, default=float))
    res["selection"] = sel
    json.dump(res, open(OUT2 / "rate_selection.json", "w"), indent=1, default=float)
    log(f"done rate {time.time() - t0:.0f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["prep", "analyse", "rate"])
    ap.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_ON_NODE", "2")))
    a = ap.parse_args()
    {"prep": prep, "analyse": analyse, "rate": rate}[a.step](a)


if __name__ == "__main__":
    main()
