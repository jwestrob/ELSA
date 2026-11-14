#!/usr/bin/env python
"""ws_calib Q2: can whitened ESM++ small similarity carry real statistics? Detector comparison per (query, genome).

usage: q2.py run --threads T
Reads work/prep and work/gpu; fits on odd target genomes (genome_code % 2 == 1), scores on even.
Writes results/q2_detectors.json, results/q2_table.tsv, results/q2_strata.tsv, results/q2_choices.json.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

RD = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111")
C = RD / "calib"
P = C / "work/prep"
GP = C / "work/gpu"
RES = C / "results"
R3 = RD / "softrec/r3"
TOK = RD / "engine/dpann_tokens/mmseqs__seq__e3c50s75_rescue3__dpann.parquet"
NB_RAW, LO_RAW, W_RAW = 1500, -0.5, 1000.0
NB_Z, LO_Z, W_Z = 2400, -8.0, 50.0
FPRS = (1e-2, 1e-3, 1e-4)
WINDOWS = {"w2_3": (1e-2, 1e-3), "w3_4": (1e-3, 1e-4)}
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def jdump(path, obj):
    path.write_text(json.dumps(obj, indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x)) + "\n")


# ------------------------------------------------------------------------------------------- tail models
class Hist:
    def __init__(self, h, lo, w):
        self.h = h.astype(np.float64)
        self.N = self.h.sum(1, keepdims=True)
        self.lo, self.w = lo, w
        nb = h.shape[1]
        # S[:, k] = P(x >= edge_k), edge_k = lo + k / w, k = 0..nb
        self.S = np.concatenate([np.cumsum(self.h[:, ::-1], 1)[:, ::-1], np.zeros((len(h), 1))], 1) / self.N
        self.nb = nb

    def surv(self, x):
        """Empirical survival at x (NQ x NG), linear within bins."""
        k = np.clip(np.floor((x - self.lo) * self.w).astype(np.int64), 0, self.nb - 1)
        frac = np.clip((self.lo + (k + 1) / self.w - x) * self.w, 0, 1)
        Sk1 = np.take_along_axis(self.S, k + 1, 1)
        hk = np.take_along_axis(self.h, k, 1) / self.N
        return Sk1 + hk * frac

    def quant(self, s):
        """Per query value c with S(c) = s (vector over queries)."""
        out = np.empty(len(self.S))
        for i in range(len(self.S)):
            Si = self.S[i]
            k = np.searchsorted(-Si, -s, side="left") - 1      # last k with S_k >= s
            k = int(np.clip(k, 0, self.nb - 1))
            hk = self.h[i, k] / self.N[i, 0]
            out[i] = self.lo + k / self.w + ((Si[k] - s) / hk if hk > 0 else 0) / self.w
        return out

    def fitted(self, x, sa, sb):
        ca, cb = self.quant(sa), self.quant(sb)
        lam = np.log(sa / sb) / np.maximum(cb - ca, 1e-6)
        Se = self.surv(x)
        St = sa * np.exp(-lam[:, None] * (x - ca[:, None]))
        return np.where(x >= ca[:, None], St, Se), dict(lam_q10_q50_q90=np.quantile(lam, [.1, .5, .9]).tolist(),
                                                         ca_q50=float(np.median(ca)), cb_q50=float(np.median(cb)))


def neglog10_p(n, S):
    """-log10 of P(max of n draws >= x) = 1 - (1 - S)^n, Poisson form 1 - exp(-n S)."""
    x = np.maximum(n * S, 1e-300)
    return np.where(x < 1e-8, -np.log10(x), -np.log10(-np.expm1(-x)))


# ------------------------------------------------------------------------------------------- metrics
def roc_points(pos, neg, xs):
    """Recall at each FPR x with linear interpolation across ties (exact). One sort per call."""
    neg = np.sort(neg)[::-1]
    out = []
    for x in xs:
        k = x * len(neg)
        thr = neg[max(int(np.ceil(k)) - 1, 0)]
        n_gt = int(np.searchsorted(-neg, -thr, side="left"))
        n_eq = int(np.searchsorted(-neg, -thr, side="right")) - n_gt
        frac = float(np.clip((k - n_gt) / max(n_eq, 1), 0, 1))
        out.append(float((pos > thr).mean() + frac * (pos == thr).mean()))
    return out


def roc_point(pos, neg, x):
    return roc_points(pos, neg, [x])[0]


def auroc(pos, neg):
    from scipy.stats import rankdata
    r = rankdata(np.r_[pos, neg])
    n1, n0 = len(pos), len(neg)
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def thr_at(neg, x):
    return float(np.quantile(neg, 1 - x, method="higher"))


class Boot:
    """Query bootstrap of ROC quantities from per-query binned counts (bins at negative-score quantiles)."""

    def __init__(self, nq, draws=2000, seed=17):
        rng = np.random.default_rng(seed)
        self.W = rng.multinomial(nq, np.full(nq, 1.0 / nq), size=draws).astype(np.float32)
        self.nq = nq

    def curves(self, jl, score, posm, negm):
        neg = score[negm]
        lv = np.r_[10 ** np.linspace(-6, -1, 500), np.linspace(0.1, 1, 200)]
        edges = np.unique(np.quantile(neg, 1 - lv, method="higher"))
        m = len(edges)
        b = np.searchsorted(edges, score, side="right")
        pc = np.bincount(jl[posm] * (m + 1) + b[posm], minlength=self.nq * (m + 1)).reshape(self.nq, m + 1).astype(np.float32)
        nc = np.bincount(jl[negm] * (m + 1) + b[negm], minlength=self.nq * (m + 1)).reshape(self.nq, m + 1).astype(np.float32)
        Pb = self.W @ pc
        Nb = self.W @ nc
        tp = np.cumsum(Pb[:, ::-1], 1)[:, ::-1]           # tp[:, k] = sum over bins >= k
        fp = np.cumsum(Nb[:, ::-1], 1)[:, ::-1]
        tpr = tp / tp[:, :1]
        fpr = fp / fp[:, :1]
        return tpr, fpr

    @staticmethod
    def at(tpr, fpr, x):
        out = np.empty(len(tpr))
        for d in range(len(tpr)):
            f, t = fpr[d, ::-1], tpr[d, ::-1]                 # increasing FPR
            out[d] = np.interp(x, np.r_[0, f], np.r_[0, t])
        return out

    @staticmethod
    def auc(tpr, fpr):
        f = np.concatenate([fpr[:, ::-1], np.ones((len(fpr), 1))], 1)
        t = np.concatenate([tpr[:, ::-1], np.ones((len(tpr), 1))], 1)
        f = np.concatenate([np.zeros((len(f), 1)), f], 1)
        t = np.concatenate([np.zeros((len(t), 1)), t], 1)
        return ((f[:, 1:] - f[:, :-1]) * (t[:, 1:] + t[:, :-1]) / 2).sum(1)


# ------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--threads", type=int, required=True)
    a = ap.parse_args()
    RES.mkdir(parents=True, exist_ok=True)
    meta = dict(np.load(R3 / "vec/meta.npz"))
    uq = meta["uq"].astype(np.int64)
    gs = meta["g_start"].astype(np.int64)
    n_g = meta["n_g"].astype(np.float64)
    n = len(uq)
    NG = len(gs)
    gc = np.repeat(np.arange(NG), np.diff(np.r_[gs, n])).astype(np.int64)
    tokdf = pd.read_parquet(TOK)
    tok = np.full(n, -1, np.int64)
    tok[tokdf.gid.to_numpy()] = tokdf.token.to_numpy()
    uf = dict(np.load(P / "uidfeat.npz"))
    A = dict(np.load(P / "ann.npz"))
    Q = dict(np.load(P / "queries.npz"))
    quid, strat = Q["uid"], Q["stratum"]
    NQ = len(quid)
    hub = dict(np.load(P / "hub.npz"))
    tok_u = uf["tok"]
    L = uf["length"]
    ntm = uf["ntm"]
    has_pfam = np.diff(A["pfam_ptr"]) > 0
    has_cdd = np.diff(A["cdd_ptr"]) > 0
    fs = uf["fam_genes"]

    # ---------------------------------------------------------------- labels
    fam_key = np.unique(tok * NG + gc)
    uid_key = np.unique(uq * NG + gc)
    FAM = np.zeros((NQ, NG), bool)
    QG = np.zeros((NQ, NG), bool)
    for j in range(NQ):
        q, tq = int(quid[j]), int(tok_u[quid[j]])
        lo_, hi_ = np.searchsorted(fam_key, [tq * NG, (tq + 1) * NG])
        FAM[j, fam_key[lo_:hi_] - tq * NG] = True
        lo_, hi_ = np.searchsorted(uid_key, [q * NG, (q + 1) * NG])
        QG[j, uid_key[lo_:hi_] - q * NG] = True

    def presence(ptr, val):
        cnt = np.diff(ptr)[uq]
        gidx = np.repeat(np.arange(n), cnt)
        starts = ptr[uq]
        # codes for every gene: concatenate val[ptr[u]:ptr[u+1]] in gene order
        off = np.repeat(starts - np.r_[0, np.cumsum(cnt)[:-1]], cnt) + np.arange(cnt.sum())
        codes = val[off].astype(np.int64)
        key = np.unique(codes * NG + gc[gidx])
        return key

    def any_present(key, codes):
        out = np.zeros(NG, bool)
        for c in codes:
            lo_, hi_ = np.searchsorted(key, [c * NG, (c + 1) * NG])
            out[key[lo_:hi_] - c * NG] = True
        return out

    gkey = presence(A["grp_ptr"], A["grp_val"])
    ckey = presence(A["cdd_ptr"], A["cdd_val"])
    CLAN = np.zeros((NQ, NG), bool)
    CDDP = np.zeros((NQ, NG), bool)
    for j in range(NQ):
        q = int(quid[j])
        CLAN[j] = any_present(gkey, A["grp_val"][A["grp_ptr"][q]:A["grp_ptr"][q + 1]])
        CDDP[j] = any_present(ckey, A["cdd_val"][A["cdd_ptr"][q]:A["cdd_ptr"][q + 1]])
    annot_q = has_pfam[quid] | has_cdd[quid]
    CLEAN = ~FAM & ~CLAN & ~CDDP & annot_q[:, None]
    valid = ~QG
    even = (np.arange(NG) % 2 == 0)[None, :]
    log("labels; FAM share", float(FAM[valid].mean()), "CLAN share (pfam queries)", float(CLAN[valid & has_pfam[quid][:, None]].mean()),
        "clean share (annotated)", float(CLEAN[valid & annot_q[:, None]].mean()))

    # ---------------------------------------------------------------- scores
    top8 = np.load(GP / "top8.npy").astype(np.float32)
    m1 = top8[:, :, 0].copy()
    mean = np.load(GP / "mean.npy").astype(np.float32)
    sd = np.load(GP / "sd.npy").astype(np.float32)
    z1 = (m1 - mean) / np.maximum(sd, 1e-4)
    del mean
    nG = n_g[None, :]
    S = {}
    S["D1_raw"] = m1
    S["D2_z"] = z1
    hr = Hist(np.load(GP / "hist_raw.npy"), LO_RAW, W_RAW)
    hz = Hist(np.load(GP / "hist_z.npy"), LO_Z, W_Z)
    log("histogram totals per query q50", float(np.median(hr.N)), float(np.median(hz.N)))
    fitinfo = {}
    for wn, (sa, sb) in WINDOWS.items():
        Sf, inf = hr.fitted(m1, sa, sb)
        S[f"D3_ev_raw_{wn}"] = neglog10_p(nG, Sf).astype(np.float32)
        fitinfo[f"raw_{wn}"] = inf
        Sf, inf = hz.fitted(z1, sa, sb)
        S[f"D4_ev_z_{wn}"] = neglog10_p(nG, Sf).astype(np.float32)
        fitinfo[f"z_{wn}"] = inf
        del Sf
    S["D5_ev_emp"] = neglog10_p(nG, hr.surv(m1)).astype(np.float32)
    S["D5b_ev_emp_z"] = neglog10_p(nG, hz.surv(z1)).astype(np.float32)
    c = top8
    sp = np.stack([(i + 1) * (c[:, :, i] - c[:, :, i + 1]) for i in range(1, 7)], 2).mean(2)
    T = (c[:, :, 0] - c[:, :, 1]) / np.maximum(sp, 1e-5)
    S["D6_ev_local"] = (6 * np.log10(1 + T / 6)).astype(np.float32)
    S["D7_gap"] = c[:, :, 0] - c[:, :, 1]
    del c, sp, T, top8
    S["D8_csls10"] = np.load(GP / "cs10.npy").astype(np.float32) - hub["r10"][quid][:, None]
    S["D8_csls50"] = np.load(GP / "cs50.npy").astype(np.float32) - hub["r50"][quid][:, None]
    recip = np.load(GP / "recip.npy")
    S["D9_rbm_then_raw"] = m1 + 2 * recip
    S["D10_centred"] = np.load(GP / "cmax.npy").astype(np.float32)
    S["D9b_ev_z_rbm"] = None      # filled after the window choice
    log("scores computed", list(S))

    jR = np.flatnonzero(strat == 0)
    odd = ~even

    def cells(js, lab, negm, par):
        """Return (pos values per score dict-able) masks over the (js x NG) block."""
        v = valid[js] & par
        return v & lab[js], v & negm[js]

    # ---------------------------------------------------------------- odd-genome choices
    choices = {}
    posm, negm = cells(jR, FAM, ~FAM, odd)
    for fam in ("D3_ev_raw", "D4_ev_z"):
        best = None
        for wn in WINDOWS:
            s = S[f"{fam}_{wn}"][jR]
            r = roc_point(s[posm], s[negm], 1e-3)
            choices[f"{fam}_{wn}_odd_recall_1e-3"] = r
            if best is None or r > best[1]:
                best = (wn, r)
        choices[fam] = best[0]
        S[fam] = S[f"{fam}_{best[0]}"]
    S["D9b_ev_z_rbm"] = S["D4_ev_z"] + 1000 * recip
    log("choices", choices)

    # combination models (odd cells of R queries)
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import HistGradientBoostingClassifier
    feat_names = ["D1_raw", "D2_z", "D4_ev_z", "D6_ev_local", "D7_gap", "D8_csls10", "recip", "D10_centred", "log_nG",
                  "log_len_q", "r10_q"]

    def feats(js, mask):
        jj, gg = np.nonzero(mask)
        cols = []
        for f in feat_names:
            if f == "recip":
                cols.append(recip[js][jj, gg].astype(np.float32))
            elif f == "log_nG":
                cols.append(np.log(n_g[gg]).astype(np.float32))
            elif f == "log_len_q":
                cols.append(np.log(L[quid[js][jj]]).astype(np.float32))
            elif f == "r10_q":
                cols.append(hub["r10"][quid[js][jj]].astype(np.float32))
            else:
                cols.append(S[f][js][jj, gg])
        return np.column_stack(cols), jj, gg

    trm = valid[jR] & odd
    Xtr, jj, gg = feats(jR, trm)
    ytr = FAM[jR][jj, gg]
    rng = np.random.default_rng(17)
    pick = rng.choice(len(ytr), min(2_000_000, len(ytr)), replace=False)
    sc = StandardScaler().fit(Xtr[pick])
    lr = LogisticRegression(max_iter=500).fit(sc.transform(Xtr[pick]), ytr[pick])
    gbm = HistGradientBoostingClassifier(random_state=17).fit(Xtr[pick], ytr[pick])
    choices["lr_coef"] = dict(zip(feat_names, lr.coef_[0].round(4).tolist()))
    del Xtr
    log("combination models fitted")

    def combo_scores(js):
        mask = np.ones((len(js), NG), bool)
        X, jj, gg = feats(js, mask)
        out_lr = np.empty((len(js), NG), np.float32)
        out_gb = np.empty((len(js), NG), np.float32)
        B = 5_000_000
        for s0 in range(0, len(X), B):
            out_lr[jj[s0:s0 + B], gg[s0:s0 + B]] = lr.decision_function(sc.transform(X[s0:s0 + B]))
            p = gbm.predict_proba(X[s0:s0 + B])[:, 1]
            out_gb[jj[s0:s0 + B], gg[s0:s0 + B]] = np.log10(np.clip(p, 1e-12, 1)) - np.log10(np.clip(1 - p, 1e-12, 1))
        return out_lr, out_gb

    # ---------------------------------------------------------------- subsets
    allq = np.arange(NQ)
    subsets = {
        "overall_FAM": (jR, FAM, ~FAM),
        "overall_FAM_cleanneg": (jR[annot_q[jR]], FAM, CLEAN),
        "overall_CLAN": (jR[has_pfam[quid[jR]]], CLAN, ~CLAN),
        "overall_CLAN_famabsent_vs_clean": (jR[annot_q[jR] & has_pfam[quid[jR]]], CLAN & ~FAM, CLEAN),
        "short_FAM": (allq[np.isin(strat, [0, 1]) & (L[quid] < 100)], FAM, ~FAM),
        "singletons_CLAN": (allq[np.isin(strat, [0, 2]) & (fs[quid] == 1) & has_pfam[quid]], CLAN, ~CLAN),
        "curated_FAM": (allq[np.isin(strat, [0, 3]) & (A["callbit"][quid] > 0)], FAM, ~FAM),
    }
    order = ["D1_raw", "D2_z", "D3_ev_raw", "D4_ev_z", "D5_ev_emp", "D5b_ev_emp_z", "D6_ev_local", "D7_gap", "D8_csls10",
             "D8_csls50", "D9_rbm_then_raw", "D9b_ev_z_rbm", "D10_centred", "D11_combo_lr", "D12_combo_gbm"]
    results = {}
    table = []
    for sname, (js, lab, negl) in subsets.items():
        lr_s, gb_s = combo_scores(js)
        loc = {k: S[k][js] for k in order if k in S}
        loc["D11_combo_lr"] = lr_s
        loc["D12_combo_gbm"] = gb_s
        posE, negE = cells(js, lab, negl, even)
        posO, negO = cells(js, lab, negl, odd)
        jl = np.broadcast_to(np.arange(len(js))[:, None], (len(js), NG))
        bt = Boot(len(js))
        res = dict(queries=int(len(js)), pos_even=int(posE.sum()), neg_even=int(negE.sum()), scores={})
        base_draws = None
        rbin = recip[js]
        res["D9_rbm_binary"] = dict(fpr=float(rbin[negE].mean()), recall=float(rbin[posE].mean()))
        for k in order:
            s = loc[k]
            pv, nv = s[posE], s[negE]
            d = dict(auroc=auroc(pv, nv))
            rp = roc_points(pv, nv, FPRS)
            nO = s[negO]
            for x, r in zip(FPRS, rp):
                d[f"recall@{x:g}"] = r
                thr = thr_at(nO, x)
                d[f"transfer@{x:g}"] = dict(fpr_even=float((nv >= thr).mean()), recall_even=float((pv >= thr).mean()))
            tpr, fpr = bt.curves(jl[posE | negE], s[posE | negE], posE[posE | negE], negE[posE | negE])
            draws = {x: Boot.at(tpr, fpr, x) for x in FPRS}
            draws["auc"] = Boot.auc(tpr, fpr)
            for x in FPRS:
                d[f"ci@{x:g}"] = np.quantile(draws[x], [0.025, 0.975]).round(4).tolist()
            d["ci_auc"] = np.quantile(draws["auc"], [0.025, 0.975]).round(4).tolist()
            if k == "D1_raw":
                base_draws = draws
            else:
                for x in FPRS:
                    dd = draws[x] - base_draws[x]
                    d[f"diff_vs_raw@{x:g}"] = [round(float(d[f"recall@{x:g}"] - res["scores"]["D1_raw"][f"recall@{x:g}"]), 4)] + \
                        np.quantile(dd, [0.025, 0.975]).round(4).tolist()
            res["scores"][k] = d
            table.append(dict(subset=sname, score=k, auroc=round(d["auroc"], 4),
                              **{f"rec@{x:g}": round(d[f"recall@{x:g}"], 4) for x in FPRS},
                              **{f"ci@{x:g}": d[f"ci@{x:g}"] for x in FPRS},
                              **{f"dvsraw@{x:g}": d.get(f"diff_vs_raw@{x:g}") for x in FPRS},
                              **{f"transferFPR@{x:g}": round(d[f"transfer@{x:g}"]["fpr_even"], 6) for x in FPRS}))
        results[sname] = res
        log(sname, "queries", len(js), "pos", res["pos_even"], "neg", res["neg_even"],
            {k: round(res["scores"][k]["recall@0.001"], 3) for k in order})
        del loc, lr_s, gb_s
    pd.DataFrame(table).to_csv(RES / "q2_table.tsv", sep="\t", index=False)

    # ---------------------------------------------------------------- nominal calibration (E-value test)
    js = jR[annot_q[jR]]
    lt = np.quantile(L[uf["valid"]], [1 / 3, 2 / 3])
    st_len = np.digitize(L[quid[js]], lt)
    st_tm = (ntm[quid[js]] >= 1).astype(int)
    st_pf = has_pfam[quid[js]].astype(int)
    stratum = st_len * 4 + st_tm * 2 + st_pf
    negE = valid[js] & even & CLEAN[js]
    negF = valid[js] & even & ~FAM[js]
    cal = {}
    lr_s, gb_s = combo_scores(js)
    allsc = {k: S[k][js] for k in order if k in S}
    allsc["D11_combo_lr"] = lr_s
    allsc["D12_combo_gbm"] = gb_s
    pvalued = ["D3_ev_raw", "D4_ev_z", "D5_ev_emp", "D5b_ev_emp_z", "D6_ev_local", "D3_ev_raw_w2_3", "D3_ev_raw_w3_4",
               "D4_ev_z_w2_3", "D4_ev_z_w3_4"]
    srows = []
    for k in pvalued + ["D1_raw", "D2_z", "D8_csls10", "D10_centred", "D11_combo_lr", "D12_combo_gbm"]:
        s = allsc[k] if k in allsc else S[k][js]
        d = {}
        if k in pvalued:
            for al in FPRS:
                th = -np.log10(al)
                d[f"nominal_{al:g}"] = dict(clean_fpr=float((s[negE] >= th).mean()), famabsent_fpr=float((s[negF] >= th).mean()))
            th = 3.0
        else:
            th = thr_at(s[negE], 1e-3)
        d["threshold_used_for_strata"] = th
        per = []
        for v in range(12):
            mm = negE & (stratum == v)[:, None]
            if mm.sum() >= 10000:
                f = float((s[mm] >= th).mean())
                per.append(f)
                srows.append(dict(score=k, stratum=v, len_tertile=v // 4, tm=(v // 2) % 2, pfam=v % 2, n_neg=int(mm.sum()), fpr=f))
        per = np.array(per)
        d["strata_n"] = int(len(per))
        d["strata_within_10x_of_1e-3"] = int(((per >= 1e-4) & (per <= 1e-2)).sum())
        d["strata_within_3x_of_1e-3"] = int(((per >= 1e-3 / 3) & (per <= 3e-3)).sum())
        d["strata_fpr_min_max"] = [float(per.min()), float(per.max())] if len(per) else None
        cal[k] = d
    pd.DataFrame(srows).to_csv(RES / "q2_strata.tsv", sep="\t", index=False)
    results["calibration"] = cal
    results["choices"] = choices
    results["fitinfo"] = fitinfo
    results["seconds"] = round(time.time() - T0, 1)
    jdump(RES / "q2_detectors.json", results)
    log("done")


if __name__ == "__main__":
    main()
