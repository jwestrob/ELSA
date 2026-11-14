#!/usr/bin/env python
"""ws_calib Q2b (POST HOC; designed after the Q1 tables and the first Q2 subset were read).

Q1 showed that genomes lacking the query's MMseqs2 family often hold homologs, so FAM negatives are contaminated. Here
every (query, genome) cell of the R queries gets a label from the best raw match itself:
  HOM   best gene alignable to the query (MMseqs2 align, Smith-Waterman, E <= 1e-3 against the 2.55 M-sequence DB) or
        sharing a Pfam family, Pfam clan, CDD superfamily or family-profile link with the query
  NEGC  both proteins annotated (Pfam or CDD), sharing none of these, and not alignable at E <= 1 (clean negative)
Detectors as in q2.py, evaluated on even target genomes (fits and thresholds on odd).
usage: q2b.py pairs --threads T  |  q2b.py eval --threads T
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import q2 as Q2  # noqa: E402
from q1 import evidence  # noqa: E402

RD = Q2.RD
C = RD / "calib"
P = C / "work/prep"
GP = C / "work/gpu"
W = C / "work/q2b"
RES = C / "results"
FPRS = (1e-2, 1e-3, 1e-4)
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def base():
    meta = dict(np.load(Q2.R3 / "vec/meta.npz"))
    Q = dict(np.load(P / "queries.npz"))
    jR = np.flatnonzero(Q["stratum"] == 0)
    uq = meta["uq"].astype(np.int64)
    gs = meta["g_start"].astype(np.int64)
    NG = len(gs)
    n = len(uq)
    gc = np.repeat(np.arange(NG), np.diff(np.r_[gs, n])).astype(np.int64)
    quid = Q["uid"][jR]
    arg1 = np.load(GP / "arg1.npy", mmap_mode="r")[jR].astype(np.int64)
    t1 = uq[gs[None, :] + arg1]
    uid_key = np.unique(uq * NG + gc)
    QG = np.zeros((len(jR), NG), bool)
    for j, q in enumerate(quid):
        lo_, hi_ = np.searchsorted(uid_key, [q * NG, (q + 1) * NG])
        QG[j, uid_key[lo_:hi_] - q * NG] = True
    return meta, Q, jR, quid, t1, QG, NG


def cmd_pairs(a):
    W.mkdir(parents=True, exist_ok=True)
    meta, Q, jR, quid, t1, QG, NG = base()
    qq = np.broadcast_to(quid[:, None], t1.shape)[~QG]
    tt = t1[~QG]
    key = np.unique(qq.astype(np.int64) * 4_000_000 + tt)
    np.save(W / "t1.npy", t1.astype(np.int32))
    with open(W / "pref.tsv", "w") as fh:
        q = key // 4_000_000
        t = key % 4_000_000
        for s in range(0, len(key), 5_000_000):
            fh.write("".join(f"{x}\t{y}\t0\t0\n" for x, y in zip(q[s:s + 5_000_000], t[s:s + 5_000_000])))
    log("cells", int((~QG).sum()), "unique pairs", len(key))


def cmd_eval(a):
    import numba
    numba.set_num_threads(a.threads)
    meta, Q, jR, quid, t1, QG, NG = base()
    uf = dict(np.load(P / "uidfeat.npz"))
    A = dict(np.load(P / "ann.npz"))
    hub = dict(np.load(P / "hub.npz"))
    n_g = meta["n_g"].astype(np.float64)
    nq = len(jR)
    # alignment E per cell
    if not (W / "aln.parquet").exists():
        pd.read_csv(W / "aln.tsv", sep="\t", header=None, names=["query", "target", "evalue"]).to_parquet(W / "aln.parquet")
    aln = pd.read_parquet(W / "aln.parquet", columns=["query", "target", "evalue"])
    akey = aln["query"].to_numpy(np.int64) * 4_000_000 + aln["target"].to_numpy(np.int64)
    o = np.argsort(akey)
    akey, aev = akey[o], aln.evalue.to_numpy()[o]
    ckey = quid[:, None].astype(np.int64) * 4_000_000 + t1
    pos = np.searchsorted(akey, ckey)
    pos = np.clip(pos, 0, len(akey) - 1)
    E = np.where(akey[pos] == ckey, aev[pos], np.inf).astype(np.float32)
    log("alignments", len(akey), "cells with an alignment reported", float(np.isfinite(E[~QG]).mean()))
    # annotation evidence per cell
    qf = np.broadcast_to(quid[:, None], t1.shape).ravel().astype(np.int64)
    tf = t1.ravel().astype(np.int64)
    ev = np.zeros((len(qf), 7), np.bool_)
    evidence(qf, tf, A["pfam_ptr"], A["pfam_val"], A["grp_ptr"], A["grp_val"], A["prof_ptr"], A["prof_val"],
             A["pdb_ptr"], A["pdb_val"], A["cdd_ptr"], A["cdd_val"], A["role_ptr"], A["role_val"], ev)
    ev = ev.reshape(nq, NG, 7)
    annP = ev[:, :, 0] | ev[:, :, 1] | ev[:, :, 2] | ev[:, :, 3]
    ALN = E <= 1e-3
    HOM = ALN | annP
    NEGC = ev[:, :, 5] & ev[:, :, 6] & ~annP & (E > 1)
    valid = ~QG
    log("cells", int(valid.sum()), "HOM share", float(HOM[valid].mean()), "aligned share", float(ALN[valid].mean()),
        "NEGC share", float(NEGC[valid].mean()), "unknown share", float((valid & ~HOM & ~NEGC).mean()))
    # scores (R queries), as q2.py
    js = jR
    top8 = np.load(GP / "top8.npy", mmap_mode="r")[js].astype(np.float32)
    m1 = top8[:, :, 0].copy()
    mean = np.load(GP / "mean.npy", mmap_mode="r")[js].astype(np.float32)
    sd = np.load(GP / "sd.npy", mmap_mode="r")[js].astype(np.float32)
    z1 = (m1 - mean) / np.maximum(sd, 1e-4)
    nG = n_g[None, :]
    S = {"D1_raw": m1, "D2_z": z1}
    hr = Q2.Hist(np.load(GP / "hist_raw.npy", mmap_mode="r")[js], Q2.LO_RAW, Q2.W_RAW)
    hz = Q2.Hist(np.load(GP / "hist_z.npy", mmap_mode="r")[js], Q2.LO_Z, Q2.W_Z)
    for wn, (sa, sb) in Q2.WINDOWS.items():
        S[f"D3_ev_raw_{wn}"] = Q2.neglog10_p(nG, hr.fitted(m1, sa, sb)[0]).astype(np.float32)
        S[f"D4_ev_z_{wn}"] = Q2.neglog10_p(nG, hz.fitted(z1, sa, sb)[0]).astype(np.float32)
    S["D5_ev_emp"] = Q2.neglog10_p(nG, hr.surv(m1)).astype(np.float32)
    S["D5b_ev_emp_z"] = Q2.neglog10_p(nG, hz.surv(z1)).astype(np.float32)
    c = top8
    sp = np.stack([(i + 1) * (c[:, :, i] - c[:, :, i + 1]) for i in range(1, 7)], 2).mean(2)
    T = (c[:, :, 0] - c[:, :, 1]) / np.maximum(sp, 1e-5)
    S["D6_ev_local"] = (6 * np.log10(1 + T / 6)).astype(np.float32)
    S["D7_gap"] = c[:, :, 0] - c[:, :, 1]
    del c, sp, T, top8
    S["D8_csls10"] = np.load(GP / "cs10.npy", mmap_mode="r")[js].astype(np.float32) - hub["r10"][quid][:, None]
    S["D8_csls50"] = np.load(GP / "cs50.npy", mmap_mode="r")[js].astype(np.float32) - hub["r50"][quid][:, None]
    recip = np.load(GP / "recip.npy", mmap_mode="r")[js]
    S["D9_rbm_then_raw"] = m1 + 2 * recip
    S["D10_centred"] = np.load(GP / "cmax.npy", mmap_mode="r")[js].astype(np.float32)
    even = (np.arange(NG) % 2 == 0)[None, :]
    odd = ~even
    choices = {}
    pO, nO = valid & odd & HOM, valid & odd & NEGC
    for fam in ("D3_ev_raw", "D4_ev_z"):
        best = None
        for wn in Q2.WINDOWS:
            s = S[f"{fam}_{wn}"]
            r = Q2.roc_point(s[pO], s[nO], 1e-3)
            choices[f"{fam}_{wn}_odd_recall_1e-3"] = r
            if best is None or r > best[1]:
                best = (wn, r)
        choices[fam] = best[0]
        S[fam] = S[f"{fam}_{best[0]}"]
    S["D9b_ev_z_rbm"] = S["D4_ev_z"] + 1000 * recip
    # combination models on odd HOM vs NEGC cells
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import HistGradientBoostingClassifier
    L = uf["length"]
    fn = ["D1_raw", "D2_z", "D4_ev_z", "D6_ev_local", "D7_gap", "D8_csls10", "recip", "D10_centred", "log_nG", "log_len_q", "r10_q"]

    def feats(mask):
        jj, gg = np.nonzero(mask)
        cols = []
        for f in fn:
            if f == "recip":
                cols.append(recip[jj, gg].astype(np.float32))
            elif f == "log_nG":
                cols.append(np.log(n_g[gg]).astype(np.float32))
            elif f == "log_len_q":
                cols.append(np.log(L[quid[jj]]).astype(np.float32))
            elif f == "r10_q":
                cols.append(hub["r10"][quid[jj]].astype(np.float32))
            else:
                cols.append(S[f][jj, gg])
        return np.column_stack(cols), jj, gg

    trm = pO | nO
    X, jj, gg = feats(trm)
    y = HOM[jj, gg]
    rng = np.random.default_rng(17)
    pick = rng.choice(len(y), min(2_000_000, len(y)), replace=False)
    sc = StandardScaler().fit(X[pick])
    lr = LogisticRegression(max_iter=500).fit(sc.transform(X[pick]), y[pick])
    gbm = HistGradientBoostingClassifier(random_state=17).fit(X[pick], y[pick])
    choices["lr_coef"] = dict(zip(fn, lr.coef_[0].round(4).tolist()))
    del X
    X, jj, gg = feats(np.ones_like(valid))
    S["D11_combo_lr"] = np.empty(valid.shape, np.float32)
    S["D12_combo_gbm"] = np.empty(valid.shape, np.float32)
    for s0 in range(0, len(X), 5_000_000):
        sl = slice(s0, s0 + 5_000_000)
        S["D11_combo_lr"][jj[sl], gg[sl]] = lr.decision_function(sc.transform(X[sl]))
        p = gbm.predict_proba(X[sl])[:, 1]
        S["D12_combo_gbm"][jj[sl], gg[sl]] = np.log10(np.clip(p, 1e-12, 1)) - np.log10(np.clip(1 - p, 1e-12, 1))
    del X
    log("scores and combinations ready", choices)
    order = ["D1_raw", "D2_z", "D3_ev_raw", "D4_ev_z", "D5_ev_emp", "D5b_ev_emp_z", "D6_ev_local", "D7_gap", "D8_csls10",
             "D8_csls50", "D9_rbm_then_raw", "D9b_ev_z_rbm", "D10_centred", "D11_combo_lr", "D12_combo_gbm"]
    lt = np.quantile(L[uf["valid"]], [1 / 3, 2 / 3])
    ntm = uf["ntm"]
    results = dict(choices=choices, label_shares=dict(HOM=float(HOM[valid].mean()), ALN=float(ALN[valid].mean()),
                                                      NEGC=float(NEGC[valid].mean())))
    table = []
    short = (L[quid] < 100)[:, None]
    for sname, posm, negm in [("HOM_vs_NEGC", HOM, NEGC), ("ALN_vs_NEGC", ALN, NEGC), ("HOM_vs_notHOM", HOM, ~HOM),
                              ("short_HOM_vs_NEGC", HOM & short, NEGC & short)]:
        posE, negE = valid & even & posm, valid & even & negm
        posO, negO = valid & odd & posm, valid & odd & negm
        jl = np.broadcast_to(np.arange(nq)[:, None], valid.shape)
        bt = Q2.Boot(nq)
        res = dict(pos_even=int(posE.sum()), neg_even=int(negE.sum()), scores={},
                   rbm_binary=dict(fpr=float(recip[negE].mean()), recall=float(recip[posE].mean())))
        base_draws = None
        for k in order:
            s = S[k]
            pv, nv = s[posE], s[negE]
            d = dict(auroc=Q2.auroc(pv, nv))
            rp = Q2.roc_points(pv, nv, FPRS)
            nOv = s[negO]
            for x, r in zip(FPRS, rp):
                d[f"recall@{x:g}"] = r
                thr = Q2.thr_at(nOv, x)
                d[f"transfer@{x:g}"] = dict(fpr_even=float((nv >= thr).mean()), recall_even=float((pv >= thr).mean()))
            mm = posE | negE
            tpr, fpr = bt.curves(jl[mm], s[mm], posE[mm], negE[mm])
            draws = {x: Q2.Boot.at(tpr, fpr, x) for x in FPRS}
            for x in FPRS:
                d[f"ci@{x:g}"] = np.quantile(draws[x], [0.025, 0.975]).round(4).tolist()
            if k == "D1_raw":
                base_draws = draws
            else:
                for x in FPRS:
                    d[f"diff_vs_raw@{x:g}"] = [round(d[f"recall@{x:g}"] - res["scores"]["D1_raw"][f"recall@{x:g}"], 4)] + \
                        np.quantile(draws[x] - base_draws[x], [0.025, 0.975]).round(4).tolist()
            res["scores"][k] = d
            table.append(dict(subset=sname, score=k, auroc=round(d["auroc"], 4), **{f"rec@{x:g}": round(d[f"recall@{x:g}"], 4) for x in FPRS},
                              **{f"ci@{x:g}": d[f"ci@{x:g}"] for x in FPRS}, **{f"dvsraw@{x:g}": d.get(f"diff_vs_raw@{x:g}") for x in FPRS},
                              **{f"transferFPR@{x:g}": round(d[f"transfer@{x:g}"]["fpr_even"], 6) for x in FPRS}))
        results[sname] = res
        log(sname, res["pos_even"], res["neg_even"], {k: round(res["scores"][k]["recall@0.001"], 3) for k in order})
    pd.DataFrame(table).to_csv(RES / "q2b_table.tsv", sep="\t", index=False)
    # nominal calibration on clean negatives, by query stratum (length tertile x TM x Pfam)
    stratum = (np.digitize(L[quid], lt) * 4 + (ntm[quid] >= 1) * 2 + (np.diff(A["pfam_ptr"])[quid] > 0)).astype(int)
    negE = valid & even & NEGC
    cal = {}
    srows = []
    pvalued = ["D3_ev_raw", "D4_ev_z", "D5_ev_emp", "D5b_ev_emp_z", "D6_ev_local"]
    for k in order:
        s = S[k]
        d = {}
        if k in pvalued:
            for al in FPRS:
                d[f"nominal_{al:g}_clean_fpr"] = float((s[negE] >= -np.log10(al)).mean())
            th = 3.0
        else:
            th = Q2.thr_at(s[negE], 1e-3)
        per = []
        for v in range(12):
            mm = negE & (stratum == v)[:, None]
            if mm.sum() >= 10000:
                f = float((s[mm] >= th).mean())
                per.append(f)
                srows.append(dict(score=k, stratum=v, len_tertile=v // 4, tm=(v // 2) % 2, pfam=v % 2, n_neg=int(mm.sum()), fpr=f))
        per = np.array(per)
        d.update(threshold=float(th), strata_n=int(len(per)), strata_within_10x=int(((per >= 1e-4) & (per <= 1e-2)).sum()),
                 strata_within_3x=int(((per >= 1e-3 / 3) & (per <= 3e-3)).sum()), strata_fpr_min_max=[float(per.min()), float(per.max())])
        cal[k] = d
    results["calibration"] = cal
    pd.DataFrame(srows).to_csv(RES / "q2b_strata.tsv", sep="\t", index=False)
    results["seconds"] = round(time.time() - T0, 1)
    (RES / "q2b.json").write_text(json.dumps(results, indent=1, default=float) + "\n")
    log("done")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--threads", type=int, required=True)
    a = ap.parse_args()
    dict(pairs=cmd_pairs, eval=cmd_eval)[a.cmd](a)


if __name__ == "__main__":
    main()
