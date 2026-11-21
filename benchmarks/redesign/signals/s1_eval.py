#!/usr/bin/env python
"""Signal 1 evaluation (biotite login node, single thread, small tables): existence strata, co-fold, curated split,
showcase operon. Reads work/s1_pairs.parquet; writes work/s1_eval.json and work/s1_cofold_rows.tsv."""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RD, W, SEED, load_tokens, auc_w, ci  # noqa: E402


def boot_median(x, draws=2000, seed=SEED):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    rng = np.random.default_rng(seed)
    b = np.median(x[rng.integers(0, len(x), (draws, len(x)))], axis=1)
    return float(np.median(x)), ci(b), len(x)


def unit_boot_auc(score, y, units, draws=2000, seed=SEED):
    """AUROC with resampling of units (family pairs); returns point, ci, draws array."""
    score, y = np.asarray(score, float), np.asarray(y, bool)
    u, inv = np.unique(units, return_inverse=True)
    rng = np.random.default_rng(seed)
    Wb = rng.multinomial(len(u), np.full(len(u), 1 / len(u)), size=draws).astype(float)
    b = np.array([auc_w(score, y, w[inv]) for w in Wb])
    return auc_w(score, y), ci(b), b


def pct_rank(x):
    return pd.Series(x).rank(pct=True).to_numpy()


def main():
    R = pd.read_parquet(W / "s1_pairs.parquet").set_index("key")
    out = {"scored_pairs": int(len(R)), "E_defined": int(R.E.notna().sum())}
    # ------------------------------------------------------------------ existence
    ex = {}
    for s in ("S_op", "S_opp", "S_unc"):
        d = R[R[s]]
        med, mci, nE = boot_median(d.E)
        ex[s] = dict(pairs=int(len(d)), E_defined=nE, median_E=med, median_E_ci=mci, mean_E=float(d.E.mean()),
                     frac_E_pos=float((d.E > 0).mean()), median_M=float(d.M.median()), median_rhoS_F=float(d.rhoS_F.median()),
                     frac_p_hom_le05=float((d.p_hom <= 0.05).mean()), frac_p_fam_le05=float((d.p_fam <= 0.05).mean()),
                     median_movable_hom_frac=float((d.movable_hom / d.n_used).median()),
                     median_cca=float(d.cca.median()), median_n=float(d.n.median()))
    a, b = R[R.S_op].E.dropna().to_numpy(), R[R.S_opp].E.dropna().to_numpy()
    rng = np.random.default_rng(SEED)
    db = [np.median(a[rng.integers(0, len(a), len(a))]) - np.median(b[rng.integers(0, len(b), len(b))]) for _ in range(2000)]
    ex["S_op_minus_S_opp_median_E"] = dict(diff=float(np.median(a) - np.median(b)), ci=ci(db))
    ex["verdict_exists"] = bool(ex["S_op"]["median_E_ci"][0] > 0 and ex["S_op"]["frac_p_hom_le05"] >= 0.10)
    ex["verdict_beyond_linkage"] = bool(ex["S_op_minus_S_opp_median_E"]["ci"][0] > 0)
    # E against engine recurrence inside S_op (descriptive)
    d = R[R.S_op].dropna(subset=["E"])
    ex["S_op_spearman_E_vs_L"] = list(map(float, spearmanr(d.E, d.L)))
    ex["S_op_spearman_E_vs_n"] = list(map(float, spearmanr(d.E, d.n)))
    out["existence"] = ex

    # ------------------------------------------------------------------ co-fold
    tok = load_tokens()
    T = int(tok.max()) + 1
    f = pd.read_parquet(RD / "grammar/prep/folded.parquet")
    cr = pd.read_parquet(RD / "grammar/eval/folded_scored_cv.parquet", columns=["fold_id", "s_primary"])
    f = f.merge(cr, on="fold_id", how="left")
    f["key"] = np.minimum(tok[f.g0], tok[f.g1]) * T + np.maximum(tok[f.g0], tok[f.g1])
    pos = pd.read_csv(RD / "signals/code/screen1_positive.tsv", sep="\t")
    ab = pos.gids.str.split(";", expand=True).astype(np.int64)
    pos["key"] = np.minimum(tok[ab[0]], tok[ab[1]]) * T + np.maximum(tok[ab[0]], tok[ab[1]])
    pos["cls"] = "positive"
    allf = pd.concat([f[["fold_id", "cls", "key", "iptm", "eng_nlogP", "s_primary"]], pos[["fold_id", "cls", "key", "iptm"]]],
                     ignore_index=True)
    allf = allf.join(R[["n", "E", "M", "p_hom", "cca"]], on="key")
    allf["hit"] = allf.iptm >= 0.75
    allf.to_csv(W / "s1_cofold_rows.tsv", sep="\t", index=False)
    cf = {}
    for nmin in (10, 30):
        for setname, mask in (("cognate+uncalled", allf.cls.isin(["cognate", "uncalled"])), ("cognate", allf.cls == "cognate"),
                              ("positive", allf.cls == "positive")):
            s = allf[mask & (allf.n >= nmin) & allf.E.notna()].reset_index(drop=True)
            r = dict(folds=int(len(s)), hits=int(s.hit.sum()), family_pairs=int(s.key.nunique()),
                     uncalled_folds=int((s.cls == "uncalled").sum()))
            if s.hit.sum() >= 2 and (~s.hit).sum() >= 2:
                aE, aEci, bE = unit_boot_auc(s.E, s.hit, s.key)
                r.update(auc_E=aE, auc_E_ci=aEci)
                aM, aMci, _ = unit_boot_auc(s.M, s.hit, s.key)
                r.update(auc_M=aM, auc_M_ci=aMci)
                r["spearman_E_iptm"] = list(map(float, spearmanr(s.E, s.iptm)))
                if setname != "positive":
                    aN, aNci, bN = unit_boot_auc(s.eng_nlogP, s.hit, s.key)
                    comb = (pct_rank(s.E) + pct_rank(s.eng_nlogP)) / 2
                    aC, aCci, bC = unit_boot_auc(comb, s.hit, s.key)
                    aK, aKci, _ = unit_boot_auc(s.s_primary, s.hit, s.key)
                    r.update(auc_engine=aN, auc_engine_ci=aNci, auc_critic=aK, auc_critic_ci=aKci, auc_comb=aC, auc_comb_ci=aCci,
                             comb_minus_engine=float(aC - aN), comb_minus_engine_ci=ci(bC - bN),
                             E_minus_engine=float(aE - aN), E_minus_engine_ci=ci(bE - bN))
            cf[f"{setname}_n{nmin}"] = r
    out["cofold"] = cf
    p = cf["cognate+uncalled_n10"]
    out["cofold_verdict_usable"] = bool(p.get("auc_E", 0) >= 0.60 and p.get("auc_E_ci", [0])[0] > 0.5 and
                                        p.get("comb_minus_engine", 0) >= 0.03 and p.get("comb_minus_engine_ci", [0])[0] > 0)
    out["cofold_refuted"] = bool(p.get("auc_E_ci", [0, 1])[0] <= 0.5)

    # ------------------------------------------------------------------ curated split
    cu = pd.read_parquet(W / "curated_pairs.parquet")
    cu = cu[cu.comp_a != cu.comp_b]
    cu["sys"] = np.where(cu.caller == "CRISPRCasTyper", "CRISPR", cu.system_type)
    kc = cu.groupby("key").agg(cls=("cls", lambda s: s.iloc[0] if s.nunique() == 1 else "both"), sys=("sys", "first"),
                               comps=("comp_a", lambda s: ""))
    kc = kc[kc.cls != "both"].join(R[["n", "E", "M", "p_hom"]], how="inner")
    cur = {}
    for nmin in (10, 30):
        for nm, msk in (("all", np.ones(len(kc), bool)), ("CRISPR", (kc.sys == "CRISPR").to_numpy()),
                        ("RM", kc.sys.isin(["RM_Type_I", "RM_Type_III", "RM_Type_II"]).to_numpy())):
            s = kc[msk & (kc.n >= nmin).to_numpy() & kc.E.notna().to_numpy()]
            y = (s.cls == "complex").to_numpy()
            r = dict(complex=int(y.sum()), noncomplex=int((~y).sum()))
            if y.sum() >= 2 and (~y).sum() >= 2:
                a_, aci, _ = unit_boot_auc(s.E, y, s.index.to_numpy())
                r.update(auc_E=a_, auc_E_ci=aci, median_E_complex=float(s.E[y].median()), median_E_noncomplex=float(s.E[~y].median()),
                         frac_p_le05_complex=float((s.p_hom[y] <= 0.05).mean()), frac_p_le05_noncomplex=float((s.p_hom[~y] <= 0.05).mean()))
                a2, a2ci, _ = unit_boot_auc(s.M, y, s.index.to_numpy())
                r.update(auc_M=a2, auc_M_ci=a2ci)
            cur[f"{nm}_n{nmin}"] = r
    out["curated"] = cur
    kc.reset_index().to_csv(W / "s1_curated_rows.tsv", sep="\t", index=False)

    # ------------------------------------------------------------------ showcase operon
    sop = R[R.S_op].E.dropna().to_numpy()
    abc = {}
    for nm, (x, y) in dict(AB=(81718, 393054), BC=(393054, 130016), AC=(81718, 130016)).items():
        k = min(x, y) * T + max(x, y)
        if k in R.index:
            r = R.loc[k]
            abc[nm] = dict(n=int(r.n), M=float(r.M), E=float(r.E), E_sd=float(r.E_sd), p_hom=float(r.p_hom), p_fam=float(r.p_fam),
                           cca=float(r.cca), rhoS_F=float(r.rhoS_F), rhoS_G=float(r.rhoS_G),
                           pct_E_in_S_op=float((sop < r.E).mean()))
    out["operon_ABC"] = abc
    json.dump(out, open(W / "s1_eval.json", "w"), indent=1, default=float)
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
