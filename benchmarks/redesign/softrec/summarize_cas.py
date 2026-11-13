#!/usr/bin/env python
"""ws_cas: summaries of the lists-step outputs (small files copied from $RD/softrec/cas/results to a local directory).

usage: summarize_cas.py DIR      reads DIR/{support_detail.npz, query_sets.npz, pairs_cas_instances.parquet,
                                 token_support_recovery.parquet, list_gene_metrics.parquet}; writes DIR/lists_summary.json
Genome bootstrap (2,000 draws, seed 17) for differences between groups; AUC by ranks (ties count one half).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

DCOLS_I = ["L_any", "L_fam", "L_rec", "L_famrec", "L_any200", "L_fam200", "L_bestrec", "mmin_any", "mmin_fam"]
DCOLS_F = ["W_any", "W_fam", "W_rec", "W_famrec", "W_any200", "W_fam200"]
STRATA = [(2, 2), (3, 3), (4, 4), (5, 7), (8, 15)]


def auc(pos, neg):
    if len(pos) == 0 or len(neg) == 0:
        return None
    r = rankdata(np.r_[pos, neg])
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def tpr(pos, neg, fpr):
    if len(pos) == 0 or len(neg) == 0:
        return None
    return float((pos > np.quantile(neg, 1 - fpr)).mean())


def boot_diff(gA, yA, gB, yB, draws=2000, seed=17):
    rng = np.random.default_rng(seed)
    allg, inv = np.unique(np.r_[gA, gB], return_inverse=True)
    ia, ib = inv[:len(gA)], inv[len(gA):]
    G = len(allg)
    sa, na = np.bincount(ia, yA, G), np.bincount(ia, minlength=G).astype(float)
    sb, nb = np.bincount(ib, yB, G), np.bincount(ib, minlength=G).astype(float)
    pick = rng.integers(0, G, size=(draws, G))
    dd = sa[pick].sum(1) / np.maximum(na[pick].sum(1), 1) - sb[pick].sum(1) / np.maximum(nb[pick].sum(1), 1)
    return [float(np.mean(yA) - np.mean(yB)), float(np.quantile(dd, .025)), float(np.quantile(dd, .975))]


def frame(z, name):
    I = z[f"{name}_I"]
    F = z[f"{name}_F"]
    df = pd.DataFrame(I, columns=DCOLS_I)
    for j, c in enumerate(DCOLS_F):
        df[c] = F[:, j]
    df["idx"] = z[f"{name}_idx"]
    return df


def q(x, qs=(0.1, 0.25, 0.5, 0.75, 0.9)):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return [round(float(v), 4) for v in np.quantile(x, qs)] if len(x) else []


def main():
    D = Path(sys.argv[1])
    z = np.load(D / "support_detail.npz")
    Q = np.load(D / "query_sets.npz")
    ci = pd.read_parquet(D / "pairs_cas_instances.parquet")
    out = {}
    cas = frame(z, "cas").merge(ci, on="idx", how="left")
    cas["Lt"] = cas.emb_Lt1000
    lost = cas[cas.tokS & ~cas.embS & (cas.Lt >= 1)].copy()
    found = cas[cas.tokS & cas.embS].copy()
    nl = frame(z, "noncas_lost")
    nl["tokL"] = Q["noncas_lost_tokL"]
    nf = frame(z, "noncas_found")
    nf["tokL"] = Q["noncas_found_tokL"]
    nulls = []
    for r in range(3):
        f = frame(z, f"null{r}")
        f["Lt"] = Q[f"null{r}_Lt"]
        nulls.append(f)
    null = pd.concat(nulls, ignore_index=True)
    # check: the kernel's L_any equals stored L - 1 + 1 (stored Lt = L + 1)
    out["check_L_any_vs_stored_cas"] = int(((lost.L_any + 1) == lost.Lt).sum()), int(len(lost))
    out["check_L_any_vs_stored_null"] = int(((null.L_any + 1) == null.Lt).sum()), int(len(null))

    # ---- recovery of the token's supporting genera in the lists (family-consistent support)
    def recov(df, tokcol="tokL"):
        other = np.maximum(df[tokcol].to_numpy() - 1, 1)
        return dict(n=int(len(df)), tok_other_genera_q=q(df[tokcol] - 1), L_fam_q=q(df.L_fam), L_fam200_q=q(df.L_fam200),
                    recovered_frac_mean=float((df.L_fam / other).mean()), recovered_all=float((df.L_fam >= other).mean()),
                    recovered_none=float((df.L_fam == 0).mean()),
                    W_fam_q=q(df.W_fam), W_fam_ge1=float((df.W_fam >= 1).mean()), W_any_q=q(df.W_any),
                    W_chance_q=q(df.W_any - df.W_fam), L_any_q=q(df.L_any), mmin_fam_q=q(df.mmin_fam[df.mmin_fam > 0]),
                    L_rec_q=q(df.L_rec), L_famrec_q=q(df.L_famrec))
    out["recovery"] = dict(cas_lost=recov(lost), cas_found=recov(found), noncas_lost=recov(nl), noncas_found=recov(nf))
    # by token L bin, Cas lost vs non-Cas lost
    by = {}
    for lo, hi in ((2, 2), (3, 3), (4, 7), (8, 15), (16, 10 ** 6)):
        a = lost[(lost.tokL >= lo) & (lost.tokL <= hi)]
        b = nl[(nl.tokL >= lo) & (nl.tokL <= hi)]
        if len(a) == 0:
            continue
        by[f"{lo}-{hi}"] = dict(cas=recov(a), noncas=recov(b))
    out["recovery_by_tokL"] = by
    # ---- reciprocity and other list-derived statistics at fixed Lt: lost Cas vs null sample
    stats = {"W_any": "W_any", "W_rec": "W_rec", "L_rec": "L_rec", "L_bestrec": "L_bestrec", "W_any200": "W_any200",
             "L_any200": "L_any200", "rec_share": None, "neg_mmin": None}
    roc = {}
    for lo, hi in STRATA:
        a = lost[(lost.Lt >= lo) & (lost.Lt <= hi)]
        b = null[(null.Lt >= lo) & (null.Lt <= hi)]
        c = nl.copy()
        c = c[(c.L_any + 1 >= lo) & (c.L_any + 1 <= hi)]
        item = dict(n_cas=int(len(a)), n_null=int(len(b)), n_noncas_lost=int(len(c)))
        for k, col in stats.items():
            def val(df):
                if k == "rec_share":
                    return (df.L_rec / np.maximum(df.L_any, 1)).to_numpy(float)
                if k == "neg_mmin":
                    return -df.mmin_any.to_numpy(float)
                return df[col].to_numpy(float)
            item[k] = dict(auc_cas=auc(val(a), val(b)), auc_noncas_lost=auc(val(c), val(b)),
                           tpr_cas_fpr1e2=tpr(val(a), val(b), 1e-2), tpr_cas_fpr1e3=tpr(val(a), val(b), 1e-3))
        # family-consistent support is an oracle quantity (needs MMseqs2 families): reported as the separation ceiling
        item["oracle_L_fam"] = dict(auc_cas=auc(a.L_fam.to_numpy(float), b.L_fam.to_numpy(float)),
                                    null_share_Lfam_ge1=float((b.L_fam >= 1).mean()) if len(b) else None,
                                    cas_share_Lfam_ge1=float((a.L_fam >= 1).mean()) if len(a) else None)
        roc[f"{lo}-{hi}"] = item
    out["auc_list_stats"] = roc
    # chance support composition of null instances vs lost Cas: share of supporting lineages that are family-consistent
    out["support_composition"] = dict(
        cas_lost_fam_share=float((lost.L_fam / np.maximum(lost.L_any, 1)).mean()),
        cas_found_fam_share=float((found.L_fam / np.maximum(found.L_any, 1)).mean()),
        null_fam_share=float((null.L_fam / np.maximum(null.L_any, 1)).mean()),
        noncas_lost_fam_share=float((nl.L_fam / np.maximum(nl.L_any, 1)).mean()))
    # ---- token-support recovery (per genus classes: both members in lists / a only / b only / neither)
    ts = pd.read_parquet(D / "token_support_recovery.parquet")
    tsum = {}
    for g, df in ts.groupby("group"):
        tg = df.tok_genera.sum()
        tsum[g] = dict(n=int(len(df)), tok_genera=int(tg), both=float(df.both.sum() / max(tg, 1)), a_only=float(df.a_only.sum() / max(tg, 1)),
                       b_only=float(df.b_only.sum() / max(tg, 1)), neither=float(df.neither.sum() / max(tg, 1)),
                       both200=float(df.both200.sum() / max(tg, 1)), W_q=q(df.W), W_ge1=float((df.W >= 1).mean()),
                       m_best_q=q(df.m_best[df.m_best > 0]), m_med_q=q(df.m_med[df.m_med > 0]),
                       cos_partner_q50=q(np.r_[df.cos_a_q50, df.cos_b_q50]), cap_cos_q=q(np.r_[df.cap_a, df.cap_b]),
                       frac_homolog_above_cap=q(np.r_[df.frac_x_above_cap, df.frac_y_above_cap]),
                       tokL_q=q(df.tokL))
    out["token_support"] = tsum
    # Cas lost vs non-Cas lost at matched token L (per-instance share of the token's genera found in lists), bootstrap
    tsl = ts[ts.group == "cas_lost"]
    tsn = ts[ts.group == "noncas_lost"]
    cmp = {}
    for lo, hi in ((2, 2), (3, 3), (4, 7), (8, 10 ** 6)):
        a = tsl[(tsl.tokL >= lo) & (tsl.tokL <= hi) & (tsl.tok_genera > 0)]
        b = tsn[(tsn.tokL >= lo) & (tsn.tokL <= hi) & (tsn.tok_genera > 0)]
        if len(a) == 0 or len(b) == 0:
            continue
        ya, yb = (a.both / a.tok_genera).to_numpy(), (b.both / b.tok_genera).to_numpy()
        cmp[f"{lo}-{hi}"] = dict(n_cas=int(len(a)), n_noncas=int(len(b)), found_share_cas=float(ya.mean()), found_share_noncas=float(yb.mean()),
                                 diff=boot_diff(a.genome.to_numpy(), ya, b.genome.to_numpy(), yb),
                                 W_ge1_cas=float((a.W >= 1).mean()), W_ge1_noncas=float((b.W >= 1).mean()))
    out["token_support_cas_vs_noncas"] = cmp
    # ---- part 3: lists of single genes
    gm = pd.read_parquet(D / "list_gene_metrics.parquet")
    p3 = {}
    for g, df in gm.groupby("group"):
        p3[g] = dict(n=int(len(df)), fam_genes_q=q(df.fam_genes), genera_out_q=q(df.genera_out), n_out_q=q(df.n_out),
                     frac_in_q=q(df.frac_in), frac_in_mean=float(df.frac_in.mean()), frac_in_cap_mean=float(df.frac_in_cap.mean()),
                     all_in=float((df.frac_in >= 1).mean()), none_in=float((df.n_in == 0)[df.n_out > 0].mean()),
                     genera_reached_frac_mean=float((df.genera_reached / np.maximum(df.genera_out, 1))[df.genera_out > 0].mean()),
                     rank_med_q=q(df.rank_med), rank_max_q=q(df.rank_max), first_fam_pos_q=q(df.first_fam_pos[df.first_fam_pos >= 0]),
                     intrusions_q=q(df.intrusions), cos_in_q=q(df.cos_in_med), cos_out_q=q(df.cos_out_med), cos_out_max_q=q(df.cos_out_max),
                     cap_cos_q=q(df.cap_cos), nonfam_q=q(df.nonfam), nonfam_cas_q=q(df.nonfam_cas), nonfam_cas_mean=float(df.nonfam_cas.mean()),
                     nonfam_casfam_mean=float(df.nonfam_casfam.mean()), nonfam_same_class_mean=float(df.nonfam_same_class.mean()),
                     top10_fam_q=q(df.top10_fam), top50_fam_q=q(df.top50_fam),
                     # family members outside the list that score above the list's last cosine (index recall misses)
                     out_above_cap=float((df.cos_out_max > df.cap_cos).mean()))
    out["part3"] = p3
    (D / "lists_summary.json").write_text(json.dumps(out, indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x)) + "\n")
    print(json.dumps(out, indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x))[:20000])


if __name__ == "__main__":
    main()
