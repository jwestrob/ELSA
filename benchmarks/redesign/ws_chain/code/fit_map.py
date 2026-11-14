#!/usr/bin/env python
"""Step 1 of the design: choose the pair score and fit P(sequence-visible | score).

Training = pool-sample queries from the six training genomes; held-out = the three genomes
named in common.HELDOUT. Unique (q, c) pairs (seeds and proposals pooled). Label =
MMseqs2 E <= 1e-3. Rule: highest training AUROC; a score within 0.005 of the best is
preferred if it comes earlier in SCORE_KINDS. Map: isotonic regression, clipped to
P in [0.005, 0.995] at use.
"""
import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import MODELS, P_CLIP, SCORE_KINDS, W, evidence, load_stats, log, pair_score, universe_keys, write_json

GATES = (0.5, 0.8, 0.95)


def labels_for(pairs, threads):
    import pairs as pr
    con = pr.connect(threads)
    pr.load_labels(con)
    con.register("x", pairs[["qkey", "tkey"]])
    lab = con.execute("SELECT x.qkey, x.tkey, lab.evalue FROM x LEFT JOIN lab USING (qkey, tkey)").df()
    return pairs.merge(lab, on=["qkey", "tkey"], how="left", validate="many_to_one")


def calib_table(p, y, bins=10):
    edges = np.unique(np.quantile(p, np.linspace(0, 1, bins + 1)))
    b = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    rows = []
    for i in range(len(edges) - 1):
        s = b == i
        if s.any():
            rows.append(dict(bin=i, n=int(s.sum()), p_mean=float(p[s].mean()), observed=float(y[s].mean())))
    return rows


def fit_one(m, threads):
    from sklearn.isotonic import IsotonicRegression
    from sklearn.metrics import average_precision_score, roc_auc_score
    qk, tk = universe_keys()
    qst, rst = load_stats(m)
    qst, rst = np.asarray(qst), np.asarray(rst)
    pool = pd.read_parquet(W / "work" / m / "pool.parquet")
    pairs = pool[pool.kind == "pair"].reset_index(drop=True)
    occ = pool[pool.kind == "occ"].reset_index(drop=True)
    pairs["qkey"] = qk[pairs.query_row_index.to_numpy()]
    pairs["tkey"] = tk[pairs.target_row_index.to_numpy()]
    pairs = labels_for(pairs, threads)
    res = dict(model=m, n_pairs=len(pairs), unscored=int(pairs.evalue.isna().sum()))
    pairs["vis"] = (pairs.evalue <= 1e-3).fillna(False).to_numpy()
    q, t, c = pairs.query_row_index.to_numpy(), pairs.target_row_index.to_numpy(), pairs.similarity.to_numpy()
    scores = {k: pair_score(k, q, t, c, qst, rst) for k in SCORE_KINDS}
    ok = np.all([np.isfinite(v) for v in scores.values()], axis=0)
    res["nonfinite_scores"] = int((~ok).sum())
    tr = (~pairs.heldout.to_numpy()) & ok
    ho = pairs.heldout.to_numpy() & ok
    seed = pairs.is_original.to_numpy()
    y = pairs.vis.to_numpy()
    res["prevalence"] = {name: float(y[mask].mean()) for name, mask in
                         [("train", tr), ("train_seeds", tr & seed), ("train_proposals", tr & ~seed),
                          ("heldout", ho), ("heldout_seeds", ho & seed), ("heldout_proposals", ho & ~seed)]}
    res["counts"] = {name: int(mask.sum()) for name, mask in
                     [("train", tr), ("train_seeds", tr & seed), ("train_proposals", tr & ~seed),
                      ("heldout", ho), ("heldout_seeds", ho & seed), ("heldout_proposals", ho & ~seed)]}
    auc = {}
    for k, s in scores.items():
        auc[k] = {}
        for name, mask in [("train", tr), ("train_seeds", tr & seed), ("train_proposals", tr & ~seed),
                           ("heldout", ho), ("heldout_seeds", ho & seed), ("heldout_proposals", ho & ~seed)]:
            auc[k][name] = dict(auroc=float(roc_auc_score(y[mask], s[mask])),
                                ap=float(average_precision_score(y[mask], s[mask])))
    res["auc"] = auc
    best = max(auc[k]["train"]["auroc"] for k in SCORE_KINDS)
    kind = next(k for k in SCORE_KINDS if auc[k]["train"]["auroc"] >= best - 0.005)
    res["kind"] = kind
    s = scores[kind]
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip")
    iso.fit(s[tr], y[tr].astype(np.float64))
    x, yy = iso.X_thresholds_.astype(np.float64), iso.y_thresholds_.astype(np.float64)
    res["map_knots"] = len(x)
    w = evidence(s, x, yy)
    p = 1 / (1 + np.exp(-w))
    res["heldout_calibration"] = {name: dict(brier=float(np.mean((p[mask] - y[mask]) ** 2)),
                                             brier_prevalence=float(np.mean((y[mask].mean() - y[mask]) ** 2)),
                                             mean_p=float(p[mask].mean()), observed=float(y[mask].mean()),
                                             bins=calib_table(p[mask], y[mask]))
                                  for name, mask in [("pooled", ho), ("seeds", ho & seed), ("proposals", ho & ~seed)]}
    # gates: own evidence of proposals, witness evidence of occurrences, and the two combined
    ow = evidence(pair_score(kind, occ.witness_query_row.to_numpy(), occ.witness_target_row.to_numpy(),
                             occ.witness_similarity.to_numpy(), qst, rst), x, yy)
    occ = occ.assign(w_wit=ow)
    best_wit = occ.groupby(["query_row_index", "target_row_index"]).w_wit.max().rename("w_wit_best").reset_index()
    pairs = pairs.assign(w=w).merge(best_wit, on=["query_row_index", "target_row_index"], how="left")
    gates = {}
    for g in GATES:
        L = np.log(g / (1 - g))
        out = {}
        for split, sm in [("train", ~pairs.heldout), ("heldout", pairs.heldout)]:
            prop = pairs[sm & pairs.is_proposed & ~pairs.is_original]
            own = prop.w >= L
            both = own & (prop.w_wit_best >= L)
            out[split] = dict(proposals=len(prop), vis_all=float(prop.vis.mean()),
                              pass_own=int(own.sum()), vis_pass_own=float(prop.vis[own].mean()) if own.any() else None,
                              pass_both=int(both.sum()), vis_pass_both=float(prop.vis[both].mean()) if both.any() else None,
                              witness_occurrences=int((occ.heldout == (split == "heldout")).sum()),
                              witness_pass=int(((occ.heldout == (split == "heldout")) & (occ.w_wit >= L)).sum()))
        gates[str(g)] = out
    res["gates"] = gates
    # published 0.85 gate and supported-first for reference: proposals with cosine >= 0.85
    for split, sm in [("train", ~pairs.heldout), ("heldout", pairs.heldout)]:
        prop = pairs[sm & pairs.is_proposed & ~pairs.is_original]
        g = prop.similarity >= 0.85
        res.setdefault("published_gate", {})[split] = dict(pass_frac=float(g.mean()), vis_pass=float(prop.vis[g].mean()))
    write_json(W / "work" / m / "map.json", dict(kind=kind, x=x.tolist(), y=yy.tolist(), clip=P_CLIP))
    write_json(W / "work" / m / "map_report.json", res)
    log(m, "kind", kind, "train AUROC", {k: round(auc[k]["train"]["auroc"], 4) for k in SCORE_KINDS})
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, required=True)
    a = ap.parse_args()
    per = max(1, a.threads // len(MODELS))
    with ProcessPoolExecutor(max_workers=len(MODELS)) as ex:
        out = list(ex.map(fit_one, MODELS, [per] * len(MODELS)))
    write_json(W / "results" / "map_summary.json", {r["model"]: {k: r[k] for k in ("kind", "auc", "prevalence", "counts", "unscored")} for r in out})
