#!/usr/bin/env python
"""Diagnostics after stage F (post hoc, descriptive).

1. Calibrated V0 (cos5) accepted anchor pairs split by origin (original seed vs supported
   proposal), with visibility, map evidence and target stratum.
2. How many of V0's accepted sequence-visible pairs each other selection also accepts.
3. Anchors of V1 blocks with negative weight (what score-first chaining removes), by visibility.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import MODELS, W, log, universe_keys, write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, required=True)
    a = ap.parse_args()
    import pairs as pr
    con = pr.connect(a.threads)
    pr.load_labels(con)
    qk, tk = universe_keys()
    u = pd.read_parquet(Path(pr.RD) / "universe/genes.parquet", columns=["side", "row_index", "genome_code"])
    g = pd.read_parquet(Path(pr.RD) / "universe/genomes.parquet", columns=["genome_code", "stratum"])
    ref_stratum = u[u.side == "ref"].sort_values("row_index").merge(g, on="genome_code", how="left").stratum.to_numpy()
    out = {}
    for m in MODELS:
        runs, res = W / "runs" / m, W / "results" / m

        def accepted(v, rule):
            calls = pd.read_parquet(res / f"calls__{v}.parquet")
            table = pd.read_csv(res / "table.tsv", sep="\t")
            an = pd.read_parquet(runs / f"anchors__{v}.parquet")
            if rule == "raw":
                keep = calls.block_id
            elif rule == "q5":
                keep = calls.block_id[calls.q <= 0.05]
            else:
                # recompute cos5 acceptance from the stored cuts
                import json
                cuts = json.load(open(res / "summary.json"))["variants"][v]["cos5_cuts"]
                s = np.minimum(calls.n_anchors, 6)
                ok = np.zeros(len(calls), bool)
                for k, t in cuts.items():
                    if t is not None:
                        ok |= (s == int(k)).to_numpy() & (calls.m.to_numpy() >= t)
                keep = calls.block_id[ok]
                exp = int(table[(table.variant == v) & (table.rule == "cos5")].blocks.iloc[0])
                assert len(keep) == exp, (v, len(keep), exp)
            return an[an.block_id.isin(set(keep))]

        def label(df):
            df = df.assign(qkey=qk[df.query_row_index.to_numpy()], tkey=tk[df.target_row_index.to_numpy()])
            con.register("x", df[["qkey", "tkey"]].drop_duplicates())
            lab = con.execute("SELECT x.qkey, x.tkey, COALESCE(lab.evalue <= 1e-3, FALSE) AS vis FROM x LEFT JOIN lab USING (qkey, tkey)").df()
            con.unregister("x")
            return df.merge(lab, on=["qkey", "tkey"], how="left")

        r = {}
        v0 = label(accepted("context60__lf", "cos5"))
        sel = pd.read_parquet(runs / "selection__context60.parquet", columns=["query_row_index", "target_row_index", "is_original"])
        v0 = v0.merge(sel, on=["query_row_index", "target_row_index"], how="left", validate="one_to_one")
        v0["stratum"] = ref_stratum[v0.target_row_index.to_numpy()]
        for name, s in [("original", v0.is_original), ("proposal", ~v0.is_original)]:
            x = v0[s]
            r[f"v0cos5_{name}"] = dict(pairs=len(x), vis=float(x.vis.mean()), w_neg=float((x.w < 0).mean()),
                                       vis_among_w_neg=float(x.vis[x.w < 0].mean()) if (x.w < 0).any() else None,
                                       by_stratum={k: dict(pairs=int(len(y)), vis=float(y.vis.mean()))
                                                   for k, y in x.groupby("stratum")})
        vis_pairs = set(zip(v0.query_row_index[v0.vis], v0.target_row_index[v0.vis]))
        r["v0cos5_visible_pairs"] = len(vis_pairs)
        for v, rule in [("seeds__lf", "raw"), ("ev_p50__lf", "cos5"), ("ev_p80__lf", "raw"), ("ev_p95__lf", "raw"),
                        ("context60__sf0", "cos5"), ("seeds__sf0", "raw")]:
            x = label(accepted(v, rule))
            xs = set(zip(x.query_row_index[x.vis], x.target_row_index[x.vis]))
            r[f"{v}_{rule}"] = dict(pairs=len(x), visible=len(xs), shared_with_v0_visible=len(xs & vis_pairs),
                                    v0_visible_missed=len(vis_pairs - xs))
        v1 = label(pd.read_parquet(runs / "anchors__seeds__lf.parquet"))
        neg = v1.w < 0
        r["v1_lf_anchors"] = dict(pairs=len(v1), w_neg=int(neg.sum()), vis_w_neg=float(v1.vis[neg].mean()),
                                  vis_w_pos=float(v1.vis[~neg].mean()))
        out[m] = r
        log(m, r)
    write_json(W / "results" / "diag.json", out)


if __name__ == "__main__":
    main()
