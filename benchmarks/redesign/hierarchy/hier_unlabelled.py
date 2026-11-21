#!/usr/bin/env python
"""Post hoc (not predeclared): what are the unlabelled hits of the hierarchy-evidence searches?

Applies round 2's evidence classification (locus_search/code/evidence.py: raw DefenseFinder profile of the
query's type in the window = df_own; engine-module agreement = c_module; supported = either) to unlabelled hits
at E <= 0.01 of each hierarchy variant on even-genome DefenseFinder queries, with 10 span-matched control
windows per hit. Observations only; no system is named for any hit."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RD = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111")
sys.path.insert(0, str(RD / "locus_search/code"))
import bench  # noqa: E402
import evidence as evm  # noqa: E402
import locus_search as ls  # noqa: E402

OUT = RD / "hierarchy/search"


def main():
    db = evm.attach_db()
    callers = bench.load_gold(db)
    queries = pd.read_parquet(RD / "locus_search/out/queries.parquet")
    H = pd.concat([pd.read_parquet(f, filters=[("evalue", "<=", 0.01)]) for f in sorted(OUT.glob("final_even_hits_E10*.parquet"))],
                  ignore_index=True)
    M = pd.concat([pd.read_parquet(f, columns=["qid", "method", "n_rel"]) for f in sorted(OUT.glob("final_even_metrics*.parquet"))])
    qset = set(M[M.n_rel > 0].qid)
    nq = len(qset)
    H = H[H.qid.isin(qset)].merge(queries[["qid", "n_members", "tcode", "type", "genome_code"]], on="qid")
    U = H[H.label == 0].reset_index(drop=True)
    ev = evm.Evidence(db)
    rng = np.random.default_rng(17)
    F, G = evm.classify(ev, U, queries, callers, rng)
    U = pd.concat([U, F], axis=1)
    G["method"] = U.method.to_numpy()[G.hit.to_numpy()]
    U.to_parquet(OUT / "unlabelled_hits_E0.01.parquet", index=False)
    res = dict(queries=nq, methods={})
    for m, g in U.groupby("method"):
        gc = G[G.method == m]
        gn = gc[~gc.overlaps_hit]
        d = dict(unlabelled_hits=int(len(g)), per_query=len(g) / nq, supported_per_query=float(g.supported.sum() / nq),
                 unsupported_per_query=float((~g.supported).sum() / nq),
                 supported_share_ci=ls.boot_mean(g.supported.astype(float), g.genome_code),
                 slots_filled_mean=float(g.n_slots_filled.mean()), single_slot_share=float((g.n_slots_filled == 1).mean()),
                 queries_with_unlabelled=int(g.qid.nunique()),
                 top10_query_share=float(g.qid.value_counts().head(10).sum() / len(g)),
                 types_top=g.type.value_counts().head(6).to_dict())
        for fl in ("df_own", "own_ge2", "c_module", "supported"):
            d[fl] = dict(hits=float(g[fl].mean()), control_not_overlapping_hit=float(gn[fl].mean()) if len(gn) else None)
        res["methods"][m] = d
        print(m, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in d.items()
                             if k in ("unlabelled_hits", "per_query", "supported_per_query", "unsupported_per_query")}),
              "df_own", round(d["df_own"]["hits"], 3), "ctrl", d["df_own"]["control_not_overlapping_hit"], flush=True)
    json.dump(res, open(OUT / "unlabelled_summary.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
