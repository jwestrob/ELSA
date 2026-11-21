#!/usr/bin/env python
"""Stage D for one model: every selection x chainer, observed and three permutations.

Selections: context60 (V0, published kernel), seeds (V1), ev_p50/ev_p80/ev_p95 (V2).
Chainers:   lf  = published length-first;  sf0 = score-first, no gap cost;
            sfc = score-first, concave gap cost 0.5*log2(g+1)   (V3).
Variant id = <selection>__<chainer>. Each permutation (seeds 17/18/19) reorders query genes
before selection and chaining. Observed runs save blocks and their anchor pairs; permuted
runs save per-block statistics.
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (PERM_SEEDS, S, SAVED_V0, TAGS, W, array_paths, base, evidence, load_map, load_meta,
                    load_published, load_stats, log, model_dirs, ordered_metadata, pair_score, write_json)

SELECTIONS = ("context60", "seeds", "ev_p50", "ev_p80", "ev_p95")
CHAINERS = (("lf", "length_first", 0.0), ("sf0", "score_first", 0.0), ("sfc", "score_first", 0.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--workers", type=int, required=True)
    ap.add_argument("--selections", default=",".join(SELECTIONS))
    ap.add_argument("--smoke", action="store_true", help="first 3,000 query rows, dummy map, runs/<m>_smoke")
    ap.add_argument("--llr", action="store_true",
                    help="post hoc: weights = logit P - logit(pool prevalence); output runs/<m>_llr")
    a = ap.parse_args()
    os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")
    import numba
    numba.set_num_threads(a.workers)
    import run_spacedust_scaling as rss
    from spacedust_prepared_extension_executor import extend_prepared_queries
    import chain_v3
    import extend_v2

    m = a.model
    work, runs = model_dirs(m)
    if a.smoke or a.llr:
        runs = W / "runs" / (f"{m}_smoke" if a.smoke else f"{m}_llr")
        runs.mkdir(parents=True, exist_ok=True)
    tmp = W / "tmp" / m
    tmp.mkdir(parents=True, exist_ok=True)
    t_all = time.time()
    paths = array_paths(m)
    qmeta, rmeta = load_meta()
    chain_mod, group_type = load_published()
    if a.smoke and not (work / "map.json").exists():
        kind, x, y = "z2", np.array([0.0, 2.0, 6.0]), np.array([0.05, 0.5, 0.97])
    else:
        kind, x, y = load_map(m)
    qst, rst = load_stats(m)
    qst_np, rst_np = np.asarray(qst), np.asarray(rst)
    offset = 0.0
    if a.llr:
        import json
        pi = json.load(open(work / "map_report.json"))["prevalence"]["train"]
        offset = float(np.log(pi / (1 - pi)))
        log(m, "LLR weights: training pool prevalence", pi, "offset", offset)
    seeds = pd.read_parquet(base(m) / "seeds.parquet")
    pp = pd.read_parquet(S / "results/permuted_panels" / f"{m}_permuted_blocks.parquet", columns=["seed"])
    saved = (len(pd.read_parquet(base(m) / "all_blocks.parquet", columns=["block_id"])),
             [int((pp.seed == s).sum()) for s in PERM_SEEDS])
    if m in SAVED_V0:
        assert saved == (SAVED_V0[m][0], SAVED_V0[m][1]), saved
    rows = np.arange(3000 if a.smoke else len(qmeta), dtype=np.int64)
    selections = a.selections.split(",")
    perm_rows, summary = [], dict(model=m, kind=kind, workers=a.workers, runs={})

    for tag in TAGS:
        meta = qmeta
        p = dict(paths)
        if tag != "observed":
            meta = ordered_metadata(qmeta, int(tag[-2:]))
            np.save(tmp / f"query_neighbors_{tag}.npy", rss.neighbors(meta))
            p["query_neighbors"] = tmp / f"query_neighbors_{tag}.npy"
        for sel in selections:
            t = time.time()
            if sel == "context60":
                ext = extend_prepared_queries(p, rows, workers=a.workers)
                anchors = ext.pop("anchors")[["query_row_index", "target_row_index", "similarity", "is_original"]]
            elif sel == "seeds":
                anchors = seeds[seeds.query_row_index <= rows.max()].assign(is_original=True)
            else:
                tail = sel.split("_p")[1]
                gate = offset if tail == "prior" else float(np.log(int(tail) / 100.0 / (1 - int(tail) / 100.0)))
                params = dict(kind=kind, x=x, y=y, gate=gate, k=60, offset=offset)
                anchors, _ = extend_v2.run("extend_v2", p, {"qstats": work / "query_stats.npy",
                                                             "rstats": work / "reference_stats.npy"},
                                           params, rows, a.workers)
            sel_seconds = time.time() - t
            cos = anchors.similarity.to_numpy(np.float64)
            w = evidence(pair_score(kind, anchors.query_row_index.to_numpy(), anchors.target_row_index.to_numpy(),
                                    cos, qst_np, rst_np), x, y) - offset
            n_sup = int(anchors["supported"].sum()) if "supported" in anchors else None
            for cname, mode, gap in CHAINERS:
                t2 = time.time()
                blocks, ba = chain_v3.chain(anchors, meta, rmeta, chain_mod, group_type, mode=mode, cos=cos,
                                            weights=w, gap_scale=gap, want_anchors=(tag == "observed"))
                ch_seconds = time.time() - t2
                v = f"{sel}__{cname}"
                rec = dict(anchors=len(anchors), supported=n_sup, original=int(anchors.is_original.sum()),
                           blocks=len(blocks), anchor_pairs=int(blocks.n_anchors.sum()) if len(blocks) else 0,
                           selection_seconds=sel_seconds, chain_seconds=ch_seconds)
                summary["runs"].setdefault(v, {})[tag] = rec
                log(m, tag, v, rec)
                if v == "context60__lf" and not a.smoke:
                    want = saved[0] if tag == "observed" else saved[1][PERM_SEEDS.index(int(tag[-2:]))]
                    assert len(blocks) == want, (tag, len(blocks), want)
                if tag == "observed":
                    blocks.to_parquet(runs / f"blocks__{v}.parquet", index=False)
                    ba = ba.assign(cos=cos[ba.anchor_row.to_numpy()], w=w[ba.anchor_row.to_numpy()])
                    ba.drop(columns="anchor_row").to_parquet(runs / f"anchors__{v}.parquet", index=False)
                elif len(blocks):
                    perm_rows.append(blocks[["n_anchors", "chain_score", "sum_cos", "sum_w"]].assign(
                        variant=v, seed=int(tag[-2:])))
            if tag == "observed":
                anchors.assign(w=w).to_parquet(runs / f"selection__{sel}.parquet", index=False)
        write_json(runs / "run_summary.json", summary)
    pd.concat(perm_rows, ignore_index=True).to_parquet(runs / "perm_blocks.parquet", index=False)
    summary["seconds"] = time.time() - t_all
    summary["weight_offset"] = offset
    write_json(runs / "run_summary.json", summary)
    log("done", m, summary["seconds"])


if __name__ == "__main__":
    main()
