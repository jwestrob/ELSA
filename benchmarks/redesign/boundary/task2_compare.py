#!/usr/bin/env python
"""Task 2 comparison: merged universe (split pairs as one position, rescue3 tokens, w3) against the default
(unmerged rescue3, w3), both scored with the engine's full locus table (score_min2), paired by instance_id.

Sets (predeclared.json task2): contains_pair = both pieces of a split pair among the members and >= 2 members after
merging; unaffected = no piece inside the first-to-last member span; remaining = >= 2 members in both universes;
touched = any member is a piece. Genome bootstrap 2,000 draws, seed 17; tune = odd, test = even genome_code.
Also: the ribosomal check on the merged universe (runs remapped) and, when given, a Task-1 rule read on both universes.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import blib as B  # noqa: E402
import rules as RU  # noqa: E402

MERGED = B.BD / "merged"


def boot_pooled(d, g, rng, draws=2000):
    return RU.boot_mean(d, g, rng, draws)


def compare(A, Bf, sets, gcode, label):
    out = {}
    ids = A.index.intersection(Bf.index)
    x, y = A.loc[ids], Bf.loc[ids]
    s = sets.set_index("instance_id").loc[ids]
    g = x.genome.map(gcode).to_numpy()
    for split, m in (("even", g % 2 == 0), ("odd", g % 2 == 1)):
        rng = np.random.default_rng(17)
        o = {}
        for name, sm in (("contains_pair", s.contains_pair.to_numpy() & (s.n_new.to_numpy() >= 2)),
                         ("touched", s.touched.to_numpy() & (s.n_new.to_numpy() >= 2)),
                         ("remaining", s.n_new.to_numpy() >= 2),
                         ("unaffected", ~s.span_has_piece.to_numpy())):
            k = m & sm
            d = dict(n=int(k.sum()))
            if k.any():
                d["f1_A"] = float(x.f1.to_numpy()[k].mean())
                d["f1_B"] = float(y.f1.to_numpy()[k].mean())
                d["d_f1"] = boot_pooled((x.f1.to_numpy() - y.f1.to_numpy())[k], g[k], rng)
                d["d_excess"] = boot_pooled((x.excess.to_numpy() - y.excess.to_numpy())[k], g[k], rng)
                d["by_caller"] = {}
                for c in RU.CALLERS:
                    kc = k & (x.caller.to_numpy() == c)
                    if kc.sum() >= 2:
                        d["by_caller"][c] = dict(n=int(kc.sum()),
                                                 d_f1=boot_pooled((x.f1.to_numpy() - y.f1.to_numpy())[kc], g[kc], rng),
                                                 d_excess=boot_pooled((x.excess.to_numpy() - y.excess.to_numpy())[kc], g[kc], rng))
            o[name] = d
        # summed DF + TXSS excess on unaffected and on remaining (dpann_compare form)
        for name, sm in (("unaffected", ~s.span_has_piece.to_numpy()), ("remaining", s.n_new.to_numpy() >= 2)):
            keep = ids[sm]
            pr = RU.paired(A.loc[keep], Bf.loc[keep], gcode, split)
            o[f"summed_{name}"] = pr["summed"]
            o[f"paired_{name}"] = pr
        out[split] = o
    B.log(label, json.dumps({k: {kk: (vv.get("n"), vv.get("d_f1")) for kk, vv in v.items() if isinstance(vv, dict) and "n" in vv}
                             for k, v in out.items()}))
    return out


def load_scores(path):
    t = pd.read_parquet(path)
    t["excess"] = t.f1 - t.control_mean_f1
    return t.set_index("instance_id")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, required=True)
    ap.add_argument("--merged-result", required=True)
    ap.add_argument("--rule", default="", help="Task-1 rule as family:json-params, read on both universes")
    a = ap.parse_args()
    import numba
    numba.set_num_threads(a.threads)
    t0 = time.time()
    gcode = B.odd_even_code()
    sets = pd.read_parquet(MERGED / "instance_sets.parquet")
    mres = Path(a.merged_result)
    default = B.ARMS["rescue3"]
    A = load_scores(mres / "score_min2/instance_scores.parquet")
    Bf = load_scores(default / "score_min2/instance_scores.parquet")
    res = dict(merged=str(mres), default=str(default), full_table=compare(A, Bf, sets, gcode, "full table"))
    # ribosomal check, conserved@2 and the ladder, merged universe
    runs = pd.read_parquet(MERGED / "ribo_runs.parquet")
    bm = np.load(mres / "boundary_support.npy")
    contigs = pd.read_parquet(MERGED / "pkg/universe/contigs.parquet").sort_values("first_gid")
    cs = contigs.first_gid.to_numpy().astype(np.int64)
    ce = cs + contigs.n_genes.to_numpy().astype(np.int64)
    rib = {}
    for lv in (2, 5, 12, 30, 80):
        s, e = B.R.runs_at_least(bm, cs, ce, float(lv), 0)
        rib[f"conserved@{lv}"] = B.ribo_summary(runs, B.ribo_score(runs, s, e))
    res["ribosomal_merged"] = rib
    B.log("ribosomal merged", {k: (v["recovered"], v["resolved"], v["parvarchaeaceae"]) for k, v in rib.items()})
    if a.rule:
        fam, params = a.rule.split(":", 1)
        params = json.loads(params)
        Lm = B.Landscape(mres, universe=MERGED / "pkg/universe", tokens=MERGED / "tokens_rescue3_merged.parquet")
        Ld = B.Landscape(default)
        out = {}
        for tag, L, pkg in (("merged", Lm, MERGED / "pkg"), ("default", Ld, B.PKG)):
            s, e = RU.Builder(L).build(fam, params)
            df = L.table(s, e)
            per, _ = B.score_table(pkg, df, a.threads)
            per.to_parquet(MERGED / f"rule_scores_{tag}.parquet", index=False)
            out[tag] = per.set_index("instance_id")
            rr = runs if tag == "merged" else B.ribo_runs()
            res[f"ribosomal_rule_{tag}"] = B.ribo_summary(rr, B.ribo_score(rr, df.gid_s.to_numpy(), df.gid_e.to_numpy()))
        res["rule"] = dict(family=fam, params=params, comparison=compare(out["merged"], out["default"], sets, gcode, "rule"))
    res["seconds"] = round(time.time() - t0, 1)
    with open(B.BD / "results/task2_compare.json", "w") as fh:
        json.dump(res, fh, indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x))
    B.log("done", res["seconds"])


if __name__ == "__main__":
    main()
