#!/usr/bin/env python
"""Design input for workstream ws_signals: how many distinct co-located instances each family pair has.

No vectors are read and no outcome is computed: only instance counts, so that the minimum instance count and the
evaluation sets can be fixed in predeclared.json before any statistic is seen.

Instances of a family pair (F, G) (rescue3 tokens, unordered): two genes on one contig at offset 1..3, one instance per
genome (the closest pair; ties -> lowest gid), then one per distinct (uid_F, uid_G) sequence pair.
Writes $RD/signals/work/pair_counts.parquet (key, lo, hi, n_genomes, n_distinct, n_off1_genomes) and prints the counts
for the evaluation sets.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

RD = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111")
PKG = RD / "ws_dpann/package"
EMB = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/embed_queue/runs/dpann__esmpp_small__20261111")
TOK = RD / "engine/dpann_tokens/mmseqs__seq__e3c50s75_rescue3__dpann.parquet"
OUT = RD / "signals/work"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    g = pd.read_parquet(PKG / "universe/genes.parquet", columns=["gid", "genome_code", "contig_code", "position_index", "strand"])
    assert (g.gid.to_numpy() == np.arange(len(g))).all()
    tok = pd.read_parquet(TOK).sort_values("gid").token.to_numpy(np.int64)
    uid = np.load(EMB / "uid_of_row.npy").astype(np.int64)
    gc = g.genome_code.to_numpy(np.int64)
    cc = g.contig_code.to_numpy(np.int64)
    n = len(g)
    T = int(tok.max()) + 1
    rows = []
    for off in (1, 2, 3):
        i = np.arange(n - off)
        j = i + off
        m = (cc[i] == cc[j]) & (tok[i] != tok[j])
        i, j = i[m], j[m]
        lo = np.minimum(tok[i], tok[j])
        swap = tok[i] > tok[j]
        a = np.where(swap, j, i)       # gene carrying token lo
        b = np.where(swap, i, j)
        rows.append(pd.DataFrame(dict(key=lo * T + np.maximum(tok[i], tok[j]), a=a, b=b, off=off, gc=gc[i])))
    d = pd.concat(rows, ignore_index=True)
    log("co-located pairs (offset 1-3, distinct tokens)", len(d))
    d = d.sort_values(["key", "gc", "off", "a"]).drop_duplicates(["key", "gc"])
    n_gen = d.groupby("key").size()
    n_off1 = d[d.off == 1].groupby("key").size()
    d["ua"], d["ub"] = uid[d.a.to_numpy()], uid[d.b.to_numpy()]
    dd = d.drop_duplicates(["key", "ua", "ub"])
    n_dist = dd.groupby("key").size()
    pc = pd.DataFrame(dict(n_genomes=n_gen, n_distinct=n_dist.reindex(n_gen.index).fillna(0).astype(int),
                           n_off1_genomes=n_off1.reindex(n_gen.index).fillna(0).astype(int)))
    pc.index.name = "key"
    pc = pc.reset_index()
    pc["lo"], pc["hi"] = pc.key // T, pc.key % T
    pc.to_parquet(OUT / "pair_counts.parquet", index=False)
    log("family pairs", len(pc))
    for k in (10, 20, 30, 50, 100, 300):
        log(f"pairs with >= {k} distinct instances: {(pc.n_distinct >= k).sum():,}")

    sig = pd.read_parquet(RD / "engine/results_dpann_repair/dpann__mmseqs__seq__e3c50s75_rescue3__w3/significant_pairs.parquet",
                          columns=["lo", "hi", "L"])
    sk = set((sig.lo.to_numpy(np.int64) * T + sig.hi.to_numpy(np.int64)).tolist())
    pc["called"] = pc.key.isin(sk)
    for k in (10, 30):
        log(f">= {k}: called {int((pc.called & (pc.n_distinct >= k)).sum()):,}, not called {int((~pc.called & (pc.n_distinct >= k)).sum()):,}")

    # folded pairs, screen 1
    f = pd.read_parquet(RD / "grammar/prep/folded.parquet")
    pos = pd.read_csv(sys.argv[1], sep="\t") if len(sys.argv) > 1 else None
    def keys(g0, g1):
        t0, t1 = tok[g0], tok[g1]
        return np.minimum(t0, t1) * T + np.maximum(t0, t1)
    ix = pc.set_index("key").n_distinct
    f["key"] = keys(f.g0.to_numpy(), f.g1.to_numpy())
    f["n_distinct"] = ix.reindex(f.key).fillna(0).astype(int).to_numpy()
    f["hit"] = f.iptm >= 0.75
    out = {}
    for cls in ("cognate", "uncalled"):
        s = f[f.cls == cls]
        out[cls] = {f">={k}": dict(folds=int((s.n_distinct >= k).sum()), hits=int((s.hit & (s.n_distinct >= k)).sum()),
                                     family_pairs=int(s[s.n_distinct >= k].key.nunique()))
                    for k in (5, 10, 20, 30, 50)}
        out[cls]["all"] = dict(folds=len(s), hits=int(s.hit.sum()), family_pairs=int(s.key.nunique()))
    if pos is not None:
        ab = pos.gids.str.split(";", expand=True).astype(np.int64)
        pos["key"] = keys(ab[0].to_numpy(), ab[1].to_numpy())
        pos["n_distinct"] = ix.reindex(pos.key).fillna(0).astype(int).to_numpy()
        pos["hit"] = pos.iptm >= 0.75
        out["positive"] = {f">={k}": dict(folds=int((pos.n_distinct >= k).sum()), hits=int((pos.hit & (pos.n_distinct >= k)).sum()),
                                          family_pairs=int(pos[pos.n_distinct >= k].key.nunique())) for k in (5, 10, 20, 30, 50)}
        out["positive"]["all"] = dict(folds=len(pos), hits=int(pos.hit.sum()), family_pairs=int(pos.key.nunique()))
    # showcase operon
    for nm, (x, y) in dict(AB=(81718, 393054), BC=(393054, 130016), AC=(81718, 130016)).items():
        k = min(x, y) * T + max(x, y)
        out[nm] = dict(n_distinct=int(ix.get(k, 0)), n_genomes=int(pc.set_index("key").n_genomes.get(k, 0)))
    print(json.dumps(out, indent=1))
    json.dump(out, open(OUT / "sizes_eval.json", "w"), indent=1)


if __name__ == "__main__":
    main()
