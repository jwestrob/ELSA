#!/usr/bin/env python
"""Task 4: are the saved levels nested? Build nested versions by top-down intersection.

For consecutive levels (ordered by token count, fine -> coarse) reports the fraction of genes whose finer cluster
lies entirely inside one coarser cluster, the fraction of finer clusters of size >= 2 that are split across coarser
clusters, and the gene-weighted share of each finer cluster in its plurality coarser cluster. Nested construction:
the coarsest level is kept; going down, each level is replaced by its intersection with the nested level above it.
Writes $RD/hierarchy/levels/nested__<set>__<level>.npy (gid order, tokens 0..T-1) and nested_<set>.json.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from levels import LEVEL_SETS, load_tokens  # noqa: E402

RD = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111")
OUT = RD / "hierarchy/levels"


def containment(fine, coarse):
    ok = (fine >= 0) & (coarse >= 0)
    f, c = fine[ok], coarse[ok]
    C = int(c.max()) + 1
    fc, cnt = np.unique(f * C + c, return_counts=True)
    ff = fc // C
    n_coarse_per_fine = np.bincount(ff)                       # distinct coarse labels per fine cluster
    size = np.bincount(f)
    inside = n_coarse_per_fine[f] == 1
    top = np.zeros(int(f.max()) + 1, np.int64)
    np.maximum.at(top, ff, cnt)
    multi = size >= 2
    return dict(genes=int(ok.sum()), genes_in_contained_fine_cluster=float(inside.mean()),
                fine_clusters_ge2=int(multi.sum()),
                fine_clusters_ge2_split=float((n_coarse_per_fine[multi] >= 2).mean()) if multi.any() else 0.0,
                gene_share_in_plurality_coarse=float(top[size > 0].sum() / size.sum()))


def nest(toks):
    """toks ordered fine -> coarse; returns nested tokens in the same order."""
    out = [None] * len(toks)
    out[-1] = toks[-1]
    for k in range(len(toks) - 2, -1, -1):
        a, b = toks[k], out[k + 1]
        ok = (a >= 0) & (b >= 0)
        t = np.full(len(a), -1, np.int64)
        _, t[ok] = np.unique(a[ok] * (int(b.max()) + 1) + b[ok], return_inverse=True)
        out[k] = t
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name in sys.argv[1:]:
        lv = LEVEL_SETS[name]
        toks = [load_tokens(x) for x in lv]
        info = dict(set=name, levels=lv, tokens=[int(len(np.unique(t[t >= 0]))) for t in toks], pairs=[])
        for k in range(len(lv) - 1):
            d = containment(toks[k], toks[k + 1])
            d.update(fine=lv[k], coarse=lv[k + 1])
            info["pairs"].append(d)
            print(name, json.dumps(d), flush=True)
        nt = nest(toks)
        info["nested_tokens"] = [int(len(np.unique(t[t >= 0]))) for t in nt]
        info["nested_check"] = [containment(nt[k], nt[k + 1])["genes_in_contained_fine_cluster"] for k in range(len(lv) - 1)]
        for x, t in zip(lv, nt):
            np.save(OUT / f"nested__{name}__{x}.npy", t.astype(np.int64))
        (OUT / f"nested_{name}.json").write_text(json.dumps(info, indent=1) + "\n")
        print(name, "tokens", info["tokens"], "nested", info["nested_tokens"], "check", info["nested_check"], flush=True)


if __name__ == "__main__":
    main()
