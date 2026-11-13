#!/usr/bin/env python
"""ws_showcase2 context, part 1 (biotite login node, seconds): for the named candidates, every realisation's member
genes and its +-3 flanking genes on the same contig (offset relative to the block, strand relative to the block's
first member), plus up to 3 control windows per realisation (same contig, same width, >= 6 genes from the block).
Writes data/context_genes.tsv (no sequences). Usage: sc2_context_genes.py CAND [CAND ...]"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RD = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111")
S = RD / "showcase2"
W = 3


def main(cands):
    g = pd.read_parquet(RD / "ws_dpann/package/universe/genes.parquet",
                        columns=["gid", "protein_id", "genome_code", "contig_code", "position_index", "strand", "length"])
    tok = pd.read_parquet(RD / "engine/dpann_tokens/mmseqs__seq__e3c50s75_rescue3__dpann.parquet").sort_values("gid").token.to_numpy()
    cc = g.contig_code.to_numpy()
    st = g.strand.to_numpy()
    n = len(g)
    b = g.groupby("contig_code").gid.agg(["min", "max"])
    cfirst = pd.Series(b["min"].to_numpy(), index=b.index)
    clast = pd.Series(b["max"].to_numpy(), index=b.index)
    R = pd.read_parquet(S / "data/realisations.parquet", columns=["cand", "gids", "genome_code", "group", "phylum"])
    R = R[R.cand.isin(cands)].reset_index(drop=True)
    rng = np.random.default_rng(17)
    rows = []
    for ri, r in R.iterrows():
        gl = [int(x) for x in r.gids.split(";")]
        lo, hi = min(gl), max(gl)
        s0 = st[gl[0]]
        for x in range(lo - W, hi + W + 1):
            if x < 0 or x >= n or cc[x] != cc[gl[0]]:
                continue
            off = 0 if lo <= x <= hi else (x - lo if x < lo else x - hi)
            rows.append(dict(cand=r.cand, real=ri, kind="block", gid=x, member=x in gl,
                             member_pos=(gl.index(x) if x in gl else -1), offset=int(off), same_strand=bool(st[x] == s0)))
        # controls: same contig, same width, >= 6 genes away from the block
        span = hi - lo + 1
        first, last = int(cfirst[cc[gl[0]]]), int(clast[cc[gl[0]]])
        starts = [s for s in range(first + W, last - W - span + 2) if s + span - 1 < lo - 6 or s > hi + 6]
        for s in (rng.choice(starts, size=min(3, len(starts)), replace=False) if starts else []):
            for x in range(s - W, s + span + W):
                off = 0 if s <= x < s + span else (x - s if x < s else x - (s + span - 1))
                rows.append(dict(cand=r.cand, real=ri, kind="control", gid=int(x), member=False, member_pos=-1,
                                 offset=int(off), same_strand=bool(st[x] == st[s])))
    out = pd.DataFrame(rows)
    out["protein_id"] = g.protein_id.to_numpy()[out.gid]
    out["length"] = g.length.to_numpy()[out.gid]
    out["token"] = tok[out.gid]
    out = out.merge(R[["cand", "genome_code", "group", "phylum"]].reset_index().rename(columns={"index": "real"}),
                    on=["cand", "real"])
    out.to_csv(S / "data/context_genes.tsv", sep="\t", index=False)
    print(out.groupby(["cand", "kind"]).size().to_string())


if __name__ == "__main__":
    main(sys.argv[1:])
