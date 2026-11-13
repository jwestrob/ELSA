#!/usr/bin/env python
"""Input adapters for the synton pipeline (Herve, Cury & Bernheim 2026; repository commit 84baf14).

Their pipeline reads two tables (README "Inputs"):
  data/genes.parquet     gene, replicon, position, orientation ('D' direct / 'C' reverse)
  data/clusters.parquet  gene, representative_50, representative_95
These adapters write them from an ELSA universe and from either an MMseqs2 linclust cascade (faithful families) or a
token table (matched-identity variant). Nothing of their logic is touched.

Identifier choices (stated as adapter decisions in REPORT.txt):
  gene      'g' + gid zero-padded to 9 digits: a string (their diversity.py writes genes as pa.string()) whose
            lexicographic order equals gid order. Their context representative is min(gene) of a context cluster;
            with gembase ids that is the alphabetically first genome code; here it is the lowest gid.
  replicon  'c' + contig_code zero-padded to 7 digits (one contig = one replicon).
  position  position_index (contiguous 0..n-1 per contig).
  representative_50/95  the gene id of the stage representative (faithful) or 't' + token (variant).

usage
  make_inputs.py genes    --universe DIR --out data/genes.parquet
  make_inputs.py cascade  --universe DIR --tsv99 A --tsv95 B --tsv80 C --tsv50 D --out data/clusters.parquet
  make_inputs.py tokens   --universe DIR --tokens T.parquet --out data/clusters.parquet
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def gname(gid):
    return pd.Series(np.asarray(gid, dtype=np.int64)).map(lambda x: f"g{x:09d}").to_numpy(dtype=object)


def load_universe(d):
    g = pd.read_parquet(Path(d) / "genes.parquet", columns=["gid", "contig_code", "position_index", "strand"])
    assert g.gid.is_unique
    assert np.array_equal(np.sort(g.gid.to_numpy()), np.arange(len(g))), "gids must be 0..n-1"
    return g.sort_values("gid").reset_index(drop=True)


def cmd_genes(a):
    g = load_universe(a.universe)
    assert set(np.unique(g.strand)) <= {-1, 1}
    out = pd.DataFrame(dict(
        gene=gname(g.gid),
        replicon=g.contig_code.map(lambda x: f"c{x:07d}").to_numpy(dtype=object),
        position=g.position_index.to_numpy(np.int64),
        orientation=np.where(g.strand.to_numpy() > 0, "D", "C").astype(object)))
    # position must be unique within replicon
    assert not out.duplicated(["replicon", "position"]).any()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(a.out, index=False)
    log("genes", len(out), "replicons", out.replicon.nunique(), "->", a.out)


def read_tsv(p):
    t = pd.read_csv(p, sep="\t", header=None, names=["rep", "mem"], dtype=np.int64)
    assert t.mem.is_unique, p
    return t


def cmd_cascade(a):
    g = load_universe(a.universe)
    n = len(g)
    maps = {}
    for lvl in ("99", "95", "80", "50"):
        t = read_tsv(getattr(a, f"tsv{lvl}"))
        m = np.full(n, -1, dtype=np.int64)
        m[t.mem.to_numpy()] = t.rep.to_numpy()
        maps[lvl] = (m, t)
        log("stage", lvl, "members", len(t), "clusters", t.rep.nunique())
    m99 = maps["99"][0]
    assert (m99 >= 0).all(), "every gene must be a member at 99%"
    r99 = m99
    r95 = maps["95"][0][r99]
    assert (r95 >= 0).all()
    r80 = maps["80"][0][r95]
    assert (r80 >= 0).all()
    r50 = maps["50"][0][r80]
    assert (r50 >= 0).all()
    out = pd.DataFrame(dict(gene=gname(g.gid), representative_50=gname(r50), representative_95=gname(r95)))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(a.out, index=False)
    sizes = pd.Series(r50).value_counts()
    info = dict(genes=n, fam99=int(len(np.unique(r99))), fam95=int(len(np.unique(r95))), fam80=int(len(np.unique(r80))),
                fam50=int(len(sizes)), fam50_singletons=int((sizes == 1).sum()),
                fam50_size_q50_q90_q99_max=[float(x) for x in np.quantile(sizes, [.5, .9, .99, 1])],
                genes_in_fam50_ge2=int(sizes[sizes >= 2].sum()))
    json.dump(info, open(Path(a.out).with_suffix(".json"), "w"), indent=1)
    log("clusters", info)


def cmd_tokens(a):
    g = load_universe(a.universe)
    t = pd.read_parquet(a.tokens, columns=["gid", "token"])
    assert len(t) == len(g) and t.gid.is_unique
    tok = np.full(len(g), -1, dtype=np.int64)
    tok[t.gid.to_numpy()] = t.token.to_numpy()
    assert (tok >= 0).all(), "every gene needs a token"
    lab = pd.Series(tok).map(lambda x: f"t{x:09d}").to_numpy(dtype=object)
    out = pd.DataFrame(dict(gene=gname(g.gid), representative_50=lab, representative_95=lab))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(a.out, index=False)
    sizes = pd.Series(tok).value_counts()
    info = dict(genes=len(g), tokens=int(len(sizes)), singletons=int((sizes == 1).sum()), source=str(a.tokens),
                size_q50_q90_q99_max=[float(x) for x in np.quantile(sizes, [.5, .9, .99, 1])])
    json.dump(info, open(Path(a.out).with_suffix(".json"), "w"), indent=1)
    log("clusters (tokens)", info)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["genes", "cascade", "tokens"])
    ap.add_argument("--universe", required=True)
    ap.add_argument("--out", required=True)
    for lvl in ("99", "95", "80", "50"):
        ap.add_argument(f"--tsv{lvl}")
    ap.add_argument("--tokens")
    a = ap.parse_args()
    {"genes": cmd_genes, "cascade": cmd_cascade, "tokens": cmd_tokens}[a.cmd](a)


if __name__ == "__main__":
    main()
