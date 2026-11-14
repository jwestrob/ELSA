#!/usr/bin/env python
"""Post hoc (after the strand split showed coherence confined to same-strand neighbours): is coherence a provenance
signal or a transcription-unit signal? Adjacent pairs of ribosomal-protein marker genes (native, highly expressed,
co-transcribed) are compared with adjacent curated defence pairs and with strand-matched adjacent pairs.
If coherence were provenance only, ribosomal pairs would sit at the background. Single thread; argument: tag."""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RD, PKG, W, log, load_genes, boot_weights, ci  # noqa: E402

G = 3564


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else ""
    g = load_genes()
    gc = g.genome_code.to_numpy(np.int64)
    cc = g.contig_code.to_numpy(np.int64)
    st = g.strand.to_numpy(np.int64)
    n = len(g)
    elig = pd.read_parquet(W / "gene_stats.parquet", columns=["eligible"]).eligible.to_numpy(bool)
    ge = np.flatnonzero(elig)
    row = np.full(n, -1, np.int64)
    row[ge] = np.arange(len(ge))
    V = np.load(W / f"s2_v_famjac_f16{tag}.npy", mmap_mode="r")
    import scipy.sparse as sp
    Sg = np.asarray(sp.csr_matrix((np.ones(len(ge)), (gc[ge], np.arange(len(ge)))), shape=(G, len(ge))) @ np.asarray(V, np.float32), np.float64)
    ng = np.bincount(gc[ge], minlength=G).astype(float)

    def coh(a, b):
        va, vb = np.asarray(V[row[a]], np.float64), np.asarray(V[row[b]], np.float64)
        ua = va - (Sg[gc[a]] - va) / np.maximum(ng[gc[a]] - 1, 1)[:, None]
        ub = vb - (Sg[gc[b]] - vb) / np.maximum(ng[gc[b]] - 1, 1)[:, None]
        return (ua * ub).sum(1) / np.maximum(np.linalg.norm(ua, axis=1) * np.linalg.norm(ub, axis=1), 1e-12)
    i = np.arange(n - 1)
    ok = (cc[i] == cc[i + 1]) & elig[i] & elig[i + 1] & (gc[i] % 2 == 0)
    i = i[ok]
    same = st[i] == st[i + 1]
    rib = np.zeros(n, bool)
    rib[pd.read_parquet(RD / "ribosomal/gold/marked.parquet", columns=["gid"]).gid.to_numpy()] = True
    mem = pd.read_parquet(PKG / "gold/members.parquet")
    df = mem[mem.caller == "DefenseFinder"]
    pi = df.merge(df, on="instance_id")
    pi = pi[pi.gid_y == pi.gid_x + 1].gid_x.to_numpy()
    rng = np.random.default_rng(17)
    bg = np.sort(rng.choice(np.flatnonzero(same), 500_000, replace=False))
    Wb = boot_weights(G)
    out = {}
    cb = coh(i[bg], i[bg] + 1)
    out["background_same_strand"] = float(cb.mean())
    for nm, mask in (("ribosomal_both", rib[i] & rib[i + 1] & same), ("ribosomal_one", (rib[i] ^ rib[i + 1]) & same),
                     ("defense_members", np.isin(i, pi) & same)):
        a = i[mask]
        c = coh(a, a + 1)
        s_, c_ = np.bincount(gc[a], weights=c, minlength=G), np.bincount(gc[a], minlength=G).astype(float)
        out[nm] = dict(pairs=int(len(a)), mean_c=float(c.mean()), ci=ci((Wb @ s_) / (Wb @ c_)))
    json.dump(out, open(W / f"s2_ribo_coh{tag}.json", "w"), indent=1)
    log(tag, json.dumps(out))


if __name__ == "__main__":
    main()
