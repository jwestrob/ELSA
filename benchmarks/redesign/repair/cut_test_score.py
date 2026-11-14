#!/usr/bin/env python3
"""Rescue error, part 2: score the cut test under the rescue rule and its stricter variants.

usage: cut_test_score.py <repair dir> <subgenic dir>
Reads cut/frag_meta.parquet, cut/hits.parquet (q = fragment id, t = uid), genes.parquet, subgenic uniq.parquet.
Variants (predeclared amendment_1): base (rescue rule), a (fragment is the shorter protein), c (a + anchored on the
kept side); b = a and d = c in this test (every fragment is a simulated edge truncation). Exploratory, reported
apart: c_either (a + anchored on either side, side unknown).
Writes cut/outcomes.parquet, cut/rates.tsv, cut/score.json.
"""
import json, sys
import numpy as np, pandas as pd

R, SG = sys.argv[1:3]
M = pd.read_parquet(f"{R}/cut/frag_meta.parquet").set_index("fid")
G = pd.read_parquet(f"{R}/genes.parquet", columns=["gid", "uid", "token", "tsize"])
U = pd.read_parquet(f"{SG}/uniq.parquet", columns=["uid", "n_genes"]).sort_values("uid")
n_genes = U.n_genes.values
gt = G[["uid", "token"]].groupby(["uid", "token"]).size().reset_index(name="n")
pl = gt.sort_values(["uid", "n", "token"], ascending=[True, False, True]).drop_duplicates("uid")
tok_u = np.full(len(U), -1, np.int64); tok_u[pl.uid.values] = pl.token.values
_v, _c = np.unique(G.token.values, return_counts=True); size_of = dict(zip(_v.tolist(), _c.tolist()))

H = pd.read_parquet(f"{R}/cut/hits.parquet")
H = H.rename(columns={"q": "fid"})
H = H[H.evalue <= 1e-5].copy()
for c in ("qstart", "qend", "qlen", "tstart", "tend", "tlen"):
    H[c] = H[c].astype(np.int64)
H["t"] = H.t.astype(np.int64)
H = H.join(M[["uid", "gid", "token", "side_kept"]].rename(columns={"uid": "uid0", "gid": "gid0", "token": "tok0"}), on="fid")
# remove the original gene from the targets
own = H.t.values == H.uid0.values
drop = own & (n_genes[H.t.values] == 1)
H = H[~drop].copy(); own = H.t.values == H.uid0.values
H["tok_t"] = tok_u[H.t.values]
if own.any():  # sequence carried by other genes too: token of the remaining genes
    Gs = G[G.uid.isin(set(H.t.values[own].tolist()))]
    col = H.columns.get_loc("tok_t")
    for i in np.flatnonzero(own):
        u, g0 = H.t.values[i], H.gid0.values[i]
        rem = Gs[(Gs.uid.values == u) & (Gs.gid.values != g0)].token.value_counts()
        H.iat[i, col] = int(sorted(zip(-rem.values, rem.index))[0][1])
H["size_t"] = np.array([size_of[t] for t in H.tok_t.values]) - (H.tok_t.values == H.tok0.values)
H["cov_short"] = np.where(H.qlen <= H.tlen, (H.qend - H.qstart + 1) / H.qlen, (H.tend - H.tstart + 1) / H.tlen)
H = H[(H.cov_short >= 0.8) & (H.size_t >= 2)].copy()
a_ok = (H.qlen <= H.tlen).values
anc_n = ((H.qstart <= 11) & (H.tstart <= 11)).values
anc_c = (((H.qlen - H.qend) <= 10) & ((H.tlen - H.tend) <= 10)).values
kept_n = (H.side_kept == "N").values
masks = {"base": np.ones(len(H), bool), "a": a_ok, "c": a_ok & np.where(kept_n, anc_n, anc_c), "c_either": a_ok & (anc_n | anc_c)}

def lbin(x, edges, labels):
    return pd.cut(x, edges, right=False, labels=labels).astype(str)

M["frag_len_bin"] = lbin(M.klen, [0, 75, 150, 300, 10 ** 9], ["<75", "75-149", "150-299", ">=300"])
M["kept_bin"] = lbin(M.kept_frac, [0, 0.35, 0.55, 0.75, 1.01], ["0.15-0.35", "0.35-0.55", "0.55-0.75", "0.75-0.95"])
out, rates = M.copy(), []
for v, m in masks.items():
    h = H[m].sort_values(["fid", "bits", "evalue", "size_t", "t"], ascending=[True, False, True, False, True]).drop_duplicates("fid")
    best = h.set_index("fid")
    tok = best.tok_t.reindex(M.index)
    o = np.where(tok.isna(), "unplaced", np.where(tok.values == M.token.values, "true_family", "wrong_family"))
    o = np.where((M.kind == "neg").values & (o != "unplaced"), "false_join", o)
    out[f"outcome_{v}"] = o
    out[f"fident_{v}"] = best.fident.reindex(M.index).values
    out[f"t_{v}"] = best.t.reindex(M.index).values
    out[f"tok_{v}"] = tok.values
    for dim in ("all", "frag_len_bin", "kept_bin", "fam_stratum", "len_stratum"):
        grp = pd.Series("all", index=M.index) if dim == "all" else M[dim]
        for kind in ("pos", "neg"):
            k = (M.kind == kind).values
            for b, idx in pd.Series(np.arange(len(M))[k], index=grp.values[k]).groupby(level=0):
                oo = o[idx.values]; n = len(oo)
                rates.append({"variant": v, "kind": kind, "dimension": dim, "bin": b, "n": n,
                              "true_family": float((oo == "true_family").mean()), "wrong_family": float((oo == "wrong_family").mean()),
                              "false_join": float((oo == "false_join").mean()), "unplaced": float((oo == "unplaced").mean())})
T = pd.DataFrame(rates)
T.to_csv(f"{R}/cut/rates.tsv", sep="\t", index=False, float_format="%.4f")
out.to_parquet(f"{R}/cut/outcomes.parquet")
summ = {v: {"positives": T[(T.variant == v) & (T.kind == "pos") & (T.dimension == "all")].iloc[0][["n", "true_family", "wrong_family", "unplaced"]].to_dict(),
            "negatives": T[(T.variant == v) & (T.kind == "neg") & (T.dimension == "all")].iloc[0][["n", "false_join", "unplaced"]].to_dict()}
        for v in masks}
summ["original_hits_dropped"] = int(drop.sum()); summ["original_sequence_kept_other_genes"] = int(own.sum())
json.dump(summ, open(f"{R}/cut/score.json", "w"), indent=1, default=float)
print(json.dumps(summ, indent=1, default=float))
pd.set_option("display.width", 200)
print(T[T.dimension.isin(["frag_len_bin", "kept_bin"])].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
