"""Exploration (read-only): composition of the locus-space uncalled dense clusters (ws_showcase2)."""
import numpy as np, pandas as pd
RD = "/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111"
PKG = f"{RD}/ws_dpann/package"
C = pd.read_parquet(f"{RD}/locus_space/out/map/cores_clusters_MEAN.parquet", columns=["genome","gid_s","gid_e","span","cluster","module_id"])
G = pd.read_parquet(f"{RD}/locus_space/out/map/cluster_table_MEAN.parquet")
dense = G[(G.cores >= 10) & (~G.called)].copy()
print("uncalled dense", len(dense), "dark>=.5", (dense.dark>=.5).sum(), "dark>=.5 & groups>=5", ((dense.dark>=.5)&(dense.groups>=5)).sum())
tok = pd.read_parquet(f"{RD}/engine/dpann_tokens/mmseqs__seq__e3c50s75_rescue3__dpann.parquet").sort_values("gid").token.to_numpy()
fl = pd.read_parquet(f"{PKG}/annotation_flags.parquet", columns=["gid","unannotated"]).sort_values("gid").unannotated.to_numpy()
g = pd.read_parquet(f"{PKG}/universe/genes.parquet", columns=["gid","length","strand","contig_code"])
L = g.length.to_numpy()
Cd = C[C.cluster.isin(dense.cluster)].copy()
print("cores in uncalled dense", len(Cd), Cd.span.value_counts().sort_index().head(12).to_dict())
Cd["toks"] = [tuple(tok[s:e+1]) for s, e in zip(Cd.gid_s, Cd.gid_e)]
Cd["ok_len"] = [bool((L[s:e+1] <= 300).all()) for s, e in zip(Cd.gid_s, Cd.gid_e)]
Cd["sumL"] = [int(L[s:e+1].sum()) for s, e in zip(Cd.gid_s, Cd.gid_e)]
Cd["dark"] = [float(fl[s:e+1].mean()) for s, e in zip(Cd.gid_s, Cd.gid_e)]
Cd["anydark"] = [bool(fl[s:e+1].any()) for s, e in zip(Cd.gid_s, Cd.gid_e)]
x = Cd[(Cd.span.between(2,5))]
print("span2-5", len(x), "all<=300", x.ok_len.sum(), "sum<=450", (x.sumL<=450).sum(), "both & anydark", (x.ok_len & (x.sumL<=450) & x.anydark).sum())
y = x[x.ok_len & (x.sumL <= 600) & x.anydark]
s = y.groupby("cluster").agg(n=("span","size"), famsets=("toks", "nunique"), span_mode=("span", lambda v: v.mode().iloc[0]), dark=("dark","mean"), sumL=("sumL","median"))
s = s.join(dense.set_index("cluster")[["cores","groups","phyla","dark"]], rsuffix="_cl")
print("clusters with eligible cores", len(s))
print(s.sort_values("n", ascending=False).head(60).to_string())
