#!/usr/bin/env python
"""lut_loci n06 (biotite login node): neighborhoods of the lanthanide-dependent methanol dehydrogenase genes (K23995, KOfam
hits of the slice_lut workstream mapped by n04) in the lut collection.

Per K23995 gene: the +-8 genes on its contig, oriented so the K23995 gene reads on +; per neighbor: oriented offset and strand,
rescue3 token, Pfam (annot_gpu, GA), Z bonds to the K23995 gene (frozen embedding-list arm), slot call, and whether the
neighbor lies in the smallest Z L4b+ locus holding the K23995 gene and at least one neighbor.
Token summary: per (token, offset, strand) the number of K23995 genes and genera; per token the share without Pfam.
Writes data/xoxf_window.parquet and data/xoxf_tokens.tsv.
"""
import numpy as np
import pandas as pd

RD = "/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111"
PK = RD + "/slice_lut/testset/package_k8"
FZ = RD + "/long_loci/frozen"
FL = 8
gn = pd.read_parquet(PK + "/universe/genes.parquet",
                     columns=["gid", "genome_code", "contig_code", "position_index", "strand", "length"])
gm = pd.read_parquet(PK + "/universe/genomes.parquet")
cid = gn.contig_code.to_numpy()
strand = gn.strand.to_numpy()
tok = pd.read_parquet(FZ + "/tok/mmseqs__seq__e3c50s75_rescue3__lut.parquet").token.to_numpy()
mdh = pd.read_parquet("data/mdh_pkg.parquet")
X = mdh[mdh.ko == "K23995"].gid.unique()
n = len(gn)
rows = []
for x in X:
    o = int(strand[x])
    for d in range(-FL, FL + 1):
        g = x + d
        if g < 0 or g >= n or cid[g] != cid[x]:
            continue
        rows.append((x, g, d * o, int(strand[g] * o)))
R = pd.DataFrame(rows, columns=["xoxf", "gid", "off", "ostrand"])
R["token"] = tok[R.gid]
R["length"] = gn.length.to_numpy()[R.gid]
R["genome_code"] = gn.genome_code.to_numpy()[R.gid]
tax = gm.set_index("genome_code")
for c in ["genus", "family", "lut_status"]:
    R[c] = tax.loc[R.genome_code, c].to_numpy()
pf = pd.read_parquet(RD + "/annot_gpu/lut/PFAM.parquet", columns=["gid", "name"])
pf = pf[pf.gid.isin(set(R.gid))].groupby("gid").name.apply(lambda s: ";".join(s))
R["pfam_names"] = R.gid.map(pf)
z = np.load(FZ + "/nodes/lut_Z/bonds_obs.npz")
k, dd, lam = z["k"], z["d"], z["lam"]
bond = {}
xs = set(X.tolist())
for a, b, l in zip(k, k + dd, lam):
    if a in xs:
        bond[(a, b)] = l
    if b in xs:
        bond[(b, a)] = l
R["lam_to_xoxf"] = [bond.get((x, g), 0.0) for x, g in zip(R.xoxf, R.gid)]
ps = pd.read_parquet(RD + "/dpann_leads/pass_port/out/lut/lut_slot_rescue_pass.parquet",
                     columns=["gid", "proposed_label_name", "support"])
R = R.merge(ps, on="gid", how="left")
R.to_parquet("data/xoxf_window.parquet")
nb = R[R.off != 0]
s = nb.groupby(["token", "off", "ostrand"]).agg(xoxf=("xoxf", "nunique"), genera=("genus", "nunique")).reset_index()
s = s.sort_values("xoxf", ascending=False)
t = nb.groupby("token").agg(genes=("gid", "nunique"), xoxf=("xoxf", "nunique"), genera=("genus", "nunique"),
                            nopfam=("pfam_names", lambda v: float(v.isna().mean())),
                            top_pfam=("pfam_names", lambda v: v.dropna().value_counts().index[0] if v.notna().any() else ""),
                            med_len=("length", "median"), bonded=("lam_to_xoxf", lambda v: float((v > 0).mean())))
best = s.drop_duplicates("token").set_index("token")
t = t.join(best[["off", "ostrand", "xoxf"]].rename(columns={"xoxf": "xoxf_at_modal"}))
t = t.sort_values("xoxf", ascending=False)
t.to_csv("data/xoxf_tokens.tsv", sep="\t")
print("K23995 genes", len(X), "genomes", R[R.off == 0].genome_code.nunique())
print(t.head(40).to_string())
