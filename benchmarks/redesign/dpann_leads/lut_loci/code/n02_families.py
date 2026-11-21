#!/usr/bin/env python
"""lut_loci n02 (Mac): embedding-list families of the lut neighborhood genes and their positional summary.

Families: connected components of the graph on window genes (n01) with an edge a-b when a and b list each other (mutual
Z-list partners, z >= 10.6, best gene per genome, own genome group excluded) or share a rescue3 token (MMseqs2 family; this
joins genes of one genome group, which never list each other). One-directional Z edges chain unrelated families (lutB,
lutD and lutI fell into one component) and are not used. Each gold component falls into exactly one family.
Per family: window genes, instances, genomes, genera, families, gold components carried, Pfam share, DUF-only share,
modal (nearest component, oriented offset, oriented strand) slot and its share, region (inside / up / down) shares,
share of genes bonded (Z bond) to an instance member, slot call labels (pass_port) and median support, tier counts.
Writes data/window_families.parquet (gene -> family) and data/families_summary.tsv.
"""
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

D = "/Users/jacob/Documents/Sandbox/elsa_test/ELSA/benchmarks/redesign/dpann_leads/lut_loci/data"
W = pd.read_parquet(D + "/window_genes.parquet")
Z = pd.read_parquet(D + "/window_zlists.parquet")
wg = np.sort(W.gid.unique())
idx = pd.Series(np.arange(len(wg)), index=wg)
rows, cols = [], []
for g, p in zip(Z.gid.to_numpy(), Z.partners.to_numpy()):
    p = np.asarray(p)
    p = p[np.isin(p, wg)]
    rows += [idx[g]] * len(p)
    cols += idx[p].tolist()
A = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(wg), len(wg))).tocsr()
M = (A.multiply(A.T) > 0).astype(np.int64)
Gu = W.drop_duplicates("gid").set_index("gid").loc[wg]
tok = pd.factorize(Gu.token)[0]
T = coo_matrix((np.ones(len(wg)), (np.arange(len(wg)), tok))).tocsr()
U = ((M + (T @ T.T > 0).astype(np.int64)) > 0).astype(np.int64)
nc, lab = connected_components(U, directed=False)
fam = pd.DataFrame({"gid": wg, "fam": lab})
W = W.merge(fam, on="gid")
# family id ordered by size
sz = W.drop_duplicates("gid").fam.value_counts()
ren = {f: i for i, f in enumerate(sz.index)}
W["fam"] = W.fam.map(ren)
W[["instance_id", "gid", "fam"]].to_parquet(D + "/window_families.parquet")


def modal(x):
    v = x.value_counts()
    return v.index[0], v.iloc[0] / len(x)


out = []
for f, g in W.groupby("fam"):
    gu = g.drop_duplicates("gid")
    slot = g.nearest_comp + ":" + g.nearest_off.astype(str) + ":" + np.where(g.ostrand > 0, "+", "-")
    ms, msh = modal(slot)
    comp = g.component[g.component != ""]
    labs = gu.proposed_label_name.dropna()
    out.append(dict(
        fam=f, genes=len(gu), instances=g.instance_id.nunique(), genomes=gu.genome_code.nunique(),
        genera=gu.genus.nunique(), tax_families=gu.family.nunique(), classes=gu["class"].nunique(),
        components=";".join(f"{k}:{v}" for k, v in comp.value_counts().items()),
        non_gold_in_comp_family=(int((gu.component == "").sum()) if len(comp) else 0),
        pfam_share=round(gu.has_pfam.mean(), 3), duf_only_share=round(gu.duf_only.mean(), 3),
        top_pfam=(gu.pfam_names.dropna().value_counts().index[0] if gu.pfam_names.notna().any() else ""),
        top_pfam_n=(int(gu.pfam_names.dropna().value_counts().iloc[0]) if gu.pfam_names.notna().any() else 0),
        modal_slot=ms, modal_slot_share=round(msh, 3),
        inside=round((g.region == "inside").mean(), 3), up=round((g.region == "up").mean(), 3),
        down=round((g.region == "down").mean(), 3),
        bonded_to_member=round((g.bonds_to_members > 0).mean(), 3),
        in_best_locus=round(g.in_best_locus.mean(), 3),
        median_len=int(gu.length.median()),
        labeled=len(labs), top_label=(labs.value_counts().index[0] if len(labs) else ""),
        top_label_n=(int(labs.value_counts().iloc[0]) if len(labs) else 0),
        support_median=(round(gu.support.dropna().median(), 3) if len(labs) else np.nan),
        tiers=";".join(f"{k}:{v}" for k, v in gu.tier.dropna().value_counts().items()),
        top_genera=";".join(f"{k}:{v}" for k, v in gu.genus.value_counts().head(4).items()),
    ))
S = pd.DataFrame(out).sort_values("instances", ascending=False)
S.to_csv(D + "/families_summary.tsv", sep="\t", index=False)
print("families", nc, "window genes", len(wg))
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
pd.set_option("display.max_colwidth", 40)
print(S.head(45)[["fam", "genes", "instances", "genomes", "genera", "components", "pfam_share", "top_pfam", "modal_slot",
                  "modal_slot_share", "inside", "bonded_to_member", "median_len", "labeled", "top_label", "support_median"]]
      .to_string(index=False))
