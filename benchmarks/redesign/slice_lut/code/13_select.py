#!/usr/bin/env python3
"""Deterministic selection of the lut test set (rules: testset/selection_rules.txt, v2 amendment; seed 17).

usage: 13_select.py TABLES_DIR OUT_DIR
Reads TABLES_DIR/{order,alpha_wide,beyond}_genomes.tsv and *_locus_classes_main.tsv (part-1 outputs).
Writes OUT_DIR/candidates.tsv (every candidate genome with lut status and rank), OUT_DIR/ladder.tsv (totals per k),
OUT_DIR/selection_k<k>.tsv for every k, and OUT_DIR/sizes.json (k chosen for 250 / 500 / 1,000).
"""
import hashlib, json, math, os, sys
import pandas as pd

T, OUT = sys.argv[1:3]
os.makedirs(OUT, exist_ok=True)
SEED = 17
FULL = set("ABCDEFGHI")
ABEF = set("ABEF")
FORCED_A3 = ["GCA_901457705.2", "GCF_001043915.1", "GCF_002752655.1"]
FORCED_C2 = ["RS_GCF_021283545.2", "RS_GCF_000341735.1"]

rows = []
for sc in ["order", "alpha_wide", "beyond"]:
    g = pd.read_csv(f"{T}/{sc}_genomes.tsv", sep="\t")
    c = pd.read_csv(f"{T}/{sc}_locus_classes_main.tsv", sep="\t", dtype={"components_present": str})
    comps = c.groupby("genome").components_present.agg(lambda v: [set(str(x).split(",")) for x in v])
    g["loci"] = g.genome.map(comps)
    g["source_scope"] = sc
    rows.append(g)
G = pd.concat(rows, ignore_index=True)
assert G.genome.is_unique, "genome in two scopes"


def status(r):
    L = r.loci if isinstance(r.loci, list) else []
    if L:
        if any(FULL <= s for s in L):
            return "S1_complete"
        if any("H" in s and len(s & ABEF) >= 2 for s in L):
            return "S2_receptor_transporter"
        if any("H" in s for s in L):
            return "S3_receptor_module"
        if r.rb_H >= 1:
            return "S4_split"
        if any(len(s & ABEF) >= 2 for s in L):
            return "S5_transporter_only"
        return "S6_other_locus"
    return "S7_receptor_only" if r.rb_H >= 1 else "S0_negative"


G["lut_status"] = G.apply(status, axis=1)
G["n_main_loci"] = G.loci.map(lambda v: len(v) if isinstance(v, list) else 0)
G["locus_components"] = G.loci.map(lambda v: "|".join(",".join(sorted(s)) for s in v) if isinstance(v, list) else "")
G["in_class"] = G["class"] == "Alphaproteobacteria"
comp = G.completeness.fillna(0); cont = G.contamination.fillna(100)
G["qtier"] = 2 - ((comp >= 70) & (cont <= 10)).astype(int) - ((comp >= 90) & (cont <= 5)).astype(int)
G["hash"] = [hashlib.sha256(f"{SEED}:{a}".encode()).hexdigest() for a in G.gtdb_id]
G = G.sort_values(["qtier", "hash"]).reset_index(drop=True)
G["rank"] = range(len(G))
cand = G.drop(columns=["loci"])
cand.to_csv(f"{OUT}/candidates.tsv", sep="\t", index=False)


def select(k):
    """rule set v2 (selection_rules.txt, AMENDMENT v2)."""
    take = {}  # genome -> layer

    def add(genomes, layer):
        for x in genomes:
            if x not in take:
                take[x] = layer

    C = G[G.in_class]
    add(C[C.lut_status == "S1_complete"].genome, "a1_complete")
    x = C[C.lut_status.isin(["S2_receptor_transporter", "S3_receptor_module"])]
    add(x.groupby("genus", sort=False).head(k).genome, "a2_receptor")
    a12_genera = set(G[G.genome.isin(take)].genus)
    x = C[C.lut_status == "S4_split"]
    add(list(G[G.genome.isin(FORCED_A3)].genome) + list(x.groupby("family", sort=False).head(k).genome), "a3_split")
    a123_fams = set(G[G.genome.isin(take)].family)
    x = C[C.lut_status == "S5_transporter_only"]
    add(x.groupby("family", sort=False).head(k).genome, "a4_transporter_only")
    a_genera = set(G[G.genome.isin(take)].genus)
    neg = C[C.lut_status == "S0_negative"]
    add(neg[neg.genus.isin(a12_genera)].groupby("genus", sort=False).head(k).genome, "b1_same_genus_negative")
    x = neg[neg.family.isin(a123_fams) & ~neg.genus.isin(a_genera)]
    firsts = x.groupby("genus", sort=False).head(1)            # first genome per genus, in rank order
    add(firsts.groupby("family", sort=False).head(k).genome, "b2_same_family_negative")
    held = set(G[G.genome.isin(take)].family)
    orders = C.groupby("ord").size().reset_index(name="n").sort_values(["n", "ord"], ascending=[False, True]).ord
    picks = []
    for o in list(orders)[: 10 * k]:
        y = C[(C.ord == o) & ~C.family.isin(held)]
        if len(y):
            picks.append(y.genome.iloc[0])
    add(picks, "c1_class_background")
    B = G[~G.in_class & (G.n_main_loci > 0)]
    fam_order = B.groupby("family").size().reset_index(name="n").sort_values(["n", "family"], ascending=[False, True]).family
    picks = []
    for f in list(fam_order)[: 2 * k]:
        y = B[B.family == f]
        r = y[y.lut_status.isin(["S2_receptor_transporter", "S3_receptor_module"])]
        picks.append((r if len(r) else y).genome.iloc[0])
    add(list(G[G.gtdb_id.isin(FORCED_C2)].genome) + picks, "c2_outside_class_module")
    S = G[G.genome.isin(take)].copy()
    S["layer"] = S.genome.map(take)
    return S


lad = []
for k in range(1, 13):
    S = select(k)
    S.drop(columns=["loci"]).to_csv(f"{OUT}/selection_k{k}.tsv", sep="\t", index=False)
    r = dict(k=k, genomes=len(S), genes=int(S.n_genes.sum()), genera=S.genus.nunique(), families=S.family.nunique(),
             max_genus=S.genus.value_counts().index[0], max_genus_n=int(S.genus.value_counts().iloc[0]))
    for L, n in S.layer.value_counts().sort_index().items():
        r[L] = int(n)
    lad.append(r)
lad = pd.DataFrame(lad).fillna(0)
lad.to_csv(f"{OUT}/ladder.tsv", sep="\t", index=False)
sizes = {}
for t in (250, 500, 1000):
    d = (lad.genomes - t).abs()
    sizes[str(t)] = int(lad.k[d == d.min()].min())
json.dump(dict(sizes=sizes, seed=SEED, candidates=len(G), status_counts=G.lut_status.value_counts().to_dict(),
               status_counts_class=G[G.in_class].lut_status.value_counts().to_dict()), open(f"{OUT}/sizes.json", "w"), indent=1)
print(lad.to_string()); print(sizes)
