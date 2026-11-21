#!/usr/bin/env python
"""galt step 17: define the divergent subfamily phylogenetically and redo the question-4 counts on it.
Subfamily D = leaves of the smallest clade holding the 64 (tree rooted on a GDP-hexose phosphorylase reference, as in g14), with every cd-hit 50% cluster
member of a named representative in that clade added; form N = every other DPANN family member (census groups
named_unlabeled + labeled genes outside the clade). Per order: genomes, genomes with D, with N, with both; GT3-GS
co-occurrence (Fisher per order, Mantel-Haenszel pooled); within Woesearchaeales: glgC (K00975) and GalU-type UGPase
(CDD cd02541 / COG1210 / TIGR01099 / PRK10122) in D genomes versus N-only genomes; D-gene adjacency to a GT3 GS.
Writes data/q4_subfamily_members.tsv, data/q4_subfamily_by_order.tsv, data/q4_subfamily_tests.tsv."""
import re
from pathlib import Path
import duckdb
import numpy as np
import pandas as pd
from ete3 import Tree
from scipy.stats import fisher_exact
from statsmodels.stats.contingency_tables import StratifiedTable

H = Path("/Users/jacob/Documents/Sandbox/elsa_test/ELSA/benchmarks/redesign/dpann_leads/galt")
T = Tree(str(H / "aln/galt_tree_treefile.txt"), format=1)
X = pd.read_csv(H / "data/tree_taxa.tsv", sep="\t").set_index("taxon")
C = pd.read_csv(H / "data/census_proteins.tsv", sep="\t")
GF = pd.read_csv(H / "data/genome_features.tsv", sep="\t")
PF = pd.read_parquet(H / "data/protein_features.parquet")
lv = T.get_leaf_names()
T.set_outgroup(T & "REF_GDPhexose_phosphorylase_GDPGP1_human")   # leaf membership only; supports read in g14
s64 = [t for t in lv if X.loc[t, "set64"]]
clade = T.get_common_ancestor(s64)
cl_leaves = set(clade.get_leaf_names())
members = {}
for l in open(H / "aln/named_c50_clstr.txt"):
    if l.startswith(">Cluster"):
        cur = []
        members.setdefault("_", []).append(cur)
    else:
        pid = re.search(r">(.+?)\.\.\.", l).group(1)
        cur.append((pid, l.rstrip().endswith("*")))
rep2mem = {}
for cl in members["_"]:
    rep = [p for p, s in cl if s][0]
    rep2mem[rep] = [p for p, _ in cl]
D = set()
for t in cl_leaves:
    if t.startswith("REF_"):
        continue
    D |= set(rep2mem.get(t, [t]))
C["subfamily"] = np.where(C.protein_id.isin(D), "D", "N")
C.loc[C.group == "ap4a_branch", "subfamily"] = "ap4a"
C[["protein_id", "bin_id", "order", "group", "subfamily", "length"]].to_csv(H / "data/q4_subfamily_members.tsv", sep="\t", index=False)
print("clade leaves", len(cl_leaves), "non-ref", sum(not t.startswith("REF_") for t in cl_leaves), "(edge support: see g14 / q1_phylogeny.tsv)",
      "| D genes", len(D), "| refs inside:", [t for t in cl_leaves if t.startswith("REF_")])
print(C.groupby(["subfamily", "group"]).size().to_string())
Dg = set(C.bin_id[C.subfamily == "D"]); Ng = set(C.bin_id[C.subfamily == "N"])
GF["D"] = GF.bin_id.isin(Dg); GF["N"] = GF.bin_id.isin(Ng); GF["GT3"] = GF.GT3_GS > 0
c = duckdb.connect()
c.execute("ATTACH '/Users/jacob/Documents/Sandbox/Sharur/Sharur/data/dpann/sharur.duckdb' AS d (READ_ONLY)")
c.execute("SET threads=2")
galu = set(c.execute("""SELECT DISTINCT p.bin_id FROM d.protein_resemblance r JOIN d.proteins p USING(protein_id)
   WHERE r.accession IN ('cd02541','COG1210','TIGR01099','PRK10122')""").fetchdf().bin_id)
GF["GalU"] = GF.bin_id.isin(galu); GF["glgC_ko"] = GF.glgC > 0
byo = GF.groupby("order").agg(genomes=("bin_id", "size"), completeness_median=("completeness", "median"), D=("D", "sum"),
                              N=("N", "sum"), both=("D", lambda s: int((s & GF.loc[s.index, "N"]).sum())), GT3=("GT3", "sum")).reset_index()
byo = byo[(byo.D > 0) | (byo.N > 0)].sort_values(["D", "N"], ascending=False)
byo.to_csv(H / "data/q4_subfamily_by_order.tsv", sep="\t", index=False)
print(byo.head(15).to_string(index=False))
tests, tabs = [], []
for o, g in GF.groupby("order"):
    if g.D.sum() == 0:
        continue
    a = int((g.GT3 & g.D).sum()); b = int((g.GT3 & ~g.D).sum()); c_ = int((~g.GT3 & g.D).sum()); d = int((~g.GT3 & ~g.D).sum())
    orr, p = fisher_exact([[a, b], [c_, d]], alternative="greater")
    tests.append(dict(test=f"GT3 x D, {o}", a=f"{a}/{a + b} GT3 genomes carry D", b=f"{c_}/{c_ + d} others", OR=round(orr, 2), p=p))
    if min(a + b, c_ + d) > 0:
        tabs.append(np.array([[a, b], [c_, d]]))
st = StratifiedTable(tabs)
tests.append(dict(test="GT3 x D, Mantel-Haenszel over orders", a="", b="", OR=round(st.oddsratio_pooled, 2), p=st.test_null_odds().pvalue))
W = GF[GF.order == "Woesearchaeales"]
for f in ("glgC_ko", "GalU", "GT3"):
    a = int((W.D & W[f]).sum()); n1 = int(W.D.sum()); cc = int((~W.D & W.N & W[f]).sum()); n2 = int((~W.D & W.N).sum())
    orr, p = fisher_exact([[a, n1 - a], [cc, n2 - cc]])
    tests.append(dict(test=f"Woesearchaeales {f}: D genomes vs N-only genomes", a=f"{a}/{n1}", b=f"{cc}/{n2}", OR=round(orr, 2), p=p))
tests.append(dict(test="genomes with D that also carry N", a=f"{len(Dg & Ng)}/{len(Dg)}", b="", OR="", p=""))
TT = pd.DataFrame(tests)
TT.to_csv(H / "data/q4_subfamily_tests.tsv", sep="\t", index=False)
print(TT.to_string(index=False))
