#!/usr/bin/env python
"""galt step 13 (questions 3 and 4): locus recurrence, genome routes, clade distribution, co-occurrence with the
glycogen synthase against a matched background. All features are OBSERVED database rows (g10 definitions).
Focal groups: lead15; labeled_unannotated (the other 2 of the 17 unannotated + 3 more labeled genes outside the 64);
labeled_named (56 labeled genes that a database source places in the family); named_unlabeled (1,245 named DPANN
family members, the named form).
Outputs
  data/q3_locus_recurrence.tsv   feature x group: share of focal genes with the feature within +-3 / +-5 genes
  data/q3_genome_routes.tsv      feature x genome set: genomes with the feature / genomes in the set
  data/q4_distribution.tsv       order x counts (genomes, completeness median, named form, divergent form, lead15)
  data/q4_cooccurrence.tsv       per order: GT3-GS genomes vs others x divergent / named form; Fisher and
                                 Mantel-Haenszel; adjacency (family gene within 3 genes of a GT3 GS) vs background
"""
from pathlib import Path
import duckdb
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from statsmodels.stats.contingency_tables import StratifiedTable

H = Path("/Users/jacob/Documents/Sandbox/elsa_test/ELSA/benchmarks/redesign/dpann_leads/galt")
C = pd.read_csv(H / "data/census_proteins.tsv", sep="\t")
N = pd.read_csv(H / "data/locus_neighbors.tsv", sep="\t")
PF = pd.read_parquet(H / "data/protein_features.parquet")
GF = pd.read_csv(H / "data/genome_features.tsv", sep="\t")
FEATS = ["GT3_GS", "GT5_GS", "GT4_GS_KO", "GH57", "GDE_C", "GH15", "GP", "NTP_transferase", "cdd_glgC", "ArchGlmU",
         "glgC", "UGP", "DUF5752", "PGM", "galE", "rmlA", "GMPP"]
pf = PF.set_index("protein_id")
grp = C.set_index("protein_id").group
N["group"] = N.focal.map(grp)
N = N[N.group.isin(["lead15", "labeled_unannotated", "labeled_named", "named_unlabeled"])]
rows = []
for w in (3, 5):
    nn = N[(N["off"] != 0) & (N["off"].abs() <= w)]
    for f in FEATS:
        has = set(pf.index[pf[f] > 0])
        hit = nn[nn.nb.isin(has)].groupby("group").focal.nunique()
        tot = N.groupby("group").focal.nunique()
        for g in tot.index:
            rows.append(dict(window=w, feature=f, group=g, genes=int(hit.get(g, 0)), total=int(tot[g]),
                             share=round(hit.get(g, 0) / tot[g], 3)))
R = pd.DataFrame(rows)
R.pivot_table(index=["window", "feature"], columns="group", values="share").to_csv(H / "data/q3_locus_recurrence.tsv", sep="\t")
R.to_csv(H / "data/q3_locus_recurrence_long.tsv", sep="\t", index=False)
print(R[R.window == 3].pivot_table(index="feature", columns="group", values="genes").astype(int).to_string())
print(R.groupby("group").total.first().to_string())

# genome routes
sets = {"lead15_genomes": set(C.bin_id[C.lead15]), "lab76_genomes": set(C.bin_id[C.lab76]),
        "named_genomes": set(C.bin_id[C.group == "named_unlabeled"])}
for o in ["Woesearchaeales", "QMZS01", "EX4484-52"]:
    sets[f"all_{o}"] = set(GF.bin_id[GF.order == o])
rt = []
for f in FEATS + ["galK", "galM", "USP", "glgB", "glgX", "SGA1", "malQ"]:
    for s, b in sets.items():
        sub = GF[GF.bin_id.isin(b)]
        rt.append(dict(feature=f, set=s, genomes=int((sub[f] > 0).sum()), total=len(sub)))
RT = pd.DataFrame(rt)
RT["cell"] = RT.genomes.astype(str) + "/" + RT.total.astype(str)
RT.pivot(index="feature", columns="set", values="cell").to_csv(H / "data/q3_genome_routes.tsv", sep="\t")
print(RT.pivot(index="feature", columns="set", values="cell").to_string())

# distribution by order
GF["named"] = GF.n_named > 0
GF["div"] = GF.n_lab76 > 0
GF["div_unann"] = GF.n_lab76_unann > 0
GF["GT3"] = GF.GT3_GS > 0
D = GF.groupby("order").agg(genomes=("bin_id", "size"), completeness_median=("completeness", "median"),
                            named_form=("named", "sum"), divergent_form=("div", "sum"), divergent_unannotated=("div_unann", "sum"),
                            lead15=("n_lead15", "sum"), GT3_GS=("GT3", "sum")).reset_index()
D = D[(D.named_form > 0) | (D.divergent_form > 0)].sort_values("genomes", ascending=False)
D.to_csv(H / "data/q4_distribution.tsv", sep="\t", index=False)
print(D.to_string(index=False))
both = GF[GF["div"] & GF.named]
print("genomes with the divergent form:", int(GF["div"].sum()), "; also carrying a named-form gene:", len(both))
print("genomes with an unannotated divergent gene:", int(GF.div_unann.sum()), "; also named form:", int((GF.div_unann & GF.named).sum()))

# co-occurrence with GT3 GS, per order, Mantel-Haenszel across orders with divergent-form genomes
co, tabs = [], []
for o, g in GF.groupby("order"):
    if g["div"].sum() == 0:
        continue
    for form in ("div", "named"):
        a = int((g.GT3 & g[form]).sum()); b = int((g.GT3 & ~g[form]).sum())
        c_ = int((~g.GT3 & g[form]).sum()); d = int((~g.GT3 & ~g[form]).sum())
        orr, p = fisher_exact([[a, b], [c_, d]], alternative="greater")
        co.append(dict(order=o, form=form, GT3_with=a, GT3_without=b, noGT3_with=c_, noGT3_without=d,
                       share_in_GT3=f"{a}/{a + b}", share_in_noGT3=f"{c_}/{c_ + d}", fisher_OR=round(orr, 2), fisher_p=p))
        if form == "div" and min(a + b, c_ + d) > 0:
            tabs.append(np.array([[a, b], [c_, d]]))
CO = pd.DataFrame(co)
st = StratifiedTable(tabs)
mh = dict(order="MH_pooled_divergent", form="div", fisher_OR=round(st.oddsratio_pooled, 2), fisher_p=st.test_null_odds().pvalue)
# adjacency: family gene within 3 genes of a GT3 GS vs background genes of genomes with a GT3 GS
c = duckdb.connect()
c.execute("ATTACH '/Users/jacob/Documents/Sandbox/Sharur/Sharur/data/dpann/sharur.duckdb' AS d (READ_ONLY)")
c.execute("SET threads=2")
gt3 = PF[PF.GT3_GS > 0][["protein_id", "bin_id", "contig_id", "gene_index"]]
c.register("gt3", gt3)
bg = c.execute("""WITH b AS (SELECT DISTINCT bin_id FROM gt3)
   SELECT p.bin_id, count(*) n_genes,
          count(*) FILTER (WHERE EXISTS (SELECT 1 FROM gt3 g WHERE g.contig_id = p.contig_id AND abs(g.gene_index - p.gene_index) BETWEEN 1 AND 3)) n_near
   FROM d.proteins p JOIN b USING(bin_id) GROUP BY 1""").fetchdf()
c.register("fam", C[["protein_id", "group"]])
near = c.execute("""SELECT f.protein_id, f."group" AS grp, p.bin_id,
   EXISTS (SELECT 1 FROM gt3 g WHERE g.contig_id = p.contig_id AND abs(g.gene_index - p.gene_index) BETWEEN 1 AND 3) near3,
   EXISTS (SELECT 1 FROM gt3 g WHERE g.bin_id = p.bin_id) genome_has_gt3
   FROM fam f JOIN d.proteins p USING(protein_id)""").fetchdf()
adj = []
fam_bins = set(C.bin_id)
bgo = bg.merge(GF[["bin_id", "order"]], on="bin_id")
for gname in ["lead15", "labeled_unannotated", "labeled_named", "named_unlabeled"]:
    x = near[(near.grp == gname) & near.genome_has_gt3]
    for o in ["Woesearchaeales", "all"]:
        xo = x if o == "all" else x[x.bin_id.isin(set(GF.bin_id[GF.order == o]))]
        bo = bgo if o == "all" else bgo[bgo.order == o]
        bo = bo[~bo.bin_id.isin(fam_bins)]
        rate = bo.n_near.sum() / bo.n_genes.sum()
        k, n = int(xo.near3.sum()), len(xo)
        from scipy.stats import binomtest
        adj.append(dict(group=gname, scope=o, near3=f"{k}/{n}", background_rate=round(rate, 4),
                        background_genomes=len(bo), fold=round((k / n) / rate, 1) if n else np.nan,
                        binom_p=binomtest(k, n, rate, alternative="greater").pvalue if n else np.nan))
ADJ = pd.DataFrame(adj)
pd.concat([CO, pd.DataFrame([mh])], ignore_index=True).to_csv(H / "data/q4_cooccurrence.tsv", sep="\t", index=False)
ADJ.to_csv(H / "data/q4_adjacency.tsv", sep="\t", index=False)
print(CO.to_string(index=False)); print(mh)
print(ADJ.to_string(index=False))
