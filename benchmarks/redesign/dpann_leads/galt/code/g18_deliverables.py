#!/usr/bin/env python
"""galt step 18: deliverable tables in the workstream folder.
  q1_phylogeny.tsv   clades of the rooted ML tree (g14) with support, composition and orders
  q2_residues.tsv    determinant positions (g16), references and the 15 by genome group, labeled-named and named form
  q3_locus.tsv       locus recurrence (+-3 genes) per focal group, and genome routes (g13)
  q4_distribution.tsv  per-order counts for subfamily D (the clade holding the 64) and the named form N, plus tests (g17)
  q5_cofold.tsv      co-fold metrics per protein x ligand set (g11), chain-averaged
  q6_identity.tsv    per labeled gene: lists, supporters, identities, profile scores (g07)
  leads.tsv          one row, brief schema"""
from pathlib import Path
import pandas as pd

H = Path("/Users/jacob/Documents/Sandbox/elsa_test/ELSA/benchmarks/redesign/dpann_leads/galt")
pd.read_csv(H / "data/q1_clades.tsv", sep="\t").to_csv(H / "q1_phylogeny.tsv", sep="\t", index=False)
pd.read_csv(H / "data/q2_table.tsv", sep="\t", keep_default_na=False).to_csv(H / "q2_residues.tsv", sep="\t", index=False)
R = pd.read_csv(H / "data/q3_locus_recurrence_long.tsv", sep="\t")
R = R[R.window == 3]
R["cell"] = R.genes.astype(str) + "/" + R.total.astype(str)
loc = R.pivot(index="feature", columns="group", values="cell").reset_index()
loc.insert(0, "table", "locus_within_3_genes")
rt = pd.read_csv(H / "data/q3_genome_routes.tsv", sep="\t")
rt.insert(0, "table", "genome_routes")
pd.concat([loc, rt], ignore_index=True).to_csv(H / "q3_locus.tsv", sep="\t", index=False)
a = pd.read_csv(H / "data/q4_subfamily_by_order.tsv", sep="\t"); a.insert(0, "table", "by_order")
b = pd.read_csv(H / "data/q4_subfamily_tests.tsv", sep="\t"); b.insert(0, "table", "tests")
adj = pd.read_csv(H / "data/q4_adjacency.tsv", sep="\t"); adj.insert(0, "table", "adjacency_GT3_within_3")
pd.concat([a, b, adj], ignore_index=True).to_csv(H / "q4_distribution.tsv", sep="\t", index=False)
G = pd.read_csv(H / "data/cofold_geometry.tsv", sep="\t")
q5 = G.groupby(["tag", "protein", "ligset", "nucleotide"]).agg(d_his_alphaP=("d_his_aP", "mean"), productive_chains=("productive", "sum"),
                                                              ligand_plddt=("lig_plddt", "mean"), pair_iptm_protein_ligand=("iptm_pl", "mean"),
                                                              curated_contact_share=("contact", "first"), ptm=("ptm", "first"),
                                                              iptm=("iptm", "first")).round(3).reset_index()
q5.to_csv(H / "q5_cofold.tsv", sep="\t", index=False)
pd.read_csv(H / "data/q6_identity.tsv", sep="\t").to_csv(H / "q6_identity.tsv", sep="\t", index=False)
print("written")
