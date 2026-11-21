#!/usr/bin/env python3
"""Numbers for REPORT.txt / leads.tsv: rung 2 coverage per gene, per-gene composite residue sets, clade denominators
(rung 6), criterion A hydrophobic fraction, unlabeled family members in TS-lacking genomes, lead 2 summary.
usage: python code/15_numbers.py"""
import duckdb, pandas as pd, numpy as np, json
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
con = duckdb.connect()
con.execute("ATTACH '/Users/jacob/Documents/Sandbox/Sharur/Sharur/data/dpann/sharur.duckdb' AS d (READ_ONLY)")
L = pd.read_csv("data/lead_genes.tsv", sep="\t")
H = pd.read_csv("data/aksha_picked/user_hmms_hits_df.tsv", sep="\t")
seqs = {}
for fn in ["data/lead1.faa", "data/lead2.faa"]:
    k = None
    for l in open(fn):
        l = l.strip()
        if l.startswith(">"): k = l[1:]; seqs[k] = ""
        else: seqs[k] += l
def cov(g, mlen, slen):
    m = np.zeros(mlen + 1, bool); s = np.zeros(slen + 1, bool)
    for r in g.itertuples():
        m[r.hmm_from:r.hmm_to + 1] = True; s[r.ali_from:r.ali_to + 1] = True
    return m.sum() / mlen, s.sum() / slen
rows = []
for r in L.itertuples():
    fam = "Thy1" if r.lab == "Thy1" else "Thymidylat_synt"
    g = H[(H.sequence_id == r.protein_id) & (H.hmm_name == fam)]
    mc, sc = cov(g, 191 if fam == "Thy1" else 266, r.len) if len(g) else (0, 0)
    s = seqs[r.protein_id]
    hyd = sum(c in "ILVFMAWC" for c in s) / len(s)
    rows.append(dict(protein_id=r.protein_id, lab=r.lab, len=r.len, tier=r.tier, best_full_E=g.evalue.min() if len(g) else np.nan,
                     full_bits=g.bitscore.max() if len(g) else np.nan, n_dom_hits=len(g), model_cov=round(mc, 2), seq_cov=round(sc, 2), hyd=round(hyd, 3)))
R = pd.DataFrame(rows)
R.to_csv("data/rung2_per_gene.tsv", sep="\t", index=False)
for lab, g in R.groupby("lab"):
    t1 = g[g.tier == "T1"]
    print(lab, "genes", len(g), "with any weak hit", int((g.n_dom_hits > 0).sum()), "| T1", len(t1), "full E range", t1.best_full_E.min(), t1.best_full_E.max(),
          "bits range", g.full_bits.min(), g.full_bits.max(), "| model cov median (T1)", t1.model_cov.median(), "range", t1.model_cov.min(), t1.model_cov.max(),
          "| seq cov median", t1.seq_cov.median(), "| hits in both halves (len>=420)", int(((g.n_dom_hits >= 2) & (g.len >= 420)).sum()), "of", int((g.len >= 420).sum()),
          "| hyd median", g.hyd.median(), "range", g.hyd.min(), g.hyd.max())

# composite residue completeness per long gene
C = pd.read_csv("data/residues_by_half.tsv", sep="\t")
C["pid"] = C.name.str.split("|").str[0]
C["grp"] = C.name.str.split("|").str[1].str.split(":").str[0]
out = []
for pid, g in C[C.grp.isin(["lead_N", "lead_C"])].groupby("pid"):
    n = g[g.grp == "lead_N"].iloc[0]; c = g[g.grp == "lead_C"].iloc[0]
    out.append(dict(pid=pid, N_S88=n.S88 == "S", N_R90=n.R90 == "R", N_H53=n.H53 == "H", N_R147=n.R147 == "R", C_R78=c.R78 == "R", C_R174=c.R174 == "R", C_H79=c.H79 == "H"))
O = pd.DataFrame(out)
O["composite_core"] = O.N_S88 & O.N_R90 & O.C_R78 & O.C_R174
print("long lead genes with N-half S88+R90 and C-half R78+R174:", int(O.composite_core.sum()), "/", len(O))
print(O.sum(numeric_only=True).to_dict())
for grp in ["ann2_N", "ann2_C"]:
    pass
A2 = C[C.grp.isin(["ann2_N", "ann2_C"])]
an = A2[A2.grp == "ann2_N"].set_index("pid"); ac = A2[A2.grp == "ann2_C"].set_index("pid")
both = an.index.intersection(ac.index)
comp = ((an.loc[both].S88 == "S") & (an.loc[both].R90 == "R") & (ac.loc[both].R78 == "R") & (ac.loc[both].R174 == "R"))
print("annotated two-domain DPANN members with the same composite set:", int(comp.sum()), "/", len(both))
lostN = ((an.loc[both].R78 != "R") & (an.loc[both].R174 != "R")).sum(); lostC = ((ac.loc[both].S88 != "S") & (ac.loc[both].R90 != "R")).sum()
print("annotated two-domain: N half lacks both R78 and R174:", int(lostN), "/", len(both), "; C half lacks both S88 and R90:", int(lostC), "/", len(both))
a1 = C[C.grp == "ann1"]
print("annotated one-domain with all of S88 R90 R78 R174:", int(((a1.S88 == "S") & (a1.R90 == "R") & (a1.R78 == "R") & (a1.R174 == "R")).sum()), "/", len(a1))

# rung 6 denominators
G = pd.read_csv("data/genome_functions.tsv", sep="\t")
tot_order = G.groupby("order").size(); tot_phy = G.groupby("phylum").size()
for lab, flag in [("Thy1", "lead1"), ("Thymidylat_synt", "lead2")]:
    g = G[G[flag] == 1]
    print(lab, "genomes by order (lead genomes / order total):", {o: f"{n}/{tot_order[o]}" for o, n in g.order.value_counts().items()})
    print(lab, "by phylum:", {o: f"{n}/{tot_phy[o]}" for o, n in g.phylum.value_counts().items()})

# unlabeled members of the lead-1 families in TS-lacking genomes outside the lead set
fams = tuple(L[L.lab == "Thy1"].perm.dropna().unique())
U = con.execute(f"""SELECT pf.protein_id, pr.bin_id, pr.sequence_length len FROM d.protein_families pf JOIN d.proteins pr USING (protein_id)
   WHERE pf.clustering_run='permissive-e3c50s75' AND pf.family_id IN {fams}
   AND pf.protein_id NOT IN (SELECT protein_id FROM d.annotations WHERE source='pfam' AND "name"='Thy1')""").df()
U = U[~U.protein_id.isin(L.protein_id)]
U = U.merge(G[["bin_id", "TS_any", "order", "completeness"]], on="bin_id", how="left")
print("unlabeled, unannotated members of lead-1 permissive families outside the lead set:", len(U), "| in genomes with no other TS:", int((U.TS_any == 0).sum()),
      "| orders", U.order.value_counts().to_dict(), "| lengths", U.len.tolist())
