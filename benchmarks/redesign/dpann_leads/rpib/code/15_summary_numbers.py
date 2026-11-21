#!/usr/bin/env python3
"""Numbers for REPORT.txt: rung 1 (support, amplification), rung 2 (T1 E-values, model and sequence coverage from the raw
Aksha domain rows), rung 6 denominators, rung 7 by tier, criterion A (ILVFMAWC fraction, length), pLDDT of the conserved
core versus the C-terminal tail in the long-form models, and the monomer TM summary."""
import duckdb, pandas as pd, numpy as np, re
pd.set_option("display.width",250)
R="/Users/jacob/Documents/Sandbox/elsa_test/ELSA/benchmarks/redesign/dpann_leads/rpib"; D=R+"/data"
lab=pd.read_csv(D+"/rpib_genes.tsv",sep="\t"); B=pd.read_csv(D+"/genome_groups.tsv",sep="\t")
con=duckdb.connect(); con.execute("ATTACH '/Users/jacob/Documents/Sandbox/Sharur/Sharur/data/dpann/sharur.duckdb' AS d (READ_ONLY)")
print("rung1 support median", lab.support.median().round(3), "IQR", lab.support.quantile([.25,.75]).round(3).tolist(), "| votes_for median", lab.votes_for.median(), "| supporting genomes median", lab.supporting_genomes.median())
print("   amp (verification table)", lab.amp.unique()[:5], "| annotated", lab.annotated.unique()[:5])
n_pf=con.execute("SELECT count(DISTINCT protein_id) FROM d.annotations WHERE source='pfam' AND name='LacAB_rpiB'").fetchone()[0]; print("   DPANN PF02502 annotated proteins", n_pf, "-> amplification 287 /", n_pf, "=", round(287/n_pf,3))
print("   support by tier", lab.groupby("tier").support.median().round(3).to_dict(), "| confidence bins", lab.confidence_bin.value_counts().to_dict())
a=pd.read_csv(D+"/checkA_rows_labeled.tsv",sep="\t"); a=a[a.hmm_name=="LacAB_rpiB"].merge(lab[["protein_id","sequence_length","tier"]],left_on="sequence_id",right_on="protein_id")
a["mcov"]=(a.hmm_to-a.hmm_from+1)/138; a["scov"]=(a.ali_to-a.ali_from+1)/a.sequence_length
print(f"rung2 T1 n={len(a)} (tier T1 {int((lab.tier=='T1').sum())}); E median {a.evalue.median():.2g} range {a.evalue.min():.2g}-{a.evalue.max():.2g}; bits {a.bitscore.min()}-{a.bitscore.max()} (GA 24.7; >=24.7: {int((a.bitscore>=24.7).sum())}, domain >=24.7: {int((a.dom_bitscore>=24.7).sum())})")
print(f"   model coverage median {a.mcov.median():.2f} (hmm {a.hmm_from.median():.0f}-{a.hmm_to.median():.0f} of 138); sequence coverage median {a.scov.median():.2f} (ali {a.ali_from.median():.0f}-{a.ali_to.median():.0f})")
mi=B[B.ord=="o__Micrarchaeales"]
print("rung6: labeled 287 / Micrarchaeales", len(mi), "/ Micrarchaeota", len(B), "/ DPANN 3564; by family:", {f:f"{int(x.LF_labeled.sum())}/{len(x)} labeled, LF any {int(x.LF_any.sum())}" for f,x in mi.groupby("fam") if len(x)>=10})
print("   LF any outside Micrarchaeales", int(B[B.ord!="o__Micrarchaeales"].LF_any.sum()), "of", int((B.ord!="o__Micrarchaeales").sum()))
print("rung7 alignment same label:", int((lab.alignment_lists_same_label=="yes").sum()), "/ 287; by tier:", lab.groupby("tier").alignment_lists_same_label.apply(lambda x:f"{int((x=='yes').sum())}/{len(x)}").to_dict())
seq=con.execute("SELECT protein_id, sequence FROM d.proteins WHERE protein_id IN (SELECT unnest(?))",[lab.protein_id.tolist()]).df()
seq["hyd"]=seq.sequence.str.rstrip("*").map(lambda s: sum(c in "ILVFMAWC" for c in s)/len(s))
print("criterionA ILVFMAWC median", round(seq.hyd.median(),3), "IQR", seq.hyd.quantile([.25,.75]).round(3).tolist(), "| length median", lab.sequence_length.median(), "range", lab.sequence_length.min(), lab.sequence_length.max(), "q10-q90", lab.sequence_length.quantile([.1,.9]).tolist())
# pLDDT core vs tail; TM summary
U=pd.read_csv(D+"/usalign_out.tsv",sep="\t"); fs=pd.read_csv(D+"/fold_set.tsv",sep="\t").set_index("protein_id")
rows=[]
for r in U[U.kind=="mono_vs_2VVP_A"].itertuples():
    pid=r.a.split("/")[-1][:-4]; pl=[]
    for l in open(f"{R}/folds/monomers/{pid}.pdb"):
        if l.startswith("ATOM") and l[12:16].strip()=="CA": pl.append(float(l[60:66]))
    qi=0; last=0
    for x,y in zip(r.aln_a,r.aln_b):
        if x!="-": qi+=1
        if x!="-" and y!="-": last=qi
    pl=np.array(pl); rows.append(dict(pid=pid,role=fs.role.get(pid,""),L=len(pl),last_aligned=last,tail=len(pl)-last,core_pLDDT=pl[:last].mean().round(1),tail_pLDDT=pl[last:].mean().round(1) if len(pl)>last else None,TM_ref=r.TM_b,TM_q=r.TM_a))
P=pd.DataFrame(rows); print(P.to_string(index=False))
lf=P[P.role.str.startswith("labeled")]
print("LF monomers: TM (2VVP_A norm) median", lf.TM_ref.median(), "range", lf.TM_ref.min(), lf.TM_ref.max(), "| tail length median", lf["tail"].median(), "| core pLDDT median", lf.core_pLDDT.median(), "| tail pLDDT median", lf.tail_pLDDT.median())
