#!/usr/bin/env python3
"""Pick fold representatives: 12 labeled long-form genes (T1 5, T2 3, T2b 4) one per genus (12 genera) over both Micrarchaeales families,
two positive controls (a same-slot long-form gene annotated with PF02502 at the gathering threshold; a canonical short
Micrarchaeales RpiB with K01808 and the E. coli-type C66/H99 pair), two negative controls of similar length from unrelated
families (Ribosomal_L6, Thymidylate_kin), and the oligomer set. Writes folds_in/*.faa and data/fold_set.tsv."""
import duckdb, pandas as pd, json, os
R="/Users/jacob/Documents/Sandbox/elsa_test/ELSA/benchmarks/redesign/dpann_leads/rpib"; D=R+"/data"; F=R+"/folds_in"; os.makedirs(F,exist_ok=True)
con=duckdb.connect(); con.execute("ATTACH '/Users/jacob/Documents/Sandbox/Sharur/Sharur/data/dpann/sharur.duckdb' AS d (READ_ONLY)")
lab=pd.read_csv(D+"/rpib_genes.tsv",sep="\t"); rc=pd.read_csv(D+"/residue_calls.tsv",sep="\t").set_index("id")
car=pd.read_csv(D+"/carriers_class.tsv",sep="\t")
seq=lambda ids: con.execute("SELECT protein_id, sequence FROM d.proteins WHERE protein_id IN (SELECT unnest(?))",[list(ids)]).df().set_index("protein_id").sequence.str.rstrip("*")
pick=[]; used_genus=set()
plan=[("T1","f__CABMGC01",3),("T1","f__Micrarchaeaceae",2),("T2","f__Micrarchaeaceae",3),("T2b","f__CABMGC01",2),("T2b","f__Micrarchaeaceae",2)]
for tier,fam,n in plan:   # one gene per genus, genera never reused, largest genera first; gene nearest the genus median length
    t=lab[(lab.tier==tier)&(lab.family==fam)]
    k=0
    for gen in t.genus.value_counts().index:
        if gen in used_genus: continue
        x=t[t.genus==gen].copy(); x["dl"]=(x.sequence_length-x.sequence_length.median()).abs()
        r=x.sort_values(["dl","completeness"],ascending=[True,False]).iloc[0]
        pick.append((r.protein_id,f"labeled_{tier}",fam,gen,int(r.sequence_length))); used_genus.add(gen); k+=1
        if k==n: break
ss=car[car.slot=="same_slot_as_labeled"].sort_values("pf02502_e").iloc[0]
pick.append((ss.protein_id,"pos_same_slot_PF02502_GA","","",int(ss.len)))
short=car[(car.slot!="same_slot_as_labeled")&car.ko.fillna("").str.contains("K01808")&(car.ord=="o__Micrarchaeales")].copy()
short=short[short.protein_id.map(lambda i: rc.loc[i,"C66"]=="C" and rc.loc[i,"H99"]=="H" and rc.loc[i,"D9"]=="D" and rc.loc[i,"H10"]=="H")]
s1=short.sort_values("comp",ascending=False).iloc[0]; pick.append((s1.protein_id,"pos_canonical_RpiB_K01808","","",int(s1.len)))
for pf in ["Ribosomal_L6","Thymidylate_kin"]:
    x=con.execute("""SELECT pr.protein_id, pr.sequence_length len FROM d.annotations a JOIN d.proteins pr USING (protein_id) JOIN d.bins b ON b.bin_id=pr.bin_id
      WHERE a.source='pfam' AND a.name=? AND b.taxonomy LIKE '%o__Micrarchaeales%' AND pr.sequence_length BETWEEN 180 AND 200 ORDER BY a.evalue, pr.protein_id LIMIT 1""",[pf]).df().iloc[0]
    pick.append((x.protein_id,f"neg_{pf}","","",int(x.len)))
P=pd.DataFrame(pick,columns=["protein_id","role","family","genus","length"])
S=seq(P.protein_id); P["sequence"]=P.protein_id.map(S)
# reference sequences (public): RPIB_MYCTU, RPIB_ECOLI
for a in ["P9WKD7","P37351"]:
    j=json.load(open(f"{D}/ref/{a}.json")); P.loc[len(P)]=[a,"ref_"+j["uniProtkbId"],"","",len(j["sequence"]["value"]),j["sequence"]["value"]]
with open(F+"/monomers.faa","w") as f:
    for r in P[~P.role.str.startswith("ref_")].itertuples(): f.write(f">{r.protein_id}\n{r.sequence}\n")
# oligomer set: 3 long-form genes (T1, T2, T2b), the canonical DPANN RpiB, and RPIB_MYCTU as method control
olig=[P[P.role=="labeled_T1"].iloc[0], P[P.role=="labeled_T2"].iloc[0], P[P.role=="labeled_T2b"].iloc[0], P[P.role=="pos_canonical_RpiB_K01808"].iloc[0], P[P.role=="ref_RPIB_MYCTU"].iloc[0]]
with open(F+"/dimers.faa","w") as f:
    for r in olig: f.write(f">{r.protein_id}__dimer_R5P\n{r.sequence}\n")
P["dimer"]=P.protein_id.isin([r.protein_id for r in olig])
P.drop(columns="sequence").to_csv(D+"/fold_set.tsv",sep="\t",index=False)
print(P.drop(columns="sequence").to_string(index=False))
