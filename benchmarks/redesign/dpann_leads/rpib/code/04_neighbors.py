#!/usr/bin/env python3
"""Conserved neighbors of the slot (strand-aware offsets -6..+6) for the 287 labeled genes and the 165 same-slot PF02502
carriers; plus the second-copy check in the 59 genomes with a PF02502 gene elsewhere. Writes data/neighbors_offsets.tsv,
data/second_copy.tsv."""
import duckdb, pandas as pd, numpy as np, subprocess, tempfile, os
pd.set_option("display.width",250); pd.set_option("display.max_rows",400); pd.set_option("display.max_colwidth",120)
D="/Users/jacob/Documents/Sandbox/elsa_test/ELSA/benchmarks/redesign/dpann_leads/rpib/data"
con=duckdb.connect()
con.execute("ATTACH '/Users/jacob/Documents/Sandbox/Sharur/Sharur/data/dpann/sharur.duckdb' AS d (READ_ONLY)")
lab=pd.read_csv(D+"/rpib_genes.tsv",sep="\t"); car=pd.read_csv(D+"/carriers_class.tsv",sep="\t")
same=car[car.slot=="same_slot_as_labeled"].protein_id
focal=pd.DataFrame({"protein_id":list(lab.protein_id)+list(same),"set":["labeled"]*len(lab)+["same_slot_PF02502"]*len(same)})
con.register("focal",focal)
W=6
N=con.execute(f"""WITH f AS (SELECT fo.protein_id fid, fo."set", pr.contig_id, pr.gene_index gi, pr.strand fs FROM focal fo JOIN d.proteins pr USING (protein_id)),
 w AS (SELECT f.fid, f."set", f.fs, p.protein_id, p.gene_index - f.gi raw, p.strand, p.sequence_length len FROM f JOIN d.proteins p ON p.contig_id=f.contig_id AND p.gene_index BETWEEN f.gi-{W} AND f.gi+{W})
 SELECT w.*, CASE WHEN fs IN ('-','-1') OR CAST(fs AS VARCHAR) IN ('-','-1') THEN -raw ELSE raw END o,
  (SELECT string_agg(DISTINCT a.name, '+' ORDER BY a.name) FROM d.annotations a WHERE a.protein_id=w.protein_id AND a.source='pfam') pfam,
  (SELECT string_agg(DISTINCT a.accession || ':' || coalesce(a.description,a.name), '+') FROM d.annotations a WHERE a.protein_id=w.protein_id AND a.source='kofam') ko,
  (SELECT arg_min(r."name", r.evalue) FROM d.protein_resemblance r WHERE r.protein_id=w.protein_id) cdd
 FROM w""").df()
print("strand values:", N.fs.astype(str).value_counts().to_dict())
N["rel_strand"]=np.where(N.strand.astype(str)==N.fs.astype(str),"same","opp")
N.to_csv(D+"/neighbors_offsets.tsv",sep="\t",index=False)
for st,g in N.groupby("set"):
    nf=g.fid.nunique(); print(f"\n##### {st}: {nf} focal genes")
    for o in range(-W,W+1):
        x=g[g.o==o]
        if o==0: continue
        top_pf=x.pfam.fillna("(no Pfam)").value_counts().head(3)
        top_ko=x.ko.dropna().str.slice(0,70).value_counts().head(2)
        print(f" {o:+d} n={len(x)} same_strand={ (x.rel_strand=='same').mean():.2f} | " + "; ".join(f"{k} {v/nf:.2f}" for k,v in top_pf.items()) + " || " + "; ".join(f"{k} {v/nf:.2f}" for k,v in top_ko.items()))
# second copy check in genomes that carry a PF02502 gene besides the labeled one
oth=con.execute("""SELECT pr.bin_id, a.protein_id other_pid, pr.contig_id other_contig, pr.gene_index other_gi, pr.sequence_length other_len, pr.sequence other_seq, min(a.evalue) other_e
  FROM d.annotations a JOIN d.proteins pr USING (protein_id) WHERE a.source='pfam' AND a.name='LacAB_rpiB' GROUP BY ALL""").df()
seqs=con.execute("SELECT protein_id, sequence FROM d.proteins WHERE protein_id IN (SELECT protein_id FROM focal)").df().set_index("protein_id").sequence
m=lab[["protein_id","bin_id","contig_id","gene_index","sequence_length"]].merge(oth,on="bin_id")
m=m.merge(car[["protein_id","slot","ko"]].rename(columns={"protein_id":"other_pid"}),on="other_pid",how="left")
def pid(a,b):
    with tempfile.TemporaryDirectory() as t:
        open(t+"/i.faa","w").write(f">a\n{a.rstrip('*')}\n>b\n{b.rstrip('*')}\n")
        o=subprocess.run(["mafft","--quiet","--globalpair","--maxiterate","100",t+"/i.faa"],capture_output=True,text=True).stdout
    s=[l for l in o.split(">") if l]; A=["".join(x.split("\n")[1:]) for x in s]
    al=[(x,y) for x,y in zip(*A) if x!="-" and y!="-"]
    return sum(x==y for x,y in al)/max(1,min(len(a),len(b)))
m["identity_to_labeled"]=[pid(seqs[r.protein_id],r.other_seq) for r in m.itertuples()]
m["same_contig"]=m.contig_id==m.other_contig
m.drop(columns="other_seq").to_csv(D+"/second_copy.tsv",sep="\t",index=False)
print(f"\n== second copies: {len(m)} PF02502 genes in {m.bin_id.nunique()} of 287 genomes")
print(m[["sequence_length","other_len","identity_to_labeled"]].describe().round(3).to_string())
print(" slot class:", m.slot.value_counts(dropna=False).to_dict(), "| K01808 on other:", int(m.ko.fillna('').str.contains('K01808').sum()), "| same contig:", int(m.same_contig.sum()))
print(" identity >= 0.9 (same gene counted twice):", int((m.identity_to_labeled>=0.9).sum()))
