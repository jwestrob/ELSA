#!/usr/bin/env python3
"""Question 4: ortholog or paralog, and conserved neighbors.
(a) For each lead-1 gene, the genes within +-4 positions and their mmseqs family ids (pfr-8a31f5747e-family, 30% id).
(b) For every DPANN protein with a Pfam Thy1 row, the same window. A lead gene and an annotated Thy1 gene in another
    genome are positional matches when their windows share >= 2 neighbor families. Report how many annotated Thy1
    genes match each lead gene, how many of those carry two Thy1 domains, and their lengths.
(c) Neighbor annotation tallies (Pfam names, else KO description, else CDD name) by offset for lead-1 genes and for
    annotated two-domain Thy1 genes of the same orders.
Writes data/neighbors_lead1.tsv, data/positional_matches.tsv; prints summaries.
usage: python code/09_neighbors.py"""
import duckdb, pandas as pd, numpy as np
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300); pd.set_option("display.max_colwidth", 50)
con = duckdb.connect()
con.execute("ATTACH '/Users/jacob/Documents/Sandbox/Sharur/Sharur/data/dpann/sharur.duckdb' AS d (READ_ONLY)")
L = pd.read_csv("data/lead_genes.tsv", sep="\t").query("lab=='Thy1'")
W = 4
con.execute("""CREATE TEMP TABLE thy AS SELECT a.protein_id, count(*) ndom FROM d.annotations a WHERE a.source='pfam' AND a."name"='Thy1' GROUP BY 1""")
con.register("lead", L[["protein_id"]])
con.execute("""CREATE TEMP TABLE focal AS SELECT pr.protein_id, pr.contig_id, pr.gene_index, pr.strand, pr.bin_id, pr.sequence_length len,
       CASE WHEN pr.protein_id IN (SELECT protein_id FROM lead) THEN 'lead' ELSE 'ann' END kind, coalesce(t.ndom, 0) ndom
       FROM d.proteins pr LEFT JOIN thy t USING (protein_id)
       WHERE pr.protein_id IN (SELECT protein_id FROM thy) OR pr.protein_id IN (SELECT protein_id FROM lead)""")
con.execute(f"""CREATE TEMP TABLE win AS SELECT f.protein_id focal, f.kind, f.bin_id, f.strand fstrand,
       (n.gene_index - f.gene_index) * CASE WHEN f.strand='-' THEN -1 ELSE 1 END rel, n.protein_id nb, n.strand
       FROM focal f JOIN d.proteins n ON n.contig_id=f.contig_id AND n.gene_index BETWEEN f.gene_index-{W} AND f.gene_index+{W}
       WHERE n.protein_id <> f.protein_id""")
con.execute("""CREATE TEMP TABLE winf AS SELECT w.*, pf.family_id fam FROM win w LEFT JOIN d.protein_families pf
       ON pf.protein_id=w.nb AND pf.clustering_run='pfr-8a31f5747e-family'""")
con.execute("""CREATE TEMP TABLE nbann AS SELECT w.nb,
       (SELECT string_agg(DISTINCT "name", '+') FROM d.annotations a WHERE a.protein_id=w.nb AND a.source='pfam') pf,
       (SELECT any_value(description) FROM d.annotations a WHERE a.protein_id=w.nb AND a.source IN ('kofam','kegg') AND description IS NOT NULL) ko,
       (SELECT any_value(accession) FROM d.annotations a WHERE a.protein_id=w.nb AND a.source='kofam') koacc,
       (SELECT arg_min("name", evalue) FROM d.protein_resemblance r WHERE r.protein_id=w.nb) cdd
       FROM (SELECT DISTINCT nb FROM winf) w""")
N = con.execute("SELECT w.*, a.pf, a.ko, a.koacc, a.cdd FROM winf w LEFT JOIN nbann a USING (nb)").df()
N["lab"] = N.pf.fillna(N.koacc.fillna("") + " " + N.ko.fillna("")).str.strip().replace("", np.nan).fillna("cdd:" + N.cdd.fillna("none"))
N[N.kind == "lead"].to_csv("data/neighbors_lead1.tsv", sep="\t", index=False)
B = con.execute("SELECT bin_id, split_part(taxonomy,';',4) ord FROM d.bins").df().set_index("bin_id").ord
F = con.execute("SELECT * FROM focal").df()

# (b) positional matches by shared neighbor families
fams = N.dropna(subset=["fam"]).groupby("focal").fam.apply(set).to_dict()
ann = F[F.kind == "ann"]
ann_f = {p: fams.get(p, set()) for p in ann.protein_id}
inv = {}
for p, s in ann_f.items():
    for x in s:
        inv.setdefault(x, []).append(p)
rows = []
for r in L.itertuples():
    s = fams.get(r.protein_id, set())
    cnt = {}
    for x in s:
        for p in inv.get(x, []):
            cnt[p] = cnt.get(p, 0) + 1
    m = [p for p, c in cnt.items() if c >= 2]
    mm = ann[ann.protein_id.isin(m) & (ann.bin_id != r.bin_id)]
    rows.append(dict(protein_id=r.protein_id, len=r.len, tier=r.tier, order=B.get(r.bin_id), n_nb_fams=len(s),
                     matched_annotated=len(mm), matched_two_domain=int((mm.ndom == 2).sum()),
                     matched_len_median=float(mm.len.median()) if len(mm) else np.nan,
                     matched_orders=";".join(sorted(set(B.reindex(mm.bin_id).fillna("?")))[:4]),
                     same_genome_other_thy1=int(((ann.bin_id == r.bin_id)).sum())))
R = pd.DataFrame(rows)
R.to_csv("data/positional_matches.tsv", sep="\t", index=False)
print(R.to_string(index=False))
print("lead genes with >=1 positional match to an annotated Thy1 gene:", int((R.matched_annotated > 0).sum()), "/", len(R),
      "| matched annotated genes two-domain:", int(R.matched_two_domain.sum()), "/", int(R.matched_annotated.sum()),
      "| lead genomes carrying another Pfam-Thy1 gene:", int((R.same_genome_other_thy1 > 0).sum()))

# (c) neighbor tallies by offset
def tally(df, title, n_focal):
    print(f"\n## {title} (n focal = {n_focal}); top labels at each offset (count of focal genes)")
    for o in [-3, -2, -1, 1, 2, 3]:
        t = df[df.rel == o].groupby("lab").focal.nunique().sort_values(ascending=False).head(4)
        print(f"  {o:+d}: " + "; ".join(f"{k[:45]} {v}" for k, v in t.items()))
tally(N[N.kind == "lead"], "lead-1 genes", L.protein_id.nunique())
two = F[(F.kind == "ann") & (F.ndom == 2)].copy()
two["ord"] = B.reindex(two.bin_id).values
for od in ["o__Pacearchaeales", "o__Woesearchaeales"]:
    ids = set(two[two.ord == od].protein_id)
    tally(N[N.focal.isin(ids)], f"annotated two-domain Thy1, {od}", len(ids))
