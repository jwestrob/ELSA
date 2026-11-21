#!/usr/bin/env python
"""galt step 8: locus table for the 76 labeled genes (64 + 12) and, for comparison, for every named DPANN family member.
Neighbors within +-6 genes on the same contig. Per neighbor: offset, strand relative to the focal gene, length, Pfam names
(Sharur Pfam run), KO (kofam, best score), KEGG KO, CDD best hit (protein_resemblance), and the HHsearch-Pfam best hit of
its protein family (family_annotations, probability >= 50) when it has no Pfam row.
Writes data/locus_neighbors.tsv (one row per focal x neighbor) and data/neighbors.faa (neighbor sequences, for dbCAN)."""
from pathlib import Path
import duckdb
import pandas as pd

H = Path("/Users/jacob/Documents/Sandbox/elsa_test/ELSA/benchmarks/redesign/dpann_leads/galt")
C = pd.read_csv(H / "data/census_proteins.tsv", sep="\t")
foc = C[C.lab76 | (C.group == "named_unlabeled") | (C.group == "ap4a_branch")][["protein_id", "group"]]
c = duckdb.connect()
c.execute("ATTACH '/Users/jacob/Documents/Sandbox/Sharur/Sharur/data/dpann/sharur.duckdb' AS d (READ_ONLY)")
c.execute("SET threads=2")
c.register("q", foc)
nb = c.execute("""SELECT q.protein_id focal, q."group" AS grp, p2.protein_id nb, p2.gene_index - p1.gene_index AS "off",
   CASE WHEN p2.strand = p1.strand THEN '+' ELSE '-' END rel_strand, p2.sequence_length length, p2.sequence, p1.bin_id
   FROM q JOIN d.proteins p1 USING(protein_id) JOIN d.proteins p2 ON p2.contig_id = p1.contig_id
   AND abs(p2.gene_index - p1.gene_index) <= 6""").fetchdf()
c.register("nbq", nb[["nb"]].drop_duplicates())
pf = c.execute("SELECT a.protein_id nb, string_agg(DISTINCT a.name, ',') pfam FROM d.annotations a JOIN nbq ON a.protein_id = nbq.nb WHERE a.source='pfam' GROUP BY 1").fetchdf()
ko = c.execute("""SELECT protein_id nb, arg_max(accession, score) ko FROM d.annotations a JOIN nbq ON a.protein_id = nbq.nb
   WHERE a.source='kofam' GROUP BY 1""").fetchdf()
kg = c.execute("""SELECT protein_id nb, string_agg(DISTINCT accession || ' ' || coalesce(name,''), '; ') kegg FROM d.annotations a JOIN nbq ON a.protein_id = nbq.nb
   WHERE a.source='kegg' GROUP BY 1""").fetchdf()
cd = c.execute("""SELECT r.protein_id nb, string_agg(DISTINCT r.accession || ':' || coalesce(r.name,''), ',') cdd FROM d.protein_resemblance r JOIN nbq ON r.protein_id = nbq.nb GROUP BY 1""").fetchdf()
hh = c.execute("""SELECT pf.protein_id nb, arg_max(fa.hit_accession || ' ' || coalesce(fa.hit_description,''), fa.probability) hhs_pfam
   FROM d.protein_families pf JOIN nbq ON pf.protein_id = nbq.nb
   JOIN d.family_annotations fa ON fa.family_id = pf.family_id AND fa.clustering_run = pf.clustering_run
   WHERE pf.clustering_run = 'permissive-e3c50s75' AND fa.source = 'hhsearch_pfam' AND fa.probability >= 50 GROUP BY 1""").fetchdf()
for t in (pf, ko, kg, cd, hh):
    nb = nb.merge(t, on="nb", how="left")
nb["hhs_pfam"] = nb.hhs_pfam.where(nb.pfam.isna())
with open(H / "data/neighbors.faa", "w") as fh:
    for r in nb[nb.off != 0].drop_duplicates("nb").itertuples():
        fh.write(f">{r.nb}\n{r.sequence.rstrip('*')}\n")
nb.drop(columns="sequence").to_csv(H / "data/locus_neighbors.tsv", sep="\t", index=False)
print(len(nb), "rows;", nb.focal.nunique(), "focal genes;", nb[nb.off != 0].nb.nunique(), "distinct neighbors")
