"""Family 17 members: atlas genes + DB coordinates, strand, sequences, flanks (+-6 genes)."""
import duckdb, pandas as pd
L = '/Users/jacob/Documents/Sandbox/elsa_test/ELSA/benchmarks/redesign/dpann_leads/large_loci_dpann'
fg = pd.read_parquet(f'{L}/data/family_genes.parquet')
f = fg[fg.family == 17].copy()
c = duckdb.connect('/Users/jacob/Documents/Sandbox/Sharur/Sharur/data/dpann/sharur.duckdb', read_only=True)
c.register('f', f[['protein_id', 'locus', 'gene_cluster']])
p = c.sql("""select p.protein_id, p.contig_id, p.bin_id, p.gene_index, p.start, p.end_coord, p.strand, p.sequence_length, f.locus, f.gene_cluster
             from proteins p join f using(protein_id)""").df()
# window per locus
w = p.groupby('locus').agg(contig_id=('contig_id', 'first'), lo=('gene_index', 'min'), hi=('gene_index', 'max')).reset_index()
w['lo'] -= 6; w['hi'] += 6
c.register('w', w)
allg = c.sql("""select w.locus, p.protein_id, p.contig_id, p.bin_id, p.gene_index, p.start, p.end_coord, p.strand, p.sequence_length, p.sequence
               from proteins p join w on p.contig_id = w.contig_id and p.gene_index between w.lo and w.hi""").df()
allg = allg.merge(f[['protein_id', 'gene_cluster']], on='protein_id', how='left')
allg['in_family'] = allg.gene_cluster.notna()
ann = c.sql("""select a.protein_id, string_agg(distinct a.name, ';') pf from annotations a join allg using(protein_id)
              where a.source='pfam' group by 1""").df()
ko = c.sql("""select a.protein_id, string_agg(distinct a.accession, ';') ko from annotations a join allg using(protein_id)
              where a.source='kofam' group by 1""").df()
allg = allg.merge(ann, how='left').merge(ko, how='left')
allg = allg.sort_values(['locus', 'gene_index'])
allg.to_parquet('data/f17_windows.parquet')
print(allg.shape, allg.locus.nunique(), allg.in_family.sum())
