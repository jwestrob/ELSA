"""Q4: Topo-VIb_trans proteins with a DUF87 protein within 8 genes, outside Micrarchaeota: what the DUF87 protein is
and what surrounds it."""
import duckdb, pandas as pd, collections
c = duckdb.connect('/Users/jacob/Documents/Sandbox/Sharur/Sharur/data/dpann/sharur.duckdb', read_only=True)
q = """
with t6 as (select distinct a.protein_id from annotations a where a.source='pfam' and a.name='Topo-VIb_trans'),
d87 as (select distinct a.protein_id from annotations a where a.source='pfam' and a.name='DUF87'),
p as (select protein_id, contig_id, bin_id, gene_index, strand, sequence_length from proteins)
select t.protein_id t6, d.protein_id duf, d.sequence_length dlen, d.gene_index - t.gene_index dist, t.bin_id, t.contig_id, t.gene_index t6i, d.gene_index di,
  regexp_extract(b.taxonomy,'p__[^;]+') phylum, regexp_extract(b.taxonomy,'o__[^;]+') ord
from p t join t6 using(protein_id) join p d on d.contig_id=t.contig_id and abs(d.gene_index-t.gene_index)<=8 and d.protein_id in (select protein_id from d87)
join bins b on b.bin_id=t.bin_id"""
d = c.sql(q).df()
print(d.groupby('phylum').agg(genomes=('bin_id', 'nunique'), pairs=('t6', 'size'), dlen_med=('dlen', 'median')))
o = d[d.phylum.isin(['p__Nanobdellota', 'p__Aenigmatarchaeota'])].copy()
print(o.groupby(['phylum', 'ord']).bin_id.nunique())
print('distance top6B->DUF87', o.dist.value_counts().head(10).to_dict())
c.register('o', o)
pf = c.sql("""select o.duf, string_agg(distinct a.name, ';' order by a.name) pf from o join annotations a on a.protein_id=o.duf and a.source='pfam' group by 1""").df()
print(pf.pf.value_counts().head(8))
# neighborhood of the DUF87 gene: Pfam names within +-4 genes
nb = c.sql("""select o.duf, n.gene_index - o.di rel, n.sequence_length L, string_agg(distinct a.name, ';' order by a.name) pf, any_value(k.accession) ko
              from o join proteins n on n.contig_id=o.contig_id and n.gene_index between o.di-4 and o.di+4
              left join annotations a on a.protein_id=n.protein_id and a.source='pfam'
              left join annotations k on k.protein_id=n.protein_id and k.source='kofam'
              group by all""").df()
for r in range(-4, 5):
    s = nb[nb.rel == r]
    print(r, len(s), s.pf.fillna('none').value_counts().head(4).to_dict())
o.to_csv('data/wider_top6_duf87.tsv', sep='\t', index=False)
