"""Q1: Is Top6B-KH a housekeeping pair independent of the machine? Neighborhood of every DPANN top6B, by phylum,
and whether the F17 machine (ATPase504 / A2Ml794 of the atlas windows) sits next to it."""
import duckdb, pandas as pd
c = duckdb.connect('/Users/jacob/Documents/Sandbox/Sharur/Sharur/data/dpann/sharur.duckdb', read_only=True)
a = pd.read_parquet('data/f17_windows.parquet')
mach = set(a[a.role.isin(['ATPase504', 'A2Ml794', 'TM277a', 'wHTH244', 'TM132'])].protein_id)
c.register('m', pd.DataFrame({'protein_id': sorted(mach)}))
q = """
with t6 as (select distinct protein_id from annotations where (source='pfam' and name='Topo-VIb_trans') or (source='kofam' and accession='K03167')),
kh as (select distinct protein_id from annotations where (source='pfam' and name like 'KH_KRR1%') or (source='kofam' and accession='K06961')),
t6a as (select distinct protein_id from annotations where (source='kofam' and accession='K03166') or (source='pfam' and name='TP6A_N')),
p as (select protein_id, contig_id, bin_id, gene_index, strand from proteins),
t as (select p.* from p join t6 using(protein_id))
select t.protein_id, t.bin_id, t.contig_id, t.gene_index, regexp_extract(b.taxonomy, 'p__[^;]+') phylum, regexp_extract(b.taxonomy, 'o__[^;]+') ord, b.completeness,
  max(case when n.protein_id in (select protein_id from kh) and abs(n.gene_index-t.gene_index)<=2 then 1 else 0 end) kh2,
  max(case when n.protein_id in (select protein_id from kh) and abs(n.gene_index-t.gene_index)=1 then 1 else 0 end) kh1,
  max(case when n.protein_id in (select protein_id from t6a) and abs(n.gene_index-t.gene_index)<=2 then 1 else 0 end) t6a2,
  max(case when n.protein_id in (select protein_id from m) and abs(n.gene_index-t.gene_index)<=3 then 1 else 0 end) mach3
from t join p n on n.contig_id=t.contig_id and n.gene_index between t.gene_index-3 and t.gene_index+3 and n.protein_id<>t.protein_id
join bins b on b.bin_id=t.bin_id group by all"""
d = c.sql(q).df()
d.to_csv('data/top6b_context.tsv', sep='\t', index=False)
g = d.groupby('phylum').agg(top6b=('protein_id', 'count'), genomes=('bin_id', 'nunique'), kh_adj=('kh1', 'sum'), kh_le2=('kh2', 'sum'),
                            top6a_le2=('t6a2', 'sum'), machine_le3=('mach3', 'sum')).sort_values('top6b', ascending=False)
print(g.to_string())
mi = d[d.phylum == 'p__Micrarchaeota']
print('Micrarchaeota: top6B with KH within 2:', mi.kh2.sum(), '/', len(mi), '; with machine within 3:', mi.mach3.sum())
print(pd.crosstab(mi.kh2, mi.mach3))
print(mi.groupby('ord').agg(n=('protein_id','count'), kh2=('kh2','sum'), mach3=('mach3','sum'), t6a=('t6a2','sum')))
