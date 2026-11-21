"""Existing Sharur family-level HHsearch (Pfam39) / HHblits (PDB100) hits for each role, via the members' families."""
import duckdb, pandas as pd
a = pd.read_parquet('data/f17_windows.parquet')
r = a[a.role != '-'][['protein_id', 'role']].drop_duplicates()
c = duckdb.connect('/Users/jacob/Documents/Sandbox/Sharur/Sharur/data/dpann/sharur.duckdb', read_only=True)
c.register('r', r)
fam = c.sql("select r.role, pf.family_id, pf.clustering_run, count(*) n from r join protein_families pf using(protein_id) group by all").df()
print(fam.clustering_run.value_counts())
fam = fam.sort_values('n', ascending=False)
c.register('fam', fam)
h = c.sql("""select fam.role, fam.family_id, fam.n, fa.source, fa.hit_name, fa.hit_description, fa.probability, fa.evalue, fa.q_start, fa.q_end, fa.q_length, fa.t_start, fa.t_end, fa.t_length
             from fam join family_annotations fa on fa.family_id = fam.family_id and fa.clustering_run = fam.clustering_run
             where fa.probability >= 50 order by fam.role, fam.n desc, fa.probability desc""").df()
h.to_csv('data/sharur_family_hits.tsv', sep='\t', index=False)
pd.set_option('display.width', 250)
for role, s in h.groupby('role'):
    print('==', role, 'families with hits', s.family_id.nunique(), 'members covered', s.drop_duplicates('family_id').n.sum())
    print(s.groupby(['source', 'hit_name']).agg(nfam=('family_id', 'nunique'), mem=('n', 'sum'), pmax=('probability', 'max'),
          desc=('hit_description', 'first')).sort_values('mem', ascending=False).head(8).to_string(max_colwidth=70))
print(fam.groupby('role').agg(nfam=('family_id', 'nunique'), mem=('n', 'sum')))
