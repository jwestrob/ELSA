"""Raw TXSScan / DefenseFinder profile rows and curated system calls on F17 window proteins, by role."""
import duckdb, pandas as pd
a = pd.read_parquet('data/f17_windows.parquet')
w = a[['protein_id', 'role', 'locus']].drop_duplicates('protein_id')
c = duckdb.connect('/Users/jacob/Documents/Sandbox/Sharur/Sharur/data/dpann/sharur.duckdb', read_only=True)
c.register('w', w)
x = c.sql("""select w.role, a.source, a.name, count(distinct a.protein_id) n, min(a.evalue) emin
             from w join annotations a using(protein_id) where a.source in ('txsscan','defensefinder') group by all order by role, n desc""").df()
print(x.to_string())
s = c.sql("""select w.role, sp.system_source, sp.profile_name, count(*) n from w join system_proteins sp using(protein_id) group by all order by n desc""").df()
print(s.to_string())
r = c.sql("""select w.role, pr.source, pr.name, pr.description, count(distinct pr.protein_id) n, min(pr.evalue) e from w join protein_resemblance pr using(protein_id)
             where w.role != '-' group by all order by role, n desc""").df()

print(r.groupby('role').head(4).to_string(max_colwidth=60))
q = c.sql("""select a.protein_id, a.name, a.score, a.evalue, a.start_aa, a.end_aa from w join annotations a using(protein_id)
             where w.role='ATPase504' and a.source in ('txsscan','defensefinder','pfam')""").df()
pv = q.pivot_table(index='protein_id', columns='name', values='score', aggfunc='max')
print(pv.describe().T[['count', '50%', 'min', 'max']])
print('virb4 > HerA_DUF score in', (pv['virb4'] > pv['Gao_Her_DUF__HerA_DUF']).sum(), 'of', pv[['virb4', 'Gao_Her_DUF__HerA_DUF']].dropna().shape[0])
print(q.groupby('name')[['start_aa', 'end_aa']].median())
