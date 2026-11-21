"""Q1/Q5: per-genome presence of each role (ortholog sets) across the 1,146 Micrarchaeota genomes; co-occurrence of the
motor module with the Top6B-KH pair; adjacency of machine to Top6B."""
import pandas as pd, numpy as np, duckdb
o = pd.read_parquet('data/role_orthologs.parquet')
c = duckdb.connect('/Users/jacob/Documents/Sandbox/Sharur/Sharur/data/dpann/sharur.duckdb', read_only=True)
B = c.sql("""select bin_id, completeness, contamination, total_length, regexp_extract(taxonomy,'o__[^;]+') ord, regexp_extract(taxonomy,'f__[^;]+') fam
             from bins where taxonomy like '%p__Micrarchaeota%'""").df()
P = o.pivot_table(index='bin_id', columns='role', values='target', aggfunc='count').reindex(B.bin_id).fillna(0).astype(int)
G = B.set_index('bin_id').join(P > 0)
# Top6B-KH adjacency from the DB-wide context table (any top6B)
t6 = pd.read_csv('data/top6b_context.tsv', sep='\t')
G['top6b_any'] = G.index.isin(t6.bin_id)
G['top6b_kh_adj'] = G.index.isin(t6[t6.kh2 == 1].bin_id)
motor = ['ATPase504', 'TM277a', 'TM132', 'p102', 'wHTH244']
G['motor'] = G[motor].sum(axis=1) >= 3
G['A2Ml'] = G['A2Ml794']
G.to_csv('data/micr_genome_roles.tsv', sep='\t')
hc = G[G.completeness >= 80]
print('Micrarchaeota genomes', len(G), '; completeness >= 80:', len(hc))
for nm, d in [('all', G), ('compl>=80', hc)]:
    print(f'-- {nm}: motor(>=3 of 5) {d.motor.sum()}; A2Ml794 {d.A2Ml.sum()}; Top6B any {d.top6b_any.sum()}; Top6B-KH adjacent {d.top6b_kh_adj.sum()}')
    print(pd.crosstab(d.motor, d.top6b_kh_adj, rownames=['motor'], colnames=['top6B-KH adj']))
    print(pd.crosstab(d.motor, d.A2Ml, rownames=['motor'], colnames=['A2Ml794']))
# pairwise co-occurrence among roles (compl >= 80): P(role2 | role1)
R = ['ATPase504', 'TM277a', 'TM278b', 'TM132', 'p102', 'wHTH244', 'p49', 'A2Ml794', 'MP4610', 'pil129', 'p244b', 'Top6B', 'KH171', 'ArsR178']
M = pd.DataFrame({r2: [((hc[r1]) & (hc[r2])).sum() / max(hc[r1].sum(), 1) for r1 in R] for r2 in R}, index=R).round(2)
print('P(column present | row present), completeness >= 80 (role orthologs at the family threshold)')
print(M.to_string())
x = hc.groupby('ord')[['motor', 'A2Ml', 'MP4610', 'pil129', 'top6b_kh_adj']].sum(); x['n'] = hc.groupby('ord').size(); print(x.to_string())
