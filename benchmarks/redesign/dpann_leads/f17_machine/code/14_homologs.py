"""Genome-wide homologs of each role (MMseqs2 role queries vs all DPANN proteins): ortholog threshold per role from the
score distribution of the atlas members themselves (5th percentile of members' best bitscore, coverage >= 0.5 both)."""
import pandas as pd, numpy as np, duckdb, sys
SP = sys.argv[1]
cols = 'query target pident alnlen qstart qend qlen tstart tend tlen evalue bits'.split()
h = pd.read_csv(f'{SP}/hits.m8', sep='\t', names=cols)
h['role'] = h['query'].str.split('|').str[0]
h['role'] = h.role.str.replace(r'MP4610_s\d', 'MP4610', regex=True)
h['qcov'] = (h.qend - h.qstart + 1) / h.qlen; h['tcov'] = (h.tend - h.tstart + 1) / h.tlen
h = h[(h.qcov >= 0.5) & (h.tcov >= 0.5)]
best = h.sort_values('bits', ascending=False).drop_duplicates('target')  # each target assigned to its best role
a = pd.read_parquet('data/f17_windows.parquet')
mem = a[a.role != '-'][['protein_id', 'role']].drop_duplicates('protein_id').rename(columns={'protein_id': 'target', 'role': 'mrole'})
best = best.merge(mem, on='target', how='left')
thr = {}
for r, s in best.groupby('role'):
    m = s[s.mrole == r]
    thr[r] = float(np.percentile(m.bits, 5)) if len(m) >= 10 else np.nan
best['thr'] = best.role.map(thr)
orth = best[best.bits >= best.thr].copy()
c = duckdb.connect('/Users/jacob/Documents/Sandbox/Sharur/Sharur/data/dpann/sharur.duckdb', read_only=True)
c.register('o', orth[['target']])
meta = c.sql("""select p.protein_id target, p.bin_id, p.contig_id, p.gene_index, p.strand, b.completeness, regexp_extract(b.taxonomy,'p__[^;]+') phylum,
                regexp_extract(b.taxonomy,'o__[^;]+') ord from proteins p join o on p.protein_id=o.target join bins b on b.bin_id=p.bin_id""").df()
orth = orth.merge(meta, on='target')
orth['in_atlas'] = orth.mrole.notna()
orth.to_parquet('data/role_orthologs.parquet')
summ = orth.groupby('role').agg(thr=('thr', 'first'), proteins=('target', 'size'), in_atlas=('in_atlas', 'sum'), genomes=('bin_id', 'nunique'),
        micr=('phylum', lambda x: (x == 'p__Micrarchaeota').sum()))
summ['other_phyla'] = orth[orth.phylum != 'p__Micrarchaeota'].groupby('role').phylum.agg(lambda x: x.value_counts().head(3).to_dict())
print(summ.to_string())
