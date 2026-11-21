"""Q4: any-score homologs (E <= 1e-5, coverage >= 0.5 both) of machine roles outside Micrarchaeota and their co-localization;
the Nanobdellota / Aenigmatarchaeota Top6B-near DUF87 proteins scored against the ATPase504 queries."""
import pandas as pd, numpy as np, duckdb, sys
SP = sys.argv[1]
cols = 'query target pident alnlen qstart qend qlen tstart tend tlen evalue bits'.split()
h = pd.read_csv(f'{SP}/hits.m8', sep='\t', names=cols)
h['role'] = h['query'].str.split('|').str[0].str.replace(r'MP4610_s\d', 'MP4610', regex=True)
h['qcov'] = (h.qend - h.qstart + 1) / h.qlen; h['tcov'] = (h.tend - h.tstart + 1) / h.tlen
h5 = h[(h.qcov >= 0.5) & (h.tcov >= 0.5)]
best = h5.sort_values('bits', ascending=False).drop_duplicates('target')
c = duckdb.connect('/Users/jacob/Documents/Sandbox/Sharur/Sharur/data/dpann/sharur.duckdb', read_only=True)
c.register('b', best[['target', 'role', 'bits', 'pident']])
m = c.sql("""select b.*, p.bin_id, p.contig_id, p.gene_index, p.sequence_length L, regexp_extract(bi.taxonomy,'p__[^;]+') phylum, regexp_extract(bi.taxonomy,'o__[^;]+') ord
             from b join proteins p on p.protein_id=b.target join bins bi on bi.bin_id=p.bin_id""").df()
M = ['ATPase504', 'TM277a', 'TM278b', 'TM132', 'p102', 'wHTH244', 'p49', 'A2Ml794', 'MP4610', 'pil129', 'p244b', 'p634']
o = m[(m.phylum != 'p__Micrarchaeota')]
print('any-score homologs outside Micrarchaeota, by role and phylum (best-role assignment):')
print(o[o.role.isin(M + ['Top6B', 'KH171', 'ArsR178'])].groupby(['role', 'phylum']).target.size().unstack(fill_value=0).to_string())
print('pident / bits of outside hits per machine role (median):')
print(o[o.role.isin(M)].groupby('role')[['pident', 'bits', 'L']].median())
# co-localization: per contig, distinct machine roles within a 15-gene window
oo = o[o.role.isin(M)].sort_values(['contig_id', 'gene_index'])
co = []
for cid, s in oo.groupby('contig_id'):
    gi, rr = s.gene_index.to_numpy(), s.role.to_numpy()
    for i in range(len(s)):
        win = set(rr[(gi >= gi[i] - 7) & (gi <= gi[i] + 7)])
        co.append((cid, s.phylum.iloc[0], len(win), ' '.join(sorted(win))))
co = pd.DataFrame(co, columns=['contig', 'phylum', 'nroles', 'roles']).sort_values('nroles', ascending=False).drop_duplicates('contig')
print('outside contigs by max distinct machine roles within +-7 genes:'); print(co.groupby(['phylum', 'nroles']).size().unstack(fill_value=0))
print(co.head(10).to_string())
# Q4 specific: the Top6B-near DUF87 proteins
w = pd.read_csv('data/wider_top6_duf87.tsv', sep='\t')
hh = h[h.target.isin(w.duf) & (h.role == 'ATPase504')].sort_values('bits', ascending=False).drop_duplicates('target')
print('Nanobdellota/Aenigmatarchaeota Top6B-near DUF87 proteins hit by ATPase504 queries:', len(hh), 'of', w.duf.nunique(),
      '; bits median', hh.bits.median(), 'pident median', hh.pident.median(), 'qcov median', round(hh.qcov.median(), 2))
bb = best[best.target.isin(w.duf)]
print('their best role:', bb.role.value_counts().to_dict())
