"""Q5: carriers vs non-carriers of the motor module within Micrarchaeota (completeness >= 80, contamination <= 10):
lineage, genome size (completeness-binned), and gene content (KO and Pfam presence) enrichment stratified by family
(CMH-style pooled odds via Mantel-Haenszel), with a permutation control (labels shuffled within family x completeness bin)."""
import pandas as pd, numpy as np, duckdb
from scipy.stats import fisher_exact, mannwhitneyu
rng = np.random.default_rng(17)
G = pd.read_csv('data/micr_genome_roles.tsv', sep='\t').set_index('bin_id')
c = duckdb.connect('/Users/jacob/Documents/Sandbox/Sharur/Sharur/data/dpann/sharur.duckdb', read_only=True)
G = G[(G.completeness >= 80) & (G.contamination <= 10)].copy()
G['cbin'] = pd.cut(G.completeness, [80, 90, 95, 101], right=False)
print('genomes', len(G), 'carriers', int(G.motor.sum()))
print(G.groupby('ord').motor.agg(['sum', 'size']).query('size >= 5').to_string())
# genome size by completeness bin
for cb, s in G.groupby('cbin', observed=True):
    a, b = s[s.motor].total_length, s[~s.motor].total_length
    if len(a) > 3 and len(b) > 3:
        print(f'size {cb}: carriers {len(a)} median {a.median()/1e6:.3f} Mb; non {len(b)} median {b.median()/1e6:.3f} Mb; MWU p={mannwhitneyu(a, b).pvalue:.2g}')
mi = G[G.ord == 'o__Micrarchaeales']
for cb, s in mi.groupby('cbin', observed=True):
    a, b = s[s.motor].total_length, s[~s.motor].total_length
    if len(a) > 3 and len(b) > 3:
        print(f'Micrarchaeales size {cb}: carriers {len(a)} {a.median()/1e6:.3f} Mb; non {len(b)} {b.median()/1e6:.3f} Mb; p={mannwhitneyu(a, b).pvalue:.2g}')
c.register('g', pd.DataFrame({'bin_id': G.index}))
F = c.sql("""select p.bin_id, a.source, a.accession, any_value(a.name) nm from proteins p join g using(bin_id) join annotations a using(protein_id)
             where a.source in ('kofam','pfam') group by all""").df()
F['feat'] = F.source + ':' + F.accession
names = F.drop_duplicates('feat').set_index('feat').nm
X = pd.crosstab(F.bin_id, F.feat).reindex(G.index).fillna(0).astype(bool)
fam = G.fam.fillna('NA')
strata = [ix for _, ix in G.groupby('fam').groups.items() if G.loc[ix, 'motor'].nunique() == 2]
print('families with both carriers and non-carriers:', len(strata), 'genomes', sum(len(i) for i in strata))
def mh(y):
    num = np.zeros(X.shape[1]); den = np.zeros(X.shape[1]); obs = np.zeros(X.shape[1]); exp = np.zeros(X.shape[1]); var = np.zeros(X.shape[1])
    for ix in strata:
        x = X.loc[ix].to_numpy(); yy = y.loc[ix].to_numpy(); n = len(ix)
        a = x[yy].sum(0); b = (~x[yy]).sum(0); cc = x[~yy].sum(0); d = (~x[~yy]).sum(0)
        num += a * d / n; den += b * cc / n
        m1 = a + cc; n1 = yy.sum(); n0 = n - n1
        obs += a; exp += m1 * n1 / n; var += m1 * (n - m1) * n1 * n0 / (n ** 2 * max(n - 1, 1))
    z = (obs - exp) / np.sqrt(np.maximum(var, 1e-9))
    return (num + 0.5) / (den + 0.5), z
y = G.motor.astype(bool)
orr, z = mh(y)
res = pd.DataFrame({'feat': X.columns, 'name': names.reindex(X.columns).values, 'OR_MH': orr, 'z': z,
                    'carr_frac': X[y].mean().values, 'non_frac': X[~y].mean().values})
# permutation control: shuffle labels within family x completeness bin, 20 rounds; count |z| >= 5 features
G['stratum'] = G.fam.fillna('NA') + '|' + G.cbin.astype(str)
nperm = []
for i in range(20):
    yp = G.groupby('stratum').motor.transform(lambda s: rng.permutation(s.to_numpy())).astype(bool)
    _, zp = mh(yp)
    nperm.append(int((np.abs(zp) >= 5).sum()))
print('features with |z| >= 5: observed', int((np.abs(z) >= 5).sum()), '; permuted (within family x completeness) median', np.median(nperm), 'max', max(nperm))
res = res.sort_values('z')
role_like = res.name.fillna('').str.contains('DUF87|Topo|KH_KRR', regex=True)
print('depleted in carriers (top 12):'); print(res.head(12).to_string(index=False))
print('enriched in carriers (top 25):'); print(res.tail(25).iloc[::-1].to_string(index=False))
res.to_csv('data/enrichment_motor.tsv', sep='\t', index=False)
# A2M partner check: PBP1C / transglycosylase / transpeptidase, any genome in Micrarchaeota
q = c.sql("""select a.source, a.accession, any_value(a.name) nm, count(distinct p.bin_id) genomes from proteins p join bins b using(bin_id) join annotations a using(protein_id)
             where b.taxonomy like '%p__Micrarchaeota%' and (a.accession in ('K05367','K05366','K05365','K03814','K03587','K05515','K08384','K01006')
             or a.name in ('Transgly','Transpeptidase','PBP_dimer','MG1','MG2','A2M','A2M_N','A2M_BRD','A2M_recep','Thiol-ester_cl','A2M_comp','PG_binding_1','SLT')) group by all""").df()
print('A2M-partner check in Micrarchaeota:'); print(q.to_string() if len(q) else 'no rows')
