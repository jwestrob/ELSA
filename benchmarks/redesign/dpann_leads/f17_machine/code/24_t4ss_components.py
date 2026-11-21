"""Q3: T4SS / Ced component Pfams in the F17 windows (+-6 genes) and anywhere in carrier genomes (Pfam rows as stored,
gathering-threshold annotation in Sharur)."""
import duckdb, pandas as pd
c = duckdb.connect('/Users/jacob/Documents/Sandbox/Sharur/Sharur/data/dpann/sharur.duckdb', read_only=True)
P = {'PF12696': 'TraG-D_C (VirD4/TraG)', 'PF10412': 'TrwB_AAD_bind (VirD4)', 'PF02534': 'T4SS-DNA_transf (VirD4)', 'PF03432': 'Relaxase',
     'PF03389': 'MobA_MobL', 'PF08751': 'TrwC', 'PF04335': 'VirB8', 'PF05101': 'VirB3', 'PF04610': 'TrbL (VirB6)', 'PF18895': 'T4SS_pilin',
     'PF04956': 'TrbC (VirB2)', 'PF27251': 'CedA1', 'PF01464': 'SLT (VirB1-like)', 'PF19044': 'P-loop_TraG (VirB4)', 'PF12846': 'AAA_10',
     'PF01935': 'DUF87', 'PF31562': 'CedB_2nd', 'PF26593': 'TraC-like/TrsD', 'PF04610': 'TrbL', 'PF03135': 'CagE_TrbE_VirB (VirB4)',
     'PF07996': 'T4SS (VirB5)', 'PF03524': 'CagX (VirB9)', 'PF03743': 'TrbI (VirB10)', 'PF00437': 'T2SSE (VirB11-like)'}
a = pd.read_parquet('data/f17_windows.parquet')
G = pd.read_csv('data/micr_genome_roles.tsv', sep='\t')
c.register('w', a[['protein_id', 'locus', 'role']])
c.register('g', G[['bin_id', 'motor']])
acc = "('" + "','".join(P) + "')"
x = c.sql(f"""select regexp_replace(an.accession,'\\..*','') acc, count(distinct w.locus) loci, count(*) n from w join annotations an using(protein_id)
              where an.source='pfam' and regexp_replace(an.accession,'\\..*','') in {acc} group by 1""").df()
y = c.sql(f"""select regexp_replace(an.accession,'\\..*','') acc, count(distinct case when g.motor then p.bin_id end) carriers, count(distinct case when not g.motor then p.bin_id end) noncarriers
              from proteins p join g using(bin_id) join annotations an using(protein_id) where an.source='pfam' and regexp_replace(an.accession,'\\..*','') in {acc} group by 1""").df()
r = pd.DataFrame({'acc': list(P), 'name': list(P.values())}).merge(x, how='left').merge(y, how='left').fillna(0)
print(f"carriers {int(G.motor.sum())}, non-carriers {int((~G.motor).sum())} (all completeness)")
print(r.to_string(index=False))
