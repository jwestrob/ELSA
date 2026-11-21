#!/usr/bin/env python
"""AbiE lead, rung 5 at locus level and criterion A: does any other gene within 3 genes of the completion toxin carry
an AbiEi-family annotation (Pfam PF13338 / PF09407 / PF09952 / PF11459 / PF17194, any DefenseFinder AbiEi
or SanaA hit)? ILVFMAWC fraction of the slot antitoxins; label amplification (labeled genes / annotated carriers)."""
from pathlib import Path
import duckdb
import pandas as pd
W = Path('/Users/jacob/Documents/Sandbox/elsa_test/ELSA/benchmarks/redesign/dpann_leads/systems')
ab = pd.read_csv(W / 'data/lead_abie_loci.tsv', sep='\t')
slot = pd.read_parquet(W / 'data/slot.parquet')
con = duckdb.connect()
con.execute("ATTACH '/Users/jacob/Documents/Sandbox/Sharur/Sharur/data/dpann/sharur.duckdb' AS d (READ_ONLY)")
ann = con.execute("""SELECT DISTINCT p.contig_id, p.gene_index, p.protein_id, a.source, a.accession FROM d.annotations a
  JOIN d.proteins p USING (protein_id) WHERE (a.source='pfam' AND a.accession IN
  ('PF13338','PF09407','PF09952','PF11459','PF17194')) OR (a.source='defensefinder' AND (a.accession LIKE 'AbiEii__AbiEi_%'
  OR a.accession LIKE 'SanaTA__SanaA%'))""").df()
pos = con.execute("SELECT protein_id, contig_id, gene_index, sequence FROM d.proteins").df().set_index('protein_id')
bycontig = ann.groupby('contig_id')
n_other = 0
for r in ab.itertuples():
    tox = [t for t in r.cluster_proteins.split(';') if t in pos.index]
    found = False
    for t in tox:
        c, gi = pos.loc[t, 'contig_id'], pos.loc[t, 'gene_index']
        if c in bycontig.groups:
            x = bycontig.get_group(c)
            if ((x.gene_index - gi).abs().between(1, 3) & (x.protein_id != r.protein_id)).any():
                found = True
    n_other += found
hyd = [sum(pos.loc[p, 'sequence'].count(a) for a in 'ILVFMAWC') / len(pos.loc[p, 'sequence']) for p in ab.protein_id]
carriers = con.execute("SELECT accession, count(DISTINCT protein_id) FROM d.annotations WHERE source='pfam' AND "
                       "accession IN ('PF13338','PF09407','PF09952') GROUP BY 1").fetchall()
lab = slot[slot.label.isin(['PF13338', 'PF09407', 'PF09952'])].label.value_counts().to_dict()
print(f'completion loci with another AbiEi-family gene (Pfam / DefenseFinder) within 3 genes of the toxin: '
      f'{n_other}/{len(ab)}')
print(f'ILVFMAWC fraction of slot antitoxins: median {pd.Series(hyd).median():.3f} '
      f'(IQR {pd.Series(hyd).quantile(.25):.3f}-{pd.Series(hyd).quantile(.75):.3f})')
print('label amplification (pass-table labeled genes / DPANN Pfam carriers):',
      {a: f'{lab.get(a, 0)}/{n} = {lab.get(a, 0) / n:.2f}' for a, n in carriers})
print('own KO among slot antitoxins:', slot[slot.protein_id.isin(ab.protein_id)].own_ko.replace('', None).notna().sum())
