#!/usr/bin/env python
"""Detail for the pattern (c) candidates and the (b) MazE / REase candidates: tier, list type, length,
ILVFMAWC fraction, species, order, and for HEPN-in-Type-I-RM the neighboring nucleotidyltransferase and the
RX4-6H motif. Writes data/pattern_leads_genes.tsv and prints a summary."""
import re
from pathlib import Path

import duckdb
import pandas as pd

W = Path('/Users/jacob/Documents/Sandbox/elsa_test/ELSA/benchmarks/redesign/dpann_leads/systems')
D = W / 'data'
DB = '/Users/jacob/Documents/Sandbox/Sharur/Sharur/data/dpann/sharur.duckdb'
win = pd.read_csv(D / 'defense_window_slot_genes.tsv', sep='\t')
ev = pd.read_csv(D / 'uncalled_completion_events.tsv', sep='\t')
slot = pd.read_parquet(D / 'slot.parquet')
con = duckdb.connect()
con.execute(f"ATTACH '{DB}' AS d (READ_ONLY)")
bins = con.execute("SELECT bin_id, taxonomy, completeness FROM d.bins").df()
bins['species'] = bins.taxonomy.str.extract(r'(s__[^;]*)')[0]
bins['order'] = bins.taxonomy.str.extract(r'(o__[^;]*)')[0]
bins['phylum'] = bins.taxonomy.str.extract(r'(p__[^;]*)')[0]

cands = {
    'HEPN_in_RM_I': win[(win.model == 'RM_Type_I') & (win.label == 'PF05168')],
    'PIN12_in_RM_I': win[(win.model == 'RM_Type_I') & (win.label == 'PF16289')],
    'DUF3800_in_RM_I': win[(win.model == 'RM_Type_I') & (win.label == 'PF12686')],
    'ZnRibbon_in_RM_I': win[(win.model == 'RM_Type_I') & (win.label == 'PF13240')],
    'PDDEXK5_by_RM_I': win[(win.model == 'RM_Type_I') & (win.label == 'PF11645')],
    'PDDEXK1_by_RM_III': win[(win.model == 'RM_Type_III') & (win.label == 'PF12705')],
    'MazE_by_CapRel': win[(win.model == 'CapRel') & (win.label == 'PF04014')],
    'Ceres_K08982': win[(win.model == 'Ceres') & (win.label == 'K08982')],
    'Ceres_DUF5676': win[(win.model == 'Ceres') & (win.label == 'PF18926')],
    'PepSY_by_dCTPdeaminase': win[(win.model == 'dCTPdeaminase') & (win.label == 'PF03413')],
    'MazE_completes_MazF': ev[(ev.model == 'MazEF') & (ev.label == 'PF04014') & ev.completes],
    'REase_RHinP1I': ev[(ev.model == 'RM_Type_II') & (ev.label == 'PF11463')],
    'REase_ThaI': ev[(ev.model == 'RM_Type_II') & (ev.label == 'PF15514')],
    'REase_HaeIII': ev[(ev.model == 'RM_Type_II') & (ev.label == 'PF09556')],
}
allp = sorted(set(p for x in cands.values() for p in x.protein_id))
con.execute("CREATE TEMP TABLE ids(protein_id VARCHAR)")
con.executemany("INSERT INTO ids VALUES (?)", [(p,) for p in allp])
seq = con.execute("SELECT p.protein_id, p.sequence, p.contig_id, p.gene_index, p.strand FROM d.proteins p "
                  "JOIN ids USING (protein_id)").df().set_index('protein_id')
hyd = lambda s: sum(s.count(c) for c in 'ILVFMAWC') / max(len(s), 1)
rows = []
for name, x in cands.items():
    x = x.drop_duplicates('protein_id').merge(
        slot[['protein_id', 'bin_id', 'tier', 'list_type', 'length', 'vfrac', 'votes_for', 'label_name']]
        .rename(columns={'tier': 'tier_', 'list_type': 'lt_', 'length': 'len_', 'vfrac': 'vf_',
                         'votes_for': 'votes_', 'label_name': 'ln_'}), on='protein_id')
    x = x.merge(bins, left_on='bin_id', right_on='bin_id', how='left')
    for r in x.itertuples():
        s = seq.loc[r.protein_id, 'sequence'].rstrip('*')
        rows.append(dict(cand=name, protein_id=r.protein_id, genome=r.bin_id, species=r.species, order=r.order,
                         phylum=r.phylum, tier=r.tier_, list_type=r.lt_, length=r.len_, vfrac=r.vf_,
                         votes=r.votes_, label=r.ln_, hyd=round(hyd(s), 3),
                         rx4_6h=bool(re.search(r'R.{4,6}H', s)),
                         contig=seq.loc[r.protein_id, 'contig_id'], gene_index=int(seq.loc[r.protein_id, 'gene_index']),
                         strand=seq.loc[r.protein_id, 'strand']))
g = pd.DataFrame(rows)

# HEPN: neighbor nucleotidyltransferase (Pfam NTP_transf_2 / MNT-like / KO) within 2 genes
hep = g[g.cand == 'HEPN_in_RM_I']
nt = con.execute("""SELECT p.contig_id, p.gene_index, p.protein_id, string_agg(DISTINCT a.name, ',') pf
  FROM d.proteins p JOIN d.annotations a USING (protein_id) WHERE a.source='pfam'
  AND (a.name LIKE 'NTP_transf%' OR a.name IN ('DUF86','MNT','Polbeta','NTase_sub_bind','DUF4037'))
  GROUP BY 1, 2, 3""").df()
ntd = nt.groupby('contig_id').apply(lambda x: list(zip(x.gene_index, x.pf))).to_dict()
g['ntase_within2'] = ''
for i, r in hep.iterrows():
    near = [pf for gi, pf in ntd.get(r.contig, []) if abs(gi - r.gene_index) <= 2 and gi != r.gene_index]
    g.loc[i, 'ntase_within2'] = ';'.join(near)
g.to_csv(D / 'pattern_leads_genes.tsv', sep='\t', index=False)
summ = g.groupby('cand').agg(genes=('protein_id', 'nunique'), genomes=('genome', 'nunique'),
                             species=('species', 'nunique'), orders=('order', 'nunique'),
                             T1=('tier', lambda s: (s == 'T1').sum()),
                             emb_only=('list_type', lambda s: (s == 'embedding_only').sum()),
                             med_len=('length', 'median'), med_hyd=('hyd', 'median'),
                             vfrac=('vfrac', 'median'), rx46h=('rx4_6h', 'mean'))
pd.set_option('display.width', 220)
print(summ.to_string())
print('HEPN genes with an NTase within 2 genes:', int((g.loc[hep.index, 'ntase_within2'] != '').sum()),
      'of', len(hep))
print(g[g.cand == 'HEPN_in_RM_I'][['protein_id', 'species', 'tier', 'length', 'rx4_6h', 'ntase_within2']]
      .to_string())
