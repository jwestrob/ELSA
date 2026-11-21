#!/usr/bin/env python
"""Summarize the Aksha rerun: per completion event, the best hit of the caller's profiles for the missing component;
per gene set, the rates of (any hit at E <= 10, full GA pass, full DefenseFinder selection pass).
Controls are scored against the profiles of the label they are matched to (union over that label's events)."""
from pathlib import Path

import pandas as pd

W = Path('/Users/jacob/Documents/Sandbox/elsa_test/ELSA/benchmarks/redesign/dpann_leads/systems')
RR = W / 'data' / 'rerun'
hits = pd.read_csv(RR / 'hits_annotated.tsv', sep='\t')
ev = pd.read_csv(RR / 'test_events.tsv', sep='\t')
sets = pd.read_csv(RR / 'gene_sets.tsv', sep='\t')
slot = pd.read_parquet(W / 'data' / 'slot.parquet')
sysdf = pd.read_parquet(W / 'data' / 'df_systems.parquet')
mg = pd.read_csv(W / 'data' / 'df_model_genes.tsv', sep='\t')
comp_prof = {(r.model, r.component): r.profiles.split(';') for r in mg.itertuples()}

h = hits.rename(columns={'sequence_id': 'protein_id'})
h['profile'] = h.profile_file
# best domain row per (gene, profile): prefer passing rows, then score
h = h.sort_values(['pass_all', 'pass_ga', 'bitscore', 'dom_bitscore'], ascending=False)


def best(pid, profs):
    x = h[(h.protein_id == pid) & (h.profile.isin(profs))]
    if not len(x):
        return dict(any_hit=False, pass_ga=False, pass_all=False, best_profile='', bitscore=None,
                    dom_bitscore=None, GA1=None, cov=None, i_evalue_genome=None)
    r = x.iloc[0]
    return dict(any_hit=True, pass_ga=bool(r.pass_ga), pass_all=bool(r.pass_all), best_profile=r.profile,
                bitscore=r.bitscore, dom_bitscore=r.dom_bitscore, GA1=r.GA1, cov=round(r['cov'], 3),
                i_evalue_genome=r.i_evalue_genome)


rows = []
for r in ev.itertuples():
    rows.append(dict(protein_id=r.protein_id, model=r.model, component=r.component, label=r.label, screen=r.screen,
                     **best(r.protein_id, r.profiles.split(';'))))
te = pd.DataFrame(rows).merge(slot[['protein_id', 'length', 'tier', 'list_type', 'label_name', 'vfrac']],
                              on='protein_id', how='left')
te.to_csv(RR / 'test_event_scores.tsv', sep='\t', index=False)

label_profs = ev.groupby('label').profiles.apply(lambda s: sorted(set(';'.join(s).split(';')))).to_dict()
crow = []
for r in sets[sets.set.isin(['neg_same', 'neg_other'])].itertuples():
    crow.append(dict(protein_id=r.protein_id, set=r.set, test_label=r.test_label,
                     **best(r.protein_id, label_profs[r.test_label])))
pos = sets[sets.set == 'pos'].merge(sysdf[['protein_id', 'model', 'component']], on='protein_id')
for r in pos.itertuples():
    crow.append(dict(protein_id=r.protein_id, set='pos', test_label=f'{r.model}:{r.component}',
                     **best(r.protein_id, comp_prof[(r.model, r.component)])))
ct = pd.DataFrame(crow)
ct.to_csv(RR / 'control_scores.tsv', sep='\t', index=False)

lines = []
P = lines.append
g = te.groupby('protein_id').agg(any_hit=('any_hit', 'max'), pass_ga=('pass_ga', 'max'), pass_all=('pass_all', 'max'))
P(f'test genes (unique) {len(g)}: any profile hit (E<=10, Z=search) {g.any_hit.sum()} ({g.any_hit.mean():.3f}); '
  f'full GA {g.pass_ga.sum()} ({g.pass_ga.mean():.3f}); GA + i-Evalue + coverage {g.pass_all.sum()} '
  f'({g.pass_all.mean():.3f})')
for s, x in ct.groupby('set'):
    P(f'  {s}: n {len(x)}; any {x.any_hit.mean():.3f}; GA {x.pass_ga.mean():.3f}; all {x.pass_all.mean():.3f}')
P('')
P('per (model, component, label) for test events; controls matched by label:')
t = te.groupby(['model', 'component', 'label', 'label_name']).agg(
    n=('protein_id', 'nunique'), any_hit=('any_hit', 'mean'), ga=('pass_ga', 'mean'), all=('pass_all', 'mean'),
    med_bits=('bitscore', 'median'), GA1=('GA1', 'median'), med_len=('length', 'median')).reset_index()
cs = ct[ct.set == 'neg_same'].groupby('test_label').agg(ns_n=('protein_id', 'size'), ns_any=('any_hit', 'mean'),
                                                        ns_ga=('pass_ga', 'mean'))
co = ct[ct.set == 'neg_other'].groupby('test_label').agg(no_n=('protein_id', 'size'), no_any=('any_hit', 'mean'),
                                                         no_ga=('pass_ga', 'mean'))
t = t.merge(cs, left_on='label', right_index=True, how='left').merge(co, left_on='label', right_index=True,
                                                                     how='left')
pd.set_option('display.width', 250)
P(t.sort_values('n', ascending=False).head(40).round(3).to_string())
P('')
pp = ct[ct.set == 'pos'].groupby('test_label').agg(n=('protein_id', 'size'), ga=('pass_ga', 'mean'),
                                                   all=('pass_all', 'mean'))
P('positive controls (called members rescored): ' + pp.sort_values('n', ascending=False).head(15).round(3).to_string())
(RR / 'rerun_summary.txt').write_text('\n'.join(lines) + '\n')
print('\n'.join(lines))
