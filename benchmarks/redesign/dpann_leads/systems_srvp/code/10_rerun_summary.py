#!/usr/bin/env python
"""Summarize the caller-profile rerun (08_rerun.py): per test event, the best hit among the profiles of the proposed
component; per gene set, rates of (any hit at E <= 10, GA pass, GA + coverage + P <= 1e-3). Controls are scored against
the profiles of the label they are matched to (union over that label's events); positives against their own component.
Outputs data/rerun/test_event_scores.tsv, control_scores.tsv, rerun_summary.txt."""
from pathlib import Path

import pandas as pd

RD = Path('/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111')
D = RD / 'dpann_leads/systems_srvp/data'
RR = D / 'rerun'
h = pd.read_csv(RR / 'hits_annotated.tsv', sep='\t')
ev = pd.read_csv(RR / 'test_events.tsv', sep='\t')
sets = pd.read_csv(RR / 'gene_sets.tsv', sep='\t')
slot = pd.read_parquet(D / 'slot.parquet')
cm = pd.read_parquet(D / 'call_members.parquet')
mg = pd.read_csv(D / 'model_genes.tsv', sep='\t')
info = pd.read_csv(D / 'model_info.tsv', sep='\t').set_index('model_key')
comp_prof = {(r.model_key, r.component): r.profiles.split(';') for r in mg.itertuples()}
h = h.sort_values(['pass_all', 'pass_ga', 'bitscore', 'dom_bitscore'], ascending=False)
H = {k: g for k, g in h.groupby('gid')}


def best(gid, profs):
    x = H.get(gid)
    x = None if x is None else x[x.profile.isin(profs)]
    if x is None or not len(x):
        return dict(any_hit=False, pass_ga=False, pass_all=False, best_profile='', bitscore=None, dom_bitscore=None,
                    GA1=None, cov=None, P=None)
    r = x.iloc[0]
    return dict(any_hit=True, pass_ga=bool(r.pass_ga), pass_all=bool(r.pass_all), best_profile=r.profile,
                bitscore=r.bitscore, dom_bitscore=r.dom_bitscore, GA1=r.GA1, cov=round(r['cov'], 3), P=r.P)


te = pd.DataFrame([dict(gid=r.gid, model_key=r.model_key, component=r.component, label=r.label, screen=r.screen,
                        **best(r.gid, r.profiles.split(';'))) for r in ev.itertuples()])
te = te.merge(slot[['gid', 'length', 'tier', 'label_name', 'support']], on='gid', how='left')
te['cls'] = te.model_key.map(info.cls)
te.to_csv(RR / 'test_event_scores.tsv', sep='\t', index=False)
label_profs = ev.groupby('label').profiles.apply(lambda s: sorted(set(';'.join(s).split(';')))).to_dict()
crow = [dict(gid=r.gid, set=r.set, test_label=r.test_label, **best(r.gid, label_profs[r.test_label]))
        for r in sets[sets.set.isin(['neg_same', 'neg_other'])].itertuples()]
pos = sets[sets.set == 'pos'].merge(cm[['gid', 'model_key', 'component']].drop_duplicates('gid'), on='gid')
crow += [dict(gid=r.gid, set='pos', test_label=f'{r.model_key}:{r.component}',
              **best(r.gid, comp_prof[(r.model_key, r.component)])) for r in pos.itertuples()]
ct = pd.DataFrame(crow)
ct.to_csv(RR / 'control_scores.tsv', sep='\t', index=False)

L = []
P = L.append
for nm, x in [('all test genes', te), ('small-system test genes', te[te.cls.str.endswith('small')])]:
    g = x.groupby('gid').agg(any_hit=('any_hit', 'max'), pass_ga=('pass_ga', 'max'), pass_all=('pass_all', 'max'))
    P(f'{nm} {len(g)}: any hit {int(g.any_hit.sum())} ({g.any_hit.mean():.3f}); GA {int(g.pass_ga.sum())} '
      f'({g.pass_ga.mean():.3f}); GA + coverage + P {int(g.pass_all.sum())} ({g.pass_all.mean():.3f})')
for s, x in ct.groupby('set'):
    P(f'  {s}: n {len(x)}; any {int(x.any_hit.sum())} ({x.any_hit.mean():.3f}); GA {int(x.pass_ga.sum())} '
      f'({x.pass_ga.mean():.3f}); all {int(x.pass_all.sum())} ({x.pass_all.mean():.3f})')
P('')
P('per (model, component, label) for small-system test events; controls matched by label:')
t = te[te.cls.str.endswith('small')].groupby(['model_key', 'component', 'label', 'label_name']).agg(
    n=('gid', 'nunique'), any_hit=('any_hit', 'sum'), ga=('pass_ga', 'sum'), allp=('pass_all', 'sum'),
    med_bits=('bitscore', 'median'), GA1=('GA1', 'median'), med_len=('length', 'median')).reset_index()
cs = ct[ct.set == 'neg_same'].groupby('test_label').agg(ns_n=('gid', 'size'), ns_any=('any_hit', 'sum'),
                                                        ns_ga=('pass_ga', 'sum'))
co = ct[ct.set == 'neg_other'].groupby('test_label').agg(no_n=('gid', 'size'), no_any=('any_hit', 'sum'),
                                                         no_ga=('pass_ga', 'sum'))
t = t.merge(cs, left_on='label', right_index=True, how='left').merge(co, left_on='label', right_index=True, how='left')
pd.set_option('display.width', 250)
P(t.sort_values('n', ascending=False).head(50).round(2).to_string(index=False))
t.to_csv(RR / 'per_component_label.tsv', sep='\t', index=False)
P('')
pp = ct[ct.set == 'pos'].groupby('test_label').agg(n=('gid', 'size'), ga=('pass_ga', 'mean'), allp=('pass_all', 'mean'))
P('positive controls (called members rescored), lowest GA rates: ' +
  pp[pp.n >= 5].sort_values('ga').head(10).round(3).to_string())
(RR / 'rerun_summary.txt').write_text('\n'.join(L) + '\n')
print('\n'.join(L))
