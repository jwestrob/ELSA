#!/usr/bin/env python
"""Global numbers for the paper: called systems examined, systems with a slot gene inside / adjacent, systems that
gain a model component, uncalled loci that become complete; split by system type, tier and list type (rung 7).
System type: DefenseFinder category RM vs other DefenseFinder; size class small (<= 4 mandatory+accessory
components) vs large (> 4, many-subunit assemblies the PI excludes)."""
from pathlib import Path

import pandas as pd

W = Path('/Users/jacob/Documents/Sandbox/elsa_test/ELSA/benchmarks/redesign/dpann_leads/systems')
D = W / 'data'
mg = pd.read_csv(D / 'df_model_genes.tsv', sep='\t')
cf = pd.read_csv(D / 'component_families.tsv', sep='\t').fillna({'families': ''})
sy = pd.read_csv(D / 'defense_systems_summary.tsv', sep='\t')
win = pd.read_csv(D / 'defense_window_slot_genes.tsv', sep='\t').fillna({'gains': '', 'same_family_as_member': ''})
ev = pd.read_csv(D / 'uncalled_completion_events.tsv', sep='\t')
sc = pd.read_csv(D / 'rerun/test_event_scores.tsv', sep='\t')
ctl = pd.read_csv(D / 'rerun/control_scores.tsv', sep='\t')
sysdf = pd.read_parquet(D / 'df_systems.parquet')
slot = pd.read_parquet(D / 'slot.parquet')
info = mg.groupby('model').agg(category=('category', 'first'), n_comp=('n_mand_acc', 'first'))
info['stype'] = ['RM' if c.startswith('RM') else 'other_defense' for c in info.category]
info['size'] = ['small' if n <= 4 else 'large' for n in info.n_comp]
cls = lambda m: f'{info.loc[m, "stype"]}/{info.loc[m, "size"]}'
sy['cls'] = sy.model.map(cls)
win['cls'] = win.model.map(cls)
ev['cls'] = ev.model.map(cls)

out = []
P = out.append
P('== DefenseFinder 2.0.2 calls (Sharur colocation engine)')
t = sy.groupby('cls').agg(systems=('system_id', 'size'), lacking_component=('n_lacking', lambda x: int((x > 0).sum())),
                          slot_inside_adjacent=('has_slot_inside_adj', 'sum'), slot_in_window=('has_slot_window', 'sum'),
                          gain_window=('gains', 'sum'), gain_adjacent=('gains_adj', 'sum'))
t.loc['ALL'] = t.sum()
P(t.to_string())
wa = win[win.dist <= 1].drop_duplicates('protein_id')
P(f'slot genes inside/adjacent to a call: {len(wa)}; tier {wa.tier.value_counts().to_dict()}; '
  f'list {wa.list_type.value_counts().to_dict()}')
g = win[win.gains != ''].drop_duplicates('protein_id')
P(f'gain genes (window): {len(g)}; tier {g.tier.value_counts().to_dict()}; list {g.list_type.value_counts().to_dict()}')

# label agreement for slot genes that are members of DefenseFinder calls
prof2 = {(r.model, r.component): set(r.families.split(';')) - {''} for r in cf.itertuples()}
m = sysdf.merge(slot[['protein_id', 'label']], on='protein_id')
m['mappable'] = [bool(prof2.get((a, b))) for a, b in zip(m.model, m.component)]
m['agree'] = [lab in prof2.get((a, b), set()) for lab, a, b in zip(m.label, m.model, m.component)]
mm = m[m.mappable]
P(f'slot genes that are DefenseFinder members: {m.protein_id.nunique()}; mappable {len(mm)}; slot label in the '
  f'component family set {int(mm.agree.sum())} ({mm.agree.mean():.3f})')

P('')
P('== Uncalled partial loci completed by a slot gene (screen b)')
c = ev[ev.completes]
t2 = c.groupby('cls').agg(loci=('cluster_id', 'nunique'), slot_genes=('protein_id', 'nunique'),
                          models=('model', 'nunique'))
P(t2.to_string())
P(f'unique loci (cluster ids) complete: {c.cluster_id.nunique()}; unique slot genes: {c.protein_id.nunique()}')
P('by model (loci complete): ' + str(c.groupby('model').cluster_id.nunique().sort_values(ascending=False)
                                     .head(15).to_dict()))
cu = c.drop_duplicates('protein_id')
P(f'completing genes by tier {cu.tier.value_counts().to_dict()}')
P(f'completing genes by list type {cu.list_type.value_counts().to_dict()}')
cs = cu[cu.cls.str.endswith('small')]
P(f'small systems only: genes {len(cs)}; tier {cs.tier.value_counts().to_dict()}; '
  f'list {cs.list_type.value_counts().to_dict()}')

P('')
P('== Caller profiles rerun on completing / gaining genes (Aksha, picked DefenseFinder profiles, E <= 10)')
g2 = sc.groupby('protein_id').agg(any_hit=('any_hit', 'max'), ga=('pass_ga', 'max'), allp=('pass_all', 'max'))
P(f'test genes {len(g2)}: sub-threshold hit {int(g2.any_hit.sum())} ({g2.any_hit.mean():.3f}); GA pass '
  f'{int(g2.ga.sum())}; full selection pass {int(g2.allp.sum())}')
for s, x in ctl.groupby('set'):
    P(f'  control {s}: n {len(x)}; any hit {x.any_hit.mean():.3f}; GA {x.pass_ga.mean():.3f}')
(D / 'global_numbers.txt').write_text('\n'.join(out) + '\n')
print('\n'.join(out))
