#!/usr/bin/env python
"""Shared context tables for systems_srvp (biotite login node, serial).

Outputs (data/):
  slot.parquet              69,701 position-labeled srvp genes: pass columns + tier + contig_code
  profile_family.parquet    caller profile -> Pfam family co-annotation: on srvp proteins carrying a selected hit to
                            profile p (03_hits.py search or a called-system member), Pfam domains (Aksha --cut_ga,
                            annot_gpu/srvp/PFAM.parquet) overlapping the profile hit by >= 50% of the shorter span
  component_families.tsv    model component -> Pfam families: union over its profiles of families with >= 3 overlapping
                            proteins and >= 20% of the profile's proteins (the DPANN systems rule)
  systems.parquet           called systems (callers' best_solution): sys_key, run, caller, model_key, genome,
                            contig_code, core span lo / hi (members with locus_num > 0), members, satisfied components
"""
from pathlib import Path

import pandas as pd

RD = Path('/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111')
W = RD / 'dpann_leads/systems_srvp'
D = W / 'data'
P = RD / 'dpann_leads/pass_port/out/srvp'
genes = pd.read_parquet(RD / 'ws_srvp_bins/package/universe/genes.parquet',
                        columns=['gid', 'genome', 'contig_code', 'position_index', 'strand', 'length'])
gi = genes.set_index('gid')

# ------------------------------------------------------------------ slot table
s = pd.read_parquet(P / 'srvp_slot_rescue_pass.parquet')
t = pd.read_parquet(P / 'tiers.parquet')[['protein_id', 'tier', 'tier_family', 'proposed_label_clans']]
s = s.merge(t, on='protein_id', how='left')
s = s.rename(columns={'proposed_label_positional_homology': 'label', 'proposed_label_name': 'label_name'})
s['contig_code'] = s.gid.map(gi.contig_code)
assert (s.position_index.values == s.gid.map(gi.position_index).values).all()
s.to_parquet(D / 'slot.parquet', index=False)
print('slot', len(s), s.tier.value_counts().to_dict())

# ------------------------------------------------------------------ profile -> Pfam family map
mg = pd.read_csv(D / 'model_genes.tsv', sep='\t')
hits = pd.read_parquet(D / 'hits.parquet')
cm = pd.read_parquet(D / 'call_members.parquet')
cm['caller'] = cm.run.map({'DefenseFinder': 'DefenseFinder', 'DefenseFinder_RM': 'DefenseFinder', 'TXSScan': 'TXSScan'})
spans = pd.concat([hits[['gid', 'caller', 'profile', 'ali_from', 'ali_to']].rename(columns={'ali_from': 's1', 'ali_to': 'e1'}),
                   cm[['gid', 'caller', 'profile', 'hit_begin_match', 'hit_end_match']].rename(
                       columns={'hit_begin_match': 's1', 'hit_end_match': 'e1'})], ignore_index=True)
spans = spans.groupby(['gid', 'caller', 'profile']).agg(s1=('s1', 'min'), e1=('e1', 'max')).reset_index()
want = set(spans.gid)
pf = pd.read_parquet(RD / 'annot_gpu/srvp/PFAM.parquet', columns=['gid', 'accession', 'name', 'ali_from', 'ali_to'],
                     filters=[('gid', 'in', sorted(want))])
pf['family'] = pf.accession.str.split('.').str[0]
pf = pf.groupby(['gid', 'family']).agg(s2=('ali_from', 'min'), e2=('ali_to', 'max'), fname=('name', 'first')).reset_index()
x = spans.merge(pf, on='gid')
ov = (x[['e1', 'e2']].min(axis=1) - x[['s1', 's2']].max(axis=1) + 1)
short = pd.concat([x.e1 - x.s1 + 1, x.e2 - x.s2 + 1], axis=1).min(axis=1)
x['overlap'] = ov >= 0.5 * short
fm = x[x.overlap].groupby(['caller', 'profile', 'family']).agg(n_overlap=('gid', 'nunique'), fname=('fname', 'first'))
fm = fm.reset_index().merge(spans.groupby(['caller', 'profile']).gid.nunique().rename('n_profile').reset_index(),
                            on=['caller', 'profile'])
fm['frac'] = fm.n_overlap / fm.n_profile
fm.to_parquet(D / 'profile_family.parquet', index=False)
ok = fm[(fm.n_overlap >= 3) & (fm.frac >= 0.20)]
p2f = ok.groupby(['caller', 'profile']).family.apply(set).to_dict()
print('profiles with srvp proteins', spans.groupby(['caller', 'profile']).ngroups, 'with >= 1 mapped family',
      len(p2f), 'mapped pairs', len(ok))

comp = mg[mg.presence.isin(['mandatory', 'accessory'])].copy()
comp['families'] = [';'.join(sorted(set().union(*[p2f.get((c, p), set()) for p in ps.split(';')])))
                    for c, ps in zip(comp.caller, comp.profiles)]
comp['n_fam'] = comp.families.map(lambda f: len([z for z in f.split(';') if z]))
comp.to_csv(D / 'component_families.tsv', sep='\t', index=False)
print('components', len(comp), 'with families', int((comp.n_fam > 0).sum()))

# ------------------------------------------------------------------ called systems
cm['position_index'] = cm.gid.map(gi.position_index)
cm['contig_code'] = cm.gid.map(gi.contig_code)
cm['sys_key'] = cm.run + ':' + cm.genome + ':' + cm.sys_id
rows = []
for k, g in cm.groupby('sys_key'):
    core = g[g.locus_num > 0]
    core = core if len(core) else g
    rows.append(dict(sys_key=k, run=g.run.iloc[0], caller=g.caller.iloc[0], model_key=g.model_key.iloc[0],
                     genome=g.genome.iloc[0], contig_code=int(core.contig_code.mode().iloc[0]),
                     lo=int(core[core.contig_code == core.contig_code.mode().iloc[0]].position_index.min()),
                     hi=int(core[core.contig_code == core.contig_code.mode().iloc[0]].position_index.max()),
                     n_members=len(g), members=sorted(g.gid.tolist()), comps=sorted(set(g.component)),
                     n_contigs=int(g.contig_code.nunique())))
sy = pd.DataFrame(rows)
sy.to_parquet(D / 'systems.parquet', index=False)
print('systems', len(sy), sy.run.value_counts().to_dict(), 'multi-contig', int((sy.n_contigs > 1).sum()))
