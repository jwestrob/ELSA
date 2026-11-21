#!/usr/bin/env python
"""Screen (a): position-labeled genes inside or next to called systems (biotite login node, serial).

For each called system (DefenseFinder 3.1.0 DefenseFinder + RM passes, TXSScan 1.1.1; callers' best_solution):
  * components the call satisfies (hit_gene_ref of its members) and model components (mandatory / accessory) it lacks
  * slot genes on the same contig, not members: inside the core span (dist 0), adjacent (dist 1), or window
    (dist 2 .. inter_gene_max_space + 1, where MacSyFinder would still cluster a hit)
  * a slot gene "gains a component" when its label is in the Pfam family set of a lacking component
Control (d): for every (system, lacking component with a family set) the same test around every called system of an
unrelated model in the same genome (unrelated = no shared component family), with the test system's window width.
System class: caller type (RM = DefenseFinder RM family; DF = other DefenseFinder; TXSS = TXSScan) x size (see the
size rule at model_info below). For a called system every lacking component is optional by construction (the call
already meets its quorum); n_mand_acc counts alternative profiles listed as separate components.
"""
from pathlib import Path

import numpy as np
import pandas as pd

RD = Path('/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111')
D = RD / 'dpann_leads/systems_srvp/data'
mg = pd.read_csv(D / 'model_genes.tsv', sep='\t')
cf = pd.read_csv(D / 'component_families.tsv', sep='\t').fillna({'families': ''})
sy = pd.read_parquet(D / 'systems.parquet')
slot = pd.read_parquet(D / 'slot.parquet')

comp_fams = {(r.model_key, r.component): set(r.families.split(';')) - {''} for r in cf.itertuples()}
model_fams = {}
for (m, c), f in comp_fams.items():
    model_fams.setdefault(m, set()).update(f)
info = mg.groupby('model_key').agg(family=('family', 'first'), igms=('inter_gene_max_space', 'first'),
                                   n_comp=('n_mand_acc', 'first'), min_mand=('min_mandatory', 'first'),
                                   min_genes=('min_genes', 'first')).reset_index()
info['ctype'] = info.family.map({'RM': 'RM', 'DefenseFinder': 'DF', 'TXSScan': 'TXSS'})
# size: TXSScan small = T1SS and T5a/b/c SS (pili, secretion and conjugation machines are large); RM family small (1-3
# defined proteins: Type I R2M2S, Type II R + M, Type IIG one protein, Type III Mod2Res, Type IV; its models list many
# alternative profiles as separate components); DefenseFinder family small when the model lists <= 4 mandatory +
# accessory genes.
TX_SMALL = {'T1SS', 'T5aSS', 'T5bSS', 'T5cSS'}
info['size'] = ['small' if (f == 'RM' or (f == 'TXSScan' and m.split('/')[-1] in TX_SMALL) or
                            (f == 'DefenseFinder' and n <= 4)) else 'large'
                for f, m, n in zip(info.family, info.model_key, info.n_comp)]
info['cls'] = info.ctype + '/' + info['size']
info.to_csv(D / 'model_info.tsv', sep='\t', index=False)
I = info.set_index('model_key')
allc = cf.groupby('model_key').component.apply(list).to_dict()
assert set(sy.model_key) <= set(I.index), set(sy.model_key) - set(I.index)
sy['igms'] = sy.model_key.map(I.igms)
sy['cls'] = sy.model_key.map(I.cls)
sy['lacking'] = [[c for c in allc.get(m, []) if c not in set(cs)] for m, cs in zip(sy.model_key, sy.comps)]
sy['n_lacking'] = sy.lacking.map(len)

SB = {}
for (g, c), x in slot.groupby(['genome', 'contig_code']):
    x = x.sort_values('position_index')
    SB[(g, c)] = (x.position_index.to_numpy(), x.gid.to_numpy(), x.label.to_numpy())
members = set(g for m in sy.members for g in m)


def window(genome, contig, lo, hi, igms):
    t = SB.get((genome, contig))
    if t is None:
        return []
    pos, gid, lab = t
    a, b = np.searchsorted(pos, lo - (igms + 1), 'left'), np.searchsorted(pos, hi + igms + 1, 'right')
    out = []
    for k in range(a, b):
        if gid[k] in members:
            continue
        x = int(pos[k])
        d = 0 if lo < x < hi else min(abs(x - lo), abs(x - hi))
        out.append((int(gid[k]), x, lab[k], d))
    return out


rows, obs_rows, ctrl_rows = [], [], []
by_genome = {g: x for g, x in sy.groupby('genome')}
for s in sy.itertuples():
    wg = window(s.genome, s.contig_code, s.lo, s.hi, s.igms)
    for gid, x, lab, d in wg:
        gains = [c for c in s.lacking if lab in comp_fams.get((s.model_key, c), set())]
        same = [c for c in s.comps if lab in comp_fams.get((s.model_key, c), set())]
        rows.append(dict(sys_key=s.sys_key, model_key=s.model_key, cls=s.cls, genome=s.genome,
                         contig_code=s.contig_code, gid=gid, position_index=x, label=lab, dist=d,
                         dist_class='inside' if d == 0 else ('adjacent' if d == 1 else 'window'),
                         gains=';'.join(gains), same_family_as_member=';'.join(same)))
    mf = model_fams.get(s.model_key, set())
    others = by_genome[s.genome]
    others = others[(others.sys_key != s.sys_key) & (others.model_key != s.model_key)]
    others = others[[len(model_fams.get(m, set()) & mf) == 0 for m in others.model_key]]
    for c in s.lacking:
        fams = comp_fams.get((s.model_key, c), set())
        hits = [g for g in wg if g[2] in fams]
        obs_rows.append(dict(sys_key=s.sys_key, model_key=s.model_key, cls=s.cls, component=c, mappable=int(bool(fams)),
                             hit=int(bool(hits)), hit_adj=int(any(g[3] <= 1 for g in hits)),
                             n_ctrl=len(others) if fams else 0))
        if not fams:
            continue
        for o in others.itertuples():
            ctl = [g for g in window(o.genome, o.contig_code, o.lo, o.hi, s.igms) if g[2] in fams]
            ctrl_rows.append(dict(sys_key=s.sys_key, model_key=s.model_key, component=c, ctrl_sys=o.sys_key,
                                  ctrl_hit=int(bool(ctl)), ctrl_hit_adj=int(any(g[3] <= 1 for g in ctl))))
win = pd.DataFrame(rows).merge(slot[['gid', 'label_name', 'length', 'tier', 'support', 'votes_for', 'supporting_genomes',
                                     'label_amplification', 'strand']], on='gid', how='left')
win.to_parquet(D / 'screen_a_window.parquet', index=False)
obs = pd.DataFrame(obs_rows)
ctrl = pd.DataFrame(ctrl_rows)
obs.to_parquet(D / 'screen_a_lacking.parquet', index=False)
ctrl.to_parquet(D / 'screen_a_control.parquet', index=False)

sy['slot_win'] = sy.sys_key.isin(win.sys_key)
sy['slot_adj'] = sy.sys_key.isin(win[win.dist <= 1].sys_key)
sy['gain_win'] = sy.sys_key.isin(win[win.gains != ''].sys_key)
sy['gain_adj'] = sy.sys_key.isin(win[(win.gains != '') & (win.dist <= 1)].sys_key)
sy.drop(columns=['members', 'comps', 'lacking']).to_parquet(D / 'screen_a_systems.parquet', index=False)

L = []
P = L.append
P(f'called systems {len(sy)} in {sy.genome.nunique()} genomes, {sy.model_key.nunique()} models')
t = sy.groupby('cls').agg(systems=('sys_key', 'size'), lacking=('n_lacking', lambda x: int((x > 0).sum())),
                          slot_inside_adj=('slot_adj', 'sum'), slot_window=('slot_win', 'sum'),
                          gain_window=('gain_win', 'sum'), gain_adjacent=('gain_adj', 'sum'))
t.loc['all'] = t.sum()
P(t.to_string())
wa = win[win.dist <= 1].drop_duplicates('gid')
P(f'slot genes inside/adjacent: {len(wa)}; tier {wa.tier.value_counts().to_dict()}')
P(f'slot genes in windows {win.gid.nunique()}; matching a component the call already has '
  f'{win[win.same_family_as_member != ""].gid.nunique()}; gaining {win[win.gains != ""].gid.nunique()}')
m = obs[obs.mappable == 1]
P(f'lacking (system, component) pairs {len(obs)}; with a family set {len(m)}; observed window {m.hit.sum()} '
  f'({m.hit.mean():.4f}), adjacent {m.hit_adj.sum()}')
if len(ctrl):
    cr = ctrl.groupby(['sys_key', 'component']).agg(r=('ctrl_hit', 'mean'), ra=('ctrl_hit_adj', 'mean'))
    mm = m.set_index(['sys_key', 'component']).join(cr, how='inner')
    P(f'control (unrelated called systems, same genome): {len(ctrl)} pairs on {len(mm)} lacking components; '
      f'expected window {mm.r.sum():.1f} vs observed {mm.hit.sum()}; expected adjacent {mm.ra.sum():.1f} vs '
      f'observed {mm.hit_adj.sum()}')
g = win[win.gains != '']
if len(g):
    P('gain events:')
    P(g[['sys_key', 'model_key', 'gains', 'gid', 'label', 'label_name', 'dist', 'tier', 'length', 'support']]
      .to_string(index=False))
(D / 'screen_a_summary.txt').write_text('\n'.join(L) + '\n')
print('\n'.join(L))
