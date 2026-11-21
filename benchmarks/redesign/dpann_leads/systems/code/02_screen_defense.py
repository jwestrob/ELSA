#!/usr/bin/env python
"""Screen (a) and (c): slot-labeled genes inside or next to called DefenseFinder systems.

For each called system (DefenseFinder 2.0.2 model, Sharur colocation call):
  * components the call satisfies (member profiles, exchangeables collapsed to the primary component)
  * components the model accepts (mandatory / accessory) that the call lacks
  * slot-labeled genes on the same contig inside the member span (inside), at distance 1 from the span (adjacent),
    or at distance 2 .. inter_gene_max_space + 1 (window, where MacSyFinder would still cluster a hit)
  * a slot gene "gains a component" when its proposed family maps to a lacking component (family map below)

Family map (profile -> Pfam / KO): on DPANN proteins carrying a DefenseFinder hit to profile p, family F is a
correspondent of p when F's domain overlaps the profile hit by >= 50% of the shorter span on >= 3 proteins and on
>= 20% of p's proteins. A component's families = union over its profiles.

Control: for each (system, lacking component) the same test on every called system of an unrelated model in the
same genome (unrelated = shares no component family with the system's model), same distance classes.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

W = Path('/Users/jacob/Documents/Sandbox/elsa_test/ELSA/benchmarks/redesign/dpann_leads/systems')
D = W / 'data'
mg = pd.read_csv(D / 'df_model_genes.tsv', sep='\t')
sysdf = pd.read_parquet(D / 'df_systems.parquet')
slot = pd.read_parquet(D / 'slot.parquet')
pf = pd.read_parquet(D / 'profile_family.parquet')

MIN_N, MIN_FRAC = 3, 0.20
pf_ok = pf[(pf.n_overlap >= MIN_N) & (pf.p_overlap_given_prof >= MIN_FRAC)]
prof2fam = pf_ok.groupby('profile').family.apply(set).to_dict()

comp = mg[mg.presence.isin(['mandatory', 'accessory'])].copy()
comp['families'] = [set().union(*[prof2fam.get(p, set()) for p in ps.split(';')]) for ps in comp.profiles]
comp['n_fam'] = comp.families.map(len)
comp_fams = {(r.model, r.component): r.families for r in comp.itertuples()}
model_fams = comp.groupby('model').families.apply(lambda s: set().union(*s)).to_dict()
model_info = mg.groupby('model').agg(igms=('inter_gene_max_space', 'first'), n_comp=('n_mand_acc', 'first'),
                                     min_mand=('min_mandatory', 'first'), min_genes=('min_genes', 'first'),
                                     category=('category', 'first')).reset_index()
comp[['model', 'component', 'presence', 'profiles', 'n_fam']].assign(
    families=[';'.join(sorted(f)) for f in comp.families]).to_csv(D / 'component_families.tsv', sep='\t', index=False)

# ---------------------------------------------------------------- per-system summary
sy = sysdf.groupby('system_id').agg(model=('model', 'first'), genome=('genome_id', 'first'),
                                    contig=('contig_id', 'first'), lo=('gene_index', 'min'),
                                    hi=('gene_index', 'max'), n_members=('protein_id', 'size'),
                                    comps=('component', lambda s: set(s))).reset_index()
sy = sy.merge(model_info, on='model', how='left')
assert sysdf.groupby('system_id').contig_id.nunique().max() == 1
allc = comp.groupby('model').component.apply(list).to_dict()
sy['lacking'] = [[c for c in allc.get(m, []) if c not in cs] for m, cs in zip(sy.model, sy.comps)]
sy['n_lacking'] = sy.lacking.map(len)

# slot genes indexed by contig
slot_by_contig = {c: g[['protein_id', 'gene_index', 'label', 'tier', 'list_type']].to_numpy()
                  for c, g in slot.groupby('contig_id')}
member_set = set(sysdf.protein_id)


def window_genes(contig, lo, hi, igms):
    arr = slot_by_contig.get(contig)
    if arr is None:
        return []
    out = []
    for pid, gi, lab, tier, lt in arr:
        if lo - (igms + 1) <= gi <= hi + igms + 1 and pid not in member_set:
            d = 0 if lo < gi < hi else min(abs(gi - lo), abs(gi - hi))
            cls = 'inside' if d == 0 else ('adjacent' if d == 1 else 'window')
            out.append((pid, int(gi), lab, tier, lt, d, cls))
    return out


rows = []
for s in sy.itertuples():
    for pid, gi, lab, tier, lt, d, cls in window_genes(s.contig, s.lo, s.hi, s.igms):
        gains = [c for c in s.lacking if lab in comp_fams.get((s.model, c), set())]
        fills_present = [c for c in s.comps if lab in comp_fams.get((s.model, c), set())]
        rows.append(dict(system_id=s.system_id, model=s.model, genome=s.genome, contig=s.contig, protein_id=pid,
                         gene_index=gi, label=lab, tier=tier, list_type=lt, dist=d, dist_class=cls,
                         gains=';'.join(gains), same_family_as_member=';'.join(fills_present)))
win = pd.DataFrame(rows)
win = win.merge(slot[['protein_id', 'label_name', 'length', 'votes_for', 'supporting_genomes', 'vfrac',
                      'proposed_evalue', 'alignment_label']], on='protein_id', how='left')
win.to_csv(D / 'defense_window_slot_genes.tsv', sep='\t', index=False)

# ---------------------------------------------------------------- control: unrelated systems, same genome
by_genome = sy.groupby('genome')
ctrl_rows = []
for s in sy.itertuples():
    if s.n_lacking == 0:
        continue
    others = by_genome.get_group(s.genome)
    mf = model_fams.get(s.model, set())
    others = others[(others.system_id != s.system_id) & (others.model != s.model)]
    others = others[[len(model_fams.get(m, set()) & mf) == 0 for m in others.model]]
    for c in s.lacking:
        fams = comp_fams.get((s.model, c), set())
        if not fams:
            continue
        obs = [g for g in window_genes(s.contig, s.lo, s.hi, s.igms) if g[2] in fams]
        for o in others.itertuples():
            ctl = [g for g in window_genes(o.contig, o.lo, o.hi, s.igms) if g[2] in fams]
            ctrl_rows.append(dict(system_id=s.system_id, model=s.model, component=c, ctrl_system=o.system_id,
                                  ctrl_model=o.model, ctrl_hit=int(len(ctl) > 0),
                                  ctrl_hit_adj=int(any(g[5] <= 1 for g in ctl))))
        if len(others) == 0:
            ctrl_rows.append(dict(system_id=s.system_id, model=s.model, component=c, ctrl_system=None,
                                  ctrl_model=None, ctrl_hit=np.nan, ctrl_hit_adj=np.nan))
ctrl = pd.DataFrame(ctrl_rows)
ctrl.to_csv(D / 'defense_control_pairs.tsv', sep='\t', index=False)

# observed per (system, lacking component with a family map)
obs_rows = []
for s in sy.itertuples():
    wg = window_genes(s.contig, s.lo, s.hi, s.igms)
    for c in s.lacking:
        fams = comp_fams.get((s.model, c), set())
        hits = [g for g in wg if g[2] in fams]
        obs_rows.append(dict(system_id=s.system_id, model=s.model, component=c, mappable=int(bool(fams)),
                             hit=int(bool(hits)), hit_adj=int(any(g[5] <= 1 for g in hits)),
                             genes=';'.join(g[0] for g in hits)))
obs = pd.DataFrame(obs_rows)
obs.to_csv(D / 'defense_lacking_components.tsv', sep='\t', index=False)

# ---------------------------------------------------------------- summary
sy['has_slot_window'] = sy.system_id.isin(win.system_id)
sy['has_slot_inside_adj'] = sy.system_id.isin(win[win.dist <= 1].system_id)
sy['gains'] = sy.system_id.isin(win[win.gains != ''].system_id)
sy['gains_adj'] = sy.system_id.isin(win[(win.gains != '') & (win.dist <= 1)].system_id)
sy.drop(columns=['comps', 'lacking']).to_csv(D / 'defense_systems_summary.tsv', sep='\t', index=False)

lines = []
P = lines.append
P(f'called DefenseFinder systems: {len(sy)} in {sy.genome.nunique()} genomes, {sy.model.nunique()} models')
P(f'  systems lacking >= 1 model component (mandatory or accessory): {(sy.n_lacking > 0).sum()}')
P(f'  systems with a slot-labeled gene inside/adjacent (dist <= 1): {sy.has_slot_inside_adj.sum()}; '
  f'within model window: {sy.has_slot_window.sum()}')
P(f'  systems that gain a model component from a slot gene: window {sy.gains.sum()}; inside/adjacent {sy.gains_adj.sum()}')
P(f'  slot genes in windows: {win.protein_id.nunique()} (rows {len(win)}); with gain: {win[win.gains != ""].protein_id.nunique()}')
P(f'  slot genes in windows whose label matches a component the call already has: '
  f'{win[win.same_family_as_member != ""].protein_id.nunique()}')
m = obs[obs.mappable == 1]
P(f'lacking (system, component) pairs: {len(obs)}; with a family map: {len(m)}; observed hit rate window '
  f'{m.hit.mean():.4f} ({m.hit.sum()}/{len(m)}), adjacent {m.hit_adj.mean():.4f} ({m.hit_adj.sum()}/{len(m)})')
c = ctrl.dropna(subset=['ctrl_hit'])
if len(c):
    # one control rate per (system, component), averaged over unrelated systems
    cr = c.groupby(['system_id', 'component']).agg(ctrl_hit=('ctrl_hit', 'mean'), ctrl_adj=('ctrl_hit_adj', 'mean'))
    mm = m.set_index(['system_id', 'component']).join(cr, how='inner')
    P(f'control (unrelated systems, same genome; {len(c)} pairs over {len(mm)} lacking components with >=1 control):'
      f' window {c.ctrl_hit.mean():.4f}, adjacent {c.ctrl_hit_adj.mean():.4f}; '
      f'matched observed on those {len(mm)}: window {mm.hit.mean():.4f}, adjacent {mm.hit_adj.mean():.4f}; '
      f'expected window hits {mm.ctrl_hit.sum():.1f} vs observed {mm.hit.sum()}')
    tab = [[int(m.hit.sum()), int(len(m) - m.hit.sum())], [int(c.ctrl_hit.sum()), int(len(c) - c.ctrl_hit.sum())]]
    P(f'  Fisher window obs vs ctrl pairs: OR {fisher_exact(tab)[0]:.2f} p {fisher_exact(tab)[1]:.2e}')
P('')
P('by model (systems / lacking-any / slot inside-adj / slot window / gain window / gain adj):')
bm = sy.groupby('model').agg(n=('system_id', 'size'), lacking=('n_lacking', lambda x: int((x > 0).sum())),
                             slot_adj=('has_slot_inside_adj', 'sum'), slot_win=('has_slot_window', 'sum'),
                             gain=('gains', 'sum'), gain_adj=('gains_adj', 'sum'), n_comp=('n_comp', 'first'))
bm = bm.sort_values('n', ascending=False)
P(bm.to_string())
P('')
g = win[win.gains != '']
if len(g):
    P('gain events (slot gene, system, component):')
    P(g[['protein_id', 'system_id', 'model', 'gains', 'label', 'label_name', 'tier', 'list_type', 'dist', 'length',
         'votes_for', 'supporting_genomes', 'vfrac']].to_string())
(W / 'data' / 'screen_a_summary.txt').write_text('\n'.join(lines) + '\n')
print('\n'.join(lines))
