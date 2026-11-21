#!/usr/bin/env python
"""Global numbers for systems_srvp (biotite login node, serial). Writes data/global_numbers.tsv (section, key, columns)
and data/global_numbers.txt.

Sections
  a_called     called systems by class: systems, lacking >= 1 model component, slot gene inside/adjacent, in window,
               gaining a component (window / adjacent), observed vs expected (unrelated called systems, same genome)
  a_tier       tiers of slot genes inside/adjacent and of gaining genes
  b_uncalled   small multi-gene models: partial loci (all / strict), with a slot gene in window, gaining, completed
               (strict), completing genes, observed vs expected (unrelated called systems; random windows)
  b_large      large models: the callers' own rejected candidates (failed quorum) and those with a slot gene inside or
               adjacent (counts only)
  b_tier       tiers of strictly completing genes
  type         by system type (DefenseFinder category): completed loci, genes, genomes called -> called or completed,
               new genomes, their phyla
  rerun        caller-profile rerun rates per gene set
"""
from pathlib import Path

import numpy as np
import pandas as pd

RD = Path('/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111')
D = RD / 'dpann_leads/systems_srvp/data'
info = pd.read_csv(D / 'model_info.tsv', sep='\t').set_index('model_key')
sa = pd.read_parquet(D / 'screen_a_systems.parquet')
wa = pd.read_parquet(D / 'screen_a_window.parquet')
la = pd.read_parquet(D / 'screen_a_lacking.parquet')
ca = pd.read_parquet(D / 'screen_a_control.parquet')
cl = pd.read_parquet(D / 'screen_b_clusters.parquet')
ev = pd.read_parquet(D / 'screen_b_events.parquet')
ob = pd.read_parquet(D / 'screen_b_obs_control.parquet')
rj = pd.read_parquet(D / 'rejected.parquet')
sy = pd.read_parquet(D / 'systems.parquet')
slot = pd.read_parquet(D / 'slot.parquet')
sc = pd.read_csv(D / 'rerun/test_event_scores.tsv', sep='\t')
ct = pd.read_csv(D / 'rerun/control_scores.tsv', sep='\t')
genes = pd.read_parquet(RD / 'ws_srvp_bins/package/universe/genes.parquet', columns=['gid', 'contig_code', 'position_index'])
gi = genes.set_index('gid')
gen = pd.read_parquet(RD / 'ws_srvp_bins/package/universe/genomes.parquet', columns=['genome', 'domain', 'phylum'])
ph = dict(zip(gen.genome, gen.phylum))
rows, txt = [], []


def add(section, key, **kv):
    rows.append(dict(section=section, key=key, **kv))


# ---------------------------------------------------------------- (a)
cr = ca.groupby(['sys_key', 'component']).agg(r=('ctrl_hit', 'mean'), ra=('ctrl_hit_adj', 'mean'))
lm = la[la.mappable == 1].set_index(['sys_key', 'component']).join(cr, how='left').reset_index()
for cls_, g in list(sa.groupby('cls')) + [('all', sa)]:
    l = lm if cls_ == 'all' else lm[lm.cls == cls_]
    lc = l.dropna(subset=['r'])
    add('a_called', cls_, systems=len(g), genomes=g.genome.nunique(), lacking_component=int((g.n_lacking > 0).sum()),
        slot_inside_adjacent=int(g.slot_adj.sum()), slot_in_window=int(g.slot_win.sum()), gain_window=int(g.gain_win.sum()),
        gain_adjacent=int(g.gain_adj.sum()), lacking_pairs_mappable=len(l), observed_window=int(l.hit.sum()),
        expected_window_unrelated=round(lc.r.sum(), 1), observed_adjacent=int(l.hit_adj.sum()),
        expected_adjacent_unrelated=round(lc.ra.sum(), 1))
w1 = wa[wa.dist <= 1].drop_duplicates('gid')
wg = wa[wa.gains != ''].drop_duplicates('gid')
for nm, x in [('inside_adjacent', w1), ('gaining', wg), ('gaining_small', wg[wg.cls.str.endswith('small')])]:
    add('a_tier', nm, genes=len(x), **{f'tier_{k}': int(v) for k, v in x.tier.value_counts().items()})

# ---------------------------------------------------------------- (b)
cl['partial'] = ~cl.quorum
evc = ev[ev.completes]
for cls_, g in list(cl.groupby('cls')) + [('all', cl)]:
    p = g[g.partial]
    e = ev if cls_ == 'all' else ev[ev.cls == cls_]
    c = e[e.completes & ~e.overlaps_call]
    o = (ob if cls_ == 'all' else ob[ob.cls == cls_])
    o = o[~o.overlaps_call]
    oc = o.dropna(subset=['ctrl_rate'])
    add('b_uncalled', cls_, clusters=len(g), partial=len(p), partial_strict=int((~p.overlaps_call).sum()),
        quorum_met_uncalled=int(g.quorum.sum()), partial_gaining=e.cluster_id.nunique(),
        completed_strict_loci=c.drop_duplicates(['genome', 'contig_code', 'lo', 'hi']).shape[0],
        completed_strict_cluster_model_pairs=c.cluster_id.nunique(), completing_genes=c.gid.nunique(),
        genomes=c.genome.nunique(), pairs_with_family_set=len(o), observed=int(o.hit.sum()),
        expected_random=round(o.random_rate.sum(), 1), pairs_with_unrelated_call=len(oc),
        observed_on_those=int(oc.hit.sum()), expected_unrelated=round(oc.ctrl_rate.sum(), 1))
# large models: callers' own rejected candidates
rj['cls'] = rj.model_key.map(info.cls)
rj['contig_code'] = rj.gid.map(gi.contig_code)
rj['pos'] = rj.gid.map(gi.position_index)
SB = {c: np.sort(g.position_index.to_numpy()) for c, g in slot.groupby('contig_code')}
for cls_, g in rj[rj.cls.str.endswith('large')].groupby('cls'):
    cand = g.groupby(['run', 'genome', 'candidate_id']).agg(contig=('contig_code', 'first'), lo=('pos', 'min'),
                                                           hi=('pos', 'max'), nc=('contig_code', 'nunique'))
    n_adj = 0
    for r in cand.itertuples():
        a = SB.get(r.contig)
        if a is not None and np.any((a >= r.lo - 1) & (a <= r.hi + 1)):
            n_adj += 1
    add('b_large', cls_, rejected_candidates=len(cand), with_slot_inside_adjacent=n_adj)
cs = ev[ev.completes & ~ev.overlaps_call].drop_duplicates('gid')
for nm, x in [('completing_strict', cs), ('completing_strict_small', cs[cs.cls.str.endswith('small')])]:
    add('b_tier', nm, genes=len(x), **{f'tier_{k}': int(v) for k, v in x.tier.value_counts().items()})

# ---------------------------------------------------------------- by type
typ = lambda m: m.split('/')[-1] if m.startswith('TXSScan') else m.split('/')[1]
c = ev[ev.completes & ~ev.overlaps_call].copy()
c['stype'] = c.model_key.map(typ)
sy['stype'] = sy.model_key.map(typ)
called = sy.groupby('stype').genome.apply(set).to_dict()
for t, g in c.groupby('stype'):
    before = called.get(t, set())
    new = set(g.genome) - before
    add('type', t, cls=info.loc[g.model_key.iloc[0], 'cls'],
        completed_loci=g.drop_duplicates(['genome', 'contig_code', 'gid']).shape[0], genes=g.gid.nunique(),
        genomes_called=len(before), genomes_after=len(before | set(g.genome)), new_genomes=len(new),
        new_genome_phyla=';'.join(f'{k}:{v}' for k, v in pd.Series([ph[x] for x in new]).value_counts().items()),
        components=';'.join(sorted(set(g.component))))

# ---------------------------------------------------------------- rerun
for nm, x in [('test_all', sc), ('test_small', sc[sc.cls.str.endswith('small')]),
              ('test_small_strict_complete', sc[sc.gid.isin(cs.gid)])]:
    g = x.groupby('gid').agg(a=('any_hit', 'max'), ga=('pass_ga', 'max'), al=('pass_all', 'max'))
    add('rerun', nm, n=len(g), any_hit=int(g.a.sum()), ga=int(g.ga.sum()), ga_cov_p=int(g.al.sum()))
for s, x in ct.groupby('set'):
    add('rerun', s, n=len(x), any_hit=int(x.any_hit.sum()), ga=int(x.pass_ga.sum()), ga_cov_p=int(x.pass_all.sum()))

t = pd.DataFrame(rows)
t.to_csv(D / 'global_numbers.tsv', sep='\t', index=False)
pd.set_option('display.width', 300)
for s, x in t.groupby('section', sort=False):
    txt.append(f'== {s}')
    txt.append(x.dropna(axis=1, how='all').drop(columns='section').to_string(index=False))
(D / 'global_numbers.txt').write_text('\n'.join(txt) + '\n')
print('\n'.join(txt))
