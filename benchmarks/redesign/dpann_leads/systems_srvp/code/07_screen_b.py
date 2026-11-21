#!/usr/bin/env python
"""Screen (b): partial loci the callers did not call, completed by a position-labeled gene (biotite login node).

Per small multi-gene model M (min_genes_required >= 2 and size 'small' by the rule in 06_screen_a.py / model_info.tsv;
every profile of these models was in the 02 search):
  * hits = selected per-protein hits (03_hits.py) to any profile of M (primary or exchangeable), minus proteins that
    are members of a called system of M
  * cluster on a contig when consecutive positions differ by <= inter_gene_max_space + 1 (MacSyFinder rule)
  * components per cluster by a matching (one protein fills one component; mandatory first, then accessory)
  * partial = quorum fails (matched mandatory < min_mandatory or matched mandatory + accessory < min_genes)
  * strict uncalled = no hit protein of the cluster is a member of any called system (any model, any caller)
  * completion = a slot gene within inter_gene_max_space + 1 of the cluster span, not a cluster hit protein, whose label
    is in the family set of a missing component, and whose addition satisfies the quorum
Controls (d) for each (partial cluster, missing component with a family set): the same label test around every called
system of an unrelated model (no shared component family) in the same genome with the cluster's window width, and a
random-window expectation n_label_in_genome x window_width / genome_genes.
"""
from pathlib import Path

import numpy as np
import pandas as pd

RD = Path('/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111')
D = RD / 'dpann_leads/systems_srvp/data'
mg = pd.read_csv(D / 'model_genes.tsv', sep='\t')
cf = pd.read_csv(D / 'component_families.tsv', sep='\t').fillna({'families': ''})
info = pd.read_csv(D / 'model_info.tsv', sep='\t').set_index('model_key')
sy = pd.read_parquet(D / 'systems.parquet')
slot = pd.read_parquet(D / 'slot.parquet')
hits = pd.read_parquet(D / 'hits.parquet')
genes = pd.read_parquet(RD / 'ws_srvp_bins/package/universe/genes.parquet', columns=['gid', 'genome'])
genome_ngenes = genes.groupby('genome').size().to_dict()

comp_fams = {(r.model_key, r.component): set(r.families.split(';')) - {''} for r in cf.itertuples()}
model_fams = {}
for (m, c), f in comp_fams.items():
    model_fams.setdefault(m, set()).update(f)
small_multi = set(info[(info.min_genes >= 2) & (info['size'] == 'small')].index)
picked = pd.read_csv(D / 'picked_profiles.tsv', sep='\t')
pres = {(r.model_key, r.component): r.presence for r in cf.itertuples()}
allcomp = cf[cf.model_key.isin(small_multi)].groupby('model_key').component.apply(list).to_dict()

pm = []
for r in mg[mg.model_key.isin(small_multi) & mg.presence.isin(['mandatory', 'accessory'])].itertuples():
    for p in r.profiles.split(';'):
        pm.append((r.caller, p, r.model_key, r.component, r.presence))
pm = pd.DataFrame(pm, columns=['caller', 'profile', 'model_key', 'component', 'presence']).drop_duplicates()
assert set(zip(pm.caller, pm.profile)) <= set(zip(picked.caller, picked.profile)), 'profile outside the search'
h = hits.merge(pm, on=['caller', 'profile'])
called_mem = set()
for r in sy.itertuples():
    for g in r.members:
        called_mem.add((r.model_key, g))
any_member = set(g for m in sy.members for g in m)
h = h[[(m, g) not in called_mem for m, g in zip(h.model_key, h.gid)]].copy()
h['igms'] = h.model_key.map(info.igms)
h = h.sort_values(['model_key', 'genome', 'contig_code', 'position_index']).reset_index(drop=True)
newc = ((h.model_key != h.model_key.shift()) | (h.genome != h.genome.shift()) |
        (h.contig_code != h.contig_code.shift()) | (h.position_index - h.position_index.shift() > h.igms + 1))
h['cidx'] = np.cumsum(newc.to_numpy())


def match(prot_comps, order):
    """Kuhn matching components -> proteins; components tried in `order` (mandatory first)."""
    owner = {}

    def aug(c, seen):
        for p, cs in prot_comps.items():
            if c in cs and p not in seen:
                seen.add(p)
                if p not in owner or aug(owner[p], seen):
                    owner[p] = c
                    return True
        return False
    for c in order:
        aug(c, set())
    return set(owner.values())


rows = []
for cidx, g in h.groupby('cidx', sort=False):
    m = g.model_key.iloc[0]
    pc = g.groupby('gid').component.apply(set).to_dict()
    comps = set(g.component)
    order = sorted(comps, key=lambda c: (pres[(m, c)] != 'mandatory', c))
    got = match(pc, order)
    nm = sum(pres[(m, c)] == 'mandatory' for c in got)
    na = len(got) - nm
    rows.append(dict(cidx=cidx, model_key=m, genome=g.genome.iloc[0], contig_code=int(g.contig_code.iloc[0]),
                     lo=int(g.position_index.min()), hi=int(g.position_index.max()), n_prot=len(pc),
                     prots=';'.join(map(str, sorted(pc))), comps=';'.join(sorted(got)), n_mand=nm, n_acc=na,
                     overlaps_call=any(p in any_member for p in pc)))
cl = pd.DataFrame(rows)
cl['igms'] = cl.model_key.map(info.igms)
cl['min_mand'] = cl.model_key.map(info.min_mand)
cl['min_genes'] = cl.model_key.map(info.min_genes)
cl['quorum'] = (cl.n_mand >= cl.min_mand) & (cl.n_mand + cl.n_acc >= cl.min_genes)
cl['cls'] = cl.model_key.map(info.cls)
cl['cluster_id'] = cl.model_key + '|' + cl.genome + '|c' + cl.contig_code.astype(str) + ':' + cl.lo.astype(str) + '-' + \
    cl.hi.astype(str)
cl['missing'] = [[c for c in allcomp[m] if c not in set(cs.split(';'))] for m, cs in zip(cl.model_key, cl.comps)]

SB = {}
for (gm, c), x in slot.groupby(['genome', 'contig_code']):
    x = x.sort_values('position_index')
    SB[(gm, c)] = (x.position_index.to_numpy(), x.gid.to_numpy(), x.label.to_numpy())


def window(genome, contig, lo, hi, igms, exclude):
    t = SB.get((genome, contig))
    if t is None:
        return []
    pos, gid, lab = t
    a, b = np.searchsorted(pos, lo - (igms + 1), 'left'), np.searchsorted(pos, hi + igms + 1, 'right')
    out = []
    for k in range(a, b):
        if gid[k] in exclude:
            continue
        x = int(pos[k])
        out.append((int(gid[k]), x, lab[k], 0 if lo < x < hi else min(abs(x - lo), abs(x - hi))))
    return out


part = cl[~cl.quorum].copy()
part['wg'] = [window(r.genome, r.contig_code, r.lo, r.hi, r.igms, set(map(int, r.prots.split(';'))))
              for r in part.itertuples()]
ev = []
for r in part.itertuples():
    have = set(r.comps.split(';')) - {''}
    for gid, x, lab, d in r.wg:
        for c in r.missing:
            if lab in comp_fams.get((r.model_key, c), set()):
                nm = r.n_mand + (pres[(r.model_key, c)] == 'mandatory')
                na = r.n_acc + (pres[(r.model_key, c)] == 'accessory')
                ev.append(dict(cluster_id=r.cluster_id, model_key=r.model_key, cls=r.cls, genome=r.genome,
                               contig_code=r.contig_code, lo=r.lo, hi=r.hi, cluster_prots=r.prots, have=r.comps,
                               gid=gid, position_index=x, label=lab, dist=d, component=c,
                               presence=pres[(r.model_key, c)],
                               completes=bool(nm >= r.min_mand and nm + na >= r.min_genes),
                               overlaps_call=r.overlaps_call, slot_is_member=gid in any_member))
ev = pd.DataFrame(ev)
if len(ev):
    ev = ev.merge(slot[['gid', 'label_name', 'length', 'strand', 'tier', 'support', 'votes_for', 'supporting_genomes',
                        'label_amplification']], on='gid', how='left')
ev.to_parquet(D / 'screen_b_events.parquet', index=False)

# controls (label sets in each called system's window, cached per (system, igms))
loci_by_genome = {g: x for g, x in sy.groupby('genome')}
_wc = {}


def sys_labels(o, igms):
    k = (o.sys_key, igms)
    if k not in _wc:
        w = window(o.genome, o.contig_code, o.lo, o.hi, igms, set())
        _wc[k] = (set(x[2] for x in w), set(x[2] for x in w if x[3] <= 1))
    return _wc[k]



n_lab_genome = slot.groupby(['genome', 'label']).size().to_dict()
ob = []
for r in part.itertuples():
    others = loci_by_genome.get(r.genome)
    if others is not None:
        mf = model_fams.get(r.model_key, set())
        others = others[(others.model_key != r.model_key) &
                        np.array([len(model_fams.get(m, set()) & mf) == 0 for m in others.model_key], dtype=bool)]
    width = (r.hi - r.lo + 1) + 2 * (r.igms + 1)
    ng = genome_ngenes[r.genome]
    for c in r.missing:
        fams = comp_fams.get((r.model_key, c), set())
        if not fams:
            continue
        hit = [x for x in r.wg if x[2] in fams]
        labs = [] if others is None else [sys_labels(o, r.igms) for o in others.itertuples()]
        cr = [bool(a & fams) for a, b in labs]
        cra = [bool(b & fams) for a, b in labs]
        nf = sum(n_lab_genome.get((r.genome, f), 0) for f in fams)
        ob.append(dict(cluster_id=r.cluster_id, model_key=r.model_key, cls=r.cls, component=c,
                       presence=pres[(r.model_key, c)], overlaps_call=r.overlaps_call, hit=int(bool(hit)),
                       hit_adj=int(any(x[3] <= 1 for x in hit)), ctrl_n=len(cr),
                       ctrl_rate=np.mean(cr) if cr else np.nan, ctrl_rate_adj=np.mean(cra) if cra else np.nan,
                       random_rate=min(1.0, nf * width / ng), random_rate_adj=min(1.0, nf * 2 / ng)))
ob = pd.DataFrame(ob)
ob.to_parquet(D / 'screen_b_obs_control.parquet', index=False)
cl.drop(columns=['missing']).assign(missing=[';'.join(x) for x in cl.missing]).to_parquet(D / 'screen_b_clusters.parquet',
                                                                                          index=False)

L = []
P = L.append
P(f'small multi-gene models with uncalled hit clusters: {cl.model_key.nunique()} of {len(small_multi)}; clusters '
  f'{len(cl)}; quorum met but uncalled {int(cl.quorum.sum())} (strict, no called member: '
  f'{int((cl.quorum & ~cl.overlaps_call).sum())}); partial {len(part)} (strict {int((~part.overlaps_call).sum())})')
P(f'partial clusters with a slot gene in window {int((part.wg.map(len) > 0).sum())}')
if len(ev):
    c = ev[ev.completes]
    cs = c[~c.overlaps_call]
    P(f'gain events {len(ev)} rows, clusters gaining {ev.cluster_id.nunique()}; complete: clusters '
      f'{c.cluster_id.nunique()}, slot genes {c.gid.nunique()}; strict uncalled: clusters {cs.cluster_id.nunique()}, '
      f'slot genes {cs.gid.nunique()}, genomes {cs.genome.nunique()}')
for nm, o in [('all partial', ob), ('strict partial', ob[~ob.overlaps_call])]:
    oc = o.dropna(subset=['ctrl_rate'])
    P(f'{nm}: (cluster, missing component) pairs with family set {len(o)}; observed {o.hit.sum()} '
      f'({o.hit.mean():.4f}); random-window expected {o.random_rate.sum():.1f}; with >= 1 unrelated called system '
      f'{len(oc)}: observed {oc.hit.sum()} vs expected {oc.ctrl_rate.sum():.1f}; adjacent observed {oc.hit_adj.sum()} '
      f'vs expected {oc.ctrl_rate_adj.sum():.1f}')
P('')
t = part.groupby('model_key').size().rename('partial').to_frame()
t['strict'] = part[~part.overlaps_call].groupby('model_key').size()
t['with_slot'] = part[part.wg.map(len) > 0].groupby('model_key').size()
if len(ev):
    t['gain'] = ev.groupby('model_key').cluster_id.nunique()
    t['complete'] = ev[ev.completes].groupby('model_key').cluster_id.nunique()
    t['complete_strict'] = ev[ev.completes & ~ev.overlaps_call].groupby('model_key').cluster_id.nunique()
P(t.fillna(0).astype(int).sort_values(['complete_strict', 'partial'] if len(ev) else ['partial'], ascending=False)
  .head(50).to_string())
(D / 'screen_b_summary.txt').write_text('\n'.join(L) + '\n')
print('\n'.join(L))
