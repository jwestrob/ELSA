#!/usr/bin/env python
"""Screen (c): slot-labeled families that recur inside / next to one called DefenseFinder system type.

For model M and slot label F:
  obs      = systems of M with >= 1 F-labeled slot gene (not a system member) within the model window
             (span +/- inter_gene_max_space + 1), plus counts inside the member span and at distance <= 1
  control  = for every system of M, the called systems of OTHER models in the same genome whose component families
             share nothing with M and do not include F; per M-system the fraction of those controls with an
             F-labeled slot gene within the same radius of their span; expected = sum over M-systems with >= 1 control
  random   = expected from the genome-wide density of F-labeled slot genes x window width
  test     = one-sided Poisson on obs vs expected (control); Benjamini-Hochberg over pairs with obs >= 3
"""
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy.stats import poisson

W = Path('/Users/jacob/Documents/Sandbox/elsa_test/ELSA/benchmarks/redesign/dpann_leads/systems')
D = W / 'data'
DB = '/Users/jacob/Documents/Sandbox/Sharur/Sharur/data/dpann/sharur.duckdb'
mg = pd.read_csv(D / 'df_model_genes.tsv', sep='\t')
cf = pd.read_csv(D / 'component_families.tsv', sep='\t').fillna({'families': ''})
model_fams = {}
for r in cf.itertuples():
    model_fams.setdefault(r.model, set()).update(set(r.families.split(';')) - {''})
sysdf = pd.read_parquet(D / 'df_systems.parquet')
slot = pd.read_parquet(D / 'slot.parquet')
con = duckdb.connect()
con.execute(f"ATTACH '{DB}' AS d (READ_ONLY)")
bins = con.execute("SELECT bin_id, taxonomy, completeness FROM d.bins").df()
species = dict(zip(bins.bin_id, bins.taxonomy.str.extract(r'(s__[^;]*)')[0].fillna(bins.bin_id)))
ngenes = con.execute("SELECT bin_id, count(*) n FROM d.proteins GROUP BY 1").df().set_index('bin_id').n.to_dict()

igms = mg.groupby('model').inter_gene_max_space.first().to_dict()
sy = sysdf.groupby('system_id').agg(model=('model', 'first'), genome=('genome_id', 'first'),
                                    contig=('contig_id', 'first'), lo=('gene_index', 'min'),
                                    hi=('gene_index', 'max')).reset_index()
sy['r'] = sy.model.map(igms) + 1
members = set(sysdf.protein_id)
SB = {}
for c, g in slot[~slot.protein_id.isin(members)].groupby('contig_id'):
    g = g.sort_values('gene_index')
    SB[c] = (g.gene_index.to_numpy(), g.label.to_numpy(), g.protein_id.to_numpy())


def labels_near(contig, lo, hi, r):
    t = SB.get(contig)
    if t is None:
        return {}
    gi, lab, pid = t
    a, b = np.searchsorted(gi, lo - r, 'left'), np.searchsorted(gi, hi + r, 'right')
    out = {}
    for k in range(a, b):
        x = int(gi[k])
        d = 0 if lo < x < hi else min(abs(x - lo), abs(x - hi))
        out[lab[k]] = min(d, out.get(lab[k], 99))
    return out


# labels near each system at its own radius; for controls the radius of the focal model
near_own = {s.system_id: labels_near(s.contig, s.lo, s.hi, s.r) for s in sy.itertuples()}
radii = sorted(sy.r.unique())
near_r = {(s.system_id, r): labels_near(s.contig, s.lo, s.hi, r) for s in sy.itertuples() for r in radii}
by_genome = {g: x for g, x in sy.groupby('genome')}
lab_count_genome = slot.groupby(['bin_id', 'label']).size().to_dict()

rows = []
for m, ms in sy.groupby('model'):
    mf = model_fams.get(m, set())
    labs = set()
    for sid in ms.system_id:
        labs.update(near_own[sid])
    for F in labs:
        obs = obs_in = obs_adj = 0
        genomes, spp, exp_ctrl, exp_rand, n_with_ctrl = set(), set(), 0.0, 0.0, 0
        exp_ctrl_adj, exp_rand_adj, spp_adj = 0.0, 0.0, set()
        for s in ms.itertuples():
            d = near_own[s.system_id].get(F)
            if d is not None:
                obs += 1
                obs_in += d == 0
                obs_adj += d <= 1
                genomes.add(s.genome)
                spp.add(species.get(s.genome))
                if d <= 1:
                    spp_adj.add(species.get(s.genome))
            oth = by_genome[s.genome]
            oth = oth[(oth.model != m)]
            ok = [(len(model_fams.get(om, set()) & mf) == 0) and (F not in model_fams.get(om, set()))
                  for om in oth.model]
            oth = oth[ok]
            if len(oth):
                n_with_ctrl += 1
                exp_ctrl += np.mean([F in near_r[(o, s.r)] for o in oth.system_id])
                exp_ctrl_adj += np.mean([near_r[(o, s.r)].get(F, 99) <= 1 for o in oth.system_id])
            width = (s.hi - s.lo + 1) + 2 * s.r
            dens = lab_count_genome.get((s.genome, F), 0) / ngenes.get(s.genome, 1)
            exp_rand += min(1.0, dens * width)
            exp_rand_adj += min(1.0, dens * (max(s.hi - s.lo - 1, 0) + 2))
        obs_ctrl_sub = sum(1 for s in ms.itertuples() if F in near_own[s.system_id]
                           and len(by_genome[s.genome][by_genome[s.genome].model != m]) > 0)
        rows.append(dict(model=m, label=F, n_systems=len(ms), obs=obs, obs_inside=obs_in, obs_adj=obs_adj,
                         genomes=len(genomes), species=len(spp), n_with_ctrl=n_with_ctrl,
                         exp_ctrl=exp_ctrl, exp_rand=exp_rand, species_adj=len(spp_adj),
                         exp_ctrl_adj=exp_ctrl_adj, exp_rand_adj=exp_rand_adj))
t = pd.DataFrame(rows)
t = t.merge(slot.groupby('label').label_name.first().rename('label_name'), left_on='label', right_index=True)
t['label_total'] = t.label.map(slot.label.value_counts())
# Poisson tail against the larger of the two expectations (conservative)
t['exp_max'] = np.maximum(t.exp_ctrl * t.n_systems / t.n_with_ctrl.clip(lower=1), t.exp_rand)
t['p'] = poisson.sf(t.obs - 1, t.exp_max.clip(lower=1e-3))
t['exp_max_adj'] = np.maximum(t.exp_ctrl_adj * t.n_systems / t.n_with_ctrl.clip(lower=1), t.exp_rand_adj)
t['p_adj'] = poisson.sf(t.obs_adj - 1, t.exp_max_adj.clip(lower=1e-3))
tt = t[t.species >= 3].copy()   # at least three species: lineage-only co-occurrence is not a pattern
mtest = len(tt)


def bh(p):
    o = np.argsort(p.to_numpy())
    q = np.empty(len(p))
    q[o] = np.minimum.accumulate((p.to_numpy()[o] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    return np.clip(q, 0, 1)


tt['q'] = bh(tt.p)
tt['q_adj'] = bh(tt.p_adj)
tt['enrich'] = tt.obs / tt.exp_max.clip(lower=1e-3)
tt['enrich_adj'] = tt.obs_adj / tt.exp_max_adj.clip(lower=1e-3)
tt = tt.sort_values('p')
t.to_csv(D / 'pattern_c_all_pairs.tsv', sep='\t', index=False)
tt.to_csv(D / 'pattern_c_tested.tsv', sep='\t', index=False)
cols = ['model', 'label', 'label_name', 'n_systems', 'obs', 'obs_inside', 'obs_adj', 'genomes', 'species',
        'species_adj', 'exp_max', 'enrich', 'q', 'exp_max_adj', 'enrich_adj', 'q_adj', 'label_total']
lines = [f'pairs (model, label) observed: {len(t)}; tested (>= 3 species): {mtest}; window q < 0.05: '
         f'{(tt.q < 0.05).sum()}; inside/adjacent q < 0.05: {(tt.q_adj < 0.05).sum()}',
         'window test, top 30:', tt[cols].head(30).to_string(),
         'inside/adjacent test, q_adj < 0.05:', tt[tt.q_adj < 0.05].sort_values('q_adj')[cols].to_string()]
(D / 'screen_c_summary.txt').write_text('\n'.join(lines) + '\n')
print('\n'.join(lines))
