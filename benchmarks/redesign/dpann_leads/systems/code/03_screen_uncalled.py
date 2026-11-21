#!/usr/bin/env python
"""Screen (b): partial DefenseFinder loci the caller did not call, completed by a slot-labeled gene.

Per multi-gene model M (min_genes_required >= 2):
  * hits = DPANN per-protein `defensefinder` rows (already GA / i-Evalue / coverage selected by Sharur) to any
    profile of M, minus proteins that are members of a called system of M
  * cluster hits on a contig when consecutive gene_index gaps are <= inter_gene_max_space + 1 (MacSyFinder rule)
  * a cluster fails quorum when satisfied mandatory < min_mandatory or satisfied mandatory+accessory < min_genes
  * completion: a slot-labeled gene within inter_gene_max_space + 1 genes of the cluster whose proposed family maps
    to a missing component (family map from 02) and whose addition satisfies the quorum

Control: for every partial cluster and missing component, the same family test around clusters / called systems
of unrelated models (no shared component family) in the same genome.
"""
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

W = Path('/Users/jacob/Documents/Sandbox/elsa_test/ELSA/benchmarks/redesign/dpann_leads/systems')
D = W / 'data'
DB = '/Users/jacob/Documents/Sandbox/Sharur/Sharur/data/dpann/sharur.duckdb'
mg = pd.read_csv(D / 'df_model_genes.tsv', sep='\t')
cf = pd.read_csv(D / 'component_families.tsv', sep='\t').fillna({'families': ''})
comp_fams = {(r.model, r.component): set(r.families.split(';')) - {''} for r in cf.itertuples()}
model_fams = {}
for (m, c), f in comp_fams.items():
    model_fams.setdefault(m, set()).update(f)
sysdf = pd.read_parquet(D / 'df_systems.parquet')
slot = pd.read_parquet(D / 'slot.parquet')

con = duckdb.connect()
con.execute(f"ATTACH '{DB}' AS d (READ_ONLY)")
hits = con.execute("""
  SELECT a.protein_id, a.accession AS profile, max(a.score) AS score, p.contig_id, p.gene_index, p.bin_id
  FROM d.annotations a JOIN d.proteins p USING (protein_id)
  WHERE a.source = 'defensefinder' GROUP BY 1, 2, 4, 5, 6
""").df()

prof_rows = []
for r in mg.itertuples():
    for p in r.profiles.split(';'):
        prof_rows.append((r.model, r.component, r.presence, p))
pmap = pd.DataFrame(prof_rows, columns=['model', 'component', 'presence', 'profile'])
info = mg.groupby('model').agg(igms=('inter_gene_max_space', 'first'), min_mand=('min_mandatory', 'first'),
                               min_genes=('min_genes', 'first'), n_comp=('n_mand_acc', 'first')).reset_index()
multi = info[info.min_genes >= 2].model
h = hits.merge(pmap[pmap.model.isin(multi)], on='profile')
called_members = set(zip(sysdf.model, sysdf.protein_id))
h = h[[(m, p) not in called_members for m, p in zip(h.model, h.protein_id)]]
in_any_call = set(sysdf.protein_id)
h['in_other_call'] = h.protein_id.isin(in_any_call)

info_d = info.set_index('model').to_dict('index')
allcomp = mg[mg.presence.isin(['mandatory', 'accessory'])].groupby('model').apply(
    lambda g: dict(zip(g.component, g.presence))).to_dict()

# vectorized clustering: new cluster at a model / contig change or a gene_index gap > inter_gene_max_space + 1
h = h.sort_values(['model', 'contig_id', 'gene_index']).reset_index(drop=True)
h['igms'] = h.model.map(lambda m: info_d[m]['igms'])
newc = ((h.model != h.model.shift()) | (h.contig_id != h.contig_id.shift()) |
        (h.gene_index - h.gene_index.shift() > h.igms + 1))
h['cidx'] = np.cumsum(newc.to_numpy())
cl = h.groupby('cidx').agg(model=('model', 'first'), contig=('contig_id', 'first'), genome=('bin_id', 'first'),
                           lo=('gene_index', 'min'), hi=('gene_index', 'max'), n_hits=('protein_id', 'nunique'),
                           any_in_other_call=('in_other_call', 'any'))
u = h.drop_duplicates(['cidx', 'component'])
cl['n_mand'] = u[u.presence == 'mandatory'].groupby('cidx').size().reindex(cl.index).fillna(0).astype(int)
cl['n_acc'] = u[u.presence == 'accessory'].groupby('cidx').size().reindex(cl.index).fillna(0).astype(int)
cl['igms'] = cl.model.map(lambda m: info_d[m]['igms'])
cl['min_mand'] = cl.model.map(lambda m: info_d[m]['min_mand'])
cl['min_genes'] = cl.model.map(lambda m: info_d[m]['min_genes'])
cl['quorum'] = (cl.n_mand >= cl.min_mand) & (cl.n_mand + cl.n_acc >= cl.min_genes)
cl = cl.reset_index()
cl['cluster_id'] = [f'{m}|{c}|{lo}-{hi}' for m, c, lo, hi in zip(cl.model, cl.contig, cl.lo, cl.hi)]


SB = {}
for c, g in slot.groupby('contig_id'):
    g = g.sort_values('gene_index')
    SB[c] = (g.gene_index.to_numpy(), g.protein_id.to_numpy(), g.label.to_numpy())


def win(contig, lo, hi, igms, exclude):
    t = SB.get(contig)
    if t is None:
        return []
    gi, pid, lab = t
    a, b = np.searchsorted(gi, lo - (igms + 1), 'left'), np.searchsorted(gi, hi + igms + 1, 'right')
    out = []
    for k in range(a, b):
        if pid[k] in exclude:
            continue
        x = int(gi[k])
        d = 0 if lo < x < hi else min(abs(x - lo), abs(x - hi))
        out.append((pid[k], x, lab[k], d))
    return out


# genome gene counts and per-genome label positions for the random-window control
gene_counts = con.execute("SELECT contig_id, count(*) n FROM d.proteins GROUP BY 1").df()
contig_n = dict(zip(gene_counts.contig_id, gene_counts.n))
contig_genome = dict(zip(slot.contig_id, slot.bin_id))
genome_ngenes = con.execute("SELECT bin_id, count(*) n FROM d.proteins GROUP BY 1").df().set_index('bin_id').n.to_dict()
part_all = cl[~cl.quorum].copy()
hp = h[h.cidx.isin(part_all.cidx)]
prot_by_c = hp.groupby('cidx').protein_id.agg(lambda x: ';'.join(sorted(set(x))))
mand_by_c = hp[hp.presence == 'mandatory'].groupby('cidx').component.agg(set)
acc_by_c = hp[hp.presence == 'accessory'].groupby('cidx').component.agg(set)
part_all['proteins'] = part_all.cidx.map(prot_by_c)
part_all['mand'] = [mand_by_c.get(i, set()) for i in part_all.cidx]
part_all['acc'] = [acc_by_c.get(i, set()) for i in part_all.cidx]
part_all['missing'] = [[c for c in allcomp.get(m, {}) if c not in a | b] for m, a, b in
                       zip(part_all.model, part_all.mand, part_all.acc)]
part_all['wg'] = [win(c.contig, c.lo, c.hi, c.igms, set(c.proteins.split(';'))) for c in part_all.itertuples()]
part_all['has_win'] = part_all.wg.map(len) > 0
n_part_win = int(part_all.has_win.sum())
part = part_all[part_all.has_win].copy()
rows = []
for c in part.itertuples():
    for pid, gi, lab, d in c.wg:
        for comp_name in c.missing:
            if lab in comp_fams.get((c.model, comp_name), set()):
                pres = allcomp[c.model][comp_name]
                mand = c.mand | ({comp_name} if pres == 'mandatory' else set())
                acc = c.acc | ({comp_name} if pres == 'accessory' else set())
                completes = (len(mand) >= c.min_mand) and (len(mand) + len(acc) >= c.min_genes)
                rows.append(dict(cluster_id=c.cluster_id, model=c.model, genome=c.genome, contig=c.contig,
                                 cluster_proteins=c.proteins, protein_id=pid, gene_index=gi, label=lab, dist=d,
                                 component=comp_name, presence=pres, completes=completes,
                                 any_in_other_call=c.any_in_other_call))
comp_ev = pd.DataFrame(rows)
if len(comp_ev):
    comp_ev = comp_ev.merge(slot[['protein_id', 'label_name', 'length', 'tier', 'list_type', 'votes_for',
                                  'supporting_genomes', 'vfrac', 'proposed_evalue']], on='protein_id', how='left')
comp_ev.to_csv(D / 'uncalled_completion_events.tsv', sep='\t', index=False)

# control: same missing-component families around unrelated clusters / called systems in the same genome
sy = sysdf.groupby('system_id').agg(model=('model', 'first'), genome=('genome_id', 'first'),
                                    contig=('contig_id', 'first'), lo=('gene_index', 'min'),
                                    hi=('gene_index', 'max')).reset_index()
loci_by_genome = {g: x for g, x in sy.groupby('genome')}
n_lab_genome = slot.groupby(['bin_id', 'label']).size().to_dict()
obs_rows = []
for c in part_all.itertuples():
    others = loci_by_genome.get(c.genome)
    if others is not None:
        mf = model_fams.get(c.model, set())
        others = others[(others.model != c.model) & [len(model_fams.get(m, set()) & mf) == 0 for m in others.model]]
    ng = genome_ngenes.get(c.genome, 1)
    for comp_name in c.missing:
        fams = comp_fams.get((c.model, comp_name), set())
        if not fams:
            continue
        hit = any(x[2] in fams for x in c.wg)
        cr = [] if others is None else [any(x[2] in fams for x in win(o.contig, o.lo, o.hi, c.igms, set()))
                                        for o in others.itertuples()]
        width = (c.hi - c.lo + 1) + 2 * (c.igms + 1)
        nf = sum(n_lab_genome.get((c.genome, f), 0) for f in fams)
        obs_rows.append(dict(cluster_id=c.cluster_id, model=c.model, component=comp_name, hit=int(hit),
                             ctrl_n=len(cr), ctrl_rate=np.mean(cr) if cr else np.nan,
                             random_window_rate=min(1.0, nf * width / ng)))
ob = pd.DataFrame(obs_rows)
ob.to_parquet(D / 'uncalled_obs_vs_control.parquet', index=False)

lines = []
P = lines.append
P(f'multi-gene DefenseFinder models with uncalled hit clusters: {cl.model.nunique()}; clusters {len(cl)}; '
  f'quorum met but uncalled (reimplementation residue): {cl.quorum.sum()}; partial (quorum failed): {len(part_all)}')
P(f'partial clusters with a slot-labeled gene in window: {n_part_win} (of {int((~cl.quorum).sum())})')
if len(comp_ev):
    P(f'completion events: {len(comp_ev)} rows; clusters gaining a missing component {comp_ev.cluster_id.nunique()};'
      f' clusters that become complete {comp_ev[comp_ev.completes].cluster_id.nunique()}')
o = ob.dropna(subset=['ctrl_rate'])
P(f'(cluster, missing component) pairs with family map: {len(ob)}; observed rate {ob.hit.mean():.4f} '
  f'({ob.hit.sum()}/{len(ob)}); with >=1 unrelated locus in genome: {len(o)}, observed {o.hit.mean():.4f}, '
  f'expected from unrelated called systems {o.ctrl_rate.sum():.1f} vs observed {o.hit.sum()}; '
  f'expected from random windows in the same genome {ob.random_window_rate.sum():.1f} vs observed {ob.hit.sum()}')
P('')
P('partial clusters by model (clusters / with gain / complete):')
t = cl[~cl.quorum].groupby('model').size().rename('partial').to_frame()
t['partial_with_slot'] = part.groupby('model').size()
if len(comp_ev):
    t['gain'] = comp_ev.groupby('model').cluster_id.nunique()
    t['complete'] = comp_ev[comp_ev.completes].groupby('model').cluster_id.nunique()
P(t.fillna(0).astype(int).sort_values('partial', ascending=False).head(60).to_string())
P('')
if len(comp_ev):
    P(comp_ev[['cluster_id', 'protein_id', 'label', 'label_name', 'component', 'completes', 'dist', 'length', 'tier',
               'list_type', 'votes_for', 'vfrac']].to_string())
(D / 'screen_b_summary.txt').write_text('\n'.join(lines) + '\n')
cl.to_parquet(D / 'uncalled_clusters.parquet', index=False)
part.drop(columns=['mand', 'acc', 'wg']).assign(missing=[';'.join(x) for x in part.missing]).to_parquet(
    D / 'uncalled_partial_with_slot.parquet', index=False)
print('\n'.join(lines))
