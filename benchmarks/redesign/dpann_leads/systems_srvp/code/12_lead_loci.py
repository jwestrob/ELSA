#!/usr/bin/env python
"""Per-locus detail for the lead completions and a label-specific adjacency control (biotite login node, serial).

Leads (system type, completing component): AbiE antitoxin and toxin, Hachiman HamA, PARIS AriB, CBASS cyclase,
HipA (PhiR5.1_HipAD and DS-6), plus routine comparators Type III RM Res and MazE.
Per locus: genome taxonomy (package genomes table), slot gene (gid, label, length, tier, support), partner hit genes
with the profiles they hit and bit scores, rerun best bits for the slot gene, whether the genome has a called system of
the same type elsewhere, distance of the locus to the contig end.
Label-specific control: for each lead label L, over strictly uncalled partial clusters of the type's models that miss
the component, observed = clusters with L within the window; expected = sum of the rate of L in same-width windows
around unrelated called systems in the same genome, and L count x width / genome genes (random).
Outputs data/lead_loci.tsv, data/lead_label_control.tsv, data/lead_models.tsv.
"""
from pathlib import Path

import numpy as np
import pandas as pd

RD = Path('/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111')
D = RD / 'dpann_leads/systems_srvp/data'
LEADS = [('AbiE', 'AbiEii__AbiEi_1'), ('AbiE', 'AbiEii__AbiEii'), ('Hachiman', 'Hachiman__HamA_1'),
         ('Paris', 'Paris_I__AriB_DUF4435_I'), ('Paris', 'Paris_II__AriB_DUF4435_II'), ('CBASS', 'CBASS__Cyclase_SMODS'),
         ('PhiR5.1_HipAD', 'PhiR5.1_HipAD__HipA'), ('DS-6', 'DS-6__DS-6A'),
         ('RM_Type_III', 'RM_Type_III__Type_III_REases_FAM_0.einsi_trimmed'), ('MazEF', 'MazEF__MazE')]
mg = pd.read_csv(D / 'model_genes.tsv', sep='\t')
cf = pd.read_csv(D / 'component_families.tsv', sep='\t').fillna({'families': ''})
info = pd.read_csv(D / 'model_info.tsv', sep='\t').set_index('model_key')
ev = pd.read_parquet(D / 'screen_b_events.parquet')
cl = pd.read_parquet(D / 'screen_b_clusters.parquet')
sy = pd.read_parquet(D / 'systems.parquet')
hits = pd.read_parquet(D / 'hits.parquet')
slot = pd.read_parquet(D / 'slot.parquet')
sc = pd.read_csv(D / 'rerun/test_event_scores.tsv', sep='\t')
genes = pd.read_parquet(RD / 'ws_srvp_bins/package/universe/genes.parquet',
                        columns=['gid', 'genome', 'contig_code', 'position_index', 'strand', 'length'])
contigs = pd.read_parquet(RD / 'ws_srvp_bins/package/universe/contigs.parquet', columns=['contig_code', 'n_genes'])
ncg = dict(zip(contigs.contig_code, contigs.n_genes))
gen = pd.read_parquet(RD / 'ws_srvp_bins/package/universe/genomes.parquet',
                      columns=['genome', 'genome_name', 'domain', 'phylum', 'class', 'ord', 'genus', 'species',
                               'completeness', 'contamination', 'platform', 'n_contigs'])
typ = lambda m: m.split('/')[1]
ev['stype'] = ev.model_key.map(typ)
cl['stype'] = cl.model_key.map(typ)
sy['stype'] = sy.model_key.map(typ)
lead_models = mg[mg.model_key.map(typ).isin({t for t, _ in LEADS})]
lead_models[['model_key', 'component', 'presence', 'profiles', 'min_mandatory', 'min_genes', 'inter_gene_max_space']] \
    .to_csv(D / 'lead_models.tsv', sep='\t', index=False)
hp = hits.groupby('gid').apply(lambda g: ';'.join(f'{p}:{b:.0f}' for p, b in zip(g.profile, g.bitscore)))
called_type = sy.groupby('stype').genome.apply(set).to_dict()
rows = []
for t, comp in LEADS:
    e = ev[(ev.stype == t) & (ev.component == comp) & ev.completes & ~ev.overlaps_call]
    e = e.sort_values(['dist', 'support'], ascending=[True, False]).drop_duplicates('gid')
    for r in e.itertuples():
        partners = [int(x) for x in r.cluster_prots.split(';')]
        s = sc[(sc.gid == r.gid) & (sc.component == comp)]
        mk = ev[(ev.gid == r.gid) & (ev.component == comp)].model_key
        rows.append(dict(stype=t, component=comp, models=';'.join(sorted(set(m.split('/')[-1] for m in mk))),
                         genome=r.genome, contig_code=r.contig_code, gid=r.gid, position_index=r.position_index,
                         label=r.label, label_name=r.label_name, length=r.length, tier=r.tier,
                         support=r.support, votes_for=r.votes_for, dist=r.dist, partners=';'.join(map(str, partners)),
                         partner_hits=' | '.join(hp.get(p, '') for p in partners),
                         rerun_bits=s.bitscore.max() if len(s) else None, rerun_profile=s.best_profile.iloc[0] if len(s) else '',
                         genome_has_type_called=r.genome in called_type.get(t, set()),
                         to_contig_end=int(min(r.position_index, ncg[r.contig_code] - 1 - r.position_index)),
                         contig_genes=ncg[r.contig_code]))
ll = pd.DataFrame(rows).merge(gen, on='genome', how='left')
ll.to_csv(D / 'lead_loci.tsv', sep='\t', index=False)

# label-specific adjacency control
SB = {}
for (g, c), x in slot.groupby(['genome', 'contig_code']):
    x = x.sort_values('position_index')
    SB[(g, c)] = (x.position_index.to_numpy(), x.label.to_numpy())


def labs(genome, contig, lo, hi, igms):
    t = SB.get((genome, contig))
    if t is None:
        return set()
    pos, lab = t
    a, b = np.searchsorted(pos, lo - (igms + 1), 'left'), np.searchsorted(pos, hi + igms + 1, 'right')
    return set(lab[a:b])


comp_fams = {(r.model_key, r.component): set(r.families.split(';')) - {''} for r in cf.itertuples()}
model_fams = {}
for (m, c), f in comp_fams.items():
    model_fams.setdefault(m, set()).update(f)
ngen = genes.groupby('genome').size().to_dict()
nlab = slot.groupby(['genome', 'label']).size().to_dict()
by_g = {g: x for g, x in sy.groupby('genome')}
crow = []
for t, comp in LEADS:
    lab_set = sorted(set(ll[(ll.stype == t) & (ll.component == comp)].label))
    p = cl[(cl.stype == t) & ~cl.quorum & ~cl.overlaps_call & cl.missing.str.split(';').map(lambda x: comp in x)]
    p = p.sort_values('cluster_id').drop_duplicates(['genome', 'contig_code', 'lo', 'hi'])
    for L in lab_set:
        obs = exp_u = exp_r = 0.0
        n_ctrl = 0
        for r in p.itertuples():
            w = (r.hi - r.lo + 1) + 2 * (r.igms + 1)
            obs += L in labs(r.genome, r.contig_code, r.lo, r.hi, r.igms)
            o = by_g.get(r.genome)
            if o is not None:
                mf = model_fams.get(r.model_key, set())
                o = o[(o.stype != t) & np.array([len(model_fams.get(m, set()) & mf) == 0 for m in o.model_key], dtype=bool)]
                if len(o):
                    n_ctrl += 1
                    exp_u += np.mean([L in labs(x.genome, x.contig_code, x.lo, x.hi, r.igms) for x in o.itertuples()])
            exp_r += min(1.0, nlab.get((r.genome, L), 0) * w / ngen[r.genome])
        crow.append(dict(stype=t, component=comp, label=L, partial_loci=len(p), loci_with_unrelated_call=n_ctrl,
                         observed=int(obs), expected_unrelated=round(exp_u, 2), expected_random=round(exp_r, 2)))
cc = pd.DataFrame(crow)
cc.to_csv(D / 'lead_label_control.tsv', sep='\t', index=False)
pd.set_option('display.width', 300)
pd.set_option('display.max_colwidth', 60)
print(cc.to_string(index=False))
print(lead_models[lead_models.presence.isin(['mandatory', 'accessory'])][['model_key', 'component', 'presence', 'min_mandatory',
                                                                           'min_genes']].to_string(index=False))
print(ll[ll.stype.isin(['Hachiman', 'Paris', 'CBASS', 'PhiR5.1_HipAD', 'DS-6'])][
    ['stype', 'models', 'gid', 'label_name', 'length', 'tier', 'support', 'partner_hits', 'rerun_bits',
     'genome_has_type_called', 'phylum', 'class', 'genus', 'completeness', 'to_contig_end']].to_string(index=False))
