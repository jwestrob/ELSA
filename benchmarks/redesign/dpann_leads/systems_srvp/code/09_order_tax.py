#!/usr/bin/env python
"""Gene order, strand and taxonomy for completed uncalled loci of small systems (biotite login node, serial).

Per completion event (screen b, completes, strict = no called member in the cluster): the slot gene against the nearest
cluster hit gene (another component): same strand, slot gene 5' of the partner in the partner's transcription direction,
adjacent (1 gene apart). The same three statistics for the matching component pair in called systems of the same model
(member of the completing component against the nearest member of another component).
Genomes per model before (called) and after (called or strictly completed), by phylum from the package genomes table
(GTDB r232 / r220 labels as given), with the phylum's genome count as denominator.
Outputs: data/order_by_component.tsv, data/genomes_by_model.tsv, data/genomes_by_model_phylum.tsv,
data/completed_loci.tsv (one row per strict completed locus, best event).
"""
from pathlib import Path

import numpy as np
import pandas as pd

RD = Path('/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111')
D = RD / 'dpann_leads/systems_srvp/data'
info = pd.read_csv(D / 'model_info.tsv', sep='\t').set_index('model_key')
ev = pd.read_parquet(D / 'screen_b_events.parquet')
cm = pd.read_parquet(D / 'call_members.parquet')
sy = pd.read_parquet(D / 'systems.parquet')
genes = pd.read_parquet(RD / 'ws_srvp_bins/package/universe/genes.parquet',
                        columns=['gid', 'genome', 'contig_code', 'position_index', 'strand'])
gi = genes.set_index('gid')
gen = pd.read_parquet(RD / 'ws_srvp_bins/package/universe/genomes.parquet',
                      columns=['genome', 'domain', 'phylum', 'class', 'ord', 'family', 'genus', 'species',
                               'completeness', 'contamination', 'platform', 'n_contigs'])

c = ev[ev.completes & ~ev.overlaps_call].copy()
c = c.sort_values(['cluster_id', 'dist', 'support'], ascending=[True, True, False]).drop_duplicates('cluster_id')


def rel(a, b):
    """a = completing gene, b = partner gene (gids)."""
    A, B = gi.loc[a], gi.loc[b]
    same = int(A.strand == B.strand)
    up = int(A.position_index < B.position_index) if B.strand == 1 else int(A.position_index > B.position_index)
    return same, up, int(abs(int(A.position_index) - int(B.position_index)) == 1)


rows = []
for r in c.itertuples():
    prots = [int(x) for x in r.cluster_prots.split(';')]
    near = min(prots, key=lambda p: abs(int(gi.loc[p].position_index) - r.position_index))
    s, u, a = rel(r.gid, near)
    rows.append(dict(cluster_id=r.cluster_id, same_strand=s, upstream=u, adjacent=a, partner=near))
c = c.merge(pd.DataFrame(rows), on='cluster_id')
c = c.merge(gen, on='genome', how='left')
c.to_csv(D / 'completed_loci.tsv', sep='\t', index=False)

# called systems: same statistics for (model, component) pairs
cm['position_index'] = cm.gid.map(gi.position_index)
cm['contig_code'] = cm.gid.map(gi.contig_code)
cm['sys_key'] = cm.run + ':' + cm.genome + ':' + cm.sys_id
want = set(zip(c.model_key, c.component))
crow = []
for k, g in cm[cm.model_key.isin(set(c.model_key))].groupby('sys_key'):
    m = g.model_key.iloc[0]
    for comp in set(g.component):
        if (m, comp) not in want:
            continue
        a = g[g.component == comp].iloc[0]
        oth = g[(g.component != comp) & (g.contig_code == a.contig_code)]
        if not len(oth):
            continue
        b = oth.iloc[np.argmin(np.abs(oth.position_index.to_numpy() - a.position_index))]
        s, u, ad = rel(int(a.gid), int(b.gid))
        crow.append(dict(model_key=m, component=comp, same_strand=s, upstream=u, adjacent=ad))
cr = pd.DataFrame(crow)
o1 = c.groupby(['model_key', 'component']).agg(loci=('cluster_id', 'size'), genes=('gid', 'nunique'),
                                               genomes=('genome', 'nunique'), same_strand=('same_strand', 'mean'),
                                               upstream_of_same=('upstream', 'mean'), adjacent=('adjacent', 'mean'))
o2 = cr.groupby(['model_key', 'component']).agg(called_n=('same_strand', 'size'), called_same=('same_strand', 'mean'),
                                                called_upstream=('upstream', 'mean'), called_adjacent=('adjacent', 'mean'))
ob = o1.join(o2, how='left').reset_index()
ob.round(3).to_csv(D / 'order_by_component.tsv', sep='\t', index=False)

# genomes per model before / after
cal = sy.groupby('model_key').genome.apply(set).to_dict()
comp_g = c.groupby('model_key').genome.apply(set).to_dict()
grow, prow = [], []
den = gen.groupby('phylum').genome.nunique()
for m in sorted(comp_g):
    before = cal.get(m, set())
    after = before | comp_g[m]
    grow.append(dict(model_key=m, cls=info.loc[m, 'cls'], genomes_called=len(before), genomes_after=len(after),
                     new_genomes=len(after - before), completed_loci=int((c.model_key == m).sum())))
    gb = gen[gen.genome.isin(before)].groupby('phylum').genome.nunique()
    ga = gen[gen.genome.isin(after)].groupby('phylum').genome.nunique()
    for ph in ga.index:
        prow.append(dict(model_key=m, phylum=ph, phylum_genomes=int(den[ph]), called=int(gb.get(ph, 0)),
                         after=int(ga[ph])))
pd.DataFrame(grow).sort_values('new_genomes', ascending=False).to_csv(D / 'genomes_by_model.tsv', sep='\t', index=False)
pd.DataFrame(prow).to_csv(D / 'genomes_by_model_phylum.tsv', sep='\t', index=False)
pd.set_option('display.width', 250)
print(ob.sort_values('loci', ascending=False).head(30).round(3).to_string(index=False))
print(pd.DataFrame(grow).sort_values('new_genomes', ascending=False).head(25).to_string(index=False))
