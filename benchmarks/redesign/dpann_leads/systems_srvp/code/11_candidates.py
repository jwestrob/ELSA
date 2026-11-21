#!/usr/bin/env python
"""Candidate table: strictly uncalled loci of small systems completed by a slot gene, grouped by system type
(DefenseFinder category, e.g. AbiE, CBASS, RM_Type_II) and completing component role (biotite login node, serial).

Per (type, component, label): unique completing genes / loci / genomes; genomes of that type before (called) and after
(called or completed); tiers; support; caller-profile rerun (any hit at E <= 10, GA) with label-matched controls
(neg_same = same label away from the model's loci, neg_other = other labels, length-matched); adjacency control from
screen (b) (observed label hits vs expectation around unrelated called systems and in random windows, all partial
clusters of the type's models missing that component); gene order (completed_loci.tsv).
Output data/candidates.tsv.
"""
from pathlib import Path

import pandas as pd

RD = Path('/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111')
D = RD / 'dpann_leads/systems_srvp/data'
info = pd.read_csv(D / 'model_info.tsv', sep='\t').set_index('model_key')
ev = pd.read_parquet(D / 'screen_b_events.parquet')
ob = pd.read_parquet(D / 'screen_b_obs_control.parquet')
sc = pd.read_csv(D / 'rerun/test_event_scores.tsv', sep='\t')
ct = pd.read_csv(D / 'rerun/control_scores.tsv', sep='\t')
cl = pd.read_csv(D / 'completed_loci.tsv', sep='\t')
sy = pd.read_parquet(D / 'systems.parquet')
gen = pd.read_parquet(RD / 'ws_srvp_bins/package/universe/genomes.parquet', columns=['genome', 'phylum'])
ph = dict(zip(gen.genome, gen.phylum))
typ = lambda m: m.split('/')[1]
for x in (ev, ob, sc, cl, sy):
    x['stype'] = x.model_key.map(typ)
ev = ev[ev.cls.str.endswith('small') & ev.completes & ~ev.overlaps_call]
called_g = sy.groupby('stype').genome.apply(set).to_dict()
rows = []
for (t, comp, lab), g in ev.groupby(['stype', 'component', 'label']):
    genes = set(g.gid)
    s = sc[sc.gid.isin(genes) & (sc.component == comp)].groupby('gid').agg(a=('any_hit', 'max'), ga=('pass_ga', 'max'),
                                                                            bits=('bitscore', 'max'), GA1=('GA1', 'max'))
    ns = ct[(ct.set == 'neg_same') & (ct.test_label == lab)]
    no = ct[(ct.set == 'neg_other') & (ct.test_label == lab)]
    o = ob[(ob.stype == t) & (ob.component == comp) & ~ob.overlaps_call].dropna(subset=['ctrl_rate'])
    o = o.drop_duplicates('cluster_id')
    c = cl[(cl.stype == t) & (cl.component == comp) & (cl.label == lab)].drop_duplicates('gid')
    before = called_g.get(t, set())
    after = before | set(g.genome)
    rows.append(dict(stype=t, component=comp, label=lab, label_name=g.label_name.iloc[0],
                     models=';'.join(sorted(set(m.split('/')[-1] for m in g.model_key))), genes=len(genes),
                     genomes=g.genome.nunique(), genomes_type_called=len(before), genomes_type_after=len(after),
                     new_type_genomes=len(after - before),
                     phyla=';'.join(f'{k}:{v}' for k, v in pd.Series([ph[x] for x in set(g.genome)]).value_counts().items()),
                     med_len=g.drop_duplicates('gid').length.median(),
                     T1=int((g.drop_duplicates('gid').tier == 'T1').sum()),
                     support_med=round(g.drop_duplicates('gid').support.median(), 3),
                     rerun_hit=int(s.a.sum()), rerun_ga=int(s.ga.sum()), rerun_n=len(s),
                     med_bits=round(s.bits.median(), 1) if len(s) else None, GA1=s.GA1.max() if len(s) else None,
                     ns_n=len(ns), ns_hit=int(ns.any_hit.sum()), no_n=len(no), no_hit=int(no.any_hit.sum()),
                     adj_obs=int(o.hit.sum()), adj_exp_unrel=round(o.ctrl_rate.sum(), 1),
                     adj_exp_random=round(o.random_rate.sum(), 1), partial_pairs=len(o),
                     same_strand=round(c.same_strand.mean(), 3), upstream=round(c.upstream.mean(), 3),
                     adjacent=round(c.adjacent.mean(), 3)))
t = pd.DataFrame(rows).sort_values('genes', ascending=False)
t.to_csv(D / 'candidates.tsv', sep='\t', index=False)
pd.set_option('display.width', 300)
pd.set_option('display.max_colwidth', 30)
print(t.drop(columns=['phyla']).to_string(index=False))
