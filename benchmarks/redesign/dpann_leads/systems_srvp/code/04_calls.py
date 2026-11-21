#!/usr/bin/env python
"""Parse the callers' own per-genome outputs for srvp (biotite login node, serial).

Sources: ws_srvp_bins/work_callers/out/{DefenseFinder,DefenseFinder_RM,TXSScan}/<genome>/best_solution.tsv and
rejected_candidates.tsv (MacSyFinder 2.1.4; DefenseFinder 3.1.0, TXSScan 1.1.1). Protein ids c<contig_code>_<k>,
k = position_index + 1 (04_callers.py).
Outputs (data/):
  call_members.parquet   one row per member hit of a called system: run, genome, sys_id, model_key, gid,
                         profile (gene_name), component (hit_gene_ref), hit_status, locus_num, scores, coverage,
                         hit_begin_match / hit_end_match
  rejected.parquet       one row per hit of a rejected candidate: run, genome, candidate_id, model_key, gid,
                         profile (gene_name), component (function), reasons
"""
from pathlib import Path

import pandas as pd

RD = Path('/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111')
W = RD / 'dpann_leads/systems_srvp'
OUTR = RD / 'ws_srvp_bins/work_callers/out'
genes = pd.read_parquet(RD / 'ws_srvp_bins/package/universe/genes.parquet',
                        columns=['gid', 'contig_code', 'position_index'])
mid2gid = dict(zip('c' + genes.contig_code.astype(str) + '_' + (genes.position_index + 1).astype(str), genes.gid))


def read_tab(path):
    hdr, rows = None, []
    for line in open(path):
        if line.startswith('#') or not line.strip():
            continue
        f = line.rstrip('\n').split('\t')
        if hdr is None:
            hdr = f
            continue
        rows.append(dict(zip(hdr, f)))
    return rows


def mkey(run, fqn):
    p = fqn.split('/')
    # defense-finder-models/DefenseFinder/<cat>/<model> ; defense-finder-models/RM/<cat>/<model> ; TXSScan/<dom>/<model>
    if run.startswith('DefenseFinder'):
        return '/'.join(p[1:])
    return 'TXSScan/' + '/'.join(p[1:])


mem, rej = [], []
for run in ('DefenseFinder', 'DefenseFinder_RM', 'TXSScan'):
    for gd in sorted((OUTR / run).iterdir()):
        for r in read_tab(gd / 'best_solution.tsv'):
            r.update(run=run, genome=gd.name)
            mem.append(r)
        rp = gd / 'rejected_candidates.tsv'
        if rp.exists():
            for r in read_tab(rp):
                r.update(run=run, genome=gd.name)
                rej.append(r)
m = pd.DataFrame(mem)
m['model_key'] = [mkey(a, b) for a, b in zip(m.run, m.model_fqn)]
m['gid'] = m.hit_id.map(mid2gid)
m = m.rename(columns={'gene_name': 'profile', 'hit_gene_ref': 'component'})
for c in ('hit_score', 'hit_i_eval', 'hit_profile_cov', 'hit_begin_match', 'hit_end_match', 'locus_num', 'hit_pos'):
    m[c] = pd.to_numeric(m[c])
assert m.gid.notna().all()
m['gid'] = m.gid.astype('int64')
m[['run', 'genome', 'sys_id', 'model_key', 'gid', 'hit_id', 'profile', 'component', 'hit_status', 'locus_num',
   'sys_wholeness', 'hit_score', 'hit_i_eval', 'hit_profile_cov', 'hit_begin_match', 'hit_end_match']].to_parquet(
    W / 'data/call_members.parquet', index=False)
j = pd.DataFrame(rej)
j['model_key'] = [mkey(a, b) for a, b in zip(j.run, j.model_fqn)]
j['gid'] = j.hit_id.map(mid2gid).astype('int64')
j = j.rename(columns={'gene_name': 'profile', 'function': 'component'})
j[['run', 'genome', 'candidate_id', 'model_key', 'cluster_id', 'gid', 'profile', 'component', 'reasons']].to_parquet(
    W / 'data/rejected.parquet', index=False)
print('call member rows', len(m), 'systems', m.groupby(['run', 'genome']).sys_id.nunique().sum(),
      'by run', m.groupby('run').apply(lambda g: g[['genome', 'sys_id']].drop_duplicates().shape[0]).to_dict())
print('rejected rows', len(j), 'candidates', j[['run', 'genome', 'candidate_id']].drop_duplicates().shape[0])
