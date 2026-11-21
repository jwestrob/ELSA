#!/usr/bin/env python
"""Per-protein hits of the callers' profiles, selected by the callers' own rules (biotite login node, serial).

Input: search/s000..s007/user_hmms_hits_df.tsv (Aksha --cut_ga, one row per reported domain).
Selection as MacSyFinder 2.1.4 applies it in the srvp runs (04_callers.py):
  full-sequence score >= GA1 and domain score >= GA2 (applied by --cut_ga);
  domain i-Evalue <= 1e-3 at the genome level: HMMER's i-Evalue = P x domZ, domZ = number of targets reported for
    that profile; MacSyFinder searched one genome per run, so i_evalue_genome = i_evalue x n_reported_in_genome /
    n_reported_in_shard (same P, the genome's own domZ);
  profile coverage (hmm_to - hmm_from + 1) / LENG >= 0.4 (DefenseFinder --coverage-profile 0.4) or 0.5 (TXSScan default).
Best selected domain per (protein, profile). Output data/hits.parquet: gid, genome, contig_code, position_index,
strand, caller, profile, bitscore, dom_bitscore, i_evalue_genome, cov, ali_from, ali_to.
"""
from pathlib import Path

import pandas as pd

RD = Path('/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111')
W = RD / 'dpann_leads/systems_srvp'
pp = pd.read_csv(W / 'data/picked_profiles.tsv', sep='\t')
genes = pd.read_parquet(RD / 'ws_srvp_bins/package/universe/genes.parquet',
                        columns=['gid', 'genome', 'contig_code', 'position_index', 'strand', 'length'])
parts = []
for s in range(8):
    d = W / f'search/s{s:03d}'
    assert (d / 'STATUS').exists(), d
    x = pd.read_csv(d / 'user_hmms_hits_df.tsv', sep='\t')
    x['shard'] = s
    parts.append(x)
h = pd.concat(parts, ignore_index=True)
n_raw = len(h)
name2 = pp.set_index('hmm_name')
h = h[h.hmm_name.isin(name2.index)].copy()
assert len(h) == n_raw, 'unknown profile names'
h['profile'] = h.hmm_name.map(name2.profile)
h['caller'] = h.hmm_name.map(name2.caller)
h['LENG'] = h.hmm_name.map(name2.LENG)
h['GA1'] = h.hmm_name.map(name2.GA1)
h['GA2'] = h.hmm_name.map(name2.GA2)
h = h.rename(columns={'sequence_id': 'gid'}).merge(genes, on='gid', how='left')
assert h.genome.notna().all()
n_shard = h.groupby(['shard', 'profile']).gid.nunique().rename('n_shard')
n_gen = h.groupby(['genome', 'profile']).gid.nunique().rename('n_genome')
h = h.join(n_shard, on=['shard', 'profile']).join(n_gen, on=['genome', 'profile'])
h['i_evalue_genome'] = h.i_evalue * h.n_genome / h.n_shard
h['cov'] = (h.hmm_to - h.hmm_from + 1) / h.LENG
h['cov_min'] = h.caller.map({'DefenseFinder': 0.4, 'TXSScan': 0.5})
h['ga_ok'] = (h.bitscore >= h.GA1) & (h.dom_bitscore >= h.GA2)
h['ie_ok'] = h.i_evalue_genome <= 1e-3
h['cov_ok'] = h['cov'] >= h.cov_min
h['sel'] = h.ga_ok & h.ie_ok & h.cov_ok
print(f'domain rows {len(h)}; GA ok {h.ga_ok.mean():.4f}; of GA ok: i-Evalue ok {h[h.ga_ok].ie_ok.mean():.4f}, '
      f'coverage ok {h[h.ga_ok].cov_ok.mean():.4f}; selected rows {h.sel.sum()}')
s = h[h.sel].sort_values('dom_bitscore', ascending=False).drop_duplicates(['gid', 'profile'])
cols = ['gid', 'genome', 'contig_code', 'position_index', 'strand', 'length', 'caller', 'profile', 'bitscore',
        'dom_bitscore', 'i_evalue_genome', 'cov', 'ali_from', 'ali_to']
s[cols].sort_values(['gid', 'profile']).to_parquet(W / 'data/hits.parquet', index=False)
allp = h.sort_values('dom_bitscore', ascending=False).drop_duplicates(['gid', 'profile'])
allp[cols + ['ga_ok', 'ie_ok', 'cov_ok', 'sel']].to_parquet(W / 'data/hits_reported.parquet', index=False)
print(f'selected (protein, profile) pairs {len(s)}; proteins {s.gid.nunique()}; profiles hit {s.profile.nunique()} '
      f'of {len(pp)}; genomes {s.genome.nunique()}')
