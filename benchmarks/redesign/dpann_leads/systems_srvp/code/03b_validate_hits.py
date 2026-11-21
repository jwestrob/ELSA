#!/usr/bin/env python
"""Validate the reconstructed per-protein hits (03_hits.py) against the callers' own member hits: for every called
member of a small multi-gene model whose profile was searched, is (gid, profile) among the selected hits?
Also: the same for hits listed in rejected_candidates.tsv (caller-selected hits of failed candidates)."""
from pathlib import Path
import pandas as pd
RD = Path('/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111')
D = RD / 'dpann_leads/systems_srvp/data'
pp = pd.read_csv(D / 'picked_profiles.tsv', sep='\t')
hits = pd.read_parquet(D / 'hits.parquet')
rep = pd.read_parquet(D / 'hits_reported.parquet')
cm = pd.read_parquet(D / 'call_members.parquet')
rj = pd.read_parquet(D / 'rejected.parquet')
L = []
for nm, x in [('called members', cm), ('rejected-candidate hits', rj)]:
    x = x.assign(caller=x.run.map({'DefenseFinder': 'DefenseFinder', 'DefenseFinder_RM': 'DefenseFinder', 'TXSScan': 'TXSScan'}))
    x = x.merge(pp[['caller', 'profile']], on=['caller', 'profile']).drop_duplicates(['gid', 'caller', 'profile'])
    sel = set(zip(hits.gid, hits.caller, hits.profile))
    rp = rep.set_index(['gid', 'caller', 'profile'])
    k = [(g, c, p) in sel for g, c, p in zip(x.gid, x.caller, x.profile)]
    x['selected'] = k
    miss = x[~x.selected]
    why = []
    for r in miss.itertuples():
        key = (r.gid, r.caller, r.profile)
        if key not in rp.index:
            why.append('not reported')
        else:
            z = rp.loc[key]
            why.append('GA' if not z.ga_ok else ('i-Evalue' if not z.ie_ok else ('coverage' if not z.cov_ok else 'other')))
    L.append(f'{nm} with a searched profile: {len(x)}; reproduced {int(x.selected.sum())} ({x.selected.mean():.4f}); '
             f'missing by reason {pd.Series(why).value_counts().to_dict()}')
(D / 'hits_validation.txt').write_text('\n'.join(L) + '\n')
print('\n'.join(L))
