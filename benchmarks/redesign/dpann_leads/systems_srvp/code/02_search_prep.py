#!/usr/bin/env python
"""Pick the caller's own profiles for the per-protein hit search (biotite login node, serial).

Picked: every profile (primary + exchangeables) of the small multi-gene models the srvp callers ran:
  DefenseFinder 3.1.0 families DefenseFinder and RM, and TXSScan 1.1.1, with min_genes_required >= 2 and
  <= 4 mandatory + accessory components. Large models (> 4 components) are counted from the callers' own outputs
  (best_solution / rejected_candidates) and not searched.
Writes hmm/picked/<profile>.hmm (byte copies; one file each, because the set mixes HMMER3/b and HMMER3/f files and a
concatenated file fails to parse) and data/picked_profiles.tsv (NAME -> file, LENG, GA).
Why a search: the callers' per-protein hmmer outputs were deleted after the run (04_callers.py keeps only the solution
tables), and single-hit partial loci of multi-gene models never reach rejected_candidates.tsv.
"""
import shutil
from pathlib import Path

import pandas as pd

RD = Path('/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111')
W = RD / 'dpann_leads/systems_srvp'
mg = pd.read_csv(W / 'data/model_genes.tsv', sep='\t')
pr = pd.read_csv(W / 'data/profiles.tsv', sep='\t')
info = mg.groupby('model_key').agg(caller=('caller', 'first'), n_comp=('n_mand_acc', 'first'),
                                   min_genes=('min_genes', 'first')).reset_index()
small_multi = set(info[(info.min_genes >= 2) & (info.n_comp <= 4)].model_key)
sel = mg[mg.model_key.isin(small_multi)]
want = sorted(set((r.caller, p) for r in sel.itertuples() for p in r.profiles.split(';')))
pr = pr.set_index(['caller', 'profile'])
HD = W / 'hmm/picked'
if HD.exists():
    shutil.rmtree(HD)
HD.mkdir(parents=True)
(W / 'hmm/picked.hmm').unlink(missing_ok=True)
rows = []
for caller, p in want:
    r = pr.loc[(caller, p)]
    txt = open(r.path).read()
    name = next(l.split()[1] for l in txt.splitlines() if l.startswith('NAME'))
    shutil.copyfile(r.path, HD / f'{p}.hmm')
    rows.append(dict(caller=caller, profile=p, hmm_name=name, LENG=r.LENG, GA1=r.GA1, GA2=r.GA2,
                     version=txt.split('\n', 1)[0][:8]))
pp = pd.DataFrame(rows)
assert pp.hmm_name.is_unique, pp[pp.hmm_name.duplicated(keep=False)]
pp.to_csv(W / 'data/picked_profiles.tsv', sep='\t', index=False)
print('small multi-gene models', len(small_multi), info[info.model_key.isin(small_multi)].caller.value_counts().to_dict())
print('picked profiles', len(pp), pp.caller.value_counts().to_dict(), 'total LENG', int(pp.LENG.sum()),
      'name != file', int((pp.hmm_name != pp.profile).sum()))
