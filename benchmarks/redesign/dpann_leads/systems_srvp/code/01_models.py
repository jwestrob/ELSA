#!/usr/bin/env python
"""Parse the caller's own model definitions as run on srvp (biotite, login node, serial).

DefenseFinder: defense-finder-models 3.1.0, families DefenseFinder and RM (the two passes the srvp callers ran;
  ws_srvp_bins/code/04_callers.py). TXSScan 1.1.1 (~/.macsyfinder/models/TXSScan), 'all'.
Outputs ($RD/dpann_leads/systems_srvp/data/):
  model_genes.tsv   one row per model component: caller, family, category, model, component (primary profile),
                    presence, profiles (primary + exchangeables), loner, multi_system, model quorum, inter_gene_max_space
  profiles.tsv      every profile file referenced: caller, profile, path, LENG, GA1, GA2 (None when absent)
"""
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

RD = Path('/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111')
OUT = RD / 'dpann_leads/systems_srvp/data'
DF = RD / 'ws_srvp_bins/models/defense-finder-models'
TX = Path('/home/jwestrob/.macsyfinder/models/TXSScan')
SETS = [('DefenseFinder', 'DefenseFinder', DF / 'definitions/DefenseFinder', DF / 'profiles'),
        ('DefenseFinder', 'RM', DF / 'definitions/RM', DF / 'profiles'),
        ('TXSScan', 'TXSScan', TX / 'definitions', TX / 'profiles')]

rows, prof_rows = [], {}
for caller, fam, ddir, pdir in SETS:
    for xml in sorted(ddir.rglob('*.xml')):
        root = ET.parse(xml).getroot()
        igms = int(root.get('inter_gene_max_space', '5'))
        minm = int(root.get('min_mandatory_genes_required', '1'))
        ming = int(root.get('min_genes_required', '1'))
        genes = root.findall('gene')
        n_ma = sum(g.get('presence', 'mandatory') in ('mandatory', 'accessory') for g in genes)
        n_mand = sum(g.get('presence', 'mandatory') == 'mandatory' for g in genes)
        maxg = root.get('max_nb_genes')
        for g in genes:
            prim = g.get('name')
            exch = [e.get('name') for e in g.findall('exchangeables/gene')]
            rows.append(dict(caller=caller, family=fam, category=str(xml.parent.relative_to(ddir)), model=xml.stem,
                             model_key=f'{fam}/{xml.parent.relative_to(ddir)}/{xml.stem}',
                             component=prim, presence=g.get('presence', 'mandatory'),
                             profiles=';'.join([prim] + exch), loner=g.get('loner', 'false'),
                             multi_system=g.get('multi_system', 'false'), gene_igms=g.get('inter_gene_max_space'),
                             inter_gene_max_space=igms, min_mandatory=minm, min_genes=ming, n_mand_acc=n_ma,
                             n_mandatory=n_mand, max_nb_genes=int(maxg) if maxg else n_ma))
            for p in [prim] + exch:
                if (caller, p) in prof_rows:
                    continue
                path = pdir / f'{p}.hmm'
                leng = ga1 = ga2 = None
                if path.exists():
                    for line in open(path):
                        if line.startswith('LENG'):
                            leng = int(line.split()[1])
                        elif line.startswith('GA'):
                            x = line.split()[1:3]
                            ga1, ga2 = float(x[0]), float(x[1].rstrip(';'))
                        elif line.startswith('HMM '):
                            break
                prof_rows[(caller, p)] = dict(caller=caller, profile=p, path=str(path), exists=path.exists(),
                                              LENG=leng, GA1=ga1, GA2=ga2)
mg = pd.DataFrame(rows)
pr = pd.DataFrame(prof_rows.values())
mg.to_csv(OUT / 'model_genes.tsv', sep='\t', index=False)
pr.to_csv(OUT / 'profiles.tsv', sep='\t', index=False)
info = mg.groupby(['caller', 'family', 'model_key']).agg(n_comp=('n_mand_acc', 'first'), min_genes=('min_genes', 'first'),
                                                         min_mand=('min_mandatory', 'first')).reset_index()
print('models', info.groupby(['caller', 'family']).size().to_dict())
print('multi-gene models (min_genes >= 2):', info[info.min_genes >= 2].groupby('family').size().to_dict())
print('small multi-gene (<= 4 mand+acc):', info[(info.min_genes >= 2) & (info.n_comp <= 4)].groupby('family').size().to_dict())
print('profiles', pr.groupby('caller').size().to_dict(), 'missing files', int((~pr.exists).sum()),
      'without GA', pr[pr.GA1.isna()].groupby('caller').size().to_dict())
multi = set(info[info.min_genes >= 2].model_key)
pm = set(p for r in mg[mg.model_key.isin(multi)].itertuples() for p in r.profiles.split(';'))
sub = pr[pr.profile.isin(pm)]
print('profiles of multi-gene models', len(sub), 'total LENG', int(sub.LENG.sum()), 'all profiles total LENG',
      int(pr.LENG.sum()))
dup = mg.groupby('model_key').size()
print('model keys duplicated across families?', mg.groupby('model').model_key.nunique().max())
