#!/usr/bin/env python
"""Fold inputs for the lead completions (biotite login node, serial). Sequences stay on biotite and in the git-ignored
local folds/ folder.

Co-folds (chain A = completing / test gene, chain B = partner), names <lead>__<role>__<gidA>_<gidB>:
  slot  completed locus: slot gene + the nearest cluster hit gene of the other component
  pos   called system of the same model: member of the same component + member of the partner component
  neg   mismatched pair: a Pfam-annotated protein from the slot gene's genome with no defense-caller hit, length within
        15% of the slot gene (seed 17), paired with the real partner of that locus
Leads: AbiE antitoxin (3 slot, phyla diverse) and toxin (2 slot), Hachiman HamA (all 6), CBASS cyclase (all 4),
PARIS AriB (all 3 events = 2 genes); 1-2 positives and 1 negative each.
Writes folds_in/complexes.tsv (name, seqA, seqB) and folds_in/manifest.tsv (name, lead, role, gidA, gidB, lengths).
"""
from pathlib import Path

import numpy as np
import pandas as pd

RD = Path('/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111')
W = RD / 'dpann_leads/systems_srvp'
D = W / 'data'
OUT = W / 'folds_in'
OUT.mkdir(exist_ok=True)
ll = pd.read_csv(D / 'lead_loci.tsv', sep='\t')
cm = pd.read_parquet(D / 'call_members.parquet')
hits = pd.read_parquet(D / 'hits.parquet')
genes = pd.read_parquet(RD / 'ws_srvp_bins/package/universe/genes.parquet',
                        columns=['gid', 'genome', 'contig_code', 'position_index', 'length'])
gi = genes.set_index('gid')
gen = pd.read_parquet(RD / 'ws_srvp_bins/package/universe/genomes.parquet', columns=['genome', 'phylum'])
ph = dict(zip(gen.genome, gen.phylum))
rng = np.random.default_rng(17)
EFFECTOR_SKIP = ('CBASS__E2', 'CBASS__Jab', 'CBASS__2TM_Gros')


def partner(r):
    ps = [int(x) for x in str(r.partners).split(';')]
    hp = hits[hits.gid.isin(ps)]
    if r.stype == 'CBASS':
        hp = hp[~hp.profile.str.startswith(EFFECTOR_SKIP)]
    if r.stype == 'Paris':
        hp = hp[hp.profile.str.contains('AriA')]
    if len(hp):
        ps = sorted(set(hp.gid))
    return min(ps, key=lambda p: abs(int(gi.loc[p].position_index) - int(r.position_index)))


rows = []
L = ll.drop_duplicates(['stype', 'component', 'gid'])
ab_at = L[(L.stype == 'AbiE') & (L.component == 'AbiEii__AbiEi_1')].sort_values(['tier', 'support'], ascending=[True, False])
pick = ab_at.groupby('phylum').head(1).sort_values('support', ascending=False).head(3)
ab_tx = L[(L.stype == 'AbiE') & (L.component == 'AbiEii__AbiEii') & L.length.between(230, 330)].sort_values(
    'support', ascending=False)   # full-length toxins (completion median 272 aa)
pick_tx = ab_tx.groupby('phylum').head(1).head(2)
sel = pd.concat([pick.assign(lead='AbiE'), pick_tx.assign(lead='AbiEtox'),
                 L[L.stype == 'Hachiman'].assign(lead='Hachiman'), L[L.stype == 'CBASS'].assign(lead='CBASS'),
                 L[L.stype == 'Paris'].drop_duplicates('gid').assign(lead='PARIS')])
for r in sel.itertuples():
    rows.append(dict(lead=r.lead, role='slot', gidA=int(r.gid), gidB=int(partner(r))))

# positives: called systems with exactly one member of each component of interest
POS = {'AbiE': ('DefenseFinder/AbiE/AbiE', 'AbiEii__AbiEi_1', 'AbiEii__AbiEii', 1),
       'AbiEtox': ('DefenseFinder/AbiE/AbiE', 'AbiEii__AbiEii', 'AbiEii__AbiEi_1', 1),
       'Hachiman': ('DefenseFinder/Hachiman/Hachiman', 'Hachiman__HamA_1', 'Hachiman__HamB', 2),
       'CBASS': ('DefenseFinder/CBASS/CBASS_I_Endonuclease', 'CBASS__Cyclase_SMODS', 'CBASS__Endonuc_small', 1),
       'CBASS2': ('DefenseFinder/CBASS/CBASS_I_TM', 'CBASS__Cyclase_SMODS', 'CBASS__Effector_2TM_S_2TMBeta', 1),
       'PARIS': ('DefenseFinder/Paris/PARIS_II', 'Paris_II__AriB_DUF4435_II', 'Paris_II__AriA_AAA21', 1)}
for lead, (m, ca, cb, n) in POS.items():
    x = cm[cm.model_key == m]
    ok = []
    for sk, g in x.groupby(['run', 'genome', 'sys_id']):
        a, b = g[g.component == ca], g[g.component == cb]
        if len(a) == 1 and len(b) == 1:
            la, lb = gi.loc[int(a.gid.iloc[0])].length, gi.loc[int(b.gid.iloc[0])].length
            if lead.startswith('AbiE') and not (150 <= min(la, lb) and max(la, lb) <= 330):
                continue   # typical AbiEi / AbiEii lengths (9HLO: 196 / 287 aa)
            ok.append((sk, int(a.gid.iloc[0]), int(b.gid.iloc[0])))
    ok = sorted(ok)
    idx = rng.choice(len(ok), size=min(n, len(ok)), replace=False) if ok else []
    for i in idx:
        rows.append(dict(lead=lead.replace('CBASS2', 'CBASS'), role='pos', gidA=ok[i][1], gidB=ok[i][2]))

# negatives: length-matched unrelated protein from the first slot locus genome, with the real partner
pf_g = set(pd.read_parquet(RD / 'annot_gpu/srvp/PFAM.parquet', columns=['gid']).gid)
defense = set(hits.gid) | set(cm.gid)
for lead in ['AbiE', 'AbiEtox', 'Hachiman', 'CBASS', 'PARIS']:
    s = [r for r in rows if r['lead'] == lead and r['role'] == 'slot'][0]
    ga = gi.loc[s['gidA']]
    pool = genes[(genes.genome == ga.genome) & (abs(genes.length - ga.length) <= 0.15 * ga.length) &
                 genes.gid.isin(pf_g) & ~genes.gid.isin(defense) &
                 ((genes.contig_code != ga.contig_code) | (abs(genes.position_index - ga.position_index) > 20))]
    rows.append(dict(lead=lead, role='neg', gidA=int(pool.gid.iloc[rng.integers(len(pool))]), gidB=s['gidB']))
man = pd.DataFrame(rows)
need = set(man.gidA) | set(man.gidB)
seq = {}
with open(RD / 'ws_srvp_bins/package/universe/proteins.faa') as f:
    g = None
    for line in f:
        if line[0] == '>':
            g = int(line[1:])
        elif g in need:
            seq[g] = line.strip().rstrip('*')
man['name'] = man.lead + '__' + man.role + '__' + man.gidA.astype(str) + '_' + man.gidB.astype(str)
man['lenA'] = man.gidA.map(lambda x: len(seq[x]))
man['lenB'] = man.gidB.map(lambda x: len(seq[x]))
man['phylum'] = man.gidA.map(lambda x: ph[gi.loc[x].genome])
assert (man.lenA + man.lenB <= 4096).all()
man.to_csv(OUT / 'manifest.tsv', sep='\t', index=False)
with open(OUT / 'complexes.tsv', 'w') as f:
    for r in man.itertuples():
        f.write(f'{r.name}\t{seq[r.gidA]}\t{seq[r.gidB]}\n')
print(man[['name', 'lenA', 'lenB', 'phylum']].to_string(index=False))
print(len(man), 'co-folds')
