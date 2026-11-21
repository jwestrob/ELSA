#!/usr/bin/env python
"""Prepare US-align inputs (Mac, gemmi): single chains of the co-fold models (folds/split/, git-ignored) and of the
public reference structures (data/refs/), two-chain reference complexes, and the pair table usalign_pairs.tsv.

Single-chain pairs (mode mono): chain A of every co-fold (slot, pos, neg) against the reference chain of its role:
  AbiE antitoxin -> 9HLO A (AbiEi, Q8DZ36); AbiE toxin -> 9HLO B (AbiEii, Q8DZ35); HamA -> 8VX9 A (Q6XGE5);
  CBASS cyclase -> 6E0K A (CdnE, G2SLH8) and 4TY0 A (DncV, Q9KVG7); AriB -> 8V46 D (AriB).
Complex pairs (mode mm, -mm 1 -ter 0): each co-fold against 9HLO A+B, 8VX9 A+B, or 8V46 D + the AriA chain with the most
contacts to D.
"""
import glob
import os
from pathlib import Path

import gemmi

W = Path('/Users/jacob/Documents/Sandbox/elsa_test/ELSA/benchmarks/redesign/dpann_leads/systems_srvp')
REF = W / 'data/refs'
SPL = W / 'folds/split'
SPL.mkdir(exist_ok=True)
RS = '/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/dpann_leads/systems_srvp/usalign'


def write_chains(st, chains, out):
    s2 = st.clone()
    s2.setup_entities()
    s2.remove_ligands_and_waters()
    for m in s2:
        for c in list(m):
            if c.name not in chains:
                m.remove_chain(c.name)
    s2.setup_entities()
    s2.write_pdb(str(out))


refs = {}
for pid in ['9HLO', '8VX9', '8V46', '6E0K', '4TY0']:
    st = gemmi.read_structure(str(REF / f'{pid}.cif'))
    st.setup_entities()
    refs[pid] = st
    for ch in st[0]:
        write_chains(st, {ch.name}, REF / f'{pid}_{ch.name}.pdb')
write_chains(refs['9HLO'], {'A', 'B'}, REF / '9HLO_AB.pdb')
write_chains(refs['8VX9'], {'A', 'B'}, REF / '8VX9_AB.pdb')
# AriA chain with the most contacts to AriB (chain D) in 8V46
st = refs['8V46']
ns = gemmi.NeighborSearch(st[0], st.cell, 5).populate()
cnt = {}
for res in st[0]['D']:
    for at in res:
        for mk in ns.find_atoms(at.pos, '\0', radius=4.5):
            cra = mk.to_cra(st[0])
            if cra.chain.name != 'D':
                cnt[cra.chain.name] = cnt.get(cra.chain.name, 0) + 1
aria = max(cnt, key=cnt.get)
write_chains(st, {'D', aria}, REF / '8V46_complex.pdb')
print('8V46 AriA chain in contact with AriB:', aria, cnt)

rows = []
for p in sorted(glob.glob(str(W / 'folds/complexes/*.pdb'))):
    name = os.path.basename(p)[:-4]
    st = gemmi.read_structure(p)
    for ch in ('A', 'B'):
        write_chains(st, {ch}, SPL / f'{name}_{ch}.pdb')
    lead = name.split('__')[0]
    q = f'{RS}/split/{name}_A.pdb'
    mono = {'AbiE': ['9HLO_A'], 'AbiEtox': ['9HLO_B'], 'Hachiman': ['8VX9_A'], 'CBASS': ['6E0K_A', '4TY0_A'],
            'PARIS': ['8V46_D']}[lead]
    for r in mono:
        rows.append((f'{name}|{r}', q, f'{RS}/refs/{r}.pdb', 'mono', ''))
    cx = {'AbiE': '9HLO_AB', 'AbiEtox': '9HLO_AB', 'Hachiman': '8VX9_AB', 'PARIS': '8V46_complex'}.get(lead)
    if cx:
        rows.append((f'{name}|{cx}', f'{RS}/complexes/{name}.pdb', f'{RS}/refs/{cx}.pdb', 'mm', ''))
# positive / negative references for the fold test: experimental chains against each other
rows.append(('ref|6E0K_A|4TY0_A', f'{RS}/refs/6E0K_A.pdb', f'{RS}/refs/4TY0_A.pdb', 'mono', ''))
with open(W / 'data/usalign_pairs.tsv', 'w') as f:
    f.write('name\tquery_pdb\tref_pdb\tmode\textra\n')
    for r in rows:
        f.write('\t'.join(r) + '\n')
print(len(rows), 'pairs')
