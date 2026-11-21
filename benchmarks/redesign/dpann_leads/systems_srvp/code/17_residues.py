#!/usr/bin/env python
"""Catalytic residues at aligned positions (Mac): reads the US-align alignment strings (folds/usalign_out.tsv) and the
reference chains (data/refs/<pdb>_<chain>.pdb), maps reference author residue numbers to query residues.

Curated residue sources (fetched by accession):
  HamA  PDB 8VXY entity 3 'HamA E138A,K140A' (the mutant of the DNA-bound HamA:HamB structure; Q6XGE5 numbering)
  CdnE  UniProt G2SLH8 (Swiss-Prot): Mg2+ binding D60, D62, D129 (6E0K chain A)
  DncV  UniProt Q9KVG7 (Swiss-Prot): Mg2+ binding D131, D133, D193 (4TY0 chain A)
A site passes when the aligned query residue is the same residue type (E/K for HamA, D or E for the CD-NTase triad).
Writes data/residue_map.tsv.
"""
from pathlib import Path

import json

import gemmi
import pandas as pd

W = Path('/Users/jacob/Documents/Sandbox/elsa_test/ELSA/benchmarks/redesign/dpann_leads/systems_srvp')
SITES = {'8VX9_A': {138: 'E', 140: 'K'}, '6E0K_A': {60: 'DE', 62: 'DE', 129: 'DE'},
         '4TY0_A': {131: 'DE', 133: 'DE', 193: 'DE'}}   # UniProt numbering
UNIPROT = {'8VX9_A': 'Q6XGE5', '6E0K_A': 'G2SLH8', '4TY0_A': 'Q9KVG7'}
u = pd.read_csv(W / 'folds/usalign_out.tsv', sep='\t')   # aligned SR-VP sequences stay in the git-ignored folds/
rows = []
for ref, sites in SITES.items():
    st = gemmi.read_structure(str(W / f'data/refs/{ref}.pdb'))
    ch = st[0][0]
    names = [gemmi.find_tabulated_residue(r.name).one_letter_code.upper() for r in ch]
    useq = json.load(open(W / f'data/refs/{UNIPROT[ref]}.json'))['sequence']['value']
    cseq = ''.join(names)
    # UniProt position of each modeled residue: anchor 12-mers of the chain in the UniProt sequence
    nums, last = [], 0
    for i in range(len(cseq)):
        k = useq.find(cseq[i:i + 12], last) if i + 12 <= len(cseq) else -1
        if k >= 0 and (not nums or nums[-1] is None or k + 1 > nums[-1]):
            nums.append(k + 1)
            last = k
        else:
            nums.append(nums[-1] + 1 if nums and nums[-1] else None)
    for k, v in sites.items():
        assert names[nums.index(k)] in v, (ref, k, names[nums.index(k)])
    for r in u[(u['mode'] == 'mono') & u.name.str.endswith('|' + ref)].itertuples():
        qi, ri = 0, 0
        amap = {}
        for a, b in zip(r.seq1, r.seq2):
            if a != '-' and b != '-':
                amap[nums[ri]] = (a, qi + 1)
            qi += a != '-'
            ri += b != '-'
        got = {k: amap.get(k, ('-', None)) for k in sites}
        ok = sum(got[k][0] in sites[k] for k in sites)
        rows.append(dict(name=r.name.split('|')[0], ref=ref, TM1=r.TM1, sites_ok=ok, n_sites=len(sites),
                         residues=';'.join(f'{names[nums.index(k)]}{k}->{got[k][0]}{got[k][1] or ""}' for k in sites)))
t = pd.DataFrame(rows)
t.to_csv(W / 'data/residue_map.tsv', sep='\t', index=False)
pd.set_option('display.width', 200)
print(t.to_string(index=False))
