#!/usr/bin/env python
"""CD-NTase substrate/cofactor co-folds (ESMFold2, 2 ATP + 2 Mg; folds/cyclase_ligand/): distance from each Mg2+ and
from the ATP alpha-phosphate to the carboxylate oxygens of the three Mg-binding Asp residues located by the US-align
residue map (data/residue_map.tsv, CdnE G2SLH8 D60/D62/D129 equivalents). The negative (no triad aligned) is measured
against its three closest Asp/Glu residues to any Mg. Writes data/ligand_cofold.tsv."""
import json
from pathlib import Path

import gemmi
import numpy as np
import pandas as pd

W = Path('/Users/jacob/Documents/Sandbox/elsa_test/ELSA/benchmarks/redesign/dpann_leads/systems_srvp')
F = W / 'folds/cyclase_ligand'
rm = pd.read_csv(W / 'data/residue_map.tsv', sep='\t')
rm = rm[rm.ref == '6E0K_A']
triad = {}
for r in rm.itertuples():
    gid = r.name.split('__')[2].split('_')[0]
    res = [x.split('->')[1] for x in r.residues.split(';')]
    triad[f"CBASS_{r.name.split('__')[1]}_{gid}"] = [int(x[1:]) for x in res if x[0] in 'DE' and x[1:]]
rows = []
for p in sorted(F.glob('*.pdb')):
    name = p.stem
    st = gemmi.read_structure(str(p))
    prot = st[0]['A']
    mg = [a.pos for ch in st[0] for res in ch for a in res if res.name == 'MG']
    pa = [a.pos for ch in st[0] for res in ch for a in res if res.name == 'ATP' and a.name in ('PA',)]
    carb = {}
    for res in prot:
        if res.name in ('ASP', 'GLU'):
            carb[res.seqid.num] = [a.pos for a in res if a.name in ('OD1', 'OD2', 'OE1', 'OE2')]
    tri = triad.get(name) or []
    if not tri and mg:
        d = {k: min(min(m.dist(o) for o in v) for m in mg) for k, v in carb.items() if v}
        tri = sorted(d, key=d.get)[:3]
    dm = [min(m.dist(o) for k in tri for o in carb.get(k, [])) for m in mg] if tri else []
    dpa = [min(x.dist(o) for k in tri for o in carb.get(k, [])) for x in pa] if tri else []
    j = json.load(open(p.with_suffix('.json')))
    ca = [a.b_iso for res in prot for a in res if a.name == 'CA']
    lig_b = [a.b_iso for ch in st[0] for res in ch for a in res if res.name in ('ATP', 'MG')]
    rows.append(dict(name=name, triad=';'.join(f'D{k}' for k in tri), from_residue_map=bool(triad.get(name)),
                     ptm=round(j.get('ptm', np.nan), 3), plddt_protein=round(np.mean(ca), 1),
                     plddt_ligand=round(np.mean(lig_b), 1) if lig_b else None,
                     mg_to_triad_A=';'.join(f'{x:.1f}' for x in dm), atp_PA_to_triad_A=';'.join(f'{x:.1f}' for x in dpa),
                     mg_within_3A=sum(x <= 3.0 for x in dm)))
t = pd.DataFrame(rows)
t.to_csv(W / 'data/ligand_cofold.tsv', sep='\t', index=False)
pd.set_option('display.width', 200)
print(t.to_string(index=False))
