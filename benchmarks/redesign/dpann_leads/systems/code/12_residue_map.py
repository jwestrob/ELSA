#!/usr/bin/env python
"""Rung 3: residues of characterized relatives (UniProt features by accession) at aligned positions.

* HinP1I (Q5I6E6, PDB 1YNM): metal-binding residues 18, 62, 81, 82 (UniProt binding-site features); mapped through
  the US-align structural alignment of each predicted model to 1YNM chain A (author numbering = UniProt numbering,
  checked against the UniProt sequence).
* MazE (P0AE72): residues 8 and 16 (mutagenesis: DNA binding lost / poor); mapped by mafft (--localpair) of the
  slot MazE, the positive-control DPANN MazE and E. coli MazE.
"""
import json
import subprocess
from pathlib import Path

import gemmi
import pandas as pd

W = Path('/Users/jacob/Documents/Sandbox/elsa_test/ELSA/benchmarks/redesign/dpann_leads/systems')
us = pd.read_csv(W / 'data/usalign_out.tsv', sep='\t')
refs = W / 'data/refs'


def atom_seq(pdb, chain):
    st = gemmi.read_structure(str(pdb))
    ch = st[0][chain]
    return [(r.seqid.num, gemmi.find_tabulated_residue(r.name).one_letter_code.upper()) for r in ch
            if gemmi.find_tabulated_residue(r.name) and gemmi.find_tabulated_residue(r.name).is_amino_acid()]


uni = json.load(open(refs / 'Q5I6E6.json'))['sequence']['value']
ref = atom_seq(W / 'data/refs_chains/1YNM_A.pdb', 'A')
assert all(uni[n - 1] == a for n, a in ref), 'numbering mismatch'
sites = [18, 62, 81, 82]
lines = [f'HinP1I sites (Q5I6E6): ' + ', '.join(f'{uni[s - 1]}{s}' for s in sites)]
for r in us[us.name.str.endswith('|1YNM_A')].itertuples():
    s1, s2 = r.seq1, r.seq2
    qi = ri = 0
    got = {}
    refnums = [n for n, _ in ref]
    for a, b in zip(s1, s2):
        if b != '-':
            rn = refnums[ri]
            if rn in sites:
                got[rn] = (a, qi + 1 if a != '-' else None)
            ri += 1
        if a != '-':
            qi += 1
    lines.append(f'  {r.name.split("|")[0]}: TM1 {r.TM1:.2f}; aligned residues at ' +
                 ', '.join(f'{uni[s - 1]}{s}->{got.get(s, ("-", None))[0]}' for s in sites))

# MazE by sequence
mz = json.load(open(refs / 'P0AE72.json'))['sequence']['value']
import duckdb
con = duckdb.connect()
con.execute("ATTACH '/Users/jacob/Documents/Sandbox/Sharur/Sharur/data/dpann/sharur.duckdb' AS d (READ_ONLY)")
ids = ['MBI2144465.1', 'HLD49479.1', 'KHO48508.1']
seqs = dict(con.execute(f"SELECT protein_id, sequence FROM d.proteins WHERE protein_id IN "
                        f"({','.join(repr(i) for i in ids)})").fetchall())
tmp = W / 'data/maze_in.faa'
tmp.write_text(f'>EcMazE_P0AE72\n{mz}\n' + ''.join(f'>{i}\n{seqs[i].rstrip("*")}\n' for i in ids))
aln = subprocess.run(['mafft', '--localpair', '--maxiterate', '1000', '--quiet', str(tmp)], capture_output=True,
                     text=True).stdout
recs, h = {}, None
for line in aln.splitlines():
    if line.startswith('>'):
        h = line[1:].strip()
        recs[h] = ''
    else:
        recs[h] += line.strip().upper()
ref_al = recs['EcMazE_P0AE72']
cols = {}
k = 0
for c, a in enumerate(ref_al):
    if a != '-':
        k += 1
        cols[k] = c
lines.append('MazE sites (P0AE72 mutagenesis, DNA binding): ' + ', '.join(f'{mz[s - 1]}{s}' for s in (8, 16)))
for i in ids:
    lines.append(f'  {i}: ' + ', '.join(f'{mz[s - 1]}{s}->{recs[i][cols[s]]}' for s in (8, 16)))
(W / 'data/residue_map.txt').write_text('\n'.join(lines) + '\n')
print('\n'.join(lines))
