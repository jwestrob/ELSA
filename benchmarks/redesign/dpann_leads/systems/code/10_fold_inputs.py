#!/usr/bin/env python
"""Write fold inputs: single-chain FASTA (folds/singles.faa) and hetero-complex table (folds/complexes.tsv).
Names encode role: <lead>__<role>__<protein_id>; roles: slot, pos (caller-accepted DPANN member at GA),
neg (DPANN protein of similar length from an unrelated Pfam family)."""
from pathlib import Path

import duckdb

W = Path('/Users/jacob/Documents/Sandbox/elsa_test/ELSA/benchmarks/redesign/dpann_leads/systems')
F = W / 'folds'
F.mkdir(exist_ok=True)
con = duckdb.connect()
con.execute("ATTACH '/Users/jacob/Documents/Sandbox/Sharur/Sharur/data/dpann/sharur.duckdb' AS d (READ_ONLY)")

singles = [
    # AbiE antitoxins (slot), positive control (AbiEi_4 at GA in a called AbiE system), negative (Methyltransf_11)
    ('AbiEi', 'slot', 'DAHVSL010000002.1_2'), ('AbiEi', 'slot', 'DAHVIM010000005.1_54'),
    ('AbiEi', 'slot', 'MCX6815689.1'), ('AbiEi', 'slot', 'VVB99915.1'), ('AbiEi', 'slot', 'MBU2576469.1'),
    ('AbiEi', 'pos', 'OIO20792.1'), ('AbiEi', 'neg', 'JAJYTZ010000018.1_15'),
    # MazE (slot), positive (MazE_antitoxin at GA in a called MazEF system), negative (Ribosomal_L7Ae)
    ('MazE', 'slot', 'MBI2144465.1'), ('MazE', 'slot', 'HLD49479.1'), ('MazE', 'pos', 'KHO48508.1'),
    ('MazE', 'neg', 'MBS1266953.1'),
    # Type II REases (slot, embedding-only), positives at GA, negative (Methyltransf_11 270 aa)
    ('RHinP1I', 'slot', 'MBU0460042.1'), ('RHinP1I', 'slot', 'MCR4323407.1'), ('RHinP1I', 'pos', 'HLC51486.1'),
    ('ThaI', 'slot', 'MEM5793242.1'), ('ThaI', 'slot', 'DAHVPS010000013.1_3'),
    ('ThaI', 'pos', 'DAHWVQ010000010.1_87'), ('REase', 'neg', 'JAJYTZ010000002.1_23'),
]
complexes = [
    # AbiEi + AbiEii from the same locus (antitoxin chain A, toxin chain B)
    ('AbiE__slot__DAHVSL010000002.1_2', ['DAHVSL010000002.1_2', 'DAHVSL010000002.1_1']),
    ('AbiE__slot__DAHVIM010000005.1_54', ['DAHVIM010000005.1_54', 'DAHVIM010000005.1_55']),
    ('AbiE__slot__MCX6815689.1', ['MCX6815689.1', 'MCX6815690.1']),
    ('AbiE__slot__VVB99915.1', ['VVB99915.1', 'VVB99913.1']),
    ('AbiE__slot__MBU2576469.1', ['MBU2576469.1', 'MBU2576468.1']),
    ('AbiE__pos__OIO20792.1', ['OIO20792.1', 'OIO20791.1']),
    ('AbiE__neg__DAHVSL010000002.1_2', ['DAHVSL010000002.1_2', 'JAJYTZ010000002.1_23']),
    # MazF2 : MazE (E. coli 1UB4 stoichiometry), toxin chains A,B; antitoxin chain C
    ('MazEF__slot__MBI2144465.1', ['MBI2144464.1*2', 'MBI2144465.1']),
    ('MazEF__slot__HLD49479.1', ['HLD49480.1*2', 'HLD49479.1']),
    ('MazEF__pos__KHO48508.1', ['KHO48509.1*2', 'KHO48508.1']),
]
ids = sorted({p for _, _, p in singles} | {s.split('*')[0] for _, ch in complexes for s in ch})
seq = dict(con.execute(f"SELECT protein_id, sequence FROM d.proteins WHERE protein_id IN "
                       f"({','.join(repr(i) for i in ids)})").fetchall())
missing = [i for i in ids if i not in seq]
assert not missing, missing
with open(F / 'singles.faa', 'w') as f:
    for lead, role, p in singles:
        f.write(f'>{lead}__{role}__{p}\n{seq[p].rstrip("*")}\n')
with open(F / 'complexes.tsv', 'w') as f:
    for name, ch in complexes:
        specs = []
        for s in ch:
            p, _, n = s.partition('*')
            specs.append(seq[p].rstrip('*') + (f'*{n}' if n else ''))
        f.write(name + '\t' + '\t'.join(specs) + '\n')
print(len(singles), 'singles;', len(complexes), 'complexes')
