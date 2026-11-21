"""Assign roles from mmseqs window clusters; layout (order, strand relative to the DUF87 ATPase, intergenic spacing)."""
import pandas as pd, numpy as np, collections
a = pd.read_parquet('data/f17_windows.parquet')
ROLE = {'NYZ75760.1': 'ATPase504', 'CAIKVX010000082.1_9': 'TM277a', 'JAGDXY010000005.1_54': 'TM278b',
        'JAJYVC010000013.1_11': 'TM132', 'DAIDBQ010000003.1_24': 'wHTH244', 'JAJZJQ010000009.1_24': 'A2Ml794',
        'CAIKVX010000082.1_7': 'p102', 'MEM3841301.1': 'p49', 'JAJYOL010000006.1_27': 'p49',
        'JAJZLT010000016.1_19': 'Top6B', 'JAJYOR010000001.1_30': 'KH171', 'MDE1860699.1': 'ArsR178',
        'CAIPUG010000085.1_13': 'pil129', 'JAKAUU010000027.1_4': 'p244b', 'MDE1871070.1': 'p634'}
for r in ['MEM3791321.1', 'JAKAVK010000018.1_19', 'JAJZAJ010000002.1_210', 'JAJZLT010000016.1_6', 'JAKASF010000001.1_18']:
    ROLE[r] = 'MP4610'
a['role'] = a.rep.map(ROLE).fillna('-')
a.to_parquet('data/f17_windows.parquet')
rows = []
orders = collections.Counter()
for loc, s in a.groupby('locus'):
    s = s.sort_values('gene_index')
    atp = s[s.role == 'ATPase504']
    if len(atp) == 0: continue
    st = atp.strand.iloc[0]
    if st == '-': s = s.iloc[::-1]
    rel = [('+' if x == st else '-') for x in s.strand]
    toks = [f'{r}{o}' for r, o in zip(s.role, rel) if r != '-']
    # collapse runs of MP4610
    out = []
    for t in toks:
        if out and t == out[-1] and t.startswith('MP'): continue
        out.append(t)
    orders[' '.join(out)] += 1
    for r, o, g, st_, en in zip(s.role, rel, s.gene_index, s.start, s.end_coord):
        rows.append((loc, r, o, g, st_, en))
d = pd.DataFrame(rows, columns=['locus', 'role', 'relstrand', 'gi', 'start', 'end'])
d.to_csv('data/layout_rows.tsv', sep='\t', index=False)
print('loci with ATPase', d.locus.nunique())
for o, n in orders.most_common(12): print(n, o)
# relative strand per role
print(d[d.role != '-'].groupby('role').relstrand.value_counts().unstack().fillna(0).astype(int))
