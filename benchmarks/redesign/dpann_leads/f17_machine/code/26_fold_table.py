"""Collect monomer and complex fold metrics into folds/fold_summary.tsv (pLDDT, pTM, ipTM, chain-pair ipTM)."""
import json, glob, os, numpy as np
rows = []
for d in ['mono', 'mono2', 'mono3']:
    for f in sorted(glob.glob(f'folds/{d}/*.json')):
        j = json.load(open(f)); n = os.path.basename(f)[:-5]
        pl = j.get('mean_plddt', j.get('plddt'))
        if isinstance(pl, list): pl = float(np.mean(pl))
        rows.append((n, 'monomer', round(pl * (100 if pl <= 1 else 1), 1), round(j.get('ptm', np.nan), 3), '', ''))
for f in sorted(glob.glob('folds/cx/*.json')):
    j = json.load(open(f)); n = os.path.basename(f)[:-5]
    pl = float(np.mean(j['plddt'])) * 100
    pc = j.get('pair_chains_iptm'); off = ''
    if pc and len(pc) == 2: off = round((pc[0][1] + pc[1][0]) / 2, 3)
    rows.append((n, 'complex', round(pl, 1), round(j['ptm'], 3), round(j['interface_ptm'], 3), off))
with open('folds/fold_summary.tsv', 'w') as fh:
    fh.write('name\tkind\tmean_pLDDT\tpTM\tipTM\tpair_iptm_AB\n')
    for r in rows: fh.write('\t'.join(map(str, r)) + '\n')
print(len(rows)); print(open('folds/fold_summary.tsv').read())
