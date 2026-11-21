#!/usr/bin/env python
"""US-align comparisons (biotite login node, serial). Input pairs.tsv: name, query_pdb, ref_pdb, mode, extra args.
mode 'mono': query chain A vs the reference chain given in extra (e.g. '-chain2 A'); TM normalized by the query (TM1)
and by the reference (TM2). mode 'mm': multimer alignment (-mm 1 -ter 0), TM normalized by query / reference complex.
Writes out.tsv with TM1, TM2, RMSD, Lali and the aligned sequence strings (for residue mapping)."""
import csv
import subprocess
import sys

US = '/shared/software/bin/USalign'
pairs, out = sys.argv[1], sys.argv[2]
rows = []
for r in csv.DictReader(open(pairs), delimiter='\t'):
    r = type('R', (), r)
    cmd = [US, r.query_pdb, r.ref_pdb] + (['-mm', '1', '-ter', '0'] if r.mode == 'mm' else []) \
        + (r.extra.split() if r.extra else [])
    res = subprocess.run(cmd, capture_output=True, text=True)
    txt = res.stdout.splitlines()
    tm = [float(l.split('=')[1].split()[0]) for l in txt if l.startswith('TM-score=')]
    ali = [l for l in txt if l.startswith('Aligned length=')]
    rec = dict(name=r.name, mode=r.mode, query=r.query_pdb.split('/')[-1], ref=r.ref_pdb.split('/')[-1],
               extra=r.extra, rc=res.returncode)
    if len(tm) >= 2 and ali:
        a = ali[0]
        rec.update(TM1=tm[0], TM2=tm[1], Lali=int(a.split('=')[1].split(',')[0]),
                   RMSD=float(a.split('RMSD=')[1].split(',')[0]))
        k = [i for i, l in enumerate(txt) if l.startswith('(":" denotes')]
        if k:
            rec.update(seq1=txt[k[0] + 1], marks=txt[k[0] + 2], seq2=txt[k[0] + 3])
    rows.append(rec)
cols = ['name', 'mode', 'query', 'ref', 'extra', 'rc', 'TM1', 'TM2', 'Lali', 'RMSD', 'seq1', 'marks', 'seq2']
with open(out, 'w') as fh:
    w = csv.DictWriter(fh, fieldnames=cols, delimiter='\t')
    w.writeheader()
    for x in rows:
        w.writerow(x)
print(len(rows), 'pairs', sum('TM1' in x for x in rows), 'ok')
