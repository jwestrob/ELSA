#!/usr/bin/env python3
"""Hetero-complex co-folds through the same Biohub ESMFold2 API and helpers as Sharur's fold_biohub.py.

fold_biohub.py folds one protein (optionally N identical copies with --copies); it has no input for two different
proteins in one complex. This wrapper imports that script's own helpers (headers, post, to_pdb_complex) and submits
fold_all_atom items whose `sequences` list holds several protein entities, model esmfold2-2026-05.

usage: cofold_pair.py OUT_DIR complexes.tsv
complexes.tsv columns: name, then chain specs `SEQ` or `SEQ*N` (N copies) in the order chains A, B, C, ...
Writes <name>.pdb (pLDDT x 100 in B-factor), <name>.json (ptm, interface_ptm, pair_chains_iptm), summary.tsv.
"""
import importlib.util
import json
import os
import string
import sys
import time
import urllib.request

spec = importlib.util.spec_from_file_location(
    'fold_biohub', '/Users/jacob/Documents/Sandbox/Sharur/Sharur/scripts/fold_biohub.py')
fb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fb)

out_dir, tsv = sys.argv[1], sys.argv[2]
os.makedirs(out_dir, exist_ok=True)
H = fb.headers()
items, names, chains_of = [], [], []
for line in open(tsv):
    if not line.strip() or line.startswith('#'):
        continue
    f = line.rstrip('\n').split('\t')
    name, specs = f[0], f[1:]
    ents, ids = [], iter(string.ascii_uppercase)
    for sp in specs:
        seq, _, n = sp.partition('*')
        for _ in range(int(n) if n else 1):
            ents.append({'type': 'protein', 'id': next(ids), 'sequence': seq.rstrip('*').upper()})
    if sum(len(e['sequence']) for e in ents) > 4096:
        sys.exit(f'{name}: complex exceeds 4,096 tokens')
    if os.path.exists(f'{out_dir}/{name}.pdb'):
        continue
    items.append({'model': 'esmfold2-2026-05', 'all_atom_input': {'sequences': ents}, 'include_pae': False,
                  'include_pair_chains_iptm': True})
    names.append(name)
    chains_of.append(tuple(e['id'] for e in ents))
if not items:
    sys.exit('nothing to fold')
batch = fb.post('/submit', {'endpoint': 'fold_all_atom', 'payload': items}, H)['data']['batch_id']
print(f'batch {batch}: {len(items)} complexes', flush=True)
pending = {i: f'{batch}-{i:05d}' for i in range(len(items))}
rows = []
while pending:
    time.sleep(15)
    for i, tid in list(pending.items()):
        d = fb.post('/status', {'task_id': tid}, H)['data']
        st = d.get('status')
        if st in ('failed', 'cancelled'):
            print(f'FAILED {names[i]}: {d.get("error", st)}', flush=True)
            del pending[i]
            continue
        if st != 'done':
            continue
        j = json.loads(urllib.request.urlopen(d['response'], timeout=120).read())
        pdb, atoms = fb.to_pdb_complex(j['complex'], {}, chains_of[i])
        open(f'{out_dir}/{names[i]}.pdb', 'w').write(pdb)
        keep = {k: j.get(k) for k in ('model', 'ptm', 'interface_ptm', 'plddt', 'pair_chains_iptm')}
        json.dump(keep, open(f'{out_dir}/{names[i]}.json', 'w'))
        ca = [a[5] for a in atoms if a[3] == 'CA']
        rows.append((names[i], round(100 * sum(ca) / max(1, len(ca)), 1), round(j['ptm'], 3),
                     j.get('interface_ptm')))
        print(f'done {names[i]} pLDDT={rows[-1][1]} pTM={rows[-1][2]} ipTM={rows[-1][3]}', flush=True)
        del pending[i]
with open(f'{out_dir}/summary.tsv', 'w') as f:
    f.write('name\tplddt\tptm\tiptm\n')
    for r in sorted(rows):
        f.write('\t'.join(map(str, r)) + '\n')
