"""TMHMM 2.0c summary per role: predicted helices (PredHel), first-60 expected TM residues, N-terminus side."""
import pandas as pd, re
a = pd.read_parquet('data/f17_windows.parquet')
role = dict(zip(a.protein_id, a.role))
rows = []
for l in open('data/tmhmm.tsv'):
    f = l.split('\t')
    if len(f) < 6: continue
    pid = f[0]; d = dict(x.split('=', 1) for x in f[1:])
    top = d['Topology'].strip()
    rows.append((pid, role.get(pid, '-'), int(d['len']), int(d['PredHel']), float(d['First60']), top[0] if top else '', top))
t = pd.DataFrame(rows, columns=['protein_id', 'role', 'len', 'predhel', 'first60', 'nterm', 'topology'])
t.to_csv('data/tmhmm_parsed.tsv', sep='\t', index=False)
g = t.groupby('role').agg(n=('len', 'size'), len_med=('len', 'median'), predhel_med=('predhel', 'median'),
                          predhel_mode=('predhel', lambda x: x.mode().iloc[0]), frac_ge1=('predhel', lambda x: round((x >= 1).mean(), 2)),
                          first60_med=('first60', 'median'), nterm_in=('nterm', lambda x: round((x == 'i').mean(), 2)))
print(g.to_string())
# example topologies for 2-TM roles
for r in ['TM277a', 'TM278b', 'TM132', 'pil129', 'A2Ml794', 'p49']:
    print(r, t[t.role == r].topology.str.replace(r'\d+', '#', regex=True).value_counts().head(3).to_dict())
