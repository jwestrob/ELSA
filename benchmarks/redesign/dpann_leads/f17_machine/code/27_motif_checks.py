"""Motif checks run inline during the analysis, collected here: A2M thioester motif, Walker A/B in ATPase504, Zn-ribbon
CxxC in TM132, AxA signal-peptide motif in pil129, SignalP-6 summary per role, Nanobdellota exception contig."""
import re, statistics, collections, pandas as pd, duckdb
def readfa(p):
    d, k = {}, None
    for l in open(p):
        l = l.strip()
        if l.startswith('>'): k = l[1:].split()[0]; d[k] = ''
        else: d[k] += l
    return d
R = 'data/roles/'
A = readfa(R + 'A2Ml794.all.faa')
print('A2Ml794 thioester C.EQ:', sum(bool(re.search('C.EQ', s)) for s in A.values()), '/', len(A), '; median Cys', statistics.median(s.count('C') for s in A.values()))
T = readfa(R + 'ATPase504.all.faa')
wa = [re.search(r'G..G.GK[ST]|[GA]....GK[ST]', s) for s in T.values()]
print('ATPase504 Walker A:', sum(m is not None for m in wa), '/', len(T), 'median start', statistics.median(m.start() + 1 for m in wa if m))
wb = 0
for s, m in zip(T.values(), wa):
    if m and re.search(r'[ILVMFA]{3}[ILVMFAC]D[ED]', s[m.end() + 100:m.end() + 400]): wb += 1
print('ATPase504 Walker B:', wb, '/', len(T))
Z = readfa(R + 'TM132.all.faa'); print('TM132 two CxxC in 1-70:', sum(len(re.findall('C..C', s[:70])) >= 2 for s in Z.values()), '/', len(Z))
P = readfa(R + 'pil129.all.faa'); print('pil129 A.A[QSAE] at 19-34:', sum(bool(re.search(r'A.A[QSAE]', s[18:34])) for s in P.values()), '/', len(P))
a = pd.read_parquet('data/f17_windows.parquet'); role = dict(zip(a.protein_id, a.role))
rows = []
for l in open('data/signalp_prediction_results.txt'):
    if l.startswith('#'): continue
    f = l.rstrip('\n').split('\t'); rows.append((f[0].split()[0], role.get(f[0].split()[0], '-'), f[1]))
d = pd.DataFrame(rows, columns=['pid', 'role', 'pred']); print(pd.crosstab(d.role, d.pred))
c = duckdb.connect('/Users/jacob/Documents/Sandbox/Sharur/Sharur/data/dpann/sharur.duckdb', read_only=True)
print(c.sql("select p.bin_id, b.completeness, b.contamination, count(*) genes from proteins p join bins b using(bin_id) where p.contig_id='QMQO01000051.1' group by all").df())
