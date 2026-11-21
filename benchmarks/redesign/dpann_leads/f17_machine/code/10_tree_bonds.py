"""Q1 (method's own view): bonds between adjacent role genes and the nested nodes of the Z L4b+ theta .02 tree inside
family-17 loci. Runs on the biotite login node (reads frozen DPANN run files; writes only to f17_machine)."""
import numpy as np, pandas as pd, collections, json
RD = '/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111'
W = RD + '/dpann_leads/f17_machine'
g = pd.read_parquet(RD + '/ws_dpann/package/universe/genes.parquet', columns=['gid', 'protein_id'])
w = pd.read_csv(W + '/window_roles.tsv', sep='\t').merge(g, on='protein_id')
role = dict(zip(w.gid, w.role))
z = np.load(RD + '/long_loci/nodes/dpann_Z/bonds_obs.npz')
B = pd.DataFrame({k: z[k] for k in ['k', 'd', 'lam', 'n', 'ng']})
B = B[B.k.isin(set(w.gid))]
B['r1'] = B.k.map(role); B['r2'] = (B.k + B.d).map(role)
B = B.dropna(subset=['r2'])
B['pair'] = [' | '.join(sorted([a, b])) for a, b in zip(B.r1, B.r2)]
b1 = B[(B.d == 1) & (B.r1 != '-') & (B.r2 != '-')]
s1 = b1.groupby('pair').agg(bonds=('lam', 'size'), lam_med=('lam', 'median'), n_med=('n', 'median'), ng_med=('ng', 'median')).sort_values('bonds', ascending=False)
print('adjacent (d=1) bonds between role genes'); print(s1.head(25).to_string())
# adjacency present without a bond: count adjacent role pairs in windows
w = w.sort_values(['contig_id', 'gene_index'])
adj = collections.Counter(); bonded = set(zip(b1.k, b1.k + 1))
for c, s in w.groupby('contig_id'):
    gi, gd, rr = s.gene_index.to_numpy(), s.gid.to_numpy(), s.role.to_numpy()
    for i in range(len(s) - 1):
        if gi[i + 1] == gi[i] + 1 and rr[i] != '-' and rr[i + 1] != '-':
            p = ' | '.join(sorted([rr[i], rr[i + 1]]))
            adj[(p, (gd[i], gd[i + 1]) in bonded)] += 1
print('adjacent role pairs: bonded / total')
for p in sorted({k[0] for k in adj}, key=lambda p: -(adj[(p, True)] + adj[(p, False)])):
    t = adj[(p, True)] + adj[(p, False)]
    if t >= 50: print(f'  {p:28s} {adj[(p, True)]:4d} / {t:4d}')
# tree nodes nested in each family-17 locus
L = pd.read_parquet(RD + '/long_loci/loci/dpann_Z/L4bp_t0.02.parquet').sort_values(['s', 'e'], ascending=[True, False])
gidset = np.zeros(int(g.gid.max()) + 2, bool); gidset[w.gid.to_numpy()] = True
rarr = np.full(len(gidset), '', object)
for k, v in role.items(): rarr[k] = v
loc_of = dict(zip(w.gid, w.locus))
cnt = collections.Counter(); outer_cnt = collections.Counter()
for s, e in zip(L.s.to_numpy(), L.e.to_numpy()):
    if not (gidset[s] and gidset[e]): continue
    if loc_of.get(s) != loc_of.get(e): continue
    rs = [r for r in rarr[s:e + 1] if r not in ('', '-')]
    if len(rs) < 2: continue
    key = ' '.join(sorted(set(rs)))
    cnt[key] += 1
print('most common role sets of tree nodes (>=2 role genes) inside family-17 windows')
for k, v in cnt.most_common(30): print(f'  {v:4d}  {k}')
# does any node contain both Top6B and a machine role, versus nodes splitting them
both = sum(v for k, v in cnt.items() if 'Top6B' in k and any(m in k for m in ['ATPase504', 'A2Ml794', 'TM277a', 'wHTH244']))
top_only = sum(v for k, v in cnt.items() if 'Top6B' in k and not any(m in k for m in ['ATPase504', 'A2Ml794', 'TM277a', 'wHTH244', 'TM132', 'p102', 'p49']))
mach_only = sum(v for k, v in cnt.items() if 'Top6B' not in k and 'KH171' not in k and 'ATPase504' in k)
print(json.dumps({'nodes_with_Top6B_and_machine': both, 'nodes_Top6B_without_machine': top_only, 'nodes_ATPase_without_Top6B_KH': mach_only}))
B.to_csv(W + '/bonds_roles.tsv', sep='\t', index=False)
