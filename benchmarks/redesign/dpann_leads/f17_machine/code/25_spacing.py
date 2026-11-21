"""Q1: intergenic distances (bp) between adjacent role genes, with relative orientation (co-directional, convergent,
divergent)."""
import pandas as pd, numpy as np
a = pd.read_parquet('data/f17_windows.parquet').sort_values(['contig_id', 'gene_index'])
rows = []
for cid, s in a.groupby('contig_id'):
    s = s.reset_index(drop=True)
    for i in range(len(s) - 1):
        x, y = s.loc[i], s.loc[i + 1]
        if y.gene_index != x.gene_index + 1 or x.role == '-' or y.role == '-': continue
        gap = int(min(y.start, y.end_coord) - max(x.start, x.end_coord))
        if x.strand == y.strand: ori = 'co'
        elif x.strand == '+' and y.strand == '-': ori = 'convergent'
        else: ori = 'divergent'
        # order the pair along the ATPase-forward reading: use role names sorted for a key
        rows.append((' | '.join(sorted([x.role, y.role])), ori, gap))
d = pd.DataFrame(rows, columns=['pair', 'orientation', 'gap_bp'])
s = d.groupby(['pair', 'orientation']).gap_bp.agg(['size', 'median', lambda v: np.percentile(v, 10), lambda v: np.percentile(v, 90)])
s.columns = ['n', 'median_bp', 'p10', 'p90']
print(s[s.n >= 50].sort_values('n', ascending=False).to_string())
