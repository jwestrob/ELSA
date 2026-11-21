"""Per-role FASTA (all members) and a <=250-sequence diverse subset for MSAs; per-role composition stats."""
import pandas as pd, numpy as np, os, subprocess
a = pd.read_parquet('data/f17_windows.parquet')
a = a[a.role != '-'].drop_duplicates('protein_id').copy()
a['seq'] = a.sequence.str.rstrip('*')
a['hyd'] = a.seq.map(lambda s: sum(c in 'ILVFMAWC' for c in s) / len(s))
os.makedirs('data/roles', exist_ok=True)
# MP4610 split by mmseqs subcluster
mp = {'MEM3791321.1': 'MP4610_s1', 'JAKAVK010000018.1_19': 'MP4610_s2', 'JAJZAJ010000002.1_210': 'MP4610_s3',
      'JAJZLT010000016.1_6': 'MP4610_s4', 'JAKASF010000001.1_18': 'MP4610_s5'}
a['grp'] = np.where(a.role == 'MP4610', a.rep.map(mp), a.role)
rng = np.random.default_rng(17)
rows = []
for g, s in a.groupby('grp'):
    med = s.sequence_length.median()
    rows.append((g, len(s), s.locus.nunique(), int(med), round(s.hyd.median(), 3)))
    open(f'data/roles/{g}.all.faa', 'w').write(''.join(f'>{r.protein_id}\n{r.seq}\n' for r in s.itertuples()))
    t = s[(s.sequence_length > 0.8 * med) & (s.sequence_length < 1.2 * med)]
    t = t.sample(min(250, len(t)), random_state=17)
    open(f'data/roles/{g}.sub.faa', 'w').write(''.join(f'>{r.protein_id}\n{r.seq}\n' for r in t.itertuples()))
st = pd.DataFrame(rows, columns=['group', 'n', 'loci', 'median_aa', 'hyd_median'])
st.to_csv('data/role_stats.tsv', sep='\t', index=False)
print(st.to_string())
