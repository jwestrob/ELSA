#!/usr/bin/env python
"""CRISPR-Cas (CCTyper port 1.9.0) and TXSScan calls: slot genes inside / next to called loci, label agreement for
slot genes that are already system members, and (a)-style gains for complete CRISPR-Cas calls.

CCTyper composition: ~/.config/Astra/CCTyper/interference.json + adaptation.json (gene-name prefixes per subtype,
an inner list = alternatives). A call lacks a gene when no member profile name starts with any alternative.
Profile -> family map: DPANN co-annotation of cctyper member profiles with Pfam / KO on the same protein
(>= 3 proteins and >= 20% of the profile's proteins), as for DefenseFinder.
Complete calls = status in (cas, crispr_cas, crispr_cas_putative); cas_putative (single accessory hits) are counted
separately. Archaeal-T4P (TXSScan) is a many-subunit machine and is reported for counts only.
"""
import json
from pathlib import Path

import duckdb
import pandas as pd

W = Path('/Users/jacob/Documents/Sandbox/elsa_test/ELSA/benchmarks/redesign/dpann_leads/systems')
D = W / 'data'
DB = '/Users/jacob/Documents/Sandbox/Sharur/Sharur/data/dpann/sharur.duckdb'
CC = Path.home() / '.config/Astra/CCTyper'
slot = pd.read_parquet(D / 'slot.parquet')
con = duckdb.connect()
con.execute(f"ATTACH '{DB}' AS d (READ_ONLY)")

mem = con.execute("""
  SELECT sp.system_id, sp.system_source, sp.protein_id, sp.profile_name, p.contig_id, p.gene_index, p.bin_id,
         c.status, c.prediction, c.best_type, s.system_type
  FROM d.system_proteins sp JOIN d.proteins p USING (protein_id)
  LEFT JOIN d.crispr_cas_systems c ON c.system_id = sp.system_id
  LEFT JOIN d.secretion_systems s ON s.system_id = sp.system_id
  WHERE sp.system_source IN ('cctyper', 'txsscan_system')
""").df()
fam = con.execute("SELECT protein_id, accession FROM d.annotations WHERE source IN ('pfam','kofam')").df()
fam_by_p = fam.groupby('protein_id').accession.apply(set).to_dict()

# profile -> family map
cc = mem[mem.system_source == 'cctyper']
rows = []
for prof, g in cc.groupby('profile_name'):
    n = g.protein_id.nunique()
    cnt = {}
    for pid in set(g.protein_id):
        for f in fam_by_p.get(pid, ()):
            cnt[f] = cnt.get(f, 0) + 1
    for f, k in cnt.items():
        rows.append((prof, f, k, n))
pfm = pd.DataFrame(rows, columns=['profile', 'family', 'n_both', 'n_profile'])
pfm = pfm[(pfm.n_both >= 3) & (pfm.n_both / pfm.n_profile >= 0.2)]
prof2fam = pfm.groupby('profile').family.apply(set).to_dict()
prefix = lambda p: p.split('_')[0]
gene2fam = {}
for p, fs in prof2fam.items():
    gene2fam.setdefault(prefix(p), set()).update(fs)

lines = []
P = lines.append
# label agreement for slot genes that are members
ms = mem.merge(slot[['protein_id', 'label', 'label_name', 'tier', 'list_type']], on='protein_id')
for src, g in ms.groupby('system_source'):
    if src == 'cctyper':
        g = g.assign(agree=[lab in prof2fam.get(p, set()) for lab, p in zip(g.label, g.profile_name)],
                     mappable=[bool(prof2fam.get(p)) for p in g.profile_name])
        cplt = g.status.isin(['cas', 'crispr_cas', 'crispr_cas_putative'])
        for nm, x in [('complete calls', g[cplt]), ('cas_putative', g[~cplt])]:
            xm = x[x.mappable]
            P(f'cctyper {nm}: slot genes that are members {x.protein_id.nunique()}; with mappable profile '
              f'{xm.protein_id.nunique()}; slot label in the profile family set {xm.agree.sum()} '
              f'({xm.agree.mean():.3f})')
    else:
        P(f'txsscan members that are slot genes: {g.protein_id.nunique()} '
          f'({g.system_type.value_counts().to_dict()}); top labels '
          f'{g.label_name.value_counts().head(5).to_dict()}')
ms.to_csv(D / 'cctyper_txss_member_slot_genes.tsv', sep='\t', index=False)

# gains for complete CRISPR-Cas calls
inter = json.load(open(CC / 'interference.json'))
adapt = json.load(open(CC / 'adaptation.json'))
comp = cc[cc.status.isin(['cas', 'crispr_cas', 'crispr_cas_putative'])]
SB = {c: g.sort_values('gene_index') for c, g in slot.groupby('contig_id')}
members = set(mem.protein_id)
grows, n_sys, n_lack, n_adj, n_win = [], 0, 0, 0, 0
for sid, g in comp.groupby('system_id'):
    n_sys += 1
    typ = g.prediction.iloc[0] if g.prediction.iloc[0] in inter else g.best_type.iloc[0]
    req = inter.get(typ, []) + adapt.get(typ, [])
    prefs = [prefix(p) for p in g.profile_name]
    lacking = [alts for alts in req if not any(pp.startswith(a.rstrip('_')) for pp in prefs for a in alts)]
    n_lack += bool(lacking)
    contig, lo, hi = g.contig_id.iloc[0], g.gene_index.min(), g.gene_index.max()
    s = SB.get(contig)
    if s is None:
        continue
    w = s[(s.gene_index >= lo - 3) & (s.gene_index <= hi + 3) & ~s.protein_id.isin(members)]
    n_win += len(w) > 0
    n_adj += len(w[(w.gene_index >= lo - 1) & (w.gene_index <= hi + 1)]) > 0
    for r in w.itertuples():
        d = 0 if lo < r.gene_index < hi else min(abs(r.gene_index - lo), abs(r.gene_index - hi))
        gains = [a[0] for a in lacking if any(r.label in gene2fam.get(x.rstrip('_'), set()) for x in a)]
        grows.append(dict(system_id=sid, type=typ, protein_id=r.protein_id, label=r.label,
                          label_name=r.label_name, dist=d, gains=';'.join(gains), tier=r.tier,
                          list_type=r.list_type, length=r.length))
gw = pd.DataFrame(grows)
gw.to_csv(D / 'cctyper_window_slot_genes.tsv', sep='\t', index=False)
P(f'complete CRISPR-Cas calls {n_sys}; lacking >= 1 interference/adaptation gene {n_lack}; with slot gene within '
  f'+/-3 genes {n_win}; inside/adjacent {n_adj}; gain events {int((gw.gains != "").sum()) if len(gw) else 0} in '
  f'{gw[gw.gains != ""].system_id.nunique() if len(gw) else 0} calls')
if len(gw):
    P('gain events by gene: ' + str(gw[gw.gains != ''].gains.value_counts().to_dict()))
    P('labels in CRISPR windows (top 15): ' + str(gw.label_name.value_counts().head(15).to_dict()))
    P(gw[gw.gains != ''][['system_id', 'type', 'protein_id', 'label_name', 'gains', 'dist', 'tier', 'list_type',
                          'length']].to_string())
t4 = mem[mem.system_type == 'Archaeal-T4P']
P(f'Archaeal-T4P calls {t4.system_id.nunique()} (many-subunit machine; counts only)')
(D / 'screen_cctyper_txss_summary.txt').write_text('\n'.join(lines) + '\n')
print('\n'.join(lines))
