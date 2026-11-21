#!/usr/bin/env python
"""Lineage denominators and neighborhood context for the CBASS and Hachiman leads (biotite login node, serial).
Per lead locus: called systems of the same type in the genome, in the genome's class and phylum (genomes with a call /
genomes in the collection); genes within +-6 of the slot gene with their caller-profile hits (03 search, small models
only) and Pfam names. Also AbiE: called vs completed genomes by phylum (top 12). Writes data/lead_context.txt."""
from pathlib import Path
import pandas as pd
RD = Path('/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111')
D = RD / 'dpann_leads/systems_srvp/data'
ll = pd.read_csv(D / 'lead_loci.tsv', sep='\t')
sy = pd.read_parquet(D / 'systems.parquet')
hits = pd.read_parquet(D / 'hits.parquet')
genes = pd.read_parquet(RD / 'ws_srvp_bins/package/universe/genes.parquet',
                        columns=['gid', 'genome', 'contig_code', 'position_index', 'strand', 'length'])
gen = pd.read_parquet(RD / 'ws_srvp_bins/package/universe/genomes.parquet',
                      columns=['genome', 'domain', 'phylum', 'class', 'ord', 'genus'])
sy['stype'] = sy.model_key.map(lambda m: m.split('/')[1])
sy = sy.merge(gen, on='genome')
L = []
for t in ['CBASS', 'Hachiman', 'Paris']:
    x = ll[ll.stype == t].drop_duplicates('gid')
    called = sy[sy.stype == t]
    L.append(f'== {t}: called systems {len(called)} in {called.genome.nunique()} genomes of 994')
    for r in x.itertuples():
        cls_ = r[ll.columns.get_loc('class') + 1]
        c_cls = called[called['class'] == cls_].genome.nunique()
        n_cls = int((gen['class'] == cls_).sum())
        c_ph = called[called.phylum == r.phylum].genome.nunique()
        n_ph = (gen.phylum == r.phylum).sum()
        L.append(f'  gid {r.gid} {r.label_name} {r.length} aa {r.phylum} / {r[ll.columns.get_loc("class") + 1]} / '
                 f'{r.genus}: genomes with a called {t} in class {c_cls}/{n_cls}, phylum {c_ph}/{n_ph}; '
                 f'other defense calls in this genome: {sy[sy.genome == r.genome].stype.value_counts().to_dict()}')
        g = genes[(genes.genome == r.genome) & (genes.contig_code == r.contig_code) &
                  genes.position_index.between(r.position_index - 6, r.position_index + 6)]
        pf = pd.read_parquet(RD / 'annot_gpu/srvp/PFAM.parquet', columns=['gid', 'name'],
                             filters=[('gid', 'in', g.gid.tolist())]).groupby('gid').name.apply(lambda s: ','.join(dict.fromkeys(s)))
        hp = hits[hits.gid.isin(g.gid)].groupby('gid').profile.apply(lambda s: ','.join(s))
        for q in g.sort_values('position_index').itertuples():
            mark = '*' if q.gid == r.gid else ' '
            L.append(f'     {mark} {q.position_index - r.position_index:+d} {"+" if q.strand == 1 else "-"} {q.length:4d} aa '
                     f'pfam[{pf.get(q.gid, "")}] caller[{hp.get(q.gid, "")}]')
ab = ll[ll.stype == 'AbiE'].drop_duplicates('gid')
cal = sy[sy.stype == 'AbiE']
den = gen.groupby('phylum').size()
before = cal.groupby('phylum').genome.nunique()
after = pd.concat([cal[['genome', 'phylum']], ab[['genome', 'phylum']]]).drop_duplicates().groupby('phylum').genome.nunique()
t = pd.DataFrame({'genomes': den, 'called_AbiE': before, 'called_or_completed': after}).fillna(0).astype(int)
L.append('== AbiE genomes by phylum (called -> called or completed / phylum genomes)')
L.append(t.sort_values('called_or_completed', ascending=False).head(14).to_string())
both = pd.concat([cal[['genome', 'domain']], ab[['genome', 'domain']]]).drop_duplicates()
n_arch_after = int((both.domain == 'Archaea').sum())
n_arch = int((gen.domain == 'Archaea').sum())
L.append(f'AbiE genomes: called {cal.genome.nunique()}, called or completed {len(both)} of 994; archaeal genomes with '
         f'AbiE: called {cal[cal.domain == "Archaea"].genome.nunique()}, after {n_arch_after} of {n_arch}')
(D / 'lead_context.txt').write_text('\n'.join(L) + '\n')
print('\n'.join(L))
