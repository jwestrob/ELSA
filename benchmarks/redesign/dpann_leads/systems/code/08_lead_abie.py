#!/usr/bin/env python
"""AbiE lead: slot genes labeled as AbiEi antitoxin families that complete uncalled AbiEii-toxin loci.

Reports: counts by tier / list type / label family; gene order and strand relative to the toxin (abiEi upstream of
abiEii on the same strand is the characterized operon order); the same statistics for the antitoxins of the 336
called AbiE systems; CRISPR context (CCTyper calls within 10 genes); taxonomy distribution with clade denominators;
genome completeness; the toxin family on the cctyper side (cctyper also carries an AbiEii profile).
"""
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

W = Path('/Users/jacob/Documents/Sandbox/elsa_test/ELSA/benchmarks/redesign/dpann_leads/systems')
D = W / 'data'
DB = '/Users/jacob/Documents/Sandbox/Sharur/Sharur/data/dpann/sharur.duckdb'
ev = pd.read_csv(D / 'uncalled_completion_events.tsv', sep='\t')
sc = pd.read_csv(D / 'rerun' / 'test_event_scores.tsv', sep='\t')
sysdf = pd.read_parquet(D / 'df_systems.parquet')
slot = pd.read_parquet(D / 'slot.parquet')
con = duckdb.connect()
con.execute(f"ATTACH '{DB}' AS d (READ_ONLY)")
prot = con.execute("SELECT protein_id, contig_id, gene_index, strand, bin_id, sequence_length FROM d.proteins").df()
pi = prot.set_index('protein_id')
bins = con.execute("SELECT bin_id, completeness, taxonomy FROM d.bins").df()
bins['phylum'] = bins.taxonomy.str.extract(r'(p__[^;]*)')[0]
bins['order'] = bins.taxonomy.str.extract(r'(o__[^;]*)')[0]
bins['species'] = bins.taxonomy.str.extract(r'(s__[^;]*)')[0]

ab = ev[(ev.model == 'AbiE') & (ev.component == 'AbiEii__AbiEi_1') & ev.completes].copy()
ab = ab.drop_duplicates('cluster_id')     # one antitoxin per locus (closest kept by order of the events table)
lines = []
P = lines.append
P(f'AbiE completions: loci {ab.cluster_id.nunique()}, slot genes {ab.protein_id.nunique()}, genomes '
  f'{ab.genome.nunique()}; labels {ab.label_name.value_counts().to_dict()}')
P(f'  tier {ab.tier.value_counts().to_dict()}')
P(f'  list type {ab.list_type.value_counts().to_dict()}')
P(f'  median length {ab.length.median():.0f} aa (IQR {ab.length.quantile(.25):.0f}-{ab.length.quantile(.75):.0f})')
P(f'  vote support (vfrac) median {ab.vfrac.median():.3f}; votes_for median {ab.votes_for.median():.0f}')


def order_strand(at_pid, tox_pids):
    a = pi.loc[at_pid]
    best = None
    for t in tox_pids:
        if t not in pi.index:
            continue
        b = pi.loc[t]
        if b.contig_id != a.contig_id:
            continue
        dd = abs(int(b.gene_index) - int(a.gene_index))
        if best is None or dd < best[0]:
            best = (dd, b)
    if best is None:
        return None
    dd, b = best
    same = str(a.strand) == str(b.strand)
    # upstream in transcription direction: on + strand lower index; on - strand higher index
    up = (a.gene_index < b.gene_index) if str(b.strand) in ('+', '1') else (a.gene_index > b.gene_index)
    return dict(dist=dd, same_strand=same, at_upstream=bool(up))


ab_os = [order_strand(r.protein_id, r.cluster_proteins.split(';')) for r in ab.itertuples()]
ab['same_strand'] = [x['same_strand'] if x else None for x in ab_os]
ab['at_upstream'] = [x['at_upstream'] if x else None for x in ab_os]
ab['gdist'] = [x['dist'] if x else None for x in ab_os]
P(f'  slot antitoxin vs toxin: same strand {ab.same_strand.mean():.3f}; antitoxin upstream (5\') '
  f'{ab[ab.same_strand == True].at_upstream.mean():.3f} of same-strand; adjacent (1 gene) {(ab.gdist == 1).mean():.3f}')
cal = sysdf[sysdf.model == 'AbiE']
rows = []
for sid, g in cal.groupby('system_id'):
    at = g[g.component == 'AbiEii__AbiEi_1'].protein_id.tolist()
    tx = g[g.component == 'AbiEii__AbiEii'].protein_id.tolist()
    if at and tx:
        x = order_strand(at[0], tx)
        if x:
            rows.append(x)
cr = pd.DataFrame(rows)
P(f'  called AbiE systems ({cal.system_id.nunique()}): same strand {cr.same_strand.mean():.3f}; antitoxin upstream '
  f'{cr[cr.same_strand].at_upstream.mean():.3f}; adjacent {(cr.dist == 1).mean():.3f}')
cfam = con.execute(f"""SELECT a.accession, any_value(a.name) nm, count(DISTINCT a.protein_id) n FROM d.annotations a
  WHERE a.source='pfam' AND a.protein_id IN ({','.join("'" + p + "'" for p in cal[cal.component ==
  'AbiEii__AbiEi_1'].protein_id)}) GROUP BY 1 ORDER BY 3 DESC LIMIT 6""").df()
P('  called AbiE antitoxins, Pfam: ' + '; '.join(f'{r.accession} {r.nm} {r.n}' for r in cfam.itertuples()))

# profile rerun on these genes
s = sc[(sc.model == 'AbiE') & (sc.component == 'AbiEii__AbiEi_1') & sc.protein_id.isin(ab.protein_id)]
s = s.groupby('protein_id').agg(any_hit=('any_hit', 'max'), bits=('bitscore', 'max'), prof=('best_profile', 'first'),
                                cov=('cov', 'max'))
P(f'  caller AbiEi profiles (Aksha, E<=10): hit {s.any_hit.sum()}/{len(s)} ({s.any_hit.mean():.3f}); median bits '
  f'{s.bits.median():.1f}; best profiles {s.prof.value_counts().head(4).to_dict()}; GA passes 0')
ab = ab.merge(s[['any_hit', 'bits']], left_on='protein_id', right_index=True, how='left')

# CRISPR context of the toxins
tox = set(p for x in ab.cluster_proteins for p in x.split(';'))
cct = con.execute("""SELECT sp.protein_id, sp.profile_name, c.status FROM d.system_proteins sp
  JOIN d.crispr_cas_systems c USING (system_id) WHERE sp.system_source='cctyper'""").df()
cpro = cct.set_index('protein_id')
tox_cct = cct[cct.protein_id.isin(tox)]
P(f'  toxins also flagged by cctyper: {tox_cct.protein_id.nunique()}/{len(tox)} '
  f'({tox_cct.profile_name.value_counts().head(3).to_dict()}; status {tox_cct.status.value_counts().to_dict()})')
complete = cct[cct.status != 'cas_putative'].merge(prot, on='protein_id')
cpos = complete.groupby('contig_id').gene_index.apply(np.array).to_dict()
near = 0
for t in tox:
    if t not in pi.index:
        continue
    r = pi.loc[t]
    arr = cpos.get(r.contig_id)
    if arr is not None and np.any(np.abs(arr - r.gene_index) <= 10):
        near += 1
P(f'  toxins within 10 genes of a complete CRISPR-Cas call: {near}/{len(tox)}')

# taxonomy and completeness
ab = ab.merge(bins, left_on='genome', right_on='bin_id', how='left')
called_g = set(cal.genome_id)
any_g = set(ab.genome) | called_g
den = bins.groupby('phylum').bin_id.nunique()
num_c = bins[bins.bin_id.isin(called_g)].groupby('phylum').bin_id.nunique()
num_any = bins[bins.bin_id.isin(any_g)].groupby('phylum').bin_id.nunique()
tx = pd.DataFrame({'genomes': den, 'with_called_AbiE': num_c, 'with_called_or_completed': num_any}).fillna(0)
tx = tx[tx.with_called_or_completed > 0].astype(int).sort_values('genomes', ascending=False)
P('  genomes with AbiE by phylum (denominator = phylum genomes in DB):')
P(tx.to_string())
P(f'  genomes with >=1 called AbiE {len(called_g)}; with >=1 called or completed {len(any_g)} of {len(bins)}')
P(f'  completion genomes: species {ab.species.nunique()}, orders {ab.order.nunique()}; completeness median '
  f'{ab.completeness.median():.1f}')
ab.to_csv(D / 'lead_abie_loci.tsv', sep='\t', index=False)
(D / 'lead_abie_summary.txt').write_text('\n'.join(lines) + '\n')
print('\n'.join(lines))
