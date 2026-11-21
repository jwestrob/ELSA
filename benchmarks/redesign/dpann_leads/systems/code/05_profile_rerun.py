#!/usr/bin/env python
"""Rerun the caller's own DefenseFinder profiles (through Aksha) on the slot genes that complete a component.

Gene sets (one FASTA, labeled by set):
  test      every slot gene from screen (a) gains and screen (b) completion events
  pos       called-system members of the same components (reference: genes the caller accepted), <= 150 / component
  neg_same  slot genes with the same proposed label located outside every window of the model's hits and calls,
            length-matched (same 20-aa length bin) to the test genes of that label, sampled 1:1 (seed 17)
  neg_other slot genes with other labels, length-matched, sampled 1:1 (seed 17)
Profiles: all profiles (primary + exchangeables) of every component involved.
Thresholds read from the profiles: GA (full sequence, domain), LENG. MacSyFinder/DefenseFinder 2.0.2 rules as used
by Sharur: full >= GA1 and domain >= GA2, i-Evalue <= 1e-3, profile coverage >= 0.4. The i-Evalue is rescaled from
the Aksha run (Z = sequences searched) to the gene's genome proteome size, as DefenseFinder searches per genome.
"""
import shutil
import subprocess
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

W = Path('/Users/jacob/Documents/Sandbox/elsa_test/ELSA/benchmarks/redesign/dpann_leads/systems')
D = W / 'data'
RR = D / 'rerun'
RR.mkdir(exist_ok=True)
DB = '/Users/jacob/Documents/Sandbox/Sharur/Sharur/data/dpann/sharur.duckdb'
ASTRA = Path.home() / '.config/Astra/DefenseFinder'
rng = np.random.default_rng(17)

mg = pd.read_csv(D / 'df_model_genes.tsv', sep='\t')
cf = pd.read_csv(D / 'component_families.tsv', sep='\t').fillna({'families': ''})
slot = pd.read_parquet(D / 'slot.parquet')
sysdf = pd.read_parquet(D / 'df_systems.parquet')
ev_b = pd.read_csv(D / 'uncalled_completion_events.tsv', sep='\t')
win_a = pd.read_csv(D / 'defense_window_slot_genes.tsv', sep='\t').fillna({'gains': ''})
ev_a = win_a[win_a.gains != ''].assign(component=lambda x: x.gains.str.split(';')).explode('component')

test = pd.concat([ev_b[['protein_id', 'model', 'component', 'label']].assign(screen='b'),
                  ev_a[['protein_id', 'model', 'component', 'label']].assign(screen='a')], ignore_index=True)
comp_prof = {(r.model, r.component): r.profiles.split(';') for r in mg.itertuples()}
test['profiles'] = [comp_prof[(m, c)] for m, c in zip(test.model, test.component)]
profiles = sorted(set(p for ps in test.profiles for p in ps))
print('test rows', len(test), 'unique genes', test.protein_id.nunique(), 'profiles', len(profiles))

# positives: called members of the same components
comps = set(zip(test.model, test.component))
pos = sysdf[[(m, c) in comps for m, c in zip(sysdf.model, sysdf.component)]]
pos = pos.groupby(['model', 'component'], group_keys=False).apply(
    lambda g: g.sample(min(len(g), 150), random_state=17))
pos = pos[['protein_id', 'model', 'component']].assign(set='pos')

# negatives
con = duckdb.connect()
con.execute(f"ATTACH '{DB}' AS d (READ_ONLY)")
dfh = con.execute("SELECT DISTINCT a.protein_id, a.accession AS profile, p.contig_id, p.gene_index FROM d.annotations a "
                  "JOIN d.proteins p USING (protein_id) WHERE a.source='defensefinder'").df()
prof_models = {}
for r in mg.itertuples():
    for p in r.profiles.split(';'):
        prof_models.setdefault(p, set()).add(r.model)
test_models = set(test.model)
near = set()   # (contig, gene_index) positions within 6 genes of any hit / call of a test model
for r in dfh.itertuples():
    if prof_models.get(r.profile, set()) & test_models:
        for k in range(-6, 7):
            near.add((r.contig_id, r.gene_index + k))
for r in sysdf[sysdf.model.isin(test_models)].itertuples():
    for k in range(-6, 7):
        near.add((r.contig_id, r.gene_index + k))
slot['lbin'] = slot.length // 20
slot['near_model'] = [(c, g) in near for c, g in zip(slot.contig_id, slot.gene_index)]
tg = test.drop_duplicates('protein_id').merge(slot[['protein_id', 'lbin', 'length']], on='protein_id')
negs, nego = [], []
for (lab, lb), g in tg.groupby(['label', 'lbin']):
    pool = slot[(slot.label == lab) & (slot.lbin == lb) & ~slot.near_model & ~slot.protein_id.isin(tg.protein_id)]
    if len(pool):
        negs.append(pool.sample(min(len(g), len(pool)), random_state=17).assign(test_label=lab))
    fams = set().union(*[set(cf[(cf.model == m)].families.str.cat(sep=';').split(';')) for m in set(g.model)])
    pool2 = slot[(~slot.label.isin(fams)) & (slot.lbin == lb) & ~slot.protein_id.isin(tg.protein_id)]
    nego.append(pool2.sample(min(len(g), len(pool2)), random_state=17).assign(test_label=lab))
neg_same = pd.concat(negs)[['protein_id', 'test_label']].assign(set='neg_same')
neg_other = pd.concat(nego)[['protein_id', 'test_label']].assign(set='neg_other')
print('pos', len(pos), 'neg_same', len(neg_same), 'neg_other', len(neg_other))

sets = pd.concat([test.drop_duplicates('protein_id')[['protein_id', 'label']].rename(columns={'label': 'test_label'})
                  .assign(set='test'), pos[['protein_id']].assign(set='pos'), neg_same, neg_other],
                 ignore_index=True)
sets.to_csv(RR / 'gene_sets.tsv', sep='\t', index=False)
test.assign(profiles=[';'.join(p) for p in test.profiles]).to_csv(RR / 'test_events.tsv', sep='\t', index=False)
ids = sorted(set(sets.protein_id))
con.execute("CREATE TEMP TABLE ids(protein_id VARCHAR)")
con.executemany("INSERT INTO ids VALUES (?)", [(i,) for i in ids])
seqs = con.execute("SELECT p.protein_id, p.sequence, p.bin_id FROM d.proteins p JOIN ids USING (protein_id)").df()
with open(RR / 'genes.faa', 'w') as f:
    for r in seqs.itertuples():
        f.write(f'>{r.protein_id}\n{r.sequence.rstrip("*")}\n')
hd = Path('/private/tmp/claude-501/-Users-jacob-Documents-Sandbox-elsa-test-ELSA/4a153818-3d15-49e9-b3de-41cb21532c73/scratchpad/rerun_hmms')  # copied profiles stay outside the workstream folder
if hd.exists():
    shutil.rmtree(hd)
hd.mkdir(parents=True)
for p in profiles:
    shutil.copy(ASTRA / f'{p}.hmm', hd / f'{p}.hmm')
out = RR / 'aksha_out'
if out.exists():
    shutil.rmtree(out)
cmd = [str(Path.home() / '.local/bin/aksha'), 'search', '--hmm_in', str(hd), '--prot_in', str(RR / 'genes.faa'),
       '--outdir', str(out), '--evalue', '10', '--threads', '4']
print(' '.join(cmd))
subprocess.run(cmd, check=True)

# thresholds
thr = {}
file_of_name = {}
for p in profiles:
    ga1 = ga2 = None
    leng = None
    name = p
    for line in open(hd / f'{p}.hmm'):
        if line.startswith('NAME'):
            name = line.split()[1]
        if line.startswith('LENG'):
            leng = int(line.split()[1])
        if line.startswith('GA'):
            x = line.split()[1:3]
            ga1, ga2 = float(x[0]), float(x[1].rstrip(';'))
        if line.startswith('HMM '):
            break
    thr[name] = (ga1, ga2, leng)
    file_of_name[name] = p
hits = pd.read_csv(out / 'user_hmms_hits_df.tsv', sep='\t')
hits['profile_file'] = hits.hmm_name.map(file_of_name)
hits['GA1'] = hits.hmm_name.map(lambda p: thr[p][0])
hits['GA2'] = hits.hmm_name.map(lambda p: thr[p][1])
hits['LENG'] = hits.hmm_name.map(lambda p: thr[p][2])
hits['cov'] = (hits.hmm_to - hits.hmm_from + 1) / hits.LENG
nprot = con.execute("SELECT bin_id, count(*) n FROM d.proteins GROUP BY 1").df().set_index('bin_id').n
bin_of = dict(zip(seqs.protein_id, seqs.bin_id))
Zsearch = len(seqs)
hits['i_evalue_genome'] = hits.i_evalue * hits.sequence_id.map(lambda x: nprot.get(bin_of[x], 1000)) / Zsearch
hits['pass_ga'] = (hits.bitscore >= hits.GA1.fillna(-1e9)) & (hits.dom_bitscore >= hits.GA2.fillna(-1e9))
hits['pass_all'] = hits.pass_ga & (hits.i_evalue_genome <= 1e-3) & (hits['cov'] >= 0.4)
hits.to_csv(RR / 'hits_annotated.tsv', sep='\t', index=False)
(RR / 'profiles_used.txt').write_text('\n'.join(profiles) + '\n')
print('hits', len(hits), 'pass_ga', hits.pass_ga.sum(), 'pass_all', hits.pass_all.sum())
