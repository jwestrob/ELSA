#!/usr/bin/env python
"""Verification (c): rerun the callers' own profiles for the proposed component on the completing / gaining slot genes,
through Aksha (picked profiles, E <= 10; biotite login node, 8 threads), with controls.

Gene sets:
  test       slot genes from screen (a) gains and screen (b) gain events (complete or not)
  pos        called-system members of the same (model, component), <= 150 per component (seed 17)
  neg_same   slot genes with the same label, > 6 genes from every hit or call of the test models, same 20-aa length
             bin as the test genes of that label, 1:1 (seed 17)
  neg_other  slot genes whose label is outside every family set of the involved models, length-matched 1:1 (seed 17)
Scores against each profile's own GA (full sequence GA1 and domain GA2) and profile coverage (DefenseFinder 0.4,
TXSScan 0.5). i-Evalue: P = i_evalue / n_reported_for_profile_in_this_search, reported as P (the genome-level
i-Evalue when the gene is the only reported target of that profile in its genome).
Outputs data/rerun/: gene_sets.tsv, test_events.tsv, hits_annotated.tsv, profiles_used.txt (sequences stay on biotite).
"""
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

RD = Path('/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111')
D = RD / 'dpann_leads/systems_srvp/data'
RR = D / 'rerun'
RR.mkdir(exist_ok=True)
mg = pd.read_csv(D / 'model_genes.tsv', sep='\t')
pr = pd.read_csv(D / 'profiles.tsv', sep='\t')
cf = pd.read_csv(D / 'component_families.tsv', sep='\t').fillna({'families': ''})
slot = pd.read_parquet(D / 'slot.parquet')
cm = pd.read_parquet(D / 'call_members.parquet')
sy = pd.read_parquet(D / 'systems.parquet')
hits = pd.read_parquet(D / 'hits.parquet')
genes = pd.read_parquet(RD / 'ws_srvp_bins/package/universe/genes.parquet',
                        columns=['gid', 'genome', 'contig_code', 'position_index'])
gi = genes.set_index('gid')

wa = pd.read_parquet(D / 'screen_a_window.parquet')
wa = wa[wa.gains != ''].assign(component=lambda x: x.gains.str.split(';')).explode('component')
eb = pd.read_parquet(D / 'screen_b_events.parquet')
test = pd.concat([wa[['gid', 'model_key', 'component', 'label']].assign(screen='a'),
                  eb[['gid', 'model_key', 'component', 'label']].assign(screen='b')], ignore_index=True)
test = test.drop_duplicates()
mcaller = mg.groupby('model_key').caller.first().to_dict()
comp_prof = {(r.model_key, r.component): r.profiles.split(';') for r in mg.itertuples()}
test['caller'] = test.model_key.map(mcaller)
test['profiles'] = [';'.join(comp_prof[(m, c)]) for m, c in zip(test.model_key, test.component)]
need = sorted(set((c, p) for c, ps in zip(test.caller, test.profiles) for p in ps.split(';')))
print('test rows', len(test), 'unique genes', test.gid.nunique(), 'profiles', len(need))

rng = np.random.default_rng(17)
comps = set(zip(test.model_key, test.component))
pos = cm[[(m, c) in comps for m, c in zip(cm.model_key, cm.component)]]
pos = pos.groupby(['model_key', 'component'], group_keys=False).apply(lambda g: g.sample(min(len(g), 150), random_state=17))
pos = pos[['gid', 'model_key', 'component']].drop_duplicates('gid')

test_models = set(test.model_key)
prof_models = {}
for r in mg.itertuples():
    for p in r.profiles.split(';'):
        prof_models.setdefault((r.caller, p), set()).add(r.model_key)
near = set()
for r in hits.itertuples():
    if prof_models.get((r.caller, r.profile), set()) & test_models:
        for k in range(-6, 7):
            near.add((r.genome, r.contig_code, r.position_index + k))
for r in sy[sy.model_key.isin(test_models)].itertuples():
    for k in range(r.lo - 6, r.hi + 7):
        near.add((r.genome, r.contig_code, k))
slot['lbin'] = slot.length // 20
slot['near_model'] = [(g, c, p) in near for g, c, p in zip(slot.genome, slot.contig_code, slot.position_index)]
tg = test.drop_duplicates('gid').merge(slot[['gid', 'lbin']], on='gid')
fam_models = cf.groupby('model_key').families.apply(lambda s: set(';'.join(s).split(';')) - {''}).to_dict()
negs, nego = [], []
for (lab, lb), g in tg.groupby(['label', 'lbin']):
    pool = slot[(slot.label == lab) & (slot.lbin == lb) & ~slot.near_model & ~slot.gid.isin(tg.gid)]
    if len(pool):
        negs.append(pool.sample(min(len(g), len(pool)), random_state=17).assign(test_label=lab))
    fams = set().union(*[fam_models.get(m, set()) for m in set(test[test.label == lab].model_key)])
    pool2 = slot[~slot.label.isin(fams) & (slot.lbin == lb) & ~slot.gid.isin(tg.gid)]
    nego.append(pool2.sample(min(len(g), len(pool2)), random_state=17).assign(test_label=lab))
neg_same = pd.concat(negs)[['gid', 'test_label']].assign(set='neg_same') if negs else pd.DataFrame()
neg_other = pd.concat(nego)[['gid', 'test_label']].assign(set='neg_other')
sets = pd.concat([test.drop_duplicates('gid')[['gid', 'label']].rename(columns={'label': 'test_label'}).assign(set='test'),
                  pos[['gid']].assign(set='pos'), neg_same, neg_other], ignore_index=True)
sets.to_csv(RR / 'gene_sets.tsv', sep='\t', index=False)
test.to_csv(RR / 'test_events.tsv', sep='\t', index=False)
print('sets', sets.set.value_counts().to_dict())

ids = set(sets.gid)
work = RD / 'dpann_leads/systems_srvp/rerun_work'
if work.exists():
    shutil.rmtree(work)
(work / 'hmm').mkdir(parents=True)
with open(RD / 'ws_srvp_bins/package/universe/proteins.faa') as f, open(work / 'genes.faa', 'w') as o:
    keep = False
    for line in f:
        if line[0] == '>':
            keep = int(line[1:]) in ids
        if keep:
            o.write(line)
prp = pr.set_index(['caller', 'profile'])
name2 = {}
for c, p in need:
    path = prp.loc[(c, p)].path
    shutil.copyfile(path, work / 'hmm' / f'{c}__{p}.hmm')
    nm = next(l.split()[1] for l in open(path) if l.startswith('NAME'))
    name2[nm] = (c, p)
subprocess.run(['/home/jwestrob/.local/bin/aksha', 'search', '--hmm_in', str(work / 'hmm'), '--prot_in',
                str(work / 'genes.faa'), '--outdir', str(work / 'out'), '--evalue', '10', '--threads', '8'], check=True,
               stdout=subprocess.DEVNULL)
h = pd.read_csv(work / 'out/user_hmms_hits_df.tsv', sep='\t').rename(columns={'sequence_id': 'gid'})
h['caller'] = h.hmm_name.map(lambda n: name2[n][0])
h['profile'] = h.hmm_name.map(lambda n: name2[n][1])
h = h.merge(pr[['caller', 'profile', 'LENG', 'GA1', 'GA2']], on=['caller', 'profile'], how='left')
h['cov'] = (h.hmm_to - h.hmm_from + 1) / h.LENG
h['n_rep'] = h.groupby('profile').gid.transform('nunique')
h['P'] = h.i_evalue / h.n_rep
h['pass_ga'] = (h.bitscore >= h.GA1) & (h.dom_bitscore >= h.GA2)
h['cov_ok'] = h['cov'] >= h.caller.map({'DefenseFinder': 0.4, 'TXSScan': 0.5})
h['pass_all'] = h.pass_ga & h.cov_ok & (h.P <= 1e-3)
h.to_csv(RR / 'hits_annotated.tsv', sep='\t', index=False)
(RR / 'profiles_used.txt').write_text('\n'.join(f'{c}\t{p}' for c, p in need) + '\n')
print('hit rows', len(h), 'pass_ga', int(h.pass_ga.sum()), 'pass_all', int(h.pass_all.sum()))
