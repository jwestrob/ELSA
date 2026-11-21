#!/usr/bin/env python
"""Build the shared context tables for the `systems` workstream (read-only inputs).

Outputs (systems/data/):
  df_model_genes.tsv      every DefenseFinder 2.0.2 model component: model, component (primary profile), presence,
                          profiles (primary + exchangeables), model quorum and inter_gene_max_space
  df_systems.parquet      called DefenseFinder systems with member proteins, gene_index, profile, component
  slot.parquet            the 175,729 slot-labeled genes joined to verification tier and DB coordinates
  profile_family.parquet  co-annotation of each DefenseFinder profile with Pfam / KO accessions on the same DPANN
                          proteins (protein level and domain-overlap level)
"""
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import duckdb
import pandas as pd

R = Path('/Users/jacob/Documents/Sandbox/elsa_test/ELSA')
OUT = R / 'benchmarks/redesign/dpann_leads/systems/data'
OUT.mkdir(parents=True, exist_ok=True)
DB = '/Users/jacob/Documents/Sandbox/Sharur/Sharur/data/dpann/sharur.duckdb'
DEFS = Path.home() / '.macsyfinder/models/defense-finder-models/definitions'

# ---------------------------------------------------------------- model definitions
rows = []
for xml in sorted(DEFS.rglob('*.xml')):
    root = ET.parse(xml).getroot()
    model = xml.stem
    cat = str(xml.parent.relative_to(DEFS))
    igms = int(root.get('inter_gene_max_space', '5'))
    minm = int(root.get('min_mandatory_genes_required', '1'))
    ming = int(root.get('min_genes_required', '1'))
    genes = root.findall('gene')
    n_ma = sum(g.get('presence', 'mandatory') in ('mandatory', 'accessory') for g in genes)
    maxg = root.get('max_nb_genes')
    for g in genes:
        prim = g.get('name')
        exch = [e.get('name') for e in g.findall('exchangeables/gene')]
        rows.append(dict(model=model, category=cat, component=prim, presence=g.get('presence', 'mandatory'),
                         profiles=';'.join([prim] + exch), loner=g.get('loner', 'false'),
                         multi_system=g.get('multi_system', 'false'),
                         gene_igms=g.get('inter_gene_max_space'), inter_gene_max_space=igms,
                         min_mandatory=minm, min_genes=ming, n_mand_acc=n_ma,
                         max_nb_genes=int(maxg) if maxg else n_ma))
mg = pd.DataFrame(rows)
mg.to_csv(OUT / 'df_model_genes.tsv', sep='\t', index=False)
print('models', mg.model.nunique(), 'component rows', len(mg))

con = duckdb.connect()
con.execute(f"ATTACH '{DB}' AS d (READ_ONLY)")
con.execute("SET threads=4")

# ---------------------------------------------------------------- called systems
sysdf = con.execute("""
  SELECT s.system_id, s.genome_id, s.system_type AS model, s.system_subtype, s.genes_count, s.contig_id AS sys_contig,
         sp.protein_id, sp.profile_name, sp.position, p.contig_id, p.gene_index, p.strand, p.sequence_length
  FROM d.defense_systems s JOIN d.system_proteins sp ON sp.system_id = s.system_id
       AND sp.system_source = 'defensefinder_system'
  JOIN d.proteins p ON p.protein_id = sp.protein_id
""").df()
# profile -> component within the system's own model
pc = {}
for r in mg.itertuples():
    for prof in r.profiles.split(';'):
        pc[(r.model, prof)] = (r.component, r.presence)
sysdf['component'] = [pc.get((m, p), (None, None))[0] for m, p in zip(sysdf.model, sysdf.profile_name)]
sysdf['presence'] = [pc.get((m, p), (None, None))[1] for m, p in zip(sysdf.model, sysdf.profile_name)]
print('system member rows', len(sysdf), 'systems', sysdf.system_id.nunique(),
      'members without component mapping', sysdf.component.isna().sum(),
      'models missing from definitions', sorted(set(sysdf.model) - set(mg.model)))
sysdf.to_parquet(OUT / 'df_systems.parquet', index=False)

# ---------------------------------------------------------------- slot table
slot = con.execute(f"""
  SELECT s.protein_id, s.genome, s.contig, s.position_index, s.strand, s.length, s.own_ko,
         s.proposed_label_positional_homology AS label, s.proposed_label_name AS label_name,
         s.proposed_label_kind AS label_kind, s.votes_for, s.voters_annotated, s.vote_share, s.supporting_genomes,
         s.supporting_groups, s.locus_id, s.locus_size, s.alignment_label, s.alignment_lists_same_label,
         s.confidence_bin, s.calibration_agreement, s.own_ko_agrees,
         v.tier, v.vfrac, v.amp, v.annotated, v.proposed_evalue, v.best_other_label, v.best_other_evalue,
         p.contig_id, p.gene_index, p.bin_id, p.sequence_length
  FROM read_parquet('{R}/benchmarks/redesign/slot_rescue/pass/dpann_slot_rescue_pass.parquet') s
  JOIN read_parquet('{R}/benchmarks/redesign/slot_rescue/verify/verification_per_protein.parquet') v USING (protein_id)
  JOIN d.proteins p USING (protein_id)
""").df()
assert len(slot) == 175729, len(slot)
assert (slot.contig == slot.contig_id).all() and (slot.position_index == slot.gene_index).all()
slot['list_type'] = slot.alignment_lists_same_label.map(
    {'yes': 'both', 'no call': 'embedding_only', 'no list': 'embedding_only', 'no': 'lists_disagree'})
slot.to_parquet(OUT / 'slot.parquet', index=False)
print('slot rows', len(slot), slot.list_type.value_counts().to_dict())

# ---------------------------------------------------------------- profile <-> family co-annotation
pf = con.execute("""
  WITH dfh AS (SELECT protein_id, accession AS profile, min(start_aa) s1, max(end_aa) e1
               FROM d.annotations WHERE source = 'defensefinder' GROUP BY 1, 2),
       fam AS (SELECT protein_id, accession AS family, source AS fam_source, min(start_aa) s2, max(end_aa) e2
               FROM d.annotations WHERE source IN ('pfam', 'kofam') GROUP BY 1, 2, 3)
  SELECT dfh.profile, fam.family, fam.fam_source, count(*) AS n_both,
         sum(CASE WHEN least(e1, e2) - greatest(s1, s2) + 1 >= 0.5 * least(e1 - s1 + 1, e2 - s2 + 1)
                  THEN 1 ELSE 0 END) AS n_overlap
  FROM dfh JOIN fam USING (protein_id) GROUP BY 1, 2, 3
""").df()
nprof = con.execute("SELECT accession AS profile, count(DISTINCT protein_id) n_profile FROM d.annotations "
                    "WHERE source='defensefinder' GROUP BY 1").df()
nfam = con.execute("SELECT accession AS family, count(DISTINCT protein_id) n_family FROM d.annotations "
                   "WHERE source IN ('pfam','kofam') GROUP BY 1").df()
pf = pf.merge(nprof, on='profile').merge(nfam, on='family')
pf['p_fam_given_prof'] = pf.n_both / pf.n_profile
pf['p_prof_given_fam'] = pf.n_both / pf.n_family
pf['p_overlap_given_prof'] = pf.n_overlap / pf.n_profile
pf.to_parquet(OUT / 'profile_family.parquet', index=False)
print('profile-family pairs', len(pf), 'profiles', pf.profile.nunique())
json.dump({'defense_finder_models': '2.0.2', 'definitions': str(DEFS)}, open(OUT / 'context_meta.json', 'w'))
