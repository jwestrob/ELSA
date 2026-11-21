lut test set, large size (package_k8): 1,032 GTDB r232 bacterial species representatives built around loci homologous
to the Methylorubrum extorquens AM1 lut cluster (workstream slice_lut, part 2; built 2026-11-13 on biotite).
Location: /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/slice_lut/testset/package_k8/
Status: untouched test set. No ELSA method, similarity pass or neighbour list has been run on it. Embeddings (ESM++
small, SR-VP/DPANN recipe) are added to embeddings/ by the queued GPU job; they are outside the seal.

Selection (selection.tsv; rules in ../selection_rules.txt, v2, seed 17, ladder step k = 8): layers a1 complete lutA-I
(80), a2 receptor + transporter or receptor module (96), a3 split arrangement (141, incl. RH AL1, M. aquaticum 22A,
M. trichosporium OB3b), a4 transporter module only (341), b1 same-genus lut-negative (166), b2 same-family lut-negative
(134), c1 class background (56), c2 outside-class module carriers (18, incl. the P. putida KT2440 species rep and
M. buryatense). GTDB taxonomy was used to build the set only.

Layout (DPANN package conventions)
  universe/genes.parquet      4,507,042 rows: gid, protein_id, genome, genome_code, contig, contig_code, position_index,
                              strand, length, start, end (protein_id = '<genome>|<Prodigal ID>')
  universe/genomes.parquet    1,032 rows: DPANN columns + gtdb_id, layer, lut_status, n_main_loci, locus_components,
                              source_scope, quality_tier, split
  universe/contigs.parquet    161,964 rows (length_nt NULL; contig_extra.max_end_nt is a lower bound)
  universe/contig_extra.parquet, universe/gene_flags.parquet (Prodigal flags, source file index)
  proteins_gid.faa            4,507,042 records, header = gid; 4,464,102 unique sequences
  split.parquet               genome_code, genome, gtdb_id, split: odd genome_code = tuning (516), even = test (516)
  gold/lut_instances.parquet  part-1 lut loci of these genomes (caller lut_homology_main 733, lut_homology_strict 121):
                              system_type = architecture class, system_subtype = components present, start/end
                              position, member_gids, member_positions, member_components (A-G lutA-lutG, H lutH,
                              M lanM, I lutI), architecture string, contig-edge flags, source locus id
  gold/lut_members.parquet    instance_id, caller, gid, component (4,585 rows)
  gold/ribosomal_marked.parquet, gold/ribosomal_runs.parquet   standing definition (Pfam 38.1 ^ribosom(al)?_ models at
                              GA; runs with <= 2 unmarked genes (g2) or none (g0); >= 8 marked kept, m16 primary):
                              g2_m16 858 runs in 855 genomes
  receipts/build.json, receipts/ribosomal.json, seal.json, MANIFEST.tsv
Genes, coordinates and protein_ids are identical to the part-1 universes (GTDB r232 Prodigal proteomes).
Naming: the lut instances are homology + co-location calls, not a curated caller; functions are UNVERIFIED.
Code: slice_lut/code/13_select.py, 14_build_testset.py, 15_ribosomal_truth.py, 16_seal.py, testset.sbatch, embed/.
