ELSA slice package: GTDB r232 order Rhizobiales (workstream slice_lut, built 2026-11-13 on biotite)
Location: /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/slice_lut/package/rhizobiales/
Status: untouched test set. No ELSA neighbourhood, word or embedding method has been run on it.

Source
  GTDB r232 species representatives in o__Rhizobiales (c__Alphaproteobacteria), 5,752 genomes in 61 families
  (Rhizobiaceae 1,546, Xanthobacteraceae 1,487, Beijerinckiaceae 996, Hyphomicrobiaceae 409, Devosiaceae 243, ...),
  from sp_clusters.tsv (rp16_run_r232). NCBI name of the order: Hyphomicrobiales.
  Proteins: GTDB r232 Prodigal proteome of each representative, trailing '*' removed; coordinates, strand and partial
  flags from the Prodigal headers. Taxonomy GTDB r232 (tax); tax_db = GlobDB r232 string (identical for all 5,752).
  Completeness / contamination: GlobDB r232 CheckM2 (found for all 5,752).

Layout (same column set as ws_dpann/package/universe; see the family package README for column definitions)
  universe/genes.parquet     27,072,085 rows (protein_id = '<genome>|<Prodigal ID>')
  universe/genomes.parquet   5,752 rows (strand-evidence columns NULL; extra column gtdb_id)
  universe/contigs.parquet   913,090 rows (length_nt NULL; contig_extra.max_end_nt is a lower bound)
  universe/contig_extra.parquet, universe/gene_flags.parquet
  proteins_gid.faa           27,072,085 records, header = gid, 7,971,672,333 residues
  tokens/mmseqs__seq__e3c50s75__rhizobiales.parquet and tokens/mmseqs__seq__e3c50s75_rescue3__rhizobiales.parquet:
                             DPANN recipe (base mmseqs cluster -s 7.5 -e 1e-3 --min-seq-id 0 -c 0.5 --cov-mode 0
                             --cluster-mode 0; rescue3 as in the family package), job 92432; check the .json next to
                             each parquet for counts once the job has finished
  receipts/universe.json, README.txt, MANIFEST.tsv (written after the tokens)

QC: duplicate (contig, position) 0; non-contiguous contigs 0; internal stops 0; 1,300,444 partial genes; 45,774
single-gene contigs; 14 genes whose coordinate order differs from the Prodigal numbering.
The family package (package/beijerinckiaceae) is the Beijerinckiaceae subset with its own gid numbering and its own tokens.
GlobDB r232 annotation rows cannot be joined to these genes by gene index (slice_lut REPORT.txt, ANSWER 7).
Code: slice_lut/code/01_parse_slice.py, 02_universe.py, tokens.sbatch, 06a_unique.py, 06b_rescue3.py, mmseqs_tokens.py.
Jobs: 91291 (parse + universe), 92432 (tokens).
