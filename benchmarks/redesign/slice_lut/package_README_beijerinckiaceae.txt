ELSA slice package: GTDB r232 family Beijerinckiaceae (workstream slice_lut, built 2026-11-13 on biotite)
Location: /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/slice_lut/package/beijerinckiaceae/
Status: untouched test set. No ELSA neighbourhood, word or embedding method has been run on it.

Source
  GTDB r232 species representatives in f__Beijerinckiaceae (o__Rhizobiales, c__Alphaproteobacteria), 996 genomes,
  from sp_clusters.tsv (rp16_run_r232). In GTDB r232 Methylorubrum is part of g__Methylobacterium (146 species reps).
  Proteins: the GTDB r232 Prodigal proteome of each representative (protein_faa_reps/bacteria/<ID>_protein.faa.gz),
  trailing '*' removed. Coordinates, strand and partial flags from the Prodigal headers.
  Taxonomy: GTDB r232 (tax); tax_db = GlobDB r232 taxonomy string (identical for all 996).
  Completeness / contamination: GlobDB r232 CheckM2 table (found for all 996).

Layout (same column set as ws_dpann/package/universe)
  universe/genes.parquet     4,517,879 rows  gid, protein_id, genome, genome_code, contig, contig_code, position_index,
                                             strand (+1/-1), length (aa), start, end (1-based nt)
                             protein_id = '<genome>|<Prodigal ID>' (globally unique)
  universe/genomes.parquet   996 rows        DPANN columns; strand-evidence columns are NULL (the gene calls are the
                                             Prodigal calls themselves); extra column gtdb_id (RS_/GB_ prefixed)
  universe/contigs.parquet   175,407 rows    length_nt is NULL (GTDB distributes proteomes only)
  universe/contig_extra.parquet               contig_code, max_end_nt (lower bound of contig length), first/last gene partial flag
  universe/gene_flags.parquet                 gid, gtdb_id, prodigal_id, prodigal_n, file_index, partial, start_type,
                                             rbs_motif, gc_cont, internal_stops
  proteins_gid.faa           4,517,879 records, header = gid, 1,320,355,524 residues
  tokens/mmseqs__seq__e3c50s75__beijerinckiaceae.parquet          base families: mmseqs cluster -s 7.5 -e 1e-3
                             --min-seq-id 0 -c 0.5 --cov-mode 0 --cluster-mode 0 (MMseqs2 release 18 8cc5ce36);
                             308,158 tokens, 206,053 singletons, largest 21,352
  tokens/mmseqs__seq__e3c50s75_rescue3__beijerinckiaceae.parquet  rescue3 (DPANN rule, repair/build_arms.py rescue part,
                             resolve_moves from repair/repair_lib.py): genes in base tokens of < 3 genes move to the token
                             of their best hit (unique-sequence prefilter -s 7.5 --max-seqs 1000; align -e 1e-5 -a;
                             >= 80% coverage of the shorter protein; target token >= 2 genes); 82,293 genes changed,
                             241,760 tokens, 155,409 singletons, largest 21,523
  receipts/universe.json     QC counts and SHA-256 of every universe file
  MANIFEST.tsv               path, rows, bytes, SHA-256 of every file

gid = rank over (genome_code, contig_code, start, end, protein_id); each contig is a contiguous gid range starting at
contigs.first_gid. QC: duplicate (contig, position) 0; non-contiguous contigs 0; internal stops 0; 252,943 partial genes;
8,717 single-gene contigs; 2 genes whose coordinate order differs from the Prodigal numbering.

GlobDB r232 annotation rows cannot be joined to these genes by gene index: GlobDB numbers its own gene calls, which
differ from the GTDB protein files (agreement 3.9% at the best offset; slice_lut REPORT.txt). Join by coordinates
(GlobDB gff_cog release) or annotate directly.
Code: slice_lut/code/01_parse_slice.py, 02_universe.py, tokens.sbatch, 06a_unique.py, 06b_rescue3.py, mmseqs_tokens.py.
Jobs: 91291 (parse + universe), 91327 (tokens).
