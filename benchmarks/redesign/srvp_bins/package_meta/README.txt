SR-VP genome-bin community test package (workstream srvp_bins)
================================================================
Location: package
Every file with its row count, size and SHA-256 is in MANIFEST.tsv; seal.json / SEAL.txt hold the combined seal.
No ELSA method has been run on this universe, nothing has been embedded and nothing has been scored.

SOURCE
Bins: the reconciled SR-VP soil genome register (Luis Valentin-Alvarado's srvp-fina, 2026-10-21):
  03_assemblies_and_genomes/joint_dereplication_v1/input_manifest.tsv (6,647 bins with a CheckM2 estimate linked by
  identifier and measured length; completeness >= 50, contamination < 10, >= 50 kb) and the joint dRep 3.5.0 run
  (fastANI, 95% ANI, 50% aligned fraction, average linkage; Slurm job 17839) -> 2,392 clusters.
Selection (selection.tsv): CheckM2 completeness >= 90 and contamination <= 5 (primary tier), one bin per 95% ANI
  cluster: the dRep representative when it passes the tier, else the passing member with the best
  completeness - 5*contamination + 0.5*log10(N50).
Taxonomy: GTDB-Tk 2.7.2 / r232 (job 18127) for representatives; tax_source names the source for every genome.
Gene calls: GTDB-Tk's identify-step Prodigal 2.6.3 calls (single mode, table 11 or 4 by coding density) for
  representatives; the same procedure (prodigal -m -p single, tables 11 and 4) for the other picks. Control: the
  procedure rerun on 6 representatives reproduced GTDB-Tk's coordinates and sequences exactly (receipt genes.json).

CONTENTS
  universe/genes.parquet         3,211,751  gid, protein_id, genome, genome_code, contig, contig_code, position_index, strand, length, start, end
  universe/gene_flags.parquet    3,211,751  gid, partial (Prodigal), start_type, truncated, at_contig_end_nt, first_on_contig, last_on_contig, contig_circular (null: not audited)
  universe/contigs.parquet         146,107  contigs carrying genes: genome, contig, contig_code, n_genes, length_nt, first_gid, file_order
  universe/contigs_all.parquet     146,154  every contig of every bin in FASTA order (file_order), length_nt, n_genes
  universe/genomes.parquet             994  completeness, contamination, taxonomy, platform, provisional sample, dereplication cluster, gene source
  universe/proteins.faa          3,211,751  header = gid
  split/genome_halves.parquet          994  test = even genome_code, tune = odd
  ribosomal/marked.parquet          53,345  genes with a Pfam 38.1 ^ribosom(al)?_ hit at GA (121 models)
  ribosomal/runs.parquet             2,111  runs g0/g2 with >= 8 marked genes; m16 flag
  selection.tsv, code/, receipts/

NUMBERS
  genomes 994; domains {'Bacteria': 780, 'Archaea': 214}; phyla 52; platform {'Illumina': 563, 'PacBio': 431}
  single-contig genomes 342; contigs 146,154 (47 without a gene call)
  genes 3,211,751; residues 988,758,684; unique protein sequences 3,170,150
  partial flags {'00': 2998350, '10': 105435, '01': 105019, '11': 2947}
  completeness median 97.01, contamination median 1.32
  ribosomal runs: {'g0_m8': 1060, 'g0_m16': 606, 'g2_m8': 1051, 'g2_m16': 803}; g2_m16 runs in 803 of 994 genomes; 113 g2_m16 runs touch a contig edge
  split: {'test': 497, 'tune': 497} genomes, {'test': 1600439, 'tune': 1611312} genes
  callers: not in this version of the package (gold/ absent); see REPORT.txt for the queued job and estimate.
