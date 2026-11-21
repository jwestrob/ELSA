#!/usr/bin/env python3
"""Write PKG/README.txt from the stage receipts (workstream srvp_bins). usage: 05_readme.py PKG_DIR"""
import json, sys
from pathlib import Path
import pandas as pd

P = Path(sys.argv[1])
r = {p.stem: json.loads(p.read_text()) for p in (P / "receipts").glob("*.json")}
g = pd.read_parquet(P / "universe" / "genomes.parquet")
u, fa, rb, sp, gn = r["universe"], r["fasta"], r["ribosomal"], r["split"], r["genes"]
cl = r.get("callers")
L = []
w = L.append
w("SR-VP genome-bin community test package (workstream srvp_bins)")
w("=" * 64)
w(f"Location: {P}")
w("Every file with its row count, size and SHA-256 is in MANIFEST.tsv; seal.json / SEAL.txt hold the combined seal.")
w("No ELSA method has been run on this universe, nothing has been embedded and nothing has been scored.")
w("")
w("SOURCE")
w("Bins: the reconciled SR-VP soil genome register (Luis Valentin-Alvarado's srvp-fina, 2026-10-21):")
w("  03_assemblies_and_genomes/joint_dereplication_v1/input_manifest.tsv (6,647 bins with a CheckM2 estimate linked by")
w("  identifier and measured length; completeness >= 50, contamination < 10, >= 50 kb) and the joint dRep 3.5.0 run")
w("  (fastANI, 95% ANI, 50% aligned fraction, average linkage; Slurm job 17839) -> 2,392 clusters.")
w("Selection (selection.tsv): CheckM2 completeness >= 90 and contamination <= 5 (primary tier), one bin per 95% ANI")
w("  cluster: the dRep representative when it passes the tier, else the passing member with the best")
w("  completeness - 5*contamination + 0.5*log10(N50).")
w("Taxonomy: GTDB-Tk 2.7.2 / r232 (job 18127) for representatives; tax_source names the source for every genome.")
w("Gene calls: GTDB-Tk's identify-step Prodigal 2.6.3 calls (single mode, table 11 or 4 by coding density) for")
w("  representatives; the same procedure (prodigal -m -p single, tables 11 and 4) for the other picks. Control: the")
w("  procedure rerun on 6 representatives reproduced GTDB-Tk's coordinates and sequences exactly (receipt genes.json).")
w("")
w("CONTENTS")
w(f"  universe/genes.parquet        {u['genes']:>10,}  gid, protein_id, genome, genome_code, contig, contig_code, position_index, strand, length, start, end")
w(f"  universe/gene_flags.parquet   {u['genes']:>10,}  gid, partial (Prodigal), start_type, truncated, at_contig_end_nt, first_on_contig, last_on_contig, contig_circular (null: not audited)")
w(f"  universe/contigs.parquet      {u['contigs_with_genes']:>10,}  contigs carrying genes: genome, contig, contig_code, n_genes, length_nt, first_gid, file_order")
w(f"  universe/contigs_all.parquet  {u['contigs_all']:>10,}  every contig of every bin in FASTA order (file_order), length_nt, n_genes")
w(f"  universe/genomes.parquet      {u['genomes']:>10,}  completeness, contamination, taxonomy, platform, provisional sample, dereplication cluster, gene source")
w(f"  universe/proteins.faa         {fa['records']:>10,}  header = gid")
w(f"  split/genome_halves.parquet   {u['genomes']:>10,}  test = even genome_code, tune = odd")
w(f"  ribosomal/marked.parquet      {rb['marked_genes']:>10,}  genes with a Pfam 38.1 ^ribosom(al)?_ hit at GA (121 models)")
w(f"  ribosomal/runs.parquet        {sum(v for k, v in rb['runs'].items() if k.endswith('m8')):>10,}  runs g0/g2 with >= 8 marked genes; m16 flag")
if cl:
    w(f"  gold/instances.parquet        {sum(cl['instances_by_caller'].values()):>10,}  DefenseFinder and TXSScan systems (MacSyFinder 2.1.4)")
    w("  gold/members.parquet                      instance_id, caller, gid")
w("  selection.tsv, code/, receipts/")
w("")
w("NUMBERS")
w(f"  genomes {u['genomes']:,}; domains {u['domains']}; phyla {u['phyla']}; platform {u['platform']}")
w(f"  single-contig genomes {u['single_contig_genomes']}; contigs {u['contigs_all']:,} ({u['contigs_without_genes']:,} without a gene call)")
w(f"  genes {u['genes']:,}; residues {fa['residues']:,}; unique protein sequences {fa['unique_sequences']:,}")
w(f"  partial flags {dict(u['partial_values'])}")
w(f"  completeness median {g.completeness.median():.2f}, contamination median {g.contamination.median():.2f}")
w(f"  ribosomal runs: {rb['runs']}; g2_m16 runs in {rb['g2_m16_genomes']} of {rb['genomes']} genomes; "
  f"{rb['g2_m16_at_contig_edge']} g2_m16 runs touch a contig edge")
w(f"  split: {sp['genomes']} genomes, {sp['genes']} genes")
if cl:
    w(f"  callers: {cl['instances_by_caller']} instances; genomes with an instance {cl['genomes_with_instance']}")
else:
    w("  callers: not in this version of the package (gold/ absent); see REPORT.txt for the queued job and estimate.")
(P / "README.txt").write_text("\n".join(L) + "\n")
print("\n".join(L))
