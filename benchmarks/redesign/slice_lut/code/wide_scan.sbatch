#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand
#SBATCH -c 24
#SBATCH -J lut_wide
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/slice_lut/logs/wide_scan_%j.log
# Workstream slice_lut, job 3: lut scan beyond the order. (a) GTDB r232 c__Alphaproteobacteria representatives outside
# o__Rhizobiales: parse, gene table + FASTA (02_universe layout, kept under work/), MMseqs2 forward search of the AM1
# references + reverse best hit against the AM1 proteome, GlobDB Pfam/KOfam rows of the class, loci (05_loci.py).
# (b) GlobDB Pfam proxy over all 346,233 GlobDB genomes (07_globdb_proxy.py). Parallel stages use $SLURM_CPUS_ON_NODE.
set -euo pipefail
RD=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111
S=$RD/slice_lut; C=$S/code; W=$S/work; X=$S/scan
PY=$RD/ws_tokens/env/bin/python
MM=/groups/banfield/users/jwestrob/bin/mmseqs_18/mmseqs/bin/mmseqs
G=/groups/banfield/projects/environmental/sr/srvp2020/gtdb/rp16_run_r232
GP=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/Chloroflexi_2026/provenance/globdb
REPS=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/giant_protein_folding/rp16_phylogeny_2026_06/gtdb_giant_scan/reps/protein_faa_reps/bacteria
T="${SLURM_CPUS_ON_NODE:?}"
export OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
stamp() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*"; }
A=$W/alpha_wide; M=$W/mm_alpha
mkdir -p $A $M $S/globdb
stamp "host=$(hostname) threads=$T"
if [[ ! -f $A/genomes_src.tsv ]]; then
  $PY -u $C/01_parse_slice.py $G/sp_clusters.tsv $REPS class Alphaproteobacteria $A $T order Rhizobiales
fi
if [[ ! -f $A/pkg/receipts/universe.json ]]; then
  $PY -u $C/02_universe.py $A/parts alpha_wide ALL $A/pkg $GP/globdb_r232_checkm2.tsv.gz $GP/globdb_r232_taxonomy.tsv.gz $T
  rm -rf $A/pkg/.duck_tmp
fi
if [[ ! -f $X/alpha_rev.tsv ]]; then
  for db in seqdb refs am1 fwd rev hitdb; do $MM rmdb $M/$db -v 1 2>/dev/null || true; done
  FMT=query,target,evalue,bits,fident,alnlen,qstart,qend,qlen,tstart,tend,tlen
  $MM createdb $A/pkg/proteins_gid.faa $M/seqdb -v 2
  $MM createdb $S/ref/lut_refs_1777_1788.faa $M/refs -v 2
  $MM createdb $S/ref/AM1_proteome.faa $M/am1 -v 2
  stamp "START forward search"
  $MM search $M/refs $M/seqdb $M/fwd $M/tmp -s 7.5 -e 1e-3 --max-seqs 2000000 -a --split-memory-limit 60G --threads $T -v 2
  $MM convertalis $M/refs $M/seqdb $M/fwd $X/alpha_fwd.tsv --format-output $FMT --threads $T -v 2
  cut -f2 $X/alpha_fwd.tsv | sort -u > $M/hit_gids.txt
  $MM createsubdb --id-mode 1 $M/hit_gids.txt $M/seqdb $M/hitdb -v 2
  stamp "START reverse search ($(wc -l < $M/hit_gids.txt) genes)"
  $MM search $M/hitdb $M/am1 $M/rev $M/tmp -s 7.5 -e 1e-3 --max-seqs 300 -a --threads $T -v 2
  $MM convertalis $M/hitdb $M/am1 $M/rev $X/alpha_rev.tsv.tmp --format-output $FMT --threads $T -v 2
  mv $X/alpha_rev.tsv.tmp $X/alpha_rev.tsv
  rm -rf $M/tmp $M/seqdb*
fi
if [[ ! -f $S/globdb/alpha_rows.parquet ]]; then
  stamp "START GlobDB rows of the class"
  ulimit -n 8192
  $PY -u $C/04b_globdb_rows_subset.py /groups/banfield/users/jwestrob/databases/globdb_r232/protein_annotations_parquet \
      $GP/globdb_r232_taxonomy.tsv.gz rank:3:c__Alphaproteobacteria $S/globdb/alpha_rows.parquet $T
  rm -rf $S/globdb/duck_tmp_alpha_rows.parquet
fi
stamp "START proxy"
$PY -u $C/07_globdb_proxy.py $S/globdb/wide_rows.parquet $GP/globdb_r232_taxonomy.tsv.gz $G/sp_clusters.tsv $S/results/proxy $T
stamp "START loci (alpha outside Rhizobiales)"
$PY -u $C/05_loci.py $S/code/loci_alpha.json
stamp "DONE"
