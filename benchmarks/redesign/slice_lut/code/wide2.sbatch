#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand
#SBATCH -c 24
#SBATCH -J lut_wide2
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/slice_lut/logs/wide2_%j.log
# Workstream slice_lut, beyond the class: GTDB r232 bacterial representatives outside c__Alphaproteobacteria listed in
# work/beyond/list.txt (every rep with a GlobDB ABEF proxy run at gap 5, 500 proxy-negative reps drawn with seed 17, and
# three literature controls). Parse, gene table + FASTA, MMseqs2 forward/reverse as for the order, GlobDB rows of the
# listed genomes, loci; then direct Pfam/KOfam annotation + MDH KOs for the alpha and beyond scopes (annot.sbatch body).
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
B=$W/beyond; M=$W/mm_beyond; mkdir -p $M
stamp "host=$(hostname) threads=$T"
if [[ ! -f $B/genomes_src.tsv ]]; then $PY -u $C/01_parse_slice.py $G/sp_clusters.tsv $REPS list $B/list.txt $B $T; fi
if [[ ! -f $B/pkg/receipts/universe.json ]]; then
  $PY -u $C/02_universe.py $B/parts beyond ALL $B/pkg $GP/globdb_r232_checkm2.tsv.gz $GP/globdb_r232_taxonomy.tsv.gz $T; rm -rf $B/pkg/.duck_tmp
fi
if [[ ! -f $X/beyond_rev.tsv ]]; then
  for db in seqdb refs am1 fwd rev hitdb; do $MM rmdb $M/$db -v 1 2>/dev/null || true; done
  FMT=query,target,evalue,bits,fident,alnlen,qstart,qend,qlen,tstart,tend,tlen
  $MM createdb $B/pkg/proteins_gid.faa $M/seqdb -v 2
  $MM createdb $S/ref/lut_refs_1777_1788.faa $M/refs -v 2
  $MM createdb $S/ref/AM1_proteome.faa $M/am1 -v 2
  $MM search $M/refs $M/seqdb $M/fwd $M/tmp -s 7.5 -e 1e-3 --max-seqs 2000000 -a --split-memory-limit 60G --threads $T -v 2
  $MM convertalis $M/refs $M/seqdb $M/fwd $X/beyond_fwd.tsv --format-output $FMT --threads $T -v 2
  cut -f2 $X/beyond_fwd.tsv | sort -u > $M/hit_gids.txt
  $MM createsubdb --id-mode 1 $M/hit_gids.txt $M/seqdb $M/hitdb -v 2
  stamp "START reverse ($(wc -l < $M/hit_gids.txt) genes)"
  $MM search $M/hitdb $M/am1 $M/rev $M/tmp -s 7.5 -e 1e-3 --max-seqs 300 -a --threads $T -v 2
  $MM convertalis $M/hitdb $M/am1 $M/rev $X/beyond_rev.tsv.tmp --format-output $FMT --threads $T -v 2
  mv $X/beyond_rev.tsv.tmp $X/beyond_rev.tsv; rm -rf $M/tmp $M/seqdb*
fi
if [[ ! -f $S/globdb/beyond_rows.parquet ]]; then
  ulimit -n 8192
  sed 's/^RS_//; s/^GB_//; s/\..*//' $B/list.txt > $B/globdb_ids.txt
  $PY -u $C/04b_globdb_rows_subset.py /groups/banfield/users/jwestrob/databases/globdb_r232/protein_annotations_parquet \
      $GP/globdb_r232_taxonomy.tsv.gz list:$B/globdb_ids.txt $S/globdb/beyond_rows.parquet $T
  rm -rf $S/globdb/duck_tmp_beyond_rows.parquet
fi
stamp "START loci (beyond, before annotation)"
$PY -u $C/05_loci.py $C/loci_beyond.json
for sc in "alpha_wide $W/alpha_wide/pkg/proteins_gid.faa $C/loci_alpha.json" "beyond $B/pkg/proteins_gid.faa $C/loci_beyond.json"; do
  set -- $sc
  stamp "START annotation $1"
  SC=$1 FAA=$2 CFG=$3 bash $C/annot_body.sh
done
stamp "DONE"
