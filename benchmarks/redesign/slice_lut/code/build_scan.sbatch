#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand
#SBATCH -c 24
#SBATCH -J lut_build
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/slice_lut/logs/build_scan_%j.log
# Workstream slice_lut, job 1: parse the GTDB r232 o__Rhizobiales species-representative proteomes, build the ELSA
# universes (family Beijerinckiaceae and order Rhizobiales), search the AM1 lut proteins against the order (MMseqs2,
# forward + reverse best hit against the AM1 proteome), pull GlobDB r232 annotation rows, and check the GlobDB gene
# index against an independent Pfam search of the family proteins. All parallel stages use $SLURM_CPUS_ON_NODE.
set -euo pipefail
RD=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111
S=$RD/slice_lut; C=$S/code; W=$S/work; P=$S/package; X=$S/scan
PY=$RD/ws_tokens/env/bin/python
MM=/groups/banfield/users/jwestrob/bin/mmseqs_18/mmseqs/bin/mmseqs
G=/groups/banfield/projects/environmental/sr/srvp2020/gtdb/rp16_run_r232
GP=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/Chloroflexi_2026/provenance/globdb
REPS=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/giant_protein_folding/rp16_phylogeny_2026_06/gtdb_giant_scan/reps/protein_faa_reps/bacteria
PFAM=/home/jwestrob/.config/Astra/PFAM/Pfam-A.hmm
T="${SLURM_CPUS_ON_NODE:?}"
export OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
stamp() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*"; }
mkdir -p $W $P $X $W/mm
stamp "host=$(hostname) threads=$T mmseqs=$($MM version)"

if [[ ! -f $W/rhizobiales/genomes_src.tsv ]]; then
  stamp "START parse"
  $PY -u $C/01_parse_slice.py $G/sp_clusters.tsv $REPS order Rhizobiales $W/rhizobiales $T
fi
for sc in "beijerinckiaceae Beijerinckiaceae" "rhizobiales ALL"; do
  set -- $sc
  if [[ ! -f $P/$1/receipts/universe.json ]]; then
    stamp "START universe $1"
    $PY -u $C/02_universe.py $W/rhizobiales/parts $1 $2 $P/$1 $GP/globdb_r232_checkm2.tsv.gz $GP/globdb_r232_taxonomy.tsv.gz $T
    rm -rf $P/$1/.duck_tmp
  fi
done

stamp "START lut search (forward)"
for db in order_seqdb refs am1 fwd self rev hitdb; do $MM rmdb $W/mm/$db -v 1 2>/dev/null || true; done
$MM createdb $P/rhizobiales/proteins_gid.faa $W/mm/order_seqdb -v 2
$MM createdb $S/ref/lut_refs_1777_1788.faa $W/mm/refs -v 2
$MM createdb $S/ref/AM1_proteome.faa $W/mm/am1 -v 2
FMT=query,target,evalue,bits,fident,alnlen,qstart,qend,qlen,tstart,tend,tlen
$MM search $W/mm/refs $W/mm/order_seqdb $W/mm/fwd $W/mm/tmp -s 7.5 -e 1e-3 --max-seqs 2000000 -a \
    --split-memory-limit 60G --threads $T -v 2
$MM convertalis $W/mm/refs $W/mm/order_seqdb $W/mm/fwd $X/order_fwd.tsv --format-output $FMT --threads $T -v 2
$MM search $W/mm/refs $W/mm/refs $W/mm/self $W/mm/tmp -s 7.5 -e 10 -a --threads $T -v 2
$MM convertalis $W/mm/refs $W/mm/refs $W/mm/self $X/refs_self.tsv --format-output $FMT --threads $T -v 2
stamp "forward rows $(wc -l < $X/order_fwd.tsv)"
stamp "START reverse search against the AM1 proteome"
cut -f2 $X/order_fwd.tsv | sort -u > $W/mm/hit_gids.txt
$MM createsubdb --id-mode 1 $W/mm/hit_gids.txt $W/mm/order_seqdb $W/mm/hitdb -v 2
$MM search $W/mm/hitdb $W/mm/am1 $W/mm/rev $W/mm/tmp -s 7.5 -e 1e-3 --max-seqs 300 -a --threads $T -v 2
$MM convertalis $W/mm/hitdb $W/mm/am1 $W/mm/rev $X/order_rev.tsv --format-output $FMT --threads $T -v 2
stamp "reverse rows $(wc -l < $X/order_rev.tsv), hit genes $(wc -l < $W/mm/hit_gids.txt)"
rm -rf $W/mm/tmp

stamp "START GlobDB rows"
ulimit -n 8192
$PY -u $C/04_globdb_rows.py /groups/banfield/users/jwestrob/databases/globdb_r232/protein_annotations_parquet \
    $GP/globdb_r232_taxonomy.tsv.gz $S/globdb $T
rm -rf $S/globdb/duck_tmp

stamp "START Pfam check on the family proteins (independent of GlobDB)"
PV=$W/pfam_check; rm -rf $PV; mkdir -p $PV/parts $PV/out
printf '%s\n' PF11684.14 PF13458.13 PF10282.16 PF22494.4 PF00005.34 PF01061.30 PF12698.14 PF12679.14 PF07715.21 \
    PF00593.31 PF13202.13 PF13727.13 PF02397.23 > $PV/keys.txt
hmmfetch -f $PFAM $PV/keys.txt > $PV/sel.hmm
awk -v n=$T -v d=$PV/parts 'BEGIN{k=-1} /^>/{k=(k+1)%n} {print > (d "/p_" k ".faa")}' $P/beijerinckiaceae/proteins_gid.faa
ls $PV/parts/*.faa | xargs -P $T -I{} sh -c 'b=$(basename {} .faa); hmmsearch --cut_ga --cpu 1 --noali -o /dev/null --domtblout '$PV'/out/$b.domtbl '$PV'/sel.hmm {}'
cat $PV/out/*.domtbl | grep -v '^#' | awk '{print $1"\t"$4"\t"$5"\t"$7"\t"$8"\t"$14"\t"$18"\t"$19}' > $X/family_pfam_check.tsv
rm -rf $PV/parts
stamp "Pfam check rows $(wc -l < $X/family_pfam_check.tsv)"
stamp "DONE"
