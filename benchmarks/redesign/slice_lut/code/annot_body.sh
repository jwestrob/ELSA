#!/bin/bash
# Body of annot.sbatch (called by wide2.sbatch inside its allocation). Env: SC, FAA, CFG, SLURM_CPUS_ON_NODE.
set -euo pipefail
RD=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111
S=$RD/slice_lut; C=$S/code; W=$S/work/annot_${SC:?}; R=$S/results/$SC
PY=$RD/ws_tokens/env/bin/python
A=/home/jwestrob/.config/Astra
T="${SLURM_CPUS_ON_NODE:?}"
export OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
stamp() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*"; }
split_faa() { rm -rf $2; mkdir -p $2; awk -v n=$T -v d=$2 'BEGIN{k=-1} /^>/{k=(k+1)%n} {print > (d "/p_" k ".faa")}' $1; }
mkdir -p $W
stamp "host=$(hostname) threads=$T scope=$SC"
$PY -u $C/10_annot.py export $R ${FAA:?} $W/loci_genes.faa main
split_faa $W/loci_genes.faa $W/parts
mkdir -p $W/pfam $W/kofam
stamp "START Pfam hmmscan (GA)"
ls $W/parts/*.faa | xargs -P $T -I{} sh -c 'b=$(basename {} .faa); hmmscan --cut_ga --cpu 1 --noali -o /dev/null --domtblout '$W'/pfam/$b.domtbl '$A'/PFAM/PFAM {}'
stamp "START KOfam hmmscan"
ls $W/parts/*.faa | xargs -P $T -I{} sh -c 'b=$(basename {} .faa); hmmscan -E 1e-3 --domE 1e-3 --cpu 1 --noali -o /dev/null --domtblout '$W'/kofam/$b.domtbl '$A'/KOFAM/KOFAM {}'
$PY -u $C/10_annot.py parse "$W/pfam/*.domtbl" "$W/kofam/*.domtbl" $A/ko_list $R/own_annot.parquet
stamp "START MDH KOs over all slice proteins"
cat $A/KOFAM/K23995.hmm $A/KOFAM/K14028.hmm $A/KOFAM/K14029.hmm $A/KOFAM/K00114.hmm > $W/mdh.hmm
split_faa $FAA $W/allparts
mkdir -p $W/mdh
ls $W/allparts/*.faa | xargs -P $T -I{} sh -c 'b=$(basename {} .faa); hmmsearch -E 1e-5 --cpu 1 --noali -o /dev/null --domtblout '$W'/mdh/$b.domtbl '$W'/mdh.hmm {}'
rm -rf $W/allparts
$PY -u $C/10_annot.py mdh "$W/mdh/*.domtbl" $A/ko_list $R/mdh_hits.parquet
stamp "START loci with direct annotation"
$PY -u $C/05_loci.py ${CFG:?}
stamp "DONE"
