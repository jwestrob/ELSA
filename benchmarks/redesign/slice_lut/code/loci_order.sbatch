#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand
#SBATCH -c 24
#SBATCH -J lut_loci
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/slice_lut/logs/loci_order_%j.log
# Workstream slice_lut: tail of job 1 (91291 failed at hmmfetch with several keys) + order loci.
# Was job 1: parse the GTDB r232 o__Rhizobiales species-representative proteomes, build the ELSA
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
stamp "START loci (order)"
$PY -u $C/05_loci.py $C/loci_order.json
stamp "DONE"
