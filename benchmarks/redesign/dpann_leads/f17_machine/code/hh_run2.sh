#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand
#SBATCH -c 24
#SBATCH -J f17_hh2
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/dpann_leads/f17_machine/logs/hh_%j.log
# f17_machine: profile-profile search of family-17 role alignments (HH-suite 3.1, local databases as used by Sharur's
# family annotation): family MSA -> hhblits UniRef30 (2 iterations) -> hhsearch vs Pfam 39 and PDB100; also the
# family-only MSA vs the same databases. hh_run2: hhblits filter options from HHOPT (empty = HH-suite defaults, used after the first run with -maxfilt/-realign_max 100000 stalled on the multi-pass membrane groups). GROUPS env var lists the role groups (or reference names) to run.
set -euo pipefail
HHOPT=${HHOPT:-}
W=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/dpann_leads/f17_machine
U=/shared/db/uniclust/30_2020_06/UniRef30_2020_06
PF=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/sharur_db/pfam39_hh/pfam39
PDB=/shared/db/rosettafold/rosettafold2/pdb100_2021Mar03/pdb100_2021Mar03
B=/shared/software/bin
cd $W/hh
for g in $GROUPS; do
  if [ -f $W/roles/$g.aln ]; then IN=$W/roles/$g.aln; MOPT="-M 50"; else IN=$W/refs/$g.faa; MOPT=""; fi
  $B/hhblits -i $IN $MOPT -d $U -n 2 -e 0.001 $HHOPT -oa3m $g.uni.a3m -o /dev/null -cpu $SLURM_CPUS_ON_NODE -v 0
  $B/hhsearch -i $g.uni.a3m -d $PF -d $PDB -o $g.uni.hhr -cpu $SLURM_CPUS_ON_NODE -v 0 -z 1 -b 1 -Z 100 -B 100
  if [ -f $W/roles/$g.aln ]; then
    $B/hhsearch -i $IN -M 50 -d $PF -d $PDB -o $g.fam.hhr -cpu $SLURM_CPUS_ON_NODE -v 0 -Z 100 -B 100
  fi
  echo "$g done $(date -u +%FT%TZ)"
done
echo "DONE $(date -u +%FT%TZ) job $SLURM_JOB_ID" > STATUS_$SLURM_JOB_ID
