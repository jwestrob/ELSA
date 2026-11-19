#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand
#SBATCH -c 24
#SBATCH -J llf_zarm
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/long_loci/frozen/logs/zarm_%j.log
# long_loci frozen run, J5b: SECONDARY Z arm before the tokens exist (fzrun.py arm: fz.cmd_arm unchanged, token-cohesion
# check tree deferred). Outputs copied to loci/lut_Z_provisional for the identity check against the later unchanged run.
set -euo pipefail
source /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/long_loci/frozen/code/env_f.sh
stamp "host=$(hostname) threads=$T job=$SLURM_JOB_ID"
cd "$C"
test -f "$FZ/logs/done_z_prep"
step z_arm_prov "$PYE" -u fzrun.py arm --threads "$T"
mkdir -p "$FZ/loci/lut_Z_provisional"
cp -p "$FZ"/loci/lut_Z/*.parquet "$FZ/loci/lut_Z_provisional/"
cp -p "$FZ/results/zbuild_lut_Z.json" "$FZ/results/zbuild_lut_Z_provisional.json"
stamp "ALL DONE"
