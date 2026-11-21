#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand
#SBATCH -c 24
#SBATCH -J annot_val_cpu
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/annot_gpu/validation/val_cpu_%j.log

# Validation gate, CPU arm: the same Aksha 0.2.3 commands as val_gpu.sbatch without --gpu-manifest.
set -euo pipefail
for v in $(compgen -e | grep -E '^(ASTRA_|_ASTRA_|PLAN7_)' || true); do unset "$v"; done
export PYTHONDONTWRITEBYTECODE=1
AKSHA=/home/jwestrob/.local/bin/aksha
A=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/annot_gpu
V=$A/validation
echo "host $(hostname)  start $(date -u +%FT%TZ)  CPUs $SLURM_CPUS_ON_NODE"
$AKSHA --version
for DB in PFAM KOFAM; do
    out=$V/cpu_${DB,,}
    test ! -e "$out"
    echo "== $DB CPU start $(date -u +%FT%TZ)"
    /usr/bin/time -v "$AKSHA" search --installed_hmms "$DB" --cut_ga --prot_in "$V/sample4000.faa" --outdir "$out" \
        --threads "$SLURM_CPUS_ON_NODE"
    echo "== $DB CPU end $(date -u +%FT%TZ)"
done
echo "VAL_CPU_DONE $(date -u +%FT%TZ)"
