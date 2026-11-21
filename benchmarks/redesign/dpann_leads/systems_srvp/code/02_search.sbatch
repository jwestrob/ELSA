#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand
#SBATCH -c 24
#SBATCH -J sysrvp_df
#SBATCH --array=0-7
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/dpann_leads/systems_srvp/logs/search_%A_%a.log

# systems_srvp: per-protein hits of the callers' own profiles (small multi-gene DefenseFinder 3.1.0 / RM / TXSScan
# models, hmm/picked/ from 02_search_prep.py) on the 8 srvp proteome shards, through Aksha with GA cutoffs.
# Selection (i-Evalue rescaled to the genome, profile coverage) is applied afterwards by 03_hits.py.
set -euo pipefail
RD=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111
W=$RD/dpann_leads/systems_srvp
S=$(printf '%03d' "$SLURM_ARRAY_TASK_ID")
OUT=$W/search/s$S
rm -rf "$OUT"
echo "CPUs $SLURM_CPUS_ON_NODE shard $S host $(hostname) start $(date -u +%FT%TZ)"
/usr/bin/time -v /home/jwestrob/.local/bin/aksha search --hmm_in $W/hmm/picked --cut_ga \
    --prot_in $RD/annot_gpu/srvp/shards/shard_$S.faa --outdir "$OUT" --threads "$SLURM_CPUS_ON_NODE"
echo "DONE $(date -u +%FT%TZ)" > "$OUT/STATUS"
