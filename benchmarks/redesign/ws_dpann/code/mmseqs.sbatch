#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand_2
#SBATCH -c 64
#SBATCH -J dpann_mmseqs
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/ws_dpann/logs/mmseqs_%j.log
# Sequence-family token arm: MMseqs2 easy-cluster of all DPANN proteins, >=30% identity, >=80% coverage of both.
set -euo pipefail
W=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/ws_dpann
PY=/home/jwestrob/miniconda3/envs/elsa/bin/python
MM=/groups/banfield/users/jwestrob/bin/mmseqs_18/mmseqs/bin/mmseqs
N=$SLURM_CPUS_ON_NODE
D=$W/work/mmseqs_id30c80
mkdir -p $D $W/package/tokens
echo "host $(hostname) cpus $N mmseqs $($MM version) start $(date -Is)"
t0=$(date +%s)
$MM easy-cluster $W/work/proteins_gid.faa $D/fam $D/tmp --min-seq-id 0.3 -c 0.8 --cov-mode 0 \
    --threads "$N" > $D/mmseqs.log 2>&1
t1=$(date +%s)
rm -rf $D/tmp
echo "mmseqs done in $((t1 - t0)) s; members $(wc -l < $D/fam_cluster.tsv)"
PARAMS="{\"tool\": \"mmseqs easy-cluster\", \"mmseqs\": \"$MM\", \"mmseqs_version\": \"$($MM version)\", \"min_seq_id\": 0.3, \"coverage\": 0.8, \"cov_mode\": 0, \"cluster_mode\": \"default (0, set cover)\", \"sensitivity\": \"default\", \"input_fasta\": \"$W/work/proteins_gid.faa\", \"threads\": $N}"
$PY $W/code/mmseqs_tokens.py --clusters $D/fam_cluster.tsv --universe $W/package/universe/genes.parquet \
    --out $W/package/tokens/mmseqs__seq__id30c80__dpann --threads "$N" --params "$PARAMS" --mmseqs-seconds $((t1 - t0))
echo "end $(date -Is)"
