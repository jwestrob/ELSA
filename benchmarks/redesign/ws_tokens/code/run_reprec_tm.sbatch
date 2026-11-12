#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand
#SBATCH -c 24
#SBATCH -J tok_tm
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/ws_tokens/logs/tm_%j.log

# One-off: recompute precision for the grid8m and models_raw batches with the parser that
# treats degenerate MMseqs rows as "no alignment", then run TM-align (chunked) for all 16 arms.
set -euo pipefail
WS=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/ws_tokens
PY=/home/jwestrob/miniconda3/envs/elsa/bin/python
T="${SLURM_CPUS_ON_NODE:?}"
cd "$WS/code"
G=mmseqs__seq__id30c80; for t in raw globalcentred genomecentred; do for K in 65536 262144 1048576; do G=$G,kmeans__esm2_8m__${t}__K$K; done; done
M=""; for m in esm2_650m esmc600m; do for K in 65536 262144 1048576; do M=$M${M:+,}kmeans__${m}__raw__K$K; done; done
"$PY" -u quality.py precision --arms "$G" --aln "$WS/work/quality/grid8m/aln.tsv" --threads "$T"
"$PY" -u quality.py precision --arms "$M" --aln "$WS/work/quality/models_raw/aln.tsv" --threads "$T"
bash run_quality.sbatch tm_a "$G,$M" 1 1
