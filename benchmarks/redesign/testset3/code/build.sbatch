#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand
#SBATCH -c 24
#SBATCH -J ts3_build
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/testset3/logs/build_%j.log
# testset3 packaging: parse the selected GTDB r232 proteomes ($SLURM_CPUS_ON_NODE workers), universe tables + FASTA
# (DuckDB, $SLURM_CPUS_ON_NODE threads), embed_queue prep for ESM++ small (sharding only, no model).
set -euo pipefail
RD=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111
T3=$RD/testset3; P=$T3/package; W=$T3/work
PY=$RD/ws_tokens/env/bin/python
T="${SLURM_CPUS_ON_NODE:?}"
RUN=$T3/embed/ts3__esmpp_small__20261119
export OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
stamp() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*"; }
stamp "host=$(hostname) threads=$T"
mkdir -p $W $T3/embed
rm -rf $W/parts; $PY -u $T3/code/ts3_parse.py $T3/selection/selection.tsv $W $T
stamp "START build"
rm -rf $P; $PY -u $T3/code/ts3_build.py $T3/selection/selection.tsv $W/parts $P $T
rm -rf $P/.duck_tmp
stamp "START embed_queue prep (no model) -> $RUN"
set +u; source /groups/banfield/projects/environmental/sr/srvp2020/Jacob/embed_queue/code/env.sh; set -u
$PYBIN -u $EQ/code/embed_queue.py prep --fasta $P/proteins_gid.faa --run $RUN --backend esmpp --model-dir $ESMPP_SMALL \
    --model-id Synthyra/ESMplusplus_small --model-revision $ESMPP_SMALL_REV --dtype bf16
stamp "DONE"
