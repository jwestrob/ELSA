#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand
#SBATCH -c 24
#SBATCH -J llf_main
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/long_loci/frozen/logs/main_%j.log
# long_loci frozen run, J4: uid gate, float32 zca10 vectors, genome groups (0.75 rule), token cohesion, PRIMARY (lw.py WA
# build: bond-FDR tau, L1, L4b trees), recurrence engine on the same tokens (w 3, genome null, --all-genome-loci), DH.
set -euo pipefail
source /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/long_loci/frozen/code/env_f.sh
stamp "host=$(hostname) threads=$T job=$SLURM_JOB_ID"
cd "$C"
TOKF=$FZ/tok/mmseqs__seq__e3c50s75_rescue3__lut.parquet
test -s "$TOKF"; test -s "$FZ/gtable/sums.npz"
step f_uidgate "$PYE" -u fl.py uidgate
step f_zca32 "$PYE" -u fl.py zca32 --threads "$T"
step f_groups "$PYE" -u fl.py groups
step f_cohesion "$PYE" -u fl.py cohesion --threads "$T"
step f_wa "$PYE" -u fl.py wa --threads "$T"
step f_engine "$PYE" -u "$RD/engine/code/recurrence.py" --arm lut__mmseqs__seq__e3c50s75_rescue3 --universe "$PKG/universe" \
    --tokens "$TOKF" --w 3 --null genome --benchmark none --all-genome-loci --out "$FZ/engine" --threads "$T"
step f_dh "$PYE" -u fl.py dh --threads "$T"
stamp "ALL DONE"
