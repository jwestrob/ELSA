#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand_2
#SBATCH -c 64
#SBATCH -J hier_leiden
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/logs/hier_leiden_%j.log
# Task 1 Leiden runs on three graphs at once (Leiden is serial per run; parallelism = runs, 19 in total).
# 64 CPUs for the 256 GB the V0 graph needs (34.6 GB graph + ~9 GB per concurrent run, measured in job 78502).
set -euo pipefail
RD=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111
PY=$RD/ws_tokens/env/bin/python
C=$RD/hierarchy/code
T="${SLURM_CPUS_ON_NODE:?}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TMPDIR="$RD/tmp"
echo "$(date -Is) host=$(hostname) cpus=$T job=$SLURM_JOB_ID"
V0="19000:17,19000:18,19000:19,19000:20,13000:17,13500:17"
MU="3000:17,6000:17,10000:17,19000:17,30000:17,50000:17"
G1="3000:17,6000:17,10000:17,19000:17,30000:17,50000:17,100000:17"
"$PY" -u $C/leiden_t1.py --graph v0 --tasks "$V0" --procs 6 > $RD/hierarchy/logs/leiden_v0.log 2>&1 &
p1=$!
"$PY" -u $C/leiden_t1.py --graph mutual --tasks "$MU" --procs 6 > $RD/hierarchy/logs/leiden_mutual.log 2>&1 &
p2=$!
"$PY" -u $C/leiden_t1.py --graph g1 --tasks "$G1" --procs 7 > $RD/hierarchy/logs/leiden_g1.log 2>&1 &
p3=$!
wait $p3; echo "g1 done $(date -Is)"; wait $p2; echo "mutual done $(date -Is)"; wait $p1; echo "v0 done $(date -Is)"
# determinism check: seed 17 at r19000 against the saved context_kernel membership
"$PY" - <<'EOF'
import numpy as np
a = np.load("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/hierarchy/families/v0_r19000_s17.npy")
b = np.load("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/context_kernel/families/sim_all_r19000.npy")
print("seed-17 rerun identical to context_kernel sim_all_r19000:", bool(np.array_equal(a, b)), len(np.unique(a)), len(np.unique(b)))
EOF
echo "$(date -Is) end"
