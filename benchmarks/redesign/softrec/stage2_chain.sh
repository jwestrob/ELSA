#!/bin/bash
# usage (login node): stage2_chain.sh SRC CAND [AFTER_JOBID]
# edges (64 CPU) -> Leiden scan 250..4000 (64 CPU) -> resolution pick + odd DH (24 CPU), chained afterok.
set -euo pipefail
SRC=$1; CAND=$2; AFTER=${3:-}
cd /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/softrec/r3/code
dep=""; [ -n "$AFTER" ] && dep="--dependency=afterok:$AFTER"
export STEPS="softrec3s.py cands --src $SRC --names $CAND --edges $CAND --redo"
J1=$(sbatch --parsable $dep --export=ALL -J e_$CAND run3_big.sbatch)
export STEPS="readings3.py families --cand $CAND --res 250,500,1000,2000,4000 --procs 5"
J2=$(sbatch --parsable --dependency=afterok:$J1 --export=ALL -J f_$CAND run3_big.sbatch)
export STEPS="readings3.py rescount --cand $CAND;readings3.py stage2 --cand $CAND --tag stage2_$CAND"
J3=$(sbatch --parsable --dependency=afterok:$J2 --export=ALL -J s_$CAND run3_py.sbatch)
echo "$CAND $J1 $J2 $J3"
