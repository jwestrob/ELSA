#!/bin/bash
# usage (login node): stage2_ex.sh C1,C2,C3,C4   -> one edges job (lists loaded once), Leiden in two jobs, then per
# candidate rescount + odd DH (24 CPU), all chained afterok.
set -euo pipefail
CANDS=$1
cd /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/softrec/r3/code
export STEPS="softrec3s.py cands --src ex --names $CANDS --edges $CANDS --redo"
J1=$(sbatch --parsable --export=ALL -J e_ex run3_big.sbatch)
IFS=',' read -r -a C <<< "$CANDS"
A="${C[0]},${C[1]}"; B="${C[2]},${C[3]}"; P="${C[4]:-}"
export STEPS="readings3.py families --cand $A --res 250,500,1000,2000,4000 --procs 10"
J2=$(sbatch --parsable --dependency=afterok:$J1 --export=ALL -J f_exA run3_big.sbatch)
export STEPS="readings3.py families --cand $B --res 250,500,1000,2000,4000 --procs 10"
J3=$(sbatch --parsable --dependency=afterok:$J1 --export=ALL -J f_exB run3_big.sbatch)
J4=""
if [ -n "$P" ]; then
  export STEPS="readings3.py families --cand $P --res 250,500,1000,2000,4000 --procs 5"
  J4=$(sbatch --parsable --dependency=afterok:$J1 --export=ALL -J f_exP run3_big.sbatch)
fi
out="$J1 $J2 $J3 $J4"
for i in "${!C[@]}"; do
  c=${C[$i]}; dep=$([ $i -lt 2 ] && echo $J2 || ([ $i -lt 4 ] && echo $J3 || echo $J4))
  export STEPS="readings3.py rescount --cand $c;readings3.py stage2 --cand $c --tag stage2_$c"
  out="$out $(sbatch --parsable --dependency=afterok:$dep --export=ALL -J s_$c run3_py.sbatch)"
done
echo "$out"
