#!/bin/bash
# ribo_audit part 4: login-node analysis chain (one thread, sequential) after job ribo4_annot.
set -euo pipefail
cd /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/ribo_audit
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1
PY=/home/jwestrob/miniconda3/envs/elsa/bin/python
for ds in ${DSLIST:-lut dpann}; do
  $PY -u code/bio4.py repro $ds
  $PY -u code/bio4.py ext $ds union
  $PY -u code/bio4.py ext $ds domain
  $PY -u code/bio4.py verdict $ds cluster
  $PY -u code/bio4.py verdict $ds standing
done
[ "${SKIP_INSERT:-0}" = 1 ] || $PY -u code/bio4.py insert lut
$PY -u code/tables4.py work4 work4/tables4.txt > /dev/null
echo CHAIN_DONE
