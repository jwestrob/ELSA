#!/bin/bash
# genome_center job chain (run on the biotite login node from $RD/genome_center/code; read the two cluster rule files first).
# lists x3 (parallel) -> grid (CV, gate, primary, frozen scans) -> nullx x3 (parallel) -> final (tables + STATUS).
# Every dependency is afterany: each stage checks its inputs, and final always writes $RD/genome_center/STATUS.
set -euo pipefail
cd /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/genome_center/code
rm -f ../STATUS
L1=$(STAGE=lists ARMS=Z,G1 sbatch --parsable --export=ALL -J gc_lists1 run_gc.sbatch)
L2=$(STAGE=lists ARMS=G05,B8 sbatch --parsable --export=ALL -J gc_lists2 run_gc.sbatch)
L3=$(STAGE=lists ARMS=B32 sbatch --parsable --export=ALL -J gc_lists3 run_gc.sbatch)
G=$(STAGE=grid sbatch --parsable --export=ALL -J gc_grid --dependency=afterany:$L1:$L2:$L3 run_gc.sbatch)
N0=$(STAGE=nullx PART=0/3 sbatch --parsable --export=ALL -J gc_null0 --dependency=afterany:$G run_gc.sbatch)
N1=$(STAGE=nullx PART=1/3 sbatch --parsable --export=ALL -J gc_null1 --dependency=afterany:$G run_gc.sbatch)
N2=$(STAGE=nullx PART=2/3 sbatch --parsable --export=ALL -J gc_null2 --dependency=afterany:$G run_gc.sbatch)
F=$(STAGE=final NPARTS=3 sbatch --parsable --export=ALL -J gc_final --dependency=afterany:$N0:$N1:$N2 run_gc.sbatch)
echo "lists $L1 $L2 $L3 grid $G nullx $N0 $N1 $N2 final $F"
