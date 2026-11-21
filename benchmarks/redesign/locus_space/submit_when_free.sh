#!/bin/bash
# Submit one run_lsp.sbatch job once the account holds < 400 running CPUs and I have < 2 lsp jobs.
# Usage: submit_when_free.sh SCRIPT STAGE "EXTRA ARGS"
S=$1; ST=$2; EX=$3
RD=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111
while true; do
  out=$(ssh -o ConnectTimeout=60 jwestrob@biotite.berkeley.edu "c=\$(squeue -u jwestrob -h -t R -o '%C' | awk '{c+=\$1} END {print c+0}'); n=\$(squeue -u jwestrob -h -n lsp | wc -l); if [ \$c -lt 400 ] && [ \$n -lt 2 ]; then cd $RD/locus_space/code && sbatch --export=ALL,SCRIPT=$S,STAGE=$ST,EXTRA='$EX' run_lsp.sbatch && echo \"cpus \$c\"; else echo WAIT \$c \$n; fi" 2>/dev/null)
  case "$out" in
    *Submitted*) echo "$out"; break;;
  esac
  sleep 60
done
