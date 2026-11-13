#!/bin/bash
# Wait until a Slurm job on biotite leaves the queue; tolerate ssh failures. Usage: wait_job.sh JOBID [LOGLINES]
J=$1
N=${2:-15}
while true; do
  r=$(ssh -o ConnectTimeout=30 -o ServerAliveInterval=15 jwestrob@biotite.berkeley.edu "squeue -h -j $J 2>/dev/null | wc -l" 2>/dev/null)
  if [ "$r" = "0" ]; then break; fi
  sleep 60
done
ssh -o ConnectTimeout=60 jwestrob@biotite.berkeley.edu "sacct -j $J -o JobID,State,Elapsed,MaxRSS -n | head -3; tail -$N /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/locus_space/logs/lsp_$J.log"
