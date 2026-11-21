#!/bin/bash
# ws_tokens watcher (Mac side). usage: tok_watch.sh JOBID [JOBID...]
# Exits as soon as ANY listed job reaches a terminal state, printing all states.
ids=$(echo "$@" | tr ' ' ',')
while true; do
  out=$(ssh -o ConnectTimeout=20 jwestrob@biotite.berkeley.edu "sacct -j $ids -X -n -P --format=JobID,JobName,State,Elapsed" 2>/dev/null)
  if echo "$out" | grep -qE '\|(COMPLETED|FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL)'; then
    echo "$out"; exit 0
  fi
  sleep 75
done
