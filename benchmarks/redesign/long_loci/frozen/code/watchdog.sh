#!/bin/bash
# long_loci frozen run: login-node watchdog (light: one squeue per registered job per minute). Cancels a registered job
# that has been RUNNING longer than its cap (about twice its expected time) or that waits on a dependency that can never
# be satisfied; logs every action; exits when no registered job is pending or running. Registry lines: jobid<TAB>cap_min<TAB>label
# (re-read every minute). Used instead of `timeout` inside the job scripts, which the account rules forbid.
REG=$1; LOG=$2
echo "$(date -u +%FT%TZ) watchdog start pid $$ registry $REG" >> "$LOG"
while true; do
  live=0
  while IFS=$'\t' read -r id cap label; do
    [[ -z "$id" || "$id" == \#* ]] && continue
    line=$(squeue -h -j "$id" -o "%T|%M|%r" 2>/dev/null)
    [[ -z "$line" ]] && continue
    live=1
    st=${line%%|*}; rest=${line#*|}; el=${rest%%|*}; reason=${rest#*|}
    if [[ "$st" == RUNNING ]]; then
      m=$(echo "$el" | awk -F'[-:]' '{ if (NF==4) print $1*1440+$2*60+$3; else if (NF==3) print $1*60+$2; else print $1 }')
      if (( m > cap )); then
        scancel "$id"; echo "$(date -u +%FT%TZ) CANCELLED $id ($label): running $el > cap $cap min" >> "$LOG"
      fi
    elif [[ "$reason" == *DependencyNeverSatisfied* ]]; then
      scancel "$id"; echo "$(date -u +%FT%TZ) CANCELLED $id ($label): DependencyNeverSatisfied" >> "$LOG"
    fi
  done < "$REG"
  if (( live == 0 )); then echo "$(date -u +%FT%TZ) no registered job pending or running; exit" >> "$LOG"; exit 0; fi
  sleep 60
done
