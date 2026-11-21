#!/bin/bash
# Submission gate for the ESM++ small embedding of the SR-VP bacteria package (workstream transfer_srvp_bac).
# Runs detached on the biotite login node (one squeue/sacct call every 5 min; no compute). It enforces the
# workstream's submission rules literally:
#   - the GPU worker is submitted only after build_a (FASTA + embed_queue prep) completed, and only when
#     `squeue -u jwestrob -p gpu` lists nothing (running or pending) for the whole account;
#   - this workstream holds at most 2 jobs (names sb_*), and the account stays under 400 CPUs after submission;
#   - before each sbatch the two remote instruction files are read into the log (Slurm preflight).
# After the GPU job is in, finalize_zca.sbatch is submitted with --dependency=afterok:<gpu job> as soon as the
# workstream holds at most one job. Then the gate exits. Usage: nohup gate.sh <build_a job> <tokens job> &
set -uo pipefail
A=${1:?build_a job id}; TK=${2:?tokens job id}
W=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/ws_srvp_bac
LOG=$W/logs/gate.log
stamp() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" >> "$LOG"; }
state() { sacct -X -n -j "$1" -o State 2>/dev/null | head -1 | tr -d ' '; }
mine() { squeue -u jwestrob -h -o "%j" | grep -c '^sb_'; }
cpus() { squeue -u jwestrob -h -t R -o "%C" | awk '{s+=$1} END {print s+0}'; }
preflight() { { echo "---- preflight $(date -u +%FT%TZ)"; cat /home/jwestrob/.claude/CLAUDE.md; cat /home/jwestrob/jwestrob/obsidian-vault/biotite-cluster-ops.md; } > "$W/logs/gate_preflight_last.txt"; stamp "preflight: read ~/.claude/CLAUDE.md and biotite-cluster-ops.md ($(wc -c < "$W/logs/gate_preflight_last.txt") bytes)"; }
stamp "gate start pid $$ build_a=$A tokens=$TK"
while true; do
  s=$(state "$A")
  case "$s" in
    COMPLETED) break ;;
    FAILED|CANCELLED*|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL) stamp "build_a $A ended $s; gate exits without submitting"; exit 1 ;;
  esac
  sleep 300
done
stamp "build_a $A COMPLETED"
[ -e "$W/embed/srvp_bac__esmpp_small__20261113/run.json" ] || { stamp "embed run not prepared; gate exits"; exit 1; }
G=""
while [ -z "$G" ]; do
  ngpu=$(squeue -u jwestrob -p gpu -h | wc -l); m=$(mine); c=$(cpus)
  if [ "$ngpu" -eq 0 ] && [ "$m" -le 1 ] && [ $((c + 12)) -lt 400 ]; then
    preflight
    if [ "$(squeue -u jwestrob -p gpu -h | wc -l)" -eq 0 ]; then
      G=$(sbatch --parsable "$W/code/embed_gpu.sbatch") && stamp "submitted GPU worker $G (account gpu jobs 0, workstream jobs $m, running CPUs $c)" || { stamp "sbatch GPU failed"; G=""; }
      [ -n "$G" ] && { sleep 10; scontrol show job "$G" | grep -E "JobState|Partition|NumCPUs|ExcNodeList|TRES" >> "$LOG"; }
    fi
  else
    stamp "waiting: account gpu jobs $ngpu, workstream jobs $m, running CPUs $c"
  fi
  [ -z "$G" ] && sleep 300
done
echo "$G" > "$W/logs/gpu_jobid.txt"
F=""
while [ -z "$F" ]; do
  m=$(mine); c=$(cpus); gs=$(state "$G")
  case "$gs" in FAILED|CANCELLED*|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL) stamp "GPU worker $G ended $gs; finalize not submitted"; exit 1 ;; esac
  if [ "$m" -le 1 ] && [ $((c + 24)) -lt 400 ]; then
    preflight
    ts=$(state "$TK"); dep="afterok:$G"
    case "$ts" in PENDING|RUNNING) dep="afterok:$G:$TK" ;; COMPLETED) ;; *) stamp "tokens $TK ended $ts; finalize still waits only on the GPU worker" ;; esac
    F=$(sbatch --parsable --dependency="$dep" "$W/code/finalize_zca.sbatch") && stamp "submitted finalize $F $dep (tokens $TK: $ts; running CPUs $c)" || { stamp "sbatch finalize failed"; F=""; }
  else
    stamp "finalize waiting: workstream jobs $m, running CPUs $c, GPU $gs, tokens $(state "$TK")"
  fi
  [ -z "$F" ] && sleep 300
done
echo "$F" > "$W/logs/finalize_jobid.txt"
stamp "gate done"
