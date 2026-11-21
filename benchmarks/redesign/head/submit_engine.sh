#!/bin/bash
# Submit engine jobs (run_recurrence_dpann_arm.sbatch, W=3, OUTDIR=results_dpann_head) for the token arms listed one
# per line in the remote file $RD/head/engine/arms.txt, at most $MAXJ (default 4) of them running or pending at once.
# Job IDs are appended to the local log $1. Exits when every submitted job has reached a terminal state.
set -uo pipefail
RD=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111
LOG=$1
MAXJ=${MAXJ:-4}
H=jwestrob@biotite.berkeley.edu
ids=()
inflight () {     # this workstream's in-flight CPU jobs: submitted engine jobs plus $OTHER (comma-separated IDs)
    local all
    all=$(printf "%s," ${ids[@]+"${ids[@]}"} ${OTHER:+${OTHER//,/ }})
    all=${all%,}; all=${all#,}
    [[ -z "$all" ]] && { echo 0; return; }
    local s
    s=$(ssh -n -o ConnectTimeout=20 $H "sacct -X -n -j $all --format=State") || { echo 99; return; }
    echo "$s" | grep -cE "PENDING|RUNNING|CONFIGURING|COMPLETING|REQUEUED|SUSPENDED"
}
arms=$(ssh -n $H "cat $RD/head/engine/arms.txt")
for arm in $arms; do
    until [[ $(inflight) -lt $MAXJ ]]; do sleep 30; done
    id=$(ssh -n -o ConnectTimeout=20 $H "cd $RD/engine && ARM=dpann__$arm TOKENS=$RD/engine/dpann_tokens/$arm.parquet W=3 OUTDIR=results_dpann_head sbatch --parsable --export=ALL $RD/engine/code/run_recurrence_dpann_arm.sbatch")
    ids+=("$id")
    echo "$(date -u +%FT%TZ) $arm $id" >> "$LOG"
    sleep 3
done
OTHER=""
until [[ $(inflight) -eq 0 ]]; do sleep 30; done
ssh -n $H "sacct -X -n -j $(IFS=,; echo "${ids[*]}") --format=JobID,JobName,State,Elapsed" >> "$LOG"
echo "ALL DONE" >> "$LOG"
cat "$LOG"
