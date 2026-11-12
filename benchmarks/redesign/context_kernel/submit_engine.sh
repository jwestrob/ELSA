#!/bin/bash
# Submit engine jobs (run_recurrence_dpann_arm.sbatch, W=3, OUTDIR=results_dpann_ctx) for the token arms
# listed one per line in $1, keeping at most $MAXJ of this workstream's jobs running or pending at once
# (other in-flight job IDs of the workstream can be passed in $OTHER, comma-separated). Job IDs go to $2.
# Exits when every submitted job has reached a terminal state.
set -uo pipefail
RD=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111
ARMS=$1
LOG=$2
MAXJ=${MAXJ:-4}
H=jwestrob@biotite.berkeley.edu
ids=()
inflight () {
    local all
    all=$(printf "%s," "${ids[@]}" ${OTHER:+${OTHER//,/ }})
    [[ -z "${all%,}" ]] && { echo 0; return; }
    local s
    s=$(ssh -n -o ConnectTimeout=20 $H "sacct -X -n -j ${all%,} --format=State") || { echo 99; return; }   # unknown -> wait
    echo "$s" | grep -cE "PENDING|RUNNING|CONFIGURING|COMPLETING|REQUEUED|SUSPENDED"
}
while read -r arm; do
    [[ -z "$arm" ]] && continue
    until [[ $(inflight) -lt $MAXJ ]]; do sleep 30; done
    id=$(ssh -n -o ConnectTimeout=20 $H "cd $RD/engine && ARM=dpann__$arm TOKENS=$RD/engine/dpann_tokens/$arm.parquet W=3 OUTDIR=results_dpann_ctx sbatch --parsable --export=ALL $RD/engine/code/run_recurrence_dpann_arm.sbatch")
    ids+=("$id")
    echo "$(date -u +%FT%TZ) $arm $id" | tee -a "$LOG"
    sleep 5
done < "$ARMS"
OTHER=""
until [[ $(inflight) -eq 0 ]]; do sleep 30; done
ssh -n $H "sacct -X -n -j $(IFS=,; echo "${ids[*]}") --format=JobID,State,Elapsed" | tee -a "$LOG"
