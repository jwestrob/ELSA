#!/bin/bash
# Move screen-3 structures and PAE blocks from a local fold directory to biotite as they arrive (files older than
# 30 s, so the collector has finished writing them); metrics.jsonl, state.json and jobs files stay local and are
# copied (not moved). Loops until the file STOP_SYNC exists in the fold directory and nothing is left to move.
# Usage: s3_sync.sh LOCAL_FOLD_DIR REMOTE_SUBDIR
set -u
LOCAL=$1
REMOTE=jwestrob@biotite.berkeley.edu:/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/cofold/s3/folds/$2
ssh -o BatchMode=yes jwestrob@biotite.berkeley.edu "mkdir -p /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/cofold/s3/folds/$2" || true
while true; do
  cd "$LOCAL" || exit 1
  find pdb pae -type f -mmin +1 2>/dev/null > .sync_list
  n=$(wc -l < .sync_list | tr -d ' ')
  if [ "$n" -gt 0 ]; then
    rsync -a --remove-source-files --files-from=.sync_list . "$REMOTE/" && echo "$(date -u +%H:%M:%S) moved $n files" \
      || echo "$(date -u +%H:%M:%S) rsync failed (will retry)"
  fi
  rsync -a metrics.jsonl state.json jobs_*.json "$REMOTE/" 2>/dev/null || true
  if [ -e STOP_SYNC ] && [ "$n" -eq 0 ] && [ -z "$(find pdb pae -type f 2>/dev/null | head -1)" ]; then
    echo "sync finished"; exit 0
  fi
  sleep 60
done
