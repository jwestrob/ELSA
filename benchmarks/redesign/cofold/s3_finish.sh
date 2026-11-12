#!/bin/bash
# Screen 3, completion of the 72 complexes > 768 aa that were queued at the Biohub API when REPORT_census.txt was written.
# Run from benchmarks/redesign/cofold after the main collector (biohub_cofold.py collect --out work/s3/folds_main) exits;
# s3_finish_watch starts it automatically. Steps (no folds are submitted here):
#  1. s3_analyze.py on the screen and follow-up folds -> work/s3/analyze_final.txt (class table, tests, block calls)
#  2. s3_followup.py fu2 -> work/s3/fold_manifest3_fu2.tsv + jobs_s3_fu2.jsonl for any new interface passing both
#     replicates (missing margin controls + monomers), to be submitted with:
#       python3 s3_submit.py submit --jobs work/s3/jobs_s3_fu2.jsonl --out work/s3/folds_fu2 --what "follow-up 2"
#       python3 biohub_cofold.py collect --out work/s3/folds_fu2
#     then rerun step 1 with work/s3/folds_fu2 added, s3_zipping.py, and for new hits s3_post.sbatch / s3_triage.sbatch
#     (inputs as in post1/), s3_split_check.py, s3_context.py.
set -u
cd "$(dirname "$0")"
D="work/s3/folds_probe work/s3/folds_main work/s3/folds_fu1"   # probe first: a main copy of cog3_002_r1 overrides it
[ -d work/s3/folds_fu2 ] && D="$D work/s3/folds_fu2"
python3 -W ignore s3_analyze.py $D > work/s3/analyze_final.txt 2>&1
python3 -W ignore s3_followup.py fu2 >> work/s3/analyze_final.txt 2>&1
echo "finished $(date -u)" >> work/s3/analyze_final.txt
