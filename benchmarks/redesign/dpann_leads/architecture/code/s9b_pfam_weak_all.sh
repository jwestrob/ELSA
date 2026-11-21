#!/bin/bash
# Architecture screen, step 9b: weak full-Pfam domain search (Aksha, Mac, 4 threads; --evalue 10 --domE 10) of all 6,916
# screen candidates (data/cand.faa) after step 9 showed the search takes seconds.
set -euo pipefail
cd "$(dirname "$0")/.."
SP=${SP:-/private/tmp/claude-501/-Users-jacob-Documents-Sandbox-elsa-test-ELSA/4a153818-3d15-49e9-b3de-41cb21532c73/scratchpad}
~/.local/bin/aksha search --hmm_in ~/.config/Astra/PFAM/Pfam-A.hmm --prot_in data/cand.faa --outdir $SP/pfamweak_all \
   --evalue 10 --domE 10 --threads 4
gzip -c $SP/pfamweak_all/user_hmms_hits_df.tsv > data/pfam_weak_hits_all.tsv.gz
zcat < data/pfam_weak_hits_all.tsv.gz | wc -l
