#!/bin/bash
# Architecture screen, step 9: weak full-Pfam domain search (Aksha, Mac, 4 threads; --evalue 10 --domE 10, no gathering
# cutoffs) of up to 3 members per conserved group, to annotate the extra region with domain coordinates.
set -euo pipefail
cd "$(dirname "$0")/.."
SP=${SP:-/private/tmp/claude-501/-Users-jacob-Documents-Sandbox-elsa-test-ELSA/4a153818-3d15-49e9-b3de-41cb21532c73/scratchpad}
~/.local/bin/aksha search --hmm_in ~/.config/Astra/PFAM/Pfam-A.hmm --prot_in data/pfam_weak_input.faa --outdir $SP/pfamweak \
   --evalue 10 --domE 10 --threads 4
cp $SP/pfamweak/user_hmms_hits_df.tsv data/pfam_weak_hits.tsv
wc -l data/pfam_weak_hits.tsv
