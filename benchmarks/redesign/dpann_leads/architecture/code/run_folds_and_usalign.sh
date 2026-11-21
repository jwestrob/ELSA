#!/bin/bash
# Commands run inline for folding (Biohub ESMFold2, background, Mac) and US-align (biotite login node, serial).
set -euo pipefail
cd "$(dirname "$0")/.."
python3 /Users/jacob/Documents/Sandbox/Sharur/Sharur/scripts/fold_biohub.py folds/out folds/fold_input.faa > folds/fold.log 2>&1      # 20 folds
python3 /Users/jacob/Documents/Sandbox/Sharur/Sharur/scripts/fold_biohub.py folds/cofold_L1 folds/cofold_L1.faa \
    --ligand CTP:1 --ligand ATP:1 --ligand MG:2 > folds/cofold_L1.log 2>&1                                                      # 2 co-folds
SP=/private/tmp/claude-501/-Users-jacob-Documents-Sandbox-elsa-test-ELSA/4a153818-3d15-49e9-b3de-41cb21532c73/scratchpad
RD=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/dpann_leads/architecture
python code/s19_structure_pairs.py
ssh jwestrob@biotite.berkeley.edu "mkdir -p $RD/us"
scp -q -r $SP/us/str $SP/us/pairs.tsv code/usalign_run.py code/usalign_residue_map.py jwestrob@biotite.berkeley.edu:$RD/us/
ssh jwestrob@biotite.berkeley.edu "cd $RD/us && python3 usalign_run.py pairs.tsv str usalign_out.tsv && python3 usalign_residue_map.py str residue_map.tsv"
scp -q jwestrob@biotite.berkeley.edu:$RD/us/usalign_out.tsv folds/usalign_out.tsv
scp -q jwestrob@biotite.berkeley.edu:$RD/us/residue_map.tsv folds/usalign_residue_map.tsv
