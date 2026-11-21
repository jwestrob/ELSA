# all-window proteins of family 17 (+-6 flanks), mmseqs2 clustering (local, 4 threads)
set -e
cd /Users/jacob/Documents/Sandbox/elsa_test/ELSA/benchmarks/redesign/dpann_leads/f17_machine
python -c "
import pandas as pd
a=pd.read_parquet('data/f17_windows.parquet').drop_duplicates('protein_id')
open('data/windows.faa','w').write(''.join(f'>{r.protein_id}\n{r.sequence.rstrip(\"*\")}\n' for r in a.itertuples()))
"
TMP=/private/tmp/claude-501/-Users-jacob-Documents-Sandbox-elsa-test-ELSA/4a153818-3d15-49e9-b3de-41cb21532c73/scratchpad/mm
mmseqs easy-cluster data/windows.faa data/mmw $TMP -s 7.5 -c 0.5 --cov-mode 0 -e 1e-3 --cluster-mode 0 --threads 4 >/dev/null
rm -f data/mmw_all_seqs.fasta data/mmw_rep_seq.fasta
