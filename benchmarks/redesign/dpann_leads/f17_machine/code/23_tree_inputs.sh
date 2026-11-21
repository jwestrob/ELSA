# Q7: VirB4 / HerA tree inputs: 40 diverse F17 ATPases (mmseqs 50% id reps), curated references fetched by accession,
# Micrarchaeota non-F17 DUF87 proteins and Nanobdellota Top6B-near DUF87 proteins (DPANN sequences stay local).
set -e
cd /Users/jacob/Documents/Sandbox/elsa_test/ELSA/benchmarks/redesign/dpann_leads/f17_machine
SP=/private/tmp/claude-501/-Users-jacob-Documents-Sandbox-elsa-test-ELSA/4a153818-3d15-49e9-b3de-41cb21532c73/scratchpad
mmseqs easy-cluster data/roles/ATPase504.all.faa $SP/atp50 $SP/mmtmp --min-seq-id 0.5 -c 0.8 --threads 4 >/dev/null
python3 - <<'PY'
import random, duckdb, json, urllib.request
random.seed(17)
reps = sorted({l.split()[0] for l in open('/private/tmp/claude-501/-Users-jacob-Documents-Sandbox-elsa-test-ELSA/4a153818-3d15-49e9-b3de-41cb21532c73/scratchpad/atp50_cluster.tsv')})
pick = random.sample(reps, min(40, len(reps)))
seqs = {}
cur = None
for l in open('data/roles/ATPase504.all.faa'):
    l = l.strip()
    if l.startswith('>'): cur = l[1:]; seqs[cur] = ''
    else: seqs[cur] += l
out = [f'>F17_{p}\n{seqs[p]}\n' for p in pick]
acc = {'Q4JAQ5': 'CedB_Sacidocaldarius', 'Q9YFN0': 'CedB_Apernix', 'A3MU65': 'TedB_Pcalidifontis', 'P17794': 'VirB4_Atumefaciens', 'O50330': 'TrwK_R388',
       'P18004': 'TraC_F', 'B0K833': 'VirB4_Tpseudethanolicus1', 'B0KAW2': 'VirB4_Tpseudethanolicus2', 'Q5G3P5': 'PrgJ_pCF10', 'Q6AG11': 'TrsE_Leifsonia',
       'Q2L5X8': 'TcpF_Cperfringens', 'Q6F059': 'pXO2-09_Banthracis', 'R4PMW4': 'VirB4_Saccharimonas', 'A0A857MM64': 'VirB4_Mycosynbacter',
       'A0ABY0FHL7': 'VirB4_Nanogingivalis', 'A0A1N7GTM9': 'VirB4_Natronorubrum', 'A0A2K5ARD6': 'VirB4_Nitrosocaldus', 'A0A1H8W697': 'VirB4_Halogranum',
       'Q97WG8': 'HerA_Ssolfataricus', 'Q96YR7': 'HerA_Stokodaii', 'F2Z5Z6': 'HerA_Sacidocaldarius', 'Q8U1P0': 'HerA_Pfuriosus', 'Q58824': 'HerA_Mjannaschii',
       'Q9YFZ4': 'HerA_Apernix', 'S6DG77': 'TraG-like_Ssolfataricus_plasmid', 'B6D922': 'TraG-like_Ahospitalis'}
for a, n in acc.items():
    s = urllib.request.urlopen(f'https://rest.uniprot.org/uniprotkb/{a}.fasta').read().decode().split('\n', 1)[1].replace('\n', '')
    out.append(f'>REF_{n}_{a}\n{s}\n')
c = duckdb.connect('/Users/jacob/Documents/Sandbox/Sharur/Sharur/data/dpann/sharur.duckdb', read_only=True)
import pandas as pd
w = pd.read_csv('data/wider_top6_duf87.tsv', sep='\t')
nano = w[w.phylum == 'p__Nanobdellota'].duf.drop_duplicates().sample(3, random_state=17).tolist() + w[w.phylum == 'p__Aenigmatarchaeota'].duf.drop_duplicates().head(1).tolist()
for pid, tag in [(p, 'DPANN_Top6Bnear') for p in nano] + [('DAHWZL010000041.1_15', 'MICR_otherDUF87'), ('DAHWZL010000002.1_1', 'MICR_otherDUF87')]:
    s = c.sql(f"select rtrim(sequence,'*') from proteins where protein_id='{pid}'").fetchone()[0]
    out.append(f'>{tag}_{pid}\n{s}\n')
open('tree/atpase_tree_in.faa', 'w').write(''.join(out))
print(len(out))
PY
mafft --localpair --maxiterate 1000 --thread 4 --quiet tree/atpase_tree_in.faa > tree/atpase.aln
trimal -in tree/atpase.aln -out tree/atpase.trim.aln -automated1
iqtree2 -s tree/atpase.trim.aln -m LG+F+G4 -B 1000 -T 4 --prefix tree/atpase -redo -quiet
