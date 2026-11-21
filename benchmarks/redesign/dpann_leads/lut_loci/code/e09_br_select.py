#!/usr/bin/env python
"""lut_loci e09 (Mac): representatives, family alignment and fold input for the beta-roll follow-up.

Family = rescue3 token 275376 (lut_loci window family 52; 17 of the 29 handed-over motif proteins; the other 12 sit in 12
unrelated tokens, 8 of them with other Pfam domains, and share only the 5-residue motif). Representatives (n07 rule, motif
carriers preferred): one per genus for the 7 largest genera of the token (motif carrier first where the genus has one, else
the first gid; length 0.8-1.25x token median), plus one motif carrier for each further motif-carrying genus: 10 in total.
Alignment: token members one per genome plus representatives, mafft --auto. Extra negative-class control: P. aeruginosa serralysin AprA (UniProt Q03023, reviewed, hemolysin-type Ca-binding
beta-roll; PDB 1KAP), public sequence, structure-only like the public positive references.
Writes sel/selection_br.tsv, msa/br_<token>_aln.faa, msa/br_genome_group_<g>.faa, folds_in/br_apo.faa.
"""
import subprocess

import pandas as pd
from Bio import SeqIO

B = "/Users/jacob/Documents/Sandbox/elsa_test/ELSA/benchmarks/redesign/dpann_leads/lut_loci"
DL = "/private/tmp/claude-501/-Users-jacob-Documents-Sandbox-elsa-test-ELSA/4a153818-3d15-49e9-b3de-41cb21532c73/scratchpad/dl_br"
M = pd.read_csv(B + "/data/br_members.tsv", sep="\t").sort_values("gid")
seqs = {int(r.id): str(r.seq).rstrip("*") for r in SeqIO.parse(B + "/data/br_members.faa", "fasta")}
T = M[M.token == 275376]
med = T.length.median()
T = T[(T.length >= 0.8 * med) & (T.length <= 1.25 * med)].sort_values(["motif", "gid"], ascending=[False, True])
order = T.genus.value_counts().index.tolist()[:7]
order += [g for g in T[T.motif].genus.unique() if g not in order]
reps = [T[T.genus == g].iloc[0] for g in order][:10]
rows = [dict(role="br", family=f"br_{int(r.token)}", gid=int(r.gid), genus=r.genus, group=r.group, length=len(seqs[int(r.gid)]))
        for r in reps]
S = pd.DataFrame(rows)
for fam in S.family.unique():
    t = int(fam.split("_")[1])
    mem = M[M.token == t].drop_duplicates("genome").gid.tolist()
    ids = list(dict.fromkeys(mem + S[S.family == fam].gid.tolist()))
    fa = f"{B}/msa/{fam}.faa"
    with open(fa, "w") as f:
        for i in ids:
            f.write(f">{i}\n{seqs[i]}\n")
    with open(f"{B}/msa/{fam}_aln.faa", "w") as fo:
        subprocess.run(["mafft", "--auto", "--thread", "4", "--quiet", fa], stdout=fo, check=True)
    print(fam, len(ids))
ref = str(next(SeqIO.parse(DL + "/Q03023.fasta", "fasta")).seq)
S = pd.concat([S, pd.DataFrame([dict(role="brneg", family="betaroll_Ca_Q03023", gid="Q03023", genus="Pseudomonas",
                                     group="", length=len(ref))])], ignore_index=True)
S.to_csv(B + "/sel/selection_br.tsv", sep="\t", index=False)
with open(B + "/folds_in/br_apo.faa", "w") as f:
    for r in S.itertuples():
        s = ref if r.gid == "Q03023" else seqs[int(r.gid)]
        f.write(f">{r.role}__{r.family}__{r.gid}\n{s}\n")
print(S.to_string(index=False))
