#!/usr/bin/env python
"""lut_loci e04 (Mac): representatives, family alignments and fold input for the EF-hand follow-up.

Representatives of the 32 query genes: one per genus (first gid), then a second for each of the three largest genera, and
the single query gene of token 248360. Same-family EF-hand comparators (fold comparison and the identical site test): label
carriers 3512381 (token 351841) and 318825 (token 248360). Families = rescue3 tokens, one member per genome (e04a) plus
representatives, mafft --auto (as s01). Writes sel/selection_ef.tsv, msa/<family>_aln.faa, folds_in/ef_apo.faa.
"""
import subprocess

import pandas as pd
from Bio import SeqIO

B = "/Users/jacob/Documents/Sandbox/elsa_test/ELSA/benchmarks/redesign/dpann_leads/lut_loci"
G = pd.read_csv(B + "/data/ef_genes.tsv", sep="\t").sort_values("gid")
seqs = {int(r.id): str(r.seq).rstrip("*") for r in SeqIO.parse(B + "/data/ef_sets.faa", "fasta")}
mem = {}
for r in SeqIO.parse(B + "/data/ef_tokens.faa", "fasta"):
    t = int(r.description.split("token=")[1])
    mem.setdefault(t, []).append(int(r.id))
    seqs[int(r.id)] = str(r.seq).rstrip("*")
order = G.genus.value_counts().index.tolist()
pick = [G[G.genus == g].gid.iloc[0] for g in order]
for g in order[:3]:
    pick.append(G[(G.genus == g) & ~G.gid.isin(pick)].gid.iloc[0])
t248 = G[G.token == 248360].gid.tolist()
pick += [x for x in t248 if x not in pick]
rows = [dict(role="efq", family=f"efq_{int(G.set_index('gid').token[g])}", gid=int(g),
             genus=G.set_index("gid").genus[g], length=len(seqs[g])) for g in pick]
rows += [dict(role="efcar", family="efcar_351841", gid=3512381, genus="Prosthecodimorpha", length=len(seqs[3512381])),
         dict(role="efcar", family="efq_248360", gid=318825, genus="Tagaea", length=len(seqs[318825]))]
S = pd.DataFrame(rows)
S.to_csv(B + "/sel/selection_ef.tsv", sep="\t", index=False)
fam_tok = {"efq_363245": 363245, "efq_248360": 248360, "efcar_351841": 351841}
for fam, t in fam_tok.items():
    ids = list(dict.fromkeys(mem[t] + S[S.family == fam].gid.tolist()))
    fa = f"{B}/msa/{fam}.faa"
    with open(fa, "w") as f:
        for i in ids:
            f.write(f">{i}\n{seqs[i]}\n")
    with open(f"{B}/msa/{fam}_aln.faa", "w") as fo:
        subprocess.run(["mafft", "--auto", "--thread", "4", "--quiet", fa], stdout=fo, check=True)
    print(fam, len(ids))
with open(B + "/folds_in/ef_apo.faa", "w") as f:
    for r in S.itertuples():
        f.write(f">{r.role}__{r.family}__{r.gid}\n{seqs[r.gid]}\n")
print(S.to_string(index=False))
