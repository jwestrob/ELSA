#!/usr/bin/env python
"""lut_loci e03 (Mac): EF-hand loops, loop by loop, for the EF-hand follow-up (sequence part, tests_rule.json T_seq patterns).

Loops = 12-residue matches of the lanthanide project's site_cn.py patterns: relaxed D.[DNS].[DNSEG]G.{5}[DE] (primary) and
strict D[^W][DNS][^ILVFYW][DENSTG][DNQGHRK][^GP][LIVMC][DENQSTAGC].{2}[DE]; overlapping matches merged (first kept).
Per loop: residue at loop position 2 (Pro = the lanmodulin feature whose Pro->Ala mutation changes its Ca response,
reference set: PMID 30351021), acidic residues among the 12, residue 9, start-to-start spacing to the next loop
(lanmodulin homologs about 24-25 residues; lanthanide project notes 2026-11-09).
Sets: query (32 genes), lanM_same_genome (lanmodulin-token proteins of the 32 genomes), ordEF_same_genome (other EF-hand Pfam
proteins of the 32 genomes), carrier (label-carrying partners), lanM_fam (all lanmodulin-token proteins of the collection).
Writes data/ef_loops.tsv (one row per loop) and data/ef_loops_protein.tsv (one row per protein).
"""
import re

import numpy as np
import pandas as pd
from Bio import SeqIO

B = "/Users/jacob/Documents/Sandbox/elsa_test/ELSA/benchmarks/redesign/dpann_leads/lut_loci"
EF = re.compile(r"(?=(D[^W][DNS][^ILVFYW][DENSTG][DNQGHRK][^GP][LIVMC][DENQSTAGC].{2}[DE]))")
EFR = re.compile(r"(?=(D.[DNS].[DNSEG]G.{5}[DE]))")
seqs = {int(r.id): str(r.seq).rstrip("*") for r in SeqIO.parse(B + "/data/ef_sets.faa", "fasta")}
R = pd.read_csv(B + "/data/ef_sets.tsv", sep="\t")
E = pd.read_csv(B + "/data/ef_genome_efhands.tsv", sep="\t")
sets = {"query": R[R.set == "query"].gid.tolist(),
        "lanM_same_genome": E[E.lanM_token].gid.tolist(),
        "ordEF_same_genome": E[~E.lanM_token].gid.tolist(),
        "carrier": R[R.set == "carrier"].gid.unique().tolist(),
        "lanM_fam": R[R.set == "lanM_fam"].gid.tolist()}
lr, pr = [], []
for st, gl in sets.items():
    for g in gl:
        s = seqs[g]
        starts = sorted({m.start(1) for m in EFR.finditer(s)} | {m.start(1) for m in EF.finditer(s)})
        keep = []
        for x in starts:
            if not keep or x - keep[-1] >= 12:
                keep.append(x)
        strict = {m.start(1) for m in EF.finditer(s)}
        for i, x in enumerate(keep):
            loop = s[x:x + 12]
            lr.append(dict(set=st, gid=g, loop_no=i + 1, start=x + 1, loop=loop, strict=x in strict, pos2=loop[1],
                           pro2=loop[1] == "P", acidic=sum(c in "DE" for c in loop), pos9=loop[8],
                           spacing_next=(keep[i + 1] - x) if i + 1 < len(keep) else np.nan))
        sp = np.diff(keep) if len(keep) > 1 else np.array([])
        pr.append(dict(set=st, gid=g, length=len(s), n_loops=len(keep), n_strict=len(strict & set(keep)),
                       n_pro2=sum(s[x + 1] == "P" for x in keep), acidic_mean=np.mean([sum(c in "DE" for c in s[x:x + 12]) for x in keep]) if keep else np.nan,
                       spacing=";".join(map(str, sp)), spacing_24_25=int(np.sum((sp >= 23) & (sp <= 26)))))
L = pd.DataFrame(lr)
P = pd.DataFrame(pr)
L.to_csv(B + "/data/ef_loops.tsv", sep="\t", index=False)
P.to_csv(B + "/data/ef_loops_protein.tsv", sep="\t", index=False)
pd.set_option("display.width", 250)
print(P.groupby("set").agg(n=("gid", "size"), len_med=("length", "median"), loops_med=("n_loops", "median"),
                           loops_ge3=("n_loops", lambda v: int((v >= 3).sum())), any_loop=("n_loops", lambda v: int((v >= 1).sum())),
                           strict_med=("n_strict", "median"), pro2_any=("n_pro2", lambda v: int((v >= 1).sum())),
                           acidic=("acidic_mean", "median"), spacing2425_any=("spacing_24_25", lambda v: int((v >= 1).sum()))).to_string())
print(L.groupby("set").agg(loops=("loop", "size"), pro2=("pro2", "sum"), acidic_med=("acidic", "median"),
                           sp_med=("spacing_next", "median")).to_string())
q = L[L.set == "query"]
print(q.groupby("loop_no").agg(n=("loop", "size"), pro2=("pro2", "sum"), acid=("acidic", "median"),
                               pos2=("pos2", lambda v: "".join(v.value_counts().head(4).index)),
                               pos9=("pos9", lambda v: "".join(v.value_counts().head(4).index)),
                               sp=("spacing_next", "median"), ex=("loop", lambda v: v.value_counts().index[0])).to_string())
m = L[L.set == "lanM_same_genome"]
print(m.groupby("loop_no").agg(n=("loop", "size"), pro2=("pro2", "sum"), acid=("acidic", "median"),
                               pos2=("pos2", lambda v: "".join(v.value_counts().head(4).index)),
                               sp=("spacing_next", "median"), ex=("loop", lambda v: v.value_counts().index[0])).to_string())
o = L[L.set == "ordEF_same_genome"]
print(o.groupby("loop_no").agg(n=("loop", "size"), pro2=("pro2", "sum"), acid=("acidic", "median"),
                               pos2=("pos2", lambda v: "".join(v.value_counts().head(4).index)),
                               sp=("spacing_next", "median"), ex=("loop", lambda v: v.value_counts().index[0])).to_string())
