#!/usr/bin/env python
"""lut_loci e02 (biotite login node): sequence sets for the EF-hand follow-up.

Writes data/ef_sets.tsv (gid, set) and data/ef_sets.faa:
  query        the 32 EF-hand_5-labeled genes (data/ef_gids.txt)
  qfam         all members of their rescue3 tokens (363245, 248360)
  carrier      label-carrying supporters (data/ef_supporters.tsv)
  genome_ef    every EF-hand-Pfam protein of the 32 genomes (data/ef_genome_efhands.tsv)
  lanM_fam     all members of the lanmodulin token 358533
  ordEF_fam    all members of token 328557 (the commonest non-lanmodulin EF-hand family of the 32 genomes)
"""
import numpy as np
import pandas as pd

RD = "/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111"
FA = RD + "/slice_lut/testset/package_k8/proteins_gid.faa"
tok = pd.read_parquet(RD + "/long_loci/frozen/tok/mmseqs__seq__e3c50s75_rescue3__lut.parquet").token.to_numpy()
q = [int(x) for x in open("data/ef_gids.txt") if x.strip()]
S = pd.read_csv("data/ef_supporters.tsv", sep="\t")
E = pd.read_csv("data/ef_genome_efhands.tsv", sep="\t")
rows = [(g, "query") for g in q]
rows += [(int(g), "qfam") for g in np.flatnonzero(np.isin(tok, [363245, 248360]))]
rows += [(int(g), "carrier") for g in S[S.carries_label].supporter.unique()]
rows += [(int(g), "genome_ef") for g in E.gid]
rows += [(int(g), "lanM_fam") for g in np.flatnonzero(tok == 358533)]
rows += [(int(g), "ordEF_fam") for g in np.flatnonzero(tok == 328557)]
R = pd.DataFrame(rows, columns=["gid", "set"])
R["token"] = tok[R.gid]
R.to_csv("data/ef_sets.tsv", sep="\t", index=False)
want = set(R.gid)
n = 0
with open(FA) as fh, open("data/ef_sets.faa", "w") as fo:
    keep = False
    for line in fh:
        if line[0] == ">":
            keep = int(line[1:].split()[0]) in want
            n += keep
        if keep:
            fo.write(line)
print(R.groupby("set").gid.nunique().to_dict(), "sequences", n)
