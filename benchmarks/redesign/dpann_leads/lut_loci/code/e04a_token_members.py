#!/usr/bin/env python
"""lut_loci e04a (biotite login node): members of given rescue3 tokens, one per genome (first by gid), as FASTA.
usage: e04a_token_members.py OUT.faa TOKEN [TOKEN ...]"""
import sys

import numpy as np
import pandas as pd

RD = "/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111"
FA = RD + "/slice_lut/testset/package_k8/proteins_gid.faa"
toks = [int(t) for t in sys.argv[2:]]
tok = pd.read_parquet(RD + "/long_loci/frozen/tok/mmseqs__seq__e3c50s75_rescue3__lut.parquet").token.to_numpy()
gc = pd.read_parquet(RD + "/slice_lut/testset/package_k8/universe/genes.parquet", columns=["genome_code"]).genome_code.to_numpy()
g = np.flatnonzero(np.isin(tok, toks))
d = pd.DataFrame({"gid": g, "gc": gc[g], "tok": tok[g]}).drop_duplicates(["tok", "gc"])
want = dict(zip(d.gid, d.tok))
with open(FA) as fh, open(sys.argv[1], "w") as fo:
    keep = False
    for line in fh:
        if line[0] == ">":
            k = int(line[1:].split()[0])
            keep = k in want
            if keep:
                line = f">{k} token={want[k]}\n"
        if keep:
            fo.write(line)
print(d.groupby("tok").size().to_dict())
