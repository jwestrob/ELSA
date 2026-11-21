"""Where the interfaces of the screen-2 hit blocks sit: per interface (i, j) and replicate, the fraction of each
chain's contact residues (CB-CB <= 8 A) that lie in the chain's first 30 residues (the class III signal /
alpha-1 region of pilin-like proteins), and the hydrophobic fraction (ILVFMAWC) of residues 1-30.
Usage: python s2_iface_location.py RANK [...]; reads the single-mode complexes; writes work/s2/s2_iface_location.tsv."""
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from s2_analyze_folds import read_pdb  # noqa: E402

W2 = HERE / "work" / "s2"
AA = dict(ALA="A", ARG="R", ASN="N", ASP="D", CYS="C", GLN="Q", GLU="E", GLY="G", HIS="H", ILE="I", LEU="L", LYS="K",
          MET="M", PHE="F", PRO="P", SER="S", THR="T", TRP="W", TYR="Y", VAL="V")


def seq_of(pdb):
    s, cur = {}, None
    for line in open(pdb):
        if line.startswith("ATOM"):
            key = (line[21], int(line[22:26]))
            if key != cur:
                cur = key
                s.setdefault(line[21], []).append(AA.get(line[17:20], "X"))
    return {k: "".join(v) for k, v in s.items()}


rows = []
for rank in map(int, sys.argv[1:]):
    for rep in (1, 2):
        pdb = next(iter(sorted(W2.glob(f"folds_s2_b*/pdb/cog2_{rank:03d}_r{rep}.pdb"))))
        ch, xyz, pl = read_pdb(pdb)
        sq = seq_of(pdb)
        ids = sorted(set(ch), key=lambda c: "ABCDEFGH".index(c))
        resi = np.concatenate([np.arange(int((ch == c).sum())) for c in ids])
        d = np.sqrt(((xyz[:, None, :] - xyz[None, :, :]) ** 2).sum(-1))
        for a, b in combinations(range(len(ids)), 2):
            ia, ib = ch == ids[a], ch == ids[b]
            cm = d[np.ix_(ia, ib)] <= 8.0
            if cm.sum() < 10:
                continue
            ra, rb = resi[ia][cm.any(1)], resi[ib][cm.any(0)]
            rows.append(dict(s2_rank=rank, rep=rep, pair=f"{ids[a]}{ids[b]}", contacts=int(cm.sum()),
                             nterm30_frac_i=round(float((ra < 30).mean()), 2), nterm30_frac_j=round(float((rb < 30).mean()), 2),
                             hyd30_i=round(sum(x in "ILVFMAWC" for x in sq[ids[a]][:30]) / 30, 2),
                             hyd30_j=round(sum(x in "ILVFMAWC" for x in sq[ids[b]][:30]) / 30, 2)))
out = pd.DataFrame(rows)
out.to_csv(W2 / "s2_iface_location.tsv", sep="\t", index=False)
print(out.to_string())
