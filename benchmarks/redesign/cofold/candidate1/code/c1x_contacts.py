"""Task 4, step 1 (local, seconds): contact / non-contact alignment-column pairs from the 11 passing ABC models.

Alignments (deep dive): A c1A_inst.aln.fa (268 rows), B c1B_inst.aln.fa (268), C C_inst.aln.fa (258); rows are
engine instance ids. Distances: CB-CB (CA for Gly) in the single-sequence ABC models c1_abc_<slot>, slots 1-3, 5-12.
Per column pair: number of models where the pair is <= 8 A ('contact') and where it is > 12 A ('far').
contact  = <= 8 A in >= 6 of 11 models; noncontact = > 12 A in >= 6 of 11 models (intermediate pairs excluded).
Intra-chain pairs need |i - j| >= 6 alignment residues apart in every model where both are present.
Writes work/coup/pairs.json and the three alignments (copied) for the remote coupling job.
"""
import json
import shutil
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.PDB import PDBParser

HERE = Path(__file__).resolve().parent.parent
DD = HERE.parent / "deepdive" / "work"
O = HERE / "work" / "coup"
O.mkdir(parents=True, exist_ok=True)
ALN = {"A": DD / "aln" / "c1A_inst.aln.fa", "B": DD / "aln" / "c1B_inst.aln.fa", "C": DD / "c1_aln_out" / "C_inst.aln.fa"}


def rf(p):
    d, h = {}, None
    for line in open(p):
        line = line.strip()
        if line.startswith(">"):
            h = line[1:].split()[0]
            d[h] = []
        elif h:
            d[h].append(line)
    return {k: "".join(v) for k, v in d.items()}


def main():
    aln = {k: rf(v) for k, v in ALN.items()}
    sel = pd.read_csv(DD / "c1_selected.tsv", sep="\t")
    P = PDBParser(QUIET=True)
    near, far, seen, seq_sep_ok = {}, {}, {}, {}
    for _, s in sel.iterrows():
        if int(s.slot) == 4:
            continue
        m = P.get_structure("m", DD / "folds_c1abc" / "pdb" / f"c1_abc_{int(s.slot):02d}.pdb")[0]
        xyz, col = {}, {}
        for ch in "ABC":
            row = aln[ch][s.instance_id]
            cmap, k = {}, 0
            for c, a in enumerate(row):
                if a != "-":
                    k += 1
                    cmap[k] = c
            pts, cols, idx = [], [], []
            for r in m[ch]:
                a = r["CB"] if ("CB" in r and r.get_resname() != "GLY") else r["CA"]
                pts.append(a.coord)
                cols.append(cmap[r.id[1]])
                idx.append(r.id[1])
            xyz[ch], col[ch] = (np.array(pts), np.array(idx)), np.array(cols)
        for c1, c2 in (("B", "C"), ("A", "B"), ("A", "C"), ("C", "C"), ("B", "B"), ("A", "A")):
            (X1, i1), (X2, i2) = xyz[c1], xyz[c2]
            D = np.sqrt(((X1[:, None] - X2[None]) ** 2).sum(-1))
            for a, b in product(range(len(i1)), range(len(i2))):
                if c1 == c2 and (i2[b] - i1[a]) < 6:
                    continue
                key = (c1, c2, int(col[c1][a]), int(col[c2][b]))
                seen[key] = seen.get(key, 0) + 1
                if D[a, b] <= 8:
                    near[key] = near.get(key, 0) + 1
                elif D[a, b] > 12:
                    far[key] = far.get(key, 0) + 1
    out = {}
    for blk in ("BC", "AB", "AC", "CC", "BB", "AA"):
        ks = [k for k in seen if k[0] + k[1] == blk]
        out[blk] = dict(contact=[[k[2], k[3]] for k in ks if near.get(k, 0) >= 6],
                        noncontact=[[k[2], k[3]] for k in ks if far.get(k, 0) >= 6])
        print(blk, "contact", len(out[blk]["contact"]), "noncontact", len(out[blk]["noncontact"]))
    json.dump(out, open(O / "pairs.json", "w"))
    for k, v in ALN.items():
        shutil.copy(v, O / f"aln_{k}.fa")
    inst = pd.read_csv(DD / "c1_instances.tsv", sep="\t")[["instance_id", "phylum", "ord", "genome"]]
    inst.to_csv(O / "instances.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
