#!/usr/bin/env python3
"""Prepare structure files and the US-align pair list (rung 4).
- Domain boundary of each long single-chain model: the cut k within +-60 of the sequence-based cut (05 pieces.tsv, or
  the Pfam domain boundary for positive controls) that minimizes CA-CA contacts (< 8 A) between residues 1..k and k+1..n.
- Writes per-half PDBs (<id>_N.pdb, <id>_C.pdb), domain-as-chain PDBs for the oligomer models (each chain's N and C
  halves become separate chains, ligands dropped), 1O26 protein-only (4 chains) and 1O26 chain A.
- Pair list data/struct/pairs.tsv: kind, a, b, mode (mono or mm1).
usage: python code/11_prepare_structs.py"""
import os, glob, json, numpy as np, pandas as pd
os.makedirs("data/struct/pdb", exist_ok=True)
OUT = "data/struct/pdb"

def read_atoms(fn):
    A = []
    for l in open(fn):
        if l.startswith("ATOM"):
            A.append((l[21], int(l[22:26]), l[12:16].strip(), l))
    return A

def ca_xyz(A, chain):
    pts = {r: l for c, r, n, l in A if c == chain and n == "CA"}
    ks = sorted(pts)
    return ks, np.array([[float(pts[k][30:38]), float(pts[k][38:46]), float(pts[k][46:54])] for k in ks])

def best_cut(fn, guess, chain="A", w=60):
    ks, X = ca_xyz(read_atoms(fn), chain)
    n = len(ks)
    D = np.sqrt(((X[:, None] - X[None]) ** 2).sum(-1)) < 8.0
    best = None
    for k in range(max(100, guess - w), min(n - 100, guess + w)):
        c = D[:k, k:].sum()
        if best is None or c < best[1]:
            best = (k, c)
    return best

def write_sub(fn, out, ranges):
    """ranges: list of (src_chain, lo, hi, new_chain)"""
    with open(out, "w") as f:
        for c, r, n, l in read_atoms(fn):
            for sc, lo, hi, nc in ranges:
                if c == sc and lo <= r <= hi:
                    f.write(l[:21] + nc + l[22:])
        f.write("END\n")

P = pd.read_csv("data/msa/pieces.tsv", sep="\t")
guess = {r.protein_id: r.end for r in P[P.group == "lead_N"].itertuples()}
guess.update({"MBU2497176.1": 264, "NYZ74576.1": 296})
cuts = {}
for fn in sorted(glob.glob("data/fold/singles/*.pdb")):
    pid = os.path.basename(fn)[:-4]
    if pid in guess:
        k, c = best_cut(fn, guess[pid])
        n = len(ca_xyz(read_atoms(fn), "A")[0])
        cuts[pid] = dict(seq_cut=guess[pid], struct_cut=k, contacts=int(c), length=n)
        write_sub(fn, f"{OUT}/{pid}_N.pdb", [("A", 1, k, "A")])
        write_sub(fn, f"{OUT}/{pid}_C.pdb", [("A", k + 1, n, "A")])
    write_sub(fn, f"{OUT}/{pid}_full.pdb", [("A", 1, 10000, "A")])
json.dump(cuts, open("data/struct/cuts.json", "w"), indent=1)
print(pd.DataFrame(cuts).T)

# oligomer and ligand models: domain-as-chain versions
for sub in ["dimer_lig", "monomer_lig", "long_tetramer"]:
    for fn in sorted(glob.glob(f"data/fold/{sub}/*.pdb")):
        pid = os.path.basename(fn)[:-4]
        if pid not in cuts:
            continue
        k = cuts[pid]["struct_cut"]
        chains = sorted({c for c, r, n, l in read_atoms(fn)})
        rng, labels = [], iter("ABCDEFGHIJKLMNOP")
        for ch in chains:
            rng += [(ch, 1, k, next(labels)), (ch, k + 1, 10000, next(labels))]
        write_sub(fn, f"{OUT}/{sub}__{pid}__dom.pdb", rng)
for fn in sorted(glob.glob("data/fold/tetramer_lig/*.pdb")) + sorted(glob.glob("data/fold/dimer_lig/DAHVAY*.pdb")):
    pid = os.path.basename(fn)[:-4]
    sub = fn.split("/")[-2]
    chains = sorted({c for c, r, n, l in read_atoms(fn)})
    write_sub(fn, f"{OUT}/{sub}__{pid}__prot.pdb", [(c, 1, 10000, c) for c in chains])

# experimental reference 1O26
write_sub("data/ref/1O26.pdb", f"{OUT}/1O26_ABCD.pdb", [(c, 1, 10000, c) for c in "ABCD"])
write_sub("data/ref/1O26.pdb", f"{OUT}/1O26_A.pdb", [("A", 1, 10000, "A")])
for pair in ["AD", "AB", "AC"]:
    write_sub("data/ref/1O26.pdb", f"{OUT}/1O26_{pair}.pdb", [(pair[0], 1, 10000, "A"), (pair[1], 1, 10000, "B")])

rows = []
ref = "1O26_A.pdb"
for fn in sorted(glob.glob(f"{OUT}/*_full.pdb")):
    rows.append(("full_vs_ref", os.path.basename(fn), ref, "mono"))
for fn in sorted(glob.glob(f"{OUT}/*_N.pdb")) + sorted(glob.glob(f"{OUT}/*_C.pdb")):
    rows.append(("half_vs_ref", os.path.basename(fn), ref, "mono"))
for pid in cuts:
    rows.append(("N_vs_C", f"{pid}_N.pdb", f"{pid}_C.pdb", "mono"))
for fn in sorted(glob.glob(f"{OUT}/dimer_lig__*__dom.pdb")) + sorted(glob.glob(f"{OUT}/long_tetramer__*__dom.pdb")) + sorted(glob.glob(f"{OUT}/*__prot.pdb")):
    rows.append(("complex_vs_1O26", os.path.basename(fn), "1O26_ABCD.pdb", "mm1"))
for fn in sorted(glob.glob(f"{OUT}/monomer_lig__*__dom.pdb")) + [f"{OUT}/{p}_full.pdb" for p in ["NMB66601.1", "MEM4152697.1", "NYZ74576.1", "MBU2497176.1", "JAGVGF010000004.1_10", "HJO15200.1"]]:
    if not os.path.exists(fn):
        continue
    b = os.path.basename(fn)
    if b.endswith("_full.pdb"):  # make a two-chain domain version of the single-chain model
        pid = b[:-9]; k = cuts[pid]["struct_cut"]
        b = f"single__{pid}__dom.pdb"
        write_sub(fn, f"{OUT}/{b}", [("A", 1, k, "A"), ("A", k + 1, 10000, "B")])
    for pair in ["AD", "AB", "AC"]:
        rows.append((f"intrachain_vs_1O26_{pair}", b, f"1O26_{pair}.pdb", "mm1"))
R = pd.DataFrame(rows, columns=["kind", "a", "b", "mode"])
R.to_csv("data/struct/pairs.tsv", sep="\t", index=False)
print(R.kind.value_counts())
