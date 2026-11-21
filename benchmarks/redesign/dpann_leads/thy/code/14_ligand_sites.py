#!/usr/bin/env python3
"""Rungs 3-4 in the co-folds: where do the predicted FAD and dUMP (CCD UMP) copies sit, and which conserved residues
(T. maritima ThyX Q9WYT0 numbering, taken from the halves alignment of 05_halves_msa.py; positive controls added to it
with mafft --add --keeplength) line each pocket. Same analysis on the experimental tetramer 1O26 for reference.
A pocket is 'composite' when the conserved residues within 4.5 A of the ligand come from an N half and a C half.
Writes data/ligand_sites.tsv and data/core_positions.tsv.
usage: python code/14_ligand_sites.py"""
import os, glob, json, subprocess, numpy as np, pandas as pd
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300); pd.set_option("display.max_colwidth", 120)
KEY = {53: "H", 78: "R", 79: "H", 80: "R", 88: "S", 90: "R", 147: "R", 165: "R", 174: "R"}

def readfa(fn):
    s, k = {}, None
    for l in open(fn):
        l = l.strip()
        if l.startswith(">"):
            k = l[1:].split()[0]; s[k] = ""
        elif k:
            s[k] += l
    return s

cuts = json.load(open("data/struct/cuts.json"))
# positive controls and extra halves -> add to the alignment
import duckdb
con = duckdb.connect()
con.execute("ATTACH '/Users/jacob/Documents/Sandbox/Sharur/Sharur/data/dpann/sharur.duckdb' AS d (READ_ONLY)")
extra = ["MBU2497176.1", "NYZ74576.1", "MBN2880691.1"]
S = dict(con.execute(f"SELECT protein_id, sequence FROM d.proteins WHERE protein_id IN {tuple(extra)}").fetchall())
if not os.path.exists("data/msa/halves_plus_aln.faa"):
    with open("data/msa/extra.faa", "w") as f:
        for p in extra:
            s = S[p].rstrip("*")
            if p in cuts:
                k = cuts[p]["struct_cut"]
                f.write(f">{p}|pos_N:1-{k}\n{s[:k]}\n>{p}|pos_C:{k+1}-{len(s)}\n{s[k:]}\n")
            else:
                f.write(f">{p}|pos_1:1-{len(s)}\n{s}\n")
    with open("data/msa/halves_plus_aln.faa", "w") as out:
        subprocess.run(["mafft", "--add", "data/msa/extra.faa", "--keeplength", "--thread", "4", "--quiet", "data/msa/halves_aln.faa"], stdout=out, check=True)
A = readfa("data/msa/halves_plus_aln.faa")
tm = A["Q9WYT0|ref"]; col_of, k = {}, 0
for i, ch in enumerate(tm):
    if ch != "-":
        k += 1; col_of[k] = i
rows = []
for name, s in A.items():
    if "|ref" in name:
        continue
    pid, rest = name.split("|"); grp, rng = rest.split(":"); start = int(rng.split("-")[0])
    idx, n = {}, start - 1
    for i, ch in enumerate(s):
        if ch != "-":
            n += 1; idx[i] = n
    for p, aa in KEY.items():
        c = col_of[p]
        if s[c] != "-":
            rows.append(dict(protein_id=pid, piece=grp, refpos=f"{aa}{p}", resnum=idx[c], aa=s[c].upper(), match=s[c].upper() == aa))
C = pd.DataFrame(rows)
C.to_csv("data/core_positions.tsv", sep="\t", index=False)

def parse(fn):
    P, Lg = [], []
    for l in open(fn):
        if l.startswith("ATOM"):
            P.append((l[21], int(l[22:26]), l[17:20], float(l[30:38]), float(l[38:46]), float(l[46:54])))
        elif l.startswith("HETATM") and l[17:20].strip() in ("FAD", "UMP"):
            Lg.append((l[21], int(l[22:26]), l[17:20].strip(), float(l[30:38]), float(l[38:46]), float(l[46:54])))
    return pd.DataFrame(P, columns=["ch", "res", "rn", "x", "y", "z"]), pd.DataFrame(Lg, columns=["ch", "res", "lig", "x", "y", "z"])

out = []
models = [("1O26", "1O26", "data/ref/1O26.pdb")]
for sub in ["dimer_lig", "monomer_lig", "tetramer_lig"]:
    for fn in sorted(glob.glob(f"data/fold/{sub}/*.pdb")):
        models.append((sub, os.path.basename(fn)[:-4], fn))
for sub, pid, fn in models:
    P, Lg = parse(fn)
    X = P[["x", "y", "z"]].to_numpy()
    cut = cuts.get(pid, {}).get("struct_cut")
    core = C[C.protein_id == pid]
    if pid in ("1O26", "THYX_THEMA_Q9WYT0"):
        core = pd.DataFrame([dict(resnum=p, refpos=f"{a}{p}", aa=a, match=True) for p, a in KEY.items()])
    cmap = {int(r.resnum): (r.refpos, r.aa) for r in core.itertuples()}
    for (lch, lres, lig), g in Lg.groupby(["ch", "res", "lig"]):
        G = g[["x", "y", "z"]].to_numpy()
        dmin = np.sqrt(((X[:, None] - G[None]) ** 2).sum(-1)).min(1)
        near = P[dmin <= 4.5].assign(d=dmin[dmin <= 4.5]).groupby(["ch", "res"]).d.min().reset_index()
        halves, hits = set(), []
        for r in near.itertuples():
            h = "" if cut is None else ("N" if r.res <= cut else "C")
            halves.add(f"{r.ch}{h}")
            if int(r.res) in cmap:
                rp, aa = cmap[int(r.res)]
                hits.append(f"{r.ch}{h}:{aa}{r.res}={rp}")
        core_halves = sorted({h.split(":")[0] for h in hits})
        out.append(dict(set=sub, model=pid, ligand=f"{lig}_{lch}{lres}", n_contact_res=len(near), chains_halves="+".join(sorted(halves)),
                        core_contacts=" ".join(sorted(hits, key=lambda t: t.split("=")[1])), n_core=len(hits),
                        composite=(cut is not None and any(h.endswith("N") for h in core_halves) and any(h.endswith("C") for h in core_halves))))
R = pd.DataFrame(out)
R.to_csv("data/ligand_sites.tsv", sep="\t", index=False)
print(R.to_string(index=False))
print(R.groupby(["set", "model"]).agg(ligands=("ligand", "size"), with_core=("n_core", lambda x: int((x >= 2).sum())), composite=("composite", "sum")).to_string())
