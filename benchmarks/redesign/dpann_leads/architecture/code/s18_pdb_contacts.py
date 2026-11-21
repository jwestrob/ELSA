#!/usr/bin/env python3
"""Step 18 (rung 3 from experimental complexes): residues of a characterized relative within 4.0 A of its bound
substrate/product in a PDB entry, mapped onto lead members through a MAFFT L-INS-i alignment of the PDB chain sequence
with the members. Also tests the pol-beta nucleotidyltransferase signature in the NTP_transf_5 part of L2.
Writes data/contacts_<pdb>_<ligand>.tsv."""
import os, re, subprocess, tempfile, math
import numpy as np, pandas as pd
os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/..")
SP = os.environ.get("SP", "/private/tmp/claude-501/-Users-jacob-Documents-Sandbox-elsa-test-ELSA/4a153818-3d15-49e9-b3de-41cb21532c73/scratchpad")
AA = dict(ALA="A", ARG="R", ASN="N", ASP="D", CYS="C", GLN="Q", GLU="E", GLY="G", HIS="H", ILE="I", LEU="L", LYS="K", MET="M", PHE="F", PRO="P",
          SER="S", THR="T", TRP="W", TYR="Y", VAL="V", MSE="M")
seqs = pd.read_parquet("data/genes.parquet", columns=["protein_id", "seq"]).set_index("protein_id").seq
def chain(pdb, ch):
    res, atoms, lig = [], {}, []
    for l in open(f"{SP}/ref/pdb/{pdb}.pdb"):
        if l.startswith("ENDMDL"): break
        if l[:6] in ("ATOM  ", "HETATM") and l[21] == ch:
            rn, num, ic = l[17:20].strip(), int(l[22:26]), l[26]
            xyz = np.array([float(l[30:38]), float(l[38:46]), float(l[46:54])])
            if l[:6] == "ATOM  " or rn == "MSE":
                key = (num, ic)
                if key not in atoms: atoms[key] = []; res.append((key, AA.get(rn, "X")))
                atoms[key].append(xyz)
            elif rn != "HOH": lig.append((rn, xyz))
    return res, atoms, lig
def contacts(pdb, ch, ligs, cut=4.0):
    res, atoms, lig = chain(pdb, ch)
    L = [x for n, x in lig if n in ligs]
    seq = "".join(a for _, a in res)
    hit = {}
    for i, (key, a) in enumerate(res):
        d = min(np.linalg.norm(np.array(atoms[key]) - x, axis=1).min() for x in L) if L else 99
        if d <= cut: hit[i + 1] = (f"{a}{key[0]}", round(float(d), 2))
    return seq, hit
def mafft(recs):
    with tempfile.NamedTemporaryFile("w", suffix=".faa", delete=False, dir=SP) as f:
        for k, s in recs: f.write(f">{k}\n{s}\n")
    out = subprocess.run(["mafft", "--localpair", "--maxiterate", "1000", "--thread", "4", "--quiet", f.name], capture_output=True, text=True).stdout
    al = {}
    for blk in out.split(">")[1:]:
        h, *s = blk.split("\n"); al[h.split()[0]] = "".join(s).upper()
    return al
def run(pdb, ch, ligs, members, tag, region=None):
    seq, hit = contacts(pdb, ch, ligs)
    recs = [(f"{pdb}_{ch}", seq)]
    for m in members:
        s = seqs[m].rstrip("*")
        if region: s = s[region[0] - 1:region[1]]
        recs.append((m, s))
    al = mafft(recs); ref = al[f"{pdb}_{ch}"]
    colmap, c = {}, 0
    for i, chh in enumerate(ref):
        if chh != "-": c += 1; colmap[c] = i
    rows = []
    for idx, (lab, d) in sorted(hit.items()):
        col = colmap[idx]; row = dict(ref_res=lab, min_dist=d)
        for m in members:
            sa = al[m]; row[m] = sa[col] if sa[col] != "-" else "-"
        row["n_identical"] = sum(row[m] == lab[0] for m in members); rows.append(row)
    R = pd.DataFrame(rows); R.to_csv(f"data/contacts_{pdb}_{tag}.tsv", sep="\t", index=False)
    print(f"\n##### {pdb} chain {ch} ligands {ligs}: {len(R)} contact residues; identical in all members: {int((R.n_identical == len(members)).sum())};"
          f" identical in >= half: {int((R.n_identical >= len(members) / 2).sum())}")
    print(R.to_string(index=False))
L1 = ["HII14378.1", "MBX4196761.1", "HLC90278.1", "MEK6912472.1", "JAFCFS010000026.1_6"]
L2 = ["MBT7106118.1", "MBN2421503.1", "MDP3698733.1", "HLC74554.1", "MBI5065264.1", "MBT5274800.1"]
run("1JYL", "A", {"CDC", "CTP", "CDP", "PC", "MG"}, L1, "L1_NTase", region=(1, 260))
run("4R78", "A", {"AMP", "ADP", "ANP", "MG", "CHT", "PC"}, L1, "L1_kinase", region=(200, 560))
run("1E2D", "A", {"TMP", "ADP", "MG"}, L2, "L2_Tmk", region=(200, 460))
run("4TMK", "A", {"T5A", "TP5", "MG"}, L2, "L2_Tmk_Ecoli", region=(200, 460))
# pol-beta superfamily NTase signature in the N-terminal (NTP_transf_5) part of L2: G[GS] followed by [DE].[DE] 7-15 residues later
rx = re.compile(r"G[GS].{7,15}[DE].[DE]")
for m in L2:
    s = seqs[m].rstrip("*")[:200]
    print("polB-NTase signature", m, [(x.start() + 1, x.group()) for x in rx.finditer(s)])
