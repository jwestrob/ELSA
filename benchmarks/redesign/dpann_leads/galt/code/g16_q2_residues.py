#!/usr/bin/env python
"""galt step 16 (question 2): substrate- and reaction-determining positions by structure-based alignment.
Positions = residues contacting a ligand moiety in an experimental structure of a characterized enzyme (g12:
1GUP E. coli GalT + UDP-galactose, 1Z84 Arabidopsis ADPGP + AMP, 4I5V yeast Apa2 + Ap4A, 6K9Z Pyrobaculum GalT + UDP)
plus the HIT motif (E. coli H164 / H166 / Q168). Each position is anchored in the structure that defines it; residues of
every other reference and of every DPANN model come from US-align (TM-align algorithm) alignments to that structure
(g15). A residue counts only when the aligned pair is within 5 A after superposition (US-align ':'), else '-'.
Defining structure residue numbers are mapped to E. coli GalT numbering through the 1Z84/4I5V/6K9Z -> 1GUP alignment.
Group consensus: residue counts across members (e.g. F6 L1). Writes data/q2_positions.tsv (long) and
data/q2_table.tsv (one row per position)."""
from collections import Counter
from pathlib import Path
import pandas as pd

H = Path("/Users/jacob/Documents/Sandbox/elsa_test/ELSA/benchmarks/redesign/dpann_leads/galt")
U = pd.read_csv(H / "data/usalign_q2.tsv", sep="\t", keep_default_na=False).set_index("pair_id")
M = pd.read_csv(H / "data/q2_models.tsv", sep="\t")
RC = pd.read_csv(H / "data/ref_contacts.tsv", sep="\t")
C = pd.read_csv(H / "data/census_proteins.tsv", sep="\t").set_index("protein_id")
AA = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
      "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
      "MSE": "M"}


def resnums(path):
    out, seen = [], set()
    for l in open(path):
        if l.startswith("ATOM"):
            k = (l[22:27])
            if k not in seen:
                seen.add(k); out.append((int(l[22:26]), AA.get(l[17:20], "X")))
    return out


REFNUM = {k: resnums(H / f"ref/chains/{k}.pdb") for k in ["1GUP_A", "5IN3_A", "6K9Z_A", "1Z84_A", "4I5V_A", "AF-E8MF11", "AF-Q6ZNW5", "AF-Q8RWE8"]}
MODNUM = {r.protein_id: resnums(r.path) for r in M.itertuples()}


def mapping(pair, qnum, rnum):
    """reference residue number -> (query residue, mark) for one US-align alignment (query = Structure_1)."""
    r = U.loc[pair]
    a, mk, b = r.aln_model, r.aln_marks, r.aln_ref
    i = j = 0
    out = {}
    for x, m, y in zip(a, mk, b):
        if x != "-" and y != "-":
            out[rnum[j][0]] = (x, m, qnum[i][0])
        if x != "-":
            i += 1
        if y != "-":
            j += 1
    return out


# positions
pos = []
for s, chain_map in [("1GUP_A", "1GUP"), ("1Z84_A", "1Z84"), ("4I5V_A", "4I5V"), ("6K9Z_A", "6K9Z")]:
    sub = RC[RC.structure == chain_map]
    for (resi, resn), g in sub.groupby(["resi", "resn"]):
        pos.append(dict(anchor=s, resi=int(resi), resn=AA[resn], moieties="+".join(sorted(set(g.moiety))),
                        other_chain="yes" if (g.chain != "A").any() else ""))
for resi, nm in [(164, "H"), (166, "H"), (168, "Q")]:
    pos.append(dict(anchor="1GUP_A", resi=resi, resn=nm, moieties="HIT_motif", other_chain=""))
P = pd.DataFrame(pos).drop_duplicates(["anchor", "resi"])
REFS = ["1GUP_A", "5IN3_A", "6K9Z_A", "AF-E8MF11", "1Z84_A", "4I5V_A", "AF-Q6ZNW5", "AF-Q8RWE8"]
LBL = {"1GUP_A": "GalT_Ecoli", "5IN3_A": "GalT_human", "6K9Z_A": "GalT_Pyrobac", "AF-E8MF11": "GalTII_Bifido",
       "1Z84_A": "ADPGP_Athal", "4I5V_A": "Apa2_yeast", "AF-Q6ZNW5": "GDPGP1_human", "AF-Q8RWE8": "VTC2_Athal"}
rows = []
for p in P.itertuples():
    rec = dict(anchor=p.anchor, anchor_res=f"{p.resn}{p.resi}", moieties=p.moieties, other_chain=p.other_chain)
    # E. coli number
    if p.anchor == "1GUP_A":
        rec["Ecoli_pos"] = p.resi
    else:
        mp = mapping(f"REF:1GUP_A|{p.anchor}", REFNUM["1GUP_A"], REFNUM[p.anchor])
        x = mp.get(p.resi)
        rec["Ecoli_pos"] = x[2] if x and x[1] == ":" else ""
    for k in REFS:
        if k == p.anchor:
            rec[LBL[k]] = f"{p.resn}{p.resi}"
            continue
        mp = mapping(f"REF:{k}|{p.anchor}", REFNUM[k], REFNUM[p.anchor])
        x = mp.get(p.resi)
        rec[LBL[k]] = f"{x[0]}{x[2]}" if x and x[1] == ":" else "-"
    for m in M.itertuples():
        mp = mapping(f"{m.protein_id}|{p.anchor}", MODNUM[m.protein_id], REFNUM[p.anchor])
        x = mp.get(p.resi)
        rec[m.protein_id] = f"{x[0]}{x[2]}" if x and x[1] == ":" else "-"
    rows.append(rec)
T = pd.DataFrame(rows)
T.to_csv(H / "data/q2_positions_wide.tsv", sep="\t", index=False)


def group_of(pid):
    o, g = C.order.get(pid, ""), C.genus.get(pid, "")
    cls = M.set_index("protein_id").cls[pid]
    if cls == "lead15":
        if o == "QMZS01":
            return "lead15_QMZS01"
        return f"lead15_Woese_{g}" if g in ("SM23-78", "GCA-2686295") else "lead15_Woese_other"
    if cls == "labeled_named":
        return "labeled_named_" + ("Woese" if o == "Woesearchaeales" else ("QMZS01" if o == "QMZS01" else "EX4484/CG10238/Aenig"))
    return "named_form"


M["grp"] = [group_of(p) for p in M.protein_id]
G = ["lead15_Woese_SM23-78", "lead15_Woese_GCA-2686295", "lead15_Woese_other", "lead15_QMZS01", "labeled_named_Woese",
     "labeled_named_QMZS01", "labeled_named_EX4484/CG10238/Aenig", "named_form"]
out = []
for r in T.itertuples():
    d = r._asdict()
    rec = {k: d[k] for k in ["anchor", "anchor_res", "moieties", "other_chain", "Ecoli_pos"] + list(LBL.values())}
    for g in G:
        ids = M.protein_id[M.grp == g]
        cnt = Counter(T.loc[r.Index, list(ids)].str[0])
        rec[f"{g} (n={len(ids)})"] = " ".join(f"{a}{n}" for a, n in cnt.most_common())
    out.append(rec)
O = pd.DataFrame(out)
O = O.sort_values(["Ecoli_pos", "anchor"], key=lambda s: pd.to_numeric(s, errors="coerce")).reset_index(drop=True)
O.to_csv(H / "data/q2_table.tsv", sep="\t", index=False)
pd.set_option("display.width", 320); pd.set_option("display.max_colwidth", 30); pd.set_option("display.max_rows", 200)
print(O.to_string(index=False))
