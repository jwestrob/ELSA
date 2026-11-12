"""Task 3, superposition (local, seconds): C of every ABC model (single sequence, slots 1-12) superposed onto the
PDB entries matched by Foldseek / HHsearch in the deep dive, with US-align (run on biotite: work/usal/*.out, *.mat).

Templates (chain A) and their catalytic dyad (UniProt 'Active site' features mapped to author numbering):
  8tl0 SAVED-CHAT (D0LTI2; no UniProt active-site annotation; aligned residues reported)
  1cvr gingipain R (RgpB, P95493 H440/C473 = author H211/C244; D-Phe-Phe-Arg-CMK bonded to C244 SG)
  6l4y asparaginyl endopeptidase (A0A0P0QM28 H180/C222; C222 oxidised, OCS)
  6nry caspase-4 (P49662 H210/C258; C258A)
  8dgz caspase-7 (P55210 H144/C186; Ac-DEVD-aldehyde bonded to C186 SG)
  4jr2 caspase-7 (H144/C186; Ac-DEVD-CMK bonded to C186 SG)
  4m9r CED-3 (P42573 H315/C358; C358S)
Per model x template: TM-score normalised by C, RMSD, aligned length; template residues aligned to C's His and
Cys; after applying the US-align transform to the whole A-B-C model: His CB - template His CB, His ring centroid -
template His ring centroid, Cys CB - template dyad CB (Cys/Ala/Ser), Cys SG - template SG.
Inhibitor templates (8dgz, 4jr2, 1cvr): distance from the warhead atom (bonded to the catalytic SG) to B's column-
48 residue backbone C and CA; nearest B residue to every inhibitor residue; cosine between the B strand vector
(CA of the column-48 residue minus CA of the residue 4 before it) and the inhibitor vector (N-terminal end to
warhead atom).
Writes work/struct/superpose.tsv and superpose_summary.json.
"""
import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.PDB import MMCIFParser, PDBParser
from Bio.SeqUtils import seq1

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent.parent
W = HERE / "work"
DD = HERE.parent / "deepdive" / "work"
DYAD = {"8tl0_A": None, "1cvr_A": (211, 244), "6l4y_A": (180, 222), "6nry_A": (210, 258), "8dgz_A": (144, 186),
        "4jr2_A": (144, 186), "4m9r_A": (315, 358)}
INH = {"8dgz_A": ("C", None), "4jr2_A": ("C", None), "1cvr_A": ("A", "H_H37")}
RING = ("CG", "ND1", "CD2", "CE1", "NE2")


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


def parse_out(p):
    txt = open(p).read()
    tm1 = float(re.search(r"TM-score= ([\d.]+) \(normalized by length of Structure_1", txt).group(1))
    m = re.search(r"Aligned length=\s*(\d+), RMSD=\s*([\d.]+)", txt)
    lines = txt.rstrip().split("\n")
    i = [k for k, l in enumerate(lines) if l.startswith('(":" denotes')][0]
    return tm1, int(m.group(1)), float(m.group(2)), lines[i + 1], lines[i + 2], lines[i + 3]


def parse_mat(p):
    rows = [l.split() for l in open(p) if re.match(r"^[012]\s", l)]
    t = np.array([float(r[1]) for r in rows])
    U = np.array([[float(x) for x in r[2:5]] for r in rows])
    return t, U


def main():
    sel = pd.read_csv(DD / "c1_selected.tsv", sep="\t")
    sm = pd.read_csv(W / "struct" / "struct_models.tsv", sep="\t")
    sm = sm[sm.kind == "abc"].set_index("slot")
    baln = rf(DD / "aln" / "c1B_inst.aln.fa")
    PP, MP = PDBParser(QUIET=True), MMCIFParser(QUIET=True)
    tmpl = {t: PP.get_structure(t, W / "tmpl" / f"{t}.pdb")[0]["A"] for t in DYAD}
    inh = {}
    for t, (chid, het) in INH.items():
        s = MP.get_structure(t, W / "tmpl_raw" / f"{t[:4]}.cif")[0]
        sg = s["A"][DYAD[t][1]]["SG"]
        if het:
            atoms = [a for r in s[chid] if r.id[0] == het for a in r]
            resatoms = {f"{het[2:]}": atoms}
        else:
            resatoms = {f"{r.get_resname()}{r.id[1]}": [a for a in r] for r in s[chid]}
            atoms = [a for v in resatoms.values() for a in v]
        war = min(atoms, key=lambda a: np.linalg.norm(a.coord - sg.coord))
        far = max(atoms, key=lambda a: np.linalg.norm(a.coord - war.coord))
        inh[t] = dict(atoms=atoms, resatoms=resatoms, war=war, far=far, sg_war=float(np.linalg.norm(sg.coord - war.coord)))
    rows = []
    for _, s in sel.iterrows():
        slot = int(s.slot)
        model = PP.get_structure("m", DD / "folds_c1abc" / "pdb" / f"c1_abc_{slot:02d}.pdb")[0]
        bmap, k = {}, 0
        for c, a in enumerate(baln[s.instance_id]):
            if a != "-":
                k += 1
                bmap[c] = k
        g48 = bmap.get(48)
        hnum = sm.at[slot, "his"]
        cnum = sm.at[slot, "cys"]
        for t, dy in DYAD.items():
            f = W / "usal" / f"C_abc_{slot:02d}__{t}"
            tm1, alen, rmsd, a1, mk, a2 = parse_out(f.with_suffix(".out"))
            tr, U = parse_mat(f.with_suffix(".mat"))
            X = lambda v: tr + U @ v  # noqa: E731
            cres = [r for r in model["C"]]
            tres = [r for r in tmpl[t] if "CA" in r]
            amap, i1, i2 = {}, 0, 0
            for x, y, z in zip(a1, mk, a2):
                if x != "-" and y != " " and z != "-":
                    amap[cres[i1].id[1]] = (tres[i2].id[1], tres[i2].get_resname(), y)
                if x != "-":
                    i1 += 1
                if z != "-":
                    i2 += 1
            r = dict(slot=slot, template=t, TM_C=tm1, aligned=alen, rmsd=rmsd)
            for nm, num in (("his", hnum), ("cys", cnum)):
                if pd.isna(num):
                    r[f"{nm}_aligned_to"] = None
                    continue
                al = amap.get(int(num))
                r[f"{nm}_aligned_to"] = f"{seq1(al[1])}{al[0]}{'(d<5)' if al[2] == ':' else ''}" if al else "unaligned"
            if dy:
                th, tc = tmpl[t][dy[0]], tmpl[t][dy[1]]
                if not pd.isna(hnum):
                    H = model["C"][int(hnum)]
                    r["his_CB_d"] = round(float(np.linalg.norm(X(H["CB"].coord) - th["CB"].coord)), 2)
                    r["his_ring_d"] = round(float(np.linalg.norm(np.mean([X(H[a].coord) for a in RING], 0)
                                                                 - np.mean([th[a].coord for a in RING], 0))), 2)
                if not pd.isna(cnum):
                    Cy = model["C"][int(cnum)]
                    r["cys_CB_d"] = round(float(np.linalg.norm(X(Cy["CB"].coord) - tc["CB"].coord)), 2)
                    tsg = tc["SG"] if "SG" in tc else (tc["OG"] if "OG" in tc else None)
                    r["cys_SG_d"] = round(float(np.linalg.norm(X(Cy["SG"].coord) - tsg.coord)), 2) if tsg else None
                    r["tmpl_dyad_aa"] = f"{seq1(th.get_resname())}{dy[0]}/{seq1(tc.get_resname())}{dy[1]}"
            if t in inh and g48:
                I = inh[t]
                G = model["B"][g48]
                r["Bcol48"] = f"{seq1(G.get_resname())}{g48}"
                r["war_to_B48_C"] = round(float(np.linalg.norm(X(G["C"].coord) - I["war"].coord)), 2)
                r["war_to_B48_CA"] = round(float(np.linalg.norm(X(G["CA"].coord) - I["war"].coord)), 2)
                bca = {rr.id[1]: X(rr["CA"].coord) for rr in model["B"]}
                near = []
                for nm_, ats in I["resatoms"].items():
                    if nm_.startswith("ACE") or nm_.startswith("HOH"):
                        continue
                    best = min(((min(np.linalg.norm(v - a.coord) for a in ats), kk) for kk, v in bca.items()))
                    near.append(f"{nm_}:B{best[1]}({best[0]:.1f})")
                r["inhibitor_nearest_B"] = ";".join(near)
                # occupancy: fraction of inhibitor heavy atoms within 3 A of any transformed B heavy atom
                batoms = np.array([X(a.coord) for rr in model["B"] for a in rr if a.element != "H"])
                ia = np.array([a.coord for a in I["atoms"] if a.element != "H"])
                dmin = np.sqrt(((ia[:, None] - batoms[None]) ** 2).sum(-1)).min(1)
                r["inh_atoms_within_2A_of_B"] = round(float((dmin <= 2.0).mean()), 2)
                r["inh_atoms_within_3A_of_B"] = round(float((dmin <= 3.0).mean()), 2)
                pre = g48 - 4
                if pre in bca:
                    vb = bca[g48] - bca[pre]
                    vi = I["war"].coord - I["far"].coord
                    r["cos_Bstrand_vs_inhibitor"] = round(float(vb @ vi / np.linalg.norm(vb) / np.linalg.norm(vi)), 2)
            rows.append(r)
    D = pd.DataFrame(rows)
    D.to_csv(W / "struct" / "superpose.tsv", sep="\t", index=False)
    ok = D[D.slot != 4]
    S = {}
    for t, x in ok.groupby("template"):
        d = dict(TM_C=[round(x.TM_C.min(), 3), round(x.TM_C.median(), 3), round(x.TM_C.max(), 3)],
                 his_aligned_to=x.his_aligned_to.value_counts().to_dict(),
                 cys_aligned_to=x.cys_aligned_to.value_counts().to_dict())
        for c in ("his_CB_d", "his_ring_d", "cys_CB_d", "cys_SG_d", "war_to_B48_C", "war_to_B48_CA",
                  "inh_atoms_within_2A_of_B", "inh_atoms_within_3A_of_B", "cos_Bstrand_vs_inhibitor"):
            if c in x and x[c].notna().any():
                d[c] = [round(float(x[c].min()), 2), round(float(x[c].median()), 2), round(float(x[c].max()), 2)]
        if "inhibitor_nearest_B" in x:
            d["inhibitor_nearest_B_examples"] = x.inhibitor_nearest_B.dropna().head(3).tolist()
        S[t] = d
    S["inhibitor_SG_warhead_dist"] = {t: round(v["sg_war"], 2) for t, v in inh.items()}
    json.dump(S, open(W / "struct" / "superpose_summary.json", "w"), indent=1, default=str)
    print(json.dumps(S, indent=1, default=str))
    pd.set_option("display.width", 300)
    pd.set_option("display.max_colwidth", 90)
    print(D[D.template.isin(["8dgz_A", "1cvr_A", "8tl0_A"])].drop(columns=["inhibitor_nearest_B"], errors="ignore").to_string())


if __name__ == "__main__":
    main()
