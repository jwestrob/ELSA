#!/usr/bin/env python
"""galt step 12: ligand-contact residues in experimental structures of characterized GalT-like enzymes (chain A).
  1GUP  E. coli GalT (P09148) + UDP-galactose (GDU)            Thoden 1997
  6K9Z  Pyrobaculum aerophilum GalT H140F (Q8ZXN7) + UDP       Ohshida 2020 (PMID 31693208)
  1Z84  Arabidopsis ADP-glucose phosphorylase (Q9FK51) + AMP   McCoy 2006 (PMID 16519510)
  4I5V  yeast Apa2 Ap4A phosphorylase (P22108) + Ap4A (B4P)    Hou 2013 (PMID 23628156)
Residue in contact = any heavy atom within 4.0 A of a moiety atom (base, ribose, phosphate, hexose), any chain
(GalT pockets are completed by the other subunit; the chain is recorded).
Writes data/ref_contacts.tsv (structure, ligand, moiety, chain, resi, resn)."""
from pathlib import Path
import numpy as np
import pandas as pd

H = Path("/Users/jacob/Documents/Sandbox/elsa_test/ELSA/benchmarks/redesign/dpann_leads/galt")
MO = {
    "GDU": {"base": "N1 C2 N3 C4 C5 C6 O2 O4", "ribose": "C1D C2D O2D C3D C4D O4D O3D C5D",
            "phosphate": "O5D PA O1A O2A O3A PB O1B O2B O3B", "hexose": "C1' C2' C3' C4' C5' C6' O2' O3' O4' O5' O6'"},
    "UDP": {"base": "N1 C2 N3 C4 C5 C6 O2 O4", "ribose": "C1' C2' O2' C3' C4' O4' O3' C5'",
            "phosphate": "O5' PA O1A O2A O3A PB O1B O2B O3B"},
    "AMP": {"base": "N9 C8 N7 C5 C6 N6 N1 C2 N3 C4", "ribose": "C1' C2' O2' C3' C4' O4' O3' C5'", "phosphate": "O5' P O1P O2P"},
    "B4P": {"base": "N9A C8A N7A C5A C6A N6A N1A C2A N3A C4A N9B C8B N7B C5B C6B N6B N1B C2B N3B C4B",
            "ribose": "C5E C4E O4E C3E O3E C2E O2E C1E C5F C4F O4F C3F O3F C2F O2F C1F",
            "phosphate": "PA O1A O2A O3A PB O1B O2B O3B PG O1G O2G O3G PD O1D O2D O5E O5F"},
}
SETS = [("1GUP", "GDU", "A"), ("6K9Z", "UDP", "A"), ("1Z84", "AMP", "A"), ("4I5V", "B4P", "A")]
rows = []
for pdb, lig, lch in SETS:
    prot, L = [], []
    for l in open(H / f"ref/pdb/{pdb}.pdb"):
        if l.startswith("ENDMDL"):
            break
        if l.startswith("ATOM") and l[76:78].strip() != "H":
            prot.append((l[21], int(l[22:26]), l[17:20], float(l[30:38]), float(l[38:46]), float(l[46:54])))
        elif l.startswith("HETATM") and l[17:20] == lig and l[21] == lch:
            L.append((l[12:16].strip(), float(l[30:38]), float(l[38:46]), float(l[46:54])))
    P = pd.DataFrame(prot, columns=["chain", "resi", "resn", "x", "y", "z"])
    Lg = pd.DataFrame(L, columns=["name", "x", "y", "z"])
    xyz = P[["x", "y", "z"]].to_numpy()
    for mo, names in MO[lig].items():
        m = Lg[Lg.name.isin(names.split())][["x", "y", "z"]].to_numpy()
        if not len(m):
            continue
        d = np.linalg.norm(xyz[:, None, :] - m[None, :, :], axis=2).min(axis=1)
        hit = P[d <= 4.0].drop_duplicates(["chain", "resi"])
        for r in hit.itertuples():
            rows.append(dict(structure=pdb, ligand=lig, moiety=mo, chain=r.chain, resi=r.resi, resn=r.resn))
T = pd.DataFrame(rows)
T.to_csv(H / "data/ref_contacts.tsv", sep="\t", index=False)
for (s, mo), g in T.groupby(["structure", "moiety"], sort=False):
    print(s, mo, " ".join(f"{r.resn}{r.resi}{'' if r.chain == 'A' else '(' + r.chain + ')'}" for r in g.sort_values("resi").itertuples()))
