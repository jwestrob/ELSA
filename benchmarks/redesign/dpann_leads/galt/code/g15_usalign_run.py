#!/usr/bin/env python
"""galt step 15: structure-based alignment of DPANN models (ESMFold2 fast, single chain) onto experimental /
AlphaFold structures of characterized members of each branch; run on the biotite login node (serial, single-threaded,
code/s04_usalign_remote.py copied from the invisible workstream).
References (chain A extracted, protein ATOM records only):
  1GUP_A E. coli GalT (P09148) | 5IN3_A human GALT (P07902) | 6K9Z_A Pyrobaculum GalT (Q8ZXN7) | AF-E8MF11 Bifidobacterium
  GalT class II | 1Z84_A Arabidopsis ADPGP (Q9FK51) | 4I5V_A yeast Apa2 (P22108) | AF-Q6ZNW5 human GDPGP1 | AF-Q8RWE8 VTC2
Models: folds listed in data/q2_models.tsv (protein_id, path, class).
Also every reference against every reference (to place reference residues in the E. coli GalT frame).
Writes data/usalign_q2.tsv (pair_id, model, ref, TM1, TM2, RMSD, Lali, alignment lines)."""
import subprocess
from pathlib import Path
import pandas as pd

H = Path("/Users/jacob/Documents/Sandbox/elsa_test/ELSA/benchmarks/redesign/dpann_leads/galt")
RS = "/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/dpann_leads/galt"
SSH = ["ssh", "jwestrob@biotite.berkeley.edu"]
REFS = {"1GUP_A": "ref/pdb/1GUP.pdb", "5IN3_A": "ref/pdb/5IN3.pdb", "6K9Z_A": "ref/pdb/6K9Z.pdb", "1Z84_A": "ref/pdb/1Z84.pdb",
        "4I5V_A": "ref/pdb/4I5V.pdb", "AF-E8MF11": "ref/pdb/AF-E8MF11.pdb", "AF-Q6ZNW5": "ref/pdb/AF-Q6ZNW5.pdb",
        "AF-Q8RWE8": "ref/pdb/AF-Q8RWE8.pdb"}
(H / "ref/chains").mkdir(exist_ok=True)
for k, f in REFS.items():
    out = []
    for l in open(H / f):
        if l.startswith("ENDMDL"):
            break
        if l.startswith("ATOM") and (l[21] == "A" or k.startswith("AF-")) and l[16] in " A":
            out.append(l[:16] + " " + l[17:])
    (H / f"ref/chains/{k}.pdb").write_text("".join(out) + "END\n")
M = pd.read_csv(H / "data/q2_models.tsv", sep="\t")
rows = []
for r in M.itertuples():
    for k in REFS:
        rows.append(dict(pair_id=f"{r.protein_id}|{k}", model=f"{RS}/q2/models/{Path(r.path).name}", ref=f"{RS}/q2/refs/{k}.pdb"))
for a in REFS:
    for b in REFS:
        if a != b:
            rows.append(dict(pair_id=f"REF:{a}|{b}", model=f"{RS}/q2/refs/{a}.pdb", ref=f"{RS}/q2/refs/{b}.pdb"))
P = pd.DataFrame(rows)
P.to_csv(H / "data/q2_pairs.tsv", sep="\t", index=False)
subprocess.run(SSH + [f"mkdir -p {RS}/q2/models {RS}/q2/refs {RS}/code"], check=True)
lst = H / "data/q2_model_paths.txt"
lst.write_text("\n".join(sorted(set(M.path))) + "\n")
subprocess.run(["rsync", "-a", "--files-from", str(lst), "--no-relative", "/", f"jwestrob@biotite.berkeley.edu:{RS}/q2/models/"], check=True)
subprocess.run(["rsync", "-a"] + [str(H / f"ref/chains/{k}.pdb") for k in REFS] + [f"jwestrob@biotite.berkeley.edu:{RS}/q2/refs/"], check=True)
subprocess.run(["rsync", "-a", str(H / "code/s04_usalign_remote.py"), str(H / "data/q2_pairs.tsv"), f"jwestrob@biotite.berkeley.edu:{RS}/code/"], check=True)
subprocess.run(SSH + [f"cd {RS} && rm -f usalign_q2_raw.tsv && python3 code/s04_usalign_remote.py code/q2_pairs.tsv usalign_q2_raw.tsv"], check=True)
subprocess.run(["rsync", "-a", f"jwestrob@biotite.berkeley.edu:{RS}/usalign_q2_raw.tsv", str(H / "data/usalign_q2.tsv")], check=True)
U = pd.read_csv(H / "data/usalign_q2.tsv", sep="\t", keep_default_na=False)
print(len(P), "pairs;", (U.TM2 != "").sum(), "scored")
