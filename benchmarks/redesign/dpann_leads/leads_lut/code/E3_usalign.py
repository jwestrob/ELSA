#!/usr/bin/env python
"""Extended tier step 3 (local driver; the comparison itself is code/s04_usalign_remote.py, unchanged, on the biotite login
node). Pairs from data/ext/selection_ext.tsv: candidate -> own reference ('own') and negative-A reference ('negA'); positive
and negB -> own; every model of a family with an experimental chain -> that chain ('exp', fold comparison only).
usage: E3_usalign.py COLLECTION     writes data/ext/usalign_ext.tsv"""
import json, subprocess, sys
from pathlib import Path
import pandas as pd

C = sys.argv[1]
RD = "/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111"
CODE = Path(__file__).resolve().parent
CFG = json.load(open(CODE / "collections_leads.json"))[C]
H = Path(CFG["local_out"].replace("R/", "/Users/jacob/Documents/Sandbox/elsa_test/ELSA/"))
E = H / "data/ext"
RS = CFG["remote"].replace("RD/", RD + "/") + "/usalign_ext"
B = "jwestrob@biotite.berkeley.edu"
S = pd.read_csv(E / "selection_ext.tsv", sep="\t", dtype={"protein_id": str})
RX = pd.read_csv(E / "references_ext.tsv", sep="\t", keep_default_na=False).set_index("func")
S = S[S.fold_path.apply(lambda p: Path(p).exists())]
rows = []
for r in S.itertuples():
    model = f"{RS}/models/{Path(r.fold_path).name}"
    rows.append(dict(pair_id=f"{r.protein_id}|{r.func}|{r.role}|own", protein_id=r.protein_id, func=r.func, role=r.role, pair="own",
                     ref_func=r.func, model=model, ref=f"{RS}/refs/{Path(RX.loc[r.func, 'ref_file']).name}"))
    if r.role == "candidate" and RX.loc[r.func, "negA_func"]:
        nf = RX.loc[r.func, "negA_func"]
        rows.append(dict(pair_id=f"{r.protein_id}|{r.func}|{r.role}|negA", protein_id=r.protein_id, func=r.func, role=r.role, pair="negA",
                         ref_func=nf, model=model, ref=f"{RS}/refs/{Path(RX.loc[nf, 'ref_file']).name}"))
    if RX.loc[r.func, "exp_file"]:
        rows.append(dict(pair_id=f"{r.protein_id}|{r.func}|{r.role}|exp", protein_id=r.protein_id, func=r.func, role=r.role, pair="exp",
                         ref_func=r.func, model=model, ref=f"{RS}/refs/{Path(RX.loc[r.func, 'exp_file']).name}"))
P = pd.DataFrame(rows).drop_duplicates("pair_id")
P.to_csv(E / "pairs_ext.tsv", sep="\t", index=False)
subprocess.run(["ssh", B, f"mkdir -p {RS}/models {RS}/refs {RS}/code"], check=True)
lst = E / "models_ext.txt"
lst.write_text("\n".join(sorted(set(S.fold_path))) + "\n")
subprocess.run(["rsync", "-a", "--files-from", str(lst), "--no-relative", "/", f"{B}:{RS}/models/"], check=True)
refs = sorted(set(RX.ref_file) | {x for x in RX.exp_file if x})
subprocess.run(["rsync", "-a"] + refs + [f"{B}:{RS}/refs/"], check=True)
subprocess.run(["rsync", "-a", str(CODE / "s04_usalign_remote.py"), str(E / "pairs_ext.tsv"), f"{B}:{RS}/code/"], check=True)
subprocess.run(["ssh", B, f"cd {RS} && python3 code/s04_usalign_remote.py code/pairs_ext.tsv usalign_raw_ext.tsv"], check=True)
subprocess.run(["rsync", "-a", f"{B}:{RS}/usalign_raw_ext.tsv", str(E) + "/"], check=True)
U = pd.read_csv(E / "usalign_raw_ext.tsv", sep="\t", keep_default_na=False)
P.merge(U, on="pair_id", how="left").to_csv(E / "usalign_ext.tsv", sep="\t", index=False)
print(len(P), "pairs;", (U.TM2 != "").sum(), "scored")
