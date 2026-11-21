#!/usr/bin/env python
"""Local driver for step 4: build the US-align pair list from data/selection_<mode>.tsv, copy models and reference
chains to the remote scratch, run code/s04_usalign_remote.py on the biotite login node (serial) and fetch the result.
Pairs: candidate -> own family reference ('own') and negative-A reference ('negA'); positive -> own; negB -> own.
usage: L5_usalign.py COLLECTION MODE   (copy of dpann_leads/invisible/code/s04_run_usalign.py, paths only)   writes data/usalign_<mode>.tsv (pairs joined with scores)."""
import subprocess, sys
from pathlib import Path
import pandas as pd

import json
C, MODE = sys.argv[1], sys.argv[2]
_CFG = json.load(open(Path(__file__).resolve().parent / "collections_leads.json"))[C]
H = Path(_CFG["local_out"].replace("R/", "/Users/jacob/Documents/Sandbox/elsa_test/ELSA/"))
# srvp: US-align output carries aligned SR-VP sequences -> kept under the git-ignored fold folder (UA), not data/
UA = (Path(_CFG["fold_dir"]) if _CFG["fold_dir"] else H / "folds") / "usalign"; UA.mkdir(parents=True, exist_ok=True)
RS = _CFG["remote"].replace("RD/", "/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/") + "/usalign"
SSH = ["ssh", "jwestrob@biotite.berkeley.edu"]
S = pd.read_csv(H / f"data/selection_{MODE}.tsv", sep="\t")
REFS = pd.read_csv(H / "data/references_testable.tsv", sep="\t").set_index("func")
S = S[S.fold_path.apply(lambda p: Path(p).exists())]
rows = []
for r in S.itertuples():
    model = f"{RS}/models/{Path(r.fold_path).name}"
    rows.append(dict(pair_id=f"{r.protein_id}|{r.func}|{r.role}|own", protein_id=r.protein_id, func=r.func, role=r.role, pair="own",
                     ref_func=r.func, model=model, ref=f"{RS}/refs/{Path(REFS.loc[r.func, 'ref_file']).name}"))
    if r.role == "candidate":
        nf = REFS.loc[r.func, "negA_func"]
        rows.append(dict(pair_id=f"{r.protein_id}|{r.func}|{r.role}|negA", protein_id=r.protein_id, func=r.func, role=r.role, pair="negA",
                         ref_func=nf, model=model, ref=f"{RS}/refs/{Path(REFS.loc[nf, 'ref_file']).name}"))
P = pd.DataFrame(rows).drop_duplicates("pair_id")
P.to_csv(H / f"data/pairs_{MODE}.tsv", sep="\t", index=False)
subprocess.run(SSH + [f"mkdir -p {RS}/models {RS}/refs {RS}/code"], check=True)
lst = H / f"data/models_{MODE}.txt"
lst.write_text("\n".join(sorted(set(S.fold_path))) + "\n")
subprocess.run(["rsync", "-a", "--files-from", str(lst), "--no-relative", "/", f"jwestrob@biotite.berkeley.edu:{RS}/models/"], check=True)
subprocess.run(["rsync", "-a"] + [str(H / f) for f in REFS.ref_file] + [f"jwestrob@biotite.berkeley.edu:{RS}/refs/"], check=True)
subprocess.run(["rsync", "-a", str(H / "code/s04_usalign_remote.py"), str(H / f"data/pairs_{MODE}.tsv"),
                f"jwestrob@biotite.berkeley.edu:{RS}/code/"], check=True)
subprocess.run(SSH + [f"cd {RS} && python3 code/s04_usalign_remote.py code/pairs_{MODE}.tsv usalign_raw_{MODE}.tsv"], check=True)
subprocess.run(["rsync", "-a", f"jwestrob@biotite.berkeley.edu:{RS}/usalign_raw_{MODE}.tsv", str(UA) + "/"], check=True)
U = pd.read_csv(UA / f"usalign_raw_{MODE}.tsv", sep="\t", keep_default_na=False)
P.merge(U, on="pair_id", how="left").to_csv(UA / f"usalign_{MODE}.tsv", sep="\t", index=False)
P.merge(U.drop(columns=["aln_model", "aln_marks", "aln_ref"], errors="ignore"), on="pair_id", how="left").to_csv(
    H / f"data/usalign_scores_{MODE}.tsv", sep="\t", index=False)          # scores only, no sequence
print(len(P), "pairs;", (U.TM2 != "").sum(), "scored")
