#!/usr/bin/env python
"""Extended tier step 3 (local driver; rule assay_rule_extended.json): US-align pairs on the biotite login node with the
unchanged code/s04_usalign_remote.py (serial). Pairs: candidate -> own trimmed reference member ('own') and the negative-A
family's reference member ('negA'); positive (held-out member) -> own; negB -> own; candidates and positives also -> the
family's experimental chain ('exp', reported only, never used for PASS) where E1 found one (ref/ext_<KO>_exp.pdb).
Models and trimmed reference members go to the remote scratch RS/usalign_ext/ (biotite only; SR-VP structures stay off the
repository). usage: E3_usalign.py COLLECTION   writes FOLD_DIR/usalign/ext_usalign.tsv (alignments) and data/ext/usalign_scores.tsv"""
import json, subprocess, sys
from pathlib import Path
import pandas as pd

C = sys.argv[1]
CFG = json.load(open(Path(__file__).resolve().parent / "collections_leads.json"))[C]
H = Path(CFG["local_out"].replace("R/", "/Users/jacob/Documents/Sandbox/elsa_test/ELSA/"))
RS = CFG["remote"].replace("RD/", "/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/") + "/usalign_ext"
B = "jwestrob@biotite.berkeley.edu"
E = H / "data/ext"
UA = (Path(CFG["fold_dir"]) if CFG["fold_dir"] else H / "folds") / "usalign"; UA.mkdir(parents=True, exist_ok=True)
S = pd.read_csv(E / "selection.tsv", sep="\t", dtype={"protein_id": str})
REFS = pd.read_csv(E / "references_ext.tsv", sep="\t").set_index("func")
S = S[S.fold_path.apply(lambda p: Path(p).exists())]
rows = []
for r in S.itertuples():
    model = f"{RS}/models/{Path(r.fold_path).name}"
    rows.append(dict(pair_id=f"{r.protein_id}|{r.func}|{r.role}|own", protein_id=r.protein_id, func=r.func, role=r.role, pair="own",
                     ref_func=r.func, model=model, ref=f"{RS}/refs/{r.func}_ref.pdb"))
    if r.role == "candidate" and isinstance(REFS.loc[r.func, "negA_func"], str) and REFS.loc[r.func, "negA_func"]:
        nf = REFS.loc[r.func, "negA_func"]
        rows.append(dict(pair_id=f"{r.protein_id}|{r.func}|{r.role}|negA", protein_id=r.protein_id, func=r.func, role=r.role, pair="negA",
                         ref_func=nf, model=model, ref=f"{RS}/refs/{nf}_ref.pdb"))
    exp = H / f"ref/ext_{r.func}_exp.pdb"
    if r.role in ("candidate", "positive") and exp.exists():
        rows.append(dict(pair_id=f"{r.protein_id}|{r.func}|{r.role}|exp", protein_id=r.protein_id, func=r.func, role=r.role, pair="exp",
                         ref_func=r.func, model=model, ref=f"{RS}/refs/ext_{r.func}_exp.pdb"))
P = pd.DataFrame(rows).drop_duplicates("pair_id")
P.to_csv(E / "pairs.tsv", sep="\t", index=False)
subprocess.run(["ssh", B, f"mkdir -p {RS}/models {RS}/refs {RS}/code"], check=True)
lst = E / "models_ext.txt"
lst.write_text("\n".join(sorted(set(S.fold_path))) + "\n")
subprocess.run(["rsync", "-a", "--files-from", str(lst), "--no-relative", "/", f"{B}:{RS}/models/"], check=True)
refs = [str(Path(REFS.loc[f, "ref_file"])) for f in REFS.index] + [str(p) for p in (H / "ref").glob("ext_*_exp.pdb")]
subprocess.run(["rsync", "-a"] + refs + [f"{B}:{RS}/refs/"], check=True)
subprocess.run(["rsync", "-a", str(H / "code/s04_usalign_remote.py"), str(E / "pairs.tsv"), f"{B}:{RS}/code/"], check=True)
subprocess.run(["ssh", B, f"cd {RS} && python3 code/s04_usalign_remote.py code/pairs.tsv usalign_raw.tsv"], check=True)
subprocess.run(["rsync", "-a", f"{B}:{RS}/usalign_raw.tsv", str(UA / "ext_usalign_raw.tsv")], check=True)
U = pd.read_csv(UA / "ext_usalign_raw.tsv", sep="\t", keep_default_na=False)
P.merge(U, on="pair_id", how="left").to_csv(UA / "ext_usalign.tsv", sep="\t", index=False)     # alignments (SR-VP sequence)
P.merge(U.drop(columns=["aln_model", "aln_marks", "aln_ref"]), on="pair_id", how="left").to_csv(E / "usalign_scores.tsv", sep="\t", index=False)
print(len(P), "pairs;", (U.TM2 != "").sum(), "scored")
