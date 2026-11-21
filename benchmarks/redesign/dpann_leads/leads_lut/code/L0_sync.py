#!/usr/bin/env python
"""Leads pipeline helper (local): push code to the remote leads folder, or pull the small remote tables into the local
data/ folder.   usage: L0_sync.py COLLECTION push|pull
push: code/ -> REMOTE/code/, gapscreen KEGG snapshot -> REMOTE/data/.
pull: REMOTE/data/{genes_function.parquet,pfset.tsv,funnel_r1.json} and REMOTE/novelty/novelty_*.tsv (+ STATUS) -> data/.
No sequence leaves biotite in either direction."""
import json, subprocess, sys
from pathlib import Path

C, MODE = sys.argv[1], sys.argv[2]
RD = "/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111"
R = Path("/Users/jacob/Documents/Sandbox/elsa_test/ELSA")
CODE = Path(__file__).resolve().parent
CFG = json.load(open(CODE / "collections_leads.json"))[C]
H = Path(CFG["local_out"].replace("R/", str(R) + "/"))
RS = CFG["remote"].replace("RD/", RD + "/")
B = "jwestrob@biotite.berkeley.edu"
(H / "data").mkdir(parents=True, exist_ok=True)
if MODE == "push":
    subprocess.run(["ssh", B, f"mkdir -p {RS}/code {RS}/data {RS}/partners {RS}/novelty {RD}/dpann_leads/logs"], check=True)
    subprocess.run(["rsync", "-a", f"{CODE}/", f"{B}:{RS}/code/"], check=True)
    g = R / "benchmarks/redesign/dpann_leads/gapscreen/data"
    subprocess.run(["rsync", "-a", str(g / "kegg_ko.tsv"), str(g / "kegg_modules.tsv"), f"{B}:{RS}/data/"], check=True)
else:
    files = [f"{RS}/data/genes_function.parquet", f"{RS}/data/pfset.tsv", f"{RS}/data/funnel_r1.json", f"{RS}/partners/prep.json"]
    files += [f"{RS}/novelty/{f}" for f in ("novelty_pfam.tsv", "novelty_kofam.tsv", "novelty_cdd.tsv", "STATUS", "rpsblast_version.txt")]
    for f in files:
        subprocess.run(["rsync", "-a", f"{B}:{f}", str(H / "data") + "/"], check=False)
print("ok", MODE, C)
