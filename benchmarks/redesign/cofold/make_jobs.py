"""fold_manifest.tsv + sequences (biotite single-pass extraction) -> jobs.jsonl for biohub_cofold.py."""
import json
import sys
from pathlib import Path

import pandas as pd

W = Path(__file__).resolve().parent / "work"


def read_fasta(p):
    seqs, h = {}, None
    for line in open(p):
        line = line.strip()
        if line.startswith(">"):
            h = int(line[1:])
            seqs[h] = []
        elif line:
            seqs[h].append(line)
    return {k: "".join(v).rstrip("*") for k, v in seqs.items()}


man = pd.read_csv(W / sys.argv[1], sep="\t")
seqs = {}
for f in sys.argv[3:]:
    seqs.update(read_fasta(W / f))
with open(W / sys.argv[2], "w") as out:
    for _, r in man.iterrows():
        gids = [int(x) for x in str(r.gids).split(";")]
        chains = [{"id": "ABCDEFGH"[i], "seq": seqs[g], "gid": g} for i, g in enumerate(gids)]
        out.write(json.dumps({"fold_id": r.fold_id, "cls": r.cls, "chains": chains}) + "\n")
print(len(man), "jobs")
