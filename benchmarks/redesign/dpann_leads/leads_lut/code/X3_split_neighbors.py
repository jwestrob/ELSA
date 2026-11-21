#!/usr/bin/env python
"""Frame-challenge test 3 (lut): for each assayed unannotated candidate, does a gene 1-3 positions away on the same contig
carry a Pfam row (GA) of the candidate's label family? Under 'split partner' (the partner's labeled domain is encoded by an
adjacent gene in the candidate genome: gene fission, frameshift or a fragmented call) the label domain sits next door in
candidates that do not cover the partner's label envelope (X2). Runs the lookup on biotite (genes table + Pfam table rows of
the candidate neighborhoods; single process), then joins with data/label_domain_overlap.tsv.
usage: X3_split_neighbors.py lut   writes data/split_neighbors.tsv; prints the comparison."""
import json, subprocess, sys
from pathlib import Path
import pandas as pd

C = sys.argv[1]
RD = "/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111"
R = Path("/Users/jacob/Documents/Sandbox/elsa_test/ELSA")
CFG = json.load(open(Path(__file__).resolve().parent / "collections_leads.json"))[C]
H = Path(CFG["local_out"].replace("R/", str(R) + "/"))
RS = CFG["remote"].replace("RD/", RD + "/")
B, PY = "jwestrob@biotite.berkeley.edu", "/home/jwestrob/miniconda3/envs/elsa/bin/python"
D = pd.read_csv(H / "data/label_domain_overlap.tsv", sep="\t", dtype={"protein_id": str})
D[["protein_id", "label"]].to_csv(H / "data/x3_in.tsv", sep="\t", index=False)
subprocess.run(["rsync", "-a", str(H / "data/x3_in.tsv"), f"{B}:{RS}/data/"], check=True)
remote = f"""
import pandas as pd
q = pd.read_csv('{RS}/data/x3_in.tsv', sep='\\t'); q['gid'] = q.protein_id.astype(int)
g = pd.read_parquet('{RD}/slice_lut/testset/package_k8/universe/genes.parquet', columns=['gid', 'contig', 'position_index', 'strand'])
gi = g.set_index('gid')
rows = []
for r in q.itertuples():
    me = gi.loc[r.gid]
    for off in (-3, -2, -1, 1, 2, 3):
        nb = r.gid + off
        if nb in gi.index and gi.loc[nb, 'contig'] == me.contig:
            rows.append(dict(protein_id=r.protein_id, label=r.label, offset=off, neighbor=nb, same_strand=int(gi.loc[nb, 'strand'] == me.strand)))
N = pd.DataFrame(rows)
pf = pd.read_parquet('{RD}/annot_gpu/{C}/PFAM.parquet', columns=['gid', 'accession', 'name'], filters=[('gid', 'in', list(N.neighbor))])
pf['pf'] = pf.accession.str.split('.').str[0]
N = N.merge(pf[['gid', 'pf', 'name']].rename(columns={{'gid': 'neighbor'}}), on='neighbor', how='left')
N['label_hit'] = N.pf == N.label
N.to_csv('{RS}/data/x3_neighbors.tsv', sep='\\t', index=False)
print(len(N))
"""
subprocess.run(["ssh", B, f"{PY} -"], input=remote, text=True, check=True)
subprocess.run(["rsync", "-a", f"{B}:{RS}/data/x3_neighbors.tsv", str(H / "data") + "/"], check=True)
N = pd.read_csv(H / "data/x3_neighbors.tsv", sep="\t", dtype={"protein_id": str})
lab = N[N.label_hit].sort_values("offset", key=abs).drop_duplicates("protein_id")
D = D.merge(lab[["protein_id", "offset", "same_strand", "name"]].rename(columns={"offset": "label_neighbor_offset", "name": "label_neighbor_pfam"}),
            on="protein_id", how="left")
D["label_next_door"] = D.label_neighbor_offset.notna()
D["covers_label"] = D.label_env_covered >= 0.5
D.to_csv(H / "data/split_neighbors.tsv", sep="\t", index=False)
pd.set_option("display.width", 200)
print(D.groupby(["covers_label"]).agg(n=("protein_id", "size"), label_next_door=("label_next_door", "sum"),
      adjacent_pm1=("label_neighbor_offset", lambda s: int(s.abs().eq(1).sum())), same_strand=("same_strand", "sum"),
      fold=("fold_match", "sum")).to_string())
print(D[~D.covers_label].groupby("func").agg(n=("protein_id", "size"), next_door=("label_next_door", "sum"),
      pm1=("label_neighbor_offset", lambda s: int(s.abs().eq(1).sum()))).sort_values("n", ascending=False).head(15).to_string())
