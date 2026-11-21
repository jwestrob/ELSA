#!/usr/bin/env python
"""Curated caller check (biotite login node, serial): DefenseFinder / TXSScan membership of given genes and of their +-5 gene
neighbors, from the SR-VP package gold tables (MacSyFinder 2.1.4; defense-finder-models 3.1.0; TXSScan 1.1.1; built by the
srvp_bins workstream). usage: r7_gold.py --collection C --gids FILE (one gid per line) --out OUT.tsv
OUT columns: gid, own_system (caller:system_type:subtype for systems that list the gene), neighbor_systems (offset:caller:type)."""
import argparse, json, os
import pandas as pd

RD = "/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111"
HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser(); ap.add_argument("--collection", required=True); ap.add_argument("--gids", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
cfg = {k: (v.replace("RD/", RD + "/") if isinstance(v, str) else v)
       for k, v in json.load(open(os.path.join(HERE, "collections_leads.json")))[a.collection].items()}
pkg = os.path.dirname(os.path.dirname(cfg["genes"]))
I = pd.read_parquet(f"{pkg}/gold/instances.parquet", columns=["instance_id", "caller", "system_type", "system_subtype"])
M = pd.read_parquet(f"{pkg}/gold/members.parquet").merge(I, on=["instance_id", "caller"])
lab = M.assign(t=M.caller + ":" + M.system_type + ":" + M.system_subtype).groupby("gid").t.apply(lambda s: ";".join(sorted(set(s))))
genes = pd.read_parquet(cfg["genes"], columns=["gid", "contig", "position_index"]).set_index("gid")
rows = []
for g in [int(x) for x in open(a.gids).read().split()]:
    me = genes.loc[g]
    nb = []
    for off in range(-5, 6):
        h = g + off          # gid is contiguous per contig (package rule); check the contig
        if off and h in genes.index and genes.loc[h, "contig"] == me.contig and h in lab.index:
            nb.append(f"{off:+d}:{lab[h]}")
    rows.append(dict(gid=g, own_system=lab.get(g, ""), neighbor_systems=" ".join(nb)))
pd.DataFrame(rows).to_csv(a.out, sep="\t", index=False)
print(pd.DataFrame(rows).to_string(index=False))
