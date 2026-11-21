#!/usr/bin/env python
"""Extended tier, remote helper (biotite login node, single-threaded): annotated members of each in-scope family
(assay_rule_extended.json): collection proteins with the family KO in the partner KOfam table (Aksha --cut_ga) AND a Pfam
row (GA) for the family label, not in the labeled table, 80-1000 aa. Label-domain envelope = the longest GA row of the label.
usage: r6_members.py --collection C --families REMOTE/data/ext_families.tsv  (columns func, label)
Writes REMOTE/data/ext_members.tsv (func, gid, length, env_from, env_to, genome, order) and REMOTE/data/ext_members.faa."""
import argparse, json, os
import pandas as pd

RD = "/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111"
HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser(); ap.add_argument("--collection", required=True); ap.add_argument("--families", required=True)
a = ap.parse_args()
cfg = {k: (v.replace("RD/", RD + "/") if isinstance(v, str) else v)
       for k, v in json.load(open(os.path.join(HERE, "collections_leads.json")))[a.collection].items()}
R = cfg["remote"]
fam = pd.read_csv(a.families, sep="\t")
lab = set(pd.read_parquet(f"{cfg['pass_dir']}/{a.collection}_slot_rescue_pass.parquet", columns=["gid"]).gid.astype(int))
K = pd.read_parquet(cfg["kofam_partners"])
pf = pd.read_parquet(cfg["pfam"], columns=["gid", "accession", "env_from", "env_to"])
pf["pf"] = pf.accession.str.split(".").str[0]
pf = pf[pf.pf.isin(set(fam.label))]
pf["span"] = pf.env_to - pf.env_from
pf = pf.sort_values("span", ascending=False).drop_duplicates(["gid", "pf"])
genes = pd.read_parquet(cfg["genes"], columns=["gid", "genome", "length"]).set_index("gid")
G = pd.read_parquet(cfg["genomes"])
ocol = "ord" if "ord" in G.columns else "order"
order_of = G.set_index("genome")[ocol].to_dict() if ocol in G.columns else {}
rows = []
for r in fam.itertuples():
    g1 = set(K[K.ko == r.func].gid.astype(int))
    d = pf[(pf.pf == r.label) & pf.gid.isin(g1) & ~pf.gid.isin(lab)]
    for x in d.itertuples():
        rows.append(dict(func=r.func, gid=int(x.gid), env_from=int(x.env_from), env_to=int(x.env_to)))
M = pd.DataFrame(rows, columns=["func", "gid", "env_from", "env_to"])
M["length"] = M.gid.map(genes.length)
M = M[M.length.between(80, 1000)]
M["genome"] = M.gid.map(genes.genome)
M["order"] = M.genome.map(order_of).fillna("NA")
want = set(M.gid)
seq, keep, name, buf = {}, False, None, []
for line in open(cfg["fasta"]):
    if line.startswith(">"):
        if keep:
            seq[name] = "".join(buf).rstrip("*")
        name = line[1:].split()[0]; keep = int(name) in want; buf = []
    elif keep:
        buf.append(line.strip())
if keep:
    seq[name] = "".join(buf).rstrip("*")
M.to_csv(f"{R}/data/ext_members.tsv", sep="\t", index=False)
with open(f"{R}/data/ext_members.faa", "w") as fh:
    for g in sorted(want):
        fh.write(f">{g}\n{seq[str(g)]}\n")
print(M.groupby("func").size().describe().to_string(), "\nfamilies with members:", M.func.nunique(), "of", len(fam))
