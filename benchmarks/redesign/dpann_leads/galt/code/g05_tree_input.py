#!/usr/bin/env python
"""galt step 5: phylogeny input. DPANN: all 76 slot-labeled genes (contains the 64 and the 15); named DPANN family
members (length >= 250) reduced to cd-hit 50% representatives (aln/named_c50.faa; cluster sizes recorded); the DPANN
Ap4A-phosphorylase-branch proteins (length >= 250); reference panel (ref/refs.faa). Names: protein IDs, REF_<branch>_<label>.
Writes aln/tree_in.faa and data/tree_taxa.tsv (taxon, group, order, length, cluster_size).
Then run (shell): mafft --localpair --maxiterate 1000 --thread 4; trimal -gt 0.5; iqtree2 (see code/g05_tree_sh.txt)."""
import re
from pathlib import Path
import pandas as pd

H = Path("/Users/jacob/Documents/Sandbox/elsa_test/ELSA/benchmarks/redesign/dpann_leads/galt")
C = pd.read_csv(H / "data/census_proteins.tsv", sep="\t").set_index("protein_id")


def rf(p):
    d, h = {}, None
    for l in open(p):
        l = l.strip()
        if l.startswith(">"):
            h = l[1:].split()[0]; d[h] = ""
        elif h:
            d[h] += l
    return d


cen = rf(H / "data/census.faa")
reps = rf(H / "aln/named_c50.faa")
# cluster sizes from the .clstr file
size, cur, rep = {}, [], None
for l in open(H / "aln/named_c50_clstr.txt"):
    if l.startswith(">Cluster"):
        if rep:
            size[rep] = len(cur)
        cur, rep = [], None
    else:
        pid = re.search(r">(.+?)\.\.\.", l).group(1)
        cur.append(pid)
        if l.rstrip().endswith("*"):
            rep = pid
if rep:
    size[rep] = len(cur)
rows, out = [], []
for pid in C.index[C.lab76]:
    rows.append(dict(taxon=pid, group=C.loc[pid, "group"], set64=bool(C.loc[pid, "set64"]), order=C.loc[pid, "order"],
                     length=C.loc[pid, "length"], cluster_size=1)); out.append((pid, cen[pid]))
for pid in reps:
    rows.append(dict(taxon=pid, group="named_rep50", set64=False, order=C.loc[pid, "order"], length=C.loc[pid, "length"],
                     cluster_size=size.get(pid, 1))); out.append((pid, cen[pid]))
for pid in C.index[(C.group == "ap4a_branch") & (C.length >= 250)]:
    rows.append(dict(taxon=pid, group="ap4a_branch", set64=False, order=C.loc[pid, "order"], length=C.loc[pid, "length"],
                     cluster_size=1)); out.append((pid, cen[pid]))
for h, s in rf(H / "ref/refs.faa").items():
    _, br, lab, acc = h.split("|")
    t = f"REF_{br}_{lab}"
    rows.append(dict(taxon=t, group="REF_" + br, set64=False, order="", length=len(s), cluster_size=1, accession=acc))
    out.append((t, s))
T = pd.DataFrame(rows)
T.to_csv(H / "data/tree_taxa.tsv", sep="\t", index=False)
with open(H / "aln/tree_in.faa", "w") as fh:
    for t, s in out:
        fh.write(f">{t}\n{s.rstrip('*')}\n")
print(T.group.value_counts().to_string(), len(T))
