#!/usr/bin/env python
"""lut_loci e07 (biotite login node): the RLWNL/LWNLT 'beta-roll' family handed over by spicy_xref (29 motif proteins,
spicy_xref/motif_betaroll.tsv -> data/br_gids.txt) and every member of their rescue3 tokens.

Per protein: token, genome group (frozen emb_hom_average_t0.75), genus, layer, length, motif flag (RLWNL or LWNLT), Pfam rows
(annot_gpu GA), distance in genes to the nearest main lut instance on the contig (0 = inside), nearest K23995 / K00114 /
K14028 gene and nearest PQQ-Pfam protein (PQQ, PQQ_2, PQQ_3) on the contig, TonB-dependent receptor (PF00593) within 3 genes,
slot call. Writes data/br_members.tsv and data/br_members.faa.
"""
import re

import numpy as np
import pandas as pd

RD = "/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111"
PK = RD + "/slice_lut/testset/package_k8"
FA = PK + "/proteins_gid.faa"
q = [int(x) for x in open("data/br_gids.txt") if x.strip()]
gn = pd.read_parquet(PK + "/universe/genes.parquet",
                     columns=["gid", "genome", "genome_code", "contig_code", "position_index", "strand", "length"])
gm = pd.read_parquet(PK + "/universe/genomes.parquet", columns=["genome_code", "genus", "family", "layer", "lut_status"])
grp = pd.read_parquet(RD + "/long_loci/frozen/groups/emb_hom_average_t0.75.parquet")[["genome_code", "group"]]
gn = gn.merge(gm, on="genome_code", how="left").merge(grp, on="genome_code", how="left")
tok = pd.read_parquet(RD + "/long_loci/frozen/tok/mmseqs__seq__e3c50s75_rescue3__lut.parquet").token.to_numpy()
gn["token"] = tok
toks = sorted(set(tok[q]))
mem = gn[gn.token.isin(toks) | gn.gid.isin(q)].copy()
want = set(mem.gid)
seqs, cur = {}, None
with open(FA) as fh:
    for line in fh:
        if line[0] == ">":
            k = int(line[1:].split()[0])
            cur = k if k in want else None
            if cur is not None:
                seqs[cur] = []
        elif cur is not None:
            seqs[cur].append(line.strip())
seqs = {k: "".join(v).rstrip("*") for k, v in seqs.items()}
mem["motif"] = mem.gid.map(lambda g: bool(re.search("RLWNL|LWNLT", seqs[g])))
mem["in_handover"] = mem.gid.isin(q)
pf = pd.read_parquet(RD + "/annot_gpu/lut/PFAM.parquet", columns=["gid", "accession", "name"])
pf["acc"] = pf.accession.str.split(".").str[0]
mem["pfam"] = mem.gid.map(pf[pf.gid.isin(want)].groupby("gid").name.apply(";".join))
I = pd.read_parquet(PK + "/gold/lut_instances.parquet")
I = I[I.definition == "main"]
mdh = pd.read_parquet("data/mdh_pkg.parquet")
pqq = set(pf[pf.acc.isin(["PF01011", "PF13360", "PF13570"])].gid)
tonb = set(pf[pf.acc == "PF00593"].gid)
gi = gn.set_index("gid")
cpos = {}
for k in ["K23995", "K00114", "K14028"]:
    cpos[k] = gi.loc[mdh[mdh.ko == k].gid.unique(), ["contig_code", "position_index"]]
pq = gi.loc[sorted(pqq), ["contig_code", "position_index"]]


def nearest(df, c, p):
    d = df[df.contig_code == c]
    return int((d.position_index - p).abs().min()) if len(d) else -1


rows = []
for r in mem.itertuples():
    ii = I[I.contig_code == r.contig_code]
    if len(ii):
        inside = ((ii.start_idx <= r.position_index) & (ii.end_idx >= r.position_index)).any()
        dl = 0 if inside else int(np.minimum((ii.start_idx - r.position_index).abs(), (ii.end_idx - r.position_index).abs()).min())
    else:
        dl = -1
    nb = gn[(gn.contig_code == r.contig_code) & ((gn.position_index - r.position_index).abs() <= 3)].gid
    rows.append(dict(gid=r.gid, dist_lut=dl, dist_K23995=nearest(cpos["K23995"], r.contig_code, r.position_index),
                     dist_K00114=nearest(cpos["K00114"], r.contig_code, r.position_index),
                     dist_K14028=nearest(cpos["K14028"], r.contig_code, r.position_index),
                     dist_PQQ_pfam=nearest(pq, r.contig_code, r.position_index), tonb_within3=bool(set(nb) & tonb)))
mem = mem.merge(pd.DataFrame(rows), on="gid")
ps = pd.read_parquet(RD + "/dpann_leads/pass_port/out/lut/lut_slot_rescue_pass.parquet",
                     columns=["gid", "proposed_label_name", "support"])
mem = mem.merge(ps, on="gid", how="left")
mem.to_csv("data/br_members.tsv", sep="\t", index=False)
with open("data/br_members.faa", "w") as fo:
    for g in mem.gid:
        fo.write(f">{g}\n{seqs[g]}\n")
print("tokens", {int(t): int((mem.token == t).sum()) for t in toks}, "members", len(mem), "motif", int(mem.motif.sum()),
      "handover", int(mem.in_handover.sum()))
