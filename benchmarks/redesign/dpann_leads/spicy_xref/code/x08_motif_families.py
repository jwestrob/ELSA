"""Cross-reference two small secreted families of the earlier notes with the lut collection by sequence motif
(observation only; motifs are the notes' own descriptors):
  beta-roll ('Marco's beta-roll', notes: 114-125 aa, xRLWNLTx motif, near LanM / xoxF, TonB -> beta-roll -> LanM)
  CKDD cluster (notes: 14 small secreted LanM-adjacent proteins incl. Leaf118 and M. extorquens) against LutI members.
Reads package_k8 proteins_gid.faa, genes, gold lut tables and lut Pfam. Writes motif_betaroll.tsv, motif_summary.json."""
import json, re
import numpy as np
import pandas as pd
RD = "/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111"
PKG = RD + "/slice_lut/testset/package_k8"
genes = pd.read_parquet(PKG + "/universe/genes.parquet", columns=["gid", "genome", "genome_code", "contig_code", "strand", "length"])
gm = pd.read_parquet(PKG + "/universe/genomes.parquet", columns=["genome", "genome_code", "tax", "layer"]).sort_values("genome_code")
gm["genus"] = gm.tax.str.extract(r"g__([^;]*)")[0]
CC = genes.contig_code.to_numpy()
mem = pd.read_parquet(PKG + "/gold/lut_members.parquet")
mem = mem[mem.caller == "lut_homology_main"]
comp = dict(zip(mem.gid, mem.component))
lutI = set(mem.gid[mem.component == "I"])
ins = pd.read_parquet(PKG + "/gold/lut_instances.parquet")
ins = ins[ins.caller == "lut_homology_main"]
inlut = np.zeros(len(genes), bool)
for g in ins.member_gids:
    inlut[min(g):max(g) + 1] = True
P = pd.read_parquet(RD + "/annot_gpu/lut/PFAM.parquet", columns=["gid", "name", "env_from", "env_to"])
pq = set(P.gid[P.name.isin(["PQQ", "PQQ_2"])]) & set(genes.gid[(genes.length >= 500) & (genes.length <= 700)])
tonb = set(P.gid[P.name.isin(["Plug", "TonB_dep_Rec_b-barrel"])])
ef = set(P.gid[P.name.str.startswith("EF-hand")]) & set(genes.gid[genes.length <= 200])
br, ck, ckI = [], 0, 0
gid, seq = None, []
def flush():
    global ck, ckI
    if gid is None:
        return
    s = "".join(seq)
    if 100 <= len(s) <= 140 and re.search(r"RLWNL|LWNLT", s):
        br.append(gid)
    if gid in lutI and "CKDD" in s:
        ckI += 1
for line in open(PKG + "/proteins_gid.faa"):
    if line.startswith(">"):
        flush()
        gid, seq = int(line[1:].split()[0]), []
    else:
        seq.append(line.strip())
flush()
def near(x, S, k):
    return [d for d in range(-k, k + 1) if d and 0 <= x + d < len(genes) and CC[x + d] == CC[x] and (x + d) in S]
rows = []
for x in br:
    g = genes.genome_code.iat[x]
    rows.append(dict(gid=x, genome=gm.genome.iat[g], genus=gm.genus.iat[g], layer=gm.layer.iat[g], length=int(genes.length.iat[x]),
                     in_lut_span=bool(inlut[x]), lut_component_within2=";".join(f"{d}:{comp[x + d]}" for d in range(-2, 3) if d and (x + d) in comp),
                     pqqdh_10=str(near(x, pq, 10)), tonb_3=str(near(x, tonb, 3)), efhand_small_3=str(near(x, ef, 3))))
B = pd.DataFrame(rows)
B.to_csv("motif_betaroll.tsv", sep="\t", index=False)
out = dict(betaroll_proteins=len(B), genomes=int(B.genome.nunique()) if len(B) else 0,
           genera=B.genus.value_counts().to_dict() if len(B) else {}, in_lut_span=int(B.in_lut_span.sum()) if len(B) else 0,
           with_pqqdh_10=int((B.pqqdh_10 != "[]").sum()), with_tonb_3=int((B.tonb_3 != "[]").sum()),
           with_small_efhand_3=int((B.efhand_small_3 != "[]").sum()),
           lutI_members=len(lutI), lutI_with_CKDD=ckI,
           leaf118_in_lut=bool(gm.genome.str.contains("920984675").any()))
json.dump(out, open("motif_summary.json", "w"), indent=1)
print(json.dumps(out, indent=1)); print(B.to_string())
