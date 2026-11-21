#!/usr/bin/env python
"""Leads pipeline step 3 parse (biotite, called by r3_novelty_sbatch.txt): per-gene hit tables of the candidate set.
  novelty_pfam.tsv   gid, pf (accession), pfname, evalue, bitscore           Aksha full Pfam-A, E <= 0.01 (relaxed, no GA)
  novelty_kofam.tsv  gid, ko, evalue, bitscore                              Aksha full KOfam, E <= 0.01 (relaxed, no ko_list)
  novelty_cdd.tsv    gid, accession, name, description, evalue, bitscore, qstart, qend
                     rpsblast CDD E <= 0.01, best model per region exactly as Sharur protein_resemblance (greedy by bit score,
                     ties lower E then lower model id; a hit is kept when its query range shares at most 0.5 of the shorter
                     range with every hit kept before it); accession / name / description = the three comma fields of stitle.
usage: r3_novelty.py --collection C"""
import argparse, json, os
import pandas as pd

RD = "/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111"
HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser(); ap.add_argument("--collection", required=True); a = ap.parse_args()
cfg = json.load(open(os.path.join(HERE, "collections_leads.json")))[a.collection]
N = cfg["remote"].replace("RD/", RD + "/") + "/novelty"
acc, nm = {}, None
for line in open(f"{N}/pfam_name_acc.txt"):
    k, v = line.split()[:2]
    if k == "NAME":
        nm = v
    else:
        acc[nm] = v.split(".")[0]
p = pd.read_csv(f"{N}/pfam_relaxed/PFAM_hits_df.tsv", sep="\t", dtype={"sequence_id": str})
p = p.sort_values("bitscore", ascending=False).drop_duplicates(["sequence_id", "hmm_name"])
pd.DataFrame(dict(gid=p.sequence_id, pf=p.hmm_name.map(acc), pfname=p.hmm_name, evalue=p.evalue, bitscore=p.bitscore)).to_csv(
    f"{N}/novelty_pfam.tsv", sep="\t", index=False)
k = pd.read_csv(f"{N}/kofam_relaxed/KOFAM_hits_df.tsv", sep="\t", dtype={"sequence_id": str})
k["ko"] = k.hmm_name.astype(str).str.extract(r"(K\d{5})")[0]
k = k.sort_values("bitscore", ascending=False).drop_duplicates(["sequence_id", "ko"])
pd.DataFrame(dict(gid=k.sequence_id, ko=k.ko, evalue=k.evalue, bitscore=k.bitscore)).to_csv(f"{N}/novelty_kofam.tsv", sep="\t", index=False)
cols = ["qseqid", "sseqid", "qlen", "qstart", "qend", "slen", "sstart", "send", "evalue", "bitscore", "stitle"]
c = pd.read_csv(f"{N}/cdd_raw.tsv", sep="\t", names=cols, dtype={"qseqid": str}, quoting=3) if os.path.getsize(f"{N}/cdd_raw.tsv") else pd.DataFrame(columns=cols)
c["mid"] = c.sseqid.astype(str).str.split("|").str[-1]
c["accession"] = c.stitle.str.split(",").str[0].str.strip()
c["name"] = c.stitle.str.split(",").str[1].str.strip()
c["description"] = c.stitle.str.extract(r"^[^,]*,[^,]*,(.*)$")[0].str.strip()
keep = []
for q, g in c.sort_values(["qseqid", "bitscore", "evalue", "mid"], ascending=[True, False, True, True]).groupby("qseqid", sort=False):
    kept = []
    for r in g.itertuples():
        ok = True
        for s, e in kept:
            ov = max(0, min(e, r.qend) - max(s, r.qstart) + 1)
            if ov > 0.5 * min(e - s + 1, r.qend - r.qstart + 1):
                ok = False; break
        if ok:
            kept.append((r.qstart, r.qend))
            keep.append(dict(gid=q, accession=r.accession, name=r.name, description=r.description, evalue=r.evalue, bitscore=r.bitscore,
                             qstart=r.qstart, qend=r.qend))
pd.DataFrame(keep, columns=["gid", "accession", "name", "description", "evalue", "bitscore", "qstart", "qend"]).to_csv(
    f"{N}/novelty_cdd.tsv", sep="\t", index=False)
print("pfam rows", len(p), "kofam rows", len(k), "cdd raw", len(c), "cdd best-per-region", len(keep))
