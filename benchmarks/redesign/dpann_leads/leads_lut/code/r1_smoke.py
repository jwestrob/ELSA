#!/usr/bin/env python
"""Smoke test of r1_partners.py merge (leads_lut copy) before the unattended Slurm run; biotite login node, single process.
Redirects the remote folder and the KOfam output to SCRATCH (nothing is written to the real data/, novelty/ or annot_gpu/),
writes a fake Aksha KOFAM table (real carrier gids from shard_00.faa: the first 3,000 get K01953, every 7th also K00873),
an out_00.done status file next to out_00/ (the glob case), copies prep.json and kegg_ko.tsv, then runs merge and checks
the outputs. usage: r1_smoke.py SCRATCH"""
import json, os, shutil, sys
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import r1_partners as R1  # noqa: E402

SCR = sys.argv[1]
real = R1.cfg_of("lut")
shutil.rmtree(SCR, ignore_errors=True)
for d in ("partners/out_00", "data", "novelty"):
    os.makedirs(f"{SCR}/{d}", exist_ok=True)
shutil.copy(f"{real['remote']}/partners/prep.json", f"{SCR}/partners/prep.json")
shutil.copy(f"{real['remote']}/partners/labeled.faa", f"{SCR}/partners/labeled.faa")
shutil.copy(f"{real['remote']}/data/kegg_ko.tsv", f"{SCR}/data/kegg_ko.tsv")
ids = [l[1:].strip() for l in open(f"{real['remote']}/partners/shard_00.faa") if l.startswith(">")][:3000]
rows = [dict(sequence_id=g, hmm_name="K01953", bitscore=300.0, evalue=1e-80) for g in ids]
rows += [dict(sequence_id=g, hmm_name="K00873", bitscore=200.0, evalue=1e-50) for g in ids[::7]]
pd.DataFrame(rows).to_csv(f"{SCR}/partners/out_00/KOFAM_hits_df.tsv", sep="\t", index=False)
open(f"{SCR}/partners/out_00.done", "w").write("DONE smoke\n")


def fake_cfg(c):
    cfg = dict(real)
    cfg["remote"] = SCR
    cfg["kofam_partners"] = f"{SCR}/KOFAM_partners_smoke.parquet"
    return cfg


R1.cfg_of = fake_cfg
R1.merge("lut")
F = pd.read_parquet(f"{SCR}/data/genes_function.parquet")
fun = json.load(open(f"{SCR}/data/funnel_r1.json"))
print("rows", len(F), "scope", int(F.kofam_scope.sum()), "max capped carriers", int(F.n_carriers2.max()),
      "strong", int(F.strong_partner.sum()), "strong out of scope", int((F.strong_partner & ~F.kofam_scope).sum()))
print("columns ok:", all(c in F.columns for c in ["tier", "tier_family", "seq_len", "ilvfmawc", "order", "phylum", "completeness",
                                                  "n_carriers_all", "ec", "enzyme"]))
print(F[F.strong_partner].tier.value_counts().to_string())
print("candidates.faa", sum(1 for l in open(f"{SCR}/novelty/candidates.faa") if l.startswith(">")))
assert len(F) == 122421 and F.n_carriers2.max() <= 12 and not (F.strong_partner & ~F.kofam_scope).any()
print("SMOKE OK")
