#!/usr/bin/env python
"""lut_loci n07 (biotite login node): proteins for the metal-handling assay (tests_rule.json), chosen by rule before any fold.

Families are rescue3 tokens (MMseqs2 families; the same definition for every tested protein).
  candidates  orf6-like 298295, orf7-like 361272, DUF3280 220448, lutC 253719 + 136200, lutG 92541, lutI 327650
  positives   lanM 358533 and lutD = LanD 349040 (the cluster's known lanthanide-binding proteins), xoxF 278478 (K23995)
  negatives A calcium-binding families by Pfam (EF-hand, Excalibur, thrombospondin type 3, hemolysin-type Ca-binding,
              Calx-beta, dockerin, FG-GAP, VCBS), protein 80-320 aa, the lanM token excluded
  negatives B proteins with no Pfam row and no slot label, outside every lut and xoxF window, length 90-300 aa
  Negative families: token with >= 10 members in >= 5 genera (candidate families all exceed this); one token per
  representative; drawn at random (seed 17) from lut-carrying genomes, 14 tokens for A (at most 3 per Pfam class) and 30 for B
  (10 per length bin 90-140, 140-190, 190-300 aa).
Representatives per tested family: up to 6 (candidates, positives) or 1 (negatives): the M. extorquens representative's
  member first where present, then one member per genus in order of genus member count (ties by gid), length within
  0.8-1.25x the family median.
Family alignment input: one member per genome (the first by gid), at most 150 members, spread over genera.
Writes sel/selection.tsv, sel/reps.faa, sel/fam_<family>_gids.txt, sel/negB_pool.tsv.
"""
import os

import numpy as np
import pandas as pd

RD = "/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111"
PK = RD + "/slice_lut/testset/package_k8"
rng = np.random.default_rng(17)
os.makedirs("sel", exist_ok=True)
CAND = {"orf6": [298295], "orf7": [361272], "DUF3280": [220448], "lutC": [253719, 136200], "lutG": [92541],
        "lutI": [327650]}
POS = {"lanM": [358533], "lutD_LanD": [349040], "xoxF": [278478]}
CA_PF = {"EF-hand": ["PF00036", "PF13202", "PF13405", "PF13499", "PF13833", "PF14658", "PF12763"],
         "Excalibur": ["PF05901"], "TSP_3": ["PF02412"], "HemolysinCabind": ["PF00353"], "Calx-beta": ["PF03160"],
         "Dockerin": ["PF00404"], "FG-GAP": ["PF01839", "PF13517"]}
MEXT = "GCA_900234795.1"

gn = pd.read_parquet(PK + "/universe/genes.parquet", columns=["gid", "genome", "genome_code", "length"])
gm = pd.read_parquet(PK + "/universe/genomes.parquet", columns=["genome_code", "genus", "lut_status"])
gn = gn.merge(gm, on="genome_code", how="left")
gn["token"] = pd.read_parquet(RD + "/long_loci/frozen/tok/mmseqs__seq__e3c50s75_rescue3__lut.parquet").token.to_numpy()
assert (gn.gid.to_numpy() == np.arange(len(gn))).all()
tstat = gn.groupby("token").agg(members=("gid", "size"), genera=("genus", "nunique"), med_len=("length", "median"))
pf = pd.read_parquet(RD + "/annot_gpu/lut/PFAM.parquet", columns=["gid", "accession"])
pf["acc"] = pf.accession.str.split(".").str[0]
haspf = np.zeros(len(gn), bool)
haspf[pf.gid.unique()] = True
lab = set(pd.read_parquet(RD + "/dpann_leads/pass_port/out/lut/lut_slot_rescue_pass.parquet", columns=["gid"]).gid)
win = set(pd.read_parquet("data/window_genes.parquet", columns=["gid"]).gid) | \
    set(pd.read_parquet("data/xoxf_window.parquet", columns=["gid"]).gid)
lutg = gm[gm.lut_status.str[:2].isin(["S1", "S2", "S3", "S4", "S5"])].genome_code
rows = []


def reps_of(name, role, toks, nmax):
    m = gn[gn.token.isin(toks)]
    med = m.length.median()
    m = m[(m.length >= 0.8 * med) & (m.length <= 1.25 * med)]
    out = []
    first = m[m.genome == MEXT]
    if len(first):
        out.append(first.iloc[0])
    gcount = m.genus.value_counts()
    for g in gcount.index:
        if len(out) >= nmax:
            break
        if any(o.genus == g for o in out):
            continue
        out.append(m[m.genus == g].sort_values("gid").iloc[0])
    for o in out:
        rows.append(dict(role=role, family=name, token=int(o.token), gid=int(o.gid), genome=o.genome, genus=o.genus,
                         length=int(o.length), fam_members=int(len(gn[gn.token.isin(toks)])),
                         fam_genera=int(gn[gn.token.isin(toks)].genus.nunique())))
    fm = gn[gn.token.isin(toks)].sort_values("gid").drop_duplicates("genome")
    if len(fm) > 150:
        fm = fm.groupby("genus", group_keys=False).apply(lambda d: d.head(3)).head(150)
    with open(f"sel/fam_{name}_gids.txt", "w") as fo:
        fo.write("\n".join(str(x) for x in fm.gid) + "\n")


for k, v in CAND.items():
    reps_of(k, "candidate", v, 6)
for k, v in POS.items():
    reps_of(k, "positive", v, 6 if k != "xoxF" else 3)
# negatives A
used = set()
for cls, accs in CA_PF.items():
    g = pf[pf.acc.isin(accs)].gid.unique()
    c = gn[gn.gid.isin(g) & (gn.length >= 80) & (gn.length <= 320) & gn.genome_code.isin(lutg) & (gn.token != 358533)]
    c = c[c.token.map(tstat.members) >= 10]
    c = c[c.token.map(tstat.genera) >= 5]
    toks = [t for t in c.token.unique() if t not in used]
    rng.shuffle(toks)
    for t in toks[:3]:
        used.add(t)
        reps_of(f"negA_{cls}_{t}", "negA", [t], 1)
# negatives B
ok = (~haspf) & ~gn.gid.isin(lab) & ~gn.gid.isin(win) & gn.genome_code.isin(lutg)
c = gn[ok & (gn.length >= 90) & (gn.length <= 300)]
c = c[(c.token.map(tstat.members) >= 10) & (c.token.map(tstat.genera) >= 5)]
# the token must be unannotated as a family (no member with a Pfam row)
tok_pf = pd.Series(haspf, index=gn.gid).groupby(gn.token.to_numpy()).any()
c = c[~c.token.map(tok_pf)]
c[["gid", "token", "length", "genus"]].to_csv("sel/negB_pool.tsv", sep="\t", index=False)
for lo, hi in [(90, 140), (140, 190), (190, 300)]:
    toks = c[(c.token.map(tstat.med_len) >= lo) & (c.token.map(tstat.med_len) < hi)].token.unique()
    toks = [t for t in toks if t not in used]
    rng.shuffle(toks)
    for t in toks[:10]:
        used.add(t)
        reps_of(f"negB_{lo}_{t}", "negB", [t], 1)
S = pd.DataFrame(rows)
S.to_csv("sel/selection.tsv", sep="\t", index=False)
print(S.groupby(["role", "family"]).size().to_string())
print(len(S), "representatives")
