#!/usr/bin/env python
"""ribo_audit part 2 tables: reads work2/ outputs of bio.py, writes tables2.txt.
usage: tables2.py WORK2DIR OUTFILE
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

W = Path(sys.argv[1])
OUT = open(sys.argv[2], "w")
ORD = ("TOK", "L4b", "WA_L4b", "Z_L4bp")
VNAME = {1: "1 split one locus", 2: "2 run glues >= 2 loci", 3: "3 output locus accurate, larger than run",
         4: "4 output extends past the locus", 5: "5 fitting locus exists, scorer picks larger", 6: "6 data problem",
         7: "7 cannot tell"}


def p(*a):
    print(*a, file=OUT)


def wilson(k, n, z=1.96):
    if n == 0:
        return "-"
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return f"{100 * ph:.1f}% [{100 * max(0, c - h):.1f}, {100 * min(1, c + h):.1f}]"


cases = pd.read_csv(W / "cases2.tsv", sep="\t")
bio = pd.read_parquet(W / "bio_scores.parquet")
ri = pd.read_parquet(W / "runinfo.parquet")
junc = {d["case_id"]: d["junctions"] for d in json.load(open(W / "junctions2.json"))}

p("VERDICTS FOR EVERY UNRESOLVED RUN (standing rule), count per output")
t = cases.groupby(["verdict", "output"]).size().unstack(fill_value=0).reindex(columns=list(ORD), fill_value=0)
p(f"  {'verdict':<44}" + "".join(f"{o:>9}" for o in ORD))
for v in range(1, 8):
    row = t.loc[v] if v in t.index else pd.Series(0, index=list(ORD))
    p(f"  {VNAME[v]:<44}" + "".join(f"{int(row[o]):>9}" for o in ORD))
p(f"  {'unresolved':<44}" + "".join(f"{int((cases.output == o).sum()):>9}" for o in ORD))
p("")
p("ANSWER A: RUNS RIGHT UNDER THE ACCURATE READING (all 1,686)")
nt = ri.n_targets.to_numpy()
p(f"  runs with >= 1 accurate target (ceiling): {int((nt > 0).sum())}; single locus {int(((ri.n_units == 1) & (nt > 0)).sum())}; "
  f"multiple loci {int(((ri.n_units > 1) & (nt > 0)).sum())}; no target {int((nt == 0).sum())}")
for o in ORD:
    d = bio[bio.output == o]
    p(f"  {o:<8} standing {int(d.std_ok.sum()):>5}   accurate {int(d.bio_ok.sum()):>5}   "
      f"(std-only {int((d.std_ok & ~d.bio_ok).sum())}, bio-only {int((~d.std_ok & d.bio_ok).sum())})")
p("")
p("BY STRATUM (verdict counts per output)")
for o in ORD:
    d = cases[cases.output == o]
    tt = d.groupby(["stratum", "verdict"]).size().unstack(fill_value=0)
    p(f"  {o}:")
    for st, r in tt.iterrows():
        p(f"    {st:<20}" + " ".join(f"v{v}:{int(r[v])}" for v in tt.columns if r[v]))
p("")
p("DECISIVE JUNCTIONS (verdicts 1 and 2): strand, distance, classes, inheritance")
for v in (1, 2):
    for o in ORD:
        d = cases[(cases.output == o) & (cases.verdict == v)]
        js = [j for cid in d.case_id for j in junc.get(cid, [])]
        if not js:
            continue
        jj = pd.DataFrame(js)
        inh = jj.own_adj.dropna()
        p(f"  v{v} {o}: {len(d)} cases, {len(jj)} junctions; same strand {int(jj.same_strand.sum())}; "
          f"dist median {jj.dist.median()}; pair classes " + str((jj.left_cls + jj.right_cls).value_counts().to_dict()) +
          f"; inherited in own group (>= 1 other genome) {int((inh >= 1).sum())}/{len(inh)}; "
          f"in >= 1 other group {int((jj.other_groups_adj.dropna() >= 1).sum())}; unannotated {int(jj.unannot.sum())}; "
          f"bonded {int((jj.bond_this.fillna(0) > 0).sum())}")
        top = (jj.left + "|" + jj.right).value_counts().head(8).to_dict()
        p(f"      commonest: {top}")
OUT.close()
