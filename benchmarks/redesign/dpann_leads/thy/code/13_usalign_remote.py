#!/usr/bin/env python3
"""Rung 4 structure comparison on the biotite login node (single-threaded serial loop, US-align).
mode mono: US-align a b (TM-align); mode mm1: US-align a b -mm 1 -ter 0 (multimer alignment, chain mapping found by
US-align). Saves the full text output of every pair (alignments included) and a parsed table.
TM1 = normalized by structure a (model), TM2 = normalized by structure b (reference / second half).
usage: 13_usalign_remote.py PAIRS.tsv PDB_DIR OUT_PREFIX"""
import re, subprocess, sys
import pandas as pd
US = "/shared/software/bin/USalign"
pairs, pdir, outp = sys.argv[1], sys.argv[2], sys.argv[3]
P = pd.read_csv(pairs, sep="\t")
rows, raw = [], open(outp + "_raw.txt", "w")
for r in P.itertuples(index=False):
    cmd = [US, f"{pdir}/{r.a}", f"{pdir}/{r.b}"] + (["-mm", "1", "-ter", "0"] if r.mode == "mm1" else [])
    res = subprocess.run(cmd, capture_output=True, text=True)
    out = res.stdout
    raw.write(f"##### {r.kind}\t{r.a}\t{r.b}\t{r.mode}\n{out}\n")
    tms = re.findall(r"TM-score= ([0-9.]+) \(normalized by length of Structure_([12])", out)
    tm = {k: float(v) for v, k in tms}
    m = re.search(r"Aligned length=\s*(\d+), RMSD=\s*([0-9.]+), Seq_ID=n_identical/n_aligned=\s*([0-9.]+)", out)
    rows.append(dict(kind=r.kind, a=r.a, b=r.b, mode=r.mode, rc=res.returncode, TM1=tm.get("1"), TM2=tm.get("2"),
                     Lali=int(m.group(1)) if m else None, RMSD=float(m.group(2)) if m else None, seqid=float(m.group(3)) if m else None))
    print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(outp + ".tsv", sep="\t", index=False)
