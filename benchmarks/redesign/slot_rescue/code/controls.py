#!/usr/bin/env python3
"""slot_rescue step 2 (Mac; read-only on the Sharur DPANN DB): matched control ORFs by the predeclared rule.

For each unique truth protein of lead_X and DL sets (sorted by protein_id): 3 controls without replacement from the same
genome; no Pfam and no KO; not first / last gene of the contig; not a truth or negative protein; length within
max(5 aa, 15%); no gene within 2 positions carries a member of either unit of the replacement screen's 1,812 conserved
pairs. Seed 20261120. Writes data/controls.tsv and data/controls_meta.json.
"""
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

DB = "/Users/jacob/Documents/Sandbox/Sharur/Sharur/data/dpann/sharur.duckdb"
RS = Path("/Users/jacob/Documents/Berkeley/Sharur/replacement_screen_2026-11-11")
OUT = Path(__file__).resolve().parents[1] / "data"
M, SEED = 3, 20261120

T = pd.read_csv(OUT / "truth.tsv", sep="\t", keep_default_na=False)
prio = {"lead_X": 0, "DL_slot": 1, "DL_profile": 2}
Q = T[T.set.isin(prio)].copy()
Q["p"] = Q.set.map(prio)
Q = Q.sort_values(["protein_id", "p", "unit"]).drop_duplicates("protein_id")
excluded = set(T.protein_id)

pairs = pd.read_csv(RS / "tables/pairs_adjacent_ge40.tsv", sep="\t")
pairs = pairs[pairs.pass_conserved]
cons = set()
for m in list(pairs.X_members) + list(pairs.Y_members):
    cons |= set(str(m).split("+"))

con = duckdb.connect(DB, read_only=True)
con.execute("PRAGMA threads=4; PRAGMA memory_limit='8GB'")
bins = sorted(set(Q.bin_id))
P = con.execute("""SELECT protein_id, bin_id, contig_id, gene_index, sequence_length AS len FROM proteins
                   WHERE bin_id IN (SELECT unnest(?::VARCHAR[])) ORDER BY contig_id, gene_index""", [bins]).df()
A = pd.read_parquet(OUT / "annot.parquet")
annotated = set(A.protein_id)
carries_cons = set(A.protein_id[[bool(set(a) & cons) or bool(set(b) & cons) for a, b in zip(A.pfam, A.ko)]])
P["annot"] = P.protein_id.isin(annotated)
P["cons"] = P.protein_id.isin(carries_cons)
cc = pd.factorize(P.contig_id)[0]
gi = P.gene_index.to_numpy()
n = len(P)
near = np.zeros(n, bool)
cs = P.cons.to_numpy()
for d in (-2, -1, 1, 2):
    j = np.arange(n) + d
    ok = (j >= 0) & (j < n)
    jj = np.where(ok, j, 0)
    ok &= (cc[jj] == cc) & (np.abs(gi[jj] - gi) == abs(d))
    near |= ok & cs[jj]
first = np.r_[True, cc[1:] != cc[:-1]]
last = np.r_[cc[1:] != cc[:-1], True]
P["eligible"] = (~P.annot) & (~first) & (~last) & (~P.protein_id.isin(excluded)) & (~near)

rng = np.random.default_rng(SEED)
E = P[P.eligible]
by_bin = {b: g for b, g in E.groupby("bin_id")}
used = set()
rows, short = [], 0
for r in Q.itertuples(index=False):
    g = by_bin.get(r.bin_id)
    L = int(r.length)
    w = max(5, 0.15 * L)
    if g is None:
        cand = []
    else:
        cand = [p for p, ln in zip(g.protein_id, g.len) if abs(int(ln) - L) <= w and p not in used]
    pick = list(rng.choice(cand, size=min(M, len(cand)), replace=False)) if cand else []
    if len(pick) < M:
        short += 1
    used.update(pick)
    for c in pick:
        rows.append(dict(protein_id=c, matched_truth=r.protein_id, matched_set=r.set, matched_unit=r.unit, R_pfam=r.R_pfam,
                         R_ko=r.R_ko, broad_class=r.broad_class, bin_id=r.bin_id, truth_length=L))
C = pd.DataFrame(rows)
C = C.merge(P[["protein_id", "len"]].rename(columns={"len": "length"}), on="protein_id", how="left")
C.to_csv(OUT / "controls.tsv", sep="\t", index=False)
meta = dict(truth_proteins=len(Q), controls=len(C), truth_with_fewer_than_3=short,
            truth_with_0=int(len(Q) - C.matched_truth.nunique()), genomes=len(bins), proteins_in_genomes=n,
            eligible=int(P.eligible.sum()), conserved_labels=len(cons), near_conserved=int(near.sum()),
            controls_by_matched_set=C.groupby("matched_set").size().to_dict(),
            length_ratio_q10_q50_q90=np.quantile(C.length / C.truth_length, [.1, .5, .9]).round(3).tolist())
json.dump(meta, open(OUT / "controls_meta.json", "w"), indent=1)
print(json.dumps(meta, indent=1))
