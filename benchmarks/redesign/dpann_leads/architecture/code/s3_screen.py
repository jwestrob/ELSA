#!/usr/bin/env python3
"""Architecture screen, step 3: filter cascade for (a) length outliers and (b) composite / internally duplicated genes.
Writes data/screen_candidates.parquet, data/screen_regions.parquet, screen_counts.tsv and cand.faa (for grouping)."""
import pandas as pd, numpy as np, os
os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/..")
G = pd.read_parquet("data/genes2.parquet")
R = pd.read_parquet("data/regions.parquet")
counts = []
def step(name, df, note=""):
    counts.append(dict(step=name, genes=len(df), genomes=df.bin_id.nunique(), labels=df.acc.nunique(), note=note)); print(counts[-1])
step("labeled genes", G)
G = G[G.n_car >= 5]; step("label has >= 5 DPANN carriers at GA", G, "stable carrier median")
# label region: own weak hit to the proposed label (i-E <= 10) or a CDD pfam-derived model of the label
cddlab = R[R.cdd_is_label].groupby("protein_id").agg(c_from=("start_aa", "min"), c_to=("end_aa", "max"))
G = G.merge(cddlab, left_on="protein_id", right_index=True, how="left")
G["L_from"] = G.lab_from.fillna(G.c_from); G["L_to"] = G.lab_to.fillna(G.c_to)
G["has_label_region"] = G.L_from.notna()
# other-family regions outside the label region
R = R.merge(G[["protein_id", "L_from", "L_to"]], on="protein_id")
ov = (np.minimum(R.end_aa, R.L_to) - np.maximum(R.start_aa, R.L_from) + 1).clip(lower=0).fillna(0)
R["ovL"] = ov / (R.end_aa - R.start_aa + 1)
R["rlen"] = R.end_aa - R.start_aa + 1
strong = ((R.source == "cdd") | ((R.source == "label_model") & (R.evalue <= 1e-3)))
other = R[strong & (R.source != "proposed_label") & ~R.cdd_is_label & (R.ovL < 0.3) & (R.rlen >= 40)]
other = other.sort_values("evalue")
# collapse overlapping other-family regions to the best per span
keep = []
for pid, g in other.groupby("protein_id", sort=False):
    taken = []
    for _, r in g.iterrows():
        if all(min(r.end_aa, b) - max(r.start_aa, a) + 1 < 0.5 * min(r.rlen, b - a + 1) for a, b in taken):
            taken.append((r.start_aa, r.end_aa)); keep.append(r)
other = pd.DataFrame(keep)
# second copy of the proposed label (internal duplication): proposed-label domains not overlapping each other
pl = R[(R.source == "proposed_label") & (R.evalue <= 1e-2)].sort_values("evalue")
dup = {}
for pid, g in pl.groupby("protein_id", sort=False):
    taken = []
    for _, r in g.iterrows():
        if all(min(r.end_aa, b) - max(r.start_aa, a) + 1 < 20 for a, b in taken): taken.append((r.start_aa, r.end_aa))
    if len(taken) >= 2: dup[pid] = len(taken)
G["n_other_regions"] = G.protein_id.map(other.groupby("protein_id").size()).fillna(0).astype(int)
G["label_copies"] = G.protein_id.map(dup).fillna(1).astype(int)
G["other_fams"] = G.protein_id.map(other.groupby("protein_id").apply(lambda g: ";".join(f"{r.fam}[{r.start_aa}-{r.end_aa}|{r.evalue:.0e}]" for _, r in g.sort_values("start_aa").iterrows())))
G["pat_len"] = G.outlier
G["pat_comp"] = G.has_label_region & (G.n_other_regions >= 1)
G["pat_dup"] = G.label_copies >= 2
A = G[G.pat_len | G.pat_comp | G.pat_dup].copy()
step("pattern (a) length outlier", G[G.pat_len]); step("pattern (b) composite: label region + other-family region", G[G.pat_comp])
step("pattern (c) proposed label twice in one chain", G[G.pat_dup]); step("union of patterns", A)
A = A[A.has_label_region]; step("label region located (own weak hit or CDD model of the label)", A)
A = A[~A.partial.isin(["01", "10", "11"])]; step("complete gene (no partial flag)", A)
memb = (A.hyd >= 0.47) | (A.tm_segs >= 3) | (A.tm_frac >= 0.2)
step("  membrane-dominated (dropped)", A[memb]); A = A[~memb]
step("composition: not membrane-dominated (ILVFMAWC < 0.47, < 3 KD TM segments, TM < 20% of length)", A)
rep = (A.lowc_frac >= 0.2) | (A.exrep_frac >= 0.15) | (A.max_fam_copies >= 4)
step("  repeat- or low-complexity-dominated (dropped)", A[rep]); A = A[~rep]
step("composition: low complexity < 20%, exact repeats < 15%, < 4 copies of one family", A)
A.to_parquet("data/screen_candidates.parquet", index=False)
other[other.protein_id.isin(A.protein_id)].to_parquet("data/screen_regions.parquet", index=False)
pd.DataFrame(counts).to_csv("screen_counts.tsv", sep="\t", index=False)
seqs = pd.read_parquet("data/genes.parquet", columns=["protein_id", "seq"]).set_index("protein_id").seq
with open("data/cand.faa", "w") as f:
    for pid in A.protein_id: f.write(f">{pid}\n{seqs[pid].rstrip('*')}\n")
print(A.groupby("lname").agg(n=("protein_id", "size"), g=("bin_id", "nunique"), L=("pat_len", "sum"), C=("pat_comp", "sum"), D=("pat_dup", "sum"),
      len=("len", "median"), car=("car_med_len", "first")).sort_values("n", ascending=False).head(50).to_string())
