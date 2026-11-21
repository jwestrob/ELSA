#!/usr/bin/env python3
"""Step 10: annotate the extra region of every member of groups that pass the known-form control, from the weak full-Pfam
search (Aksha, i-E <= 1e-3, domain outside the label region by >= 70% of its envelope) plus CDD regions; consensus per
group = share of members with the family in the extra region. Writes data/extra_annot.parquet and groups_pass.tsv."""
import pandas as pd, numpy as np, os
from collections import Counter
os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/..")
S = pd.read_parquet("data/known_form.parquet")
P = S[~S.known_form].copy()
A = pd.read_parquet("data/screen_candidates_grouped.parquet")
A = A[A.rep.isin(P.rep)]
H = pd.read_csv("data/pfam_weak_hits_all.tsv.gz", sep="\t")
H = H[H.sequence_id.isin(A.protein_id)]
H = H.merge(A[["protein_id", "rep", "L_from", "L_to", "len", "lname"]], left_on="sequence_id", right_on="protein_id")
ov = (np.minimum(H.env_to, H.L_to) - np.maximum(H.env_from, H.L_from) + 1).clip(lower=0)
H["ovL"] = ov / (H.env_to - H.env_from + 1)
H["side_"] = np.where(H.ovL >= 0.3, "label", np.where(H.env_to <= H.L_from, "N", "C"))
R = pd.read_parquet("data/regions.parquet")
R = R[R.protein_id.isin(A.protein_id) & (R.source == "cdd")].merge(A[["protein_id", "rep", "L_from", "L_to"]], on="protein_id")
ovc = (np.minimum(R.end_aa, R.L_to) - np.maximum(R.start_aa, R.L_from) + 1).clip(lower=0)
R["ovL"] = ovc / (R.end_aa - R.start_aa + 1)
rows = []
for rep, g in A.groupby("rep"):
    h = H[(H.rep == rep) & (H.i_evalue <= 1e-3)]
    n = len(g)
    ex = h[h.side_ != "label"].groupby("hmm_name").agg(m=("protein_id", "nunique"), side=("side_", lambda x: x.value_counts().index[0]),
            ie=("i_evalue", "median"), f=("env_from", "median"), t=("env_to", "median")).sort_values("m", ascending=False)
    lab = h[h.side_ == "label"].groupby("hmm_name").protein_id.nunique().sort_values(ascending=False)
    c = R[(R.rep == rep) & (R.ovL < 0.3)].groupby("fam").protein_id.nunique().sort_values(ascending=False)
    rows.append(dict(rep=rep,
        extra_pfam=";".join(f"{k}({r.side},{int(r.f)}-{int(r.t)}):{r.m}/{n}" for k, r in ex.head(4).iterrows()),
        extra_pfam_top_share=(ex.m.iloc[0] / n) if len(ex) else 0.0,
        label_region_pfam=";".join(f"{k}:{v}/{n}" for k, v in lab.head(3).items()),
        extra_cdd=";".join(f"{k}:{v}/{n}" for k, v in c.head(3).items()),
        extra_cdd_top_share=(c.iloc[0] / n) if len(c) else 0.0))
X = pd.DataFrame(rows)
P = P.merge(X, on="rep")
P["extra_annotated"] = (P.extra_pfam_top_share >= 0.5) | (P.extra_cdd_top_share >= 0.5)
P.to_parquet("data/extra_annot.parquet", index=False)
cols = ["rep", "label", "genes", "genomes", "genera", "top_order", "len_med", "car_med", "L_from_med", "L_to_med", "carriers_fl", "ncar_all",
        "arch_annotated_in_carriers", "extra_annotated", "extra_pfam", "label_region_pfam", "extra_cdd"]
P[cols + ["hyd_med", "t1", "support_med", "emb_only", "pat_len", "pat_comp", "pat_dup", "phyla", "orders", "members"]].to_csv("groups_pass.tsv", sep="\t", index=False)
pd.set_option("display.width", 330); pd.set_option("display.max_colwidth", 75); pd.set_option("display.max_rows", 200)
print("groups passing known-form", len(P), "| extra region annotated in >= 50% of members", int(P.extra_annotated.sum()))
print(P[["rep", "label", "genes", "genera", "len_med", "car_med", "L_from_med", "L_to_med", "carriers_fl", "extra_pfam", "label_region_pfam", "extra_cdd"]].to_string(index=False))
