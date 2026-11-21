#!/usr/bin/env python3
"""Architecture screen, step 7: family-level remote homology for members of groups that pass the known-form control.
Members map to their permissive MMseqs2 family (permissive-e3c50s75); Sharur family_annotations holds HHsearch (Pfam 39)
and HHblits (PDB100) hits of the family MSA with query coordinates on the family representative (only Pfam-dark families
with >= 5 genomes were searched). Hits are placed on the representative and classified as label region / extra region
using the representative's own label region when it is a screened gene, else the member coordinates scaled by length.
Writes data/family_hh.parquet."""
import duckdb, pandas as pd, numpy as np, os
os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/..")
S = pd.read_parquet("data/known_form.parquet")
S = S[~S.known_form]
A = pd.read_parquet("data/screen_candidates_grouped.parquet")
A = A[A.rep.isin(S.rep)]
con = duckdb.connect(); con.execute("SET threads=4")
con.execute("ATTACH '/Users/jacob/Documents/Sandbox/Sharur/Sharur/data/dpann/sharur.duckdb' AS d (READ_ONLY)")
con.register("m", A[["protein_id", "rep", "len", "L_from", "L_to"]])
F = con.execute("""SELECT m.*, pf.family_id FROM m JOIN d.protein_families pf ON pf.protein_id=m.protein_id AND pf.clustering_run='permissive-e3c50s75'""").df()
con.register("f", F[["family_id"]].drop_duplicates())
HH = con.execute("""SELECT fa.family_id, fa.source, fa.hit_accession, fa.hit_name, fa.hit_description, fa.probability, fa.evalue, fa.q_start, fa.q_end, fa.q_length,
                    fa.t_start, fa.t_end, fa.t_length FROM d.family_annotations fa JOIN f USING (family_id) WHERE fa.probability >= 80""").df()
F["fam_rep"] = F.family_id.str.split(":").str[2]
fam_cnt = F.groupby(["rep", "family_id"]).size().rename("members_in_family").reset_index()
X = fam_cnt.merge(HH, on="family_id")
# label region on the representative: median of members in this family (scaled to q_length)
lr = F.groupby("family_id").apply(lambda g: pd.Series(dict(Lf=np.median(g.L_from / g.len), Lt=np.median(g.L_to / g.len))), include_groups=False).reset_index()
X = X.merge(lr, on="family_id")
X["Lf_q"] = X.Lf * X.q_length; X["Lt_q"] = X.Lt * X.q_length
ov = (np.minimum(X.q_end, X.Lt_q) - np.maximum(X.q_start, X.Lf_q)).clip(lower=0)
X["in_label_frac"] = ov / (X.q_end - X.q_start + 1)
X["region"] = np.where(X.in_label_frac >= 0.5, "label", "extra")
X.to_parquet("data/family_hh.parquet", index=False)
print("members", len(A), "with permissive family", len(F), "families", F.family_id.nunique(), "families with HH hits (prob >= 80)", HH.family_id.nunique())
best = X.sort_values("probability", ascending=False).groupby(["rep", "region", "source"]).head(3)
pd.set_option("display.width", 300); pd.set_option("display.max_colwidth", 60); pd.set_option("display.max_rows", 500)
print(best[["rep", "region", "source", "hit_name", "hit_description", "probability", "q_start", "q_end", "q_length", "members_in_family"]].sort_values(["rep", "region"]).to_string(index=False))
