#!/usr/bin/env python
"""galt step 4: profile-based branch classification (Aksha, thresholds off, E <= 1000).

Models (downloaded by accession from the InterPro API): PANTHER family HMMs of the five GalT-like branches
(PTHR11943 GalT class I, PTHR39191 GalT class II, PTHR42763 ADP-glucose phosphorylase, PTHR38420 Ap4A phosphorylase,
PTHR20884 GDP-hexose phosphorylase), NCBIfam TIGR00209 (galT_1) and TIGR01239 (galT_2), Pfam PF01087 / PF02744
(GalT N / C domains), PF19327 / PF09830 (Ap4A phosphorylase), PF26216 / PF26217 (GDPGP1), PF01230 (HIT).
Full-sequence bit score per model; branch call = best PANTHER family model; margin = best - second best.
Control: the reference panel (branch known) must be called correctly.
Writes data/profile_scores.tsv (one row per sequence, a score column per model)."""
from pathlib import Path
import pandas as pd

H = Path("/Users/jacob/Documents/Sandbox/elsa_test/ELSA/benchmarks/redesign/dpann_leads/galt")
A = pd.read_csv(H / "data/aksha_branch/user_hmms_hits_df.tsv", sep="\t")
A["hmm"] = A.hmm_name.str.replace(".orig.30.pir", "", regex=False)
S = A.groupby(["sequence_id", "hmm"]).bitscore.max().unstack()
# model coverage of the best domain for the Pfam family models
cov = A.assign(c=(A.hmm_to - A.hmm_from + 1)).groupby(["sequence_id", "hmm"]).c.max().unstack()
PAN = {"PTHR11943": "GalT_I", "PTHR39191": "GalT_II", "PTHR42763": "ADPGP", "PTHR38420": "Ap4A_phosphorylase",
       "PTHR20884": "GDPhexose_phosphorylase"}
C = pd.read_csv(H / "data/census_proteins.tsv", sep="\t")
ids = pd.Series(S.index, index=S.index)
allids = list(C.protein_id) + [l[1:].strip() for l in open(H / "ref/refs.faa") if l.startswith(">")]
S = S.reindex(allids)
P = S[list(PAN)].fillna(0)
best = P.idxmax(axis=1)
srt = pd.DataFrame(sorted(r, reverse=True)[:2] for r in P.to_numpy())
out = S.round(1).copy()
out.columns = [f"bits_{c}" for c in out.columns]
for m in ["GalP_UDP_transf", "GalP_UDP_tr_C"]:
    out[f"hmmcov_{m}"] = cov.reindex(allids)[m] if m in cov.columns else pd.NA
out["branch_call"] = [PAN[b] if v > 0 else "none" for b, v in zip(best, P.max(axis=1))]
out["branch_margin"] = (srt[0] - srt[1]).round(1).to_numpy()
out["ADPGP_minus_GalTI"] = (P["PTHR42763"] - P["PTHR11943"]).round(1)
out = out.reset_index().rename(columns={"index": "sequence_id"})
out["sequence_id"] = allids
meta = C.set_index("protein_id")[["group", "order", "length", "set64", "lead15"]]
out = out.merge(meta, left_on="sequence_id", right_index=True, how="left")
ref = out.sequence_id.str.startswith("REF|")
out.loc[ref, "group"] = "REF_" + out.loc[ref, "sequence_id"].str.split("|").str[1]
out.to_csv(H / "data/profile_scores.tsv", sep="\t", index=False)
pd.set_option("display.width", 250)
print("REFERENCE CONTROL (true branch x call):")
print(pd.crosstab(out[ref].group, out[ref].branch_call))
print("\nDPANN groups x call:")
print(pd.crosstab(out[~ref].group, out[~ref].branch_call))
g = out[~ref].groupby("group")
cols = ["bits_GalP_UDP_transf", "bits_GalP_UDP_tr_C", "bits_PTHR11943", "bits_PTHR42763", "bits_galT_1", "bits_PTHR38420", "bits_PTHR20884", "ADPGP_minus_GalTI", "branch_margin"]
print("\nmedian scores by group (NaN = no hit at E<=1000 counted as missing):")
print(g[cols].median().round(1))
print(g[cols].count())
print("\nlead15 rows:")
print(out[out.lead15 == True][["sequence_id"] + cols + ["hmmcov_GalP_UDP_transf", "branch_call"]].to_string(index=False))
print("\nreferences:")
print(out[ref][["sequence_id"] + cols + ["branch_call"]].to_string(index=False))
