#!/usr/bin/env python3
"""Rung 2 for the worked-up leads: each labeled gene's own weak hit to the proposed family model (Aksha check A, E <= 10),
with model coverage and sequence coverage (best domain envelope) from the raw Aksha table (data/rung2_hits.tsv)."""
from pathlib import Path
import pandas as pd
H = Path(__file__).resolve().parents[1]
LENG = {"Amidohydro_1": 345, "GCS2": 377, "Pro_CA": 156, "K23370": 203}
LEADS = {"L1_dihydroorotase": ("K01465", "Micrarchaeales", "Amidohydro_1"), "L2_beta_carbonic_anhydrase": ("K01673", "Micrarchaeales", "Pro_CA"),
         "L3_ADP_serine_kinase": ("K23370", "Anstonellales", "K23370"), "L5_gamma_glutamylcysteine_ligase_GCS2": ("K01919", "Woesearchaeales", "GCS2")}
S = pd.read_parquet(H / "data/genes_function.parquet")
A = pd.read_csv(H / "data/rung2_hits.tsv", sep="\t")
A = A.groupby(["sequence_id", "hmm_name"]).agg(bitscore=("bitscore", "max"), evalue=("evalue", "min"), hmm_from=("hmm_from", "min"),
                                                hmm_to=("hmm_to", "max"), env_from=("env_from", "min"), env_to=("env_to", "max")).reset_index()
rows = []
W = pd.read_csv(H / "data/workup_genes.tsv", sep="\t")
for lead, (ko, clade, model) in LEADS.items():
    g = S[(S.ko_top == ko) & (S.order == clade)].merge(A[A.hmm_name == model], left_on="protein_id", right_on="sequence_id", how="left")
    g["model_cov"] = (g.hmm_to - g.hmm_from + 1) / LENG[model]
    g["seq_cov"] = (g.env_to - g.env_from + 1) / g.length
    hit = g[g.evalue.notna()]
    rows.append(dict(lead=lead, model=model, genes=len(g), with_hit=len(hit), T1=int((g.tier == "T1").sum()),
                     median_evalue=float(hit.evalue.median()), median_bits=float(hit.bitscore.median()),
                     median_model_cov=round(float(hit.model_cov.median()), 3), median_seq_cov=round(float(hit.seq_cov.median()), 3)))
    for r in g[g.protein_id.isin(W.protein_id)].itertuples():
        rows.append(dict(lead=lead, model=model, protein_id=r.protein_id, evalue=r.evalue, bits=r.bitscore,
                         model_cov=round(r.model_cov, 3), seq_cov=round(r.seq_cov, 3)))
R = pd.DataFrame(rows)
R.to_csv(H / "data/rung2.tsv", sep="\t", index=False)
pd.set_option("display.width", 250)
print(R.to_string(index=False))
