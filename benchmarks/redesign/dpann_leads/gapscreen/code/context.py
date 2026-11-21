#!/usr/bin/env python3
"""Rungs 5-7 for the worked-up leads: per-step presence of the KEGG module around the gap (lead genomes vs the whole
clade), completeness of gap genomes vs the clade, function rate in clade genomes without a labeled gene, distribution
across orders (genomes with a labeled gene / order genome count), embedding-only share. Writes data/context.tsv and
data/distribution.tsv; prints the step table."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
H = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(H / "code"))
import screen as SC  # noqa: E402  (parser and evaluator reused)
D = H / "data"
LEADS = {"L1_dihydroorotase": ("K01465", "Micrarchaeales"), "L2_beta_carbonic_anhydrase": ("K01673", "Micrarchaeales"),
         "L3_ADP_serine_kinase": ("K23370", "Anstonellales"), "L4_small_alarmone_synthetase": ("K07816", "Woesearchaeales"),
         "L5_gamma_glutamylcysteine_ligase_GCS2": ("K01919", "Woesearchaeales")}
mods = pd.read_csv(D / "kegg_modules.tsv", sep="\t")
MOD = {m: [SC.P(s).p_or() for s in SC.split_top(str(d))] for m, d in zip(mods.module, mods.definition)}
MODS = {m: SC.split_top(str(d)) for m, d in zip(mods.module, mods.definition)}
KO = pd.read_parquet(D / "genome_ko.parquet").groupby("bin_id").ko.apply(set).to_dict()
bins = pd.read_parquet(D / "bins.parquet").set_index("bin_id")
S = pd.read_parquet(D / "genes_function.parquet")
T = pd.read_csv(H / "screen_leads.tsv", sep="\t")
crow, drow = [], []
for lead, (ko, clade) in LEADS.items():
    g = S[(S.ko_top == ko) & (S.order == clade)]
    gap_b = sorted(set(g[g.gap].bin_id))
    cl = bins.index[bins["order"] == clade]
    m = g.ctx_module.mode().iloc[0] if g.ctx_module.notna().any() and (g.ctx_module != "").any() else ""
    rec = dict(lead=lead, ko=ko, clade=clade, genes=len(g), genomes=g.bin_id.nunique(), gap_genomes=len(gap_b),
               clade_genomes=len(cl), completeness_gap_median=round(float(bins.loc[gap_b].completeness.median()), 1) if gap_b else None,
               completeness_clade_median=round(float(bins.loc[cl].completeness.median()), 1),
               gap_genomes_ge90=int((bins.loc[gap_b].completeness >= 90).sum()) if gap_b else 0,
               embedding_only_share=round(float(g.embedding_only.mean()), 3),
               own_ko_share=round(float((g.own_ko != "").mean()), 3), module=m)
    row = T[(T.ko == ko) & (T.clade == clade)]
    rec["clade_rate_function"] = float(row.clade_rate_function.iloc[0]) if len(row) else None
    crow.append(rec)
    if m:
        print(f"== {lead} {m}: step presence, gap genomes (n={len(gap_b)}) vs all {clade} genomes (n={len(cl)})")
        for i, (st, txt) in enumerate(zip(MOD[m], MODS[m])):
            a = np.mean([SC.ev(st, KO.get(b, set())) for b in gap_b]) if gap_b else np.nan
            c = np.mean([SC.ev(st, KO.get(b, set())) for b in cl])
            print(f"   step {i + 1}: {txt[:70]:70s} gap genomes {a:.2f}   clade {c:.2f}")
            rec[f"step{i + 1}"] = f"{a:.2f}/{c:.2f}"
    for r in T[T.ko == ko].itertuples():
        drow.append(dict(lead=lead, ko=ko, order=r.clade, phylum=r.phylum, genes=r.genes, genomes=r.genomes, order_genomes=r.clade_genomes,
                         gap_genomes=r.gap_genomes, tier_T1=r.tier_T1, clade_rate_function=r.clade_rate_function))
pd.DataFrame(crow).to_csv(D / "context.tsv", sep="\t", index=False)
pd.DataFrame(drow).to_csv(D / "distribution.tsv", sep="\t", index=False)
pd.set_option("display.width", 250)
print(pd.DataFrame(crow).drop(columns=[c for c in pd.DataFrame(crow).columns if c.startswith("step")]).to_string(index=False))
print(pd.DataFrame(drow).to_string(index=False))
