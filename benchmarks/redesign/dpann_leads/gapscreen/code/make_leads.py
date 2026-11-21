#!/usr/bin/env python3
"""Assemble leads.tsv: every (KO, order) lead of the screen (screen_leads.tsv, ranked) in the brief's columns, with the
ladder results of the five worked-up leads and the three known families marked."""
from pathlib import Path
import pandas as pd
H = Path(__file__).resolve().parents[1]
T = pd.read_csv(H / "screen_leads.tsv", sep="\t", keep_default_na=False, na_values=[""])
T["ko_name"] = T.ko_name.fillna("")
WK = {("K01465", "Micrarchaeales"): ("L1_dihydroorotase", "yes: Zn ligands H17 H19 K103 H140 H178 and D251 identical (3/3 genes; P05020)", "0.79 / 0.76 (1J79; +ctrl 0.68, -ctrl 0.34)", "SUPPORTED",
                                     "T1 12/18; Zn site intact and in place; fold matches E. coli PyrC; gap in 17/18 genomes that carry ATCase (17/17) and DHODH (17/17)"),
      ("K01673", "Micrarchaeales"): ("L2_beta_carbonic_anhydrase", "yes: Zn ligands C42 D44 H98 C101 identical (3/3; P61517)", "0.45 / 0.51 (1I6P; +ctrl 0.44 / 0.50, -ctrl 0.31)", "SUPPORTED",
                                     "T1 10/18; Zn site intact and in place; fold at the annotated-control level; embedding-only 94%; no pathway context (single-step enzyme)"),
      ("K23370", "Anstonellales"): ("L3_ADP_serine_kinase", "partial: E30 and Mg D69 identical, 12/19 curated residues (Q5JD03; +ctrl same 12/19)", "0.68-0.72 / 0.75-0.78 (5X0B; +ctrl 0.74 / 0.81, -ctrl 0.29)", "SUPPORTED",
                                    "T1 101/101 but 79/101 already carry the KO: assignment holds, only 22 genomes gain the function"),
      ("K07816", "Woesearchaeales"): ("L4_small_alarmone_synthetase", "no: Mg D72, ATP R46 K56 R77 replaced (2-3 of 3 genes; O31611); +ctrl keeps them", "not folded (killed at rung 3)", "KILLED",
                                      "RelA_SpoT relatives that lack the synthetase catalytic residues; label-level function only"),
      ("K01919", "Woesearchaeales"): ("L5_gamma_glutamylcysteine_ligase_GCS2", "untestable: curated record P77213 (putative) lists no residues", "0.66-0.70 / 0.83-0.86 (1R8G; +ctrl 0.70 / 0.88, -ctrl 0.25)", "WEAK",
                                      "fold matches YbdK; rung 3 untestable and the reference function itself is putative; GCS2 otherwise absent from Woesearchaeales (2.1%)")}
KNOWN = {"K01808": "rpib", "K03465": "thy", "K00560": "thy", "K01641": "hmgcs"}
rows = []
for r in T.itertuples():
    w = WK.get((r.ko, r.clade))
    rec = dict(lead=w[0] if w else f"{r.ko}|{r.clade}", family_label=r.family_label, clade=r.clade, genes=r.genes, genomes=r.genomes,
               clade_genomes=r.clade_genomes, median_length=r.median_length, tier_T1=r.tier_T1, support_median=r.support_median,
               catalytic_residues_ok=w[1] if w else "", structure_TM_to_reference=w[2] if w else "",
               function_absent_otherwise=f"{r.gap_genomes}/{r.genomes}", embedding_only_share=r.embedding_only_share,
               verdict=w[3] if w else ("KNOWN (other workstream: " + KNOWN[r.ko] + ")" if r.ko in KNOWN else "UNVERIFIED (screen only)"),
               one_line_reason=w[4] if w else (f"{r.ko_name[:70]}; gap {r.gap_genomes}/{r.genomes} genomes; in-pathway {r.in_context_genomes}"
                                                f" (>=0.5: {r.ctx50_genomes}); clade rate {r.clade_rate_function}"),
               rank=r.rank, ko=r.ko, ko_name=r.ko_name, ec=r.ec, func_source=r.func_source, gap_genomes=r.gap_genomes,
               in_context_genomes=r.in_context_genomes, ctx50_genomes=r.ctx50_genomes, ctx_module=r.ctx_module,
               clade_rate_function=r.clade_rate_function, completeness_median_gap=r.completeness_median_gap,
               clade_completeness_median=r.clade_completeness_median, ilvfmawc_median=r.ilvfmawc_median, enzyme=r.enzyme,
               housekeeping=r.housekeeping, subunit_or_component=r.subunit_or_component, criteria_prefilter=r.criteria_prefilter)
    rows.append(rec)
L = pd.DataFrame(rows)
L.to_csv(H / "leads.tsv", sep="\t", index=False)
print(len(L), "leads;", L.verdict.value_counts().to_dict())
