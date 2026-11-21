#!/usr/bin/env python
"""Write top20_maps.txt from families.tsv and data/core_clusters.tsv (one line per core gene)."""
import pandas as pd

W = "/Users/jacob/Documents/Sandbox/elsa_test/ELSA/benchmarks/redesign/dpann_leads/large_loci_dpann"
F = pd.read_csv(W + "/families.tsv", sep="\t", keep_default_na=False)
CL = pd.read_csv(W + "/data/core_clusters.tsv", sep="\t", keep_default_na=False)
MANUAL = {
    64: "textbook: Mbh-type membrane-bound [NiFe]-hydrogenase / Mrp antiporter operon (HydDB call on the catalytic subunit in 74% of members); 109-aa 2-TM gene without annotation",
    77: "housekeeping mix: ScpAB condensin, Lon-like AAA+, TldD/PmbA pair, archaeosortase-like, ERCC4, DEAD helicase; one 280-aa unannotated gene",
    59: "consistent with a metallocluster module: Nitro_FeMo-Co (NifX/NafY-type) + DUF134 + DUF5320 + COG1149 MinD-type ATPase + CcdA (UNVERIFIED); 3 core genes only",
    50: "textbook: archaeal N-glycosylation (AglB/STT3 oligosaccharyltransferase, GT2, DUF2304)",
    29: "Orc1/Cdc6 + PolD (DP1, DP2) replication-origin module plus 3 conserved unannotated genes (folds: SecB-like chaperone, two winged HTH)",
    36: "archaeal type IV pilus locus (T2SSE/T2SSF + pilin-like unannotated genes)",
    15: "archaeal type IV pilus locus (T2SSE/T2SSF + class III signal pilins)",
    17: "Micrarchaeota: topo VI B-anchored locus with ~795-aa membrane-anchored A2M-like protein + DUF87 (HerA/VirB4-family) ATPase + 5 unannotated membrane/soluble genes",
    0: "archaeal type IV pilus loci (T2SSE + arCOG01814 platform), loose family (5 core genes in 1,309 loci)",
    75: "textbook: LysW-mediated alpha-aminoadipate lysine (or arginine) biosynthesis (LysW, LysX, LysZ, LysY, LysJ, homoaconitase, HCS, HICDH)",
    49: "textbook: GatCAB / GatDE tRNA-dependent amidotransferase with AspRS-type synthetase",
    74: "archaeal type IV pilus locus",
    33: "archaeal type IV pilus locus (Micrarchaeales) with TrkA_N and M50 peptidase",
    43: "Pacearchaeales: AP_endonuc_2 (IolE-type sugar isomerase) + Polbeta-type NTase + 3 unannotated genes that fold poorly (pLDDT 37-72)",
    96: "MOBILE: head-tail provirus (HK97 capsid, connector, tail terminator, tail tube, tape measure; folds confirm), 16/23 on contig edges",
    72: "textbook: NAD salvage (PncA, PncB, NadM, NadE, NadK, NUDIX); pathway complete, 327-aa unannotated gene is extra",
    198: "Aenigmatarchaeota thiol / Fe-S module (5 loci, 3 groups)",
    39: "S-layer protein + GT2 glycosylation + TruD",
    118: "textbook system in an unexpected clade: gas vesicle gvp cluster (GvpA, F/L, G, K, N, O, I-like) in 15 DPANN genomes",
    150: "Aenigmatarchaeota: 4 unannotated genes + two ParA/MinD ATPases + ArsR; all 10 members on contig edges",
}
F = F[F["rank"] != ""].copy()
F["rank"] = F["rank"].astype(float)
F = F.sort_values("rank").head(20)
lines = ["TOP 20 LARGE-LOCUS FAMILIES (not KNOWN; >= 3 genome groups, >= 2 orders, >= 5 loci), ranked by the predeclared score",
         "One line per core gene (present in >= 50% of member loci), in consensus order. strand = relative to the oriented",
         "locus (agree = share of members on that strand); aa = median length; annotation = plurality Pfam architecture,",
         "else KO, else CDD best hit, else DefenseFinder / TXSScan profile (OBSERVED rows, 'resembles' for CDD), share of",
         "the cluster's genes carrying it; unann = share with no row from any source; slot = position-based label of the",
         "slot-rescue pass and its tier (hypothesis from bonded partners); hyd = median fraction ILVFMAWC; tm = median",
         "TMHMM helices; pres = share of member loci carrying the gene. [class] = automatic class (amended rule, predeclared.json\n" "amendment_3); 'reading' = manual reading of the composition (OBSERVED domains; names are 'consistent with').",
         ""]
for r in F.itertuples():
    lines.append(f"#{int(r.rank)} family {r.family} [{r.cls}] loci {r.loci}, genomes {r.genomes}, genome groups {r.groups}, "
                 f"phyla {r.phyla}, orders {r.orders}; size median {r.size_median:g} ({r.size_min}-{r.size_max}); core {r.core} "
                 f"of {r.gene_clusters} clusters; order conservation {r.order_cons}; unannotated {r.unannotated_share}; "
                 f"edge rate {r.edge_rate}; completeness median {r.completeness_median}")
    lines.append(f"   reading: {MANUAL.get(r.family, '-')}")
    lines.append(f"   phyla: {r.phyla_list}; top order {r.top_order}; curated overlap {r.curated_loci_share} "
                 f"({r.curated_top or 'none'}); mobile genes {r.mobile_gene_share}; example {r.example_locus}")
    sub = CL[CL.family == r.family].sort_values("core_pos")
    for c in sub.itertuples():
        ann = c.annotation if c.annotation != "none" else "none"
        slot = f"{c.slot_label} {c.slot_tier}".strip() or "-"
        lines.append(f"   {c.core_pos:>3} {c.strand} ({c.strand_agree:.2f}) {c.median_aa:>5} aa  pres {c.presence:.2f}  "
                     f"unann {c.unannotated_share:.2f}  hyd {c.hyd:.2f} tm {c.n_tm:g}  {ann[:70]} ({c.annotation_share:.2f})"
                     f"  slot: {slot[:40]}")
    lines.append("")
FF = pd.read_csv(W + "/families.tsv", sep="\t", keep_default_na=False)
FF["manual_reading"] = FF.family.map(MANUAL).fillna("")
FF.to_csv(W + "/families.tsv", sep="\t", index=False)
open(W + "/top20_maps.txt", "w").write("\n".join(lines) + "\n")
print(len(lines), "lines")
