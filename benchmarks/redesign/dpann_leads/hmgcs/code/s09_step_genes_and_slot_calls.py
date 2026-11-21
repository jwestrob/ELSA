#!/usr/bin/env python3
"""(a) The exact KO/Pfam calls behind each isoprenoid step in the 17 lead genomes (which alternative enzyme fills each
step); (b) slot calls in the pass table whose label corresponds to an isoprenoid/lipid step, DPANN-wide and in genomes
where that step is otherwise blank (any tier); (c) the same for the 17 genomes."""
import duckdb, pandas as pd
pd.set_option("display.width", 320); pd.set_option("display.max_columns", 40); pd.set_option("display.max_colwidth", 70); pd.set_option("display.max_rows", 500)
R = "/Users/jacob/Documents/Sandbox/elsa_test/ELSA/benchmarks/redesign"
W = f"{R}/dpann_leads/hmgcs"
con = duckdb.connect()
con.execute("ATTACH '/Users/jacob/Documents/Sandbox/Sharur/Sharur/data/dpann/sharur.duckdb' AS d (READ_ONLY)")
con.execute(f"CREATE VIEW pass AS SELECT * FROM read_parquet('{R}/slot_rescue/pass/dpann_slot_rescue_pass.parquet')")
con.execute(f"CREATE VIEW ver AS SELECT * FROM read_parquet('{R}/slot_rescue/verify/verification_per_protein.parquet')")
G = pd.read_parquet(f"{W}/genome_steps.parquet")
lead_bins = pd.read_csv(f"{R}/dpann_leads/candidates_all.tsv", sep="\t").query("label=='HMG_CoA_synt_C'").bin_id.tolist()
KOS = ["K00626", "K01641", "K00021", "K00054", "K00869", "K18689", "K18690", "K22813", "K17942", "K25517", "K09128", "K25518", "K03186", "K00938", "K13273", "K01597",
       "K06981", "K01823", "K13787", "K13789", "K00804", "K00795", "K00096", "K17104", "K17105", "K19664", "K17830"]
print("# (a) KO calls in the 17 lead genomes (genomes with the KO, and protein lengths)")
a = con.execute("""SELECT a.accession, a.source, count(DISTINCT pr.bin_id) genomes, median(pr.sequence_length) len_med, string_agg(DISTINCT a.name, ' | ') nm
                   FROM d.annotations a JOIN d.proteins pr USING (protein_id) WHERE pr.bin_id IN (SELECT unnest(?)) AND a.accession IN (SELECT unnest(?))
                   GROUP BY 1,2 ORDER BY 1""", [lead_bins, KOS]).df()
print(a.to_string(index=False))
# Pfam families relevant to steps present in the 17 genomes
PF = ["HMG_CoA_synt_N", "HMG_CoA_synt_C", "HMG-CoA_red", "Thiolase_N", "Thiolase_C", "GHMP_kinases_N", "GHMP_kinases_C", "AA_kinase", "IPP_isom_2", "FMN_dh", "NUDIX",
      "polyprenyl_synt", "PcrB", "UbiA", "CTP_transf_1", "Fe-ADH", "Fe-ADH_2", "OB_ChsH2_C", "zf-ChsH2", "Pyr_redox_2", "FAD_binding_3"]
print("\n# Pfam calls (GA) in the 17 lead genomes")
print(con.execute("""SELECT a.name, count(DISTINCT pr.bin_id) genomes, count(*) genes, median(pr.sequence_length) len_med FROM d.annotations a JOIN d.proteins pr USING (protein_id)
                     WHERE pr.bin_id IN (SELECT unnest(?)) AND a.source='pfam' AND a.name IN (SELECT unnest(?)) GROUP BY 1 ORDER BY 1""", [lead_bins, PF]).df().to_string(index=False))
# (b) slot-call labels mapping to steps
STEPLAB = {"2_HMGCS": ["PF01154", "PF08540", "K01641"], "3_HMGCR": ["PF00368", "K00021", "K00054"], "1_thiolase": ["K00626", "PF00108", "PF02803"],
           "4_MVK/GHMP": ["K00869", "K18689", "K18690", "PF00288", "PF08544"], "5_decarb": ["K17942", "K22813", "K01597", "K00938", "K13273", "K25517", "K09128", "K25518"],
           "6_IPK": ["K06981"], "7_IDI": ["K01823"], "8_GGPS": ["K13787", "K13789", "K00804", "K00795"], "9_G1PDH": ["K00096"], "10_GGGPS": ["K17104", "PF01884"],
           "11_DGGGPS": ["K17105"], "12_CarS": ["K19664"], "13_GGR": ["K17830"], "DUF35": ["PF01796", "PF12172"]}
lab2step = {l: s for s, ls in STEPLAB.items() for l in ls}
p = con.execute("""SELECT p.protein_id, p.gid, pr.bin_id, p.proposed_label_positional_homology lab, p.proposed_label_name lname, p.vote_share, p.votes_for, p.supporting_genomes,
                   p.alignment_lists_same_label, v.tier, pr.sequence_length len FROM pass p JOIN d.proteins pr USING (protein_id) LEFT JOIN ver v USING (protein_id)
                   WHERE p.proposed_label_positional_homology IN (SELECT unnest(?))""", [list(lab2step)]).df()
p["step"] = p.lab.map(lab2step)
p = p.join(G[["phylum", "fam", "genus", "completeness", "2_HMGCS_canonical", "2b_HMGS_like_560fam", "3_HMGCR", "1_thiolase_AACT", "MVK_or_alt", "decarb_any", "6_IPK", "7_IDI", "8_GGPS", "lipid_specific_any"]], on="bin_id")
stepcol = {"2_HMGCS": "2_HMGCS_canonical", "3_HMGCR": "3_HMGCR", "1_thiolase": "1_thiolase_AACT", "4_MVK/GHMP": "MVK_or_alt", "5_decarb": "decarb_any", "6_IPK": "6_IPK", "7_IDI": "7_IDI", "8_GGPS": "8_GGPS"}
p["step_blank_in_genome"] = [(not r[stepcol[r.step]]) if r.step in stepcol else None for _, r in p.iterrows()]
print("\n# (b) slot calls with step labels, DPANN-wide: genes by step, label, phylum, tier; and whether the genome lacks the step otherwise")
print(p.groupby(["step", "lname", "phylum"]).agg(genes=("protein_id", "size"), genomes=("bin_id", "nunique"), blank_genomes=("step_blank_in_genome", "sum"),
      T1=("tier", lambda s: (s == "T1").sum()), emb_only=("alignment_lists_same_label", lambda s: (s != "yes").sum()), len_med=("len", "median")).to_string())
p.to_csv(f"{W}/slot_calls_isoprenoid_steps.tsv", sep="\t", index=False)
print("\n# (c) slot calls with step labels inside the 17 lead genomes")
print(p[p.bin_id.isin(lead_bins)][["protein_id", "bin_id", "step", "lname", "vote_share", "votes_for", "tier", "len", "alignment_lists_same_label", "step_blank_in_genome"]].to_string(index=False))
# all slot calls in the 17 genomes (any label), to see what else the pass table proposes there
print("\n# every slot call in the 17 lead genomes: label counts")
q = con.execute("""SELECT p.proposed_label_name lname, p.proposed_label_kind kind, count(*) genes, count(DISTINCT pr.bin_id) genomes, median(pr.sequence_length) len_med
                   FROM pass p JOIN d.proteins pr USING (protein_id) WHERE pr.bin_id IN (SELECT unnest(?)) GROUP BY 1,2 ORDER BY genomes DESC, genes DESC""", [lead_bins]).df()
print(len(q), "labels;", q.genes.sum(), "genes")
print(q.head(60).to_string(index=False))
