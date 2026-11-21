#!/usr/bin/env python3
"""Step after lead 1 (HMG-CoA -> mevalonate). (a) Co-occurrence statistics of the ~560-aa HMGS-like family with the
canonical HMGCS and with HMGCR in Micrarchaeota (Fisher exact tests). (b) Phylogenetic profiling inside
f__Micrarchaeaceae (completeness >= 70, genomes that carry a mevalonate-consuming step: MVK or the mevalonate-3-kinase
route): protein families (permissive rescue3c run) present in most HMGCR-blank genomes and rare in HMGCR-carrying
genomes, as candidates for a non-homologous replacement; with their annotations and length. (c) Leucine-catabolism KOs
that also produce HMG-CoA, in the 17 lead genomes."""
import duckdb, pandas as pd, numpy as np
from scipy.stats import fisher_exact
pd.set_option("display.width", 320); pd.set_option("display.max_columns", 30); pd.set_option("display.max_colwidth", 80); pd.set_option("display.max_rows", 200)
R = "/Users/jacob/Documents/Sandbox/elsa_test/ELSA/benchmarks/redesign"
W = f"{R}/dpann_leads/hmgcs"
G = pd.read_parquet(f"{W}/genome_steps.parquet")
mic = G[G.phylum == "p__Micrarchaeota"]
for lab, df in [("Micrarchaeota all", mic), ("Micrarchaeota comp>=70", mic[mic.completeness >= 70]), ("Micrarchaeaceae comp>=70", mic[(mic.fam == "f__Micrarchaeaceae") & (mic.completeness >= 70)])]:
    t = pd.crosstab(df["2b_HMGS_like_560fam"], df["2_HMGCS_canonical"])
    o, p = fisher_exact(t.values)
    exp = df["2b_HMGS_like_560fam"].sum() * df["2_HMGCS_canonical"].sum() / len(df)
    t2 = pd.crosstab(df["2b_HMGS_like_560fam"], df["3_HMGCR"])
    o2, p2 = fisher_exact(t2.values)
    hm560 = df[df["2b_HMGS_like_560fam"]]["3_HMGCR"].mean(); hmcan = df[df["2_HMGCS_canonical"]]["3_HMGCR"].mean()
    print(f"{lab}: n={len(df)}; 560fam={df['2b_HMGS_like_560fam'].sum()}, canonical={df['2_HMGCS_canonical'].sum()}, both={(df['2b_HMGS_like_560fam'] & df['2_HMGCS_canonical']).sum()} (expected {exp:.1f}), Fisher OR={o:.3f} p={p:.2e}; "
          f"HMGCR in 560fam genomes {df[df['2b_HMGS_like_560fam']]['3_HMGCR'].sum()}/{df['2b_HMGS_like_560fam'].sum()} ({hm560:.2f}) vs canonical genomes {df[df['2_HMGCS_canonical']]['3_HMGCR'].sum()}/{df['2_HMGCS_canonical'].sum()} ({hmcan:.2f})")
# genera of 560fam genomes with HMGCR
m70 = mic[mic.completeness >= 70]
x = m70[m70["2b_HMGS_like_560fam"]].groupby(["fam", "genus"]).agg(n=("3_HMGCR", "size"), HMGCR=("3_HMGCR", "sum"), M3K=("4alt_M3K_route", "sum"), MVK=("4_MVK", "sum"), canon=("2_HMGCS_canonical", "sum")).sort_values("n", ascending=False)
print("\n# 560-family genomes (comp >= 70) by genus: HMGCR / M3K route / MVK / canonical HMGCS")
print(x.head(30).to_string())
# (b) profiling
con = duckdb.connect()
con.execute("ATTACH '/Users/jacob/Documents/Sandbox/Sharur/Sharur/data/dpann/sharur.duckdb' AS d (READ_ONLY)")
S = mic[(mic.fam == "f__Micrarchaeaceae") & (mic.completeness >= 70) & (mic["4_MVK"] | mic["4alt_M3K_route"])].copy()
plus, minus = S.index[S["3_HMGCR"]].tolist(), S.index[~S["3_HMGCR"]].tolist()
print(f"\n# profiling set: Micrarchaeaceae comp>=70 with MVK or M3K route: HMGCR+ {len(plus)}, HMGCR- {len(minus)}")
con.execute("CREATE TEMP TABLE gp AS SELECT unnest(?) bin_id, true plus", [plus])
con.execute("INSERT INTO gp SELECT unnest(?) bin_id, false plus", [minus])
F = con.execute("""SELECT f.family_id, sum(CASE WHEN g.plus THEN 1 ELSE 0 END) n_plus, sum(CASE WHEN g.plus THEN 0 ELSE 1 END) n_minus FROM
                   (SELECT DISTINCT family_id, bin_id FROM d.protein_families WHERE clustering_run='permissive-e3c50s75-rescue3c') f JOIN gp g USING (bin_id) GROUP BY 1""").df()
F["prev_plus"] = F.n_plus / len(plus); F["prev_minus"] = F.n_minus / len(minus)
F["diff"] = F.prev_minus - F.prev_plus
top = F[(F.prev_minus >= 0.6) & (F.prev_plus <= 0.2)].sort_values("diff", ascending=False).head(25)
def desc(fid):
    r = con.execute("""SELECT median(pr.sequence_length) len,
        (SELECT string_agg(nm, '; ') FROM (SELECT a.name nm, count(*) c FROM d.annotations a JOIN d.protein_families f2 USING (protein_id) WHERE f2.family_id=? AND a.source IN ('pfam','kofam') GROUP BY 1 ORDER BY c DESC LIMIT 3)) ann,
        (SELECT string_agg(nm, '; ') FROM (SELECT r.name nm, count(*) c FROM d.protein_resemblance r JOIN d.protein_families f2 USING (protein_id) WHERE f2.family_id=? GROUP BY 1 ORDER BY c DESC LIMIT 2)) cdd
        FROM d.protein_families f JOIN d.proteins pr USING (protein_id) WHERE f.family_id=?""", [fid, fid, fid]).fetchone()
    return pd.Series(dict(len_med=r[0], annot=r[1], cdd=r[2]))
top = pd.concat([top.reset_index(drop=True), top.family_id.apply(desc).reset_index(drop=True)], axis=1)
print("# families enriched in HMGCR-blank genomes (prev_minus >= 0.6, prev_plus <= 0.2)")
print(top.round(2).to_string(index=False))
top.to_csv(f"{W}/hmgcr_profiling_candidates.tsv", sep="\t", index=False)
# (c) leucine catabolism route to HMG-CoA in the 17 genomes
lead_bins = pd.read_csv(f"{R}/dpann_leads/candidates_all.tsv", sep="\t").query("label=='HMG_CoA_synt_C'").bin_id.tolist()
print("\n# leucine-catabolism / HMG-CoA lyase KOs in the 17 lead genomes")
print(con.execute("""SELECT a.accession, count(DISTINCT pr.bin_id) genomes FROM d.annotations a JOIN d.proteins pr USING (protein_id)
                     WHERE pr.bin_id IN (SELECT unnest(?)) AND a.accession IN ('K01968','K01969','K05607','K01640','K00249','K00253') GROUP BY 1""", [lead_bins]).df())
