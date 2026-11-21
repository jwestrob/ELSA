#!/usr/bin/env python3
"""Genome context of the ~560-aa HMGS-like family: for every member of the permissive family that holds the 17 leads
(fam:5ad133d455:DAIDBP010000029.1_20, 288 Micrarchaeota genes), list the genes within +-6 positions with Pfam, KO,
CDD best hit, slot label and default-run family id; then count which neighbor families recur, separately for the 17
leads and for the other members. Partner families checked by name: thiolase, DUF35 scaffold (OB_ChsH2_C / zf-ChsH2),
ACP (PP-binding), KS (ketoacyl-synt), ECH (ECH_1/ECH_2), HMG-CoA reductase."""
import duckdb, pandas as pd
pd.set_option("display.width", 320); pd.set_option("display.max_columns", 40); pd.set_option("display.max_colwidth", 60)
pd.set_option("display.max_rows", 2000)
R = "/Users/jacob/Documents/Sandbox/elsa_test/ELSA/benchmarks/redesign"
W = f"{R}/dpann_leads/hmgcs"
FAM = "fam:5ad133d455:DAIDBP010000029.1_20"
con = duckdb.connect()
con.execute("ATTACH '/Users/jacob/Documents/Sandbox/Sharur/Sharur/data/dpann/sharur.duckdb' AS d (READ_ONLY)")
con.execute(f"CREATE VIEW pass AS SELECT * FROM read_parquet('{R}/slot_rescue/pass/dpann_slot_rescue_pass.parquet')")
lead = [l[1:].strip() for l in open(f"{W}/lead17.faa") if l.startswith(">")]
con.execute("CREATE TEMP TABLE mem AS SELECT f.protein_id, pr.contig_id, pr.gene_index, pr.strand, pr.bin_id FROM d.protein_families f JOIN d.proteins pr USING (protein_id) WHERE f.family_id=?", [FAM])
con.execute("CREATE TEMP TABLE lead AS SELECT unnest(?) protein_id", [lead])
nb = con.execute("""
  SELECT m.protein_id anchor, (m.protein_id IN (SELECT protein_id FROM lead)) is_lead, m.bin_id,
         (n.gene_index - m.gene_index) * (CASE WHEN m.strand IN ('-','-1') THEN -1 ELSE 1 END) rel,
         n.protein_id, n.gene_index, n.strand, n.sequence_length len,
         (SELECT string_agg(DISTINCT a.name, ',') FROM d.annotations a WHERE a.protein_id=n.protein_id AND a.source='pfam') pfam,
         (SELECT string_agg(DISTINCT a.accession || ':' || a.name, ',') FROM d.annotations a WHERE a.protein_id=n.protein_id AND a.source IN ('kofam','kegg')) ko,
         (SELECT arg_min(r.name, r.evalue) FROM d.protein_resemblance r WHERE r.protein_id=n.protein_id) cdd,
         (SELECT p.proposed_label_name || '|' || p.vote_share FROM pass p WHERE p.protein_id=n.protein_id) slot,
         (SELECT f.family_id FROM d.protein_families f WHERE f.protein_id=n.protein_id AND f.clustering_run='permissive-e3c50s75-rescue3c') pfam_id
  FROM mem m JOIN d.proteins n ON n.contig_id=m.contig_id AND abs(n.gene_index - m.gene_index) <= 6
  ORDER BY m.protein_id, rel""").df()
nb.to_csv(f"{W}/neighbors_pm6.tsv", sep="\t", index=False)
# print a few leads in full (one per species) and two non-lead GA members
for a in ["OJT94780.1", "MCL5680306.1", "JAJZNR010000018.1_8", "WP_203134402.1"]:
    print(f"\n### {a}")
    print(nb[nb.anchor == a][["rel", "protein_id", "strand", "len", "pfam", "ko", "cdd", "slot", "pfam_id"]].to_string(index=False))
# recurrence of neighbor families by relative position
def recur(sub, label):
    n_anchor = sub.anchor.nunique()
    x = sub[sub.rel != 0].copy()
    x["desc"] = x.pfam.fillna("") + " / " + x.ko.fillna("") + " / " + x.cdd.fillna("")
    g = x.groupby("pfam_id").agg(anchors=("anchor", "nunique"), rel_mode=("rel", lambda s: s.mode().iloc[0]),
                                 rel_set=("rel", lambda s: sorted(set(s))[:6]), len_med=("len", "median"),
                                 desc=("desc", lambda s: s.value_counts().index[0][:90]))
    g = g.sort_values("anchors", ascending=False)
    g["frac"] = (g.anchors / n_anchor).round(2)
    print(f"\n## {label}: {n_anchor} anchors; neighbor families (permissive rescue3c run) in >= 25% of anchors")
    print(g[g.frac >= 0.25].to_string())
    return g
g1 = recur(nb[nb.is_lead], "17 leads")
g2 = recur(nb[~nb.is_lead], "other family members")
g1.to_csv(f"{W}/neighbor_recurrence_leads.tsv", sep="\t")
g2.to_csv(f"{W}/neighbor_recurrence_others.tsv", sep="\t")
# named partner families anywhere within +-6
pat = {"thiolase": "Thiolase|K00626|K00632", "DUF35_scaffold": "OB_ChsH2|zf-ChsH2|DUF35", "ACP": "PP-binding|K02078|Acyl_carrier",
       "KS": "ketoacyl-synt|Ketoacyl-synt", "ECH": "ECH_|K01692|K01715", "HMGCR": "HMG-CoA_red|K00021|K00054",
       "canonical_HMGCS": "HMG_CoA_synt", "GHMP": "GHMP|K00869"}
x = nb[nb.rel != 0].copy()
x["all"] = x.pfam.fillna("") + "|" + x.ko.fillna("") + "|" + x.cdd.fillna("") + "|" + x.slot.fillna("")
rows = []
for k, p in pat.items():
    hit = x[x["all"].str.contains(p, regex=True)]
    rows.append(dict(partner=k, leads_with=hit[hit.is_lead].anchor.nunique(), others_with=hit[~hit.is_lead].anchor.nunique()))
print("\n## named partner families within +-6 genes (anchors with at least one):", nb[nb.is_lead].anchor.nunique(), "leads,", nb[~nb.is_lead].anchor.nunique(), "others")
print(pd.DataFrame(rows).to_string(index=False))
