#!/usr/bin/env python3
"""Tier selection and dereplication over the reconciled SR-VP genome register (workstream srvp_bins).

usage: 02_select.py OUT_DIR
Rules (NOTEBOOK.txt, written before counting):
  universe U   joint dRep input set (joint_dereplication_v1/input_manifest.tsv): sequence-verified SR-VP soil bins whose
               CheckM2 estimate is linked by identifier and measured length; completeness >= 50, contamination < 10.
  primary      completeness >= 90 and contamination <= 5
  secondary    completeness >= 70 and contamination <= 10 (cumulative 'ge70' and 'secondary_only' reported)
  derep        joint 95% ANI clusters (job 17839); pick = dRep representative when it passes the tier, else the tier member
               with the best completeness - 5*contamination + 0.5*log10(N50) (ties: fewer contigs, then genome_id).
Writes small tables (no sequences): selection_undereplicated.tsv, selection_derep_<tier>.tsv, counts.json and per-axis
count tables.
"""
import json, math, os, sys
import numpy as np, pandas as pd

F = "/groups/banfield/projects/environmental/sr/luis/srvp-all/srvp-fina"
G = f"{F}/03_assemblies_and_genomes"
MAN = f"{G}/joint_dereplication_v1/input_manifest.tsv"
MEM = f"{G}/joint_dereplication_results_v1/cluster_membership.tsv"
REG = f"{G}/genome_reconciliation_v1/master_genomes.tsv"
R232 = f"{F}/05_taxonomy_and_community/r232_integrated_results_v1/genome_master_r232.tsv"
OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)

man = pd.read_csv(MAN, sep="\t", dtype=str, keep_default_na=False)
mem = pd.read_csv(MEM, sep="\t", dtype=str, keep_default_na=False)
reg = pd.read_csv(REG, sep="\t", dtype=str, keep_default_na=False)
r232 = pd.read_csv(R232, sep="\t", dtype=str, keep_default_na=False)
for df in (man, reg):
    for c in ("completeness", "contamination", "genome_bp", "contigs"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
man["n50_bp"] = pd.to_numeric(man["n50_bp"], errors="coerce")
assert man.master_id.is_unique and len(man) == 6647, len(man)
assert set(mem.master_id) == set(man.master_id), "membership and manifest disagree"

m = man.merge(mem[["master_id", "cluster_id", "representative", "representative_master_id"]], on="master_id", how="left")
m["is_rep"] = m.representative == "True"
# r232 taxonomy (representatives only); member-level r220 from the manifest
r = r232[["master_id", "domain_r232", "phylum_r232", "class_r232", "order_r232", "family_r232", "genus_r232",
          "species_r232", "classification_r232", "gtdb_has_warning", "gtdb_translation_table"]]
m = m.merge(r, on="master_id", how="left")
crep = r232[["cluster_id", "domain_r232", "phylum_r232", "classification_r232"]].rename(
    columns={"domain_r232": "cluster_rep_domain_r232", "phylum_r232": "cluster_rep_phylum_r232",
             "classification_r232": "cluster_rep_classification_r232"})
m = m.merge(crep, on="cluster_id", how="left")


def strip(s):
    return s.split("__", 1)[1] if isinstance(s, str) and "__" in s else (s or "")


# phylum used for counting: r232 of the bin itself if it is a representative, else its own r220 phylum, else the
# cluster representative's r232 phylum (marked as context)
m["phylum_own"] = np.where(m.phylum_r232.fillna("") != "", m.phylum_r232.fillna("").map(strip), m.phylum)
m["phylum_source"] = np.where(m.phylum_r232.fillna("") != "", "r232_own",
                              np.where(m.phylum != "", "r220_own", "r232_cluster_rep"))
# counting uses one release for every bin: the r232 call of its 95% ANI cluster representative (GTDB-Tk 2.7.2,
# job 18127, all 2,392 representatives classified); the bin's own label is kept in phylum_own / phylum_source
m["phylum_count"] = m.cluster_rep_phylum_r232.fillna("").map(strip)
m["domain_count"] = m.cluster_rep_domain_r232.fillna("").map(strip)
m["phylum_own_vs_rep_differ"] = (m.phylum_own != "") & (m.phylum_own != m.phylum_count)
m["primary"] = (m.completeness >= 90) & (m.contamination <= 5)
m["ge70"] = (m.completeness >= 70) & (m.contamination <= 10)
m["secondary_only"] = m.ge70 & ~m.primary
m["multi_contig"] = m.contigs >= 2
m["score"] = m.completeness - 5 * m.contamination + 0.5 * np.log10(m.n50_bp.clip(lower=1))
m["sample"] = m.sample_identifier.replace("", "unresolved")

cols = ["master_id", "genome_id", "cluster_id", "is_rep", "representative_master_id", "platform", "cohort", "sample",
        "sample_key", "domain_count", "phylum_count", "phylum_source", "classification", "taxonomy_release",
        "classification_r232", "cluster_rep_classification_r232", "phylum_own", "phylum_own_vs_rep_differ", "completeness", "contamination", "genome_bp",
        "contigs", "n50_bp", "multi_contig", "reported_circular", "score", "primary", "ge70", "secondary_only",
        "preferred_path", "input_path", "file_sha256", "sequence_sha256", "quality_source_path", "gtdb_translation_table",
        "gtdb_has_warning"]
m[cols].sort_values("master_id").to_csv(f"{OUT}/universe_U.tsv", sep="\t", index=False)


def pick(df, tier):
    t = df[df[tier]].copy()
    t["_rep_first"] = ~t.is_rep
    t = t.sort_values(["cluster_id", "_rep_first", "score", "contigs", "genome_id"],
                      ascending=[True, True, False, True, True])
    # rep first only when it passes; otherwise best score
    return t.drop_duplicates("cluster_id").drop(columns="_rep_first")


counts = {"universe_U": len(m), "clusters_total": int(m.cluster_id.nunique())}
sel = {}
for tier in ("primary", "ge70", "secondary_only"):
    und = m[m[tier]]
    d = pick(m, tier) if tier != "secondary_only" else None
    if tier == "secondary_only":
        # clusters reaching >=70/10 but with no primary member; pick by the ge70 rule
        prim_clusters = set(m[m.primary].cluster_id)
        d = pick(m, "ge70")
        d = d[~d.cluster_id.isin(prim_clusters)]
    sel[tier] = (und, d)
    counts[tier] = dict(
        undereplicated=len(und), dereplicated=len(d),
        derep_pick_is_drep_rep=int(d.is_rep.sum()),
        undereplicated_by_platform=und.platform.value_counts().to_dict(),
        dereplicated_by_platform=d.platform.value_counts().to_dict(),
        undereplicated_by_domain=und.domain_count.value_counts().to_dict(),
        dereplicated_by_domain=d.domain_count.value_counts().to_dict(),
        undereplicated_multi_contig=int(und.multi_contig.sum()),
        dereplicated_multi_contig=int(d.multi_contig.sum()),
        undereplicated_samples=int(und["sample"].nunique()),
        dereplicated_phyla=int(d.phylum_count.replace("", np.nan).nunique()),
        dereplicated_phylum_source=d.phylum_source.value_counts().to_dict(),
        undereplicated_phyla=int(und.phylum_count.replace("", np.nan).nunique()),
        undereplicated_own_phylum_differs_from_cluster_rep=int(und.phylum_own_vs_rep_differ.sum()),
        dereplicated_bp=int(d.genome_bp.sum()), undereplicated_bp=int(und.genome_bp.sum()),
        dereplicated_contigs=int(d.contigs.sum()), undereplicated_contigs=int(und.contigs.sum()),
        dereplicated_contigs_median=float(d.contigs.median()),
        dereplicated_completeness_median=float(d.completeness.median()),
    )
    und[cols].sort_values("master_id").to_csv(f"{OUT}/selection_undereplicated_{tier}.tsv", sep="\t", index=False)
    d[cols].sort_values("cluster_id").to_csv(f"{OUT}/selection_derep_{tier}.tsv", sep="\t", index=False)

# per-phylum and per-sample tables
rows = []
for tier in ("primary", "secondary_only", "ge70"):
    und, d = sel[tier]
    for lvl, df in (("undereplicated", und), ("dereplicated", d)):
        for (dom, ph), n in df.groupby(["domain_count", "phylum_count"]).size().items():
            rows.append((tier, lvl, dom, ph or "unassigned", n))
pd.DataFrame(rows, columns=["tier", "level", "domain", "phylum", "n"]).to_csv(f"{OUT}/counts_by_phylum.tsv", sep="\t", index=False)
rows = []
for tier in ("primary", "secondary_only", "ge70"):
    und, d = sel[tier]
    for lvl, df in (("undereplicated", und), ("dereplicated", d)):
        for (s, p), n in df.groupby(["sample", "platform"]).size().items():
            rows.append((tier, lvl, s, p, n))
pd.DataFrame(rows, columns=["tier", "level", "sample", "platform", "n"]).to_csv(f"{OUT}/counts_by_sample.tsv", sep="\t", index=False)

# bins outside U that would pass a tier on a recorded (unlinked or boundary) quality estimate
reg_seq = reg[reg.sequence_status == "Sequence verified"]
outside = reg_seq[~reg_seq.master_id.isin(m.master_id)]
counts["register_rows"] = len(reg)
counts["register_sequence_verified"] = len(reg_seq)
counts["outside_U_with_quality_primary"] = int(((outside.completeness >= 90) & (outside.contamination <= 5)).sum())
counts["outside_U_with_quality_ge70"] = int(((outside.completeness >= 70) & (outside.contamination <= 10)).sum())
counts["outside_U_contamination_eq_10_ge70"] = int(((outside.completeness >= 70) & (outside.contamination == 10)).sum())
counts["outside_U_ge70_by_link_basis"] = outside[(outside.completeness >= 70) & (outside.contamination <= 10)
                                                 ].quality_link_basis.value_counts().to_dict()
counts["outside_U_ge70_by_platform"] = outside[(outside.completeness >= 70) & (outside.contamination <= 10)
                                               ].platform.value_counts().to_dict()
counts["outside_U_ge70_shorter_50kb"] = int(((outside.completeness >= 70) & (outside.contamination <= 10)
                                              & (outside.genome_bp < 50000)).sum())
json.dump(counts, open(f"{OUT}/counts.json", "w"), indent=1, default=int)
print(json.dumps(counts, indent=1, default=int))
