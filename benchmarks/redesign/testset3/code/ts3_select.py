#!/usr/bin/env python3
"""testset3 selection from GTDB r232 metadata only (rules: testset3/selection_rules.txt, seed 17).

usage: ts3_select.py OUT_DIR THREADS RULES_FILE
Writes OUT_DIR/selection.tsv (one row per selected genome, in selection order), OUT_DIR/eligible_counts.tsv,
OUT_DIR/phylum_table.tsv, OUT_DIR/selection.json. Reads no protein or annotation content.
"""
import hashlib, json, os, sys, time
import duckdb, pandas as pd

OUT, TH, RULES = sys.argv[1], int(sys.argv[2]), sys.argv[3]
ARCH_CAP = int(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4] != "scan" else None
SCAN = len(sys.argv) > 4 and sys.argv[4] == "scan"
os.makedirs(OUT, exist_ok=True)
RD = "/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111"
G = "/groups/banfield/projects/environmental/sr/srvp2020/Jacob/giant_protein_folding/rp16_phylogeny_2026_06/gtdb_giant_scan"
META = {"Bacteria": f"{G}/meta/bac120_metadata.tsv.gz", "Archaea": f"{G}/meta/ar53_metadata.tsv.gz"}
PDIR = {"Bacteria": f"{G}/reps/protein_faa_reps/bacteria", "Archaea": f"{G}/reps/protein_faa_reps/archaea"}
TARGET = {"Bacteria": 700, "Archaea": 300}
SEED = 17
t0 = time.time()
h = lambda s: hashlib.sha256(s.encode()).hexdigest()
con = duckdb.connect(); con.execute(f"PRAGMA threads={TH}")
cols = ("accession, gtdb_representative, gtdb_taxonomy, ncbi_assembly_level, checkm2_completeness, checkm2_contamination, "
        "checkm_completeness, checkm_contamination, contig_count, protein_count, genome_size, ncbi_organism_name")
frames = []
for dom, f in META.items():
    d = con.execute(f"SELECT {cols} FROM read_csv('{f}', delim='\t', header=true, all_varchar=true) "
                    f"WHERE gtdb_representative = 't'").df()
    d["domain"] = dom
    have = set(x[: -len("_protein.faa.gz")] for x in os.listdir(PDIR[dom]) if x.endswith("_protein.faa.gz"))
    d["proteome"] = d.accession.isin(have)
    frames.append(d)
M = pd.concat(frames, ignore_index=True)
for c in ["checkm2_completeness", "checkm2_contamination", "checkm_completeness", "checkm_contamination"]:
    M[c] = pd.to_numeric(M[c], errors="coerce")
for c in ["contig_count", "protein_count", "genome_size"]:
    M[c] = pd.to_numeric(M[c], errors="coerce").astype("Int64")
tx = M.gtdb_taxonomy.str.split(";", expand=True)
for i, r in enumerate(["domain_tax", "phylum", "class", "ord", "family", "genus", "species"]):
    M[r] = tx[i].str[3:]
M["id9"] = M.accession.str.extract(r"GC[AF]_(\d{9})")[0]
info = dict(reps_in_metadata=M.groupby("domain").size().to_dict(),
            reps_with_proteome=M[M.proteome].groupby("domain").size().to_dict())
# exclusions
ex = {}
lut = pd.read_csv(f"{RD}/slice_lut/testset/package_k8/selection.tsv", sep="\t", usecols=["gtdb_id"])
ex["lut_package_k8"] = set(lut.gtdb_id.str.extract(r"GC[AF]_(\d{9})")[0])
dp = pd.read_parquet(f"{RD}/ws_dpann/package/universe/genomes.parquet", columns=["genome", "phylum"])
ex["dpann_universe"] = set(dp.genome.str.extract(r"GC[AF]_(\d{9})")[0].dropna())
sd = pd.read_parquet(f"{RD}/universe/genomes.parquet", columns=["genome"])
ex["spacedust_collection"] = set(sd.genome.str.extract(r"GC[AF]_(\d{9})")[0].dropna())
info["exclusion_list_sizes"] = {k: len(v) for k, v in ex.items()}
info["exclusion_list_rows"] = dict(lut_package_k8=len(lut), dpann_universe=len(dp), spacedust_collection=len(sd))
M["excl"] = ""
for k, v in ex.items():
    M.loc[M.id9.isin(v), "excl"] += k + ";"
sp = M.species.fillna("")
reserved = sp.str.match(r"^(Escherichia coli|Bacillus subtilis)(_[A-Z]+)?$")
M.loc[reserved, "excl"] += "reserved_species;"
info["reserved_species_reps"] = sorted(M.loc[reserved, "species"].unique().tolist())
dpann_phyla = sorted(dp.phylum.dropna().unique().tolist())
info["dpann_phyla"] = dpann_phyla
base = M.proteome & (M.excl == "")
complete = base & (M.ncbi_assembly_level == "Complete Genome") & (M.checkm2_completeness >= 95) & (M.checkm2_contamination <= 5)
fill = base & (M.domain == "Archaea") & (M.ncbi_assembly_level != "Complete Genome") & (M.checkm2_completeness >= 98) \
    & (M.checkm2_contamination <= 2) & (M.contig_count <= 10)
M["tier"] = None
M.loc[complete, "tier"] = "complete"; M.loc[fill, "tier"] = "fill"
# eligibility table (before caps)
ec = M[M.proteome].assign(complete_any=(M.ncbi_assembly_level == "Complete Genome"),
                          complete_q=complete | ((M.ncbi_assembly_level == "Complete Genome") & (M.checkm2_completeness >= 95)
                                                 & (M.checkm2_contamination <= 5)))
ect = ec.groupby("domain").agg(reps=("accession", "count"), complete_genome=("complete_any", "sum"),
                               complete_q95c5=("complete_q", "sum"), eligible_after_exclusion=("tier", lambda v: int((v == "complete").sum())),
                               fill_eligible=("tier", lambda v: int((v == "fill").sum())))
excl_counts = {k: int(((M.ncbi_assembly_level == "Complete Genome") & (M.checkm2_completeness >= 95) & (M.checkm2_contamination <= 5)
                       & M.proteome & M.excl.str.contains(k)).sum()) for k in list(ex) + ["reserved_species"]}
info["complete_q_removed_by_exclusion"] = excl_counts
ect.to_csv(f"{OUT}/eligible_counts.tsv", sep="\t")
M["h_acc"] = [h(f"{SEED}:{a}") for a in M.accession]
M["h_gen"] = [h(f"{SEED}:genus:{g}") for g in M.genus.fillna("")]
M["h_fam"] = [h(f"{SEED}:family:{f}") for f in M.family.fillna("")]


def picking(df, tier):
    if tier == "complete":
        df = df.sort_values(["h_gen", "h_acc"])
    else:
        df = df.sort_values(["h_gen", "checkm2_completeness", "checkm2_contamination", "contig_count", "h_acc"],
                            ascending=[True, False, True, True, True])
    df = df.copy()
    df["rank_in_genus"] = df.groupby("genus", sort=False).cumcount()
    gorder = {g: i for i, g in enumerate(dict.fromkeys(df.genus))}
    df["gpos"] = df.genus.map(gorder)
    return df.sort_values(["rank_in_genus", "gpos"])


def walk(dom, cap):
    out = []
    per_phylum, per_family, n = {}, {}, 0
    for tier in (["complete"] if dom == "Bacteria" else ["complete", "fill"]):
        pool = M[(M.domain == dom) & (M.tier == tier)]
        for fam in pool.drop_duplicates("family").sort_values("h_fam").family:
            if n >= TARGET[dom]:
                return out
            for _, r in picking(pool[pool.family == fam], tier).iterrows():
                if n >= TARGET[dom] or per_family.get(fam, 0) >= 4:
                    break
                if per_phylum.get(r.phylum, 0) >= cap:
                    continue
                per_phylum[r.phylum] = per_phylum.get(r.phylum, 0) + 1
                per_family[fam] = per_family.get(fam, 0) + 1
                n += 1
                out.append((tier, r.phylum))
    return out


if SCAN:
    rows = []
    for cap in list(range(45, 160, 5)) + [10 ** 6]:
        w = walk("Archaea", cap)
        rows.append(dict(cap=cap, archaea=len(w), complete=sum(t == "complete" for t, _ in w), fill=sum(t == "fill" for t, _ in w),
                         max_phylum=max(pd.Series([p for _, p in w]).value_counts()) if w else 0))
    print(pd.DataFrame(rows).to_string()); sys.exit(0)

sel = []
for dom in ["Bacteria", "Archaea"]:
    cap = int(0.15 * TARGET[dom]) if (dom == "Bacteria" or ARCH_CAP is None) else ARCH_CAP
    per_phylum, per_family, n = {}, {}, 0
    for tier in (["complete"] if dom == "Bacteria" else ["complete", "fill"]):
        if n >= TARGET[dom]:
            break
        pool = M[(M.domain == dom) & (M.tier == tier)]
        fam_order = pool.drop_duplicates("family").sort_values("h_fam").family
        for fam in fam_order:
            if n >= TARGET[dom]:
                break
            for _, r in picking(pool[pool.family == fam], tier).iterrows():
                if n >= TARGET[dom] or per_family.get(fam, 0) >= 4:
                    break
                if per_phylum.get(r.phylum, 0) >= cap:
                    continue
                per_phylum[r.phylum] = per_phylum.get(r.phylum, 0) + 1
                per_family[fam] = per_family.get(fam, 0) + 1
                n += 1
                row = dict(order=len(sel), domain=dom, tier=tier, accession=r.accession, genome=r.accession[3:])
                for k in ["phylum", "class", "ord", "family", "genus", "species", "ncbi_assembly_level", "checkm2_completeness",
                          "checkm2_contamination", "checkm_completeness", "checkm_contamination", "contig_count",
                          "protein_count", "genome_size", "gtdb_taxonomy", "ncbi_organism_name"]:
                    row[k] = r[k]
                row["dpann_lineage"] = bool(dom == "Archaea" and r.phylum in dpann_phyla)
                sel.append(row)
S = pd.DataFrame(sel)
S.to_csv(f"{OUT}/selection.tsv", sep="\t", index=False)
pt = S.groupby(["domain", "phylum"]).agg(genomes=("accession", "count"), families=("family", "nunique"), genera=("genus", "nunique"),
                                         complete=("tier", lambda v: int((v == "complete").sum())), fill=("tier", lambda v: int((v == "fill").sum())),
                                         proteins=("protein_count", "sum")).reset_index()
pt["frac_of_domain"] = (pt.genomes / pt.domain.map(S.groupby("domain").size())).round(3)
pt.sort_values(["domain", "genomes"], ascending=[True, False]).to_csv(f"{OUT}/phylum_table.tsv", sep="\t", index=False)
info.update(selected=S.groupby(["domain", "tier"]).size().reset_index().values.tolist(), total=len(S),
            families=S.groupby("domain").family.nunique().to_dict(), genera=S.groupby("domain").genus.nunique().to_dict(),
            phyla=S.groupby("domain").phylum.nunique().to_dict(),
            per_family_hist=S.groupby(["domain", "family"]).size().groupby("domain").value_counts().sort_index().reset_index().values.tolist(),
            dpann_archaea=int(S.dpann_lineage.sum()), dpann_by_phylum=S[S.dpann_lineage].phylum.value_counts().to_dict(),
            proteins_expected=int(S.protein_count.sum()), proteins_by_domain=S.groupby("domain").protein_count.sum().astype(int).to_dict(),
            contigs_total=int(S.contig_count.sum()), max_contigs=int(S.contig_count.max()),
            rules_sha256=hashlib.sha256(open(RULES, "rb").read()).hexdigest(),
            selection_sha256=hashlib.sha256(open(f"{OUT}/selection.tsv", "rb").read()).hexdigest(),
            seconds=round(time.time() - t0, 1), written_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
json.dump(info, open(f"{OUT}/selection.json", "w"), indent=1, default=str)
print(json.dumps(info, indent=1, default=str)); print(ect.to_string()); print(pt.to_string())
