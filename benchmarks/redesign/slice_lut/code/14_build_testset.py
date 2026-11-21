#!/usr/bin/env python3
"""lut test set: counts for candidate sizes, and the package for one selection (workstream slice_lut, part 2).

usage: 14_build_testset.py count SELECTION_TSV [SELECTION_TSV ...] --out JSON --threads N
       14_build_testset.py build SELECTION_TSV --pkg DIR --threads N
Genes come from the part-1 Parquet parts (GTDB r232 Prodigal proteomes, one row per protein with sequence) of the
three scanned slices; numbering follows the DPANN package rules: genome_code = rank of genome accession, contig_code =
rank of (genome, contig), gid = rank over (genome_code, contig_code, start, end, protein_id), position_index = rank on
the contig. A selected genome's genes, coordinates and protein_ids are identical to the part-1 universes.
build writes: universe/{genes,genomes,contigs,gene_flags,contig_extra}.parquet, proteins_gid.faa, selection.tsv,
gold/lut_instances.parquet + gold/lut_members.parquet (part-1 main and strict loci mapped to package gids),
split.parquet (odd genome_code = tuning, even = test), receipts/build.json.
"""
import argparse, hashlib, json, os, sys, time
import duckdb, pandas as pd, pyarrow.parquet as pq

S = "/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/slice_lut"
SRC = {"order": f"{S}/work/rhizobiales", "alpha_wide": f"{S}/work/alpha_wide", "beyond": f"{S}/work/beyond"}
GTAX = "/groups/banfield/projects/environmental/sr/srvp2020/Jacob/Chloroflexi_2026/provenance/globdb/globdb_r232_taxonomy.tsv.gz"
CM2 = "/groups/banfield/projects/environmental/sr/srvp2020/Jacob/Chloroflexi_2026/provenance/globdb/globdb_r232_checkm2.tsv.gz"
ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=["count", "build"])
ap.add_argument("selection", nargs="+")
ap.add_argument("--out"); ap.add_argument("--pkg"); ap.add_argument("--threads", type=int, required=True)
a = ap.parse_args()
t0 = time.time()
con = duckdb.connect()
con.execute(f"PRAGMA threads={a.threads}; SET preserve_insertion_order=false")
parts = ", ".join(f"'{d}/parts/*.parquet'" for d in SRC.values())
con.execute(f"CREATE VIEW allp AS SELECT * FROM read_parquet([{parts}], union_by_name=true)")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(16 << 20), b""):
            h.update(b)
    return h.hexdigest()


if a.mode == "count":
    out = {}
    for sel in a.selection:
        con.execute(f"CREATE OR REPLACE TABLE sel AS SELECT gtdb_id FROM read_csv('{sel}', delim='\t', header=true, all_varchar=true)")
        r = con.execute("""SELECT count(DISTINCT p.gtdb_id), count(*), sum(p.length), count(DISTINCT md5(p.seq)),
                                  sum(p.length) FILTER (WHERE rn = 1)
                           FROM (SELECT *, row_number() OVER (PARTITION BY md5(seq)) AS rn FROM allp
                                 WHERE gtdb_id IN (SELECT gtdb_id FROM sel)) p""").fetchone()
        n_sel = con.execute("SELECT count(*) FROM sel").fetchone()[0]
        out[os.path.basename(sel)] = dict(selected=n_sel, genomes_found=r[0], proteins=r[1], residues=int(r[2]),
                                          unique_sequences=r[3], unique_residues=int(r[4]))
        print(os.path.basename(sel), out[os.path.basename(sel)], flush=True)
    json.dump(out, open(a.out, "w"), indent=1)
    sys.exit(0)

# ------------------------------------------------------------------------------------------------ build
P = a.pkg
for d in ("universe", "gold", "receipts"):
    os.makedirs(f"{P}/{d}", exist_ok=True)
con.execute(f"SET temp_directory='{P}/.duck_tmp'")
sel = pd.read_csv(a.selection[0], sep="\t", dtype=str)
sel.to_csv(f"{P}/selection.tsv", sep="\t", index=False)
con.register("sel_df", sel)
con.execute("CREATE TABLE sel AS SELECT * FROM sel_df")
con.execute("""CREATE TABLE p AS SELECT a.* EXCLUDE (seq), a.genome || '|' || a.prodigal_id AS protein_id
               FROM allp a WHERE a.gtdb_id IN (SELECT gtdb_id FROM sel)""")
con.execute("CREATE TABLE gm AS SELECT genome, row_number() OVER (ORDER BY genome) - 1 AS genome_code FROM (SELECT DISTINCT genome FROM p)")
con.execute("""CREATE TABLE cg AS SELECT genome, contig, row_number() OVER (ORDER BY genome, contig) - 1 AS contig_code
               FROM (SELECT DISTINCT genome, contig FROM p)""")
con.execute("""CREATE TABLE u AS
    SELECT row_number() OVER (ORDER BY gm.genome_code, cg.contig_code, p.start, p."end", p.protein_id) - 1 AS gid,
           p.protein_id, p.genome, gm.genome_code::INTEGER AS genome_code, p.contig, cg.contig_code,
           row_number() OVER (PARTITION BY cg.contig_code ORDER BY p.start, p."end", p.protein_id) - 1 AS position_index,
           p.strand::TINYINT AS strand, p.length::BIGINT AS length, p.start::BIGINT AS start, p."end"::BIGINT AS "end",
           p.gtdb_id, p.prodigal_id, p.prodigal_n, p.file_index, p.partial, p.start_type, p.rbs_motif, p.gc_cont, p.internal_stops
    FROM p JOIN gm USING (genome) JOIN cg USING (genome, contig)""")
q = lambda s: con.execute(s).fetchone()
qc = dict(selected=len(sel))
qc["genes"], qc["genomes"], qc["contigs"] = q("SELECT count(*), count(DISTINCT genome), count(DISTINCT contig_code) FROM u")
assert qc["genomes"] == len(sel), (qc["genomes"], len(sel))
qc["dup_contig_position"] = q("SELECT count(*) FROM (SELECT contig_code, position_index FROM u GROUP BY ALL HAVING count(*) > 1)")[0]
qc["contigs_noncontiguous_gid"] = q("""SELECT count(*) FROM (SELECT contig_code, max(gid) - min(gid) + 1 AS w, count(*) AS n,
        max(position_index) AS mp FROM u GROUP BY contig_code) WHERE w <> n OR mp <> n - 1""")[0]
qc["dup_protein_id"] = q("SELECT count(*) - count(DISTINCT protein_id) FROM u")[0]
assert qc["dup_contig_position"] == 0 and qc["contigs_noncontiguous_gid"] == 0 and qc["dup_protein_id"] == 0
con.execute(f"""COPY (SELECT gid::BIGINT AS gid, protein_id, genome, genome_code, contig, contig_code::BIGINT AS contig_code,
        position_index::BIGINT AS position_index, strand, length, start, "end" FROM u ORDER BY gid)
        TO '{P}/universe/genes.parquet' (FORMAT parquet)""")
con.execute(f"""COPY (SELECT gid::BIGINT AS gid, gtdb_id, prodigal_id, prodigal_n, file_index, partial, start_type, rbs_motif,
        gc_cont, internal_stops FROM u ORDER BY gid) TO '{P}/universe/gene_flags.parquet' (FORMAT parquet)""")
con.execute(f"""COPY (SELECT genome, contig, contig_code::BIGINT AS contig_code, count(*)::BIGINT AS n_genes,
        NULL::INTEGER AS length_nt, min(gid)::BIGINT AS first_gid FROM u GROUP BY ALL ORDER BY contig_code)
        TO '{P}/universe/contigs.parquet' (FORMAT parquet)""")
con.execute(f"""COPY (SELECT contig_code::BIGINT AS contig_code, max("end")::BIGINT AS max_end_nt,
        arg_min(partial, position_index) AS first_partial, arg_max(partial, position_index) AS last_partial
        FROM u GROUP BY 1 ORDER BY 1) TO '{P}/universe/contig_extra.parquet' (FORMAT parquet)""")
srcs = " UNION ALL ".join(f"SELECT * FROM read_csv('{d}/genomes_src.tsv', delim='\t', header=true, all_varchar=true)" for d in SRC.values())
con.execute(f"CREATE TABLE src AS {srcs}")
con.execute(f"CREATE TABLE cm AS SELECT Name AS id, Completeness::FLOAT AS completeness, Contamination::FLOAT AS contamination FROM read_csv('{CM2}', delim='\t', header=true, all_varchar=true)")
con.execute(f"CREATE TABLE gt AS SELECT column0 AS id, column1 AS tax_db FROM read_csv('{GTAX}', delim='\t', header=false, all_varchar=true)")
con.execute(f"""CREATE TABLE genomes AS
    WITH g AS (SELECT genome, any_value(gtdb_id) AS gtdb_id, genome_code, count(*) AS n_genes, count(DISTINCT contig_code) AS n_contigs,
                      sum((start <= 0)::INT)::INTEGER AS n_start_le0, avg((strand = -1)::INT)::DOUBLE AS minus_fraction
               FROM u GROUP BY genome, genome_code)
    SELECT g.genome, g.genome_code, g.n_genes, g.n_contigs, s.tax, 'gtdb_r232_species_representative' AS tax_source, gt.tax_db,
           replace(split_part(s.tax, ';', 2), 'p__', '') AS phylum, replace(split_part(s.tax, ';', 3), 'c__', '') AS class,
           replace(split_part(s.tax, ';', 4), 'o__', '') AS ord, replace(split_part(s.tax, ';', 5), 'f__', '') AS family,
           replace(split_part(s.tax, ';', 6), 'g__', '') AS genus, replace(split_part(s.tax, ';', 7), 's__', '') AS species,
           cm.completeness, cm.contamination, TRUE AS order_trusted, '' AS order_flags, 0 AS n_wp, g.n_start_le0,
           0 AS n_contig_missing, NULL::INTEGER AS n_beyond_contig, g.minus_fraction,
           NULL::INTEGER AS n_stop_matched, NULL::INTEGER AS n_strand_agree, NULL::INTEGER AS n_strand_disagree,
           NULL::DOUBLE AS strand_agreement, g.gtdb_id,
           sl.layer, sl.lut_status, sl.n_main_loci::INTEGER AS n_main_loci, sl.locus_components, sl.source_scope,
           sl.qtier::INTEGER AS quality_tier, CASE WHEN g.genome_code % 2 = 1 THEN 'tuning' ELSE 'test' END AS split
    FROM g JOIN src s ON s.gtdb_id = g.gtdb_id JOIN sel sl ON sl.gtdb_id = g.gtdb_id
    LEFT JOIN cm ON cm.id = split_part(g.genome, '.', 1) LEFT JOIN gt ON gt.id = split_part(g.genome, '.', 1)
    ORDER BY g.genome_code""")
con.execute(f"COPY genomes TO '{P}/universe/genomes.parquet' (FORMAT parquet)")
con.execute(f"COPY (SELECT genome_code, genome, gtdb_id, split FROM genomes ORDER BY genome_code) TO '{P}/split.parquet' (FORMAT parquet)")
qc["split"] = dict(con.execute("SELECT split, count(*) FROM genomes GROUP BY 1").fetchall())
qc["layers"] = dict(con.execute("SELECT layer, count(*) FROM genomes GROUP BY 1 ORDER BY 1").fetchall())

# FASTA
con.execute(f"""COPY (SELECT u.gid::BIGINT AS gid, a.seq FROM u JOIN (SELECT genome, prodigal_id, seq FROM allp WHERE gtdb_id IN (SELECT gtdb_id FROM sel)) a ON a.genome = u.genome AND a.prodigal_id = u.prodigal_id
                ORDER BY u.gid) TO '{P}/.seq.parquet' (FORMAT parquet, ROW_GROUP_SIZE 200000)""")
n = 0; res = 0
with open(f"{P}/.faa.tmp", "w") as fh:
    for b in pq.ParquetFile(f"{P}/.seq.parquet").iter_batches(batch_size=200000):
        for g, s in zip(b.column(0).to_pylist(), b.column(1).to_pylist()):
            assert g == n
            fh.write(f">{g}\n{s}\n"); n += 1; res += len(s)
os.replace(f"{P}/.faa.tmp", f"{P}/proteins_gid.faa"); os.remove(f"{P}/.seq.parquet")
assert n == qc["genes"]
qc["fasta_records"], qc["fasta_residues"] = n, res
qc["unique_sequences"] = q(f"SELECT count(DISTINCT md5(seq)) FROM allp WHERE gtdb_id IN (SELECT gtdb_id FROM sel)")[0]

# lut truth: part-1 main and strict loci of the selected genomes, mapped by protein_id
def arch_class(cp):
    c = set(str(cp).split(","))
    abef = len(c & {"A", "B", "E", "F"})
    if {"A", "B", "C", "D", "E", "F", "G", "H", "I"} <= c:
        return "1_complete_AtoI" + ("_M" if "M" in c else "")
    if "H" in c and abef >= 2:
        return "2_receptor_plus_transporter"
    if "H" in c:
        return "4_receptor_module_without_transporter"
    if abef >= 2:
        return "3_transporter_module_without_receptor"
    return "5_other"


L, M = [], []
gset = set(sel.genome)
for sc in SRC:
    l = pd.read_csv(f"{S}/results/{sc}/loci.tsv", sep="\t", dtype={"components_present": str})
    l = l[l.definition.isin(["main", "strict"]) & l.genome.isin(gset)].copy()
    l["source_scope"] = sc
    L.append(l)
    g = pd.read_csv(f"{S}/results/{sc}/locus_genes.tsv", sep="\t", usecols=["locus_id", "role", "protein_id", "label", "best_component"],
                    low_memory=False)
    g = g[(g.role == "component") & g.locus_id.isin(set(l.locus_id))].copy()
    g["source_scope"] = sc
    M.append(g)
L = pd.concat(L, ignore_index=True); M = pd.concat(M, ignore_index=True)
gid_of = con.execute("SELECT protein_id, gid, position_index, contig_code FROM u").df().set_index("protein_id")
M = M.join(gid_of, on="protein_id")
assert M.gid.notna().all(), "unmapped lut member"
L = L.sort_values(["definition", "genome", "contig", "start_pos", "source_scope", "locus_id"]).reset_index(drop=True)
L["instance_id"] = range(len(L))
M = M.merge(L[["locus_id", "source_scope", "instance_id"]], on=["locus_id", "source_scope"])
mem = M.sort_values(["instance_id", "position_index"]).groupby("instance_id").agg(
    member_gids=("gid", lambda v: [int(x) for x in v]), member_positions=("position_index", lambda v: [int(x) for x in v]),
    member_components=("label", list), contig_code=("contig_code", "first"), start_idx=("position_index", "min"),
    end_idx=("position_index", "max"))
L = L.join(mem, on="instance_id")
L["caller"] = "lut_homology_" + L.definition
L["arch_class"] = L.components_present.map(arch_class)
gc = con.execute("SELECT genome, genome_code FROM genomes").df().set_index("genome").genome_code
L["genome_code"] = L.genome.map(gc)
L["span_genes"] = L.end_idx - L.start_idx + 1
inst = L[["instance_id", "caller", "definition", "arch_class", "components_present", "components_missing", "architecture",
          "genome", "genome_code", "contig", "contig_code", "start_idx", "end_idx", "start_nt", "end_nt", "span_genes",
          "member_gids", "member_positions", "member_components", "n_component_genes", "lanm_like_in_locus",
          "mdh_ko_within_25_genes", "genes_to_contig_start", "genes_to_contig_end", "at_contig_edge_le2", "family", "genus",
          "source_scope", "locus_id"]].rename(columns={"locus_id": "source_locus_id", "components_present": "system_subtype",
                                                       "arch_class": "system_type"})
inst.to_parquet(f"{P}/gold/lut_instances.parquet", index=False)
M2 = M[["instance_id", "gid", "label"]].rename(columns={"label": "component"})
M2.insert(1, "caller", M2.instance_id.map(L.set_index("instance_id").caller))
M2.sort_values(["instance_id", "gid"]).to_parquet(f"{P}/gold/lut_members.parquet", index=False)
qc["lut_instances"] = L.groupby(["definition", "arch_class"]).size().rename("n").reset_index().to_dict("records")
qc["lut_members"] = int(len(M2))
qc["seconds"] = round(time.time() - t0, 1)
rec = dict(stage="build", selection=a.selection[0], selection_sha256=sha(a.selection[0]), qc=qc,
           code_sha256=sha(__file__), date=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
json.dump(rec, open(f"{P}/receipts/build.json", "w"), indent=1, default=str)
print(json.dumps(rec, indent=1, default=str), flush=True)
