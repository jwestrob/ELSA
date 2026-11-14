#!/usr/bin/env python3
"""Build an ELSA gene universe (layout of ws_dpann/package/universe) for a GTDB slice (workstream slice_lut).

usage: 02_universe.py PARTS_DIR SCOPE FAMILY_FILTER PKG_DIR GLOBDB_CHECKM2 GLOBDB_TAX THREADS
  SCOPE          package name (e.g. beijerinckiaceae, rhizobiales)
  FAMILY_FILTER  GTDB family name to keep (e.g. Beijerinckiaceae), or ALL
Writes PKG_DIR/universe/{genes,genomes,contigs}.parquet in the DPANN layout, plus
       PKG_DIR/universe/gene_flags.parquet  (gid, gtdb_id, prodigal_id, prodigal_n, file_index, partial, start_type,
                                             rbs_motif, gc_cont, internal_stops)
       PKG_DIR/universe/contig_extra.parquet (contig_code, max_end_nt, first_partial, last_partial)
       PKG_DIR/proteins_gid.faa (header = gid), PKG_DIR/receipts/universe.json
gid = rank over (genome_code, contig_code, start, end, protein_id); genome_code = rank of genome accession;
contig_code = rank of (genome, contig); position_index = rank on the contig by (start, end, protein_id).
"""
import hashlib, json, os, sys, time
import duckdb
import pyarrow.parquet as pq

PARTS, SCOPE, FAM, PKG, CM2, GTAX, TH = sys.argv[1:8]
t0 = time.time()
for d in ("universe", "receipts", "tokens"):
    os.makedirs(f"{PKG}/{d}", exist_ok=True)
con = duckdb.connect()
con.execute(f"PRAGMA threads={TH}; SET preserve_insertion_order=false")
con.execute(f"SET temp_directory='{PKG}/.duck_tmp'")
where = "" if FAM == "ALL" else f"WHERE split_part(s.tax, ';', 5) = 'f__{FAM}'"
con.execute(f"""CREATE TABLE src AS SELECT * FROM read_csv('{os.path.dirname(PARTS.rstrip('/'))}/genomes_src.tsv',
                delim='\t', header=true, all_varchar=true)""")
con.execute(f"""CREATE TABLE p AS SELECT p.* EXCLUDE (seq), p.genome || '|' || p.prodigal_id AS protein_id
                FROM read_parquet('{PARTS}/*.parquet') p JOIN src s ON s.gtdb_id = p.gtdb_id {where}""")
con.execute("CREATE TABLE gm AS SELECT genome, row_number() OVER (ORDER BY genome) - 1 AS genome_code FROM (SELECT DISTINCT genome FROM p)")
con.execute("""CREATE TABLE cg AS SELECT genome, contig, row_number() OVER (ORDER BY genome, contig) - 1 AS contig_code
               FROM (SELECT DISTINCT genome, contig FROM p)""")
con.execute("""CREATE TABLE u AS
    SELECT row_number() OVER (ORDER BY gm.genome_code, cg.contig_code, p.start, p."end", p.protein_id) - 1 AS gid,
           p.protein_id, p.genome, gm.genome_code::INTEGER AS genome_code, p.contig, cg.contig_code,
           row_number() OVER (PARTITION BY cg.contig_code ORDER BY p.start, p."end", p.protein_id) - 1 AS position_index,
           p.strand::TINYINT AS strand, p.length::BIGINT AS length, p.start::BIGINT AS start, p."end"::BIGINT AS "end",
           p.gtdb_id, p.prodigal_id, p.prodigal_n, p.file_index, p.partial, p.start_type, p.rbs_motif, p.gc_cont,
           p.internal_stops
    FROM p JOIN gm USING (genome) JOIN cg USING (genome, contig)""")
q = lambda s: con.execute(s).fetchone()
qc = {}
qc["genes"], qc["genomes"], qc["contigs"] = q("SELECT count(*), count(DISTINCT genome), count(DISTINCT contig_code) FROM u")
qc["dup_contig_position"] = q("SELECT count(*) FROM (SELECT contig_code, position_index FROM u GROUP BY ALL HAVING count(*) > 1)")[0]
qc["contigs_noncontiguous_gid"] = q("""SELECT count(*) FROM (SELECT contig_code, max(gid) - min(gid) + 1 AS w, count(*) AS n,
        max(position_index) AS mp FROM u GROUP BY contig_code) WHERE w <> n OR mp <> n - 1""")[0]
qc["dup_protein_id"] = q("SELECT count(*) - count(DISTINCT protein_id) FROM u")[0]
qc["position_vs_prodigal_order_disagree"] = q("""SELECT count(*) FROM (SELECT position_index,
        row_number() OVER (PARTITION BY contig_code ORDER BY prodigal_n) - 1 AS r FROM u) WHERE r <> position_index""")[0]
qc["single_gene_contigs"] = q("SELECT count(*) FROM (SELECT contig_code FROM u GROUP BY 1 HAVING count(*) = 1)")[0]
qc["identical_coordinate_pairs"] = q("""SELECT count(*) FROM (SELECT contig_code, start, "end" FROM u GROUP BY ALL HAVING count(*) > 1)""")[0]
qc["start_le0"] = q("SELECT count(*) FROM u WHERE start <= 0")[0]
qc["internal_stop_genes"] = q("SELECT count(*) FROM u WHERE internal_stops > 0")[0]
qc["empty_sequences"] = q("SELECT count(*) FROM u WHERE length = 0")[0]
qc["partial_genes"] = q("SELECT count(*) FROM u WHERE partial <> '00'")[0]
qc["strand_plus_minus"] = q("SELECT sum((strand = 1)::INT), sum((strand = -1)::INT) FROM u")
assert qc["dup_contig_position"] == 0 and qc["contigs_noncontiguous_gid"] == 0 and qc["dup_protein_id"] == 0
assert qc["empty_sequences"] == 0

con.execute(f"COPY (SELECT gid::BIGINT AS gid, protein_id, genome, genome_code, contig, contig_code::BIGINT AS contig_code, "
            f"position_index::BIGINT AS position_index, strand, length, start, \"end\" FROM u ORDER BY gid) "
            f"TO '{PKG}/universe/genes.parquet' (FORMAT parquet)")
con.execute(f"COPY (SELECT gid::BIGINT AS gid, gtdb_id, prodigal_id, prodigal_n, file_index, partial, start_type, rbs_motif, "
            f"gc_cont, internal_stops FROM u ORDER BY gid) TO '{PKG}/universe/gene_flags.parquet' (FORMAT parquet)")
con.execute(f"""COPY (SELECT genome, contig, contig_code::BIGINT AS contig_code, count(*)::BIGINT AS n_genes,
        NULL::INTEGER AS length_nt, min(gid)::BIGINT AS first_gid FROM u GROUP BY ALL ORDER BY contig_code)
        TO '{PKG}/universe/contigs.parquet' (FORMAT parquet)""")
con.execute(f"""COPY (SELECT contig_code::BIGINT AS contig_code, max("end")::BIGINT AS max_end_nt,
        arg_min(partial, position_index) AS first_partial, arg_max(partial, position_index) AS last_partial
        FROM u GROUP BY 1 ORDER BY 1) TO '{PKG}/universe/contig_extra.parquet' (FORMAT parquet)""")
con.execute(f"""CREATE TABLE cm AS SELECT Name AS id, Completeness::FLOAT AS completeness, Contamination::FLOAT AS contamination
                FROM read_csv('{CM2}', delim='\t', header=true, all_varchar=true)""")
con.execute(f"""CREATE TABLE gt AS SELECT column0 AS id, column1 AS tax_db
                FROM read_csv('{GTAX}', delim='\t', header=false, all_varchar=true)""")
con.execute(f"""COPY (
    WITH g AS (SELECT genome, any_value(gtdb_id) AS gtdb_id, genome_code, count(*) AS n_genes, count(DISTINCT contig_code) AS n_contigs,
                      sum((start <= 0)::INT)::INTEGER AS n_start_le0, avg((strand = -1)::INT)::DOUBLE AS minus_fraction
               FROM u GROUP BY genome, genome_code)
    SELECT g.genome, g.genome_code, g.n_genes, g.n_contigs, s.tax, 'gtdb_r232_species_representative' AS tax_source,
           gt.tax_db,
           replace(split_part(s.tax, ';', 2), 'p__', '') AS phylum, replace(split_part(s.tax, ';', 3), 'c__', '') AS class,
           replace(split_part(s.tax, ';', 4), 'o__', '') AS ord, replace(split_part(s.tax, ';', 5), 'f__', '') AS family,
           replace(split_part(s.tax, ';', 6), 'g__', '') AS genus, replace(split_part(s.tax, ';', 7), 's__', '') AS species,
           cm.completeness, cm.contamination, TRUE AS order_trusted, '' AS order_flags, 0 AS n_wp, g.n_start_le0,
           0 AS n_contig_missing, NULL::INTEGER AS n_beyond_contig, g.minus_fraction,
           NULL::INTEGER AS n_stop_matched, NULL::INTEGER AS n_strand_agree, NULL::INTEGER AS n_strand_disagree,
           NULL::DOUBLE AS strand_agreement, g.gtdb_id
    FROM g JOIN src s ON s.gtdb_id = g.gtdb_id
    LEFT JOIN cm ON cm.id = split_part(g.genome, '.', 1)
    LEFT JOIN gt ON gt.id = split_part(g.genome, '.', 1)
    ORDER BY g.genome_code) TO '{PKG}/universe/genomes.parquet' (FORMAT parquet)""")
gq = con.execute(f"""SELECT count(*), sum((completeness IS NULL)::INT), sum((tax_db IS NULL)::INT),
                     sum((tax_db <> tax)::INT) FROM '{PKG}/universe/genomes.parquet'""").fetchone()
qc["genomes_rows"], qc["genomes_without_checkm2"], qc["genomes_without_globdb_tax"], qc["globdb_tax_differs"] = gq
qc["genes_per_genome_q05_q50_q95"] = q("SELECT quantile_cont(n_genes, [0.05, 0.5, 0.95]) FROM '" + PKG + "/universe/genomes.parquet'")[0]

# FASTA in gid order
con.execute(f"""COPY (SELECT u.gid::BIGINT AS gid, p.seq FROM u JOIN read_parquet('{PARTS}/*.parquet') p
                ON p.genome = u.genome AND p.prodigal_id = u.prodigal_id ORDER BY u.gid)
                TO '{PKG}/.seq_by_gid.parquet' (FORMAT parquet, ROW_GROUP_SIZE 200000)""")
tmp = f"{PKG}/.proteins_gid.faa.tmp"
n = 0; expect = 0; res = 0
with open(tmp, "w") as fh:
    pf = pq.ParquetFile(f"{PKG}/.seq_by_gid.parquet")
    for b in pf.iter_batches(batch_size=200000):
        g = b.column(0).to_pylist(); s = b.column(1).to_pylist()
        for gi, si in zip(g, s):
            assert gi == expect; expect += 1
            fh.write(f">{gi}\n{si}\n"); n += 1; res += len(si)
os.replace(tmp, f"{PKG}/proteins_gid.faa")
os.remove(f"{PKG}/.seq_by_gid.parquet")
assert n == qc["genes"]
qc["fasta_records"], qc["fasta_residues"] = n, res


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(16 << 20), b""):
            h.update(blk)
    return h.hexdigest()


files = {f: sha(f"{PKG}/{f}") for f in ["universe/genes.parquet", "universe/genomes.parquet", "universe/contigs.parquet",
                                         "universe/gene_flags.parquet", "universe/contig_extra.parquet", "proteins_gid.faa"]}
rec = dict(stage="universe", scope=SCOPE, family_filter=FAM, parts=PARTS, qc=qc, sha256=files,
           code_sha256=sha(__file__), seconds=round(time.time() - t0, 1), date=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
json.dump(rec, open(f"{PKG}/receipts/universe.json", "w"), indent=1, default=str)
print(json.dumps(rec, indent=1, default=str), flush=True)
