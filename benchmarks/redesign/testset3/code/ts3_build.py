#!/usr/bin/env python3
"""testset3 package: universe tables + protein FASTA in the layout of slice_lut/testset/package_k8 (from 14_build_testset.py).

usage: ts3_build.py SELECTION_TSV PARTS_DIR PKG_DIR THREADS
Numbering (DPANN rules): genome_code = rank of genome accession, contig_code = rank of (genome, contig),
gid = rank over (genome_code, contig_code, start, end, protein_id), position_index = rank on the contig.
Writes universe/{genes,genomes,contigs,gene_flags,contig_extra}.parquet, proteins_gid.faa (header = gid, trailing '*'
removed), selection.tsv, split.parquet (odd genome_code = tuning, even = test), receipts/build.json.
No truth tables, no annotation, no token arms.
"""
import hashlib, json, os, sys, time
import duckdb, pandas as pd, pyarrow.parquet as pq

SEL, PARTS, P, TH = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
t0 = time.time()
for d in ("universe", "receipts"):
    os.makedirs(f"{P}/{d}", exist_ok=True)
con = duckdb.connect()
con.execute(f"PRAGMA threads={TH}; SET preserve_insertion_order=false; SET temp_directory='{P}/.duck_tmp'")
sel = pd.read_csv(SEL, sep="\t", dtype=str)
sel.to_csv(f"{P}/selection.tsv", sep="\t", index=False)
con.register("sel_df", sel)
con.execute("CREATE TABLE sel AS SELECT * FROM sel_df")
con.execute(f"CREATE VIEW allp AS SELECT * FROM read_parquet('{PARTS}/*.parquet')")
con.execute("""CREATE TABLE p AS SELECT a.* EXCLUDE (seq), a.genome || '|' || a.prodigal_id AS protein_id
               FROM allp a WHERE a.gtdb_id IN (SELECT accession FROM sel)""")
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
qc["internal_stop_genes"] = q("SELECT count(*) FROM u WHERE internal_stops > 0")[0]
qc["empty_sequences"] = q("SELECT count(*) FROM u WHERE length = 0")[0]
qc["position_vs_prodigal_order_disagree"] = q("""SELECT count(*) FROM (SELECT position_index,
        row_number() OVER (PARTITION BY contig_code ORDER BY prodigal_n) - 1 AS r FROM u) WHERE r <> position_index""")[0]
qc["single_gene_contigs"] = q("SELECT count(*) FROM (SELECT contig_code FROM u GROUP BY 1 HAVING count(*) = 1)")[0]
qc["partial_genes"] = q("SELECT count(*) FROM u WHERE partial <> '00'")[0]
assert qc["dup_contig_position"] == 0 and qc["contigs_noncontiguous_gid"] == 0 and qc["dup_protein_id"] == 0 and qc["empty_sequences"] == 0
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
con.execute("""CREATE TABLE genomes AS
    WITH g AS (SELECT genome, any_value(gtdb_id) AS gtdb_id, genome_code, count(*) AS n_genes, count(DISTINCT contig_code) AS n_contigs,
                      sum((start <= 0)::INT)::INTEGER AS n_start_le0, avg((strand = -1)::INT)::DOUBLE AS minus_fraction
               FROM u GROUP BY genome, genome_code)
    SELECT g.genome, g.genome_code, g.n_genes, g.n_contigs, s.gtdb_taxonomy AS tax, 'gtdb_r232_species_representative' AS tax_source,
           s.gtdb_taxonomy AS tax_db, s.domain, s.phylum, s."class", s.ord, s.family, s.genus, s.species,
           s.checkm2_completeness::FLOAT AS completeness, s.checkm2_contamination::FLOAT AS contamination,
           TRUE AS order_trusted, '' AS order_flags, 0 AS n_wp, g.n_start_le0, 0 AS n_contig_missing,
           NULL::INTEGER AS n_beyond_contig, g.minus_fraction, NULL::INTEGER AS n_stop_matched, NULL::INTEGER AS n_strand_agree,
           NULL::INTEGER AS n_strand_disagree, NULL::DOUBLE AS strand_agreement, g.gtdb_id,
           s.ncbi_assembly_level AS assembly_level, s.contig_count::INTEGER AS contig_count_metadata,
           s.protein_count::INTEGER AS protein_count_metadata, s.genome_size::BIGINT AS genome_size, s.tier AS selection_tier,
           s."order"::INTEGER AS selection_order, s.dpann_lineage = 'True' AS dpann_lineage,
           s.checkm_completeness::FLOAT AS checkm1_completeness, s.checkm_contamination::FLOAT AS checkm1_contamination,
           CASE WHEN g.genome_code % 2 = 1 THEN 'tuning' ELSE 'test' END AS split
    FROM g JOIN sel s ON s.accession = g.gtdb_id ORDER BY g.genome_code""")
con.execute(f"COPY genomes TO '{P}/universe/genomes.parquet' (FORMAT parquet)")
con.execute(f"COPY (SELECT genome_code, genome, gtdb_id, domain, split FROM genomes ORDER BY genome_code) TO '{P}/split.parquet' (FORMAT parquet)")
qc["split"] = dict(con.execute("SELECT domain || ':' || split, count(*) FROM genomes GROUP BY 1 ORDER BY 1").fetchall())
qc["contigs_vs_metadata_mismatch"] = q("SELECT count(*) FROM genomes WHERE n_contigs > contig_count_metadata")[0]
qc["proteins_vs_metadata_mismatch"] = q("SELECT count(*) FROM genomes WHERE n_genes <> protein_count_metadata")[0]
qc["by_domain"] = con.execute("SELECT domain, count(*), sum(n_genes), sum(n_contigs) FROM genomes GROUP BY 1 ORDER BY 1").fetchall()
con.execute(f"""COPY (SELECT u.gid::BIGINT AS gid, a.seq FROM u JOIN (SELECT genome, prodigal_id, seq FROM allp
                WHERE gtdb_id IN (SELECT accession FROM sel)) a ON a.genome = u.genome AND a.prodigal_id = u.prodigal_id
                ORDER BY u.gid) TO '{P}/.seq.parquet' (FORMAT parquet, ROW_GROUP_SIZE 200000)""")
n = 0; res = 0
with open(f"{P}/.faa.tmp", "w") as fh:
    for b in pq.ParquetFile(f"{P}/.seq.parquet").iter_batches(batch_size=200000):
        for g, s in zip(b.column(0).to_pylist(), b.column(1).to_pylist()):
            assert g == n and not s.endswith("*")
            fh.write(f">{g}\n{s}\n"); n += 1; res += len(s)
os.replace(f"{P}/.faa.tmp", f"{P}/proteins_gid.faa"); os.remove(f"{P}/.seq.parquet")
assert n == qc["genes"]
qc["fasta_records"], qc["fasta_residues"] = n, res
qc["unique_sequences"], qc["unique_residues"] = q("""SELECT count(*), sum(l) FROM (SELECT md5(seq) h, any_value(length) l FROM allp
        WHERE gtdb_id IN (SELECT accession FROM sel) GROUP BY 1)""")
qc["seconds"] = round(time.time() - t0, 1)
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
rec = dict(stage="build", selection=SEL, selection_sha256=sha(SEL), qc=qc, code_sha256=sha(__file__),
           date=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
json.dump(rec, open(f"{P}/receipts/build.json", "w"), indent=1, default=str)
print(json.dumps(rec, indent=1, default=str), flush=True)
