#!/usr/bin/env python
"""Build the shared gene universe for the redesign experiments (Spacedust collection).

One row per protein, reference first then query:
  gid = ELSA reference row_index                 for the 4,190,383 reference proteins
  gid = 4,190,383 + ELSA query row_index         for the 67,148 query proteins
Every experiment addresses genes by gid, so token files and results join exactly.
"""
import argparse
from pathlib import Path

import duckdb

ROOT = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/spacedust")
N = ROOT / "native_run_20260910"
G = "/groups/banfield/projects/environmental/sr/srvp2020/gtdb/rp16_run_r232"
NREF = 4190383


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--threads", type=int, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"PRAGMA threads={a.threads}")
    idx = "columns={'key':'BIGINT','off':'BIGINT','len':'BIGINT'}"
    con.execute(f"""CREATE TABLE rfs AS SELECT key FROM read_csv('{N}/reference/keggclusterdb_foldseek_ca.index',
                    delim='\t', header=false, {idx})""")
    con.execute(f"""CREATE TABLE qfs AS SELECT key FROM read_csv('{N}/query/querydb_foldseek_ca.index',
                    delim='\t', header=false, {idx})""")
    con.execute(f"""
        CREATE TABLE gtdb AS
        SELECT regexp_extract(acc, 'GC[AF]_([0-9]{{9}})', 1) AS id9, any_value(tax) AS tax
        FROM read_csv(['{G}/bac120_taxonomy_r232.tsv', '{G}/ar53_taxonomy_r232.tsv'],
                      delim='\t', header=false, columns={{'acc':'VARCHAR','tax':'VARCHAR'}}) GROUP BY ALL""")
    con.execute(f"""
        CREATE TABLE genes AS
        SELECT r.row_index AS gid, 'ref' AS side, r.row_index, r.source_sequence_db_key AS mmseqs_key,
               r.gene_id, r.sample_id AS genome, r.contig_id AS contig, r.position_index, r.strand,
               r.sequence_length AS length, (r.source_sequence_db_key IN (SELECT key FROM rfs)) AS has_struct
        FROM read_parquet('{ROOT}/results/esm2_t6/shard_000/export/metadata.parquet') r
        UNION ALL
        SELECT {NREF} + q.row_index, 'query', q.row_index, q.key, q.gene_id, q.sample_id, q.contig_id,
               q.position_index, q.strand, q.sequence_length, (q.key IN (SELECT key FROM qfs))
        FROM read_parquet('{N}/query/native_lookup_crosswalk.parquet') q""")
    con.execute("""
        CREATE TABLE genomes AS
        WITH g AS (SELECT genome, any_value(side) AS side, count(*) AS n_genes,
                          count(DISTINCT contig) AS n_contigs,
                          regexp_extract(genome, 'GC[AF]_([0-9]{9})', 1) AS id9
                   FROM genes GROUP BY genome)
        SELECT g.genome, g.side, g.n_genes, g.n_contigs, gtdb.tax,
               replace(split_part(gtdb.tax, ';', 2), 'p__', '') AS phylum,
               replace(split_part(gtdb.tax, ';', 3), 'c__', '') AS class,
               replace(split_part(gtdb.tax, ';', 4), 'o__', '') AS ord,
               replace(split_part(gtdb.tax, ';', 5), 'f__', '') AS family,
               replace(split_part(gtdb.tax, ';', 6), 'g__', '') AS genus,
               replace(split_part(gtdb.tax, ';', 7), 's__', '') AS species
        FROM g LEFT JOIN gtdb ON g.id9 <> '' AND gtdb.id9 = g.id9""")
    ref_phylum, ref_class = con.execute("""SELECT phylum, class FROM genomes WHERE genus = 'Streptomyces'
                                           GROUP BY ALL ORDER BY count(*) DESC LIMIT 1""").fetchone()
    con.execute(f"""
        ALTER TABLE genomes ADD COLUMN stratum VARCHAR;
        UPDATE genomes SET stratum = CASE
            WHEN side = 'query' THEN 'query'
            WHEN tax IS NULL THEN '9_no_gtdb'
            WHEN genus = 'Streptomyces' THEN '0_genus_Streptomyces'
            WHEN class = '{ref_class}' AND phylum = '{ref_phylum}' THEN '1_same_class_other_genus'
            WHEN phylum = '{ref_phylum}' THEN '2_same_phylum_other_class'
            ELSE '3_other_phylum' END;
        ALTER TABLE genomes ADD COLUMN genome_code INTEGER;
        UPDATE genomes SET genome_code = r.rn FROM (
            SELECT genome, row_number() OVER (ORDER BY side DESC, genome) - 1 AS rn FROM genomes) r
        WHERE genomes.genome = r.genome""")
    con.execute("""
        CREATE TABLE contigs AS
        SELECT genome, contig, row_number() OVER (ORDER BY genome, contig) - 1 AS contig_code, count(*) AS n_genes
        FROM genes GROUP BY genome, contig""")
    con.execute("""
        CREATE TABLE universe AS
        SELECT g.gid, g.side, g.row_index, g.mmseqs_key, g.gene_id, g.genome, gm.genome_code, g.contig,
               c.contig_code, g.position_index, g.strand, g.length, g.has_struct
        FROM genes g JOIN genomes gm USING (genome) JOIN contigs c USING (genome, contig)
        ORDER BY g.gid""")
    n, nq, ng, nc = con.execute("""SELECT count(*), sum((side='query')::INT), count(DISTINCT genome_code),
                                          count(DISTINCT contig_code) FROM universe""").fetchone()
    assert n == NREF + 67148 and nq == 67148, (n, nq)
    assert con.execute("SELECT count(*) = count(DISTINCT gid) AND min(gid) = 0 AND max(gid) = count(*) - 1 FROM universe").fetchone()[0]
    dup = con.execute("""SELECT count(*) FROM (SELECT contig_code, position_index FROM universe
                                              GROUP BY ALL HAVING count(*) > 1)""").fetchone()[0]
    assert dup == 0, f"{dup} duplicate positions"
    con.execute(f"COPY universe TO '{a.out}/genes.parquet' (FORMAT parquet)")
    con.execute(f"COPY (SELECT * FROM genomes ORDER BY genome_code) TO '{a.out}/genomes.parquet' (FORMAT parquet)")
    con.execute(f"COPY (SELECT * FROM contigs ORDER BY contig_code) TO '{a.out}/contigs.parquet' (FORMAT parquet)")
    print("genes", n, "query", nq, "genomes", ng, "contigs", nc, "ref rank names", ref_phylum, ref_class)
    print(con.execute("SELECT stratum, count(*) AS genomes, sum(n_genes) AS genes FROM genomes GROUP BY ALL ORDER BY 1").df())
    print(con.execute("SELECT genome, n_genes, n_contigs FROM genomes WHERE side='query' ORDER BY 1").df())


if __name__ == "__main__":
    main()
