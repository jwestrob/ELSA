#!/usr/bin/env python
"""Build the DPANN benchmark package for the ELSA redesign (workstream ws_dpann).

Stages (run in order; each writes under --pkg and a JSON receipt under --pkg/receipts):
  verify      SHA-256 of the canonical DuckDB copy against the Sharur seal.
  universe    genes/contigs/genomes tables in the bacterial-universe shape, plus QC.
  prodigal    parse the Oct 2026 Prodigal re-call headers (independent strand evidence).
  strandqc    compare canonical strand with Prodigal at shared stop codons; per-genome trust.
  taxonomy    GTDB r232 taxonomy per genome, with the database string beside it.
  embed       align March ESM2-8M vectors to the canonical proteins by exact identity.
  gold        curated caller instances (system loci) and per-protein annotation flags.
  fasta       proteins.faa with >gid headers for MMseqs2.

All DuckDB files are opened read_only. Nothing under the Sharur tree is written.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

SEAL_ID = "805c93a69776eb746bb9f642d71bfd52055540968f3a1f8b40f6987f7fcc29e5"
MARCH_DB = "/groups/banfield/users/jwestrob/bin/Sharur/data/dpann/sharur.duckdb"
H5 = "/groups/banfield/users/jwestrob/bin/Sharur/data/dpann/embeddings/protein_embeddings.h5"
H5_MANIFEST = "/groups/banfield/users/jwestrob/bin/Sharur/data/dpann/embeddings/embedding_manifest.json"
PRODIGAL_DIR = "/groups/banfield/users/jwestrob/dpann_reingest_2026-10-14/proteins"
GTDB_AR = "/groups/banfield/projects/environmental/sr/srvp2020/gtdb/rp16_run_r232/ar53_taxonomy_r232.tsv"


def sha256(path, chunk=16 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def receipt(pkg: Path, stage: str, payload: dict):
    d = pkg / "receipts"
    d.mkdir(parents=True, exist_ok=True)
    payload = dict(stage=stage, written=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                   code_sha256=sha256(__file__), **payload)
    tmp = d / f".{stage}.json.tmp"
    tmp.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    tmp.rename(d / f"{stage}.json")
    print(json.dumps(payload, indent=2, default=str), flush=True)


def connect(a):
    con = duckdb.connect()
    con.execute(f"PRAGMA threads={a.threads}")
    con.execute(f"SET temp_directory='{a.work}/duck_tmp'")
    con.execute(f"ATTACH '{a.db}' AS c (READ_ONLY)")
    return con


def copy(con, sql, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name("." + path.name + ".tmp")
    con.execute(f"COPY ({sql}) TO '{tmp}' (FORMAT parquet, COMPRESSION zstd)")
    tmp.rename(path)


# --------------------------------------------------------------------------- verify
def st_verify(a):
    seal = json.loads(Path(a.seal).read_text())
    assert seal["dataset_id"] == SEAL_ID, seal["dataset_id"]
    rec = [x for x in seal["identity"]["canonical_artifacts"] if x["kind"] == "duckdb"][0]
    size = os.path.getsize(a.db)
    digest = sha256(a.db)
    ok = digest == rec["digest"]["value"] and size == rec["size"]
    con = connect(a)
    counts = {t["table"]: t["rows"] for t in seal["identity"]["database"]["tables"]}
    live = {}
    for t in ["proteins", "contigs", "bins", "defense_systems", "secretion_systems", "system_proteins",
              "crispr_cas_systems", "crispr_array_types", "loci", "hydrogenase_classifications", "annotations"]:
        live[t] = con.execute(f"SELECT count(*) FROM c.{t}").fetchone()[0]
    mism = {t: (live[t], counts.get(t)) for t in live if counts.get(t) != live[t]}
    receipt(Path(a.pkg), "verify", dict(db=a.db, seal_id=seal["dataset_id"], seal_generated=seal["generated_at"],
                                        sealed_sha256=rec["digest"]["value"], file_sha256=digest, size=size,
                                        sha_match=ok, live_row_counts=live, row_count_mismatches=mism))
    assert ok and not mism, "canonical copy does not match the seal"


# --------------------------------------------------------------------------- universe
STRAND_SQL = """CASE WHEN trim(p.strand) IN ('+','1','+1') THEN 1
                     WHEN trim(p.strand) IN ('-','-1') THEN -1 ELSE 0 END"""


def st_universe(a):
    pkg = Path(a.pkg)
    con = connect(a)
    raw_strands = con.execute("SELECT strand, count(*) FROM c.proteins GROUP BY 1 ORDER BY 2 DESC").fetchall()
    con.execute(f"""
        CREATE TABLE p AS
        SELECT p.protein_id, p.bin_id AS genome, p.contig_id AS contig, p.gene_index, p.start, p.end_coord,
               p.strand AS strand_raw, {STRAND_SQL} AS strand, p.sequence_length AS length, p.partial,
               (ct.contig_id IS NOT NULL) AS contig_in_table, ct.length AS contig_length
        FROM c.proteins p LEFT JOIN c.contigs ct ON ct.contig_id = p.contig_id AND ct.bin_id = p.bin_id""")
    con.execute("""
        CREATE TABLE gm AS SELECT genome, row_number() OVER (ORDER BY genome) - 1 AS genome_code
        FROM (SELECT DISTINCT genome FROM p)""")
    con.execute("""
        CREATE TABLE cg AS SELECT genome, contig, row_number() OVER (ORDER BY genome, contig) - 1 AS contig_code
        FROM (SELECT DISTINCT genome, contig FROM p)""")
    con.execute("""
        CREATE TABLE u AS
        SELECT row_number() OVER (ORDER BY gm.genome_code, cg.contig_code, p.start, p.end_coord, p.protein_id) - 1 AS gid,
               p.protein_id, p.genome, gm.genome_code::INTEGER AS genome_code, p.contig, cg.contig_code,
               (row_number() OVER (PARTITION BY cg.contig_code ORDER BY p.start, p.end_coord, p.protein_id) - 1) AS position_index,
               p.strand::TINYINT AS strand, p.length, p.start, p.end_coord AS "end", p.gene_index AS db_gene_index,
               p.strand_raw, p.partial, p.contig_in_table, p.contig_length
        FROM p JOIN gm USING (genome) JOIN cg USING (genome, contig)""")
    q = lambda s: con.execute(s).fetchone()
    n, n_pid = q("SELECT count(*), count(DISTINCT protein_id) FROM u")
    assert n == n_pid, (n, n_pid)
    assert q("SELECT min(gid) = 0 AND max(gid) = count(*) - 1 AND count(DISTINCT gid) = count(*) FROM u")[0]
    dup_pos = q("SELECT count(*) FROM (SELECT contig_code, position_index FROM u GROUP BY ALL HAVING count(*) > 1)")[0]
    # gid is contiguous within a contig and follows position_index
    assert q("""SELECT count(*) FROM (SELECT contig_code, max(gid) - min(gid) + 1 AS w, count(*) AS n,
                    max(position_index) AS mp FROM u GROUP BY contig_code) WHERE w <> n OR mp <> n - 1""")[0] == 0
    qc = {}
    qc["proteins"] = n
    qc["genomes"] = q("SELECT count(DISTINCT genome) FROM u")[0]
    qc["contigs_with_genes"] = q("SELECT count(DISTINCT contig_code) FROM u")[0]
    qc["contigs_in_db"] = q("SELECT count(*) FROM c.contigs")[0]
    qc["duplicate_positions"] = dup_pos
    qc["raw_strand_values"] = raw_strands
    qc["strand_minus_fraction"] = q("SELECT avg((strand = -1)::DOUBLE) FROM u")[0]
    qc["strand_unparsed"] = q("SELECT count(*) FROM u WHERE strand = 0")[0]
    qc["single_gene_contigs"] = q("SELECT count(*) FROM (SELECT contig_code FROM u GROUP BY 1 HAVING count(*) = 1)")[0]
    qc["genes_on_single_gene_contigs"] = qc["single_gene_contigs"]
    qc["contig_size_hist"] = con.execute("""SELECT CASE WHEN n=1 THEN '1' WHEN n<=4 THEN '2-4' WHEN n<=9 THEN '5-9'
            WHEN n<=49 THEN '10-49' ELSE '50+' END AS bin, count(*) AS contigs, sum(n) AS genes
            FROM (SELECT contig_code, count(*) n FROM u GROUP BY 1) GROUP BY 1 ORDER BY 1""").fetchall()
    qc["identical_coordinate_groups"] = q("""SELECT count(*), coalesce(sum(k),0) FROM (SELECT contig_code, start, "end",
            count(*) k FROM u GROUP BY ALL HAVING count(*) > 1)""")
    qc["same_start_groups"] = q("""SELECT count(*) FROM (SELECT contig_code, start FROM u GROUP BY ALL HAVING count(*) > 1)""")[0]
    # agreement of the stored gene_index with the coordinate rank
    con.execute("""CREATE TABLE gi AS SELECT gid, contig_code, position_index, db_gene_index,
                   dense_rank() OVER (PARTITION BY contig_code ORDER BY db_gene_index) - 1 AS gi_rank FROM u""")
    qc["db_gene_index_null"] = q("SELECT count(*) FROM u WHERE db_gene_index IS NULL")[0]
    qc["db_gene_index_duplicate_groups"] = q("""SELECT count(*), coalesce(sum(k),0) FROM (SELECT contig_code, db_gene_index,
            count(*) k FROM u GROUP BY ALL HAVING count(*) > 1)""")
    qc["db_gene_index_rank_disagree"] = q("SELECT count(*) FROM gi WHERE gi_rank <> position_index")[0]
    qc["contigs_with_rank_disagreement"] = q("SELECT count(DISTINCT contig_code) FROM gi WHERE gi_rank <> position_index")[0]
    # coordinate sanity
    qc["start_le_zero"] = q("SELECT count(*) FROM u WHERE start <= 0 OR start IS NULL")[0]
    qc["end_lt_start"] = q('SELECT count(*) FROM u WHERE "end" < start')[0]
    qc["contig_not_in_contigs_table"] = q("SELECT count(*) FROM u WHERE NOT contig_in_table")[0]
    qc["end_beyond_contig_length"] = q('SELECT count(*) FROM u WHERE contig_length IS NOT NULL AND "end" > contig_length')[0]
    qc["contig_named_WP"] = q("SELECT count(*), count(DISTINCT genome) FROM u WHERE contig = 'WP'")
    qc["cds_length_not_3x"] = q('SELECT count(*) FROM u WHERE ("end" - start + 1) <> 3 * (length + 1) AND ("end" - start + 1) <> 3 * length')[0]
    # per-genome coordinate flags
    con.execute("""
        CREATE TABLE gflag AS
        SELECT genome, genome_code, count(*) AS n_genes, count(DISTINCT contig_code) AS n_contigs,
               sum((contig = 'WP')::INT) AS n_wp, sum((start <= 0)::INT) AS n_start_le0,
               sum((NOT contig_in_table)::INT) AS n_contig_missing,
               sum((contig_length IS NOT NULL AND "end" > contig_length)::INT) AS n_beyond_contig,
               sum((strand = 0)::INT) AS n_strand_bad,
               avg((strand = -1)::DOUBLE) AS minus_fraction
        FROM u GROUP BY ALL""")
    qc["genomes_with_WP_or_placeholder"] = con.execute("""SELECT genome, n_genes, n_wp, n_start_le0, n_contig_missing
            FROM gflag WHERE n_wp > 0 OR n_start_le0 > 0 OR n_contig_missing > 0 ORDER BY genome""").fetchall()
    qc["genomes_minus_fraction_quantiles"] = q("""SELECT quantile_cont(minus_fraction, [0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0])
            FROM gflag""")[0]
    work = Path(a.work)
    copy(con, "SELECT * FROM u ORDER BY gid", work / "universe_full.parquet")
    copy(con, "SELECT * FROM gflag ORDER BY genome_code", work / "genome_flags.parquet")
    copy(con, """SELECT gid::BIGINT AS gid, protein_id, genome, genome_code, contig, contig_code::BIGINT AS contig_code,
                        position_index::BIGINT AS position_index, strand, length::BIGINT AS length,
                        start::BIGINT AS start, "end"::BIGINT AS "end"
                 FROM u ORDER BY gid""", pkg / "universe" / "genes.parquet")
    copy(con, """SELECT genome, contig, contig_code::BIGINT AS contig_code, count(*)::BIGINT AS n_genes,
                        any_value(contig_length) AS length_nt, min(gid)::BIGINT AS first_gid
                 FROM u GROUP BY ALL ORDER BY contig_code""", pkg / "universe" / "contigs.parquet")
    receipt(pkg, "universe", qc)


# --------------------------------------------------------------------------- prodigal
def _parse_faa(path):
    genome = Path(path).stem
    rows = []
    with open(path) as f:
        for line in f:
            if line[0] != ">":
                continue
            parts = line[1:].split(" # ")
            pid = parts[0]
            rows.append((genome, pid.rsplit("_", 1)[0], int(parts[1]), int(parts[2]), int(parts[3])))
    return rows


def st_prodigal(a):
    files = sorted(str(p) for p in Path(PRODIGAL_DIR).glob("*.faa"))
    with ProcessPoolExecutor(a.threads) as ex:
        chunks = list(ex.map(_parse_faa, files, chunksize=16))
    df = pd.DataFrame([r for c in chunks for r in c], columns=["genome_file", "contig", "start", "end", "strand"])
    # stage-03 file names use GCA_xxx_1 for GCA_xxx.1
    df["genome"] = df.genome_file.str.replace(r"_(\d+)$", r".\1", regex=True)
    out = Path(a.work) / "prodigal_calls.parquet"
    df.drop(columns="genome_file").to_parquet(out, index=False)
    receipt(Path(a.pkg), "prodigal", dict(files=len(files), calls=len(df), genomes=int(df.genome.nunique()),
                                         strand_counts=df.strand.value_counts().to_dict()))


# --------------------------------------------------------------------------- strand QC
def st_strandqc(a):
    pkg, work = Path(a.pkg), Path(a.work)
    con = connect(a)
    con.execute(f"CREATE VIEW u AS SELECT * FROM read_parquet('{work}/universe_full.parquet')")
    con.execute(f"CREATE VIEW pr AS SELECT * FROM read_parquet('{work}/prodigal_calls.parquet')")
    # stop codon position implied by each strand: + -> end, - -> start
    con.execute("""
        CREATE TABLE m AS
        SELECT u.gid, u.genome, u.strand,
               max((pp.start IS NOT NULL)::INT) AS plus_stop_match,
               max((pm.start IS NOT NULL)::INT) AS minus_stop_match,
               max((pp.start = u.start)::INT) AS plus_exact,
               max((pm."end" = u."end")::INT) AS minus_exact
        FROM u
        LEFT JOIN pr pp ON pp.genome = u.genome AND pp.contig = u.contig AND pp.strand = 1 AND pp."end" = u."end"
        LEFT JOIN pr pm ON pm.genome = u.genome AND pm.contig = u.contig AND pm.strand = -1 AND pm.start = u.start
        GROUP BY ALL""")
    con.execute("""
        CREATE TABLE m2 AS SELECT *, CASE WHEN plus_stop_match = 1 AND minus_stop_match = 0 THEN 1
                                          WHEN minus_stop_match = 1 AND plus_stop_match = 0 THEN -1 ELSE 0 END AS implied
        FROM m""")
    q = lambda s: con.execute(s).fetchone()
    res = {}
    res["proteins"] = q("SELECT count(*) FROM m2")[0]
    res["unambiguous_stop_match"] = q("SELECT count(*) FROM m2 WHERE implied <> 0")[0]
    res["both_orientations_match"] = q("SELECT count(*) FROM m2 WHERE plus_stop_match = 1 AND minus_stop_match = 1")[0]
    res["no_match"] = q("SELECT count(*) FROM m2 WHERE plus_stop_match = 0 AND minus_stop_match = 0")[0]
    res["agree"] = q("SELECT count(*) FROM m2 WHERE implied <> 0 AND implied = strand")[0]
    res["disagree"] = q("SELECT count(*) FROM m2 WHERE implied <> 0 AND implied <> strand")[0]
    res["agreement_fraction"] = res["agree"] / max(1, res["agree"] + res["disagree"])
    res["disagree_by_db_strand"] = con.execute("""SELECT strand, implied, count(*) FROM m2 WHERE implied <> 0 AND implied <> strand
                                                  GROUP BY ALL ORDER BY ALL""").fetchall()
    res["agree_by_db_strand"] = con.execute("""SELECT strand, count(*) FROM m2 WHERE implied <> 0 AND implied = strand
                                               GROUP BY ALL ORDER BY ALL""").fetchall()
    res["exact_coordinate_matches"] = q("SELECT sum(greatest(plus_exact, minus_exact)) FROM m2")[0]
    con.execute("""
        CREATE TABLE gs AS SELECT genome, sum((implied <> 0)::INT) AS n_stop_matched,
               sum((implied <> 0 AND implied = strand)::INT) AS n_strand_agree,
               sum((implied <> 0 AND implied <> strand)::INT) AS n_strand_disagree
        FROM m2 GROUP BY genome""")
    con.execute(f"""
        CREATE TABLE gt AS
        SELECT f.*, coalesce(gs.n_stop_matched, 0) AS n_stop_matched, coalesce(gs.n_strand_agree, 0) AS n_strand_agree,
               coalesce(gs.n_strand_disagree, 0) AS n_strand_disagree,
               gs.n_strand_agree / nullif(gs.n_stop_matched, 0) AS strand_agreement,
               gs.n_stop_matched / f.n_genes AS stop_match_fraction
        FROM read_parquet('{work}/genome_flags.parquet') f LEFT JOIN gs USING (genome)""")
    con.execute("""
        ALTER TABLE gt ADD COLUMN order_flags VARCHAR;
        UPDATE gt SET order_flags = concat_ws(';',
            CASE WHEN n_wp > 0 THEN 'synthetic_WP_contig' END,
            CASE WHEN n_start_le0 > 0 THEN 'start_le_0' END,
            CASE WHEN n_contig_missing > 0 THEN 'contig_not_in_assembly' END,
            CASE WHEN n_beyond_contig > 0 THEN 'gene_beyond_contig_end' END,
            CASE WHEN n_strand_bad > 0 THEN 'unparsed_strand' END,
            CASE WHEN n_stop_matched >= 20 AND strand_agreement < 0.98 THEN 'strand_disagrees_with_prodigal' END,
            CASE WHEN n_stop_matched < 20 THEN 'low_prodigal_evidence' END);
        ALTER TABLE gt ADD COLUMN order_trusted BOOLEAN;
        UPDATE gt SET order_trusted = (n_wp = 0 AND n_start_le0 = 0 AND n_contig_missing = 0 AND n_beyond_contig = 0
                                       AND n_strand_bad = 0 AND (n_stop_matched < 20 OR strand_agreement >= 0.98));""")
    res["genomes"] = q("SELECT count(*) FROM gt")[0]
    res["genomes_order_trusted"] = q("SELECT sum(order_trusted::INT) FROM gt")[0]
    res["genomes_by_flag"] = con.execute("""SELECT order_flags, count(*), sum(n_genes) FROM gt GROUP BY 1 ORDER BY 2 DESC""").fetchall()
    res["genome_strand_agreement_quantiles"] = q("""SELECT quantile_cont(strand_agreement, [0.0, 0.01, 0.05, 0.5]) FROM gt
                                                    WHERE n_stop_matched >= 20""")[0]
    res["untrusted_genomes"] = con.execute("""SELECT genome, n_genes, order_flags, n_stop_matched, strand_agreement FROM gt
                                              WHERE NOT order_trusted ORDER BY genome""").fetchall()
    copy(con, "SELECT gid, strand, plus_stop_match, minus_stop_match, implied AS prodigal_strand FROM m2 ORDER BY gid",
         work / "strand_evidence.parquet")
    copy(con, "SELECT * FROM gt ORDER BY genome_code", work / "genome_trust.parquet")
    receipt(pkg, "strandqc", res)


# --------------------------------------------------------------------------- taxonomy
def st_taxonomy(a):
    pkg, work = Path(a.pkg), Path(a.work)
    con = connect(a)
    con.execute(f"""
        CREATE TABLE gtdb AS SELECT regexp_replace(acc, '^(RS_|GB_)', '') AS acc, tax
        FROM read_csv('{GTDB_AR}', delim='\t', header=false, columns={{'acc':'VARCHAR','tax':'VARCHAR'}})""")
    con.execute(f"""
        CREATE TABLE gt AS SELECT * FROM read_parquet('{work}/genome_trust.parquet')""")
    con.execute("""
        CREATE TABLE tx AS
        SELECT gt.genome, gt.genome_code, b.taxonomy AS tax_db, b.completeness, b.contamination,
               g1.tax AS tax_exact, g2.tax AS tax_9digit,
               coalesce(g1.tax, g2.tax) AS tax_gtdb
        FROM gt LEFT JOIN c.bins b ON b.bin_id = gt.genome
        LEFT JOIN gtdb g1 ON g1.acc = gt.genome
        LEFT JOIN (SELECT regexp_extract(acc, 'GC[AF]_([0-9]{9})', 1) AS id9, any_value(tax) AS tax FROM gtdb GROUP BY 1) g2
               ON g2.id9 = regexp_extract(gt.genome, 'GC[AF]_([0-9]{9})', 1) AND g1.tax IS NULL""")
    con.execute("""
        CREATE TABLE tx2 AS SELECT *,
            CASE WHEN tax_exact IS NOT NULL THEN 'gtdb_r232_exact_accession'
                 WHEN tax_9digit IS NOT NULL THEN 'gtdb_r232_accession_9digit'
                 WHEN tax_db IS NOT NULL AND tax_db LIKE 'd__%' THEN 'sharur_bins_taxonomy'
                 ELSE 'none' END AS tax_source,
            CASE WHEN tax_gtdb IS NOT NULL THEN tax_gtdb WHEN tax_db LIKE 'd__%' THEN tax_db END AS tax
        FROM tx""")
    for i, r in enumerate(["domain", "phylum", "class", "ord", "family", "genus", "species"]):
        con.execute(f"ALTER TABLE tx2 ADD COLUMN {r} VARCHAR")
        con.execute(f"UPDATE tx2 SET {r} = nullif(regexp_replace(trim(split_part(tax, ';', {i + 1})), '^[a-z]__', ''), '')")
    q = lambda s: con.execute(s).fetchall()
    res = dict(source_counts=q("SELECT tax_source, count(*) FROM tx2 GROUP BY 1 ORDER BY 2 DESC"),
               phylum_counts=q("SELECT phylum, count(*) FROM tx2 GROUP BY 1 ORDER BY 2 DESC"),
               order_count=q("SELECT count(DISTINCT ord) FROM tx2")[0][0],
               db_vs_gtdb_phylum_disagree=q("""SELECT count(*) FROM tx2 WHERE tax_gtdb IS NOT NULL AND tax_db LIKE 'd__%'
                    AND split_part(tax_gtdb, ';', 2) <> split_part(tax_db, ';', 2)""")[0][0],
               db_vs_gtdb_order_disagree=q("""SELECT count(*) FROM tx2 WHERE tax_gtdb IS NOT NULL AND tax_db LIKE 'd__%'
                    AND split_part(tax_gtdb, ';', 4) <> split_part(tax_db, ';', 4)""")[0][0],
               db_tax_examples=q("SELECT tax_db FROM tx2 LIMIT 3"),
               gtdb_file=GTDB_AR, gtdb_sha256=sha256(GTDB_AR))
    copy(con, """SELECT gt.genome, gt.genome_code, gt.n_genes, gt.n_contigs, tx2.tax, tx2.tax_source, tx2.tax_db,
                        tx2.phylum, tx2.class, tx2.ord, tx2.family, tx2.genus, tx2.species,
                        tx2.completeness, tx2.contamination,
                        gt.order_trusted, gt.order_flags, gt.n_wp::INTEGER AS n_wp, gt.n_start_le0::INTEGER AS n_start_le0,
                        gt.n_contig_missing::INTEGER AS n_contig_missing, gt.n_beyond_contig::INTEGER AS n_beyond_contig,
                        gt.minus_fraction, gt.n_stop_matched::INTEGER AS n_stop_matched,
                        gt.n_strand_agree::INTEGER AS n_strand_agree, gt.n_strand_disagree::INTEGER AS n_strand_disagree,
                        gt.strand_agreement
                 FROM gt JOIN tx2 USING (genome, genome_code) ORDER BY gt.genome_code""",
         pkg / "universe" / "genomes.parquet")
    receipt(pkg, "taxonomy", res)


# --------------------------------------------------------------------------- embeddings
def st_embed(a):
    import h5py
    pkg, work = Path(a.pkg), Path(a.work)
    man = json.loads(Path(H5_MANIFEST).read_text())
    with h5py.File(H5, "r") as f:
        ids = np.array([x.decode() for x in f["protein_ids"][:]], dtype=object)
        attrs = {k: (v.item() if hasattr(v, "item") else v) for k, v in f.attrs.items()}
        shape = f["embeddings"].shape
    assert len(set(ids)) == len(ids), "duplicate IDs in h5"
    con = connect(a)
    con.execute(f"ATTACH '{MARCH_DB}' AS m (READ_ONLY)")
    con.register("h5df", pd.DataFrame({"protein_id": ids, "h5_row": np.arange(len(ids), dtype=np.int64)}))
    con.execute("CREATE TABLE h5ids AS SELECT * FROM h5df")
    con.execute(f"CREATE VIEW u AS SELECT * FROM read_parquet('{work}/universe_full.parquet')")
    # embedded sequence = March DB sequence of the h5 protein id
    con.execute("""
        CREATE TABLE hs AS SELECT h.protein_id, h.h5_row, mp.sequence AS seq, mp.bin_id AS march_genome
        FROM h5ids h LEFT JOIN m.proteins mp USING (protein_id)""")
    h5_missing_in_march = con.execute("SELECT count(*) FROM hs WHERE seq IS NULL").fetchone()[0]
    con.execute("CREATE TABLE hsh AS SELECT protein_id, h5_row, md5(seq) AS h, length(seq) AS L FROM hs WHERE seq IS NOT NULL")
    con.execute("""
        CREATE TABLE cs AS SELECT u.gid, u.protein_id, u.genome, cp.sequence AS seq, md5(cp.sequence) AS h
        FROM u JOIN c.proteins cp USING (protein_id)""")
    con.execute("""
        CREATE TABLE idm AS SELECT cs.gid, hs.h5_row, (hs.seq = cs.seq) AS same_seq,
               (rtrim(hs.seq, '*') = rtrim(cs.seq, '*')) AS same_seq_mod_stop
        FROM cs JOIN hs USING (protein_id) WHERE hs.seq IS NOT NULL""")
    # sequence rescue: identical sequence embedded under another ID
    con.execute("""
        CREATE TABLE resc AS
        SELECT cs.gid, min(hsh.h5_row) AS h5_row
        FROM cs JOIN hsh ON hsh.h = cs.h
        JOIN hs ON hs.h5_row = hsh.h5_row
        WHERE cs.gid NOT IN (SELECT gid FROM idm WHERE same_seq) AND hs.seq = cs.seq
        GROUP BY cs.gid""")
    con.execute("""
        CREATE TABLE em AS
        SELECT u.gid, u.genome, u.length,
               CASE WHEN i.same_seq THEN i.h5_row WHEN r.h5_row IS NOT NULL THEN r.h5_row END AS h5_row,
               CASE WHEN i.same_seq THEN 'id_and_sequence'
                    WHEN r.h5_row IS NOT NULL THEN 'sequence_only'
                    WHEN i.gid IS NOT NULL AND i.same_seq_mod_stop THEN 'id_sequence_differs_only_stop'
                    WHEN i.gid IS NOT NULL THEN 'id_sequence_changed'
                    ELSE 'id_absent_from_h5' END AS match_kind
        FROM u LEFT JOIN idm i USING (gid) LEFT JOIN resc r USING (gid)""")
    q = lambda s: con.execute(s).fetchall()
    res = dict(h5=H5, h5_sha256=sha256(H5), h5_shape=list(shape), h5_attrs=attrs,
               manifest_model=man.get("model_name"), manifest_dim=man.get("embedding_dim"),
               manifest_created=man.get("created_at"), manifest_total=man.get("total_proteins"),
               march_db=MARCH_DB, march_db_sha256=sha256(MARCH_DB),
               h5_ids_missing_from_march_db=h5_missing_in_march,
               match_kinds=q("SELECT match_kind, count(*) FROM em GROUP BY 1 ORDER BY 2 DESC"),
               h5_ids_absent_from_canonical=q("SELECT count(*) FROM h5ids WHERE protein_id NOT IN (SELECT protein_id FROM u)")[0][0])
    em = con.execute("SELECT gid, genome, length, h5_row, match_kind FROM em ORDER BY gid").df()
    n = len(em)
    rows = em.h5_row.astype("float64").to_numpy()
    have = ~np.isnan(rows)
    res["covered_id_and_sequence"] = int((em.match_kind == "id_and_sequence").sum())
    res["covered_any"] = int(have.sum())
    res["coverage_id_and_sequence"] = res["covered_id_and_sequence"] / n
    res["coverage_any"] = res["covered_any"] / n
    # build matrix
    D = shape[1]
    out = np.full((n, D), np.nan, dtype=np.float32)
    with h5py.File(H5, "r") as f:
        X = f["embeddings"][:]  # 4.3 GB, fits in the 96 GB allocation
    sel = rows[have].astype(np.int64)
    V = X[sel].astype(np.float32)
    norms = np.linalg.norm(V, axis=1, keepdims=True)
    res["raw_norm_quantiles"] = np.quantile(norms, [0, 0.01, 0.5, 0.99, 1]).tolist()
    res["zero_norm_rows"] = int((norms[:, 0] == 0).sum())
    V /= np.where(norms == 0, 1, norms)
    out[have] = V
    del X
    edir = pkg / "embeddings"
    edir.mkdir(parents=True, exist_ok=True)
    tmp = edir / ".esm2_8m_march.npy.tmp.npy"
    np.save(tmp, out)
    tmp.rename(edir / "esm2_8m_march.npy")
    em["h5_row"] = em.h5_row.astype("Int64")
    em["truncated_1022"] = em.length > 1022
    em.to_parquet(edir / "embedding_match.parquet", index=False)
    per = em.assign(cov_id=em.match_kind.eq("id_and_sequence"), cov_any=em.h5_row.notna()).groupby("genome").agg(
        n_genes=("gid", "size"), covered_id_and_sequence=("cov_id", "sum"), covered_any=("cov_any", "sum")).reset_index()
    per["coverage_id_and_sequence"] = per.covered_id_and_sequence / per.n_genes
    per["coverage_any"] = per.covered_any / per.n_genes
    per.to_csv(edir / "coverage_per_genome.tsv", sep="\t", index=False)
    res["genome_coverage_any_quantiles"] = per.coverage_any.quantile([0, 0.01, 0.05, 0.5, 1]).tolist()
    res["genomes_coverage_any_ge_0.8"] = int((per.coverage_any >= 0.8).sum())
    res["genomes_coverage_any_ge_0.9"] = int((per.coverage_any >= 0.9).sum())
    res["genomes_coverage_any_eq_0"] = int((per.coverage_any == 0).sum())
    res["genomes"] = len(per)
    res["covered_with_length_gt_1022"] = int((em.h5_row.notna() & (em.length > 1022)).sum())
    res["npy_shape"] = list(out.shape)
    receipt(pkg, "embed", res)


# --------------------------------------------------------------------------- gold loci
def st_gold(a):
    pkg, work = Path(a.pkg), Path(a.work)
    con = connect(a)
    con.execute(f"CREATE VIEW u AS SELECT * FROM read_parquet('{work}/universe_full.parquet')")
    con.execute(f"CREATE VIEW gt AS SELECT * FROM read_parquet('{work}/genome_trust.parquet')")
    q = lambda s: con.execute(s).fetchall()
    census = {}
    census["tables"] = q("""SELECT table_name, estimated_size, column_count FROM duckdb_tables() WHERE database_name='c'
                            ORDER BY 1""")
    census["views"] = q("SELECT view_name FROM duckdb_views() WHERE database_name='c' AND NOT internal ORDER BY 1")
    census["annotation_sources"] = q("""SELECT source, count(*), count(DISTINCT protein_id) FROM c.annotations
                                        GROUP BY 1 ORDER BY 2 DESC""")
    census["system_proteins_sources"] = q("SELECT system_source, count(*), count(DISTINCT system_id) FROM c.system_proteins GROUP BY 1")
    census["crispr_cas_callers"] = q("SELECT caller, status, count(*) FROM c.crispr_cas_systems GROUP BY ALL ORDER BY ALL")
    census["loci_types"] = q("SELECT locus_type, count(*) FROM c.loci GROUP BY 1")
    census["crispr_array_callers"] = q("SELECT caller, count(*) FROM c.crispr_array_types GROUP BY 1")
    census["hydrogenase_outcomes"] = q("SELECT outcome, count(*) FROM c.hydrogenase_classifications GROUP BY 1 ORDER BY 2 DESC")
    # members, one row per (instance, protein) -- from system_proteins for defense/secretion, from the
    # protein_ids list for CRISPR-Cas (CCTyper). Lists are checked against system_proteins where both exist.
    con.execute("""
        CREATE TABLE inst AS
        SELECT 'DefenseFinder' AS caller, 'defense_systems' AS source_table, system_id, genome_id AS genome,
               contig_id AS contig_db, system_type, system_subtype, NULL::VARCHAR AS status, activity AS extra,
               genes_count, protein_ids FROM c.defense_systems
        UNION ALL
        SELECT 'TXSScan', 'secretion_systems', system_id, genome_id, contig_id, system_type, system_subtype, NULL, NULL,
               genes_count, protein_ids FROM c.secretion_systems
        UNION ALL
        SELECT 'CRISPRCasTyper', 'crispr_cas_systems', system_id, genome_id, contig_id, prediction, best_type,
               status, caller, genes_count, protein_ids FROM c.crispr_cas_systems""")
    census["instances_by_caller"] = q("SELECT caller, count(*), count(DISTINCT system_id) FROM inst GROUP BY 1")
    census["protein_ids_examples"] = q("SELECT caller, any_value(protein_ids) FROM inst GROUP BY 1")
    con.execute("""
        CREATE TABLE mem_list AS
        SELECT caller, system_id, trim(trim(x), '"[]'' ') AS protein_id
        FROM (SELECT caller, system_id, unnest(string_split(regexp_replace(coalesce(protein_ids, ''), '[\\[\\]"'']', '', 'g'), ',')) AS x
              FROM inst) WHERE trim(x) <> ''""")
    con.execute("""
        CREATE TABLE mem_sp AS SELECT CASE system_source WHEN 'defensefinder_system' THEN 'DefenseFinder'
                                                          WHEN 'txsscan_system' THEN 'TXSScan'
                                                          WHEN 'cctyper' THEN 'CRISPRCasTyper' END AS caller,
               system_id, protein_id, system_source, position, profile_name FROM c.system_proteins""")
    unk = q("SELECT DISTINCT system_source FROM c.system_proteins WHERE system_source NOT IN ('defensefinder_system', 'txsscan_system', 'cctyper')")
    assert not unk, f"unmapped system_source values {unk}"
    census["system_proteins_vs_list"] = q("""
        SELECT l.caller, count(*) AS list_rows, sum((s.protein_id IS NOT NULL)::INT) AS in_system_proteins
        FROM mem_list l LEFT JOIN (SELECT DISTINCT caller, system_id, protein_id FROM mem_sp) s USING (caller, system_id, protein_id)
        GROUP BY 1""")
    census["system_proteins_not_in_list"] = q("""
        SELECT s.caller, count(*) FROM (SELECT DISTINCT caller, system_id, protein_id FROM mem_sp) s
        LEFT JOIN mem_list l USING (caller, system_id, protein_id) WHERE l.protein_id IS NULL GROUP BY 1""")
    con.execute("""
        CREATE TABLE mem AS
        SELECT DISTINCT caller, system_id, protein_id FROM mem_list
        UNION SELECT DISTINCT caller, system_id, protein_id FROM mem_sp""")
    con.execute("""
        CREATE TABLE memu AS SELECT m.caller, m.system_id, m.protein_id, u.gid, u.genome, u.contig, u.contig_code,
               u.position_index, u.start, u."end"
        FROM mem m LEFT JOIN u USING (protein_id)""")
    census["member_rows"] = q("SELECT caller, count(*), sum((gid IS NULL)::INT) AS unresolved FROM memu GROUP BY 1")
    con.execute("""
        CREATE TABLE g AS
        SELECT i.caller, i.source_table, i.system_id, i.genome AS genome_db, i.contig_db, i.system_type, i.system_subtype,
               i.status, i.extra, i.genes_count,
               count(m.protein_id) AS n_members_listed, count(m.gid) AS n_members_resolved,
               count(DISTINCT m.contig_code) AS n_contigs, any_value(m.genome) AS genome, any_value(m.contig) AS contig,
               any_value(m.contig_code) AS contig_code, min(m.position_index) AS start_idx, max(m.position_index) AS end_idx,
               min(m.start) AS start_nt, max(m."end") AS end_nt,
               list(m.gid ORDER BY m.position_index) FILTER (WHERE m.gid IS NOT NULL) AS member_gids,
               list(m.position_index ORDER BY m.position_index) FILTER (WHERE m.gid IS NOT NULL) AS member_positions
        FROM inst i LEFT JOIN memu m ON m.caller = i.caller AND m.system_id = i.system_id
        GROUP BY ALL""")
    con.execute("""
        CREATE TABLE g2 AS SELECT g.*, coalesce(n_members_resolved > 0 AND n_members_resolved = n_members_listed AND n_contigs = 1
                                       AND (genome = genome_db OR genome_db IS NULL), false) AS resolved,
               end_idx - start_idx + 1 AS span_genes, t.order_trusted
        FROM g LEFT JOIN (SELECT genome, order_trusted FROM gt) t USING (genome)""")
    con.execute("""
        CREATE TABLE g3 AS SELECT row_number() OVER (ORDER BY caller, genome, contig_code, start_idx, system_id) - 1 AS instance_id, *
        FROM g2""")
    census["instances"] = q("""SELECT caller, count(*), sum(resolved::INT), sum((n_members_resolved=1)::INT) AS single_member,
                                      sum((resolved AND order_trusted)::INT) FROM g3 GROUP BY 1 ORDER BY 1""")
    census["unresolved_reasons"] = q("""SELECT caller,
            sum((n_members_listed = 0)::INT) AS no_members, sum((n_members_resolved < n_members_listed)::INT) AS member_missing,
            sum((n_contigs > 1)::INT) AS multi_contig, sum((genome <> genome_db)::INT) AS genome_mismatch
            FROM g3 WHERE NOT resolved GROUP BY 1""")
    census["contig_db_mismatch"] = q("SELECT caller, sum((contig_db IS NOT NULL AND contig <> contig_db)::INT) FROM g3 GROUP BY 1")
    census["system_types"] = q("""SELECT caller, count(DISTINCT system_type) FROM g3 GROUP BY 1""")
    census["genes_count_vs_members"] = q("""SELECT caller, sum((genes_count = n_members_listed)::INT), count(*) FROM g3 GROUP BY 1""")
    census["span_vs_members"] = q("""SELECT caller, quantile_cont(span_genes, [0.5, 0.9, 0.99, 1.0]),
            quantile_cont(n_members_resolved, [0.5, 0.9, 1.0]) FROM g3 WHERE resolved GROUP BY 1""")
    gdir = pkg / "gold"
    # per-protein annotation flags
    srcs = [r[0] for r in census["annotation_sources"]]
    flag_cols = ",\n".join(f"bool_or(source = '{s}') AS has_{''.join(ch if ch.isalnum() else '_' for ch in s.lower())}" for s in srcs)
    con.execute(f"""CREATE TABLE af AS SELECT protein_id, {flag_cols} FROM c.annotations GROUP BY protein_id""")
    # per-domain profile sources; *_system rows are caller memberships and hyddb_subgroup derives from hyddb
    domain_srcs = [s for s in srcs if s.lower() in ("pfam", "dbcan", "hyddb", "defensefinder", "txsscan", "tigrfam", "cazy")]
    orth_srcs = [s for s in srcs if s.lower() in ("kofam", "kegg")]
    census["sources_not_in_domain_or_orthology"] = [s for s in srcs if s not in domain_srcs + orth_srcs]
    col = lambda s: "has_" + "".join(ch if ch.isalnum() else "_" for ch in s.lower())
    dom_expr = " OR ".join(f"coalesce(af.{col(s)}, false)" for s in domain_srcs) or "false"
    orth_expr = " OR ".join(f"coalesce(af.{col(s)}, false)" for s in orth_srcs) or "false"
    pf_expr = "coalesce(af.has_pfam, false)" if "pfam" in [s.lower() for s in srcs] else "false"
    ko_expr = " OR ".join(f"coalesce(af.{col(s)}, false)" for s in orth_srcs) or "false"
    copy(con, f"""SELECT u.gid, {pf_expr} AS has_pfam, ({dom_expr}) AS has_domain, ({orth_expr}) AS has_orthology,
                         NOT ({dom_expr} OR {orth_expr}) AS unannotated,
                         NOT ({pf_expr} OR {ko_expr}) AS no_pfam_no_ko,
                         {", ".join(f"coalesce(af.{col(s)}, false) AS src_{col(s)[4:]}" for s in srcs)}
                  FROM u LEFT JOIN af USING (protein_id) ORDER BY u.gid""", pkg / "annotation_flags.parquet")
    con.execute(f"CREATE VIEW fl AS SELECT * FROM read_parquet('{pkg}/annotation_flags.parquet')")
    census["domain_sources"] = domain_srcs
    census["orthology_sources"] = orth_srcs
    census["flags"] = q("""SELECT count(*), sum(has_pfam::INT), sum(has_domain::INT), sum(has_orthology::INT),
                                  sum(unannotated::INT) FROM fl""")[0]
    con.execute("""
        CREATE TABLE g4 AS
        SELECT g3.*, coalesce(x.n_unannotated, 0)::INTEGER AS n_members_unannotated,
               coalesce(x.n_no_pfam, 0)::INTEGER AS n_members_no_pfam,
               coalesce(y.n_span_unannotated, 0)::INTEGER AS n_span_unannotated
        FROM g3
        LEFT JOIN (SELECT t.instance_id, sum(fl.unannotated::INT) AS n_unannotated, sum((NOT fl.has_pfam)::INT) AS n_no_pfam
                   FROM (SELECT instance_id, unnest(member_gids) AS gid FROM g3) t JOIN fl ON fl.gid = t.gid GROUP BY 1) x
                  USING (instance_id)
        LEFT JOIN (SELECT g3.instance_id, sum(fl.unannotated::INT) AS n_span_unannotated
                   FROM g3 JOIN u ON u.contig_code = g3.contig_code AND u.position_index BETWEEN g3.start_idx AND g3.end_idx
                   JOIN fl ON fl.gid = u.gid WHERE g3.resolved GROUP BY 1) y USING (instance_id)""")
    census["instance_unannotated_members"] = q("""SELECT caller, sum(n_members_resolved), sum(n_members_unannotated),
            sum(span_genes), sum(n_span_unannotated) FROM g4 WHERE resolved GROUP BY 1 ORDER BY 1""")
    copy(con, "SELECT * FROM g4 ORDER BY instance_id", gdir / "instances.parquet")
    copy(con, """SELECT instance_id, caller, unnest(member_gids) AS gid FROM g4 WHERE resolved ORDER BY 1, 3""",
         gdir / "members.parquet")
    receipt(pkg, "gold", census)


# --------------------------------------------------------------------------- fasta
def st_fasta(a):
    pkg, work = Path(a.pkg), Path(a.work)
    con = connect(a)
    con.execute(f"CREATE VIEW u AS SELECT * FROM read_parquet('{work}/universe_full.parquet')")
    df = con.execute("""SELECT u.gid, rtrim(cp.sequence, '*') AS seq FROM u JOIN c.proteins cp USING (protein_id)
                        ORDER BY u.gid""").df()
    assert len(df) == con.execute("SELECT count(*) FROM u").fetchone()[0]
    empty = int((df.seq.str.len().fillna(0) == 0).sum())
    internal_stop = int(df.seq.str.contains(r"\*", regex=True).sum())
    out = work / "proteins_gid.faa"
    tmp = work / ".proteins_gid.faa.tmp"
    with open(tmp, "w") as f:
        for gid, s in zip(df.gid.to_numpy(), df.seq.to_numpy()):
            f.write(f">{gid}\n{s}\n")
    tmp.rename(out)
    receipt(pkg, "fasta", dict(path=str(out), records=len(df), empty_sequences=empty,
                               internal_stop_sequences=internal_stop, sha256=sha256(out)))


STAGES = dict(verify=st_verify, universe=st_universe, prodigal=st_prodigal, strandqc=st_strandqc,
              taxonomy=st_taxonomy, embed=st_embed, gold=st_gold, fasta=st_fasta)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--seal", required=True)
    ap.add_argument("--pkg", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--threads", type=int, required=True)
    ap.add_argument("stages", nargs="+", choices=list(STAGES))
    a = ap.parse_args()
    Path(a.work, "duck_tmp").mkdir(parents=True, exist_ok=True)
    for s in a.stages:
        t0 = time.time()
        print(f"== stage {s} start {time.strftime('%H:%M:%S')}", flush=True)
        STAGES[s](a)
        print(f"== stage {s} done in {time.time() - t0:.1f} s", flush=True)


if __name__ == "__main__":
    main()
