#!/usr/bin/env python3
"""Split unification, stage 1: X candidates from the shorter piece's hits; explicit pair list for stage 2.

usage: split_candidates.py <repair dir> <subgenic dir>
Reads pairs.parquet, hits/Q_*.parquet, subgenic uniq.parquet and predeclared.json.
Writes split/cand.parquet (pair, X, stage-1 region) and split/pref.tsv (uid_l, X, 0, 0; sorted by uid_l) for
`mmseqs tsv2db --output-dbtype 7` + `mmseqs align`.
"""
import json, os, sys, time
import duckdb

R, SG = sys.argv[1], sys.argv[2]
PD = json.load(open(f"{R}/code/predeclared.json"))
T = os.environ.get("SLURM_CPUS_ON_NODE", "2")
YC, RMIN, OV, NCAND = 0.7, 80, 20, 30          # predeclared split.piece_hit / x_rule / search_route
assert "0.7" in PD["split"]["piece_hit"] and ">= 80 aa" in PD["split"]["piece_hit"] and "at most 30" in PD["split"]["search_route"]
os.makedirs(f"{R}/split", exist_ok=True); os.makedirs(f"{R}/tmp/duck", exist_ok=True)
con = duckdb.connect(f"{R}/tmp/split_cand.duckdb")
con.execute(f"SET threads={T}; SET temp_directory='{R}/tmp/duck'; SET preserve_insertion_order=false")
t0 = time.time()
def log(*x): print(f"[{time.time() - t0:7.1f}s]", *x, flush=True)

# pairs: N-terminal piece in genomic order (+ strand: lower position; - strand: higher position)
con.execute(f"""CREATE OR REPLACE TABLE pr AS SELECT ga, gb, genome_code, strand, gap_nt,
     CASE WHEN short_is_a THEN uid_a ELSE uid_b END us, CASE WHEN short_is_a THEN uid_b ELSE uid_a END ul,
     CASE WHEN short_is_a THEN len_a ELSE len_b END ls, CASE WHEN short_is_a THEN len_b ELSE len_a END ll,
     ((strand > 0) = short_is_a) s_is_n
  FROM '{R}/pairs.parquet'""")
log("pairs", con.execute("SELECT count(*) FROM pr").fetchone()[0])
# stage-1 piece hits (pair-independent filters)
con.execute(f"""CREATE OR REPLACE TABLE h1 AS SELECT q::BIGINT q, t::BIGINT t, evalue, bits, fident,
     qstart::INT qs, qend::INT qe, qlen::INT ql, tstart::INT ts, tend::INT te, tlen::INT tl
  FROM read_parquet('{R}/hits/Q_*.parquet')
  WHERE evalue <= 1e-5 AND (qend::INT - qstart::INT + 1) >= {YC} * qlen::INT
    AND (tend::INT - tstart::INT + 1) >= {RMIN} AND (tend::INT - tstart::INT + 1) <= tlen::INT - ({RMIN} - {OV})
    AND q IN (SELECT DISTINCT us FROM pr)""")
log("stage-1 piece hits", con.execute("SELECT count(*), count(DISTINCT q) FROM h1").fetchone())
con.execute(f"CREATE OR REPLACE TABLE uq AS SELECT uid, n_genomes, genome_code_first FROM '{SG}/uniq.parquet'")
# pair-dependent necessary conditions of the x_rule:
#  room for the other piece's region on the correct side (>= RMIN - OV), uncovered outside the union <= 0.2 len(X),
#  len(X) <= (region_s + 1.5 len_l) / 0.8, X not the long piece, X present in another genome
con.execute(f"""CREATE OR REPLACE TABLE cand AS
  SELECT * FROM (
    SELECT p.ga, p.gb, p.genome_code, p.strand, p.gap_nt, p.us, p.ul, p.ls, p.ll, p.s_is_n, h.t x, h.tl xlen, h.ts s_ts, h.te s_te,
           h.bits s_bits, h.evalue s_e, h.fident s_fid,
           row_number() OVER (PARTITION BY p.ga, p.gb ORDER BY h.bits DESC, h.evalue, h.t) rk
    FROM pr p JOIN h1 h ON h.q = p.us JOIN uq u ON u.uid = h.t
    WHERE h.t <> p.ul
      AND NOT (u.n_genomes = 1 AND u.genome_code_first = p.genome_code)
      AND CASE WHEN p.s_is_n THEN (h.tl - h.te) >= {RMIN - OV} AND (h.ts - 1) <= 0.2 * h.tl
               ELSE (h.ts - 1) >= {RMIN - OV} AND (h.tl - h.te) <= 0.2 * h.tl END
      AND h.tl <= ((h.te - h.ts + 1) + 1.5 * p.ll) / 0.8)
  WHERE rk <= {NCAND}""")
n = con.execute("SELECT count(*), count(DISTINCT (ga, gb)), count(DISTINCT (ul, x)) FROM cand").fetchone()
log("candidates rows / pairs / distinct (long piece, X)", n)
con.execute(f"COPY (SELECT * FROM cand ORDER BY ga, gb, rk) TO '{R}/split/cand.parquet' (FORMAT parquet)")
con.execute(f"COPY (SELECT DISTINCT ul, x, 0, 0 FROM cand ORDER BY ul, x) TO '{R}/split/pref.tsv' (DELIMITER '\t', HEADER false)")
json.dump({"pairs": con.execute("SELECT count(*) FROM pr").fetchone()[0],
           "stage1_piece_hits": con.execute("SELECT count(*) FROM h1").fetchone()[0],
           "cand_rows": n[0], "cand_pairs": n[1], "stage2_alignments": n[2], "seconds": round(time.time() - t0, 1)},
          open(f"{R}/split/cand.json", "w"), indent=1)
log("done")
