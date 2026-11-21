#!/usr/bin/env python3
"""Summarize the search benchmark: hits, candidate hits lost against the most sensitive setting,
prefilter saturation, timing and extrapolation to the full query sets."""
import argparse, json, os, re
import duckdb

ap = argparse.ArgumentParser()
ap.add_argument("--work", required=True)
a = ap.parse_args()
W = a.work; B = f"{W}/bench"
prep = json.load(open(f"{W}/prep.json"))
con = duckdb.connect(); con.execute(f"SET threads={os.environ.get('SLURM_CPUS_ON_NODE', '4')}")
COLS = "{'q':'BIGINT','t':'BIGINT','evalue':'DOUBLE','bits':'DOUBLE','alnlen':'BIGINT','qstart':'BIGINT','qend':'BIGINT','qlen':'BIGINT','tstart':'BIGINT','tend':'BIGINT','tlen':'BIGINT'}"

def tparse(log):
    out = {}
    for line in open(log, errors="ignore"):
        m = re.match(r"Time for (.+?): (\d+)h (\d+)m (\d+)s (\d+)ms", line.strip())
        if m:
            k = m.group(1); out[k] = out.get(k, 0) + int(m.group(2)) * 3600 + int(m.group(3)) * 60 + int(m.group(4)) + int(m.group(5)) / 1000
        m = re.search(r"k-mer size[:\s]+(\d+)|Kmer size[:\s]+(\d+)|^k-mer length[:\s]+(\d+)", line)
        if m:
            out["kmer"] = [g for g in m.groups() if g][0]
    return out

res = {"prep": prep, "configs": {}}
names = [n for n in ["s75_m4000", "s57_m4000", "s57_m1000", "s75_m1000", "giant_s75_m20000"] if os.path.exists(f"{B}/{n}/timing.json")]
for n in names:
    O = f"{B}/{n}"
    tm = json.load(open(f"{O}/timing.json"))
    con.execute(f"""CREATE OR REPLACE TABLE h_{n} AS SELECT * FROM read_csv('{O}/aln.tsv', delim='\t', header=false, columns={COLS}) WHERE q<>t""")
    con.execute(f"CREATE OR REPLACE TABLE lc_{n} AS SELECT * FROM read_csv('{O}/pref_lc.tsv', delim='\t', header=false, columns={{'q':'BIGINT','n':'BIGINT'}})")
    r = con.execute(f"""SELECT count(*) n_hits, count(DISTINCT q) n_q_with_hit,
        sum(((qend-qstart+1)>=80 AND (qend-qstart+1)<=qlen-60 AND (tend-tstart+1)>=0.7*tlen)::int) comp_cand,
        sum(((tend-tstart+1)>=0.8*tlen AND (qend-qstart+1)<=0.5*qlen)::int) frag_cand,
        sum((alnlen>=100)::int) aln100
        FROM h_{n}""").fetchone()
    lc = con.execute(f"SELECT count(*), avg(n), quantile_cont(n,0.5), quantile_cont(n,0.99), sum((n>={tm['max_seqs']})::int) FROM lc_{n}").fetchone()
    d = dict(tm)
    d.update(tparse(f"{O}/prefilter.log"))
    d.update({"n_hits_nonself": r[0], "n_queries_with_hit": r[1], "composite_candidate_hits": r[2],
              "fragment_candidate_hits": r[3], "hits_alnlen_ge100": r[4],
              "prefilter_queries": lc[0], "prefilter_mean": lc[1], "prefilter_median": lc[2], "prefilter_q99": lc[3],
              "prefilter_saturated": lc[4], "aln_tsv_bytes": os.path.getsize(f"{O}/aln.tsv")})
    res["configs"][n] = d

ref = "s75_m4000"
if ref in names:
    for n in names:
        if n == ref or n.startswith("giant"):
            continue
        r = con.execute(f"""
          WITH R AS (SELECT q,t,(qend-qstart+1)>=80 AND (qend-qstart+1)<=qlen-60 c, (tend-tstart+1)>=0.8*tlen AND (qend-qstart+1)<=0.5*qlen f FROM h_{ref}),
               X AS (SELECT q,t FROM h_{n})
          SELECT count(*) n_ref, sum((X.q IS NOT NULL)::int) found,
                 sum(c::int) n_c, sum((c AND X.q IS NOT NULL)::int) found_c,
                 sum(f::int) n_f, sum((f AND X.q IS NOT NULL)::int) found_f
          FROM R LEFT JOIN X USING (q,t)""").fetchone()
        extra = con.execute(f"SELECT count(*) FROM h_{n} X ANTI JOIN h_{ref} R USING (q,t)").fetchone()[0]
        res["configs"][n]["vs_ref"] = {"ref": ref, "ref_hits": r[0], "recovered_frac": r[1] / r[0],
                                       "composite_cand_recovered_frac": r[3] / max(r[2], 1),
                                       "fragment_cand_recovered_frac": r[5] / max(r[4], 1),
                                       "hits_not_in_ref": extra}

# extrapolation: per-query cost scales with query residues; index init is a fixed cost
fac_ng = prep["residues_nongiants"] / prep["bench20k_residues"]
fac_g = prep["residues_giants"] / prep["bench_giant200_residues"]
for n, d in res["configs"].items():
    fac = fac_g if n.startswith("giant") else fac_ng
    fixed = d.get("index table init", 0.0)
    var = d["prefilter_s"] - fixed + d["align_s"] + d["convert_s"]
    d["extrapolation_factor"] = fac
    d["extrapolated_hours_full"] = (fixed + var * fac) / 3600
    d["extrapolated_hits_full"] = d["n_hits_nonself"] * fac
    d["extrapolated_tsv_GB_full"] = d["aln_tsv_bytes"] * fac / 1e9
json.dump(res, open(f"{B}/bench_summary.json", "w"), indent=1, default=float)
print(json.dumps(res["configs"], indent=1, default=float))
