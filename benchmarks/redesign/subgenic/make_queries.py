#!/usr/bin/env python3
"""Query set for the sampled design.

S = genes of a random sample of genomes (seed 17)  U  gold neighbourhoods (every member gene of a gold
instance with >= 2 resolved members, plus the genes within 3 positions of it on its contig).
Query uids = unique sequences of S below the giant length (giants are searched separately, search B).
Every measurement that is reported for a gene needs only that gene's own hits (and, for the split side,
its neighbours' hits), so genes in S are measured exactly; fractions are extrapolated from the genome sample.
"""
import argparse, json, os
import numpy as np, duckdb

ap = argparse.ArgumentParser()
ap.add_argument("--work", required=True); ap.add_argument("--pkg", required=True)
ap.add_argument("--n-genomes", type=int, required=True); ap.add_argument("--seed", type=int, default=17)
ap.add_argument("--giant", type=int, default=2046); ap.add_argument("--dry", action="store_true")
a = ap.parse_args(); W, P = a.work, a.pkg
con = duckdb.connect(); con.execute(f"SET threads={os.environ.get('SLURM_CPUS_ON_NODE', '2')}")
gcodes = con.execute(f"SELECT genome_code FROM '{P}/universe/genomes.parquet' ORDER BY genome_code").fetchnumpy()["genome_code"]
rng = np.random.default_rng(a.seed)
samp = np.sort(rng.choice(gcodes, a.n_genomes, replace=False))
con.register("sg", {"genome_code": samp})
con.execute(f"""CREATE TABLE g AS SELECT g.gid, g.genome_code, g.contig_code, g.position_index pos, m.uid, u.length
  FROM '{P}/universe/genes.parquet' g JOIN '{W}/gid2uid.parquet' m USING (gid) JOIN '{W}/uniq.parquet' u USING (uid)""")
con.execute(f"""CREATE TABLE mem AS SELECT DISTINCT mb.gid FROM '{P}/gold/members.parquet' mb
  JOIN '{P}/gold/instances.parquet' i USING (instance_id) WHERE i.n_members_resolved >= 2""")
con.execute("""CREATE TABLE nb AS SELECT DISTINCT g2.gid FROM mem JOIN g g1 USING (gid)
  JOIN g g2 ON g2.contig_code = g1.contig_code AND abs(g2.pos - g1.pos) <= 3""")
con.execute("""CREATE TABLE S AS SELECT g.*, (g.genome_code IN (SELECT genome_code FROM sg)) in_sample,
  (g.gid IN (SELECT gid FROM nb)) in_gold_nbhd FROM g WHERE g.genome_code IN (SELECT genome_code FROM sg) OR g.gid IN (SELECT gid FROM nb)""")
r = con.execute(f"""SELECT count(*) genes, sum(in_sample::int) sample_genes, sum(in_gold_nbhd::int) gold_nbhd_genes,
  (SELECT count(*) FROM mem) gold_member_genes,
  count(DISTINCT uid) uids, count(DISTINCT uid) FILTER (WHERE length < {a.giant}) uids_nongiant,
  sum(length) FILTER (WHERE length < {a.giant}) res_genes_nongiant,
  (SELECT sum(length) FROM (SELECT DISTINCT uid, length FROM S WHERE length < {a.giant})) res_uids_nongiant
  FROM S""").df().iloc[0].to_dict()
r = {k: int(v) for k, v in r.items()}; r["n_genomes_sampled"] = int(a.n_genomes); r["seed"] = a.seed
print(json.dumps(r))
if not a.dry:
    np.savetxt(f"{W}/lists/sample_genomes.txt", samp, fmt="%d")
    q = con.execute(f"SELECT DISTINCT uid FROM S WHERE length < {a.giant} ORDER BY uid").fetchnumpy()["uid"]
    np.savetxt(f"{W}/lists/S.uid", q, fmt="%d")
    con.execute(f"COPY (SELECT gid, in_sample, in_gold_nbhd FROM S ORDER BY gid) TO '{W}/lists/S_genes.parquet' (FORMAT parquet)")
    json.dump(r, open(f"{W}/lists/S.json", "w"), indent=1)
