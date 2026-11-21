#!/usr/bin/env python3
"""rescue3 token arm for a slice package (workstream slice_lut). Same rule and code path as repair/build_arms.py
(rescue part, code sha256 86064004...) with resolve_moves copied unchanged from repair/repair_lib.py:
a gene whose base token has fewer than 3 genes moves to the token of its best qualifying hit (E <= 1e-5, alignment
covers >= 80% of the shorter protein, target token has >= 2 genes and differs from its own; best = highest bits,
then lowest E, larger target token, lower target uid), following pointers when the target gene itself moves.

usage: 06b_rescue3.py WORK_DIR HITS_PARQUET BASE_TOKEN_PARQUET OUT_PATH_NO_EXT THREADS
"""
import hashlib, json, os, sys, time
import numpy as np, pandas as pd, duckdb
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from repair_lib import resolve_moves

W, HITS, BASE, OUT, T = sys.argv[1:6]
t0 = time.time()
COVS = 0.8
con = duckdb.connect()
con.execute(f"SET threads={T}; SET preserve_insertion_order=false; SET temp_directory='{W}/duck_tmp'")
g2u = pd.read_parquet(f"{W}/gid2uid.parquet").sort_values("gid")
uid_of = g2u.uid.values.astype(np.int64); NG = len(uid_of)
tok = pd.read_parquet(BASE).sort_values("gid"); assert np.array_equal(tok.gid.values, np.arange(NG))
base_tok = tok.token.values.astype(np.int64)
NU = int(uid_of.max()) + 1
v, inv, c = np.unique(base_tok, return_inverse=True, return_counts=True)
base_size = c[inv]
size_of_token = dict(zip(v.tolist(), c.tolist()))
df = pd.DataFrame({"uid": uid_of, "tok": base_tok}).groupby(["uid", "tok"]).size().reset_index(name="n")
df = df.sort_values(["uid", "n", "tok"], ascending=[True, False, True]).drop_duplicates("uid")
tok_u = np.full(NU, -1, np.int64); tok_u[df.uid.values] = df.tok.values
assert (tok_u >= 0).all()
tsize_u = np.array([size_of_token[t] for t in tok_u], np.int64)
con.register("tku", pd.DataFrame({"uid": np.arange(NU), "tok_t": tok_u, "size_t": tsize_u}))
q_genes = pd.DataFrame({"gid": np.flatnonzero(base_size < 3)}); q_genes["uid"] = uid_of[q_genes.gid.values]
q_genes["tok"] = base_tok[q_genes.gid.values]; q_genes["qsize"] = base_size[q_genes.gid.values]
con.register("qg", q_genes)
con.execute(f"""CREATE OR REPLACE TABLE rh AS
  SELECT h.q::BIGINT q, h.t::BIGINT t, h.evalue, h.bits, h.fident, h.qstart::INT qs, h.qend::INT qe, h.qlen::INT ql,
         h.tstart::INT ts, h.tend::INT te, h.tlen::INT tl, k.tok_t, k.size_t,
         CASE WHEN h.qlen <= h.tlen THEN (h.qend::INT - h.qstart::INT + 1) / h.qlen::DOUBLE
              ELSE (h.tend::INT - h.tstart::INT + 1) / h.tlen::DOUBLE END cov_short
  FROM read_parquet('{HITS}') h JOIN tku k ON k.uid = h.t
  WHERE h.evalue <= 1e-5 AND k.size_t >= 2 AND h.q IN (SELECT DISTINCT uid FROM qg)""")
con.execute(f"DELETE FROM rh WHERE cov_short < {COVS}")
best = con.execute("""SELECT * FROM (
    SELECT qg.gid, qg.qsize, rh.*, row_number() OVER (PARTITION BY qg.gid ORDER BY rh.bits DESC, rh.evalue, rh.size_t DESC, rh.t) rk
    FROM qg JOIN rh ON rh.q = qg.uid WHERE rh.tok_t <> qg.tok) WHERE rk = 1""").df()
b = best[best.qsize < 3].copy()
rep = pd.DataFrame({"gid": np.arange(NG), "uid": uid_of, "tok": base_tok})
rep = rep[rep.uid.isin(b.t.unique())]
rep = rep[rep.tok.values == tok_u[rep.uid.values]].drop_duplicates("uid")
rep_of = dict(zip(rep.uid.values, rep.gid.values))
nxt = {int(g): int(rep_of[t]) for g, t in zip(b.gid.values, b.t.values)}
final, n_chain, n_cycle = resolve_moves(nxt, base_tok, size_of_token)
new = base_tok.copy()
if final:
    gids = np.fromiter(final.keys(), np.int64); new[gids] = np.fromiter(final.values(), np.int64)
ch = new != base_tok
sz = np.unique(new, return_counts=True)[1]
tmp = OUT + ".parquet.tmp"
pd.DataFrame({"gid": np.arange(NG, dtype=np.int64), "token": new.astype(np.int64)}).to_parquet(tmp, index=False)
os.replace(tmp, OUT + ".parquet")
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
a = {"arm": os.path.basename(OUT), "repair": "rescue3", "rows": NG, "n_tokens": int(len(sz)), "n_assigned": NG,
     "query_genes": int((base_size < 3).sum()), "genes_with_qualifying_hit": int(len(b)), "chains_followed": n_chain,
     "cycles": n_cycle, "genes_changed": int(ch.sum()), "genes_changed_frac": float(ch.mean()),
     "base_tokens_emptied": int(len(np.setdiff1d(np.unique(base_tok), np.unique(new)))),
     "receiving_tokens": int(len(np.unique(new[ch]))), "singletons": int((sz == 1).sum()),
     "largest_before": int(base_size.max()), "largest_after": int(sz.max()),
     "token_size_q50_q90_q99": np.quantile(sz, [0.5, 0.9, 0.99]).tolist(),
     "rule": "rescue3 (repair/build_arms.py, rescue part; resolve_moves from repair/repair_lib.py)",
     "search": "uniq-sequence prefilter -s 7.5 --max-seqs 1000; align -e 1e-5 -a (repair/search_shard.sbatch)",
     "base": BASE, "hits": HITS, "code_sha256": sha(__file__), "parquet_sha256": sha(OUT + ".parquet"),
     "date": time.strftime("%Y-%m-%d"), "seconds": round(time.time() - t0, 1)}
json.dump(a, open(OUT + ".json", "w"), indent=1)
pd.DataFrame({"gid": np.flatnonzero(ch), "base_token": base_tok[ch], "token": new[ch]}).to_parquet(f"{W}/changed_rescue3.parquet", index=False)
print(json.dumps(a, indent=1), flush=True)
