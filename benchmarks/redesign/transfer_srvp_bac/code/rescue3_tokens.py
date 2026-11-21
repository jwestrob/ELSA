#!/usr/bin/env python3
"""rescue3 fragment rescue applied to a base MMseqs2 family table (workstream transfer_srvp_bac).

The rescue part of $RD/repair/code/build_arms.py (sha256 86064004...), copied line for line with the split repair,
the rescue1 arm and the sizing-study validation removed. Rule (repair/code/predeclared.json, adopted by Jacob on
2026-11-12 12:43 UTC): a gene in a base token with < 3 members joins the base token of its best hit when the hit has
E <= 1e-5, the alignment covers >= 80% of the shorter protein, and the hit's token (plurality token of the hit
sequence's genes) has >= 2 members and differs from the gene's own; best hit by bits, then E, then larger target
token, then lower target uid; moves resolved together with pointer following (repair_lib.resolve_moves).

usage: rescue3_tokens.py --gid2uid P --uniq P --base P --hits GLOB --out PATH_NO_EXT --tmp DIR [--expect P]
--expect: a token parquet that the result must equal row for row (replay check on DPANN inputs).
"""
import argparse, hashlib, json, os, sys, time
import numpy as np, pandas as pd, duckdb
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from repair_lib import resolve_moves

ap = argparse.ArgumentParser()
ap.add_argument("--gid2uid", required=True)
ap.add_argument("--uniq", required=True)
ap.add_argument("--base", required=True)
ap.add_argument("--hits", required=True, help="glob of hit parquets (q, t uids; evalue, bits, fident, qstart..tlen)")
ap.add_argument("--out", required=True, help="output path without extension; arm name = basename")
ap.add_argument("--tmp", required=True)
ap.add_argument("--expect", default=None)
a = ap.parse_args()
T = os.environ.get("SLURM_CPUS_ON_NODE", "2")
os.makedirs(a.tmp, exist_ok=True); os.makedirs(f"{a.tmp}/duck", exist_ok=True)
dbf = f"{a.tmp}/rescue3.duckdb"
if os.path.exists(dbf):
    os.remove(dbf)
con = duckdb.connect(dbf)
con.execute(f"SET threads={T}; SET temp_directory='{a.tmp}/duck'; SET preserve_insertion_order=false")
t0 = time.time()
def log(*x): print(f"[{time.time() - t0:7.1f}s]", *x, flush=True)
def jd(o):
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, np.floating): return float(o)
    if isinstance(o, np.ndarray): return o.tolist()
    return str(o)
COVS = 0.8      # predeclared rescue.hit_rule
res = {}

M = pd.read_parquet(a.gid2uid).sort_values("gid").reset_index(drop=True)
B = pd.read_parquet(a.base).sort_values("gid").reset_index(drop=True)
NG = len(M); assert np.array_equal(M.gid.values, np.arange(NG)) and np.array_equal(B.gid.values, np.arange(NG))
base_tok = B.token.values.astype(np.int64); uid_of = M.uid.values.astype(np.int64)
assert (base_tok >= 0).all()
U = pd.read_parquet(a.uniq).sort_values("uid"); NU = len(U)
assert np.array_equal(U.uid.values, np.arange(NU)) and uid_of.max() == NU - 1

def tok_sizes(tok):
    v, inv, c = np.unique(tok, return_inverse=True, return_counts=True)
    return c[inv]

def uid_plurality_token(tok):
    """per uid: token held by most of its genes (ties: lower token id)."""
    df = pd.DataFrame({"uid": uid_of, "tok": tok}).groupby(["uid", "tok"]).size().reset_index(name="n")
    df = df.sort_values(["uid", "n", "tok"], ascending=[True, False, True]).drop_duplicates("uid")
    out = np.full(NU, -1, np.int64); out[df.uid.values] = df.tok.values
    return out

base_size = tok_sizes(base_tok)
tok_u = uid_plurality_token(base_tok)
assert (tok_u >= 0).all()
_v, _c = np.unique(base_tok, return_counts=True); size_of_token = dict(zip(_v.tolist(), _c.tolist()))
tsize_u = np.array([size_of_token[t] for t in tok_u], np.int64)
res["uids_with_genes_in_more_than_one_token"] = int((pd.DataFrame({"u": uid_of, "t": base_tok}).drop_duplicates().groupby("u").size() > 1).sum())

con.register("tku", pd.DataFrame({"uid": np.arange(NU), "tok_t": tok_u, "size_t": tsize_u}))
q_genes = pd.DataFrame({"gid": np.flatnonzero(base_size < 3)}); q_genes["uid"] = uid_of[q_genes.gid.values]
q_genes["tok"] = base_tok[q_genes.gid.values]; q_genes["qsize"] = base_size[q_genes.gid.values]
con.register("qg", q_genes)
res["hit_rows_total"], res["hit_queries_total"] = con.execute(f"SELECT count(*), count(DISTINCT q) FROM read_parquet('{a.hits}')").fetchone()
con.execute(f"""CREATE OR REPLACE TABLE rh AS
  SELECT h.q::BIGINT q, h.t::BIGINT t, h.evalue, h.bits, h.fident, h.qstart::INT qs, h.qend::INT qe, h.qlen::INT ql,
         h.tstart::INT ts, h.tend::INT te, h.tlen::INT tl, k.tok_t, k.size_t,
         CASE WHEN h.qlen <= h.tlen THEN (h.qend::INT - h.qstart::INT + 1) / h.qlen::DOUBLE
              ELSE (h.tend::INT - h.tstart::INT + 1) / h.tlen::DOUBLE END cov_short
  FROM read_parquet('{a.hits}') h JOIN tku k ON k.uid = h.t
  WHERE h.evalue <= 1e-5 AND k.size_t >= 2 AND h.q IN (SELECT DISTINCT uid FROM qg)""")
con.execute(f"DELETE FROM rh WHERE cov_short < {COVS}")
res["qualifying_uid_hits"] = con.execute("SELECT count(*), count(DISTINCT q) FROM rh").fetchone()
log("rescue qualifying uid hits", res["qualifying_uid_hits"])
best = con.execute("""SELECT * FROM (
    SELECT qg.gid, qg.qsize, rh.*, row_number() OVER (PARTITION BY qg.gid ORDER BY rh.bits DESC, rh.evalue, rh.size_t DESC, rh.t) rk
    FROM qg JOIN rh ON rh.q = qg.uid WHERE rh.tok_t <> qg.tok) WHERE rk = 1""").df()
log("rescue best hits", len(best))

def rescue_tokens(kmax, tag):
    """genes with base token size < kmax move to the token of their best hit; pointer following for moved targets."""
    b = best[best.qsize < kmax].copy()
    rep = pd.DataFrame({"gid": np.arange(NG), "uid": uid_of, "tok": base_tok})
    rep = rep[rep.uid.isin(b.t.unique())]
    rep = rep[rep.tok.values == tok_u[rep.uid.values]].drop_duplicates("uid")
    rep_of = dict(zip(rep.uid.values, rep.gid.values))
    nxt = {int(g): int(rep_of[t]) for g, t in zip(b.gid.values, b.t.values)}
    final, n_chain, n_cycle = resolve_moves(nxt, base_tok, size_of_token)
    tok = base_tok.copy()
    if final:
        gids = np.fromiter(final.keys(), np.int64); tok[gids] = np.fromiter(final.values(), np.int64)
    res[f"{tag}_query_genes"] = int((base_size < kmax).sum())
    res[f"{tag}_genes_with_qualifying_hit"] = int(len(b))
    res[f"{tag}_chains_followed"] = n_chain; res[f"{tag}_cycles"] = n_cycle
    if len(b):
        res[f"{tag}_query_shorter_frac"] = float((b.ql <= b.tl).mean())
        res[f"{tag}_fident_q10_q50_q90"] = b.fident.quantile([0.1, 0.5, 0.9]).tolist()
        res[f"{tag}_len_ratio_short_over_long_q10_q50_q90"] = (np.minimum(b.ql, b.tl) / np.maximum(b.ql, b.tl)).quantile([0.1, 0.5, 0.9]).tolist()
        res[f"{tag}_target_token_size_q50_q90_max"] = [float(b.size_t.median()), float(b.size_t.quantile(0.9)), int(b.size_t.max())]
    return tok, b

tok, b3 = rescue_tokens(3, "rescue3")
log("rescue3 built")
ch = tok != base_tok
sz = tok_sizes(tok)
emptied = np.setdiff1d(np.unique(base_tok), np.unique(tok))
out = a.out
arm = os.path.basename(out)
meta = {"arm": arm, "repair": "rescue3", "rows": NG, "n_tokens": int(len(np.unique(tok))), "n_assigned": NG,
        "genes_changed": int(ch.sum()), "genes_changed_frac": float(ch.mean()),
        "base_tokens_emptied": int(len(emptied)), "receiving_tokens": int(len(np.unique(tok[ch]))),
        "singletons": int((sz == 1).sum()), "largest_before": int(base_size.max()), "largest_after": int(sz.max()),
        "token_size_q50_q90_q99": np.quantile(np.bincount(np.unique(tok, return_inverse=True)[1]), [0.5, 0.9, 0.99]).tolist(),
        "max_growth_of_a_token": int((pd.Series(tok[ch]).value_counts().max()) if ch.any() else 0),
        "base": a.base, "hits": a.hits, "gid2uid": a.gid2uid, "uniq": a.uniq,
        "rule": "repair/code/predeclared.json rescue (variant rescue3), copied from repair/code/build_arms.py",
        "code_sha256": hashlib.sha256(open(__file__, "rb").read()).hexdigest(),
        "repair_lib_sha256": hashlib.sha256(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "repair_lib.py"), "rb").read()).hexdigest(),
        "date": time.strftime("%Y-%m-%d"), "seconds": round(time.time() - t0, 1)}
meta.update(res)
if a.expect:
    ex = pd.read_parquet(a.expect).sort_values("gid")
    meta["expect"] = a.expect
    meta["expect_identical"] = bool(np.array_equal(ex.gid.values, np.arange(NG)) and np.array_equal(ex.token.values.astype(np.int64), tok))
    meta["expect_rows_differ"] = int((ex.token.values.astype(np.int64) != tok).sum())
    log("replay identical to expected arm:", meta["expect_identical"], "rows differing", meta["expect_rows_differ"])
tmp = f"{out}.parquet.tmp"
pd.DataFrame({"gid": np.arange(NG, dtype=np.int64), "token": tok.astype(np.int64)}).to_parquet(tmp, index=False)
os.replace(tmp, f"{out}.parquet")
meta["parquet_sha256"] = hashlib.sha256(open(f"{out}.parquet", "rb").read()).hexdigest()
pd.DataFrame({"gid": np.flatnonzero(ch), "base_token": base_tok[ch], "token": tok[ch]}).to_parquet(f"{a.tmp}/{arm}.changed.parquet", index=False)
json.dump(meta, open(f"{out}.json", "w"), indent=1, default=jd)
con.close(); os.remove(dbf)
print(json.dumps(meta, indent=1, default=jd))
if a.expect and not meta["expect_identical"]:
    sys.exit("replay differs from the expected arm")
