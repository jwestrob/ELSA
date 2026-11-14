#!/usr/bin/env python3
"""Rescue error, part 3: stricter rescue3 variants a-d applied genome-wide on the existing hits; token arms.

usage: build_variants.py <repair dir> <subgenic dir> <base token parquet> <token out dir>
Variants (predeclared amendment_1_rescue_error.variants_on_rescue3):
  a: moving gene is the shorter protein; b: a + truncated (partial flag, else contig end);
  c: a + anchored within 10 residues of both termini on every intact side ('11' cannot pass); d: b and c.
The unfiltered rule is replayed first and must reproduce the rescue3 arm exactly.
Writes <out>/mmseqs__seq__e3c50s75_rescue3{a,b,c,d}__dpann.{parquet,json}, results/changed_<arm>.parquet,
results/variants.json.
"""
import hashlib, json, os, sys, time
import numpy as np, pandas as pd, duckdb
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from repair_lib import resolve_moves

R, SG, BASE, OUTD = sys.argv[1:5]
t0 = time.time()
con = duckdb.connect(); con.execute(f"SET threads={os.environ.get('SLURM_CPUS_ON_NODE', '2')}; SET temp_directory='{R}/tmp/duck'")
G = pd.read_parquet(f"{R}/genes.parquet")
NG = len(G); base_tok = G.token.values.astype(np.int64); uid_of = G.uid.values.astype(np.int64)
PF = pd.read_parquet(f"{R}/partial.parquet"); assert (PF.gid.values == np.arange(NG)).all()
U = pd.read_parquet(f"{SG}/uniq.parquet", columns=["uid"]); NU = len(U)
_v, _inv, _c = np.unique(base_tok, return_inverse=True, return_counts=True)
base_size = _c[_inv]; size_of = dict(zip(_v.tolist(), _c.tolist()))
gt = pd.DataFrame({"uid": uid_of, "tok": base_tok}).groupby(["uid", "tok"]).size().reset_index(name="n")
gt = gt.sort_values(["uid", "n", "tok"], ascending=[True, False, True]).drop_duplicates("uid")
tok_u = np.full(NU, -1, np.int64); tok_u[gt.uid.values] = gt.tok.values
tsize_u = np.array([size_of[t] for t in tok_u], np.int64)
con.register("tku", pd.DataFrame({"uid": np.arange(NU), "tok_t": tok_u, "size_t": tsize_u}))
qg = pd.DataFrame({"gid": np.flatnonzero(base_size < 3)}); qg["uid"] = uid_of[qg.gid.values]
qg["tok"] = base_tok[qg.gid.values]; qg["qsize"] = base_size[qg.gid.values]
qg["trunc"] = PF.trunc.values[qg.gid.values]; qg["intact_n"] = PF.intact_n.values[qg.gid.values]; qg["intact_c"] = PF.intact_c.values[qg.gid.values]
con.register("qg", qg)
H = con.execute(f"""SELECT qg.gid, qg.qsize, qg.trunc, qg.intact_n, qg.intact_c, h.t::BIGINT t, h.evalue, h.bits, h.fident,
     h.qstart::INT qs, h.qend::INT qe, h.qlen::INT ql, h.tstart::INT ts, h.tend::INT te, h.tlen::INT tl, k.tok_t, k.size_t
  FROM read_parquet('{R}/hits/Q_*.parquet') h JOIN tku k ON k.uid = h.t JOIN qg ON qg.uid = h.q
  WHERE h.evalue <= 1e-5 AND k.size_t >= 2 AND k.tok_t <> qg.tok
    AND CASE WHEN h.qlen <= h.tlen THEN (h.qend::INT - h.qstart::INT + 1) >= 0.8 * h.qlen::INT
             ELSE (h.tend::INT - h.tstart::INT + 1) >= 0.8 * h.tlen::INT END""").df()
print(f"[{time.time() - t0:.0f}s] qualifying gene-hit rows", len(H), flush=True)
a_ok = (H.ql <= H.tl).values
anc_n = ((H.qs <= 11) & (H.ts <= 11)).values
anc_c = (((H.ql - H.qe) <= 10) & ((H.tl - H.te) <= 10)).values
inn, inc = H.intact_n.values, H.intact_c.values
c_ok = a_ok & (~inn | anc_n) & (~inc | anc_c) & (inn | inc)
b_ok = a_ok & H.trunc.values
masks = {"replay": np.ones(len(H), bool), "a": a_ok, "b": b_ok, "c": c_ok, "d": b_ok & c_ok}

rep_cache = {}
def rep_gene(uids):
    need = [u for u in uids if u not in rep_cache]
    if need:
        df = pd.DataFrame({"gid": np.arange(NG), "uid": uid_of, "tok": base_tok})
        df = df[df.uid.isin(need)]
        df = df[df.tok.values == tok_u[df.uid.values]].drop_duplicates("uid")
        rep_cache.update(dict(zip(df.uid.values.tolist(), df.gid.values.tolist())))
    return rep_cache

res = {"predeclared": "amendment_1_rescue_error", "query_genes": int(len(qg)), "qualifying_rows": int(len(H))}
r3 = pd.read_parquet(f"{OUTD}/mmseqs__seq__e3c50s75_rescue3__dpann.parquet").token.values
code_hash = hashlib.sha256(open(__file__, "rb").read()).hexdigest()
for v, m in masks.items():
    b = H[m].sort_values(["gid", "bits", "evalue", "size_t", "t"], ascending=[True, False, True, False, True]).drop_duplicates("gid")
    rc = rep_gene(b.t.unique().tolist())
    nxt = {int(g): int(rc[int(t)]) for g, t in zip(b.gid.values, b.t.values)}
    final, n_chain, n_cycle = resolve_moves(nxt, base_tok, size_of)
    tok = base_tok.copy()
    if final:
        tok[np.fromiter(final.keys(), np.int64)] = np.fromiter(final.values(), np.int64)
    ch = tok != base_tok
    if v == "replay":
        res["replay_identical_to_rescue3"] = bool(np.array_equal(tok, r3))
        assert res["replay_identical_to_rescue3"], "replay differs from the rescue3 arm"
        continue
    arm = f"mmseqs__seq__e3c50s75_rescue3{v}__dpann"
    sz = np.unique(tok, return_counts=True)[1]
    a = {"arm": arm, "variant": v, "rows": NG, "n_tokens": int(len(sz)), "n_assigned": NG,
         "genes_with_qualifying_hit": int(len(b)), "genes_changed": int(ch.sum()),
         "changed_truncated": int((ch & PF.trunc.values).sum()), "changed_from_partial_flag": int((ch & (PF.flag_source.values == "partial")).sum()),
         "chains": n_chain, "cycles": n_cycle, "largest_after": int(sz.max()), "singletons": int((sz == 1).sum()),
         "best_hit_fident_q10_q50_q90": b.fident.quantile([0.1, 0.5, 0.9]).tolist(),
         "best_hit_len_ratio_q10_q50_q90": (b.ql / b.tl).quantile([0.1, 0.5, 0.9]).tolist(),
         "base": BASE, "code_sha256": code_hash, "date": time.strftime("%Y-%m-%d")}
    tmp = f"{OUTD}/{arm}.parquet.tmp"
    pd.DataFrame({"gid": np.arange(NG, dtype=np.int64), "token": tok.astype(np.int64)}).to_parquet(tmp, index=False)
    os.replace(tmp, f"{OUTD}/{arm}.parquet")
    a["parquet_sha256"] = hashlib.sha256(open(f"{OUTD}/{arm}.parquet", "rb").read()).hexdigest()
    json.dump(a, open(f"{OUTD}/{arm}.json", "w"), indent=1)
    pd.DataFrame({"gid": np.flatnonzero(ch), "base_token": base_tok[ch], "token": tok[ch], "reason": "rescue"}).to_parquet(
        f"{R}/results/changed_{arm}.parquet", index=False)
    res[arm] = a
    print(f"[{time.time() - t0:.0f}s]", arm, {k: a[k] for k in ("genes_changed", "changed_truncated", "n_tokens", "largest_after")}, flush=True)
res["seconds"] = round(time.time() - t0, 1)
json.dump(res, open(f"{R}/results/variants.json", "w"), indent=1)
