#!/usr/bin/env python3
"""Apply the predeclared repairs genome-wide and write the token arms (workstream repair).

usage: build_arms.py <repair dir> <subgenic dir> <package dir> <base token parquet> <token out dir>
Reads genes.parquet, pairs.parquet, hits/Q_*.parquet, split/cand.parquet, split/stage2.parquet.
Writes <token out dir>/<arm>.parquet + .json, results/changed_<arm>.parquet (gid, base_token, token, reason),
results/arms.json (all counts, validation, inspection), results/inspect_rescue1.tsv, results/split_pairs.parquet.
"""
import hashlib, json, os, sys, time
import numpy as np, pandas as pd, duckdb
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from repair_lib import resolve_moves

R, SG, P, BASE, OUTD = sys.argv[1:6]
PD = json.load(open(f"{R}/code/predeclared.json"))
T = os.environ.get("SLURM_CPUS_ON_NODE", "2")
os.makedirs(f"{R}/results", exist_ok=True); os.makedirs(f"{R}/tmp/duck", exist_ok=True)
con = duckdb.connect(f"{R}/tmp/arms.duckdb")
con.execute(f"SET threads={T}; SET temp_directory='{R}/tmp/duck'; SET preserve_insertion_order=false")
t0 = time.time()
def log(*x): print(f"[{time.time() - t0:7.1f}s]", *x, flush=True)
def jd(o):
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, np.floating): return float(o)
    if isinstance(o, np.ndarray): return o.tolist()
    return str(o)
COVS, YC, RMIN, OV, XGAP, JCOV = 0.8, 0.7, 80, 20, 10, 0.8      # predeclared rescue.hit_rule, split.*
res = {"predeclared_written_utc": PD["written_utc"]}

G = pd.read_parquet(f"{R}/genes.parquet")
NG = len(G); assert np.array_equal(G.gid.values, np.arange(NG))
base_tok = G.token.values.astype(np.int64); uid_of = G.uid.values.astype(np.int64)
genome_of = G.genome_code.values.astype(np.int64)
U = pd.read_parquet(f"{SG}/uniq.parquet").sort_values("uid"); NU = len(U)
ulen = U.length.values

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

# ------------------------------------------------------------------ rescue
con.register("tku", pd.DataFrame({"uid": np.arange(NU), "tok_t": tok_u, "size_t": tsize_u}))
q_genes = pd.DataFrame({"gid": np.flatnonzero(base_size < 3)}); q_genes["uid"] = uid_of[q_genes.gid.values]
q_genes["tok"] = base_tok[q_genes.gid.values]; q_genes["qsize"] = base_size[q_genes.gid.values]
con.register("qg", q_genes)
con.execute(f"""CREATE OR REPLACE TABLE rh AS
  SELECT h.q::BIGINT q, h.t::BIGINT t, h.evalue, h.bits, h.fident, h.qstart::INT qs, h.qend::INT qe, h.qlen::INT ql,
         h.tstart::INT ts, h.tend::INT te, h.tlen::INT tl, k.tok_t, k.size_t,
         CASE WHEN h.qlen <= h.tlen THEN (h.qend::INT - h.qstart::INT + 1) / h.qlen::DOUBLE
              ELSE (h.tend::INT - h.tstart::INT + 1) / h.tlen::DOUBLE END cov_short
  FROM read_parquet('{R}/hits/Q_*.parquet') h JOIN tku k ON k.uid = h.t
  WHERE h.evalue <= 1e-5 AND k.size_t >= 2 AND h.q IN (SELECT DISTINCT uid FROM qg)""")
con.execute(f"DELETE FROM rh WHERE cov_short < {COVS}")
log("rescue qualifying uid hits", con.execute("SELECT count(*), count(DISTINCT q) FROM rh").fetchone())
best = con.execute("""SELECT * FROM (
    SELECT qg.gid, qg.qsize, rh.*, row_number() OVER (PARTITION BY qg.gid ORDER BY rh.bits DESC, rh.evalue, rh.size_t DESC, rh.t) rk
    FROM qg JOIN rh ON rh.q = qg.uid WHERE rh.tok_t <> qg.tok) WHERE rk = 1""").df()
log("rescue best hits", len(best))

def rescue_tokens(kmax, tag):
    """genes with base token size < kmax move to the token of their best hit; pointer following for moved targets."""
    b = best[best.qsize < kmax].copy()
    tgt_tok = dict(zip(b.gid.values, b.tok_t.values))
    # representative gene of each target uid inside its plurality token (lowest gid)
    rep = pd.DataFrame({"gid": np.arange(NG), "uid": uid_of, "tok": base_tok})
    rep = rep[rep.uid.isin(b.t.unique())]
    rep = rep[rep.tok.values == tok_u[rep.uid.values]].drop_duplicates("uid")
    rep_of = dict(zip(rep.uid.values, rep.gid.values))
    nxt = {int(g): int(rep_of[t]) for g, t in zip(b.gid.values, b.t.values)}
    final, n_chain, n_cycle = resolve_moves(nxt, base_tok, size_of_token)
    tok = base_tok.copy()
    gids = np.fromiter(final.keys(), np.int64); tok[gids] = np.fromiter(final.values(), np.int64)
    res[f"{tag}_query_genes"] = int((base_size < kmax).sum())
    res[f"{tag}_genes_with_qualifying_hit"] = int(len(b))
    res[f"{tag}_chains_followed"] = n_chain; res[f"{tag}_cycles"] = n_cycle
    res[f"{tag}_query_shorter_frac"] = float((b.ql <= b.tl).mean())
    res[f"{tag}_fident_q10_q50_q90"] = b.fident.quantile([0.1, 0.5, 0.9]).tolist()
    res[f"{tag}_len_ratio_short_over_long_q10_q50_q90"] = (np.minimum(b.ql, b.tl) / np.maximum(b.ql, b.tl)).quantile([0.1, 0.5, 0.9]).tolist()
    res[f"{tag}_target_token_size_q50_q90_max"] = [float(b.size_t.median()), float(b.size_t.quantile(0.9)), int(b.size_t.max())]
    return tok, b

tok_r1, b1 = rescue_tokens(2, "rescue1")
tok_r3, b3 = rescue_tokens(3, "rescue3")
log("rescue arms built")

# 20 random rescue1 genes for inspection (seed 17)
rng = np.random.default_rng(17)
ins = b1.iloc[np.sort(rng.choice(len(b1), 20, replace=False))].copy()
ins["qcov"] = (ins.qe - ins.qs + 1) / ins.ql; ins["tcov"] = (ins.te - ins.ts + 1) / ins.tl
ins["shorter"] = np.where(ins.ql <= ins.tl, "query", "target")
ins = ins[["gid", "q", "t", "ql", "tl", "shorter", "qcov", "tcov", "cov_short", "fident", "evalue", "bits", "tok_t", "size_t"]]
ins.to_csv(f"{R}/results/inspect_rescue1.tsv", sep="\t", index=False)

# ------------------------------------------------------------------ split
cand = con.execute(f"SELECT * FROM '{R}/split/cand.parquet'").df()
s2 = con.execute(f"""SELECT q::BIGINT ul, t::BIGINT x, evalue l_e, bits l_bits, fident l_fid, qstart::INT l_qs, qend::INT l_qe,
     qlen::INT l_ql, tstart::INT l_ts, tend::INT l_te, tlen::INT l_tl FROM '{R}/split/stage2.parquet'
     WHERE evalue <= 1e-5""").df()
log("split candidates", len(cand), "stage-2 alignments E<=1e-5", len(s2))
m = cand.merge(s2, on=["ul", "x"], how="inner")
m = m[((m.l_qe - m.l_qs + 1) >= YC * m.l_ql) & ((m.l_te - m.l_ts + 1) >= RMIN) & ((m.l_te - m.l_ts + 1) <= m.xlen - (RMIN - OV))]
sN = np.where(m.s_is_n, m.s_ts, m.l_ts); eN = np.where(m.s_is_n, m.s_te, m.l_te)
sC = np.where(m.s_is_n, m.l_ts, m.s_ts); eC = np.where(m.s_is_n, m.l_te, m.s_te)
ov = np.minimum(eN, eC) - np.maximum(sN, sC) + 1
m["overlap"] = ov; m["xgap"] = sC - eN - 1
m["jcov"] = ((eN - sN + 1) + (eC - sC + 1) - np.maximum(ov, 0)) / m.xlen
m["ok"] = (sN < sC) & (ov <= OV) & (m.xgap <= XGAP) & (m.jcov >= JCOV)
res["split_candidates_with_long_piece_hit"] = int(len(m))
res["split_candidate_rejections"] = {"order": int((sN >= sC).sum()), "overlap": int((ov > OV).sum()),
                                     "xgap": int((m.xgap > XGAP).sum()), "jcov": int((m.jcov < JCOV).sum())}
q = m[m.ok].copy(); q["bits_sum"] = q.s_bits + q.l_bits
q = q.sort_values(["ga", "gb", "jcov", "bits_sum", "x"], ascending=[True, True, False, False, True]).drop_duplicates(["ga", "gb"])
q.to_parquet(f"{R}/results/split_pairs.parquet")
res["split_qualifying_pairs"] = int(len(q))
res["split_pairs_overlap_nt_frac"] = float((q.gap_nt < 0).mean()) if "gap_nt" in q else None
log("split qualifying pairs", len(q))
# gene -> best pair
gp = pd.concat([q[["ga", "gb", "x", "jcov", "bits_sum"]].rename(columns={"ga": "gid"}),
                q[["ga", "gb", "x", "jcov", "bits_sum"]].rename(columns={"gb": "gid"})])
gp = gp.sort_values(["gid", "jcov", "bits_sum", "x"], ascending=[True, False, False, True]).drop_duplicates("gid")
res["split_genes_assigned"] = int(len(gp))
res["split_genes_in_two_pairs"] = int(len(pd.concat([q.ga, q.gb])) - len(gp))

def apply_split(tok_in):
    tu = uid_plurality_token(tok_in)
    tok = tok_in.copy(); tok[gp.gid.values] = tu[gp.x.values]
    return tok

tok_sp = apply_split(base_tok)
tok_r1sp = apply_split(tok_r1)
tok_r3sp = apply_split(tok_r3)

# ------------------------------------------------------------------ write arms
arms = {"mmseqs__seq__e3c50s75_rescue1__dpann": (tok_r1, "rescue1"), "mmseqs__seq__e3c50s75_rescue3__dpann": (tok_r3, "rescue3"),
        "mmseqs__seq__e3c50s75_split__dpann": (tok_sp, "split"), "mmseqs__seq__e3c50s75_rescue1split__dpann": (tok_r1sp, "rescue1+split"),
        "mmseqs__seq__e3c50s75_rescue3split__dpann": (tok_r3sp, "rescue3+split")}
code_hash = hashlib.sha256(open(__file__, "rb").read()).hexdigest()
res["base"] = {"tokens": int(len(np.unique(base_tok))), "largest": int(base_size.max()), "singletons": int((base_size == 1).sum())}
idx = []
for arm, (tok, kind) in arms.items():
    ch = tok != base_tok
    sz = tok_sizes(tok)
    emptied = np.setdiff1d(np.unique(base_tok), np.unique(tok))
    a = {"arm": arm, "repair": kind, "rows": NG, "n_tokens": int(len(np.unique(tok))), "n_assigned": NG,
         "genes_changed": int(ch.sum()), "genes_changed_frac": float(ch.mean()),
         "base_tokens_emptied": int(len(emptied)), "receiving_tokens": int(len(np.unique(tok[ch]))),
         "singletons": int((sz == 1).sum()), "largest_before": int(base_size.max()), "largest_after": int(sz.max()),
         "token_size_q50_q90_q99": np.quantile(np.bincount(np.unique(tok, return_inverse=True)[1]), [0.5, 0.9, 0.99]).tolist(),
         "max_growth_of_a_token": int((pd.Series(tok[ch]).value_counts().max()) if ch.any() else 0),
         "base": BASE, "predeclared": f"{R}/code/predeclared.json", "code_sha256": code_hash,
         "date": time.strftime("%Y-%m-%d"), "seconds": round(time.time() - t0, 1)}
    tmp = f"{OUTD}/{arm}.parquet.tmp"
    pd.DataFrame({"gid": np.arange(NG, dtype=np.int64), "token": tok.astype(np.int64)}).to_parquet(tmp, index=False)
    os.replace(tmp, f"{OUTD}/{arm}.parquet")
    a["parquet_sha256"] = hashlib.sha256(open(f"{OUTD}/{arm}.parquet", "rb").read()).hexdigest()
    json.dump(a, open(f"{OUTD}/{arm}.json", "w"), indent=1, default=jd)
    reason = np.full(NG, "", object)
    if "rescue1" in kind: reason[tok_r1 != base_tok] = "rescue"
    if "rescue3" in kind: reason[tok_r3 != base_tok] = "rescue"
    if "split" in kind: reason[gp.gid.values] = np.where(reason[gp.gid.values] == "rescue", "split_over_rescue", "split")
    pd.DataFrame({"gid": np.flatnonzero(ch), "base_token": base_tok[ch], "token": tok[ch], "reason": reason[ch]}).to_parquet(
        f"{R}/results/changed_{arm}.parquet", index=False)
    res[arm] = a
    idx.append(f"{arm}\t{a['n_tokens']}\t{NG}\t{a['singletons']}\t{a['largest_after']}\t{a['genes_changed']}\t{a['date']}")
    log(arm, {k: a[k] for k in ("genes_changed", "n_tokens", "base_tokens_emptied", "largest_after")})
open(f"{R}/results/arms_index.tsv", "w").write("arm\tn_tokens\tn_assigned\tsingletons\tmax_size\tgenes_changed\tdate\n" + "\n".join(idx) + "\n")

# ------------------------------------------------------------------ validation against the sizing study's search S (max-seqs 8000)
SGn = pd.read_parquet(f"{SG}/lists/S_genes.parquet", columns=["gid", "in_sample"])
in_sample = np.zeros(NG, bool); in_sample[SGn.gid.values[SGn.in_sample.values]] = True
qs = q_genes[(q_genes.qsize == 1) & in_sample[q_genes.gid.values]].copy()
con.register("qs1", qs)
bS = con.execute(f"""SELECT * FROM (
   SELECT qs1.gid, k.tok_t, h.bits, row_number() OVER (PARTITION BY qs1.gid ORDER BY h.bits DESC, h.evalue, k.size_t DESC, h.t) rk
   FROM qs1 JOIN read_parquet('{SG}/hits/S_*.parquet') h ON h.q = qs1.uid JOIN tku k ON k.uid = h.t
   WHERE h.q <> h.t AND h.evalue <= 1e-5 AND k.size_t >= 2 AND k.tok_t <> qs1.tok
     AND CASE WHEN h.qlen <= h.tlen THEN (h.qend::INT - h.qstart::INT + 1) >= {COVS} * h.qlen::INT
              ELSE (h.tend::INT - h.tstart::INT + 1) >= {COVS} * h.tlen::INT END) WHERE rk = 1""").df()
new = dict(zip(b1.gid.values, b1.tok_t.values)); old = dict(zip(bS.gid.values, bS.tok_t.values))
gs = set(qs.gid.values)
both = [g for g in gs if g in new and g in old]
res["validation_rescue1_sample_query_genes"] = len(gs)
res["validation_rescue1_moved_new_search"] = int(sum(g in new for g in gs))
res["validation_rescue1_moved_S_search"] = int(len(old))
res["validation_rescue1_moved_both_same_token"] = int(sum(new[g] == old[g] for g in both))
res["validation_rescue1_moved_both_different_token"] = int(sum(new[g] != old[g] for g in both))
res["validation_rescue1_moved_only_new"] = int(sum(g in new and g not in old for g in gs))
res["validation_rescue1_moved_only_S"] = int(sum(g in old and g not in new for g in gs))
moved_union = (set(new) & gs) | (set(old) & gs)
res["validation_rescue1_disagreement_frac_of_moved_union"] = float(
    (res["validation_rescue1_moved_both_different_token"] + res["validation_rescue1_moved_only_new"] + res["validation_rescue1_moved_only_S"])
    / max(1, len(moved_union)))
# split: sizing-study broken-call-like loci with joint coverage >= 0.8 in sample genomes
L = pd.read_parquet(f"{SG}/results/split_loci.parquet")
ref = L[L.in_sample & L.broken_like & (L.jcov >= JCOV)][["ya", "yb"]].drop_duplicates()
mine = set(zip(q.ga.values, q.gb.values))
hit = np.array([(a, b) in mine for a, b in zip(ref.ya.values, ref.yb.values)])
res["validation_split_reference_pairs"] = int(len(ref)); res["validation_split_recovered"] = int(hit.sum())
res["validation_split_recall"] = float(hit.mean()) if len(ref) else None
qin = q[in_sample[q.ga.values]]
refset = set(zip(ref.ya.values, ref.yb.values))
res["validation_split_sample_pairs_new"] = int(len(qin))
res["validation_split_sample_pairs_new_not_in_reference"] = int(sum((a, b) not in refset for a, b in zip(qin.ga.values, qin.gb.values)))
# how often the rescued gene is a sizing-study fragment (sample genomes)
ff = np.load(f"{SG}/results/fragment_flags.npz")["frag"]
mv1 = np.zeros(NG, bool); mv1[b1.gid.values] = True
res["rescue1_moved_sample_genes"] = int((mv1 & in_sample).sum())
res["rescue1_moved_sample_genes_that_are_fragments"] = int((mv1 & in_sample & ff).sum())
res["rescue1_sample_fragments_singleton"] = int((ff & in_sample & (base_size == 1)).sum())
res["rescue1_sample_fragments_singleton_moved"] = int((ff & in_sample & (base_size == 1) & mv1).sum())
res["seconds"] = round(time.time() - t0, 1)
json.dump(res, open(f"{R}/results/arms.json", "w"), indent=1, default=jd)
print(json.dumps(res, indent=1, default=jd))
