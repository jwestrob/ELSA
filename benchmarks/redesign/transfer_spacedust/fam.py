#!/usr/bin/env python
"""Family tokens for the Spacedust collection with the DPANN recipe (workstream transfer_spacedust).

Base families = the stored MMseqs2 clustering of all 4,257,531 genes (-s 7.5 -e 1e-3 -c 0.5 --cov-mode 0,
$RD/tokens/mmseqs__seq__e3c50s75.parquet). The rescue3 step is the repair workstream's rule
(repair/build_arms.py rescue_tokens(3), repair_lib.resolve_moves), copied:
  queries  unique sequences of genes in a base token with < 3 members
  search   MMseqs2 18 prefilter -s 7.5 --max-seqs 1000; align -e 1e-5 -a; against all unique sequences
  rule     hit at E <= 1e-5 covering >= 80% of the shorter protein, target uid's plurality token has >= 2 members and
           differs from the gene's own token; best hit by bits (ties: E, larger target token, lower uid); moves applied
           together, a gene whose best hit itself moved follows it (chains; cycles take the largest base token)

usage
  fam.py uniq --out DIR                 uid table from all.faa (normalised: uppercase, one terminal '*' removed)
  fam.py rescue3 --dir DIR --threads T  hits/*.parquet -> tokens rescue3 (gid, token)
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

RD = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111")
TS = RD / "transfer_spacedust"
ALLFAA = RD / "ws_tokens/work/mmseqs/all.faa"
BASE = RD / "tokens/mmseqs__seq__e3c50s75.parquet"
N = 4257531
COVS = 0.8


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def cmd_uniq(a):
    t0 = time.time()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    gid2uid = np.full(N, -1, np.int64)
    first = {}
    lens = []
    raw_first = []
    g = -1
    seq = []

    def flush(g, seq):
        s = b"".join(seq).upper()
        if s.endswith(b"*"):
            s = s[:-1]
        h = hashlib.blake2b(s, digest_size=16).digest()
        u = first.get(h)
        if u is None:
            u = len(first)
            first[h] = u
            lens.append(len(s))
            raw_first.append(s)
        gid2uid[g] = u

    with open(ALLFAA, "rb") as fh:
        for line in fh:
            if line.startswith(b">"):
                if g >= 0:
                    flush(g, seq)
                g = int(line[1:].strip())
                seq = []
            else:
                seq.append(line.strip())
        flush(g, seq)
    assert (gid2uid >= 0).all()
    U = len(first)
    with open(out / "uniq.faa", "wb") as fh:
        for u, s in enumerate(raw_first):
            fh.write(b">%d\n%s\n" % (u, s))
    np.save(out / "gid2uid.npy", gid2uid.astype(np.int32))
    np.save(out / "uid_len.npy", np.asarray(lens, np.int32))
    cnt = np.bincount(gid2uid, minlength=U)
    np.save(out / "uid_count.npy", cnt.astype(np.int32))
    # rescue queries: uids of genes in base tokens with < 3 members
    t = pd.read_parquet(BASE)
    tok = np.full(N, -1, np.int64)
    tok[t.gid.to_numpy()] = t.token.to_numpy()
    assert (tok >= 0).all()
    size = np.bincount(tok)[tok]
    q = np.unique(gid2uid[size < 3])
    np.savetxt(out / "Q.uid", q, fmt="%d")
    info = dict(genes=N, unique_sequences=int(U), duplicate_genes=int(N - U), uid_count_max=int(cnt.max()),
                genes_in_tokens_lt3=int((size < 3).sum()), query_uids=int(len(q)),
                query_residues=int(np.asarray(lens)[q].sum()), residues_unique=int(np.sum(lens)),
                base_tokens=int(tok.max() + 1), seconds=round(time.time() - t0, 1))
    (out / "uniq.json").write_text(json.dumps(info, indent=1) + "\n")
    log("uniq", info)


def resolve_moves(nxt, base_tok, size_of_token):
    """repair_lib.resolve_moves, copied verbatim (final token of every moved gene; chains followed, cycles resolved)."""
    final, n_chain, n_cycle = {}, 0, 0
    for g0 in nxt:
        if g0 in final:
            continue
        path, seen, g = [], {}, g0
        while g in nxt and g not in final and g not in seen:
            seen[g] = len(path)
            path.append(g)
            g = nxt[g]
        if g in final:
            f = final[g]
        elif g in seen:
            cyc = path[seen[g]:]
            n_cycle += 1
            f = sorted((-size_of_token[int(base_tok[c])], int(base_tok[c])) for c in cyc)[0][1]
            for c in cyc:
                final[c] = f
            path = path[:seen[g]]
        else:
            f = int(base_tok[g])
        for c in path:
            final[c] = f
        n_chain += int(len(path) > 1)
    return final, n_chain, n_cycle


def cmd_rescue3(a):
    import duckdb
    t0 = time.time()
    D = Path(a.dir)
    con = duckdb.connect()
    con.execute(f"SET threads={a.threads}; SET preserve_insertion_order=false")
    gid2uid = np.load(D / "gid2uid.npy").astype(np.int64)
    t = pd.read_parquet(BASE)
    base_tok = np.full(N, -1, np.int64)
    base_tok[t.gid.to_numpy()] = t.token.to_numpy()
    _v, _c = np.unique(base_tok, return_counts=True)
    size_of_token = dict(zip(_v.tolist(), _c.tolist()))
    base_size = np.bincount(base_tok)[base_tok]
    NU = int(gid2uid.max()) + 1
    # uid plurality token (ties: lower token id), as repair uid_plurality_token
    df = pd.DataFrame({"uid": gid2uid, "tok": base_tok}).groupby(["uid", "tok"]).size().reset_index(name="n")
    df = df.sort_values(["uid", "n", "tok"], ascending=[True, False, True]).drop_duplicates("uid")
    tok_u = np.full(NU, -1, np.int64)
    tok_u[df.uid.values] = df.tok.values
    assert (tok_u >= 0).all()
    tsize_u = np.array([size_of_token[x] for x in tok_u], np.int64)
    multi = int((pd.DataFrame({"u": gid2uid, "t": base_tok}).drop_duplicates().groupby("u").size() > 1).sum())
    con.register("tku", pd.DataFrame({"uid": np.arange(NU), "tok_t": tok_u, "size_t": tsize_u}))
    qg = pd.DataFrame({"gid": np.flatnonzero(base_size < 3)})
    qg["uid"] = gid2uid[qg.gid.values]
    qg["tok"] = base_tok[qg.gid.values]
    qg["qsize"] = base_size[qg.gid.values]
    con.register("qg", qg)
    con.execute(f"""CREATE OR REPLACE TABLE rh AS
      SELECT h.q::BIGINT q, h.t::BIGINT t, h.evalue, h.bits, h.fident, h.qstart::INT qs, h.qend::INT qe, h.qlen::INT ql,
             h.tstart::INT ts, h.tend::INT te, h.tlen::INT tl, k.tok_t, k.size_t,
             CASE WHEN h.qlen <= h.tlen THEN (h.qend::INT - h.qstart::INT + 1) / h.qlen::DOUBLE
                  ELSE (h.tend::INT - h.tstart::INT + 1) / h.tlen::DOUBLE END cov_short
      FROM read_parquet('{D}/hits/Q_*.parquet') h JOIN tku k ON k.uid = h.t
      WHERE h.evalue <= 1e-5 AND k.size_t >= 2 AND h.q IN (SELECT DISTINCT uid FROM qg)""")
    con.execute(f"DELETE FROM rh WHERE cov_short < {COVS}")
    nq = con.execute("SELECT count(*), count(DISTINCT q) FROM rh").fetchone()
    log("rescue qualifying uid hits", nq)
    best = con.execute("""SELECT * FROM (
        SELECT qg.gid, qg.qsize, rh.*, row_number() OVER (PARTITION BY qg.gid ORDER BY rh.bits DESC, rh.evalue, rh.size_t DESC, rh.t) rk
        FROM qg JOIN rh ON rh.q = qg.uid WHERE rh.tok_t <> qg.tok) WHERE rk = 1""").df()
    log("rescue best hits", len(best))
    b = best[best.qsize < 3].copy()
    rep = pd.DataFrame({"gid": np.arange(N), "uid": gid2uid, "tok": base_tok})
    rep = rep[rep.uid.isin(b.t.unique())]
    rep = rep[rep.tok.values == tok_u[rep.uid.values]].drop_duplicates("uid")
    rep_of = dict(zip(rep.uid.values, rep.gid.values))
    nxt = {int(g): int(rep_of[u]) for g, u in zip(b.gid.values, b.t.values)}
    final, n_chain, n_cycle = resolve_moves(nxt, base_tok, size_of_token)
    tok = base_tok.copy()
    gids = np.fromiter(final.keys(), np.int64)
    tok[gids] = np.fromiter(final.values(), np.int64)
    # dense relabel
    _, tok_d = np.unique(tok, return_inverse=True)
    out = TS / "tok"
    out.mkdir(parents=True, exist_ok=True)
    arm = "mmseqs__seq__e3c50s75_rescue3__spacedust"
    pd.DataFrame({"gid": np.arange(N), "token": tok_d.astype(np.int64)}).to_parquet(out / f"{arm}.parquet", index=False)
    ch = tok != base_tok
    pd.DataFrame({"gid": np.flatnonzero(ch), "base_token": base_tok[ch], "token": tok[ch]}).to_parquet(
        out / f"changed_{arm}.parquet", index=False)
    sz = np.bincount(tok_d)
    res = dict(arm=arm, rows=N, base_tokens=int(len(_v)), tokens=int(len(sz)), singletons=int((sz == 1).sum()),
               doubletons=int((sz == 2).sum()), largest=int(sz.max()), genes_changed=int(ch.sum()),
               genes_changed_frac=float(ch.mean()), base_tokens_emptied=int(len(_v) - len(sz)),
               receiving_tokens=int(len(np.unique(tok[ch]))), query_genes=int((base_size < 3).sum()),
               genes_with_qualifying_hit=int(len(b)), chains_followed=n_chain, cycles=n_cycle,
               uids_with_genes_in_more_than_one_token=multi, qualifying_uid_hits=int(nq[0]),
               fident_q10_q50_q90=b.fident.quantile([.1, .5, .9]).tolist() if len(b) else [],
               target_token_size_q50_q90_max=[float(b.size_t.median()), float(b.size_t.quantile(.9)), int(b.size_t.max())] if len(b) else [],
               seconds=round(time.time() - t0, 1),
               recipe="repair/build_arms.py rescue_tokens(3) + repair_lib.resolve_moves, copied; base " + str(BASE))
    (out / f"{arm}.json").write_text(json.dumps(res, indent=1) + "\n")
    log("rescue3", res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["uniq", "rescue3"])
    ap.add_argument("--out", default=str(TS / "fam"))
    ap.add_argument("--dir", default=str(TS / "fam"))
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    {"uniq": cmd_uniq, "rescue3": cmd_rescue3}[a.cmd](a)


if __name__ == "__main__":
    main()
