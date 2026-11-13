#!/usr/bin/env python3
"""Unique sequences and rescue queries for the rescue3 step (workstream slice_lut; same layout as subgenic/prep_unique.py).

usage: 06a_unique.py PKG_DIR BASE_TOKEN_PARQUET WORK_DIR
Writes WORK_DIR/uniq.faa (header = uid, 0-based in order of first gid), WORK_DIR/gid2uid.parquet (gid, uid),
       WORK_DIR/uniq.parquet (uid, length, rep_gid, n_genes), WORK_DIR/Q.uid (uids of genes whose base token has < 3 genes),
       WORK_DIR/unique.json
"""
import json, os, sys, time
import numpy as np, pyarrow as pa, pyarrow.parquet as pq

PKG, BASE, W = sys.argv[1:4]
os.makedirs(W, exist_ok=True)
t0 = time.time()
seq2uid, uid_len, gid_uid = {}, [], []
with open(f"{PKG}/proteins_gid.faa") as fh, open(f"{W}/.uniq.faa.tmp", "w") as out:
    gid, s = None, None
    for line in fh:
        if line.startswith(">"):
            gid = int(line[1:]); assert gid == len(gid_uid)
        else:
            s = line.rstrip("\n")
            u = seq2uid.get(s)
            if u is None:
                u = len(uid_len); seq2uid[s] = u; uid_len.append(len(s)); out.write(f">{u}\n{s}\n")
            gid_uid.append(u)
os.replace(f"{W}/.uniq.faa.tmp", f"{W}/uniq.faa")
del seq2uid
gid_uid = np.asarray(gid_uid, np.int64); U = len(uid_len); lens = np.asarray(uid_len, np.int64)
rep = np.full(U, -1, np.int64)
first = np.unique(gid_uid, return_index=True)[1]
rep[gid_uid[first]] = first
pq.write_table(pa.table({"gid": np.arange(len(gid_uid), dtype=np.int64), "uid": gid_uid}), f"{W}/gid2uid.parquet")
pq.write_table(pa.table({"uid": np.arange(U, dtype=np.int64), "length": lens, "rep_gid": rep,
                         "n_genes": np.bincount(gid_uid, minlength=U)}), f"{W}/uniq.parquet")
tok = pq.read_table(BASE).to_pandas().sort_values("gid")
assert np.array_equal(tok.gid.values, np.arange(len(gid_uid)))
t = tok.token.values
size = np.bincount(t)[t]
Q = np.unique(gid_uid[size < 3])
np.savetxt(f"{W}/Q.uid", Q, fmt="%d")
res = dict(genes=int(len(gid_uid)), unique=int(U), residues_unique=int(lens.sum()), query_genes_token_lt3=int((size < 3).sum()),
           query_uids=int(len(Q)), query_residues=int(lens[Q].sum()), seconds=round(time.time() - t0, 1))
json.dump(res, open(f"{W}/unique.json", "w"), indent=1)
print(json.dumps(res), flush=True)
