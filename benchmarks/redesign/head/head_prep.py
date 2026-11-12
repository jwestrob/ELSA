#!/usr/bin/env python
"""Workstream head, step 1: teacher data for a learned head on ESM++ small zca10 vectors.

Writes $RD/head/data/:
  split.npz        genome roles (0 even/test, 1 odd training, 2 odd validation), per-uid flags
                   (train, val, test), rescue3 family per uid, prefilter line counts of the S and Q searches
  pos_csr.npz      symmetrised training alignments (both ends training sequences, E <= 1e-5, no self),
                   per anchor at most CAP_POS partners with the highest weight; weight w, bits, min coverage
  fam.npz          family co-membership among training sequences: start, size and position per uid
  hardneg.npy      (n_uid, N_HARD) zca10 kNN neighbours of training anchors that are training sequences of
                   another family with no alignment either way, where at least one end has a complete
                   (unsaturated) alignment list; -1 padded
  anchors.npy      training sequences with >= 1 alignment partner or >= 1 family co-member
  eval_val.npz, eval_test.npz   query uids (S search, unsaturated, >= 1 alignment), CSR of aligned uids + bits
  zca10_f16__uid.npy, l2_f16__uid.npy   transformed vectors (float16) for training / evaluation on the GPU
  prep.json        counts, checks, seconds
"""
from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

RD = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111")
H = RD / "head"
DATA = H / "data"
EMB = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/embed_queue/runs/dpann__esmpp_small__20261111")
PKG = RD / "ws_dpann/package"
KNN = RD / "lab/knn/knn__esmpp_small__zca10__k50__hnsw.npz"
FAM = RD / "engine/dpann_tokens/mmseqs__seq__e3c50s75_rescue3__dpann.parquet"
HITS = [(RD / f"subgenic/hits/S_{k}.parquet", "S") for k in range(4)] + \
       [(RD / f"repair/hits/Q_{k}.parquet", "Q") for k in range(4)]
LC = {"S": ([RD / f"subgenic/run/S_{k}_pref_lc.tsv" for k in range(4)], 8000),
      "Q": ([RD / f"repair/run/Q_{k}_pref_lc.tsv" for k in range(4)], 1000)}
SEED = 17
VAL_FRAC = 0.2
N_EVAL = {"val": 4000, "test": 5000}
N_CAND = {"val": 12000, "test": 15000}
CAP_POS = 2000
N_HARD = 8

sys.path.insert(0, str(Path(__file__).resolve().parent))
from embed_tokens import load_run, transform  # noqa: E402

G = {}


def stamp(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def code_sha():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def read_task(task):
    """One row group of one hit file: training pairs and evaluation-candidate hits."""
    import pyarrow.parquet as pq
    fi, rg = task
    path, src = HITS[fi]
    pf = pq.ParquetFile(path)
    t = pf.read_row_group(rg, columns=["q", "t", "bits", "qstart", "qend", "qlen", "tstart", "tend", "tlen"])
    q = t["q"].to_numpy().astype(np.int64)
    tt = t["t"].to_numpy().astype(np.int64)
    bits = t["bits"].to_numpy().astype(np.float32)
    ns = q != tt
    m = ns & G["is_train"][q] & G["is_train"][tt]
    if m.any():
        qc = (t["qend"].to_numpy()[m].astype(np.float32) - t["qstart"].to_numpy()[m] + 1) / t["qlen"].to_numpy()[m]
        tc = (t["tend"].to_numpy()[m].astype(np.float32) - t["tstart"].to_numpy()[m] + 1) / t["tlen"].to_numpy()[m]
        cov = np.clip(np.round(100 * np.minimum(qc, tc)), 0, 100).astype(np.uint8)
    else:
        cov = np.empty(0, np.uint8)
    tr = (q[m].astype(np.uint32), tt[m].astype(np.uint32), bits[m], cov)
    if src == "S":
        e = ns & G["is_cand"][q]
        ev = (q[e].astype(np.uint32), tt[e].astype(np.uint32), bits[e])
    else:
        ev = (np.empty(0, np.uint32), np.empty(0, np.uint32), np.empty(0, np.float32))
    return fi, len(q), int(ns.size - ns.sum()), tr, ev


def main():
    import pyarrow.parquet as pq
    T = int(os.environ.get("SLURM_CPUS_ON_NODE", "2"))
    t0 = time.time()
    DATA.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    info = dict(code_sha256=code_sha(), slurm_job=os.environ.get("SLURM_JOB_ID"), threads=T, seed=SEED,
                val_frac=VAL_FRAC, cap_pos=CAP_POS, n_hard=N_HARD, knn=str(KNN), fam=str(FAM))

    # ---------------------------------------------------------------- universe, uids, genome split
    g = pq.read_table(PKG / "universe/genes.parquet", columns=["gid", "genome_code"])
    gid = g["gid"].to_numpy()
    assert (gid == np.arange(len(gid))).all()
    gcode = g["genome_code"].to_numpy().astype(np.int64)
    u = np.load(EMB / "uid_of_row.npy").astype(np.int64)
    g2u = pq.read_table(RD / "subgenic/gid2uid.parquet").to_pandas().sort_values("gid")
    assert (g2u.gid.to_numpy() == gid).all()
    info["uid_of_row_equals_subgenic_gid2uid"] = bool((g2u.uid.to_numpy() == u).all())
    assert info["uid_of_row_equals_subgenic_gid2uid"]
    U = int(u.max()) + 1
    n_genomes = int(gcode.max()) + 1
    odd = np.arange(1, n_genomes, 2)
    perm = rng.permutation(odd)
    n_val = int(round(VAL_FRAC * len(odd)))
    role = np.zeros(n_genomes, np.int8)                 # 0 even (test), 1 odd training, 2 odd validation
    role[odd] = 1
    role[perm[:n_val]] = 2
    r = role[gcode]
    in_even = np.bincount(u[r == 0], minlength=U) > 0
    in_tr = np.bincount(u[r == 1], minlength=U) > 0
    in_va = np.bincount(u[r == 2], minlength=U) > 0
    is_train = in_tr & ~in_va & ~in_even
    is_val = in_va & ~in_even
    is_test = in_even
    assert not (is_train & is_test).any() and not (is_train & is_val).any()
    info["genomes"] = dict(total=n_genomes, even=int((role == 0).sum()), odd_train=int((role == 1).sum()),
                           odd_val=int((role == 2).sum()))
    info["uids"] = dict(total=U, train=int(is_train.sum()), val=int(is_val.sum()), test=int(is_test.sum()),
                        odd_train_genome_uids_also_elsewhere=int((in_tr & ~is_train).sum()))
    info["genes_of_train_uids"] = int(is_train[u].sum())
    stamp("split", info["genomes"], info["uids"])

    # ---------------------------------------------------------------- families (rescue3), per uid
    f = pq.read_table(FAM).to_pandas().sort_values("gid")
    assert (f.gid.to_numpy() == gid).all()
    tok = f.token.to_numpy().astype(np.int64)
    rep = pq.read_table(RD / "subgenic/uniq.parquet", columns=["uid", "rep_gid"]).to_pandas().sort_values("uid")
    assert (rep.uid.to_numpy() == np.arange(U)).all()
    fam = tok[rep.rep_gid.to_numpy()]
    info["family_genes_disagreeing_with_their_uid_rep"] = int((tok != fam[u]).sum())
    info["families"] = int(len(np.unique(fam[fam >= 0])))

    # ---------------------------------------------------------------- saturation of the searches
    lc = {}
    for src, (files, cap) in LC.items():
        a = np.full(U, -1, np.int64)
        for fn in files:
            x = np.loadtxt(fn, dtype=np.int64, delimiter="\t", ndmin=2)
            a[x[:, 0]] = x[:, 1]
        lc[src] = a
        info[f"queries_{src}"] = dict(n=int((a >= 0).sum()), saturated=int((a >= cap).sum()), cap=cap)
    unsat_S = (lc["S"] >= 0) & (lc["S"] < LC["S"][1])
    unsat_Q = (lc["Q"] >= 0) & (lc["Q"] < LC["Q"][1])
    complete = unsat_S | unsat_Q
    info["train_uids_with_complete_list"] = int((complete & is_train).sum())
    info["train_uids_queried"] = int((((lc["S"] >= 0) | (lc["Q"] >= 0)) & is_train).sum())

    # evaluation candidates: S-search queries with an unsaturated prefilter
    cand = {}
    is_cand = np.zeros(U, bool)
    for name, flag in (("val", is_val), ("test", is_test)):
        c = np.flatnonzero(flag & unsat_S)
        c = c[rng.permutation(len(c))][:N_CAND[name]]
        cand[name] = c
        is_cand[c] = True
        info[f"eval_{name}_candidates"] = dict(pool=int((flag & unsat_S).sum()), drawn=int(len(c)))
    np.savez(DATA / "split.npz", genome_role=role, is_train=is_train, is_val=is_val, is_test=is_test, fam=fam,
             lc_S=lc["S"], lc_Q=lc["Q"], complete=complete)

    # ---------------------------------------------------------------- read all alignments
    G.update(is_train=is_train, is_cand=is_cand)
    tasks = [(fi, rg) for fi, (p, _) in enumerate(HITS) for rg in range(pq.ParquetFile(p).metadata.num_row_groups)]
    t1 = time.time()
    TR, EV = [], []
    rows = np.zeros(len(HITS), np.int64)
    selfhits = 0
    with mp.get_context("fork").Pool(min(T, 48)) as pool:
        for fi, n, ns, tr, ev in pool.imap_unordered(read_task, tasks, chunksize=1):
            rows[fi] += n
            selfhits += ns
            TR.append(tr)
            EV.append(ev)
    info["hit_rows"] = {HITS[i][0].name: int(rows[i]) for i in range(len(HITS))}
    info["hit_rows_total"] = int(rows.sum())
    info["self_hits"] = int(selfhits)
    info["read_seconds"] = round(time.time() - t1, 1)
    stamp("read", info["hit_rows_total"], "rows", info["read_seconds"], "s")

    q = np.concatenate([x[0] for x in TR]).astype(np.int64)
    t = np.concatenate([x[1] for x in TR]).astype(np.int64)
    bits = np.concatenate([x[2] for x in TR])
    cov = np.concatenate([x[3] for x in TR])
    del TR
    info["train_pairs_directed_raw"] = int(len(q))
    w = (np.log1p(bits / 40.0) * (0.2 + 0.8 * cov.astype(np.float32) / 100.0)).astype(np.float32)
    # symmetrise; duplicate (a, b) keep the highest weight
    a = np.concatenate([q, t])
    b = np.concatenate([t, q])
    w2 = np.concatenate([w, w])
    bt2 = np.concatenate([bits, bits])
    cv2 = np.concatenate([cov, cov])
    del q, t, w, bits, cov
    key = a * U + b
    del a, b
    t1 = time.time()
    o = np.lexsort((-w2, key))
    key, w2, bt2, cv2 = key[o], w2[o], bt2[o], cv2[o]
    del o
    first = np.r_[True, key[1:] != key[:-1]]
    key, w2, bt2, cv2 = key[first], w2[first], bt2[first], cv2[first]
    del first
    info["train_pairs_symmetric_unique"] = int(len(key))
    np.save(DATA / "pair_keys_sorted.npy", key)              # complete key set, for the hard-negative check
    a = key // U
    b = key % U
    # cap per anchor: keep the CAP_POS highest weights
    o = np.lexsort((-w2, a))
    a, b, w2, bt2, cv2 = a[o], b[o], w2[o], bt2[o], cv2[o]
    del o
    cnt = np.bincount(a, minlength=U)
    start = np.r_[0, np.cumsum(cnt)[:-1]]
    rank = np.arange(len(a)) - start[a]
    k = rank < CAP_POS
    a, b, w2, bt2, cv2 = a[k], b[k], w2[k], bt2[k], cv2[k]
    cntk = np.bincount(a, minlength=U)
    indptr = np.r_[0, np.cumsum(cntk)].astype(np.int64)
    np.savez(DATA / "pos_csr.npz", indptr=indptr, tgt=b.astype(np.int32), w=w2, bits=bt2, cov=cv2)
    has = cntk > 0
    info["positives"] = dict(pairs_after_cap=int(len(a)), anchors_with_alignment=int(has.sum()),
                             anchors_capped=int((cnt > CAP_POS).sum()),
                             per_anchor_q10_q50_q90_q99=np.quantile(cntk[has], [.1, .5, .9, .99]).tolist(),
                             bits_q10_q50_q90=np.quantile(bt2, [.1, .5, .9]).tolist(),
                             mincov_q10_q50_q90=np.quantile(cv2, [.1, .5, .9]).tolist(),
                             w_q10_q50_q90=np.quantile(w2, [.1, .5, .9]).tolist(),
                             seconds=round(time.time() - t1, 1))
    stamp("positives", info["positives"])
    del a, b, w2, bt2, cv2, rank, k

    # ---------------------------------------------------------------- families among training sequences
    tu = np.flatnonzero(is_train & (fam >= 0))
    o = np.lexsort((tu, fam[tu]))
    mem = tu[o]
    ff = fam[mem]
    brk = np.r_[0, np.flatnonzero(ff[1:] != ff[:-1]) + 1, len(ff)]
    sizes = np.diff(brk)
    fstart = np.full(U, -1, np.int64)
    fn = np.zeros(U, np.int32)
    fpos = np.full(U, -1, np.int32)
    gidx = np.repeat(np.arange(len(sizes)), sizes)
    fstart[mem] = brk[:-1][gidx]
    fn[mem] = sizes[gidx]
    fpos[mem] = (np.arange(len(mem)) - brk[:-1][gidx]).astype(np.int32)
    np.savez(DATA / "fam.npz", members=mem.astype(np.int32), start=fstart, n=fn, pos=fpos)
    info["family_training"] = dict(families_with_train_members=int(len(sizes)),
                                   families_ge2=int((sizes >= 2).sum()),
                                   train_uids_with_comember=int((fn >= 2).sum()),
                                   size_q50_q90_q99_max=np.quantile(sizes, [.5, .9, .99, 1]).tolist())
    anchors = np.flatnonzero(is_train & (has | (fn >= 2)))
    np.save(DATA / "anchors.npy", anchors.astype(np.int32))
    info["anchors"] = dict(n=int(len(anchors)), alignment_only=int((has[anchors] & (fn[anchors] < 2)).sum()),
                           family_only=int((~has[anchors] & (fn[anchors] >= 2)).sum()),
                           both=int((has[anchors] & (fn[anchors] >= 2)).sum()))
    stamp("families", info["family_training"], info["anchors"])

    # ---------------------------------------------------------------- hard negatives from the zca10 kNN
    t1 = time.time()
    I = np.load(KNN)["I"]
    assert I.shape[0] == U
    hn = np.full((U, N_HARD), -1, np.int32)
    n_valid_tot = 0
    step = 200_000
    for s in range(0, len(anchors), step):
        A = anchors[s:s + step].astype(np.int64)
        C = I[A].astype(np.int64)
        ok = C >= 0
        Cc = np.where(ok, C, 0)
        ok &= Cc != A[:, None]
        ok &= is_train[Cc]
        fa = fam[A][:, None]
        ok &= ~((fa >= 0) & (fam[Cc] == fa))
        ok &= complete[A][:, None] | complete[Cc]
        kk = A[:, None] * U + Cc
        pos = np.searchsorted(key, kk)
        pos = np.minimum(pos, len(key) - 1)
        ok &= key[pos] != kk
        rk = np.cumsum(ok, axis=1)
        sel = ok & (rk <= N_HARD)
        rr, cc = np.nonzero(sel)
        hn[A[rr], rk[rr, cc] - 1] = C[rr, cc]
        n_valid_tot += int(ok.sum())
    del I
    np.save(DATA / "hardneg.npy", hn)
    nh = (hn[anchors] >= 0).sum(1)
    info["hard_negatives"] = dict(anchors_with_any=int((nh > 0).sum()), mean_per_anchor=float(nh.mean()),
                                  eligible_pairs=n_valid_tot, seconds=round(time.time() - t1, 1))
    stamp("hard negatives", info["hard_negatives"])
    del key

    # ---------------------------------------------------------------- evaluation sets
    eq = np.concatenate([x[0] for x in EV]).astype(np.int64)
    et = np.concatenate([x[1] for x in EV]).astype(np.int64)
    eb = np.concatenate([x[2] for x in EV])
    del EV
    o = np.lexsort((-eb, et, eq))
    eq, et, eb = eq[o], et[o], eb[o]
    first = np.r_[True, (eq[1:] != eq[:-1]) | (et[1:] != et[:-1])]
    eq, et, eb = eq[first], et[first], eb[first]
    ne = np.bincount(eq, minlength=U)
    for name in ("val", "test"):
        c = cand[name]
        c = c[ne[c] > 0][:N_EVAL[name]]
        m = np.isin(eq, c)
        qq, tt_, bb = eq[m], et[m], eb[m]
        # CSR in the (random) order of c
        rank_of = np.full(U, -1, np.int64)
        rank_of[c] = np.arange(len(c))
        o = np.lexsort((-bb, rank_of[qq]))
        qq, tt_, bb = qq[o], tt_[o], bb[o]
        cnt = np.bincount(rank_of[qq], minlength=len(c))
        ip = np.r_[0, np.cumsum(cnt)].astype(np.int64)
        np.savez(DATA / f"eval_{name}.npz", q=c.astype(np.int32), indptr=ip, tgt=tt_.astype(np.int32), bits=bb)
        info[f"eval_{name}"] = dict(queries=int(len(c)), hits=int(len(tt_)),
                                    candidates_without_hit=int((ne[cand[name]] == 0).sum()),
                                    aligned_per_query_q10_q50_q90_max=np.quantile(cnt, [.1, .5, .9, 1]).tolist(),
                                    targets_that_are_train_uids=float(is_train[tt_].mean()))
        stamp("eval", name, info[f"eval_{name}"])

    # ---------------------------------------------------------------- vectors
    t1 = time.time()
    X, uu, valid, cnt = load_run(EMB)
    assert (uu == u).all()
    info["vectors"] = dict(shape=list(X.shape), invalid_rows=int((~valid).sum()))
    L = X.copy()
    transform(L, valid, cnt, "l2")
    np.save(DATA / "l2_f16__uid.npy", L.astype(np.float16))
    del L
    info["zca10"] = transform(X, valid, cnt, "zca10")
    z = np.load(KNN)
    Iz, Dz = z["I"], z["D"]
    qq = rng.choice(np.flatnonzero(Iz[:, 0] >= 0), 20000, replace=False)
    rec = np.einsum("ij,ij->i", X[qq], X[Iz[qq, 0]])
    info["zca10_knn_rank1_recompute_max_abs_err"] = float(np.abs(rec - Dz[qq, 0]).max())
    info["knn_has_self_in_column0"] = float((Iz[qq, 0] == qq).mean())
    assert info["zca10_knn_rank1_recompute_max_abs_err"] < 1e-3
    del Iz, Dz, z
    np.save(DATA / "zca10_f16__uid.npy", X.astype(np.float16))
    info["vectors"]["seconds"] = round(time.time() - t1, 1)
    stamp("vectors", info["vectors"], info["zca10_knn_rank1_recompute_max_abs_err"])
    info["seconds"] = round(time.time() - t0, 1)
    (DATA / "prep.json").write_text(json.dumps(info, indent=1, default=float) + "\n")
    stamp("done", info["seconds"], "s")


if __name__ == "__main__":
    main()
