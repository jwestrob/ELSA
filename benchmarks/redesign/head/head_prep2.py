#!/usr/bin/env python
"""Workstream head, step 1b (amendment 1, written before any training or evaluation number).

Prefilter saturation is the norm in the teacher searches (-s 7.5 fills the max-seqs slots with k-mer candidates):
96% of S queries reach 8,000 prefilter candidates and 99.3% of Q queries reach 1,000, so 'unsaturated prefilter'
selects an atypical 4% of queries. Amended rule: a query's alignment list is complete when its prefilter is
unsaturated OR its number of alignments (self excluded) is at most half the prefilter cap (S <= 4,000, Q <= 500):
then the k-mer ranking ran well past the alignable set. Recomputed with this rule: split.npz['complete'],
eval_val.npz / eval_test.npz (S-search queries), hardneg.npy (stored 50-NN set); positives are unchanged.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import head_prep as hp  # noqa: E402

G = {}


def count_task(task):
    import pyarrow.parquet as pq
    fi, rg = task
    path, src = hp.HITS[fi]
    t = pq.ParquetFile(path).read_row_group(rg, columns=["q", "t", "bits"])
    q = t["q"].to_numpy().astype(np.int64)
    tt = t["t"].to_numpy().astype(np.int64)
    ns = q != tt
    cnt = np.bincount(q[ns], minlength=G["U"]).astype(np.int32)
    if src == "S":
        e = ns & G["is_cand"][q]
        ev = (q[e].astype(np.uint32), tt[e].astype(np.uint32), t["bits"].to_numpy()[e].astype(np.float32))
    else:
        ev = None
    return src, cnt, ev


def main():
    import pyarrow.parquet as pq
    T = int(os.environ.get("SLURM_CPUS_ON_NODE", "2"))
    t0 = time.time()
    D = hp.DATA
    s = dict(np.load(D / "split.npz"))
    U = len(s["fam"])
    is_train, is_val, is_test, fam = s["is_train"], s["is_val"], s["is_test"], s["fam"]
    info = dict(amendment=1, rule="complete = prefilter unsaturated OR alignments (self excluded) <= cap / 2 (S 4,000; Q 500)",
                code_sha256=hp.hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), slurm_job=os.environ.get("SLURM_JOB_ID"))
    # S queries: eval candidates need their hits; candidates = all val / test S queries (complete decided below)
    lcS, lcQ = s["lc_S"], s["lc_Q"]
    is_cand = (lcS >= 0) & (is_val | is_test)
    G.update(U=U, is_cand=is_cand)
    tasks = [(fi, rg) for fi, (p, _) in enumerate(hp.HITS) for rg in range(pq.ParquetFile(p).metadata.num_row_groups)]
    nS = np.zeros(U, np.int64)
    nQ = np.zeros(U, np.int64)
    EV = []
    with mp.get_context("fork").Pool(min(T, 48)) as pool:
        for src, cnt, ev in pool.imap_unordered(count_task, tasks, chunksize=1):
            if src == "S":
                nS += cnt
                EV.append(ev)
            else:
                nQ += cnt
    cS = (lcS >= 0) & ((lcS < hp.LC["S"][1]) | (nS <= hp.LC["S"][1] // 2))
    cQ = (lcQ >= 0) & ((lcQ < hp.LC["Q"][1]) | (nQ <= hp.LC["Q"][1] // 2))
    complete = cS | cQ
    for name, c, n, lc in (("S", cS, nS, lcS), ("Q", cQ, nQ, lcQ)):
        qm = lc >= 0
        info[f"queries_{name}"] = dict(n=int(qm.sum()), complete=int(c.sum()),
                                       alignments_q10_q50_q90_q99=np.quantile(n[qm], [.1, .5, .9, .99]).tolist(),
                                       train_complete=int((c & is_train).sum()))
    info["train_uids_complete"] = int((complete & is_train).sum())
    s["complete"] = complete
    s["complete_S"] = cS
    s["complete_Q"] = cQ
    s["n_aln_S"] = nS.astype(np.int32)
    s["n_aln_Q"] = nQ.astype(np.int32)
    np.savez(D / "split.npz", **s)
    hp.stamp("complete", info["queries_S"], info["queries_Q"], info["train_uids_complete"])

    # ---------------------------------------------------------------- evaluation sets (S queries, complete)
    rng = np.random.default_rng(hp.SEED + 1)
    eq = np.concatenate([x[0] for x in EV]).astype(np.int64)
    et = np.concatenate([x[1] for x in EV]).astype(np.int64)
    eb = np.concatenate([x[2] for x in EV])
    del EV
    for name, flag in (("val", is_val), ("test", is_test)):
        pool_ = np.flatnonzero(flag & cS)
        c = pool_[rng.permutation(len(pool_))]
        c = c[nS[c] > 0][:hp.N_EVAL[name]]
        m = np.isin(eq, c)
        qq, tt_, bb = eq[m], et[m], eb[m]
        o = np.lexsort((-bb, tt_, qq))
        qq, tt_, bb = qq[o], tt_[o], bb[o]
        first = np.r_[True, (qq[1:] != qq[:-1]) | (tt_[1:] != tt_[:-1])]
        qq, tt_, bb = qq[first], tt_[first], bb[first]
        rank_of = np.full(U, -1, np.int64)
        rank_of[c] = np.arange(len(c))
        o = np.lexsort((-bb, rank_of[qq]))
        qq, tt_, bb = qq[o], tt_[o], bb[o]
        cnt = np.bincount(rank_of[qq], minlength=len(c))
        assert (cnt > 0).all()
        ip = np.r_[0, np.cumsum(cnt)].astype(np.int64)
        np.savez(D / f"eval_{name}.npz", q=c.astype(np.int32), indptr=ip, tgt=tt_.astype(np.int32), bits=bb)
        info[f"eval_{name}"] = dict(pool=int(len(pool_)), pool_with_alignment=int((nS[pool_] > 0).sum()),
                                    queries=int(len(c)), hits=int(len(tt_)),
                                    aligned_per_query_q10_q50_q90_max=np.quantile(cnt, [.1, .5, .9, 1]).tolist(),
                                    prefilter_saturated_share=float((lcS[c] >= hp.LC["S"][1]).mean()),
                                    query_genomes_note="S search = 356 random genomes + gold-locus neighbourhoods")
        hp.stamp("eval", name, info[f"eval_{name}"])

    # ---------------------------------------------------------------- stored 50-NN hard negatives, amended rule
    key = np.load(D / "pair_keys_sorted.npy")
    anchors = np.load(D / "anchors.npy").astype(np.int64)
    I = np.load(hp.KNN)["I"]
    hn = np.full((U, hp.N_HARD), -1, np.int32)
    for s0 in range(0, len(anchors), 200_000):
        A = anchors[s0:s0 + 200_000]
        C = I[A].astype(np.int64)
        ok = C >= 0
        Cc = np.where(ok, C, 0)
        ok &= (Cc != A[:, None]) & is_train[Cc]
        fa = fam[A][:, None]
        ok &= ~((fa >= 0) & (fam[Cc] == fa))
        ok &= complete[A][:, None] | complete[Cc]
        kk = A[:, None] * U + Cc
        p = np.minimum(np.searchsorted(key, kk), len(key) - 1)
        ok &= key[p] != kk
        rk = np.cumsum(ok, axis=1)
        sel = ok & (rk <= hp.N_HARD)
        rr, cc = np.nonzero(sel)
        hn[A[rr], rk[rr, cc] - 1] = C[rr, cc]
    np.save(D / "hardneg.npy", hn)
    nh = (hn[anchors] >= 0).sum(1)
    info["hard_negatives_50nn"] = dict(anchors_with_any=int((nh > 0).sum()), mean_per_anchor=float(nh.mean()))
    hp.stamp("hard negatives", info["hard_negatives_50nn"])
    info["seconds"] = round(time.time() - t0, 1)
    (D / "prep_amend1.json").write_text(json.dumps(info, indent=1, default=float) + "\n")
    hp.stamp("done", info["seconds"])


if __name__ == "__main__":
    main()
