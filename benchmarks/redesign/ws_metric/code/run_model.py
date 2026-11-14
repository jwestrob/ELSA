#!/usr/bin/env python
"""Per-model stages: stats, train, eval.

stats  one pass over the reference: finite mask, global mean, per-genome means, covariance
       eigendecomposition, and the CSLS radius r_t (mean top-10 cosine of each reference
       protein to all query proteins), exact. The query radius r_q (mean of the exact top-10
       over the reference) comes from the raw arm's exact search in eval.
train  every learned arm in common.LEARNED (training query genomes x Actinomycetota targets).
eval   transforms the held-out queries, streams the reference once, and for every arm keeps
       the exact top-60 over (a) the whole reference and (b) other-phylum reference only;
       scores the labelled pair populations with every arm.
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from common import (K_RETRIEVE, LEARNED, MODELS, N_QUERY, N_REF, SETS, TRAIN, UNTRAINED, W,
                    Transforms, code_hash, heldout, log, make_head, readers, ref_tables)

CH = 262_144      # reference rows per chunk
QB = 2048         # query rows per matmul block


def finite_rows(X):
    return np.isfinite(X).all(axis=1)


# ======================================================================================
# stats
# ======================================================================================

def stage_stats(model, out, threads, smoke):
    import torch
    qv, rv = readers(model)
    T = ref_tables()
    d = rv.shape[1]
    n_ref = N_REF if not smoke else 300_000
    ncode = T["n_codes"]
    gsum = np.zeros((ncode, d), np.float64)
    gcnt = np.zeros(ncode, np.int64)
    xtx = torch.zeros((d, d), dtype=torch.float64)
    fin_ref = np.zeros(N_REF, bool)
    r_t = np.full(N_REF, np.nan, np.float32)
    Q = qv.read(0, N_QUERY).copy()
    qfin = finite_rows(Q)
    Q[~qfin] = 0.0
    Qt = torch.from_numpy(Q)
    t0 = time.time()
    for c0 in range(0, n_ref, CH):
        c1 = min(c0 + CH, n_ref)
        X = rv.read(c0, c1)
        fin = finite_rows(X)
        fin_ref[c0:c1] = fin
        rows = np.arange(c0, c1)[fin]
        X = X[fin]
        codes = T["ref_codes"][rows]
        order = np.argsort(codes, kind="stable")
        cs = codes[order]
        starts = np.r_[0, np.flatnonzero(np.diff(cs)) + 1]
        gsum[cs[starts]] += np.add.reduceat(X[order].astype(np.float64), starts, axis=0)
        gcnt[cs[starts]] += np.diff(np.r_[starts, len(cs)])
        Xt = torch.from_numpy(X)
        xtx += (Xt.T @ Xt).double()
        rt_chunk = np.empty(len(X), np.float32)
        for b0 in range(0, len(X), 8192):
            M = Xt[b0:b0 + 8192] @ Qt.T                       # (b, n_query)
            rt_chunk[b0:b0 + 8192] = M.topk(10, dim=1).values.mean(dim=1).numpy()
        r_t[rows] = rt_chunk
        log(model, "stats chunk", c0, f"{time.time()-t0:.0f}s")
    n = int(gcnt.sum())
    mu = gsum.sum(axis=0) / n
    cov = xtx.numpy() / n - np.outer(mu, mu)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    evals, evecs = evals[order], evecs[:, order]
    # query genome means
    qcodes = T["query_codes"]
    for c in np.unique(qcodes):
        sel = (qcodes == c) & qfin
        gsum[c] = Q[sel].astype(np.float64).sum(axis=0)
        gcnt[c] = sel.sum()
    gmeans = (gsum / np.maximum(gcnt, 1)[:, None]).astype(np.float32)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "stats.npz", mu=mu.astype(np.float32), gmeans=gmeans, gcount=gcnt,
             evals=evals, evecs=evecs.astype(np.float32))
    np.save(out / "r_t.npy", r_t)
    np.save(out / "ref_finite.npy", fin_ref)
    np.save(out / "query_finite.npy", qfin)
    ev = evals / evals.sum()
    info = dict(model=model, d=d, n_ref_finite=int(fin_ref.sum()), n_ref_scanned=n_ref,
                n_query_finite=int(qfin.sum()), mu_norm=float(np.linalg.norm(mu)),
                explained_top=[float(x) for x in ev[:10]], explained_cum_256=float(ev[:256].sum()),
                r_t_mean=float(np.nanmean(r_t)),
                seconds=round(time.time() - t0, 1), smoke=smoke)
    json.dump(info, open(out / "stats.json", "w"), indent=1)
    log(model, "stats", info)


# ======================================================================================
# train
# ======================================================================================

def stage_train(model, stats_dir, out, threads, smoke, only=None):
    import torch
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    torch.set_num_threads(threads)
    out.mkdir(parents=True, exist_ok=True)
    qv, rv = readers(model)
    st = np.load(stats_dir / "stats.npz")
    mu = st["mu"].astype(np.float32)
    d = rv.shape[1]
    H = heldout()
    T = ref_tables()
    pos = pd.read_parquet(SETS / f"pos_{model}.parquet")
    pos = pos[~pos.q_heldout & pos.tstratum.isin(H["actino_strata"])].reset_index(drop=True)
    # pool = every protein of the training query genomes and Actinomycetota reference genomes
    ref_rows = np.flatnonzero(T["ref_actino"])
    q_rows = np.flatnonzero(np.isin(T["query_codes"], H["training_codes"]))
    Xr = rv.rows(ref_rows) if not smoke else rv.rows(ref_rows[:200_000])
    if smoke:
        ref_rows = ref_rows[:200_000]
    Xq = qv.rows(q_rows)
    pool_gid = np.r_[ref_rows, N_REF + q_rows]
    pool_code = np.r_[T["ref_codes"][ref_rows], T["query_codes"][q_rows]]
    X = np.vstack([Xr, Xq])
    del Xr, Xq
    fin = finite_rows(X)
    X, pool_gid, pool_code = X[fin], pool_gid[fin], pool_code[fin]
    X_raw = torch.from_numpy(X)
    del X
    tfm = Transforms(stats_dir)
    pool_code_t = torch.from_numpy(pool_code)
    inputs = {}

    def head_input(inp):
        # same preprocessing as Transforms.head_input, applied to the whole pool
        if inp not in inputs:
            inputs.clear()
            inputs[inp] = tfm.head_input(inp, X_raw, pool_code_t)
        return inputs[inp]
    g2p = pd.Series(np.arange(len(pool_gid)), index=pool_gid)
    pos = pos[pos.qgid.isin(g2p.index) & pos.tgid.isin(g2p.index)].reset_index(drop=True)
    # genome -> pool slice for hard negatives
    order = np.argsort(pool_code, kind="stable")
    sc = pool_code[order]
    gstart = np.searchsorted(sc, np.arange(T["n_codes"]), side="left")
    gend = np.searchsorted(sc, np.arange(T["n_codes"]), side="right")
    order_t = torch.from_numpy(order)
    log(model, "pool", tuple(X_raw.shape), "positives (train genomes x Actinomycetota)", len(pos))

    summary = {}
    for name, cfg in LEARNED.items():
        if only and name not in only:
            continue
        t0 = time.time()
        seed = cfg.get("seed", TRAIN["seed"])
        X = head_input(cfg.get("inp", "gcenter"))
        sel = {"ge6": pos.src_ge6, "acc05": pos.src_acc05, "vis": pos.src_ge6 & pos.vis}[cfg["pos"]].to_numpy()
        P = pos[sel].reset_index(drop=True)
        pq = g2p.loc[P.qgid].to_numpy()
        pt = g2p.loc[P.tgid].to_numpy()
        # same-family mask: connected components of the positive graph (query and target nodes)
        nodes, inv = np.unique(np.r_[pq, pt], return_inverse=True)
        a, b = inv[:len(pq)], inv[len(pq):]
        G = coo_matrix((np.ones(len(a)), (a, b)), shape=(len(nodes), len(nodes)))
        ncomp, lab = connected_components(G, directed=False)
        comp = lab[a]
        csize = np.bincount(comp)
        # validation split by query protein
        rng = np.random.default_rng(seed)
        uq = np.unique(pq)
        valq = set(rng.choice(uq, max(1, int(len(uq) * TRAIN["val_frac"])), replace=False).tolist())
        isval = np.array([x in valq for x in pq])
        tr_idx, va_idx = np.flatnonzero(~isval), np.flatnonzero(isval)
        torch.manual_seed(seed)
        head = make_head(cfg["kind"], d, cfg["d_out"])
        opt = torch.optim.Adam(head.parameters(), lr=TRAIN["lr"])
        steps = TRAIN["steps"] if not smoke else 60
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
        bs = min(TRAIN["batch"], len(tr_idx))
        pq_t, pt_t, comp_t = torch.from_numpy(pq), torch.from_numpy(pt), torch.from_numpy(comp)
        pcode_t = torch.from_numpy(pool_code)
        gstart_t, glen_t = torch.from_numpy(gstart), torch.from_numpy(gend - gstart)
        temp = cfg["temp"]
        hard = cfg["neg"] == "hard"
        nh = cfg.get("n_hard", TRAIN["n_hard"])
        gen = torch.Generator().manual_seed(seed)

        def batch_loss(idx, train_mode):
            i_q, i_t = pq_t[idx], pt_t[idx]
            zq = torch.nn.functional.normalize(head(X[i_q]), dim=1)
            zt = torch.nn.functional.normalize(head(X[i_t]), dim=1)
            logits = zq @ zt.T / temp
            c = comp_t[idx]
            mask = (c[:, None] == c[None, :])
            mask.fill_diagonal_(False)
            logits = logits.masked_fill(mask, float("-inf"))
            lq, lt = logits, logits.T
            if hard:
                def sample(gcode):
                    r = torch.rand(len(gcode), nh, generator=gen)
                    off = (r * glen_t[gcode][:, None]).long()
                    return order_t[gstart_t[gcode][:, None] + off]
                ht = sample(pcode_t[i_t])                      # negatives from the target's genome
                hq = sample(pcode_t[i_q])                      # negatives from the query's genome
                ht = ht.masked_fill(ht == i_t[:, None], 0)     # drop the positive itself (rare)
                hq = hq.masked_fill(hq == i_q[:, None], 0)
                zht = torch.nn.functional.normalize(head(X[ht.reshape(-1)]), dim=1).view(len(idx), nh, -1)
                zhq = torch.nn.functional.normalize(head(X[hq.reshape(-1)]), dim=1).view(len(idx), nh, -1)
                lq = torch.cat([lq, (zq[:, None, :] * zht).sum(-1) / temp], dim=1)
                lt = torch.cat([lt, (zt[:, None, :] * zhq).sum(-1) / temp], dim=1)
            target = torch.arange(len(idx))
            return (torch.nn.functional.cross_entropy(lq, target) +
                    torch.nn.functional.cross_entropy(lt, target)) / 2

        curve = []
        perm = rng.permutation(tr_idx)
        p0 = 0
        for step in range(steps):
            if p0 + bs > len(perm):
                perm = rng.permutation(tr_idx)
                p0 = 0
            idx = torch.from_numpy(perm[p0:p0 + bs])
            p0 += bs
            loss = batch_loss(idx, True)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            if step % 250 == 0 or step == steps - 1:
                with torch.no_grad():
                    vb = torch.from_numpy(rng.choice(va_idx, min(bs, len(va_idx)), replace=False))
                    vl = float(batch_loss(vb, False))
                curve.append(dict(step=step, train=loss.item(), val=vl))
                log(model, name, step, round(loss.item(), 4), "val", round(vl, 4))
        torch.save(head.state_dict(), out / f"{name}.pt")
        summary[name] = dict(cfg=cfg, n_pos=int(len(P)), n_train=int(len(tr_idx)), n_val=int(len(va_idx)),
                             vis_rate=float(P.vis.mean()), n_components=int(ncomp),
                             largest_component_pairs=int(csize.max()), top5_component_pairs=[int(x) for x in np.sort(csize)[::-1][:5]],
                             steps=steps, batch=bs, epochs=round(steps * bs / max(len(tr_idx), 1), 2),
                             curve=curve, seconds=round(time.time() - t0, 1))
        json.dump(summary, open(out / ("train_summary.json" if not only else f"train_summary_{'_'.join(sorted(only))}.json"), "w"), indent=1)
    log(model, "train done")


# ======================================================================================
# eval
# ======================================================================================

SEARCH = os.environ.get("WSM_SEARCH", "faiss")


def _merge(tops, key, v, gi, K):
    import torch
    tv, ti = tops[key]
    cv = torch.cat([tv, v], dim=1)
    ci = torch.cat([ti, gi], dim=1)
    nv, j = cv.topk(K, dim=1)
    tops[key] = (nv, torch.gather(ci, 1, j))


def search_chunk(Q, Y, rows_t, n_oth, tops, a, K, method=None):
    """Exact inner-product top-K of every query against the chunk rows Y (whole chunk,
    and the leading n_oth other-phylum rows), merged into the running tops."""
    import torch
    method = method or SEARCH
    for s in ("all", "other"):
        ncol = len(Y) if s == "all" else n_oth
        k = min(K, ncol)
        if k == 0:
            continue
        if method == "faiss":
            import faiss
            D, I = faiss.knn(np.ascontiguousarray(Q.numpy()), np.ascontiguousarray(Y[:ncol].numpy()), k,
                             metric=faiss.METRIC_INNER_PRODUCT)
            v, i = torch.from_numpy(D), torch.from_numpy(I)
            _merge(tops, (a, s), v, rows_t[i], K)
        else:
            for b0 in range(0, len(Q), QB):
                Sb = Q[b0:b0 + QB] @ Y[:ncol].T
                v, i = Sb.topk(k, dim=1)
                tv, ti = tops[(a, s)]
                sub = {(a, s): (tv[b0:b0 + QB], ti[b0:b0 + QB])}
                _merge(sub, (a, s), v, rows_t[i], K)
                tv[b0:b0 + QB], ti[b0:b0 + QB] = sub[(a, s)]

def stage_eval(model, stats_dir, heads_dir, out, threads, smoke, only=None):
    import torch
    torch.set_num_threads(threads)
    import faiss
    faiss.omp_set_num_threads(threads)
    out.mkdir(parents=True, exist_ok=True)
    qv, rv = readers(model)
    T = ref_tables()
    tf = Transforms(stats_dir, heads_dir)
    arms = [a for a in tf.arms() if not only or a in only]
    log(model, "arms", arms)
    fin_ref = np.load(stats_dir / "ref_finite.npy")
    qs = pd.read_parquet(SETS / "qset.parquet")
    qs = qs[qs.heldout & qs.fin_all].reset_index(drop=True)
    if smoke:
        qs = qs.iloc[::10].reset_index(drop=True)
    qrows = qs.qrow.to_numpy()
    nq = len(qrows)
    Qraw = torch.from_numpy(qv.rows(qrows))
    qcodes = torch.from_numpy(T["query_codes"][qrows])
    Qa = {a: tf.apply(a, Qraw, qcodes, "q") for a in arms}
    K = K_RETRIEVE
    tops = {(a, s): (torch.full((nq, K), float("-inf")), torch.full((nq, K), -1, dtype=torch.long))
            for a in arms for s in ("all", "other")}
    n_ref = N_REF if not smoke else 300_000
    t0 = time.time()
    ref_codes_t = torch.from_numpy(T["ref_codes"])
    for c0 in range(0, n_ref, CH):
        c1 = min(c0 + CH, n_ref)
        rows = np.arange(c0, c1)[fin_ref[c0:c1]]
        if not len(rows):
            continue
        # other-phylum rows first, so the other-phylum search is a leading column slice
        oth = T["ref_other"][rows]
        perm = np.r_[np.flatnonzero(oth), np.flatnonzero(~oth)]
        rows = rows[perm]
        n_oth = int(oth.sum())
        Xr = torch.from_numpy(rv.read(c0, c1)[fin_ref[c0:c1]][perm])
        rows_t = torch.from_numpy(rows)
        codes = ref_codes_t[rows_t]
        for a in arms:
            Y = tf.apply(a, Xr, codes, "t", rows_t)
            search_chunk(Qa[a], Y, rows_t, n_oth, tops, a, K)
        log(model, "eval chunk", c0, f"{time.time()-t0:.0f}s")
    hits_dir = out / "hits"
    hits_dir.mkdir(exist_ok=True)
    qgid = qs.qgid.to_numpy()
    for a in arms:
        frames = []
        for s in ("all", "other"):
            tv, ti = tops[(a, s)]
            v, i = tv.numpy(), ti.numpy()
            ok = np.isfinite(v) & (i >= 0)
            qq = np.repeat(qgid, K).reshape(nq, K)
            rk = np.tile(np.arange(1, K + 1, dtype=np.int16), (nq, 1))
            frames.append(pd.DataFrame(dict(search=s, qgid=qq[ok], rank=rk[ok], tgid=i[ok], score=v[ok])))
        pd.concat(frames).to_parquet(hits_dir / f"{a}.parquet", index=False)
    # raw top-60 statistics: r_q (CSLS radius = mean of the exact top-10) and z_q moments
    rv60 = tops[("raw", "all")][0] if "raw" in arms else None
    zstats = None
    if rv60 is not None:
        zstats = pd.DataFrame(dict(qgid=qgid, r_q=rv60[:, :10].mean(dim=1).numpy(),
                                   mean60=rv60.mean(dim=1).numpy(), sd60=rv60.std(dim=1).numpy()))
        zstats.to_parquet(out / "raw_top60_stats.parquet", index=False)
    elif (out / "raw_top60_stats.parquet").exists():
        zstats = pd.read_parquet(out / "raw_top60_stats.parquet")

    score_pairs(model, tf, arms, qv, rv, T, fin_ref, qs, zstats, out, smoke)
    json.dump(dict(model=model, arms=arms, n_queries=nq, seconds=round(time.time() - t0, 1), smoke=smoke,
                   code=code_hash(__file__, Path(__file__).with_name("common.py"))),
              open(out / ("eval_info.json" if not only else f"eval_info_{'_'.join(sorted(only))}.json"), "w"), indent=1)
    log(model, "eval done")


def score_pairs(model, tf, arms, qv, rv, T, fin_ref, qs, zstats, out, smoke, pops_only=None):
    """Score the labelled pair populations with every arm. With a subset of arms, new
    columns are merged into an existing pairs file."""
    import torch
    ref_codes_t = torch.from_numpy(T["ref_codes"])
    pops = {"popA": SETS / f"popA_{model}.parquet", "xphy": SETS / "xphy.parquet", "popB": SETS / "popB.parquet"}
    for pop, path in pops.items():
        if pops_only and pop not in pops_only:
            continue
        if not path.exists():
            log("missing population", pop)
            continue
        df = pd.read_parquet(path)
        if "heldout" in df.columns:
            df = df[df.heldout]
        df = df[df.qgid.isin(qs.qgid)].reset_index(drop=True)
        df = df[fin_ref[df.tgid.to_numpy()]].reset_index(drop=True)
        if not len(df):
            continue
        qrow = df.qgid.to_numpy() - N_REF
        uq = np.unique(qrow)
        qX = torch.from_numpy(qv.rows(uq))
        qinv = torch.from_numpy(np.searchsorted(uq, qrow))
        ut = np.unique(df.tgid.to_numpy())
        tX = torch.from_numpy(rv.rows(ut))
        tinv = torch.from_numpy(np.searchsorted(ut, df.tgid.to_numpy()))
        qc = torch.from_numpy(T["query_codes"][uq])
        tc = ref_codes_t[torch.from_numpy(ut)]
        sc = pd.DataFrame(dict(qgid=df.qgid, tgid=df.tgid))
        for a in arms:
            if a == "csls2":
                continue
            A = tf.apply(a, qX, qc, "q")[qinv]
            B = tf.apply(a, tX, tc, "t", torch.from_numpy(ut))[tinv]
            sc[a] = (A * B).sum(dim=1).numpy()
        rt = tf.r_t.numpy()[df.tgid.to_numpy()]
        if "raw" in sc and zstats is not None:
            z = sc[["qgid"]].merge(zstats, on="qgid", how="left")
            rq = z.r_q.to_numpy()
            sc["csls_q"] = sc.raw.to_numpy() - rq
            sc["csls2"] = 2 * sc.raw.to_numpy() - rq - rt
            sc["z_q"] = (sc.raw.to_numpy() - z.mean60.to_numpy()) / (z.sd60.to_numpy() + 1e-6)
        if pop == "popA":
            sc["rank_q"] = -df.rank_q.to_numpy().astype(np.float64)
        f = out / f"pairs_{pop}.parquet"
        if f.exists():
            old = pd.read_parquet(f)
            if len(old) == len(sc) and (old.qgid.to_numpy() == sc.qgid.to_numpy()).all() and \
                    (old.tgid.to_numpy() == sc.tgid.to_numpy()).all():
                for c in sc.columns[2:]:
                    old[c] = sc[c]
                sc = old
        sc.to_parquet(f, index=False)
        log(model, "pairs", pop, len(sc), list(sc.columns[2:]))


def stage_pairs(model, stats_dir, heads_dir, out, threads, smoke, only=None, pops_only=None):
    import torch
    torch.set_num_threads(threads)
    qv, rv = readers(model)
    T = ref_tables()
    tf = Transforms(stats_dir, heads_dir)
    arms = [a for a in tf.arms() if not only or a in only]
    fin_ref = np.load(stats_dir / "ref_finite.npy")
    qs = pd.read_parquet(SETS / "qset.parquet")
    qs = qs[qs.heldout & qs.fin_all].reset_index(drop=True)
    if smoke:
        qs = qs.iloc[::10].reset_index(drop=True)
    zstats = pd.read_parquet(out / "raw_top60_stats.parquet")
    score_pairs(model, tf, arms, qv, rv, T, fin_ref, qs, zstats, out, smoke, pops_only)
    log(model, "pairs done")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=MODELS)
    ap.add_argument("--stages", default="stats,train,eval")
    ap.add_argument("--threads", type=int, required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--only", default="", help="comma list of arms (train/eval)")
    ap.add_argument("--pops", default="", help="comma list of pair populations (pairs stage)")
    a = ap.parse_args()
    import torch
    torch.set_num_threads(a.threads)
    tag = "smoke" if a.smoke else "full"
    base = W / tag
    sdir, hdir, edir = base / "stats" / a.model, base / "heads" / a.model, base / "eval" / a.model
    only = set(x for x in a.only.split(",") if x) or None
    for stage in a.stages.split(","):
        if stage == "stats":
            stage_stats(a.model, sdir, a.threads, a.smoke)
        elif stage == "train":
            stage_train(a.model, sdir, hdir, a.threads, a.smoke, only)
        elif stage == "eval":
            stage_eval(a.model, sdir, hdir, edir, a.threads, a.smoke, only)
        elif stage == "pairs":
            stage_pairs(a.model, sdir, hdir, edir, a.threads, a.smoke, only,
                        set(x for x in a.pops.split(",") if x) or None)
