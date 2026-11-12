"""Evidence-gated neighbourhood selection (V2) and full candidate-pool dumps.

Candidate generation is copied from `spacedust_seed_extension.extend_query` (private copy in
code/scale/): six query neighbours, their original seeds as witnesses, three strand-consistent
target steps, same-sample and orientation filters, eligible vectors. What changes in V2:
- the 0.85 cosine cutoff on new pairs is removed;
- a proposal is supported only if its witness pair and the proposed pair both have
  calibrated evidence w >= gate (gate = logit p);
- the witness is the proposer with the highest w (ties: witness query row, then target row);
- the 60 retained targets are ranked by w(q,c) + w(witness) for supported proposals and
  w(q,c) for unsupported originals, then cosine, then target row. Originals stay eligible
  without a gate, exactly as in the published selection.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import math
import multiprocessing as mp
import time

import numpy as np
import pandas as pd

OFFSETS = np.array([-3, -2, -1, 1, 2, 3], dtype=np.int64)
ARRAY_NAMES = ("seed_offsets", "seed_targets", "seed_scores", "query_neighbors", "reference_neighbors",
               "query_strand", "reference_strand", "query_sample", "reference_sample",
               "query_vectors", "reference_vectors")
_W = {}


def _eligible(values):
    return np.isfinite(values).all(axis=-1) & np.any(values != 0, axis=-1)


def _seeds_for(query, A):
    lo, hi = int(A["seed_offsets"][query]), int(A["seed_offsets"][query + 1])
    targets = np.asarray(A["seed_targets"][lo:hi], dtype=np.int64)
    scores = np.asarray(A["seed_scores"][lo:hi])
    order = np.argsort(targets, kind="stable")
    targets, scores = targets[order], scores[order]
    unique = np.r_[True, targets[1:] != targets[:-1]] if len(targets) else np.empty(0, dtype=bool)
    targets, scores = targets[unique], scores[unique]
    keep = ((A["reference_sample"][targets] != A["query_sample"][query]) &
            _eligible(A["reference_vectors"][targets]) & bool(_eligible(A["query_vectors"][query])))
    return targets[keep], scores[keep]


def _candidates(qrow, A):
    """Originals and every proposal occurrence (no cosine cutoff), as in extend_query."""
    originals, original_scores = _seeds_for(qrow, A)
    neighbors = np.asarray(A["query_neighbors"][qrow])
    qs, rs = A["query_strand"], A["reference_strand"]
    qsamp, rsamp = A["query_sample"], A["reference_sample"]
    pt, wq, wt, ws = [], [], [], []
    for offset, neighbor in zip(OFFSETS, neighbors):
        if neighbor < 0:
            continue
        assert qsamp[neighbor] == qsamp[qrow]
        targets, scores = _seeds_for(int(neighbor), A)
        if not len(targets):
            continue
        orientation = qs[neighbor] * rs[targets]
        local = np.asarray(A["reference_neighbors"][targets])
        forward = -np.sign(offset) * orientation > 0
        cand = np.where(forward[:, None], local[:, 3:], local[:, :3]).ravel()
        owners = np.repeat(targets, 3)
        supports = np.repeat(scores, 3)
        orients = np.repeat(orientation, 3)
        valid = cand >= 0
        cand, owners, supports, orients = cand[valid], owners[valid], supports[valid], orients[valid]
        valid = ((qs[qrow] * rs[cand] == orients) & (rsamp[cand] != qsamp[qrow]) & (rsamp[cand] == rsamp[owners]))
        pt.append(cand[valid])
        wq.append(np.full(int(valid.sum()), neighbor, dtype=np.int64))
        wt.append(owners[valid])
        ws.append(supports[valid])
    cat = lambda xs, dt: np.concatenate(xs).astype(dt) if xs else np.empty(0, dtype=dt)
    return (originals, original_scores, cat(pt, np.int64), cat(wq, np.int64), cat(wt, np.int64),
            cat(ws, np.float64))


def _cosines(qrow, targets, originals, original_scores, A):
    values = A["reference_vectors"][targets]
    valid = _eligible(values)
    targets, values = targets[valid], values[valid]
    scores = np.empty(len(targets), dtype=np.float64)
    is_original = np.isin(targets, originals)
    scores[is_original] = original_scores[np.searchsorted(originals, targets[is_original])]
    new = ~is_original
    scores[new] = values[new] @ A["query_vectors"][qrow]
    return targets, scores, is_original


def extend_v2(qrow, A, P):
    from common import pair_score, evidence
    empty = pd.DataFrame({c: np.empty(0, dtype=t) for c, t in [
        ("query_row_index", np.int64), ("target_row_index", np.int64), ("similarity", np.float64),
        ("is_original", bool), ("supported", bool), ("witness_query_row", np.int64),
        ("witness_target_row", np.int64), ("w_own", np.float64), ("w_wit", np.float64)]})
    if not bool(_eligible(A["query_vectors"][qrow])):
        return empty
    originals, oscores, pt, wq, wt, ws = _candidates(qrow, A)
    qst, rst, kind, x, y, gate = P["qstats"], P["rstats"], P["kind"], P["x"], P["y"], P["gate"]
    if len(pt):
        we = evidence(pair_score(kind, wq, wt, ws, qst, rst), x, y)
        ok = we >= gate
        pt, wq, wt, we = pt[ok], wq[ok], wt[ok], we[ok]
        order = np.lexsort((wt, wq, -we, pt))
        pt, wq, wt, we = pt[order], wq[order], wt[order], we[order]
        first = np.r_[True, pt[1:] != pt[:-1]] if len(pt) else np.empty(0, dtype=bool)
        pt, wq, wt, we = pt[first], wq[first], wt[first], we[first]
    targets = np.union1d(originals, pt)
    if not len(targets):
        return empty
    targets, scores, is_original = _cosines(qrow, targets, originals, oscores, A)
    w_own = evidence(pair_score(kind, np.full(len(targets), qrow), targets, scores, qst, rst), x, y)
    proposed = np.isin(targets, pt)
    supported = proposed & (w_own >= gate)
    keep = is_original | supported
    targets, scores, is_original, supported, w_own = (a[keep] for a in (targets, scores, is_original, supported, w_own))
    where = np.searchsorted(pt, targets)
    where = np.minimum(where, max(len(pt) - 1, 0))
    w_wit = np.where(supported, we[where] if len(pt) else np.nan, np.nan)
    wit_q = np.where(supported, wq[where] if len(pt) else -1, -1)
    wit_t = np.where(supported, wt[where] if len(pt) else -1, -1)
    off = P.get("offset", 0.0)
    rank = (w_own - off) + np.where(supported, w_wit - off, 0.0)
    order = np.lexsort((targets, -scores, -rank))[:P["k"]]
    return pd.DataFrame(dict(query_row_index=np.full(len(order), qrow, dtype=np.int64),
                             target_row_index=targets[order], similarity=scores[order],
                             is_original=is_original[order], supported=supported[order],
                             witness_query_row=wit_q[order].astype(np.int64),
                             witness_target_row=wit_t[order].astype(np.int64),
                             w_own=w_own[order], w_wit=w_wit[order]))


def pool_dump(qrow, A, P):
    """Every candidate (originals + all proposals, no cutoff, no cap) and every witness occurrence."""
    if not bool(_eligible(A["query_vectors"][qrow])):
        return pd.DataFrame()
    originals, oscores, pt, wq, wt, ws = _candidates(qrow, A)
    targets = np.union1d(originals, pt)
    if not len(targets):
        return pd.DataFrame()
    targets, scores, is_original = _cosines(qrow, targets, originals, oscores, A)
    up, n_occ = np.unique(pt, return_counts=True)
    is_prop = np.isin(targets, up)
    occ = np.zeros(len(targets), dtype=np.int64)
    occ[is_prop] = n_occ[np.searchsorted(up, targets[is_prop])]
    pairs = pd.DataFrame(dict(kind="pair", query_row_index=qrow, target_row_index=targets, similarity=scores,
                              is_original=is_original, is_proposed=is_prop, n_occ=occ,
                              witness_query_row=-1, witness_target_row=-1, witness_similarity=np.nan))
    keep = np.isin(pt, targets)  # drop occurrences whose target vector is ineligible
    occs = pd.DataFrame(dict(kind="occ", query_row_index=qrow, target_row_index=pt[keep], similarity=np.nan,
                             is_original=np.isin(pt[keep], originals), is_proposed=True, n_occ=1,
                             witness_query_row=wq[keep], witness_target_row=wt[keep],
                             witness_similarity=ws[keep]))
    return pd.concat([pairs, occs], ignore_index=True)


FUNCS = {"extend_v2": extend_v2, "pool_dump": pool_dump}


def _init(paths, extra, params):
    from threadpoolctl import threadpool_limits
    _W.clear()
    _W["limiter"] = threadpool_limits(limits=1)
    _W["A"] = {name: np.load(paths[name], mmap_mode="r", allow_pickle=False) for name in ARRAY_NAMES}
    P = dict(params)
    for key, path in extra.items():
        P[key] = np.load(path, mmap_mode="r", allow_pickle=False)
    _W["P"] = P


def _chunk(task):
    number, fn, rows = task
    t0 = time.process_time()
    frames = [FUNCS[fn](int(r), _W["A"], _W["P"]) for r in rows]
    frames = [f for f in frames if len(f)]
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return number, out, time.process_time() - t0


def run(fn, paths, extra, params, rows, workers, chunk_size=None):
    """Process-parallel map of `fn` over query rows; returns (frame sorted by query, target), seconds."""
    t0 = time.perf_counter()
    rows = np.asarray(rows, dtype=np.int64)
    chunk_size = chunk_size or max(1, min(512, math.ceil(len(rows) / (4 * workers))))
    tasks = [(i, fn, rows[lo:lo + chunk_size]) for i, lo in enumerate(range(0, len(rows), chunk_size))]
    results = []
    cpu = 0.0
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"),
                             initializer=_init, initargs=({k: str(v) for k, v in paths.items()},
                                                          {k: str(v) for k, v in extra.items()}, params)) as ex:
        for fut in as_completed([ex.submit(_chunk, t) for t in tasks]):
            number, frame, c = fut.result()
            results.append((number, frame))
            cpu += c
    results.sort(key=lambda r: r[0])
    frames = [f for _, f in results if len(f)]
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return out, dict(seconds=time.perf_counter() - t0, worker_cpu_seconds=cpu, chunks=len(tasks))
