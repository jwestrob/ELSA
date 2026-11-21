#!/usr/bin/env python
"""Synthetic self-test of locus_search.py (small; runs on a laptop in seconds).

1. scan() equals a brute-force enumeration of windows + greedy one-to-one filling + greedy NMS.
2. topk_rows() equals numpy sorting, with tail ranks counted over ties.
3. label_hits() on hand-made instances.
4. search() on a synthetic universe with a planted 3-gene locus in 30 of 60 genomes: planted copies
   come back with small E-values, and the null keeps random windows at E >= 1.
"""
import math
import sys
import tempfile
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import locus_search as ls  # noqa: E402


def brute_scan(eg, es_, ev, n_slots, cc, pos, W, tau):
    genes = np.unique(eg)
    wins = []
    for ia, a in enumerate(genes):
        for b in genes[ia:]:
            if cc[b] != cc[a] or pos[b] - pos[a] + 1 > W:
                continue
            L = pos[b] - pos[a] + 1
            corr = math.log(L)
            m = (eg >= a) & (eg <= b)
            ent = sorted(zip(ev[m], es_[m], eg[m]), key=lambda x: -x[0])
            us, ug_, sc, ua, ub = set(), set(), 0.0, False, False
            for v, s, g in ent:
                if v < tau + corr:
                    break
                if s in us or g in ug_:
                    continue
                us.add(s); ug_.add(g); sc += v - corr
                ua |= g == a; ub |= g == b
            if ua and ub and sc > 0:
                wins.append((sc, a, b))
    wins.sort(key=lambda x: -x[0])
    used, kept = set(), []
    for sc, a, b in wins:
        span = set(g for g in genes if a <= g <= b)
        if span & used:
            continue
        used |= span
        kept.append((a, b, sc))
    return kept


def test_scan(rng):
    N = 400
    cc = np.repeat(np.arange(8), 50).astype(np.int64)
    pos = np.tile(np.arange(50), 8).astype(np.int64)
    for trial in range(40):
        n_slots = int(rng.integers(1, 5))
        k = int(rng.integers(5, 80))
        eg = rng.integers(0, N, k).astype(np.int64)
        es_ = rng.integers(0, n_slots, k).astype(np.int64)
        ev = np.round(rng.uniform(0, 8, k), 3)
        o = np.argsort(eg, kind="stable")
        eg, es_, ev = eg[o], es_[o], ev[o]
        # one entry per (gene, slot)
        key = eg * 10 + es_
        _, first = np.unique(key, return_index=True)
        eg, es_, ev = eg[first], es_[first], ev[first]
        W, tau = int(rng.integers(1, 8)), float(rng.choice([0.0, 1.0, 2.5]))
        ha, hb, sc, _ = ls.scan(eg, es_, ev, n_slots, cc, pos, W, tau, 0)
        ref = brute_scan(eg, es_, ev, n_slots, cc, pos, W, tau)
        got = sorted(zip(ha.tolist(), hb.tolist(), np.round(sc, 9).tolist()))
        exp = sorted((int(a), int(b), round(s, 9)) for a, b, s in ref)
        assert got == exp, (trial, got[:5], exp[:5])
    print("scan == brute force on 40 random cases")


def test_topk(rng):
    S = rng.normal(size=(5, 300)).astype(np.float32)
    S[:, 10:13] = S[:, [20]]           # ties
    K = 40
    lo = np.array([0, 50, 100, 0, 290], np.int64)
    hi = np.array([0, 60, 130, 5, 300], np.int64)
    oi, oc, os_ = np.empty((5, K), np.int64), np.empty((5, K), np.float32), np.empty((5, K), np.float32)
    S2 = S.copy()
    ls.topk_rows(S2, K, lo, hi, oi, oc, os_)
    for r in range(5):
        row = S[r].copy()
        row[lo[r]:hi[r]] = -np.inf
        n_db = 300 - (hi[r] - lo[r])
        top = np.sort(row)[::-1][:K]
        assert np.allclose(oc[r], top)
        for t in range(K):
            rank = int((row >= oc[r, t]).sum())
            assert abs(os_[r, t] - math.log(n_db / rank)) < 1e-5
        assert not np.any((oi[r] >= lo[r]) & (oi[r] < hi[r]))
    print("topk_rows == numpy sort, tail ranks over ties")


def test_label():
    # genes 0..19; instances: 0 type 0 members {2,3,4}; 1 type 1 members {10,11}; 2 type 0 members {15} (non-target)
    members = {0: [2, 3, 4], 1: [10, 11], 2: [15]}
    rows = sorted((g, t) for t, ms in members.items() for g in ms)
    gptr = np.searchsorted(np.array([g for g, _ in rows]), np.arange(21)).astype(np.int64)
    ginst = np.array([t for _, t in rows], np.int64)
    itype = np.array([0, 1, 0], np.int64)
    nmem = np.array([3, 2, 1], np.int64)
    target = np.array([True, True, False])
    ha = np.array([2, 3, 4, 10, 15, 18, 1], np.int64)
    hb = np.array([3, 4, 4, 10, 15, 19, 2], np.int64)
    lab, match, _ = ls.label_hits(ha, hb, 0, gptr, ginst, itype, nmem, target)
    assert lab.tolist() == [1, 1, 3, 2, 3, 0, 3], lab
    assert match.tolist()[:2] == [0, 0]
    print("label_hits on hand-made instances")


def synthetic_pkg(root, rng, n_genomes=60, contigs=3, genes_per=80, D=32):
    rows, gid = [], 0
    for g in range(n_genomes):
        for c in range(contigs):
            for p in range(genes_per):
                rows.append((gid, f"G{g}", g, f"G{g}_c{c}", g * contigs + c, p))
                gid += 1
    genes = pd.DataFrame(rows, columns=["gid", "genome", "genome_code", "contig", "contig_code", "position_index"])
    N = len(genes)
    n_fam = 400
    tok = rng.integers(0, n_fam, N)
    fam_vec = rng.normal(size=(n_fam + 3, D))
    planted = []
    for g in range(0, n_genomes, 2):                       # locus in even genomes: families 400, 401, 402
        start = g * contigs * genes_per + int(rng.integers(0, genes_per - 5))
        tok[start:start + 3] = [n_fam, n_fam + 1, n_fam + 2]
        planted.append(start)
    # make a third of the planted middle genes token-invisible (new token) but embedding-close
    vec = fam_vec[np.minimum(tok, n_fam + 2)] + 0.6 * rng.normal(size=(N, D))
    for i, s in enumerate(planted):
        if i % 3 == 0:
            tok[s + 1] = n_fam + 3 + i
    X = (vec / np.linalg.norm(vec, axis=1, keepdims=True)).astype(np.float32)
    (root / "universe").mkdir(parents=True)
    genes.assign(strand=1).to_parquet(root / "universe/genes.parquet")
    genes.groupby(["genome", "contig", "contig_code"]).size().rename("n_genes").reset_index().to_parquet(
        root / "universe/contigs.parquet")
    pd.DataFrame(dict(genome=[f"G{g}" for g in range(n_genomes)], genome_code=range(n_genomes))).to_parquet(
        root / "universe/genomes.parquet")
    pd.DataFrame(dict(gid=np.arange(N), token=tok)).to_parquet(root / "tokens.parquet")
    np.save(root / "emb.npy", X)
    return planted


def test_search(rng):
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        planted = synthetic_pkg(root, rng)
        db = ls.Database(root, root / "tokens.parquet", root / "emb.npy", vectors="ram")
        q = np.arange(planted[1], planted[1] + 3)              # a copy whose middle gene keeps its token
        hits = {}
        for method in ("tok", "emb", "comb"):
            h = ls.search(db, q, method=method, W=5, tau=1.0, K=200, R=100, max_E=1e9)
            truth = set(planted) - {planted[1]}
            got = h[(h.evalue <= 0.01)]
            found = sum(any((r.start_gid <= p + 1 <= r.end_gid) for r in got.itertuples()) for p in truth)
            fp = sum(not any(r.start_gid <= p + 1 <= r.end_gid for p in truth) for r in got.itertuples())
            hits[method] = (found, fp, len(truth))
            print(f"  {method}: planted copies found at E<=0.01: {found}/{len(truth)}, other hits at E<=0.01: {fp}")
        # 10 of the 29 copies have a token-invisible middle gene; in this 14,400-gene universe two-slot
        # chance windows are common (about 1.75 per replicate), so token-only finds the 19 intact copies.
        assert hits["tok"][0] >= 19 and hits["emb"][0] >= 25 and hits["comb"][0] >= 25
        assert all(v[1] <= 3 for v in hits.values())
    print("search: planted locus recovered, few null-level hits")


if __name__ == "__main__":
    rng = np.random.default_rng(3)
    test_label()
    test_scan(rng)
    test_topk(rng)
    test_search(rng)
    print("ALL PASSED")
