#!/usr/bin/env python
"""Negative control for the phase-0 headroom: the same reach fractions (f_tok, f_cos, f_rank) for
ordered pairs of instances of DIFFERENT types of the same caller (different genomes), sampled.
A reach criterion that is homology-specific should give same-type fractions far above these.
usage: phase0_control.py --out DIR --threads T [--pairs 200000]"""
import argparse
import json
import math
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench  # noqa: E402
import locus_search as ls  # noqa: E402

G = {}


def work(chunk):
    out = []
    C, tok, top, Xm, row, cstar = G["C"], G["tok"], G["top"], G["Xm"], G["row"], G["cstar"]
    for q, t in chunk:
        mq, mt = C["mem"][q], C["mem"][t]
        ft = float(np.isin(tok[mt], tok[mq]).mean())
        cos = Xm[[row[int(g)] for g in mq]] @ Xm[[row[int(g)] for g in mt]].T
        cv = cos.max(0) >= cstar
        fc = float(cv.mean())
        hit = np.zeros(len(mt), bool)
        for a in mq:
            lst = top[row[int(a)]]
            p = np.searchsorted(lst, mt)
            hit |= (p < len(lst)) & (lst[np.minimum(p, len(lst) - 1)] == mt)
        fr = float(hit.mean())
        out.append((ft, fc, fr, float((cv | hit).mean())))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ls.RD / "locus_search/out")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--pairs", type=int, default=200000)
    a = ap.parse_args()
    db = ls.Database(vectors="mmap")
    callers = bench.load_gold(db)
    d = a.out / "topk"
    gids, idx, s = np.load(d / "gids.npy"), np.load(d / "idx.npy"), np.load(d / "s.npy")
    rs = math.log(1000.0)
    top = [np.sort(idx[k][s[k] >= rs].astype(np.int64)) for k in range(len(gids))]
    cstar = json.load(open(a.out / "phase0_headroom.json"))["cos_threshold_0.999"]
    Xm = np.asarray(db.X[gids], dtype=np.float32)
    G.update(tok=db.tok, top=top, Xm=Xm, row={int(g): k for k, g in enumerate(gids)}, cstar=cstar)
    rng = np.random.default_rng(17)
    res = {}
    for c in bench.CALLERS:
        C = callers[c]
        G["C"] = C
        tr = np.flatnonzero(C["target"])
        q = rng.choice(tr, a.pairs * 3)
        t = rng.choice(tr, a.pairs * 3)
        ok = (C["tcode"][q] != C["tcode"][t]) & (C["gc"][q] != C["gc"][t])
        pairs = list(zip(q[ok][:a.pairs].tolist(), t[ok][:a.pairs].tolist()))
        chunks = [pairs[i:i + 2000] for i in range(0, len(pairs), 2000)]
        with mp.get_context("fork").Pool(a.threads) as pool:
            v = np.array([x for ch in pool.map(work, chunks) for x in ch])
        if not len(v):
            res[c] = dict(pairs=0)
            continue
        ft, fc, fr, fe = v[:, 0], v[:, 1], v[:, 2], v[:, 3]
        res[c] = dict(pairs=int(len(v)), f_tok_mean=float(ft.mean()), tok_visible=float((ft >= 0.5).mean()),
                      f_cos_visible=float((fc >= 0.5).mean()), f_rank_visible=float((fr >= 0.5).mean()),
                      emb_visible=float((fe >= 0.5).mean()),
                      tok_invisible_and_rank_visible=float(((ft < 0.5) & (fr >= 0.5)).mean()),
                      tok_invisible_and_cos_visible=float(((ft < 0.5) & (fc >= 0.5)).mean()),
                      tok_invisible_and_emb_visible=float(((ft < 0.5) & (fe >= 0.5)).mean()),
                      union_visible=float(((ft >= 0.5) | (fe >= 0.5)).mean()))
        print(c, json.dumps(res[c]), flush=True)
    json.dump(res, open(a.out / "phase0_control.json", "w"), indent=1)


if __name__ == "__main__":
    main()
