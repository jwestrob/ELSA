#!/usr/bin/env python
"""Task 6 of round 2: a library of the odd-genome significant token pairs, applied to even genomes.

Library = the 91,522 significant token pairs (rule P <= 1e-4 and L >= 2, w = 3) of the recurrence engine
run on odd genome_code genomes only (round-1 phase 2, job 78273), each with its odd-genome lineage
count L, its odd-genome incidence (genomes where the pair co-occurs within 3 genes) and its odd module.
Application to the even genomes, with no recurrence statistic recomputed:
  1. every pair of even-genome genes <= 3 apart on a contig whose token pair is in the library is an
     instance; each boundary carries the largest library L of the instances spanning it (the engine's
     support landscape, with odd-genome L as the support value);
  2. loci are read exactly as the engine reads them for its all-genome table: conserved runs at
     support >= 2, 5, 12, 30, 80 (gap 0 and gap 2) and profile units (Jaccard of the odd-genome
     incidence >= 0.5); module = plurality odd module of the locus' genes;
  3. the loci are scored with the package scorer (min_members 2, 200 controls, seed 17) and compared,
     instance by instance on even genomes, with the full engine and with the round-1 module library.

usage: pairlib.py --threads T
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import locus_search as ls  # noqa: E402
import phase2  # noqa: E402
from locus_search import log  # noqa: E402

sys.path.insert(0, str(ls.RD / "engine/code"))
import recurrence as rec  # noqa: E402

OUT1 = Path(os.environ.get("LS_OUT1", ls.RD / "locus_search/out"))
OUT2 = Path(os.environ.get("LS_OUT2", ls.RD / "locus_search/out2"))
ENG_ODD = OUT1 / "phase2/engine/odd__mmseqs__seq__e3c50s75__w3"
W = 3


def instances(cc, tok, T, w):
    keys, lefts, dists = [], [], []
    for d in range(1, w + 1):
        left = np.flatnonzero(cc[:-d] == cc[d:])
        ta, tb = tok[left], tok[left + d]
        ok = (ta >= 0) & (tb >= 0) & (ta != tb)
        left, ta, tb = left[ok], ta[ok], tb[ok]
        keys.append(np.minimum(ta, tb) * T + np.maximum(ta, tb))
        lefts.append(left)
        dists.append(np.full(len(left), d, np.int64))
    return np.concatenate(keys), np.concatenate(lefts), np.concatenate(dists)


def contig_bounds(cc):
    ch = np.flatnonzero(np.diff(cc)) + 1
    return np.r_[0, ch].astype(np.int64), np.r_[ch, len(cc)].astype(np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_ON_NODE", "2")))
    a = ap.parse_args()
    import numba
    numba.set_num_threads(a.threads)
    OUT2.mkdir(parents=True, exist_ok=True)
    out = OUT2 / "pairlib"
    out.mkdir(exist_ok=True)
    t0 = time.time()
    db = ls.Database(vectors=None)
    names = pd.read_parquet(ls.PKG / "universe/genes.parquet", columns=["genome", "contig"])
    sig = pd.read_parquet(ENG_ODD / "significant_pairs.parquet")
    tm = pd.read_parquet(ENG_ODD / "token_modules.parquet")
    T = int(db.tok.max()) + 1
    lo = np.minimum(sig.lo.to_numpy(np.int64), sig.hi.to_numpy(np.int64))
    hi = np.maximum(sig.lo.to_numpy(np.int64), sig.hi.to_numpy(np.int64))
    skey = lo * T + hi
    order = np.argsort(skey)
    skey_sorted = skey[order]
    Lval = sig.L.to_numpy(np.float32)
    token_module = np.full(T, -1, np.int64)
    token_module[tm.token.to_numpy()] = tm.module.to_numpy()

    def match(key):
        i = np.searchsorted(skey_sorted, key)
        i = np.minimum(i, len(skey_sorted) - 1)
        hit = skey_sorted[i] == key
        return hit, order[i]                      # sig row (= rank) of each matched instance

    # odd-genome incidence bitsets of the library pairs (for profile units)
    odd = np.flatnonzero(db.gc % 2 == 1)
    k_o, l_o, _ = instances(db.cc[odd], db.tok[odd], T, W)
    hit_o, r_o = match(k_o)
    gidx = np.searchsorted(np.unique(db.gc[odd]), db.gc[odd][l_o[hit_o]])
    n_odd_g = len(np.unique(db.gc[odd]))
    rg = np.unique(r_o[hit_o] * n_odd_g + gidx)
    bits = rec.fill_bitsets(rg // n_odd_g, rg % n_odd_g, len(sig), (n_odd_g + 63) // 64)
    # consistency with the odd engine: genomes per pair equals its G
    G_rec = np.bincount(rg // n_odd_g, minlength=len(sig))
    info = dict(library_pairs=int(len(sig)), odd_genomes=int(n_odd_g),
                odd_G_matches_engine=float((G_rec == sig.G.to_numpy()).mean()))
    log("odd incidence", info, f"{time.time() - t0:.0f}s")

    # even genomes: instances, support landscape, loci
    even = np.flatnonzero(db.gc % 2 == 0)
    cc_e, tok_e = db.cc[even], db.tok[even]
    k_e, l_e, d_e = instances(cc_e, tok_e, T, W)
    hit_e, r_e = match(k_e)
    li, di, ri = l_e[hit_e], d_e[hit_e], r_e[hit_e]
    b, arg = rec.boundary_support_argmax(li, di, Lval[ri], ri, len(even))
    cs, ce = contig_bounds(cc_e)
    gene_module = np.where(tok_e >= 0, token_module[np.maximum(tok_e, 0)], -1).astype(np.int64)
    genome_arr = names.genome.to_numpy()[even]
    contig_arr = names.contig.to_numpy()[even]
    pos = db.pos[even]
    covered = np.zeros(len(even), bool)
    covered[li] = True
    covered[li + di] = True
    info.update(even_genes=int(len(even)), even_instances=int(hit_e.sum()),
                library_pairs_seen_on_even=int(len(np.unique(ri))), even_genes_covered=float(covered.mean()))
    parts = []

    def frame(s_idx, e_idx, kind, level):
        return pd.DataFrame(dict(genome=genome_arr[s_idx], contig=contig_arr[s_idx], start=pos[s_idx], end=pos[e_idx],
                                 kind=kind, level=float(level), module_id=rec.locus_modules(s_idx, e_idx, gene_module)))
    for gap in (0, 2):
        for level in (2, 5, 12, 30, 80):
            s_idx, e_idx = rec.runs_at_least(b, cs, ce, float(level), gap)
            parts.append(frame(s_idx, e_idx, "conserved" if gap == 0 else f"conserved_g{gap}", level))
    s_idx, e_idx = rec.profile_units(arg, bits, cs, ce, 0.5)
    parts.append(frame(s_idx, e_idx, "unit", 0.5))
    loci = pd.concat(parts, ignore_index=True)
    loci = loci[loci.end > loci.start].drop_duplicates(["genome", "contig", "start", "end", "kind"]).reset_index(drop=True)
    loci.to_parquet(out / "pairlib_loci_even.parquet", index=False)
    info["loci"] = {k: int(v) for k, v in loci.groupby("kind").size().items()}
    info["seconds_library_application"] = round(time.time() - t0, 1)
    log("even loci", info)

    # scoring
    scorer = phase2.load_scorer()
    gm = pd.read_parquet(ls.PKG / "universe/genomes.parquet", columns=["genome", "genome_code"])
    gcode_of = dict(zip(gm.genome, gm.genome_code))
    base = pd.read_parquet(phase2.ENGINE_FULL / "score_min2/instance_scores.parquet")
    base["excess"] = base.f1 - base.control_mean_f1
    base = base[base.genome.map(gcode_of) % 2 == 0].set_index("instance_id")
    modlib = pd.read_parquet(OUT1 / "phase2/instance_scores_tok_fill2_v2.parquet")
    if "excess" not in modlib:
        modlib["excess"] = modlib.f1 - modlib.control_mean_f1
    eng_loci = pd.read_parquet(phase2.ENGINE_FULL / "loci_all_genomes.parquet")
    eng_loci = eng_loci[eng_loci.genome.map(gcode_of) % 2 == 0]
    info["engine_full_loci_on_even"] = {k: int(v) for k, v in eng_loci.groupby("kind").size().items()}
    res = dict(info=info, runs={})
    for name, sel in (("all_kinds", np.ones(len(loci), bool)), ("conserved_gap0", (loci.kind == "conserved").to_numpy())):
        t1 = time.time()
        L = loci[sel]
        per, rep, _ = scorer.score(ls.PKG, L[["genome", "contig", "start", "end", "module_id"]], controls=200, seed=17,
                                   threads=a.threads, min_members=2)
        per["excess"] = per.f1 - per.control_mean_f1
        per = per[per.genome.map(gcode_of) % 2 == 0].set_index("instance_id")
        per.to_parquet(out / f"instance_scores_{name}.parquet")
        res["runs"][name] = dict(loci=int(len(L)), span_sum=int((L.end - L.start + 1).sum()),
                                 vs_engine=phase2.paired(per, base, gcode_of),
                                 vs_module_library=phase2.paired(per, modlib, gcode_of),
                                 module_consistency={k: v.get("module_consistency_weighted") for k, v in rep["by_caller"].items()},
                                 scorer_seconds=round(time.time() - t1, 1))
        log(name, json.dumps(res["runs"][name]["vs_engine"], default=float)[:700])
        log(name, json.dumps(res["runs"][name]["vs_module_library"], default=float)[:700])
    res["span_sum_engine_even"] = int((eng_loci.end - eng_loci.start + 1).sum())
    json.dump(res, open(out / "pairlib_summary.json", "w"), indent=1, default=float)
    log(f"done {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
