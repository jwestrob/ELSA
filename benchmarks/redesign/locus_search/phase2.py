#!/usr/bin/env python
"""Phase 2: a profile library built on odd genomes annotates even genomes without recomputing recurrence.

  phase2.py prep --out DIR        odd-genome universe + tokens for the recurrence engine (reindexed)
  (the sbatch wrapper then runs engine/code/recurrence.py on it: modules from odd genomes only)
  phase2.py scan --out DIR        one profile per odd-genome module, scanned over even genomes;
                                  loci scored with the package scorer against the engine's own loci

Profile of module m: one slot per token of m; slot evidence = the token (members in even genomes)
and, for the combined profile, the centroid of up to 200 odd-genome genes carrying the token
(top-K cosine among even-genome genes). Window W_m = 75th percentile of the span of the module's
conserved loci on odd genomes (2 <= W_m <= 60). Database = every gene of every even genome, in a
compact index (the circular-shift null runs over even genomes only).
"""
from __future__ import annotations

import argparse

import json
import math
import multiprocessing as mp
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import locus_search as ls  # noqa: E402
from locus_search import log  # noqa: E402

CTX = {}
ARM = "odd__mmseqs__seq__e3c50s75"
ENGINE_FULL = Path(__import__("os").environ.get("LS_ENGINE_FULL",          # override for synthetic tests only
                                               ls.RD / "engine/results_dpann_v5/dpann__mmseqs__seq__e3c50s75__w3"))


def prep(a):
    out = a.out / "phase2/odd_universe"
    out.mkdir(parents=True, exist_ok=True)
    g = pd.read_parquet(ls.PKG / "universe/genes.parquet")
    gm = pd.read_parquet(ls.PKG / "universe/genomes.parquet")
    odd = g[g.genome_code % 2 == 1].copy()
    gm_odd = gm[gm.genome_code % 2 == 1].sort_values("genome_code").copy()
    gmap = dict(zip(gm_odd.genome_code, range(len(gm_odd))))
    cmap = {c: i for i, c in enumerate(np.unique(odd.contig_code))}
    odd["orig_gid"] = odd.gid
    odd = odd.sort_values("gid")
    odd["gid"] = np.arange(len(odd))
    odd["genome_code"] = odd.genome_code.map(gmap).astype(np.int32)
    odd["contig_code"] = odd.contig_code.map(cmap).astype(np.int64)
    gm_odd["genome_code"] = gm_odd.genome_code.map(gmap).astype(np.int32)
    odd.to_parquet(out / "genes.parquet", index=False)
    gm_odd.to_parquet(out / "genomes.parquet", index=False)
    t = pd.read_parquet(ls.TOKENS)
    tok = np.full(len(g), -1, np.int64)
    tok[t.gid.to_numpy()] = t.token.to_numpy()
    pd.DataFrame(dict(gid=odd.gid.to_numpy(), token=tok[odd.orig_gid.to_numpy()])).to_parquet(out / "tokens.parquet", index=False)
    log(f"odd universe: {len(odd):,} genes, {len(gm_odd):,} genomes, {len(cmap):,} contigs -> {out}")


def module_profiles(db, engine_dir, max_genes=200, seed=17):
    tm = pd.read_parquet(engine_dir / "token_modules.parquet")
    loci = pd.read_parquet(engine_dir / "loci_all_genomes.parquet")
    c = loci[(loci.kind == "conserved") & (loci.module_id >= 0)]
    span = (c.end - c.start + 1).groupby(c.module_id).quantile(0.75)
    rng = np.random.default_rng(seed)
    odd = db.gc % 2 == 1
    prof = {}
    for m, T in tm.groupby("module"):
        toks = np.sort(T.token.to_numpy())
        W = int(min(60, max(2, math.ceil(span.get(m, 3)))))
        prof[int(m)] = dict(tokens=toks, W=W)
    # centroid of each module token from odd-genome genes
    all_tok = np.unique(tm.token.to_numpy())
    cent = np.zeros((len(all_tok), db.X.shape[1]), np.float32)
    n_used = np.zeros(len(all_tok), np.int64)
    for i, t in enumerate(all_tok):
        mem = db.tok_order[db.tok_ptr[t]:db.tok_ptr[t + 1]]
        mem = mem[odd[mem]]
        if len(mem) > max_genes:
            mem = np.sort(rng.choice(mem, max_genes, replace=False))
        if len(mem):
            v = db.X[mem].mean(0)
            cent[i] = v / np.linalg.norm(v)
            n_used[i] = len(mem)
    return prof, all_tok, cent, n_used


def scan_module(m):
    e = CTX["even"]
    p = CTX["prof"][m]
    lists = []
    for t in p["tokens"]:
        mem = e.tok_order[e.tok_ptr[t]:e.tok_ptr[t + 1]] if t < len(e.tok_ptr) - 1 else np.empty(0, np.int64)
        st = np.full(len(mem), math.log(e.N / len(mem)) if len(mem) else 0.0)
        if CTX["mode"] == "tok":
            lists.append((mem, st))
            continue
        k = CTX["tok_row"].get(int(t), -1)
        if k < 0:
            lists.append((mem, st))
            continue
        ge, se = CTX["c_idx"][k].astype(np.int64), CTX["c_s"][k].astype(np.float64) * CTX["w"]
        ok = ge >= 0
        g = np.concatenate([mem, ge[ok]])
        s = np.concatenate([st, se[ok]])
        o = np.lexsort((-s, g))
        g, s = g[o], s[o]
        first = np.r_[True, g[1:] != g[:-1]] if len(g) else np.zeros(0, bool)
        lists.append((g[first], s[first]))
    t0 = time.time()
    eg, es_, ev = ls.assemble(lists)
    ha, hb, sc, nf = ls.scan(eg, es_, ev, len(lists), e.cc, e.pos, p["W"], CTX["tau"], 0)
    tail = ls.null_tail(eg, es_, ev, len(lists), e.cc, e.pos, e.N, 0, 0, p["W"], CTX["tau"], CTX["R"], 17 + m, 64)
    E, _ = ls.evalues(sc, tail, CTX["R"])
    keep = (E <= 1.0) | (nf >= 2)          # library call: >= 2 slots of the module co-located, any E
    return dict(module=m, ha=ha[keep], hb=hb[keep], score=sc[keep], E=E[keep], nfill=nf[keep],
                n_entries=int(len(eg)), n_slots=len(lists), W=p["W"], seconds=time.time() - t0)


def load_scorer():
    # imported by name so that its process pool can pickle score_contig
    sys.path.insert(0, str(ls.PKG / "scorer"))
    import score_dpann_loci
    return score_dpann_loci


def paired(A, B, gcode_of, rng_seed=17):
    ids = A.index.intersection(B.index)
    rows = []
    for c in ("DefenseFinder", "TXSScan", "CRISPRCasTyper"):
        k = ids[(A.loc[ids, "caller"] == c).to_numpy()]
        if not len(k):
            continue
        g = A.loc[k, "genome"].map(gcode_of).to_numpy()
        d = {}
        for col in ("f1", "excess", "covered50", "localized"):
            x = A.loc[k, col].astype(float).to_numpy()
            y = B.loc[k, col].astype(float).to_numpy()
            d[col] = dict(A=float(x.mean()), B=float(y.mean()), diff=ls.boot_mean(x - y, g, seed=rng_seed))
        rows.append(dict(caller=c, instances=int(len(k)), **d))
    return rows


def scan(a):
    out = a.out / "phase2"
    t0 = time.time()
    db = ls.Database(vectors="ram")
    eng = out / "engine" / f"{ARM}__w3"
    prof, all_tok, cent, n_used = module_profiles(db, eng)
    log(f"profiles: {len(prof):,} modules, {len(all_tok):,} tokens, slots per module median "
        f"{np.median([len(p['tokens']) for p in prof.values()]):.0f}, max {max(len(p['tokens']) for p in prof.values())}; "
        f"W median {np.median([p['W'] for p in prof.values()]):.0f}; {time.time() - t0:.0f}s")
    even_g = np.flatnonzero(db.gc % 2 == 0)
    tok_e = db.tok[even_g]
    order = np.argsort(tok_e, kind="stable")
    order = order[tok_e[order] >= 0]
    e = SimpleNamespace(N=len(even_g), cc=db.cc[even_g], pos=db.pos[even_g], tok_order=order.astype(np.int64),
                        tok_ptr=np.searchsorted(tok_e[order], np.arange(db.n_tokens + 1)).astype(np.int64))
    best = json.load(open(a.out / "best_configs.json"))["DefenseFinder"]["M2"]
    CTX.update(even=e, prof=prof, tau=float(best["tau"]), w=float(best["w"]), R=a.R)
    # combined profiles: centroid top-K among even-genome genes
    t1 = time.time()
    Xe = db.X[even_g]
    shim = SimpleNamespace(X=Xe, g_lo=np.zeros(1, np.int64), g_hi=np.zeros(1, np.int64))
    ok = n_used > 0
    c_idx = np.full((len(all_tok), a.K), -1, np.int32)
    c_s = np.zeros((len(all_tok), a.K), np.float32)
    ci, _, cs = ls.topk_cosine(shim, cent[ok], np.zeros(int(ok.sum()), np.int64), K=a.K)
    c_idx[ok], c_s[ok] = ci, cs
    del Xe
    secs_topk = time.time() - t1
    log(f"centroid top-K: {int(ok.sum()):,} tokens, K={a.K}, {secs_topk:.0f}s")
    CTX.update(tok_row={int(t): i for i, t in enumerate(all_tok)}, c_idx=c_idx, c_s=c_s)
    import numba
    numba.set_num_threads(a.threads)
    z = np.array([0, 1], np.int64)
    ls.scan(z, np.zeros(2, np.int64), np.ones(2), 1, e.cc, e.pos, 3, 0.0, 0)
    ls.null_tail(z, np.zeros(2, np.int64), np.ones(2), 1, e.cc, e.pos, e.N, 0, 0, 3, 0.0, 2, 1, 4)
    scorer = load_scorer()
    gm = pd.read_parquet(ls.PKG / "universe/genomes.parquet", columns=["genome", "genome_code"])
    gcode_of = dict(zip(gm.genome, gm.genome_code))
    base = pd.read_parquet(ENGINE_FULL / "score_min2/instance_scores.parquet")
    base["excess"] = base.f1 - base.control_mean_f1
    base = base[base.genome.map(gcode_of) % 2 == 0].set_index("instance_id")
    summary = dict(modules=len(prof), tokens=int(len(all_tok)), tokens_with_centroid=int(ok.sum()), K=a.K, R=a.R,
                   tau=CTX["tau"], w=CTX["w"], even_genomes=int(len(np.unique(db.gc[even_g]))), even_genes=int(e.N),
                   centroid_topk_seconds=round(secs_topk, 1), engine_odd=json.load(open(eng / "summary.json")), runs={})
    for mode in ("tok", "comb"):
        CTX["mode"] = mode
        t2 = time.time()
        with mp.get_context("fork").Pool(a.threads) as pool:
            res = list(pool.imap_unordered(scan_module, sorted(prof, key=lambda m: -len(prof[m]["tokens"])), chunksize=1))
        wall = time.time() - t2
        frames = []
        for r in res:
            if len(r["ha"]):
                frames.append(pd.DataFrame(dict(module_id=r["module"], gid_a=even_g[r["ha"]], gid_b=even_g[r["hb"]],
                                                score=r["score"], evalue=r["E"], n_slots_filled=r["nfill"])))
        H = pd.concat(frames, ignore_index=True)
        c = db.contigs.loc[db.cc[H.gid_a.to_numpy()]]
        H["genome"], H["contig"] = c.genome.to_numpy(), c.contig.to_numpy()
        H["start"], H["end"] = db.pos[H.gid_a.to_numpy()], db.pos[H.gid_b.to_numpy()]
        H.to_parquet(out / f"profile_hits_{mode}{a.tag}.parquet", index=False)
        run = dict(wall_seconds=round(wall, 1), cpu_seconds=round(sum(r["seconds"] for r in res), 1),
                   even_genomes_per_wall_second=summary["even_genomes"] / wall, hits_saved=int(len(H)))
        sels = {"E0.01": H.evalue <= 0.01, "E1": H.evalue <= 1.0, "fill2": H.n_slots_filled >= 2,
                "E1_or_fill2": (H.evalue <= 1.0) | (H.n_slots_filled >= 2)}
        for name, sel in sels.items():
            L = H[sel][["genome", "contig", "start", "end", "module_id"]].drop_duplicates()
            per, rep, _ = scorer.score(ls.PKG, L, controls=200, seed=17, threads=a.threads, min_members=2)
            per["excess"] = per.f1 - per.control_mean_f1
            per = per[per.genome.map(gcode_of) % 2 == 0].set_index("instance_id")
            per.to_parquet(out / f"instance_scores_{mode}_{name}{a.tag}.parquet")
            run[name] = dict(loci=int(len(L)), span_sum=int(sum(L.end - L.start + 1)),
                             paired_vs_engine=paired(per, base, gcode_of),
                             module_consistency={k: v.get("module_consistency_weighted") for k, v in rep["by_caller"].items()})
            log(mode, name, json.dumps(run[name]["paired_vs_engine"], default=float)[:600])
        summary["runs"][mode] = run
    eng_loci = pd.read_parquet(ENGINE_FULL / "loci_all_genomes.parquet")
    eng_loci = eng_loci[eng_loci.genome.map(gcode_of) % 2 == 0]
    summary["engine_full_loci_on_even"] = int(len(eng_loci))
    json.dump(summary, open(out / f"phase2_summary{a.tag}.json", "w"), indent=1, default=float)
    log(f"phase 2 done {time.time() - t0:.0f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["prep", "scan"])
    ap.add_argument("--out", type=Path, default=ls.RD / "locus_search/out")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--K", type=int, default=2000)
    ap.add_argument("--R", type=int, default=100)
    ap.add_argument("--tag", default="", help="suffix of the summary file")
    a = ap.parse_args()
    prep(a) if a.step == "prep" else scan(a)


if __name__ == "__main__":
    main()
