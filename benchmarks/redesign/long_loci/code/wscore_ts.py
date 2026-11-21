#!/usr/bin/env python
"""Readings of a family-word run and source207 scoring (workstream transfer_spacedust).

Readings (vocab/vscore.py, vscore2.py, vscore2a.py, copied):
  V-top    top-level word instances
  V-nest   V-top + their direct children that are words
  V-mob    V-top + maximal sub-instances whose type is a top-level type in >= 2 units (groups in group mode, genomes else)
  V-phr    V-top + phrases joined where every gap boundary is covered by the run's own round-0 significant occurrences
  V-coh<t> V-top + phrases of consecutive top words (<= 2 genes apart) joined where every boundary has token cohesion
           kappa >= t or NaN (kappa = this collection's ws_context token cohesion, emb/kappa_var_tokens.npy)
  CC1/CC2  connected-component baseline (amendment_0): maximal runs of boundaries covered by round-0 significant
           occurrences at distance <= 2 / <= 3 (no merging rounds)
Intervals with one gene are dropped (as vscore Ctx.table). Scored on query-genome intervals with s207.Bench.

usage: wscore.py --spec syn --tau 8 --primary V-coh0.1 --tag P2 [--compare P1:V-coh0.1,...] --threads T
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s207  # noqa: E402

RD, TS = s207.RD, s207.TS
log = s207.log
COH = (0.02, 0.05, 0.1, 0.2)
KAPPA = TS / "emb/kappa_var_tokens.npy"


@njit(cache=True)
def mobile_cores(top_inst, ia, ib, isym, mobile):
    out = np.empty(len(ia), dtype=np.int64)
    n = 0
    stack = np.empty(len(ia) + 2, dtype=np.int64)
    for t in range(len(top_inst)):
        sp = 0
        r = top_inst[t]
        if ia[r] >= 0:
            stack[sp] = ia[r]
            sp += 1
        if ib[r] >= 0:
            stack[sp] = ib[r]
            sp += 1
        while sp > 0:
            sp -= 1
            x = stack[sp]
            if mobile[isym[x]]:
                out[n] = x
                n += 1
            else:
                if ia[x] >= 0:
                    stack[sp] = ia[x]
                    sp += 1
                if ib[x] >= 0:
                    stack[sp] = ib[x]
                    sp += 1
    return out[:n]


@njit(cache=True)
def phrases_cov(ts, te, tc, covered):
    n = len(ts)
    out_s = np.empty(n, dtype=np.int64)
    out_e = np.empty(n, dtype=np.int64)
    m = 0
    i = 0
    while i < n:
        s, e = ts[i], te[i]
        j = i + 1
        while j < n and tc[j] == tc[i] and ts[j] - e - 1 <= 2:
            ok = True
            for k in range(e, ts[j]):
                if not covered[k]:
                    ok = False
                    break
            if not ok:
                break
            e = te[j]
            j += 1
        out_s[m] = s
        out_e[m] = e
        m += 1
        i = j
    return out_s[:m], out_e[:m]


@njit(cache=True)
def phrases_coh(ts, te, tc, kappa, theta):
    n = len(ts)
    out_s = np.empty(n, dtype=np.int64)
    out_e = np.empty(n, dtype=np.int64)
    m = 0
    i = 0
    while i < n:
        s, e = ts[i], te[i]
        j = i + 1
        while j < n and tc[j] == tc[i] and ts[j] - e - 1 <= 2:
            ok = True
            for k in range(e, ts[j]):
                if kappa[k] < theta:
                    ok = False
                    break
            if not ok:
                break
            e = te[j]
            j += 1
        out_s[m] = s
        out_e[m] = e
        m += 1
        i = j
    return out_s[:m], out_e[:m]


@njit(cache=True)
def type_size_depth(T, wsym, wa, wb, S):
    size = np.zeros(S, dtype=np.int64)
    depth = np.zeros(S, dtype=np.int64)
    size[:T] = 1
    for i in range(len(wsym)):
        x, y = wa[i], wb[i]
        size[wsym[i]] = size[x] + size[y]
        depth[wsym[i]] = 1 + max(depth[x], depth[y])
    return size, depth


def covered_runs(k, d, n):
    """Maximal runs of boundaries covered by occurrences (k, k+d) -> gene intervals [s, e]."""
    cov = np.zeros(n + 1, dtype=np.int32)
    np.add.at(cov, k, 1)
    np.add.at(cov, k + d, -1)
    b = np.cumsum(cov)[:n] > 0
    x = np.r_[False, b, False].astype(np.int8)
    st = np.flatnonzero(np.diff(x) == 1)
    en = np.flatnonzero(np.diff(x) == -1)
    return st, en          # boundaries st..en-1 covered -> genes st..en


class Run:
    def __init__(self, d):
        self.d = Path(d)
        self.meta = json.load(open(self.d / "run.json"))
        self.words = pd.read_parquet(self.d / "words.parquet")
        self.inst = dict(np.load(self.d / "inst.npz"))
        self.live = dict(np.load(self.d / "live.npz"))
        self.b0 = np.load(self.d / "b0.npy")
        self.sig0 = dict(np.load(self.d / "sig0.npz"))
        self.T = self.meta["T"]
        top = ~self.live["term"]
        self.top_inst = self.live["ninst"][top]
        self.top_s, self.top_e = self.live["ns"][top], self.live["ne"][top]
        self.top_sym = self.live["sym"][top]


def readings(run, unit_of_genome, gc, cc, n):
    I = run.inst
    rd = {"V-top": (run.top_s, run.top_e)}
    ch = np.r_[I["a"][run.top_inst], I["b"][run.top_inst]]
    ch = ch[ch >= 0]
    rd["V-nest"] = (np.r_[run.top_s, I["s"][ch]], np.r_[run.top_e, I["e"][ch]])
    lab = I["label"]
    S = int(max(lab.max() if len(lab) else 0, run.top_sym.max() if len(run.top_sym) else 0, run.T)) + 1
    nu = int(unit_of_genome.max()) + 1
    tg = np.unique(run.top_sym * nu + unit_of_genome[gc[run.top_s]])
    mobile = np.bincount(tg // nu, minlength=S) >= 2
    mc = mobile_cores(run.top_inst.astype(np.int64), I["a"], I["b"], lab, mobile)
    rd["V-mob"] = (np.r_[run.top_s, I["s"][mc]], np.r_[run.top_e, I["e"][mc]])
    o = np.argsort(run.top_s)
    ts, te, tcc = run.top_s[o], run.top_e[o], cc[run.top_s[o]]
    ps, pe = phrases_cov(ts, te, tcc, run.b0)
    rd["V-phr"] = (np.r_[run.top_s, ps], np.r_[run.top_e, pe])
    words = set(zip(run.top_s.tolist(), run.top_e.tolist()))
    info = dict(top=int(len(run.top_s)), mobile_types=int(mobile.sum()), mobile_cores=int(len(mc)), nest_children=int(len(ch)))
    pinfo = {}
    multi = np.array([(x, y) not in words for x, y in zip(ps.tolist(), pe.tolist())])
    pinfo["V-phr"] = dict(multi=int(multi.sum()), span_q50_q90_max=np.quantile((pe - ps + 1)[multi], [.5, .9, 1]).tolist() if multi.any() else [])
    kappa = np.load(KAPPA).astype(np.float64)
    assert len(kappa) == n
    for th in COH:
        cs_, ce_ = phrases_coh(ts, te, tcc, kappa, th)
        rd[f"V-coh{th:g}"] = (np.r_[run.top_s, cs_], np.r_[run.top_e, ce_])
        mm = np.array([(x, y) not in words for x, y in zip(cs_.tolist(), ce_.tolist())])
        pinfo[f"V-coh{th:g}"] = dict(multi=int(mm.sum()), genes_in_multi=int((ce_ - cs_ + 1)[mm].sum()) if mm.any() else 0,
                                     span_q50_q90_max=np.quantile((ce_ - cs_ + 1)[mm], [.5, .9, 1]).tolist() if mm.any() else [])
    k, d = run.sig0["k"], run.sig0["d"]
    ccs = [("CC1", 2), ("CC2", 3)]
    if len(d) and int(d.max()) > 3:          # wider-window development runs: also the run's own full window
        ccs.append((f"CC{int(d.max()) - 1}", int(d.max())))
    for nm, dmax in ccs:
        m = d <= dmax
        st, en = covered_runs(k[m], d[m], n)
        rd[nm] = (st, en)
        pinfo[nm] = dict(loci=int(len(st)), span_q50_q90_max=np.quantile(en - st + 1, [.5, .9, 1]).tolist() if len(st) else [])
    info["phrases"] = pinfo
    out = {}
    for name, (s, e) in rd.items():
        df = pd.DataFrame(dict(s=s, e=e)).drop_duplicates().sort_values(["s", "e"])
        out[name] = df[df.e > df.s]
    return out, info


def locus_stats(df, n, qmask):
    s, e = df.s.to_numpy(), df.e.to_numpy()
    cov = np.zeros(n + 1, np.int32)
    np.add.at(cov, s, 1)
    np.add.at(cov, e + 1, -1)
    inl = np.cumsum(cov)[:n] > 0
    sp = e - s + 1
    q = qmask[s]
    return dict(loci=int(len(s)), genes_in_loci_frac=float(inl.mean()), query_genes_in_loci_frac=float(inl[qmask].mean()),
                span_q10_q50_q90_q99_max=np.quantile(sp, [.1, .5, .9, .99, 1]).tolist() if len(sp) else [],
                query_loci=int(q.sum()),
                query_span_q10_q50_q90_q99_max=np.quantile(sp[q], [.1, .5, .9, .99, 1]).tolist() if q.any() else [])


def vocab_stats(run, n, qmask):
    w = run.words.sort_values("sym")
    S = int(w.sym.max()) + 1 if len(w) else run.T
    size, depth = type_size_depth(run.T, w.sym.to_numpy(np.int64), w.a.to_numpy(np.int64), w.b.to_numpy(np.int64), S)
    ws, wd = size[w.sym.to_numpy()], depth[w.sym.to_numpy()]
    top_len = run.top_e - run.top_s + 1
    top_slots = size[run.top_sym]
    q = qmask[run.top_s]
    return dict(word_types=int(len(w)), instances=int(len(run.inst["sym"])), top_instances=int(len(run.top_s)),
                top_types=int(len(np.unique(run.top_sym))),
                types_by_slots={str(k): int(v) for k, v in zip(*np.unique(np.minimum(ws, 50), return_counts=True))},
                types_by_depth={str(k): int(v) for k, v in zip(*np.unique(wd, return_counts=True))},
                max_slots=int(ws.max()) if len(ws) else 0, max_depth=int(wd.max()) if len(wd) else 0,
                top_span_q10_q25_q50_q75_q90_q99_max=np.quantile(top_len, [.1, .25, .5, .75, .9, .99, 1]).tolist(),
                top_span_query_q50_q90_q99_max=np.quantile(top_len[q], [.5, .9, .99, 1]).tolist() if q.any() else [],
                query_top_instances=int(q.sum()),
                genes_in_top=int(top_len.sum()), genes_in_top_frac=float(top_len.sum() / n),
                slot_genes=int(top_slots.sum()), inserted_genes=int(top_len.sum() - top_slots.sum()),
                lam_q01_q10_q50_q90_q99=np.quantile(w.lam, [.01, .1, .5, .9, .99]).tolist() if len(w) else [],
                n_units_q10_q50_q90_q99=np.quantile(w.n, [.1, .5, .9, .99]).tolist() if len(w) else [],
                rounds=len(run.meta["rounds"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--tau", type=float, required=True)
    ap.add_argument("--root", default=str(TS / "runs"))
    ap.add_argument("--primary", default="V-coh0.1")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--groups", default=str(TS / "groups/emb_hom_average_t0.75.parquet"))
    ap.add_argument("--compare", default="")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    import numba
    numba.set_num_threads(a.threads)
    t0 = time.time()
    g = s207.load_universe_genes()
    n = len(g)
    gc = g.genome_code.to_numpy(np.int64)
    cc = g.contig_code.to_numpy(np.int64)
    qmask = (g.side == "query").to_numpy()
    run = Run(Path(a.root) / a.spec / f"tau{a.tau:g}" / "obs")
    ng = int(gc.max()) + 1
    if a.spec == "group":
        m = pd.read_parquet(a.groups).sort_values("genome_code")
        unit = pd.factorize(m.group.to_numpy())[0].astype(np.int64)
    else:
        unit = np.arange(ng, dtype=np.int64)
    rds, info = readings(run, unit, gc, cc, n)
    B = s207.Bench(a.threads)
    refs = {k: pd.read_parquet(TS / f"results/pr_{k}.parquet") for k in ("engine_ref", "engine_single", "spacedust")}
    refs.update({c: pd.read_parquet(TS / f"results/pr_{c.replace(':', '__')}.parquet")
                 for c in a.compare.split(",") if c and (TS / f"results/pr_{c.replace(':', '__')}.parquet").exists()})
    res = dict(spec=a.spec, tau=a.tau, tag=a.tag, primary=a.primary, run_meta=dict(rounds=len(run.meta["rounds"]),
               summary=run.meta["summary"]), info=info, vocab=vocab_stats(run, n, qmask), readings={})
    # gold member genes outside every top-level word
    regions = B.regions
    qg = g[qmask]
    key = dict(zip(zip(qg.genome, qg.contig, qg.position_index), qg.gid))
    mem = [key[(r.sample_id, r.contig_id, p)] for r in regions.itertuples() for p in json.loads(r.member_position_indices_json)]
    inword = np.zeros(n, bool)
    cov = np.zeros(n + 1, np.int32)
    np.add.at(cov, run.top_s, 1)
    np.add.at(cov, run.top_e + 1, -1)
    inword = np.cumsum(cov)[:n] > 0
    res["gold_member_genes_outside_words"] = float(1 - inword[np.asarray(mem)].mean())
    for name, df in rds.items():
        spans = s207.gene_spans(g, df.s.to_numpy(), df.e.to_numpy())
        r, pr = B.score(spans, f"{a.tag} {name}", np.random.default_rng(17))
        pr.to_parquet(TS / f"results/pr_{a.tag}__{name}.parquet", index=False)
        item = dict(score=r, loci=locus_stats(df, n, qmask))
        for rk, rp in refs.items():
            p = s207.paired(pr, rp)
            item[f"vs_{rk}"] = dict(paired=p, verdict=s207.verdict(p))
        res["readings"][name] = item
        log(a.tag, name, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()
                          if k in ("n_intervals", "f1_observed", "f1_random", "f1_excess", "ci_lo", "ci_hi",
                                   "median_boundary_precision", "median_coverage", "query_gene_coverage")},
            {k: [round(x, 4) for x in v["paired"]["excess"]] + [v["verdict"]] for k, v in item.items() if k.startswith("vs_")})
    res["seconds"] = round(time.time() - t0, 1)
    res["written"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (TS / f"results/score_{a.tag}.json").write_text(json.dumps(res, indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x)) + "\n")
    log("wrote score", a.tag, f"{time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
