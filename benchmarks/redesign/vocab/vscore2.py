#!/usr/bin/env python
"""Stage-2 readings and scoring (and the post hoc phrase reading on the stage-1 vocabulary).

V-phr: V-top + phrases. A phrase is a maximal run of consecutive top-level words on a contig (<= 2 genes apart) whose
every gap is spanned by a significant round-0 adjacency: a same-level label pair of two genes within 3 positions with
n >= 2 groups and Lambda >= tau (the learner's own statistic on the original sequence).

usage (biotite, Slurm):
  vscore2.py odd  --src runs2/vt1k_main --tau T [--alphabet vt1k]     odd only; selects V-mob vs V-phr
  vscore2.py even --src runs2/vt1k_main --tau T --primary V-phr        even verdict, typing, vocabulary, level pairs
  vscore2.py odd  --src runs/main --tau 10 --alphabet tokens --tag s1phr   (stage-1 attribution control, post hoc)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

RD = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111")
VD = RD / "vocab"
sys.path.insert(0, str(VD / "code"))
import vscore as S1  # noqa: E402
import vocab as V  # noqa: E402
import vocab2 as V2  # noqa: E402

B, RU = S1.B, S1.RU
log = B.log
CALLERS = S1.CALLERS


class Run2:
    def __init__(self, src, tau):
        d = VD / src / f"tau{tau:g}" / "obs"
        self.d = d
        self.words = pd.read_parquet(d / "words.parquet")
        self.inst = dict(np.load(d / "inst.npz"))
        self.live = dict(np.load(d / "live.npz"))
        self.T = json.load(open(d / "run.json"))["T"]
        top = ~self.live["term"]
        self.top_inst = self.live["ninst"][top]
        self.top_s, self.top_e = self.live["ns"][top], self.live["ne"][top]
        self.top_sym = self.inst["sym"][self.top_inst]     # stage 1: word symbol; stage 2: type id


def significant_boundaries(lab0, cc, gc, grp, tau, w=3, min_n=2):
    """Boundary k (between genes k, k+1) is covered if a significant same-level round-0 adjacency spans it."""
    n, L = lab0.shape
    ng = int(grp.max()) + 1
    n_genomes = len(grp)
    order = np.lexsort((np.arange(n_genomes), grp))
    rank = np.empty(n_genomes, dtype=np.int64)
    rank[order] = np.arange(n_genomes)
    grp_of_rank = grp[order].astype(np.int64)
    lab = lab0.copy()
    S = int(lab0.max()) + 1
    for l in range(L):
        x = lab0[:, l]
        okx = x >= 0
        tg = np.unique(x[okx] * ng + grp[gc[okx]])
        cnt = np.bincount(tg // ng, minlength=S)
        lab[okx & (cnt[np.maximum(x, 0)] < min_n), l] = -1
    ks, ds, keys = [], [], []
    for d in range(1, w + 1):
        base = np.flatnonzero(cc[:-d] == cc[d:])
        for l in range(L):
            a, b = lab[base, l], lab[base + d, l]
            ok = (a >= 0) & (b >= 0) & (a != b)
            ks.append(base[ok])
            ds.append(np.full(int(ok.sum()), d, dtype=np.int64))
            keys.append((np.minimum(a[ok], b[ok]) << V2.SHIFT) | np.maximum(a[ok], b[ok]))
    k, d, key = np.concatenate(ks), np.concatenate(ds), np.concatenate(keys)
    ukey, kinv = np.unique(key, return_inverse=True)
    kg = np.unique(kinv * ng + grp[gc[k]])
    nG = np.bincount(kg // ng, minlength=len(ukey))
    cand = np.flatnonzero(nG >= min_n)
    lo, hi = ukey[cand] >> V2.SHIFT, ukey[cand] & V2.MASK
    sym = lab.ravel()
    gen = np.repeat(rank[gc], L)
    okl = sym >= 0
    sr = np.unique(sym[okl] * n_genomes + gen[okl], return_counts=True)
    s_of, r_of = sr[0] // n_genomes, sr[0] % n_genomes
    indptr = np.searchsorted(s_of, np.arange(S + 1)).astype(np.int64)
    m_rank = np.bincount(rank[gc], minlength=n_genomes).astype(np.float64)
    E = V.expected_groups(lo, hi, indptr, r_of.astype(np.int64), sr[1].astype(np.float64), m_rank, grp_of_rank, float(w))
    nc_ = nG[cand]
    lam = np.where(nc_ > E, nc_ * np.log(nc_ / np.maximum(E, 1e-300)) - nc_ + E, 0.0)
    sig = np.zeros(len(ukey), dtype=bool)
    sig[cand[lam >= tau]] = True
    si = sig[kinv]
    cov = np.zeros(n + 1, dtype=np.int32)
    np.add.at(cov, k[si], 1)
    np.add.at(cov, k[si] + d[si], -1)
    covered = np.cumsum(cov)[:n] > 0          # boundary k covered
    return covered, dict(significant_keys=int(sig.sum()), significant_instances=int(si.sum()),
                         boundaries_covered=int(covered.sum()))


@njit(cache=True)
def phrases(ts, te, tc, covered):
    """Maximal chains of consecutive top words (sorted by start) joined when the gap <= 2 genes and every boundary
    from the end of one word to the start of the next is covered."""
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


def readings2(run, grp, gc, cc, covered):
    rd, info = S1.readings(run, grp, gc)
    o = np.argsort(run.top_s)
    ps, pe = phrases(run.top_s[o], run.top_e[o], cc[run.top_s[o]], covered)
    se = pd.DataFrame(dict(s=np.r_[run.top_s, ps], e=np.r_[run.top_e, pe],
                           m=np.r_[run.top_sym, np.full(len(ps), -1, dtype=np.int64)]))
    rd["V-phr"] = se.drop_duplicates(["s", "e"]).sort_values(["s", "e"])
    words = set(zip(run.top_s.tolist(), run.top_e.tolist()))
    multi = np.array([(x, y) not in words for x, y in zip(ps.tolist(), pe.tolist())])
    info.update(phrases=int(len(ps)), multi_word_phrases=int(multi.sum()),
                multi_word_phrase_span_q=np.quantile((pe - ps + 1)[multi], [.5, .9, .99, 1]).tolist() if multi.any() else [],
                genes_in_multi_word_phrases=int((pe - ps + 1)[multi].sum()))
    return rd, info


def load_ctx(a):
    ctx = S1.Ctx()
    grp = V.load_groups(a.arm, int(ctx.gc.max()) + 1)
    cc, gc, _ = V.load_universe()
    lab = V2.load_labels(a.alphabet, len(cc))
    covered, cinfo = significant_boundaries(lab, cc, gc, grp, a.tau)
    run = Run2(a.src, a.tau)
    rds, info = readings2(run, grp, gc, cc, covered)
    info["round0"] = cinfo
    return ctx, grp, cc, gc, lab, run, rds, info


def cmd_odd(a):
    ctx, grp, cc, gc, lab, run, rds, info = load_ctx(a)
    ref = json.load(open(VD / "results" / "refcheck.json"))
    assert ref["reproduced"]
    dh_odd = ref["DH"]["summary_odd"]
    dh_per = pd.read_parquet(S1.DH_SCORES).set_index("instance_id")
    res = dict(src=a.src, tau=a.tau, alphabet=a.alphabet, info=info, readings={})
    for name, se in rds.items():
        df = ctx.table(se)
        per, summ, sc, rib, loc = S1.score(ctx, df, a.threads)
        pr = RU.paired(per.set_index("instance_id"), dh_per, ctx.gcode, "odd")
        cond = S1.odd_conditions(summ["odd"], rib["odd"], dh_odd)
        res["readings"][name] = dict(summary_odd=summ["odd"], ribo_odd=rib["odd"], loci=loc, vs_DH_odd=pr,
                                     conditions_odd=cond, score=[sum(cond), summ["odd"]["summed_DF_TXSS_excess"]])
        log(name, "odd summed", round(summ["odd"]["summed_DF_TXSS_excess"], 4), "vs DH", [round(x, 4) for x in pr["summed"]],
            {c: round(summ["odd"][c]["f1"], 3) for c in CALLERS}, "ribo odd", rib["odd"]["recovered"],
            rib["odd"]["resolved"], rib["odd"]["parvarchaeaceae"], "cond", cond, "loci", loc)
    cands = {k: v for k, v in res["readings"].items() if k in ("V-mob", "V-phr")}
    best = max(cands, key=lambda k: tuple(cands[k]["score"]))
    res["selected_primary"] = best
    res["written"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    log("selected primary (odd, between V-mob and V-phr):", best, "| info", info)
    S1.jdump(res, VD / "results" / f"odd2_{a.src.replace('/', '_')}_tau{a.tau:g}{('_' + a.tag) if a.tag else ''}.json")


@njit(cache=True)
def inst_shape(ia, ib):
    n = len(ia)
    slots = np.zeros(n, dtype=np.int64)
    depth = np.zeros(n, dtype=np.int64)
    for i in range(n):
        sa = slots[ia[i]] if ia[i] >= 0 else 1
        sb = slots[ib[i]] if ib[i] >= 0 else 1
        da = depth[ia[i]] if ia[i] >= 0 else 0
        db = depth[ib[i]] if ib[i] >= 0 else 0
        slots[i] = sa + sb
        depth[i] = 1 + max(da, db)
    return slots, depth


def vocab_stats2(ctx, run, mem):
    I = run.inst
    slots, depth = inst_shape(I["a"], I["b"])
    w = run.words
    tslots = pd.Series(slots).groupby(I["sym"]).max()
    tdepth = pd.Series(depth).groupby(I["sym"]).max()
    n = len(ctx.gc)
    top_len = run.top_e - run.top_s + 1
    out = dict(word_types=int(len(w)), instances=int(len(I["sym"])), top_instances=int(len(run.top_s)),
               top_types=int(len(np.unique(run.top_sym))),
               types_by_slots={str(k): int(v) for k, v in zip(*np.unique(np.minimum(tslots.to_numpy(), 50), return_counts=True))},
               types_by_depth={str(k): int(v) for k, v in zip(*np.unique(tdepth.to_numpy(), return_counts=True))},
               top_instance_span_q=np.quantile(top_len, [.5, .9, .99, 1]).tolist(),
               genes_in_top=int(top_len.sum()), genes_in_top_frac=float(top_len.sum() / n),
               slot_genes=int(slots[run.top_inst].sum()), lam_q=np.quantile(w.lam, [.01, .5, .99]).tolist(),
               n_groups_q=np.quantile(w.n, [.1, .5, .9, .99]).tolist())
    if "la" in w:
        out["types_by_level_pair"] = {f"{a_}{b_}": int(c) for (a_, b_), c in w.groupby(["la", "lb"]).size().items()}
        # slot radius by gene class: first-level instances (both children terminal), level of each slot
        first = (I["a"] < 0) & (I["b"] < 0)
        tl = w.set_index("type")
        la = tl.la.reindex(I["sym"][first]).to_numpy()
        lb = tl.lb.reindex(I["sym"][first]).to_numpy()
        ga, gb = -I["a"][first] - 1, -I["b"][first] - 1
        # which slot of the key each gene filled is ambiguous when levels differ; report the pair, by gene class of gene a
        cls = np.full(n, "other", dtype=object)
        runs = ctx.runs
        for rs, re_ in zip(runs.gid_start.to_numpy(), runs.gid_end.to_numpy()):
            cls[rs:re_ + 1] = "ribosomal"
        for c in CALLERS:
            cls[mem[mem.caller == c].gid.to_numpy()] = c
        lp = pd.DataFrame(dict(cls=cls[ga], pair=[f"{min(x, y)}{max(x, y)}" for x, y in zip(la, lb)]))
        out["first_level_pairs_by_class"] = {c: g.pair.value_counts(normalize=True).round(4).to_dict()
                                             for c, g in lp.groupby("cls")}
    return out


def cmd_even(a):
    ctx, grp, cc, gc, lab, run, rds, info = load_ctx(a)
    ref = json.load(open(VD / "results" / "refcheck.json"))
    assert ref["reproduced"]
    dh_per = pd.read_parquet(S1.DH_SCORES).set_index("instance_id")
    r1_per = pd.read_parquet(S1.R1_SCORES).set_index("instance_id")
    dh_sc = B.ribo_score(ctx.runs, *[pd.read_parquet(S1.DH_LOCI)[c].to_numpy() for c in ("gid_s", "gid_e")])
    res = dict(src=a.src, tau=a.tau, alphabet=a.alphabet, primary=a.primary, info=info, readings={})
    pers = {}
    tagb = f"{a.src.replace('/', '_')}_tau{a.tau:g}{('_' + a.tag) if a.tag else ''}"
    for name, se in rds.items():
        df = ctx.table(se)
        df.to_parquet(VD / "loci" / f"{tagb}_{name}.parquet", index=False)
        per, summ, sc, rib, loc = S1.score(ctx, df, a.threads)
        per.to_parquet(VD / "scores" / f"{tagb}_{name}.parquet", index=False)
        pers[name] = per
        P = per.set_index("instance_id")
        pr_dh = RU.paired(P, dh_per, ctx.gcode, "even")
        item = dict(summary_even=summ["even"], summary_odd=summ["odd"], ribo=rib, loci=loc, vs_DH_even=pr_dh,
                    vs_R1_even=RU.paired(P, r1_per, ctx.gcode, "even"), ribo_resolved_vs_DH=RU.ribo_paired(sc, dh_sc),
                    verdict_vs_DH=S1.verdict(pr_dh, rib["all"], ref["DH"]["ribo_all"]))
        if a.compare:
            cmp_per = pd.read_parquet(VD / "scores" / f"{a.compare}_{name}.parquet").set_index("instance_id")
            item["vs_compare_even"] = RU.paired(P, cmp_per, ctx.gcode, "even")
        res["readings"][name] = item
        log(name, "EVEN summed", round(summ["even"]["summed_DF_TXSS_excess"], 4), "vs DH", [round(x, 4) for x in pr_dh["summed"]],
            "dF1", {c[:4]: [round(x, 3) for x in pr_dh[c]["d_f1"]] for c in CALLERS}, "ribo", rib["all"]["recovered"],
            rib["all"]["resolved"], rib["all"]["parvarchaeaceae"], rib["all"]["parvarchaeaceae_resolved"],
            "verdict", item["verdict_vs_DH"]["label"],
            "vs compare", ([round(x, 4) for x in item["vs_compare_even"]["summed"]],
                           {c[:4]: [round(x, 3) for x in item["vs_compare_even"][c]["d_f1"]] for c in CALLERS})
            if a.compare else "")
    mem = pd.read_parquet(B.PKG / "gold/members.parquet")
    res["typing"] = S1.typing(ctx, run, pers, a.threads)
    res["vocabulary"] = vocab_stats2(ctx, run, mem)
    # gene status of CRISPR-Cas members (S2-H2): free = not inside any top-level word
    inword = np.zeros(len(ctx.gc), dtype=bool)
    lens = run.top_e - run.top_s + 1
    idx = np.repeat(np.arange(len(run.top_s)), lens)
    off = np.r_[0, np.cumsum(lens)[:-1]]
    inword[run.top_s[idx] + (np.arange(len(idx)) - off[idx])] = True
    k = mem.groupby("instance_id").gid.transform("size")
    mm = mem[k >= 2]
    res["members_outside_words"] = {c: float(1 - inword[g.gid.to_numpy()].mean()) for c, g in mm.groupby("caller")}
    log("members outside words", res["members_outside_words"])
    res["written"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    S1.jdump(res, VD / "results" / f"even2_{tagb}.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["odd", "even"])
    ap.add_argument("--src", required=True)
    ap.add_argument("--arm", default="main")
    ap.add_argument("--alphabet", default="vt1k")
    ap.add_argument("--tau", type=float, required=True)
    ap.add_argument("--primary", default="V-phr")
    ap.add_argument("--compare", default="", help="score-file prefix of another output to pair against (same reading)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    import numba
    numba.set_num_threads(a.threads)
    {"odd": cmd_odd, "even": cmd_even}[a.cmd](a)


if __name__ == "__main__":
    main()
