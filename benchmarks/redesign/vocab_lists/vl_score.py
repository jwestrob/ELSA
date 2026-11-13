#!/usr/bin/env python
"""ws_vocab2b readings, word classes, scoring, verdicts and typing.

usage (biotite, Slurm):
  vl_score.py refcheck                       rescore the stored token DH loci; must reproduce .4270 and the per-instance scores
  vl_score.py odd  --arms ctx,ctxshuf,...    readings V-top / V-nest scored on ODD genomes only; per-arm reading selection
  vl_score.py select                         main list source among {ctx, z10.6, z13.7} by the odd rule (amendment_1)
  vl_score.py even --arms ctx,...            even verdicts, paired differences, ribosomal, word classes, typing
Readings use vscore.py of ws_vocab (read-only import) for scoring, conditions, verdict and typing.
The odd command never writes or prints an even number.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit, prange

RD = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111")
VL = RD / "vocab_lists"
sys.path.insert(0, str(RD / "vocab/code"))
import vscore as VS  # noqa: E402  (read-only import; writes nothing by itself)

B, RU = VS.B, VS.RU
CALLERS = VS.CALLERS
REFS = {"stage1_Vmob": RD / "vocab/scores/main_tau10_V-mob.parquet",
        "softrec_cas_fix_e8": RD / "softrec/cas/scores/fix__gate_p1e-8__DH.parquet",
        "softrec_post": RD / "softrec/cas/scores/post__DH.parquet"}
MAIN_POOL = ("grp", "z10.6", "z13.7")      # amendment_2: grp is the main unit
log = B.log


def jdump(obj, fn):
    tmp = Path(str(fn) + ".tmp")
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x))
    tmp.replace(fn)
    log("wrote", fn)


class Run:
    def __init__(self, arm):
        d = VL / "runs" / arm / "obs"
        self.d, self.arm = d, arm
        z = np.load(d / "live.npz")
        self.live = {k: z[k] for k in z.files}
        self.inst = pd.read_parquet(d / "inst.parquet")
        top = ~self.live["nterm"]
        self.top_s = self.live["ns"][top]
        self.top_e = self.live["ne"][top]
        self.top_inst = self.live["ninst"][top]
        self.top_lslot = self.live["lslot"][top]
        self.info = json.load(open(d / "run.json"))
        self.top_sym = np.arange(len(self.top_s))       # replaced by word classes in the even command


# ------------------------------------------------------------------------------------------------ word classes
@njit(parallel=True, cache=True)
def class_edges(lslot, wptr, ws, we, topid, phase, optr, out_cnt, out_v):
    T = len(lslot)
    for t in prange(T):
        q = lslot[t]
        b0, b1 = wptr[q], wptr[q + 1]
        tot = 0
        for x in range(b0, b1):
            tot += we[x] - ws[x] + 1
        buf = np.empty(max(tot, 1), np.int64)
        cand = np.empty(max(b1 - b0, 1) * 2, np.int64)
        nc = 0
        for x in range(b0, b1):
            s, e = ws[x], we[x]
            L = e - s + 1
            m = 0
            for g in range(s, e + 1):
                if topid[g] >= 0:
                    buf[m] = topid[g]
                    m += 1
            if m == 0:
                continue
            vv = np.sort(buf[:m])
            i = 0
            while i < m:
                j = i
                while j < m and vv[j] == vv[i]:
                    j += 1
                if 2 * (j - i) >= L and vv[i] != t:
                    if nc < len(cand):
                        cand[nc] = vv[i]
                        nc += 1
                i = j
        u = np.unique(cand[:nc]) if nc > 0 else cand[:0]
        if phase == 0:
            out_cnt[t] = len(u)
        else:
            for k in range(len(u)):
                out_v[optr[t] + k] = u[k]


def word_classes(run, n_genes, threads):
    fn = run.d / "classes.npz"
    if fn.exists():
        z = np.load(fn)
        return z["cls"], json.loads(str(z["info"]))
    t0 = time.time()
    tl = np.load(run.d / "toplists.npz")
    T = len(run.top_s)
    topid = np.full(n_genes, -1, np.int64)
    lens = run.top_e - run.top_s + 1
    idx = np.repeat(np.arange(T), lens)
    off = np.r_[0, np.cumsum(lens)[:-1]]
    topid[run.top_s[idx] + (np.arange(len(idx)) - off[idx])] = idx
    cnt = np.zeros(T, np.int64)
    args = (run.top_lslot.astype(np.int64), tl["wptr"].astype(np.int64), tl["ws"].astype(np.int64),
            tl["we"].astype(np.int64), topid)
    class_edges(*args, 0, np.zeros(1, np.int64), cnt, np.zeros(1, np.int64))
    optr = np.r_[0, np.cumsum(cnt)].astype(np.int64)
    dst = np.zeros(optr[-1], np.int64)
    class_edges(*args, 1, optr, cnt, dst)
    src = np.repeat(np.arange(T, dtype=np.int64), cnt)
    key = src * T + dst
    rev = dst * T + src
    ks = np.sort(key)
    pos = np.searchsorted(ks, rev)
    mutual = (pos < len(ks)) & (ks[np.minimum(pos, len(ks) - 1)] == rev)
    a, b = src[mutual], dst[mutual]
    keep = a < b
    a, b = a[keep], b[keep]
    import random
    import igraph as ig
    G = ig.Graph(n=T, edges=np.column_stack([a, b]), directed=False)
    random.seed(17)
    ig.set_random_number_generator(random)
    part = G.community_leiden(objective_function="modularity", resolution=1.0, n_iterations=-1)
    cls = np.asarray(part.membership, np.int64)
    comp = np.asarray(G.connected_components().membership, np.int64)
    sizes = np.bincount(cls)
    info = dict(top_words=int(T), directed_edges=int(len(src)), mutual_edges=int(len(a)),
                words_with_mutual_edge=int(len(np.unique(np.r_[a, b]))), classes=int(len(sizes)),
                classes_ge2=int((sizes >= 2).sum()), class_size_q50_q90_q99_max=np.quantile(sizes, [.5, .9, .99, 1]).tolist(),
                components=int(comp.max() + 1), seconds=round(time.time() - t0, 1))
    np.savez(fn, cls=cls, comp=comp, info=json.dumps(info))
    log("classes", run.arm, info)
    return cls, info


# ------------------------------------------------------------------------------------------------ readings
def readings(run, cls):
    I = run.inst
    ia, ib = I.a.to_numpy(np.int64), I.b.to_numpy(np.int64)
    s_all, e_all = I.s.to_numpy(np.int64), I.e.to_numpy(np.int64)
    ti = run.top_inst
    ch, chc = [], []
    for side in (ia, ib):
        c = side[ti]
        k = c >= 0
        ch.append(c[k])
        chc.append(cls[k])
    ch, chc = np.concatenate(ch), np.concatenate(chc)
    rd = {"V-top": (run.top_s, run.top_e, cls),
          "V-nest": (np.r_[run.top_s, s_all[ch]], np.r_[run.top_e, e_all[ch]], np.r_[cls, chc])}
    out = {}
    for k, (s, e, m) in rd.items():
        out[k] = pd.DataFrame(dict(s=s, e=e, m=m)).drop_duplicates(["s", "e"]).sort_values(["s", "e"])
    return out, dict(nest_children=int(len(ch)))


def ref_check_ok():
    ref = json.load(open(VL / "results" / "refcheck.json"))
    assert ref["reproduced"], "token DH reference not reproduced"
    return ref


# ------------------------------------------------------------------------------------------------ commands
def cmd_refcheck(a):
    ctx = VS.Ctx()
    out = {}
    for name, lp, sp in (("DH", VS.DH_LOCI, VS.DH_SCORES), ("R1", VS.R1_LOCI, VS.R1_SCORES)):
        df = pd.read_parquet(lp)
        per, summ, sc, rib, loc = VS.score(ctx, df, a.threads)
        stored = pd.read_parquet(sp).set_index("instance_id")
        p = per.set_index("instance_id").loc[stored.index]
        same = {c: bool(np.array_equal(p[c].to_numpy(), stored[c].to_numpy())) for c in ("f1", "control_mean_f1", "coverage")}
        out[name] = dict(summary_even=summ["even"], summary_odd=summ["odd"], ribo_all=rib["all"], ribo_odd=rib["odd"],
                         loci=loc, per_instance_equal_stored=same)
        log(name, "summed even", round(summ["even"]["summed_DF_TXSS_excess"], 4),
            {c: round(summ["even"][c]["f1"], 3) for c in CALLERS}, "ribo", rib["all"]["recovered"], rib["all"]["resolved"],
            rib["all"]["parvarchaeaceae"], rib["all"]["parvarchaeaceae_resolved"], "equal stored", same)
    dh = out["DH"]
    R = VS.REF_EVEN
    ok = (round(dh["summary_even"]["summed_DF_TXSS_excess"], 4) == R["summed"]
          and all(round(dh["summary_even"][c]["f1"], 3) == R["f1"][c] for c in CALLERS)
          and dh["ribo_all"]["recovered"] == R["recovered"] and dh["ribo_all"]["resolved"] == R["resolved"]
          and VS.parv(dh["ribo_all"]["parvarchaeaceae_resolved"]) == R["parv_resolved"]
          and all(dh["per_instance_equal_stored"].values()))
    out["reproduced"] = bool(ok)
    log("token DH reproduced exactly:", ok)
    jdump(out, VL / "results" / "refcheck.json")


def cmd_odd(a):
    ctx = VS.Ctx()
    ref = ref_check_ok()
    dh_odd, dh_rib_odd = ref["DH"]["summary_odd"], ref["DH"]["ribo_odd"]
    dh_per = pd.read_parquet(VS.DH_SCORES).set_index("instance_id")
    for arm in [x for x in a.arms.split(",") if x]:
        run = Run(arm)
        rds, info = readings(run, np.arange(len(run.top_s)))
        res = dict(arm=arm, run=run.info["summary"], info=info, DH_odd=dict(summary=dh_odd, ribo=dh_rib_odd), readings={})
        for name, se in rds.items():
            df = ctx.table(se)
            per, summ, sc, rib, loc = VS.score(ctx, df, a.threads)
            pr = RU.paired(per.set_index("instance_id"), dh_per, ctx.gcode, "odd")
            cond = VS.odd_conditions(summ["odd"], rib["odd"], dh_odd)
            res["readings"][name] = dict(summary_odd=summ["odd"], ribo_odd=rib["odd"], vs_DH_odd=pr, conditions_odd=cond,
                                         score=[sum(cond), summ["odd"]["summed_DF_TXSS_excess"]],
                                         loci=dict(loci=loc["loci"]))
            log(arm, name, "ODD summed", round(summ["odd"]["summed_DF_TXSS_excess"], 4), "vs DH odd",
                [round(x, 4) for x in pr["summed"]], {c[:4]: round(summ["odd"][c]["f1"], 3) for c in CALLERS},
                "ribo odd", rib["odd"]["recovered"], rib["odd"]["resolved"], rib["odd"]["parvarchaeaceae"], "cond", cond)
        best = max(res["readings"], key=lambda k: tuple(res["readings"][k]["score"]))
        res["selected_reading"] = best
        res["written"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        log(arm, "selected reading (odd):", best, res["readings"][best]["score"])
        jdump(res, VL / "results" / f"odd_{arm}.json")


def cmd_select(a):
    rows = {}
    for arm in MAIN_POOL:
        fn = VL / "results" / f"odd_{arm}.json"
        if fn.exists():
            r = json.load(open(fn))
            rows[arm] = dict(reading=r["selected_reading"], score=r["readings"][r["selected_reading"]]["score"])
    order = sorted(rows, key=lambda k: (-rows[k]["score"][0], -rows[k]["score"][1], MAIN_POOL.index(k)))
    out = dict(pool=rows, ranking=order, main=(order[0] if order else None),
               rule="number of odd conditions met, then odd summed excess; ties keep grp",
               written=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    log("MAIN ARM (odd selection):", out)
    jdump(out, VL / "results" / "selection.json")


def cmd_even(a):
    ctx = VS.Ctx()
    ref = ref_check_ok()
    dh_per = pd.read_parquet(VS.DH_SCORES).set_index("instance_id")
    r1_per = pd.read_parquet(VS.R1_SCORES).set_index("instance_id")
    refs = {k: pd.read_parquet(v).set_index("instance_id") for k, v in REFS.items() if v.exists()}
    dh_sc = B.ribo_score(ctx.runs, *[pd.read_parquet(VS.DH_LOCI)[c].to_numpy() for c in ("gid_s", "gid_e")])
    (VL / "loci").mkdir(exist_ok=True)
    (VL / "scores").mkdir(exist_ok=True)
    arms = [x for x in a.arms.split(",") if x]
    for arm in arms:
        run = Run(arm)
        odd = json.load(open(VL / "results" / f"odd_{arm}.json"))
        cls, cinfo = word_classes(run, len(ctx.gc), a.threads)
        run.top_sym = cls
        rds, info = readings(run, cls)
        res = dict(arm=arm, run=run.info["summary"], classes=cinfo, primary=odd["selected_reading"], info=info, readings={})
        pers = {}
        for name, se in rds.items():
            df = ctx.table(se)
            df.to_parquet(VL / "loci" / f"{arm}_{name}.parquet", index=False)
            per, summ, sc, rib, loc = VS.score(ctx, df, a.threads)
            per.to_parquet(VL / "scores" / f"{arm}_{name}.parquet", index=False)
            pers[name] = per
            np.save(VL / "scores" / f"{arm}_{name}_ribo_resolved.npy", sc.resolved.to_numpy())
            P = per.set_index("instance_id")
            pr_dh = RU.paired(P, dh_per, ctx.gcode, "even")
            item = dict(summary_even=summ["even"], summary_odd=summ["odd"], ribo=rib, loci=loc, vs_DH_even=pr_dh,
                        vs_R1_even=RU.paired(P, r1_per, ctx.gcode, "even"),
                        ribo_resolved_vs_DH=RU.ribo_paired(sc, dh_sc),
                        verdict_vs_DH=VS.verdict(pr_dh, rib["all"], ref["DH"]["ribo_all"]),
                        vs_refs={k: RU.paired(P, v, ctx.gcode, "even") for k, v in refs.items()})
            res["readings"][name] = item
            log(arm, name, "EVEN summed", round(summ["even"]["summed_DF_TXSS_excess"], 4), "vs DH",
                [round(x, 4) for x in pr_dh["summed"]], "dF1", {c[:4]: [round(x, 3) for x in pr_dh[c]["d_f1"]] for c in CALLERS},
                "ribo", rib["all"]["recovered"], rib["all"]["resolved"], rib["all"]["parvarchaeaceae"],
                rib["all"]["parvarchaeaceae_resolved"], "verdict", item["verdict_vs_DH"]["label"])
        # H7: share of gold member genes (instances with >= 2 members, even genomes) outside every top-level word
        inw = np.zeros(len(ctx.gc) + 1, np.int64)
        np.add.at(inw, run.top_s, 1)
        np.add.at(inw, run.top_e + 1, -1)
        inw = np.cumsum(inw)[:len(ctx.gc)] > 0
        mem = pd.read_parquet(B.PKG / "gold/members.parquet")
        gi = pd.read_parquet(B.PKG / "gold/instances.parquet", columns=["instance_id", "caller", "genome"]).set_index("instance_id")
        mem = mem[mem.instance_id.isin(gi.index)]
        nmem = mem.groupby("instance_id").size()
        mem = mem[nmem.loc[mem.instance_id].to_numpy() >= 2]
        mc = gi.loc[mem.instance_id]
        ev_ = mc.genome.map(ctx.gcode).to_numpy() % 2 == 0
        res["members_outside_words_even"] = {c: float(1 - inw[mem.gid.to_numpy()[ev_ & (mc.caller.to_numpy() == c)]].mean())
                                            for c in CALLERS}
        log(arm, "members outside words (even)", res["members_outside_words_even"])
        try:
            res["typing"] = VS.typing(ctx, run, pers, a.threads)
        except Exception as ex:  # typing is reported separately if it fails
            res["typing_error"] = repr(ex)
            log("typing failed", repr(ex))
        res["written"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        jdump(res, VL / "results" / f"even_{arm}.json")


def cmd_pairs(a):
    """Paired even differences between arms at their selected readings (A - B)."""
    ctx = VS.Ctx()
    out = {}
    for pair in [x for x in a.pairs.split(",") if x]:
        A, Bn = pair.split(":")
        ra = json.load(open(VL / "results" / f"odd_{A}.json"))["selected_reading"]
        rb = json.load(open(VL / "results" / f"odd_{Bn}.json"))["selected_reading"]
        pa = pd.read_parquet(VL / "scores" / f"{A}_{ra}.parquet").set_index("instance_id")
        pb = pd.read_parquet(VL / "scores" / f"{Bn}_{rb}.parquet").set_index("instance_id")
        sa = np.load(VL / "scores" / f"{A}_{ra}_ribo_resolved.npy")
        sb = np.load(VL / "scores" / f"{Bn}_{rb}_ribo_resolved.npy")
        rp = [float(sa.sum() - sb.sum())]
        out[pair] = dict(readings=[ra, rb], paired=RU.paired(pa, pb, ctx.gcode, "even"), ribo_resolved_diff=rp)
        pr = out[pair]["paired"]
        log(pair, "summed", [round(x, 4) for x in pr["summed"]], {c[:4]: [round(x, 3) for x in pr[c]["d_f1"]] for c in CALLERS})
    jdump(out, VL / "results" / f"pairs_{a.tag}.json")


# ------------------------------------------------------------------------------------------------ typing 2: vector classes
RES_GRID = (0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0)     # ws_locusspace map recipe (lspace_map.py), copied
K_NN = 15


@njit(parallel=True, cache=True)
def span_means(X, uq, s, e, out):
    d = X.shape[1]
    for t in prange(len(s)):
        acc = np.zeros(d)
        c = 0
        for g in range(s[t], e[t] + 1):
            u = uq[g]
            if u >= 0:
                for k in range(d):
                    acc[k] += X[u, k]
                c += 1
        nn = 0.0
        for k in range(d):
            nn += acc[k] * acc[k]
        nn = np.sqrt(nn) if nn > 0 else 1.0
        for k in range(d):
            out[t, k] = acc[k] / nn


def knn_flat(V, k, threads):
    import faiss
    faiss.omp_set_num_threads(threads)
    ix = faiss.IndexFlatIP(V.shape[1])
    ix.add(V)
    I = np.empty((V.shape[0], k + 1), np.int64)
    S = np.empty((V.shape[0], k + 1), np.float32)
    for c0 in range(0, V.shape[0], 20_000):
        Ss, Ii = ix.search(V[c0:c0 + 20_000], k + 1)
        I[c0:c0 + 20_000], S[c0:c0 + 20_000] = Ii, Ss
    return I, S


def graph_from_knn(I, S):
    import igraph as ig
    import scipy.sparse as sp
    n = I.shape[0]
    rows, cols, w = [], [], []
    for j in range(I.shape[1]):
        ok = (I[:, j] >= 0) & (I[:, j] != np.arange(n))
        rows.append(np.flatnonzero(ok))
        cols.append(I[ok, j])
        w.append(np.maximum(S[ok, j], 0) + 1e-12)
    r, c, w = np.concatenate(rows), np.concatenate(cols), np.concatenate(w)
    A = sp.csr_matrix((w, (r, c)), shape=(n, n))
    A = A.maximum(A.T).tocoo()
    m = A.row < A.col
    return ig.Graph(n=n, edges=np.c_[A.row[m], A.col[m]].tolist(), edge_attrs={"weight": A.data[m].tolist()})


_LG = {}


def _leiden_task(res):
    """Modularity with resolution (the RB-configuration objective of ws_locusspace's leidenalg call), solved with
    igraph's C implementation (leidenalg took > 8 min per resolution on 0.5 M words; deviation stated in the report)."""
    import random
    import igraph as ig
    random.seed(17)
    ig.set_random_number_generator(random)
    p = _LG["g"].community_leiden(objective_function="modularity", weights="weight", resolution=res, n_iterations=-1)
    return res, np.array(p.membership, np.int64)


def gold_on_words(ctx, run):
    """Each gold instance (>= 2 members) -> the top word holding the plurality of its member genes, >= 50% of them."""
    inst = pd.read_parquet(B.PKG / "gold/instances.parquet", columns=["instance_id", "caller", "system_type",
                                                                       "system_subtype", "genome"])
    mem = pd.read_parquet(B.PKG / "gold/members.parquet")
    n = len(ctx.gc)
    topid = np.full(n, -1, np.int64)
    lens = run.top_e - run.top_s + 1
    idx = np.repeat(np.arange(len(run.top_s)), lens)
    off = np.r_[0, np.cumsum(lens)[:-1]]
    topid[run.top_s[idx] + (np.arange(len(idx)) - off[idx])] = idx
    m = mem.copy()
    m["w"] = topid[m.gid.to_numpy()]
    tot = m.groupby("instance_id").size()
    mm = m[m.w >= 0].groupby(["instance_id", "w"]).size().rename("k").reset_index()
    mm = mm.sort_values(["instance_id", "k", "w"], ascending=[True, False, True]).drop_duplicates("instance_id")
    mm["frac"] = mm.k.to_numpy() / tot.loc[mm.instance_id].to_numpy()
    mm = mm[(mm.frac >= 0.5) & (tot.loc[mm.instance_id].to_numpy() >= 2)]
    g = inst.set_index("instance_id").loc[mm.instance_id]
    out = pd.DataFrame(dict(instance_id=mm.instance_id.to_numpy(), word=mm.w.to_numpy(), caller=g.caller.to_numpy(),
                            genome=g.genome.to_numpy()))
    out["label"] = np.where(out.caller == "CRISPRCasTyper", g.system_subtype.fillna("NA").astype(str).to_numpy(),
                            g.system_type.fillna("NA").astype(str).to_numpy())
    out["genome_code"] = out.genome.map(ctx.gcode).to_numpy()
    gm = pd.read_parquet(RD / "lineagefree/groups/emb_hom_average_t0.75.parquet", columns=["genome_code", "group"])
    out["grp"] = out.genome_code.map(dict(zip(gm.genome_code, gm.group))).to_numpy()
    out["split"] = np.where(out.genome_code % 2 == 0, "even", "odd")
    return out


def label_metrics(y, c):
    from sklearn.metrics import adjusted_mutual_info_score, homogeneity_completeness_v_measure
    h, cm, _ = homogeneity_completeness_v_measure(y, c)
    return dict(n=int(len(y)), n_true=int(len(np.unique(y))), n_pred=int(len(np.unique(c))),
                ami=float(adjusted_mutual_info_score(y, c)), homogeneity=float(h), completeness=float(cm))


def nn_accuracy(Gd, V, caller):
    """Leave-genome-group-out 1-NN label accuracy for even gold instances of a caller (references: gold instances of the
    caller in other groups, any parity), cosine of word vectors; and class-majority accuracy for given class arrays."""
    D = Gd[Gd.caller == caller].reset_index(drop=True)
    W = V[D.word.to_numpy()]
    ev = np.flatnonzero(D.split.eq("even").to_numpy())
    grp = D.grp.to_numpy()
    lab = D.label.to_numpy()
    sim = W[ev] @ W.T
    sim[grp[ev][:, None] == grp[None, :]] = -np.inf
    nn = sim.argmax(1)
    ok = np.isfinite(sim.max(1))
    acc = float((lab[ev][ok] == lab[nn[ok]]).mean()) if ok.any() else None
    return dict(n=int(ok.sum()), accuracy=acc)


def class_majority_accuracy(Gd, cls, caller):
    D = Gd[Gd.caller == caller].reset_index(drop=True)
    c = cls[D.word.to_numpy()]
    grp, lab = D.grp.to_numpy(), D.label.to_numpy()
    ev = np.flatnonzero(D.split.eq("even").to_numpy())
    hit, tot = 0, 0
    for i in ev:
        ref = (c == c[i]) & (grp != grp[i])
        if not ref.any():
            tot += 1
            continue
        vals, cnt = np.unique(lab[ref], return_counts=True)
        hit += int(vals[np.argmax(cnt)] == lab[i])
        tot += 1
    return dict(n=int(tot), accuracy=hit / max(tot, 1))


def cmd_typing2(a):
    """Word classes from mean-pooled zca10 vectors (ws_locusspace recipe) and from mutual listing; typing for both."""
    import multiprocessing as mp
    sys.path.insert(0, str(VL / "code"))
    sys.path.insert(0, str(RD / "softrec/code"))
    import vl as VLm
    ctx = VS.Ctx()
    Vv = VLm.load_vectors()
    for arm in [x for x in a.arms.split(",") if x]:
        t0 = time.time()
        run = Run(arm)
        cls_list, cinfo = word_classes(run, len(ctx.gc), a.threads)
        V = np.zeros((len(run.top_s), Vv.X.shape[1]), np.float32)
        span_means(Vv.X, Vv.uq, run.top_s.astype(np.int64), run.top_e.astype(np.int64), V)
        Gd = gold_on_words(ctx, run)
        I, S = knn_flat(V, K_NN, a.threads)
        log(arm, "15-NN done", f"{time.time() - t0:.0f}s")
        g = graph_from_knn(I, S)
        _LG["g"] = g
        parts = {}
        for r_ in RES_GRID:                       # sequential: no fork after OpenMP initialisation
            parts[r_] = _leiden_task(r_)[1]
            log(arm, "leiden", r_, len(np.unique(parts[r_])), f"{time.time() - t0:.0f}s")
        res = dict(arm=arm, top_words=int(len(run.top_s)), list_classes=cinfo, graph_edges=int(g.ecount()),
                   gold_on_words={f"{c}|{sp}": int(((Gd.caller == c) & (Gd.split == sp)).sum())
                                  for c in CALLERS for sp in ("odd", "even")}, grid={})
        dfo = Gd[(Gd.caller == "DefenseFinder") & (Gd.split == "odd")]
        for r_, memb in sorted(parts.items()):
            res["grid"][str(r_)] = dict(clusters=int(len(np.unique(memb))),
                                        odd_DF_ami=label_metrics(dfo.label.to_numpy(), memb[dfo.word.to_numpy()])["ami"])
        best = max(parts, key=lambda r_: res["grid"][str(r_)]["odd_DF_ami"])
        cls_vec = parts[best]
        res["vector_resolution_odd"] = best
        np.savez(run.d / "vector_classes.npz", cls=cls_vec, V=V.astype(np.float16))
        for caller in ("DefenseFinder", "CRISPRCasTyper"):
            de = Gd[(Gd.caller == caller) & (Gd.split == "even")]
            y = de.label.to_numpy()
            res[caller] = dict(list_classes=label_metrics(y, cls_list[de.word.to_numpy()]),
                               vector_classes=label_metrics(y, cls_vec[de.word.to_numpy()]),
                               nn_word_vector=nn_accuracy(Gd, V, caller),
                               majority_list_classes=class_majority_accuracy(Gd, cls_list, caller),
                               majority_vector_classes=class_majority_accuracy(Gd, cls_vec, caller))
            log(arm, caller, {k: (v.get("ami"), v.get("accuracy")) for k, v in res[caller].items()})
        res["seconds"] = round(time.time() - t0, 1)
        jdump(res, VL / "results" / f"typing2_{arm}.json")


def phrases(run):
    """amendment_3 diagnostic reading: chains of consecutive live nodes joined where the junction gene pair passes."""
    z = np.load(run.d / "junctions.npz")
    lv = run.live
    m = len(lv["ns"])
    parent = np.arange(m)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for j, k in zip(z["j"][z["gene_ok"]], z["k"][z["gene_ok"]]):
        a_, b_ = find(int(j)), find(int(k))
        if a_ != b_:
            parent[max(a_, b_)] = min(a_, b_)
    root = np.array([find(i) for i in range(m)])
    df = pd.DataFrame(dict(root=root, s=lv["ns"], e=lv["ne"], word=~lv["nterm"]))
    g = df.groupby("root").agg(s=("s", "min"), e=("e", "max"), nodes=("s", "size"), words=("word", "sum"))
    g = g[(g.nodes >= 2) & (g.words >= 1)]
    return g.s.to_numpy(np.int64), g.e.to_numpy(np.int64)


def cmd_phr(a):
    ctx = VS.Ctx()
    ref = ref_check_ok()
    dh_per = pd.read_parquet(VS.DH_SCORES).set_index("instance_id")
    for arm in [x for x in a.arms.split(",") if x]:
        run = Run(arm)
        z = np.load(run.d / "classes.npz") if (run.d / "classes.npz").exists() else None
        cls = z["cls"] if z is not None else np.arange(len(run.top_s))
        rds, _ = readings(run, cls)
        ps, pe = phrases(run)
        nest = rds["V-nest"]
        se = pd.concat([nest, pd.DataFrame(dict(s=ps, e=pe, m=-1))]).drop_duplicates(["s", "e"]).sort_values(["s", "e"])
        df = ctx.table(se)
        df.to_parquet(VL / "loci" / f"{arm}_V-phr.parquet", index=False)
        per, summ, sc, rib, loc = VS.score(ctx, df, a.threads)
        per.to_parquet(VL / "scores" / f"{arm}_V-phr.parquet", index=False)
        P = per.set_index("instance_id")
        pr = RU.paired(P, dh_per, ctx.gcode, "even")
        res = dict(arm=arm, status="diagnostic reading (amendment_3); not selectable", phrases=int(len(ps)),
                   phrase_span_q=np.quantile(pe - ps + 1, [.5, .9, .99, 1]).tolist() if len(ps) else [],
                   summary_even=summ["even"], summary_odd=summ["odd"], ribo=rib, loci=loc, vs_DH_even=pr,
                   verdict_vs_DH=VS.verdict(pr, rib["all"], ref["DH"]["ribo_all"]))
        log(arm, "V-phr EVEN summed", round(summ["even"]["summed_DF_TXSS_excess"], 4), [round(x, 4) for x in pr["summed"]],
            {c[:4]: round(summ["even"][c]["f1"], 3) for c in CALLERS}, "ribo", rib["all"]["recovered"],
            rib["all"]["resolved"], rib["all"]["parvarchaeaceae"], rib["all"]["parvarchaeaceae_resolved"])
        jdump(res, VL / "results" / f"phr_{arm}.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["refcheck", "odd", "select", "even", "pairs", "typing2", "phr"])
    ap.add_argument("--arms", default="ctx")
    ap.add_argument("--pairs", default="")
    ap.add_argument("--tag", default="main")
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    import numba
    numba.set_num_threads(a.threads)
    {"refcheck": cmd_refcheck, "odd": cmd_odd, "select": cmd_select, "even": cmd_even, "pairs": cmd_pairs,
     "typing2": cmd_typing2, "phr": cmd_phr}[a.cmd](a)


if __name__ == "__main__":
    main()
