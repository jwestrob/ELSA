#!/usr/bin/env python
"""DPANN scoring for long_loci outputs (scorer, ribosomal check and paired bootstrap of vocab/vscore.py, copied as
vscore_v.py; blib / rules / ribosomal_check imported read-only) and the label-free ribosomal benchmark diagnosis.

usage (Slurm)
  dscore.py refcheck                         token DH rescored: must equal the stored reference exactly
  dscore.py odd  --tag dpann_group           every locus table of loci/<tag>/ scored; ODD summaries only
  dscore.py even --tag dpann_group --names L1,L2_t0.05,...   even verdicts, paired intervals, ribosomal diagnosis
  dscore.py rdiag                            ribosomal diagnosis of token DH and the benchmark structure only
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import vscore_v as S1  # noqa: E402  (copy of vocab/vscore.py)

B, RU = S1.B, S1.RU
RC = B.RC
RD = S1.RD
LL = RD / "long_loci"
CALLERS = S1.CALLERS
WORDS_VMOB = RD / "vocab/scores/main_tau10_V-mob.parquet"
WORDS_COH = RD / "vocab/scores/runs_main_tau10_s1coh_V-coh0.1.parquet"
WORDS_VMOB_LOCI = RD / "vocab/loci/main_tau10_V-mob.parquet"
WORDS_COH_LOCI = RD / "vocab/loci/runs_main_tau10_s1coh_V-coh0.1.parquet"
DUCK = RD / "ws_dpann/input/sharur.duckdb"
log = B.log


def jdump(obj, fn):
    Path(fn).write_text(json.dumps(obj, indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x)) + "\n")
    log("wrote", fn)


def table(ctx, df):
    se = pd.DataFrame(dict(s=df.s.to_numpy(np.int64), e=df.e.to_numpy(np.int64), m=np.full(len(df), -1, np.int64)))
    return ctx.table(se)


def ref_odd():
    return json.load(open(LL / "results/refcheck_dpann.json"))


# =========================================================================== commands
def cmd_refcheck(a):
    ctx = S1.Ctx()
    df = pd.read_parquet(S1.DH_LOCI)
    per, summ, sc, rib, loc = S1.score(ctx, df, a.threads)
    stored = pd.read_parquet(S1.DH_SCORES).set_index("instance_id")
    p = per.set_index("instance_id").loc[stored.index]
    same = {c: bool(np.array_equal(p[c].to_numpy(), stored[c].to_numpy())) for c in ("f1", "control_mean_f1", "coverage")}
    out = dict(summary_even=summ["even"], summary_odd=summ["odd"], ribo_all=rib["all"], ribo_odd=rib["odd"], loci=loc,
               per_instance_equal_stored=same)
    R = S1.REF_EVEN
    ok = (round(summ["even"]["summed_DF_TXSS_excess"], 4) == R["summed"]
          and all(round(summ["even"][c]["f1"], 3) == R["f1"][c] for c in CALLERS)
          and rib["all"]["recovered"] == R["recovered"] and rib["all"]["resolved"] == R["resolved"]
          and S1.parv(rib["all"]["parvarchaeaceae_resolved"]) == R["parv_resolved"] and all(same.values()))
    out["reproduced"] = bool(ok)
    log("DH even summed", round(summ["even"]["summed_DF_TXSS_excess"], 4), {c: round(summ["even"][c]["f1"], 3) for c in CALLERS},
        "ribo", rib["all"]["recovered"], rib["all"]["resolved"], rib["all"]["parvarchaeaceae"],
        rib["all"]["parvarchaeaceae_resolved"], "per-instance equal", same, "REPRODUCED", ok)
    jdump(out, LL / "results/refcheck_dpann.json")
    assert ok


def sel_score(summ_odd, rib_odd, ref):
    cond = S1.odd_conditions(summ_odd, rib_odd, ref["summary_odd"])
    return [int(sum(cond)), int(rib_odd["resolved"]), float(summ_odd["summed_DF_TXSS_excess"])], cond


def cmd_odd(a):
    ctx = S1.Ctx()
    ref = ref_odd()
    assert ref["reproduced"]
    dh_per = pd.read_parquet(S1.DH_SCORES).set_index("instance_id")
    names = sorted(p.stem for p in (LL / "loci" / a.tag).glob("*.parquet"))
    if a.names:
        names = [x for x in a.names.split(",") if x]
    res = dict(tag=a.tag, DH_odd=dict(summary=ref["summary_odd"], ribo=ref["ribo_odd"]), outputs={})
    for nm in names:
        df = pd.read_parquet(LL / "loci" / a.tag / f"{nm}.parquet")
        per, summ, sc, rib, loc = S1.score(ctx, table(ctx, df), a.threads)
        pr = RU.paired(per.set_index("instance_id"), dh_per, ctx.gcode, "odd")
        score, cond = sel_score(summ["odd"], rib["odd"], ref)
        res["outputs"][nm] = dict(summary_odd=summ["odd"], ribo_odd=rib["odd"], loci=loc, vs_DH_odd=pr, conditions_odd=cond,
                                  score=score)
        log(a.tag, nm, "ODD resolved", rib["odd"]["resolved"], "recovered", rib["odd"]["recovered"], "parv",
            rib["odd"]["parvarchaeaceae"], "summed", round(summ["odd"]["summed_DF_TXSS_excess"], 4), "vs DH",
            [round(x, 4) for x in pr["summed"]], {c[:4]: round(summ["odd"][c]["f1"], 3) for c in CALLERS}, "cond", cond,
            "genes", loc["genes_covered"])
    # predeclared selection (amendment_1): theta by L2; L4 vs L4a at that theta
    sel = {}
    l2 = {k: v for k, v in res["outputs"].items() if k.startswith("L2_t")}
    if l2:
        best = max(l2, key=lambda k: tuple(l2[k]["score"]))
        th = best.split("_t")[1]
        sel["theta"] = th
        cands = {k: res["outputs"][k] for k in (f"L4_t{th}", f"L4a_t{th}") if k in res["outputs"]}
        if cands:
            sel["L4_variant"] = max(cands, key=lambda k: tuple(cands[k]["score"]))
        sel["scores"] = {k: v["score"] for k, v in res["outputs"].items()}
    res["selection"] = sel
    res["written"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    log("SELECTION (odd)", sel)
    jdump(res, LL / "results" / f"odd_{a.tag}{a.out_suffix}.json")


def cmd_even(a):
    ctx = S1.Ctx()
    ref = ref_odd()
    assert ref["reproduced"]
    dh_per = pd.read_parquet(S1.DH_SCORES).set_index("instance_id")
    cmp = {"DH": dh_per, "words_Vmob": pd.read_parquet(WORDS_VMOB).set_index("instance_id"),
           "words_Vcoh0.1": pd.read_parquet(WORDS_COH).set_index("instance_id")}
    for c in [x for x in a.compare.split(",") if x]:          # label=tag/name (per-instance scores of this workstream)
        lab, ref_ = c.split("=")
        cmp[lab] = pd.read_parquet(LL / "scores" / f"{ref_}.parquet").set_index("instance_id")
    dh_sc = B.ribo_score(ctx.runs, *[pd.read_parquet(S1.DH_LOCI)[c].to_numpy() for c in ("gid_s", "gid_e")])
    names = [x for x in a.names.split(",") if x]
    res = dict(tag=a.tag, outputs={})
    (LL / "scores" / a.tag).mkdir(parents=True, exist_ok=True)
    loci = {}
    for nm in names:
        df = pd.read_parquet(LL / "loci" / a.tag / f"{nm}.parquet")
        loci[nm] = df
        per, summ, sc, rib, loc = S1.score(ctx, table(ctx, df), a.threads)
        per.to_parquet(LL / "scores" / a.tag / f"{nm}.parquet", index=False)
        P = per.set_index("instance_id")
        item = dict(summary_even=summ["even"], summary_odd=summ["odd"], ribo=rib, loci=loc,
                    genes_in_loci_frac=loc["genes_covered"] / len(ctx.gc),
                    ribo_resolved_vs_DH=RU.ribo_paired(sc, dh_sc), verdict_vs_DH=None)
        for k, v in cmp.items():
            item[f"vs_{k}_even"] = RU.paired(P, v, ctx.gcode, "even")
        item["verdict_vs_DH"] = S1.verdict(item["vs_DH_even"], rib["all"], ref["ribo_all"])
        item["ribosomal_success"] = bool(rib["all"]["resolved"] > 1512 and rib["all"]["recovered"] >= 1650
                                         and S1.parv(rib["all"]["parvarchaeaceae"]) >= 110
                                         and item["verdict_vs_DH"]["summed_ok"] and all(item["verdict_vs_DH"]["f1_ok"].values()))
        res["outputs"][nm] = item
        log(a.tag, nm, "EVEN ribo", rib["all"]["recovered"], rib["all"]["resolved"], "vs DH resolved",
            [round(x) for x in item["ribo_resolved_vs_DH"]], "parv", rib["all"]["parvarchaeaceae"], "summed",
            round(summ["even"]["summed_DF_TXSS_excess"], 4),
            {k: [round(x, 4) for x in item[f"vs_{k}_even"]["summed"]] for k in cmp},
            "F1", {c[:4]: round(summ["even"][c]["f1"], 3) for c in CALLERS},
            "bprec", {c[:4]: summ["even"][c]["median_bprec_covered"] for c in CALLERS},
            "genes", round(item["genes_in_loci_frac"], 3), "success", item["ribosomal_success"])
    if a.extra:
        res["extra"] = extra_diag(ctx, loci, a)
    if a.rdiag:
        loci["token_DH"] = pd.read_parquet(S1.DH_LOCI).rename(columns={"gid_s": "s", "gid_e": "e"})[["s", "e"]]
        res["ribosomal_diagnosis"] = rdiag(ctx, loci, a.threads)
    res["written"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    jdump(res, LL / "results" / f"even_{a.tag}{a.out_suffix}.json")


# =========================================================================== ribosomal diagnosis
def extra_diag(ctx, loci, a):
    """Part 2: gold member genes outside every locus (even genomes, instances with >= 2 members) per caller; the runs
    that part-1 L4b (theta .1) leaves under-covered: how many have every internal boundary bonded in this arm (best_frac
    1 under the arm's L1) and how many each output resolves."""
    mem = pd.read_parquet(B.PKG / "gold/members.parquet")
    inst = pd.read_parquet(B.PKG / "gold/instances.parquet", columns=["instance_id", "genome"])
    even = set(inst.instance_id[inst.genome.map(ctx.gcode).to_numpy() % 2 == 0])
    k = mem.groupby("instance_id").gid.transform("size")
    mm = mem[(k >= 2) & mem.instance_id.isin(even)]
    n = len(ctx.gc)
    p1 = pd.read_parquet(LL / "loci/dpann_group/L4b_t0.1.parquet")
    sc1 = B.ribo_score(ctx.runs, p1.s.to_numpy(), p1.e.to_numpy())
    u56 = sc1.best_frac.to_numpy() < 0.8
    l1 = pd.read_parquet(LL / "loci" / a.tag / "L1.parquet")
    scl1 = B.ribo_score(ctx.runs, l1.s.to_numpy(), l1.e.to_numpy())
    out = dict(part1_L4b_under_covered=int(u56.sum()),
               of_those_fully_bonded_in_this_arm=int((u56 & (scl1.best_frac.to_numpy() >= 1.0)).sum()),
               of_those_best_frac_q50_under_arm_L1=float(np.median(scl1.best_frac.to_numpy()[u56])), outputs={})
    for nm, df in loci.items():
        cov = np.zeros(n + 1, np.int32)
        np.add.at(cov, df.s.to_numpy(), 1)
        np.add.at(cov, df.e.to_numpy() + 1, -1)
        inl = np.cumsum(cov)[:n] > 0
        sc = B.ribo_score(ctx.runs, df.s.to_numpy(), df.e.to_numpy())
        out["outputs"][nm] = dict(
            members_outside_every_locus={c: float(1 - inl[g.gid.to_numpy()].mean()) for c, g in mm.groupby("caller")},
            part1_under56_resolved=int((u56 & sc.resolved.to_numpy()).sum()),
            part1_under56_recovered_best_frac_ge_08=int((u56 & (sc.best_frac.to_numpy() >= 0.8)).sum()))
        log("extra", nm, out["outputs"][nm])
    log("extra", {k_: v for k_, v in out.items() if k_ != "outputs"})
    return out


def cmd_pairs(a):
    """Paired even differences between per-instance score files of this workstream: --pairs A=tag/name:B=tag/name,..."""
    ctx = S1.Ctx()
    res = {}
    for pr in [x for x in a.pairs.split(",") if x]:
        A, Bn = pr.split(":")
        PA = pd.read_parquet(LL / "scores" / f"{A}.parquet").set_index("instance_id")
        PB = pd.read_parquet(LL / "scores" / f"{Bn}.parquet").set_index("instance_id")
        res[pr] = RU.paired(PA, PB, ctx.gcode, "even")
        la = pd.read_parquet(LL / "loci" / f"{A}.parquet")
        lb_ = pd.read_parquet(LL / "loci" / f"{Bn}.parquet")
        sa = B.ribo_score(ctx.runs, la.s.to_numpy(), la.e.to_numpy())
        sb = B.ribo_score(ctx.runs, lb_.s.to_numpy(), lb_.e.to_numpy())
        res[pr]["ribo_resolved"] = RU.ribo_paired(sa, sb)
        log("PAIR", pr, "summed", [round(x, 4) for x in res[pr]["summed"]],
            {c[:4]: [round(x, 3) for x in res[pr][c]["d_f1"]] for c in CALLERS}, "resolved", res[pr]["ribo_resolved"])
    jdump(res, LL / "results" / f"pairs_{a.tag}.json")


def g2_runs():
    r = pd.read_parquet(RC.CACHE / "runs.parquet")
    return r[r.run_set == "g2"].reset_index(drop=True)


def pfam_names(pids, threads):
    import duckdb
    con = duckdb.connect(str(DUCK), read_only=True)
    con.execute(f"SET threads={threads}")
    con.register("q", pd.DataFrame(dict(protein_id=pd.unique(pids))))
    d = con.execute("SELECT a.protein_id, a.name FROM annotations a JOIN q USING (protein_id) "
                    "WHERE a.source = 'pfam'").df()
    con.close()
    return d.drop_duplicates()


def merged_targets(runs, maxgap=5):
    """Chains of benchmark runs on one contig with gaps of <= maxgap genes -> merged targets (gap genes as members)."""
    r = runs.sort_values("gid_start").reset_index()
    cc = r.contig_code.to_numpy() if "contig_code" in r else None
    gs, ge = r.gid_start.to_numpy(), r.gid_end.to_numpy()
    gap = np.r_[gs[1:] - ge[:-1] - 1, 10 ** 9]
    same = np.r_[cc[1:] == cc[:-1], False]
    link = same & (gap <= maxgap)
    tid = np.r_[0, np.cumsum(~link[:-1])]
    t = pd.DataFrame(dict(tid=tid, s=gs, e=ge)).groupby("tid").agg(s=("s", "min"), e=("e", "max"), k=("s", "size"))
    run_tid = np.empty(len(r), np.int64)
    run_tid[r["index"].to_numpy()] = tid
    return t.reset_index(drop=True), run_tid, gap[:-1][same[:-1]]


def rdiag(ctx, loci, threads):
    genes = pd.read_parquet(B.PKG / "universe/genes.parquet", columns=["gid", "protein_id", "contig_code"])
    cc = genes.contig_code.to_numpy()
    runs = ctx.runs.copy()
    runs["contig_code"] = cc[runs.gid_start.to_numpy()]
    allg2 = g2_runs()
    allg2["contig_code"] = cc[allg2.gid_start.to_numpy()]
    n = len(cc)
    # gene -> benchmark run id / small g2 run (8-15 marked) / marked gene
    in_run = np.full(n, -1, np.int64)
    for i, (s, e) in enumerate(zip(runs.gid_start.to_numpy(), runs.gid_end.to_numpy())):
        in_run[s:e + 1] = i
    small = allg2[allg2.n_marked < 16]
    in_small = np.zeros(n, bool)
    for s, e in zip(small.gid_start.to_numpy(), small.gid_end.to_numpy()):
        in_small[s:e + 1] = True
    marked = np.zeros(n, bool)
    marked[pd.read_parquet(RC.CACHE / "marked.parquet").gid.to_numpy()] = True
    # gap genes: between two consecutive benchmark runs on one contig
    rs = runs.sort_values("gid_start")
    in_gap = np.zeros(n, bool)
    a_e, b_s = rs.gid_end.to_numpy()[:-1], rs.gid_start.to_numpy()[1:]
    samec = rs.contig_code.to_numpy()[:-1] == rs.contig_code.to_numpy()[1:]
    for e0, s1 in zip(a_e[samec], b_s[samec]):
        in_gap[e0 + 1:s1] = True
    # (3) neighbouring runs and merged targets. Among benchmark runs (>= 16 marked):
    tg, run_tid, gaps = merged_targets(runs, 5)
    gs, ge = runs.gid_start.to_numpy(), runs.gid_end.to_numpy()
    order = np.argsort(gs)
    gsrt, gesrt, ccs = gs[order], ge[order], runs.contig_code.to_numpy()[order]
    gapn = np.r_[gsrt[1:] - gesrt[:-1] - 1]
    sm = ccs[1:] == ccs[:-1]
    has_nb = np.zeros(len(runs), bool)
    close = sm & (gapn >= 1) & (gapn <= 5)
    has_nb[order[:-1][close]] = True
    has_nb[order[1:][close]] = True
    gap_hist = {str(g): int((gapn[sm] == g).sum()) for g in range(1, 11)}
    # ... and among marked fragments of any size (the same g2 rule without the >= 8 / >= 16 floor; read-only build_runs)
    mg_ = pd.read_parquet(RC.CACHE / "marked.parquet").gid.to_numpy()
    fr = RC.build_runs(mg_, cc, 2).sort_values("gid_start").reset_index(drop=True)
    fs, fe, fm = fr.gid_start.to_numpy(), fr.gid_end.to_numpy(), fr.n_marked.to_numpy()
    fcc = cc[fs]
    fgap = fs[1:] - fe[:-1] - 1
    flink = (fcc[1:] == fcc[:-1]) & (fgap <= 5)
    cid = np.r_[0, np.cumsum(~flink)]
    ch = pd.DataFrame(dict(cid=cid, s=fs, e=fe, m=fm)).groupby("cid").agg(s=("s", "min"), e=("e", "max"),
                                                                           m=("m", "sum"), k=("s", "size"))
    rf = np.searchsorted(fs, gs)
    assert np.array_equal(fs[rf], gs) and np.array_equal(fe[rf], ge)
    rch = cid[rf]
    ts, te = ch.s.to_numpy()[rch], ch.e.to_numpy()[rch]
    has_frag = ch.k.to_numpy()[rch] > 1
    extra_marked = ch.m.to_numpy()[rch] - fm[rf]
    ratio = (ge - gs + 1) / (te - ts + 1)
    tdf = pd.DataFrame(dict(gid_start=ts, gid_end=te))
    in_target = np.zeros(n, np.int64) - 1
    for i in range(len(runs)):
        in_target[ts[i]:te[i] + 1] = i
    fsz = fm[~np.isin(np.arange(len(fs)), rf)]
    struct = dict(runs=int(len(runs)), runs_with_neighbour_run_within_1_5=int(has_nb.sum()),
                  consecutive_run_gap_hist_1_10=gap_hist,
                  marked_fragments=int(len(fs)), fragments_by_marked={"1": int((fsz == 1).sum()), "2-3": int(((fsz >= 2) & (fsz <= 3)).sum()),
                                                                     "4-7": int(((fsz >= 4) & (fsz <= 7)).sum()),
                                                                     "8-15": int(((fsz >= 8) & (fsz <= 15)).sum())},
                  runs_with_fragment_within_3_5=int(has_frag.sum()),
                  extra_marked_genes_q50_q90_max_among_those=np.quantile(extra_marked[has_frag], [.5, .9, 1]).tolist()
                  if has_frag.any() else [],
                  run_over_neighbourhood_len_q10_q50_among_those=np.quantile(ratio[has_frag], [.1, .5]).tolist()
                  if has_frag.any() else [],
                  runs_whose_neighbourhood_fails_precision=int((ratio < 0.8).sum()),
                  neighbourhood_len_q50_q90_max=np.quantile(te - ts + 1, [.5, .9, 1]).tolist())
    # (4) oracles under the standing definition
    exact = B.ribo_score(runs, gs, ge)
    orc = B.ribo_score(runs, ts, te)
    struct["oracle_exact_span"] = dict(recovered=int(exact.recovered.sum()), resolved=int(exact.resolved.sum()))
    struct["oracle_merged_neighbourhood"] = dict(recovered=int(orc.recovered.sum()), resolved=int(orc.resolved.sum()),
                                                 resolved_among_runs_with_fragment=int(orc.resolved.to_numpy()[has_frag].sum()))
    out = dict(structure=struct, outputs={})
    pfq = {}
    for nm, df in loci.items():
        s, e = df.s.to_numpy(np.int64), df.e.to_numpy(np.int64)
        sc = B.ribo_score(runs, s, e)
        unres = ~sc.resolved.to_numpy()
        under = unres & (sc.best_frac.to_numpy() < 0.8)
        over = unres & (sc.best_frac.to_numpy() >= 0.8) & (sc.precision.to_numpy() < 0.8)
        el, er = sc.ext_left.to_numpy(), sc.ext_right.to_numpy()
        og = []
        per_run = []
        for i in np.flatnonzero(over):
            left = np.arange(gs[i] - el[i], gs[i])
            right = np.arange(ge[i] + 1, ge[i] + er[i] + 1)
            x = np.r_[left, right]
            per_run.append(len(x))
            og.append(np.c_[np.full(len(x), i), x])
        og = np.concatenate(og) if og else np.zeros((0, 2), np.int64)
        ri, gi = og[:, 0], og[:, 1]
        cat_a = (in_run[gi] >= 0) & (in_run[gi] != ri)
        cat_b = ~cat_a & in_gap[gi]
        cat_c = ~cat_a & ~cat_b
        c_nb = cat_c & (in_target[gi] == ri)                 # inside the run's own merged neighbourhood
        c_mk = cat_c & ~c_nb & marked[gi]                    # Ribosomal_* marker beyond it
        d = dict(resolved=int((~unres).sum()), unresolved=int(unres.sum()), under_covered=int(under.sum()),
                 over_extended=int(over.sum()),
                 under_covered_no_locus=int((under & (sc.fragments.to_numpy() == 0)).sum()),
                 under_covered_best_frac_q50=float(np.median(sc.best_frac.to_numpy()[under])) if under.any() else None,
                 under_covered_fragments_q50=float(np.median(sc.fragments.to_numpy()[under])) if under.any() else None,
                 over_extended_outside_genes=int(len(gi)),
                 outside_per_run_q10_q50_q90_max=np.quantile(per_run, [.1, .5, .9, 1]).tolist() if per_run else [],
                 outside_frac_a_other_run=float(cat_a.mean()) if len(gi) else None,
                 outside_frac_b_gap_between_runs=float(cat_b.mean()) if len(gi) else None,
                 outside_frac_c_other=float(cat_c.mean()) if len(gi) else None,
                 outside_c_in_small_g2_run=float((cat_c & in_small[gi]).mean()) if len(gi) else None,
                 outside_c_marked_outside_runs=float((cat_c & ~in_small[gi] & marked[gi]).mean()) if len(gi) else None,
                 outside_c1_within_merged_neighbourhood=float(c_nb.mean()) if len(gi) else None,
                 outside_c2_marked_beyond_neighbourhood=float(c_mk.mean()) if len(gi) else None,
                 outside_c3_rest=float((cat_c & ~c_nb & ~c_mk).mean()) if len(gi) else None,
                 over_extended_runs_with_fragment=int((over & has_frag).sum()),
                 unresolved_runs_with_fragment=int((unres & has_frag).sum()),
                 over_extended_runs_reaching_other_run=int(len(np.unique(ri[cat_a]))),
                 over_extended_runs_with_neighbour_1_5=int((over & has_nb).sum()),
                 unresolved_runs_with_neighbour_1_5=int((unres & has_nb).sum()))
        # merged-neighbourhood sensitivity: each run's target = its fragment chain (gaps <= 5, gap genes as members),
        # scored under the standing rule (best_frac >= .8 and precision >= .8 against the target)
        tsc = B.ribo_score(tdf, s, e)
        d["merged_targets_resolved"] = int(tsc.resolved.sum())
        d["merged_targets_recovered"] = int(tsc.recovered.sum())
        d["merged_targets_resolved_among_runs_with_fragment"] = int(tsc.resolved.to_numpy()[has_frag].sum())
        d["standing_resolved_among_runs_with_fragment"] = int(sc.resolved.to_numpy()[has_frag].sum())
        d["runs_resolved_standing_or_merged"] = int((sc.resolved.to_numpy() | tsc.resolved.to_numpy()).sum())
        pfq[nm] = gi
        out["outputs"][nm] = d
        log("rdiag", nm, d)
    allg = np.unique(np.concatenate([v for v in pfq.values() if len(v)])) if pfq else np.zeros(0, np.int64)
    if len(allg):
        pf = pfam_names(genes.protein_id.to_numpy()[allg], threads)
        pid_of = genes.protein_id.to_numpy()
        for nm, gi in pfq.items():
            if not len(gi):
                continue
            q = pd.DataFrame(dict(protein_id=pid_of[gi]))
            m = q.merge(pf, on="protein_id", how="left")
            has = m.dropna(subset=["name"])
            top = has.groupby("name").size().sort_values(ascending=False).head(30)
            out["outputs"][nm]["outside_pfam_top30_gene_counts"] = [[k, int(v)] for k, v in top.items()]
            out["outputs"][nm]["outside_genes_without_pfam"] = int((~q.protein_id.isin(pf.protein_id)).sum())
    return out


def has_nb_multi(run_tid, tg):
    return tg.k.to_numpy()[run_tid] > 1


def cmd_rdiag(a):
    ctx = S1.Ctx()
    loci = {"token_DH": pd.read_parquet(S1.DH_LOCI).rename(columns={"gid_s": "s", "gid_e": "e"})[["s", "e"]],
            "words_Vmob": pd.read_parquet(WORDS_VMOB_LOCI).rename(columns={"gid_s": "s", "gid_e": "e"})[["s", "e"]]}
    if WORDS_COH_LOCI.exists():
        loci["words_Vcoh0.1"] = pd.read_parquet(WORDS_COH_LOCI).rename(columns={"gid_s": "s", "gid_e": "e"})[["s", "e"]]
    res = rdiag(ctx, loci, a.threads)
    res["written"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    jdump(res, LL / "results/rdiag_reference.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["refcheck", "odd", "even", "rdiag", "pairs"])
    ap.add_argument("--compare", default="")
    ap.add_argument("--extra", action="store_true")
    ap.add_argument("--pairs", default="")
    ap.add_argument("--tag", default="dpann_group")
    ap.add_argument("--names", default="")
    ap.add_argument("--rdiag", action="store_true")
    ap.add_argument("--out-suffix", dest="out_suffix", default="")
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    import numba
    numba.set_num_threads(a.threads)
    {"refcheck": cmd_refcheck, "odd": cmd_odd, "even": cmd_even, "rdiag": cmd_rdiag, "pairs": cmd_pairs}[a.cmd](a)


if __name__ == "__main__":
    main()
