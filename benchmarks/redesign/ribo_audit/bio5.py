#!/usr/bin/env python
"""ribo_audit part 5: strand-aware cluster-level ribosomal runs (predeclared5.json). DEVELOPMENT on all three datasets.

usage (biotite login node, one thread):
  bio5.py score   DS            connector, targets under standing / part4 / a / b / c, scores of Z / WA / DH, chance,
                                accurate-reading comparison
  bio5.py verdict DS VARIANT    part-2 strict verdicts on every unresolved target of VARIANT
DS = lut | dpann | srvp. Writes $RD/ribo_audit/work5/<DS>/. Reads part-4 annotation (work4 / work3); no new search.
"""
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bio as Bm  # noqa: E402
import bio4 as B4  # noqa: E402
import audit as A  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

W5 = A.AD / "work5"
log = A.log
OUTS = B4.OUTS
DEFS = ("standing", "part4", "a", "b", "c")
ALIAS = {"L4e": "L4", "L26": "L24", "L5e": "L18", "L18p": "L18", "L27A": "L15", "S29e": "S14", "L23e": "L14", "S12_S23": "S12"}
MINMARK = 16


def norm(x):
    x = re.sub(r"(?i)^ribosom(al)?_", "", str(x))
    x = re.sub(r"_(C|N)$", "", x)
    return ALIAS.get(x, x)


def family_names(B):
    lit = np.array([str(x) for x in B.lit], dtype=object)
    fam = np.full(B.D.n, "", dtype=object)
    r = np.flatnonzero(B.isR)
    fam[r] = [norm(lit[i]) if lit[i] else "RP?" for i in r]
    nn = np.flatnonzero(B.cls == "N")
    fam[nn] = lit[nn]
    return fam


def connector(B, fam):
    """Junctions = consecutive members of base chains. Returns junction frame (index t into member list) with the
    excluded-own-group connector, and pair-level table per domain."""
    D = B.D
    mk = np.flatnonzero(B.member)
    base_brk = np.r_[True, (np.diff(mk) > 3) | (D.cc[mk[1:]] != D.cc[mk[:-1]])]
    t = np.flatnonzero(~base_brk)                      # junction between mk[t-1] and mk[t]
    i, j = mk[t - 1], mk[t]
    fa, fb = fam[i], fam[j]
    key = np.where(fa <= fb, fa + "|" + fb, fb + "|" + fa)
    J = pd.DataFrame(dict(t=t, i=i, j=j, key=key, grp=D.grp[D.gc[i]], dom=D.domain_of_genome[D.gc[i]],
                          same=(D.strand[i] == D.strand[j])))
    G = J.groupby(["dom", "key", "grp"]).same.mean().rename("f").reset_index()
    P = G.groupby(["dom", "key"]).agg(S=("f", "sum"), N=("f", "size")).reset_index()
    J = J.merge(P, on=["dom", "key"], how="left").merge(G, on=["dom", "key", "grp"], how="left")
    J["conn"] = np.where(J.N > 1, (J.S - J.f) / (J.N - 1).clip(lower=1), np.nan)
    J["n_other"] = J.N - 1
    P["conn_all"] = P.S / P.N
    return mk, base_brk, J, P


def chains(B, mk, base_brk, J, variant):
    D = B.D
    brk = base_brk.copy()
    if variant in ("a", "b", "c"):
        flip = np.r_[False, D.strand[mk[1:]] != D.strand[mk[:-1]]]
        if variant == "c":
            keep = np.zeros(len(mk), bool)
            jj = J[(~J.same) & (J.conn <= 1 / 3) & (J.n_other >= 5)]
            keep[jj.t.to_numpy()] = True
            flip &= ~keep
        brk |= flip
        if variant == "b":
            for d in (1, 2):
                k = np.minimum(mk[:-1] + d, D.n - 1)
                valid = k < mk[1:]
                brk[1:] |= valid & (D.strand[k] != D.strand[mk[:-1]])
    cs = mk[brk]
    ce = mk[np.r_[brk[1:], True]]
    cid = np.cumsum(brk) - 1
    return cs, ce, cid


def run_targets(B, mk, cs, ce, cid, runs):
    D = B.D
    pos = np.full(D.n, -1, np.int64)
    pos[mk] = np.arange(len(mk))
    cm = np.r_[0, np.cumsum(D.marked)]
    out = []
    for s, e in zip(runs.gid_start.to_numpy(), runs.gid_end.to_numpy()):
        ids = sorted(set(int(cid[pos[k]]) for k in range(s, e + 1) if D.marked[k]))
        pieces = [(int(cs[c]), int(ce[c]), int(cm[ce[c] + 1] - cm[cs[c]])) for c in ids]
        out.append(pieces)
    return out


def cmd_score(ds):
    t0 = time.time()
    B = B4.Bio4(ds, "union")
    D = B.D
    runs = D.runs
    Ls, lr = B4.loci(ds)
    (W5 / ds).mkdir(parents=True, exist_ok=True)
    fam = family_names(B)
    mk, base_brk, J, P = connector(B, fam)
    # targets per definition
    TG = {"standing": [[(int(s), int(e), -1)] for s, e in zip(runs.gid_start, runs.gid_end)]}
    for v in ("part4", "a", "b", "c"):
        cs, ce, cid = chains(B, mk, base_brk, J, v)
        pcs = run_targets(B, mk, cs, ce, cid, runs)
        if v == "part4":
            TG[v] = [[p for p in ps if p[0] <= s <= p[1]] for ps, s in zip(pcs, runs.gid_start)]
        else:
            TG[v] = [[p for p in ps if p[2] >= MINMARK] for ps in pcs]
        if v != "part4":
            TG[v + "_pieces"] = pcs
    # connector report at part-4 target junctions
    tj = np.zeros(len(mk), bool)
    pos = np.full(D.n, -1, np.int64)
    pos[mk] = np.arange(len(mk))
    for tg in TG["part4"]:
        for (s, e, _) in tg:
            a_, b_ = pos[s], pos[e]
            tj[a_ + 1:b_ + 1] = True
    J["in_target"] = tj[J.t.to_numpy()]
    J["cls"] = np.where(J.same, "same", np.where((J.conn <= 1 / 3) & (J.n_other >= 5), "opp_conserved",
                                                 np.where((J.conn > 0.5) & (J.n_other >= 5), "opp_lineage_specific", "opp_other")))
    J.to_parquet(W5 / ds / "junctions5.parquet", index=False)
    P.to_parquet(W5 / ds / "connector5_pairs.parquet", index=False)
    # accurate-reading units of each run (part-2 strict, bio4 annotation)
    acc = [B.span_units(int(s), int(e))[4] for s, e in zip(runs.gid_start, runs.gid_end)]
    rows = []
    res = dict(dataset=ds, loci_reading=lr, defs={})
    for d in DEFS:
        rng = np.random.default_rng(17)
        ch = {n: {} for n in OUTS}
        for ri in range(len(runs)):
            tg = TG[d][ri]
            row = dict(run_id=int(runs.run_id.iat[ri]), dom=runs.domain.iat[ri], gtype=runs.gtype.iat[ri], definition=d,
                       n_targets=len(tg), kept=len(tg) > 0, targets=";".join(f"{s}..{e}" for s, e, _ in tg),
                       target_len=sum(e - s + 1 for s, e, _ in tg),
                       equals_accurate=bool(len(tg) > 0 and [(s, e) for s, e, _ in tg] == [tuple(x) for x in acc[ri]]))
            if d in ("a", "b", "c"):
                row["pieces"] = len(TG[d + "_pieces"][ri])
                row["pieces_marked"] = ",".join(str(m) for _, _, m in TG[d + "_pieces"][ri])
            for n in OUTS:
                st = [A.standard(Ls[n], s, e) for s, e, _ in tg]
                row[f"{n}_recovered"] = bool(tg) and all(x["recovered"] for x in st)
                row[f"{n}_resolved"] = bool(tg) and all(x["resolved"] for x in st)
                row[f"{n}_anyfit"] = bool(tg) and all(Bm.any_fit(Ls[n], s, e) for s, e, _ in tg)
                if d == "standing":
                    row[f"{n}_accurate"] = bool(acc[ri]) and all(Bm.score_target(Ls[n], s, e)["ok"] for s, e in acc[ri])
            rows.append(row)
            for s, e, _ in sorted(tg):
                cf, cl = int(D.cfirst[s]), int(D.clast[s])
                ln = e - s + 1
                cand = [x for x in range(cf, cl - ln + 2) if x + ln - 1 < s or x > e]
                if cand:
                    for stt in rng.choice(cand, size=5, replace=True):
                        for n in OUTS:
                            c_ = ch[n].setdefault(runs.domain.iat[ri], [0, 0])
                            c_[0] += int(Bm.score_target(Ls[n], int(stt), int(stt) + ln - 1)["ok"])
                            c_[1] += 1
        res["defs"][d] = {n: {dm: dict(resolved=v[0], windows=v[1], chance=v[0] / max(v[1], 1)) for dm, v in ch[n].items()} for n in OUTS}
    df = pd.DataFrame(rows)
    df.to_parquet(W5 / ds / "score5.parquet", index=False)
    summ = {}
    for (d, dm), x in df.groupby(["definition", "dom"]):
        r = dict(runs=int(len(x)), kept=int(x.kept.sum()), below_min=int((~x.kept).sum()), targets=int(x.n_targets.sum()),
                 multi_target_runs=int((x.n_targets > 1).sum()), equals_accurate=int(x.equals_accurate.sum()),
                 target_len_median=float(x[x.kept].target_len.median()))
        if d in ("a", "b", "c"):
            r["runs_cut"] = int((x.pieces > 1).sum())
        for n in OUTS:
            r[n] = dict(recovered=int(x[f"{n}_recovered"].sum()), resolved=int(x[f"{n}_resolved"].sum()),
                        anyfit=int(x[f"{n}_anyfit"].sum()), chance=res["defs"][d][n].get(dm, {}).get("chance"),
                        chance_windows=res["defs"][d][n].get(dm, {}).get("windows"))
            if d == "standing":
                r[n]["accurate_reading_resolved"] = int(x[f"{n}_accurate"].sum())
        summ[f"{d}|{dm}"] = r
    res["summary"] = summ
    # connector summary
    cr = {}
    for dm in sorted(J.dom.unique()):
        jt = J[(J.dom == dm) & J.in_target]
        pairs = jt.key.unique()
        pp = P[(P.dom == dm) & P.key.isin(pairs) & (P.N >= 5)]
        hist = np.histogram(pp.conn_all, bins=[0, .1, .2, .3, .4, .5, .6, .7, .8, .9, .95, 1.0001])[0].tolist()
        low = pp[pp.conn_all < 0.5].sort_values("N", ascending=False)
        cover = jt.groupby("key").size()
        cr[dm] = dict(target_junctions=int(len(jt)), target_pairs=int(len(pairs)), pairs_with_5_groups=int(len(pp)),
                      hist_bins=[0, .1, .2, .3, .4, .5, .6, .7, .8, .9, .95, 1.0], hist=hist,
                      junction_classes=jt.cls.value_counts().to_dict(),
                      opposite_in_most=[dict(pair=k, conn=round(float(c), 3), groups=int(nn), target_junctions=int(cover.get(k, 0)))
                                        for k, c, nn in zip(low.key, low.conn_all, low.N)])
    res["connector"] = cr
    json.dump(res, open(W5 / ds / "score5.json", "w"), indent=1, default=str)
    log(ds, json.dumps({k: (v["kept"], v["below_min"], v["targets"], {n: (v[n]["resolved"], v[n]["anyfit"], round(v[n]["chance"] or 0, 4))
                                                                       for n in OUTS}) for k, v in summ.items()}), f"{time.time()-t0:.0f}s")


def cmd_verdict(ds, variant):
    t0 = time.time()
    B = B4.Bio4(ds, "union")
    D = B.D
    runs = D.runs
    Ls, _ = B4.loci(ds)
    S = pd.read_parquet(W5 / ds / "score5.parquet")
    S = S[S.definition == variant].reset_index(drop=True)
    cases = []
    for ri, r in S.iterrows():
        if not r.kept:
            continue
        for n in OUTS:
            if r[f"{n}_resolved"]:
                continue
            for tg in r.targets.split(";"):
                s, e = map(int, tg.split(".."))
                st = A.standard(Ls[n], s, e)
                if st["resolved"]:
                    continue
                mk_, comps, single, n_units, targets = B.span_units(s, e)
                L = Ls[n]
                c = dict(run_id=int(r.run_id), output=n, dom=r.dom, gtype=r.gtype, span_s=s, span_e=e, span_len=e - s + 1,
                         std_bf=float(st["best_frac"]), n_units=n_units, n_targets=len(targets),
                         genome_code=int(runs.genome_code.iat[ri]))
                decisive, kind = [], None
                if not targets:
                    verdict = 7
                else:
                    if n_units > 1:
                        uid = [int(B.comp_of[k]) if B.comp_of[k] >= 0 else -10 - k for k in mk_]
                        for x, y, ux, uy in zip(mk_[:-1], mk_[1:], uid[:-1], uid[1:]):
                            if ux != uy:
                                p = next((q for q in range(x, y) if not B.link[q]), x)
                                decisive.append(("glue", p))
                        kind = "glue"
                    else:
                        ts, te = targets[0]
                        sc = Bm.score_target(L, ts, te)
                        c.update(C_len=te - ts + 1, C_minus_span=(te - ts) - (e - s))
                        if sc["ok"]:
                            kind = "ok"
                        elif Bm.any_fit(L, ts, te):
                            kind = "fit"
                        elif sc["bf"] < 0.8:
                            kind = "split"
                            if sc["a"] > ts:
                                decisive.append(("cut", sc["a"] - 1))
                            if 0 <= sc["b"] < te:
                                decisive.append(("cut", sc["b"]))
                            if sc["a"] < 0:
                                decisive.append(("cut", ts))
                        else:
                            kind = "ext"
                    dat = [x for (_, p) in decisive for x in (p - 1, p, p + 1, p + 2)
                           if D.cfirst[s] <= x <= D.clast[s] and B.anom[x]]
                    verdict = 6 if (dat and kind in ("split", "glue")) else {"glue": 2, "ok": 3, "fit": 5, "split": 1, "ext": 4}[kind]
                c["verdict"] = verdict
                c["junctions"] = " ; ".join(f"{kd}@{p - s} {Bm.gene_label(B, p)}|{Bm.gene_label(B, p + 1)} "
                                            f"{'same' if D.strand[p] == D.strand[p + 1] else 'OPP'} d={int(B.dist[p]) if B.dist[p] < 10 ** 8 else None}"
                                            for kd, p in decisive)
                if verdict == 2:
                    c["subtype"] = "+".join(sorted({"strand" if D.strand[p] != D.strand[p + 1] else
                                                    ("mem>500" if B.cls[p] in "RN" and B.cls[p + 1] in "RN" else "spacer")
                                                    for _, p in decisive}))
                c["genes"] = " ".join(Bm.gene_label(B, k) + ("<" if D.strand[k] < 0 else ">") for k in range(s, e + 1))
                cases.append(c)
    cdf = pd.DataFrame(cases)
    cdf.to_csv(W5 / ds / f"cases5_{variant}.tsv", sep="\t", index=False)
    below = S[~S.kept][["run_id", "dom", "gtype", "pieces", "pieces_marked"]]
    below.to_csv(W5 / ds / f"below_min5_{variant}.tsv", sep="\t", index=False)
    summ = cdf.groupby(["output", "dom", "verdict"]).size().unstack(fill_value=0) if len(cdf) else pd.DataFrame()
    json.dump({f"{a}|{b}": {int(k): int(v) for k, v in r.items()} for (a, b), r in summ.iterrows()},
              open(W5 / ds / f"verdicts5_{variant}.json", "w"), indent=1)
    log(ds, variant, summ.to_dict("index"), "below minimum", len(below), f"{time.time()-t0:.0f}s")


if __name__ == "__main__":
    cmd, ds = sys.argv[1], sys.argv[2]
    (W5 / ds).mkdir(parents=True, exist_ok=True)
    if cmd == "score":
        cmd_score(ds)
    elif cmd == "verdict":
        cmd_verdict(ds, sys.argv[3])
