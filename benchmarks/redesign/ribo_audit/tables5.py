#!/usr/bin/env python
"""ribo_audit part 5 tables (predeclared5.json; DEVELOPMENT). Usage: tables5.py WORK5DIR WORK4DIR OUTFILE"""
import json
import sys
from pathlib import Path

import pandas as pd

W5, W4 = Path(sys.argv[1]), Path(sys.argv[2])
out = []
pr = out.append
DSS = ("lut", "dpann", "srvp")
STRATA = [("lut", "Bacteria"), ("srvp", "Bacteria"), ("dpann", "Archaea"), ("srvp", "Archaea")]
DEFS = ("standing", "part4", "a", "b", "c")
OUTS = ("Z", "WA", "DH")
J = {ds: json.load(open(W5 / ds / "score5.json")) for ds in DSS}

pr("ALL DEFINITIONS (DEVELOPMENT numbers: every dataset has been seen). resolved / orig = resolved against the original run count")
pr("(runs below minimum count unresolved); any-fit and chance on kept runs / targets.")
pr(f"  {'output':4s} {'dataset':7s} {'domain':9s} {'definition':9s} {'runs':>5s} {'kept':>5s} {'below':>5s} {'targets':>7s} "
   f"{'recov':>6s} {'resolved':>8s} {'res/orig':>9s} {'any-fit':>7s} {'chance':>7s}")
rows = []
for n in OUTS:
    for ds, dm in STRATA:
        for d in DEFS:
            s = J[ds]["summary"].get(f"{d}|{dm}")
            if s is None:
                continue
            x = s[n]
            pr(f"  {n:4s} {ds:7s} {dm:9s} {d:9s} {s['runs']:5d} {s['kept']:5d} {s['below_min']:5d} {s['targets']:7d} {x['recovered']:6d} "
               f"{x['resolved']:8d} {x['resolved'] / s['runs']:9.1%} {x['anyfit']:7d} {x['chance'] if x['chance'] is not None else float('nan'):7.2%}")
            rows.append(dict(output=n, ds=ds, dom=dm, d=d, runs=s["runs"], kept=s["kept"], below=s["below_min"], targets=s["targets"],
                             resolved=x["resolved"], anyfit=x["anyfit"], chance=x["chance"], recovered=x["recovered"]))
T = pd.DataFrame(rows)

pr("")
pr("TARGET STRUCTURE (kept runs; runs cut = runs whose marked genes fall in >= 2 chains; equals accurate = targets identical to the")
pr("part-2 accurate units of the run)")
for ds, dm in STRATA:
    for d in DEFS:
        s = J[ds]["summary"].get(f"{d}|{dm}")
        pr(f"  {ds:5s} {dm:9s} {d:9s} runs {s['runs']:5d} cut {s.get('runs_cut', 0):4d} below minimum {s['below_min']:4d} "
           f"multi-target runs {s['multi_target_runs']:3d} targets {s['targets']:5d} target length median {s['target_len_median']:5.1f} "
           f"equals accurate {s['equals_accurate']:5d}")
    s = J[ds]["summary"][f"standing|{dm}"]
    pr(f"  {ds:5s} {dm:9s} accurate reading (part-2 strict, same annotation): " +
       ", ".join(f"{n} {s[n]['accurate_reading_resolved']}" for n in OUTS))

# verdicts
V = {}
for ds in DSS:
    for d in ("a", "b", "c"):
        p = W5 / ds / f"cases5_{d}.tsv"
        if p.exists() and p.stat().st_size > 1:
            V[(ds, d)] = pd.read_csv(p, sep="\t")
    p = W4 / ds / "cases4_standing_union.tsv"
    c = pd.read_csv(p, sep="\t")
    c["dom"] = c.gtype.map(lambda g: "Archaea" if g == "Archaea" else "Bacteria")
    V[(ds, "standing")] = c
    p = W4 / ds / "cases4_cluster_union.tsv"
    c = pd.read_csv(p, sep="\t")
    c["dom"] = c.gtype.map(lambda g: "Archaea" if g == "Archaea" else "Bacteria")
    V[(ds, "part4")] = c


def defattr(ds, d, dm, n):
    c = V.get((ds, d))
    if c is None:
        return None
    x = c[(c.output == n) & (c.dom == dm) & c.verdict.isin([2, 3])]
    return int(x.run_id.nunique())


pr("")
pr("SOUNDNESS (predeclared S1-S4; Z and WA)")
best_score = {}
for d in ("part4", "a", "b", "c"):
    ok_all, tot = True, 0
    for ds, dm in STRATA:
        for n in ("Z", "WA"):
            r = T[(T.output == n) & (T.ds == ds) & (T.dom == dm)]
            st = int(r[r.d == "standing"].resolved.iloc[0])
            p4 = int(r[r.d == "part4"].resolved.iloc[0])
            x = r[r.d == d].iloc[0]
            thr = max(st, p4) - 0.02 * x.runs
            s1 = x.resolved >= thr
            s2 = x.chance is not None and x.chance <= 0.05
            va, vs = defattr(ds, d, dm, n), defattr(ds, "standing", dm, n)
            s3 = va is not None and va <= vs
            s4 = x.below <= 0.02 * x.runs
            ok_all &= bool(s1 and s2 and s3 and s4)
            tot += int(x.resolved)
            pr(f"  {d:6s} {ds:5s} {dm:9s} {n:2s}: S1 {x.resolved} >= {thr:.1f} {'ok' if s1 else 'FAIL'}; S2 chance {x.chance:.2%} "
               f"{'ok' if s2 else 'FAIL'}; S3 v2+v3 runs {va} vs standing {vs} {'ok' if s3 else 'FAIL'}; S4 below {x.below} "
               f"{'ok' if s4 else 'FAIL'}")
    best_score[d] = (tot, ok_all)
    pr(f"  {d}: sum Z + WA resolved (original denominators) {tot}; sound on all four strata: {ok_all}")
cand = {d: v for d, v in best_score.items() if d in ("a", "b", "c")}
eligible = [d for d in ("a", "b", "c") if all(
    (T[(T.output == n) & (T.ds == ds) & (T.dom == dm) & (T.d == d)].chance.iloc[0] or 0) <= 0.05 for ds, dm in STRATA for n in ("Z", "WA"))]
best = max(eligible, key=lambda d: (cand[d][0], -("abc".index(d)))) if eligible else None
pr(f"  BEST strand-aware variant by the predeclared rule: {best} (eligible {eligible}; sums {({d: cand[d][0] for d in cand})})")

pr("")
pr("VERDICTS FOR RUNS UNRESOLVED UNDER EACH STRAND-AWARE VARIANT (part-2 strict on each unresolved target; runs counted)")
for d in ("a", "b", "c"):
    for ds, dm in STRATA:
        c = V.get((ds, d))
        if c is None:
            continue
        for n in OUTS:
            x = c[(c.output == n) & (c.dom == dm)]
            vc = x.drop_duplicates(["run_id", "verdict"]).verdict.value_counts()
            pr(f"  {d} {ds:5s} {dm:9s} {n:2s} unresolved runs {x.run_id.nunique():4d} (targets {len(x):4d}): " +
               " ".join(f"v{k} {int(vc.get(k, 0))}" for k in range(1, 8)))
    if (("dpann", d) in V):
        allc = pd.concat([V[(ds, d)] for ds in DSS if (ds, d) in V])
        pr(f"  {d} verdict-2 subtypes (all): {allc[allc.verdict == 2].subtype.value_counts().to_dict()}")

pr("")
pr("SAME-STRAND CONNECTOR (cross-lineage relative orientation of adjacent member families; output-independent)")
for ds in DSS:
    for dm, c in J[ds]["connector"].items():
        pr(f"  {ds:5s} {dm:9s} target junctions {c['target_junctions']}, distinct pairs {c['target_pairs']}, pairs in >= 5 groups "
           f"{c['pairs_with_5_groups']}")
        pr(f"      connector histogram (bins {c['hist_bins']}): {c['hist']}")
        pr(f"      target junctions by class: {c['junction_classes']}")
        low = c["opposite_in_most"]
        pr(f"      pairs opposite-strand in most groups (connector < 0.5, >= 5 groups): {len(low)}")
        for x in low[:25]:
            pr(f"        {x['pair']:28s} connector {x['conn']:.3f} groups {x['groups']:4d} target junctions {x['target_junctions']}")

Path(sys.argv[3]).write_text("\n".join(out) + "\n")
print("\n".join(out))
