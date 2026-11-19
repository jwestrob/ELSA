#!/usr/bin/env python
"""ribo_audit part 4 tables (predeclared4.json). Usage: tables4.py WORK4DIR OUTFILE (WORK4DIR/<ds>/ with the bio4.py outputs)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

W = Path(sys.argv[1])
out = []
pr = out.append
DSS = ("lut", "dpann", "srvp")
NAME = {"Z": "embedding-list tree (Z)", "WA": "family-bond adaptive tree (WA)", "DH": "engine reference (DH)"}
OUTS = ("Z", "WA", "DH")

pr("REPRODUCTION OF THE STANDING COUNTS (bio4.py repro; loci reading that reproduces is used for both definitions)")
for ds in DSS:
    r = json.load(open(W / ds / "repro4.json"))
    rd = r["loci_reading"]
    pr(f"  {ds:5s} runs {r['runs']}  reading {rd}  reproduced {r['reproduced']}  " + "  ".join(
        f"{n} {r[rd][n]['recovered']}/{r[rd][n]['resolved']} (any-fit {r[rd][n]['any_fit']}, target {r[rd][n]['target']}, "
        f"ribosomal_check equal {r[rd][n]['equal_rc']})" for n in OUTS))

pr("")
pr("MAIN TABLE: output x dataset x domain (PRIMARY reading = union member list for every genome, as in part 3)")
pr("  standing = package run; cluster = extended run (frozen definition); chance = 5 random same-contig windows per run of the")
pr("  target length (seed 17). lut and DPANN: check of the definition on data it was not developed on; SR-VP: development data.")
hdr = f"  {'output':32s} {'dataset':7s} {'stratum':12s} {'runs':>5s} {'std res':>8s} {'cl res':>7s} {'cl rec':>7s} {'cl anyfit':>9s} " \
      f"{'std anyfit':>10s} {'cl chance':>9s} {'std chance':>10s} {'gain':>5s} {'lost':>5s}"
pr(hdr)
J = {ds: json.load(open(W / ds / "ext4_union.json")) for ds in DSS}
rows = []
for n in OUTS:
    for ds in DSS:
        strata = [s for s in ("Bacteria", "Archaea") if s in J[ds]["strata"]]
        if ds == "srvp":
            strata = ["Bacteria", "Bact_closed", "Bact_multi", "Archaea"]
        for s in strata:
            v = J[ds]["strata"][s]
            x = v[n]
            pr(f"  {NAME[n]:32s} {ds:7s} {s:12s} {v['runs']:5d} {x['std_resolved']:8d} {x['cl_resolved']:7d} {x['cl_recovered']:7d} "
               f"{x['cl_anyfit']:9d} {x['std_anyfit']:10d} {x['cl_chance']:9.2%} {x['std_chance']:10.2%} {x['gained']:5d} {x['lost']:5d}")
            rows.append(dict(output=n, dataset=ds, stratum=s, runs=v["runs"], **x))
T = pd.DataFrame(rows)

pr("")
pr("DOMAIN READING (archaeal genomes: archaeal list only; bacterial genomes: archaeal list + bacterial supplement)")
for ds in ("dpann", "srvp"):
    p = W / ds / "ext4_domain.json"
    if not p.exists():
        continue
    jd = json.load(open(p))
    for s in ("Archaea",):
        if s not in jd["strata"]:
            continue
        v = jd["strata"][s]
        pr(f"  {ds:5s} {s}: runs {v['runs']}; ext length median {v['ext_len_median']} (union {J[ds]['strata'][s]['ext_len_median']}); " +
           "; ".join(f"{n} cluster resolved {v[n]['cl_resolved']} (union {J[ds]['strata'][s][n]['cl_resolved']}), any-fit "
                     f"{v[n]['cl_anyfit']}, chance {v[n]['cl_chance']:.2%}" for n in OUTS))

pr("")
pr("EXTENSION OF THE TARGET (genes; medians) and runs whose extended run equals the run")
for ds in DSS:
    for s, v in J[ds]["strata"].items():
        pr(f"  {ds:5s} {s:12s} runs {v['runs']:5d}  run {v['run_len_median']:5.1f} -> extended {v['ext_len_median']:5.1f} (left "
           f"{v['added_left_median']}, right {v['added_right_median']}); unchanged {v['ext_equals_run']}")

# ------------------------------------------------------------------ holds-up criteria
pr("")
pr("PREDECLARED CRITERIA (lut and DPANN; frozen outputs Z and WA)")
for ds in ("lut", "dpann"):
    dom = "Bacteria" if ds == "lut" else "Archaea"
    for n in ("Z", "WA"):
        x = T[(T.output == n) & (T.dataset == ds) & (T.stratum == dom)].iloc[0]
        thr = x.std_resolved - 0.02 * x.runs
        pr(f"  H1 {ds:5s} {n:2s}: cluster {x.cl_resolved} vs standing {x.std_resolved} - 2% ({thr:.1f}) -> "
           f"{'MET' if x.cl_resolved >= thr else 'NOT MET'}")
for ds in ("lut", "dpann"):
    c = T[T.dataset == ds]
    mx = c.cl_chance.max()
    pr(f"  H2 {ds:5s}: maximum cluster-level chance {mx:.2%} over {len(c)} output x stratum cells -> {'MET' if mx <= 0.05 else 'NOT MET'}")
VS = {}
for ds in DSS:
    for d in ("cluster", "standing"):
        p = W / ds / f"cases4_{d}_union.tsv"
        VS[(ds, d)] = pd.read_csv(p, sep="\t") if p.exists() and p.stat().st_size > 1 else pd.DataFrame(columns=["output", "verdict", "gtype"])
for ds in ("lut", "dpann"):
    for n in ("Z", "WA"):
        a = VS[(ds, "cluster")]
        b = VS[(ds, "standing")]
        ca = int(a[(a.output == n) & a.verdict.isin([2, 3])].shape[0])
        cb = int(b[(b.output == n) & b.verdict.isin([2, 3])].shape[0])
        pr(f"  H3 {ds:5s} {n:2s}: verdicts 2 + 3 under cluster {ca} vs standing {cb} -> {'MET' if ca <= cb else 'NOT MET'}")

# ------------------------------------------------------------------ verdicts
pr("")
pr("VERDICTS (part-2 strict rules; span = extended run under 'cluster', package run under 'standing'; every unresolved run)")
for ds in DSS:
    for d in ("cluster", "standing"):
        c = VS[(ds, d)]
        if not len(c):
            pr(f"  {ds:5s} {d:8s}: no unresolved runs")
            continue
        for n in OUTS:
            x = c[c.output == n]
            for g in sorted(x.gtype.unique()):
                y = x[x.gtype == g]
                vc = y.verdict.value_counts()
                pr(f"  {ds:5s} {d:8s} {n:2s} {g:11s} unresolved {len(y):4d}: " + " ".join(f"v{k} {int(vc.get(k, 0))}" for k in range(1, 8)))
    c = VS[(ds, "cluster")]
    if len(c):
        st = c[c.verdict == 2].subtype.value_counts().to_dict()
        s4 = c[c.verdict == 4].subtype.value_counts().to_dict() if "subtype" in c.columns else {}
        pr(f"  {ds:5s} cluster verdict-2 subtypes (all outputs): {st}; verdict-4 subtypes: {s4}")

# ------------------------------------------------------------------ insert
def cls3(k):
    if k in ("0", "1", "2"):
        return "<=2"
    if k in ("3", "4", ">=5"):
        return ">=3"
    if k.startswith("alpha not"):
        return "no alpha within 15"
    return "no L15"


pr("")
pr("STEP 3: L15 -> ALPHA-BLOCK INSERT (unmarked genes between L15 and the first of S13 / S11 / S4 / rpoA / L17; output-independent)")
INS = {}
for ds in ("lut", "srvp"):
    p = W / ds / "insert4.parquet"
    if not p.exists():
        continue
    I = pd.read_parquet(p)
    E = pd.read_parquet(W / ds / "ext4_union.parquet")
    I = I.merge(E[["run_id", "Z_std_resolved", "WA_std_resolved", "DH_std_resolved", "Z_resolved", "WA_resolved"]], on="run_id")
    INS[ds] = I
    order = ["0", "1", "2", "3", "4", ">=5", "alpha not within 15 (contig end within 15)", "alpha not within 15 (contig continues)",
             "no L15 in run"]
    pr(f"  {ds} (Bacteria runs {len(I)}):")
    for k in order:
        y = I[I.ins_class == k]
        if not len(y):
            continue
        pr(f"    {k:44s} runs {len(y):4d} ({len(y) / len(I):5.1%})  standing resolved Z {int(y.Z_std_resolved.sum()):4d} "
           f"({y.Z_std_resolved.mean():5.1%}) WA {int(y.WA_std_resolved.sum()):4d} ({y.WA_std_resolved.mean():5.1%})  cluster Z "
           f"{int(y.Z_resolved.sum())} WA {int(y.WA_resolved.sum())}")
    f = I[I.cls == "alpha found"]
    pr(f"    unmarked insert median {f.insert_unmarked.median()}, mean {f.insert_unmarked.mean():.2f} (runs with alpha found {len(f)})")
    pr("    commonest insert contents: " + str(f.insert_genes.value_counts().head(12).to_dict()))
    if "gtype" in I.columns:
        pr("    by genome type: " + str(I.groupby("gtype").ins_class.value_counts().unstack(fill_value=0).to_dict("index")))
    if "phylum" in I.columns:
        tx = I.assign(c3=I.ins_class.map(cls3)).groupby("phylum").agg(runs=("run_id", "size"), le2=("c3", lambda s: int((s == "<=2").sum())),
                                                                       ge3=("c3", lambda s: int((s == ">=3").sum())),
                                                                       Z=("Z_std_resolved", "sum"))
        pr("    description by phylum (runs, insert <= 2, insert >= 3, Z standing resolved): " +
           str({k: tuple(int(x) for x in v) for k, v in tx.sort_values("runs", ascending=False).head(10).iterrows()}))
        if ds == "lut":
            tc = I.assign(c3=I.ins_class.map(cls3)).groupby("tclass").agg(runs=("run_id", "size"), le2=("c3", lambda s: int((s == "<=2").sum())),
                                                                           Z=("Z_std_resolved", "sum"))
            pr("    lut by class (runs, insert <= 2, Z standing resolved): " + str({k: tuple(int(x) for x in v) for k, v in tc.iterrows()}))


if "lut" in INS and "srvp" in INS:
    pr("  CRITERION (predeclared): within-class standing resolved rates, and SR-VP rates reweighted to lut's class distribution")
    for n in ("Z", "WA"):
        col = f"{n}_std_resolved"
        a = INS["lut"].assign(c3=INS["lut"].ins_class.map(cls3))
        b = INS["srvp"].assign(c3=INS["srvp"].ins_class.map(cls3))
        ra = a.groupby("c3")[col].agg(["mean", "size"])
        rb = b.groupby("c3")[col].agg(["mean", "size"])
        within = []
        for k in ra.index.union(rb.index):
            ma = ra["mean"].get(k, np.nan)
            mb = rb["mean"].get(k, np.nan)
            within.append((k, ma, int(ra["size"].get(k, 0)), mb, int(rb["size"].get(k, 0))))
        wa = ra["size"] / ra["size"].sum()
        rew = float(sum(wa.get(k, 0) * rb["mean"].get(k, np.nan) for k in wa.index if k in rb.index and np.isfinite(rb["mean"].get(k, np.nan))))
        cover = float(sum(wa.get(k, 0) for k in wa.index if k in rb.index))
        obs = float(a[col].mean())
        pr(f"    {n}: " + "; ".join(f"{k}: lut {ma:.1%} (n {na}) vs SR-VP {mb:.1%} (n {nb})" for k, ma, na, mb, nb in within))
        diffs = [abs(ma - mb) for k, ma, na, mb, nb in within if na > 0 and nb > 0 and k in ("<=2", ">=3", "no alpha within 15")]
        pr(f"       lut observed {obs:.1%}; SR-VP observed {b[col].mean():.1%}; SR-VP reweighted to lut classes {rew:.1%} "
           f"(classes covering {cover:.1%} of lut runs); max within-class gap (<= 2, >= 3, no alpha within 15) "
           f"{max(diffs) if diffs else float('nan'):.1%} -> accounts for: "
           f"{'YES' if diffs and max(diffs) <= 0.10 and abs(rew - obs) <= 0.10 else 'NO'}")

Path(sys.argv[2]).write_text("\n".join(out) + "\n")
print("\n".join(out))
