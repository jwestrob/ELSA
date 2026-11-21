#!/usr/bin/env python
"""ribo_audit tables: reads the work/ outputs of audit.py and writes tables.txt (all numbers quoted in REPORT.txt).

usage: tables.py WORKDIR OUTFILE
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

W = Path(sys.argv[1])
OUT = open(sys.argv[2], "w")
ORDER = ("TOK", "WCP", "EMB")
CATS = ["DEF_SCORING", "DEF_NEARBY_RIBO", "DEF_OTHER_KIND", "DEF_ASSOC_FLANK", "DEF_JUNCTION", "DATA_CONTIG_EDGE",
        "DATA_GENE_CALL", "METHOD_SPLIT", "METHOD_UNCOVERED", "METHOD_EXTENSION", "OTHER"]
KIND = {c: ("definition" if c.startswith("DEF") else "data" if c.startswith("DATA") else
            "method" if c.startswith("METHOD") else "other") for c in CATS}
VARS = {"pre": "label", "corr": "c_label", "strict": "x_label"}
VNAME = {"pre": "PREDECLARED", "corr": "CORRECTED (post hoc: OTHER-KIND = marker atypical for family or length > 2x)",
         "strict": "STRICT (post hoc: corrected + nearby Ribosomal_* genes admissible only if run-associated)"}


def p(*a):
    print(*a, file=OUT)


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan, np.nan)
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return ph, max(0.0, c - h), min(1.0, c + h)


def fmt(k, n):
    ph, lo, hi = wilson(k, n)
    return f"{k:>3}/{n:<3} {ph:5.1%} [{lo:5.1%},{hi:5.1%}]"


cases = pd.read_csv(W / "cases.tsv", sep="\t")
cen = pd.read_parquet(W / "census.parquet")
rw = pd.read_parquet(W / "random_windows.parquet")
full = json.load(open(W / "full.json"))
ext = pd.read_csv(W / "ext_genes.tsv", sep="\t")
lay = pd.read_csv(W / "layouts.tsv", sep="\t")

# ------------------------------------------------------------------------------------------ 1. sample categories
for v, col in VARS.items():
    p("=" * 118)
    p(f"SAMPLE CATEGORY TABLE, {VNAME[v]}")
    p("  count/n  share [Wilson 95%]; U = 40 unresolved per output; A3 = 20 runs unresolved in all three, per output")
    hdr = f"  {'category':<18}" + "".join(f"{s + '-' + o:>30}" for s in ("U",) for o in ORDER) + "".join(
        f"{'A3-' + o:>30}" for o in ORDER)
    p(hdr)
    groups = [("U", o) for o in ORDER] + [("A3", o) for o in ORDER]
    for c in CATS:
        row = f"  {c:<18}"
        anyk = False
        for s, o in groups:
            d = cases[(cases["sample"] == s) & (cases.output == o)]
            k = int((d[col] == c).sum())
            anyk |= k > 0
            row += f"{fmt(k, len(d)):>30}"
        if anyk:
            p(row)
    for kind in ("definition", "data", "method"):
        row = f"  {'= ' + kind:<18}"
        for s, o in groups:
            d = cases[(cases["sample"] == s) & (cases.output == o)]
            k = int(d[col].map(KIND).eq(kind).sum())
            row += f"{fmt(k, len(d)):>30}"
        p(row)
    row = f"  {'= method+SCORING':<18}"
    for s, o in groups:
        d = cases[(cases["sample"] == s) & (cases.output == o)]
        k = int((d[col].map(KIND).eq("method") | (d[col] == "DEF_SCORING")).sum())
        row += f"{fmt(k, len(d)):>30}"
    p(row)
    # pooled over outputs (U samples, 120 case evaluations)
    d = cases[cases["sample"] == "U"]
    p(f"  pooled U (120): definition {fmt(int(d[col].map(KIND).eq('definition').sum()), len(d))}; "
      f"method {fmt(int(d[col].map(KIND).eq('method').sum()), len(d))}; "
      f"data {fmt(int(d[col].map(KIND).eq('data').sum()), len(d))}")
    p("")

# ------------------------------------------------------------------------------------------ 2. census cross-check
p("=" * 118)
p("CENSUS OF ALL UNRESOLVED RUNS (same automated rules; cross-check of the sample)")
for v, col in VARS.items():
    p(f"  {v}:")
    for o in ORDER:
        d = cen[(cen.output == o) & ~cen.std_ok]
        vc = d[col].value_counts()
        kinds = d[col].map(KIND).value_counts()
        n = len(d)
        p(f"    {o} n={n}: " + ", ".join(f"{k} {vc[k]}" for k in CATS if k in vc))
        p(f"         definition {fmt(int(kinds.get('definition', 0)), n)}  data {fmt(int(kinds.get('data', 0)), n)}  "
          f"method {fmt(int(kinds.get('method', 0)), n)}  method+SCORING "
          f"{fmt(int(kinds.get('method', 0)) + int(vc.get('DEF_SCORING', 0)), n)}")
p("")
p("  Failure type at the standing rule (census): coverage only / precision only / both")
for o in ORDER:
    d = cen[(cen.output == o) & ~cen.std_ok]
    vc = d.fail_type.value_counts()
    p(f"    {o}: {vc.get('coverage', 0)} / {vc.get('precision', 0)} / {vc.get('both', 0)}")
p("")
p("  Labels by stratum (census, corrected rubric): definition / data / method")
for o in ORDER:
    d = cen[(cen.output == o) & ~cen.std_ok].copy()
    runs = pd.read_parquet(W / "nearby.parquet")[["run_id", "stratum"]]
    d = d.merge(runs, on="run_id")
    d["kind"] = d.c_label.map(KIND)
    t = d.groupby(["stratum", "kind"]).size()
    p(f"    {o}: " + "; ".join(f"{st} {int(t.get((st, 'definition'), 0))}/{int(t.get((st, 'data'), 0))}/"
                             f"{int(t.get((st, 'method'), 0))}" for st in sorted(d.stratum.unique())))
p("")

# ------------------------------------------------------------------------------------------ 3. predeclared vs corrected
p("=" * 118)
p("WHERE THE PREDECLARED DEF_OTHER_KIND CASES GO UNDER THE CORRECTED AND STRICT RULES (census)")
for o in ORDER:
    d = cen[(cen.output == o) & (cen.label == "DEF_OTHER_KIND")]
    p(f"  {o} ({len(d)}): corrected " + str(d.c_label.value_counts().to_dict()) + " | strict " +
      str(d.x_label.value_counts().to_dict()))
p("  Predeclared DEF_NEARBY_RIBO under strict (nearby marked genes all non-associated in "
  + ", ".join(f"{o} {int(((cen.output == o) & (cen.label == 'DEF_NEARBY_RIBO') & (cen.n_r5_marked_out_assoc == 0)).sum())}"
              f"/{int(((cen.output == o) & (cen.label == 'DEF_NEARBY_RIBO')).sum())}" for o in ORDER) + "):")
for o in ORDER:
    d = cen[(cen.output == o) & (cen.c_label == "DEF_NEARBY_RIBO")]
    p(f"    {o} ({len(d)}): strict " + str(d.x_label.value_counts().to_dict()))
p("")

# ------------------------------------------------------------------------------------------ 4. SCORING detail
p("=" * 118)
p("DEF_SCORING DETAIL (census): runs where a resolving locus exists but the largest-overlap locus is chosen")
for o in ORDER:
    d = cen[(cen.output == o) & (cen.label == "DEF_SCORING")]
    if len(d) == 0:
        continue
    ext_len = (np.maximum(0, d.std_b - (d.std_a + 0)) + 1)  # best length
    p(f"  {o} {len(d)}: best-locus precision median {d.std_prec.median():.3f} (q10 {d.std_prec.quantile(.1):.3f}); "
      f"best extends beyond strict-admissible flank in {int(((d.x_s3_prec < 0.8) | d.x_s3_prec.isna()).sum())} "
      f"(so without the scoring step the strict chain labels: {d.x_s3_ok.map({True: 'resolved', False: 'unresolved'}).value_counts().to_dict()})")
# nested check on sampled scoring cases


def parse(lr):
    out = []
    for t in str(lr).split(";"):
        if ".." in t:
            a, b = t.split("..")
            out.append((int(a), int(b)))
    return out


rows = []
for r in cases[(cases.label == "DEF_SCORING")].itertuples():
    n = r.run_genes
    bl = parse(r.best_rel)[0]
    good = []
    for (a, b) in parse(r.loci_rel):
        ov = max(0, min(b, n - 1) - max(a, 0) + 1)
        if ov / n >= 0.8 and ov / (b - a + 1) >= 0.8:
            good.append((a, b))
    nested = any(a >= bl[0] and b <= bl[1] for (a, b) in good)
    rows.append((r.output, nested, len(good)))
t = pd.DataFrame(rows, columns=["output", "nested", "n_good"])
p("  sampled DEF_SCORING cases: resolving locus nested inside the chosen best locus: " +
  str(t.groupby("output").nested.agg(['sum', 'size']).to_dict('index')))
p("")

# ------------------------------------------------------------------------------------------ 5. controls vs failures
p("=" * 118)
p("FEATURES IN FAILURES (U samples) AND CONTROLS (C samples, resolved, phylum-matched); count/n share [95%]")
feats = [("nearby Ribosomal_* gene <=5 outside (f_nearby)", lambda d: d.f_nearby),
         ("member flagged OTHER-KIND, predeclared rule", lambda d: d.n_other_kind > 0),
         ("member flagged OTHER-KIND, corrected rule", lambda d: d.c_n_other_kind > 0),
         ("non-recurring junction (break) inside run", lambda d: d.n_breaks_run > 0),
         ("run touches contig end", lambda d: d.f_edge),
         ("gene-call anomaly inside run", lambda d: d.n_anom_run > 0),
         ("best locus extends beyond run", lambda d: d.f_ext),
         ("best locus extends beyond corrected adm", lambda d: d.f_ext_beyond_s3),
         ("another locus not inside best touches run", lambda d: d.f_multi)]
p(f"  {'feature':<46}" + "".join(f"{s + '-' + o:>28}" for o in ORDER for s in ("U", "C")))
for name, f in feats:
    row = f"  {name:<46}"
    for o in ORDER:
        for s in ("U", "C"):
            d = cases[(cases["sample"] == s) & (cases.output == o)]
            row += f"{fmt(int(f(d).sum()), len(d)):>28}"
    p(row)
for o in ORDER:
    for s in ("U", "C"):
        d = cases[(cases["sample"] == s) & (cases.output == o)]
        mao = d.ext_M_a_o.str.split("/", expand=True).astype(int).sum()
        p(f"  {s}-{o}: extension genes of best loci: median {d.ext_genes.median():.0f} per case; total M/a/o "
          f"{mao[0]}/{mao[1]}/{mao[2]}; median precision {d.std_prec.median():.3f}")
p("")
p("  Same features over ALL runs (census): unresolved vs resolved")
fe2 = [("f_nearby", lambda d: d.f_nearby), ("OTHER-KIND pre", lambda d: d.n_other_kind > 0),
       ("OTHER-KIND corr", lambda d: d.c_n_other_kind > 0), ("break", lambda d: d.n_breaks_run > 0),
       ("contig end", lambda d: d.f_edge), ("anomaly", lambda d: d.n_anom_run > 0)]
for o in ORDER:
    d = cen[cen.output == o]
    p(f"    {o}: " + "; ".join(f"{nm} {f(d[~d.std_ok]).mean():.1%} vs {f(d[d.std_ok]).mean():.1%}" for nm, f in fe2))
p("")

# ------------------------------------------------------------------------------------------ 6. extension families / domains
p("=" * 118)
p("COMMONEST OUTSIDE-OF-RUN FAMILIES IN BEST-LOCUS EXTENSIONS OF FAILURES (U + A3, unique genes)")
p("  n_other / E / Lambda: flank association with the case's own group removed (median over genes); class a = run-")
p("  associated (n' >= 3, Lambda >= 10), M = Ribosomal_* gene, o = neither")
f = ext[ext["sample"].isin(["U", "A3"])].drop_duplicates("gid").copy()
f["pf"] = f.pfam.fillna("(no Pfam hit)")
p(f"  unique extension genes {len(f)}: " + str(f.cls.value_counts().to_dict()))
by_out = ext[ext["sample"].isin(["U", "A3"])].groupby("output").cls.value_counts().unstack(fill_value=0)
p("  by output (gene-case pairs): " + "; ".join(f"{o} M/a/o {by_out.loc[o].get('M', 0)}/{by_out.loc[o].get('a', 0)}/"
                                               f"{by_out.loc[o].get('o', 0)}" for o in ORDER))
t = f.groupby(["cls", "token"]).agg(genes=("gid", "size"), cases=("case_id", "nunique"),
                                     strata=("case_id", lambda s: ""), n_other=("n_other", "median"),
                                     E=("E_other", "median"), Lam=("Lambda", "median"),
                                     pfam=("pf", lambda s: s.value_counts().index[0])).reset_index()
t = t.sort_values(["genes"], ascending=False)
p(f"  {'cls':<4}{'token':>8}{'genes':>7}{'n_oth':>7}{'E':>8}{'Lambda':>9}  Pfam (commonest architecture)")
for cl in ("a", "M", "o"):
    for r in t[t.cls == cl].head(15).itertuples():
        p(f"  {cl:<4}{r.token:>8}{r.genes:>7}{r.n_other:>7.0f}{r.E:>8.2f}{r.Lam:>9.1f}  {r.pfam[:70]}")
p("")
p("  Pfam domains on extension genes (count of genes carrying the domain; top 25), split a / M / o")
dd = f.assign(dom=f.pfam.fillna("").str.split(",")).explode("dom")
dd = dd[dd.dom != ""].drop_duplicates(["gid", "dom"])
tab = dd.groupby(["dom", "cls"]).size().unstack(fill_value=0)
tab["tot"] = tab.sum(axis=1)
for dom, r in tab.sort_values("tot", ascending=False).head(25).iterrows():
    p(f"    {dom:<28} a {r.get('a', 0):>4}  M {r.get('M', 0):>4}  o {r.get('o', 0):>4}")
p(f"  extension genes without any Pfam hit: {int(f.pfam.isna().sum())} of {len(f)} "
  f"(a {int(f[f.cls == 'a'].pfam.isna().sum())}, M {int(f[f.cls == 'M'].pfam.isna().sum())}, "
  f"o {int(f[f.cls == 'o'].pfam.isna().sum())})")
p("")

# ------------------------------------------------------------------------------------------ 7. nearby runs (full set)
p("=" * 118)
p("NEARBY RUNS, ALL 1,686 RUNS")
p(f"  pairs of g2_m16 runs on one contig: {full['m16_pairs_same_contig']} (no genome holds two); within 5 genes: "
  f"{full['m16_pairs_within5']}")
n = full["runs"]
p(f"  runs with another Ribosomal_* gene <= 5 genes outside the span: {fmt(full['runs_d_any_le5'], n)}; <= 10: "
  f"{fmt(full['runs_d_any_le10'], n)}")
p(f"  runs with a g2 cluster of >= 2 marked genes <= 5 genes outside: {fmt(full['runs_d_ge2_le5'], n)}; <= 10: "
  f"{fmt(full['runs_d_ge2_le10'], n)}")
p(f"  runs with a g2 run of 8-15 marked genes <= 5 genes outside: {full['runs_d_m8_le5']}; <= 10: {full['runs_d_m8_le10']}")
p("  by stratum (<= 5, any marked gene): " + "; ".join(f"{k} {v}/{full['by_stratum_runs'][k]}"
                                                       for k, v in full["by_stratum_d_any_le5"].items()))
p("")

# ------------------------------------------------------------------------------------------ 8. sensitivity
p("=" * 118)
p("SENSITIVITY: RESOLVED RUNS (of 1,686; Parvarchaeaceae of 114) UNDER ALTERNATIVE DEFINITIONS; random-window chance")
runs_parv = pd.read_parquet(W / "nearby.parquet")[["run_id", "stratum"]]
cen2 = cen.merge(runs_parv, on="run_id")
defs = [("standing (best locus, run span)", lambda d: d.std_ok),
        ("any locus (brief's wording)", lambda d: d.std_ok | d.any_ok),
        ("strict assoc-flank precision (x S3)", lambda d: d.std_ok | d.x_s3_ok),
        ("  + junction pieces (x S4)", lambda d: d.std_ok | d.x_s3_ok | (d.x_label == "DEF_JUNCTION")),
        ("  + any locus", lambda d: d.std_ok | d.x_s3_ok | (d.x_label == "DEF_JUNCTION") | d.any_ok),
        ("corrected chain incl. nearby marked (c S3)", lambda d: d.std_ok | d.c_s3_ok),
        ("  + junction pieces (c S4)", lambda d: d.std_ok | d.c_s3_ok | (d.c_label == "DEF_JUNCTION")),
        ("predeclared chain (S3)", lambda d: d.std_ok | d.s3_ok)]
p(f"  {'definition':<44}" + "".join(f"{o:>16}" for o in ORDER) + "".join(f"{'Parv ' + o:>11}" for o in ORDER))
for nm, fn in defs:
    row = f"  {nm:<44}"
    for o in ORDER:
        d = cen2[cen2.output == o]
        row += f"{int(fn(d).sum()):>16,}"
    for o in ORDER:
        d = cen2[(cen2.output == o) & (cen2.stratum == "Parvarchaeaceae")]
        row += f"{int(fn(d).sum()):>11}"
    p(row)
lost = {o: int((cen[cen.output == o].std_ok & ~cen[cen.output == o].x_s3_ok).sum()) for o in ORDER}
p(f"  standing-resolved runs that the strict chain alone would not resolve (kept by the OR): {lost}")
p("  random windows (5 length-matched draws per run, 8,325 windows; fraction 'resolved'):")
rwt = rw.groupby("output")[["std_ok", "any_ok", "x_s3_ok", "s3_ok"]].mean()
for o in ORDER:
    r = rwt.loc[o]
    p(f"    {o}: standing {r.std_ok:.4f}; any locus {r.any_ok:.4f}; strict chain {r.x_s3_ok:.4f}; corrected chain "
      f"{r.s3_ok:.4f}")
p("")

# ------------------------------------------------------------------------------------------ 9. per-case table index
p("=" * 118)
p("UNCLASSIFIABLE / LOW-CONFIDENCE SAMPLED CASES")
low = cases[cases[["label", "c_label", "x_label"]].apply(lambda r: any(str(x).startswith("DATA") for x in r), axis=1)]
for r in low.itertuples():
    p(f"  {r.case_id} run {r.run_id} {r.stratum}: pre {r.label} / corr {r.c_label} / strict {r.x_label}; "
      f"bf {r.std_bf} prec {r.std_prec}")
OUT.close()
