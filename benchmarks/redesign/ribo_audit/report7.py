#!/usr/bin/env python
"""ribo_audit part 7: tables for REPORT7 from score6.py outputs (no rescoring; reads score6.json, score6_runs.parquet,
cases6_*.tsv and the ribosomal receipt). usage: report7.py SCOREDIR RIBODIR GENOMES.parquet OUTFILE"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

S, R, GM, OUT = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], Path(sys.argv[4])
res = json.load(open(S / "score6.json"))
df = pd.read_parquet(S / "score6_runs.parquet")
L = []
pr = L.append
NAME = {"Z": "PRIMARY embedding-list bond tree", "WA": "SECONDARY family-bond adaptive tree", "DH": "ENGINE reference (recurrence engine DH)"}
x = df[df.definition == "a"]
st = df[df.definition == "standing"]
doms = [d for d in ("Bacteria", "Archaea") if d in set(x.dom)]


def boot(d, n=2000, seed=17):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    s = d[idx].sum(1)
    return float(d.sum()), float(np.quantile(s, .025)), float(np.quantile(s, .975))


pr("BARS (frozen definition (a), predeclared6.json; denominators = all runs, below-minimum runs counted unresolved / not recovered)")
for n in ("Z", "WA", "DH"):
    pr(f"  {NAME[n]}")
    rec = int(x[f"{n}_recovered"].sum())
    pr(f"    R1 recovered overall {rec} / {len(x)} = {rec / len(x):.2%}  bar >= 97%  {'PASS' if rec / len(x) >= .97 else 'FAIL'}  "
       f"(" + "; ".join(f"{d} {int(x[x.dom == d][f'{n}_recovered'].sum())} / {int((x.dom == d).sum())}" for d in doms) + ")")
    for d, bar, tag in (("Bacteria", .90, "R2"), ("Archaea", .85, "R3")):
        y = x[x.dom == d]
        if len(y):
            v = int(y[f"{n}_resolved"].sum())
            pr(f"    {tag} resolved {d} {v} / {len(y)} = {v / len(y):.2%}  bar >= {bar:.0%}  {'PASS' if v / len(y) >= bar else 'FAIL'}")
        else:
            pr(f"    {tag} resolved {d}: no runs")
    if n != "DH":
        b = res["bars"]["primary" if n == "Z" else "secondary"]["R4"]
        tot, lo, hi = boot(x[f"{n}_resolved"].astype(int).to_numpy() - x["DH_resolved"].astype(int).to_numpy())
        assert abs(tot - b["difference"]) < 1e-9 and abs(lo - b["ci95"][0]) < 1e-9
        pr(f"    R4 resolved minus engine {tot:+.0f} runs, 95% interval [{lo:+.0f}, {hi:+.0f}] (paired bootstrap over {len(x)} runs, "
           f"2,000 resamples, seed 17)  bar: interval above 0  {'PASS' if lo > 0 else 'FAIL'}")
    else:
        pr("    R4 not applicable (reference itself)")
    ch = res["definitions"]["a"]
    r5 = []
    for d in doms:
        c = ch[d][n]["chance"]
        r5.append(c <= .05)
        pr(f"    R5 chance {d} {c:.2%}  bar <= 5%  {'PASS' if c <= .05 else 'FAIL'}")
    if n != "DH":
        b = res["bars"]["primary" if n == "Z" else "secondary"]
        pr(f"    OVERALL (all five) {'PASS' if b['PASS'] else 'FAIL'}")
    else:
        ok = (rec / len(x) >= .97) and all(r5) and all(
            x[x.dom == d][f"{n}_resolved"].mean() >= bar for d, bar in (("Bacteria", .90), ("Archaea", .85)) if (x.dom == d).any())
        pr(f"    R1-R3 + R5 for the reference: {'PASS' if ok else 'FAIL'} (R4 not applicable)")

pr("")
pr("RUN COUNTS")
rec = json.load(open(R / "receipt.json"))
gm = pd.read_parquet(GM)
dom = gm.domain if "domain" in gm.columns else gm.tax.str.split(";").str[0].str.replace("d__", "")
pr(f"  genomes {len(gm)} ({dom.value_counts().to_dict()}); marked genes {rec['marked_genes']}; runs {rec['runs']}")
pr(f"  g2 runs with >= 16 marked (scored): {len(x)} ({x.dom.value_counts().to_dict()}) in {rec['g2_m16_genomes']} genomes; "
   f"genomes with no scored run {rec['genomes_without_g2_m16']} ({rec['genomes_without_g2_m16_by_domain']})")
bm = x[~x.kept]
pr(f"  below-minimum runs under (a), kept as unresolved: {len(bm)} ({bm.dom.value_counts().to_dict()}); pieces (marked genes): "
   + ", ".join(f"run {r.run_id}: {r.pieces_marked}" for r in bm.itertuples()))
pr(f"  runs with > 1 target under (a): {int((x.n_targets > 1).sum())}; closed (1 contig) / multi-contig bacterial runs: "
   f"{int(((x.dom == 'Bacteria') & x.closed).sum())} / {int(((x.dom == 'Bacteria') & ~x.closed).sum())}")

pr("")
pr("STANDING DEFINITION BESIDE THE FROZEN DEFINITION (resolved / any fitting locus / recovered; chance)")
for d in doms + ["all"]:
    for n in ("Z", "WA", "DH"):
        a = res["definitions"]["a"][d][n]
        s = res["definitions"]["standing"][d][n]
        p4 = res["definitions"]["part4"][d][n]
        ch_ = (lambda e: f"{e['chance']:.2%}" if e['chance'] is not None else "-")
        pr(f"  {d:9s} {n:2s}: standing {s['resolved']} / {s['anyfit']} / {s['recovered']} chance {ch_(s)}; part 4 {p4['resolved']}; "
           f"frozen (a) {a['resolved']} / {a['anyfit']} / {a['recovered']} chance {ch_(a)}; accurate reading {s.get('accurate_reading_resolved')}")

pr("")
pr("VERDICTS FOR EVERY UNRESOLVED RUN OF THE PRIMARY (part-2 strict procedure; below-minimum runs listed separately)")
for n in ("Z", "WA", "DH"):
    p = S / f"cases6_{n}.tsv"
    c = pd.read_csv(p, sep="\t") if p.exists() and p.stat().st_size > 1 else pd.DataFrame(columns=["run_id", "dom", "verdict"])
    for d in doms:
        y = c[c.dom == d]
        vc = y.drop_duplicates(["run_id", "verdict"]).verdict.astype(str).value_counts()
        pr(f"  {n:2s} {d:9s} unresolved runs {y.run_id.nunique():4d}: " + " ".join(f"v{k} {int(vc.get(str(k), 0))}" for k in range(1, 8))
           + f" below-minimum {int(vc.get('below_minimum', 0))}")

pr("")
pr("PHYLUM TABLE (frozen definition (a); runs, resolved Z / WA / DH, below minimum)")
for r in sorted(res["phylum"], key=lambda r: (r["dom"], -r["runs"])):
    pr(f"  {r['dom']:9s} {str(r['phylum']):30s} {r['runs']:5d} {r['Z']:5d} {r['WA']:5d} {r['DH']:5d} {r['below']:3d}")
pr("")
pr("CLOSED VS MULTI-CONTIG BACTERIAL GENOMES (a): " + json.dumps(res["closed_vs_multi"]))
OUT.write_text("\n".join(L) + "\n")
print("\n".join(L))
