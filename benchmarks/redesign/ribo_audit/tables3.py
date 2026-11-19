#!/usr/bin/env python
"""ribo_audit part 3 tables: reads work3/ (and work3/loose/) outputs of bio3.py, writes tables3.txt.
usage: tables3.py WORK3DIR OUTFILE
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

W = Path(sys.argv[1])
OUT = open(sys.argv[2], "w")
GT = ("Bact_closed", "Bact_multi", "Archaea")
VN = {1: "1 split one locus", 2: "2 run glues >= 2 loci", 3: "3 output correct, larger than run",
      4: "4 output extends past the locus", 5: "5 fitting locus, scorer picks larger", 6: "6 data", 7: "7 cannot tell"}


def p(*a):
    print(*a, file=OUT)


def wil(k, n, z=1.96):
    if n == 0:
        return "-"
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return f"{100 * ph:.1f}% [{100 * max(0, c - h):.1f}, {100 * min(1, c + h):.1f}]"


P = pd.read_parquet(W / "pair_shares.parquet")
R = pd.read_parquet(W / "run_order.parquet")
S = pd.read_parquet(W / "stretch.parquet")
rep = json.load(open(W / "repro3.json"))
for var, d in (("strict", W), ("loose", W / "loose")):
    cases = pd.read_csv(d / "cases3.tsv", sep="\t")
    bio = pd.read_parquet(d / "bio3_scores.parquet")
    junc = {x["case_id"]: x["junctions"] for x in json.load(open(d / "junctions3.json"))}
    p("=" * 116)
    p(f"VERDICTS, {var.upper()} reading (part-2 rules{'; loose = 500 nt for every same-strand pair, POST HOC in part 2' if var == 'loose' else ''})")
    for out in ("PRIMARY", "SECONDARY"):
        c = cases[cases.output == out]
        p(f"  {out}: {'':<38}" + "".join(f"{g:>14}" for g in GT) + f"{'Bacteria':>12}")
        for v in range(1, 8):
            row = [int(((c.gtype == g) & (c.verdict == v)).sum()) for g in GT]
            p(f"    {VN[v]:<44}" + "".join(f"{x:>14}" for x in row) + f"{row[0] + row[1]:>12}")
        row = [int((c.gtype == g).sum()) for g in GT]
        p(f"    {'unresolved':<44}" + "".join(f"{x:>14}" for x in row) + f"{row[0] + row[1]:>12}")
        for v in (2, 4):
            t = c[(c.verdict == v) & c.gtype.str.startswith("Bact")].groupby(["gtype", "subtype"]).size()
            p(f"    v{v} subtypes (Bacteria): {t.to_dict()}")
    p("  ACCURATE SCORING (runs right / runs):")
    for out in ("PRIMARY", "SECONDARY", "ENGINE"):
        b = bio[bio.output == out]
        p(f"    {out:<10}" + "  ".join(f"{g} {int(b[b.gtype == g].bio_ok.sum())}/{int((b.gtype == g).sum())} (std {int(b[b.gtype == g].std_ok.sum())})"
                                       for g in GT) + f"  Bacteria {int(b[b.gtype != 'Archaea'].bio_ok.sum())}/{int((b.gtype != 'Archaea').sum())}")
    # rare junction at the decisive point
    p("  decisive junctions (verdicts 1 and 2) at a RARE marker pair (share < 0.10 of other same-domain groups):")
    for out in ("PRIMARY", "SECONDARY"):
        for v in (1, 2):
            c = cases[(cases.output == out) & (cases.verdict == v) & cases.gtype.str.startswith("Bact")]
            n_rare = n_tot = 0
            for r in c.itertuples():
                pr = P[P.run_id == r.run_id]
                hit = False
                for j in junc.get(r.case_id, []):
                    q = pr[(pr.rel_x <= j["rel"]) & (pr.rel_y > j["rel"])]
                    if len(q) and (q.share < 0.10).any():
                        hit = True
                n_rare += int(hit)
                n_tot += 1
            p(f"    {out} v{v} Bacteria: {n_rare} of {n_tot}")
p("=" * 116)
p("REPRODUCTION (standing): " + "; ".join(f"{o} " + ", ".join(f"{k} {v['recovered']}/{v['resolved']} of {v['runs']} (any-fit {v['any_fit']})"
                                                       for k, v in rep[o].items() if isinstance(v, dict)) for o in rep))
p("")
p("ORDER VARIABILITY INSIDE RUNS (output-independent; marker pairs = consecutive Ribosomal_* genes of the run)")
Rr = R.merge(S, on=["run_id", "gtype"])
for g in GT:
    x = Rr[Rr.gtype == g]
    pp = P[P.gtype == g]
    p(f"  {g:<12} runs {len(x)}: pair share median {np.nanmedian(pp.share):.3f} (q10 {np.nanquantile(pp.share, .1):.3f}); "
      f"run median share median {np.nanmedian(x.median_share):.3f}; run min share median {np.nanmedian(x.min_share):.3f}; "
      f"runs with >= 1 rare pair {int((x.n_rare >= 1).sum())} ({100 * (x.n_rare >= 1).mean():.1f}%); rare pairs per run mean {x.n_rare.mean():.2f}")
for dom, gs in (("Bacteria", ("Bact_closed", "Bact_multi")), ("Archaea", ("Archaea",))):
    x = R[R.gtype.isin(gs)]
    vc = x.core_order.value_counts()
    grp_orders = x.groupby("group").core_order.agg(lambda s: s.value_counts().index[0])
    p(f"  {dom}: core-marker orders distinct {len(vc)} over {len(x)} runs; modal order share {vc.iloc[0] / len(x):.3f}; "
      f"runs with all 18 core markers {int((x.n_core == 18).sum())}; distinct orders among those {x[x.n_core == 18].core_order.nunique()}; "
      f"groups {x.group.nunique()}, distinct modal orders over groups {grp_orders.nunique()}")
    full = x[x.n_core == 18].core_order.value_counts()
    p(f"    top orders (all 18 core): " + " | ".join(f"{n}x {o[:80]}" for o, n in full.head(3).items()))
p("  runs resolved vs unresolved by PRIMARY (Bacteria): share of runs with >= 1 rare pair")
std = pd.read_parquet(W / "std3_PRIMARY.parquet")[["run_id", "resolved"]]
xr = R.merge(std, on="run_id")
for g in ("Bact_closed", "Bact_multi"):
    a = xr[(xr.gtype == g) & xr.resolved]
    b = xr[(xr.gtype == g) & ~xr.resolved]
    p(f"    {g}: resolved {int((a.n_rare >= 1).sum())}/{len(a)} ({100 * (a.n_rare >= 1).mean():.1f}%), unresolved "
      f"{int((b.n_rare >= 1).sum())}/{len(b)} ({100 * (b.n_rare >= 1).mean():.1f}%)")
p("")
p("CONSERVED STRETCH AROUND THE RUN (identity pairs <= 2 apart in >= 25% of other same-domain groups holding both; window 40)")
for g in GT:
    x = S[S.gtype == g]
    tot = x.stretch_left + x.stretch_right
    p(f"  {g:<12} run length median {x.run_len.median():.0f}; genes added left / right median {x.stretch_left.median():.0f} / "
      f"{x.stretch_right.median():.0f}; total q25 / q50 / q75 {tot.quantile(.25):.0f} / {tot.median():.0f} / {tot.quantile(.75):.0f}; "
      f"reaching the 40-gene window edge on a side {int(((x.stretch_left >= 39) | (x.stretch_right >= 39)).sum())}")
OUT.close()
