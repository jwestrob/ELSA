#!/usr/bin/env python
"""Part-2 even table: seven bars per output. usage: table2.py RESULTS_DIR"""
import json
import sys
from pathlib import Path

R = Path(sys.argv[1])
C = ("DefenseFinder", "TXSScan", "CRISPRCasTyper")
DH = dict(summed=.4270, f1={"DefenseFinder": .744, "TXSScan": .548, "CRISPRCasTyper": .479})


def f(x, n=3, s=False):
    if x is None:
        return "-"
    t = f"{x:+.{n}f}" if s else f"{x:.{n}f}"
    return t.replace("0.", ".", 1)


def iv(v):
    return f"{f(v[0], 3, True)} [{f(v[1], 3, True)},{f(v[2], 3, True)}]"


rows = []
for arm, tag in (("F", "dpann_group"), ("Z", "dpann_Z"), ("A", "dpann_A"), ("Zs", "dpann_Zs"), ("As", "dpann_As"),
                 ("FZ", "dpann_FZ")):
    fn = R / f"even_{tag}_p2.json"
    if not fn.exists():
        continue
    d = json.load(open(fn))
    ex = d.get("extra", {})
    for nm, o in d["outputs"].items():
        s, r = o["summary_even"], o["ribo"]["all"]
        v = o["vs_DH_even"]
        bars = dict(summed=v["summed"][0] >= -0.01, **{c[:4]: v[c]["d_f1"][0] >= -0.02 for c in C},
                    rec=r["recovered"] >= 1650, res=r["resolved"] > 1512, parv=int(r["parvarchaeaceae"].split("/")[0]) >= 110)
        e = ex.get("outputs", {}).get(nm, {})
        rows.append((arm, nm, r, o, bars, e, ex))
print(f"{'arm':3s} {'output':14s} {'rec/res':>9s} {'d res vs DH':>17s} {'d res vs p1L4b':>16s} parv  {'summed':>6s} "
      f"{'d vs DH':>22s} {'d vs part-1 L4b':>22s} {'F1 DF/TXSS/CCT':>15s} {'dF1 CCT vs DH':>22s} bars  bprec DF/TX/CC  genes  CasOut")
for arm, nm, r, o, bars, e, ex in rows:
    s = o["summary_even"]
    rr = o["ribo_resolved_vs_DH"]
    pr1 = o.get("vs_p1L4b_even", {})
    mo = e.get("members_outside_every_locus", {})
    print(f"{arm:3s} {nm:14s} {r['recovered']:>4d}/{r['resolved']:<4d} {f'{rr[0]:+.0f} [{rr[1]:+.0f},{rr[2]:+.0f}]':>17s} "
          f"{'':>16s} {r['parvarchaeaceae'].split('/')[0]:>4s}  {f(s['summed_DF_TXSS_excess']):>6s} "
          f"{iv(o['vs_DH_even']['summed']):>22s} {iv(pr1['summed']) if pr1 else '-':>22s} "
          f"{'/'.join(f(s[c]['f1']) for c in C):>15s} {iv(o['vs_DH_even']['CRISPRCasTyper']['d_f1']):>22s} "
          f"{sum(bars.values())}/7  {'/'.join(f(s[c]['median_bprec_covered'], 2) for c in C)}  {f(o['genes_in_loci_frac'])} "
          f"{f(mo.get('CRISPRCasTyper'))}  56: resolved {e.get('part1_under56_resolved')}")
    print(" " * 19 + "failed bars: " + (", ".join(k for k, v in bars.items() if not v) or "none")
          + f"; dF1 DF {iv(o['vs_DH_even']['DefenseFinder']['d_f1'])} TXSS {iv(o['vs_DH_even']['TXSScan']['d_f1'])}")
seen = set()
for arm, nm, r, o, bars, e, ex in rows:
    if arm in seen:
        continue
    seen.add(arm)
    print(f"{arm}: part-1 L4b under-covered runs {ex.get('part1_L4b_under_covered')}, fully bonded in this arm "
          f"{ex.get('of_those_fully_bonded_in_this_arm')}, best_frac under arm L1 q50 {f(ex.get('of_those_best_frac_q50_under_arm_L1'))}")
for fn in (R / "pairs_p2a.json", R / "pairs_p2.json"):
    if not fn.exists():
        continue
    for k, v in json.load(open(fn)).items():
        print("PAIR", k, "summed", iv(v["summed"]), "dF1", " ".join(f"{c[:4]} {iv(v[c]['d_f1'])}" for c in C),
              "resolved", v["ribo_resolved"])
