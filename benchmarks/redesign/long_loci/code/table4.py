#!/usr/bin/env python
"""Part-4 tables (DPANN even, source207). usage: table4.py RESULTS_DIR"""
import json
import sys
from pathlib import Path

R = Path(sys.argv[1])
C = ("DefenseFinder", "TXSScan", "CRISPRCasTyper")


def f(x, n=3, s=False):
    if x is None:
        return "-"
    t = f"{x:+.{n}f}" if s else f"{x:.{n}f}"
    return t.replace("0.", ".", 1)


def iv(v):
    return f"{f(v[0], 3, True)} [{f(v[1], 3, True)},{f(v[2], 3, True)}]"


pairs = json.load(open(R / "pairs_p4.json")) if (R / "pairs_p4.json").exists() else {}
print("DPANN EVEN")
print(f"{'arm output':18s} {'rec/res':>9s} {'d res vs DH':>17s} {'d res vs p1':>17s} parv {'summed':>6s} {'d vs DH':>22s} "
      f"{'d vs p1 L4b':>22s} {'F1 DF/TXSS/CCT':>15s} {'dF1 CCT vs DH':>22s} bars bprec DF/TX/CC genes")
rows = [("WA", "dpann_WA", "L4b_t0.2"), ("WA", "dpann_WA", "L4b_t0.1"), ("W3b", "dpann_W3b", "L4b_t0.02"),
        ("W5", "dpann_W5", "L4b_t0.02"), ("W8", "dpann_W8", "L4b_t0.1")]
for arm, tag, nm in rows:
    fn = R / f"even_{tag}_p4.json"
    if not fn.exists():
        continue
    o = json.load(open(fn))["outputs"][nm]
    s, r = o["summary_even"], o["ribo"]["all"]
    v = o["vs_DH_even"]
    bars = [v["summed"][0] >= -.01] + [v[c]["d_f1"][0] >= -.02 for c in C] + [
        r["recovered"] >= 1650, r["resolved"] > 1512, int(r["parvarchaeaceae"].split("/")[0]) >= 110]
    pk = f"{tag}/{nm}:dpann_group/L4b_t0.1"
    rp = pairs.get(pk, {}).get("ribo_resolved")
    rr = o["ribo_resolved_vs_DH"]
    print(f"{arm + ' ' + nm:18s} {r['recovered']:>4d}/{r['resolved']:<4d} {f'{rr[0]:+.0f} [{rr[1]:+.0f},{rr[2]:+.0f}]':>17s} "
          f"{(f'{rp[0]:+.0f} [{rp[1]:+.0f},{rp[2]:+.0f}]' if rp else '-'):>17s} {r['parvarchaeaceae'].split('/')[0]:>4s} "
          f"{f(s['summed_DF_TXSS_excess']):>6s} {iv(v['summed']):>22s} {iv(o['vs_p1L4b_even']['summed']):>22s} "
          f"{'/'.join(f(s[c]['f1']) for c in C):>15s} {iv(v['CRISPRCasTyper']['d_f1']):>22s} {sum(bars)}/7  "
          f"{'/'.join(f(s[c]['median_bprec_covered'], 2) for c in C)}  {f(o['genes_in_loci_frac'])}")
    print(" " * 19 + "dF1 vs DH DF " + iv(v["DefenseFinder"]["d_f1"]) + " TXSS " + iv(v["TXSScan"]["d_f1"])
          + " | vs p1 L4b dF1 " + " ".join(f"{c[:4]} {iv(o['vs_p1L4b_even'][c]['d_f1'])}" for c in C)
          + f" | failed: {', '.join(n for n, b in zip(['summed', 'DF', 'TXSS', 'CCT', 'rec', 'res', 'parv'], bars) if not b) or 'none'}")
for k, v in pairs.items():
    print("PAIR", k, "summed", iv(v["summed"]), "resolved", v["ribo_resolved"])

print("\nSOURCE207")
REF = ("p1L4b", "p1L1", "CC2", "engine_ref", "spacedust")
print(f"{'arm output':16s} {'F1':>5s} {'ctrl':>5s} {'excess [95%]':>22s} " + " ".join(f"{'d vs ' + r:>22s}" for r in REF)
      + "  union/best by size 1-5 | 6-10 | 11-20 | 21+   (>=11)   genes")
for arm, fn in (("WA", "s207_sd_WA.json"), ("W3", "s207_sd_group_p4.json"), ("W3b", "s207_sd_W3b.json"),
                ("W5", "s207_sd_W5.json"), ("W8", "s207_sd_W8.json")):
    if not (R / fn).exists():
        continue
    d = json.load(open(R / fn))
    for nm, o in d["outputs"].items():
        s, fr = o["score"], o["fragmentation"]
        by = fr["by_size"]
        print(f"{arm + ' ' + nm:16s} {f(s['f1_observed']):>5s} {f(s['f1_random']):>5s} {iv([s['f1_excess'], s['ci_lo'], s['ci_hi']]):>22s} "
              + " ".join(f"{iv(o['vs_' + r]['paired']['excess']):>22s}" for r in REF)
              + "  " + " | ".join(f"{f(by[k]['union_q50'], 2)}/{f(by[k]['best_q50'], 2)}" for k in ("1-5", "6-10", "11-20", "21+"))
              + f"   ({f(fr['union_cov_q50_ge11'], 2)}/{f(fr['best_cov_q50_ge11'], 2)})   {f(o['genes_in_loci_frac'])}"
              + f"  {d.get('label', '')}")
        print(" " * 17 + "excess by size " + " ".join(f"{k} {f(v['excess'], 3, True)}" for k, v in s["by_size"].items())
              + f"; bprec {f(s['median_boundary_precision'], 2)}; d F1 vs engine {iv(o['vs_engine_ref']['paired']['f1'])}"
              + f"; verdict vs engine {o['vs_engine_ref']['verdict']}")
