#!/usr/bin/env python
"""Compact even-genome sensitivity rows (DH and R1) for the report.  usage: compact.py RESULTS_DIR"""
import glob
import json
import re
import sys
from pathlib import Path

C = ("DefenseFinder", "TXSScan", "CRISPRCasTyper")


def key(t):
    m = re.match(r"(.+)_([gw])([0-9.]+)$", t)
    return (m.group(1), m.group(2), float(m.group(3))) if m else (t, "", 0.0)


rows = [json.load(open(f)) for f in glob.glob(str(Path(sys.argv[1]) / "tok_*.json")) if not f.endswith(".odd.json")]
rows.sort(key=lambda x: key(x["tag"]))
print(f"{'unit':<18}{'units':>6}{'sigpairs':>9} | DH: d summed [95%]          dF1 DF/TXSS/CCT       res  ok | R1: d summed   res")
for x in rows:
    e = x["engine"]
    u = e.get("lineages") or round(e["effective_genomes"])
    d, r = x["readings"]["DH"], x["readings"]["R1"]
    pr, pr1 = d["vs_genus_even"], r["vs_genus_even"]
    ok = "Y" if d["conditions_vs_genus_even"]["replaces"] else "-"
    print(f"{x['tag']:<18}{u:>6}{e['significant_pairs']:>9} | {pr['summed'][0]:+.3f} [{pr['summed'][1]:+.3f},{pr['summed'][2]:+.3f}]  "
          + "/".join(f"{pr[c]['d_f1'][0]:+.3f}" for c in C) + f"  {d['ribo']['resolved']:>5}  {ok}  | {pr1['summed'][0]:+.3f}  {r['ribo']['resolved']:>5}")
