#!/usr/bin/env python
"""Tables of the lineagefree token-engine runs.  usage: collect.py odd|even DIR_WITH_tok_*.json [--groups GROUPS_JSON_DIR]
odd  reads only tok_<tag>.odd.json (selection); even reads tok_<tag>.json (verdict and sensitivity)."""
import glob
import json
import re
import sys
from pathlib import Path

CALLERS = ("DefenseFinder", "TXSScan", "CRISPRCasTyper")


def key(tag):
    m = re.match(r"(.+)_([gw])([0-9.]+)$", tag)
    if not m:
        return (tag, "", 0.0)
    return (m.group(1), m.group(2), float(m.group(3)))


def fmt_ci(x):
    return f"{x[0]:+.4f} [{x[1]:+.4f},{x[2]:+.4f}]"


def main():
    split, d = sys.argv[1], Path(sys.argv[2])
    files = sorted(glob.glob(str(d / ("tok_*.odd.json" if split == "odd" else "tok_*.json"))))
    files = [f for f in files if (split == "odd") == f.endswith(".odd.json")]
    rows = []
    for f in files:
        x = json.load(open(f))
        rows.append(x)
    rows.sort(key=lambda x: key(x["tag"]))
    print(f"{'tag':<18} {'units':>7} {'rule':<14} {'fdr':>6} {'sigpairs':>8} {'genes':>9} | reading  summed  d vs genus [95%]            dF1 DF/TXSS/CCT          ribo rec/res/parv(res)   cond")
    for x in rows:
        e = x["engine"]
        units = e.get("lineages") or (round(e["effective_genomes"]) if e.get("effective_genomes") else "-")
        for rn in ("R1", "DH"):
            r = x["readings"][rn]
            if split == "odd":
                s, pr, rib, cond = r["summary_odd"], r["vs_genus_odd"], r["ribo_odd"], r["conditions_vs_genus_odd"]
            else:
                s, pr, rib, cond = r["summary"]["even"], r["vs_genus_even"], r["ribo"], r["conditions_vs_genus_even"]
            head = (f"{x['tag']:<18} {str(units):>7} {str(e['rule']):<14} {e['fdr']:.4f} {e['significant_pairs']:>8} "
                    f"{e['genes_covered']:>9}") if rn == "R1" else " " * 68
            df1 = "/".join(f"{pr[c]['d_f1'][0]:+.3f}" for c in CALLERS)
            ribo = f"{rib['recovered']}/{rib['resolved']}/{rib['parvarchaeaceae']}({rib.get('parvarchaeaceae_resolved')})"
            cd = cond.get("n_met", cond.get("replaces"))
            print(f"{head} | {rn:<3} {s['summed_DF_TXSS_excess']:.4f}  {fmt_ci(pr['summed']):<30} {df1:<22} {ribo:<24} {cd}")


if __name__ == "__main__":
    main()
