#!/usr/bin/env python
"""Even-genome verdict table for named token-engine runs.  usage: verdict.py RESULTS_DIR tag1 tag2 ..."""
import json
import sys
from pathlib import Path

C = ("DefenseFinder", "TXSScan", "CRISPRCasTyper")
d = Path(sys.argv[1])
for tag in sys.argv[2:]:
    x = json.load(open(d / f"tok_{tag}.json"))
    e = x["engine"]
    u = e.get("lineages") or f"Neff {e['effective_genomes']:.0f}"
    print(f"{tag}  units {u}  rule {e['rule']}  fdr {e['fdr']:.4f}  significant pairs {e['significant_pairs']:,}  genes covered {e['genes_covered']:,}")
    for rn in ("DH", "R1"):
        r = x["readings"][rn]
        s, pr, rib, cond = r["summary"]["even"], r["vs_genus_even"], r["ribo"], r["conditions_vs_genus_even"]
        f1 = " ".join(f"{c[:4]} {s[c]['f1']:.3f} ({pr[c]['d_f1'][0]:+.3f} [{pr[c]['d_f1'][1]:+.3f},{pr[c]['d_f1'][2]:+.3f}])" for c in C)
        rv = r["ribo_resolved_vs_genus"]
        print(f"   {rn}: summed {s['summed_DF_TXSS_excess']:.4f} d {pr['summed'][0]:+.4f} [{pr['summed'][1]:+.4f},{pr['summed'][2]:+.4f}] | {f1}")
        print(f"       ribo rec {rib['recovered']} res {rib['resolved']} (vs genus {rv[0]:+.0f} [{rv[1]:+.0f},{rv[2]:+.0f}]) Parv rec {rib['parvarchaeaceae']} res {rib['parvarchaeaceae_resolved']} | loci {r['loci']['loci']:,} genes {r['loci']['genes_covered']:,} | cond {cond}")
