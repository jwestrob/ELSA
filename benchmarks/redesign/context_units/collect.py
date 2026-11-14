#!/usr/bin/env python
"""context_units: odd-genome table and predeclared selection (reads results/*.odd.json only), or the even verdict table
(reads results/full/*.json; run only after the selection is written to NOTEBOOK.txt).

usage
  collect.py odd   [--pool raw,gc,raw_z2,gc_z2] [--t 2]      -> results/odd_table.txt, results/selection.json
  collect.py even  --tags TAG,TAG,...                         -> results/even_table.txt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

RD = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111")
CU = RD / "context_units"
CALLERS = ("DefenseFinder", "TXSScan", "CRISPRCasTyper")


def parv(r, key="parvarchaeaceae"):
    return str(r.get(key, "")).split("/")[0]


def odd_row(tag, d):
    r = d["readings"]["DH"]
    pr = r["vs_genus_odd"]
    c = r["conditions_odd"]
    ro = r["ribo_odd"]
    return (f"{tag:24s} {d.get('significant_pairs', 0):>8} {r['summary_odd']['summed_DF_TXSS_excess']:.4f} "
            f"{pr['summed'][0]:+.4f} [{pr['summed'][1]:+.3f},{pr['summed'][2]:+.3f}] "
            + " ".join(f"{pr[x]['d_f1'][0]:+.4f}" for x in CALLERS)
            + f"  {ro['recovered']:>4}/{ro['resolved']:>4}/{parv(ro)}  met {c['n_met']} {'M' if c['matches'] else '-'}"), c, pr


def cmd_odd(a):
    rows, cands = [], []
    for f in sorted((CU / "results").glob("*.odd.json")):
        tag = f.name[:-len(".odd.json")]
        if tag.startswith(("cuts_", "softctx_")):
            continue
        d = json.load(open(f))
        line, c, pr = odd_row(tag, d)
        rows.append(line)
        v = d.get("variant", "")
        z = d.get("z", 0) or 0
        vz = v + (f"_z{z:g}" if z else "") + ("_ext" if "_ext" in tag else "")
        if vz in a.pool.split(",") and abs(float(d.get("t", 0)) - a.t) < 1e-9:
            cands.append((c["n_met"], pr["summed"][0], tag, vz, d.get("W")))
    hdr = ("tag                      sigpairs oddDH  d_summed vs genus [95%]     dF1 DF / TXSS / CCT        "
           "ribo odd rec/res/Parv")
    txt = hdr + "\n" + "\n".join(rows)
    (CU / "results" / "odd_table.txt").write_text(txt + "\n")
    print(txt)
    if cands:
        best = max(cands)
        sel = dict(rule="most odd conditions met, then odd summed DH excess", pool=a.pool, t=a.t,
                   candidates=[dict(n_met=x[0], d_summed=x[1], tag=x[2]) for x in sorted(cands, reverse=True)],
                   selected=dict(tag=best[2], variant=best[3], W=best[4], n_met=best[0], d_summed_odd=best[1]))
        (CU / "results" / "selection.json").write_text(json.dumps(sel, indent=1) + "\n")
        print("\nSELECTED", sel["selected"])


def cmd_even(a):
    out = []
    for tag in a.tags.split(","):
        f = CU / "results" / "full" / f"{tag}.json"
        d = json.load(open(f))
        for rn in ("DH", "R1"):
            r = d["readings"][rn]
            ev = r["summary"]["even"]
            line = [f"{tag:24s} {rn:2s} {ev['summed_DF_TXSS_excess']:.4f}"]
            for cmp_ in ("genus", "lf_emb_hom_g0.75", "lf_emb_mean_g0.55", "lf_fam_jac_g0.2", "lf_genome"):
                pr = r.get(f"vs_{cmp_}_even")
                if pr is None:
                    continue
                line.append(f"{cmp_}: {pr['summed'][0]:+.4f} [{pr['summed'][1]:+.3f},{pr['summed'][2]:+.3f}] "
                            + "/".join(f"{pr[x]['d_f1'][0]:+.3f}" for x in CALLERS))
            rb = r["ribo"]
            line.append(f"ribo {rb['recovered']}/{rb['resolved']}/{parv(rb)}/{parv(rb, 'parvarchaeaceae_resolved')}")
            line.append(f"F1 " + "/".join(f"{ev[x]['f1']:.3f}" for x in CALLERS))
            line.append("cond " + json.dumps(r["conditions_even"]))
            out.append("  ".join(line))
    txt = "\n".join(out)
    (CU / "results" / "even_table.txt").write_text(txt + "\n")
    print(txt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("odd", "even"))
    ap.add_argument("--pool", default="raw,gc,raw_z2,gc_z2")
    ap.add_argument("--t", type=float, default=2.0)
    ap.add_argument("--tags", default="")
    a = ap.parse_args()
    {"odd": cmd_odd, "even": cmd_even}[a.cmd](a)


if __name__ == "__main__":
    main()
