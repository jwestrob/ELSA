#!/usr/bin/env python
"""Print the report tables from the result JSONs (local copies in results/)."""
import json
import sys
from pathlib import Path

R = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parent / "results")


def f(x, nd=3):
    if x is None:
        return "-"
    if isinstance(x, list):
        if len(x) == 3 and x[1] is not None:
            return f"{x[0]:.{nd}f} [{x[1]:.{nd}f}, {x[2]:.{nd}f}]"
        return f"{x[0]:.{nd}f}"
    return f"{x:.{nd}f}"


def sgn(x, nd=3):
    if x is None:
        return "-"
    return f"{x[0]:+.{nd}f} [{x[1]:+.{nd}f}, {x[2]:+.{nd}f}]"


def typing(g, setname):
    s = g[setname]["summary"]
    s70 = g[setname]["summary_g070"]
    sel = g[setname]["selection"]
    order = ["J", "MOD", "MODJ", "MEAN", "CH", "CHr", "ALN", "ALNs", sel["SK"], sel["RFF"]]
    print(f"\n== {setname}: 1-NN typing, even queries, emb_hom 0.75 groups held out "
          f"(selection on odd: SK {sel['SK']}, RFF {sel['RFF']}, best {sel['best']}, indexable {sel['indexable']})")
    print(f"{'rep':14s} | {'DF acc':24s} {'DF fam':7s} {'DF 5NN':7s} {'DF macro':8s} | {'CCT acc':24s} {'CCT type':8s} "
          f"{'CCT 5NN':7s} {'CCT macro':9s} | odd DF / CCT | g070 DF / CCT")
    for r in order:
        if r not in s:
            continue
        d, c = s[r]["DefenseFinder"], s[r]["CRISPRCasTyper"]
        d70, c70 = s70[r]["DefenseFinder"]["even"], s70[r]["CRISPRCasTyper"]["even"]
        print(f"{r:14s} | {f(d['even']['acc']):24s} {f(d['even']['acc_hi']):7s} {f(d['even']['acc5']):7s} "
              f"{f(d['even']['macro']):8s} | {f(c['even']['acc']):24s} {f(c['even']['acc_hi']):8s} {f(c['even']['acc5']):7s} "
              f"{f(c['even']['macro']):9s} | {f(d['odd']['acc'])} / {f(c['odd']['acc'])} | "
              f"{f(d70['acc'])} / {f(c70['acc'])}")
    print("paired (even):")
    for k, v in g[setname]["paired"].items():
        if k.endswith("|correct"):
            print(f"  {k:45s} {sgn(v)}")
    if "clustering" in g[setname]:
        print(f"clustering (Leiden on kNN-15; resolution by odd AMI; even instances):")
        for r, d in sorted(g[setname]["clustering"].items(), key=lambda kv: -kv[1]["even"]["ami"]):
            e = d["even"]
            print(f"  {r:14s} res {str(d['resolution']):5s} AMI {e['ami']:.3f} [{e['ci']['ami'][0]:.3f}, {e['ci']['ami'][1]:.3f}] "
                  f"hom {e['homogeneity']:.3f} compl {e['completeness']:.3f} clusters {e['clusters']:5d} "
                  f"cross-group pair P {e['pair_precision_cross_group']:.3f} R {e['pair_recall_cross_group']:.3f}")


def disjoint(g):
    print("\n== disjoint-family and sibling tests (even queries; other genome groups)")
    D = g["disjoint"]
    sel = g["gold"]["selection"]
    reps = ["J", "MOD", "MODJ", "MEAN", "CH", "CHr", "ALN", sel["SK"], sel["RFF"]]
    for test, by in D.items():
        for caller, d in by.items():
            print(f"  {test:18s} {caller:15s} n {d['n']:5d} chance {f(d['chance'])}")
            for r in reps:
                if r in d:
                    print(f"      {r:14s} {f(d[r])}   minus chance {sgn(d[r + '_minus_chance'])}")


def main():
    g = json.load(open(R / "gold_results.json"))
    print("timings", json.dumps(g.get("timings", {}))[:2000])
    for setname in ("gold", "engine"):
        if setname in g:
            typing(g, setname)
    if "engine_coverage50" in g:
        print("engine coverage50", g["engine_coverage50"], "F1 median", g["engine_f1_median"], "span q", g["engine_span_q"])
    if "disjoint" in g:
        disjoint(g)


if __name__ == "__main__":
    main()
