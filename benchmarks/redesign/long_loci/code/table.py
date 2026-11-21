#!/usr/bin/env python
"""Tables from long_loci result JSONs (run anywhere on copies of $RD/long_loci/results).

usage: table.py RESULTS_DIR odd|even|s207|rdiag [tag]
"""
import json
import sys
from pathlib import Path

C = ("DefenseFinder", "TXSScan", "CRISPRCasTyper")


def f(x, n=3, sign=False):
    if x is None:
        return "-"
    s = f"{x:+.{n}f}" if sign else f"{x:.{n}f}"
    return s.replace("0.", ".", 1) if abs(x) < 1 else s


def iv(v, n=3):
    return f"{f(v[0], n, True)} [{f(v[1], n, True)}, {f(v[2], n, True)}]"


def odd(R, tag):
    d = json.load(open(R / f"odd_{tag}.json"))
    print(f"ODD ({tag}); DH odd summed {f(d['DH_odd']['summary']['summed_DF_TXSS_excess'], 4)}, ribo "
          f"{d['DH_odd']['ribo']['recovered']}/{d['DH_odd']['ribo']['resolved']} parv {d['DH_odd']['ribo']['parvarchaeaceae']}")
    print(f"{'output':14s} {'rec/res':>9s} {'parv':>7s} {'summed':>7s} {'d vs DH [95%]':>26s} {'F1 DF/TXSS/CCT':>16s} cond  genes")
    for nm, o in d["outputs"].items():
        s, r = o["summary_odd"], o["ribo_odd"]
        print(f"{nm:14s} {r['recovered']:>4d}/{r['resolved']:<4d} {r['parvarchaeaceae']:>7s} "
              f"{f(s['summed_DF_TXSS_excess'], 4):>7s} {iv(o['vs_DH_odd']['summed']):>26s} "
              f"{'/'.join(f(s[c]['f1']) for c in C):>16s} {sum(o['conditions_odd'])}     {o['loci']['genes_covered']}")
    print("selection:", {k: v for k, v in d["selection"].items() if k != "scores"})


def even(R, tag):
    d = json.load(open(R / f"even_{tag}.json"))
    print(f"EVEN ({tag})")
    hdr = (f"{'output':12s} {'rec/res':>9s} {'d res vs DH [95%]':>20s} {'parv':>7s} {'summed':>6s} {'d vs DH':>24s} "
           f"{'d vs words V-mob':>24s} {'d vs words+coh':>24s} {'F1 DF/TXSS/CCT':>15s} {'bprec DF/TX/CC':>15s} genes")
    print(hdr)
    for nm, o in d["outputs"].items():
        s, r = o["summary_even"], o["ribo"]["all"]
        rr = o["ribo_resolved_vs_DH"]
        print(f"{nm:12s} {r['recovered']:>4d}/{r['resolved']:<4d} {f'{rr[0]:+.0f} [{rr[1]:+.0f}, {rr[2]:+.0f}]':>20s} "
              f"{r['parvarchaeaceae']:>7s} {f(s['summed_DF_TXSS_excess']):>6s} {iv(o['vs_DH_even']['summed']):>24s} "
              f"{iv(o['vs_words_Vmob_even']['summed']):>24s} {iv(o['vs_words_Vcoh0.1_even']['summed']):>24s} "
              f"{'/'.join(f(s[c]['f1']) for c in C):>15s} "
              f"{'/'.join(f(s[c]['median_bprec_covered'], 2) for c in C):>15s} {f(o['genes_in_loci_frac'], 3)}"
              f"  {o['verdict_vs_DH']['label']}{' RIBO-SUCCESS' if o['ribosomal_success'] else ''}")
        print(" " * 13 + "dF1 vs DH " + " ".join(f"{c[:4]} {iv(o['vs_DH_even'][c]['d_f1'])}" for c in C))
    if "ribosomal_diagnosis" in d:
        rdiag_print(d["ribosomal_diagnosis"])


def rdiag_print(x):
    print("RIBOSOMAL DIAGNOSIS structure:", json.dumps(x["structure"]))
    for nm, o in x["outputs"].items():
        print(f"  {nm:14s} resolved {o['resolved']} unresolved {o['unresolved']} = under {o['under_covered']} "
              f"(no locus {o['under_covered_no_locus']}) + over {o['over_extended']}; outside genes {o['over_extended_outside_genes']}"
              f" per run q10/50/90/max {o['outside_per_run_q10_q50_q90_max']}; a/b/c {f(o['outside_frac_a_other_run'])}/"
              f"{f(o['outside_frac_b_gap_between_runs'])}/{f(o['outside_frac_c_other'])} (c: small run "
              f"{f(o['outside_c_in_small_g2_run'])}, marked {f(o['outside_c_marked_outside_runs'])}); unresolved with "
              f"neighbour {o['unresolved_runs_with_neighbour_1_5']}; merged targets resolved {o['merged_targets_resolved']}, "
              f"standing-or-merged {o['runs_resolved_standing_or_merged']}; with-fragment runs: standing {o.get('standing_resolved_among_runs_with_fragment')}"
              f" merged {o.get('merged_targets_resolved_among_runs_with_fragment')}; outside c1 in neighbourhood "
              f"{f(o.get('outside_c1_within_merged_neighbourhood'))} c2 marked beyond {f(o.get('outside_c2_marked_beyond_neighbourhood'))}"
              f" c3 rest {f(o.get('outside_c3_rest'))}; under q50 best_frac {f(o['under_covered_best_frac_q50'])} frags {o['under_covered_fragments_q50']}")
        if "outside_pfam_top30_gene_counts" in o:
            print("     pfam top30:", ", ".join(f"{k} {v}" for k, v in o["outside_pfam_top30_gene_counts"]),
                  "| no pfam:", o.get("outside_genes_without_pfam"))


def s207(R, tag):
    fn = R / f"s207_{tag}.json"
    d = json.load(open(fn))
    print(f"SOURCE207 ({tag}) {d.get('label', '')}")
    print(f"{'output':12s} {'F1':>5s} {'ctrl':>5s} {'excess [95%]':>24s} {'bprec':>5s} {'cov':>4s} {'qgenes':>6s} "
          f"{'union/best(>=11)':>18s}  d excess vs P2 / CC1 / CC2 / engine / Spacedust")
    for nm, o in d["outputs"].items():
        s, fr = o["score"], o["fragmentation"]
        print(f"{nm:12s} {f(s['f1_observed']):>5s} {f(s['f1_random']):>5s} "
              f"{iv([s['f1_excess'], s['ci_lo'], s['ci_hi']]):>24s} {f(s['median_boundary_precision'], 2):>5s} "
              f"{f(s['median_coverage'], 2):>4s} {f(s['query_gene_coverage']):>6s} "
              f"{f(fr['union_cov_q50'], 2)}({f(fr['union_cov_q50_ge11'], 2)})/{f(fr['best_cov_q50'], 2)}({f(fr['best_cov_q50_ge11'], 2)})  "
              + " | ".join(iv(o[f"vs_{k}"]["paired"]["excess"]) for k in ("P2_Vcoh0.1", "CC1", "CC2", "engine_ref", "spacedust")))
        print(" " * 13 + "d F1: " + " | ".join(iv(o[f"vs_{k}"]["paired"]["f1"]) for k in
                                                 ("P2_Vcoh0.1", "CC1", "CC2", "engine_ref", "spacedust"))
              + "  by size excess: " + " ".join(f"{k} {f(v['excess'], 3, True)}" for k, v in s["by_size"].items()))


if __name__ == "__main__":
    R = Path(sys.argv[1])
    cmd = sys.argv[2]
    tag = sys.argv[3] if len(sys.argv) > 3 else "dpann_group"
    if cmd == "rdiag":
        rdiag_print(json.load(open(R / "rdiag_reference.json")))
    else:
        {"odd": odd, "even": even, "s207": s207}[cmd](R, tag)
