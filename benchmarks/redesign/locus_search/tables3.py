#!/usr/bin/env python
"""Print the round-3 tables from the out3 JSON summaries. usage: tables3.py DIR (holding cv_summary3.json, summary_final3.json,
rates3.json, phase0_zca10.json, m3z_summary.json, hnsw_sweep.json, bench_m1z.json, lists.json)."""
import json
import sys
from pathlib import Path

D = Path(sys.argv[1] if len(sys.argv) > 1 else ".")


def J(name):
    p = D / name
    return json.load(open(p)) if p.exists() else None


def f(v, n=3):
    if v is None:
        return "-"
    if isinstance(v, (list, tuple)) and len(v) == 3:
        return f"{v[0]:+.{n}f} [{v[1]:+.{n}f}, {v[2]:+.{n}f}]"
    if isinstance(v, float):
        return f"{v:.{n}f}"
    return str(v)


def m(v, n=3):
    return f"{v[0]:.{n}f}" if isinstance(v, (list, tuple)) else f(v, n)


cv = J("cv_summary3.json")
if cv:
    print("== TASK 1 (CV) ==", "folds same as round 2:", cv["folds_reproduce_round2"], "S3 vs round 2 max diff:",
          cv["S3_heldout_vs_round2_max_abs_diff"], "queries", cv["n_queries"], "genomes", cv["n_genomes"])
    t = cv["task1"]
    print("M1z-S1", f(t["M1z_minus_S1"]), "M1z-S3", f(t["M1z_minus_S3"]), "cond_i", t["cond_i"], "cond_ii", t["cond_ii"],
          "CONFIRMED", t["confirmed"])
    print("per fold", t["per_fold"])
    for s in ("all", "even", "odd", "min3"):
        print(f"-- held-out AP, stratum {s}")
        for fam, d in cv["tables"][s].items():
            print(f"   {fam:7s} n={d['queries']:5d} AP {m(d['ap'])} strict {m(d['ap_strict'])}")
        for k, d in cv["paired"][s].items():
            print(f"   {k:14s} dAP {f(d['ap'])}  dstrict {f(d['ap_strict'])}")
    print("chosen per fold:")
    for fam, cs in cv["chosen"].items():
        print(f"   {fam:7s}", [(c['w'], c['wx'], c['tau']) for c in cs], "odd:", (cv["odd"][fam]["w"], cv["odd"][fam]["wx"], cv["odd"][fam]["tau"]))
    if "task6" in cv:
        print("task6 M1p-M1z even", f(cv["task6"]["M1p_minus_M1z_even"]["ap"]), "survives", cv["task6"]["survives"])

sf = J("summary_final3.json")
if sf:
    print("\n== FINAL (200-replicate) ==")
    print("self-check", sf["selfcheck"])
    for key in ("task1_cv", "task1_cv_even"):
        print("--", key)
        for fam, d in sf[key]["tables"].items():
            print(f"   {fam:6s} n={d['queries']} " + " ".join(f"{c}={m(d[c])}" for c in ("ap", "ap_strict", "recall_E0.01", "recall_E1",
                                                                                           "n_wrong_E0.01", "n_unlabelled_E0.01")))
        for k, d in sf[key]["paired"].items():
            print(f"   {k:10s} " + " ".join(f"{c}={f(d[c])}" for c in ("ap", "recall_E0.01", "recall_E1", "n_wrong_E0.01")))
    print("-- generality (even, odd-genome DF configs); general =", sf["general"])
    for c, d in sf["generality"].items():
        print("  ", c, "same sign", d["same_sign"])
        for fam, t in d["tables"].items():
            print(f"     {fam:5s} n={t['queries']} " + " ".join(f"{k}={m(t[k])}" for k in ("ap", "ap_strict", "recall_E0.01", "n_wrong_E0.01",
                                                                                         "n_unlabelled_E0.01")))
        for k, t in d["paired"].items():
            print(f"     {k:7s} " + " ".join(f"{c}={f(t[c])}" for c in ("ap", "ap_strict", "recall_E0.01", "n_wrong_E0.01")))
    print("-- task 4b token-invisible recall (even, odd configs)")
    for fam, d in sf["task4b"]["even_odd_configs"].items():
        print(f"   {fam:5s} micro inv E0.01 {d['micro_recall_inv_E0.01']:.4f} E1 {d['micro_recall_inv_E1']:.4f} found {d['found_inv_E0.01']}"
              f"/{d['found_inv_E1']} of {d['invisible_targets']}; vis E0.01 {d['micro_recall_vis_E0.01']:.4f}")
    for k, d in sf["task4b"]["paired_even"].items():
        print(f"   {k:8s} inv E0.01 {f(d['recall_inv_E0.01'])} E1 {f(d['recall_inv_E1'])} n={d['queries']}")
    print("timing", sf["timing_cpu_seconds_per_query"])

r = J("rates3.json")
if r:
    print("\n== TASK 3 RATES (fast null) ==")
    for fam, d in r["calibration_even"].items():
        print(f"   calib even {fam:5s} " + " ".join(f"E{x}: {d[x]['ratio']:.2f} [{d[x]['ci'][0]:.2f},{d[x]['ci'][1]:.2f}]"
                                                   for x in ("0.001", "0.01", "0.1", "1.0")), "curve@0.01", round(r["curves_ratio_at_E"][fam]["0.01"], 2))
    for fam, d in r["at_E0_01"].items():
        print(f"   {fam:5s} AP {m(d['ap_full'])} strict {m(d['ap_strict_full'])} rec.01 {m(d['recall_E0_01'])} rec1 {m(d['recall_E1'])} "
              f"unl {m(d['unlabelled_per_query'])} unsup {m(d['unsupported_per_query'])} wrong {m(d['wrong_per_query'])}")
    print("-- matched 0.03 unsupported / query (threshold on odd)")
    for fam, d in r["matched"].items():
        print(f"   {fam:5s} t={d['threshold_E_fast']:.3g} reached={d['rate_reached_below_E10']} odd={d['odd_unsupported_per_query']:.3f} "
              f"recall {m(d['recall'])} AP {m(d['ap'])} unsup {m(d['unsupported_per_query'])} unl {m(d['unlabelled_per_query'])} wrong {m(d['wrong_per_query'])}")
    for k, d in r["paired_matched"].items():
        print(f"   {k:9s} recall {f(d['recall'])} AP {f(d['ap'])} unsup {f(d['unsup'])} wrong {f(d['wrong'])}")
    print("-- paired at E<=0.01")
    for k, d in r["paired_E0_01"].items():
        print(f"   {k:9s} recall {f(d['recall'])} unl {f(d['unl'])} unsup {f(d['unsup'])} wrong {f(d['wrong'])}")

p = J("phase0_zca10.json")
if p:
    print("\n== PHASE 0 zca10 ==", "cos q", p["random_pair_cosine_quantiles"], "same pairs", p["same_pairs_and_f_tok_as_round1"])
    for c in ("DefenseFinder", "TXSScan", "CRISPRCasTyper"):
        d = p[c]
        print(f"   {c}: pairs {d['ordered_pairs']:,} tok {d['f_tok']['frac_ge_half']:.3f} cos {d['f_cos']['frac_ge_half']:.3f} "
              f"rank {d['f_rank']['frac_ge_half']:.3f} emb {d['f_emb']['frac_ge_half']:.3f} union {d['union_ge_half']:.3f} "
              f"inv&emb {d['token_invisible_and_f_emb_ge_half']:.3f}")
        print("      control", p["control_different_type"].get(c))
        o = p["round1_l2"]["headroom"][c]
        print(f"      round1 l2: emb {o['f_emb']['frac_ge_half']:.3f} inv&emb {o['token_invisible_and_f_emb_ge_half']:.3f}",
              "control", p["round1_l2"]["control"].get(c, {}).get("tok_invisible_and_emb_visible"))

m3 = J("m3z_summary.json")
if m3:
    print("\n== M3 with zca10 ==", m3["seed_config"])
    for k, d in m3["tables"].items():
        print(f"   {k:16s} " + " ".join(f"{c}={m(d[c])}" for c in ("ap", "ap_strict", "recall_E0.01", "recall_E1", "n_wrong_E0.01",
                                                                  "n_unlabelled_E0.01", "n_unsupported_E0.01", "precision_E0.01",
                                                                  "precision_strict_E0.01")), "inv", round(d["micro_recall_inv_E0.01"], 4))
    print("   drift", m3["drift"])
    print("   round1 raw", m3["round1_raw"])

h = J("hnsw_sweep.json")
if h:
    print("\n== HNSW ==", {k: v for k, v in h.items() if not isinstance(v, dict)})
    for k, v in h.items():
        if k.startswith("hnsw_k"):
            print("  ", k, v)
    for k, v in h.get("ap", {}).items():
        print(f"   {k:22s} MAP {v['map']:.4f} d {f(v['delta_vs_exact20000'], 4)}")
    print("   chosen", h.get("chosen"))

b = J("bench_m1z.json")
if b:
    print("\n== CLI ==", json.dumps(b, indent=1))
li = J("lists.json")
if li:
    print("\n== lists ==", json.dumps(li, indent=1))
