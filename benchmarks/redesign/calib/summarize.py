#!/usr/bin/env python
"""Format ws_calib result JSONs into report tables (local; reads benchmarks/redesign/calib/results/)."""
import json
import sys
from pathlib import Path

R = Path(__file__).resolve().parent / "results"


def q1():
    r = json.loads((R / "q1_classes.json").read_text())
    x = json.loads((R / "q1_extra.json").read_text()) if (R / "q1_extra.json").exists() else None
    print("Q1 class table, R queries (point [95% query bootstrap]); all pairs | pairs annotated on both sides")
    print(f"{'band':14s} {'n':>7s} {'HOM':>21s} {'align E<=1e-3':>21s} {'Pfam fam':>7s} {'clan':>6s} {'CDD sf':>7s} "
          f"{'profile':>7s} {'kindK':>6s} {'contra':>7s} {'annot2':>7s} | {'HOM_eval':>8s} {'A_eval':>7s} {'contra_eval':>11s}")
    for g in ["rank1_10", "rank11_100", "rank101_1000", "cos20_32", "cos32_45", "cos45_60", "cos60_100", "uniform"] + \
             ["lenmatch_" + k for k in ["rank1_10", "rank101_1000", "cos20_32", "cos32_45", "cos60_100"]]:
        d, e = r["R"][g], r["R_evaluable"][g]
        f = lambda v: f"{v[0]:.3f} [{v[1]:.3f},{v[2]:.3f}]"
        print(f"{g:14s} {d['n']:7d} {f(d['HOM']):>21s} {f(d['A_align']):>21s} {d['P_pfam'][0]:7.3f} {d['C_clan_only'][0]:6.3f} "
              f"{d['D_cdd'][0]:7.3f} {d['F_profile'][0]:7.3f} {d['K_kind'][0]:6.3f} {d['contradicted_any'][0]:7.3f} "
              f"{d['both_annotated'][0]:7.3f} | {e['HOM'][0]:8.3f} {e['A_align'][0]:7.3f} {e['contradicted_any'][0]:11.3f}")
    print("\nfine cosine bins, family-absent best matches (R): HOM / align / K | length-matched HOM")
    for k, d in r["R_cos_fine"].items():
        b = r["R_cos_fine_lenmatch"][k]
        print(f"  {k:10s} n {d['n']:6d} HOM {d['HOM'][0]:.3f} [{d['HOM'][1]:.3f},{d['HOM'][2]:.3f}] align {d['A_align'][0]:.3f} "
              f"annot2 {d['both_annotated'][0]:.3f} contra {d['contradicted_any'][0]:.3f} | base HOM {b['HOM'][0]:.3f}")
    if x:
        print("\nheadline (R): remote homology / same kind, annotation contradicts / unexplained (of which unannotated side)")
        for k, d in x["d_headline"].items():
            print(f"  {k:14s} {d['remote_homology']} {d['same_kind_contradicted']} {d['unexplained']} {d['unexplained_with_unannotated_side']} diffdom {d['different_domains_not_same_kind']}")
        print("\nkind given no homology evidence (R): band | length-matched")
        for k in ["rank1_10", "rank11_100", "rank101_1000", "cos20_32", "cos32_45", "cos45_60", "cos60_100"]:
            a, b = x["a_kind_given_no_homology"][k], x["a_kind_given_no_homology"]["lenmatch_" + k]
            print(f"  {k:14s} K {a['K_kind']} vs {b['K_kind']}; sameTM {a['same_TM_class'][0]} vs {b['same_TM_class'][0]}; "
                  f"len {a['len_ratio_ge_067'][0]} vs {b['len_ratio_ge_067'][0]}; comp {a['comp_le_p25'][0]} vs {b['comp_le_p25'][0]}; "
                  f"bothTM {a['both_TM'][0]} vs {b['both_TM'][0]}")
        print("\nby annotation side (share of pairs, HOM, align)")
        for k, v in x["b_unannotated"].items():
            print(f"  {k:14s} " + "; ".join(f"{a}: {b['share_of_pairs']:.3f} {b['HOM'][0]:.3f} {b['A_align'][0]:.3f}" for a, b in v.items()))
        print("\nalignment geometry of alignable pairs")
        for k, v in x["c_alignment"].items():
            print(f"  {k:14s} {v}")


def q2():
    r = json.loads((R / "q2_detectors.json").read_text())
    for s, d in r.items():
        if not isinstance(d, dict) or "scores" not in d:
            continue
        print(f"\n{s}: queries {d['queries']} pos {d['pos_even']:,} neg {d['neg_even']:,}; RBM binary {d['D9_rbm_binary']}")
        for k, v in d["scores"].items():
            dv = lambda x: v.get(f"diff_vs_raw@{x}")
            print(f"  {k:16s} AUC {v['auroc']:.4f} rec@1e-2 {v['recall@0.01']:.3f} {v['ci@0.01']} rec@1e-3 {v['recall@0.001']:.3f} "
                  f"{v['ci@0.001']} rec@1e-4 {v['recall@0.0001']:.3f} {v['ci@0.0001']} | d1e-3 {dv('0.001')} d1e-2 {dv('0.01')} "
                  f"d1e-4 {dv('0.0001')} | transferFPR {v['transfer@0.001']['fpr_even']:.2e}")
    print("\ncalibration")
    for k, v in r["calibration"].items():
        print(f"  {k:16s}", {kk: vv for kk, vv in v.items()})
    print("\nchoices", r["choices"])
    print("fitinfo", r["fitinfo"])


if __name__ == "__main__":
    {"q1": q1, "q2": q2}[sys.argv[1]]()
