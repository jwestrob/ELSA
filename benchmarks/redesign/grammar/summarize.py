#!/usr/bin/env python
"""Print the report tables from a results directory (default results/; pass results/interim for the interim set)."""
import json
import sys
from pathlib import Path

import pandas as pd

RR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "results"
R = RR / "eval" if (RR / "eval").exists() else RR
pd.set_option("display.width", 250)
M = ["primary", "pmi_a1", "pmi_a01", "pmi_a10", "G3_train", "nlogP_train", "eng_nlogP", "raw_cos", "shuf", "glob", "w3", "primary_sym"]


def eval1():
    e = pd.read_csv(R / "eval1.tsv", sep="\t")
    e = e[e.method.notna()]
    for split in ("cv", "phylum"):
        for neg in ("nogold", "straddle"):
            x = e[(e.split == split) & (e.negset == neg)]
            if x.empty:
                continue
            t = x.pivot_table(index=["caller", "stratum"], columns="method", values="auc")
            n = x.groupby(["caller", "stratum"])[["n_pos", "n_neg"]].first()
            d = x[x.method == "pmi_a1"].set_index(["caller", "stratum"])[["ref_minus", "rm_lo", "rm_hi"]]
            d.columns = ["prim-pmi", "lo", "hi"]
            out = n.join(t[[m for m in M if m in t.columns]].round(3)).join(d.round(3))
            print(f"\n=== eval1 {split} vs {neg} (AUROC; prim-pmi = primary minus pmi_a1 with 95% genome bootstrap)")
            print(out.to_string())
        x = e[(e.split == split) & (e.negset == "nogold") & (e.stratum == "all") & (e.method.isin(["primary", "pmi_a1", "pmi_a01", "eng_nlogP", "raw_cos", "shuf"]))]
        print(f"\n--- eval1 {split} nogold all: AUROC CI and AP")
        print(x[["caller", "method", "auc", "auc_lo", "auc_hi", "ap", "prevalence"]].round(4).to_string(index=False))


def eval2():
    e = pd.read_csv(R / "eval2.tsv", sep="\t")
    keep = ["s_primary", "s_w3", "s_glob", "s_shuf", "eng_nlogP", "neg_len_sum", "s_primary_minus_eng_nlogP",
            "s_primary_minus_s_shuf", "s_primary_minus_neg_len_sum"] + [f"s_sel_c{k}" for k in range(1, 7)]
    print("\n=== eval2 (AUROC for ipTM >= 0.75)")
    print(e[e.method.isin(keep)][["split", "set", "method", "n", "n_hit", "auc", "auc_lo", "auc_hi"]].round(3).to_string(index=False))
    s = json.load(open(R / "eval_summary.json"))
    for tag in ("cv", "phylum"):
        if f"{tag}_eval2" in s:
            v = s[f"{tag}_eval2"]
            print(f"\n{tag}: uncalled pairs n={v['uncalled_n']}; the four co-folders (fold, ipTM, rank of primary among uncalled, "
                  "percentile primary, percentile engine):")
            for r in v["four_cofolders"]:
                print("   ", r["fold_id"], round(r["iptm"], 3), int(r["rank_primary"]), round(r["pct_primary"], 3), round(r["pct_engine"], 3))
            print("  per-fold DefenseFinder vs no-gold:", {k: (v_["n_pos"], round(v_["primary"], 3), round(v_["pmi_a1"], 3))
                                                           for k, v_ in s.get(f"{tag}_per_fold_DF_nogold", {}).items()})


def eval3():
    e = pd.read_csv(R / "eval3.tsv", sep="\t")
    print("\n=== eval3 (same-strand <= 50 nt vs opposite strands)")
    print(e.round(4).to_string(index=False))


def dc():
    f = R / "eval1_dc.tsv"
    if not f.exists():
        return
    e = pd.read_csv(f, sep="\t")
    e = e[e.method.notna()]
    print("\n=== exploratory decomposition: interaction-only (dc_) and marginal-only (marg_) AUROC")
    for neg in ("nogold", "straddle"):
        x = e[e.negset == neg]
        t = x.pivot_table(index=["split", "caller", "stratum"], columns="method", values="auc").round(3)
        d = x[x.method == "pmi_a1"].set_index(["split", "caller", "stratum"])[["ref_minus", "rm_lo", "rm_hi"]].round(3)
        d.columns = ["dc-pmi", "lo", "hi"]
        print(f"--- vs {neg}")
        print(t.join(d).to_string())
    for nm in ("eval2_dc.tsv", "eval3_dc.tsv"):
        if (R / nm).exists():
            print(f"--- {nm}")
            print(pd.read_csv(R / nm, sep="\t").round(3).to_string(index=False))


def runs():
    print("\n=== training (validation AUROC, best epoch)")
    for s in ["cv0", "cv1", "cv2", "cv3", "cv4", "phylum"]:
        f = RR / "runs" / s / "summary.json"
        if f.exists():
            j = json.load(open(f))
            print(s, j["selected_config"], j["genes"], j["pairs_offset1"], {m: (round(r["best_val_auc"], 4), r["best_epoch"])
                                                                           for m, r in j["training"].items()}, j["seconds"])


def loci():
    f = RR / "loci" / "loci_info.json"
    if not f.exists():
        return
    print("\n=== loci")
    print(json.dumps(json.load(open(f)), indent=1))
    for d in sorted((RR / "loci").glob("*/")):
        c = d / "compare_vs_engine.txt"
        if c.exists():
            print(f"--- {d.name}")
            print(c.read_text())
        rb = d / "ribosomal_check.txt"
        if rb.exists():
            print(rb.read_text())


def learned():
    f = R / "learned_top30.tsv"
    if f.exists():
        t = pd.read_csv(f, sep="\t")
        print("\n=== learned top 30")
        print(t.to_string(index=False))
        print(json.load(open(R / "learned_summary.json")))


for fn in (runs, eval1, eval2, eval3, dc, loci, learned):
    try:
        fn()
    except FileNotFoundError as e:
        print("missing", e)
