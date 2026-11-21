#!/usr/bin/env python3
"""Workstream repair: predeclared criteria for each token arm against the permissive base.

usage: evaluate.py <repair dir> <engine results dir> <base results dir> <arm> [<arm> ...]
(arm = token arm name without the __dpann suffix, e.g. mmseqs__seq__e3c50s75_rescue1)
Touched instance = gold instance (score_min2, >= 2 resolved members) with >= 1 member gene whose token differs
between arm and base; secondary set = >= 1 gene of its span (start_idx..end_idx) changed.
Paired genome bootstrap, 2,000 draws, seed 17, genomes resampled; split all / tune (odd genome_code) / test (even).
"""
import json, re, sys
from pathlib import Path
import numpy as np, pandas as pd

R, ENG, BASE = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
ARMS = sys.argv[4:]
RD = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111")
P = RD / "ws_dpann/package"
PD = json.load(open(R / "code/predeclared.json"))

gen = pd.read_parquet(P / "universe/genomes.parquet", columns=["genome", "genome_code"])
code = dict(zip(gen.genome, gen.genome_code))
inst = pd.read_parquet(P / "gold/instances.parquet", columns=["instance_id", "contig_code", "start_idx", "end_idx"])
ctg = pd.read_parquet(P / "universe/contigs.parquet", columns=["contig_code", "first_gid"])
inst = inst.merge(ctg, on="contig_code")
mem = pd.read_parquet(P / "gold/members.parquet", columns=["instance_id", "gid"])
B = pd.read_parquet(BASE / "score_min2/instance_scores.parquet").set_index("instance_id")
B["excess"] = B.f1 - B.control_mean_f1
gcode_all = B.genome.map(code).to_numpy()
mem = mem[mem.instance_id.isin(B.index)]
span = inst[inst.instance_id.isin(B.index)]
span_ids = np.repeat(span.instance_id.values, (span.end_idx - span.start_idx + 1).values)
span_gid = np.concatenate([np.arange(f + s, f + e + 1) for f, s, e in zip(span.first_gid.values, span.start_idx.values, span.end_idx.values)])

def boot(d, g, draws=2000):
    codes, inv = np.unique(g, return_inverse=True)
    rng = np.random.default_rng(17)
    pick = rng.integers(0, len(codes), size=(draws, len(codes)))
    out = []
    for v in d:
        s = np.bincount(inv, weights=v, minlength=len(codes)); n = np.bincount(inv, minlength=len(codes)).astype(float)
        m = s[pick].sum(1) / n[pick].sum(1)
        out.append([float(v.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))])
    return out

def summ(path):
    s = json.load(open(path / "summary.json"))
    return {k: s.get(k) for k in ("tokens", "significant_pairs", "fdr", "genes_covered", "genes_covered_frac",
                                  "significant_pairs_cross_phylum", "tokens_in_significant_pairs")} | \
           {"modules": s.get("modules", {}).get("modules"), "largest_module": s.get("modules", {}).get("largest")}

out = {"base": {"dir": str(BASE), **summ(BASE)}, "arms": {}}
rows = []
for arm in ARMS:
    d = ENG / f"dpann__{arm}__w3"
    A = pd.read_parquet(d / "score_min2/instance_scores.parquet").set_index("instance_id").loc[B.index]
    A["excess"] = A.f1 - A.control_mean_f1
    ch = pd.read_parquet(R / f"results/changed_{arm}__dpann.parquet")
    chs = set(ch.gid.values.tolist())
    touched_m = set(mem.instance_id.values[mem.gid.isin(chs).values])
    touched_s = set(span_ids[np.isin(span_gid, ch.gid.values)])
    res = {"dir": str(d), **summ(d), "genes_changed": int(len(ch)),
           "touched_member_instances": len(touched_m), "touched_span_instances": len(touched_s)}
    cmp = (d / "compare_vs_base.txt").read_text()
    for split in ("all", "tune \\(odd\\)", "test \\(even\\)"):
        m = re.search(rf"{split}\s+DefenseFinder\+TXSScan summed excess: A ([+-][\d.]+) B ([+-][\d.]+)\s+A-B ([+-][\d.]+) \[([+-][\d.]+), ([+-][\d.]+)\]", cmp)
        res[f"summed_excess_{split.split()[0]}"] = [float(x) for x in m.groups()]
        for caller in ("DefenseFinder", "TXSScan", "CRISPRCasTyper"):
            m = re.search(rf"{split}\s+{caller}\s+n=\s*([\d,]+)\s+F1 A ([\d.]+) B ([\d.]+) A-B ([+-][\d.]+) \[([+-][\d.]+), ([+-][\d.]+)\]\s+"
                          rf"excess A ([+-][\d.]+) B ([+-][\d.]+) A-B ([+-][\d.]+) \[([+-][\d.]+), ([+-][\d.]+)\]", cmp)
            res[f"all_instances_{split.split()[0]}_{caller}"] = {"n": int(m.group(1).replace(",", "")), "f1_A": float(m.group(2)),
                "f1_B": float(m.group(3)), "f1_diff": [float(m.group(i)) for i in (4, 5, 6)],
                "excess_A": float(m.group(7)), "excess_B": float(m.group(8)), "excess_diff": [float(m.group(i)) for i in (9, 10, 11)]}
    for setname, ts in (("member", touched_m), ("span", touched_s)):
        tmask = B.index.isin(list(ts))
        for split, sm in (("all", np.ones(len(B), bool)), ("odd", gcode_all % 2 == 1), ("even", gcode_all % 2 == 0)):
            for caller in ("pooled", "DefenseFinder", "TXSScan", "CRISPRCasTyper"):
                k = tmask & sm & ((B.caller.values == caller) if caller != "pooled" else True)
                if k.sum() == 0:
                    continue
                f1d, ctd, exd = boot([(A.f1.values - B.f1.values)[k], (A.control_mean_f1.values - B.control_mean_f1.values)[k],
                                      (A.excess.values - B.excess.values)[k]], gcode_all[k])
                r = {"arm": arm, "set": setname, "split": split, "caller": caller, "n": int(k.sum()),
                     "f1_A": float(A.f1.values[k].mean()), "f1_B": float(B.f1.values[k].mean()),
                     "f1_diff": f1d[0], "f1_lo": f1d[1], "f1_hi": f1d[2],
                     "ctrl_diff": ctd[0], "ctrl_lo": ctd[1], "ctrl_hi": ctd[2],
                     "excess_diff": exd[0], "excess_lo": exd[1], "excess_hi": exd[2]}
                rows.append(r)
                if setname == "member" and split == "even" and caller == "pooled":
                    res["c2_touched_even_pooled"] = r
    se = res["summed_excess_test"]
    c1 = se[3] > -0.005
    c2r = res.get("c2_touched_even_pooled")
    c2 = bool(c2r and c2r["f1_diff"] >= 0.03 and c2r["f1_lo"] > 0)
    res["c1_summed_excess_even_lo_gt_-0.005"] = bool(c1); res["c2_touched_f1_even_ge_0.03_ci_excl_0"] = c2
    res["adopted"] = bool(c1 and c2)
    out["arms"][arm] = res
T = pd.DataFrame(rows)
T.to_csv(R / "results/touched_instances.tsv", sep="\t", index=False, float_format="%.4f")
json.dump(out, open(R / "results/evaluation.json", "w"), indent=1)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(T[(T.set == "member") & (T.caller == "pooled")].to_string(index=False, float_format=lambda v: f"{v:+.3f}"))
for arm, r in out["arms"].items():
    print(arm, "genes_changed", r["genes_changed"], "touched", r["touched_member_instances"], "/", r["touched_span_instances"],
          "summed_excess_even", r["summed_excess_test"], "c1", r["c1_summed_excess_even_lo_gt_-0.005"],
          "c2", r["c2_touched_f1_even_ge_0.03_ci_excl_0"], "ADOPT", r["adopted"],
          "sig_pairs", r["significant_pairs"], "fdr", round(r["fdr"], 5), "genes_covered", r["genes_covered"])
print("base", out["base"])
