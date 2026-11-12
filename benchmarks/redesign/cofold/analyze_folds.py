"""Step 2 analysis: folds.tsv (every fold with its class), class distributions, per-candidate hit calls
against predeclared_criteria.json. Usage: python analyze_folds.py work/folds_set1 [more fold dirs]"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu

HERE = Path(__file__).resolve().parent
W = HERE / "work"
CRIT = json.load(open(HERE / "predeclared_criteria.json"))["hit"]


def load_metrics(dirs):
    rows = []
    for d in dirs:
        for line in open(Path(d) / "metrics.jsonl"):
            m = json.loads(line)
            pc = m.get("pair_chains_iptm") or [[None, None], [None, None]]
            rows.append(dict(fold_id=m["fold_id"], iptm=m["iptm"], ptm=m["ptm"], pc_ab=pc[0][1], pc_ba=pc[1][0],
                             iface_pae=m.get("iface_pae"), pae_inter_min=m.get("pae_inter_min"),
                             pae_inter_mean=m.get("pae_inter_mean"), contacts=m["contacts"],
                             interface_plddt=m.get("interface_plddt"), plddt_A=m["chain_plddt"][0],
                             plddt_B=m["chain_plddt"][1], iface_A=m["iface_residues"].get("A"),
                             iface_B=m["iface_residues"].get("B"), fold_dir=str(d)))
    return pd.DataFrame(rows).drop_duplicates("fold_id", keep="last")


def crit13(df):
    return ((df.iptm >= CRIT["iptm_min_both_replicates"])
            & (df.iface_pae.fillna(99) <= CRIT["iface_pae_max_both_replicates"])
            & (df.contacts >= CRIT["contacts_min_both_replicates"])
            & (df.interface_plddt.fillna(0) >= CRIT["interface_plddt_min_both_replicates"]))


def main():
    dirs = sys.argv[1:] or [W / "folds_set1"]
    man = pd.read_csv(W / "fold_manifest.tsv", sep="\t")
    met = load_metrics(dirs)
    f = man.merge(met, on="fold_id", how="left")
    f["folded"] = f.iptm.notna()
    f["pass_c123"] = crit13(f) & f.folded
    f.to_csv(W / "folds.tsv", sep="\t", index=False)

    # ---- class distributions --------------------------------------------------------------
    out = {}
    q = [0.1, 0.25, 0.5, 0.75, 0.9]
    rows = []
    for cls, d in f[f.folded].groupby("cls"):
        r = dict(cls=cls, n=len(d), **{f"iptm_q{int(x*100)}": round(float(d.iptm.quantile(x)), 3) for x in q},
                 iptm_ge_0p5=int((d.iptm >= 0.5).sum()), iptm_ge_0p75=int((d.iptm >= 0.75).sum()),
                 pass_c123=int(d.pass_c123.sum()), frac_c123=round(float(d.pass_c123.mean()), 4),
                 iface_pae_med=round(float(d.iface_pae.median()), 2), contacts_med=float(d.contacts.median()))
        rows.append(r)
    cd = pd.DataFrame(rows).set_index("cls")
    print(cd.to_string())
    cd.to_csv(W / "class_distributions.tsv", sep="\t")
    tests = {}
    cog = f[(f.cls == "cognate") & f.folded]
    for other in ("noncognate", "uncalled", "positive"):
        o = f[(f.cls == other) & f.folded]
        if len(o) == 0:
            continue
        u = mannwhitneyu(cog.iptm, o.iptm, alternative="two-sided")
        t = [[int(cog.pass_c123.sum()), int((~cog.pass_c123).sum())], [int(o.pass_c123.sum()), int((~o.pass_c123).sum())]]
        fe = fisher_exact(t)
        tests[f"cognate_vs_{other}"] = dict(mw_p=float(u.pvalue), median_cog=float(cog.iptm.median()),
                                            median_other=float(o.iptm.median()), c123_table=t,
                                            fisher_or=float(fe.statistic), fisher_p=float(fe.pvalue))
    print(json.dumps(tests, indent=1))

    # ---- per candidate ----------------------------------------------------------------------
    cand = pd.read_csv(W / "candidates_all.tsv", sep="\t")
    top = cand[cand.nonredundant_rank.notna() & (cand.nonredundant_rank <= 150)].copy()
    rows = []
    for _, c in top.iterrows():
        rk = int(c.nonredundant_rank)
        r = dict(cand_rank=rk, type_key=c.type_key)
        for cls, tag in (("cognate", "cog"), ("noncognate", "non"), ("uncalled", "unc")):
            for rep in (1, 2):
                x = f[(f.cls == cls) & (f.cand_rank == rk) & (f.rep == rep)]
                if len(x) and x.folded.iloc[0]:
                    x = x.iloc[0]
                    r[f"{tag}{rep}_iptm"] = x.iptm
                    if tag == "cog":
                        for k in ("iface_pae", "contacts", "interface_plddt", "plddt_A", "plddt_B", "pc_ab", "pc_ba",
                                  "genome", "phylum", "ord", "lengths"):
                            r[f"cog{rep}_{k}"] = x[k]
                        r[f"cog{rep}_c123"] = bool(x.pass_c123)
                else:
                    r[f"{tag}{rep}_iptm"] = np.nan
        r["delta1"] = r["cog1_iptm"] - r["non1_iptm"]
        r["delta2"] = r["cog2_iptm"] - r["non2_iptm"]
        r["c1_both"] = bool(r.get("cog1_iptm", 0) >= CRIT["iptm_min_both_replicates"] and r.get("cog2_iptm", 0) >= CRIT["iptm_min_both_replicates"])
        r["c123_both"] = bool(r.get("cog1_c123", False) and r.get("cog2_c123", False))
        r["c4_both"] = bool(r["delta1"] >= CRIT["cognate_minus_noncognate_iptm_min_each_replicate"]
                            and r["delta2"] >= CRIT["cognate_minus_noncognate_iptm_min_each_replicate"])
        r["promiscuous_kept_chain"] = bool((r["non1_iptm"] >= 0.6) or (r["non2_iptm"] >= 0.6))
        r["hit_predeclared"] = r["c123_both"] and r["c4_both"]
        r["min_cog_iptm"] = np.nanmin([r["cog1_iptm"], r["cog2_iptm"]])
        rows.append(r)
    pc = pd.DataFrame(rows)
    keep = ["n_genomes", "n_genus", "n_family", "n_ord", "n_class", "n_phylum", "phyla", "len_med", "fam_hyd_med",
            "intergenic_med_nt", "same_strand_frac", "gap0_frac_min", "block_integrity", "in_conserved_locus_frac", "modules"]
    pc = pc.merge(top[["type_key"] + keep], on="type_key")
    pc = pc.sort_values(["hit_predeclared", "min_cog_iptm"], ascending=[False, False])
    pc.to_csv(W / "per_candidate_folds.tsv", sep="\t", index=False)
    summ = dict(folded=int(f.folded.sum()), total=len(f), class_counts=f[f.folded].cls.value_counts().to_dict(),
                c1_both=int(pc.c1_both.sum()), c123_both=int(pc.c123_both.sum()), c4_both=int(pc.c4_both.sum()),
                hits=int(pc.hit_predeclared.sum()), promiscuous=int(pc.promiscuous_kept_chain.sum()), tests=tests)
    json.dump(summ, open(W / "fold_summary.json", "w"), indent=1, default=float)
    print(json.dumps({k: v for k, v in summ.items() if k != "tests"}, indent=1))
    cols = ["cand_rank", "type_key", "cog1_iptm", "cog2_iptm", "non1_iptm", "non2_iptm", "unc1_iptm", "unc2_iptm",
            "cog1_iface_pae", "cog2_iface_pae", "cog1_c123", "cog2_c123", "c4_both", "hit_predeclared", "n_genomes", "n_ord",
            "n_phylum", "len_med"]
    print(pc[cols].head(30).to_string())

    # positives by type
    pos = f[(f.cls == "positive") & f.folded]
    print(pos.groupby("type_key").agg(n=("iptm", "size"), iptm_med=("iptm", "median"), iptm_max=("iptm", "max"),
                                      c123=("pass_c123", "sum")).sort_values("iptm_med", ascending=False).to_string())


if __name__ == "__main__":
    main()
