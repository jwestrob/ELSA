#!/usr/bin/env python3
"""Summary tables for the lut slice (workstream slice_lut). Runs on the small result tables.

usage: 08_summarize.py RESULTS_DIR TABLES_DIR SCOPE [SCOPE ...]
Reads RESULTS_DIR/<scope>/{loci.tsv, genomes.tsv}; writes TABLES_DIR/<scope>_*.tsv:
  by_rank_<def>.tsv       genomes and loci per genus / family / order (with slice denominators)
  copies_<def>.tsv        loci per genome distribution
  architectures_<def>.tsv architecture strings (AM1 orientation, '.' = gene without a component call, '-' = opposite strand)
                          and their component-only core strings, with counts
  components_<def>.tsv    fraction of loci carrying each component (core >= 0.9 of loci)
  mdh_<def>.tsv           K23995 / K14028 / K00114 within 25 genes and in the genome, for genomes with and without loci
"""
import os, sys
import pandas as pd

R, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
COMP = ["A", "B", "C", "D", "E", "F", "G", "H", "M", "I"]


def arch_class(cp):
    c = set(str(cp).split(","))
    abef = len(c & {"A", "B", "E", "F"})
    if {"A", "B", "C", "D", "E", "F", "G", "H", "I"} <= c:
        return "1_complete_AtoI" + ("_M" if "M" in c else "")
    if "H" in c and abef >= 2:
        return "2_receptor_plus_transporter"
    if "H" in c:
        return "4_receptor_module_without_transporter"
    if abef >= 2:
        return "3_transporter_module_without_receptor"
    return "5_other"
for sc in sys.argv[3:]:
    L = pd.read_csv(f"{R}/{sc}/loci.tsv", sep="\t", dtype={"components_present": str, "components_missing": str})
    G = pd.read_csv(f"{R}/{sc}/genomes.tsv", sep="\t")
    for d in ["main", "strict", "loose"]:
        l = L[L.definition == d].copy()
        l["components_present"] = l.components_present.fillna("")
        rows = []
        for rank in ["ord", "family", "genus"]:
            den = G.groupby(rank).size().rename("slice_genomes")
            gl = l.groupby(rank).agg(loci=("locus_id", "count"), genomes_with_locus=("genome", "nunique"))
            t = den.to_frame().join(gl, how="left").fillna(0).astype(int).reset_index().rename(columns={rank: "taxon"})
            t.insert(0, "rank", {"ord": "order", "family": "family", "genus": "genus"}[rank])
            t["frac_genomes_with_locus"] = (t.genomes_with_locus / t.slice_genomes).round(4)
            rows.append(t.sort_values(["genomes_with_locus", "slice_genomes"], ascending=False))
        pd.concat(rows).to_csv(f"{OUT}/{sc}_by_rank_{d}.tsv", sep="\t", index=False)
        cp = l.groupby("genome").size().value_counts().sort_index().rename_axis("loci_per_genome").rename("genomes").reset_index()
        cp.to_csv(f"{OUT}/{sc}_copies_{d}.tsv", sep="\t", index=False)
        if len(l) == 0:
            continue
        l["core"] = l.architecture.map(lambda a: " ".join(t for t in a.split() if t not in (".", "-.")))
        l["arch_class"] = l.components_present.map(arch_class)
        ac = l.groupby("arch_class").agg(loci=("locus_id", "count"), genomes=("genome", "nunique")).reset_index()
        ac["families"] = ac.arch_class.map(lambda c: "; ".join(f"{k} {v}" for k, v in l[l.arch_class == c].family.value_counts().head(6).items()))
        ac.sort_values("loci", ascending=False).to_csv(f"{OUT}/{sc}_arch_classes_{d}.tsv", sep="\t", index=False)
        l[["locus_id", "genome", "family", "genus", "components_present", "arch_class"]].to_csv(f"{OUT}/{sc}_locus_classes_{d}.tsv", sep="\t", index=False)
        # receptor outside the locus: genomes whose loci lack H but that carry a reverse-best lutH homolog elsewhere
        gl = l.groupby("genome").components_present.agg(lambda v: set(",".join(v).split(",")))
        noH = gl[gl.map(lambda v: "H" not in v)].index
        if "rb_H" in G:
            gH = G.set_index("genome").rb_H
            split = pd.DataFrame({"genome": noH, "rb_H_elsewhere": [int(gH.get(g, 0)) for g in noH]})
            split = split.merge(G[["genome", "family", "genus"]], on="genome")
            split.to_csv(f"{OUT}/{sc}_receptor_elsewhere_{d}.tsv", sep="\t", index=False)
        ar = l.groupby(["core"]).agg(loci=("locus_id", "count"), genomes=("genome", "nunique"),
                                      families=("family", lambda x: ",".join(sorted(x.value_counts().index[:4]))),
                                      example=("architecture", lambda x: x.value_counts().index[0])).reset_index()
        ar.sort_values("loci", ascending=False).to_csv(f"{OUT}/{sc}_architectures_{d}.tsv", sep="\t", index=False)
        cf = pd.DataFrame({"component": COMP,
                           "loci_with": [int(l.components_present.str.split(",").map(lambda v: c in v).sum()) for c in COMP]})
        cf["frac"] = (cf.loci_with / len(l)).round(4); cf["core_ge_0.9"] = cf.frac >= 0.9
        cf.to_csv(f"{OUT}/{sc}_components_{d}.tsv", sep="\t", index=False)
        has = G.genome.isin(set(l.genome))
        m = []
        for k in ["K23995", "K14028", "K00114"]:
            col = f"ko_{k}"
            if col not in G:
                continue
            m.append(dict(ko=k, genomes_with_locus=int(has.sum()), with_locus_and_ko=int(((G[col] > 0) & has).sum()),
                          genomes_without_locus=int((~has).sum()), without_locus_with_ko=int(((G[col] > 0) & ~has).sum()),
                          loci_with_ko_within_25=int(l.mdh_ko_within_25_genes.fillna("").str.contains(k).sum()), loci=len(l)))
        pd.DataFrame(m).to_csv(f"{OUT}/{sc}_mdh_{d}.tsv", sep="\t", index=False)
    print(sc, "done")
