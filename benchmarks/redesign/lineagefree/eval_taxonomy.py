#!/usr/bin/env python
"""lineagefree, EVALUATION ONLY: the places where this workstream reads taxonomy.

usage
  eval_taxonomy.py genusmap --out FILE
      genome_code -> group = the deployed engine's lineage (GTDB genus, one lineage per genome when missing), in the
      format recurrence_lf.py --lineage-map reads; used only to show that the lineage-map code reproduces v1.5.
  eval_taxonomy.py describe --groups g1.parquet [g2.parquet ...] --out JSON
      Each data-derived unit against GTDB ranks (species, genus, family): adjusted mutual information (sklearn,
      arithmetic normalisation), group count, share of groups that are pure at the rank, and the largest genus
      (Parvarchaeum) split. Descriptive; no threshold is chosen from it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

RD = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111")
GM = RD / "ws_dpann/package/universe/genomes.parquet"


def genus_codes():
    gm = pd.read_parquet(GM).sort_values("genome_code").reset_index(drop=True)
    genus = gm.genus.where(gm.genus.notna() & (gm.genus != ""), "genome:" + gm.genome)
    return gm, pd.factorize(genus)[0]


def cmd_genusmap(a):
    gm, g = genus_codes()
    pd.DataFrame(dict(genome=gm.genome, genome_code=gm.genome_code, group=g, weight=np.nan)).to_parquet(a.out, index=False)
    print("genera", int(g.max()) + 1)


def cmd_describe(a):
    from sklearn.metrics import adjusted_mutual_info_score as ami, adjusted_rand_score as ari
    gm = pd.read_parquet(GM).sort_values("genome_code").reset_index(drop=True)
    ranks = {r: pd.factorize(gm[c].fillna("na:" + gm.genome))[0] for r, c in (("species", "species"), ("genus", "genus"),
                                                                              ("family", "family"))}
    parv = (gm.genus == "Parvarchaeum").to_numpy()
    out = json.loads(Path(a.out).read_text()) if Path(a.out).exists() else {}
    for f in a.groups:
        d = pd.read_parquet(f).sort_values("genome_code").reset_index(drop=True)
        lab = pd.factorize(d.group)[0]
        item = dict(groups=int(lab.max()) + 1, effective_genomes_from_weights=float(d.weight.sum()) if "weight" in d and d.weight.notna().all() else None)
        for r, x in ranks.items():
            pure = pd.DataFrame(dict(g=lab, r=x)).groupby("g").r.nunique()
            item[r] = dict(n=int(x.max()) + 1, AMI=float(ami(x, lab)), ARI=float(ari(x, lab)),
                           groups_pure=float((pure == 1).mean()),
                           ranks_split=float((pd.DataFrame(dict(g=lab, r=x)).groupby("r").g.nunique() > 1).mean()))
        item["parvarchaeum"] = dict(genomes=int(parv.sum()), groups=int(len(np.unique(lab[parv]))),
                                    largest_group_share=float(np.bincount(lab[parv]).max() / parv.sum()))
        if "weight" in d and d.weight.notna().all():
            w = d.weight.to_numpy()
            item["parvarchaeum"]["weight_sum"] = float(w[parv].sum())
            gw = pd.Series(w).groupby(ranks["genus"]).sum()
            item["weight_sum_per_genus_q10_q50_q90_max"] = np.quantile(gw, [0.1, 0.5, 0.9, 1]).tolist()
        out[Path(f).stem] = item
        print(Path(f).stem, json.dumps(item))
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("genusmap")
    p.add_argument("--out", required=True)
    p = sub.add_parser("describe")
    p.add_argument("--groups", nargs="+", required=True)
    p.add_argument("--out", required=True)
    a = ap.parse_args()
    {"genusmap": cmd_genusmap, "describe": cmd_describe}[a.cmd](a)


if __name__ == "__main__":
    main()
