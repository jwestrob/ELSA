#!/usr/bin/env python
"""Final tables (lut, local): (1) extended-tier 'fold kept, site absent' families, by the fixed rule's family-level
definition applied to the extended read-out (assayed family, >= 3 unannotated candidates with fold match, <= 20% of them
with the site in place; dropped families included and flagged); (2) assay rows appended to funnel.tsv (fixed assay and
extended tier, unannotated T2/T2b/rest stratum, counts as genes); (3) frame-challenge summary data/frame_challenge.json.
usage: X4_tables.py lut"""
import json, sys
from pathlib import Path
import pandas as pd

C = sys.argv[1]
R = Path("/Users/jacob/Documents/Sandbox/elsa_test/ELSA")
H = Path(json.load(open(Path(__file__).resolve().parent / "collections_leads.json"))[C]["local_out"].replace("R/", str(R) + "/"))
E = H / "data/ext"
O = pd.read_csv(E / "readout_ext.tsv", sep="\t", dtype={"protein_id": str})
FE = pd.read_csv(H / "families_extended.tsv", sep="\t")
pop = pd.read_parquet(H / "data/population.parquet").assign(protein_id=lambda d: d.gid.astype(str))
c = O[(O.role == "candidate") & (O.pair == "own")].merge(pop[["protein_id", "named"]], on="protein_id")
c = c[~c.named]
rows = []
for f, d in c.groupby("func"):
    fm = d[d.fold_match == True]  # noqa: E712
    st = FE.set_index("func").status.get(f, "")
    if len(fm) >= 3 and fm.site_ok.mean() <= 0.2:
        rows.append(dict(func=f, label_name=FE.set_index("func").label_name.get(f, ""), family_status=st, candidates_folded=len(d),
                         fold_match=len(fm), site_in_place=int(fm.site_ok.sum()), site_counts=",".join(sorted(fm.site)),
                         TM_ref_median=round(fm.TM_ref.median(), 3), positives=FE.set_index("func").positives.get(f, "")))
pd.DataFrame(rows).to_csv(E / "absent_site_families_ext.tsv", sep="\t", index=False)
print(pd.DataFrame(rows).to_string(index=False))
# funnel rows
FT = pd.read_csv(H / "funnel.tsv", sep="\t")
FT = FT[~FT.step.str.startswith(("fixed:", "extended:"))]
REF = pd.read_csv(H / "data/references.tsv", sep="\t")
P = pop[pop.tier.isin(["T2", "T2b", "rest"]) & ~pop.named]
st7 = json.load(open(H / "data/stats.json"))
fx = pd.read_csv(H / "data/readout_full.tsv", sep="\t", dtype={"protein_id": str})
fxc = fx[(fx.role == "candidate") & (fx.pair == "own")].merge(pop[["protein_id", "named", "tier"]], on="protein_id")
fxc = fxc[~fxc.named]
ste = json.load(open(E / "stats_ext.json"))
exc = c.merge(pop[["protein_id", "tier"]], on="protein_id")
BR = json.load(open(E / "budget_rule.json"))
EF = pd.read_csv(E / "families.tsv", sep="\t")
add = []
cnt = lambda d, name: add.append(dict(step=name, **{t: int((d.tier == t).sum()) for t in ["T2", "T2b", "rest"]}, main_T2_T2b_rest=len(d)))
cnt(P[P.func.isin(REF[REF.status == "ok"].func)], "fixed: unannotated genes in families with a curated reference")
cnt(fxc, "fixed: unannotated genes folded (one per 90% cluster)")
cnt(fxc[fxc.func.isin(st7["informative_families"])], "fixed: folded, informative families (positives >= 50%)")
cnt(fxc[fxc.func.isin(st7["informative_families"]) & fxc.PASS], "fixed: PASS")
cnt(P[P.func.isin(EF[EF.in_scope == True].func)], "extended: unannotated genes in in-scope families")  # noqa: E712
cnt(P[P.func.isin(BR["families"])], "extended: in budget families (32)")
cnt(exc, "extended: folded (cap 10 per family)")
cnt(exc[exc.func.isin(ste["kept_families"])], "extended: folded, kept families")
cnt(exc[exc.func.isin(ste["kept_families"]) & exc.PASS], "extended: PASS (conserved-core residues in place)")
FT = pd.concat([FT, pd.DataFrame(add)], ignore_index=True)
FT.to_csv(H / "funnel.tsv", sep="\t", index=False)
print(FT.to_string(index=False))
# frame challenge summary
D = pd.read_csv(H / "data/split_neighbors.tsv", sep="\t", dtype={"protein_id": str})
T = pd.read_csv(H / "data/partner_homology.tsv", sep="\t", dtype={"protein_id": str})
fc = dict(assayed_unannotated=len(T), homolog_of_partner=int(T.homolog.sum()),
          partner_score_gt_negB=round(float((T.partner_score > T.negB_score).mean()), 3),
          cover_partner_label_domain=int((D.label_env_covered >= 0.5).sum()),
          not_cover_label_next_door=int(D[(D.label_env_covered < 0.5)].label_next_door.sum()),
          not_cover=int((D.label_env_covered < 0.5).sum()),
          cover_label_next_door=int(D[(D.label_env_covered >= 0.5)].label_next_door.sum()),
          fold_match_cover=f"{int(D[D.label_env_covered >= 0.5].fold_match.sum())}/{int((D.label_env_covered >= 0.5).sum())}",
          fold_match_not_cover=f"{int(D[D.label_env_covered < 0.5].fold_match.sum())}/{int((D.label_env_covered < 0.5).sum())}",
          pass_cover=f"{int(D[D.label_env_covered >= 0.5].PASS.sum())}/{int((D.label_env_covered >= 0.5).sum())}")
json.dump(fc, open(H / "data/frame_challenge.json", "w"), indent=1)
print(json.dumps(fc, indent=1))
