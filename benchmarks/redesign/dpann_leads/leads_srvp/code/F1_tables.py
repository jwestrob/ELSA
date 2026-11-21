#!/usr/bin/env python
"""Final tables (local). usage: F1_tables.py COLLECTION
families.tsv      fixed-assay family rows (L7_stats.py output, assay = fixed) followed by extended-tier rows for the 98 in-scope
                  families (assay = extended; status kept / dropped_<reason> / not_planned_budget) and the out-of-scope reasons.
data/absent_site_all.tsv  'fold kept, site residues absent': family informative / kept, >= 3 candidates (both strata) with a fold
                  match, <= 20% of them with every site residue in place (fixed rule's family_level_absent_site; the same
                  counting applied to the extended tier's site). Observation only.
funnel.tsv        L2 funnel plus assay steps for both tiers.
data/summary.json counts quoted in REPORT.txt."""
import json, sys
from pathlib import Path
import pandas as pd

C = sys.argv[1]
H = Path(json.load(open(Path(__file__).resolve().parent / "collections_leads.json"))[C]["local_out"].replace(
    "R/", "/Users/jacob/Documents/Sandbox/elsa_test/ELSA/"))
E = H / "data/ext"
FX = pd.read_csv(H / "families.tsv", sep="\t") if "assay" not in pd.read_csv(H / "families.tsv", sep="\t", nrows=1).columns else \
    pd.read_csv(H / "families.tsv", sep="\t").query("assay == 'fixed'").drop(columns=["assay"])
FX.insert(0, "assay", "fixed")
E1 = pd.read_csv(E / "families.tsv", sep="\t")
plan = json.load(open(E / "budget_plan.json"))
planned = {p["func"]: p for p in plan["plan"]}
FR = pd.read_csv(E / "families_ext.tsv", sep="\t").set_index("func")
REFNOW = pd.read_csv(H / "data/references.tsv", sep="\t").set_index("func").status
pop = pd.read_parquet(H / "data/population.parquet")
pm = pop[pop.tier.isin(["T2", "T2b", "rest"])]
rows = []
for r in E1.itertuples():
    moved = REFNOW.get(r.func, "") in ("ok", "ok_but_length_incompatible")
    if not r.in_scope or moved:
        st = "out_of_scope: " + ("fixed reference after EC fix" if moved else str(r.reason))
    elif r.func not in planned:
        st = "in_scope_not_planned_budget"
    elif r.func not in FR.index:
        st = "planned_no_reference_member"
    else:
        st = "kept" if FR.loc[r.func, "kept"] else "dropped: " + str(FR.loc[r.func, "drop_reason"]).strip()
    d = pm[pm.func == r.func]
    rec = dict(assay="extended", func=r.func, label=r.label, label_name=r.label_name, status=st, genes=len(d), genomes=d.genome.nunique(),
               unannotated_genes=int((~d.named).sum()), n_members=r.n_members, n_reps=getattr(r, "n_reps", ""),
               curated_record=r.curated_record if isinstance(getattr(r, "curated_record", ""), str) else "",
               exp_chain=r.exp_chain if isinstance(getattr(r, "exp_chain", ""), str) else "")
    if r.func in FR.index:
        x = FR.loc[r.func]
        rec.update(ref_member=x.ref_member, ref_family_len=x.trimmed_len, n_site=x.n_site, n_curated=x.n_curated, positives=x.positives,
                   negA=x.negA, negB=x.negB, unannotated_folded=x.unannotated_folded, unannotated_fold_match=x.unannotated_fold_match,
                   unannotated_site_ok=x.unannotated_site_ok, unannotated_PASS=x.unannotated_PASS,
                   unannotated_TM_ref_median=x.unannotated_TM_ref_median, unannotated_TM_exp_median=x.unannotated_TM_exp_median,
                   named_folded=x.named_folded, named_PASS=x.named_PASS)
    rows.append(rec)
EX = pd.DataFrame(rows).sort_values(["status", "unannotated_genes", "genes"], ascending=[True, False, False])
pd.concat([FX, EX], ignore_index=True).to_csv(H / "families.tsv", sep="\t", index=False)
# absent-site table, both tiers
ab = []
O = pd.read_csv(H / "data/readout_full.tsv", sep="\t")
O["set"] = O.role.where(O.pair == "own", "negA"); O.loc[(O.role == "candidate") & (O.pair == "own"), "set"] = "candidate"
inf = set(json.load(open(H / "data/stats.json"))["informative_families"])
for f, d in O[(O.set == "candidate") & O.func.isin(inf)].groupby("func"):
    fm = d[d.fold_match == True]  # noqa: E712
    if len(fm) >= 3 and fm.cat_all.mean() <= 0.2:
        ab.append(dict(assay="fixed", func=f, candidates=len(d), fold_match=len(fm), all_site_in_place=int(fm.cat_all.sum()),
                       TM_ref_median=round(fm.TM_ref.median(), 3)))
EO = pd.read_csv(E / "readout.tsv", sep="\t", dtype={"protein_id": str})
ER = pd.read_csv(E / "residues.tsv", sep="\t", dtype={"protein_id": str}, keep_default_na=False)
kept = set(FR[FR.kept].index)
for f, d in EO[(EO.set == "candidate") & EO.func.isin(kept)].groupby("func"):
    fm = d[d.fold_match == True]  # noqa: E712
    if len(fm) >= 3 and fm.site_ok.mean() <= 0.2:
        rr = ER[ER.pair_id.isin(fm.pair_id) & (ER.pair == "own")]
        per = "; ".join(f"{a}{p}: {int(g.in_place.sum())}/{len(g)} ({''.join(sorted(set(g.model_residue)))})"
                        for (p, a), g in rr.groupby(["member_pos", "ref_residue"]))
        pos = EO[(EO.func == f) & (EO.set == "positive")]
        ab.append(dict(assay="extended", func=f, label_name=FR.loc[f, "label_name"], candidates=len(d), fold_match=len(fm),
                       fold_match_unannotated=int((fm.stratum == "unannotated").sum()), all_site_in_place=int(fm.site_ok.sum()),
                       TM_ref_median=round(fm.TM_ref.median(), 3), positives=f"{int(pos.PASS.sum())}/{len(pos)}", site_positions=per))
AB = pd.DataFrame(ab)
AB.to_csv(H / "data/absent_site_all.tsv", sep="\t", index=False)
# funnel
FU = pd.read_csv(H / "funnel.tsv", sep="\t")
FU = FU[FU.step.str.contains("assay") == False] if "step" in FU.columns else FU  # noqa: E712
st = json.load(open(H / "data/stats.json")); est = json.load(open(E / "stats.json"))
REF = pd.read_csv(H / "data/references.tsv", sep="\t")
add = []
okf = set(REF[REF.status == "ok"].func)
add.append(dict(step="fixed assay: function has a curated reference (status ok)", main_T2_T2b_rest=int(pm.func.isin(okf).sum()),
                note=f"unannotated {int((pm.func.isin(okf) & ~pm.named).sum())}; {len(okf)} functions"))
add.append(dict(step="fixed assay: informative family (positives >= 50%)", main_T2_T2b_rest=int(pm.func.isin(inf).sum()),
                note=f"unannotated {int((pm.func.isin(inf) & ~pm.named).sum())}; {len(inf)} families"))
add.append(dict(step="fixed assay: unannotated pass in informative families", main_T2_T2b_rest=None, note=st["candidates_unannotated_informative"]))
ins = EX[~EX.status.str.startswith("out_of_scope")]
add.append(dict(step="extended tier: in scope (>= 15 reps at 70%, >= 3 site positions)", main_T2_T2b_rest=int(ins.genes.sum()),
                note=f"unannotated {int(ins.unannotated_genes.sum())}; {len(ins)} families"))
pl = EX[EX.status.isin(["kept"]) | EX.status.str.startswith("dropped")]
add.append(dict(step="extended tier: assayed under the fold budget", main_T2_T2b_rest=int(pl.genes.sum()),
                note=f"unannotated {int(pl.unannotated_genes.sum())}; {len(pl)} families"))
kp = EX[EX.status == "kept"]
add.append(dict(step="extended tier: kept family (positives >= 80%, no negative passes)", main_T2_T2b_rest=int(kp.genes.sum()),
                note=f"unannotated {int(kp.unannotated_genes.sum())}; {len(kp)} families"))
add.append(dict(step="extended tier: unannotated pass in kept families (one gene per 90% cluster)", main_T2_T2b_rest=None, note=est["unannotated_kept"]))
FU2 = pd.concat([FU, pd.DataFrame(add)], ignore_index=True)
FU2.to_csv(H / "funnel.tsv", sep="\t", index=False)
summ = dict(fixed=st, extended=est, absent_site=ab)
json.dump(summ, open(H / "data/summary.json", "w"), indent=1, default=str)
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 120)
print(FU2.to_string(index=False)); print(AB.to_string(index=False))
print(EX.status.str.split(":").str[0].value_counts().to_string())
