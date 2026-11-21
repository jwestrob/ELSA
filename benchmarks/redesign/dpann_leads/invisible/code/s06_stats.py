#!/usr/bin/env python
"""Step 6: control rates, candidate pass rates (overall, tier, support bin, list type, function source) with Wilson 95%
intervals, family table (families.tsv) and the 'fold kept, catalytic residues absent' table. Rule: assay_rule.json.
Writes families.tsv, data/rates.tsv, data/absent_site_families.tsv, data/stats.json."""
import json, math
from pathlib import Path
import pandas as pd
from scipy.stats import fisher_exact

H = Path("/Users/jacob/Documents/Sandbox/elsa_test/ELSA/benchmarks/redesign/dpann_leads/invisible")
O = pd.read_csv(H / "data/readout_full.tsv", sep="\t")
S = pd.read_csv(H / "data/selection_full.tsv", sep="\t")
RC = pd.read_csv(H / "data/candidate_ranking.tsv", sep="\t")
REFALL = pd.read_csv(H / "data/references.tsv", sep="\t")
PA = pd.read_parquet(H / "data/population_amended.parquet")
PA = PA[PA.tier.isin(["T2", "T2b", "rest"])]
nv = PA[~PA.placed_any]                       # profile-invisible, unannotated (amended novelty filter)
nv1 = PA[~PA.placed_any_v1]                   # first-pass novelty filter (used for sampling)
K = pd.read_csv(H / "../gapscreen/data/kegg_ko.tsv", sep="\t", keep_default_na=False).set_index("ko")
gate = json.load(open(H / "data/pilot_gate.json"))


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(max(0, c - h), 3), round(min(1, c + h), 3))


O["set"] = O.role.where(O.pair == "own", "negA")
O.loc[(O.role == "candidate") & (O.pair == "own"), "set"] = "candidate"
for c in ["PASS", "fold_match", "cat_all"]:
    O[c] = O[c].fillna(False).astype(bool)
pos = O[O.set == "positive"].groupby("func").PASS.agg(["sum", "size"])
informative = sorted(f for f, r in pos.iterrows() if r["sum"] / r["size"] >= 0.5)
inf_set = set(informative)
meta = nv1.set_index("protein_id")[["tier", "list_type", "support", "order", "phylum", "func_source", "genome", "proposed_label_name",
                                   "votes_for", "supporting_genomes"]]
cand = O[O.set == "candidate"].join(meta, on="protein_id")
cand = cand.merge(RC[["func", "protein_id", "cluster_size"]], on=["func", "protein_id"], how="left")
cand["named_by_amendment"] = cand.protein_id.isin(set(PA[PA.placed_any].protein_id))
cand["support_bin"] = pd.cut(cand.support, [-1, 0.05, 0.2, 0.5, 2], right=False, labels=["<5%", "5-20%", "20-50%", ">=50%"])
ci_all = cand[cand.func.isin(inf_set)]
ci = ci_all[~ci_all.named_by_amendment]
rows = []
add = lambda grp, key, d: rows.append(dict(group=grp, level=key, n=len(d), PASS=int(d.PASS.sum()), rate=round(d.PASS.mean(), 3) if len(d) else None,
                                           ci95=wilson(int(d.PASS.sum()), len(d)), fold_match=int(d.fold_match.sum()), cat_all=int(d.cat_all.sum())))
for s_ in ["positive", "negA", "negB"]:
    add("control_informative", s_, O[(O.set == s_) & O.func.isin(inf_set)])
    add("control_all_families", s_, O[O.set == s_])
negpool = O[O.set.isin(["negA", "negB"]) & O.func.isin(inf_set)]
add("control_informative", "negA+negB", negpool)
add("candidates", "all informative", ci)
add("candidates", "all families incl. uninformative (unannotated)", cand[~cand.named_by_amendment])
add("candidates", "named by CDD/PDB text (amendment stratum), informative", ci_all[ci_all.named_by_amendment])
for t in ["T2", "T2b", "rest"]:
    add("tier", t, ci[ci.tier == t])
for b in ["<5%", "5-20%", "20-50%", ">=50%"]:
    add("support", b, ci[ci.support_bin == b])
for lt in ["embedding_only", "both", "lists_disagree", "no_alignment_list"]:
    add("list_type", lt, ci[ci.list_type == lt])
for fs in ["partners", "label"]:
    add("func_source", fs, ci[ci.func_source == fs])
RT = pd.DataFrame(rows)
RT.to_csv(H / "data/rates.tsv", sep="\t", index=False)
k1, n1 = int(ci.PASS.sum()), len(ci)
k0, n0 = int(negpool.PASS.sum()), len(negpool)
st = dict(informative_families=informative, uninformative_families=sorted(set(pos.index) - inf_set),
          candidates_informative=f"{k1}/{n1}", negatives_pooled=f"{k0}/{n0}",
          fisher_candidates_vs_negatives_one_sided=fisher_exact([[k1, n1 - k1], [k0, n0 - k0]], alternative="greater")[1],
          weighted_candidate_rate=round(float((ci.PASS * ci.cluster_size).sum() / ci.cluster_size.sum()), 3) if n1 else None,
          pilot_gate=gate)
# families table: every function of the novel population
fam_rows = []
gn = nv.groupby("func").agg(novel_genes=("protein_id", "size"), novel_genomes=("genome", "nunique"),
                            embedding_only=("list_type", lambda s: int((s == "embedding_only").sum())),
                            partners=("func_source", lambda s: int((s == "partners").sum())),
                            orders=("order", lambda s: ",".join(s.value_counts().index[:3])))
absent = []
for r in REFALL.itertuples():
    f = r.func
    g = gn.loc[f] if f in gn.index else None
    d = O[O.func == f]
    c_ = d[(d.set == "candidate") & ~d.protein_id.isin(set(PA[PA.placed_any].protein_id))]; p_ = d[d.set == "positive"]; a_ = d[d.set == "negA"]; b_ = d[d.set == "negB"]
    if r.status != "ok":
        status = r.status.split(" (")[0]
    elif f not in pos.index:
        status = "not_assayed"
    elif f in inf_set:
        status = "informative"
    else:
        status = "uninformative_positive_control_failed"
    fm = c_[c_.fold_match]
    rec = dict(func=f, function=K.name.get(f, "")[:70], label=r.label, label_name=r.label_name, status=status,
               novel_genes=int(g.novel_genes) if g is not None else 0, novel_genomes=int(g.novel_genomes) if g is not None else 0,
               novel_embedding_only=int(g.embedding_only) if g is not None else 0, novel_partner_function=int(g.partners) if g is not None else 0,
               top_orders=g.orders if g is not None else "", ref_acc=r.ref_acc if r.status.startswith("ok") else "",
               ref_entry=r.ref_entry if r.status.startswith("ok") else "", ref_chain=r.pdb_chain if r.status.startswith("ok") else "",
               ref_region=r.region if r.status.startswith("ok") else "", ref_family_len=r.ref_fam_len if r.status.startswith("ok") else "",
               n_catalytic=r.n_active_residues if r.status.startswith("ok") else "", n_binding=r.n_binding_residues if r.status.startswith("ok") else "",
               novel_genes_first_pass=int((nv1.func == f).sum()),
               candidates_folded_named_by_amendment=int(((d.set == "candidate") & d.protein_id.isin(set(PA[PA.placed_any].protein_id))).sum()),
               candidates_folded=len(c_), cand_fold_match=int(c_.fold_match.sum()), cand_cat_all=int(c_.cat_all.sum()), cand_PASS=int(c_.PASS.sum()),
               cand_TM_ref_median=round(c_.TM_ref.median(), 3) if len(c_) else "",
               positives=f"{int(p_.PASS.sum())}/{len(p_)}" if len(p_) else "", negA=f"{int(a_.PASS.sum())}/{len(a_)}" if len(a_) else "",
               negB=f"{int(b_.PASS.sum())}/{len(b_)}" if len(b_) else "",
               fold_match_cat_all_share=round(fm.cat_all.mean(), 3) if len(fm) else "")
    fam_rows.append(rec)
    c_all = d[d.set == "candidate"]
    fm_all = c_all[c_all.fold_match]
    if status == "informative" and len(fm_all) >= 3 and fm_all.cat_all.mean() <= 0.2:
        named = set(PA[PA.placed_any].protein_id)
        absent.append(dict(func=f, function=rec["function"], label_name=r.label_name, ref=f"{r.ref_acc} {r.pdb_chain}",
                           stratum="all folded candidates", candidates_folded=len(c_all), fold_match=len(fm_all),
                           fold_match_named_by_amendment=int(fm_all.protein_id.isin(named).sum()),
                           all_catalytic_in_place=int(fm_all.cat_all.sum()), TM_ref_median=round(fm_all.TM_ref.median(), 3),
                           positives=rec["positives"], novel_genes=rec["novel_genes"], novel_genes_first_pass=rec["novel_genes_first_pass"],
                           novel_genomes=rec["novel_genomes"]))
    if False:
        absent.append(dict(func=f, function=rec["function"], label_name=r.label_name, ref=f"{r.ref_acc} {r.pdb_chain}",
                           candidates_folded=len(c_), fold_match=len(fm), all_catalytic_in_place=int(fm.cat_all.sum()),
                           TM_ref_median=round(fm.TM_ref.median(), 3), positives=rec["positives"], novel_genes=rec["novel_genes"],
                           novel_genomes=rec["novel_genomes"]))
FT = pd.DataFrame(fam_rows).sort_values(["status", "novel_genes"], ascending=[True, False])
FT.to_csv(H / "families.tsv", sep="\t", index=False)
pd.DataFrame(absent).to_csv(H / "data/absent_site_families.tsv", sep="\t", index=False)
st["absent_site_families"] = absent
json.dump(st, open(H / "data/stats.json", "w"), indent=1, default=str)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
print(RT.to_string(index=False))
print(json.dumps({k: v for k, v in st.items() if k != "pilot_gate"}, indent=1, default=str))
print(FT[FT.status.isin(["informative", "uninformative_positive_control_failed"])][
    ["func", "label_name", "status", "novel_genes", "candidates_folded", "cand_fold_match", "cand_cat_all", "cand_PASS", "cand_TM_ref_median",
     "positives", "negA", "negB", "fold_match_cat_all_share"]].to_string(index=False))
