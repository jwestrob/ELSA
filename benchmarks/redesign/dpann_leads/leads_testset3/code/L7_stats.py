#!/usr/bin/env python
"""Leads pipeline step 7 (local): control rates, pass rates of the unannotated and the already-named strata (overall, tier,
support bin) with Wilson 95% intervals, family table (families.tsv) and the 'fold kept, catalytic residues absent' table,
following dpann_leads/invisible/code/s06_stats.py (no list-type split: the collections have no alignment lists).
usage: L7_stats.py COLLECTION [full|pilot]   writes families.tsv, data/rates.tsv, data/absent_site_families.tsv, data/stats.json"""
import json, math, sys
from pathlib import Path
import pandas as pd
from scipy.stats import fisher_exact

C = sys.argv[1]
MODE = sys.argv[2] if len(sys.argv) > 2 else "full"   # 'pilot' when the fixed assay stopped at the pilot gate
H = Path(json.load(open(Path(__file__).resolve().parent / "collections_leads.json"))[C]["local_out"].replace(
    "R/", "/Users/jacob/Documents/Sandbox/elsa_test/ELSA/"))
O = pd.read_csv(H / f"data/readout_{MODE}.tsv", sep="\t")
RC = pd.read_csv(H / "data/candidate_ranking.tsv", sep="\t")
REFALL = pd.read_csv(H / "data/references.tsv", sep="\t")
pop = pd.read_parquet(H / "data/population.parquet")
pop = pop[pop.tier.isin(["T2", "T2b", "rest"])]
K = pd.read_csv("/Users/jacob/Documents/Sandbox/elsa_test/ELSA/benchmarks/redesign/dpann_leads/gapscreen/data/kegg_ko.tsv", sep="\t",
                keep_default_na=False).set_index("ko")
gate = json.load(open(H / "data/pilot_gate.json"))


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(max(0, c - h), 3), round(min(1, c + h), 3))


O["set"] = O.role.where(O.pair == "own", "negA")
O.loc[(O.role == "candidate") & (O.pair == "own"), "set"] = "candidate"
for c in ["PASS", "fold_match", "cat_all"]:
    O[c] = O[c].fillna(False).astype(bool)
pos = O[O.set == "positive"].groupby("func").PASS.agg(["sum", "size"])
inf = sorted(f for f, r in pos.iterrows() if r["sum"] / r["size"] >= 0.5)
inf_set = set(inf)
meta = pop.assign(protein_id=pop.gid.astype(int)).set_index("protein_id")[["tier", "support", "order", "phylum", "named", "genome"]]
cand = O[O.set == "candidate"].join(meta, on="protein_id")
cand = cand.merge(RC[["func", "protein_id", "cluster_size"]], on=["func", "protein_id"], how="left")
cand["support_bin"] = pd.cut(cand.support, [-1, 0.05, 0.2, 0.5, 2], right=False, labels=["<5%", "5-20%", "20-50%", ">=50%"])
cand["stratum"] = cand.named.map({True: "named", False: "unannotated"})
ci_all = cand[cand.func.isin(inf_set)]
ci = ci_all[ci_all.stratum == "unannotated"]
rows = []
add = lambda grp, key, d: rows.append(dict(group=grp, level=key, n=len(d), PASS=int(d.PASS.sum()), rate=round(d.PASS.mean(), 3) if len(d) else None,
                                           ci95=wilson(int(d.PASS.sum()), len(d)), fold_match=int(d.fold_match.sum()), cat_all=int(d.cat_all.sum())))
for s_ in ["positive", "negA", "negB"]:
    add("control_informative", s_, O[(O.set == s_) & O.func.isin(inf_set)])
    add("control_all_families", s_, O[O.set == s_])
negpool = O[O.set.isin(["negA", "negB"]) & O.func.isin(inf_set)]
add("control_informative", "negA+negB", negpool)
add("candidates", "unannotated, informative families", ci)
add("candidates", "already named, informative families", ci_all[ci_all.stratum == "named"])
add("candidates", "unannotated, all families", cand[cand.stratum == "unannotated"])
for t in ["T2", "T2b", "rest"]:
    add("tier (unannotated)", t, ci[ci.tier == t])
for b in ["<5%", "5-20%", "20-50%", ">=50%"]:
    add("support (unannotated)", b, ci[ci.support_bin == b])
RT = pd.DataFrame(rows)
RT.to_csv(H / "data/rates.tsv", sep="\t", index=False)
k1, n1 = int(ci.PASS.sum()), len(ci)
k0, n0 = int(negpool.PASS.sum()), len(negpool)
st = dict(collection=C, mode=MODE, informative_families=inf, uninformative_families=sorted(set(pos.index) - inf_set),
          candidates_unannotated_informative=f"{k1}/{n1}", negatives_pooled=f"{k0}/{n0}",
          fisher_unannotated_vs_negatives_one_sided=fisher_exact([[k1, n1 - k1], [k0, n0 - k0]], alternative="greater")[1] if n1 and n0 else None,
          fisher_vs_dpann_rate_two_sided=fisher_exact([[k1, n1 - k1], [25, 107]])[1] if n1 else None,
          weighted_rate=round(float((ci.PASS * ci.cluster_size).sum() / ci.cluster_size.sum()), 3) if n1 else None,
          pass_by_family={f: f"{int(d.PASS.sum())}/{len(d)}" for f, d in ci.groupby("func")}, pilot_gate=gate)
fam_rows, absent = [], []
gn = pop.groupby("func").agg(genes=("gid", "size"), genomes=("genome", "nunique"), unannotated=("named", lambda s: int((~s).sum())),
                             orders=("order", lambda s: ",".join(s.astype(str).value_counts().index[:3])))
for r in REFALL.itertuples():
    f = r.func
    g = gn.loc[f] if f in gn.index else None
    d = O[O.func == f]
    cu = d[(d.set == "candidate") & d.protein_id.isin(set(cand[cand.stratum == "unannotated"].protein_id))]
    cn = d[(d.set == "candidate") & d.protein_id.isin(set(cand[cand.stratum == "named"].protein_id))]
    p_, a_, b_ = d[d.set == "positive"], d[d.set == "negA"], d[d.set == "negB"]
    status = (r.status.split(" (")[0] if r.status != "ok" else "not_assayed" if f not in pos.index else
              "informative" if f in inf_set else "uninformative_positive_control_failed")
    okref = str(r.status).startswith("ok")
    fm = cu[cu.fold_match]
    rec = dict(func=f, function=K.name.get(f, "")[:70], label=r.label, label_name=r.label_name, status=status,
               genes=int(g.genes) if g is not None else 0, genomes=int(g.genomes) if g is not None else 0,
               unannotated_genes=int(g.unannotated) if g is not None else 0, top_orders=g.orders if g is not None else "",
               ref_acc=r.ref_acc if okref else "", ref_entry=r.ref_entry if okref else "", ref_chain=r.pdb_chain if okref else "",
               ref_family_len=r.ref_fam_len if okref else "", n_catalytic=r.n_active_residues if okref else "",
               n_binding=r.n_binding_residues if okref else "",
               unannotated_folded=len(cu), unannotated_fold_match=int(cu.fold_match.sum()), unannotated_cat_all=int(cu.cat_all.sum()),
               unannotated_PASS=int(cu.PASS.sum()), unannotated_TM_ref_median=round(cu.TM_ref.median(), 3) if len(cu) else "",
               named_folded=len(cn), named_PASS=int(cn.PASS.sum()),
               positives=f"{int(p_.PASS.sum())}/{len(p_)}" if len(p_) else "", negA=f"{int(a_.PASS.sum())}/{len(a_)}" if len(a_) else "",
               negB=f"{int(b_.PASS.sum())}/{len(b_)}" if len(b_) else "",
               fold_match_cat_all_share=round(fm.cat_all.mean(), 3) if len(fm) else "")
    fam_rows.append(rec)
    c_all = d[d.set == "candidate"]
    fm_all = c_all[c_all.fold_match]
    if status == "informative" and len(fm_all) >= 3 and fm_all.cat_all.mean() <= 0.2:
        absent.append(dict(func=f, function=rec["function"], label_name=r.label_name, ref=f"{r.ref_acc} {r.pdb_chain}",
                           candidates_folded=len(c_all), fold_match=len(fm_all), all_catalytic_in_place=int(fm_all.cat_all.sum()),
                           fold_match_named=int(fm_all.protein_id.isin(set(cn.protein_id)).sum()),
                           TM_ref_median=round(fm_all.TM_ref.median(), 3), positives=rec["positives"], genes=rec["genes"]))
FT = pd.DataFrame(fam_rows).sort_values(["status", "genes"], ascending=[True, False])
FT.to_csv(H / "families.tsv", sep="\t", index=False)
pd.DataFrame(absent).to_csv(H / "data/absent_site_families.tsv", sep="\t", index=False)
st["absent_site_families"] = absent
json.dump(st, open(H / "data/stats.json", "w"), indent=1, default=str)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
print(RT.to_string(index=False))
print(json.dumps({k: v for k, v in st.items() if k != "pilot_gate"}, indent=1, default=str))
print(FT[FT.status.isin(["informative", "uninformative_positive_control_failed"])][
    ["func", "label_name", "status", "genes", "unannotated_genes", "unannotated_folded", "unannotated_fold_match", "unannotated_cat_all",
     "unannotated_PASS", "named_folded", "named_PASS", "positives", "negA", "negB"]].to_string(index=False))
