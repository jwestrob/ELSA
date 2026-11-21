#!/usr/bin/env python
"""Extended tier step 5 (local): family controls (keep: held-out positives >= 80% pass; negatives A and B each <= 2%, no pass
allowed below 50 negatives of a kind), pass rates of unannotated and already-named candidates in kept families (overall,
tier, support bin) with Wilson 95% intervals, family table. Never pooled with the fixed assay.
usage: E5_stats.py COLLECTION   writes families_extended.tsv, data/ext/rates_ext.tsv, data/ext/stats_ext.json"""
import json, math, sys
from pathlib import Path
import pandas as pd
from scipy.stats import fisher_exact

C = sys.argv[1]
H = Path(json.load(open(Path(__file__).resolve().parent / "collections_leads.json"))[C]["local_out"].replace(
    "R/", "/Users/jacob/Documents/Sandbox/elsa_test/ELSA/"))
E = H / "data/ext"
O = pd.read_csv(E / "readout_ext.tsv", sep="\t", dtype={"protein_id": str})
FAM = pd.read_csv(E / "families.tsv", sep="\t")
RX = pd.read_csv(E / "references_ext.tsv", sep="\t", keep_default_na=False)
pop = pd.read_parquet(H / "data/population.parquet")
pop = pop[pop.tier.isin(["T2", "T2b", "rest"])].assign(protein_id=lambda d: d.gid.astype(str))


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(max(0, c - h), 3), round(min(1, c + h), 3))


O["set"] = O.role.where(O.pair == "own", "negA")
O.loc[(O.role == "candidate") & (O.pair == "own"), "set"] = "candidate"
for c in ("PASS", "fold_match", "site_ok"):
    O[c] = O[c].fillna(False).astype(bool)
O = O.merge(pop[["protein_id", "tier", "support", "named", "order"]], on="protein_id", how="left")
neg_ok = lambda d: (d.PASS.sum() == 0) if len(d) < 50 else (d.PASS.mean() <= 0.02)
kept, fam_rows = [], []
for f in sorted(set(FAM.func)):
    fr = FAM[FAM.func == f].iloc[0]
    d = O[O.func == f]
    p_, a_, b_ = d[d.set == "positive"], d[d.set == "negA"], d[d.set == "negB"]
    cu = d[(d.set == "candidate") & (d.named == False)]  # noqa: E712
    cn = d[(d.set == "candidate") & (d.named == True)]  # noqa: E712
    if not fr.in_scope:
        status = f"out_of_scope: {fr.reason}"
    elif f not in set(RX.func):
        status = "no_reference_model"
    elif len(p_) == 0:
        status = "not_assayed"
    else:
        reasons = []
        if p_.PASS.mean() < 0.8:
            reasons.append(f"positives {int(p_.PASS.sum())}/{len(p_)}")
        if len(a_) and not neg_ok(a_):
            reasons.append(f"negA {int(a_.PASS.sum())}/{len(a_)}")
        if len(b_) and not neg_ok(b_):
            reasons.append(f"negB {int(b_.PASS.sum())}/{len(b_)}")
        status = "kept" if not reasons else "dropped: " + ", ".join(reasons)
        if not reasons:
            kept.append(f)
    rx = RX[RX.func == f]
    fam_rows.append(dict(func=f, label_name=fr.label_name, ecs=fr.ecs, fixed_status=fr.fixed_status, status=status, genes=fr.genes,
                         unannotated=fr.unannotated, n_members=fr.n_members, n_reps=fr.get("n_reps", ""),
                         n_site=int(rx.n_site.iloc[0]) if len(rx) else "", n_curated=int(rx.n_curated.iloc[0]) if len(rx) else "",
                         curated_record=fr.get("curated_record", ""), exp_chain=fr.get("exp_chain", ""),
                         ref_member=int(rx.ref_member.iloc[0]) if len(rx) else "",
                         positives=f"{int(p_.PASS.sum())}/{len(p_)}" if len(p_) else "", negA=f"{int(a_.PASS.sum())}/{len(a_)}" if len(a_) else "",
                         negB=f"{int(b_.PASS.sum())}/{len(b_)}" if len(b_) else "",
                         unannotated_folded=len(cu), unannotated_fold=int(cu.fold_match.sum()), unannotated_site=int(cu.site_ok.sum()),
                         unannotated_PASS=int(cu.PASS.sum()), named_folded=len(cn), named_PASS=int(cn.PASS.sum()),
                         cand_TM_median=round(cu.TM_ref.median(), 3) if len(cu) else ""))
FT = pd.DataFrame(fam_rows).sort_values(["status", "genes"], ascending=[True, False])
FT.to_csv(H / "families_extended.tsv", sep="\t", index=False)
K = O[O.func.isin(kept)]
cand = K[K.set == "candidate"].copy()
cand["support_bin"] = pd.cut(cand.support, [-1, 0.05, 0.2, 0.5, 2], right=False, labels=["<5%", "5-20%", "20-50%", ">=50%"])
cu, cn = cand[cand.named == False], cand[cand.named == True]  # noqa: E712
rows = []
add = lambda g, l, d: rows.append(dict(group=g, level=l, n=len(d), PASS=int(d.PASS.sum()), rate=round(d.PASS.mean(), 3) if len(d) else None,
                                       ci95=wilson(int(d.PASS.sum()), len(d)), fold=int(d.fold_match.sum()), site=int(d.site_ok.sum())))
for s_ in ("positive", "negA", "negB"):
    add("control_kept_families", s_, K[K.set == s_])
    add("control_all_assayed_families", s_, O[O.set == s_])
negs = K[K.set.isin(["negA", "negB"])]
add("control_kept_families", "negA+negB", negs)
add("candidates", "unannotated, kept families", cu)
add("candidates", "already named, kept families", cn)
for t in ("T2", "T2b", "rest"):
    add("tier (unannotated)", t, cu[cu.tier == t])
for b in ("<5%", "5-20%", "20-50%", ">=50%"):
    add("support (unannotated)", b, cu[cu.support_bin == b])
RT = pd.DataFrame(rows)
RT.to_csv(E / "rates_ext.tsv", sep="\t", index=False)
k1, n1, k0, n0 = int(cu.PASS.sum()), len(cu), int(negs.PASS.sum()), len(negs)
st = dict(collection=C, kept_families=kept, dropped=[r.func for r in FT.itertuples() if str(r.status).startswith("dropped")],
          unannotated=f"{k1}/{n1}", negatives=f"{k0}/{n0}",
          fisher_one_sided=fisher_exact([[k1, n1 - k1], [k0, n0 - k0]], alternative="greater")[1] if n1 and n0 else None,
          status_counts=FT.status.str.split(":").str[0].value_counts().to_dict())
json.dump(st, open(E / "stats_ext.json", "w"), indent=1, default=str)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
print(RT.to_string(index=False))
print(json.dumps(st, indent=1, default=str))
print(FT[~FT.status.str.startswith("out_of_scope")][["func", "label_name", "status", "genes", "unannotated", "n_reps", "n_site",
      "n_curated", "positives", "negA", "negB", "unannotated_folded", "unannotated_PASS", "named_folded", "named_PASS"]].to_string(index=False))
