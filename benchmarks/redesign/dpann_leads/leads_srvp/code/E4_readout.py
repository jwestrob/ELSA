#!/usr/bin/env python
"""Extended tier step 4 (local; rule assay_rule_extended.json): site read-out, PASS, family controls and rates.
In place = model residue identical to the reference-member residue at the structurally aligned position and the pair marked
':' by US-align (C-alpha < 5 A). PASS = TM normalized by the trimmed reference member >= 0.50 AND site share >= 0.90 (all
positions when there are 5 or fewer) AND mean pLDDT of the model residues aligned to site positions >= 70 (an unaligned site
position counts as not in place). Family kept: held-out positives >= 80% pass AND no negative A and no negative B passes
(fewer than 50 negatives of each kind per family, so the 2% ceiling means zero). Rates: unannotated stratum in kept families,
Wilson 95%, by tier and support bin; reported beside the fixed assay, never pooled with it.
usage: E4_readout.py COLLECTION   writes data/ext/{readout.tsv, residues.tsv, families_ext.tsv, rates.tsv, stats.json}"""
import json, math, sys
from pathlib import Path
import pandas as pd
from scipy.stats import fisher_exact

C = sys.argv[1]
CFG = json.load(open(Path(__file__).resolve().parent / "collections_leads.json"))[C]
H = Path(CFG["local_out"].replace("R/", "/Users/jacob/Documents/Sandbox/elsa_test/ELSA/"))
UA = (Path(CFG["fold_dir"]) if CFG["fold_dir"] else H / "folds") / "usalign"
E = H / "data/ext"
U = pd.read_csv(UA / "ext_usalign.tsv", sep="\t", keep_default_na=False, dtype={"protein_id": str})
S = pd.read_csv(E / "selection.tsv", sep="\t", dtype={"protein_id": str})
SR = pd.read_csv(E / "site_ref.tsv", sep="\t", keep_default_na=False)
FAM = pd.read_csv(E / "families.tsv", sep="\t")
REFS = pd.read_csv(E / "references_ext.tsv", sep="\t")
path_of = dict(zip(S.protein_id, S.fold_path))


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(max(0, c - h), 3), round(min(1, c + h), 3))


def plddt_of(p):
    return [float(l[60:66]) for l in open(p) if l.startswith("ATOM") and l[12:16].strip() == "CA"]


out, res_rows, exp_tm = [], [], {}
for r in U.itertuples():
    if r.pair == "exp":
        exp_tm[(r.protein_id, r.func)] = float(r.TM2) if r.TM2 != "" else None
        continue
    rec = dict(pair_id=r.pair_id, protein_id=r.protein_id, func=r.func, role=r.role, pair=r.pair, ref_func=r.ref_func)
    if r.TM2 == "" or not r.aln_ref:
        rec["status"] = "no_alignment"; out.append(rec); continue
    pl = plddt_of(path_of[r.protein_id])
    am, mk, ar = r.aln_model, r.aln_marks, r.aln_ref
    col_of_ref, model_at_col, i1, i2 = {}, {}, -1, -1
    for cidx, (x, y) in enumerate(zip(am, ar)):
        if x != "-":
            i1 += 1; model_at_col[cidx] = i1
        if y != "-":
            i2 += 1; col_of_ref[i2] = cidx
    site = SR[SR.func == r.ref_func]
    ok, n, pls = 0, 0, []
    for x in site.itertuples():
        col = col_of_ref.get(int(x.trimmed_index))
        m = am[col] if col is not None else "-"
        mark = mk[col] if col is not None else ""
        mi = model_at_col.get(col) if col is not None else None
        inplace = m == x.residue and mark == ":"
        n += 1; ok += inplace
        if mi is not None:
            pls.append(pl[mi])
        res_rows.append(dict(pair_id=r.pair_id, protein_id=r.protein_id, role=r.role, pair=r.pair, ref_func=r.ref_func, kind=x.kind,
                             member_pos=x.member_pos, ref_residue=x.residue, model_residue=m, mark=mark,
                             model_pos=(mi + 1) if mi is not None else "", plddt=pl[mi] if mi is not None else "", in_place=inplace,
                             note=x.note))
    share = ok / n if n else 0.0
    need = 1.0 if n <= 5 else 0.9
    site_pl = sum(pls) / len(pls) if pls else 0.0
    tm2 = float(r.TM2)
    rec.update(status="ok", TM_model=float(r.TM1), TM_ref=tm2, RMSD=float(r.RMSD), mean_plddt=round(sum(pl) / len(pl), 1),
               site=f"{ok}/{n}", site_share=round(share, 3), site_plddt=round(site_pl, 1), fold_match=tm2 >= 0.5,
               site_ok=share >= need - 1e-9)
    rec["PASS"] = bool(rec["fold_match"] and rec["site_ok"] and site_pl >= 70)
    out.append(rec)
O = pd.DataFrame(out)
O["TM_exp"] = [exp_tm.get((p, f)) for p, f in zip(O.protein_id, O.func)]
O["set"] = O.role.where(O.pair == "own", "negA")
O.loc[(O.role == "candidate") & (O.pair == "own"), "set"] = "candidate"
for c in ["PASS", "fold_match", "site_ok"]:
    O[c] = O[c].fillna(False).astype(bool)
meta = S[S.role == "candidate"].drop_duplicates(["func", "protein_id"]).set_index(["func", "protein_id"])
O["stratum"] = [meta.stratum.get((f, p), "") if s == "candidate" else "" for f, p, s in zip(O.func, O.protein_id, O.set)]
O.to_csv(E / "readout.tsv", sep="\t", index=False)
pd.DataFrame(res_rows).to_csv(E / "residues.tsv", sep="\t", index=False)
# family controls
fr = []
for f in REFS.func:
    d = O[O.func == f]
    p_, a_, b_ = d[d.set == "positive"], d[d.set == "negA"], d[d.set == "negB"]
    cu = d[(d.set == "candidate") & (d.stratum == "unannotated")]
    cn = d[(d.set == "candidate") & (d.stratum == "named")]
    pos_ok = len(p_) > 0 and p_.PASS.mean() >= 0.8
    neg_ok = all((len(x) >= 50 and x.PASS.mean() <= 0.02) or (len(x) < 50 and x.PASS.sum() == 0) for x in (a_, b_))
    fm = cu[cu.fold_match]
    fr.append(dict(func=f, kept=bool(pos_ok and neg_ok),
                   drop_reason="" if pos_ok and neg_ok else ("positives < 80%" if not pos_ok else "") + (" negatives pass" if not neg_ok else ""),
                   positives=f"{int(p_.PASS.sum())}/{len(p_)}", negA=f"{int(a_.PASS.sum())}/{len(a_)}", negB=f"{int(b_.PASS.sum())}/{len(b_)}",
                   positives_fold=f"{int(p_.fold_match.sum())}/{len(p_)}", positives_site=f"{int(p_.site_ok.sum())}/{len(p_)}",
                   unannotated_folded=len(cu), unannotated_fold_match=int(cu.fold_match.sum()), unannotated_site_ok=int(cu.site_ok.sum()),
                   unannotated_PASS=int(cu.PASS.sum()), unannotated_TM_ref_median=round(cu.TM_ref.median(), 3) if len(cu) else "",
                   unannotated_TM_exp_median=round(cu.TM_exp.dropna().median(), 3) if cu.TM_exp.notna().any() else "",
                   named_folded=len(cn), named_PASS=int(cn.PASS.sum()),
                   fold_match_site_ok_share=round(fm.site_ok.mean(), 3) if len(fm) else ""))
FR = pd.DataFrame(fr).merge(REFS[["func", "label_name", "ref_member", "trimmed_len", "n_site", "n_curated", "negA_func"]], on="func")
FR = FR.merge(FAM[["func", "curated_record", "exp_chain", "n_members", "n_reps"]], on="func", how="left")
FR.to_csv(E / "families_ext.tsv", sep="\t", index=False)
kept = set(FR[FR.kept].func)
pop = pd.read_parquet(H / "data/population.parquet")
pm = pop.assign(protein_id=pop.gid.astype(str)).set_index("protein_id")[["tier", "support"]]
cand = O[O.set == "candidate"].join(pm, on="protein_id")
cand["support_bin"] = pd.cut(cand.support, [-1, 0.05, 0.2, 0.5, 2], right=False, labels=["<5%", "5-20%", "20-50%", ">=50%"])
ci = cand[cand.func.isin(kept) & (cand.stratum == "unannotated")]
rows = []
add = lambda g, l, d: rows.append(dict(group=g, level=l, n=len(d), PASS=int(d.PASS.sum()), rate=round(d.PASS.mean(), 3) if len(d) else None,
                                       ci95=wilson(int(d.PASS.sum()), len(d)), fold_match=int(d.fold_match.sum()), site_ok=int(d.site_ok.sum())))
for s_ in ["positive", "negA", "negB"]:
    add("control_kept_families", s_, O[(O.set == s_) & O.func.isin(kept)])
    add("control_all_families", s_, O[O.set == s_])
neg = O[O.set.isin(["negA", "negB"]) & O.func.isin(kept)]
add("control_kept_families", "negA+negB", neg)
add("candidates", "unannotated, kept families", ci)
add("candidates", "already named, kept families", cand[cand.func.isin(kept) & (cand.stratum == "named")])
add("candidates", "unannotated, all in-scope families", cand[cand.stratum == "unannotated"])
for t in ["T2", "T2b", "rest"]:
    add("tier (unannotated, kept)", t, ci[ci.tier == t])
for b in ["<5%", "5-20%", "20-50%", ">=50%"]:
    add("support (unannotated, kept)", b, ci[ci.support_bin == b])
RT = pd.DataFrame(rows)
RT.to_csv(E / "rates.tsv", sep="\t", index=False)
k1, n1, k0, n0 = int(ci.PASS.sum()), len(ci), int(neg.PASS.sum()), len(neg)
st = dict(kept_families=sorted(kept), dropped_families=FR[~FR.kept][["func", "drop_reason"]].values.tolist(),
          unannotated_kept=f"{k1}/{n1}", negatives_kept=f"{k0}/{n0}",
          fisher_one_sided=fisher_exact([[k1, n1 - k1], [k0, n0 - k0]], alternative="greater")[1] if n1 and n0 else None,
          pass_by_family={f: f"{int(d.PASS.sum())}/{len(d)}" for f, d in ci.groupby("func")})
json.dump(st, open(E / "stats.json", "w"), indent=1, default=str)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(RT.to_string(index=False))
print(FR[["func", "label_name", "kept", "drop_reason", "positives", "negA", "negB", "unannotated_folded", "unannotated_fold_match",
          "unannotated_site_ok", "unannotated_PASS", "named_folded", "named_PASS", "n_site", "n_curated"]].to_string(index=False))
print(json.dumps(st, indent=1, default=str))
