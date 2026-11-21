"""Census tables from s3_census.py outputs (work/s3/census/): writes s3_tables/census_*.tsv and prints them.

Block category (from chain statuses; order not_in_tier < no_hit < unknown_only < cdd_only < resembles_characterised
< annotated):
  R1 annotated family members      most-recognised chain = annotated
  R2 resembles characterised       most-recognised chain = resembles_characterised
  R3 CDD only                      most-recognised chain = cdd_only
  U0 partly outside the tier       every chain no_hit / unknown_only / not_in_tier, >= 1 not_in_tier
  U1 unrecognised, unknown-fn only every chain no_hit / unknown_only, >= 1 unknown_only
  U2 unrecognised, no hit          every chain no_hit
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
C = HERE / "work" / "s3" / "census"
OUT = HERE / "s3_tables"
ORDER = ["not_in_tier", "no_hit", "unknown_only", "cdd_only", "resembles_characterised", "annotated"]
CATS = ["R1 annotated family members", "R2 resembles characterised (profile)", "R3 CDD only",
        "U0 unrecognised, >=1 chain outside the tier", "U1 unrecognised, unknown-function only", "U2 unrecognised, no hit"]


def category(r):
    st = r.chain_status.split(";")
    if r.most_status == "annotated":
        return CATS[0]
    if r.most_status == "resembles_characterised":
        return CATS[1]
    if r.most_status == "cdd_only":
        return CATS[2]
    if "not_in_tier" in st:
        return CATS[3]
    if "unknown_only" in st:
        return CATS[4]
    return CATS[5]


def bins(x, edges, labels):
    return pd.cut(x, edges, right=False, labels=labels)


def main():
    OUT.mkdir(exist_ok=True)
    ct = pd.read_parquet(C / "census_types.parquet")
    inst = pd.read_parquet(C / "census_instances.parquet", columns=["type_key", "gids"])
    st = pd.read_parquet(C / "token_status.parquet")
    summ = json.load(open(C / "census_summary.json"))
    ct["category"] = ct.apply(category, axis=1)
    ct["orders_bin"] = bins(ct.n_ord, [1, 2, 3, 5, 10, 10 ** 6], ["1", "2", "3-4", "5-9", ">=10"])
    ct["phyla_bin"] = bins(ct.n_phylum, [1, 2, 3, 4, 10 ** 6], ["1", "2", "3", ">=4"])
    ct["k_bin"] = ct.k.map(lambda k: str(k) if k <= 4 else "5-8")
    cat_of = ct.set_index("type_key").category
    inst["category"] = cat_of.reindex(inst.type_key).to_numpy()
    rec = ct.set_index("type_key").pass_recurrence
    inst["rec"] = rec.reindex(inst.type_key).to_numpy()

    def ugenes(mask):
        s = set()
        for x in inst.gids[mask]:
            s.update(int(v) for v in x)
        return len(s)

    rows = []
    for c in CATS + ["all"]:
        d = ct if c == "all" else ct[ct.category == c]
        m = np.ones(len(inst), bool) if c == "all" else (inst.category == c).to_numpy()
        rows.append(dict(category=c, types=len(d), types_ge5g_ge2ord=int(d.pass_recurrence.sum()),
                         types_pass_fold=int(d.pass_fold.sum()), instances=int(d.n_instances.sum()),
                         unique_genes=ugenes(m), unique_genes_ge5g_ge2ord=ugenes(m & inst.rec.to_numpy()),
                         folded_screens_1_2=int((d.folded_s12 != "").sum())))
    t1 = pd.DataFrame(rows)
    t1["types_frac"] = (t1.types / len(ct)).round(4)
    t1.to_csv(OUT / "census_paper_table.tsv", sep="\t", index=False)

    def cross(col, sub=None):
        d = ct if sub is None else ct[sub]
        a = pd.crosstab(d[col], d.category).reindex(columns=CATS, fill_value=0)
        a["all"] = a.sum(1)
        a["unrecognised_U1_U2_frac"] = ((a[CATS[4]] + a[CATS[5]]) / a["all"]).round(3)
        return a
    least = ct.groupby("least_status").agg(types=("k", "size"), instances=("n_instances", "sum")).reindex(ORDER)
    most = ct.groupby("most_status").agg(types=("k", "size"), instances=("n_instances", "sum")).reindex(ORDER)
    lm = pd.crosstab(ct.least_status, ct.most_status).reindex(index=ORDER, columns=ORDER, fill_value=0)
    least.to_csv(OUT / "census_by_least_status.tsv", sep="\t")
    most.to_csv(OUT / "census_by_most_status.tsv", sep="\t")
    lm.to_csv(OUT / "census_least_x_most.tsv", sep="\t")
    tabs = {"k": cross("k_bin"), "orders": cross("orders_bin"), "phyla": cross("phyla_bin"),
            "k_ge5g_ge2ord": cross("k_bin", ct.pass_recurrence), "orders_ge5g_ge2ord": cross("orders_bin", ct.pass_recurrence),
            "phyla_ge5g_ge2ord": cross("phyla_bin", ct.pass_recurrence)}
    for k, v in tabs.items():
        v.to_csv(OUT / f"census_by_{k}.tsv", sep="\t")
    # filters among unrecognised blocks
    u = ct[ct.category.isin(CATS[4:])]
    filt = {c: int(u[c].sum()) for c in ["pass_darkfam", "pass_len2", "pass_hyd", "pass_lowcomplex", "pass_copies",
                                          "pass_geometry", "pass_recurrence", "pass_fold"]}
    filt["types"] = len(u)
    filt["pass_fold_folded_screens_1_2"] = int((u.pass_fold & (u.folded_s12 != "")).sum())
    filt["pass_fold_new"] = int((u.pass_fold & (u.folded_s12 == "")).sum())
    # family level
    fam = st.status.value_counts().reindex(ORDER).to_frame("families")
    fam.to_csv(OUT / "census_family_status.tsv", sep="\t")
    nit = st[st.status == "not_in_tier"].groupby(["not_in_tier_reason", "not_in_tier_cdd"]).size()
    out = dict(summary=summ, unrecognised_filters=filt, not_in_tier={f"{a}|{b}": int(v) for (a, b), v in nit.items()},
               family_status=fam.families.to_dict(),
               family_status_sensitivity=dict(
                   no_hit_with_best_tier_prob_ge50=int(((st.status == "no_hit") & (st.tier_best_prob >= 50)).sum()),
                   no_hit_with_best_tier_prob_ge90_short=int(((st.status == "no_hit") & (st.tier_best_prob >= 90)).sum()),
                   no_hit_with_member_family_char=int(((st.status == "no_hit") & (st.frac_members_char_family > 0)).sum()),
                   no_hit_with_cdd_any_member=int(((st.status == "no_hit") & (st.cdd_members_any > 0)).sum()),
                   no_hit=int((st.status == "no_hit").sum())))
    json.dump(out, open(OUT / "census_extra.json", "w"), indent=1, default=str)
    pd.set_option("display.width", 250)
    print(t1.to_string(index=False))
    print(least.to_string(), "\n", most.to_string(), "\n", lm.to_string())
    for k, v in tabs.items():
        print("\n==", k)
        print(v.to_string())
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
