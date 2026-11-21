#!/usr/bin/env python3
"""slot_rescue step 4 (Mac, light): bottom line, per-lead rows, controls, negatives, DL sets from results/calls.parquet.

Classes per predeclared.json: right / broad (wrong family, same broad class) / other (wrong) / none. Controls: false
rescue = right for the matched truth family or a CORE label. Genome bootstrap 2,000 draws, seed 17.
Writes results/bottom_line.tsv, per_lead.tsv, controls.tsv, negatives.tsv, dl.tsv, ranks.tsv, flanks.tsv,
per_gene_truth.tsv.gz, summary.json.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
D, R = HERE / "data", HERE / "results"
calls = pd.read_parquet(R / "calls.parquet")
T = pd.read_csv(D / "truth.tsv", sep="\t", keep_default_na=False)
U = pd.read_csv(D / "units.tsv", sep="\t", keep_default_na=False)
L = pd.read_csv(D / "labels.tsv", sep="\t", keep_default_na=False)
C = pd.read_csv(D / "controls.tsv", sep="\t", keep_default_na=False)
PRED = ["ribosomal_protein", "sec_pathway", "rna_polymerase", "atp_synthesis", "trna_modification", "membrane", "cytochrome_bd_oxidase"]
pred = {p: set(L.label[L[p] == 1]) for p in PRED}
lname = dict(zip(L.label, L.name))
CORE = set(pred["ribosomal_protein"])
for r in U[U.set.isin(["lead_X", "DL_slot", "DL_profile"])].itertuples():
    CORE |= {x for x in (r.R_pfam.split(";") + r.R_ko.split(";")) if x}

ROUTES = {"list_Z": ("Z", "k10"), "list_A": ("A", "k10"), "slot_Z": ("Z", "slot_pri"), "slot_A": ("A", "slot_pri"),
          "comb_Z": ("Z", "comb"), "comb_A": ("A", "comb")}
SECOND = {"list50_Z": ("Z", "k50"), "list50_A": ("A", "k50"), "slotsec_Z": ("Z", "slot_sec"), "slotsec_A": ("A", "slot_sec")}
ALLR = {**ROUTES, **SECOND}


def label_col(call):
    return {"k10": "k10_label", "k50": "k50_label", "comb": "comb_label", "slot_pri": "slotname_label", "slot_sec": "slotname_label"}[call]


# wide table: one row per query row, columns <route>_class / <route>_label
w = calls[calls.arm == "Z"][["row_id", "protein_id", "role", "set", "unit", "R_pfam", "R_ko", "broad_class", "gid"]].set_index("row_id")
for name, (arm, call) in ALLR.items():
    x = calls[calls.arm == arm].set_index("row_id")
    w[f"{name}_raw"] = x[f"{call}_class"]
    w[f"{name}_label"] = x[label_col(call)]
for arm in ("Z", "A"):
    x = calls[calls.arm == arm].set_index("row_id")
    for c in ("list_len", "first_right_rank", "comb_len", "comb_first_right_rank", "inlocus_pri", "inlocus_sec", "flank_all_pri",
              "flank_any_pri", "bonded_to_flank", "n_bonds", "slot_n_support", "n_flank_gids", "locus_size_pri"):
        w[f"{c}_{arm}"] = x[c]


def cls(raw, lab, broad):
    if raw == "right":
        return "right"
    if raw == "other":
        return "broad" if broad and lab in pred.get(broad, set()) else "other"
    return "none"


for name in ALLR:
    w[name] = [cls(a, b, c) for a, b, c in zip(w[f"{name}_raw"], w[f"{name}_label"], w.broad_class)]

# protein status for lead_X
lx = T[T.set == "lead_X"].merge(U[U.set == "lead_X"][["unit", "verification"]], on="unit", suffixes=("", "_u"))
st = {}
for p, g in lx.groupby("protein_id"):
    if g.R_pfam.nunique() > 1 or g.R_ko.nunique() > 1:
        st[p] = "ambiguous"
    elif (g.verification == "study_call").any():
        st[p] = "verified"
    else:
        st[p] = "UNVERIFIED"
w["status"] = [st.get(p, "") if s == "lead_X" else "" for p, s in zip(w.protein_id, w.set)]
bin_of = dict(zip(T.protein_id, T.bin_id))
bin_of.update(dict(zip(C.protein_id, C.bin_id)))
w["bin_id"] = w.protein_id.map(bin_of)
match = dict(zip(C.protein_id, C.matched_truth))
w["matched_truth"] = w.protein_id.map(match)

rng = np.random.default_rng(17)
NB = 2000


def boot(df, cols, fn):
    """Genome bootstrap of fn(df) per column; returns dict col -> (point, lo, hi)."""
    bins = df.bin_id.to_numpy()
    ub, inv = np.unique(bins, return_inverse=True)
    out = {}
    M = np.stack([fn(df[c].to_numpy()) for c in cols], axis=1).astype(float)   # per-row values
    S = np.zeros((len(ub), len(cols)))
    np.add.at(S, inv, M)
    cnt = np.bincount(inv, minlength=len(ub)).astype(float)
    draws = rng.integers(0, len(ub), size=(NB, len(ub)))
    num = np.stack([S[d].sum(0) for d in draws])
    den = np.array([cnt[d].sum() for d in draws])
    est = num / den[:, None]
    for j, c in enumerate(cols):
        out[c] = (M[:, j].mean(), *np.quantile(est[:, j], [.025, .975]))
    return out


def boot_diff(df, a, b, fn):
    bins = df.bin_id.to_numpy()
    ub, inv = np.unique(bins, return_inverse=True)
    v = fn(df[a].to_numpy()).astype(float) - fn(df[b].to_numpy()).astype(float)
    S = np.bincount(inv, weights=v, minlength=len(ub))
    cnt = np.bincount(inv, minlength=len(ub)).astype(float)
    draws = rng.integers(0, len(ub), size=(NB, len(ub)))
    est = np.array([S[d].sum() / cnt[d].sum() for d in draws])
    return v.mean(), *np.quantile(est, [.025, .975])


isright = lambda x: x == "right"
summary = {}
rows = []


def block(name, df, ctrl=None):
    """bottom-line row: n, right count/share per route (+ boot), and the matching controls' false-rescue rate."""
    df = df.drop_duplicates("protein_id")
    rec = dict(set=name, n=len(df))
    if len(df):
        b = boot(df, list(ROUTES), isright) if df.bin_id.nunique() > 1 else {}
        for r_ in ROUTES:
            k = int((df[r_] == "right").sum())
            rec[r_] = f"{k} ({k / len(df):.1%}" + (f" [{b[r_][1]:.1%}, {b[r_][2]:.1%}])" if r_ in b else ")")
            rec[r_ + "_n"] = k
            rec[r_ + "_broad"] = int((df[r_] == "broad").sum())
            rec[r_ + "_other"] = int((df[r_] == "other").sum())
            rec[r_ + "_none"] = int((df[r_] == "none").sum())
    if ctrl is not None and len(ctrl):
        rec["n_controls"] = len(ctrl)
        for r_ in ROUTES:
            fr = ((ctrl[r_] == "right") | ((ctrl[f"{r_}_raw"] == "other") & ctrl[f"{r_}_label"].isin(CORE)))
            rec[r_ + "_ctrl_false"] = f"{int(fr.sum())} ({fr.mean():.1%})"
            rec[r_ + "_ctrl_false_n"] = int(fr.sum())
    rows.append(rec)
    return rec


tr = w[w.role == "truth"]
ct = w[w.role == "control"]
ver = tr[(tr.set == "lead_X") & (tr.status == "verified")]
ver_ids = set(ver.protein_id)
block("lead_X verified", ver, ct[ct.matched_truth.isin(ver_ids)])
unv = tr[(tr.set == "lead_X") & (tr.status == "UNVERIFIED")]
block("lead_X UNVERIFIED", unv, ct[ct.matched_truth.isin(set(unv.protein_id))])
amb = tr[(tr.set == "lead_X") & (tr.status == "ambiguous")]
block("lead_X ambiguous", amb, ct[ct.matched_truth.isin(set(amb.protein_id))])
for s in ("DL_slot", "DL_profile"):
    sub = tr[tr.set == s]
    for u in sorted(sub.unit.unique()):
        su = sub[sub.unit == u]
        block(f"{s} {u}", su, ct[ct.matched_truth.isin(set(su.protein_id)) & (ct.set == f"control_{s}")])
    block(f"{s} total", sub, ct[ct.matched_truth.isin(set(sub.protein_id)) & (ct.set == f"control_{s}")])
block("controls (all, scored as truth)", ct.assign(), None)
BL = pd.DataFrame(rows)
BL.to_csv(R / "bottom_line.tsv", sep="\t", index=False)

# paired differences on verified lead_X
diffs = {}
for a, b in (("list_Z", "list_A"), ("slot_Z", "slot_A"), ("comb_Z", "comb_A"), ("slot_Z", "list_Z"), ("comb_Z", "list_Z"),
             ("slot_A", "list_A"), ("comb_A", "list_A"), ("comb_Z", "slot_Z")):
    v = boot_diff(ver.drop_duplicates("protein_id"), a, b, isright)
    diffs[f"{a} - {b}"] = [round(x, 4) for x in v]
summary["paired_differences_verified"] = diffs

# control false-rescue split, per route, with bootstrap on the verified-matched controls
cv = ct[ct.matched_truth.isin(ver_ids)].copy()
crow = []
for r_ in ALLR:
    raw, lab = cv[f"{r_}_raw"], cv[f"{r_}_label"]
    m = cv[r_] == "right"
    rib = (raw == "other") & lab.isin(pred["ribosomal_protein"])
    oc = (raw == "other") & lab.isin(CORE) & ~lab.isin(pred["ribosomal_protein"])
    noncore = (raw == "other") & ~lab.isin(CORE)
    cv["_f"] = m | rib | oc
    b = boot(cv, ["_f"], lambda x: x.astype(float))["_f"]
    crow.append(dict(route=r_, n=len(cv), named_matched_family=int(m.sum()), named_other_ribosomal=int(rib.sum()),
                     named_other_core=int(oc.sum()), false_rescue=int(cv._f.sum()), rate=f"{b[0]:.2%} [{b[1]:.2%}, {b[2]:.2%}]",
                     named_noncore=int(noncore.sum()), no_call=int((cv[r_] == "none").sum())))
CT = pd.DataFrame(crow)
CT.to_csv(R / "controls.tsv", sep="\t", index=False)

# negatives
own = {}
for p, g in lx.groupby("protein_id"):
    own[p] = {x for r in g.itertuples() for x in (r.R_pfam.split(";") + r.R_ko.split(";")) if x}
ng = tr[tr.set == "lead_negative"]
nrow = []
for u in sorted(ng.unit.unique()) + ["all"]:
    su = ng if u == "all" else ng[ng.unit == u]
    rec = dict(unit=u, n=len(su), also_lead_X=int(su.protein_id.isin(own).sum()))
    for r_ in ROUTES:
        raw, lab = su[f"{r_}_raw"], su[f"{r_}_label"]
        ownc = np.array([raw_ == "other" and lab_ in own.get(p, set()) for raw_, lab_, p in zip(raw, lab, su.protein_id)])
        rec[f"{r_}_namedX"] = int((raw == "right").sum())
        rec[f"{r_}_own"] = int(ownc.sum())
        rec[f"{r_}_core"] = int(((raw == "other") & lab.isin(CORE) & ~ownc).sum())
        rec[f"{r_}_noncore"] = int(((raw == "other") & ~lab.isin(CORE)).sum())
        rec[f"{r_}_none"] = int(raw.str.startswith("none").sum())
    nrow.append(rec)
NG = pd.DataFrame(nrow)
NG.to_csv(R / "negatives.tsv", sep="\t", index=False)

# per lead (proteins counted in each of their leads)
prow = []
for u in U[U.set.str.startswith("lead")].itertuples():
    su = tr[tr.unit == u.unit]
    rec = dict(unit=u.unit, set=u.set, call=u.study_call, flank=u.flank_names, verification=u.verification, ev=u.best_evidence_fraction,
               len_ratio=u.best_length_ratio, n=len(su))
    for r_ in ROUTES:
        rec[r_] = int((su[r_] == "right").sum())
        rec[r_ + "_broad"] = int((su[r_] == "broad").sum())
    for arm in ("Z", "A"):
        rec[f"empty_{arm}"] = int((su[f"list_len_{arm}"] == 0).sum())
        rec[f"inlocus_{arm}"] = int((su[f"inlocus_pri_{arm}"] == 1).sum())
        rec[f"flank_{arm}"] = int((su[f"flank_all_pri_{arm}"] == 1).sum())
    prow.append(rec)
PL = pd.DataFrame(prow)
PL.to_csv(R / "per_lead.tsv", sep="\t", index=False)

# ranks and flanks
rk = []
sets = {"lead_X verified": ver, "lead_X UNVERIFIED": unv, "DL_slot": tr[tr.set == "DL_slot"], "DL_profile": tr[tr.set == "DL_profile"],
        "controls (lead_X verified matched)": cv}
for nm, df in sets.items():
    df = df.drop_duplicates("protein_id")
    for arm in ("Z", "A"):
        for kind in ("first_right_rank", "comb_first_right_rank"):
            r_ = df[f"{kind}_{arm}"].to_numpy()
            ll = df[f"list_len_{arm}" if kind == "first_right_rank" else f"comb_len_{arm}"].to_numpy()
            hit = r_ > 0
            rk.append(dict(set=nm, arm=arm, list=("full" if kind == "first_right_rank" else "slot-restricted"), n=len(df),
                           empty=int((ll == 0).sum()), rank1=int((r_ == 1).sum()), le10=int(((r_ >= 1) & (r_ <= 10)).sum()),
                           le50=int(((r_ >= 1) & (r_ <= 50)).sum()), any=int(hit.sum()),
                           median_rank=float(np.median(r_[hit])) if hit.any() else None,
                           median_len=float(np.median(ll))))
RK = pd.DataFrame(rk)
RK.to_csv(R / "ranks.tsv", sep="\t", index=False)
fl = []
for nm, df in {"lead_X verified": ver, "lead_X UNVERIFIED": unv, "DL_slot": tr[tr.set == "DL_slot"], "DL_profile": tr[tr.set == "DL_profile"],
               "controls (lead_X verified matched)": cv}.items():
    df = df.drop_duplicates("protein_id")
    for arm in ("Z", "A"):
        hasf = df[f"n_flank_gids_{arm}"] > 0
        fl.append(dict(set=nm, arm=arm, n=len(df), in_locus_pri=int(df[f"inlocus_pri_{arm}"].sum()), in_locus_sec=int(df[f"inlocus_sec_{arm}"].sum()),
                       bonded=int((df[f"n_bonds_{arm}"] > 0).sum()), with_flanks=int(hasf.sum()),
                       locus_with_all_flanks=int(df[f"flank_all_pri_{arm}"].sum()), locus_with_a_flank=int(df[f"flank_any_pri_{arm}"].sum()),
                       bonded_to_flank=int(df[f"bonded_to_flank_{arm}"].sum()),
                       median_locus_size=float(df[f"locus_size_pri_{arm}"][df[f"inlocus_pri_{arm}"] == 1].median())
                       if (df[f"inlocus_pri_{arm}"] == 1).any() else None))
FL = pd.DataFrame(fl)
FL.to_csv(R / "flanks.tsv", sep="\t", index=False)

# wrong calls on verified truth: which labels
wl = {}
for r_ in ROUTES:
    x = ver.drop_duplicates("protein_id")
    x = x[x[r_].isin(["broad", "other"])]
    wl[r_] = (x[f"{r_}_label"].map(lambda a: f"{a} {lname.get(a, '')}").value_counts().head(8)).to_dict()
summary["wrong_labels_verified"] = wl
cl = {}
for r_ in ROUTES:
    x = cv[(cv[f"{r_}_raw"] == "other") & cv[f"{r_}_label"].isin(CORE) | (cv[r_] == "right")]
    cl[r_] = x.apply(lambda q: "matched:" + q.unit if q[r_] == "right" else f"{q[f'{r_}_label']} {lname.get(q[f'{r_}_label'], '')}",
                     axis=1).value_counts().head(8).to_dict() if len(x) else {}
summary["control_false_labels"] = cl
summary["core_labels"] = len(CORE)
summary["counts"] = dict(verified=len(ver_ids), unverified=int(unv.protein_id.nunique()), ambiguous=int(amb.protein_id.nunique()),
                         negatives=int(ng.protein_id.nunique()), controls=int(ct.protein_id.nunique()),
                         controls_verified_matched=len(cv))
json.dump(summary, open(R / "summary.json", "w"), indent=1, default=str)
w.reset_index().to_csv(R / "per_gene.tsv.gz", sep="\t", index=False)
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 80)
print(BL[["set", "n"] + list(ROUTES) + [r + "_ctrl_false" for r in ROUTES if r + "_ctrl_false" in BL]].to_string())
print(CT.to_string())
print(NG.to_string())
print(json.dumps(summary, indent=1, default=str)[:5000])
