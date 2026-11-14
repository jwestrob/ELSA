"""Genomic context of the candidate blocks (local, seconds). Usage: python dd_context.py c1|c2

Windows (dd_prep.py): the +-5 genes around every engine block instance, and around up to 3 control cores per
instance on the same contig (same gene count; first gene length within +-25 % of the block's first gene; >= 6
genes away). Labels per neighbour gene, observed annotations only (Sharur DPANN database, seal 805c93a6):
  tok:<engine token>, mod:<engine module>, pfam:<name>, ko:<KOfam accession>, txss:<raw TXSScan profile>,
  df:<raw DefenseFinder profile>, sys:<curated system type> (system_proteins of txsscan_system,
  defensefinder_system, cctyper), dark (no annotation row at all).
Enrichment: windows (instances) with the label among real windows vs control windows, restricted to instances
that have >= 1 control; Fisher exact (one-sided, greater) with Benjamini-Hochberg over labels in >= 3 real
windows. Real-window frequencies over all instances are reported as well.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

HERE = Path(__file__).resolve().parent
W = HERE / "work"
CW = HERE.parent / "work"


def bh(p):
    p = np.asarray(p, float)
    n = len(p)
    o = np.argsort(p)
    q = p[o] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n)
    out[o] = np.minimum(q, 1)
    return out


def labels():
    ann = pd.read_csv(W / "annotations.tsv", sep="\t")
    cur = pd.read_csv(W / "curated_members.tsv", sep="\t")
    lab = []
    for src, pref, col in (("pfam", "pfam", "name"), ("kofam", "ko", "accession"), ("txsscan", "txss", "name"),
                           ("defensefinder", "df", "name")):
        d = ann[ann.source == src]
        lab.append(pd.DataFrame({"protein_id": d.protein_id, "label": pref + ":" + d[col].astype(str)}))
    cur["st"] = cur.system_type.fillna(cur.system_source)
    lab.append(pd.DataFrame({"protein_id": cur.protein_id, "label": "sys:" + cur.st.astype(str)}))
    lab = pd.concat(lab).drop_duplicates()
    ko_desc = ann[ann.source == "kofam"].drop_duplicates("accession").set_index("accession").description
    return lab, set(ann.protein_id), ko_desc


def enrich(win, lab, annotated, tag):
    win = win.copy()
    win["wid"] = win.instance_id + "#" + win.ctrl.astype(str)
    L = [win[["wid", "kind", "instance_id", "protein_id", "offset"]].merge(lab, on="protein_id")]
    L.append(pd.DataFrame({"wid": win.wid, "kind": win.kind, "instance_id": win.instance_id,
                           "protein_id": win.protein_id, "offset": win.offset,
                           "label": "tok:" + win.token.astype(str)}))
    L.append(pd.DataFrame({"wid": win.wid, "kind": win.kind, "instance_id": win.instance_id,
                           "protein_id": win.protein_id, "offset": win.offset,
                           "label": "mod:" + win.module.astype(str)}))
    dk = win[~win.protein_id.isin(annotated)]
    L.append(pd.DataFrame({"wid": dk.wid, "kind": dk.kind, "instance_id": dk.instance_id,
                           "protein_id": dk.protein_id, "offset": dk.offset, "label": "dark"}))
    X = pd.concat(L)
    X = X[X.label != "mod:-1"]
    with_ctrl = set(win.instance_id[win.kind == "control"])
    real_all = win[win.kind == "real"].wid.nunique()
    real_wc = win[(win.kind == "real") & win.instance_id.isin(with_ctrl)].wid.nunique()
    ctrl_n = win[win.kind == "control"].wid.nunique()
    pres = X.drop_duplicates(["wid", "label"])
    ra = pres[pres.kind == "real"].groupby("label").size()
    rw = pres[(pres.kind == "real") & pres.instance_id.isin(with_ctrl)].groupby("label").size()
    cw = pres[pres.kind == "control"].groupby("label").size()
    T = pd.DataFrame({"real_all": ra, "real_wc": rw, "ctrl": cw}).fillna(0).astype(int)
    T["frac_real_all"] = (T.real_all / real_all).round(3)
    T["frac_real_wc"] = (T.real_wc / max(real_wc, 1)).round(3)
    T["frac_ctrl"] = (T.ctrl / max(ctrl_n, 1)).round(3)
    T["ratio"] = ((T.real_wc / max(real_wc, 1) + 1e-9) / (T.ctrl / max(ctrl_n, 1) + 1e-9)).round(2)
    T["p"] = [fisher_exact([[r, real_wc - r], [c, ctrl_n - c]], alternative="greater").pvalue
              for r, c in zip(T.real_wc, T.ctrl)]
    T["q"] = np.nan
    m = T.real_wc >= 3
    T.loc[m, "q"] = bh(T.p[m])
    # modal offset of the label among real windows
    off = X[X.kind == "real"].groupby("label").offset.agg(lambda s: s.value_counts().index[0])
    T["modal_offset"] = off
    T = T.sort_values(["q", "p"])
    T.to_csv(W / f"{tag}_context_enrichment.tsv", sep="\t")
    print(f"{tag}: real windows {real_all} (with controls {real_wc}); control windows {ctrl_n}")
    return T, X


def module_report(mod, lab, tag, win):
    g = pd.read_parquet(CW / "genes_dark.parquet")
    tm = pd.read_parquet(CW / "data/token_modules.parquet")
    toks = tm.token[tm.module == mod].tolist()
    d = g[g.token.isin(toks)]
    lab_by = lab.groupby("protein_id").label.agg(lambda s: sorted(set(s)))
    rows = []
    real = win[win.kind == "real"]
    for t, x in d.groupby("token"):
        labs = pd.Series([l for p in x.protein_id for l in lab_by.get(p, []) if not l.startswith("sys:")])
        sysl = pd.Series([l for p in x.protein_id for l in lab_by.get(p, []) if l.startswith("sys:")])
        inw = real[real.token == t]
        rows.append(dict(token=t, members=len(x), genomes=x.genome_code.nunique(), len_med=float(x.length.median()),
                         dark_frac=round(float(x.dark.mean()), 3),
                         top_labels=";".join(f"{k}:{v}" for k, v in labs.value_counts().head(4).items()),
                         curated=";".join(f"{k}:{v}" for k, v in sysl.value_counts().head(3).items()),
                         in_windows=inw.instance_id.nunique(),
                         offsets=";".join(f"{k}:{v}" for k, v in inw.offset.value_counts().head(3).items())))
    R = pd.DataFrame(rows).sort_values("members", ascending=False)
    R.to_csv(W / f"{tag}_module{mod}.tsv", sep="\t", index=False)
    return R


def main(tag):
    win = pd.read_csv(W / "ctx_windows.tsv", sep="\t")
    lab, annotated, ko_desc = labels()
    if tag == "c1":
        w = win[win.cand == "c1"]
        T, X = enrich(w, lab, annotated, "c1")
        R = module_report(1167, lab, "c1", w)
    else:
        w = win[win.cand.isin(["s34", "s33", "s7"])]
        T, X = enrich(w, lab, annotated, "c2")
        R = module_report(227, lab, "c2", w)
    pd.set_option("display.width", 250)
    show = T[(T.real_all >= 3)].head(40).copy()
    show["desc"] = [ko_desc.get(l[3:], "") if l.startswith("ko:") else "" for l in show.index]
    print(show.to_string())
    print(R.head(30).to_string())
    # per-offset composition of the immediate neighbours
    real = X[X.kind == "real"]
    for off in (-2, -1, 1, 2):
        s = real[(real.offset == off) & ~real.label.str.startswith(("tok:", "mod:"))].drop_duplicates(
            ["wid", "label"]).label.value_counts().head(8)
        print(off, s.to_dict())


if __name__ == "__main__":
    main(sys.argv[1])
