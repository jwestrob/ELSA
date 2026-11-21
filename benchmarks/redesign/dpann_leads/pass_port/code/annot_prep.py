#!/usr/bin/env python
"""pass_port: label tables for one collection from the whole-proteome annotation ($RD/annot_gpu/<collection>/), in the
form the pass reads (slot_rescue/data layout): annot.parquet (protein_id = gid string, pfam = Pfam accessions without
version, ko = KOs), ko_proxy.tsv (KO -> Pfams on > 50% of the KO's proteins, computed on this collection as on DPANN),
label_names.tsv (Pfam names from the annotation rows, KO names from the KEGG snapshot copy).

Input rows: every Parquet in the annotation directory; the database of a row is read from its accession (PFxxxxx or
Kxxxxx). Rows are used as delivered (the annotation agent's gathering / KO thresholds) unless --pfam-ga is given, which
keeps Pfam rows with bitscore >= the model's sequence GA read as text from Aksha's Pfam-A.hmm (no search tool).
--smoke-testset3 builds partial tables from the testset3 ribosomal audit (121 ribosomal Pfam models, Pfam architectures
of genes near ribosomal runs, selected KOs) to test plumbing only.

usage: annot_prep.py --collection NAME --out DIR [--src DIR] [--pfam-ga] [--smoke-testset3]
"""
import argparse
import glob
import gzip
import json
import os

import numpy as np
import pandas as pd

RD = "/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111"
PP = RD + "/dpann_leads/pass_port"


def pfam_ga():
    ga, acc = {}, None
    for line in open("/home/jwestrob/.config/Astra/PFAM/Pfam-A.hmm"):
        if line.startswith("ACC "):
            acc = line.split()[1].split(".")[0]
        elif line.startswith("GA ") and acc:
            ga[acc] = float(line.split()[1])
    return ga


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--src")
    ap.add_argument("--pfam-ga", action="store_true")
    ap.add_argument("--smoke-testset3", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    clans = pd.read_csv(PP + "/data/Pfam-A.clans.tsv", sep="\t", header=None, names=["acc", "clan", "clan_id", "name", "desc"])
    name2acc = dict(zip(clans.name, clans.acc))
    info = dict(collection=a.collection)
    if a.smoke_testset3:
        W = RD + "/ribo_audit/work6/testset3"
        rm = pd.read_parquet(W + "/ribosomal/marked.parquet")
        wp = pd.read_parquet(W + "/annot/window_pfam.parquet")
        pf = pd.concat([pd.DataFrame(dict(gid=rm.gid, name=rm.markers.str.split(";"))),
                        pd.DataFrame(dict(gid=wp.gid, name=wp.arch.str.split(",")))]).explode("name")
        pf["acc"] = pf.name.map(name2acc)
        info["pfam_names_unmapped"] = int(pf.acc.isna().sum())
        pf = pf.dropna(subset=["acc"]).drop_duplicates(["gid", "acc"])
        ga_ = pd.read_parquet(W + "/annot/genes_annot.parquet", columns=["gid", "ko"]).dropna()
        ex = pd.read_parquet(W + "/annot/extra_ko.parquet", columns=["gid", "ko"])
        ko = pd.concat([ga_, ex]).dropna().drop_duplicates()
        ko = ko[ko.ko.str.match(r"^K\d{5}$")]
        names = dict(zip(pf.acc, pf.name))
        info["source"] = "SMOKE partial: testset3 ribosomal audit (marked.parquet, window_pfam.parquet, genes_annot.parquet, extra_ko.parquet)"
    else:
        files = sorted(glob.glob(os.path.join(a.src, "*.parquet")))
        assert files, f"no parquet in {a.src}"
        rows = pd.concat([pd.read_parquet(f).assign(_file=os.path.basename(f)) for f in files], ignore_index=True)
        rows["acc0"] = rows.accession.astype(str).str.split(".").str[0]
        info["files"] = {os.path.basename(f): int((rows._file == os.path.basename(f)).sum()) for f in files}
        pfr = rows[rows.acc0.str.match(r"^PF\d{5}$")].copy()
        kor = rows[rows.acc0.str.match(r"^K\d{5}$")].copy()
        info["rows_other"] = int(len(rows) - len(pfr) - len(kor))
        if a.pfam_ga:
            ga = pfam_ga()
            keep = pfr.bitscore >= pfr.acc0.map(ga)
            info["pfam_ga_filter"] = dict(rows_in=int(len(pfr)), kept=int(keep.sum()), models_without_ga=int(pfr.acc0.map(ga).isna().sum()))
            pfr = pfr[keep]
        else:
            ga = pfam_ga()
            info["pfam_rows_below_ga_as_delivered"] = int((pfr.bitscore < pfr.acc0.map(ga)).sum())
        pf = pd.DataFrame(dict(gid=pfr.protein_id.astype(np.int64), acc=pfr.acc0, name=pfr.name)).drop_duplicates(["gid", "acc"])
        ko = pd.DataFrame(dict(gid=kor.protein_id.astype(np.int64), ko=kor.acc0)).drop_duplicates()
        names = dict(zip(pf.acc, pf.name))
        info["source"] = a.src
    P = pf.groupby("gid").acc.apply(lambda s: sorted(set(s)))
    K = ko.groupby("gid").ko.apply(lambda s: sorted(set(s)))
    ids = sorted(set(P.index) | set(K.index))
    ann = pd.DataFrame(dict(protein_id=[str(i) for i in ids], pfam=[P.get(i, []) for i in ids], ko=[K.get(i, []) for i in ids]))
    ann.to_parquet(a.out + "/annot.parquet", index=False)
    # KO -> Pfam proxy (> 50% of the KO's proteins), as computed on DPANN
    kp = ko.merge(pf[["gid", "acc"]], on="gid")
    nk = ko.groupby("ko").gid.nunique()
    c = kp.groupby(["ko", "acc"]).gid.nunique().reset_index(name="c")
    c["n"] = c.ko.map(nk)
    prox = c[c.c > 0.5 * c.n].rename(columns={"acc": "pf"})[["ko", "pf", "c", "n"]].sort_values(["ko", "pf"])
    prox.to_csv(a.out + "/ko_proxy.tsv", sep="\t", index=False)
    kn = pd.read_csv(PP + "/data/ko_names.tsv", sep="\t", header=None, names=["ko", "name"])
    ln = pd.concat([pd.DataFrame(dict(label=list(names), name=list(names.values()))),
                    kn[kn.ko.isin(set(ko.ko))].rename(columns={"ko": "label"})])
    ln.to_csv(a.out + "/label_names.tsv", sep="\t", index=False)
    info.update(genes_annotated=len(ann), genes_with_pfam=int((ann.pfam.str.len() > 0).sum()), genes_with_ko=int((ann.ko.str.len() > 0).sum()),
                pfam_labels=int(pf.acc.nunique()), kos=int(ko.ko.nunique()), ko_with_proxy=int(prox.ko.nunique()))
    json.dump(info, open(a.out + "/annot_prep.json", "w"), indent=1)
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
