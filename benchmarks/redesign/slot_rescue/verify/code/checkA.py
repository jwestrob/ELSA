#!/usr/bin/env python
"""slot_rescue verify, check A (predeclared_verify.json): weak profile hits of the proposed labels.

  checkA.py models  --out DIR             label models (Pfam-A.hmm by accession, KOFAM <KO>.hmm) -> DIR/labels.hmm,
                                          DIR/model_map.tsv, DIR/missing.tsv (text copy; no search tool)
  checkA.py subset  --n N --out FAA       first N labeled proteins (smoke)
  checkA.py tables  --hits TSV --out DIR [--n N]   per-protein measures, null, strata tables
The search itself is `aksha search` in checkA.sbatch.
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd

VD = "/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/slot_rescue/verify"
PASS = "/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/slot_rescue/pass/dpann_slot_rescue_pass.parquet"
PFAM = "/home/jwestrob/.config/Astra/PFAM/Pfam-A.hmm"
KOFAM = "/home/jwestrob/.config/Astra/KOFAM"
SEED = 20261121


def cmd_models(a):
    d = pd.read_parquet(PASS, columns=["proposed_label_positional_homology", "proposed_label_kind"])
    labs = d.drop_duplicates("proposed_label_positional_homology")
    pf = set(labs.proposed_label_positional_homology[labs.proposed_label_kind == "pfam"])
    ko = sorted(labs.proposed_label_positional_homology[labs.proposed_label_kind == "ko"])
    os.makedirs(a.out, exist_ok=True)
    out = open(f"{a.out}/labels.hmm", "w")
    rows, found = [], set()
    rec, name, acc = [], None, None
    for line in open(PFAM):
        rec.append(line)
        if line.startswith("NAME "):
            name = line.split()[1]
        elif line.startswith("ACC "):
            acc = line.split()[1].split(".")[0]
        elif line.startswith("//"):
            if acc in pf:
                out.writelines(rec)
                rows.append(dict(model=name, label=acc, source="PFAM"))
                found.add(acc)
            rec, name, acc = [], None, None
    for k in ko:
        p = f"{KOFAM}/{k}.hmm"
        if os.path.exists(p):
            txt = open(p).read()
            nm = next(l.split()[1] for l in txt.splitlines() if l.startswith("NAME "))
            out.write(txt if txt.endswith("\n") else txt + "\n")
            rows.append(dict(model=nm, label=k, source="KOFAM"))
            found.add(k)
    out.close()
    M = pd.DataFrame(rows)
    M.to_csv(f"{a.out}/model_map.tsv", sep="\t", index=False)
    miss = sorted((pf | set(ko)) - found)
    pd.DataFrame(dict(label=miss)).to_csv(f"{a.out}/missing.tsv", sep="\t", index=False)
    assert M.model.is_unique
    print(json.dumps(dict(labels=len(pf) + len(ko), models=len(M), missing=len(miss), pfam=int((M.source == "PFAM").sum()),
                          kofam=int((M.source == "KOFAM").sum()))))


def cmd_subset(a):
    n, k = 0, 0
    with open(f"{VD}/data/labeled.faa") as f, open(a.out, "w") as o:
        for line in f:
            if line.startswith(">"):
                n += 1
                if n > a.n:
                    break
            o.write(line)
    print("subset", min(n, a.n))


def rate(x):
    return round(float(np.mean(x)), 4) if len(x) else None


def cmd_tables(a):
    t0 = time.time()
    P = pd.read_parquet(PASS, columns=["protein_id", "proposed_label_positional_homology", "proposed_label_kind", "length",
                                       "confidence_bin", "vote_share", "votes_for", "own_ko_agrees"])
    if a.n:
        P = P.head(a.n).copy()
    P = P.rename(columns={"proposed_label_positional_homology": "label"})
    M = pd.read_csv(f"{a.models}/model_map.tsv", sep="\t")
    has_model = set(M.label)
    m2l = dict(zip(M.model, M.label))
    H = pd.read_csv(a.hits, sep="\t", usecols=["sequence_id", "hmm_name", "bitscore", "evalue"])
    n_rows = len(H)
    H["label"] = H.hmm_name.map(m2l)
    assert H.label.notna().all(), "hit from an unmapped model"
    H = H.sort_values(["sequence_id", "label", "bitscore"], ascending=[True, True, False]).drop_duplicates(["sequence_id", "label"])
    H = H[H.evalue <= 10.0]
    H = H.sort_values(["sequence_id", "bitscore", "evalue", "label"], ascending=[True, False, True, True])
    H["rank"] = H.groupby("sequence_id").cumcount() + 1
    key = H.set_index(["sequence_id", "label"])[["rank", "evalue", "bitscore"]]
    nhit = H.groupby("sequence_id").size()
    # null label: same length decile, different proposed label
    L = P.length.to_numpy()
    edges = np.quantile(L, np.arange(1, 10) / 10)
    dec = np.searchsorted(edges, L, side="right")
    lab = P.label.to_numpy()
    rng = np.random.default_rng(SEED)
    members = {k: np.flatnonzero(dec == k) for k in range(10)}
    null = np.empty(len(P), dtype=object)
    for i in range(len(P)):
        mem = members[dec[i]]
        while True:
            j = mem[rng.integers(len(mem))]
            if lab[j] != lab[i]:
                null[i] = lab[j]
                break
    P["null_label"] = null
    P["length_decile"] = dec

    def lookup(pids, labels):
        idx = pd.MultiIndex.from_arrays([pids, labels])
        return key.reindex(idx)
    pr = lookup(P.protein_id.to_numpy(), P.label.to_numpy())
    nl = lookup(P.protein_id.to_numpy(), P.null_label.to_numpy())
    P["has_model"] = P.label.isin(has_model)
    P["n_label_hits"] = P.protein_id.map(nhit).fillna(0).astype(int).to_numpy()
    P["proposed_rank"] = pr["rank"].to_numpy()
    P["proposed_evalue"] = pr["evalue"].to_numpy()
    P["proposed_bitscore"] = pr["bitscore"].to_numpy()
    P["null_rank"] = nl["rank"].to_numpy()
    P["null_evalue"] = nl["evalue"].to_numpy()
    P["null_has_model"] = P.null_label.isin(has_model)
    # best other label
    H2 = H.merge(P[["protein_id", "label"]].rename(columns={"protein_id": "sequence_id", "label": "plabel"}), on="sequence_id")
    bo = H2[H2.label != H2.plabel].drop_duplicates("sequence_id").set_index("sequence_id")
    P["best_other_label"] = P.protein_id.map(bo.label)
    P["best_other_evalue"] = P.protein_id.map(bo.evalue)
    P["best_other_bitscore"] = P.protein_id.map(bo.bitscore)
    P["proposed_top"] = P.proposed_rank == 1
    P["proposed_any"] = P.proposed_rank.notna()
    P["proposed_any_e3"] = P.proposed_evalue <= 1e-3
    P["proposed_e5"] = P.proposed_evalue <= 1e-5
    P["null_top"] = P.null_rank == 1
    P["null_any"] = P.null_rank.notna()
    P["null_any_e3"] = P.null_evalue <= 1e-3
    P["no_hit_any"] = P.n_label_hits == 0
    P["share_gt_0.8"] = P.vote_share > 0.8
    os.makedirs(a.out, exist_ok=True)
    P.to_parquet(f"{a.out}/checkA_per_protein.parquet", index=False)

    def block(df, name, val):
        return dict(stratum=name, value=val, n=len(df), with_model=int(df.has_model.sum()),
                    proposed_top=rate(df.proposed_top), proposed_any=rate(df.proposed_any), proposed_any_e3=rate(df.proposed_any_e3),
                    null_top=rate(df.null_top), null_any=rate(df.null_any), null_any_e3=rate(df.null_any_e3),
                    no_hit_any=rate(df.no_hit_any), proposed_e5=int(df.proposed_e5.sum()),
                    median_proposed_evalue=float(df.proposed_evalue.median()) if df.proposed_any.any() else None)
    rows = [block(P, "overall", "all")]
    for v, g in P.groupby("confidence_bin"):
        rows.append(block(g, "confidence_bin", v))
    for v, g in P.groupby("share_gt_0.8"):
        rows.append(block(g, "vote_share", "> 0.8" if v else "<= 0.8"))
    for v, g in P.groupby("proposed_label_kind"):
        rows.append(block(g, "label_kind", v))
    for v, g in P.groupby("own_ko_agrees"):
        rows.append(block(g, "own_ko_agrees", v))
    top = P.label.value_counts().head(30).index
    for v in top:
        rows.append(block(P[P.label == v], "top30_label", v))
    S = pd.DataFrame(rows)
    S.to_csv(f"{a.out}/checkA_strata.tsv", sep="\t", index=False)
    run = dict(hit_rows=n_rows, label_pairs_e10=int(len(H)), proteins=len(P), seconds=round(time.time() - t0, 1),
               length_decile_edges=edges.tolist(), null_equal_label=int((P.null_label == P.label).sum()))
    json.dump(run, open(f"{a.out}/checkA_run.json", "w"), indent=1)
    print(S.head(20).to_string())
    print(json.dumps(run))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["models", "subset", "tables"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--hits")
    ap.add_argument("--models", default=f"{VD}/A/models")
    a = ap.parse_args()
    {"models": cmd_models, "subset": cmd_subset, "tables": cmd_tables}[a.cmd](a)


if __name__ == "__main__":
    main()
