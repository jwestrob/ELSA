#!/usr/bin/env python
"""pass_port weak-hit check (slot_rescue check A design) for one collection's labeled genes.

  weakhit.py models  --collection C     label models of C's pass table: Pfam by accession from Aksha's Pfam-A.hmm, KOs from
                                        Aksha's KOFAM store (text copy into OUT/weakhit/models/labels.hmm; no search tool)
  weakhit.py fasta   --collection C     sequences of C's labeled genes from the collection FASTA (header = gid), for
                                        aksha search (the sequences stay on biotite)
  weakhit.py tables  --collection C     per protein: proposed label rank among label-model hits (full-sequence bit score),
                                        E-value, score, hit_labels (all label models hit at E <= 10, best first), best
                                        other label, null label (same length decile, different label, default_rng(20261121))
The search between `fasta` and `tables` is `aksha search --hmm_in labels.hmm --prot_in labeled.faa --evalue 10 --domE 10
--threads $SLURM_CPUS_ON_NODE` in weakhit_sbatch.txt (no --cut_* flag).
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

RD = "/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111"
PFAM = "/home/jwestrob/.config/Astra/PFAM/Pfam-A.hmm"
KOFAM = "/home/jwestrob/.config/Astra/KOFAM"


def P_(p):
    return p.replace("RD/", RD + "/") if isinstance(p, str) else p


def paths(c):
    cfg = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "collections.json")))[c]
    out = P_(cfg["out"])
    return cfg, out, f"{out}/{c}_slot_rescue_pass.parquet", f"{out}/weakhit"


def cmd_models(c):
    cfg, out, passf, W = paths(c)
    d = pd.read_parquet(passf, columns=["proposed_label_positional_homology", "proposed_label_kind"]).drop_duplicates()
    pf = set(d.proposed_label_positional_homology[d.proposed_label_kind == "pfam"])
    ko = sorted(d.proposed_label_positional_homology[d.proposed_label_kind == "ko"])
    os.makedirs(W + "/models", exist_ok=True)
    o = open(W + "/models/labels.hmm", "w")
    rows, rec, name, acc = [], [], None, None
    for line in open(PFAM):
        rec.append(line)
        if line.startswith("NAME "):
            name = line.split()[1]
        elif line.startswith("ACC "):
            acc = line.split()[1].split(".")[0]
        elif line.startswith("//"):
            if acc in pf:
                o.writelines(rec)
                rows.append(dict(model=name, label=acc))
            rec, name, acc = [], None, None
    for k in ko:
        p = f"{KOFAM}/{k}.hmm"
        if os.path.exists(p):
            t = open(p).read()
            o.write(t if t.endswith("\n") else t + "\n")
            rows.append(dict(model=next(l.split()[1] for l in t.splitlines() if l.startswith("NAME ")), label=k))
    o.close()
    M = pd.DataFrame(rows)
    M.to_csv(W + "/models/model_map.tsv", sep="\t", index=False)
    miss = sorted((pf | set(ko)) - set(M.label))
    pd.DataFrame(dict(label=miss)).to_csv(W + "/models/missing.tsv", sep="\t", index=False)
    print(json.dumps(dict(labels=len(pf) + len(ko), models=len(M), missing=len(miss))))


def cmd_fasta(c):
    cfg, out, passf, W = paths(c)
    gids = set(pd.read_parquet(passf, columns=["gid"]).gid.astype(int).tolist())
    os.makedirs(W, exist_ok=True)
    n, keep = 0, False
    with open(P_(cfg["fasta"])) as f, open(W + "/labeled.faa", "w") as o:
        for line in f:
            if line.startswith(">"):
                keep = int(line[1:].split()[0]) in gids
                n += keep
            if keep:
                o.write(line if not line.startswith(">") else line)
    print(json.dumps(dict(labeled=len(gids), written=n)))
    assert n == len(gids)


def cmd_tables(c):
    cfg, out, passf, W = paths(c)
    P = pd.read_parquet(passf, columns=["gid", "protein_id", "proposed_label_positional_homology", "proposed_label_kind", "length",
                                        "vote_share", "votes_for", "supporting_genomes"])
    P = P.rename(columns={"proposed_label_positional_homology": "label"})
    P["seq_id"] = P.gid.astype(str)
    M = pd.read_csv(W + "/models/model_map.tsv", sep="\t")
    m2l = dict(zip(M.model, M.label))
    H = pd.read_csv(W + "/aksha/user_hmms_hits_df.tsv", sep="\t", usecols=["sequence_id", "hmm_name", "bitscore", "evalue"])
    H["sequence_id"] = H.sequence_id.astype(str)
    H["label"] = H.hmm_name.map(m2l)
    H = H.sort_values(["sequence_id", "label", "bitscore"], ascending=[True, True, False]).drop_duplicates(["sequence_id", "label"])
    H = H[H.evalue <= 10.0].sort_values(["sequence_id", "bitscore", "evalue", "label"], ascending=[True, False, True, True])
    H["rank"] = H.groupby("sequence_id").cumcount() + 1
    key = H.set_index(["sequence_id", "label"])[["rank", "evalue", "bitscore"]]
    hl = H.groupby("sequence_id").label.apply(lambda s: ";".join(s))
    L = P.length.to_numpy()
    dec = np.searchsorted(np.quantile(L, np.arange(1, 10) / 10), L, side="right")
    lab = P.label.to_numpy()
    rng = np.random.default_rng(20261121)
    mem = {k: np.flatnonzero(dec == k) for k in range(10)}
    null = []
    for i in range(len(P)):
        while True:
            j = mem[dec[i]][rng.integers(len(mem[dec[i]]))]
            if lab[j] != lab[i] or len(set(lab[mem[dec[i]]])) == 1:
                null.append(lab[j])
                break
    P["null_label"] = null
    pr = key.reindex(pd.MultiIndex.from_arrays([P.seq_id, P.label]))
    nl = key.reindex(pd.MultiIndex.from_arrays([P.seq_id, P.null_label]))
    P["has_model"] = P.label.isin(set(M.label))
    P["hit_labels"] = P.seq_id.map(hl).fillna("")
    P["n_label_hits"] = P.hit_labels.str.count(";") + (P.hit_labels != "")
    P["proposed_rank"], P["proposed_evalue"], P["proposed_bitscore"] = pr["rank"].to_numpy(), pr["evalue"].to_numpy(), pr["bitscore"].to_numpy()
    P["null_rank"], P["null_evalue"] = nl["rank"].to_numpy(), nl["evalue"].to_numpy()
    P["proposed_any"] = P.proposed_rank.notna()
    P["proposed_top"] = P.proposed_rank == 1
    P["null_any"] = P.null_rank.notna()
    P["no_hit_any"] = P.n_label_hits == 0
    P.drop(columns=["seq_id"]).to_parquet(W + "/weakhit_per_protein.parquet", index=False)
    s = dict(proteins=len(P), with_model=int(P.has_model.sum()), proposed_any=round(float(P.proposed_any.mean()), 4),
             proposed_top=round(float(P.proposed_top.mean()), 4), null_any=round(float(P.null_any.mean()), 4),
             no_hit_any=round(float(P.no_hit_any.mean()), 4))
    json.dump(s, open(W + "/weakhit_summary.json", "w"), indent=1)
    print(json.dumps(s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["models", "fasta", "tables"])
    ap.add_argument("--collection", required=True)
    a = ap.parse_args()
    {"models": cmd_models, "fasta": cmd_fasta, "tables": cmd_tables}[a.cmd](a.collection)


if __name__ == "__main__":
    main()
