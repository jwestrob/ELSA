#!/usr/bin/env python
"""Control for the embedding identity rule: re-embed sampled canonical sequences with the March model
(facebook/esm2_t6_8M_UR50D, CPU) and compare with the March h5 vector assigned to them.

Strata (seeded sample):
  id_and_sequence, length <= 1022      expect cosine >= 0.999 (rule correct, same pooling)
  id_and_sequence, length  > 1022      tests the truncation convention
  sequence_only (rescued by sequence)  expect cosine >= 0.999
  id_sequence_changed                  canonical sequence vs the h5 vector of the same ID; expect lower
Two poolings are computed: residue mean (Sharur 06_esm2_embeddings.py today) and mean over all tokens
including BOS/EOS.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import duckdb
import h5py
import numpy as np
import pandas as pd
import torch

H5 = "/groups/banfield/users/jwestrob/bin/Sharur/data/dpann/embeddings/protein_embeddings.h5"
MODEL = "facebook/esm2_t6_8M_UR50D"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkg", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--db", required=True)
    ap.add_argument("--threads", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL).eval()
    em = pd.read_parquet(a.pkg / "embeddings" / "embedding_match.parquet")
    rng = np.random.default_rng(17)
    plan = [("id_and_sequence", em.match_kind.eq("id_and_sequence") & (em.length <= 1022), 300),
            ("id_and_sequence_gt1022", em.match_kind.eq("id_and_sequence") & (em.length > 1022), 40),
            ("sequence_only", em.match_kind.eq("sequence_only"), 60),
            ("id_sequence_changed", em.match_kind.eq("id_sequence_changed"), 60)]
    picks = []
    for name, mask, k in plan:
        idx = np.flatnonzero(mask.to_numpy())
        if len(idx):
            for i in rng.choice(idx, size=min(k, len(idx)), replace=False):
                picks.append((name, int(em.gid.iloc[i])))
    pk = pd.DataFrame(picks, columns=["stratum", "gid"])
    con = duckdb.connect()
    con.execute(f"PRAGMA threads={a.threads}")
    con.execute(f"ATTACH '{a.db}' AS c (READ_ONLY)")
    con.register("pk", pk)
    seqs = con.execute(f"""SELECT pk.stratum, pk.gid, u.protein_id, rtrim(cp.sequence, '*') AS seq
                           FROM pk JOIN read_parquet('{a.work}/universe_full.parquet') u USING (gid)
                           JOIN c.proteins cp USING (protein_id)""").df()
    # h5 vector for each pick: assigned row, or for id_sequence_changed the row of the same protein id
    with h5py.File(H5, "r") as f:
        ids = np.array([x.decode() for x in f["protein_ids"][:]])
        rowof = {p: i for i, p in enumerate(ids)}
        X = f["embeddings"]
        h5row = []
        for r in seqs.itertuples():
            if r.stratum == "id_sequence_changed":
                h5row.append(rowof[r.protein_id])
            else:
                h5row.append(int(em.h5_row.iloc[r.gid]))
        order = np.argsort(h5row)
        vec = np.empty((len(h5row), X.shape[1]), np.float32)
        vec[order] = X[np.asarray(h5row)[order]]
    out = []
    with torch.no_grad():
        for r, v in zip(seqs.itertuples(), vec):
            enc = tok([r.seq], return_tensors="pt", truncation=True, max_length=1024, return_special_tokens_mask=True)
            sm = enc.pop("special_tokens_mask")[0].bool()
            h = model(**enc).last_hidden_state[0]
            res = h[~sm].mean(0).numpy()
            allt = h.mean(0).numpy()
            cos = lambda x: float(np.dot(x, v) / (np.linalg.norm(x) * np.linalg.norm(v)))
            out.append(dict(stratum=r.stratum, gid=r.gid, length=len(r.seq), cos_residue_mean=cos(res),
                            cos_all_token_mean=cos(allt),
                            rel_l2_residue=float(np.linalg.norm(res - v) / np.linalg.norm(v))))
    df = pd.DataFrame(out)
    a.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out / "reembed_control.tsv", sep="\t", index=False)
    summ = {}
    for s, g in df.groupby("stratum"):
        summ[s] = dict(n=len(g), cos_residue_min=float(g.cos_residue_mean.min()),
                       cos_residue_median=float(g.cos_residue_mean.median()),
                       frac_cos_residue_ge_0999=float((g.cos_residue_mean >= 0.999).mean()),
                       cos_alltoken_median=float(g.cos_all_token_mean.median()),
                       rel_l2_residue_median=float(g.rel_l2_residue.median()))
    (a.out / "reembed_control.json").write_text(json.dumps(dict(model=MODEL, summary=summ), indent=2) + "\n")
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
