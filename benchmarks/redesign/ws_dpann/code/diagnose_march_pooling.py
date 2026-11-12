#!/usr/bin/env python
"""Test whether the March h5 vectors are the Feb-Mar 2026 Sharur pooling applied to padded batches.

Sharur commit fc03c29 (src/ingest/06_esm2_embeddings.py, unchanged until 4592775 on 30 Mar 2026) did
    inputs = tokenizer(batch, padding=True, truncation=True, max_length=1024)
    vec = model(**inputs).last_hidden_state.mean(dim=1)
i.e. the mean over BOS, residues, EOS *and padding positions* of the batch, batch size 16 on MPS.
Hypothesis H: h5 rows are in processing order, so rows [16b, 16b+16) were one batch, and re-running
exactly that batch reproduces every vector of the batch (cosine >= 0.999).
Alternatives scored on the same proteins: batch sizes 8 and 32 with the same alignment, single-protein
all-token mean, single-protein residue mean.
"""
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
MARCH_DB = "/groups/banfield/users/jwestrob/bin/Sharur/data/dpann/sharur.duckdb"
MODEL = "facebook/esm2_t6_8M_UR50D"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--batches", type=int, default=12)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL).eval()
    with h5py.File(H5, "r") as f:
        ids = np.array([x.decode() for x in f["protein_ids"][:]])
        X = f["embeddings"]
        n = len(ids)
        rng = np.random.default_rng(17)
        starts32 = rng.choice(n // 32 - 1, size=a.batches, replace=False) * 32  # aligned to 8, 16 and 32
        rows = sorted({r for s in starts32 for r in range(s, s + 32)})
        V = {r: X[r] for r in rows}
    con = duckdb.connect()
    con.execute(f"PRAGMA threads={a.threads}")
    con.execute(f"ATTACH '{MARCH_DB}' AS m (READ_ONLY)")
    con.register("q", pd.DataFrame({"protein_id": ids[rows], "row": rows}))
    seq = dict(con.execute("SELECT q.row, mp.sequence FROM q JOIN m.proteins mp USING (protein_id)").fetchall())
    cos = lambda x, y: float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)))
    res = []
    with torch.no_grad():
        def batch_mean(rr):
            enc = tok([seq[r] for r in rr], return_tensors="pt", padding=True, truncation=True, max_length=1024)
            return model(**enc).last_hidden_state.mean(dim=1).numpy()
        for s in starts32:
            for bs in (8, 16, 32):
                for b0 in range(s, s + 32, bs):
                    rr = list(range(b0, b0 + bs))
                    E = batch_mean(rr)
                    for r, e in zip(rr, E):
                        res.append(dict(row=r, variant=f"padded_batch_{bs}", cos=cos(e, V[r]),
                                        length=len(seq[r]), batch_max=max(len(seq[x]) for x in rr)))
            for r in range(s, s + 16):
                enc = tok([seq[r]], return_tensors="pt", truncation=True, max_length=1024, return_special_tokens_mask=True)
                sm = enc.pop("special_tokens_mask")[0].bool()
                h = model(**enc).last_hidden_state[0]
                res.append(dict(row=r, variant="single_all_token_mean", cos=cos(h.mean(0).numpy(), V[r]), length=len(seq[r])))
                res.append(dict(row=r, variant="single_residue_mean", cos=cos(h[~sm].mean(0).numpy(), V[r]), length=len(seq[r])))
    df = pd.DataFrame(res)
    a.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out / "march_pooling_diagnosis.tsv", sep="\t", index=False)
    summ = {v: dict(n=int(len(g)), cos_min=float(g.cos.min()), cos_median=float(g.cos.median()),
                    frac_ge_0999=float((g.cos >= 0.999).mean()))
            for v, g in df.groupby("variant")}
    out = dict(hypothesis="h5 row order = processing order; padded batch of 16; mean over all positions incl. pads",
               batches_sampled_32_aligned=[int(x) for x in starts32], summary=summ)
    (a.out / "march_pooling_diagnosis.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
