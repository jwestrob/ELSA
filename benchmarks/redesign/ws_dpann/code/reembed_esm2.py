#!/usr/bin/env python
"""Re-embed DPANN proteins with ESM2 on CPU using the bacterial-redesign pooling convention.

Convention (same as $SCALE/code/embed_spacedust_scaling.py for ESM2): final hidden state, arithmetic
mean over residue positions only (BOS, EOS and padding excluded), float32, then L2 normalization;
sequences longer than 1,022 residues are skipped (NaN row). Batches are length-sorted under a token
budget so padding is minimal and never enters the mean.

Modes:
  --sample N      embed a seeded random sample of N proteins and report throughput (timing benchmark),
                  and check batch invariance: the same proteins embedded one at a time.
  --shard i/n     embed every gid with gid % n == i; write <out>/shard_i_of_n.npy (+ gids).
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

MODEL = "facebook/esm2_t6_8M_UR50D"
MAXRES = 1022


def read_fasta(path, keep=None):
    gids, seqs = [], []
    with open(path) as f:
        g = None
        for line in f:
            if line[0] == ">":
                g = int(line[1:])
            else:
                if keep is None or keep(g):
                    gids.append(g)
                    seqs.append(line.strip())
    return np.asarray(gids, dtype=np.int64), seqs


def embed(model, tok, seqs, token_budget=16384):
    out = np.full((len(seqs), model.config.hidden_size), np.nan, dtype=np.float32)
    idx = [i for i, s in enumerate(seqs) if len(s) <= MAXRES]
    idx.sort(key=lambda i: len(seqs[i]))
    b = 0
    with torch.inference_mode():
        while b < len(idx):
            L = len(seqs[idx[b]]) + 2
            e = b
            while e < len(idx) and (e - b + 1) * (len(seqs[idx[e]]) + 2) <= max(token_budget, L):
                e += 1
            batch = idx[b:e]
            enc = tok([seqs[i] for i in batch], return_tensors="pt", padding=True, return_special_tokens_mask=True)
            special = enc.pop("special_tokens_mask").bool()
            keep = enc["attention_mask"].bool() & ~special
            h = model(**enc).last_hidden_state
            w = keep.unsqueeze(-1).to(h.dtype)
            v = (h * w).sum(1) / w.sum(1)
            v = torch.nn.functional.normalize(v.float(), dim=1)
            out[batch] = v.numpy()
            b = e
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--threads", type=int, required=True)
    ap.add_argument("--sample", type=int)
    ap.add_argument("--shard")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL, add_pooling_layer=False).eval()
    a.out.mkdir(parents=True, exist_ok=True)
    if a.sample:
        gids, seqs = read_fasta(a.fasta)
        n_total, aa_total = len(seqs), int(sum(len(s) for s in seqs if len(s) <= MAXRES))
        pick = np.sort(np.random.default_rng(17).choice(len(seqs), size=a.sample, replace=False))
        sg, ss = gids[pick], [seqs[i] for i in pick]
        t0 = time.time()
        V = embed(model, tok, ss)
        dt = time.time() - t0
        aa = int(sum(len(s) for s in ss if len(s) <= MAXRES))
        # batch invariance on 50 of them
        sub = [i for i in range(len(ss)) if len(ss[i]) <= MAXRES][:50]
        single = np.vstack([embed(model, tok, [ss[i]]) for i in sub])
        cos = (single * V[sub]).sum(1)
        res = dict(model=MODEL, threads=a.threads, sample=a.sample, seconds=dt, residues=aa,
                   residues_per_second=aa / dt, proteins_per_second=a.sample / dt,
                   proteins_total=n_total, residues_total_le1022=aa_total,
                   projected_cpu_hours_total=aa_total / (aa / dt) * a.threads / 3600,
                   projected_wall_hours_on_this_allocation=aa_total / (aa / dt) / 3600,
                   skipped_gt1022_in_sample=int(np.isnan(V[:, 0]).sum()),
                   batch_invariance_min_cosine=float(cos.min()))
        (a.out / "timing_benchmark.json").write_text(json.dumps(res, indent=2) + "\n")
        print(json.dumps(res, indent=2))
        return
    i, n = map(int, a.shard.split("/"))
    gids, seqs = read_fasta(a.fasta, keep=lambda g: g % n == i)
    t0 = time.time()
    V = embed(model, tok, seqs)
    np.save(a.out / f".shard_{i}_of_{n}.tmp.npy", V)
    os.replace(a.out / f".shard_{i}_of_{n}.tmp.npy", a.out / f"shard_{i}_of_{n}.npy")
    np.save(a.out / f"shard_{i}_of_{n}.gids.npy", gids)
    print(json.dumps(dict(shard=a.shard, proteins=len(seqs), seconds=time.time() - t0)))


if __name__ == "__main__":
    main()
