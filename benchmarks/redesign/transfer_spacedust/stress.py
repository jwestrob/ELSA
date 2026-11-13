#!/usr/bin/env python
"""Duplicate-genome stress test (amendment_0, secondary, label-free; workstream transfer_spacedust).

Clade: GTDB family chosen by the predeclared rule (no genus has 20-50 reference genomes; families with 26 tie, tie-break =
more genes in total, NOTEBOOK 08:3x). k = 0, 1, 3, 9 exact copies of every clade genome are appended (new genome and
contig codes, rescue3 tokens copied). The family word procedure is rerun under three counting rules with tau fixed at
each rule's k = 0 tau*:
  genome  every genome its own unit
  group   average linkage at 0.75 on emb_hom augmented with the copies (copy rows / columns duplicated, copy-original
          and copy-copy similarity 1), mean-symmetrised
  syn     synteny-retention N_eff with S recomputed on the augmented corpus
Reported per run, for word types involving the clade (an instance in a clade genome or a copy) and for the rest:
word types, share of genes in top-level words (clade originals; other genomes), Jaccard overlap with k = 0 of the type sets
(type = canonical composition: unordered merge tree over slot tokens).

usage: stress.py --taus genome:T,group:T,syn:T --threads P
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import emb as EM  # noqa: E402
import words as W  # noqa: E402

TS = W.TS
KS = (0, 1, 3, 9)


def pick_clade(gm):
    r = gm[gm.side == "ref"]
    vc = r.genus.value_counts()
    c = vc[(vc >= 20) & (vc <= 50)]
    if len(c):
        top = c[c == c.max()].index
        rank, col = "genus", "genus"
    else:
        fc = r.family.value_counts()
        c = fc[(fc >= 20) & (fc <= 50)]
        top = c[c == c.max()].index
        rank, col = "family", "family"
    genes = {t: int(r[r[col] == t].n_genes.sum()) for t in top}
    name = max(genes, key=lambda t: (genes[t], t))
    codes = np.sort(r[r[col] == name].genome_code.to_numpy())
    return rank, name, codes, dict(candidates={k: int(c[k]) for k in c.index}, tie_genes=genes)


def augment(tok, cc, gc, clade, k):
    if k == 0:
        return tok, cc, gc, np.zeros(0, np.int64)
    ng = int(gc.max()) + 1
    ncc = int(cc.max()) + 1
    ptok, pcc, pgc, src = [tok], [cc], [gc], []
    j = 0
    for rep in range(k):
        for G in clade:
            m = np.flatnonzero(gc == G)
            ptok.append(tok[m])
            u, inv = np.unique(cc[m], return_inverse=True)
            pcc.append(ncc + inv)
            ncc += len(u)
            pgc.append(np.full(len(m), ng + j, np.int64))
            src.append(G)
            j += 1
    return np.concatenate(ptok), np.concatenate(pcc), np.concatenate(pgc), np.asarray(src, np.int64)


def groups_aug(H, src):
    """emb_hom with copies: copy rows / columns = the original's; copy-original and copy-copy = 1."""
    ng = len(H)
    idx = np.r_[np.arange(ng), src]
    A = H[np.ix_(idx, idx)].astype(np.float64)
    orig = np.r_[np.arange(ng), src]
    same = orig[:, None] == orig[None, :]
    A[same] = 1.0
    Z = EM.symmetrise(A)
    return EM.groups_at(EM.linkage_of(Z), 0.75)


def comp_ids(words, T, table):
    """canonical composition id per word symbol (shared table across runs)."""
    w = words.sort_values("sym")
    cid = {}

    def c(x):
        return ("t", int(x)) if x < T else cid[int(x)]
    for s, a, b in zip(w.sym.to_numpy(), w.a.to_numpy(), w.b.to_numpy()):
        ca, cb = c(a), c(b)
        key = (min(ca, cb, key=str), max(ca, cb, key=str))
        cid[int(s)] = table.setdefault(key, ("w", len(table)))
    return cid


def measure(res, gc_aug, clade_mask_genome, n_orig_genomes, T, table):
    cid = comp_ids(res["words"], T, table)
    I = res["inst"]
    ext = np.ones(int(gc_aug.max()) + 1, bool)              # copies (codes >= n_orig_genomes) are clade genomes
    ext[:n_orig_genomes] = clade_mask_genome
    in_clade = ext[gc_aug[I["s"]]]
    inv = {}
    for s_, f in zip(I["sym"], in_clade):
        inv[int(s_)] = inv.get(int(s_), False) or bool(f)
    keys = {cid[s]: inv.get(s, False) for s in cid}
    live = res["live"]
    top = ~live["term"]
    n = len(gc_aug)
    cov = np.zeros(n + 1, np.int32)
    np.add.at(cov, live["ns"][top], 1)
    np.add.at(cov, live["ne"][top] + 1, -1)
    inw = np.cumsum(cov)[:n] > 0
    g_orig_clade = clade_mask_genome[np.minimum(gc_aug, n_orig_genomes - 1)] & (gc_aug < n_orig_genomes)
    g_rest = (~clade_mask_genome[np.minimum(gc_aug, n_orig_genomes - 1)]) & (gc_aug < n_orig_genomes)
    return keys, dict(word_types=int(len(cid)), clade_types=int(sum(keys.values())),
                      rest_types=int(len(keys) - sum(keys.values())),
                      genes_in_words_clade_originals=float(inw[g_orig_clade].mean()),
                      genes_in_words_rest=float(inw[g_rest].mean()), seconds=res["seconds"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--taus", required=True)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    import numba
    numba.set_num_threads(a.threads)
    t0 = time.time()
    taus = {k: float(v) for k, v in (x.split(":") for x in a.taus.split(","))}
    gm = pd.read_parquet(W.UNIV / "genomes.parquet").sort_values("genome_code").reset_index(drop=True)
    rank, name, clade, cinfo = pick_clade(gm)
    tok, cc, gc = W.load_base()
    T = int(tok.max()) + 1
    ng = int(gc.max()) + 1
    cmask = np.zeros(ng, bool)
    cmask[clade] = True
    H = np.load(TS / "groups/emb_hom.npy")
    table = {}
    out = dict(clade=dict(rank=rank, name=name, genomes=int(len(clade)), genes=int(np.isin(gc, clade).sum()), **cinfo),
               taus=taus, runs={})
    base_keys = {}
    for k in KS:
        tk, ck, gk, src = augment(tok, cc, gc, clade, k)
        for unit in ("genome", "group", "syn"):
            t1 = time.time()
            grp = groups_aug(H, src) if unit == "group" else None
            S = W.synteny_matrix(tk, ck, gk) if unit == "syn" else None
            res = W.run_words(tk, ck, gk, grp, unit, taus[unit], S=S)
            keys, m = measure(res, gk, cmask, ng, T, table)
            if unit == "group":
                m["groups"] = int(grp.max()) + 1
                m["clade_groups"] = int(len(np.unique(grp[clade])))
            if k == 0:
                base_keys[unit] = keys
            else:
                b = base_keys[unit]
                anyc = lambda x: keys.get(x, False) or b.get(x, False)   # noqa: E731  involved in either run
                for part, flag in (("clade", True), ("rest", False)):
                    A = {x for x in keys if anyc(x) == flag}
                    Bs = {x for x in b if anyc(x) == flag}
                    m[f"jaccard_{part}_vs_k0"] = len(A & Bs) / max(len(A | Bs), 1)
                m["clade_types_vs_k0"] = m["clade_types"] / max(sum(b.values()), 1)
                m["rest_types_vs_k0"] = m["rest_types"] / max(len(b) - sum(b.values()), 1)
            m["seconds_total"] = round(time.time() - t1, 1)
            out["runs"][f"{unit}_k{k}"] = m
            W.log("stress", unit, k, m)
            (TS / "results/stress.json").write_text(json.dumps(out, indent=1) + "\n")
    out["seconds"] = round(time.time() - t0, 1)
    (TS / "results/stress.json").write_text(json.dumps(out, indent=1) + "\n")
    W.log("stress done", f"{time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
