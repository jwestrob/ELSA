#!/usr/bin/env python
"""End-to-end synthetic run of bench.py stage1 + stage2 (small; laptop, 2 processes).

Builds a package with 80 genomes x 3 contigs x 80 genes, random family tokens and family-structured
vectors, and gold loci of three callers: DefenseFinder types A (3 genes) and B (2 genes), TXSScan one
type (4 genes), CRISPRCasTyper subtypes (2 genes). A third of type-A copies carry a token-invisible
middle gene that stays embedding-close. Usage: selftest_bench.py <empty dir>
"""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def build(root, seed=5, n_genomes=80, contigs=3, genes_per=80, D=32):
    rng = np.random.default_rng(seed)
    rows, gid = [], 0
    for g in range(n_genomes):
        for c in range(contigs):
            for p in range(genes_per):
                rows.append((gid, f"G{g}", g, f"G{g}_c{c}", g * contigs + c, p))
                gid += 1
    genes = pd.DataFrame(rows, columns=["gid", "genome", "genome_code", "contig", "contig_code", "position_index"])
    N = len(genes)
    n_fam = 600
    tok = rng.integers(0, n_fam, N)
    loci = {"A": [n_fam, n_fam + 1, n_fam + 2], "B": [n_fam + 3, n_fam + 4], "T": [n_fam + 5, n_fam + 6, n_fam + 7, n_fam + 8],
            "C1": [n_fam + 9, n_fam + 10], "C2": [n_fam + 11, n_fam + 12]}
    fam_vec = rng.normal(size=(n_fam + 13, D))
    inst = []
    inv = []
    for g in range(n_genomes):
        base = g * contigs * genes_per
        layout = [("DefenseFinder", "A", 5), ("DefenseFinder", "B", 20), ("TXSScan", "T", 35),
                  ("CRISPRCasTyper", "C1" if g % 3 else "C2", genes_per + 10)]
        for caller, ty, off in layout:
            if rng.random() < 0.25:
                continue
            s = base + off + int(rng.integers(0, 8))
            fam = loci[ty]
            tok[s:s + len(fam)] = fam
            inst.append(dict(caller=caller, label=ty, members=list(range(s, s + len(fam))), genome=f"G{g}"))
            if ty == "A" and g % 3 == 0:
                inv.append(s + 1)
    vec = fam_vec[tok] + 0.6 * rng.normal(size=(N, D))
    for i, x in enumerate(inv):
        tok[x] = n_fam + 13 + i
    X = (vec / np.linalg.norm(vec, axis=1, keepdims=True)).astype(np.float32)
    (root / "universe").mkdir(parents=True)
    (root / "gold").mkdir()
    genes.to_parquet(root / "universe/genes.parquet")
    cont = genes.groupby(["genome", "contig", "contig_code"]).agg(n_genes=("gid", "size"), first_gid=("gid", "min")).reset_index()
    cont.to_parquet(root / "universe/contigs.parquet")
    pd.DataFrame(dict(genome=[f"G{g}" for g in range(n_genomes)], genome_code=range(n_genomes))).to_parquet(
        root / "universe/genomes.parquet")
    I = pd.DataFrame(inst)
    I["instance_id"] = np.arange(len(I))
    I["system_type"] = np.where(I.caller == "CRISPRCasTyper", "False", I.label)
    I["system_subtype"] = I.label
    I["member_gids"] = I.members
    I["n_members_resolved"] = I.members.map(len)
    I.drop(columns=["members", "label"]).to_parquet(root / "gold/instances.parquet")
    pd.DataFrame(dict(gid=np.arange(N), token=tok)).to_parquet(root / "tokens.parquet")
    np.save(root / "emb.npy", X)


if __name__ == "__main__":
    root = Path(sys.argv[1])
    build(root / "pkg")
    env = dict(os.environ, LS_PKG=str(root / "pkg"), LS_TOKENS=str(root / "pkg/tokens.parquet"),
               LS_EMB=str(root / "pkg/emb.npy"), LS_K="300", LS_R="50", OMP_NUM_THREADS="2", NUMBA_NUM_THREADS="2")
    here = Path(__file__).resolve().parent
    for st in ("stage1", "stage2"):
        subprocess.run([sys.executable, "-B", str(here / "bench.py"), st, "--out", str(root / "out"), "--threads", "2"],
                       env=env, check=True)
