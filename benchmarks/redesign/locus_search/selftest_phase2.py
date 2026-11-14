#!/usr/bin/env python
"""Synthetic test of phase2.py scan, on the package written by selftest_bench.py (run that first).

Adds the scorer's gold columns, a fake odd-genome engine output (modules = the planted loci's
tokens, loci = gold spans on odd genomes) and a fake full-engine score table, then runs the scan.
Usage: selftest_phase2.py <dir used by selftest_bench.py>
"""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

root = Path(sys.argv[1])
pkg = root / "pkg"
here = Path(__file__).resolve().parent
genes = pd.read_parquet(pkg / "universe/genes.parquet")
I = pd.read_parquet(pkg / "gold/instances.parquet")
pos = genes.position_index.to_numpy()
I["resolved"] = True
I["order_trusted"] = True
I["contig_code"] = [int(genes.contig_code[m[0]]) for m in I.member_gids]
I["contig"] = [genes.contig[m[0]] for m in I.member_gids]
I["member_positions"] = [pos[np.asarray(m)].tolist() for m in I.member_gids]
I["start_idx"] = [min(p) for p in I.member_positions]
I["end_idx"] = [max(p) for p in I.member_positions]
I["span_genes"] = I.end_idx - I.start_idx + 1
I.to_parquet(pkg / "gold/instances.parquet")
tok = pd.read_parquet(pkg / "tokens.parquet").token.to_numpy()
eng = root / "out/phase2/engine/odd__mmseqs__seq__e3c50s75__w3"
eng.mkdir(parents=True, exist_ok=True)
mods = {0: [600, 601, 602], 1: [603, 604], 2: [605, 606, 607, 608], 3: [609, 610, 611, 612]}
pd.DataFrame([(t, m) for m, ts in mods.items() for t in ts], columns=["token", "module"]).to_parquet(eng / "token_modules.parquet")
gc = {f"G{g}": g for g in range(200)}
odd = I[I.genome.map(gc) % 2 == 1]
mod_of = {"A": 0, "B": 1, "T": 2, "C1": 3, "C2": 3}
pd.DataFrame(dict(genome=odd.genome, contig=odd.contig, start=odd.start_idx, end=odd.end_idx, kind="conserved", level=2.0,
                  module_id=odd.system_subtype.map(mod_of))).to_parquet(eng / "loci_all_genomes.parquet")
(eng / "summary.json").write_text('{"fake": true}')
full = root / "fake_engine_full"
(full / "score_min2").mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(pkg / "scorer"))
import score_dpann_loci as sc  # noqa: E402
L = pd.DataFrame(dict(genome=I.genome, contig=I.contig, start=np.maximum(I.start_idx - 1, 0), end=I.end_idx, module_id=1))
per, _, _ = sc.score(pkg, L, controls=50, seed=17, threads=1, min_members=2)
per.to_parquet(full / "score_min2/instance_scores.parquet")
L.to_parquet(full / "loci_all_genomes.parquet")
env = dict(os.environ, LS_PKG=str(pkg), LS_TOKENS=str(pkg / "tokens.parquet"), LS_EMB=str(pkg / "emb.npy"),
           LS_ENGINE_FULL=str(full), OMP_NUM_THREADS="2", NUMBA_NUM_THREADS="2")
subprocess.run([sys.executable, "-B", str(here / "phase2.py"), "scan", "--out", str(root / "out"), "--threads", "2",
                "--K", "200", "--R", "30"], env=env, check=True)
