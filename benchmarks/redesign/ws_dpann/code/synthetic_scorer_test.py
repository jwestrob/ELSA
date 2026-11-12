#!/usr/bin/env python
"""Tiny synthetic package to exercise score_dpann_loci.py before the real gold table exists.

Two contigs (20 and 30 genes), four instances with hand-computed answers.
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from score_dpann_loci import score  # noqa: E402

d = Path(tempfile.mkdtemp())
(d / "universe").mkdir()
(d / "gold").mkdir()
pd.DataFrame(dict(genome=["G1", "G1"], contig=["c1", "c2"], contig_code=[0, 1], n_genes=[20, 30])).to_parquet(
    d / "universe" / "contigs.parquet")
inst = pd.DataFrame(dict(
    instance_id=[0, 1, 2, 3], caller=["X"] * 4, system_type=["A", "A", "B", "B"], genome="G1",
    contig=["c1", "c1", "c2", "c2"], contig_code=[0, 0, 1, 1],
    start_idx=[2, 10, 5, 20], end_idx=[4, 15, 6, 29],
    n_members_resolved=[3, 3, 2, 2], resolved=True, order_trusted=True,
    member_positions=[[2, 3, 4], [10, 12, 15], [5, 6], [20, 29]]))
inst.to_parquet(d / "gold" / "instances.parquet")
loci = pd.DataFrame(dict(genome="G1", contig=["c1", "c1", "c2", "c2"], start=[2, 9, 0, 20], end=[4, 16, 9, 20],
                         module_id=["m1", "m1", "m2", "m3"]))
per, rep, raw = score(d, loci, controls=50, seed=17)
per = per.set_index("instance_id")
exp = {0: (1.0, 1.0, 1.0), 1: (1.0, 3 / 8, 6 / 11), 2: (1.0, 2 / 10, 4 / 12), 3: (0.5, 1.0, 2 / 3)}
for i, (cov, prec, f1) in exp.items():
    r = per.loc[i]
    assert abs(r.coverage - cov) < 1e-12 and abs(r.boundary_precision - prec) < 1e-12 and abs(r.f1 - f1) < 1e-12, (i, r)
assert per.loc[3].covered50 and per.loc[3].localized
assert per.loc[1].span_precision == 6 / 8, per.loc[1].span_precision
assert per.loc[2].span_precision == 2 / 10
print(per[["coverage", "boundary_precision", "f1", "span_precision", "best_module", "control_mean_f1"]])
print(rep["by_type"]["X"]["A"]["module_plurality_share_hit"], rep["by_type"]["X"]["B"]["module_plurality_share_hit"])
assert rep["by_type"]["X"]["A"]["module_plurality_share_hit"] == 1.0
assert rep["by_type"]["X"]["B"]["module_plurality_share_hit"] == 0.5
print("synthetic scorer test PASSED")
