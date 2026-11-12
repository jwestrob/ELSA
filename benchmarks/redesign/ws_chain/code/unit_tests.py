#!/usr/bin/env python
"""Synthetic checks of chain_v3 (score-first DP, gap cost, trimming) and common.qvalues."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_published, qvalues
import chain_v3

chain_mod, group_type = load_published()
nq = nt = 30
qmeta = pd.DataFrame(dict(row_index=range(nq), position_index=range(nq), sample_id="Q", contig_id="q1",
                          gene_id=[f"q{i}" for i in range(nq)], strand=1))
rmeta = pd.DataFrame(dict(row_index=range(nt), position_index=range(nt), sample_id="T", contig_id="t1",
                          gene_id=[f"t{i}" for i in range(nt)], strand=1))
pairs = [(i, i, 2.0) for i in range(10)] + [(2, 7, -1.0), (20, 25, 3.0), (21, 26, 3.0), (22, 29, -4.0)]
pairs[5] = (5, 5, -3.0)
a = pd.DataFrame(pairs, columns=["query_row_index", "target_row_index", "w"])
cos = np.full(len(a), 0.9)
for mode, gap, want in [("score_first", 0.0, {9: 18.0, 2: 6.0}), ("score_first", 0.5, {9: 17.5, 2: 6.0}),
                        ("length_first", 0.0, {10: 9.0, 3: 2.7})]:
    b, ba = chain_v3.chain(a, qmeta, rmeta, chain_mod, group_type, mode=mode, cos=cos, weights=a.w.to_numpy(),
                           gap_scale=gap)
    got = dict(zip(b.n_anchors, b.chain_score))
    print(mode, gap, got, b.sum_w.tolist())
    assert got == want, (mode, gap, got, want)
obs = np.array([10, 9, 8, 7, 1, 1, 0.5]); perm = np.array([10, 2, 2, 1, 1, 1, 1, 0.5, 0.5])
q = qvalues(obs, perm, 3)
print("q", q.round(3).tolist())
assert q[0] <= q[3] <= q[4] and np.isclose(q[0], (1 / 3) / 4)
print("unit tests passed")
