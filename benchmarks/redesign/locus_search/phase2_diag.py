#!/usr/bin/env python
"""Phase-2 diagnostic (small; login node, 2 threads, < 30 s): for gold instances (>= 2 members) on even
genomes, the fraction of members whose token belongs to any odd-genome module, and the fraction of
instances whose plurality module holds >= 50% of the members (what one module profile can reach)."""
import json
import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
import pandas as pd

RD = "/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111"
P = f"{RD}/ws_dpann/package"
E = f"{RD}/locus_search/out/phase2/engine/odd__mmseqs__seq__e3c50s75__w3"
inst = pd.read_parquet(f"{P}/gold/instances.parquet", columns=["instance_id", "caller", "genome", "member_gids", "n_members_resolved"])
gm = pd.read_parquet(f"{P}/universe/genomes.parquet", columns=["genome", "genome_code"])
inst = inst[(inst.n_members_resolved >= 2) & (inst.genome.map(dict(zip(gm.genome, gm.genome_code))) % 2 == 0)]
t = pd.read_parquet(f"{RD}/engine/dpann_tokens/mmseqs__seq__e3c50s75__dpann.parquet")
tok = np.full(int(t.gid.max()) + 1, -1, np.int64)
tok[t.gid.to_numpy()] = t.token.to_numpy()
tm = pd.read_parquet(f"{E}/token_modules.parquet")
mod = np.full(int(tok.max()) + 1, -1, np.int64)
mod[tm.token.to_numpy()] = tm.module.to_numpy()
out = {}
for c, g in inst.groupby("caller"):
    in_mod, plural = [], []
    for m in g.member_gids:
        mm = mod[tok[np.asarray(m, np.int64)]]
        in_mod.append((mm >= 0).mean())
        v = mm[mm >= 0]
        plural.append(np.bincount(v).max() / len(mm) if len(v) else 0.0)
    in_mod, plural = np.array(in_mod), np.array(plural)
    out[c] = dict(instances=int(len(g)), mean_member_frac_in_any_module=float(in_mod.mean()),
                  inst_ge_half_members_in_modules=float((in_mod >= 0.5).mean()),
                  inst_ge_half_members_in_one_module=float((plural >= 0.5).mean()),
                  inst_all_members_in_one_module=float((plural == 1).mean()))
print(json.dumps(out, indent=1))
