"""Level registry (workstream hierarchy): level name -> token source, and the predeclared level sets.

V0 levels r<res> = Leiden (modularity, seed 17) memberships of the context_kernel all-edge similarity graph;
sequence levels = MMseqs2 clusterings; nested__<set>__<level> = top-down intersections written by nested.py.
Every list is ordered by decreasing token count (finest first)."""
from pathlib import Path

import numpy as np

RD = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111")
FAM = RD / "context_kernel/families"
SEQ = {"id30c80": RD / "ws_dpann/package/tokens/mmseqs__seq__id30c80__dpann.parquet",
       "e3c50s75": RD / "engine/dpann_tokens/mmseqs__seq__e3c50s75__dpann.parquet",
       "e3c30s75": RD / "engine/dpann_tokens/mmseqs__seq__e3c30s75__dpann.parquet"}
V0_ALL = ["r300000", "r100000", "r30000", "r26000", "r23000", "r19000", "r15000", "r11000", "r10000", "r7500",
          "r3000", "r1000", "r300"]
L8 = ["r30000", "r23000", "r19000", "r15000", "r11000", "r7500", "r3000", "r1000"]
S3 = ["id30c80", "e3c50s75", "e3c30s75"]

LEVEL_SETS = {
    "L8": L8, "L13": V0_ALL, "S3": S3,
    # task 2 candidates (predeclared.json)
    "E1": ["r30000", "r19000", "r11000"], "E2": ["r19000", "r11000", "r3000"],
    "E3": ["r30000", "r19000", "r11000", "r3000"], "E4": ["r30000", "r19000", "r11000", "r3000", "r1000"],
    "E5": ["r30000", "r23000", "r19000", "r15000", "r11000"], "E6": ["r19000", "r7500", "r1000"],
    "S1": S3, "S2a": ["e3c50s75", "e3c30s75"], "S2b": ["id30c80", "e3c50s75"], "S2c": ["id30c80", "e3c30s75"],
}
TOKENS = {"r300000": 3562093, "r100000": 2946570, "r30000": 709751, "r26000": 546020, "r23000": 432675,
          "r19000": 297708, "r15000": 189089, "r11000": 111780, "r10000": 97174, "r7500": 67420, "r3000": 28798,
          "r1000": 13391, "r300": 7099, "id30c80": 674172, "e3c50s75": 430471, "e3c30s75": 380979}


def path_of(name):
    if name.startswith("nested__"):
        return RD / "hierarchy/levels" / f"{name}.npy"
    if name in SEQ:
        return SEQ[name]
    return FAM / f"sim_all_{name}.npy"


def by_tokens(names):
    """Sort level names by decreasing token count (finest first)."""
    return sorted(names, key=lambda x: -TOKENS[x])


def load_tokens(name, n=None):
    """Tokens in gid order (gid = 0..N-1), 0..T-1, -1 unassigned."""
    p = path_of(name)
    if p.suffix == ".npy":
        m = np.load(p).astype(np.int64)
        if name.startswith("nested__"):
            return m
        _, t = np.unique(m, return_inverse=True)
        return t.astype(np.int64)
    import pandas as pd
    d = pd.read_parquet(p)
    N = len(d) if n is None else n
    t = np.full(N, -1, np.int64)
    t[d.gid.to_numpy()] = d.token.to_numpy()
    return t


def spec(names):
    """--levels string for recurrence_ml.py."""
    return ",".join(f"{x}={path_of(x)}" for x in names)
