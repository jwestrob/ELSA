#!/usr/bin/env python
"""Login-node schema inspection of the DPANN gold set (small files only, <30 s, 2 threads)."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
import pandas as pd
import pyarrow.parquet as pq

RD = "/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111"
P = f"{RD}/ws_dpann/package"
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)
inst = pd.read_parquet(f"{P}/gold/instances.parquet")
print(inst.dtypes)
print(inst.head(3).T)
for c in ("caller", "status", "resolved", "order_trusted"):
    if c in inst:
        print(inst[c].value_counts(dropna=False).head(10))
m2 = inst[inst.n_members_resolved >= 2]
print("instances >=2 members by caller:\n", m2.caller.value_counts())
print("instances >=3 members by caller:\n", inst[inst.n_members_resolved >= 3].caller.value_counts())
for c in ("DefenseFinder", "TXSScan", "CRISPRCasTyper"):
    s = m2[m2.caller == c]
    print(c, "types", s.system_type.nunique(), "subtypes", s.system_subtype.nunique() if "system_subtype" in s else None)
    print(s.system_type.value_counts().head(15))
    if "system_subtype" in s:
        print(s.system_subtype.value_counts().head(10))
mem = pd.read_parquet(f"{P}/gold/members.parquet")
print(mem.head(), len(mem))
print(pq.read_schema(f"{P}/universe/genes.parquet"))
print(pq.read_schema(f"{P}/universe/genomes.parquet"))
print(pq.read_schema(f"{RD}/engine/dpann_tokens/mmseqs__seq__e3c50s75__dpann.parquet"))
print(pq.read_schema(f"{RD}/engine/results_dpann_v5/dpann__mmseqs__seq__e3c50s75__w3/loci_all_genomes.parquet"))
print(pq.read_schema(f"{RD}/engine/results_dpann_v5/dpann__mmseqs__seq__e3c50s75__w3/token_modules.parquet"))
