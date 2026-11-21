#!/usr/bin/env python3
"""Export sequences (read-only on the Sharur DPANN DB) for the 175,729 labeled proteins -> data/labeled.faa (trailing '*'
stripped, header = protein_id)."""
from pathlib import Path
import duckdb, pandas as pd
H = Path(__file__).resolve().parents[1]
d = pd.read_parquet(H.parent / "pass/dpann_slot_rescue_pass.parquet", columns=["protein_id"])
con = duckdb.connect("/Users/jacob/Documents/Sandbox/Sharur/Sharur/data/dpann/sharur.duckdb", read_only=True)
con.execute("CREATE TEMP TABLE q AS SELECT unnest(?::VARCHAR[]) AS protein_id", [d.protein_id.tolist()])
s = con.execute("SELECT protein_id, sequence FROM proteins JOIN q USING (protein_id)").df()
assert len(s) == len(d) and s.sequence.notna().all()
s = s.set_index("protein_id").loc[d.protein_id]
n_int = 0
with open(H / "data/labeled.faa", "w") as f:
    for p, q in zip(s.index, s.sequence):
        q = q.rstrip("*")
        n_int += "*" in q
        f.write(f">{p}\n{q}\n")
print(len(s), "sequences;", n_int, "with an internal '*'")
