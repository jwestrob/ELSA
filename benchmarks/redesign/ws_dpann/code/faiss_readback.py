#!/usr/bin/env python
"""Read vectors back from a Sharur FAISS sidecar and compare them with the h5 rows they index."""
import argparse
import json
import sqlite3
from pathlib import Path

import faiss
import h5py
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--index", required=True)
ap.add_argument("--ids", required=True)
ap.add_argument("--h5", required=True)
ap.add_argument("--rows-json", required=True, help="pooling JSON whose windows_complete gives the h5 rows")
ap.add_argument("--out", type=Path, required=True)
a = ap.parse_args()
starts = json.loads(Path(a.rows_json).read_text())["windows_complete"]
rows = [s + i for s in starts for i in range(32)]
with h5py.File(a.h5, "r") as f:
    pids = [f["protein_ids"][r].decode() for r in rows]
    H = np.vstack([f["embeddings"][r] for r in rows])
con = sqlite3.connect(f"file:{a.ids}?mode=ro", uri=True)
frow = [con.execute("SELECT row_index FROM vector_ids WHERE protein_id = ?", (p,)).fetchone() for p in pids]
index = faiss.read_index(a.index)
ivf = faiss.extract_index_ivf(index)
ivf.make_direct_map()
res = dict(index=a.index, index_type=type(index).__name__, ntotal=int(index.ntotal), sampled=len(rows),
           found_in_id_map=sum(x is not None for x in frow))
F = np.vstack([index.reconstruct(int(x[0])) for x in frow if x is not None])
Hn = H[[i for i, x in enumerate(frow) if x is not None]]
cos = (F * Hn).sum(1) / (np.linalg.norm(F, axis=1) * np.linalg.norm(Hn, axis=1))
res.update(faiss_vs_h5_cos_min=float(cos.min()), faiss_vs_h5_cos_median=float(np.median(cos)),
           faiss_norm_median=float(np.median(np.linalg.norm(F, axis=1))))
a.out.write_text(json.dumps(res, indent=2) + "\n")
print(json.dumps(res, indent=2))
