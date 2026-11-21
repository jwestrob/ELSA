#!/usr/bin/env python3
"""Repeat the index-backend sweep at DPANN scale.

`benchmarks/evaluation/index_backend_recall.csv` swept backends at 90,566 genes
and put faiss_ivfflat ahead on recall, which is why it is the default. ELSA now
runs corpora forty times that size, where the picture may differ: IVF-Flat's
query is ~8x slower and ELSA searches every gene once, so the cost lands in full
rather than being amortised.

Recall is measured against exact inner-product neighbours, and search time is
the full all-vs-all pass extrapolated from a query sample, so the totals are
what a real run pays.
"""
from __future__ import annotations

import argparse, os, time
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
STORE = Path("/Users/jacob/Documents/Sandbox/Sharur/Sharur/data/dpann/synteny/store_v2")


def recall_at_k(got: np.ndarray, truth: np.ndarray, k: int) -> float:
    return float(np.mean([len(set(g[:k]) & set(t[:k])) / k for g, t in zip(got, truth)]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--embeddings", type=Path, default=STORE / "embeddings.npy")
    ap.add_argument("--queries", type=int, default=2000)
    ap.add_argument("-k", type=int, default=60)
    ap.add_argument("--threads", type=int, default=10)
    ap.add_argument("--output", type=Path,
                    default=REPO / "benchmarks/evaluation/index_backend_recall_at_scale.csv")
    args = ap.parse_args()

    import faiss, hnswlib
    faiss.omp_set_num_threads(args.threads)

    X = np.ascontiguousarray(np.load(args.embeddings, mmap_mode="r"), dtype=np.float32)
    X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)
    n, dim = X.shape
    print(f"corpus {n:,} x {dim}  threads {args.threads}", flush=True)

    rng = np.random.default_rng(17)
    qs = np.sort(rng.choice(n, args.queries, replace=False))
    flat = faiss.IndexFlatIP(dim); flat.add(X)
    _, truth = flat.search(X[qs], args.k)
    del flat
    print("exact ground truth ready", flush=True)

    rows = []

    def record(name, build_s, query_s, got):
        # ELSA queries every gene once, so the pass a run pays is n queries.
        full = query_s * (n / len(qs))
        rows.append({"backend": name, "build_time_s": round(build_s, 1),
                     "query_time_s_2000": round(query_s, 3),
                     "full_search_s": round(full, 1),
                     "total_s": round(build_s + full, 1),
                     f"recall@{args.k}": round(recall_at_k(got, truth, args.k), 4),
                     "n_genes": n, "dim": dim})
        print(f"  {name:<28} build {build_s:7.1f}s  full search {full:8.1f}s  "
              f"total {build_s+full:8.1f}s  recall {rows[-1][f'recall@{args.k}']:.4f}",
              flush=True)

    for ef in (128, 256, 512):
        t = time.perf_counter()
        h = hnswlib.Index(space="cosine", dim=dim)
        h.init_index(max_elements=n, ef_construction=200, M=32)
        h.add_items(X, np.arange(n, dtype=np.int32))
        build = time.perf_counter() - t
        h.set_ef(ef)
        t = time.perf_counter(); lab, _ = h.knn_query(X[qs], k=args.k)
        record(f"hnsw (M=32, ef={ef})", build, time.perf_counter() - t, lab)
        del h

    nlist = max(16, min(int(np.sqrt(n)), 4096))
    t = time.perf_counter()
    quant = faiss.IndexFlatIP(dim)
    ivf = faiss.IndexIVFFlat(quant, dim, nlist, faiss.METRIC_INNER_PRODUCT)
    ivf.train(X); ivf.add(X)
    build = time.perf_counter() - t
    for nprobe in (16, 32, 64, 128):
        ivf.nprobe = nprobe
        t = time.perf_counter(); _, lab = ivf.search(X[qs], args.k)
        record(f"faiss_ivfflat (nprobe={nprobe})", build, time.perf_counter() - t, lab)

    frame = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(f"\n{frame.to_string(index=False)}\n\nwrote {args.output}")


if __name__ == "__main__":
    main()
