#!/usr/bin/env python
"""Frozen run 3 scorer (GTDB complete genomes; FROZEN3.json 'ribosomal_scoring' / 'descriptive_statistics').
No curated truth on this set: ribosomal runs (standing definition + any fitting locus) and descriptive statistics only.
Writes results/score_gtdb3.json and results/locus_tables.json (paths + sha256 of the locus tables for the audit scorer).
usage: fscore3.py score --threads T --runs PATH
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

RD = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111")
FZ = RD / "long_loci/frozen3"
PKG = RD / "testset3/package"
sys.path.insert(0, str(RD / "ribosomal/code"))
import ribosomal_check as RC  # noqa: E402

NB = 10_000
OUTPUTS = {"PRIMARY_Z_L4bp_t0.02": FZ / "loci/gtdb3_Z/L4bp_t0.02.parquet", "Z_L1": FZ / "loci/gtdb3_Z/L1.parquet",
           "SECONDARY_WA_L4b_t0.2": FZ / "loci/gtdb3_WA/L4b_t0.2.parquet", "WA_L1": FZ / "loci/gtdb3_WA/L1.parquet",
           "engine_DH": FZ / "loci/gtdb3_engine/DH.parquet"}
PAIRS = [("PRIMARY_Z_L4bp_t0.02", "engine_DH"), ("SECONDARY_WA_L4b_t0.2", "engine_DH"), ("PRIMARY_Z_L4bp_t0.02", "SECONDARY_WA_L4b_t0.2"),
         ("PRIMARY_Z_L4bp_t0.02", "Z_L1"), ("SECONDARY_WA_L4b_t0.2", "WA_L1")]


def log(*a):
    print(time.strftime("%H:%M:%S", time.gmtime()), *a, flush=True)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(16 << 20), b""):
            h.update(b)
    return h.hexdigest()


@njit(cache=True)
def _any_fit(a, b, maxlen, qs, qe):
    out = np.zeros(len(qs), np.int8)
    for r in range(len(qs)):
        s, e = qs[r], qe[r]
        j = np.searchsorted(a, s - maxlen + 1)
        n = e - s + 1
        while j < len(a) and a[j] <= e:
            if b[j] >= s:
                ov = min(b[j], e) - max(a[j], s) + 1
                if ov >= 0.8 * n and ov >= 0.8 * (b[j] - a[j] + 1):
                    out[r] = 1
                    break
            j += 1
    return out


def cmd_score(a):
    t0 = time.time()
    genes = pd.read_parquet(PKG / "universe/genes.parquet", columns=["gid", "genome_code"]).sort_values("gid")
    n = len(genes)
    gm = pd.read_parquet(PKG / "universe/genomes.parquet").sort_values("genome_code").reset_index(drop=True)
    dom_col = "domain" if "domain" in gm else None
    gdom = gm[dom_col].to_numpy()[genes.genome_code.to_numpy()] if dom_col else np.full(n, "all", object)
    runs = pd.read_parquet(a.runs)
    if "run_set" in runs:
        runs = runs[(runs.run_set == "g2") & (runs.n_marked >= 16)]
    runs = runs.sort_values("gid_start").reset_index(drop=True)
    rdom = gm[dom_col].to_numpy()[runs.genome_code.to_numpy()] if dom_col else np.full(len(runs), "all", object)
    rsub = {"all": np.ones(len(runs), bool)}
    for d in sorted(set(rdom)):
        rsub[str(d)] = rdom == d
    if "split" in gm:
        sp = gm.split.to_numpy()[runs.genome_code.to_numpy()]
        for v in sorted(set(sp)):
            rsub[f"split_{v}"] = sp == v
    else:
        rsub["even_genome_code"] = (runs.genome_code % 2 == 0).to_numpy()
    res = dict(runs_source=str(a.runs), runs_sha256=sha(a.runs), runs={k: int(v.sum()) for k, v in rsub.items()}, outputs={}, pairs={},
               locus_tables={})
    ribs = {}
    for label, path in OUTPUTS.items():
        if not path.exists():
            log("absent", label)
            continue
        df = pd.read_parquet(path)
        s, e = df.s.to_numpy(np.int64), df.e.to_numpy(np.int64)
        k = e > s
        s, e = s[k], e[k]
        o = np.lexsort((e, s))
        s, e = s[o], e[o]
        res["locus_tables"][label] = dict(path=str(path), sha256=sha(path), rows=int(len(df)), columns=list(df.columns))
        ls = dict(a=s, b=e, mod=np.full(len(s), -1, np.int64), maxlen=int((e - s + 1).max()) if len(s) else 1, n=int(len(s)))
        sc = RC.score(ls, runs.gid_start.to_numpy(), runs.gid_end.to_numpy())
        sc["any_fit"] = _any_fit(s, e, ls["maxlen"], runs.gid_start.to_numpy(np.int64), runs.gid_end.to_numpy(np.int64)).astype(bool)
        ribs[label] = sc
        rec = sc.recovered.to_numpy()
        item = {"ribosomal": {kk: dict(runs=int(m.sum()), recovered=int(rec[m].sum()), resolved=int(sc.resolved.to_numpy()[m].sum()),
                                       any_fit=int(sc.any_fit.to_numpy()[m].sum()),
                                       median_precision_recovered=float(sc.precision[m & rec].median()) if (m & rec).any() else None)
                              for kk, m in rsub.items()}}
        cov = np.zeros(n + 1, np.int32)
        np.add.at(cov, s, 1)
        np.add.at(cov, e + 1, -1)
        inl = np.cumsum(cov)[:n] > 0
        item["coverage"] = dict(loci=int(len(s)), genes_in_loci_frac=float(inl.mean()),
                                by_domain={str(d): float(inl[gdom == d].mean()) for d in sorted(set(gdom))},
                                span_q50_q90_q99_max=np.quantile(e - s + 1, [.5, .9, .99, 1]).tolist() if len(s) else None)
        res["outputs"][label] = item
        log(label, item["ribosomal"]["all"], "genes", round(item["coverage"]["genes_in_loci_frac"], 3))
        pd.DataFrame(dict(run_id=runs.run_id.to_numpy() if "run_id" in runs else np.arange(len(runs)), best_frac=sc.best_frac,
                          precision=sc.precision, recovered=sc.recovered, resolved=sc.resolved, any_fit=sc.any_fit)).to_parquet(
            FZ / "results" / f"ribo_{label}.parquet", index=False)
    for A_, B_ in PAIRS:
        if A_ in ribs and B_ in ribs:
            d = ribs[A_].resolved.to_numpy().astype(int) - ribs[B_].resolved.to_numpy().astype(int)
            rng = np.random.default_rng(17)
            m = d[rng.integers(0, len(d), size=(NB, len(d)))].sum(1)
            res["pairs"][f"{A_}__vs__{B_}"] = dict(resolved=[int(d.sum()), float(np.quantile(m, .025)), float(np.quantile(m, .975))])
    res["written"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    res["seconds"] = round(time.time() - t0, 1)
    Path(FZ / "results/score_gtdb3.json").write_text(json.dumps(res, indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x)) + "\n")
    Path(FZ / "results/locus_tables.json").write_text(json.dumps(res["locus_tables"], indent=1) + "\n")
    log("wrote", FZ / "results/score_gtdb3.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["score"])
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--runs", required=True)
    a = ap.parse_args()
    cmd_score(a)


if __name__ == "__main__":
    main()
