#!/usr/bin/env python
"""Ribosomal-locus acceptance check on an arbitrary table of per-genome intervals.

Input side adapted from ribosomal/code/ribosomal_check.py (copied here unchanged as
ribosomal_check.py and imported): the locus set is every distinct (genome, contig, start, end)
interval of a table (legacy: both sides of every block; engine: loci_all_genomes.parquet, either
all kinds and levels or one kind@level). Gold runs and the random-window controls are the cached ones
in $RD/ribosomal/gold/. Definitions kept: recovered = best interval covers >= 50% of the run;
resolved = best interval covers >= 80% at precision >= 0.8 (best = largest overlap, ties: shorter);
fragments = intervals overlapping the run. No module ids (-1), so module purity is not computed.

usage: ribo_blocks.py --loci TABLE --name NAME --out DIR [--kind K --level L] [--threads N]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ribosomal_check as rc   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loci", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--kind")
    ap.add_argument("--level", type=float)
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    t0 = time.time()
    import numba
    numba.set_num_threads(a.threads)
    runs = pd.read_parquet(rc.CACHE / "runs.parquet")
    rnd = pd.read_parquet(rc.CACHE / "controls_random.parquet")
    contigs = pd.read_parquet(rc.PKG / "universe/contigs.parquet").sort_values("first_gid").reset_index(drop=True)
    l = pd.read_parquet(a.loci)
    if a.kind is not None:
        l = l[(l.kind == a.kind) & (l.level == a.level)]
    l = l.drop_duplicates(["genome", "contig", "start", "end"])
    assert contigs.contig.is_unique
    cat = pd.Categorical(l.contig.to_numpy(), categories=contigs.contig.to_numpy())
    code = cat.codes
    assert (code >= 0).all()
    assert (contigs.genome.to_numpy()[code] == l.genome.to_numpy()).all()
    first = contigs.first_gid.to_numpy()[code]
    A = first + l.start.to_numpy()
    Bv = first + l.end.to_numpy()
    o = np.argsort(A, kind="stable")
    A, Bv = A[o].astype(np.int64), Bv[o].astype(np.int64)
    ls = dict(a=A, b=Bv, mod=np.full(len(A), -1, np.int64), maxlen=int((Bv - A + 1).max()), n=int(len(A)))
    res = dict(name=a.name, loci=str(a.loci), kind=a.kind, level=a.level, intervals=int(len(A)),
               gold_cache=str(rc.CACHE), results={})
    for rs, mm in (("g2", 16), ("g2", 8)):
        key = f"{rs}_m{mm}"
        rr = runs[(runs.run_set == rs) & (runs.n_marked >= mm)].reset_index(drop=True)
        sc = rc.score(ls, rr.gid_start.to_numpy(), rr.gid_end.to_numpy())
        s = rc.summarize(rr, sc)
        rcs = rnd[rnd.run_id.isin(rr.run_id)]
        c1 = rc.score(ls, rcs.gid_start.to_numpy(), rcs.gid_end.to_numpy())
        s["random"] = rc.summarize(None, c1, with_tax=False)
        if key == "g2_m16":
            fam = (rr.family == "Parvarchaeaceae").to_numpy()
            s["parvarchaeaceae_runs"] = int(fam.sum())
            s["parvarchaeaceae_recovered_runs"] = int(sc.recovered.to_numpy()[fam].sum())
            s["parvarchaeaceae_resolved_runs"] = int(sc.resolved.to_numpy()[fam].sum())
            per = pd.concat([rr[["run_id", "genome_code", "phylum", "family", "n_genes"]], sc], axis=1)
            Path(a.out).mkdir(parents=True, exist_ok=True)
            per.to_parquet(Path(a.out) / "ribosomal_runs.parquet", index=False)
        res["results"][key] = s
    res["seconds"] = round(time.time() - t0, 1)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "ribosomal_check.json", "w") as fh:
        json.dump(res, fh, indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x))
    p = res["results"]["g2_m16"]
    r = p["random"]
    line = (f"RIBOSOMAL {a.name}: intervals={len(A):,} runs={p['runs']} recovered={p['recovered']} "
            f"resolved={p['resolved']} (random windows {r['recovered_frac']:.3f}/{r['resolved_frac']:.3f}) "
            f"median_coverage={p['median_coverage']:.3f} median_fragments={p['median_fragments']:g} "
            f"median_precision={p['median_precision_recovered'] or 0:.3f} "
            f"median_best_len={p['median_best_len_recovered']} "
            f"Parvarchaeaceae={p['parvarchaeaceae']} resolved {p['parvarchaeaceae_resolved']} [{res['seconds']}s]")
    (out / "ribosomal_check.txt").write_text(line + "\n")
    print(line, flush=True)


if __name__ == "__main__":
    main()
