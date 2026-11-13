#!/usr/bin/env python
"""POST HOC (decided after the co-fold result): DPANN homolog count of every class-a and class-b chain, so that the
co-fold rate can be stratified by whether a chain has any detectable homolog. MMseqs2 18, -s 7.5, E <= 1e-3,
--max-seqs 10000 against all 3,585,303 DPANN proteins; a homolog = a target other than the gene itself (and other
than identical-sequence copies) with >= 50% query coverage. Writes $RD/kinds/passers/homologs_ab.tsv."""
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

RD = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111")
K = RD / "kinds"
OUT = K / "passers"
MM = "/groups/banfield/users/jwestrob/bin/mmseqs_18/mmseqs/bin/mmseqs"
FAA = RD / "ws_dpann/work/proteins_gid.faa"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, required=True)
    a = ap.parse_args()
    f = pd.read_csv(K / "cofold/folds_kinds.tsv", sep="\t")
    f = f[f.cls.isin(["a_high", "b_low"])]
    jobs = {json.loads(l)["fold_id"]: json.loads(l) for l in open(K / "cofold/jobs.jsonl")}
    q = OUT / "ab_queries.faa"
    seen = set()
    with open(q, "w") as fh:
        for r in f.itertuples():
            for ch, gid in zip(jobs[r.fold_id]["chains"], (r.gid_A, r.gid_B)):
                if gid not in seen:
                    seen.add(gid)
                    fh.write(f">{gid}\n{ch['seq']}\n")
    tmp = OUT / "mm_ab"
    tmp.mkdir(exist_ok=True)
    hits = OUT / "ab_hits.tsv"
    T = str(a.threads)
    if not hits.exists():
        for cmd in ([MM, "createdb", str(q), str(tmp / "qdb")],
                    [MM, "createdb", str(FAA), str(tmp / "tdb"), "--createdb-mode", "1"],
                    [MM, "search", str(tmp / "qdb"), str(tmp / "tdb"), str(tmp / "aln"), str(tmp / "t"), "-s", "7.5",
                     "-e", "1e-3", "--max-seqs", "10000", "--threads", T],
                    [MM, "convertalis", str(tmp / "qdb"), str(tmp / "tdb"), str(tmp / "aln"), str(hits),
                     "--format-output", "query,target,evalue,pident,qstart,qend,qlen", "--threads", T]):
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    h = pd.read_csv(hits, sep="\t", header=None, names=["q", "t", "evalue", "pident", "qs", "qe", "ql"])
    uq = np.load(K / "prep/uq.npy")
    h = h[(h.evalue <= 1e-3) & ((h.qe - h.qs + 1) / h.ql >= 0.5) & (uq[h.t.to_numpy()] != uq[h.q.to_numpy()])]
    n = h.groupby("q").t.nunique()
    out = pd.DataFrame(dict(gid=sorted(seen)))
    out["homologs"] = out.gid.map(n).fillna(0).astype(int)
    out.to_csv(OUT / "homologs_ab.tsv", sep="\t", index=False)
    print(out.homologs.describe(), (out.homologs == 0).mean())


if __name__ == "__main__":
    main()
