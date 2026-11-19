#!/usr/bin/env python
"""ribo_audit part 3, supplementary annotation on the SR-VP package (predeclared3.json).

(1) the bacterial literature members (16 KOfam profiles, ko_list thresholds) over every package protein;
(2) Pfam-A 38.1 at GA over every gene within 40 genes of a g2_m16 run span (description and annotated flag).
Full KOfam on the windows was dropped (15 GB of profiles; would exceed the 2-hour bound). hmmsearch per protein chunk in
$SLURM_CPUS_ON_NODE parallel processes. Writes $RD/ribo_audit/work3/annot/.
"""
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

RD = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111")
PKG = RD / "ws_srvp_bins/package"
AS = Path("/groups/banfield/users/jwestrob/.config/Astra")
OUT = RD / "ribo_audit/work3/annot"
HS = "/shared/software/bin/hmmsearch"
EXTRA = ["K02601", "K03043", "K03046", "K02358", "K02355", "K00939", "K01265", "K02518", "K03073", "K01883", "K03685",
         "K03218", "K16786", "K16787", "K16785", "K06173"]
FLANK = 40


def log(*a):
    print(time.strftime("%H:%M:%S", time.gmtime()), *a, flush=True)


def write_chunks(seqs, tag, T):
    paths = [OUT / f"work/{tag}_{i}.faa" for i in range(T)]
    fhs = [open(p, "w") for p in paths]
    for i, (h, s) in enumerate(seqs):
        fhs[i % T].write(f">{h}\n{s}\n")
    for f in fhs:
        f.close()
    return paths


def run_parallel(hmm, chunks, tag, extra_args):
    procs = []
    for i, c in enumerate(chunks):
        tbl = OUT / f"work/{tag}_{i}.tbl"
        procs.append((subprocess.Popen([HS, "--cpu", "1", "--noali", "-o", "/dev/null", *extra_args(tbl), str(hmm), str(c)]),
                      tbl))
    for p, _ in procs:
        p.wait()
        assert p.returncode == 0, "hmmsearch failed"
    return [t for _, t in procs]


def main():
    t0 = time.time()
    T = int(os.environ["SLURM_CPUS_ON_NODE"])
    (OUT / "work").mkdir(parents=True, exist_ok=True)
    genes = pd.read_parquet(PKG / "universe/genes.parquet", columns=["gid", "contig_code"]).sort_values("gid")
    cc = genes.contig_code.to_numpy()
    first = pd.Series(genes.gid.to_numpy()).groupby(cc).min()
    last = pd.Series(genes.gid.to_numpy()).groupby(cc).max()
    runs = pd.read_parquet(PKG / "ribosomal/runs.parquet")
    runs = runs[(runs.run_set == "g2") & runs.m16]
    win = np.zeros(len(genes), bool)
    for r in runs.itertuples():
        lo = max(int(first[r.contig_code]), r.gid_start - FLANK)
        hi = min(int(last[r.contig_code]), r.gid_end + FLANK)
        win[lo:hi + 1] = True
    np.save(OUT / "window_gids.npy", np.flatnonzero(win))
    log("runs", len(runs), "window genes", int(win.sum()))
    allseq, winseq = [], []
    h, buf = None, []
    with open(PKG / "universe/proteins.faa") as fh:
        for line in fh:
            if line.startswith(">"):
                if h is not None:
                    s = "".join(buf).rstrip("*")
                    allseq.append((h, s))
                    if win[int(h)]:
                        winseq.append((h, s))
                h, buf = line[1:].strip().split()[0], []
            else:
                buf.append(line.strip())
        s = "".join(buf).rstrip("*")
        allseq.append((h, s))
        if win[int(h)]:
            winseq.append((h, s))
    log("proteins", len(allseq), "window proteins", len(winseq))
    # (1) bacterial members genome-wide
    kl = pd.read_csv(AS / "ko_list", sep="\t")
    kl["threshold"] = pd.to_numeric(kl.threshold, errors="coerce")
    kl = kl[kl.knum.isin(EXTRA)].set_index("knum")
    with open(OUT / "work/extra.hmm", "w") as fh:
        for k in EXTRA:
            fh.write((AS / "KOFAM" / f"{k}.hmm").read_text())
    ch = write_chunks(allseq, "all", T)
    tbls = run_parallel(OUT / "work/extra.hmm", ch, "extra", lambda tbl: ["--tblout", str(tbl), "-E", "1e-5"])
    rows = []
    for t in tbls:
        for line in open(t):
            if line.startswith("#"):
                continue
            f = line.split()
            gid, ko, full, dom = int(f[0]), f[2], float(f[5]), float(f[8])
            th, st = kl.loc[ko, "threshold"], kl.loc[ko, "score_type"]
            sc = full if st == "full" else dom
            if np.isfinite(th) and sc >= th:
                rows.append((gid, ko, sc, sc / th))
    ex = pd.DataFrame(rows, columns=["gid", "ko", "score", "ratio"]).sort_values("ratio", ascending=False).drop_duplicates("gid")
    ex.to_parquet(OUT / "extra_ko.parquet", index=False)
    log("extra KO hits", len(ex), ex.ko.value_counts().to_dict(), f"{time.time()-t0:.0f}s")
    # (2) Pfam on the windows
    ch = write_chunks(winseq, "win", T)
    tbls = run_parallel(AS / "PFAM/Pfam-A.hmm", ch, "pfam", lambda tbl: ["--domtblout", str(tbl), "--cut_ga"])
    rows = []
    for t in tbls:
        for line in open(t):
            if line.startswith("#"):
                continue
            f = line.split()
            rows.append((int(f[0]), f[3], float(f[13]), int(f[19])))
    pf = pd.DataFrame(rows, columns=["gid", "name", "dom_score", "env_from"])
    pf.to_parquet(OUT / "window_pfam_domains.parquet", index=False)
    arch = pf.sort_values(["gid", "env_from"]).groupby("gid").name.agg(",".join)
    top = pf.sort_values("dom_score", ascending=False).drop_duplicates("gid").set_index("gid").name
    out = pd.DataFrame({"gid": arch.index, "arch": arch.to_numpy(), "top": top.reindex(arch.index).to_numpy()})
    out.to_parquet(OUT / "window_pfam.parquet", index=False)
    log("window genes with Pfam", len(out), "of", len(winseq), f"{time.time()-t0:.0f}s")
    for p in (OUT / "work").glob("*.faa"):
        p.unlink()
    log("done", f"{time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
