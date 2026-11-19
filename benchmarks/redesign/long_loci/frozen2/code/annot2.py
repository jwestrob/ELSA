#!/usr/bin/env python
"""Frozen run 2: restricted annotation for ribo_audit's accurate reading on the SR-VP package (FROZEN2.json).

Members of the audit's rules (bio.py, imported read-only for N_RULES / short_rp): R = ribosomal Pfam at GA (package
ribosomal/marked.parquet) or a KOfam KO whose definition contains 'ribosomal protein'; N = N_RULES members by KO or Pfam;
EF2 = EFG_C or EFG_IV together with GTP_EFTU. Profiles searched: every 'ribosomal protein' KO and every N_RULES KO
(KOfam HMMs, ko_list thresholds by score_type), and the N_RULES / EF2 Pfam 38.1 models at GA. hmmsearch over all package
proteins in $SLURM_CPUS_ON_NODE parallel processes (one protein chunk each). Writes frozen2/annot/genes_annot.parquet.
Approximation (stated): a gene's best KO is taken among the searched KOs only.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

RD = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111")
FZ = RD / "long_loci/frozen2"
PKG = RD / "ws_srvp_bins/package"
AS = Path("/groups/banfield/users/jwestrob/.config/Astra")
OUT = FZ / "annot"
sys.path.insert(0, str(RD / "ribo_audit/code"))
import bio  # noqa: E402  (read-only: N_RULES, short_rp)


def log(*a):
    print(time.strftime("%H:%M:%S", time.gmtime()), *a, flush=True)


def main():
    t0 = time.time()
    T = int(os.environ.get("SLURM_CPUS_ON_NODE", "4"))
    OUT.mkdir(parents=True, exist_ok=True)
    W = OUT / "work"
    W.mkdir(exist_ok=True)
    kl = pd.read_csv(AS / "ko_list", sep="\t")
    kl["threshold"] = pd.to_numeric(kl.threshold, errors="coerce")
    nk = set().union(*[k for _, k, _ in bio.N_RULES])
    sel = kl[kl.definition.str.contains("ribosomal protein", na=False) | kl.knum.isin(nk)].copy()
    sel = sel[[(AS / "KOFAM" / f"{k}.hmm").exists() for k in sel.knum]]
    with open(W / "ko_sel.hmm", "w") as fh:
        for k in sel.knum:
            fh.write((AS / "KOFAM" / f"{k}.hmm").read_text())
    pf_names = sorted(set().union(*[p for _, _, p in bio.N_RULES]) | {"EFG_C", "EFG_IV", "GTP_EFTU"})
    got = []
    with open(W / "pfam_sel.hmm", "w") as fh:
        for nm in pf_names:
            r = subprocess.run(["hmmfetch", str(AS / "PFAM/Pfam-A.hmm"), nm], capture_output=True, text=True)
            if r.returncode == 0 and r.stdout.startswith("HMMER3"):
                fh.write(r.stdout)
                got.append(nm)
    log("KOs", len(sel), "of which N_RULES", int(sel.knum.isin(nk).sum()), "Pfam models", got, "missing", sorted(set(pf_names) - set(got)))
    # chunks
    chunks = [open(W / f"chunk{i}.faa", "w") for i in range(T)]
    i = -1
    with open(PKG / "universe/proteins.faa") as fh:
        for line in fh:
            if line.startswith(">"):
                i += 1
            chunks[i % T].write(line)
    for c in chunks:
        c.close()
    log("chunks", T, "proteins", i + 1, f"{time.time()-t0:.0f}s")
    ts = time.time()
    procs = []
    for j in range(T):
        procs.append(subprocess.Popen(["hmmsearch", "--cpu", "1", "--noali", "-o", "/dev/null", "-E", "1e-3",
                                       "--tblout", str(W / f"ko{j}.tbl"), str(W / "ko_sel.hmm"), str(W / f"chunk{j}.faa")]))
        procs.append(subprocess.Popen(["hmmsearch", "--cpu", "1", "--noali", "-o", "/dev/null", "--cut_ga",
                                       "--tblout", str(W / f"pf{j}.tbl"), str(W / "pfam_sel.hmm"), str(W / f"chunk{j}.faa")]))
        while sum(p.poll() is None for p in procs) >= T - 1:
            time.sleep(5)
    rc = [p.wait() for p in procs]
    assert all(x == 0 for x in rc), rc
    hmm_s = time.time() - ts
    log("hmmsearch done", f"{hmm_s:.0f}s")

    def read_tbl(pattern):
        rows = []
        for j in range(T):
            for line in open(W / pattern.format(j=j)):
                if line.startswith("#"):
                    continue
                f = line.split()
                rows.append((int(f[0]), f[2], f[3], float(f[5]), float(f[8])))
        return pd.DataFrame(rows, columns=["gid", "qname", "qacc", "full", "dom"])

    ko = read_tbl("ko{j}.tbl")
    ko["knum"] = ko.qname
    ko = ko.merge(sel[["knum", "threshold", "score_type", "definition"]], on="knum", how="left")
    ok = np.where(ko.score_type == "domain", ko.dom >= ko.threshold, ko.full >= ko.threshold)
    ko = ko[ok & ko.threshold.notna()].sort_values("full", ascending=False).drop_duplicates("gid")
    pf = read_tbl("pf{j}.tbl")
    n = int(pd.read_parquet(PKG / "universe/genes.parquet", columns=["gid"]).gid.max()) + 1
    marked = np.zeros(n, bool)
    marked[pd.read_parquet(PKG / "ribosomal/marked.parquet").gid.to_numpy()] = True
    g = pd.DataFrame({"gid": np.arange(n)})
    g["ko"] = pd.Series(ko.knum.to_numpy(), index=ko.gid.to_numpy()).reindex(g.gid).to_numpy()
    g["ko_name"] = pd.Series(ko.definition.to_numpy(), index=ko.gid.to_numpy()).reindex(g.gid).to_numpy()
    R = marked | g.ko_name.fillna("").str.contains("ribosomal protein").to_numpy()
    lit = np.array([""] * n, dtype=object)
    mk = pd.read_parquet(PKG / "ribosomal/marked.parquet").set_index("gid").marker
    top = mk.reindex(g.gid).to_numpy()
    for i_ in np.flatnonzero(R):
        lit[i_] = bio.short_rp(g.ko_name.iat[i_], top[i_])
    lab = pd.Series([""] * n, dtype=object)
    for name, kos, pfs in reversed(bio.N_RULES):
        hit = np.zeros(n, bool)
        if kos:
            hit |= g.ko.isin(kos).to_numpy()
        if pfs:
            hit[pf[pf.qname.isin(pfs)].gid.unique()] = True
        lab[hit] = name
    efg = set(pf[pf.qname.isin(["EFG_C", "EFG_IV"])].gid.unique())
    gtp = set(pf[pf.qname == "GTP_EFTU"].gid.unique())
    lab0 = lab.to_numpy()
    for x in efg & gtp:
        if lab0[x] == "":
            lab0[x] = "EF2"
    N = (lab0 != "") & ~R
    lit[N] = lab0[N]
    g["cls"] = np.where(R, "R", np.where(N, "N", "X"))
    g["lit"] = lit
    g.to_parquet(OUT / "genes_annot.parquet", index=False)
    res = dict(kos_searched=int(len(sel)), n_rules_kos=int(sel.knum.isin(nk).sum()), pfam_models=got,
               pfam_missing=sorted(set(pf_names) - set(got)), ko_assigned=int(g.ko.notna().sum()),
               cls=g.cls.value_counts().to_dict(), marked=int(marked.sum()), R_by_ko_only=int((R & ~marked).sum()),
               hmmsearch_seconds=round(hmm_s, 1), processes=T, seconds=round(time.time() - t0, 1))
    (OUT / "annot.json").write_text(json.dumps(res, indent=1) + "\n")
    log(res)
    for f in W.glob("chunk*.faa"):
        f.unlink()


if __name__ == "__main__":
    main()
