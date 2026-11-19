#!/usr/bin/env python
"""ribo_audit part 4: the part-3 restricted annotation on the lut test set and on DPANN (predeclared4.json).

Per dataset, exactly the part-3 procedures:
  base (frozen2 annot2.py): 226 KOfam profiles (201 'ribosomal protein' + 25 N_RULES; ko_list thresholds, -E 1e-3, best KO
       among the searched ones) and the 20 N_RULES / EF2 Pfam models (--cut_ga) over every protein -> genes_annot.parquet
  supplement 1 (annot3.py): 16 bacterial-member KOs, -E 1e-5, ko_list thresholds, best ratio per gene -> extra_ko.parquet
  supplement 2 (annot3.py): Pfam-A 38.1 --cut_ga on genes within 40 of a run -> window_pfam(_domains).parquet
All hmmsearch work of both datasets goes through one pool of $SLURM_CPUS_ON_NODE single-CPU processes (round-robin protein
chunks). Writes $RD/ribo_audit/work4/<dataset>/annot/. Packages are read only.
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
AS = Path("/groups/banfield/users/jwestrob/.config/Astra")
HS = "/shared/software/bin/hmmsearch"
HF = "/shared/software/bin/hmmfetch"
W4 = RD / "ribo_audit/work4"
sys.path.insert(0, str(RD / "ribo_audit/code"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(RD / "ribo_audit/numba_cache"))
sys.dont_write_bytecode = True
import bio  # noqa: E402  (read-only: N_RULES, short_rp)

EXTRA = ["K02601", "K03043", "K03046", "K02358", "K02355", "K00939", "K01265", "K02518", "K03073", "K01883", "K03685",
         "K03218", "K16786", "K16787", "K16785", "K06173"]
FLANK = 40
DS = {
    "lut": dict(genes=RD / "slice_lut/testset/package_k8/universe/genes.parquet",
                contigs=RD / "slice_lut/testset/package_k8/universe/contigs.parquet",
                marked=RD / "slice_lut/testset/package_k8/gold/ribosomal_marked.parquet",
                runs=RD / "slice_lut/testset/package_k8/gold/ribosomal_runs.parquet",
                faa=RD / "slice_lut/testset/package_k8/proteins_gid.faa"),
    "dpann": dict(genes=RD / "ws_dpann/package/universe/genes.parquet",
                  contigs=RD / "ws_dpann/package/universe/contigs.parquet",
                  marked=RD / "ribosomal/gold/marked.parquet",
                  runs=RD / "ribosomal/gold/runs.parquet",
                  db=RD / "ws_dpann/input/sharur.duckdb"),
}


def log(*a):
    print(time.strftime("%H:%M:%S", time.gmtime()), *a, flush=True)


def proteins(ds, cfg):
    """Yield (gid, sequence) in gid order."""
    if ds == "lut":
        h, buf = None, []
        with open(cfg["faa"]) as fh:
            for line in fh:
                if line.startswith(">"):
                    if h is not None:
                        yield h, "".join(buf).rstrip("*")
                    h, buf = int(line[1:].split()[0]), []
                else:
                    buf.append(line.strip())
        yield h, "".join(buf).rstrip("*")
    else:
        import duckdb
        g = pd.read_parquet(cfg["genes"], columns=["gid", "protein_id"]).sort_values("gid")
        con = duckdb.connect(str(cfg["db"]), read_only=True)
        con.execute("SET threads=4")
        p = con.execute("SELECT protein_id, sequence, partial FROM proteins").df()
        con.close()
        p = g.merge(p, on="protein_id", how="left")
        assert p.sequence.notna().all(), "missing sequences"
        p[["gid", "partial"]].to_parquet(W4 / ds / "annot/flags.parquet", index=False)
        for gid, s in zip(p.gid.to_numpy(), p.sequence.to_numpy()):
            yield int(gid), s.rstrip("*")


def prepare(ds, T):
    cfg = DS[ds]
    out = W4 / ds / "annot"
    (out / "work").mkdir(parents=True, exist_ok=True)
    genes = pd.read_parquet(cfg["genes"], columns=["gid", "contig_code"]).sort_values("gid")
    n = len(genes)
    c = pd.read_parquet(cfg["contigs"]).set_index("contig_code")
    runs = pd.read_parquet(cfg["runs"])
    runs = runs[(runs.run_set == "g2") & (runs.n_marked >= 16)]
    win = np.zeros(n, bool)
    for r in runs.itertuples():
        f = int(c.first_gid.at[r.contig_code])
        l_ = f + int(c.n_genes.at[r.contig_code]) - 1
        win[max(f, r.gid_start - FLANK):min(l_, r.gid_end + FLANK) + 1] = True
    np.save(out / "window_gids.npy", np.flatnonzero(win))
    fa = [open(out / f"work/all_{i}.faa", "w") for i in range(T)]
    fw = [open(out / f"work/win_{i}.faa", "w") for i in range(T)]
    k = j = 0
    for gid, s in proteins(ds, cfg):
        fa[k % T].write(f">{gid}\n{s}\n")
        k += 1
        if win[gid]:
            fw[j % T].write(f">{gid}\n{s}\n")
            j += 1
    for f in fa + fw:
        f.close()
    log(ds, "runs", len(runs), "proteins", k, "of", n, "window proteins", j)
    assert k == n
    return n


def main():
    t0 = time.time()
    T = int(os.environ["SLURM_CPUS_ON_NODE"])
    hdir = W4 / "hmm"
    hdir.mkdir(parents=True, exist_ok=True)
    kl = pd.read_csv(AS / "ko_list", sep="\t")
    kl["threshold"] = pd.to_numeric(kl.threshold, errors="coerce")
    nk = set().union(*[k for _, k, _ in bio.N_RULES])
    sel = kl[kl.definition.str.contains("ribosomal protein", na=False) | kl.knum.isin(nk)].copy()
    sel = sel[[(AS / "KOFAM" / f"{k}.hmm").exists() for k in sel.knum]]
    with open(hdir / "ko_sel.hmm", "w") as fh:
        for k in sel.knum:
            fh.write((AS / "KOFAM" / f"{k}.hmm").read_text())
    pf_names = sorted(set().union(*[p for _, _, p in bio.N_RULES]) | {"EFG_C", "EFG_IV", "GTP_EFTU"})
    with open(hdir / "pfam_sel.hmm", "w") as fh:
        for nm in pf_names:
            r = subprocess.run([HF, str(AS / "PFAM/Pfam-A.hmm"), nm], capture_output=True, text=True)
            assert r.returncode == 0 and r.stdout.startswith("HMMER3"), nm
            fh.write(r.stdout)
    with open(hdir / "extra.hmm", "w") as fh:
        for k in EXTRA:
            fh.write((AS / "KOFAM" / f"{k}.hmm").read_text())
    log("KOs", len(sel), "N_RULES KOs", int(sel.knum.isin(nk).sum()), "Pfam models", len(pf_names), "extra", len(EXTRA))
    sizes = {ds: prepare(ds, T) for ds in DS}
    log("prepared", sizes, f"{time.time()-t0:.0f}s")
    jobs = []
    for ds in DS:
        w = W4 / ds / "annot/work"
        for i in range(T):
            jobs.append([HS, "--cpu", "1", "--noali", "-o", "/dev/null", "-E", "1e-3", "--tblout", str(w / f"ko_{i}.tbl"),
                         str(hdir / "ko_sel.hmm"), str(w / f"all_{i}.faa")])
            jobs.append([HS, "--cpu", "1", "--noali", "-o", "/dev/null", "--cut_ga", "--tblout", str(w / f"pf_{i}.tbl"),
                         str(hdir / "pfam_sel.hmm"), str(w / f"all_{i}.faa")])
            jobs.append([HS, "--cpu", "1", "--noali", "-o", "/dev/null", "-E", "1e-5", "--tblout", str(w / f"extra_{i}.tbl"),
                         str(hdir / "extra.hmm"), str(w / f"all_{i}.faa")])
            jobs.append([HS, "--cpu", "1", "--noali", "-o", "/dev/null", "--cut_ga", "--domtblout", str(w / f"pfamA_{i}.tbl"),
                         str(AS / "PFAM/Pfam-A.hmm"), str(w / f"win_{i}.faa")])
    # longest first: Pfam-A windows, then the 226-KO searches
    jobs.sort(key=lambda c: (0 if "Pfam-A.hmm" in c[-2] else 1 if "ko_sel" in c[-2] else 2))
    running, done = [], 0
    ts = time.time()
    while jobs or running:
        while jobs and len(running) < T:
            running.append(subprocess.Popen(jobs.pop(0)))
        time.sleep(5)
        still = []
        for p in running:
            rc = p.poll()
            if rc is None:
                still.append(p)
            else:
                assert rc == 0, f"hmmsearch failed: {p.args}"
                done += 1
        running = still
    log("hmmsearch jobs", done, f"{time.time()-ts:.0f}s")
    summary = {}
    for ds, n in sizes.items():
        out = W4 / ds / "annot"
        w = out / "work"

        def tbl(pattern):
            rows = []
            for i in range(T):
                for line in open(w / pattern.format(i=i)):
                    if line.startswith("#"):
                        continue
                    f = line.split()
                    rows.append((int(f[0]), f[2], float(f[5]), float(f[8])))
            return pd.DataFrame(rows, columns=["gid", "qname", "full", "dom"])
        ko = tbl("ko_{i}.tbl").rename(columns={"qname": "knum"})
        ko = ko.merge(sel[["knum", "threshold", "score_type", "definition"]], on="knum", how="left")
        ok = np.where(ko.score_type == "domain", ko.dom >= ko.threshold, ko.full >= ko.threshold)
        ko = ko[ok & ko.threshold.notna()].sort_values("full", ascending=False).drop_duplicates("gid")
        pf = tbl("pf_{i}.tbl")
        mkdf = pd.read_parquet(DS[ds]["marked"])
        marked = np.zeros(n, bool)
        marked[mkdf.gid.to_numpy()] = True
        g = pd.DataFrame({"gid": np.arange(n)})
        g["ko"] = pd.Series(ko.knum.to_numpy(), index=ko.gid.to_numpy()).reindex(g.gid).to_numpy()
        g["ko_name"] = pd.Series(ko.definition.to_numpy(), index=ko.gid.to_numpy()).reindex(g.gid).to_numpy()
        R = marked | g.ko_name.fillna("").str.contains("ribosomal protein").to_numpy()
        lit = np.array([""] * n, dtype=object)
        top = mkdf.set_index("gid").marker.reindex(g.gid).to_numpy()
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
        g.to_parquet(out / "genes_annot.parquet", index=False)
        # supplement 1
        kx = kl[kl.knum.isin(EXTRA)].set_index("knum")
        ex = tbl("extra_{i}.tbl")
        th = kx.threshold.reindex(ex.qname).to_numpy()
        st = kx.score_type.reindex(ex.qname).to_numpy()
        sc = np.where(st == "full", ex.full.to_numpy(), ex.dom.to_numpy())
        keep = np.isfinite(th) & (sc >= th)
        ex = pd.DataFrame(dict(gid=ex.gid.to_numpy()[keep], ko=ex.qname.to_numpy()[keep], score=sc[keep], ratio=sc[keep] / th[keep]))
        ex = ex.sort_values("ratio", ascending=False).drop_duplicates("gid")
        ex.to_parquet(out / "extra_ko.parquet", index=False)
        # supplement 2
        rows = []
        for i in range(T):
            for line in open(w / f"pfamA_{i}.tbl"):
                if line.startswith("#"):
                    continue
                f = line.split()
                rows.append((int(f[0]), f[3], float(f[13]), int(f[19])))
        pa = pd.DataFrame(rows, columns=["gid", "name", "dom_score", "env_from"])
        pa.to_parquet(out / "window_pfam_domains.parquet", index=False)
        arch = pa.sort_values(["gid", "env_from"]).groupby("gid").name.agg(",".join)
        tp = pa.sort_values("dom_score", ascending=False).drop_duplicates("gid").set_index("gid").name
        pd.DataFrame({"gid": arch.index, "arch": arch.to_numpy(), "top": tp.reindex(arch.index).to_numpy()}).to_parquet(
            out / "window_pfam.parquet", index=False)
        summary[ds] = dict(genes=n, ko_assigned=int(g.ko.notna().sum()), cls=g.cls.value_counts().to_dict(),
                           marked=int(marked.sum()), R_by_ko_only=int((R & ~marked).sum()), extra_hits=int(len(ex)),
                           extra_by_ko=ex.ko.value_counts().to_dict(), window_genes=int(len(np.load(out / "window_gids.npy"))),
                           window_with_pfam=int(len(arch)))
        log(ds, summary[ds])
        for p in w.glob("*.faa"):
            p.unlink()
    summary["seconds"] = round(time.time() - t0, 1)
    summary["processes"] = T
    (W4 / "annot4.json").write_text(json.dumps(summary, indent=1) + "\n")
    log("done", f"{time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
