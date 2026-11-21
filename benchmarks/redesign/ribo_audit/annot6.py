#!/usr/bin/env python
"""ribo_audit part 6: the part-3 / part-4 restricted annotation through Aksha (Jacob's standing rule), thresholds applied here.

Profiles (identical files to part 4): work4/hmm/ko_sel.hmm (226 KOs: 201 'ribosomal protein' + 25 N_RULES), work4/hmm/pfam_sel.hmm
(20 N_RULES / EF2 Pfam models), work4/hmm/extra.hmm (16 bacterial-member KOs), Pfam-A.hmm (38.1) on genes within 40 of a run.
Aksha runs WITHOUT cutoffs (--evalue 1000 --domE 1000); this script applies:
  KO   : ko_list threshold by score_type (full: sequence bitscore; domain: best domain bitscore), AND the E-value cut of the
         developed procedure (hmmsearch -E 1e-3 base / -E 1e-5 supplement over 24 round-robin chunks), emulated exactly from
         the profile's STATS LOCAL FORWARD (tau, lambda): E = (N_total / 24) * exp(-lambda * (bitscore - tau));
         best KO per gene by sequence bitscore (base), best KO per gene by score / threshold (supplement)
  Pfam : GA from the HMM file: sequence passes if bitscore >= GA1; a domain is kept if its sequence passes and dom_bitscore >= GA2
Outputs (same schema as part 4): genes_annot.parquet, extra_ko.parquet, window_pfam.parquet, window_pfam_domains.parquet,
window_gids.npy.

usage:
  annot6.py run  --faa PROTEINS.faa --genes genes.parquet --contigs contigs.parquet --marked marked.parquet --runs runs.parquet
                 --out OUTDIR [--threads N] [--gids GIDS.npy]
     (FASTA headers = gid; --gids restricts the search to a gid subset: used for the equivalence test)
  annot6.py equiv OUTDIR        compare OUTDIR with the stored SR-VP annotation (frozen2 genes_annot, work3 extra / window Pfam)
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

RD = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111")
AD = RD / "ribo_audit"
AS = Path(os.path.expanduser("~/.config/Astra"))
AKSHA = os.environ.get("AKSHA", "/home/jwestrob/.local/bin/aksha")
HMM = AD / "work4/hmm"
sys.path.insert(0, str(AD / "code"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(AD / "numba_cache"))
sys.dont_write_bytecode = True
import bio  # noqa: E402  (read-only: N_RULES, short_rp)

FLANK = 40
EXTRA = ["K02601", "K03043", "K03046", "K02358", "K02355", "K00939", "K01265", "K02518", "K03073", "K01883", "K03685",
         "K03218", "K16786", "K16787", "K16785", "K06173"]


def log(*a):
    print(time.strftime("%H:%M:%S", time.gmtime()), *a, flush=True)


def ga_table(hmm_path):
    """NAME -> (GA1, GA2) from an HMM file (Pfam names; GA lines)."""
    out, name = {}, None
    with open(hmm_path) as fh:
        for line in fh:
            if line.startswith("NAME "):
                name = line.split()[1]
            elif line.startswith("GA ") and name:
                v = re.findall(r"[-\d.]+", line[3:])
                out[name] = (float(v[0]), float(v[1]))
    return out


def aksha(hmm, faa, outdir, threads):
    outdir.mkdir(parents=True, exist_ok=True)
    f = outdir / "user_hmms_hits_df.tsv"
    if os.environ.get("AKSHA_REUSE") == "1" and f.exists():
        df = pd.read_csv(f, sep="\t")
        df["gid"] = df.sequence_id.astype(np.int64)
        log("reused", f, "rows", len(df))
        return df
    cmd = [AKSHA, "search", "--hmm_in", str(hmm), "--prot_in", str(faa), "--outdir", str(outdir), "--evalue", "1000",
           "--domE", "1000", "--threads", str(threads)]
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    f = outdir / "user_hmms_hits_df.tsv"
    df = pd.read_csv(f, sep="\t")
    log("aksha", hmm.name, "rows", len(df), f"{time.time()-t0:.0f}s")
    df["gid"] = df.sequence_id.astype(np.int64)
    return df


def fwd_stats(hmm_path):
    """NAME -> (tau, lambda) of STATS LOCAL FORWARD."""
    out, name = {}, None
    with open(hmm_path) as fh:
        for line in fh:
            if line.startswith("NAME "):
                name = line.split()[1]
            elif line.startswith("STATS LOCAL FORWARD") and name:
                f = line.split()
                out[name] = (float(f[3]), float(f[4]))
    return out


def e_emul(seq, hmm_path, z):
    st = fwd_stats(hmm_path)
    tau = np.array([st[h][0] for h in seq.hmm_name])
    lam = np.array([st[h][1] for h in seq.hmm_name])
    return z * np.exp(-lam * (seq.full.to_numpy() - tau))


def per_seq(df):
    """sequence-level rows: full bitscore and best domain bitscore per (gid, hmm)."""
    return df.groupby(["gid", "hmm_name"]).agg(full=("bitscore", "max"), dom=("dom_bitscore", "max")).reset_index()


def cmd_run(a):
    t0 = time.time()
    out = Path(a.out)
    (out / "aksha").mkdir(parents=True, exist_ok=True)
    genes = pd.read_parquet(a.genes, columns=["gid", "contig_code"]).sort_values("gid")
    n = int(genes.gid.max()) + 1
    c = pd.read_parquet(a.contigs).set_index("contig_code")
    runs = pd.read_parquet(a.runs)
    if "m16" in runs.columns and "n_marked" not in runs.columns:
        runs = runs[(runs.run_set == "g2") & runs.m16]
    else:
        runs = runs[(runs.run_set == "g2") & (runs.n_marked >= 16)]
    sub = np.ones(n, bool)
    if a.gids:
        sub[:] = False
        sub[np.load(a.gids)] = True
    win = np.zeros(n, bool)
    for r in runs.itertuples():
        f = int(c.first_gid.at[r.contig_code])
        l_ = f + int(c.n_genes.at[r.contig_code]) - 1
        win[max(f, r.gid_start - FLANK):min(l_, r.gid_end + FLANK) + 1] = True
    win &= sub
    np.save(out / "window_gids.npy", np.flatnonzero(win))
    fa, fw = open(out / "aksha/all.faa", "w"), open(out / "aksha/win.faa", "w")
    h, buf, k, j = None, [], 0, 0

    def flush():
        nonlocal k, j
        if h is not None and sub[h]:
            s = "".join(buf).rstrip("*")
            fa.write(f">{h}\n{s}\n")
            k += 1
            if win[h]:
                fw.write(f">{h}\n{s}\n")
                j += 1
    with open(a.faa) as fh:
        for line in fh:
            if line.startswith(">"):
                flush()
                h, buf = int(line[1:].split()[0]), []
            else:
                buf.append(line.strip())
    flush()
    fa.close()
    fw.close()
    log("proteins", k, "window proteins", j)
    T = a.threads
    kl = pd.read_csv(AS / "ko_list", sep="\t")
    kl["threshold"] = pd.to_numeric(kl.threshold, errors="coerce")
    kl = kl.set_index("knum")
    # base KOs
    ko = per_seq(aksha(HMM / "ko_sel.hmm", out / "aksha/all.faa", out / "aksha/ko", T))
    th = kl.threshold.reindex(ko.hmm_name).to_numpy()
    st = kl.score_type.reindex(ko.hmm_name).to_numpy()
    ztot = (a.z_total or k) / 24.0
    ok = np.isfinite(th) & np.where(st == "domain", ko.dom.to_numpy() >= th, ko.full.to_numpy() >= th)
    ok &= e_emul(ko, HMM / "ko_sel.hmm", ztot) <= 1e-3
    ko = ko[ok].sort_values("full", ascending=False).drop_duplicates("gid")
    # base Pfam
    ga = ga_table(HMM / "pfam_sel.hmm")
    pf = per_seq(aksha(HMM / "pfam_sel.hmm", out / "aksha/all.faa", out / "aksha/pf", T))
    pf = pf[pf.full.to_numpy() >= np.array([ga[x][0] for x in pf.hmm_name])]
    mkdf = pd.read_parquet(a.marked)
    marked = np.zeros(n, bool)
    marked[mkdf.gid.to_numpy()] = True
    g = pd.DataFrame({"gid": np.arange(n)})
    g["ko"] = pd.Series(ko.hmm_name.to_numpy(), index=ko.gid.to_numpy()).reindex(g.gid).to_numpy()
    g["ko_name"] = kl.definition.reindex(g.ko.fillna("")).to_numpy()
    g.loc[g.ko.isna(), "ko_name"] = None
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
            hit[pf[pf.hmm_name.isin(pfs)].gid.unique()] = True
        lab[hit] = name
    efg = set(pf[pf.hmm_name.isin(["EFG_C", "EFG_IV"])].gid.unique())
    gtp = set(pf[pf.hmm_name == "GTP_EFTU"].gid.unique())
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
    ex = per_seq(aksha(HMM / "extra.hmm", out / "aksha/all.faa", out / "aksha/extra", T))
    th = kl.threshold.reindex(ex.hmm_name).to_numpy()
    st = kl.score_type.reindex(ex.hmm_name).to_numpy()
    sc = np.where(st == "full", ex.full.to_numpy(), ex.dom.to_numpy())
    keep = np.isfinite(th) & (sc >= th) & (e_emul(ex, HMM / "extra.hmm", ztot) <= 1e-5)
    ex = pd.DataFrame(dict(gid=ex.gid.to_numpy()[keep], ko=ex.hmm_name.to_numpy()[keep], score=sc[keep], ratio=sc[keep] / th[keep]))
    ex = ex.sort_values("ratio", ascending=False).drop_duplicates("gid")
    ex.to_parquet(out / "extra_ko.parquet", index=False)
    # supplement 2: Pfam-A on windows
    gaA = ga_table(AS / "PFAM/Pfam-A.hmm")
    pa = aksha(AS / "PFAM/Pfam-A.hmm", out / "aksha/win.faa", out / "aksha/pfamA", T)
    g1 = np.array([gaA[x][0] for x in pa.hmm_name])
    g2 = np.array([gaA[x][1] for x in pa.hmm_name])
    pa = pa[(pa.bitscore.to_numpy() >= g1) & (pa.dom_bitscore.to_numpy() >= g2)]
    pa = pd.DataFrame(dict(gid=pa.gid.to_numpy(), name=pa.hmm_name.to_numpy(), dom_score=pa.dom_bitscore.to_numpy(),
                           env_from=pa.env_from.to_numpy().astype(np.int64)))
    pa.to_parquet(out / "window_pfam_domains.parquet", index=False)
    arch = pa.sort_values(["gid", "env_from"]).groupby("gid").name.agg(",".join)
    tp = pa.sort_values("dom_score", ascending=False).drop_duplicates("gid").set_index("gid").name
    pd.DataFrame({"gid": arch.index, "arch": arch.to_numpy(), "top": tp.reindex(arch.index).to_numpy()}).to_parquet(
        out / "window_pfam.parquet", index=False)
    res = dict(proteins=k, z_total=int(a.z_total or k), window_proteins=j, ko_assigned=int(g.ko.notna().sum()), cls=g.cls.value_counts().to_dict(),
               extra_hits=int(len(ex)), window_with_pfam=int(len(arch)), threads=T, seconds=round(time.time() - t0, 1))
    (out / "annot6.json").write_text(json.dumps(res, indent=1) + "\n")
    log(res)


def cmd_equiv(outdir):
    out = Path(outdir)
    sub = np.load(out / "sample_gids.npy")
    win = np.load(out / "window_gids.npy")
    new = pd.read_parquet(out / "genes_annot.parquet").set_index("gid").loc[sub]
    old = pd.read_parquet(RD / "long_loci/frozen2/annot/genes_annot.parquet").set_index("gid").loc[sub]
    res = {"genes": int(len(sub)), "window_genes": int(len(win))}
    for col in ("ko", "cls", "lit"):
        a, b = new[col].fillna(""), old[col].fillna("")
        d = a != b
        res[f"base_{col}_diff"] = int(d.sum())
        if d.any():
            res[f"base_{col}_examples"] = pd.DataFrame(dict(new=a[d], old=b[d])).head(10).reset_index().to_dict("records")
    en = pd.read_parquet(out / "extra_ko.parquet").set_index("gid")
    eo = pd.read_parquet(AD / "work3/annot/extra_ko.parquet")
    eo = eo[eo.gid.isin(sub)].set_index("gid")
    both = en.index.intersection(eo.index)
    res["extra_new"], res["extra_old"], res["extra_common"] = int(len(en)), int(len(eo)), int(len(both))
    res["extra_ko_diff"] = int((en.loc[both].ko != eo.loc[both].ko).sum())
    res["extra_score_maxabsdiff"] = float((en.loc[both].score - eo.loc[both].score).abs().max()) if len(both) else 0.0
    res["extra_only_new"] = en.index.difference(eo.index).tolist()[:20]
    res["extra_only_old"] = eo.index.difference(en.index).tolist()[:20]
    wn = pd.read_parquet(out / "window_pfam.parquet").set_index("gid")
    wo = pd.read_parquet(AD / "work3/annot/window_pfam.parquet")
    wo = wo[wo.gid.isin(win)].set_index("gid")
    bw = wn.index.intersection(wo.index)
    res["winpfam_new"], res["winpfam_old"], res["winpfam_common"] = int(len(wn)), int(len(wo)), int(len(bw))
    res["winpfam_arch_diff"] = int((wn.loc[bw].arch != wo.loc[bw].arch).sum())
    res["winpfam_top_diff"] = int((wn.loc[bw].top != wo.loc[bw].top).sum())
    dd = wn.loc[bw][wn.loc[bw].arch != wo.loc[bw].arch]
    res["winpfam_arch_examples"] = [dict(gid=int(i), new=wn.at[i, "arch"], old=wo.at[i, "arch"]) for i in dd.index[:10]]
    res["winpfam_only_new"] = wn.index.difference(wo.index).tolist()[:20]
    res["winpfam_only_old"] = wo.index.difference(wn.index).tolist()[:20]
    dn = pd.read_parquet(out / "window_pfam_domains.parquet")
    do = pd.read_parquet(AD / "work3/annot/window_pfam_domains.parquet")
    do = do[do.gid.isin(win)]
    m = dn.merge(do, on=["gid", "name", "env_from"], how="outer", suffixes=("_new", "_old"), indicator=True)
    res["domains_new"], res["domains_old"] = int(len(dn)), int(len(do))
    res["domains_match"] = int((m._merge == "both").sum())
    res["domains_score_maxabsdiff"] = float((m.dom_score_new - m.dom_score_old).abs().max())
    (out / "equiv6.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd")
    p.add_argument("outdir", nargs="?")
    for k_ in ("faa", "genes", "contigs", "marked", "runs", "out", "gids"):
        p.add_argument("--" + k_)
    p.add_argument("--z_total", type=int, default=0, help="protein count of the whole dataset (default: proteins searched)")
    p.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_ON_NODE", "4")))
    a = p.parse_args()
    if a.cmd == "run":
        cmd_run(a)
    else:
        cmd_equiv(a.outdir)
