#!/usr/bin/env python
"""ribo_audit part 7: ribosomal marking and runs for a package that lacks them, by the package rule, through Aksha.

Rule (ws_srvp_bins/package/code/03_build_package.py st_ribosomal = slice_lut package_k8/code/15_ribosomal_truth.py):
  marked gene = a domain row of hmmsearch --cut_ga --domtblout for any of the 121 Pfam 38.1 ^ribosom(al)?_ models
  (file slice_lut/testset/work/ribo.hmm, sha256 b2ce8369...6e49), i.e. sequence score >= GA1 and a domain score >= GA2;
  marker = model with the highest full-sequence score; markers = sorted set of hit models;
  runs: per contig, maximal runs of marked genes with <= gap unmarked genes inside (g0, g2); runs with >= 8 marked kept;
  m16 = >= 16 marked.
Here the search runs through Aksha (--hmm_in ribo.hmm --cut_ga); GA1 / GA2 are re-applied from the HMM file to the rows
Aksha writes, so the result is the same whether Aksha reports every domain of a hit or only those >= GA2.
usage:
  mark7.py run   --pkg PKG --faa FAA --out OUTDIR --threads N [--gids GIDS.npy] [--ready READY.json]
  mark7.py equiv OUTDIR REF_MARKED.parquet REF_RUNS.parquet   (compare a sample against a package's stored tables)
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

RIBO_HMM = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/slice_lut/testset/work/ribo.hmm")
AKSHA = os.environ.get("AKSHA", "/home/jwestrob/.local/bin/aksha")


def log(*a):
    print(time.strftime("%H:%M:%S", time.gmtime()), *a, flush=True)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def ga_table(path):
    out, name = {}, None
    for line in open(path):
        if line.startswith("NAME "):
            name = line.split()[1]
        elif line.startswith("GA ") and name:
            v = re.findall(r"[-\d.]+", line[3:])
            out[name] = (float(v[0]), float(v[1]))
    return out


def cmd_run(a):
    t0 = time.time()
    out = Path(a.out)
    (out / "aksha").mkdir(parents=True, exist_ok=True)
    (out / "ribosomal").mkdir(parents=True, exist_ok=True)
    P = Path(a.pkg)
    faa = Path(a.faa)
    if a.gids:
        keep = set(np.load(a.gids).tolist())
        sub = out / "aksha/sample.faa"
        with open(faa) as fh, open(sub, "w") as fo:
            w = False
            for line in fh:
                if line.startswith(">"):
                    w = int(line[1:].split()[0]) in keep
                if w:
                    fo.write(line)
        faa = sub
    cmd = [AKSHA, "search", "--hmm_in", str(RIBO_HMM), "--prot_in", str(faa), "--outdir", str(out / "aksha/ribo"),
           "--cut_ga", "--threads", str(a.threads)]
    log("aksha", " ".join(cmd))
    ts = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    aksha_s = time.time() - ts
    h = pd.read_csv(out / "aksha/ribo/user_hmms_hits_df.tsv", sep="\t")
    h["gid"] = h.sequence_id.astype(np.int64)
    ga = ga_table(RIBO_HMM)
    g1 = np.array([ga[x][0] for x in h.hmm_name])
    g2 = np.array([ga[x][1] for x in h.hmm_name])
    n_rows = len(h)
    h = h[(h.bitscore.to_numpy() >= g1) & (h.dom_bitscore.to_numpy() >= g2)]
    assert h.hmm_name.str.lower().str.match(r"^ribosom(al)?_").all()
    log("aksha rows", n_rows, "after GA1 / GA2", len(h), f"{aksha_s:.0f}s")
    genes = pd.read_parquet(P / "universe/genes.parquet", columns=["gid", "genome_code", "contig_code", "position_index"]).sort_values("gid")
    assert np.array_equal(genes.gid.to_numpy(), np.arange(len(genes)))
    contigs = pd.read_parquet(P / "universe/contigs.parquet").set_index("contig_code")
    gm = pd.read_parquet(P / "universe/genomes.parquet").set_index("genome_code")
    top = h.sort_values("bitscore", ascending=False).drop_duplicates("gid").set_index("gid").hmm_name
    alln = h.groupby("gid").hmm_name.apply(lambda s: ";".join(sorted(set(s))))
    marked = pd.DataFrame({"gid": top.index.to_numpy(), "marker": top.to_numpy()})
    marked["markers"] = alln.reindex(marked.gid).to_numpy()
    marked = marked.sort_values("gid").reset_index(drop=True)
    marked.to_parquet(out / "ribosomal/marked.parquet", index=False)
    cc = genes.contig_code.to_numpy()
    pos = genes.position_index.to_numpy()
    gcode = genes.genome_code.to_numpy()
    frames = []
    for gap in (0, 2):
        m = np.sort(marked.gid.to_numpy())
        new = np.r_[True, (np.diff(m) > gap + 1) | (np.diff(cc[m]) != 0)]
        rid = np.cumsum(new) - 1
        last = np.r_[new[1:], True]
        rr = pd.DataFrame(dict(gid_start=m[new], gid_end=m[last], n_marked=np.bincount(rid)))
        rr = rr[rr.n_marked >= 8].reset_index(drop=True)
        rr.insert(0, "run_set", f"g{gap}")
        frames.append(rr)
    runs = pd.concat(frames, ignore_index=True)
    runs.insert(1, "run_id", range(len(runs)))
    runs["m16"] = runs.n_marked >= 16
    runs["n_genes"] = runs.gid_end - runs.gid_start + 1
    runs["contig_code"] = cc[runs.gid_start]
    runs["genome_code"] = gcode[runs.gid_start]
    runs["pos_start"] = pos[runs.gid_start]
    runs["pos_end"] = pos[runs.gid_end]
    ci = contigs.loc[runs.contig_code]
    runs["contig"] = ci.contig.to_numpy()
    runs["genome"] = ci.genome.to_numpy()
    runs["contig_first_gid"] = ci.first_gid.to_numpy()
    runs["contig_n_genes"] = ci.n_genes.to_numpy()
    runs["at_contig_edge"] = (runs.pos_start == 0) | (runs.pos_end == runs.contig_n_genes - 1)
    gg = gm.loc[runs.genome_code]
    for c in ["domain", "phylum", "class", "ord", "family", "genus", "completeness", "platform"]:
        if c in gg.columns:
            runs[c] = gg[c].to_numpy()
    runs.to_parquet(out / "ribosomal/runs.parquet", index=False)
    p = runs[(runs.run_set == "g2") & runs.m16]
    rec = dict(rule_source="ws_srvp_bins/package/code/03_build_package.py st_ribosomal (= slice_lut package_k8/code/15_ribosomal_truth.py)",
               tool="Aksha " + AKSHA + " search --hmm_in ribo.hmm --cut_ga --threads " + str(a.threads),
               hmm=str(RIBO_HMM), hmm_sha256=sha(RIBO_HMM), aksha_seconds=round(aksha_s, 1), aksha_rows=n_rows, hits=int(len(h)),
               marked_genes=int(len(marked)), models_hit=int(h.hmm_name.nunique()),
               runs={f"{rs}_m{mm}": int(((runs.run_set == rs) & (runs.n_marked >= mm)).sum()) for rs in ("g0", "g2") for mm in (8, 16)},
               g2_m16_genomes=int(p.genome_code.nunique()), genomes=int(len(gm)),
               g2_m16_by_domain=p.domain.value_counts().to_dict() if "domain" in p.columns else None,
               genomes_without_g2_m16=int(len(gm) - p.genome_code.nunique()),
               genomes_without_g2_m16_by_domain=(gm[~gm.index.isin(p.genome_code)].domain.value_counts().to_dict()
                                                 if "domain" in gm.columns else None),
               seconds=round(time.time() - t0, 1))
    (out / "ribosomal/receipt.json").write_text(json.dumps(rec, indent=1) + "\n")
    log(rec)
    if a.ready:
        files = {k: out / "ribosomal" / f for k, f in (("marked", "marked.parquet"), ("runs", "runs.parquet"))}
        ready = dict(written_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), package=str(P),
                     rule=rec["rule_source"], tool=rec["tool"],
                     files={k: dict(path=str(v), sha256=sha(v), rows=int(len(pd.read_parquet(v)))) for k, v in files.items()},
                     g2_m16_runs=int(len(p)), g2_m16_by_domain=rec["g2_m16_by_domain"],
                     standing_selection="run_set == 'g2' and m16 (n_marked >= 16)")
        Path(a.ready).write_text(json.dumps(ready, indent=1) + "\n")
        log("RUNS_READY written", a.ready)


def cmd_equiv(outdir, ref_marked, ref_runs):
    out = Path(outdir)
    sub = set(np.load(out / "sample_gids.npy").tolist())
    n = pd.read_parquet(out / "ribosomal/marked.parquet")
    o = pd.read_parquet(ref_marked)
    o = o[o.gid.isin(sub)]
    m = n.merge(o, on="gid", how="outer", suffixes=("_new", "_old"), indicator=True)
    both = m[m._merge == "both"]
    res = dict(sample_genes=len(sub), marked_new=int(len(n)), marked_old=int(len(o)), marked_both=int(len(both)),
               only_new=m[m._merge == "left_only"].gid.tolist()[:20], only_old=m[m._merge == "right_only"].gid.tolist()[:20],
               marker_diff=int((both.marker_new != both.marker_old).sum()), markers_diff=int((both.markers_new != both.markers_old).sum()))
    rn = pd.read_parquet(out / "ribosomal/runs.parquet")
    ro = pd.read_parquet(ref_runs)
    gl = pd.read_parquet(out / "ribosomal/runs.parquet")
    ro = ro[ro.gid_start.isin(sub)]
    key = ["run_set", "gid_start", "gid_end", "n_marked"]
    j = rn[key].merge(ro[key], on=key, how="outer", indicator=True)
    res.update(runs_new=int(len(rn)), runs_old=int(len(ro)), runs_identical=int((j._merge == "both").sum()))
    (out / "equiv7.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd")
    p.add_argument("args", nargs="*")
    for k in ("pkg", "faa", "out", "gids", "ready"):
        p.add_argument("--" + k)
    p.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_ON_NODE", "4")))
    a = p.parse_args()
    if a.cmd == "run":
        cmd_run(a)
    else:
        cmd_equiv(*a.args)
