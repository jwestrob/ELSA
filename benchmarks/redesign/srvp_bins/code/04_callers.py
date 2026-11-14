#!/usr/bin/env python3
"""Curated system callers on the SR-VP genome-bin package (workstream srvp_bins): DefenseFinder and TXSScan models
through MacSyFinder 2.1.4 (callers environment srvp_callers), one genome per process.

usage: 04_callers.py PKG_DIR --work DIR --threads N
  DefenseFinder  defense-finder-models 3.1.0 (the model set of the SR-VP Sharur reload, biotite install
                 srvp_research_2026-10-23/defensefinder/models), families DefenseFinder and RM in separate passes as the
                 defense-finder CLI runs them; --coverage-profile 0.4 --exchangeable-weight 1 (defense-finder defaults).
  TXSScan        TXSScan models in ~/.macsyfinder/models, 'all', MacSyFinder defaults (Sharur run_txsscan_systems.py).
  Both: --db-type gembase, one replicon per contig, --replicon-topology linear (MacSyFinder's default 'circular' lets a
  locus wrap across the two ends of a draft contig; the pilot showed an 11-gene T2SS 'single locus' spanning 2,667 genes);
  gene order = position_index. Protein ids c<contig_code>_<k>.
Writes gold/instances.parquet, gold/members.parquet (DPANN gold layout; annotation columns omitted), receipts/callers.json.
"""
import argparse, hashlib, json, os, shutil, subprocess, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np, pandas as pd

ENV = "/groups/banfield/users/jwestrob/envs/srvp_callers/bin"
# a directory holding only the symlink defense-finder-models -> srvp_research_2026-10-23/defensefinder/models/
# defense-finder-models (3.1.0); the original models dir also holds download/, which MacSyFinder scans as a family
DF_MODELS_DIR = "/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/ws_srvp_bins/models"
TXSS_MODELS_DIR = "/home/jwestrob/.macsyfinder/models"
RUNS = {
    "DefenseFinder": ["--models", "defense-finder-models/DefenseFinder", "all", "--models-dir", DF_MODELS_DIR,
                      "--coverage-profile", "0.4", "--exchangeable-weight", "1"],
    "DefenseFinder_RM": ["--models", "defense-finder-models/RM", "all", "--models-dir", DF_MODELS_DIR,
                         "--coverage-profile", "0.4", "--exchangeable-weight", "1"],
    "TXSScan": ["--models", "TXSScan", "all", "--models-dir", TXSS_MODELS_DIR],
}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(16 << 20), b""):
            h.update(b)
    return h.hexdigest()


def version(models_dir, family):
    for l in open(Path(models_dir) / family / "metadata.yml"):
        if l.startswith("vers"):
            return l.split(":", 1)[1].strip()


def parse_best(path):
    rows = []
    if not path.exists():
        return None
    for line in open(path):
        if line.startswith("#") or not line.strip():
            continue
        f = line.rstrip("\n").split("\t")
        if f[0] == "replicon":
            hdr = f; continue
        rows.append(dict(zip(hdr, f)))
    return rows


def genome_job(args):
    genome, faa, outroot, reuse = args
    env = dict(os.environ, PATH=ENV + ":" + os.environ["PATH"], OMP_NUM_THREADS="1")
    res, hits = dict(genome=genome), []
    for run, extra in RUNS.items():
        out = Path(outroot) / run / genome
        if reuse and (out / "best_solution.tsv").exists():
            best = parse_best(out / "best_solution.tsv")
            res[f"{run}_rc"] = 0; res[f"{run}_s"] = 0.0; res[f"{run}_ok"] = True
            for r in best:
                r["run"] = run; r["genome"] = genome
            hits += best
            continue
        shutil.rmtree(out, ignore_errors=True)
        t0 = time.time()
        p = subprocess.run([f"{ENV}/macsyfinder", "--db-type", "gembase", "--replicon-topology", "linear",
                            "--sequence-db", faa, "--out-dir", str(out), "-w", "1", "--mute"] + extra,
                           capture_output=True, text=True, env=env)
        best = parse_best(out / "best_solution.tsv")
        res[f"{run}_rc"] = p.returncode; res[f"{run}_s"] = round(time.time() - t0, 1)
        res[f"{run}_ok"] = best is not None
        if best is None:
            res[f"{run}_err"] = (p.stderr or "")[-400:]
            continue
        for r in best:
            r["run"] = run; r["genome"] = genome
        hits += best
        # keep only the solution tables (hmmer raw output is large)
        for sub in ("hmmer_results",):
            shutil.rmtree(out / sub, ignore_errors=True)
    return res, hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pkg"); ap.add_argument("--work", required=True); ap.add_argument("--threads", type=int, required=True)
    ap.add_argument("--genomes", type=int, default=0, help="first N genomes only (timing pilot)")
    ap.add_argument("--reuse", action="store_true", help="parse existing best_solution.tsv files instead of rerunning")
    a = ap.parse_args()
    P, W = Path(a.pkg), Path(a.work)
    (W / "faa").mkdir(parents=True, exist_ok=True); (W / "out").mkdir(exist_ok=True)
    genes = pd.read_parquet(P / "universe" / "genes.parquet", columns=["gid", "genome", "contig", "contig_code", "position_index", "start", "end"])
    seqs = {}
    with open(P / "universe" / "proteins.faa") as f:
        gid = None
        for line in f:
            if line.startswith(">"):
                gid = int(line[1:])
            else:
                seqs[gid] = line.strip()
    genes["mid"] = "c" + genes.contig_code.astype(str) + "_" + (genes.position_index + 1).astype(str)
    order = sorted(genes.genome.unique())
    if a.genomes:
        order = order[:a.genomes]
    jobs = []
    for g, d in genes[genes.genome.isin(order)].groupby("genome", sort=True):
        d = d.sort_values(["contig_code", "position_index"])
        fa = W / "faa" / f"{g}.faa"
        with open(fa, "w") as f:
            for mid, gid in zip(d.mid, d.gid):
                f.write(f">{mid}\n{seqs[gid]}\n")
        jobs.append((g, str(fa), str(W / "out"), a.reuse))
    del seqs
    t0 = time.time()
    recs, allhits = [], []
    with ProcessPoolExecutor(a.threads) as ex:
        for res, hits in ex.map(genome_job, jobs, chunksize=1):
            recs.append(res); allhits += hits
    wall = time.time() - t0
    rec = pd.DataFrame(recs)
    rec.to_csv(W / "caller_records.tsv", sep="\t", index=False)
    h = pd.DataFrame(allhits)
    mid2 = genes.set_index("mid")
    callers = {"DefenseFinder": "DefenseFinder", "DefenseFinder_RM": "DefenseFinder", "TXSScan": "TXSScan"}
    inst_rows, mem_rows = [], []
    if len(h):
        h["gid"] = h.hit_id.map(mid2.gid)
        gidx = genes.set_index("gid")
        h["caller"] = h.run.map(callers)
        for (run, genome, sys_id), d0 in h.groupby(["run", "genome", "sys_id"], sort=True):
            d = d0[d0.gid.notna()].copy()
            g = gidx.loc[d.gid.astype(int)]
            fqn = d.model_fqn.iloc[0].split("/")
            mv = (f"defense-finder-models {version(DF_MODELS_DIR, 'defense-finder-models')}" if run.startswith("DefenseFinder")
                  else f"TXSScan {version(TXSS_MODELS_DIR, 'TXSScan')}") + " / MacSyFinder 2.1.4"
            pos = sorted(g.position_index.tolist()); gids = sorted(d.gid.astype(int).unique().tolist())
            # MacSyFinder loners / out-of-cluster components carry locus_num -1; the core span uses loci members only
            ln = d.locus_num.astype(int).to_numpy() if "locus_num" in d else np.ones(len(d), int)
            cpos = sorted(g.position_index.to_numpy()[ln > 0].tolist()) or pos
            inst_rows.append(dict(
                caller=callers[run], source_table=f"macsyfinder_best_solution:{run}", system_id=f"{genome}:{sys_id}",
                genome_db=genome, contig_db=d.replicon.iloc[0], system_type=fqn[-2] if run.startswith("DefenseFinder") else fqn[-1],
                system_subtype=fqn[-1] if run.startswith("DefenseFinder") else "/".join(fqn[1:]),
                status=d.sys_wholeness.iloc[0] if "sys_wholeness" in d else None, extra=mv,
                genes_count=int(len(d0)), n_members_listed=int(len(d0)), n_members_resolved=int(len(gids)),
                n_contigs=int(g.contig_code.nunique()), genome=genome, contig=g.contig.iloc[0],
                contig_code=int(g.contig_code.iloc[0]), start_idx=int(min(pos)), end_idx=int(max(pos)),
                start_nt=int(g.start.min()), end_nt=int(g["end"].max()), member_gids=gids, member_positions=pos,
                resolved=bool(g.contig_code.nunique() == 1 and len(d) == len(d0)),
                span_genes=int(max(pos) - min(pos) + 1), n_out_of_locus=int((ln <= 0).sum()),
                core_start_idx=int(min(cpos)), core_end_idx=int(max(cpos)), core_span_genes=int(max(cpos) - min(cpos) + 1),
                sys_loci=int(d.sys_loci.iloc[0]) if "sys_loci" in d else None,
                sys_score=float(d.sys_score.iloc[0]) if "sys_score" in d else None, order_trusted=True))
    inst = pd.DataFrame(inst_rows)
    if len(inst):
        inst = inst.sort_values(["caller", "genome", "contig_code", "start_idx", "system_id"]).reset_index(drop=True)
        inst.insert(0, "instance_id", np.arange(len(inst)))
        mem = inst[["instance_id", "caller", "member_gids"]].explode("member_gids").rename(columns={"member_gids": "gid"})
        mem["gid"] = mem.gid.astype(np.int64)
    else:
        mem = pd.DataFrame(columns=["instance_id", "caller", "gid"])
    if not a.genomes:
        (P / "gold").mkdir(exist_ok=True)
        inst.to_parquet(P / "gold" / "instances.parquet", index=False, compression="zstd")
        mem.to_parquet(P / "gold" / "members.parquet", index=False, compression="zstd")
    else:
        inst.to_parquet(W / "pilot_instances.parquet", index=False)
    s = dict(genomes=len(jobs), wall_seconds=round(wall, 1), threads=a.threads,
             cpu_seconds_by_run={r: float(rec[f"{r}_s"].sum()) for r in RUNS},
             failed_by_run={r: int((~rec[f"{r}_ok"]).sum()) for r in RUNS},
             instances_by_caller=inst.caller.value_counts().to_dict() if len(inst) else {},
             instances_by_run=inst.source_table.value_counts().to_dict() if len(inst) else {},
             genomes_with_instance=inst.groupby("caller").genome.nunique().to_dict() if len(inst) else {},
             top_types={c: d.system_type.value_counts().head(15).to_dict() for c, d in inst.groupby("caller")} if len(inst) else {},
             span_quantiles={c: d.span_genes.quantile([0.5, 0.9, 1.0]).tolist() for c, d in inst.groupby("caller")} if len(inst) else {},
             core_span_quantiles={c: d.core_span_genes.quantile([0.5, 0.9, 1.0]).tolist() for c, d in inst.groupby("caller")} if len(inst) else {},
             instances_with_out_of_locus_members={c: int((d.n_out_of_locus > 0).sum()) for c, d in inst.groupby("caller")} if len(inst) else {},
             members_unresolved=int(h.gid.isna().sum()) if len(h) else 0,
             df_models=f"{DF_MODELS_DIR}/defense-finder-models {version(DF_MODELS_DIR, 'defense-finder-models')}",
             txss_models=f"{TXSS_MODELS_DIR}/TXSScan {version(TXSS_MODELS_DIR, 'TXSScan')}",
             runs=RUNS, code_sha256=sha(__file__), written_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    (P / "receipts").mkdir(exist_ok=True)
    out = (P / "receipts" / "callers.json") if not a.genomes else (W / "pilot_callers.json")
    out.write_text(json.dumps(s, indent=1, default=str) + "\n")
    print(json.dumps(s, default=str)[:3000], flush=True)


if __name__ == "__main__":
    main()
