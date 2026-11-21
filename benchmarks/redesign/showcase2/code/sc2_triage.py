#!/usr/bin/env python3
"""ws_showcase2 step 2: enriched remote-homology triage of the candidate families, in the order that worked in the
earlier screens (s2_triage_remote.py, copied here unchanged in its search commands; only the working directory and
the CDD query set differ).

  fasta     [login node, serial, seconds] per rescue3 token carried by a replicate chain (data/reps.tsv): query = the
            replicate chain (rep 1 first), then up to 249 further unique member sequences (one per genome first, then
            random, seed 17) -> triage/fasta/fam_<token>.faa. Tokens whose replicate gene carries a curated Pfam
            hit without unknown-function wording are listed in triage/pfam_annotated.tsv and get no HH search (the
            gene is recognised already); every other replicate token is searched.
  families  [Slurm] mafft --auto (1 thread) -> HHblits UniRef30_2020_06 (-n 2, -e 1e-3, -M first) -> enriched A3M;
            HHsearch of the enriched A3M and of the DPANN-only alignment against Pfam 39.0 (HH, screen-2 build) and
            PDB100 2021-03-03. Workers = CPUs // 4, 4 threads each.
  hmm       [Slurm] hmmsearch Pfam 39.0 (HMMER3) of every triage member, -E 0.01 --domE 0.01 and --cut_ga.
  cdd       [Slurm] RPS-BLAST CDD v3.20 of the query plus up to 9 random members per family (seed 17), -evalue 1.
"""
import glob
import os
import random
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor

RD = "/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111"
S = f"{RD}/showcase2/triage"
HH = "/shared/software/hhsuite/latest/bin"
UNIREF = "/shared/db/uniclust/30_2020_06/UniRef30_2020_06"
PDB = "/shared/db/rosettafold/rosettafold2/pdb100_2021Mar03/pdb100_2021Mar03"
PFAMHH = f"{RD}/cofold/s2/db/pfam39/pfam39"
MARK = f"{RD}/cofold/s2/db/pfam39/SELFTEST_OK"
PFAMHMM = "/shared/db/pfam/39.0/Pfam-A.hmm"
CDD = "/shared/db/cdd/latest/blast/Cdd"
NCPU = int(os.environ.get("SLURM_CPUS_ON_NODE", "2"))
UNKNOWN = r"DUF\d|UPF\d|unknown function|uncharacteri|hypothetical|function unknown|unknown protein"


def sh(cmd, **kw):
    return subprocess.run(cmd, check=True, **kw)


def cmd_fasta():
    import re
    import duckdb
    import numpy as np
    import pandas as pd
    reps = pd.read_csv(f"{RD}/showcase2/data/reps.tsv", sep="\t").sort_values(["rank", "rep"])
    tok = pd.read_parquet(f"{RD}/engine/dpann_tokens/mmseqs__seq__e3c50s75_rescue3__dpann.parquet").sort_values("gid").token.to_numpy()
    g = pd.read_parquet(f"{RD}/ws_dpann/package/universe/genes.parquet", columns=["gid", "genome_code", "protein_id"])
    g["token"] = tok
    qg = {}
    for gl in reps.gids:
        for x in str(gl).split(";"):
            qg.setdefault(int(tok[int(x)]), int(x))
    # curated Pfam hits on the replicate genes (Sharur annotations, source pfam)
    pids = g.protein_id.to_numpy()
    want_p = pd.DataFrame(dict(protein_id=[pids[x] for x in qg.values()], gid=list(qg.values())))
    con = duckdb.connect(f"{RD}/ws_dpann/input/sharur.duckdb", read_only=True, config={"threads": 2})
    con.register("w", want_p)
    pf = con.execute("select w.gid, a.name, a.description from annotations a join w using (protein_id) "
                     "where a.source = 'pfam'").fetchdf()
    con.close()
    rx = re.compile(UNKNOWN, re.I)
    pf["unknown"] = [bool(rx.search(f"{n} {d}")) for n, d in zip(pf.name, pf.description.fillna(""))]
    char_gid = set(pf[~pf.unknown].gid)
    os.makedirs(f"{S}/fasta", exist_ok=True)
    ann = []
    toks = []
    for t, q in qg.items():
        if q in char_gid:
            ann.append(dict(token=t, query_gid=q, pfam=";".join(sorted(set(pf[pf.gid == q].name)))))
        else:
            toks.append(t)
    pd.DataFrame(ann).to_csv(f"{S}/pfam_annotated.tsv", sep="\t", index=False)
    mem = g[g.token.isin(toks)]
    want = set(mem.gid) | set(qg.values())
    seqs, keep, h, buf = {}, False, None, []
    with open(f"{RD}/ws_dpann/work/proteins_gid.faa") as f:
        for line in f:
            if line[0] == ">":
                if keep:
                    seqs[h] = "".join(buf).rstrip("*")
                h = int(line[1:])
                keep = h in want
                buf = []
            elif keep:
                buf.append(line.strip())
    if keep:
        seqs[h] = "".join(buf).rstrip("*")
    rng = np.random.default_rng(17)
    for t in sorted(toks):
        m = mem[mem.token == t].copy()
        m["seq"] = m.gid.map(seqs)
        q = qg[t]
        qseq = seqs[q]
        u = m.drop_duplicates("seq")
        u = u[u.seq != qseq]
        if len(u) > 249:
            first = u.drop_duplicates("genome_code")
            if len(first) >= 249:
                u = first.iloc[rng.permutation(len(first))[:249]]
            else:
                rest = u[~u.gid.isin(first.gid)]
                u = pd.concat([first, rest.iloc[rng.permutation(len(rest))[:249 - len(first)]]])
        with open(f"{S}/fasta/fam_{t}.faa", "w") as fh:
            fh.write(f">q{q}\n{qseq}\n")
            for gid, s in zip(u.gid, u.seq):
                fh.write(f">g{gid}\n{s}\n")
    print("tokens", len(qg), "searched", len(toks), "pfam-annotated (characterised wording)", len(ann), flush=True)


# ------------------------------------------------------------------ per family (s2_triage_remote._family, unchanged)
def nseq(path):
    return sum(1 for l in open(path) if l.startswith(">"))


def complete(tok):
    d = f"{S}/out/{tok}"
    return all(os.path.exists(f"{d}/{x}.hhr") for x in ("pfam_enr", "pfam_raw", "pdb_enr", "pdb_raw"))


def _family(fa):
    tok = os.path.basename(fa)[4:-4]
    d = f"{S}/out/{tok}"
    os.makedirs(d, exist_ok=True)
    if complete(tok):
        return tok, "ok"
    try:
        os.mkdir(f"{d}/.lock")
    except FileExistsError:
        return tok, "locked"
    aln = f"{d}/aln.fa"
    try:
        if not os.path.exists(aln):
            if nseq(fa) == 1:
                sh(["cp", fa, aln])
            else:
                with open(aln + ".tmp", "w") as o:
                    sh(["mafft", "--auto", "--thread", "1", "--quiet", fa], stdout=o)
                os.replace(aln + ".tmp", aln)
        if not os.path.exists(f"{d}/enr.a3m"):
            sh([f"{HH}/hhblits", "-i", aln, "-M", "first", "-d", UNIREF, "-n", "2", "-e", "0.001", "-cpu", "4",
                "-o", f"{d}/uniref.hhr", "-oa3m", f"{d}/enr.a3m.tmp", "-v", "0"])
            os.replace(f"{d}/enr.a3m.tmp", f"{d}/enr.a3m")
        for dbn, db in (("pdb", PDB), ("pfam", PFAMHH)):
            for tag, inp, extra in (("enr", f"{d}/enr.a3m", []), ("raw", aln, ["-M", "first"])):
                out = f"{d}/{dbn}_{tag}.hhr"
                if not os.path.exists(out):
                    sh([f"{HH}/hhsearch", "-i", inp, *extra, "-d", db, "-o", out + ".tmp", "-cpu", "4", "-v", "0",
                        "-Z", "100", "-B", "100"])
                    os.replace(out + ".tmp", out)
        return tok, "ok" if complete(tok) else "incomplete"
    except subprocess.CalledProcessError as e:
        return tok, f"error: {e}"
    finally:
        os.rmdir(f"{d}/.lock")


def cmd_families():
    assert os.path.exists(MARK), "Pfam 39 HH database self-test marker missing"
    fas = sorted(glob.glob(f"{S}/fasta/fam_*.faa"), key=lambda p: -os.path.getsize(p))
    nw = max(1, NCPU // 4)
    print(len(fas), "families;", nw, "workers x 4 threads", flush=True)
    with ProcessPoolExecutor(nw) as ex:
        for tok, st in ex.map(_family, fas):
            print(tok, st, flush=True)
    print("families complete:", sum(complete(os.path.basename(fa)[4:-4]) for fa in fas), "of", len(fas), flush=True)


def read_fa(p):
    out = []
    for line in open(p):
        line = line.strip()
        if line.startswith(">"):
            out.append([line[1:].split()[0], ""])
        elif line:
            out[-1][1] += line
    return out


def cmd_hmm():
    allf = f"{S}/members_all.faa"
    with open(allf, "w") as o:
        for fa in sorted(glob.glob(f"{S}/fasta/fam_*.faa")):
            tok = os.path.basename(fa)[4:-4]
            for h, s in read_fa(fa):
                o.write(f">{tok}|{h}\n{s}\n")
    for tag, flags in (("e01", ["-E", "0.01", "--domE", "0.01"]), ("ga", ["--cut_ga"])):
        out = f"{S}/pfam39_hmmsearch_{tag}.domtbl"
        if not os.path.exists(out):
            sh(["hmmsearch", "--cpu", str(NCPU), "--noali", "-o", "/dev/null", "--domtblout", out + ".tmp", *flags,
                PFAMHMM, allf])
            os.replace(out + ".tmp", out)
    print("hmmsearch done", flush=True)


def cmd_cdd():
    rng = random.Random(17)
    q = f"{S}/cdd_queries.faa"
    n = 0
    with open(q, "w") as o:
        for fa in sorted(glob.glob(f"{S}/fasta/fam_*.faa")):
            tok = os.path.basename(fa)[4:-4]
            recs = read_fa(fa)
            pick = recs[:1] + rng.sample(recs[1:], min(9, len(recs) - 1))
            for h, s in pick:
                o.write(f">{tok}|{h}\n{s}\n")
                n += 1
    print("cdd queries", n, flush=True)
    out = f"{S}/cdd_hits.tsv"
    if not os.path.exists(out):
        sh(["rpsblast", "-query", q, "-db", CDD, "-evalue", "1", "-num_threads", str(NCPU), "-outfmt",
            "6 qseqid sseqid qlen qstart qend evalue bitscore stitle", "-out", out + ".tmp"])
        os.replace(out + ".tmp", out)
    print("rpsblast done", flush=True)


if __name__ == "__main__":
    {"fasta": cmd_fasta, "families": cmd_families, "hmm": cmd_hmm, "cdd": cmd_cdd}[sys.argv[1]]()
