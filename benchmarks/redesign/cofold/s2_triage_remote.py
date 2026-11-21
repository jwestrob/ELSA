#!/usr/bin/env python3
"""Remote-homology triage of block families (biotite, Slurm; threads from $SLURM_CPUS_ON_NODE).

Subcommands (all resumable; finished outputs are skipped):
  pfam      HH-suite Pfam 39.0 database from the Pfam-A.seed alignments of the same release
            (EBI releases/Pfam39.0/Pfam-A.seed.gz): one aligned FASTA per family ('.' -> '-', upper case;
            the seed marks no insert columns, so -M a2m fails for 88 % of families), hhmake -M 50, NAME line
            "<ACC> <ID>; <DE>", ffindex_build -> db/pfam39/pfam39_hhm.ff{data,index}.
            (The HMMER3 models of /shared/db/pfam/39.0 cannot be used directly: HH-suite 3.1 hhsearch
            segfaults on them, tested 12 Nov.)
  selftest  hhsearch of a Pfam seed sequence against the new database; fails unless its own family ranks first.
  families  per family FASTA (query first): mafft --auto (1 thread) -> hhblits -M first against UniRef30_2020_06
            (-n 2, -e 1e-3) -> enriched A3M; hhsearch of the enriched A3M and of the DPANN-only alignment
            against Pfam 39 (HH) and PDB100 (2021-03-03, HH-suite, RoseTTAFold copy). Workers =
            CPUS // 4, 4 threads each.
  hmm       hmmsearch Pfam 39.0 (HMMER3, /shared/db/pfam/39.0) against every member sequence, once with
            -E 0.01 --domE 0.01 and once with --cut_ga.
  cdd       rpsblast against CDD (/shared/db/cdd/latest/blast/Cdd, v3.20) for up to 10 members per family
            (query first, then random, seed 17), -evalue 1; sequences identical to the orchestrator's
            20 chains (cofold/cdd_check/queries.faa) are left out (their hits are merged at analysis).
"""
import glob
import gzip
import os
import random
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor

C = "/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/cofold"
S2 = f"{C}/s2"
HH = "/shared/software/hhsuite/latest/bin"
UNIREF = "/shared/db/uniclust/30_2020_06/UniRef30_2020_06"
PDB = "/shared/db/rosettafold/rosettafold2/pdb100_2021Mar03/pdb100_2021Mar03"
PFAMHH = f"{S2}/db/pfam39/pfam39"
PFAMHMM = "/shared/db/pfam/39.0/Pfam-A.hmm"
CDD = "/shared/db/cdd/latest/blast/Cdd"
NCPU = int(os.environ.get("SLURM_CPUS_ON_NODE", "2"))


def sh(cmd, **kw):
    return subprocess.run(cmd, check=True, **kw)


# ------------------------------------------------------------------ Pfam HH database
def _hhmake(args):
    acc, name, a2m, hhm = args
    if os.path.exists(hhm) and os.path.getsize(hhm) > 0:
        return 0
    r = subprocess.run([f"{HH}/hhmake", "-i", a2m, "-o", hhm, "-name", name, "-M", "50", "-v", "0"],
                       capture_output=True, text=True)
    return r.returncode


MARK = f"{S2}/db/pfam39/SELFTEST_OK"


def cmd_pfam():
    """Several jobs may call this: the first takes db/pfam39/build.lock and builds; the others wait for the
    self-test marker."""
    d = f"{S2}/db/pfam39"
    if os.path.exists(MARK):
        print("pfam39 HH database ready", flush=True)
        return
    try:
        os.mkdir(f"{d}/build.lock")
    except FileExistsError:
        print("another job is building pfam39; waiting for the self-test marker", flush=True)
        while not os.path.exists(MARK):
            time.sleep(60)
        return
    os.makedirs(f"{d}/fa", exist_ok=True)
    os.makedirs(f"{d}/hhm", exist_ok=True)
    jobs = []
    with gzip.open(f"{d}/Pfam-A.seed.gz", "rt", encoding="latin-1") as f:
        acc = idd = de = None
        seqs, order = {}, []
        for line in f:
            if line.startswith("#=GF AC"):
                acc = line.split()[2]
            elif line.startswith("#=GF ID"):
                idd = line.split()[2]
            elif line.startswith("#=GF DE"):
                de = line[10:].strip()
            elif line.startswith("//"):
                base = acc.split(".")[0]
                a2m = f"{d}/fa/{base}.fa"
                if not os.path.exists(a2m):
                    with open(a2m, "w") as o:
                        for k in order:
                            o.write(f">{k}\n{''.join(seqs[k]).replace('.', '-').upper()}\n")
                jobs.append((acc, f"{acc} {idd}; {de}", a2m, f"{d}/hhm/{base}.hhm"))
                acc = idd = de = None
                seqs, order = {}, []
            elif line.startswith("#") or not line.strip():
                continue
            else:
                k, s = line.split()
                if k not in seqs:
                    seqs[k] = []
                    order.append(k)
                seqs[k].append(s)
    print(len(jobs), "Pfam families", flush=True)
    with ProcessPoolExecutor(NCPU) as ex:
        rc = list(ex.map(_hhmake, jobs, chunksize=50))
    bad = [j[0] for j, r in zip(jobs, rc) if r != 0]
    print("hhmake failures:", len(bad), bad[:20], flush=True)
    for ext in ("ffdata", "ffindex"):
        p = f"{PFAMHH}_hhm.{ext}"
        if os.path.exists(p):
            os.remove(p)
    lst = f"{d}/hhm_files.txt"
    with open(lst, "w") as o:
        for j in jobs:
            if os.path.exists(j[3]) and os.path.getsize(j[3]) > 0:
                o.write(j[3] + "\n")
    sh([f"{HH}/ffindex_build", "-s", "-f", lst, f"{PFAMHH}_hhm.ffdata", f"{PFAMHH}_hhm.ffindex"])
    n = sum(1 for _ in open(f"{PFAMHH}_hhm.ffindex"))
    print("pfam39_hhm entries:", n, flush=True)


def cmd_selftest():
    d = f"{S2}/db/pfam39"
    if os.path.exists(MARK):
        print("selftest already passed", flush=True)
        return
    t = f"{S2}/triage/selftest"
    os.makedirs(t, exist_ok=True)
    # first sequence of three seed alignments, gaps removed
    ok = 0
    for base in ("PF00004", "PF01850", "PF02604"):
        lines = open(f"{d}/fa/{base}.fa").read().split("\n")
        seq = lines[1].replace("-", "").replace(".", "").upper()
        open(f"{t}/{base}.fa", "w").write(f">{base}_seed1\n{seq}\n")
        sh([f"{HH}/hhsearch", "-i", f"{t}/{base}.fa", "-d", PFAMHH, "-o", f"{t}/{base}.hhr", "-cpu", "2", "-v", "0"])
        top = [l for l in open(f"{t}/{base}.hhr") if l[:3].strip().isdigit()][0]
        print(base, "top hit:", top.rstrip(), flush=True)
        ok += base in top
    assert ok == 3, "Pfam HH self-test failed"
    open(MARK, "w").write("ok\n")
    print("selftest passed", flush=True)


# ------------------------------------------------------------------ per family
def nseq(path):
    return sum(1 for l in open(path) if l.startswith(">"))


def complete(tok):
    d = f"{S2}/triage/out/{tok}"
    return all(os.path.exists(f"{d}/{x}.hhr") for x in ("pfam_enr", "pfam_raw", "pdb_enr", "pdb_raw"))


def _family(fa):
    """One family; several jobs share the work through a lock directory per family (out/<token>/.lock)."""
    tok = os.path.basename(fa)[4:-4]
    d = f"{S2}/triage/out/{tok}"
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
            if dbn == "pfam" and not os.path.exists(MARK):
                continue
            for tag, inp, extra in (("enr", f"{d}/enr.a3m", []), ("raw", aln, ["-M", "first"])):
                out = f"{d}/{dbn}_{tag}.hhr"
                if not os.path.exists(out):
                    sh([f"{HH}/hhsearch", "-i", inp, *extra, "-d", db, "-o", out + ".tmp", "-cpu", "4", "-v", "0",
                        "-Z", "100", "-B", "100"])
                    os.replace(out + ".tmp", out)
        return tok, "ok" if complete(tok) else "pfam_pending"
    except subprocess.CalledProcessError as e:
        return tok, f"error: {e}"
    finally:
        os.rmdir(f"{d}/.lock")


def cmd_families():
    fas = sorted(glob.glob(f"{S2}/triage/fasta/fam_*.faa"))
    pri = os.environ.get("FAMILY_ORDER")      # optional file of tokens to take first; the rest in reverse order
    if pri:
        first = [f"{S2}/triage/fasta/fam_{t.strip()}.faa" for t in open(pri) if t.strip()]
        first = [f for f in first if f in set(fas)]
        fas = first + [f for f in reversed(fas) if f not in set(first)]
    nw = max(1, NCPU // 4)
    print(len(fas), "families;", nw, "workers x 4 threads", flush=True)
    with ProcessPoolExecutor(nw) as ex:
        for tok, st in ex.map(_family, fas):
            print(tok, st, flush=True)
    # second pass: Pfam searches once the database is ready (other jobs may hold some families)
    while True:
        todo = [fa for fa in fas if not complete(os.path.basename(fa)[4:-4])]
        if not todo:
            break
        if not os.path.exists(MARK):
            time.sleep(60)
            continue
        with ProcessPoolExecutor(nw) as ex:
            res = list(ex.map(_family, todo))
        for tok, st in res:
            print(tok, st, flush=True)
        if all(st == "locked" for _, st in res):
            print("remaining families are held by other jobs:", len(res), flush=True)
            break
    print("families complete:", sum(complete(os.path.basename(fa)[4:-4]) for fa in fas), "of", len(fas), flush=True)


# ------------------------------------------------------------------ Pfam (HMMER) and CDD
def read_fa(p):
    out, h = [], None
    for line in open(p):
        line = line.strip()
        if line.startswith(">"):
            h = line[1:].split()[0]
            out.append([h, ""])
        elif line:
            out[-1][1] += line
    return out


def cmd_hmm():
    t = f"{S2}/triage"
    allf = f"{t}/members_all.faa"
    with open(allf, "w") as o:
        for fa in sorted(glob.glob(f"{t}/fasta/fam_*.faa")):
            tok = os.path.basename(fa)[4:-4]
            for h, s in read_fa(fa):
                o.write(f">{tok}|{h}\n{s}\n")
    for tag, flags in (("e01", ["-E", "0.01", "--domE", "0.01"]), ("ga", ["--cut_ga"])):
        out = f"{t}/pfam39_hmmsearch_{tag}.domtbl"
        if not os.path.exists(out):
            sh(["hmmsearch", "--cpu", str(NCPU), "--noali", "-o", "/dev/null", "--domtblout", out + ".tmp", *flags,
                PFAMHMM, allf])
            os.replace(out + ".tmp", out)
    print("hmmsearch done", flush=True)


def cmd_cdd():
    t = f"{S2}/triage"
    skip = {s for _, s in read_fa(f"{C}/cdd_check/queries.faa")}
    rng = random.Random(17)
    q = f"{t}/cdd_queries.faa"
    n = nskip = 0
    with open(q, "w") as o:
        for fa in sorted(glob.glob(f"{t}/fasta/fam_*.faa")):
            tok = os.path.basename(fa)[4:-4]
            recs = read_fa(fa)
            pick = recs[:1] + rng.sample(recs[1:], min(9, len(recs) - 1))
            for h, s in pick:
                if s in skip:
                    nskip += 1
                    continue
                o.write(f">{tok}|{h}\n{s}\n")
                n += 1
    print("cdd queries", n, "skipped (orchestrator already ran)", nskip, flush=True)
    out = f"{t}/cdd_hits.tsv"
    if not os.path.exists(out):
        sh(["rpsblast", "-query", q, "-db", CDD, "-evalue", "1", "-num_threads", str(NCPU), "-outfmt",
            "6 qseqid sseqid qlen qstart qend evalue bitscore stitle", "-out", out + ".tmp"])
        os.replace(out + ".tmp", out)
    print("rpsblast done", flush=True)


if __name__ == "__main__":
    {"pfam": cmd_pfam, "selftest": cmd_selftest, "families": cmd_families, "hmm": cmd_hmm, "cdd": cmd_cdd}[sys.argv[1]]()
