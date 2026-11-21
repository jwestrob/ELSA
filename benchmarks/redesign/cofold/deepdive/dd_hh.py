#!/usr/bin/env python3
"""Deep dive: profile searches for candidate 1 (biotite, Slurm; threads from $SLURM_CPUS_ON_NODE).

Usage: dd_hh.py u|ub
  u   HHblits 3 iterations against UniRef30_2020_06 (5 queries in parallel)
  ub  HHblits 3 iterations against UniRef30_2020_06 + BFD (metaclust, id30 c90) together (2 queries at a time:
      the BFD cs219 prefilter file is 16.8 GB and is read per process)
Queries (hh_in/): c1A, c1B (token members, aligned with mafft --auto), c1AB (block instances, A and B aligned
separately and concatenated row by row, so each row is one genome's A followed by its B), c1F1, c1F2 (fused
homologs, single sequence). Each resulting A3M is searched with HHsearch against Pfam 39.0 (HH database built
by ws_cofold2 from Pfam-A.seed with hhmake -M 50; self-test passed) and PDB100 (2021-03-03).
The DPANN-only alignments (iteration 0) of c1AB and of c1A/c1B are also searched against both (mode u only).
"""
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor

DD = "/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/cofold/deepdive"
HH = "/shared/software/hhsuite/latest/bin"
UNIREF = "/shared/db/uniclust/30_2020_06/UniRef30_2020_06"
BFD = "/shared/db/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt"
PDB = "/shared/db/rosettafold/rosettafold2/pdb100_2021Mar03/pdb100_2021Mar03"
PFAM = ("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/cofold/s2/db/pfam39/"
        "pfam39")
NCPU = int(os.environ.get("SLURM_CPUS_ON_NODE", "2"))
IN = f"{DD}/work/hh_in"
OUT = f"{DD}/hh"


def read_fasta(p):
    seqs, h = {}, None
    for line in open(p):
        line = line.rstrip()
        if line.startswith(">"):
            h = line[1:].split()[0]
            seqs[h] = []
        elif line:
            seqs[h].append(line)
    return {k: "".join(v) for k, v in seqs.items()}


def mafft(src, dst, threads):
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return
    r = subprocess.run(["mafft", "--auto", "--thread", str(threads), "--quiet", src], capture_output=True, text=True,
                       check=True)
    open(dst, "w").write("\n".join(l if l.startswith(">") else l.upper() for l in r.stdout.splitlines()) + "\n")


def prepare():
    os.makedirs(OUT, exist_ok=True)
    if os.path.exists(f"{OUT}/c1AB.aln.fa"):
        return
    for q in ("c1A", "c1B", "c1A_inst", "c1B_inst"):
        mafft(f"{IN}/{q}.fa", f"{OUT}/{q}.aln.fa", NCPU)
    a = read_fasta(f"{OUT}/c1A_inst.aln.fa")
    b = read_fasta(f"{OUT}/c1B_inst.aln.fa")
    order = list(read_fasta(f"{IN}/c1A_inst.fa"))
    with open(f"{OUT}/c1AB.aln.fa", "w") as f:
        for k in order:
            f.write(f">{k}\n{a[k]}{b[k]}\n")
    for q in ("c1F1", "c1F2"):
        open(f"{OUT}/{q}.aln.fa", "w").write(open(f"{IN}/{q}.fa").read())


def run(args):
    q, mode, threads = args
    base = f"{OUT}/{q}.{mode}"
    if not (os.path.exists(f"{base}.a3m") and os.path.getsize(f"{base}.a3m") > 0):
        dbs = ["-d", UNIREF] + (["-d", BFD] if mode == "ub" else [])
        cmd = [f"{HH}/hhblits", "-i", f"{OUT}/{q}.aln.fa", *dbs, "-n", "3", "-e", "0.001", "-cpu", str(threads),
               "-oa3m", f"{base}.a3m.tmp", "-o", f"{base}.hhr", "-v", "1", "-maxfilt", "100000",
               "-realign_max", "100000", "-B", "100000", "-Z", "100000"]
        if not q.startswith("c1F"):
            cmd[3:3] = ["-M", "first"]
        subprocess.run(cmd, check=True)
        os.replace(f"{base}.a3m.tmp", f"{base}.a3m")
    for tag, db in (("pfam", PFAM), ("pdb", PDB)):
        o = f"{base}.{tag}.hhr"
        if not os.path.exists(o):
            subprocess.run([f"{HH}/hhsearch", "-i", f"{base}.a3m", "-d", db, "-o", o, "-cpu", str(threads), "-v", "1",
                            "-B", "500", "-Z", "500"], check=True)
    return q, mode


def run_dpann_only(args):
    q, threads = args
    for tag, db in (("pfam", PFAM), ("pdb", PDB)):
        o = f"{OUT}/{q}.dpann.{tag}.hhr"
        if not os.path.exists(o):
            cmd = [f"{HH}/hhsearch", "-i", f"{OUT}/{q}.aln.fa", "-d", db, "-o", o, "-cpu", str(threads), "-v", "1",
                   "-B", "500", "-Z", "500"]
            if not q.startswith("c1F"):
                cmd[3:3] = ["-M", "first"]
            subprocess.run(cmd, check=True)
    return q


def main(mode):
    prepare()
    qs = ["c1A", "c1B", "c1AB", "c1F1", "c1F2"]
    if mode == "u":
        t = max(1, NCPU // len(qs))
        with ProcessPoolExecutor(len(qs)) as ex:
            for r in ex.map(run, [(q, "u", t) for q in qs]):
                print("done", r, flush=True)
        with ProcessPoolExecutor(len(qs)) as ex:
            for r in ex.map(run_dpann_only, [(q, t) for q in qs]):
                print("done dpann-only", r, flush=True)
    else:
        t = max(1, NCPU // 2)
        with ProcessPoolExecutor(2) as ex:
            for r in ex.map(run, [(q, "ub", t) for q in qs]):
                print("done", r, flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "prepare":
        prepare()
    else:
        main(sys.argv[1])
