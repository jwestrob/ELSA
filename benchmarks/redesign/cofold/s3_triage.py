#!/usr/bin/env python3
"""Screen 3: enriched remote-homology triage of selected families (the screen-2 search, s2_triage_remote.py,
unchanged; only its working directory is redirected to $RD/cofold/s3/triage; the Pfam 39 HH database and its
self-test marker of screen 2 are reused).

  fasta TOKENS_FILE   per rescue3 token: query = the replicate-1 chain of the screen-3 block that brought it
                      (work/s3/design/reps.tsv), then up to 249 further unique member sequences (one per genome
                      first, then random, seed 17) -> triage/fasta/fam_<token>.faa   [login node, serial]
  families | hmm | cdd   s2_triage_remote.py subcommands                              [Slurm]
"""
import os
import sys

import numpy as np

RD = "/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111"
S3 = f"{RD}/cofold/s3"
sys.path.insert(0, f"{S3}/code")


def cmd_fasta(tokfile):
    import pandas as pd
    toks = sorted({int(x) for x in open(tokfile).read().split()})
    tok = pd.read_parquet(f"{RD}/engine/dpann_tokens/mmseqs__seq__e3c50s75_rescue3__dpann.parquet").sort_values("gid").token.to_numpy()
    g = pd.read_parquet(f"{RD}/ws_dpann/package/universe/genes.parquet", columns=["gid", "genome_code"])
    g["token"] = tok
    reps = pd.read_csv(f"{S3}/design/reps.tsv", sep="\t")
    qg = {}
    for _, r in reps.sort_values(["s3_rank", "rep"]).iterrows():
        for x in str(r.gids).split(";"):
            qg.setdefault(int(tok[int(x)]), int(x))
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
    out = f"{S3}/triage/fasta"
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(17)
    for t in toks:
        m = mem[mem.token == t].copy()
        m["seq"] = m.gid.map(seqs)
        q = qg.get(t, int(m.gid.iloc[0]))
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
        with open(f"{out}/fam_{t}.faa", "w") as fh:
            fh.write(f">q{q}\n{qseq}\n")
            for gid, s in zip(u.gid, u.seq):
                fh.write(f">g{gid}\n{s}\n")
        print(t, "members", len(m), "in fasta", 1 + len(u), flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "fasta":
        cmd_fasta(sys.argv[2])
    else:
        import s2_triage_remote as tr
        tr.S2 = S3          # working directory: S3/triage/{fasta,out}; PFAMHH and MARK stay on the screen-2 database
        {"families": tr.cmd_families, "hmm": tr.cmd_hmm, "cdd": tr.cmd_cdd}[sys.argv[1]]()
