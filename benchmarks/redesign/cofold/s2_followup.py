"""Screen 2 follow-up fold jobs (predeclared_criteria_screen2.json, refutations).

  mono  RANK [RANK ...]   single-chain folds of every chain of both replicates: mono2_<rank>_r<rep>_<chain>
                          (disorder-zipping check)
  msa   RANK [RANK ...]   genome-paired MSAs from the block instances (one instance per genome, mafft --auto with
                          one thread per family, query row first, columns where the query has a residue), all k
                          chains: msa2_cog_<rank>_r<rep>; and the replicate's rep-1/rep-2 non-cognate swap controls
                          with the same MSAs for the kept chains and the donor's own family MSA for the replaced
                          chain (rows truncated to the shortest depth): msa2_non_<rank>_r<rep>_s<slot>
Writes work/s2/jobs_mono.jsonl or work/s2/jobs_msa.jsonl (appends nothing; rewrites the file each call).
"""
import json
import subprocess
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
W = HERE / "work"
W2 = W / "s2"
OUT = W2 / "msa"
SHARUR = "/Users/jacob/Documents/Sandbox/Sharur/Sharur/data/dpann/sharur.duckdb"


def seqs_for(gids):
    g = pd.read_parquet(W / "genes_dark.parquet", columns=["gid", "protein_id"]).set_index("gid")
    pid = g.loc[list(gids), "protein_id"]
    sh = duckdb.connect(SHARUR, read_only=True)
    sh.execute("SET threads=1")
    sh.register("p", pd.DataFrame({"protein_id": pid.values}))
    s = dict(sh.execute("SELECT q.protein_id, rtrim(q.sequence,'*') FROM proteins q JOIN p USING (protein_id)").fetchall())
    return {int(k): s[v] for k, v in pid.items()}


def cmd_mono(ranks):
    man = pd.read_csv(W2 / "fold_manifest2.tsv", sep="\t")
    cog = man[(man.cls == "cognate") & man.s2_rank.isin(ranks)]
    gids = {int(x) for s in cog.gids for x in s.split(";")}
    sq = seqs_for(gids)
    n = 0
    with open(W2 / "jobs_mono.jsonl", "w") as f:
        for _, r in cog.iterrows():
            for c, x in zip("ABCDEFGH", r.gids.split(";")):
                f.write(json.dumps({"fold_id": f"mono2_{r.s2_rank:03d}_r{r.rep}_{c}", "cls": "monomer",
                                    "chains": [{"id": "A", "seq": sq[int(x)], "gid": int(x)}]}) + "\n")
                n += 1
    print(n, "monomer jobs")


def mafft(seqs, tag):
    OUT.mkdir(parents=True, exist_ok=True)
    fin = OUT / f"{tag}.in.faa"
    with open(fin, "w") as f:
        for k, s in seqs.items():
            f.write(f">{k}\n{s}\n")
    res = subprocess.run(["mafft", "--auto", "--thread", "1", "--quiet", str(fin)], capture_output=True, text=True, check=True)
    aln, h = {}, None
    for line in res.stdout.splitlines():
        if line.startswith(">"):
            h = line[1:].split()[0]
            aln[h] = []
        elif line.strip():
            aln[h].append(line.strip().upper())
    aln = {k: "".join(v) for k, v in aln.items()}
    with open(OUT / f"{tag}.aln.faa", "w") as f:
        for k, s in aln.items():
            f.write(f">{k}\n{s}\n")
    return aln


def project(aln, order, qk, qseq):
    q = aln[qk]
    cols = [i for i, ch in enumerate(q) if ch != "-"]
    rows = ["".join(aln[k][i] for i in cols) for k in order]
    assert rows[0].replace("-", "") == qseq
    rows[0] = qseq
    return rows


def cmd_msa(ranks):
    man = pd.read_csv(W2 / "fold_manifest2.tsv", sep="\t")
    inst = pd.read_parquet(W2 / "block_instances.parquet")
    fam = pd.read_parquet(W2 / "fam_stats.parquet")
    cache = {}

    def block_alignment(rank):
        if rank in cache:
            return cache[rank]
        cg = man[(man.cls == "cognate") & (man.s2_rank == rank)].sort_values("rep")
        tk = cg.type_key.iloc[0]
        tt = [int(x) for x in tk.split("-")]
        flen = [fam.at[t, "fam_len_med"] for t in tt]
        df = inst[inst.type_key == tk].copy()
        reps = {int(r.rep): [int(x) for x in r.gids.split(";")] for _, r in cg.iterrows()}
        df["is_rep"] = df.gids.map(lambda gl: any(list(gl) == v for v in reps.values()))
        df["lenok"] = [all(0.8 * f <= L <= 1.25 * f for L, f in zip(l, flen)) for l in df.lengths]
        df["comp"] = df.completeness.fillna(0)
        df = df.sort_values(["is_rep", "n_x", "lenok", "comp"], ascending=[False, True, False, False]).drop_duplicates("genome_code")
        rows = [list(map(int, gl)) for gl in df.gids]
        sq = seqs_for({x for r in rows for x in r})
        keys = ["_".join(map(str, r)) for r in rows]
        alns = [mafft({k: sq[r[c]] for k, r in zip(keys, rows)}, f"s2r{rank:03d}_{c}") for c in range(len(tt))]
        cache[rank] = (keys, rows, alns, reps, sq)
        return cache[rank]

    jobs = []
    for rank in ranks:
        keys, rows, alns, reps, sq = block_alignment(rank)
        for rep, gl in reps.items():
            qk = "_".join(map(str, gl))
            order = [qk] + [k for k in keys if k != qk]
            msas = [project(alns[c], order, qk, sq[gl[c]]) for c in range(len(gl))]
            jobs.append(dict(fold_id=f"msa2_cog_{rank:03d}_r{rep}", cls="msa_cognate",
                             chains=[dict(id="ABCDEFGH"[c], seq=sq[gl[c]], gid=gl[c], msa=msas[c]) for c in range(len(gl))]))
            # non-cognate controls of this replicate, with MSAs
            for _, nc in man[(man.cls == "noncognate") & (man.s2_rank == rank) & (man.rep == rep)].iterrows():
                s = int(nc.slot)
                donor_rank = int(nc.match_level.split(";")[0].split("=")[1])
                dgid = int(nc.gids.split(";")[s])
                dkeys, drows, dalns, dreps, dsq = block_alignment(donor_rank)
                dg = dreps[rep]
                dc = dg.index(dgid)
                dqk = "_".join(map(str, dg))
                dorder = [dqk] + [k for k in dkeys if k != dqk]
                dmsa = project(dalns[dc], dorder, dqk, dsq[dgid])
                ms = list(msas)
                ms[s] = dmsa
                depth = min(len(m) for m in ms)
                ch = []
                for c in range(len(gl)):
                    x = dgid if c == s else gl[c]
                    ch.append(dict(id="ABCDEFGH"[c], seq=dsq[dgid] if c == s else sq[gl[c]], gid=x, msa=ms[c][:depth]))
                jobs.append(dict(fold_id=f"msa2_non_{rank:03d}_r{rep}_s{s}", cls="msa_noncognate", chains=ch))
            print(f"rank {rank} rep {rep}: depth {len(order)}", flush=True)
    with open(W2 / "jobs_msa.jsonl", "w") as f:
        for j in jobs:
            f.write(json.dumps(j) + "\n")
    print(len(jobs), "MSA jobs")


if __name__ == "__main__":
    {"mono": cmd_mono, "msa": cmd_msa}[sys.argv[1]]([int(x) for x in sys.argv[2:]])
