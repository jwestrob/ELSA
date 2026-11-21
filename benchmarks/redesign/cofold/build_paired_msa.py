"""Step 3: genome-paired MSAs built from the ELSA block instances themselves, and the MSA fold jobs.

For a candidate block (A, B): one instance per genome carrying the block (no X; lengths within
0.8-1.25 x the family median when possible; then highest completeness). Family A and family B
sequences are aligned separately with mafft --auto (one thread, local). Row i of the A alignment
and row i of the B alignment come from the same genome. For a replicate, its own genome is row 0,
columns where the query has a gap are removed (insertions relative to the query are dropped).

Jobs written per candidate and replicate:
  msa_cog_<rank>_r<rep>      cognate, genome-paired rows
  msa_shuf_<rank>_r<rep>     cognate, same alignments, B rows (except the query) permuted (seed 17):
                             tests whether row pairing matters to the model
  msa_non_<rank>_r<rep>      the screen's non-cognate pair, each chain with its own family alignment
                             (rows truncated to the shorter depth; pairing meaningless)
  mono_<rank>_r<rep>_<A|B>   single-sequence monomer of each chain (disorder-zipping check)
Also writes Protenix v2 input JSONs (pairedMsaPath / unpairedMsaPath as a3m) for later confirmation.

Usage: python build_paired_msa.py RANK [RANK ...]
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
W = HERE / "work"
OUT = W / "msa"
RD_REMOTE = "/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/cofold/protenix"


def read_fasta(p):
    seqs, h = {}, None
    for line in open(p):
        line = line.strip()
        if line.startswith(">"):
            h = line[1:].split()[0]
            seqs[h] = []
        elif line:
            seqs[h].append(line)
    return {k: "".join(v).rstrip("*") for k, v in seqs.items()}


def mafft(seqs, tag):
    OUT.mkdir(parents=True, exist_ok=True)
    fin = OUT / f"{tag}.in.faa"
    with open(fin, "w") as f:
        for k, s in seqs.items():
            f.write(f">{k}\n{s}\n")
    res = subprocess.run(["mafft", "--auto", "--thread", "1", "--quiet", str(fin)], capture_output=True, text=True,
                         check=True)
    aln = {}
    h = None
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


def project(aln, order, query_key, query_seq):
    """Rows in `order` (query first), columns where the query has a residue."""
    q = aln[query_key]
    cols = [i for i, ch in enumerate(q) if ch != "-"]
    rows = ["".join(aln[k][i] for i in cols) for k in order]
    assert rows[0].replace("-", "") == query_seq, f"query mismatch for {query_key}"
    assert all(len(r) == len(query_seq) for r in rows)
    rows[0] = query_seq
    return rows


def main(ranks):
    man = pd.read_csv(W / "fold_manifest.tsv", sep="\t")
    cand = pd.read_csv(W / "candidates_all.tsv", sep="\t")
    fam = pd.read_parquet(W / "fam_stats.parquet")
    inst = pd.read_parquet(W / "block_instances.parquet")
    seqs = read_fasta(W / "instance_seqs.faa")
    seqs.update(read_fasta(W / "fold_set1.faa"))
    jobs, prot = [], []
    families = {}
    rng = np.random.default_rng(17)

    def family_alignment(rank):
        if rank in families:
            return families[rank]
        c = cand[cand.nonredundant_rank == rank].iloc[0]
        tt = [int(x) for x in c.tokens.split(";")]
        flen = [fam.at[t, "fam_len_med"] for t in tt]
        df = inst[inst.type_key == c.type_key].copy()
        df["lenok"] = [all(0.8 * f <= L <= 1.25 * f for L, f in zip(l, flen)) for l in df.lengths]
        df["comp"] = df.completeness.fillna(0)
        reps = man[(man.cls == "cognate") & (man.cand_rank == rank)]
        rep_gids = {int(r.rep): [int(x) for x in r.gids.split(";")] for _, r in reps.iterrows()}
        df["is_rep"] = df.gids.map(lambda g: any(list(g) == v for v in rep_gids.values()))
        df = df.sort_values(["is_rep", "n_x", "lenok", "comp"], ascending=[False, True, False, False])
        df = df.drop_duplicates("genome_code")
        keys = [f"{g[0]}_{g[1]}" for g in df.gids]
        A = {k: seqs[str(g[0])] for k, g in zip(keys, df.gids)}
        B = {k: seqs[str(g[1])] for k, g in zip(keys, df.gids)}
        alnA, alnB = mafft(A, f"r{rank:03d}_A"), mafft(B, f"r{rank:03d}_B")
        families[rank] = (keys, alnA, alnB, rep_gids, A, B)
        return families[rank]

    for rank in ranks:
        keys, alnA, alnB, rep_gids, A, B = family_alignment(rank)
        for rep, (ga, gb) in rep_gids.items():
            qk = f"{ga}_{gb}"
            order = [qk] + [k for k in keys if k != qk]
            ra = project(alnA, order, qk, A[qk])
            rb = project(alnB, order, qk, B[qk])
            perm = [0] + list(rng.permutation(np.arange(1, len(rb))))
            rb_shuf = [rb[i] for i in perm]
            jobs.append(dict(fold_id=f"msa_cog_{rank:03d}_r{rep}", cls="msa_cognate", chains=[
                dict(id="A", seq=A[qk], gid=ga, msa=ra), dict(id="B", seq=B[qk], gid=gb, msa=rb)]))
            jobs.append(dict(fold_id=f"msa_shuf_{rank:03d}_r{rep}", cls="msa_shuffled", chains=[
                dict(id="A", seq=A[qk], gid=ga, msa=ra), dict(id="B", seq=B[qk], gid=gb, msa=rb_shuf)]))
            jobs.append(dict(fold_id=f"mono_{rank:03d}_r{rep}_A", cls="monomer", chains=[dict(id="A", seq=A[qk], gid=ga)]))
            jobs.append(dict(fold_id=f"mono_{rank:03d}_r{rep}_B", cls="monomer", chains=[dict(id="A", seq=B[qk], gid=gb)]))
            # non-cognate with MSAs: the screen's control for this replicate
            nc = man[(man.cls == "noncognate") & (man.cand_rank == rank) & (man.rep == rep)].iloc[0]
            pr = int(nc.partner_rank)
            pk, palnA, palnB, prep, pA, pB = family_alignment(pr)
            pga, pgb = prep[rep]
            pqk = f"{pga}_{pgb}"
            porder = [pqk] + [k for k in pk if k != pqk]
            if rep == 1:     # chain 1 kept, chain 2 from partner's chain 2
                c1, c2 = ra, project(palnB, porder, pqk, pB[pqk])
                s1, s2, g1, g2 = A[qk], pB[pqk], ga, pgb
            else:            # chain 2 kept, chain 1 from partner's chain 1
                c1, c2 = project(palnA, porder, pqk, pA[pqk]), rb
                s1, s2, g1, g2 = pA[pqk], B[qk], pga, gb
            assert [g1, g2] == [int(x) for x in nc.gids.split(";")], "non-cognate gids differ from the screen"
            d = min(len(c1), len(c2))
            jobs.append(dict(fold_id=f"msa_non_{rank:03d}_r{rep}", cls="msa_noncognate", chains=[
                dict(id="A", seq=s1, gid=g1, msa=c1[:d]), dict(id="B", seq=s2, gid=g2, msa=c2[:d])]))
            # Protenix v2 input (MSA files staged on biotite under RD_REMOTE)
            name = f"cofold_{rank:03d}_r{rep}"
            pdir = OUT / "protenix" / name
            pdir.mkdir(parents=True, exist_ok=True)
            for ch, rows in (("A", ra), ("B", rb)):
                with open(pdir / f"{ch}_paired.a3m", "w") as f:
                    for k, r in zip(order, rows):
                        f.write(f">{k}\n{r}\n")
            prot.append(dict(name=name, modelSeeds=[101, 202], sequences=[
                {"proteinChain": {"sequence": A[qk], "count": 1, "pairedMsaPath": f"{RD_REMOTE}/{name}/A_paired.a3m",
                                  "unpairedMsaPath": f"{RD_REMOTE}/{name}/A_paired.a3m"}},
                {"proteinChain": {"sequence": B[qk], "count": 1, "pairedMsaPath": f"{RD_REMOTE}/{name}/B_paired.a3m",
                                  "unpairedMsaPath": f"{RD_REMOTE}/{name}/B_paired.a3m"}}]))
            print(f"rank {rank} rep {rep}: depth {len(ra)} rows; partner {pr}", flush=True)
    with open(W / "jobs_msa.jsonl", "w") as f:
        for j in jobs:
            f.write(json.dumps(j) + "\n")
    json.dump(prot, open(OUT / "protenix" / "protenix_inputs.json", "w"), indent=1)
    print(len(jobs), "jobs;", len(prot), "protenix entries")


if __name__ == "__main__":
    main([int(x) for x in sys.argv[1:]])
