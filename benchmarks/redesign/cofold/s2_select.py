"""Screen 2, step 2: replicate instances for every pool candidate, and the family table for triage.

Replicates: exactly the screen-1 rule (select_folds.py), generalised to k chains: instance quality
4*(no X) + 2*(every chain hydrophobic fraction <= 0.45) + 2*(every length within 0.8-1.25 x family
median) + 1*(not at a contig edge) + 1*(Prodigal partial 00) + completeness/100; candidates are the
top 60 by quality plus the best instance of every order; the pair of instances from different
genomes maximising (taxonomic distance, min quality, summed quality) becomes replicates 1 and 2.
Chains are in canonical type order (build_candidates), which is gene order on the contig.

Family table (work/s2/families.tsv): every token of the pool (screen 2) and of the Part A set
(screen-1 dossier ranks 27, 13 + family 164787, 67, 38, 106, 71, 6, 14, 9, 1 and the four high-scoring
uncalled pairs). Query sequence of a family = its replicate-1 chain (screen 2), the screen-1
replicate-1 chain (Part A), or the uncalled gene itself. Members: every DPANN gene of the token,
identical sequences collapsed; above 250 unique sequences, one per genome first then random
(seed 17) to 250. Writes work/s2/triage/fasta/fam_<token>.faa (query first).
"""
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from select_folds import taxdist  # noqa: E402

W = HERE / "work"
W2 = W / "s2"
D = W / "data"
SHARUR = "/Users/jacob/Documents/Sandbox/Sharur/Sharur/data/dpann/sharur.duckdb"
TAX = ["phylum", "class", "ord", "family", "genus"]
CAP = 250
PART_A = {27: "100239-199393", 13: "221251-351854", 67: "229034-423021", 38: "23083-68571", 106: "27009-27010",
          71: "171614-322104", 6: "322564-326340", 14: "48264-404027", 9: "114055-318532", 1: "81718-393054"}
UNCALLED = ["unc_002_r1", "unc_069_r2", "unc_071_r2", "unc_087_r1"]


def replicates(pool, inst, fam):
    rows = []
    for _, c in pool.iterrows():
        tt = [int(x) for x in str(c.tokens).split(";")]
        flen = [fam.at[t, "fam_len_med"] for t in tt]
        df = inst[inst.type_key == c.type_key].copy()
        q = []
        for _, r in df.iterrows():
            okx = r.n_x == 0
            okh = max(r.hyds) <= 0.45
            okl = all(0.8 * f <= L <= 1.25 * f for L, f in zip(r.lengths, flen))
            oke = r.edge_dist >= 1
            okp = all(p in ("00", "na") for p in r.partials)
            comp_ = 0 if pd.isna(r.completeness) else r.completeness
            q.append(4 * okx + 2 * okh + 2 * okl + 1 * oke + 1 * okp + comp_ / 100)
        df["quality"] = q
        df = df.sort_values("quality", ascending=False)
        df = pd.concat([df.head(60), df.drop_duplicates("ord")]).drop_duplicates(["genome_code", "pos_start"])
        recs = df.to_dict("records")
        best, bscore = None, None
        for i in range(len(recs)):
            for j in range(i + 1, len(recs)):
                if recs[i]["genome_code"] == recs[j]["genome_code"]:
                    continue
                s = (taxdist(recs[i], recs[j]), min(recs[i]["quality"], recs[j]["quality"]),
                     recs[i]["quality"] + recs[j]["quality"])
                if bscore is None or s > bscore:
                    best, bscore = (recs[i], recs[j]), s
        for rep, r in enumerate(best, 1):
            rows.append(dict(s2_rank=int(c.s2_rank), pool=c.pool, type_key=c.type_key, k=int(c.k), rep=rep,
                             gids=";".join(map(str, r["gids"])), genome=r["genome"], phylum=r["phylum"],
                             cls_=r["class"], ord=r["ord"], family=r["family"], genus=r["genus"],
                             completeness=r["completeness"], edge_dist=int(r["edge_dist"]),
                             quality=round(r["quality"], 3), replicate_taxdist=bscore[0],
                             lengths=";".join(map(str, r["lengths"])), hyds=";".join(f"{h:.3f}" for h in r["hyds"]),
                             partials=";".join(r["partials"]), n_x=int(r["n_x"]),
                             intergenic_nt=";".join(map(str, r["intergenic_nt"]))))
    return pd.DataFrame(rows).rename(columns={"cls_": "class"})


def main():
    pool = pd.read_csv(W2 / "pool.tsv", sep="\t")
    inst = pd.read_parquet(W2 / "block_instances.parquet")
    inst = inst[inst.type_key.isin(pool.type_key)]
    fam = pd.read_parquet(W2 / "fam_stats.parquet")
    reps = replicates(pool, inst, fam)
    reps.to_csv(W2 / "s2_reps.tsv", sep="\t", index=False)
    print("replicates", len(reps), "taxdist", reps.groupby("s2_rank").replicate_taxdist.first().value_counts().to_dict())

    # ---- family table -------------------------------------------------------------------
    g = pd.read_parquet(W / "genes_dark.parquet", columns=["gid", "protein_id", "genome_code", "token", "length"])
    fams = {}
    for _, r in reps[reps.rep == 1].iterrows():
        for t, gid in zip(r.type_key.split("-"), r.gids.split(";")):
            fams.setdefault(int(t), dict(query_gid=int(gid), sources=set()))["sources"].add(f"s2:{r.s2_rank}")
    man = pd.read_csv(W / "fold_manifest.tsv", sep="\t")
    for rk, tk in PART_A.items():
        row = man[(man.cls == "cognate") & (man.cand_rank == rk) & (man.rep == 1)].iloc[0]
        for t, gid in zip(tk.split("-"), row.gids.split(";")):
            e = fams.setdefault(int(t), dict(query_gid=int(gid), sources=set()))
            e["query_gid"] = int(gid)          # Part A: the folded screen-1 chain is the query
            e["sources"].add(f"A:{rk}")
    # rank 13 third gene family 164787: the chain folded in the triple (jobs_extra)
    e = fams.setdefault(164787, dict(query_gid=None, sources=set()))
    e["sources"].add("A:13C")
    if e["query_gid"] is None:
        e["query_gid"] = int(g[g.token == 164787].gid.iloc[0])
    for fid in UNCALLED:
        row = man[man.fold_id == fid].iloc[0]
        for gid in row.gids.split(";"):
            t = int(g.token.values[int(gid)])
            e = fams.setdefault(t, dict(query_gid=int(gid), sources=set()))
            e["query_gid"] = int(gid)
            e["sources"].add(f"U:{fid}")
    ft = pd.DataFrame([dict(token=t, query_gid=v["query_gid"], sources=";".join(sorted(v["sources"])))
                       for t, v in fams.items()]).sort_values("token")
    # members
    mem = g[g.token.isin(ft.token)]
    sh = duckdb.connect(SHARUR, read_only=True)
    sh.execute("SET threads=1")
    sh.register("m", mem[["protein_id"]])
    seq = dict(sh.execute("SELECT p.protein_id, rtrim(p.sequence,'*') FROM proteins p JOIN m USING (protein_id)").fetchall())
    mem = mem.assign(seq=mem.protein_id.map(seq))
    assert mem.seq.notna().all()
    out = W2 / "triage" / "fasta"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(17)
    nrow = []
    for _, f in ft.iterrows():
        m = mem[mem.token == f.token]
        qseq = m[m.gid == f.query_gid].seq.iloc[0] if (m.gid == f.query_gid).any() else \
            seq.get(g.protein_id.values[f.query_gid])
        u = m.drop_duplicates("seq")
        u = u[u.seq != qseq]
        if len(u) > CAP - 1:
            first = u.drop_duplicates("genome_code")
            if len(first) >= CAP - 1:
                u = first.iloc[rng.permutation(len(first))[:CAP - 1]]
            else:
                rest = u[~u.gid.isin(first.gid)]
                u = pd.concat([first, rest.iloc[rng.permutation(len(rest))[:CAP - 1 - len(first)]]])
        with open(out / f"fam_{f.token}.faa", "w") as fh:
            fh.write(f">q{f.query_gid}\n{qseq}\n")
            for gid, s in zip(u.gid, u.seq):
                fh.write(f">g{gid}\n{s}\n")
        nrow.append(dict(token=f.token, members=len(m), genomes=m.genome_code.nunique(), unique_seqs=m.seq.nunique(),
                         in_fasta=1 + len(u), query_len=len(qseq), len_med=float(m.seq.str.len().median())))
    ft = ft.merge(pd.DataFrame(nrow), on="token")
    ft.to_csv(W2 / "families.tsv", sep="\t", index=False)
    print(len(ft), "families;", ft.sources.str.contains("s2:").sum(), "from the screen-2 pool;",
          ft.sources.str.contains("A:|U:").sum(), "Part A")
    print(ft.describe().to_string())


if __name__ == "__main__":
    main()
