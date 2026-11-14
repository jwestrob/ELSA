"""Screen 2 split-gene / fused-homolog check per interface (rule of split_gene_check.py, screen 1).

Input: work/s2/split/hits.tsv (s2_splitgene.sbatch: MMseqs2 release 18, -s 7.5, E <= 1e-3, --max-seqs 10000;
the chains of the screened candidates' replicates against all 3,585,303 DPANN proteins).
For candidate interface (i, j): a fused homolog is a target, outside every instance of the block type, hit by a
chain-i query and a chain-j query (either replicate), each covering >= 50 % of its query, segments
overlapping by <= 10 % of the shorter. Artefact if fused homologs occur in >= 25 % as many genomes as the
block or in the genus of a replicate. Writes work/s2/s2_split_check.tsv.
Usage: python s2_split_check.py [--all]   (default: interfaces passing in both replicates; --all: every interface)
"""
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
W = HERE / "work"
W2 = W / "s2"
D = W / "data"


def main(all_ifaces):
    h = pd.read_csv(W2 / "split" / "hits.tsv", sep="\t", header=None,
                    names=["q", "t", "evalue", "bits", "pident", "qs", "qe", "ql", "ts", "te", "tl"])
    h = h[h.evalue <= 1e-3]
    h["qcov"] = (h.qe - h.qs + 1) / h.ql
    hh = h[h.qcov >= 0.5]
    g = pd.read_parquet(W / "genes_dark.parquet", columns=["gid", "genome_code", "dark", "length"]).set_index("gid")
    gm = pd.read_parquet(D / "genomes.parquet").set_index("genome_code")
    man = pd.read_csv(W2 / "fold_manifest2.tsv", sep="\t")
    pool = pd.read_csv(W2 / "pool.tsv", sep="\t").set_index("s2_rank")
    calls = pd.read_csv(W2 / "s2_calls.tsv", sep="\t")
    calls = calls.sort_values("pass_both", ascending=False).drop_duplicates(["s2_rank", "i", "j"])
    if not all_ifaces:
        calls = calls[calls.pass_both]
    inst = pd.read_parquet(W2 / "block_instances.parquet", columns=["type_key", "gids"])
    own_by_type = inst.groupby("type_key").gids.agg(lambda s: {int(x) for gl in s for x in gl})
    rows = []
    for _, c in calls.iterrows():
        cg = man[(man.cls == "cognate") & (man.s2_rank == c.s2_rank)].sort_values("rep")
        reps = [[int(x) for x in s.split(";")] for s in cg.gids]
        A = {r[c.i] for r in reps}
        B = {r[c.j] for r in reps}
        own = own_by_type.get(c.type_key, set())
        ha = hh[hh.q.isin(A) & ~hh.t.isin(own)]
        hb = hh[hh.q.isin(B) & ~hh.t.isin(own)]
        fused = []
        for t in set(ha.t) & set(hb.t):
            sa = ha[ha.t == t].sort_values("bits", ascending=False).iloc[0]
            sb = hb[hb.t == t].sort_values("bits", ascending=False).iloc[0]
            ov = max(0, min(sa.te, sb.te) - max(sa.ts, sb.ts) + 1)
            if ov <= 0.1 * min(sa.te - sa.ts + 1, sb.te - sb.ts + 1):
                fused.append(t)
        fg = {g.at[t, "genome_code"] for t in fused}
        f_gen = {gm.at[x, "genus"] for x in fg}
        r_gen = {gm.at[g.at[r[0], "genome_code"], "genus"] for r in reps}
        nb = int(pool.at[c.s2_rank, "n_genomes"])
        rows.append(dict(s2_rank=c.s2_rank, i=c.i, j=c.j, block_genomes=nb, fused_targets=len(fused), fused_genomes=len(fg),
                         fused_genome_ratio=round(len(fg) / nb, 4), fused_same_genus=bool(f_gen & r_gen),
                         split_gene_artefact=bool(len(fg) >= 0.25 * nb or (f_gen & r_gen)),
                         fused_lengths=";".join(str(int(g.at[t, "length"])) for t in fused[:10]),
                         homologs_i_outside=ha.t.nunique(), homologs_i_annotated=int((~g.loc[list(set(ha.t)), "dark"]).sum()) if len(ha) else 0,
                         homologs_j_outside=hb.t.nunique(), homologs_j_annotated=int((~g.loc[list(set(hb.t)), "dark"]).sum()) if len(hb) else 0))
    out = pd.DataFrame(rows)
    out.to_csv(W2 / "s2_split_check.tsv", sep="\t", index=False)
    print(out.to_string())


if __name__ == "__main__":
    main("--all" in sys.argv)
