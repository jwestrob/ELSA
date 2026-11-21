"""Split-gene / fused-homolog check and homolog annotation status for the cognate chains.

Input: work/split/hits.tsv from splitgene_search.sbatch (MMseqs2 release 18, -s 7.5, E <= 1e-3,
--max-seqs 10000; 600 cognate chains against all 3,585,303 DPANN proteins).
A fused homolog of candidate (a, b) is a target protein, other than any protein of a block instance of
the candidate, hit by an a-query and a b-query (either replicate) with each hit covering >= 50 % of
its query, E <= 1e-3, and the two target segments overlapping by <= 10 % of the shorter segment.
Order: whether the a-segment precedes the b-segment on the target, against the transcription order of
the pair in the replicate genomes.
Homolog annotation: among targets hit by a query at >= 50 % query coverage, how many are annotated
(not dark), with the Pfam accessions observed on them (observations, not names for the candidate).
Writes work/split_gene_check.tsv.
"""
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
W = HERE / "work"
D = W / "data"
SHARUR = "/Users/jacob/Documents/Sandbox/Sharur/Sharur/data/dpann/sharur.duckdb"


def main():
    h = pd.read_csv(W / "split" / "hits.tsv", sep="\t", header=None,
                    names=["q", "t", "evalue", "bits", "pident", "qs", "qe", "ql", "ts", "te", "tl"])
    h = h[(h.evalue <= 1e-3)]
    h["qcov"] = (h.qe - h.qs + 1) / h.ql
    g = pd.read_parquet(W / "genes_dark.parquet", columns=["gid", "protein_id", "genome_code", "strand", "position_index", "dark", "length"])
    gm = pd.read_parquet(D / "genomes.parquet").set_index("genome_code")
    man = pd.read_csv(W / "fold_manifest.tsv", sep="\t")
    cog = man[man.cls == "cognate"]
    cand = pd.read_csv(W / "candidates_all.tsv", sep="\t").set_index("type_key")
    inst = pd.read_parquet(W / "block_instances.parquet", columns=["type_key", "gids"])
    inst_gids = inst.groupby("type_key").gids.agg(lambda s: {int(x) for gl in s for x in gl})
    sh = duckdb.connect(SHARUR, read_only=True)
    sh.execute("SET threads=1")
    hh = h[h.qcov >= 0.5]
    tpid = g.set_index("gid").protein_id.reindex(hh.t.unique())
    sh.register("tp", pd.DataFrame({"protein_id": tpid.dropna().values}))
    pf = sh.execute("SELECT a.protein_id, a.accession, a.name FROM annotations a JOIN tp USING (protein_id) "
                    "WHERE a.source='pfam'").df()
    pf_by_pid = pf.groupby("protein_id").name.agg(lambda s: sorted(set(s)))
    gi = g.set_index("gid")
    rows = []
    for (rank, tk), d in cog.groupby(["cand_rank", "type_key"]):
        reps = [[int(x) for x in s.split(";")] for s in d.sort_values("rep").gids]
        A = {r[0] for r in reps}
        B = {r[1] for r in reps}
        own = inst_gids.get(tk, set())
        ha = hh[hh.q.isin(A) & ~hh.t.isin(own)]
        hb = hh[hh.q.isin(B) & ~hh.t.isin(own)]
        both = set(ha.t) & set(hb.t)
        fused = []
        for t in both:
            sa = ha[ha.t == t].sort_values("bits", ascending=False).iloc[0]
            sb = hb[hb.t == t].sort_values("bits", ascending=False).iloc[0]
            ov = max(0, min(sa.te, sb.te) - max(sa.ts, sb.ts) + 1)
            short = min(sa.te - sa.ts + 1, sb.te - sb.ts + 1)
            if ov <= 0.1 * short:
                fused.append((t, "a_first" if sa.ts < sb.ts else "b_first"))
        # transcription order in the replicate genomes: a upstream if (strand + and pos_a < pos_b) or (strand - and pos_a > pos_b)
        tx = []
        for a_, b_ in reps:
            s = gi.at[a_, "strand"]
            up_a = (gi.at[a_, "position_index"] < gi.at[b_, "position_index"]) == (s > 0)
            tx.append("a_first" if up_a else "b_first")
        fg = [gi.at[t, "genome_code"] for t, _ in fused]
        f_genera = {gm.at[x, "genus"] for x in fg}
        rep_genera = {gm.at[gi.at[r[0], "genome_code"], "genus"] for r in reps}
        nb = int(cand.at[tk, "n_genomes"])
        homA = set(ha.t)
        homB = set(hb.t)
        annA = [t for t in homA if not gi.at[t, "dark"]]
        annB = [t for t in homB if not gi.at[t, "dark"]]
        pfA = pd.Series([n for t in annA for n in pf_by_pid.get(gi.at[t, "protein_id"], [])]).value_counts()
        pfB = pd.Series([n for t in annB for n in pf_by_pid.get(gi.at[t, "protein_id"], [])]).value_counts()
        rows.append(dict(cand_rank=rank, type_key=tk, block_genomes=nb, transcription_order=";".join(tx),
                         fused_targets=len(fused), fused_genomes=len(set(fg)),
                         fused_order=";".join(sorted({o for _, o in fused})),
                         fused_same_genus_as_replicate=bool(f_genera & rep_genera),
                         fused_genome_ratio=round(len(set(fg)) / nb, 4),
                         split_gene_artefact=bool(len(set(fg)) >= 0.25 * nb or (f_genera & rep_genera)),
                         fused_lengths=";".join(str(int(gi.at[t, "length"])) for t, _ in fused[:10]),
                         homologs_A_outside_block=len(homA), homologs_A_annotated=len(annA),
                         homologs_B_outside_block=len(homB), homologs_B_annotated=len(annB),
                         pfam_on_homologs_A=";".join(f"{k}:{v}" for k, v in pfA.head(5).items()),
                         pfam_on_homologs_B=";".join(f"{k}:{v}" for k, v in pfB.head(5).items())))
    out = pd.DataFrame(rows).sort_values("cand_rank")
    out.to_csv(W / "split_gene_check.tsv", sep="\t", index=False)
    print(out[["cand_rank", "block_genomes", "fused_targets", "fused_genomes", "split_gene_artefact",
               "homologs_A_outside_block", "homologs_A_annotated", "homologs_B_outside_block", "homologs_B_annotated"]].head(40).to_string())
    print("split-gene artefacts:", int(out.split_gene_artefact.sum()), "of", len(out),
          "; candidates with any fused homolog:", int((out.fused_targets > 0).sum()),
          "; with annotated homologs (A or B):", int(((out.homologs_A_annotated > 0) | (out.homologs_B_annotated > 0)).sum()))


if __name__ == "__main__":
    main()
