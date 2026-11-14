#!/usr/bin/env python
"""ws_showcase2 split-gene / fused-homolog check per interface (census rule, s3_split_check.py, unchanged; biotite
login node, seconds). Input: post/<dir>/split/hits.tsv (MMseqs2 18, -s 7.5, E <= 1e-3, --max-seqs 10000; the chains of
both replicates of blocks with an interface passing both replicates vs all 3,585,303 DPANN proteins).
For interface (i, j): a fused homolog is a target outside every realisation of the candidate, hit by a chain-i query
and a chain-j query (either replicate), each covering >= 50 % of its query, segments overlapping by <= 10 % of the
shorter. Artefact if fused homologs occur in >= 25 % as many genomes as the candidate's realisations or in the genus
of a replicate. Usage: sc2_split_check.py POST_DIR; writes POST_DIR/split_check.tsv."""
import sys
from pathlib import Path

import pandas as pd

RD = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111")
S = RD / "showcase2"


def main(d):
    d = Path(d)
    h = pd.read_csv(d / "split/hits.tsv", sep="\t", header=None,
                    names=["q", "t", "evalue", "bits", "pident", "qs", "qe", "ql", "ts", "te", "tl"])
    h = h[h.evalue <= 1e-3].drop_duplicates(["q", "t", "qs", "ts"])
    h["qcov"] = (h.qe - h.qs + 1) / h.ql
    hh = h[h.qcov >= 0.5]
    g = pd.read_parquet(RD / "ws_dpann/package/universe/genes.parquet", columns=["gid", "genome_code", "length"]).set_index("gid")
    gm = pd.read_parquet(RD / "ws_dpann/package/universe/genomes.parquet").set_index("genome_code")
    reps = pd.read_csv(S / "data/reps.tsv", sep="\t")
    cand = pd.read_csv(S / "data/candidates.tsv", sep="\t").set_index("rank")
    R = pd.read_parquet(S / "data/realisations.parquet", columns=["cand", "gids"])
    own_by = R.groupby("cand").gids.agg(lambda s: {int(x) for gl in s for x in gl.split(";")})
    calls = pd.read_csv(d / "calls.tsv", sep="\t")
    calls = calls[calls.pass_both]
    rows = []
    for _, c in calls.iterrows():
        cn = cand.at[c["rank"], "cand"]
        rr = reps[reps.cand == cn].sort_values("rep")
        gl = [[int(x) for x in s.split(";")] for s in rr.gids]
        Aq = {r[c.i] for r in gl}
        Bq = {r[c.j] for r in gl}
        own = own_by.get(cn, set()) | {x for r in gl for x in r}
        ha = hh[hh.q.isin(Aq) & ~hh.t.isin(own)]
        hb = hh[hh.q.isin(Bq) & ~hh.t.isin(own)]
        fused = []
        for t in set(ha.t) & set(hb.t):
            sa = ha[ha.t == t].sort_values("bits", ascending=False).iloc[0]
            sb = hb[hb.t == t].sort_values("bits", ascending=False).iloc[0]
            ov = max(0, min(sa.te, sb.te) - max(sa.ts, sb.ts) + 1)
            if ov <= 0.1 * min(sa.te - sa.ts + 1, sb.te - sb.ts + 1):
                fused.append(t)
        fg = {g.at[t, "genome_code"] for t in fused}
        f_gen = {gm.at[x, "genus"] for x in fg}
        r_gen = {gm.at[g.at[r[0], "genome_code"], "genus"] for r in gl}
        nb = int(cand.at[c["rank"], "genomes"])
        rows.append(dict(rank=int(c["rank"]), cand=cn, i=int(c.i), j=int(c.j), block_genomes=nb, fused_targets=len(fused),
                         fused_genomes=len(fg), fused_genome_ratio=round(len(fg) / nb, 4), fused_same_genus=bool(f_gen & r_gen),
                         split_gene_artefact=bool(len(fg) >= 0.25 * nb or (f_gen & r_gen)),
                         fused_lengths=";".join(str(int(g.at[t, "length"])) for t in fused[:10]),
                         homologs_i_outside=ha.t.nunique(), homologs_j_outside=hb.t.nunique()))
    out = pd.DataFrame(rows)
    out.to_csv(d / "split_check.tsv", sep="\t", index=False)
    pd.set_option("display.width", 250)
    print(out.to_string())


if __name__ == "__main__":
    main(sys.argv[1])
