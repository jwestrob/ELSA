"""Distribution, copy number and geometry (local, seconds).

Presence definitions per genome (3,564 DPANN genomes, GTDB r232):
  block   an engine block instance of the type (A immediately followed by B in transcription order)
  tokA / tokB   any member of the engine token
  hmmA / hmmB   any protein hit by the family HMM (hmmsearch, full-sequence E <= 1e-5, built from the token
                members' mafft alignment), i.e. a token-independent member set
  pairHMM       an hmmA gene and an hmmB gene adjacent on one contig
Completeness correction: expected presence if every genome carrying the block in a lineage were complete is
estimated as n_present / sum(completeness/100) over the lineage's genomes (capped at 1), and presence is also
reported for genomes >= 90 % complete.
Geometry over every adjacent hmmA-hmmB pair: co-orientation, transcription order, overlap (nt) between the
upstream stop and the downstream start.
Usage: python dd_distribution.py c1 | c2
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
W = HERE / "work"
CW = HERE.parent / "work"


def read_domtbl(p, evalue=1e-5):
    rows = []
    for line in open(p):
        if line.startswith("#"):
            continue
        f = line.split()
        rows.append(dict(gid=int(f[0]), tlen=int(f[2]), hmm=f[3], qlen=int(f[5]), E=float(f[6]), score=float(f[7]),
                         hmm_from=int(f[15]), hmm_to=int(f[16]), env_from=int(f[19]), env_to=int(f[20])))
    d = pd.DataFrame(rows)
    return d[d.E <= evalue]


def lineage_table(gm, present, level, extra_cols):
    rows = []
    for name, d in gm.groupby(level):
        r = dict(level=level, lineage=name, genomes=len(d), complete90=int((d.completeness >= 90).sum()),
                 sum_completeness=round(float((d.completeness / 100).sum()), 1))
        for c in extra_cols:
            n = int(d[c].sum())
            r[f"n_{c}"] = n
            r[f"frac_{c}"] = round(n / len(d), 3)
            c90 = d[d.completeness >= 90]
            r[f"frac90_{c}"] = round(float(c90[c].mean()), 3) if len(c90) else np.nan
            r[f"corr_{c}"] = round(min(1.0, n / max(1e-9, (d.completeness / 100).sum())), 3)
        rows.append(r)
    return pd.DataFrame(rows)


def c1():
    g = pd.read_parquet(CW / "genes_dark.parquet")
    gm = pd.read_parquet(CW / "data/genomes.parquet")
    inst = pd.read_csv(W / "c1_instances.tsv", sep="\t")
    h = read_domtbl(W / "hmm/c1_vs_dpann.domtbl")
    best = h.sort_values("score", ascending=False).drop_duplicates("gid")
    ga = set(best.gid[best.hmm == "c1A"])
    gb = set(best.gid[best.hmm == "c1B"])
    g["hmmA"] = g.gid.isin(ga)
    g["hmmB"] = g.gid.isin(gb)
    g["tokA"] = g.token == 81718
    g["tokB"] = g.token == 393054
    # adjacent hmmA-hmmB pairs (same contig, consecutive positions)
    nxt = g.shift(-1)
    same = (g.genome_code == nxt.genome_code) & (g.contig_code == nxt.contig_code)
    pairs = []
    for i in np.flatnonzero(same.values & ((g.hmmA.values & nxt.hmmB.values.astype(bool))
                                           | (g.hmmB.values & nxt.hmmA.values.astype(bool)))):
        x, y = g.iloc[i], g.iloc[i + 1]
        a, b = (x, y) if x.hmmA else (y, x)
        co = x.strand == y.strand
        if co:
            up, dn = (x, y) if x.strand > 0 else (y, x)
            order = "A_first" if up.gid == a.gid else "B_first"
            ov = int(up.end - dn.start + 1) if x.strand > 0 else int(dn.end - up.start + 1)
        else:
            order, ov = "opposite", np.nan
        pairs.append(dict(genome_code=x.genome_code, gidA=int(a.gid), gidB=int(b.gid), co_oriented=bool(co),
                          order=order, overlap_nt=ov, lenA=int(a.length), lenB=int(b.length)))
    P = pd.DataFrame(pairs)
    P.to_csv(W / "c1_hmm_pairs.tsv", sep="\t", index=False)
    per = g.groupby("genome_code").agg(tokA=("tokA", "sum"), tokB=("tokB", "sum"), hmmA=("hmmA", "sum"),
                                       hmmB=("hmmB", "sum"))
    blk = inst.groupby("genome_code").size().rename("block")
    per = per.join(blk).fillna({"block": 0})
    per = per.join(P.groupby("genome_code").size().rename("pairHMM")).fillna({"pairHMM": 0})
    gm = gm.set_index("genome_code").join(per)
    for c in ("block", "tokA", "tokB", "hmmA", "hmmB", "pairHMM"):
        gm[f"has_{c}"] = gm[c] > 0
    cols = ["has_block", "has_tokA", "has_tokB", "has_hmmA", "has_hmmB", "has_pairHMM"]
    T = pd.concat([lineage_table(gm, None, lv, cols) for lv in ("phylum", "class", "ord")])
    T.to_csv(W / "c1_distribution.tsv", sep="\t", index=False)
    # completeness bins over the phyla that carry it
    gm["cbin"] = pd.cut(gm.completeness, [0, 50, 70, 90, 100.01], right=False,
                        labels=["<50", "50-70", "70-90", ">=90"])
    CB = gm.groupby(["phylum", "cbin"], observed=True).agg(genomes=("has_block", "size"),
                                                           block=("has_block", "mean"),
                                                           pairHMM=("has_pairHMM", "mean")).round(3)
    CB.to_csv(W / "c1_completeness_bins.tsv", sep="\t")
    # copy number
    cp = gm[gm.has_block | gm.has_pairHMM]
    out = dict(
        genomes_total=len(gm), genomes_block=int(gm.has_block.sum()), genomes_pairHMM=int(gm.has_pairHMM.sum()),
        genomes_tokA=int(gm.has_tokA.sum()), genomes_tokB=int(gm.has_tokB.sum()), genomes_hmmA=int(gm.has_hmmA.sum()),
        genomes_hmmB=int(gm.has_hmmB.sum()),
        genes_hmmA=len(ga), genes_hmmB=len(gb), genes_tokA=int(g.tokA.sum()), genes_tokB=int(g.tokB.sum()),
        hmmA_in_tokA=len(ga & set(g.gid[g.tokA])), hmmB_in_tokB=len(gb & set(g.gid[g.tokB])),
        block_copies=gm.loc[gm.has_block, "block"].value_counts().sort_index().to_dict(),
        pairHMM_copies=gm.loc[gm.has_pairHMM, "pairHMM"].value_counts().sort_index().to_dict(),
        hmmA_copies=gm.loc[gm.has_hmmA, "hmmA"].value_counts().sort_index().to_dict(),
        hmmB_copies=gm.loc[gm.has_hmmB, "hmmB"].value_counts().sort_index().to_dict(),
        pairs=len(P), pairs_co_oriented=int(P.co_oriented.sum()),
        pairs_order=P.order.value_counts().to_dict(),
        pairs_overlap_gt0=int((P.overlap_nt > 0).sum()), pairs_overlap_4=int((P.overlap_nt == 4).sum()),
        pairs_overlap_1=int((P.overlap_nt == 1).sum()), overlap_dist=P.overlap_nt.value_counts().head(12).to_dict(),
        # among genomes with both an hmmA and an hmmB gene, how many have them adjacent
        genomes_both_hmm=int((gm.has_hmmA & gm.has_hmmB).sum()),
        genomes_both_hmm_adjacent=int((gm.has_hmmA & gm.has_hmmB & gm.has_pairHMM).sum()),
        hmmA_genes_adjacent_to_hmmB=int(P.gidA.nunique()), hmmB_genes_adjacent_to_hmmA=int(P.gidB.nunique()),
        engine_instances=len(inst), engine_instance_overlap4=int((inst.overlaps_nt.astype(str) == "4").sum()),
    )
    lone = g[g.hmmA & ~g.gid.isin(P.gidA)]
    out["hmmA_unpaired_genes"] = len(lone)
    out["hmmA_unpaired_contig_edge_frac"] = None
    # are unpaired A genes at contig edges? (B could be off the contig end)
    cn = g.groupby(["genome_code", "contig_code"]).position_index.max().rename("maxpos")
    lone = lone.join(cn, on=["genome_code", "contig_code"])
    out["hmmA_unpaired_at_edge"] = int(((lone.position_index == 0) | (lone.position_index == lone.maxpos)).sum())
    import json
    json.dump(out, open(W / "c1_distribution_summary.json", "w"), indent=1, default=str)
    print(json.dumps(out, indent=1, default=str))
    ph = T[T.level == "phylum"].sort_values("n_has_block", ascending=False)
    print(ph[["lineage", "genomes", "complete90", "n_has_block", "frac_has_block", "frac90_has_block",
              "corr_has_block", "n_has_pairHMM", "frac_has_pairHMM", "frac90_has_pairHMM"]].to_string())
    od = T[(T.level == "ord") & (T.n_has_pairHMM > 0)].sort_values("frac_has_pairHMM", ascending=False)
    print(od[["lineage", "genomes", "complete90", "n_has_block", "frac_has_block", "n_has_pairHMM",
              "frac_has_pairHMM", "frac90_has_pairHMM", "corr_has_pairHMM"]].to_string())
    print(CB.to_string())


def c2():
    """Candidate 2. Family HMMs built from the members of the 8 cassette tokens (mafft, hmmbuild) searched against
    all DPANN proteins; each protein assigned to its best-scoring HMM (full-sequence E <= 1e-5). Groups:
    G7 (DUF7287-like by screen-2 HHsearch: tokens 15180, 287285, 55170), G8 (DUF7288-like: 354137, 410168,
    142248), GD (~270-aa partner: 96948, 55171). A cassette = a G7 gene and a G8 gene <= 2 genes apart on one
    contig, same strand; a full cassette additionally has a GD gene within the 6-gene window around them.
    Engine block presence (screen-2 types of ranks 34, 33, 11, 7) is reported alongside."""
    g = pd.read_parquet(CW / "genes_dark.parquet")
    gm = pd.read_parquet(CW / "data/genomes.parquet")
    inst = pd.read_csv(W / "c2_instances.tsv", sep="\t")
    h = read_domtbl(W / "hmm/c2/c2_tokens_vs_dpann.domtbl")
    best = h.sort_values("score", ascending=False).drop_duplicates("gid")
    grp = {"tok15180": "G7", "tok287285": "G7", "tok55170": "G7", "tok354137": "G8", "tok410168": "G8",
           "tok142248": "G8", "tok96948": "GD", "tok55171": "GD"}
    best["grp"] = best.hmm.map(grp)
    g = g.merge(best[["gid", "grp", "hmm"]], on="gid", how="left")
    g = g.sort_values("gid").reset_index(drop=True)
    G = g.grp.fillna("").values
    gc, cc, st = g.genome_code.values, g.contig_code.values, g.strand.values
    cas, full = [], []
    for i in np.flatnonzero(G == "G7"):
        for j in range(max(0, i - 2), min(len(g), i + 3)):
            if j != i and G[j] == "G8" and gc[j] == gc[i] and cc[j] == cc[i] and st[j] == st[i]:
                lo, hi = min(i, j), max(i, j)
                win = [k for k in range(max(0, lo - 4), min(len(g), hi + 5)) if gc[k] == gc[i] and cc[k] == cc[i]]
                hasD = any(G[k] == "GD" for k in win)
                cas.append(dict(genome_code=gc[i], g7=int(g.gid[i]), g8=int(g.gid[j]), full=hasD))
    C = pd.DataFrame(cas).drop_duplicates(["g7", "g8"])
    per = C.groupby("genome_code").agg(cassette=("g7", "size"), full=("full", "sum"))
    for k, v in (("G7", "nG7"), ("G8", "nG8"), ("GD", "nGD")):
        per = per.join(g[g.grp == k].groupby("genome_code").size().rename(v), how="outer")
    blk = inst[inst.cand.isin(["s34", "s33", "s11"])].groupby("genome_code").size().rename("block_nano")
    blk7 = inst[inst.cand == "s7"].groupby("genome_code").size().rename("block_s7")
    per = per.join(blk, how="outer").join(blk7, how="outer").fillna(0)
    gm = gm.set_index("genome_code").join(per).fillna({c: 0 for c in per.columns})
    for c in ("cassette", "full", "nG7", "nG8", "nGD", "block_nano", "block_s7"):
        gm[f"has_{c}"] = gm[c] > 0
    cols = ["has_cassette", "has_full", "has_nG7", "has_nG8", "has_nGD", "has_block_nano", "has_block_s7"]
    T = pd.concat([lineage_table(gm, None, lv, cols) for lv in ("phylum", "class", "ord")])
    T.to_csv(W / "c2_distribution.tsv", sep="\t", index=False)
    gm["cbin"] = pd.cut(gm.completeness, [0, 50, 70, 90, 100.01], right=False, labels=["<50", "50-70", "70-90", ">=90"])
    CB = gm.groupby(["phylum", "cbin"], observed=True).agg(genomes=("has_cassette", "size"),
                                                           cassette=("has_cassette", "mean"),
                                                           full=("has_full", "mean")).round(3)
    CB.to_csv(W / "c2_completeness_bins.tsv", sep="\t")
    import json
    out = dict(genes_G7=int((g.grp == "G7").sum()), genes_G8=int((g.grp == "G8").sum()), genes_GD=int((g.grp == "GD").sum()),
               cassettes=len(C), cassettes_full=int(C.full.sum()), genomes_cassette=int(gm.has_cassette.sum()),
               genomes_full=int(gm.has_full.sum()), genomes_block_nano=int(gm.has_block_nano.sum()),
               genomes_block_s7=int(gm.has_block_s7.sum()),
               cassette_copies=gm.loc[gm.has_cassette, "cassette"].value_counts().sort_index().to_dict())
    json.dump(out, open(W / "c2_distribution_summary.json", "w"), indent=1, default=str)
    print(json.dumps(out, indent=1, default=str))
    ph = T[T.level == "phylum"].sort_values("n_has_cassette", ascending=False)
    print(ph[["lineage", "genomes", "complete90", "n_has_cassette", "frac_has_cassette", "frac90_has_cassette",
              "corr_has_cassette", "n_has_full", "frac_has_full", "n_has_nG7", "n_has_block_nano", "n_has_block_s7"]].to_string())
    od = T[(T.level == "ord") & (T.n_has_cassette > 0) & (T.genomes >= 10)].sort_values("frac_has_cassette", ascending=False)
    print(od[["lineage", "genomes", "complete90", "n_has_cassette", "frac_has_cassette", "frac90_has_cassette",
              "corr_has_cassette", "frac_has_full"]].head(20).to_string())
    print(CB.to_string())


if __name__ == "__main__":
    {"c1": c1, "c2": c2}[sys.argv[1]]()
