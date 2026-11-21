"""Deep dive, step 1 (local, seconds): instance tables, geometry, distribution and context windows for
candidate 1 (screen-1 rank 1, tokens 81718-393054) and candidate 2 (DUF7287/DUF7288 cassette, screen-2
ranks 34/33/11 in Nanobdellota and rank 7 in Micrarchaeota/Iainarchaeota).

Inputs (ws_cofold work/, built from the DPANN package and engine results_dpann_v5):
  genes_dark.parquet (gid, protein_id, genome_code, contig_code, position_index, strand, length, start, end,
  partial, dark, gold, token), data/genomes.parquet, data/token_modules.parquet,
  block_instances.parquet (screen 1) and s2/block_instances.parquet (screen 2).
Outputs (deepdive/work/):
  c1_instances.tsv, c2_instances.tsv      every engine block instance with geometry and taxonomy
  ctx_windows.tsv                          +-5 neighbours of every real instance and of up to 3 length-matched
                                           control cores per instance on the same contig (seed 17)
  seq_gids.txt                             gids whose sequences are pulled on biotite
  ctx_protein_ids.txt                      protein IDs whose annotations are pulled from Sharur on biotite
Control rule: a control core has the block's gene count k, starts at a gene on the same contig whose length is
within +-25 % of the block's first (transcription-order) gene, runs k genes in that gene's transcription
direction, and lies >= 6 genes from the real core (no core overlap, cores outside each other's +-5 window).
"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
W = HERE.parent / "work"
OUT = HERE / "work"
OUT.mkdir(exist_ok=True)
FLANK = 5
RNG = np.random.default_rng(17)

C1 = ["81718-393054"]
C2 = {"15180-287285-354137-96948": "s34", "15180-287285-142248-96948": "s33", "96948-354137-287285": "s11",
      "55170-410168-55171": "s7"}


def main():
    g = pd.read_parquet(W / "genes_dark.parquet")
    gm = pd.read_parquet(W / "data/genomes.parquet")
    tm = pd.read_parquet(W / "data/token_modules.parquet").set_index("token").module
    g["module"] = g.token.map(tm).fillna(-1).astype(int)
    # contig index: gids are contiguous per contig, ordered by position
    first = g.groupby(["genome_code", "contig_code"]).gid.min().rename("first_gid")
    ncont = g.groupby(["genome_code", "contig_code"]).gid.size().rename("n_genes")
    gi = g.set_index("gid")

    bi1 = pd.read_parquet(W / "block_instances.parquet")
    bi2 = pd.read_parquet(W / "s2/block_instances.parquet")
    c1 = bi1[bi1.type_key.isin(C1)].copy()
    c1["cand"] = "c1"
    c2 = bi2[bi2.type_key.isin(list(C2))].copy()
    c2["cand"] = c2.type_key.map(C2)

    rows = []
    for _, r in pd.concat([c1, c2]).iterrows():
        gids = list(r.gids)          # transcription order (the engine writes reversed blocks in transcription order)
        s = int(gi.at[gids[0], "strand"])
        pos = [int(gi.at[x, "position_index"]) for x in gids]
        st = [int(gi.at[x, "start"]) for x in gids]
        en = [int(gi.at[x, "end"]) for x in gids]
        # transcription order check: on + strand positions increase, on - strand they decrease
        tx_ok = all((b - a) == (1 if s > 0 else -1) for a, b in zip(pos, pos[1:]))
        ov = []
        for i in range(len(gids) - 1):
            if s > 0:
                ov.append(en[i] - st[i + 1] + 1)      # > 0: overlap in nt
            else:
                ov.append(en[i + 1] - st[i] + 1)
        rows.append(dict(cand=r.cand, type_key=r.type_key, gids=";".join(map(str, gids)),
                         proteins=";".join(gi.loc[gids, "protein_id"]), genome=r.genome, genome_code=r.genome_code,
                         contig_code=r.contig_code, strand=s, pos_min=min(pos), pos_max=max(pos), tx_order_ok=tx_ok,
                         same_strand=bool(r.same_strand), overlaps_nt=";".join(map(str, ov)),
                         lengths=";".join(str(int(x)) for x in gi.loc[gids, "length"]),
                         partials=";".join(str(x) for x in gi.loc[gids, "partial"]),
                         contig_n_genes=int(r.contig_n_genes), edge_dist=int(r.edge_dist),
                         phylum=r.phylum, cls=r["class"], ord=r.ord, family=r.family, genus=r.genus,
                         completeness=r.completeness, in_conserved_locus=r.in_conserved_locus))
    inst = pd.DataFrame(rows)
    inst["instance_id"] = [f"{c}_{i:04d}" for c, i in zip(inst.cand, inst.groupby("cand").cumcount())]
    inst[inst.cand == "c1"].to_csv(OUT / "c1_instances.tsv", sep="\t", index=False)
    inst[inst.cand != "c1"].to_csv(OUT / "c2_instances.tsv", sep="\t", index=False)
    print(inst.groupby("cand").agg(n=("gids", "size"), genomes=("genome", "nunique"), txok=("tx_order_ok", "mean")))

    # ---------------------------------------------------------------- context windows
    win = []
    ctrl_cov = []
    for _, r in inst.iterrows():
        gids = [int(x) for x in r.gids.split(";")]
        k = len(gids)
        key = (r.genome_code, r.contig_code)
        f0, n = int(first[key]), int(ncont[key])
        s = r.strand

        def window(core_lo, core_hi, strand):
            out = []
            for p in range(core_lo - FLANK, core_hi + FLANK + 1):
                if p < 0 or p >= n or core_lo <= p <= core_hi:
                    continue
                off = (p - core_hi) if p > core_hi else (p - core_lo)   # +1..+5 after, -1..-5 before (contig order)
                off = off if strand > 0 else -off                         # transcription orientation
                out.append((f0 + p, off))
            return out

        for gid_, off in window(r.pos_min, r.pos_max, s):
            win.append(dict(cand=r.cand, instance_id=r.instance_id, kind="real", ctrl=0, gid=gid_, offset=off))
        # controls
        L0 = float(gi.at[gids[0], "length"])
        cg = g.iloc[f0:f0 + n]
        ok = cg[(cg.length >= 0.75 * L0) & (cg.length <= 1.25 * L0)]
        cand_ctrl = []
        for _, c in ok.iterrows():
            p = int(c.position_index)
            lo, hi = (p, p + k - 1) if c.strand > 0 else (p - k + 1, p)
            if lo < 0 or hi >= n:
                continue
            if hi < r.pos_min - 5 or lo > r.pos_max + 5:     # >= 6 genes away on either side
                cand_ctrl.append((lo, hi, int(c.strand)))
        pick = [cand_ctrl[i] for i in RNG.permutation(len(cand_ctrl))[:3]] if cand_ctrl else []
        ctrl_cov.append(dict(instance_id=r.instance_id, cand=r.cand, n_ctrl_candidates=len(cand_ctrl), n_ctrl=len(pick)))
        for j, (lo, hi, cs) in enumerate(pick, 1):
            for gid_, off in window(lo, hi, cs):
                win.append(dict(cand=r.cand, instance_id=r.instance_id, kind="control", ctrl=j, gid=gid_, offset=off))
    win = pd.DataFrame(win)
    win = win.merge(g[["gid", "protein_id", "token", "module", "length", "dark", "gold", "strand"]], on="gid")
    win.to_csv(OUT / "ctx_windows.tsv", sep="\t", index=False)
    cc = pd.DataFrame(ctrl_cov)
    cc.to_csv(OUT / "ctx_control_coverage.tsv", sep="\t", index=False)
    print(cc.groupby("cand").agg(inst=("n_ctrl", "size"), with_ctrl=("n_ctrl", lambda x: int((x > 0).sum())),
                                 ctrl_windows=("n_ctrl", "sum")))

    # ---------------------------------------------------------------- sequence and annotation lists
    seq = set()
    seq |= set(g.gid[g.token.isin([81718, 393054])])
    for s_ in inst.gids:
        seq |= {int(x) for x in s_.split(";")}
    seq |= {2628908, 1218213, 1650759}                      # fused homologs of candidate 1 (MMseqs2, screen 1)
    seq |= set(g.gid[g.module == 227])                      # engine module of the candidate-2 tokens
    seq |= set(g.gid[g.module == 1167])                     # engine module of the candidate-1 tokens
    with open(OUT / "seq_gids.txt", "w") as f:
        for x in sorted(seq):
            f.write(f"{x}\n")
    pids = set(win.protein_id) | set(g.protein_id[g.gid.isin(seq)])
    with open(OUT / "ctx_protein_ids.txt", "w") as f:
        for x in sorted(pids):
            f.write(f"{x}\n")
    print("sequences", len(seq), "protein ids for annotation", len(pids), "window rows", len(win))


if __name__ == "__main__":
    main()
