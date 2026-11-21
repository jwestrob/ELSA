"""Part A.5: why the engine left the four high-scoring uncalled-neighbour pairs uncalled.

Recomputes, for each uncalled pair's token pair, the engine v1.5 pair statistics exactly as
engine/recurrence.py pair_statistics does (window w = 3, genome-level counting):
  G  genomes in which the two tokens sit within 3 genes on one contig
  L  genera (lineages; genome name when genus missing) among those genomes;  NP phyla
  F  genomes carrying each token;  J = G / min(F)
  E  sum over genomes carrying both of min(1, 2 w c_a c_b / n_genes);  P = Poisson P(X >= G | E)
Rule of the run: P <= 1e-4 and L >= 2. Also: token sizes (members), whether either token is a
singleton, the gene-level darkness of each gene and its family, and whether the pair is in
significant_pairs.parquet (it must not be). Writes uncalled_diag.tsv.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson

HERE = Path(__file__).resolve().parent
W = HERE / "work"
D = W / "data"
WIN = 3


def main():
    g = pd.read_parquet(W / "genes_dark.parquet")
    gm = pd.read_parquet(D / "genomes.parquet").set_index("genome_code")
    genus = gm.genus.where(gm.genus.notna() & (gm.genus != ""), "genome:" + gm.genome)
    phylum = gm.phylum.where(gm.phylum.notna() & (gm.phylum != ""), "unk:" + gm.genome)
    sp = pd.read_parquet(D / "significant_pairs.parquet")
    T = int(g.token.max()) + 1
    fam = pd.read_parquet(W / "fam_stats.parquet")
    man = pd.read_csv(W / "fold_manifest.tsv", sep="\t")
    f = pd.read_csv(W / "folds.tsv", sep="\t")
    ids = ["unc_002_r1", "unc_069_r2", "unc_071_r2", "unc_087_r1"]
    tok = g.token.to_numpy()
    cc = g.contig_code.to_numpy()
    gc = g.genome_code.to_numpy()
    rows = []
    for fid in ids:
        r = man[man.fold_id == fid].iloc[0]
        a, b = [int(x) for x in r.gids.split(";")]
        ta, tb = int(tok[a]), int(tok[b])
        lo, hi = min(ta, tb), max(ta, tb)
        # co-location instances within WIN genes
        gen_co = set()
        for d in range(1, WIN + 1):
            left = np.flatnonzero(cc[:-d] == cc[d:])
            x, y = tok[left], tok[left + d]
            m = ((x == lo) & (y == hi)) | ((x == hi) & (y == lo))
            gen_co |= set(gc[left[m]].tolist())
        G = len(gen_co)
        L = len({genus[k] for k in gen_co})
        NP = len({phylum[k] for k in gen_co})
        ca = g[g.token == ta].groupby("genome_code").size()
        cb = g[g.token == tb].groupby("genome_code").size()
        both = ca.index.intersection(cb.index)
        E = float(sum(min(1.0, 2.0 * WIN * ca[k] * cb[k] / gm.at[k, "n_genes"]) for k in both))
        P = float(poisson.sf(G - 1, max(E, 1e-12)))
        key = lo * T + hi
        fo = f[f.fold_id == fid].iloc[0]
        rows.append(dict(
            fold_id=fid, matched_to_cand=int(r.cand_rank), match_level=r.match_level, gids=r.gids,
            genome=r.genome, lengths=r.lengths, iptm=round(float(fo.iptm), 3), iface_pae=fo.iface_pae,
            contacts=int(fo.contacts), token_a=ta, token_b=tb,
            members_a=int(fam.at[ta, "fam_n"]) if ta in fam.index else 1,
            members_b=int(fam.at[tb, "fam_n"]) if tb in fam.index else 1,
            genomes_a=int(len(ca)), genomes_b=int(len(cb)), genomes_both=int(len(both)),
            genera_of_G=";".join(sorted({str(genus[k]) for k in gen_co})),
            G=G, L=L, NP=NP, J=round(G / max(1, min(len(ca), len(cb))), 3), E=round(E, 4), P=f"{P:.2e}",
            passes_rule=bool(P <= 1e-4 and L >= 2), in_significant_pairs=bool(key in set(sp.key.values)),
            dark_a=bool(g.dark.values[a]), dark_b=bool(g.dark.values[b]),
            fam_dark_frac_a=round(float(fam.at[ta, "fam_dark_frac"]), 3) if ta in fam.index else None,
            fam_dark_frac_b=round(float(fam.at[tb, "fam_dark_frac"]), 3) if tb in fam.index else None,
            strand=f"{int(g.strand.values[a])}/{int(g.strand.values[b])}",
            intergenic_nt=int(max(g.start.values[a], g.start.values[b])
                              - min(g.end.values[a], g.end.values[b]) - 1)))
    out = pd.DataFrame(rows)
    out.to_csv(HERE / "uncalled_diag.tsv", sep="\t", index=False)
    pd.set_option("display.width", 250)
    print(out.T.to_string())


if __name__ == "__main__":
    main()
