#!/usr/bin/env python3
"""Interpretation checks for the split (Y-side) events on the 356 sample genomes.
1. per-genome split rate vs completeness, contamination, contig count (Spearman);
2. recurrence: does the same token pair (tok(Y1), tok(Y2)) appear split-adjacent in other sample genomes
   (genera)? Shared splits point to real fission, genome-specific splits to call or assembly breaks;
3. intergenic gap (nt) of split pairs at distance 1 vs all co-oriented neighbour pairs.
"""
import json, os, sys
import numpy as np, pandas as pd, duckdb
from scipy.stats import spearmanr
W, P, TOK = sys.argv[1], sys.argv[2], sys.argv[3]
R = f"{W}/results"
con = duckdb.connect(); con.execute(f"SET threads={os.environ.get('SLURM_CPUS_ON_NODE', '2')}")
g = con.execute(f"""SELECT g.gid, g.genome_code, g.contig_code, g.position_index pos, g.strand, g.start, g."end" stop, t.token
                    FROM '{P}/universe/genes.parquet' g JOIN '{TOK}' t USING (gid) ORDER BY gid""").df()
gen = con.execute(f"SELECT genome_code, genus, family, completeness, contamination, n_contigs FROM '{P}/universe/genomes.parquet'").df()
fl = np.load(f"{R}/composite_flags.npz"); fr = np.load(f"{R}/fragment_flags.npz")
samp = np.loadtxt(f"{W}/lists/sample_genomes.txt", dtype=np.int64)
ins = np.isin(g.genome_code.values, samp)
res = {}
# 1. per-genome rates
df = pd.DataFrame({"genome_code": g.genome_code.values[ins], "y": fl["Y_adj"][ins], "x": fl["X_adj"][ins], "f": fr["frag"][ins]})
pg = df.groupby("genome_code").agg(genes=("y", "size"), y=("y", "mean"), x=("x", "mean"), f=("f", "mean")).reset_index().merge(gen, on="genome_code")
res["per_genome_split_rate_q10_q50_q90"] = pg.y.quantile([0.1, 0.5, 0.9]).tolist()
res["per_genome_fragment_rate_q10_q50_q90"] = pg.f.quantile([0.1, 0.5, 0.9]).tolist()
for c in ("completeness", "contamination", "n_contigs"):
    for v in ("y", "f"):
        r = spearmanr(pg[v], pg[c], nan_policy="omit"); res[f"spearman_{v}_vs_{c}"] = [float(r.correlation), float(r.pvalue)]
top = pg.sort_values("y", ascending=False)
res["share_of_split_genes_in_top10pct_genomes"] = float((top.y * top.genes).head(len(top) // 10).sum() / (top.y * top.genes).sum())
pg.to_csv(f"{R}/per_genome_rates.tsv", sep="\t", index=False)
# 2. recurrence of split token pairs among sample genomes
gy = pd.read_parquet(f"{R}/composite_groups_y.parquet")
gy = gy[gy.adj & np.isin(gy.B.values, samp)]
t1 = g.token.values[gy.y1.values]; t2 = g.token.values[gy.y2.values]
pairs = pd.DataFrame({"a": np.minimum(t1, t2), "b": np.maximum(t1, t2), "B": gy.B.values, "y1": gy.y1.values, "y2": gy.y2.values})
pairs = pairs.merge(gen[["genome_code", "genus"]], left_on="B", right_on="genome_code")
ev = pairs.drop_duplicates(["a", "b", "B", "y1", "y2"])
loc = ev.drop_duplicates(["B", "y1", "y2"])                       # one row per split locus
ng = ev.groupby(["a", "b"]).agg(genomes=("B", "nunique"), genera=("genus", "nunique")).reset_index()
loc = loc.merge(ng, on=["a", "b"])
res["split_loci_sample"] = int(len(loc))
res["split_loci_token_pair_in_ge2_sample_genomes"] = float((loc.genomes >= 2).mean())
res["split_loci_token_pair_in_ge2_genera"] = float((loc.genera >= 2).mean())
res["split_loci_token_pair_in_ge5_sample_genomes"] = float((loc.genomes >= 5).mean())
# 3. intergenic gaps (same strand, adjacent positions)
def gap(a_, b_):
    s1, e1, s2, e2 = g.start.values[a_], g.stop.values[a_], g.start.values[b_], g.stop.values[b_]
    first = s1 <= s2
    return np.where(first, s2 - e1 - 1, s1 - e2 - 1)
d1 = loc[(np.abs(g.pos.values[loc.y1.values] - g.pos.values[loc.y2.values]) == 1)]
gs = gap(d1.y1.values, d1.y2.values)
idx = np.flatnonzero(ins[:-1] & (g.contig_code.values[:-1] == g.contig_code.values[1:]) & (g.strand.values[:-1] == g.strand.values[1:]))
gb = gap(idx, idx + 1)
q = [0.1, 0.25, 0.5, 0.75, 0.9]
res["gap_nt_split_pairs_q"] = np.quantile(gs, q).tolist(); res["gap_nt_split_pairs_n"] = int(len(gs))
res["gap_nt_all_coriented_neighbours_q"] = np.quantile(gb, q).tolist(); res["gap_nt_all_coriented_neighbours_n"] = int(len(gb))
res["frac_split_pairs_gap_le_10nt"] = float((gs <= 10).mean()); res["frac_all_neighbours_gap_le_10nt"] = float((gb <= 10).mean())
res["frac_split_pairs_overlap"] = float((gs < 0).mean()); res["frac_all_neighbours_overlap"] = float((gb < 0).mean())
json.dump(res, open(f"{R}/interp.json", "w"), indent=1)
print(json.dumps(res, indent=1))
