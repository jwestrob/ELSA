"""Gene-level darkness: a gene is dark when it has no row in any annotation source of the Sharur
DPANN DB (the package src_* flags: pfam, kofam, kegg, defensefinder, defensefinder_system, txsscan,
txsscan_system, hyddb, hyddb_subgroup), no system_proteins row (incl. cctyper), no derived
_locus/_validation atom, and no gold-instance membership. Writes work/genes_dark.parquet and
work/token_stats.parquet."""
from pathlib import Path

import numpy as np
import pandas as pd

W = Path(__file__).resolve().parent / "work"
D = W / "data"

g = pd.read_parquet(D / "genes.parquet")
assert (g.gid.values == np.arange(len(g))).all()
af = pd.read_parquet(D / "annotation_flags.parquet").sort_values("gid")
ex = pd.read_parquet(D / "sharur_extra_flags.parquet")
tok = pd.read_parquet(D / "mmseqs__seq__e3c50s75__dpann.parquet").sort_values("gid")
mem = pd.read_parquet(D / "members.parquet")
g = g.merge(ex, on="protein_id", how="left")
assert g.src_cctyper.notna().all()
src = [c for c in af.columns if c.startswith("src_")]
g["any_src"] = af[src].any(axis=1).values
assert (g.any_src.values == g.in_annotations.values).all()
g["gold"] = g.gid.isin(mem.gid)
g["dark"] = ~(g.any_src | g.src_cctyper | g.src_derived_locus_validation | g.in_system_proteins | g.gold)
g["token"] = tok.token.values
print("sources:", src + ["src_cctyper", "src_derived_locus_validation", "in_system_proteins", "gold"])
print("dark genes", int(g.dark.sum()), round(float(g.dark.mean()), 4))
t = g.groupby("token").agg(n=("gid", "size"), ndark=("dark", "sum"), ngen=("genome_code", "nunique"),
                           medlen=("length", "median"))
t["dark_frac"] = t.ndark / t.n
t["copies"] = t.n / t.ngen
g[["gid", "protein_id", "genome_code", "contig_code", "position_index", "strand", "length", "start", "end",
   "partial", "dark", "gold", "token"]].to_parquet(W / "genes_dark.parquet")
t.to_parquet(W / "token_stats.parquet")
