#!/usr/bin/env python3
"""Ribosomal runs for a slice package by the standing definition (ribosomal/code/ribosomal_check.py, PREDECLARE.txt).

usage: 15_ribosomal_truth.py PKG_DIR DOMTBL_GLOB
Marked gene: a Pfam 38.1 hit at the gathering threshold of any model whose name matches ^ribosom(al)?_ (case-insensitive;
121 models), from hmmsearch over every package protein (marker = best-scoring model). Runs: per contig, maximal runs of
marked genes with at most `gap` unmarked genes inside (g0 strict, g2 tolerant); runs with >= 8 marked genes kept, the
>= 16 subset (m16) is the primary set. Writes gold/ribosomal_marked.parquet, gold/ribosomal_runs.parquet,
receipts/ribosomal.json.
"""
import glob, json, sys
import numpy as np, pandas as pd

P, PAT = sys.argv[1:3]
rows = []
for fn in sorted(glob.glob(PAT)):
    for line in open(fn):
        if line.startswith("#"):
            continue
        f = line.split()
        rows.append((int(f[0]), f[3], f[4], float(f[7])))
h = pd.DataFrame(rows, columns=["gid", "name", "acc", "score"])
genes = pd.read_parquet(f"{P}/universe/genes.parquet", columns=["gid", "genome_code", "contig_code", "position_index"]).sort_values("gid")
assert np.array_equal(genes.gid.to_numpy(), np.arange(len(genes)))
contigs = pd.read_parquet(f"{P}/universe/contigs.parquet").set_index("contig_code")
gm = pd.read_parquet(f"{P}/universe/genomes.parquet").set_index("genome_code")
top = h.sort_values("score", ascending=False).drop_duplicates("gid").set_index("gid").name
alln = h.groupby("gid").name.apply(lambda s: ";".join(sorted(set(s))))
marked = pd.DataFrame({"gid": top.index.to_numpy(), "marker": top.to_numpy()})
marked["markers"] = alln.reindex(marked.gid).to_numpy()
marked = marked.sort_values("gid").reset_index(drop=True)
marked.to_parquet(f"{P}/gold/ribosomal_marked.parquet", index=False)
cc = genes.contig_code.to_numpy(); pos = genes.position_index.to_numpy(); gcode = genes.genome_code.to_numpy()
frames = []
for gap in (0, 2):
    m = np.sort(marked.gid.to_numpy())
    new = np.r_[True, (np.diff(m) > gap + 1) | (np.diff(cc[m]) != 0)]
    rid = np.cumsum(new) - 1
    last = np.r_[new[1:], True]
    r = pd.DataFrame(dict(gid_start=m[new], gid_end=m[last], n_marked=np.bincount(rid)))
    r = r[r.n_marked >= 8].reset_index(drop=True)
    r.insert(0, "run_set", f"g{gap}")
    frames.append(r)
runs = pd.concat(frames, ignore_index=True)
runs.insert(1, "run_id", range(len(runs)))
runs["m16"] = runs.n_marked >= 16
runs["n_genes"] = runs.gid_end - runs.gid_start + 1
runs["contig_code"] = cc[runs.gid_start]; runs["genome_code"] = gcode[runs.gid_start]
runs["pos_start"] = pos[runs.gid_start]; runs["pos_end"] = pos[runs.gid_end]
ci = contigs.loc[runs.contig_code]
runs["contig"] = ci.contig.to_numpy(); runs["genome"] = ci.genome.to_numpy()
runs["contig_first_gid"] = ci.first_gid.to_numpy(); runs["contig_n_genes"] = ci.n_genes.to_numpy()
runs["at_contig_edge"] = (runs.pos_start == 0) | (runs.pos_end == runs.contig_n_genes - 1)
g = gm.loc[runs.genome_code]
for c in ["phylum", "class", "ord", "family", "genus", "completeness", "layer", "split"]:
    runs[c] = g[c].to_numpy()
runs.to_parquet(f"{P}/gold/ribosomal_runs.parquet", index=False)
s = dict(hits=len(h), marked_genes=len(marked), models_hit=int(h.name.nunique()),
         runs={f"{rs}_m{mm}": int(((runs.run_set == rs) & (runs.n_marked >= mm)).sum()) for rs in ("g0", "g2") for mm in (8, 16)},
         genomes_with_g2_m16=int(runs[(runs.run_set == "g2") & runs.m16].genome_code.nunique()), genomes=len(gm))
json.dump(s, open(f"{P}/receipts/ribosomal.json", "w"), indent=1)
print(json.dumps(s), flush=True)
