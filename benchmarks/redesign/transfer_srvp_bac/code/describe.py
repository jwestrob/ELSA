#!/usr/bin/env python
"""Descriptive statistics of the SR-VP bacteria package (no method output is read or produced).

Universe size, contig length in genes, taxonomic breadth, gold instances per caller (all, resolved, the evaluation
set = resolved + order trusted + >= 2 members) with span distributions, ribosomal runs, family-size distributions
of the token tables, embedding run summary. Writes <pkg>/receipts/describe.json.

usage: describe.py --pkg DIR [--run EMBED_RUN_DIR]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--pkg", type=Path, required=True)
ap.add_argument("--run", type=Path)
a = ap.parse_args()
P = a.pkg
Q = [0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1]


def qd(x, qs=Q):
    x = np.asarray(x, dtype=float)
    return {str(q): float(np.quantile(x, q)) for q in qs} if len(x) else {}


def hist(x, edges, labels):
    c = pd.cut(pd.Series(x), edges, right=False, labels=labels)
    return {str(k): int(v) for k, v in c.value_counts(sort=False).items()}


R = dict(written=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
genes = pd.read_parquet(P / "universe/genes.parquet", columns=["gid", "genome_code", "contig_code", "strand", "length"])
contigs = pd.read_parquet(P / "universe/contigs.parquet")
genomes = pd.read_parquet(P / "universe/genomes.parquet")
gf = pd.read_parquet(P / "universe/gene_flags.parquet")
R["universe"] = dict(
    genomes=int(len(genomes)), contigs=int(len(contigs)), proteins=int(len(genes)),
    circular_contigs=int(gf.drop_duplicates("gid").merge(genes[["gid", "contig_code"]]).groupby("contig_code").contig_circular.first().sum()),
    strand_plus=int((genes.strand == 1).sum()), strand_minus=int((genes.strand == -1).sum()),
    partial=gf.partial.value_counts(dropna=False).to_dict(), truncated=int(gf.truncated.sum()),
    protein_length_aa=qd(genes.length),
    contig_genes=qd(contigs.n_genes),
    contig_genes_hist=hist(contigs.n_genes, [0, 500, 1000, 2000, 3000, 5000, 10 ** 7],
                           ["<500", "500-999", "1000-1999", "2000-2999", "3000-4999", ">=5000"]),
    contig_nt=qd(contigs.length_nt),
    order_trusted_genomes=int(genomes.order_trusted.sum()),
    order_flags=genomes.order_flags.fillna("").value_counts().to_dict(),
    strand_agreement_q=qd(genomes.strand_agreement.dropna(), [0, 0.01, 0.5, 1]))
tax = {}
for r in ["phylum", "class", "ord", "family", "genus", "species"]:
    tax[r] = dict(distinct=int(genomes[r].nunique(dropna=True)), genomes_named=int(genomes[r].notna().sum()))
tax["tax_source"] = genomes.tax_source.value_counts().to_dict()
tax["genomes_per_phylum"] = genomes.phylum.fillna("NA").value_counts().to_dict()
tax["orders_per_phylum"] = genomes.groupby(genomes.phylum.fillna("NA")).ord.nunique().sort_values(ascending=False).to_dict()
tax["completeness"] = qd(genomes.completeness.dropna(), [0, 0.05, 0.5, 0.95, 1])
tax["contamination"] = qd(genomes.contamination.dropna(), [0, 0.05, 0.5, 0.95, 1])
R["taxonomy"] = tax
halves = pd.read_parquet(P / "split/genome_halves.parquet")
R["split"] = dict(genomes=halves.half.value_counts().to_dict(),
                  genes=genes.merge(halves[["genome_code", "half"]]).half.value_counts().to_dict(),
                  phyla=genomes.merge(halves[["genome_code", "half"]]).groupby("half").phylum.nunique().to_dict())

inst = pd.read_parquet(P / "gold/instances.parquet")
inst = inst.merge(halves[["genome", "half"]], on="genome", how="left")
gold = {}
for c, g in inst.groupby("caller"):
    ev = g[g.resolved.fillna(False) & g.order_trusted.fillna(False) & (g.n_members_resolved >= 2)]
    d = dict(instances=int(len(g)), resolved=int(g.resolved.fillna(False).sum()),
             resolved_single_member=int((g.resolved.fillna(False) & (g.n_members_resolved == 1)).sum()),
             unresolved=int((~g.resolved.fillna(False)).sum()),
             system_types_all=int(g.system_type.nunique()),
             eval_set_ge2_members=int(len(ev)), eval_set_system_types=int(ev.system_type.nunique()),
             eval_set_by_half=ev.half.value_counts().to_dict(),
             eval_set_genomes=int(ev.genome.nunique()),
             eval_span_genes=qd(ev.span_genes, [0, 0.5, 0.9, 0.99, 1]),
             eval_span_mean=float(ev.span_genes.mean()) if len(ev) else None,
             eval_members=qd(ev.n_members_resolved, [0, 0.5, 0.9, 0.99, 1]),
             eval_contiguous=int((ev.span_genes == ev.n_members_resolved).sum()),
             eval_size_bins=hist(ev.n_members_resolved, [2, 4, 7, 10 ** 6], ["2-3", "4-6", "7+"]),
             eval_members_unannotated=int(ev.n_members_unannotated.sum()),
             eval_top_types=ev.system_type.value_counts().head(15).to_dict(),
             status=g.status.fillna("NA").value_counts().to_dict() if g.status.notna().any() else None)
    gold[c] = d
R["gold"] = gold
rj = P / "ribosomal/gold.json"
if rj.exists():
    rg = json.loads(rj.read_text())
    R["ribosomal"] = dict(marker_hits=rg["marker_hits"], marked_genes=rg["marked_genes"],
                          marker_names=len(rg["marker_names"]),
                          sets={k: {kk: v[kk] for kk in ("runs", "genomes_with_run", "genomes_with_2plus_runs",
                                                         "n_genes_quantiles", "n_marked_quantiles", "at_contig_edge")}
                                for k, v in rg["sets"].items()},
                          controls=rg["controls"])
    runs = pd.read_parquet(P / "ribosomal/runs.parquet")
    m16 = runs[(runs.run_set == "g2") & (runs.n_marked >= 16)]
    R["ribosomal"]["g2_m16_phyla"] = int(m16.phylum.nunique())
    R["ribosomal"]["g2_m16_genera"] = int(m16.genus.nunique())
    R["ribosomal"]["g2_m16_by_half"] = m16.merge(halves[["genome_code", "half"]]).half.value_counts().to_dict()
fam = {}
for f in sorted((P / "tokens").glob("*.parquet")):
    t = pd.read_parquet(f).merge(genes[["gid", "genome_code"]], on="gid")
    s = t.groupby("token").size()
    ng = t.groupby("token").genome_code.nunique()
    fam[f.stem] = dict(tokens=int(len(s)), singletons=int((s == 1).sum()), genes_in_singletons=int((s == 1).sum()),
                       genes_in_ge2=int(s[s >= 2].sum()), largest=int(s.max()),
                       size_q=qd(s, [0.5, 0.9, 0.99, 0.999, 1]),
                       size_hist=hist(s, [1, 2, 3, 5, 10, 50, 100, 1000, 10 ** 9],
                                      ["1", "2", "3-4", "5-9", "10-49", "50-99", "100-999", ">=1000"]),
                       genes_by_family_size=hist(np.repeat(s.values, s.values), [1, 2, 3, 5, 10, 50, 100, 1000, 10 ** 9],
                                                 ["1", "2", "3-4", "5-9", "10-49", "50-99", "100-999", ">=1000"]),
                       tokens_in_ge2_genomes=int((ng >= 2).sum()), tokens_in_ge5_genomes=int((ng >= 5).sum()),
                       tokens_in_ge50_genomes=int((ng >= 50).sum()), tokens_in_ge500_genomes=int((ng >= 500).sum()))
R["families"] = fam
if a.run and (a.run / "run.json").exists():
    rj = json.loads((a.run / "run.json").read_text())
    R["embedding_run"] = {k: rj[k] for k in ("name", "rows", "unique_sequences", "duplicate_rows", "residues_rows",
                                             "residues_unique", "shards", "length_quantiles", "unique_longer_than_2046",
                                             "substituted_symbols", "model_id", "model_revision", "dtype")}
    mf = a.run / "final/MANIFEST.json"
    if mf.exists():
        m = json.loads(mf.read_text())
        R["embedding_run"]["final"] = {k: m.get(k) for k in ("finalized", "dim", "unique", "rows", "unique_failed",
                                                              "rows_without_vector", "embedding_seconds", "by_device",
                                                              "shard_audit", "assembly")}
zj = P / "embeddings/zca10.json"
if zj.exists():
    z = json.loads(zj.read_text())
    R["zca10"] = {k: z[k] for k in ("unique", "valid", "invalid", "transform", "participation_ratio_after", "check")}
out = P / "receipts/describe.json"
out.write_text(json.dumps(R, indent=1, default=str) + "\n")
print(json.dumps(R, indent=1, default=str))
