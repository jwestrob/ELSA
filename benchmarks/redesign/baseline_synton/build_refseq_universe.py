#!/usr/bin/env python
"""Universe for the RefSeq complete-genome cohort of Herve et al. 2026 (locked public test; no method is run here).

Input  $W/genomes/<assembly>/<name>_feature_table.txt.gz and _protein.faa.gz (NCBI RefSeq FTP, fetched 2026-11-13),
       $W/meta/candidates_complete_le_2023-05-31.tsv (RefSeq bacteria Complete Genome, seq_rel_date <= 2023-05-31,
       current + historical assemblies live on 2023-05-31).
Output $W/universe/{genes,contigs,genomes}.parquet in the layout of ws_dpann/package/universe:
         genes   gid, protein_id, genome, genome_code, contig, contig_code, position_index, strand (int8), length,
                 start, end
         contigs genome, contig, contig_code, n_genes, length_nt (NaN: feature tables carry no replicon length),
                 first_gid
         genomes genome, genome_code, n_genes, n_contigs, organism_name, taxid, species_taxid, src, version_status,
                 seq_rel_date, asm_not_live_date, annotation_date, n_pseudo_cds
       $W/universe/proteins_wp.faa.gz  one record per distinct protein accession (header = accession)
       $W/universe/build.json           counts, missing files, timing
Genes = CDS rows of class with_protein (pseudogenes without protein excluded); position_index = rank on the replicon by
(start, end, protein_id); gid in (genome_code, contig order of first appearance, position) order.

usage (Slurm): build_refseq_universe.py --w $W --threads $SLURM_CPUS_ON_NODE
"""
import argparse
import gzip
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

FT_COLS = ["feature", "class", "assembly", "assembly_unit", "seq_type", "chromosome", "genomic_accession", "start", "end",
           "strand", "product_accession", "non_redundant_refseq", "related_accession", "name", "symbol", "GeneID",
           "locus_tag", "feature_interval_length", "product_length", "attributes"]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def parse_one(args):
    acc, d = args
    d = Path(d)
    ft = list(d.glob("*_feature_table.txt.gz"))
    if not ft:
        return acc, None, "no feature table"
    nbytes = sum(p.stat().st_size for p in d.glob("*.gz"))
    t = pd.read_csv(ft[0], sep="\t", header=None, names=FT_COLS, comment="#", dtype=str, low_memory=False)
    cds = t[t.feature == "CDS"]
    n_pseudo = int((cds["class"] != "with_protein").sum())
    cds = cds[(cds["class"] == "with_protein") & cds.product_accession.notna()].copy()
    cds["start"] = cds.start.astype(np.int64)
    cds["end"] = cds.end.astype(np.int64)
    cds["length"] = pd.to_numeric(cds.product_length, errors="coerce").fillna(-1).astype(np.int64)
    cds["strand"] = np.where(cds.strand == "+", 1, -1).astype(np.int8)
    order = {c: i for i, c in enumerate(pd.unique(t.genomic_accession))}
    cds["corder"] = cds.genomic_accession.map(order)
    cds = cds.sort_values(["corder", "start", "end", "product_accession"], kind="stable")
    cds["position_index"] = cds.groupby("genomic_accession", sort=False).cumcount().astype(np.int64)
    out = pd.DataFrame(dict(protein_id=cds.product_accession.to_numpy(), genome=acc,
                            contig=cds.genomic_accession.to_numpy(), position_index=cds.position_index.to_numpy(),
                            strand=cds.strand.to_numpy(), length=cds.length.to_numpy(), start=cds.start.to_numpy(),
                            end=cds.end.to_numpy()))
    return acc, (out, n_pseudo, nbytes), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--w", required=True)
    ap.add_argument("--threads", type=int, required=True)
    a = ap.parse_args()
    t0 = time.time()
    W = Path(a.w)
    U = W / "universe"
    U.mkdir(exist_ok=True)
    cand = pd.read_csv(W / "meta/candidates_complete_le_2023-05-31.tsv", sep="\t", dtype=str)
    cand = cand.drop_duplicates("assembly_accession").sort_values("assembly_accession").reset_index(drop=True)
    tasks = [(acc, W / "genomes" / acc) for acc in cand.assembly_accession]
    parts, missing, pseudo, nbytes = {}, {}, {}, {}
    with ProcessPoolExecutor(max_workers=a.threads) as ex:
        for i, (acc, res, err) in enumerate(ex.map(parse_one, tasks, chunksize=16)):
            if err:
                missing[acc] = err
            else:
                parts[acc], pseudo[acc], nbytes[acc] = res
            if i % 2000 == 0:
                log("parsed", i, "missing", len(missing), f"{time.time()-t0:.0f}s")
    accs = [x for x in cand.assembly_accession if x in parts]
    genes = pd.concat([parts[x] for x in accs], ignore_index=True)
    gcode = {x: i for i, x in enumerate(accs)}
    genes["genome_code"] = genes.genome.map(gcode).astype(np.int32)
    ckey = genes.genome + "\t" + genes.contig
    genes["contig_code"] = pd.factorize(ckey)[0].astype(np.int64)
    genes = genes.sort_values(["genome_code", "contig_code", "position_index"], kind="stable").reset_index(drop=True)
    genes.insert(0, "gid", np.arange(len(genes), dtype=np.int64))
    genes = genes[["gid", "protein_id", "genome", "genome_code", "contig", "contig_code", "position_index", "strand",
                   "length", "start", "end"]]
    genes.to_parquet(U / "genes.parquet", index=False)
    contigs = genes.groupby("contig_code").agg(genome=("genome", "first"), contig=("contig", "first"),
                                               n_genes=("gid", "size"), first_gid=("gid", "min")).reset_index()
    contigs["length_nt"] = np.nan
    contigs[["genome", "contig", "contig_code", "n_genes", "length_nt", "first_gid"]].to_parquet(U / "contigs.parquet",
                                                                                                index=False)
    gm = genes.groupby("genome_code").agg(genome=("genome", "first"), n_genes=("gid", "size"),
                                          n_contigs=("contig_code", "nunique")).reset_index()
    meta = cand.set_index("assembly_accession")
    for c in ("organism_name", "src", "version_status", "seq_rel_date", "asm_not_live_date", "annotation_date"):
        gm[c] = meta.loc[gm.genome, c].to_numpy()
    gm["n_pseudo_cds"] = gm.genome.map(pseudo).astype(np.int64)
    gm[["genome", "genome_code", "n_genes", "n_contigs", "organism_name", "src", "version_status", "seq_rel_date",
        "asm_not_live_date", "annotation_date", "n_pseudo_cds"]].to_parquet(U / "genomes.parquet", index=False)
    log("tables written", len(genes), "genes", len(gm), "genomes", f"{time.time()-t0:.0f}s")
    # distinct protein sequences by accession
    need = set(genes.protein_id.unique())
    seen = set()
    n_rec = 0
    with gzip.open(U / "proteins_wp.faa.gz", "wt", compresslevel=3) as out:
        for acc in accs:
            fa = list((W / "genomes" / acc).glob("*_protein.faa.gz"))
            if not fa:
                missing.setdefault(acc, "no protein.faa")
                continue
            keep = False
            with gzip.open(fa[0], "rt") as fh:
                for line in fh:
                    if line.startswith(">"):
                        pid = line[1:].split(None, 1)[0]
                        keep = pid in need and pid not in seen
                        if keep:
                            seen.add(pid)
                            out.write(">" + pid + "\n")
                            n_rec += 1
                    elif keep:
                        out.write(line)
    info = dict(candidates=len(cand), genomes=len(gm), genes=int(len(genes)), contigs=int(len(contigs)),
                downloaded_gz_bytes=int(sum(nbytes.values())),
                genomes_current=int((gm.src == "current").sum()), genomes_historical=int((gm.src == "historical").sum()),
                distinct_protein_ids=len(need), proteins_written=n_rec, proteins_missing=len(need - seen),
                missing_assemblies=missing, genes_on_contigs_ge48=int(contigs.n_genes[contigs.n_genes >= 48].sum()),
                seconds=round(time.time() - t0, 1))
    json.dump(info, open(U / "build.json", "w"), indent=1)
    log("done", {k: v for k, v in info.items() if k != "missing_assemblies"}, "missing", len(missing))


if __name__ == "__main__":
    main()
