#!/usr/bin/env python
"""ribo_audit part 6 B: the frozen definition (predeclared6.json, variant (a)) on E. coli K-12 MG1655 and B. subtilis 168.

Genes = RefSeq CDS with a protein (GCF_000005845.2, GCF_000009045.1), ordered by replicon and start; proteins from RefSeq.
Markers: the 121 Pfam-A 38.1 models whose NAME matches ^[Rr]ibosom(al)?_ (extracted from Pfam-A.hmm in Python), searched
through Aksha without cutoffs, sequence bitscore >= GA1. Runs: g2 rule (<= 2 unmarked genes between marked genes, >= 16 marked).
Member annotation: annot6.py (Aksha route of the frozen procedure) with N_total = proteins of the two genomes.
Targets: variant (a). Writes work6/ext/genes6.tsv (one row per gene with classes, run and target ids) and ext6.json.
usage: ext6.py prep | ext6.py markers | ext6.py targets   (annot6.py run between markers and targets)
"""
import gzip
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RD = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111")
AD = RD / "ribo_audit"
E = AD / "work6/ext"
AS = Path(os.path.expanduser("~/.config/Astra"))
AKSHA = "/home/jwestrob/.local/bin/aksha"
G = {"ecoli": "GCF_000005845.2_ASM584v2", "bsub": "GCF_000009045.1_ASM904v1"}


def attrs(s):
    return dict(kv.split("=", 1) for kv in s.strip().split(";") if "=" in kv)


def cmd_prep():
    rows, seqs = [], {}
    for code, (sp, acc) in enumerate(G.items()):
        with gzip.open(E / f"dl_{sp}/{acc}_genomic.gff.gz", "rt") as fh:
            for line in fh:
                if line.startswith("#"):
                    continue
                f = line.rstrip("\n").split("\t")
                if len(f) < 9 or f[2] != "CDS":
                    continue
                a = attrs(f[8])
                if "protein_id" not in a:
                    continue
                rows.append(dict(genome=sp, genome_code=code, contig=f[0], start=int(f[3]), end=int(f[4]),
                                 strand=1 if f[6] == "+" else -1, protein_id=a["protein_id"], gene=a.get("gene", ""),
                                 locus_tag=a.get("locus_tag", ""), product=a.get("product", "")))
        with gzip.open(E / f"dl_{sp}/{acc}_protein.faa.gz", "rt") as fh:
            h, buf = None, []
            for line in fh:
                if line.startswith(">"):
                    if h:
                        seqs[h] = "".join(buf)
                    h, buf = line[1:].split()[0], []
                else:
                    buf.append(line.strip())
            seqs[h] = "".join(buf)
    g = pd.DataFrame(rows).drop_duplicates(["genome", "contig", "start", "end", "strand"])
    g = g.sort_values(["genome_code", "contig", "start"]).reset_index(drop=True)
    g["gid"] = np.arange(len(g))
    g["contig_code"] = pd.factorize(g.genome + ":" + g.contig)[0]
    g["position_index"] = g.groupby("contig_code").cumcount()
    g["length"] = (g.end - g.start + 1) // 3
    g.to_parquet(E / "genes.parquet", index=False)
    c = g.groupby("contig_code").agg(first_gid=("gid", "min"), n_genes=("gid", "size"), genome=("genome", "first"),
                                     contig=("contig", "first")).reset_index()
    c.to_parquet(E / "contigs.parquet", index=False)
    with open(E / "proteins.faa", "w") as fh:
        for gid, pid in zip(g.gid, g.protein_id):
            fh.write(f">{gid}\n{seqs[pid]}\n")
    # 121 ribosomal Pfam models
    out, keep, n = [], False, 0
    with open(AS / "PFAM/Pfam-A.hmm") as fh, open(E / "ribo121.hmm", "w") as fo:
        block = []
        for line in fh:
            block.append(line)
            if line.startswith("//"):
                name = next(x.split()[1] for x in block if x.startswith("NAME "))
                if re.match(r"^[Rr]ibosom(al)?_", name):
                    fo.writelines(block)
                    n += 1
                block = []
    print("genes", len(g), g.groupby("genome").size().to_dict(), "contigs", len(c), "ribosomal Pfam models", n)


def cmd_markers():
    sys.path.insert(0, str(AD / "code"))
    import annot6 as A6
    df = A6.aksha(E / "ribo121.hmm", E / "proteins.faa", E / "aksha_ribo", 8)
    ga = A6.ga_table(E / "ribo121.hmm")
    s = A6.per_seq(df)
    s = s[s.full.to_numpy() >= np.array([ga[x][0] for x in s.hmm_name])]
    top = s.sort_values("full", ascending=False).drop_duplicates("gid")
    allm = s.sort_values(["gid", "hmm_name"]).groupby("gid").hmm_name.agg(";".join)
    mk = pd.DataFrame(dict(gid=top.gid.to_numpy(), marker=top.hmm_name.to_numpy()))
    mk["markers"] = allm.reindex(mk.gid).to_numpy()
    mk = mk.sort_values("gid")
    mk.to_parquet(E / "marked.parquet", index=False)
    g = pd.read_parquet(E / "genes.parquet")
    marked = np.zeros(len(g), bool)
    marked[mk.gid.to_numpy()] = True
    m = np.flatnonzero(marked)
    cc = g.contig_code.to_numpy()
    new = np.r_[True, (np.diff(m) > 3) | (cc[m[1:]] != cc[m[:-1]])]
    cid = np.cumsum(new) - 1
    rows = []
    for c in np.unique(cid):
        mm = m[cid == c]
        rows.append(dict(run_set="g2", gid_start=int(mm[0]), gid_end=int(mm[-1]), n_marked=len(mm),
                         contig_code=int(cc[mm[0]]), genome_code=int(g.genome_code.iat[mm[0]])))
    r = pd.DataFrame(rows)
    r["m16"] = r.n_marked >= 16
    r["run_id"] = np.arange(len(r))
    r.to_parquet(E / "runs.parquet", index=False)
    print("marked", len(mk), "runs g2 with >= 16 marked", int(r.m16.sum()), r[r.m16].to_dict("records"))


def cmd_targets():
    g = pd.read_parquet(E / "genes.parquet")
    an = pd.read_parquet(E / "annot/genes_annot.parquet")
    ex = pd.read_parquet(E / "annot/extra_ko.parquet")
    names = {"K02601": "nusG", "K03043": "rpoB", "K03046": "rpoC", "K02358": "tufA", "K02355": "fusA", "K00939": "adk",
             "K01265": "map", "K02518": "infA", "K03073": "secE", "K01883": "cysS", "K03685": "rnc", "K03218": "rlmB",
             "K16786": "ecfA1", "K16787": "ecfA2", "K16785": "ecfT", "K06173": "truA"}
    cls = an.cls.to_numpy(dtype=object).copy()
    lit = an.lit.fillna("").to_numpy(dtype=object).copy()
    for gid, k in zip(ex.gid, ex.ko):
        if cls[gid] == "X":
            cls[gid], lit[gid] = "N", names[k]
    mkd = pd.read_parquet(E / "marked.parquet")
    marked = np.zeros(len(g), bool)
    marked[mkd.gid.to_numpy()] = True
    runs = pd.read_parquet(E / "runs.parquet")
    runs = runs[runs.m16].reset_index(drop=True)
    member = np.isin(cls, ["R", "N"])
    cc, st = g.contig_code.to_numpy(), g.strand.to_numpy()
    mk = np.flatnonzero(member)
    brk = np.r_[True, (np.diff(mk) > 3) | (cc[mk[1:]] != cc[mk[:-1]]) | (st[mk[1:]] != st[mk[:-1]])]
    cs, ce = mk[brk], mk[np.r_[brk[1:], True]]
    cid = np.cumsum(brk) - 1
    pos = np.full(len(g), -1)
    pos[mk] = np.arange(len(mk))
    cm = np.r_[0, np.cumsum(marked)]
    # part-4 (strand-blind) chains for comparison
    brk4 = np.r_[True, (np.diff(mk) > 3) | (cc[mk[1:]] != cc[mk[:-1]])]
    cs4, ce4 = mk[brk4], mk[np.r_[brk4[1:], True]]
    cid4 = np.cumsum(brk4) - 1
    g["cls"], g["lit"], g["marked"] = cls, lit, marked
    g["run"], g["target"], g["piece"], g["part4"] = -1, "", "", -1
    out = []
    for r in runs.itertuples():
        g.loc[r.gid_start:r.gid_end, "run"] = r.run_id
        ids = sorted(set(int(cid[pos[k]]) for k in range(r.gid_start, r.gid_end + 1) if marked[k]))
        pieces = [(int(cs[c]), int(ce[c]), int(cm[ce[c] + 1] - cm[cs[c]])) for c in ids]
        c4 = int(cid4[pos[r.gid_start]])
        g.loc[cs4[c4]:ce4[c4], "part4"] = r.run_id
        for j, (a, b, nm) in enumerate(pieces):
            lab = f"run{r.run_id}.p{j}" + ("" if nm >= 16 else "(below16)")
            g.loc[a:b, "piece"] = lab
            if nm >= 16:
                g.loc[a:b, "target"] = f"T{r.run_id}.{j}"
        out.append(dict(run_id=int(r.run_id), genome=g.genome.iat[r.gid_start], run=[int(r.gid_start), int(r.gid_end)],
                        run_genes=f"{g.gene.iat[r.gid_start]}..{g.gene.iat[r.gid_end]}", n_marked=int(r.n_marked),
                        pieces=[dict(s=a, e=b, marked=nm, genes=f"{g.gene.iat[a]}..{g.gene.iat[b]}", qualifies=nm >= 16)
                                for a, b, nm in pieces],
                        part4=dict(s=int(cs4[c4]), e=int(ce4[c4]), genes=f"{g.gene.iat[cs4[c4]]}..{g.gene.iat[ce4[c4]]}")))
    lo = {}
    for r in out:
        sp = r["genome"]
        a = min([r["run"][0]] + [p["s"] for p in r["pieces"]] + [r["part4"]["s"]]) - 6
        b = max([r["run"][1]] + [p["e"] for p in r["pieces"]] + [r["part4"]["e"]]) + 6
        lo.setdefault(sp, []).append((a, b))
    keep = np.zeros(len(g), bool)
    for sp, ivs in lo.items():
        for a, b in ivs:
            keep[max(a, 0):b + 1] = True
    cols = ["gid", "genome", "contig", "start", "end", "strand", "locus_tag", "gene", "product", "cls", "lit", "marked", "run",
            "piece", "target", "part4"]
    g[keep][cols].to_csv(E / "genes6.tsv", sep="\t", index=False)
    json.dump(out, open(E / "ext6.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    {"prep": cmd_prep, "markers": cmd_markers, "targets": cmd_targets}[sys.argv[1]]()
