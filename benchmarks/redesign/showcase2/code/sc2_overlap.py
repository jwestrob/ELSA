#!/usr/bin/env python
"""ws_showcase2 (Mac, one thread): overlapping junctions of the passing blocks (method of cofold/overlap_check.py,
reduced): for every same-strand realisation, the upstream and downstream genes in transcription order, the overlap
in nt, the frame of the downstream gene relative to the upstream gene, its start codon, and whether the downstream
gene's frame is closed by an in-frame stop between the upstream gene's start and the downstream start (an open frame
would allow a fused read-through product). Contigs from the Sharur DPANN genome FASTAs (genomes_fna, genomes_fna_new).
Input analysis/context_annot.tsv + analysis/block_geometry.tsv; writes analysis/overlap.tsv."""
import os
from pathlib import Path

import duckdb
import pandas as pd

HERE = Path(__file__).resolve().parent.parent
DPANN = Path("/Users/jacob/Documents/Sandbox/Sharur/Sharur/data/dpann")
DB = DPANN / "sharur.duckdb"
STOP = {"TAA", "TAG", "TGA"}
COMP = str.maketrans("ACGTN", "TGCAN")


def rc(s):
    return s.translate(COMP)[::-1]


def contigs(genome, want):
    for base in ("genomes_fna", "genomes_fna_new"):
        p = DPANN / base / f"{genome}.fna"
        if p.exists():
            out, name, buf = {}, None, []
            for line in open(p):
                if line.startswith(">"):
                    if name in want:
                        out[name] = "".join(buf).upper()
                    name, buf = line[1:].split()[0], []
                else:
                    buf.append(line.strip())
            if name in want:
                out[name] = "".join(buf).upper()
            return out
    return {}


def main():
    a = pd.read_csv(HERE / "analysis/context_annot.tsv", sep="\t")
    geo = pd.read_csv(HERE / "analysis/block_geometry.tsv", sep="\t")
    geo = geo[geo.same_strand & geo.cand.isin(["kd_kA_091", "ls_c764_k2"])]
    m = a[a.member & (a.kind == "block")].merge(geo[["cand", "real"]], on=["cand", "real"])
    con = duckdb.connect(str(DB), read_only=True, config={"threads": 1})
    con.register("w", pd.DataFrame(dict(protein_id=m.protein_id.unique())))
    p = con.execute("""select p.protein_id, p.contig_id, p.start, p.end_coord, p.strand, b.bin_id
                       from proteins p join w using (protein_id) join bins b on b.bin_id = p.bin_id""").fetchdf()
    con.close()
    m = m.merge(p, on="protein_id")
    rows = []
    for genome, d in m.groupby("bin_id"):
        cs = contigs(genome, set(d.contig_id))
        for (cand, real), e in d.groupby(["cand", "real"]):
            if len(e) != 2 or e.contig_id.iloc[0] not in cs:
                continue
            s = cs[e.contig_id.iloc[0]]
            plus = e.strand.iloc[0] in ("+", 1, "1")
            e = e.sort_values("start", ascending=plus)
            up, dn = e.iloc[0], e.iloc[1]
            if plus:
                u0, d0 = up.start - 1, dn.start - 1                     # 0-based start positions
                ovl = up.end_coord - dn.start + 1
                frame = (d0 - u0) % 3
                startc = s[d0:d0 + 3]
                # downstream frame codons from the upstream start to the downstream start
                k = d0
                closed = False
                while k - 3 >= u0:
                    k -= 3
                    if s[k:k + 3] in STOP:
                        closed = True
                        break
            else:
                t = rc(s)
                L = len(s)
                u0, d0 = L - up.end_coord, L - dn.end_coord
                ovl = dn.end_coord - up.start + 1
                frame = (d0 - u0) % 3
                startc = t[d0:d0 + 3]
                k = d0
                closed = False
                while k - 3 >= u0:
                    k -= 3
                    if t[k:k + 3] in STOP:
                        closed = True
                        break
            rows.append(dict(cand=cand, real=real, genome=genome, up_member_pos=int(up.member_pos),
                             overlap_nt=int(ovl), frame_offset=int(frame), dn_start=startc, dn_frame_closed_upstream=closed))
    o = pd.DataFrame(rows)
    o.to_csv(HERE / "analysis/overlap.tsv", sep="\t", index=False)
    for c, d in o.groupby("cand"):
        print(c, "n", len(d), "upstream member pos", d.up_member_pos.value_counts().to_dict(),
              "overlap>0", int((d.overlap_nt > 0).sum()), "overlap median", float(d[d.overlap_nt > 0].overlap_nt.median()) if (d.overlap_nt > 0).any() else None,
              "frame", d.frame_offset.value_counts().to_dict(), "start", d.dn_start.value_counts().head(4).to_dict(),
              "downstream frame closed by a stop between the two starts", int(d.dn_frame_closed_upstream.sum()))


if __name__ == "__main__":
    main()
