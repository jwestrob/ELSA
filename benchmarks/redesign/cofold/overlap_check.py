"""Part A.4: nucleotide check of the overlapping reading frames of ranks 27 and 106.

Inputs: work/s2/overlap_instances.tsv (instances, gene coordinates), work/s2/overlap_contigs.fna
(contigs from the DPANN genome FASTAs, extracted on biotite by overlap_extract_nt.py), the Sharur
DPANN DB (read-only: protein sequences and assembly_contig_provenance SHA-256 of each contig).

Per instance (transcript orientation of the pair):
  - contig identity: SHA-256 of the extracted contig equals assembly_contig_provenance
  - both CDS translate (table 11) to the stored proteins (start codon read as M)
  - start codon of B; B start position relative to A; overlap (nt); frame of B relative to A
    ((B_start - A_start) mod 3: 1 = B in A's +1 frame, 2 = B in A's -1 frame)
  - B's frame upstream of its start: distance to the nearest in-frame stop codon (how far B's
    reading frame is open into A) and any alternative in-frame start codons in that open stretch
  - A's frame downstream of A's stop is irrelevant; A's stop codon and its position inside B
  - -1 slippery heptamers X XXY YYZ (XXX identical, YYY in AAA/TTT, zero-frame codons XXY YYZ in
    A's frame) and +1 shift motifs (CTT TGA / CCC TGA / TTT TGA, i.e. a slippery codon followed by an
    in-frame stop) anywhere between 60 nt upstream of B's start and A's stop codon
  - Shine-Dalgarno-like purine run (AGGAGG, GGAGG, AGGAG, GAGG, GGAG, AGGA) 4-15 nt upstream of B's start
  - fusion product under a hypothetical shift at each slippery site: length of A-frame prefix +
    B-frame continuation to B's stop (a fused ORF would read A then B)
Writes overlap_check.tsv and prints the per-rank summary.
"""
import hashlib
import re
from pathlib import Path

import duckdb
import pandas as pd

HERE = Path(__file__).resolve().parent
W = HERE / "work" / "s2"
SHARUR = "/Users/jacob/Documents/Sandbox/Sharur/Sharur/data/dpann/sharur.duckdb"
CODON = {}
b = "TCAG"
aa = "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"
for i, x in enumerate(b):
    for j, y in enumerate(b):
        for k, z in enumerate(b):
            CODON[x + y + z] = aa[16 * i + 4 * j + k]
STARTS = {"ATG", "GTG", "TTG"}
STOPS = {"TAA", "TAG", "TGA"}


def rc(s):
    return s[::-1].translate(str.maketrans("ACGTN", "TGCAN"))


def tr(s):
    return "".join(CODON.get(s[i:i + 3], "X") for i in range(0, len(s) - 2, 3))


def read_fasta(p):
    out, h = {}, None
    for line in open(p):
        if line.startswith(">"):
            h = line[1:].strip()
            out[h] = []
        else:
            out[h].append(line.strip())
    return {k: "".join(v) for k, v in out.items()}


def main():
    inst = pd.read_csv(W / "overlap_instances.tsv", sep="\t")
    fa = read_fasta(W / "overlap_contigs.fna")
    contigs = {}
    for h, s in fa.items():
        genome, contig, sha, _ = h.split("|")
        contigs[(genome, contig)] = (s, sha)
    sh = duckdb.connect(SHARUR, read_only=True)
    sh.execute("SET threads=1")
    pids = inst.protein_id.tolist()
    prot = dict(sh.execute("SELECT protein_id, rtrim(sequence,'*') FROM proteins WHERE protein_id IN ("
                           + ",".join("?" * len(pids)) + ")", pids).fetchall())
    prov = {(r[1], r[0]): r[2] for r in sh.execute(
        "SELECT contig_id, bin_id, source_sequence_sha256 FROM assembly_contig_provenance").fetchall()
        if (r[1], r[0]) in contigs}
    rows = []
    for (rank, genome, contig), d in inst.groupby(["rank", "genome", "contig"]):
        # one or more instances per contig; pair by order of appearance in the tsv
        recs = d.to_dict("records")
        for k in range(0, len(recs), 2):
            A, B = recs[k], recs[k + 1]
            assert A["slot"] == "A" and B["slot"] == "B"
            seq, sha = contigs[(genome, contig)]
            sha_ok = (hashlib.sha256(seq.encode()).hexdigest() == sha == prov.get((genome, contig)))
            strand = int(A["strand"])
            assert strand == int(B["strand"])
            # transcript coordinates: take a window spanning both genes, oriented 5'->3'
            lo = min(A["start"], B["start"]) - 200
            hi = max(A["end"], B["end"]) + 200
            lo0 = max(1, lo)
            hi0 = min(len(seq), hi)
            win = seq[lo0 - 1:hi0]
            if strand > 0:
                t = win
                pos = lambda x: x - lo0                      # 1-based contig coord -> 0-based transcript
                a0, a1 = pos(A["start"]), pos(A["end"])
                b0, b1 = pos(B["start"]), pos(B["end"])
            else:
                t = rc(win)
                pos = lambda x: hi0 - x
                a0, a1 = pos(A["end"]), pos(A["start"])
                b0, b1 = pos(B["end"]), pos(B["start"])
            a_cds, b_cds = t[a0:a1 + 1], t[b0:b1 + 1]
            pa, pb = tr(a_cds), tr(b_cds)
            ok_a = ("M" + pa[1:].rstrip("*")) == prot[A["protein_id"]] or pa.rstrip("*")[1:] == prot[A["protein_id"]][1:]
            ok_b = ("M" + pb[1:].rstrip("*")) == prot[B["protein_id"]] or pb.rstrip("*")[1:] == prot[B["protein_id"]][1:]
            upstream_first = a0 < b0
            assert upstream_first, "A must be upstream (transcription order a_first for ranks 27 and 106)"
            overlap = a1 - b0 + 1
            frame = (b0 - a0) % 3
            # B's frame upstream of its start: nearest in-frame stop
            i = b0 - 3
            open_codons, alt_starts = 0, []
            while i >= 0:
                c = t[i:i + 3]
                if c in STOPS:
                    break
                if c in STARTS:
                    alt_starts.append(-(b0 - i) // 3)
                open_codons += 1
                i -= 3
            upstream_stop_nt = b0 - i if i >= 0 else None
            # slippery motifs between b0-60 and A's stop codon, in A's frame
            sl_m1, sl_p1, fus = [], [], []
            for s in range(max(a0, b0 - 60), a1 - 5):
                hept = t[s:s + 7]
                if len(hept) < 7:
                    continue
                if (s + 1 - a0) % 3 == 0 and hept[0] == hept[1] == hept[2] and hept[3] == hept[4] == hept[5] \
                        and hept[3:6] in ("AAA", "TTT"):
                    sl_m1.append(f"{s - b0:+d}:{hept}")
                    # -1 shift after the YYZ codon's P-site: continue in frame (s + 4 - 1) ... B frame?
                    shift_at = s + 4 - 1
                    if (shift_at - b0) % 3 == 0:
                        fus.append(f"-1@{s - b0:+d}->Bframe")
                if (s - a0) % 3 == 0 and hept[:3] in ("CTT", "CCC", "TTT") and hept[3:6] in STOPS:
                    sl_p1.append(f"{s - b0:+d}:{hept[:6]}")
                    if (s + 4 - b0) % 3 == 0:
                        fus.append(f"+1@{s - b0:+d}->Bframe")
            sd = ""
            up = t[max(0, b0 - 15):max(0, b0 - 4)]
            for m in ("AGGAGG", "GGAGG", "AGGAG", "GAGG", "GGAG", "AGGA"):
                if m in up:
                    sd = m
                    break
            rows.append(dict(rank=rank, genome=genome, contig=contig, strand=strand, a_gid=A["gid"], b_gid=B["gid"],
                             contig_sha256_ok=sha_ok, a_translation_matches=ok_a, b_translation_matches=ok_b,
                             a_len_aa=len(pa.rstrip("*")), b_len_aa=len(pb.rstrip("*")),
                             a_stop=a_cds[-3:], b_start_codon=b_cds[:3], overlap_nt=overlap,
                             b_frame_vs_a={1: "+1", 2: "-1", 0: "0"}[frame],
                             b_upstream_open_codons=open_codons, b_upstream_stop_nt=upstream_stop_nt,
                             b_alt_inframe_starts_codons=";".join(map(str, alt_starts[:5])),
                             sd_like_upstream_of_b=sd, b_nterm=pb[:15], a_cterm=pa.rstrip("*")[-15:],
                             overlap_seq=t[b0:a1 + 1], slippery_m1=";".join(sl_m1), shift_p1_motifs=";".join(sl_p1),
                             shift_to_b_frame=";".join(fus)))
    out = pd.DataFrame(rows)
    out.to_csv(HERE / "overlap_check.tsv", sep="\t", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 60)
    for rk, d in out.groupby("rank"):
        print(f"\n== rank {rk}: {len(d)} instances")
        print(d[["genome", "contig_sha256_ok", "a_translation_matches", "b_translation_matches", "a_stop", "b_start_codon",
                 "overlap_nt", "b_frame_vs_a", "b_upstream_open_codons", "b_alt_inframe_starts_codons", "sd_like_upstream_of_b",
                 "b_nterm", "slippery_m1", "shift_p1_motifs", "shift_to_b_frame"]].to_string())


if __name__ == "__main__":
    main()
