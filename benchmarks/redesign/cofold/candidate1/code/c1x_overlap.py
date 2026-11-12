"""Task 5 (local, seconds): nucleotide scan of the A->B and B->C junctions (method of cofold/overlap_check.py),
plus the split-C junctions (N-terminal C piece followed by a C-terminal C piece) and the slot-4 C region.

Inputs: work/ovl/junctions.tsv (A->B: 325 HMM pairs; B->C: 330 B-HMM genes directly upstream of a C-family gene),
work/ovl/contigs.fna (contigs from the DPANN genome FASTAs, biotite, overlap_extract_nt.py), Sharur DPANN DB
(read-only: stored proteins, assembly_contig_provenance SHA-256), ../work/genes_dark.parquet.
Per junction, in transcript orientation, upstream gene U first, downstream gene D:
  contig SHA-256 = provenance; U and D translate (table 11, start read as M) to the stored proteins
  overlap (nt; > 0 overlap), D start codon, frame of D relative to U ((D0 - U0) mod 3: 0 in frame, 1 = +1, 2 = -1)
  D's frame upstream of its start: codons to the nearest in-frame stop ('open stretch'), alternative starts in it
  -1 slippery heptamers X XXY YYZ (XXX identical, YYY AAA/TTT; XXY YYZ in U's frame) and +1 motifs (CTT/CCC/TTT
  followed by a stop in U's frame), between 60 nt upstream of D's start and U's stop
  fusion-competent slip site: the shifted frame is D's frame AND D's frame has no stop between the slip site and
  D's start (a frameshift there would read U then D as one polypeptide)
  in-frame junction (frame 0): U's stop is in D's frame (a read-through would fuse U and D)
  Shine-Dalgarno-like purine run 4-15 nt upstream of D's start
Slot-4 C: 3-frame translation of C start .. C end + 600 nt, local alignment of each frame (stops kept) to the C
of slot 1 (full length) to locate where C-terminal homology continues.
Writes work/ovl/overlap_scan.tsv, work/ovl/overlap_summary.json.
"""
import hashlib
import json
from pathlib import Path

import duckdb
import pandas as pd
from Bio import Align
from Bio.Align import substitution_matrices

HERE = Path(__file__).resolve().parent.parent
W = HERE / "work" / "ovl"
DD = HERE.parent / "deepdive" / "work"
SHARUR = "/Users/jacob/Documents/Sandbox/Sharur/Sharur/data/dpann/sharur.duckdb"
B_ = "TCAG"
AA_ = "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"
CODON = {x + y + z: AA_[16 * i + 4 * j + k] for i, x in enumerate(B_) for j, y in enumerate(B_) for k, z in enumerate(B_)}
STARTS = {"ATG", "GTG", "TTG"}
STOPS = {"TAA", "TAG", "TGA"}
SD = ("AGGAGG", "GGAGG", "AGGAG", "GAGG", "GGAG", "AGGA")


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


def transcript(seq, strand, starts_ends, pad=700):
    lo = max(1, min(x for se in starts_ends for x in se) - pad)
    hi = min(len(seq), max(x for se in starts_ends for x in se) + pad)
    win = seq[lo - 1:hi]
    if strand > 0:
        t, pos = win, (lambda x: x - lo)
        coords = [(pos(s), pos(e)) for s, e in starts_ends]
    else:
        t, pos = rc(win), (lambda x: hi - x)
        coords = [(pos(e), pos(s)) for s, e in starts_ends]
    return t, coords


def scan(t, u0, u1, d0, d1):
    frame = (d0 - u0) % 3
    i, open_c, alt = d0 - 3, 0, []
    while i >= 0:
        c = t[i:i + 3]
        if c in STOPS:
            break
        if c in STARTS:
            alt.append(-(d0 - i) // 3)
        open_c += 1
        i -= 3
    up_stop = d0 - i if i >= 0 else None          # nt from D's start back to its upstream in-frame stop
    open_from = i + 3 if i >= 0 else 0            # first nt of D's open stretch
    sl_m1, sl_p1, fus = [], [], []
    for s in range(max(u0, d0 - 60), u1 - 5):
        h = t[s:s + 7]
        if len(h) < 7:
            continue
        if (s + 1 - u0) % 3 == 0 and h[0] == h[1] == h[2] and h[3] == h[4] == h[5] and h[3:6] in ("AAA", "TTT"):
            sl_m1.append(f"{s - d0:+d}:{h}")
            if frame == 2 and s >= open_from:
                fus.append(f"-1@{s - d0:+d}")
        if (s - u0) % 3 == 0 and h[:3] in ("CTT", "CCC", "TTT") and h[3:6] in STOPS:
            sl_p1.append(f"{s - d0:+d}:{h[:6]}")
            if frame == 1 and s >= open_from:
                fus.append(f"+1@{s - d0:+d}")
    sd = next((m for m in SD if m in t[max(0, d0 - 15):max(0, d0 - 4)]), "")
    inframe_rt = frame == 0 and d0 > u1          # same frame, D after U's stop: read-through would fuse
    rt_gap_stops = None
    if inframe_rt:
        seg = t[u1 + 1:d0]
        rt_gap_stops = sum(seg[k:k + 3] in STOPS for k in range(0, len(seg) - 2, 3)) + 1   # incl. U's own stop
    return dict(overlap_nt=u1 - d0 + 1, frame={0: "0", 1: "+1", 2: "-1"}[frame], d_start=t[d0:d0 + 3],
                u_stop=t[u1 - 2:u1 + 1], d_open_codons_upstream=open_c, d_upstream_stop_nt=up_stop,
                d_alt_starts=";".join(map(str, alt[:5])), slippery_m1=";".join(sl_m1), shift_p1=";".join(sl_p1),
                fusion_competent_slip=";".join(fus), inframe_readthrough=inframe_rt, readthrough_stops=rt_gap_stops,
                sd_like=sd, overlap_seq=t[d0:u1 + 1] if u1 >= d0 else "")


def main():
    J = pd.read_csv(W / "junctions.tsv", sep="\t")
    g = pd.read_parquet(HERE.parent / "work" / "genes_dark.parquet").set_index("gid", drop=False)
    # split-C junctions: C-family gene (token 190823 or any C hit with HMM end <= 120) followed by a C-family
    # C-terminal piece on the same strand
    V = pd.read_csv(HERE / "work" / "variants" / "C_family_context.tsv", sep="\t").set_index("gid")
    extra = []
    for gid, r in V.iterrows():
        dn = r.dn1
        if pd.notna(dn) and int(dn) in V.index and g.at[int(dn), "strand"] == g.at[gid, "strand"]:
            hf_up = int(r.C_hmm.split("-")[1])
            hf_dn = int(V.at[int(dn), "C_hmm"].split("-")[0])
            if hf_dn > hf_up - 30:
                U, D = g.loc[gid], g.loc[int(dn)]
                extra.append(dict(junction="Csplit", genome_code=U.genome_code, contig_code=U.contig_code,
                                  strand=int(U.strand), up_gid=gid, up_pid=U.protein_id, up_start=int(U.start),
                                  up_end=int(U.end), dn_gid=int(dn), dn_pid=D.protein_id, dn_start=int(D.start),
                                  dn_end=int(D.end), genome=None, contig=None, hmm_up=r.C_hmm,
                                  hmm_dn=V.at[int(dn), "C_hmm"]))
    E = pd.DataFrame(extra)
    if len(E):
        key = J.drop_duplicates(["genome_code", "contig_code"]).set_index(["genome_code", "contig_code"])[["genome", "contig"]]
        E["genome"] = [key.loc[(a, b), "genome"] if (a, b) in key.index else None for a, b in zip(E.genome_code, E.contig_code)]
        E["contig"] = [key.loc[(a, b), "contig"] if (a, b) in key.index else None for a, b in zip(E.genome_code, E.contig_code)]
        E = E[E.genome.notna()]
    J = pd.concat([J, E], ignore_index=True)
    fa = read_fasta(W / "contigs.fna")
    contigs = {}
    for h, s in fa.items():
        genome, contig, sha, _ = h.split("|")
        contigs[(genome, contig)] = (s, sha)
    sh = duckdb.connect(SHARUR, read_only=True)
    sh.execute("SET threads=2")
    pids = list(set(J.up_pid) | set(J.dn_pid))
    prot = dict(sh.execute("SELECT protein_id, rtrim(sequence,'*') FROM proteins WHERE protein_id IN ("
                           + ",".join("?" * len(pids)) + ")", pids).fetchall())
    prov = {(r[1], r[0]): r[2] for r in sh.execute(
        "SELECT contig_id, bin_id, source_sequence_sha256 FROM assembly_contig_provenance").fetchall()
        if (r[1], r[0]) in contigs}
    rows = []
    for r in J.itertuples():
        seq, sha = contigs[(r.genome, r.contig)]
        t, ((u0, u1), (d0, d1)) = transcript(seq, r.strand, [(r.up_start, r.up_end), (r.dn_start, r.dn_end)])
        pu, pd_ = tr(t[u0:u1 + 1]), tr(t[d0:d1 + 1])
        ok_u = pu.rstrip("*")[1:] == prot.get(r.up_pid, "")[1:]
        ok_d = pd_.rstrip("*")[1:] == prot.get(r.dn_pid, "")[1:]
        x = dict(junction=r.junction, genome=r.genome, contig=r.contig, up_gid=r.up_gid, dn_gid=r.dn_gid,
                 sha_ok=hashlib.sha256(seq.encode()).hexdigest() == sha == prov.get((r.genome, r.contig)),
                 up_translation_ok=ok_u, dn_translation_ok=ok_d)
        x.update(scan(t, u0, u1, d0, d1))
        if r.junction == "Csplit":
            x["hmm_up"], x["hmm_dn"] = r.hmm_up, r.hmm_dn
        rows.append(x)
    R = pd.DataFrame(rows)
    R.to_csv(W / "overlap_scan.tsv", sep="\t", index=False)
    S = {}
    for j, x in R.groupby("junction"):
        S[j] = dict(n=len(x), sha_ok=int(x.sha_ok.sum()), translations_ok=int((x.up_translation_ok & x.dn_translation_ok).sum()),
                    overlap_gt0=int((x.overlap_nt > 0).sum()), overlap_nt=x.overlap_nt.value_counts().head(8).to_dict(),
                    frame=x.frame.value_counts().to_dict(), d_start=x.d_start.value_counts().to_dict(),
                    ATGA=int(((x.overlap_nt == 4) & (x.overlap_seq == "ATGA")).sum()),
                    d_open_codons_upstream_median=float(x.d_open_codons_upstream.median()),
                    d_open_codons_upstream_q90=float(x.d_open_codons_upstream.quantile(0.9)),
                    any_slippery_m1=int((x.slippery_m1.fillna("") != "").sum()),
                    any_shift_p1=int((x.shift_p1.fillna("") != "").sum()),
                    fusion_competent_slip=int((x.fusion_competent_slip.fillna("") != "").sum()),
                    fusion_competent_examples=x[x.fusion_competent_slip.fillna("") != ""][
                        ["genome", "overlap_nt", "frame", "fusion_competent_slip", "d_open_codons_upstream"]].head(10).to_dict("records"),
                    inframe_readthrough=int(x.inframe_readthrough.sum()),
                    sd_like=int((x.sd_like.fillna("") != "").sum()))
    # slot-4 C region
    sel = pd.read_csv(DD / "c1_selected.tsv", sep="\t")
    caln = read_fasta(DD / "c1_aln_out" / "C_inst.aln.fa")
    ref = caln[sel[sel.slot == 1].instance_id.iloc[0]].replace("-", "")
    s4 = g.loc[1512020]
    j4 = J[(J.dn_gid == 1512020)].iloc[0]
    seq, _ = contigs[(j4.genome, j4.contig)]
    t, ((c0, c1),) = transcript(seq, int(s4.strand), [(int(s4.start), int(s4.end))], pad=600)
    al = Align.PairwiseAligner(mode="local", open_gap_score=-11, extend_gap_score=-1)
    al.substitution_matrix = substitution_matrices.load("BLOSUM62")
    s4out = dict(C_len=len(tr(t[c0:c1 + 1]).rstrip("*")), C_stop=t[c1 - 2:c1 + 1], nt_after_stop=len(t) - c1 - 1)
    for f in range(3):
        start = c1 + 1 + f - 3                     # frames anchored just before C's stop
        prot_f = tr(t[start:])
        a = al.align(prot_f.replace("*", "X"), ref)[0]
        (qs, qe), (rs, re_) = a.aligned[0][0], a.aligned[1][-1]
        s4out[f"frame_{f}"] = dict(score=float(a.score), ref_from=int(a.aligned[1][0][0] + 1), ref_to=int(re_),
                                   query_aa_from=int(a.aligned[0][0][0]), stops_in_first_100aa=prot_f[:100].count("*"),
                                   translation_head=prot_f[:60])
    S["slot4_C"] = s4out
    # the slot-4 C itself against the reference
    a = al.align(tr(t[c0:c1 + 1]).rstrip("*"), ref)[0]
    S["slot4_C"]["C_vs_ref"] = dict(score=float(a.score), ref_from=int(a.aligned[1][0][0] + 1), ref_to=int(a.aligned[1][-1][1]),
                                    ref_len=len(ref), ref_his_cys="H/C at ref residues %d/%d" % (
                                        [i for i, c in enumerate(ref, 1) if c == "H"][0], 0))
    json.dump(S, open(W / "overlap_summary.json", "w"), indent=1, default=str)
    print(json.dumps(S, indent=1, default=str))


if __name__ == "__main__":
    main()
