"""Task 5 control (local, seconds): the c1x_overlap.py scan on every other adjacent co-oriented gene pair of the same
337 contigs (A-, B- and C-family genes excluded), stratified by junction class, to give the background rate of
-1 slippery heptamers, fusion-competent slip sites, in-frame read-through junctions and SD-like motifs in these
genomes. Classes: 'ATGA' (4-nt overlap ATGA), 'overlap_other' (1-40 nt, not ATGA), 'gap' (0 to -40 nt).
Writes work/ovl/overlap_control.tsv and adds 'control' to work/ovl/overlap_summary.json.
"""
import json
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import fisher_exact

sys.path.insert(0, str(Path(__file__).resolve().parent))
from c1x_overlap import read_fasta, scan, transcript  # noqa: E402

HERE = Path(__file__).resolve().parent.parent
W = HERE / "work" / "ovl"


def cls(o, seq):
    if o == 4 and seq == "ATGA":
        return "ATGA"
    if 1 <= o <= 40:
        return "overlap_other"
    if -40 <= o <= 0:
        return "gap"
    return "other"


def main():
    J = pd.read_csv(W / "junctions.tsv", sep="\t")
    R = pd.read_csv(W / "overlap_scan.tsv", sep="\t")
    g = pd.read_parquet(HERE.parent / "work" / "genes_dark.parquet")
    V = pd.read_csv(HERE / "work" / "variants" / "C_family_context.tsv", sep="\t")
    fam = set(J.up_gid) | set(J.dn_gid) | set(V.gid)
    hp = pd.read_csv(HERE.parent / "deepdive" / "work" / "c1_hmm_pairs.tsv", sep="\t")
    fam |= set(hp.gidA) | set(hp.gidB)
    fa = read_fasta(W / "contigs.fna")
    contigs = {(h.split("|")[0], h.split("|")[1]): s for h, s in fa.items()}
    key = J.drop_duplicates(["genome_code", "contig_code"])[["genome_code", "contig_code", "genome", "contig"]]
    rows = []
    for k in key.itertuples():
        seq = contigs[(k.genome, k.contig)]
        x = g[(g.genome_code == k.genome_code) & (g.contig_code == k.contig_code)].sort_values("position_index")
        x = x.to_dict("records")
        for a, b in zip(x, x[1:]):
            if a["strand"] != b["strand"] or a["gid"] in fam or b["gid"] in fam:
                continue
            U, D = (a, b) if a["strand"] > 0 else (b, a)
            t, ((u0, u1), (d0, d1)) = transcript(seq, int(U["strand"]), [(U["start"], U["end"]), (D["start"], D["end"])])
            if not (u0 < d0):
                continue
            r = scan(t, u0, u1, d0, d1)
            r.update(genome=k.genome, contig=k.contig, up_gid=U["gid"], dn_gid=D["gid"])
            rows.append(r)
    C = pd.DataFrame(rows)
    C["cls"] = [cls(o, s) for o, s in zip(C.overlap_nt, C.overlap_seq.fillna(""))]
    R["cls"] = [cls(o, s) for o, s in zip(R.overlap_nt, R.overlap_seq.fillna(""))]
    C.to_csv(W / "overlap_control.tsv", sep="\t", index=False)
    S = json.load(open(W / "overlap_summary.json"))
    out = {}
    for j in ("AB", "BC"):
        x = R[R.junction == j]
        for c in ("ATGA", "overlap_other", "gap", "all"):
            xs = x if c == "all" else x[x.cls == c]
            cs = C[C.cls.isin(["ATGA", "overlap_other", "gap"])] if c == "all" else C[C.cls == c]
            if len(xs) == 0:
                continue
            d = {}
            for m, f in (("slippery_m1", lambda z: z.slippery_m1.fillna("") != ""),
                         ("fusion_competent", lambda z: z.fusion_competent_slip.fillna("") != ""),
                         ("inframe_readthrough", lambda z: z.inframe_readthrough.astype(bool)),
                         ("sd_like", lambda z: z.sd_like.fillna("") != "")):
                a, b = int(f(xs).sum()), int(f(cs).sum())
                orr, p = fisher_exact([[a, len(xs) - a], [b, len(cs) - b]])
                d[m] = dict(test=f"{a}/{len(xs)}", control=f"{b}/{len(cs)}", OR=round(orr, 2), p=float(f"{p:.2g}"))
            out[f"{j}_{c}"] = d
    out["control_classes"] = C.cls.value_counts().to_dict()
    out["control_frames_ATGA"] = C[C.cls == "ATGA"].frame.value_counts().to_dict()
    out["control_ATGA_fraction_of_adjacent_pairs"] = f"{int((C.cls == 'ATGA').sum())}/{len(C)}"
    S["control"] = out
    json.dump(S, open(W / "overlap_summary.json", "w"), indent=1, default=str)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
