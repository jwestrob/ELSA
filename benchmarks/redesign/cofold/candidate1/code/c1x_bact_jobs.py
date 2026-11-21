"""Task 6 (local, seconds): Biohub ESMFold2 jobs for the bacterial A-B(-C) loci (public sequences from NCBI).

Locus selection: every public locus whose +-3 window holds A, B (or a fused A|B ORF) and a C-family CDS (family
HMM E <= 1e-5) of >= 180 aa, one locus per lineage (organism name without strain) in nb_per_locus order, at most
14; plus the two B-C loci without A outside CPR (L03 Bathyarchaeota, L23 Omnitrophota): bx_bc, bx_bX (B + C'),
bx_Xc (B' + C).
Per locus (single sequence, as the DPANN screen):
  bx_abc   A + B + C                  bx_ab  A + B             bx_bc  B + C
  bx_abX   A + B + C'  (C' = CDS of the same record, length within +-15 % of C, no A/B/C family hit, fewest Pfam
                        domains; else +-25 %)
  bx_aXc   A + B' + C  (B' = same rule against B)
Fused A|B loci: bx_fc (fused + C) and bx_fX (fused + C').
Writes work/bact/jobs_bact.jsonl and work/bact/manifest.tsv.
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent.parent
NB = HERE / "work" / "nb"
O = HERE / "work" / "bact"
O.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(17)


def rf(p):
    d, h = {}, None
    for line in open(p):
        line = line.strip()
        if line.startswith(">"):
            h = line[1:].split()[0]
            d[h] = []
        elif h:
            d[h].append(line)
    return {k: "".join(v) for k, v in d.items()}


def pick(rec, L, exclude):
    for tol in (0.15, 0.25):
        c = rec[(rec.length >= (1 - tol) * L) & (rec.length <= (1 + tol) * L) & ~rec.key.isin(exclude) &
                ~(rec.isA | rec.isB | rec.isC | rec.weakA | rec.weakB | rec.weakC)].copy()
        if len(c):
            c["npf"] = c.pfam.fillna("").map(lambda s: 0 if not s else len(s.split("+")))
            c = c.sort_values(["npf", "length"]).iloc[:5]
            return c.iloc[RNG.integers(len(c))]
    return None


def main():
    seqs = rf(NB / "contig_proteins.faa")
    Wn = pd.read_csv(NB / "nb_windows.tsv", sep="\t")
    P = pd.read_csv(NB / "nb_per_locus.tsv", sep="\t")
    cds = pd.read_csv(NB / "cds.tsv", sep="\t")
    cds["key"] = cds.nuc + "|" + cds.protein_id
    fam = Wn[["nuc", "protein_id", "fam", "pfam"]].drop_duplicates()
    allc = cds.merge(fam, on=["nuc", "protein_id"], how="left")
    # family flags for every CDS (not only window members): recompute from the domtbl via nb_windows' parser output
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from c1x_nb_analyze import dom  # noqa: E402
    fd = dom(NB / "nb_c1abc.domtbl").sort_values("E").drop_duplicates(["prot", "hmm"])
    pf = dom(NB / "nb_pfam.domtbl", pfam=True)
    pf = pf[pf.iE <= 1e-5].sort_values("af").groupby("prot").hmm.apply(lambda x: "+".join(x)).to_dict()
    allc["pfam"] = allc.key.map(lambda k: pf.get(k, ""))
    for t in "ABC":
        s = set(fd[(fd.hmm == f"c1{t}") & (fd.E <= 1e-5)].prot)
        w = set(fd[(fd.hmm == f"c1{t}") & (fd.E > 1e-5) & (fd.E <= 1e-3)].prot)
        allc[f"is{t}"] = allc.key.isin(s)
        allc[f"weak{t}"] = allc.key.isin(w)
    jobs, man, seen_org = [], [], set()
    for _, p in P.iterrows():
        w = Wn[Wn.locus == p.locus]
        bc_only = (w.isC.any() and w.isB.any() and not w.isA.any() and not p.fused
                   and p.locus in ("L03", "L23"))          # Bathyarchaeota and Omnitrophota B-C loci
        if not ((w.isC.any()) and ((w.isA.any() and w.isB.any()) or p.fused or bc_only)):
            continue
        Cg0 = w[w.isC].sort_values("offset").iloc[0]
        if len(seqs[f"{p.nuc}|{Cg0.protein_id}"]) < 180:   # C fragment (split or truncated C): skip locus
            continue
        org = re.sub(r"\s+(GW2011|RIFCSP|RIFOXY|RBG|CG\d|CG_|bacterium).*", "", p.organism).strip()
        if org in seen_org or (len(seen_org) >= 14 and not bc_only):
            continue
        seen_org.add(org)
        rec = allc[allc.nuc == p.nuc]
        Cg = w[w.isC].sort_values("offset").iloc[0]
        kC = f"{p.nuc}|{Cg.protein_id}"
        sC = seqs[kC]
        tag = p.locus
        if p.fused:
            Fg = w[w.is_target].iloc[0]
            kF = f"{p.nuc}|{Fg.protein_id}"
            cp = pick(rec, len(sC), {kC, kF})
            jobs.append(dict(fold_id=f"bx_fc_{tag}", chains=[dict(id="A", seq=seqs[kF]), dict(id="B", seq=sC)]))
            if cp is not None:
                jobs.append(dict(fold_id=f"bx_fX_{tag}", chains=[dict(id="A", seq=seqs[kF]), dict(id="B", seq=seqs[cp.key])]))
            man.append(dict(locus=tag, organism=p.organism, kind="fused", A=kF, B="", C=kC, Cprime=None if cp is None else cp.key,
                            Bprime=None, lenA=len(seqs[kF]), lenB=0, lenC=len(sC)))
        if bc_only:
            Bg = w[w.isB].iloc[0]
            kB = f"{p.nuc}|{Bg.protein_id}"
            sB = seqs[kB]
            cp = pick(rec, len(sC), {kB, kC})
            bp = pick(rec, len(sB), {kB, kC})
            jobs.append(dict(fold_id=f"bx_bc_{tag}", chains=[dict(id="A", seq=sB), dict(id="B", seq=sC)]))
            if cp is not None:
                jobs.append(dict(fold_id=f"bx_bX_{tag}", chains=[dict(id="A", seq=sB), dict(id="B", seq=seqs[cp.key])]))
            if bp is not None:
                jobs.append(dict(fold_id=f"bx_Xc_{tag}", chains=[dict(id="A", seq=seqs[bp.key]), dict(id="B", seq=sC)]))
            man.append(dict(locus=tag, organism=p.organism, kind="BC", A="", B=kB, C=kC,
                            Cprime=None if cp is None else cp.key, Bprime=None if bp is None else bp.key,
                            lenA=0, lenB=len(sB), lenC=len(sC)))
        if w.isA.any() and w.isB.any():
            Ag = w[w.isA].iloc[0]
            Bg = w[w.isB].iloc[0]
            kA, kB = f"{p.nuc}|{Ag.protein_id}", f"{p.nuc}|{Bg.protein_id}"
            if kA == kB:
                continue
            sA, sB = seqs[kA], seqs[kB]
            cp = pick(rec, len(sC), {kA, kB, kC})
            bp = pick(rec, len(sB), {kA, kB, kC})
            ch = lambda *xs: [dict(id=i, seq=s) for i, s in zip("ABC", xs)]  # noqa: E731
            jobs += [dict(fold_id=f"bx_abc_{tag}", chains=ch(sA, sB, sC)),
                     dict(fold_id=f"bx_ab_{tag}", chains=ch(sA, sB)),
                     dict(fold_id=f"bx_bc_{tag}", chains=ch(sB, sC))]
            if cp is not None:
                jobs.append(dict(fold_id=f"bx_abX_{tag}", chains=ch(sA, sB, seqs[cp.key])))
            if bp is not None:
                jobs.append(dict(fold_id=f"bx_aXc_{tag}", chains=ch(sA, seqs[bp.key], sC)))
            man.append(dict(locus=tag, organism=p.organism, kind="ABC", A=kA, B=kB, C=kC,
                            Cprime=None if cp is None else cp.key, Bprime=None if bp is None else bp.key,
                            lenA=len(sA), lenB=len(sB), lenC=len(sC),
                            lenCprime=None if cp is None else len(seqs[cp.key]),
                            lenBprime=None if bp is None else len(seqs[bp.key])))
    with open(O / "jobs_bact.jsonl", "w") as f:
        for j in jobs:
            f.write(json.dumps(j) + "\n")
    M = pd.DataFrame(man)
    M.to_csv(O / "manifest.tsv", sep="\t", index=False)
    pd.set_option("display.width", 250)
    print(len(jobs), "jobs")
    print(M.to_string())


if __name__ == "__main__":
    main()
