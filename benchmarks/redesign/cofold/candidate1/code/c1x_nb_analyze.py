"""Task 1, step 3 (local, seconds): neighbourhoods of the 40 public loci (bacteria + 1 non-DPANN archaeon).

Inputs (work/nb/): loci.tsv, cds.tsv (every CDS of each nucleotide record, NCBI), nb_c1abc.domtbl (A, B, C family
HMMs vs every CDS, E <= 10 reported), nb_pfam.domtbl (Pfam 39.0 hmmscan, E <= 1e-3).
Family calls: full-sequence E <= 1e-5 ('A', 'B', 'C'); 1e-5 < E <= 1e-3 reported as weak ('c?').
Window: +-3 CDS around the locus core (the target CDS, or the span of the targets when A and B targets are both
present), offsets in the transcription orientation of the first target.
Control windows: up to 3 per locus on the same record, centred on a CDS whose length is within +-25 % of the
target's, >= 6 CDS outside the real window, same +-3 rule (seed 17).
Writes work/nb/nb_windows.tsv, nb_summary.json.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

HERE = Path(__file__).resolve().parent.parent
NB = HERE / "work" / "nb"
RNG = np.random.default_rng(17)


def dom(p, pfam=False):
    rows = []
    for line in open(p):
        if line.startswith("#"):
            continue
        f = line.split()
        if len(f) < 22:
            continue
        if pfam:   # hmmscan: target = Pfam model, query = protein
            rows.append(dict(prot=f[3], hmm=f[0], acc=f[1], E=float(f[6]), iE=float(f[12]), hf=int(f[15]),
                             ht=int(f[16]), af=int(f[17]), at_=int(f[18])))
        else:      # hmmsearch: target = protein, query = family HMM
            rows.append(dict(prot=f[0], hmm=f[3], E=float(f[6]), iE=float(f[12]), hf=int(f[15]), ht=int(f[16]),
                             af=int(f[17]), at_=int(f[18])))
    return pd.DataFrame(rows)


def main():
    L = pd.read_csv(NB / "loci.tsv", sep="\t")
    C = pd.read_csv(NB / "cds.tsv", sep="\t")
    C["key"] = C.nuc + "|" + C.protein_id
    fam = dom(NB / "nb_c1abc.domtbl")
    pf = dom(NB / "nb_pfam.domtbl", pfam=True)
    pf = pf[pf.iE <= 1e-5]
    famb = fam.sort_values("E").drop_duplicates(["prot", "hmm"])
    call = {}
    for r in famb.itertuples():
        tag = r.hmm[-1]
        lab = tag if r.E <= 1e-5 else (tag.lower() + "?" if r.E <= 1e-3 else None)
        if lab:
            call.setdefault(r.prot, []).append(f"{lab}(E{r.E:.0e}:{r.hf}-{r.ht})")
    pfd = pf.sort_values("af").groupby("prot").hmm.apply(lambda x: "+".join(x)).to_dict()
    C["fam"] = C.key.map(lambda k: ";".join(call.get(k, [])))
    C["pfam"] = C.key.map(lambda k: pfd.get(k, ""))
    for t in "ABC":
        C[f"is{t}"] = C.fam.str.contains(rf"(?:^|;){t}\(", regex=True)
        C[f"weak{t}"] = C.fam.str.contains(rf"(?:^|;){t.lower()}\?\(", regex=True)
    rows, ctrl = [], []
    for locus, x in L.groupby("locus", sort=False):
        nuc = x.nuc.iloc[0]
        rec = C[C.nuc == nuc].sort_values("idx").reset_index(drop=True)
        n = len(rec)
        lo, hi = int(x.idx.min()), int(x.idx.max())
        st = int(x.target_strand.iloc[0])
        t0 = lo if st > 0 else hi
        win = rec[(rec.idx >= lo - 3) & (rec.idx <= hi + 3)].copy()
        win["offset"] = (win.idx - t0) * st
        win["same_strand"] = win.strand == st
        win["locus"] = locus
        win["is_target"] = win.protein_id.isin(set(x.pid))
        rows.append(win)
        # controls
        tl = int(x.target_len.iloc[0])
        cand = rec[(rec.length >= 0.75 * tl) & (rec.length <= 1.25 * tl) &
                   ((rec.idx < lo - 3 - 6) | (rec.idx > hi + 3 + 6))]
        pick = cand.iloc[RNG.permutation(len(cand))[:3]] if len(cand) else cand
        for j, c in enumerate(pick.itertuples(), 1):
            w = rec[(rec.idx >= c.idx - 3) & (rec.idx <= c.idx + 3) & (rec.idx != c.idx)].copy()
            w["locus"], w["ctrl"] = locus, j
            ctrl.append(w)
    Wn = pd.concat(rows)
    Cn = pd.concat(ctrl) if ctrl else pd.DataFrame()
    Wn.to_csv(NB / "nb_windows.tsv", sep="\t", index=False)
    S = {"n_loci": int(L.locus.nunique())}
    per = []
    for locus, w in Wn.groupby("locus", sort=False):
        core = w[w.is_target]
        nonT = w[~w.is_target]
        r = dict(locus=locus, nuc=w.nuc.iloc[0], organism=L[L.locus == locus].organism.iloc[0],
                 cls=L[L.locus == locus].scope_cls.iloc[0], fused=bool(L[L.locus == locus].fused.any()),
                 n_window=len(w), n_record=int(L[L.locus == locus].n_cds_record.iloc[0]),
                 A_in_window=int(w.isA.sum()), B_in_window=int(w.isB.sum()), C_in_window=int(w.isC.sum()),
                 weakC_in_window=int(w.weakC.sum()),
                 C_offsets=";".join(f"{o:+d}{'' if s else '(opp)'}" for o, s in zip(w[w.isC].offset, w[w.isC].same_strand)),
                 A_offsets=";".join(f"{o:+d}" for o in w[w.isA].offset),
                 B_offsets=";".join(f"{o:+d}" for o in w[w.isB].offset),
                 window_edge=bool(w.idx.min() == 0 or w.idx.max() == int(L[L.locus == locus].n_cds_record.iloc[0]) - 1),
                 layout=" | ".join(f"{o:+d}{'>' if s else '<'}{l}aa[{fm or '-'}]{{{p[:40]}}}" for o, s, l, fm, p in
                                   zip(w.offset, w.same_strand, w.length, w.fam.str.replace(r'\(E[^)]*\)', '', regex=True), w.pfam)))
        per.append(r)
    P = pd.DataFrame(per)
    P.to_csv(NB / "nb_per_locus.tsv", sep="\t", index=False)
    real_C = int((P.C_in_window > 0).sum())
    ctrl_C = int(Cn.groupby(["locus", "ctrl"]).isC.any().sum()) if len(Cn) else 0
    n_ctrl = int(Cn.groupby(["locus", "ctrl"]).ngroups) if len(Cn) else 0
    orr, p = fisher_exact([[real_C, len(P) - real_C], [ctrl_C, n_ctrl - ctrl_C]])
    S["loci_with_C_pm3"] = f"{real_C}/{len(P)}"
    S["control_windows_with_C"] = f"{ctrl_C}/{n_ctrl}"
    S["fisher_C"] = dict(OR=orr, p=p)
    S["loci_with_weakC_only"] = int(((P.C_in_window == 0) & (P.weakC_in_window > 0)).sum())
    S["loci_with_A_and_B"] = int(((P.A_in_window > 0) & (P.B_in_window > 0)).sum())
    S["loci_with_fused_AB"] = int(P.fused.sum())
    S["C_offsets"] = P.C_offsets.value_counts().to_dict()
    # recurring Pfam domains in real vs control windows (A/B/C-family CDS excluded)
    def dcount(df, groups):
        d = {}
        for _, w in df.groupby(groups):
            for dm in set("+".join(w[~(w.isA | w.isB | w.isC)].pfam).split("+")) - {""}:
                d[dm] = d.get(dm, 0) + 1
        return d
    dr = dcount(Wn, "locus")
    dc = dcount(Cn, ["locus", "ctrl"]) if len(Cn) else {}
    rec = []
    for dm, k in dr.items():
        kc = dc.get(dm, 0)
        o, pp = fisher_exact([[k, len(P) - k], [kc, n_ctrl - kc]])
        rec.append(dict(domain=dm, real=k, ctrl=kc, real_frac=k / len(P), ctrl_frac=kc / max(n_ctrl, 1), p=pp))
    D = pd.DataFrame(rec).sort_values(["real", "p"], ascending=[False, True])
    D.to_csv(NB / "nb_domain_recurrence.tsv", sep="\t", index=False)
    S["recurring_domains_top"] = D.head(15).to_dict("records")
    S["n_control_windows"] = n_ctrl
    json.dump(S, open(NB / "nb_summary.json", "w"), indent=1, default=str)
    pd.set_option("display.width", 400)
    pd.set_option("display.max_colwidth", 400)
    print(json.dumps({k: v for k, v in S.items() if k != "recurring_domains_top"}, indent=1, default=str))
    print(D.head(15).to_string())
    print(P[["locus", "organism", "fused", "A_offsets", "B_offsets", "C_offsets", "weakC_in_window", "window_edge"]].to_string())


if __name__ == "__main__":
    main()
