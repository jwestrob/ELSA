#!/usr/bin/env python
"""lut_loci e10 (Mac): the fixed site test (tests_rule.json T_struct, code s02 functions unchanged) on the beta-roll follow-up
models (folds/br_apo). Writes data/sites_br.tsv."""
import sys

import numpy as np
import pandas as pd
from Bio import SeqIO

B = "/Users/jacob/Documents/Sandbox/elsa_test/ELSA/benchmarks/redesign/dpann_leads/lut_loci"
sys.path.insert(0, B + "/code")
import s02_sites as S2  # noqa: E402

rows = []
for fa in SeqIO.parse(B + "/folds_in/br_apo.faa", "fasta"):
    nm = fa.id
    role, fam, gid = nm.split("__")
    seq = str(fa.seq)
    P = S2.parse(f"{B}/folds/br_apo/{nm}.pdb")
    P = P[~P.het]
    pl = P[P["name"] == "CA"].set_index("resi").b.to_dict()
    if role == "brneg":  # public reference without a collection family: structure-only, as the public positives
        cons, ncons, nfam = None, 0, 0
    else:
        cons, ncons, nfam, defrac = S2.conservation(fam, gid, seq)
    C = S2.centers(P, cons, pl)
    r = dict(name=nm, role=role, family=fam, gid=gid, length=len(seq), mean_plddt=round(float(np.mean(list(pl.values()))), 1),
             fam_n=nfam, cons_acidic_per100=round(100 * ncons / len(seq), 2), ef_strict=len(S2.EF.findall(seq)),
             ef_relaxed=len(S2.EF_RELAXED.findall(seq)))
    ok = C[C.site_plddt >= S2.PLDDT_MIN] if len(C) else C
    r["score_struct"] = int(ok.cap.max()) if len(ok) else 0
    r["score_cons"] = int(ok.cons.max()) if len(ok) else 0
    if len(ok):
        b = ok.sort_values(["cons", "cap", "site_plddt"], ascending=False).iloc[0]
        r.update(best_res=b.res, best_site_plddt=round(b.site_plddt, 1))
        xyz = P[["x", "y", "z"]].to_numpy()
        d = np.linalg.norm(xyz - np.array([b.x, b.y, b.z]), axis=1)
        near = P[d <= 3.0]
        r["best_O_within3"] = int((near.el == "O").sum())
        r["best_bbO_within3"] = int((near["name"] == "O").sum())
    r["PASS_site"] = bool(r["score_cons"] >= S2.PASS_CONS)
    rows.append(r)
D = pd.DataFrame(rows)
D.to_csv(B + "/data/sites_br.tsv", sep="\t", index=False)
pd.set_option("display.width", 250)
print(D.to_string(index=False))
