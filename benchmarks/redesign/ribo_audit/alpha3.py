#!/usr/bin/env python
"""ribo_audit part 3, POST HOC: the secY-adk-map-infA insert and contig edges for every bacterial run.

For each g2_m16 run: genes between the run and its contig ends; whether secY / the alpha-operon markers (S13, S11, S4, L17,
L36) lie inside the run or within 12 genes past it on the same contig; for runs that contain the alpha block, the
identities of the unmarked genes between L15 and the next marker. Annotation: frozen2 genes_annot (lit) + work3 extra KOs.
Writes work3/alpha3.parquet.
"""
from pathlib import Path

import numpy as np
import pandas as pd

RD = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111")
PKG = RD / "ws_srvp_bins/package"
FZ = RD / "long_loci/frozen2"
W3 = RD / "ribo_audit/work3"
ALPHA = {"S13", "S11", "S4", "L17", "L36"}
EXTRA = {"K02601": "nusG", "K03043": "rpoB", "K03046": "rpoC", "K02358": "tufA", "K02355": "fusA", "K00939": "adk",
         "K01265": "map", "K02518": "infA", "K03073": "secE", "K01883": "cysS", "K03685": "rnc", "K03218": "rlmB",
         "K16786": "ecfA1", "K16787": "ecfA2", "K16785": "ecfT", "K06173": "truA"}

g = pd.read_parquet(PKG / "universe/genes.parquet", columns=["gid", "contig_code"]).sort_values("gid")
cc = g.contig_code.to_numpy()
an = pd.read_parquet(FZ / "annot/genes_annot.parquet", columns=["gid", "cls", "lit", "ko"]).sort_values("gid")
lab = an.lit.fillna("").to_numpy().astype(object)
xk = pd.read_parquet(W3 / "annot/extra_ko.parquet")
kcol = "ko" if "ko" in xk.columns else [c for c in xk.columns if c.lower().startswith("k")][0]
for gid_, k in zip(xk.gid.to_numpy(), xk[kcol].to_numpy()):
    if not lab[gid_]:
        lab[gid_] = EXTRA.get(k, k)
cstart = pd.Series(np.arange(len(cc))).groupby(cc).min().to_dict()
cend = pd.Series(np.arange(len(cc))).groupby(cc).max().to_dict()
gm = pd.read_parquet(PKG / "universe/genomes.parquet").sort_values("genome_code")
nct = gm.n_contigs.to_numpy()
runs = pd.read_parquet(PKG / "ribosomal/runs.parquet")
runs = runs[(runs.run_set == "g2") & runs.m16 & (runs.domain != "Archaea")].reset_index(drop=True)
rows = []
for r in runs.itertuples():
    s, e = r.gid_start, r.gid_end
    c = cc[s]
    inside = [lab[k] for k in range(s, e + 1)]
    after_r = [lab[k] for k in range(e + 1, min(e + 13, cend[c] + 1))]
    after_l = [lab[k] for k in range(max(s - 12, cstart[c]), s)]
    alpha_in = sorted(set(inside) & ALPHA)
    between = ""
    if "L15" in inside and alpha_in:
        i = inside.index("L15")
        j = next((t for t in range(i + 1, len(inside)) if inside[t] in ALPHA), None)
        if j is None:
            i = len(inside) - 1 - inside[::-1].index("L15")
            j = next((t for t in range(i - 1, -1, -1) if inside[t] in ALPHA), None)
            if j is not None:
                i, j = j, i
        if j is not None:
            between = "-".join(x if x else "?" for x in inside[i + 1:j])
    rows.append(dict(run_id=r.run_id, gtype="Bact_closed" if nct[r.genome_code] == 1 else "Bact_multi",
                     to_contig_start=int(s - cstart[c]), to_contig_end=int(cend[c] - e),
                     secY_in_run="secY" in inside, alpha_in_run=",".join(alpha_in),
                     alpha_next12=",".join(sorted((set(after_r) | set(after_l)) & ALPHA)),
                     secY_next12="secY" in after_r or "secY" in after_l, between_L15_alpha=between))
A = pd.DataFrame(rows)
A.to_parquet(W3 / "alpha3.parquet", index=False)
print(A.groupby("gtype").agg(runs=("run_id", "size"), alpha_in=("alpha_in_run", lambda s: (s != "").sum()),
                             alpha_next=("alpha_next12", lambda s: (s != "").sum())).to_string())
print(A[A.between_L15_alpha != ""].between_L15_alpha.value_counts().head(15).to_string())
