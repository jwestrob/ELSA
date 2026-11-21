#!/usr/bin/env python
"""ribo_audit part 3: order variability with ribosomal-protein names normalised to one identity per protein.

Identity of a Ribosomal_* gene = KO-derived name (frozen2 annotation 'lit') or Pfam marker, normalised: suffixes _C / _N
removed; L4e -> L4, L26 -> L24 (Pfam Ribosomal_L26 = archaeal uL24), L5e / L18p -> L18 (uL18), L27A -> L15 (Pfam family
of uL15 and L18e), S29e -> S14 (archaeal uS14), L23e -> L14, Ribosom_S12_S23 -> S12. Output-independent.
Writes work3/pair_shares_norm.parquet, run_order_norm.parquet, rearr_norm.json.
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

RD = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111")
PKG = RD / "ws_srvp_bins/package"
FZ = RD / "long_loci/frozen2"
W3 = RD / "ribo_audit/work3"
ALIAS = {"L4e": "L4", "L26": "L24", "L5e": "L18", "L18p": "L18", "L27A": "L15", "S29e": "S14", "L23e": "L14", "S12_S23": "S12"}
CORE = ("L3", "L4", "L23", "L2", "S19", "L22", "S3", "L29", "S17", "L14", "L24", "L5", "S14", "S8", "L6", "L18", "S5", "L15")


def norm(x):
    x = re.sub(r"(?i)^ribosom(al)?_", "", str(x))
    x = re.sub(r"_(C|N)$", "", x)
    return ALIAS.get(x, x)


g = pd.read_parquet(PKG / "universe/genes.parquet", columns=["gid", "genome_code", "contig_code"]).sort_values("gid")
cc, gc = g.contig_code.to_numpy(), g.genome_code.to_numpy()
an = pd.read_parquet(FZ / "annot/genes_annot.parquet", columns=["gid", "cls", "lit"]).sort_values("gid")
mk = pd.read_parquet(PKG / "ribosomal/marked.parquet")
marked = np.zeros(len(g), bool)
marked[mk.gid.to_numpy()] = True
isR = an.cls.to_numpy() == "R"
lit = an.lit.fillna("").to_numpy()
mname = pd.Series(mk.marker.to_numpy(), index=mk.gid.to_numpy())
name = np.array([""] * len(g), dtype=object)
for i in np.flatnonzero(isR):
    name[i] = norm(lit[i] if lit[i] else mname.get(i, "RP"))
gm = pd.read_parquet(PKG / "universe/genomes.parquet").sort_values("genome_code")
dom = gm.domain.to_numpy()
grp = pd.factorize(pd.read_parquet(FZ / "groups/emb_hom_average_t0.75.parquet").sort_values("genome_code").group.to_numpy())[0]
gdom = {}
for c in range(len(dom)):
    gdom[int(grp[c])] = dom[c]
R = np.flatnonzero(isR)
hold, adj = {}, {}
for x in R:
    hold.setdefault(name[x], set()).add(int(grp[gc[x]]))
for d in (1, 2, 3):
    i = R[R + d < len(g)]
    j = i + d
    ok = isR[j] & (cc[i] == cc[j])
    for x, y in zip(i[ok], j[ok]):
        adj.setdefault(tuple(sorted((name[x], name[y]))), set()).add(int(grp[gc[x]]))
runs = pd.read_parquet(PKG / "ribosomal/runs.parquet")
runs = runs[(runs.run_set == "g2") & runs.m16].reset_index(drop=True)
nct = gm.n_contigs.to_numpy()
prow, rrow = [], []
for r in runs.itertuples():
    q = int(grp[r.genome_code])
    d_ = r.domain
    gt = "Archaea" if d_ == "Archaea" else ("Bact_closed" if nct[r.genome_code] == 1 else "Bact_multi")
    m = [k for k in range(r.gid_start, r.gid_end + 1) if marked[k] and isR[k]]
    shs = []
    for x, y in zip(m[:-1], m[1:]):
        a, b = name[x], name[y]
        both = {h for h in (hold.get(a, set()) & hold.get(b, set())) if h != q and gdom[h] == d_}
        sh = len({h for h in adj.get(tuple(sorted((a, b))), set()) if h in both}) / len(both) if both else np.nan
        prow.append(dict(run_id=r.run_id, gtype=gt, a=a, b=b, share=sh, n_groups=len(both), rel_x=x - r.gid_start,
                         rel_y=y - r.gid_start))
        if np.isfinite(sh):
            shs.append(sh)
    core = [name[k] for k in m if name[k] in CORE]
    dd = [c for i_, c in enumerate(core) if i_ == 0 or c != core[i_ - 1]]
    canon = min(tuple(dd), tuple(reversed(dd))) if dd else ()
    rrow.append(dict(run_id=r.run_id, gtype=gt, group=q, n_pairs=len(m) - 1, median_share=np.median(shs) if shs else np.nan,
                     min_share=np.min(shs) if shs else np.nan, n_rare=sum(1 for s in shs if s < 0.10),
                     core_order="-".join(canon), n_core=len(set(core))))
P = pd.DataFrame(prow)
Rr = pd.DataFrame(rrow)
P.to_parquet(W3 / "pair_shares_norm.parquet", index=False)
Rr.to_parquet(W3 / "run_order_norm.parquet", index=False)
out = {}
for gt in ("Bact_closed", "Bact_multi", "Archaea"):
    p = P[P.gtype == gt]
    x = Rr[Rr.gtype == gt]
    out[gt] = dict(runs=int(len(x)), pairs=int(len(p)), pair_share_median=float(np.nanmedian(p.share)),
                   pair_share_q10=float(np.nanquantile(p.share, .1)), pair_share_q25=float(np.nanquantile(p.share, .25)),
                   rare_pairs=int((p.share < 0.10).sum()), runs_with_rare=int((x.n_rare >= 1).sum()),
                   run_min_share_median=float(np.nanmedian(x.min_share)))
for d_, gts in (("Bacteria", ("Bact_closed", "Bact_multi")), ("Archaea", ("Archaea",))):
    x = Rr[Rr.gtype.isin(gts)]
    full = x[x.n_core == 18]
    out["domain_" + d_] = dict(runs=int(len(x)), distinct_orders=int(x.core_order.nunique()),
                   modal_share=float(x.core_order.value_counts().iloc[0] / len(x)), complete_core_runs=int(len(full)),
                   distinct_complete=int(full.core_order.nunique()),
                   top_complete=full.core_order.value_counts().head(3).to_dict(),
                   n_core_distribution={int(k): int(v) for k, v in x.n_core.value_counts().sort_index().items()},
                   groups=int(x.group.nunique()),
                   groups_modal_orders=int(x.groupby("group").core_order.agg(lambda s: s.value_counts().index[0]).nunique()))
json.dump(out, open(W3 / "rearr_norm.json", "w"), indent=1)
print(json.dumps(out, indent=1))
