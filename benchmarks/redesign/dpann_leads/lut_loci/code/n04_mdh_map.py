#!/usr/bin/env python
"""lut_loci n04 (biotite login node): map the slice_lut part-1 methanol-dehydrogenase KOfam hits to package_k8 gids.

Part-1 hits (results/<scope>/mdh_hits.parquet; hmmsearch with ko_list thresholds over every protein of each slice, run by the
slice_lut workstream: K23995 lanthanide-dependent MDH xoxF, K00114 PQQ alcohol dehydrogenase, K14028 mxaF, K14029 mxaI) are
keyed by part-1 gids. Join part-1 gid -> (genome, protein_id) -> package gid. Writes data/mdh_pkg.parquet
(gid, ko, score, threshold, scope) and data/mdh_map.json (counts, unmapped).
"""
import json

import pandas as pd

RD = "/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111"
UNI = {"order": RD + "/slice_lut/package/rhizobiales/universe/genes.parquet",
       "alpha_wide": RD + "/slice_lut/work/alpha_wide/pkg/universe/genes.parquet",
       "beyond": RD + "/slice_lut/work/beyond/pkg/universe/genes.parquet"}
pk = pd.read_parquet(RD + "/slice_lut/testset/package_k8/universe/genes.parquet", columns=["gid", "genome", "protein_id"])
pk_key = pk.set_index(["genome", "protein_id"]).gid
gen = set(pk.genome)
out, rep = [], {}
for s, f in UNI.items():
    m = pd.read_parquet(RD + f"/slice_lut/results/{s}/mdh_hits.parquet")
    u = pd.read_parquet(f, columns=["gid", "genome", "protein_id"]).set_index("gid")
    m["genome"] = u.genome.reindex(m.gid).to_numpy()
    m["protein_id"] = u.protein_id.reindex(m.gid).to_numpy()
    m = m[m.genome.isin(gen)].copy()
    key = pd.MultiIndex.from_arrays([m.genome, m.protein_id])
    m["pkg_gid"] = pk_key.reindex(key).to_numpy()
    rep[s] = dict(hits_in_package_genomes=len(m), unmapped=int(m.pkg_gid.isna().sum()))
    m = m.dropna(subset=["pkg_gid"])
    m["scope"] = s
    out.append(m[["pkg_gid", "ko", "score", "threshold", "scope"]].rename(columns={"pkg_gid": "gid"}))
M = pd.concat(out).astype({"gid": "int64"}).drop_duplicates(["gid", "ko"])
M.to_parquet("data/mdh_pkg.parquet")
rep["by_ko"] = M.ko.value_counts().to_dict()
rep["genomes_by_ko"] = M.merge(pk[["gid", "genome"]], on="gid").groupby("ko").genome.nunique().to_dict()
json.dump(rep, open("data/mdh_map.json", "w"), indent=1)
print(json.dumps(rep, indent=1))
