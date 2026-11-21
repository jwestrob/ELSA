#!/usr/bin/env python
"""Curated complex-subunit pairs vs co-located non-subunit pairs (signal 1, test b). Design table only: no vectors.

Pairs: two member genes of one curated instance at offset 1..3 on the contig. Component = the caller's profile name
(system_proteins.profile_name), normalised to a component symbol by the rules below. Class rules (fixed in
predeclared.json; literature architecture of the named systems, applied only to the caller's own component calls):

  complex (subunits of one stable complex)
    DefenseFinder RM_Type_I   : any two of HsdR / HsdM / HsdS                 (R2M2S1 holoenzyme)
    DefenseFinder RM_Type_III : Mod + Res                                    (Res-Mod2 complex)
    DefenseFinder MazEF       : MazE + MazF                                  (type II toxin-antitoxin complex)
    DefenseFinder Gabija      : GajA + GajB                                  (GajA-GajB supramolecular complex)
    CRISPRCasTyper            : two effector-complex subunits (Cas5, Cas7, Cas8*, Cas10, Cas11*, Csm2-5, Cmr1-6,
                                Csa5) or two adaptation-complex subunits (Cas1, Cas2, Cas4)
  noncomplex (same system, separate complexes or separate enzymes)
    DefenseFinder RM_Type_II  : REase + MTase                                (independent enzymes)
    CRISPRCasTyper            : one adaptation subunit + one effector-complex subunit
  excluded: everything else (Cas3, Cas6, accessory and regulatory Cas genes, other defence types, TXSScan).
Writes $RD/signals/work/curated_pairs.parquet (instance_id, caller, system_type, gid_a, gid_b, comp_a, comp_b, cls,
key, n_distinct) and prints counts by class.
"""
import re
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

RD = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111")
PKG = RD / "ws_dpann/package"
TOK = RD / "engine/dpann_tokens/mmseqs__seq__e3c50s75_rescue3__dpann.parquet"
OUT = RD / "signals/work"

EFFECTOR = re.compile(r"^(cas5|cas7|cas8\w*|cas10|cas11\w*|csm[2-5]|cmr[1-6]|csa5)$")
ADAPT = re.compile(r"^(cas1|cas2|cas4)$")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def comp_of(caller, stype, prof):
    p = prof
    if caller == "CRISPRCasTyper":
        s = re.sub(r"_\d+_.*$", "", p)          # Cas1_6_CAS-I-II... -> Cas1
        s = re.sub(r"gr\d+$", "", s).lower()    # Csm3gr7 -> csm3
        return s
    s = p.split("__")[-1]                       # e.g. RM__Type_I_S, RM_Type_II__Type_II_REase18, MazEF__MazF
    if stype == "RM_Type_I":
        return "HsdS" if "Type_I_S" in s else "HsdM" if "MTases" in s else "HsdR" if "REases" in s else s
    if stype == "RM_Type_III":
        return "Mod" if "MTases" in s else "Res" if "REases" in s else s
    if stype == "RM_Type_II":
        return "MTase" if "MTases" in s else "REase" if "REase" in s else s
    return re.sub(r"_\d+$", "", s)


def cls_of(caller, stype, a, b):
    if caller == "CRISPRCasTyper":
        ea, eb = bool(EFFECTOR.match(a)), bool(EFFECTOR.match(b))
        aa, ab = bool(ADAPT.match(a)), bool(ADAPT.match(b))
        if (ea and eb) or (aa and ab):
            return "complex"
        if (ea and ab) or (aa and eb):
            return "noncomplex"
        return None
    if stype == "RM_Type_I":
        ok = {"HsdR", "HsdM", "HsdS"}
        return "complex" if a in ok and b in ok else None
    if stype == "RM_Type_III":
        return "complex" if {a, b} == {"Mod", "Res"} else None
    if stype == "MazEF":
        return "complex" if {a, b} == {"MazE", "MazF"} else None
    if stype == "Gabija":
        return "complex" if {a, b} == {"GajA", "GajB"} else None
    if stype == "RM_Type_II":
        return "noncomplex" if {a, b} == {"REase", "MTase"} else None
    return None


def main():
    g = pd.read_parquet(PKG / "universe/genes.parquet", columns=["gid", "protein_id", "genome_code", "contig_code", "position_index"])
    tok = pd.read_parquet(TOK).sort_values("gid").token.to_numpy(np.int64)
    T = int(tok.max()) + 1
    inst = pd.read_parquet(PKG / "gold/instances.parquet", columns=["instance_id", "caller", "system_id", "system_type"])
    mem = pd.read_parquet(PKG / "gold/members.parquet")
    con = duckdb.connect(str(RD / "ws_dpann/input/sharur.duckdb"), read_only=True, config={"threads": 2})
    sp = con.execute("SELECT system_id, protein_id, profile_name FROM system_proteins").df()
    con.close()
    pid2gid = pd.Series(g.gid.to_numpy(), index=g.protein_id.to_numpy())
    sp["gid"] = pid2gid.reindex(sp.protein_id).to_numpy()
    sp = sp.dropna(subset=["gid"])
    sp["gid"] = sp.gid.astype(np.int64)
    m = mem.merge(inst, on=["instance_id", "caller"]).merge(sp[["system_id", "gid", "profile_name"]], on=["system_id", "gid"], how="left")
    log("members", len(mem), "with profile", int(m.profile_name.notna().sum()))
    print(m.groupby("caller").profile_name.apply(lambda s: s.isna().mean()))
    m = m.dropna(subset=["profile_name"])
    m["comp"] = [comp_of(c, s, p) for c, s, p in zip(m.caller, m.system_type, m.profile_name)]
    cc = g.contig_code.to_numpy()
    rows = []
    for iid, d in m.groupby("instance_id"):
        d = d.sort_values("gid")
        gids, comps = d.gid.to_numpy(), d.comp.to_numpy()
        caller, stype = d.caller.iat[0], d.system_type.iat[0]
        for x in range(len(d)):
            for y in range(x + 1, len(d)):
                if gids[y] - gids[x] > 3 or cc[gids[y]] != cc[gids[x]]:
                    continue
                c = cls_of(caller, stype, comps[x], comps[y])
                if c is None or tok[gids[x]] == tok[gids[y]]:
                    continue
                rows.append((iid, caller, stype, gids[x], gids[y], comps[x], comps[y], c))
    P = pd.DataFrame(rows, columns=["instance_id", "caller", "system_type", "gid_a", "gid_b", "comp_a", "comp_b", "cls"])
    ta, tb = tok[P.gid_a.to_numpy()], tok[P.gid_b.to_numpy()]
    P["key"] = np.minimum(ta, tb) * T + np.maximum(ta, tb)
    pc = pd.read_parquet(OUT / "pair_counts.parquet", columns=["key", "n_distinct"]).set_index("key").n_distinct
    P["n_distinct"] = pc.reindex(P.key).fillna(0).astype(int).to_numpy()
    P.to_parquet(OUT / "curated_pairs.parquet", index=False)
    P["sys"] = np.where(P.caller == "CRISPRCasTyper", "CRISPR", P.system_type)
    for k in (1, 10, 20, 30):
        s = P[P.n_distinct >= k]
        log(f"n_distinct >= {k}: pairs {len(s)}; token pairs by class/system:")
        print(s.groupby(["cls", "sys"]).key.nunique().to_string())
    s = P.copy()
    s["cp"] = [";".join(sorted(x)) for x in zip(s.comp_a, s.comp_b)]
    print(s.groupby(["cls", "sys"]).cp.value_counts().groupby(level=[0, 1]).head(6).to_string())


if __name__ == "__main__":
    main()
