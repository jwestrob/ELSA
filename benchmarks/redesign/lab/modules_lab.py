#!/usr/bin/env python
"""Module definitions for the recurrence engine, scored against curated system types.

A module is a set of tokens that recur together. Candidates are built from one engine run's
significant_pairs.parquet (token graph, edge = significant pair):
  uf        the engine's current definition (union-find over pairs with J >= 0.5; token_modules.parquet)
  mod<r>    Leiden, modularity objective, resolution r, edge weight J
  cpm<r>    Leiden, CPM objective, resolution r, edge weight J
Scoring needs no locus caller: each curated instance with >= 2 member genes gets the plurality
module of its members' tokens. Reported per caller: share of instances with a module, number of
modules used, adjusted mutual information with the caller's system type, homogeneity (a module
holds one type) and completeness (a type sits in one module). Chance level of AMI is 0.

usage: modules_lab.py --run <engine result dir> --tokens <gid,token parquet> --pkg <ws_dpann package>
"""
import argparse
import random
from pathlib import Path

import igraph as ig
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_mutual_info_score, homogeneity_completeness_v_measure


def plurality(mods):
    m = mods[mods >= 0]
    if len(m) == 0:
        return -1
    v, c = np.unique(m, return_counts=True)
    return int(v[c.argmax()])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--tokens", type=Path, required=True)
    ap.add_argument("--pkg", type=Path, required=True)
    ap.add_argument("--mod", default="1,5,20,100,500")
    ap.add_argument("--cpm", default="0.02,0.05,0.1,0.2,0.4")
    ap.add_argument("--weight", default="J", choices=("J", "one", "logL", "JlogL"))
    ap.add_argument("--louvain", default="", help="networkx Louvain resolutions")
    ap.add_argument("--save", default="", help="definition to write as token_modules__<name>.parquet in --run")
    a = ap.parse_args()
    sp = pd.read_parquet(a.run / "significant_pairs.parquet", columns=["lo", "hi", "J", "L"])
    tok = pd.read_parquet(a.tokens).sort_values("gid").token.to_numpy()
    T = int(tok.max()) + 1
    nodes, inv = np.unique(np.r_[sp.lo.to_numpy(), sp.hi.to_numpy()], return_inverse=True)
    es, ed = inv[:len(sp)], inv[len(sp):]
    J, L = sp.J.to_numpy(float), sp.L.to_numpy(float)
    w = {"J": J, "one": np.ones(len(sp)), "logL": np.log2(L), "JlogL": J * np.log2(L)}[a.weight]
    G = ig.Graph(n=len(nodes), edges=np.column_stack([es, ed]).tolist(), directed=False)
    G.es["weight"] = w.tolist()
    defs = {}
    uf = pd.read_parquet(a.run / "token_modules.parquet")
    m = np.full(T, -1, dtype=np.int64)
    m[uf.token.to_numpy()] = uf.module.to_numpy()
    defs["uf (engine, J>=0.5)"] = m
    for kind, obj, grid in (("mod", "modularity", a.mod), ("cpm", "CPM", a.cpm)):
        for r in [float(x) for x in grid.split(",") if x]:
            random.seed(17)
            ig.set_random_number_generator(random)
            memb = np.asarray(G.community_leiden(objective_function=obj, weights="weight", resolution=r, n_iterations=-1).membership)
            size = np.bincount(memb)
            memb = np.where(size[memb] >= 2, memb, -1)          # a module needs two tokens
            m = np.full(T, -1, dtype=np.int64)
            m[nodes] = memb
            defs[f"{kind}{r:g}"] = m
    if a.louvain:
        import networkx as nx
        H = nx.Graph()
        H.add_weighted_edges_from(zip(es.tolist(), ed.tolist(), w.tolist()))
        for r in [float(x) for x in a.louvain.split(",") if x]:
            comms = nx.community.louvain_communities(H, weight="weight", resolution=r, seed=17)
            m = np.full(T, -1, dtype=np.int64)
            for i, c in enumerate(comms):
                if len(c) >= 2:
                    m[nodes[np.fromiter(c, dtype=np.int64)]] = i
            defs[f"louvain{r:g}"] = m
    members = pd.read_parquet(a.pkg / "gold/members.parquet")
    inst = pd.read_parquet(a.pkg / "gold/instances.parquet", columns=["instance_id", "caller", "system_type", "genome"]).set_index("instance_id")
    gcode = dict(pd.read_parquet(a.pkg / "universe/genomes.parquet", columns=["genome", "genome_code"]).itertuples(index=False))
    grp = members.groupby("instance_id").gid.apply(lambda s: s.to_numpy())
    grp = grp[grp.apply(len) >= 2]
    callers = inst.loc[grp.index, "caller"].to_numpy()
    types = inst.loc[grp.index, "system_type"].astype(str).to_numpy()
    odd = inst.loc[grp.index, "genome"].map(gcode).to_numpy() % 2 == 1
    print(f"run {a.run.name}: {len(sp):,} significant pairs over {len(nodes):,} tokens; instances with >= 2 members: {len(grp):,}; weight {a.weight}")
    print(f"{'definition':<22}{'modules':>8}{'largest':>9}{'tokens in':>10} | " + " | ".join(f"{c[:13]:<13} cov   n   AMI  hom  comp" for c in ("DefenseFinder", "TXSScan", "CRISPRCasTyper")))
    for name, m in defs.items():
        used = m[m >= 0]
        sizes = np.bincount(np.unique(used, return_inverse=True)[1]) if len(used) else np.zeros(1, int)
        pm = np.array([plurality(m[tok[g]]) for g in grp.to_numpy()])
        cells = []
        for c in ("DefenseFinder", "TXSScan", "CRISPRCasTyper"):
            k = callers == c
            hit = k & (pm >= 0)
            if hit.sum() < 2:
                cells.append(f"{'':<13} {0:4.2f} {0:3d} {0:5.2f} {0:4.2f} {0:4.2f}")
                continue
            ami = adjusted_mutual_info_score(types[hit], pm[hit])
            h, cpl, _ = homogeneity_completeness_v_measure(types[hit], pm[hit])
            cells.append(f"{'':<13} {hit.sum() / k.sum():4.2f} {len(np.unique(pm[hit])):3d} {ami:5.2f} {h:4.2f} {cpl:4.2f}")
        df = callers == "DefenseFinder"
        split = [adjusted_mutual_info_score(types[df & (pm >= 0) & h], pm[df & (pm >= 0) & h]) for h in (odd, ~odd)]
        print(f"{name:<22}{len(sizes):>8,}{int(sizes.max()):>9,}{len(used):>10,} | " + " | ".join(cells)
              + f" | DefenseFinder AMI tune {split[0]:.3f} test {split[1]:.3f}")
        if a.save and name == a.save:
            pd.DataFrame({"token": np.flatnonzero(m >= 0), "module": m[m >= 0]}).to_parquet(a.run / f"token_modules__{name}.parquet", index=False)


if __name__ == "__main__":
    main()
