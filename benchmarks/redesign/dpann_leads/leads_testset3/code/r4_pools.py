#!/usr/bin/env python
"""Leads pipeline step 4b helpers (biotite, login node, single-threaded).
  r4_pools.py candidates --collection C
      sequences of the population genes (REMOTE/novelty/candidates.faa already holds them) -> nothing to do; kept for symmetry.
  r4_pools.py positives --collection C --families REMOTE/data/families_pfam.tsv
      positive-control pool per family (assay_rule.json): proteins of the collection with the family KO in the partner
      KOfam table (Aksha --cut_ga) AND a Pfam row (GA) for one of the family Pfams, 100-600 aa, ILVFMAWC < 0.45, not in the
      labeled table. families_pfam.tsv columns: func, pfams (comma list). -> REMOTE/data/positives_pool.tsv (func, gid,
      length, order, genome) and REMOTE/data/positives_pool.faa
  r4_pools.py negpool --collection C [--n 4000]
      negative-B pool: random proteins (numpy default_rng(17)) with no Pfam row, absent from the labeled table, 100-600 aa,
      ILVFMAWC < 0.45 -> REMOTE/data/negpool.tsv (gid, length, genome) and REMOTE/data/negpool.faa
  r4_pools.py fetch --collection C --ids FILE --out FAA     sequences by gid (one per line)."""
import argparse, json, os
import numpy as np
import pandas as pd

RD = "/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111"
HERE = os.path.dirname(os.path.abspath(__file__))
SOL = set("ILVFMAWC")


def cfg_of(c):
    cfg = json.load(open(os.path.join(HERE, "collections_leads.json")))[c]
    return {k: (v.replace("RD/", RD + "/") if isinstance(v, str) else v) for k, v in cfg.items()}


def fasta_subset(path, want):
    out, keep, name, buf = {}, False, None, []
    for line in open(path):
        if line.startswith(">"):
            if keep:
                out[name] = "".join(buf).rstrip("*")
            name = line[1:].split()[0]
            keep = int(name) in want
            buf = []
        elif keep:
            buf.append(line.strip())
    if keep:
        out[name] = "".join(buf).rstrip("*")
    return out


comp = lambda s: sum(ch in SOL for ch in s) / max(1, len(s))
ap = argparse.ArgumentParser()
ap.add_argument("cmd", choices=["positives", "negpool", "fetch"])
ap.add_argument("--collection", required=True)
ap.add_argument("--families")
ap.add_argument("--n", type=int, default=4000)
ap.add_argument("--ids"); ap.add_argument("--out")
a = ap.parse_args()
cfg = cfg_of(a.collection)
R = cfg["remote"]
os.makedirs(f"{R}/data", exist_ok=True)
lab = set(pd.read_parquet(f"{cfg['pass_dir']}/{a.collection}_slot_rescue_pass.parquet", columns=["gid"]).gid.astype(int))
genes = pd.read_parquet(cfg["genes"], columns=["gid", "genome", "length"])
G = pd.read_parquet(cfg["genomes"])
ocol = "ord" if "ord" in G.columns else "order"
order_of = G.set_index("genome")[ocol].to_dict() if ocol in G.columns else {}
if a.cmd == "positives":
    fam = pd.read_csv(a.families, sep="\t")
    K = pd.read_parquet(cfg["kofam_partners"])
    pf = pd.read_parquet(cfg["pfam"], columns=["gid", "accession"])
    pf["pf"] = pf.accession.str.split(".").str[0]
    rows = []
    for r in fam.itertuples():
        pfs = set(str(r.pfams).split(","))
        g1 = set(K[K.ko == r.func].gid.astype(int))
        g2 = set(pf[pf.pf.isin(pfs)].gid.astype(int))
        for g in sorted((g1 & g2) - lab):
            rows.append(dict(func=r.func, gid=g))
    P = pd.DataFrame(rows, columns=["func", "gid"])
    seq = fasta_subset(cfg["fasta"], set(P.gid))
    P["length"] = P.gid.astype(str).map(lambda g: len(seq.get(g, "")))
    P["ilv"] = P.gid.astype(str).map(lambda g: comp(seq.get(g, "")))
    P = P[P.length.between(100, 600) & (P.ilv < 0.45)]
    P["genome"] = P.gid.map(genes.set_index("gid").genome)
    P["order"] = P.genome.map(order_of).fillna("NA")
    P.to_csv(f"{R}/data/positives_pool.tsv", sep="\t", index=False)
    with open(f"{R}/data/positives_pool.faa", "w") as fh:
        for g in sorted(set(P.gid)):
            fh.write(f">{g}\n{seq[str(g)]}\n")
    print(P.groupby("func").size().to_string())
elif a.cmd == "negpool":
    pfg = set(pd.read_parquet(cfg["pfam"], columns=["gid"]).gid.astype(int))
    elig = genes[(genes.length.between(100, 601)) & ~genes.gid.isin(pfg) & ~genes.gid.isin(lab)]
    rng = np.random.default_rng(17)
    pick = elig.iloc[rng.permutation(len(elig))[: a.n * 2]]
    seq = fasta_subset(cfg["fasta"], set(pick.gid.astype(int)))
    pick = pick.assign(sequence=pick.gid.astype(str).map(seq))
    pick = pick[pick.sequence.notna()]
    pick["length"] = pick.sequence.str.len()
    pick = pick[pick.length.between(100, 600) & (pick.sequence.map(comp) < 0.45)].head(a.n)
    pick[["gid", "length", "genome"]].to_csv(f"{R}/data/negpool.tsv", sep="\t", index=False)
    with open(f"{R}/data/negpool.faa", "w") as fh:
        for r in pick.itertuples():
            fh.write(f">{r.gid}\n{r.sequence}\n")
    print("eligible", len(elig), "pool", len(pick))
else:
    ids = {int(x) for x in open(a.ids).read().split()}
    seq = fasta_subset(cfg["fasta"], ids)
    with open(a.out, "w") as fh:
        for g in sorted(ids):
            fh.write(f">{g}\n{seq[str(g)]}\n")
    print(len(seq), "sequences")
