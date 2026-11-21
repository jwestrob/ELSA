#!/usr/bin/env python
"""Leads pipeline, step 1 (runs on biotite with the elsa env; Aksha search in r1_partners_sbatch.txt between the two
subcommands).

  r1_partners.py prep  --collection C [--shards 5]
      unique supporters that carry the winning label (carries_label true) of every labeled gene of C's pass, with their
      sequences from the collection FASTA -> REMOTE/partners/shard_NN.faa; labeled-gene sequences -> REMOTE/partners/labeled.faa;
      counts -> REMOTE/partners/prep.json (proteins, residues, estimated CPU-hours at 1.1 CPU-s per protein).
  r1_partners.py merge --collection C
      Aksha KOFAM --cut_ga hits of the shards -> KOFAM_partners.parquet (config key kofam_partners; gid, ko, bitscore, evalue).
      Partner function per labeled gene exactly as the DPANN gap screen: carriers = label-carrying supporters; each carrier
      counts each of its KOs once; ko_top = plurality KO (ties -> lowest KO); ko_share = ko_top_n / n_carriers_ko;
      STRONG = ko_top_n >= 3 and ko_share >= 0.70. EC from the local KEGG snapshot (gapscreen/data/kegg_ko.tsv copy);
      enzyme = ko_top has an EC. PFSET per KO = Pfams (collection Pfam table, GA) carried by >= 30% of the KO's carriers
      (>= 3 carriers). Population (step 2): tiers T2 / T2b / rest (T1c and contradicted kept as side populations), STRONG
      enzyme function, 100-600 aa, ILVFMAWC < 0.45 -> REMOTE/novelty/candidates.faa.
      Writes REMOTE/data/genes_function.parquet, REMOTE/data/pfset.tsv, REMOTE/data/funnel_r1.json.
"""
import argparse, glob, json, os
from collections import Counter
import numpy as np
import pandas as pd

RD = "/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111"
HERE = os.path.dirname(os.path.abspath(__file__))
SOL = set("ILVFMAWC")


def cfg_of(c):
    cfg = json.load(open(os.path.join(HERE, "collections_leads.json")))[c]
    return {k: (v.replace("RD/", RD + "/") if isinstance(v, str) else v) for k, v in cfg.items()}


def read_fasta_subset(path, want):
    out, keep, name, buf = {}, False, None, []
    with open(path) as fh:
        for line in fh:
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


def load_pass(cfg, c):
    P = pd.read_parquet(f"{cfg['pass_dir']}/{c}_slot_rescue_pass.parquet")
    S = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{cfg['pass_dir']}/supporters/part_*.parquet"))], ignore_index=True)
    return P, S


SCOPES = [["T2", "T2b", "rest"], ["T2", "T2b", "rest", "T1c"], ["T2", "T2b", "rest", "T1c", "contradicted"]]


def prep(c, nshards, budget_h=60.0):
    """srvp adaptation: the unique label carriers of ALL labeled genes exceed the 60 CPU-hour KOfam ceiling, so the run is
    restricted to the label carriers of the genes that are assayed or reported (tiers T2, T2b, rest; T1c and contradicted
    added in that order while the estimate stays under budget_h at 1.1 CPU-s per protein). Counts of every scope go to
    prep.json; the chosen scope is recorded there and read back by merge."""
    cfg = cfg_of(c)
    out = f"{cfg['remote']}/partners"
    os.makedirs(out, exist_ok=True)
    P, S = load_pass(cfg, c)
    T = pd.read_parquet(f"{cfg['pass_dir']}/tiers.parquet")[["protein_id", "tier"]]
    P = P.merge(T, on="protein_id", how="left")
    S = S[S.gid.isin(set(P.gid))].merge(P[["gid", "tier"]], on="gid")
    scope_counts = []
    for sc in SCOPES + [sorted(set(P.tier.dropna()))]:
        ss = S[S.tier.isin(sc)]
        n = len({int(s) for sup, cl in zip(ss.supporters, ss.carries_label) for s, k in zip(sup, cl) if k})
        scope_counts.append(dict(tiers=sc, labeled_genes=int(P.tier.isin(sc).sum()), unique_label_carriers=n,
                                 est_cpu_hours=round(n * 1.1 / 3600, 1)))
    chosen = [d for d in scope_counts[:3] if d["est_cpu_hours"] < budget_h][-1]["tiers"]
    S = S[S.tier.isin(chosen)]
    P = P[P.tier.isin(chosen)]
    carriers = sorted({int(s) for sup, cl in zip(S.supporters, S.carries_label) for s, k in zip(sup, cl) if k})
    seq = read_fasta_subset(cfg["fasta"], set(carriers) | set(P.gid.astype(int)))
    ids = [str(g) for g in carriers]
    chunks = np.array_split(np.array(ids), nshards)
    for i, ch in enumerate(chunks):
        with open(f"{out}/shard_{i:02d}.faa", "w") as fh:
            for g in ch:
                fh.write(f">{g}\n{seq[g]}\n")
    with open(f"{out}/labeled.faa", "w") as fh:
        for g in P.gid.astype(int):
            fh.write(f">{g}\n{seq[str(g)]}\n")
    res = sum(len(seq[g]) for g in ids)
    js = dict(collection=c, kofam_scope_tiers=chosen, scope_counts=scope_counts, labeled_genes=len(P),
              genes_with_supporters=len(S), unique_label_carrying_supporters=len(ids),
              residues=res, shards=nshards, est_cpu_hours_at_1p1s=round(len(ids) * 1.1 / 3600, 1))
    json.dump(js, open(f"{out}/prep.json", "w"), indent=1)
    print(json.dumps(js))


def merge(c):
    cfg = cfg_of(c)
    R = cfg["remote"]
    os.makedirs(f"{R}/data", exist_ok=True)
    os.makedirs(f"{R}/novelty", exist_ok=True)
    hits = []
    for d in sorted(x for x in glob.glob(f"{R}/partners/out_*") if os.path.isdir(x)):  # directories only (coordinator fix: out_NN.done files also match)
        f = f"{d}/KOFAM_hits_df.tsv"
        if not os.path.exists(f):
            raise SystemExit(f"missing {f}")
        h = pd.read_csv(f, sep="\t", dtype={"sequence_id": str})
        hits.append(h)
    H = pd.concat(hits, ignore_index=True)
    H["gid"] = H.sequence_id.astype(np.int64)
    H["ko"] = H.hmm_name.astype(str).str.extract(r"(K\d{5})")[0]
    K = H.dropna(subset=["ko"]).sort_values("bitscore", ascending=False).drop_duplicates(["gid", "ko"])
    K = K[["gid", "ko", "bitscore", "evalue"]].sort_values(["gid", "ko"]).reset_index(drop=True)
    K.to_parquet(cfg["kofam_partners"], index=False)
    kos_of = K.groupby("gid").ko.apply(lambda s: sorted(set(s))).to_dict()
    P, S = load_pass(cfg, c)
    S = S.set_index("gid")
    kegg = pd.read_csv(f"{R}/data/kegg_ko.tsv", sep="\t", keep_default_na=False).set_index("ko")
    rows = []
    for r in P.itertuples(index=False):
        g = int(r.gid)
        if g in S.index:
            sup, cl = S.loc[g, "supporters"], S.loc[g, "carries_label"]
            carriers = [int(s) for s, k in zip(sup, cl) if k]
        else:
            carriers = []
        cnt, first, nko = Counter(), {}, 0
        for p in carriers:
            ks = kos_of.get(p, [])
            nko += bool(ks)
            for k in ks:
                cnt[k] += 1
                first.setdefault(k, p)
        top = sorted(cnt.items(), key=lambda t: (-t[1], t[0]))
        rows.append(dict(gid=g, n_carriers2=len(carriers), n_carriers_ko2=nko, ko_top=top[0][0] if top else "",
                         ko_top_n=top[0][1] if top else 0, ko_second=top[1][0] if len(top) > 1 else "",
                         ko_second_n=top[1][1] if len(top) > 1 else 0, ko_counts=";".join(f"{k}:{n}" for k, n in top[:6]),
                         rep_partner_gid=first[top[0][0]] if top else -1))
    F = P.merge(pd.DataFrame(rows), on="gid")
    F["ko_share"] = np.where(F.n_carriers_ko2 > 0, F.ko_top_n / F.n_carriers_ko2.clip(lower=1), np.nan)
    F["strong_partner"] = (F.ko_top_n >= 3) & (F.ko_share >= 0.70)
    F["func"] = np.where(F.strong_partner, F.ko_top, "")
    F["func_source"] = np.where(F.strong_partner, "partners", "")
    F["ec"] = F.func.map(lambda k: kegg.ec.get(k, "") if k else "")
    F["enzyme"] = F.ec.str.len() > 0
    T = pd.read_parquet(f"{cfg['pass_dir']}/tiers.parquet")[["protein_id", "tier", "tier_family"]]
    F = F.merge(T, on="protein_id", how="left")
    scope = json.load(open(f"{R}/partners/prep.json"))["kofam_scope_tiers"]   # srvp: KOfam ran on these tiers' carriers only
    F = F[F.tier.isin(scope)].copy()
    seq = {}
    name = None
    for line in open(f"{R}/partners/labeled.faa"):
        if line.startswith(">"):
            name = line[1:].strip()
        else:
            seq[name] = line.strip()
    F["seq_len"] = F.gid.astype(str).map(lambda g: len(seq.get(g, "")))
    F["ilvfmawc"] = F.gid.astype(str).map(lambda g: round(sum(ch in SOL for ch in seq.get(g, "")) / max(1, len(seq.get(g, ""))), 3))
    G = pd.read_parquet(cfg["genomes"])
    ocol = "ord" if "ord" in G.columns else "order"
    gmap = G.set_index("genome")
    F["order"] = F.genome.map(gmap[ocol]) if ocol in gmap.columns else ""
    F["phylum"] = F.genome.map(gmap["phylum"]) if "phylum" in gmap.columns else ""
    F["completeness"] = F.genome.map(gmap["completeness"]) if "completeness" in gmap.columns else np.nan
    F.to_parquet(f"{R}/data/genes_function.parquet", index=False)
    # PFSET per KO from the collection Pfam table
    pf = pd.read_parquet(cfg["pfam"], columns=["gid", "accession", "name"])
    pf["pf"] = pf.accession.str.split(".").str[0]
    kp = K[["gid", "ko"]].merge(pf[["gid", "pf", "name"]].drop_duplicates(["gid", "pf"]), on="gid")
    ncar = K.groupby("ko").gid.nunique()
    ps = kp.groupby(["ko", "pf"]).agg(n=("gid", "nunique"), pfname=("name", "first")).reset_index()
    ps["n_carriers"] = ps.ko.map(ncar)
    ps = ps[(ps.n_carriers >= 3) & (ps.n >= 0.3 * ps.n_carriers)]
    ps.to_csv(f"{R}/data/pfset.tsv", sep="\t", index=False)
    # funnel and population
    tiers = ["T2", "T2b", "rest", "T1c", "contradicted", "T1"]
    cnt = lambda d: {t: int((d.tier == t).sum()) for t in tiers}
    fun = dict(collection=c, labeled=cnt(F), kofam_rows=len(K), kofam_proteins=int(K.gid.nunique()),
               with_carrier_ko=cnt(F[F.n_carriers_ko2 > 0]), strong_partner=cnt(F[F.strong_partner]),
               strong_enzyme=cnt(F[F.strong_partner & F.enzyme]))
    pop = F[F.strong_partner & F.enzyme & F.seq_len.between(100, 600) & (F.ilvfmawc < 0.45)]
    fun["len100_600_soluble"] = cnt(pop)
    pop = pop[pop.tier.isin(["T2", "T2b", "rest", "T1c", "contradicted"])]
    with open(f"{R}/novelty/candidates.faa", "w") as fh:
        for g in pop.gid.astype(str):
            fh.write(f">{g}\n{seq[g]}\n")
    fun["novelty_input_proteins"] = len(pop)
    json.dump(fun, open(f"{R}/data/funnel_r1.json", "w"), indent=1)
    print(json.dumps(fun, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prep", "merge"])
    ap.add_argument("--collection", required=True)
    ap.add_argument("--shards", type=int, default=5)
    ap.add_argument("--budget-h", dest="budget_h", type=float, default=60.0)
    a = ap.parse_args()
    prep(a.collection, a.shards, a.budget_h) if a.cmd == "prep" else merge(a.collection)
