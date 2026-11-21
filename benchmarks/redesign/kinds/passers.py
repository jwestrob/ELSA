#!/usr/bin/env python
"""Artefact checks and observations for every co-fold passer (classes a and b), as in screens 1-3.

1. Split gene: MMseqs2 18 (-s 7.5, E <= 1e-3, --max-seqs 10000) of both chains against all 3,585,303 DPANN
   proteins. Fused homolog = a target other than the two genes hit by both chains, each hit >= 50% query
   coverage, segments overlapping by <= 10% of the shorter one. Artefact if fused homologs occur in >= 25% as many
   genomes as carry the pair's adjacent MMseqs2 family pair (engine G at window 3, at least 1), or in the folded
   genome's genus.
2. Observed domains: Pfam on the support pairs of the pair (depth 1000, observed order; side assigned by list
   membership) and on the 10 nearest Pfam-annotated genes of each chain; curated calls (gold members) among
   support-pair genes. Observations only.
3. Critic score (held-out order) and its rank among all adjacencies; architecture (arch/H_pairs.parquet).
Writes $RD/kinds/passers/{passers.tsv, support_domains.tsv, mmseqs_hits.tsv}.
"""
import argparse
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kinds as KD  # noqa: E402

RD = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111")
PKG = RD / "ws_dpann/package"
K = RD / "kinds"
OUT = K / "passers"
MM = "/groups/banfield/users/jwestrob/bin/mmseqs_18/mmseqs/bin/mmseqs"
FAA = RD / "ws_dpann/work/proteins_gid.faa"


def log(*a):
    KD.log(*a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, required=True)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    f = pd.read_csv(K / "cofold/folds_kinds.tsv", sep="\t")
    p = f[f.pass_c123 & f.cls.isin(["a_high", "b_low"])].copy()
    jobs = {}
    import json
    for l in open(K / "cofold/jobs.jsonl"):
        j = json.loads(l)
        jobs[j["fold_id"]] = j
    with open(OUT / "queries.faa", "w") as fh:
        for r in p.itertuples():
            for ch, gid in zip(jobs[r.fold_id]["chains"], (r.gid_A, r.gid_B)):
                fh.write(f">{gid}\n{ch['seq']}\n")
    # ---------------------------------------------------------------- MMseqs2
    hits = OUT / "mmseqs_hits.tsv"
    if not hits.exists():
        tmp = OUT / "mm"
        tmp.mkdir(exist_ok=True)
        T = str(a.threads)
        for cmd in ([MM, "createdb", str(OUT / "queries.faa"), str(tmp / "qdb")],
                    [MM, "createdb", str(FAA), str(tmp / "tdb"), "--createdb-mode", "1"],
                    [MM, "search", str(tmp / "qdb"), str(tmp / "tdb"), str(tmp / "aln"), str(tmp / "t"), "-s", "7.5",
                     "-e", "1e-3", "--max-seqs", "10000", "--threads", T],
                    [MM, "convertalis", str(tmp / "qdb"), str(tmp / "tdb"), str(tmp / "aln"), str(hits),
                     "--format-output", "query,target,evalue,bits,pident,qstart,qend,qlen,tstart,tend,tlen",
                     "--threads", T]):
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    h = pd.read_csv(hits, sep="\t", header=None, names=["q", "t", "evalue", "bits", "pident", "qs", "qe", "ql", "ts",
                                                         "te", "tl"])
    h = h[(h.evalue <= 1e-3) & ((h.qe - h.qs + 1) / h.ql >= 0.5)]
    gc = np.load(K / "prep/gc.npy")
    gm = pd.read_parquet(PKG / "universe/genomes.parquet").sort_values("genome_code").reset_index(drop=True)
    genus = gm.genus.to_numpy()
    sc = pd.read_parquet(K / "main/scores.parquet", columns=["i", "eng_G", "eng_L"])
    rowlut = pd.Series(np.arange(len(sc)), index=sc.i.to_numpy())
    z = np.load(RD / "grammar/eval/assembled_cv.npz")
    crit = z["primary"]
    crit_rank = pd.Series(crit).rank(pct=True).to_numpy()
    rows = []
    for r in p.itertuples():
        ha = h[h.q == r.gid_A]
        hb = h[h.q == r.gid_B]
        m = ha.merge(hb, on="t", suffixes=("_a", "_b"))
        m = m[~m.t.isin([r.gid_A, r.gid_B])]
        lo = np.maximum(m.ts_a, m.ts_b)
        hi = np.minimum(m.te_a, m.te_b)
        ov = np.maximum(hi - lo + 1, 0)
        short = np.minimum(m.te_a - m.ts_a + 1, m.te_b - m.ts_b + 1)
        fused = m[ov <= 0.1 * short]
        fg = set(gc[fused.t.to_numpy()]) if len(fused) else set()
        rw = int(rowlut[min(r.gid_A, r.gid_B)])
        G = max(int(sc.eng_G.iloc[rw]), 1)
        g0 = gc[r.gid_A]
        same_genus = any(genus[x] == genus[g0] for x in fg) if isinstance(genus[g0], str) else False
        rows.append(dict(fold_id=r.fold_id, gid_A=r.gid_A, gid_B=r.gid_B, homologs_A=int(ha.t.nunique()),
                         homologs_B=int(hb.t.nunique()), fused_targets=int(fused.t.nunique()),
                         fused_genomes=len(fg), pair_genomes_G=G, fused_same_genus=same_genus,
                         split_gene_artefact=bool(len(fg) >= 0.25 * G or same_genus),
                         critic=float(crit[rw]), critic_pct=float(crit_rank[rw]), engine_L=int(sc.eng_L.iloc[rw])))
    S = pd.DataFrame(rows)
    # ---------------------------------------------------------------- domains on support pairs and neighbours
    d = KD.Data()
    genes = pd.read_parquet(PKG / "universe/genes.parquet", columns=["gid", "protein_id"])
    import duckdb
    con = duckdb.connect(str(RD / "ws_dpann/input/sharur.duckdb"), read_only=True)
    con.execute(f"SET threads={a.threads}")
    pf = con.execute("SELECT protein_id, name FROM annotations WHERE source='pfam'").fetchdf().merge(genes, on="protein_id")
    dom = pf.groupby("gid").name.apply(lambda s: sorted(set(s))).to_dict()
    mem = pd.read_parquet(PKG / "gold/members.parquet").merge(
        pd.read_parquet(PKG / "gold/instances.parquet", columns=["instance_id", "caller", "system_type"]),
        on=["instance_id", "caller"])
    gold = (mem.caller + ":" + mem.system_type.astype(str)).groupby(mem.gid).apply(lambda s: sorted(set(s))).to_dict()
    ptr, left = KD.support_lists(p.gid_A.to_numpy(np.int64), p.gid_B.to_numpy(np.int64), d.uq, d.I, 1000, d.mptr,
                                 d.members, d.gc, d.cid, d.U, d.n, d.bufcap)
    drow = []
    for k, r in enumerate(p.itertuples()):
        la = set(d.I[d.uq[r.gid_A], :1000].tolist()) | {int(d.uq[r.gid_A])}
        sa, sb = [], []
        for l in left[ptr[k]:ptr[k + 1]]:
            x, y = (l, l + 1) if d.uq[l] in la else (l + 1, l)
            sa.append(x)
            sb.append(y)
        def summ(gl):
            c, g = Counter(), Counter()
            for x in gl:
                for t in dom.get(x, ()):
                    c[t] += 1
                for t in gold.get(x, ()):
                    g[t] += 1
            n = len(gl)
            return (n, sum(1 for x in gl if x in dom), "; ".join(f"{t} {v}" for t, v in c.most_common(5)),
                    "; ".join(f"{t} {v}" for t, v in g.most_common(3)))
        def nn_dom(gid):
            c, seen = Counter(), 0
            for U in d.I[d.uq[gid], :1000]:
                for t in range(d.mptr[U], d.mptr[U + 1]):
                    gg = d.members[t]
                    if gg in dom:
                        for x in dom[gg]:
                            c[x] += 1
                        seen += 1
                        break
                if seen >= 10:
                    break
            return "; ".join(f"{t} {v}/10" for t, v in c.most_common(4))
        na, pa_, da, ga = summ(sa)
        nb, pb_, db, gb = summ(sb)
        drow.append(dict(fold_id=r.fold_id, support_pairs=na, A_with_pfam=pa_, A_support_domains=da, A_curated=ga,
                         B_with_pfam=pb_, B_support_domains=db, B_curated=gb, A_nn10_domains=nn_dom(r.gid_A),
                         B_nn10_domains=nn_dom(r.gid_B)))
    D = pd.DataFrame(drow)
    arch = pd.read_parquet(K / "arch/H_pairs.parquet", columns=["i", "arch"]) if (K / "arch/H_pairs.parquet").exists() else None
    out = p[["fold_id", "cls", "gid_A", "gid_B", "protein_A", "protein_B", "genome", "len_A", "len_B", "iptm", "iface_pae",
             "contacts", "interface_plddt", "kappa", "R", "singleton", "tokC_A", "tokC_B", "hyd_A", "hyd_B"]] \
        .merge(S, on=["fold_id", "gid_A", "gid_B"]).merge(D, on="fold_id")
    if arch is not None:
        out["arch"] = out[["gid_A", "gid_B"]].min(1).map(pd.Series(arch.arch.to_numpy(), index=arch.i.to_numpy()))
    out.to_csv(OUT / "passers.tsv", sep="\t", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 80)
    log(out.drop(columns=["protein_A", "protein_B"]).to_string(index=False))


if __name__ == "__main__":
    main()
