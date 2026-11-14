"""Candidate 2: curated system calls near the DUF7287/DUF7288 cassette, and overlap with earlier leads (local).

Curated calls: the gold set of the DPANN package (data/instances.parquet, data/members.parquet; TXSScan
`secretion_systems`, DefenseFinder `defense_systems`, CRISPRCasTyper), i.e. system_proteins joined to the curated
system tables; raw profile rows are not used. For every cassette instance (screen-2 block types of ranks 34, 33,
11 and 7): whether any cassette gene is a member of a call, whether the cassette lies inside a call's gene span on
the same contig, and the distance in genes from the cassette to the nearest member of a call on the same contig
(0 = adjacent gene). Same statistic for the length-matched control cores of dd_prep.py on the same contigs.
Earlier leads: cluster:1043 and cluster:3913 loci of run elsa-04bdd1970b90463e955d (source_loci.tsv of the
2026-09-09 neighbourhood controls), mapped through start/end protein IDs; overlap and distance in genes.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
W = HERE / "work"
CW = HERE.parent / "work"
EV = HERE.parents[2] / "evaluation"


def main():
    g = pd.read_parquet(CW / "genes_dark.parquet")
    pid2gid = g.set_index("protein_id").gid
    inst = pd.read_parquet(CW / "data/instances.parquet")
    mem = pd.read_parquet(CW / "data/members.parquet")
    mem = mem.merge(inst[["instance_id", "caller", "system_type", "start_idx", "end_idx"]], on=["instance_id", "caller"])
    mem = mem.merge(g[["gid", "genome_code", "contig_code", "position_index"]], on="gid")
    c2 = pd.read_csv(W / "c2_instances.tsv", sep="\t")
    win = pd.read_csv(W / "ctx_windows.tsv", sep="\t")
    rows = []
    for _, r in c2.iterrows():
        key = (r.genome_code, r.contig_code)
        m = mem[(mem.genome_code == key[0]) & (mem.contig_code == key[1])]
        gids = [int(x) for x in r.gids.split(";")]
        res = dict(cand=r.cand, instance_id=r.instance_id, genome=r.genome, phylum=r.phylum, ord=r.ord,
                   contig_n_genes=r.contig_n_genes, pos_min=r.pos_min, pos_max=r.pos_max)
        for cal in ("TXSScan", "DefenseFinder", "CRISPRCasTyper"):
            mm = m[m.caller == cal]
            res[f"{cal}_member"] = bool(mm.gid.isin(gids).any())
            if len(mm):
                d = np.where(mm.position_index < r.pos_min, r.pos_min - mm.position_index,
                             np.where(mm.position_index > r.pos_max, mm.position_index - r.pos_max, 0))
                j = int(np.argmin(d))
                res[f"{cal}_dist"] = int(d[j])
                res[f"{cal}_nearest"] = f"{mm.iloc[j].system_type}:{mm.iloc[j].instance_id}"
                spans = mm.drop_duplicates("instance_id")
                inside = spans[(spans.start_idx <= r.pos_min) & (spans.end_idx >= r.pos_max)]
                res[f"{cal}_inside_span"] = len(inside) > 0
            else:
                res[f"{cal}_dist"] = None
                res[f"{cal}_nearest"] = ""
                res[f"{cal}_inside_span"] = False
            res[f"{cal}_calls_on_contig"] = int(mm.instance_id.nunique()) if len(mm) else 0
        rows.append(res)
    R = pd.DataFrame(rows)
    R.to_csv(W / "c2_curated_proximity.tsv", sep="\t", index=False)
    pd.set_option("display.width", 250)
    print(R[["cand", "instance_id", "genome", "phylum", "contig_n_genes", "TXSScan_member", "TXSScan_inside_span",
             "TXSScan_dist", "TXSScan_nearest", "DefenseFinder_dist", "CRISPRCasTyper_dist"]].to_string())
    for c, x in R.groupby("cand"):
        d = x.TXSScan_dist.dropna()
        print(c, "instances", len(x), "TXSScan member", int(x.TXSScan_member.sum()), "inside span",
              int(x.TXSScan_inside_span.sum()), "TXSScan call on contig", int((x.TXSScan_calls_on_contig > 0).sum()),
              "dist<=5", int((d <= 5).sum()), "dist<=10", int((d <= 10).sum()), "dists", sorted(d.astype(int).tolist()))
    # controls: control cores on the same contigs (distance from control core to nearest TXSScan member)
    ctrl = win[(win.kind == "control") & win.cand.isin(["s34", "s33", "s11", "s7"])]
    # reconstruct control core bounds from window offsets is indirect; use the per-control windows' gids
    cc = []
    for (iid, j), x in ctrl.groupby(["instance_id", "ctrl"]):
        r = c2[c2.instance_id == iid].iloc[0]
        gg = g.set_index("gid").loc[x.gid]
        pos = gg.position_index
        # control core = positions between the two flanks
        lo_f, hi_f = pos[x.offset.values < 0], pos[x.offset.values > 0]
        allp = sorted(pos)
        m = mem[(mem.genome_code == r.genome_code) & (mem.contig_code == r.contig_code) & (mem.caller == "TXSScan")]
        if not len(m):
            cc.append(dict(cand=r.cand, instance_id=iid, ctrl=j, dist=None))
            continue
        # core: gap in the window positions
        cand_core = [p for p in range(min(allp), max(allp) + 1) if p not in set(allp)]
        lo, hi = (min(cand_core), max(cand_core)) if cand_core else (min(allp), max(allp))
        d = np.where(m.position_index < lo, lo - m.position_index, np.where(m.position_index > hi, m.position_index - hi, 0))
        cc.append(dict(cand=r.cand, instance_id=iid, ctrl=j, dist=int(d.min())))
    C = pd.DataFrame(cc)
    for c, x in C.groupby("cand"):
        d = x.dist.dropna()
        print("control", c, "control cores", len(x), "with TXSScan call on contig", len(d), "dist<=5", int((d <= 5).sum()),
              "dist<=10", int((d <= 10).sum()))
    # ---------------- earlier leads
    out = {}
    for name, d in (("c1043", EV / "c1043_neighborhood_positive_control_2026-09-09/source_loci.tsv"),
                    ("c3913", EV / "c3913_neighborhood_2026-09-09_v3/source_loci.tsv")):
        L = pd.read_csv(d, sep="\t")
        L["g0"] = L.start_protein_id.map(pid2gid)
        L["g1"] = L.end_protein_id.map(pid2gid)
        L = L.dropna(subset=["g0", "g1"])
        lo = np.minimum(L.g0, L.g1).astype(int)
        hi = np.maximum(L.g0, L.g1).astype(int)
        lead_gids = set()
        for a, b in zip(lo, hi):
            lead_gids |= set(range(a, b + 1))
        cas_gids = {int(x) for s in c2.gids for x in s.split(";")}
        ov = cas_gids & lead_gids
        # distance in genes (same contig) from each cassette instance to the nearest lead locus
        gi = g.set_index("gid")
        dists = []
        for _, r in c2.iterrows():
            gg = [int(x) for x in r.gids.split(";")]
            same = [(a, b) for a, b in zip(lo, hi) if gi.at[a, "genome_code"] == r.genome_code
                    and gi.at[a, "contig_code"] == r.contig_code]
            if same:
                dd = min(max(0, a - max(gg), min(gg) - b) for a, b in same)
                dists.append((r.instance_id, dd))
        out[name] = dict(lead_loci=len(L), lead_genes=len(lead_gids), cassette_genes=len(cas_gids),
                         shared_genes=len(ov), cassette_instances_with_lead_on_contig=len(dists),
                         distances=sorted(d for _, d in dists))
    # Pfam 39 DUF7287/7288/7289 (hmmsearch, E <= 1e-3) on cassette genes and on cluster:1043 genes
    dom = []
    for line in open(W / "hmm/duf728x_e3.domtbl"):
        if not line.startswith("#"):
            f = line.split()
            dom.append((int(f[0]), f[3], float(f[6])))
    dom = pd.DataFrame(dom, columns=["gid", "hmm", "E"])
    out["duf728x_genes_total"] = dom.gid.nunique()
    out["duf728x_on_cassette_genes"] = dom[dom.gid.isin(cas_gids)].groupby("hmm").gid.nunique().to_dict()
    L = pd.read_csv(EV / "c1043_neighborhood_positive_control_2026-09-09/source_loci.tsv", sep="\t")
    L["g0"] = L.start_protein_id.map(pid2gid)
    L["g1"] = L.end_protein_id.map(pid2gid)
    L = L.dropna(subset=["g0", "g1"])
    c1043 = set()
    for a, b in zip(np.minimum(L.g0, L.g1).astype(int), np.maximum(L.g0, L.g1).astype(int)):
        c1043 |= set(range(a, b + 1))
    out["duf728x_on_c1043_genes"] = dom[dom.gid.isin(c1043)].groupby("hmm").gid.nunique().to_dict()
    # token overlap: tokens of the cassette vs tokens of c1043 loci
    tok = g.set_index("gid").token
    out["c1043_tokens_top"] = pd.Series([tok[x] for x in c1043]).value_counts().head(8).to_dict()
    out["cassette_tokens"] = pd.Series([tok[x] for x in cas_gids]).value_counts().head(12).to_dict()
    out["shared_tokens_c1043_cassette"] = sorted(set(tok[list(c1043)]) & set(tok[list(cas_gids)]))
    json.dump(out, open(W / "c2_prior_leads.json", "w"), indent=1, default=int)
    print(json.dumps(out, indent=1, default=int))


if __name__ == "__main__":
    main()
