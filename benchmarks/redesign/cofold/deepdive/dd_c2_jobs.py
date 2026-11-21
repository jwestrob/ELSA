"""Candidate 2 (DUF7287/DUF7288 cassette; screen-2 rank 34 in Nanobdellota, rank 7 in Micrarchaeota /
Iainarchaeota): fold design (local, seconds). Writes work/jobs_c2.jsonl and work/c2_fold_manifest.tsv.

Cut rule (every chain, cassette and control alike): remove residues 1..E, where E is the end of the first run of
>= 10 consecutive positions whose centred 7-residue window holds >= 4 of I/L/V/F/M/A/W/C, the run starting within
the first 40 residues. This removes the class III leader peptide and the hydrophobic alpha1-N segment together
(E = 22-47 for the 14 cassette chains). Chains without such a run are not truncated (none among the cassette).

Folds, per replicate (screen-2 replicates: rank 34 r1 GCA_016188045.1, r2 GCA_016784165.1; rank 7 r1
GCA_013426015.1, r2 GCA_024277355.1):
  c2_cplxT_<rank>_r<k>        whole cassette, every chain truncated, single sequence
  c2_cplxTmsa_<rank>_r<k>     the same with the screen-2 block-paired MSA (10 rows rank 34, 29 rows rank 7), columns
                              of the removed residues dropped
  c2_pair_<rank>_r<k>_<XY>    every chain pair, full length, single sequence
  c2_pairT_<rank>_r<k>_<XY>   every chain pair, truncated
  c2_ctl_<rank>_r<k>_<XY>_<s><d>   chain X with a prepilin-like control in place of Y (s = 'Y'), or a control in place
                              of X with Y kept (s = 'X'); draw d = 1, 2; full length
  c2_ctlT_...                 the same, truncated (draw 1 only)
Control pool ('prepilin-like'): SignalP 6.0 Sec/SPIII (PILIN) prediction, an N-terminal hydrophobic run by the
cut rule, and either membership of a curated TXSScan Archaeal-T4P call (tier 1) or a Pfam Class_IIIsignal /
N_methyl / Pilin_N row (tier 2); tokens of the cassette families and of the 14 families triaged as DUF7287/7288-
like in screen 2 left out; length within +-15 % (then +-25 %) of the chain replaced; same phylum first, tier 1
first, each token used at most once per replicate (seed 17).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
W = HERE / "work"
CW = HERE.parent / "work"
sys.path.insert(0, str(HERE.parent))
from build_paired_msa import read_fasta  # noqa: E402

HYD = set("ILVFMAWC")
REPS = {("034", 1): [590052, 590051, 590050, 590049], ("034", 2): [722343, 722342, 722341, 722340],
        ("007", 1): [429575, 429576, 429577], ("007", 2): [1761769, 1761768, 1761767]}
PHYLA = {("034", 1): ["Nanobdellota"], ("034", 2): ["Nanobdellota"],
         ("007", 1): ["Micrarchaeota", "Iainarchaeota"], ("007", 2): ["Iainarchaeota", "Micrarchaeota"]}
EXCL_TOK = {15180, 287285, 354137, 96948, 142248, 55170, 410168, 55171, 111963, 117903, 119468, 120270, 131677,
            212044, 238910, 321912, 383280}
PIL = {"arCOG05987", "arCOG01824", "arCOG01829_Pil", "arCOG03926", "arCOG06117", "arCOG03872", "arCOG03870",
       "arCOG03871", "arCOG01822", "arCOG02420", "arCOG03821", "arCOG03360", "arCOG02911"}


def hyd_end(s, w=7, k=4, start_max=40, min_run=10):
    n = len(s)
    flag = []
    for i in range(n):
        a, b = max(0, i - w // 2), min(n, i + w // 2 + 1)
        flag.append(sum(c in HYD for c in s[a:b]) >= k)
    i = 0
    while i < min(start_max, n):
        if flag[i]:
            j = i
            while j < n and flag[j]:
                j += 1
            if j - i >= min_run:
                return j
            i = j
        else:
            i += 1
    return 0


def main():
    S = read_fasta(W / "dd_seqs.faa")
    g = pd.read_parquet(CW / "genes_dark.parquet", columns=["gid", "token", "genome_code", "length"])
    gm = pd.read_parquet(CW / "data/genomes.parquet").set_index("genome_code").phylum
    sp = pd.read_csv(W / "sp/prediction_results.txt", sep="\t", skiprows=1)
    sp.columns = [c.strip("# ") for c in sp.columns]
    sp["gid"] = sp.ID.astype(int)
    t4p = pd.read_csv(W / "t4p_members.tsv", sep="\t")
    c3 = pd.read_csv(W / "classIII_pfam.tsv", sep="\t")
    c2i = pd.read_csv(W / "c2_instances.tsv", sep="\t")
    cas_gids = {int(x) for s in c2i.gids for x in s.split(";")}
    pool = sp[sp.Prediction == "PILIN"].merge(g, on="gid")
    pool = pool[~pool.token.isin(EXCL_TOK) & ~pool.gid.isin(cas_gids)].copy()
    pool["seq"] = pool.gid.astype(str).map(S)
    pool = pool[pool.seq.notna() & ~pool.seq.str.contains("X")]
    pool["cut"] = pool.seq.map(hyd_end)
    pool = pool[pool.cut > 0]
    pool["len"] = pool.seq.str.len()
    pool["phylum"] = pool.genome_code.map(gm)
    t1 = set(t4p.gid[t4p.profile_name.str.replace("Archaeal-T4P_", "").isin(PIL)])
    pool["tier"] = np.where(pool.gid.isin(t1), 1, np.where(pool.gid.isin(c3.gid), 2, 3))
    pool = pool[pool.tier <= 2]
    rng = np.random.default_rng(17)
    pool = pool.iloc[rng.permutation(len(pool))]
    pool.drop(columns=["seq"]).to_csv(W / "c2_control_pool.tsv", sep="\t", index=False)
    print("control pool", len(pool), pool.tier.value_counts().to_dict(), pool.phylum.value_counts().head(6).to_dict())

    msa = {}
    for f in ("jobs_msa_b1_go.jsonl", "jobs_msa_b3_go.jsonl"):
        for line in open(CW / "s2" / f):
            j = json.loads(line)
            for (rk, rep) in REPS:
                if j["fold_id"] == f"msa2_cog2_{rk}_r{rep}":
                    msa[(rk, rep)] = {c["gid"]: c["msa"] for c in j["chains"]}
    J, man, ctl_rows = [], [], []

    def add(fid, cls, key, chains, extra=None):
        J.append(dict(fold_id=fid, cls=cls, chains=chains))
        man.append(dict(fold_id=fid, cls=cls, rank=key[0], rep=key[1], chains=";".join(c["id"] for c in chains),
                        gids=";".join(str(c["gid"]) for c in chains), lengths=";".join(str(len(c["seq"])) for c in chains),
                        msa_depth=len(chains[0]["msa"]) if chains[0].get("msa") else 0, **(extra or {})))

    for key, gids in REPS.items():
        rk, rep = key
        ids = "ABCD"[:len(gids)]
        seqs = {x: S[str(x)] for x in gids}
        cut = {x: hyd_end(seqs[x]) for x in gids}
        for x in gids:
            print(rk, rep, x, len(seqs[x]), "cut", cut[x], seqs[x][:cut[x]], "|", seqs[x][cut[x]:cut[x] + 10])
        add(f"c2_cplxT_{rk}_r{rep}", "complex_trunc", key,
            [dict(id=c, seq=seqs[x][cut[x]:], gid=x, cut=cut[x]) for c, x in zip(ids, gids)])
        m = msa[key]
        add(f"c2_cplxTmsa_{rk}_r{rep}", "complex_trunc_msa", key,
            [dict(id=c, seq=seqs[x][cut[x]:], gid=x, cut=cut[x], msa=[r[cut[x]:] for r in m[x]])
             for c, x in zip(ids, gids)])
        used = set()
        for i in range(len(gids)):
            for k in range(i + 1, len(gids)):
                X, Y = gids[i], gids[k]
                pr = ids[i] + ids[k]
                add(f"c2_pair_{rk}_r{rep}_{pr}", "pair", key,
                    [dict(id="A", seq=seqs[X], gid=X), dict(id="B", seq=seqs[Y], gid=Y)], dict(pair=pr))
                add(f"c2_pairT_{rk}_r{rep}_{pr}", "pair_trunc", key,
                    [dict(id="A", seq=seqs[X][cut[X]:], gid=X), dict(id="B", seq=seqs[Y][cut[Y]:], gid=Y)], dict(pair=pr))
                for side, keep, repl in (("Y", X, Y), ("X", Y, X)):
                    L = len(seqs[repl])
                    picks = []
                    for tol in (0.15, 0.25, 0.4):
                        c = pool[(pool.len.between((1 - tol) * L, (1 + tol) * L)) & ~pool.token.isin(used)
                                 & ~pool.gid.isin([p.gid for p in picks])]
                        c = c.assign(ph=~c.phylum.isin(PHYLA[key])).sort_values(["ph", "tier"], kind="stable")
                        for _, p in c.iterrows():
                            if len(picks) < 2 and p.token not in {q.token for q in picks}:
                                picks.append(p)
                        if len(picks) >= 2:
                            break
                    for d, p in enumerate(picks, 1):
                        used.add(int(p.token))
                        ps = p.seq
                        ctl_rows.append(dict(rank=rk, rep=rep, pair=pr, side=side, draw=d, kept_gid=keep, replaced_gid=repl,
                                             replaced_len=L, ctrl_gid=int(p.gid), ctrl_token=int(p.token), ctrl_len=len(ps),
                                             ctrl_tier=int(p.tier), ctrl_phylum=p.phylum, ctrl_cut=int(p.cut)))
                        if side == "Y":
                            ch = [dict(id="A", seq=seqs[X], gid=X), dict(id="B", seq=ps, gid=int(p.gid))]
                            chT = [dict(id="A", seq=seqs[X][cut[X]:], gid=X), dict(id="B", seq=ps[p.cut:], gid=int(p.gid))]
                        else:
                            ch = [dict(id="A", seq=ps, gid=int(p.gid)), dict(id="B", seq=seqs[Y], gid=Y)]
                            chT = [dict(id="A", seq=ps[p.cut:], gid=int(p.gid)), dict(id="B", seq=seqs[Y][cut[Y]:], gid=Y)]
                        ex = dict(pair=pr, side=side, draw=d, ctrl_tier=int(p.tier))
                        add(f"c2_ctl_{rk}_r{rep}_{pr}_{side}{d}", "control", key, ch, ex)
                        if d == 1:
                            add(f"c2_ctlT_{rk}_r{rep}_{pr}_{side}{d}", "control_trunc", key, chT, ex)
    with open(W / "jobs_c2.jsonl", "w") as f:
        for j in J:
            f.write(json.dumps(j) + "\n")
    M = pd.DataFrame(man)
    M.to_csv(W / "c2_fold_manifest.tsv", sep="\t", index=False)
    C = pd.DataFrame(ctl_rows)
    C.to_csv(W / "c2_controls.tsv", sep="\t", index=False)
    print(M.cls.value_counts().to_string(), "\ntotal", len(M))
    print(C.groupby(["rank", "rep"]).agg(n=("ctrl_gid", "size"), tier1=("ctrl_tier", lambda s: int((s == 1).sum())),
                                         same_phylum=("ctrl_phylum", "size")).to_string())
    print(C[["rank", "rep", "pair", "side", "draw", "replaced_len", "ctrl_len", "ctrl_tier", "ctrl_phylum"]].to_string())


if __name__ == "__main__":
    main()
