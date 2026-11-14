"""Candidate 1 (screen-1 rank 1, tokens 81718 A / 393054 B): fold design (local, seconds).

  prep   select 12 instances over the 7 phyla, choose length/hydrophobicity-matched swap donors, write the
         FASTAs that need aligning on biotite (donor token families for the MSA swap controls; the three fused
         homologs added to the concatenated A|B instance alignment)
  jobs   after alignment: write work/jobs_c1.jsonl (biohub_cofold.py format) and work/c1_fold_manifest.tsv

Instances (12): quotas Nanobdellota 3, Micrarchaeota 2, Aenigmatarchaeota 2, Iainarchaeota 2, EX4484-52 1,
Altiarchaeota 1, Nanohalarchaeota 1. Screen-1 replicates (366161;366162 Nanobdellota, 1310380;1310379
EX4484-52) are kept. Within a phylum, different orders first (largest order first), then no X, both lengths
within 0.85-1.20 x the family median, distance >= 1 gene from the contig edge, highest completeness.
Folds per instance: AB single sequence; AB with the block-paired MSA (all 268 instances, A and B rows from the
same instance, query first, columns = query residues); A alone; B alone; swap-B (A + donor matched to B) and
swap-A (donor matched to A + B), single sequence. Donors: members of 104 dark engine tokens (>= 98 % dark,
median 50-90 aa, >= 100 members, not module 1167) that occur in dark conserved block types of screen 1;
length within +-10 % and hydrophobic fraction (ILVFMAWC) within +-0.05 of the replaced chain, same phylum as
the instance when available, each donor token used once (seed 17).
MSA swap controls (12): instances 1-6 swap B, 7-12 swap A; the kept chain keeps its block-paired rows, the
donor takes its own token family (mafft, <= 267 other unique members), both truncated to the shallower depth.
Stoichiometry (3 instances: the Nanobdellota replicate, the first Micrarchaeota and first Aenigmatarchaeota
pick): A2B2, A2B, AB2, and the homodimer controls A2, B2, single sequence.
Fused homologs (3): single sequence, and with the concatenated A|B instance alignment as MSA (fused sequence
added with mafft --add, rows projected onto its residues).
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
from build_paired_msa import project, read_fasta  # noqa: E402  (reused from screen 1)

QUOTA = {"Nanobdellota": 3, "Micrarchaeota": 2, "Aenigmatarchaeota": 2, "Iainarchaeota": 2, "EX4484-52": 1,
         "Altiarchaeota": 1, "Nanohalarchaeota": 1}
FORCE = ["366161;366162", "1310380;1310379"]
FUSED = ["1218213", "1650759", "2628908"]


def hyd(s):
    return sum(c in "ILVFMAWC" for c in s) / len(s)


def read_aln(p):
    return read_fasta(p)


def select(c1, S):
    medA, medB = 62, 73
    c1 = c1.copy()
    a = [S[g.split(";")[0]] for g in c1.gids]
    b = [S[g.split(";")[1]] for g in c1.gids]
    c1["noX"] = [("X" not in x) and ("X" not in y) for x, y in zip(a, b)]
    c1["lenok"] = [0.85 * medA <= len(x) <= 1.2 * medA and 0.85 * medB <= len(y) <= 1.2 * medB for x, y in zip(a, b)]
    c1["edgeok"] = c1.edge_dist >= 1
    picks = []
    for ph, q in QUOTA.items():
        d = c1[c1.phylum == ph]
        forced = d[d.gids.isin(FORCE)]
        chosen = list(forced.instance_id)
        used_ord = set(forced.ord)
        d = d[~d.instance_id.isin(chosen)].sort_values(["noX", "lenok", "edgeok", "completeness"],
                                                       ascending=False)
        ords = d.ord.value_counts().index.tolist()
        while len(chosen) < q:
            fresh = [o for o in ords if o not in used_ord]
            pool = d[d.ord.isin(fresh)] if fresh else d
            pool = pool[pool.noX & pool.lenok] if (pool.noX & pool.lenok).any() else pool
            r = pool.iloc[0]
            chosen.append(r.instance_id)
            used_ord.add(r.ord)
            d = d[d.instance_id != r.instance_id]
        picks += chosen
    return c1.set_index("instance_id").loc[picks].reset_index()


def prep():
    S = read_fasta(W / "dd_seqs.faa")
    c1 = pd.read_csv(W / "c1_instances.tsv", sep="\t")
    sel = select(c1, S)
    sel["slot"] = range(1, len(sel) + 1)
    # ---- donors
    D = read_fasta(W / "donor_seqs.faa")
    g = pd.read_parquet(CW / "genes_dark.parquet", columns=["gid", "token", "genome_code"])
    gm = pd.read_parquet(CW / "data/genomes.parquet").set_index("genome_code").phylum
    dp = g[g.gid.astype(str).isin(D)].copy()
    dp["seq"] = dp.gid.astype(str).map(D)
    dp = dp[~dp.seq.str.contains("X")]
    dp["len"] = dp.seq.str.len()
    dp["hyd"] = dp.seq.map(hyd)
    dp["phylum"] = dp.genome_code.map(gm)
    rng = np.random.default_rng(17)
    dp = dp.iloc[rng.permutation(len(dp))]
    used = set()
    rows = []
    for _, r in sel.iterrows():
        ga, gb = r.gids.split(";")
        for side, gk in (("B", gb), ("A", ga)):
            s = S[gk]
            L, h = len(s), hyd(s)
            m = dp[(dp.len.between(0.9 * L, 1.1 * L)) & ((dp.hyd - h).abs() <= 0.05) & ~dp.token.isin(used)]
            same = m[m.phylum == r.phylum]
            pick = (same if len(same) else m).iloc[0]
            used.add(int(pick.token))
            rows.append(dict(slot=r.slot, instance_id=r.instance_id, side=side, replaced_gid=int(gk),
                             replaced_len=L, replaced_hyd=round(h, 3), donor_gid=int(pick.gid),
                             donor_token=int(pick.token), donor_len=int(pick.len), donor_hyd=round(pick.hyd, 3),
                             donor_phylum=pick.phylum, same_phylum=pick.phylum == r.phylum))
    don = pd.DataFrame(rows)
    sel.to_csv(W / "c1_selected.tsv", sep="\t", index=False)
    don.to_csv(W / "c1_donors.tsv", sep="\t", index=False)
    print(sel[["slot", "instance_id", "gids", "phylum", "ord", "lengths", "completeness", "edge_dist"]].to_string())
    print(don.to_string())
    # ---- FASTAs to align on biotite
    out = W / "c1_aln_in"
    out.mkdir(exist_ok=True)
    msa_sw = don[((don.slot <= 6) & (don.side == "B")) | ((don.slot > 6) & (don.side == "A"))]
    for _, r in msa_sw.iterrows():
        mem = g.gid[g.token == r.donor_token].astype(str).tolist()
        first = str(r.donor_gid)
        seen, rows_ = {D[first]}, [(first, D[first])]
        for x in rng.permutation(mem):
            s = D.get(x)
            if s and "X" not in s and s not in seen:
                seen.add(s)
                rows_.append((x, s))
            if len(rows_) >= 268:
                break
        with open(out / f"donor_{r.donor_token}.fa", "w") as f:
            for k, s in rows_:
                f.write(f">{k}\n{s}\n")
    with open(out / "fused.fa", "w") as f:
        for k in FUSED:
            f.write(f">fused_{k}\n{S[k]}\n")
    print("MSA swap donors:", len(msa_sw))


def jobs():
    S = read_fasta(W / "dd_seqs.faa")
    sel = pd.read_csv(W / "c1_selected.tsv", sep="\t")
    don = pd.read_csv(W / "c1_donors.tsv", sep="\t")
    D = read_fasta(W / "donor_seqs.faa")
    A = read_aln(W / "aln/c1A_inst.aln.fa")
    B = read_aln(W / "aln/c1B_inst.aln.fa")
    order_all = list(A)
    J, man = [], []

    def add(fid, cls, slot, chains, extra=None):
        J.append(dict(fold_id=fid, cls=cls, chains=chains))
        man.append(dict(fold_id=fid, cls=cls, slot=slot, chains=";".join(c["id"] for c in chains),
                        gids=";".join(str(c["gid"]) for c in chains),
                        lengths=";".join(str(len(c["seq"])) for c in chains),
                        msa_depth=len(chains[0]["msa"]) if chains[0].get("msa") else 0, **(extra or {})))

    for _, r in sel.iterrows():
        i = int(r.slot)
        ga, gb = r.gids.split(";")
        sa, sb = S[ga], S[gb]
        qk = r.instance_id
        order = [qk] + [k for k in order_all if k != qk]
        ra = project(A, order, qk, sa)
        rb = project(B, order, qk, sb)
        add(f"c1_cog_{i:02d}", "cognate", i, [dict(id="A", seq=sa, gid=ga), dict(id="B", seq=sb, gid=gb)])
        add(f"c1_cogmsa_{i:02d}", "cognate_msa", i, [dict(id="A", seq=sa, gid=ga, msa=ra),
                                                     dict(id="B", seq=sb, gid=gb, msa=rb)])
        add(f"c1_monoA_{i:02d}", "monomer_A", i, [dict(id="A", seq=sa, gid=ga)])
        add(f"c1_monoB_{i:02d}", "monomer_B", i, [dict(id="A", seq=sb, gid=gb)])
        for _, d in don[don.slot == i].iterrows():
            ds = D[str(d.donor_gid)]
            if d.side == "B":
                ch = [dict(id="A", seq=sa, gid=ga), dict(id="B", seq=ds, gid=int(d.donor_gid))]
            else:
                ch = [dict(id="A", seq=ds, gid=int(d.donor_gid)), dict(id="B", seq=sb, gid=gb)]
            add(f"c1_sw{d.side}_{i:02d}", f"swap_{d.side}", i, ch, dict(donor_token=int(d.donor_token)))
            msa_side = "B" if i <= 6 else "A"
            if d.side == msa_side:
                dal = read_aln(W / f"c1_aln_out/donor_{d.donor_token}.aln.fa")
                dk = str(d.donor_gid)
                dorder = [dk] + [k for k in dal if k != dk]
                rd = project(dal, dorder, dk, ds)
                if d.side == "B":
                    m1, m2 = ra, rd
                else:
                    m1, m2 = rd, rb
                depth = min(len(m1), len(m2))
                for c, m in zip(ch, (m1, m2)):
                    c = c
                ch2 = [dict(ch[0], msa=m1[:depth]), dict(ch[1], msa=m2[:depth])]
                add(f"c1_sw{d.side}msa_{i:02d}", f"swap_{d.side}_msa", i, ch2, dict(donor_token=int(d.donor_token)))
    # stoichiometry
    st = sel[sel.phylum.isin(["Nanobdellota", "Micrarchaeota", "Aenigmatarchaeota"])].groupby("phylum").head(1)
    for _, r in st.iterrows():
        i = int(r.slot)
        ga, gb = r.gids.split(";")
        sa, sb = S[ga], S[gb]
        shapes = {"A2B2": "AABB", "A2B": "AAB", "AB2": "ABB", "A2": "AA", "B2": "BB"}
        for nm, comp in shapes.items():
            ch = [dict(id="ABCD"[j], seq=sa if c == "A" else sb, gid=ga if c == "A" else gb)
                  for j, c in enumerate(comp)]
            add(f"c1_st_{i:02d}_{nm}", f"stoich_{nm}", i, ch, dict(composition=comp))
    # fused
    F = read_aln(W / "c1_aln_out/fused_in_AB.aln.fa")
    for k in FUSED:
        s = S[k]
        add(f"c1_fused_{k}", "fused", 0, [dict(id="A", seq=s, gid=k)])
        fk = f"fused_{k}"
        order = [fk] + [x for x in F if not x.startswith("fused_")]
        rows = project(F, order, fk, s)
        add(f"c1_fusedmsa_{k}", "fused_msa", 0, [dict(id="A", seq=s, gid=k, msa=rows)])
    with open(W / "jobs_c1.jsonl", "w") as f:
        for j in J:
            f.write(json.dumps(j) + "\n")
    m = pd.DataFrame(man)
    m.to_csv(W / "c1_fold_manifest.tsv", sep="\t", index=False)
    print(m.cls.value_counts().to_string(), "\ntotal", len(m))


def abc():
    """Third gene C (+1 neighbour of B, token 130016 / 190823 / 399854 in 258 of 267 instances): ABC single
    sequence, ABC with the block-paired MSA (rows = the 258 instances carrying C, query first), AB + C' with
    C' a length (+-10 %) and hydrophobicity (+-0.05) matched member of a dark screen-1 block token of median
    170-270 aa (not module 1167), same phylum when available, each token once (seed 17); C alone."""
    S = read_fasta(W / "dd_seqs.faa")
    sel = pd.read_csv(W / "c1_selected.tsv", sep="\t")
    cc = pd.read_csv(W / "c1_C_genes.tsv", sep="\t").set_index("instance_id").gid
    A = read_aln(W / "aln/c1A_inst.aln.fa")
    B = read_aln(W / "aln/c1B_inst.aln.fa")
    C = {k: v.upper() for k, v in read_aln(W / "c1_aln_out/C_inst.aln.fa").items()}
    D = read_fasta(W / "donorC_seqs.faa")
    g = pd.read_parquet(CW / "genes_dark.parquet", columns=["gid", "token", "genome_code"])
    gm = pd.read_parquet(CW / "data/genomes.parquet").set_index("genome_code").phylum
    dp = g[g.gid.astype(str).isin(D)].copy()
    dp["seq"] = dp.gid.astype(str).map(D)
    dp = dp[~dp.seq.str.contains("X")]
    dp["len"] = dp.seq.str.len()
    dp["hyd"] = dp.seq.map(hyd)
    dp["phylum"] = dp.genome_code.map(gm)
    rng = np.random.default_rng(17)
    dp = dp.iloc[rng.permutation(len(dp))]
    used = set()
    J, man = [], []
    keys = [k for k in A if k in C]
    for _, r in sel.iterrows():
        i = int(r.slot)
        ga, gb = r.gids.split(";")
        gc = str(cc[r.instance_id])
        sa, sb, sc = S[ga], S[gb], S[gc]
        qk = r.instance_id
        order = [qk] + [k for k in keys if k != qk]
        ra, rb, rc = project(A, order, qk, sa), project(B, order, qk, sb), project(C, order, qk, sc)
        L, h = len(sc), hyd(sc)
        m = dp[(dp.len.between(0.9 * L, 1.1 * L)) & ((dp.hyd - h).abs() <= 0.05) & ~dp.token.isin(used)]
        same = m[m.phylum == r.phylum]
        pk = (same if len(same) else m).iloc[0]
        used.add(int(pk.token))
        items = [
            (f"c1_abc_{i:02d}", "abc", [dict(id="A", seq=sa, gid=ga), dict(id="B", seq=sb, gid=gb),
                                         dict(id="C", seq=sc, gid=gc)], {}),
            (f"c1_abcmsa_{i:02d}", "abc_msa", [dict(id="A", seq=sa, gid=ga, msa=ra), dict(id="B", seq=sb, gid=gb, msa=rb),
                                                dict(id="C", seq=sc, gid=gc, msa=rc)], {}),
            (f"c1_abX_{i:02d}", "abc_swapC", [dict(id="A", seq=sa, gid=ga), dict(id="B", seq=sb, gid=gb),
                                               dict(id="C", seq=pk.seq, gid=int(pk.gid))],
             dict(donor_token=int(pk.token), donor_len=int(pk.len), c_len=L, donor_same_phylum=pk.phylum == r.phylum)),
            (f"c1_monoC_{i:02d}", "monomer_C", [dict(id="A", seq=sc, gid=gc)], {}),
        ]
        for fid, cls, ch, ex in items:
            J.append(dict(fold_id=fid, cls=cls, chains=ch))
            man.append(dict(fold_id=fid, cls=cls, slot=i, chains=";".join(c["id"] for c in ch),
                            gids=";".join(str(c["gid"]) for c in ch), lengths=";".join(str(len(c["seq"])) for c in ch),
                            msa_depth=len(ch[0]["msa"]) if ch[0].get("msa") else 0, **ex))
    with open(W / "jobs_c1_abc.jsonl", "w") as f:
        for j in J:
            f.write(json.dumps(j) + "\n")
    m = pd.DataFrame(man)
    m.to_csv(W / "c1_abc_manifest.tsv", sep="\t", index=False)
    print(m.to_string())


if __name__ == "__main__":
    {"prep": prep, "jobs": jobs, "abc": abc}[sys.argv[1]]()
