"""Task 2 (local, seconds): variants of the A-B-C unit inside DPANN.

Inputs
  ../work/genes_dark.parquet, ../work/data/{genomes,contigs,instances,members}.parquet  (DPANN package copies)
  ../deepdive/work/hmm/c1_vs_dpann.domtbl   A and B family HMMs vs all 3,585,303 DPANN proteins (deep dive)
  work/hmm/c1C_vs_dpann.domtbl              C family HMM (258-row C instance alignment) vs the same proteins
  ../deepdive/work/c1_instances.tsv, c1_C_genes.tsv, c1_hmm_pairs.tsv, c1_selected.tsv
  Sharur DPANN DuckDB (read-only) for Pfam rows of neighbouring genes.
Family membership: full-sequence E <= 1e-5 (A, B as in the deep dive; C likewise). C hits are also split by bit
score (>= 150 'core'); HMM coverage = union of domain hmm intervals / 215.
Outputs work/variants/*.tsv and work/variants/summary.json.
"""
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent.parent
CO = HERE.parent
DD = CO / "deepdive" / "work"
O = HERE / "work" / "variants"
O.mkdir(parents=True, exist_ok=True)
SHARUR = "/Users/jacob/Documents/Sandbox/Sharur/Sharur/data/dpann/sharur.duckdb"
CTOK = {130016, 190823, 399854}
HLEN = 215


def domtbl(p):
    rows = []
    for line in open(p):
        if line.startswith("#"):
            continue
        f = line.split()
        rows.append(dict(gid=int(f[0]), tlen=int(f[2]), hmm=f[3], E=float(f[6]), score=float(f[7]),
                         dE=float(f[12]), hf=int(f[15]), ht=int(f[16]), af=int(f[17]), al_to=int(f[18])))
    d = pd.DataFrame(rows)
    out = []
    for (gid, hmm), x in d.groupby(["gid", "hmm"]):
        cov = np.zeros(400, bool)
        for _, r in x[x.dE <= 1e-3].iterrows():
            cov[r.hf:r.ht + 1] = True
        out.append(dict(gid=gid, hmm=hmm, E=x.E.iloc[0], score=x.score.iloc[0], hmm_cov=int(cov.sum()),
                        hmm_from=int(x.hf.min()), hmm_to=int(x.ht.max()), ali_from=int(x.af.min()),
                        ali_to=int(x.al_to.max()), ndom=len(x)))
    return pd.DataFrame(out)


def main():
    g = pd.read_parquet(CO / "work" / "genes_dark.parquet")
    gm = pd.read_parquet(CO / "work" / "data" / "genomes.parquet")
    ct = pd.read_parquet(CO / "work" / "data" / "contigs.parquet")
    g = g.merge(gm[["genome_code", "genome", "phylum", "ord", "completeness"]], on="genome_code")
    first = g.groupby(["genome_code", "contig_code"]).gid.agg(["min", "size"]).rename(columns={"min": "f0", "size": "n"})
    g = g.join(first, on=["genome_code", "contig_code"])
    g = g.set_index("gid", drop=False).sort_index()
    ab = domtbl(DD / "hmm" / "c1_vs_dpann.domtbl")
    c = domtbl(HERE / "work" / "hmm" / "c1C_vs_dpann.domtbl")
    A = ab[(ab.hmm == "c1A") & (ab.E <= 1e-5)].set_index("gid")
    B = ab[(ab.hmm == "c1B") & (ab.E <= 1e-5)].set_index("gid")
    C = c[c.E <= 1e-5].set_index("gid")
    Call = c.set_index("gid")
    S = {"n_A_hmm": len(A), "n_B_hmm": len(B), "n_C_hmm_E1e-5": len(C), "n_C_hmm_E1e-3": len(Call)}

    def rel(gid, k):
        """gene k steps downstream (k > 0) or upstream (k < 0) of gid in gid's transcription orientation"""
        r = g.loc[gid]
        p = r.position_index + (k if r.strand > 0 else -k)
        if p < 0 or p >= r.n:
            return None
        return int(r.f0 + p)

    # ---------------------------------------------------------------- engine C tokens and score calibration
    cg = pd.read_csv(DD / "c1_C_genes.tsv", sep="\t")
    cg = cg.merge(g[["gid", "token", "length"]].reset_index(drop=True), on="gid")
    cg["C_score"] = cg.gid.map(Call.score)
    cg["C_E"] = cg.gid.map(Call.E)
    cg["C_cov"] = cg.gid.map(Call.hmm_cov)
    S["engine_C_score_by_token"] = {int(t): dict(n=len(x), min=float(x.C_score.min()), median=float(x.C_score.median()),
                                                 hit_E1e5=int((x.C_E <= 1e-5).sum()))
                                    for t, x in cg.groupby("token")}

    # ---------------------------------------------------------------- (a) every C-family gene: what precedes it
    rows = []
    for gid, h in C.iterrows():
        r = g.loc[gid]
        up1, up2, dn1 = rel(gid, -1), rel(gid, -2), rel(gid, 1)

        def same(x):
            return x is not None and g.at[x, "strand"] == r.strand
        b1 = up1 is not None and up1 in B.index and same(up1)
        a2 = up2 is not None and up2 in A.index and same(up2)
        a1 = up1 is not None and up1 in A.index and same(up1)
        # A or B anywhere within +-5
        near = [rel(gid, k) for k in range(-5, 6) if k != 0]
        nearAB = [x for x in near if x is not None and (x in A.index or x in B.index)]
        if b1 and a2:
            ctx = "A-B-C"
        elif b1:
            ctx = "B-C (no A at -2)"
        elif a1:
            ctx = "A-C (no B)"
        elif up1 is None:
            ctx = "contig edge upstream"
        elif nearAB:
            ctx = "A/B within 5, rearranged"
        else:
            ctx = "no A/B within 5"
        gen_has_AB = bool(((A.index.isin(g.gid[g.genome_code == r.genome_code])).any())
                          and (B.index.isin(g.gid[g.genome_code == r.genome_code])).any())
        rows.append(dict(gid=gid, protein_id=r.protein_id, genome=r.genome, phylum=r.phylum, ord=r.ord,
                         length=int(r.length), token=int(r.token), partial=r.partial, C_E=h.E, C_score=h.score,
                         C_cov=h.hmm_cov, C_hmm=f"{h.hmm_from}-{h.hmm_to}", ctx=ctx,
                         up1=up1, up1_token=None if up1 is None else int(g.at[up1, "token"]),
                         up1_len=None if up1 is None else int(g.at[up1, "length"]),
                         up1_same_strand=None if up1 is None else bool(same(up1)),
                         up1_dark=None if up1 is None else bool(g.at[up1, "dark"]),
                         up2=up2, dn1=dn1, genome_has_AB=gen_has_AB,
                         edge_up=(r.position_index if r.strand > 0 else r.n - 1 - r.position_index),
                         edge_dn=(r.n - 1 - r.position_index if r.strand > 0 else r.position_index)))
    V = pd.DataFrame(rows)
    sh = duckdb.connect(SHARUR, read_only=True)
    sh.execute("SET threads=2")
    pids = list(set(g.loc[[x for x in V.up1.dropna().astype(int)], "protein_id"]))
    pf = sh.execute("SELECT protein_id, string_agg(name, '+' ORDER BY start_aa) FROM annotations WHERE source='pfam' "
                    "AND protein_id IN (" + ",".join("?" * len(pids)) + ") GROUP BY protein_id", pids).fetchall()
    pf = dict(pf)
    V["up1_pfam"] = V.up1.map(lambda x: pf.get(g.at[int(x), "protein_id"], "") if pd.notna(x) else "")
    V["core"] = V.C_score >= 150
    V.to_csv(O / "C_family_context.tsv", sep="\t", index=False)
    S["a_C_context_all"] = V.ctx.value_counts().to_dict()
    S["a_C_context_core150"] = V[V.core].ctx.value_counts().to_dict()
    S["a_C_context_noncore"] = V[~V.core].ctx.value_counts().to_dict()
    S["a_score_hist"] = np.histogram(V.C_score, bins=[0, 50, 100, 150, 200, 250, 350])[0].tolist()
    noab = V[V.ctx != "A-B-C"]
    S["a_noABC_by_phylum"] = noab.groupby(["phylum", "core"]).size().unstack(fill_value=0).to_dict()
    S["a_noABC_genome_has_AB"] = int(noab.genome_has_AB.sum())
    S["a_noABC_up1_pfam_top"] = noab.up1_pfam.replace("", "(none)").value_counts().head(15).to_dict()
    S["a_noABC_up1_token_top"] = noab.up1_token.value_counts().head(10).to_dict()
    S["a_noABC_up1_dark"] = int(noab.up1_dark.fillna(False).sum())
    S["a_ABC_by_phylum"] = V[V.ctx == "A-B-C"].phylum.value_counts().to_dict()

    # ---------------------------------------------------------------- (b) engine instances without C at +1
    inst = pd.read_csv(DD / "c1_instances.tsv", sep="\t")
    rowsb = []
    for _, r in inst.iterrows():
        gA, gB = [int(x) for x in r.gids.split(";")]
        d1, d2 = rel(gB, 1), rel(gB, 2)
        rowsb.append(dict(instance_id=r.instance_id, gidB=gB, phylum=r.phylum, ord=r.ord, dn1=d1,
                          dn1_token=None if d1 is None else int(g.at[d1, "token"]),
                          dn1_len=None if d1 is None else int(g.at[d1, "length"]),
                          dn1_same_strand=None if d1 is None else bool(g.at[d1, "strand"] == g.at[gB, "strand"]),
                          dn1_C_E=None if d1 is None else Call.E.get(d1, np.nan),
                          dn1_partial=None if d1 is None else g.at[d1, "partial"],
                          dn2=d2, dn2_C_E=None if d2 is None else Call.E.get(d2, np.nan),
                          dn2_len=None if d2 is None else int(g.at[d2, "length"]),
                          B_edge_dn=(g.at[gB, "n"] - 1 - g.at[gB, "position_index"]) if g.at[gB, "strand"] > 0
                          else g.at[gB, "position_index"]))
    Bt = pd.DataFrame(rowsb)
    pids = list(set(g.loc[[int(x) for x in Bt.dn1.dropna()], "protein_id"]))
    pf2 = dict(sh.execute("SELECT protein_id, string_agg(name, '+' ORDER BY start_aa) FROM annotations WHERE "
                          "source='pfam' AND protein_id IN (" + ",".join("?" * len(pids)) + ") GROUP BY protein_id",
                          pids).fetchall())
    Bt["dn1_pfam"] = Bt.dn1.map(lambda x: pf2.get(g.at[int(x), "protein_id"], "") if pd.notna(x) else "")
    Bt["dn1_is_Ctok"] = Bt.dn1_token.isin(CTOK)
    Bt.to_csv(O / "engine_B_downstream.tsv", sep="\t", index=False)
    S["b_engine_dn1_Ctoken"] = int(Bt.dn1_is_Ctok.sum())
    S["b_engine_no_dn1"] = int(Bt.dn1.isna().sum())
    S["b_engine_dn1_notC_tok_but_Chmm"] = int(((~Bt.dn1_is_Ctok) & (Bt.dn1_C_E <= 1e-5)).sum())
    S["b_rows_without_Ctok"] = Bt[~Bt.dn1_is_Ctok].drop(columns=["gidB"]).to_dict("records")
    # HMM-level pairs (325)
    hp = pd.read_csv(DD / "c1_hmm_pairs.tsv", sep="\t")
    hpc = []
    for _, r in hp.iterrows():
        d1 = rel(int(r.gidB), 1)
        d2 = rel(int(r.gidB), 2)
        st = g.at[int(r.gidB), "strand"]
        hpc.append("C at +1" if (d1 is not None and d1 in C.index and g.at[d1, "strand"] == st) else
                   "contig edge after B" if d1 is None else
                   "C at +2" if (d2 is not None and d2 in C.index) else "other")
    hp["after_B"] = hpc
    hp.to_csv(O / "hmm_pairs_downstream.tsv", sep="\t", index=False)
    S["b_hmm_pairs_after_B"] = hp.after_B.value_counts().to_dict()

    # ---------------------------------------------------------------- (c) Borvo calls
    gi = pd.read_parquet(CO / "work" / "data" / "instances.parquet")
    bo = gi[(gi.caller == "DefenseFinder") & (gi.system_type == "Borvo")].copy()
    bo["gid"] = bo.member_gids.map(lambda x: int(list(x)[0]))
    bo["n_members"] = bo.member_gids.map(len)
    bo = bo.merge(V[["gid", "ctx", "C_E", "C_score", "core"]], on="gid", how="left")
    bo["ctx"] = bo.ctx.fillna("not a C-HMM hit (E <= 1e-5)")
    bo["length"] = bo.gid.map(g.length)
    bo["phylum"] = bo.gid.map(g.phylum)
    bo["C_E_any"] = bo.gid.map(Call.E)
    bo[["instance_id", "genome", "contig", "gid", "length", "phylum", "n_members", "ctx", "C_E_any", "C_score"]].to_csv(
        O / "borvo_calls.tsv", sep="\t", index=False)
    S["c_borvo_n"] = len(bo)
    S["c_borvo_members"] = bo.n_members.value_counts().to_dict()
    S["c_borvo_ctx"] = bo.ctx.value_counts().to_dict()
    S["c_borvo_C_hit_any_E1e-3"] = int(bo.C_E_any.notna().sum())
    S["c_borvo_len_median"] = float(bo.length.median())
    S["c_borvo_by_phylum"] = bo.phylum.value_counts().to_dict()

    # ---------------------------------------------------------------- (d) minority C tokens
    rowsd = []
    for tok in (130016, 190823, 399854):
        for gid in g.gid[g.token == tok]:
            r = g.loc[gid]
            up1, dn1 = rel(gid, -1), rel(gid, 1)
            h = Call.loc[gid] if gid in Call.index else None
            dnh = Call.loc[dn1] if (dn1 is not None and dn1 in Call.index) else None
            rowsd.append(dict(token=tok, gid=gid, length=int(r.length), partial=r.partial, phylum=r.phylum,
                              C_E=None if h is None else h.E, C_hmm=None if h is None else f"{h.hmm_from}-{h.hmm_to}",
                              C_cov=None if h is None else h.hmm_cov,
                              ctx=V.set_index("gid").ctx.get(gid, "not C-HMM E<=1e-5"),
                              edge_dn=(r.n - 1 - r.position_index if r.strand > 0 else r.position_index),
                              edge_up=(r.position_index if r.strand > 0 else r.n - 1 - r.position_index),
                              dn1_len=None if dn1 is None else int(g.at[dn1, "length"]),
                              dn1_same=None if dn1 is None else bool(g.at[dn1, "strand"] == r.strand),
                              dn1_C=None if dnh is None else f"E{dnh.E:.1e}:{dnh.hmm_from}-{dnh.hmm_to}",
                              dn1_token=None if dn1 is None else int(g.at[dn1, "token"]),
                              nt_gap_dn1=None if dn1 is None else (int(g.at[dn1, "start"]) - int(r.end) if r.strand > 0
                                                                   else int(r.start) - int(g.at[dn1, "end"]))))
    D = pd.DataFrame(rowsd)
    D.to_csv(O / "C_tokens.tsv", sep="\t", index=False)
    for tok, x in D.groupby("token"):
        S[f"d_token_{tok}"] = dict(n=len(x), len_median=float(x.length.median()), len_range=[int(x.length.min()),
                                   int(x.length.max())], C_hit=int(x.C_E.notna().sum()),
                                   C_cov_median=None if x.C_cov.isna().all() else float(x.C_cov.median()),
                                   C_hmm_ranges=x.C_hmm.value_counts().head(5).to_dict(),
                                   ctx=x.ctx.value_counts().to_dict(), partial=x.partial.astype(str).value_counts().to_dict(),
                                   edge_dn0=int((x.edge_dn == 0).sum()),
                                   dn1_C=int(x.dn1_C.notna().sum()), dn1_C_examples=x.dn1_C.dropna().head(5).tolist())

    # ---------------------------------------------------------------- (e) slot 4 (c1_0142) third gene
    sel = pd.read_csv(DD / "c1_selected.tsv", sep="\t")
    i4 = sel[sel.slot == 4].instance_id.iloc[0]
    gC = int(cg[cg.instance_id == i4].gid.iloc[0])
    r = g.loc[gC]
    cr = ct[(ct.genome == r.genome)]
    contig = g.loc[gC]
    clen = ct[(ct.contig_code == r.contig_code)].length_nt
    dn1 = rel(gC, 1)
    S["e_slot4"] = dict(instance=i4, gid=gC, protein_id=r.protein_id, length=int(r.length), token=int(r.token),
                        partial=r.partial, strand=int(r.strand), start=int(r.start), end=int(r.end),
                        contig_genes=int(r.n), position_index=int(r.position_index),
                        edge_dn_genes=int(r.n - 1 - r.position_index if r.strand > 0 else r.position_index),
                        contig_len_nt=int(clen.iloc[0]) if len(clen) else None,
                        C_E=float(Call.E.get(gC, np.nan)), C_hmm=f"{Call.hmm_from.get(gC)}-{Call.hmm_to.get(gC)}",
                        dn1=None if dn1 is None else dict(gid=dn1, length=int(g.at[dn1, "length"]),
                                                          strand=int(g.at[dn1, "strand"]), token=int(g.at[dn1, "token"]),
                                                          start=int(g.at[dn1, "start"]), end=int(g.at[dn1, "end"]),
                                                          C_E=float(Call.E.get(dn1, np.nan)),
                                                          C_hmm=f"{Call.hmm_from.get(dn1)}-{Call.hmm_to.get(dn1)}"))
    json.dump(S, open(O / "summary.json", "w"), indent=1, default=str)
    print(json.dumps({k: v for k, v in S.items() if k != "b_rows_without_Ctok"}, indent=1, default=str))
    pd.set_option("display.width", 250)
    print(Bt[~Bt.dn1_is_Ctok].to_string())


if __name__ == "__main__":
    main()
