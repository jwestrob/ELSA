"""Candidate 1 fold and sequence analysis (local, seconds) -> work/c1_folds.tsv, work/c1_summary.json,
work/c1_interface_columns.tsv.

Folds: 12 instances (7 phyla) AB single / MSA, swap controls, monomers, stoichiometries, ABC trimer and AB+C'
controls, fused homologs. Pass rule (screen-1 criteria 1-3): ipTM >= 0.75, interface PAE <= 5 A, >= 10 contacts,
interface pLDDT >= 70.
Interface conservation: interface residues (CB-CB <= 8 A to the partner chain) of every cognate MSA model mapped
to the columns of the 268-instance alignments; column conservation = frequency of the most common residue among
all 268 rows (gaps count as non-matching); interface versus other columns with >= 70 % occupancy, Mann-Whitney.
Sequence features of every instance chain: length, net charge (K+R-D-E), Kyte-Doolittle maximum over 19-residue
windows (>= 1.6 = candidate transmembrane segment), fraction ILVFMAWC, cysteine and histidine counts, SignalP 6.0.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from dd_pairs import pair_metrics

HERE = Path(__file__).resolve().parent
W = HERE / "work"
KD = dict(A=1.8, R=-4.5, N=-3.5, D=-3.5, C=2.5, Q=-3.5, E=-3.5, G=-0.4, H=-3.2, I=4.5, L=3.8, K=-3.9, M=1.9, F=2.8,
          P=-1.6, S=-0.8, T=-0.7, W=-0.9, Y=-1.3, V=4.2)


def read_fasta(p):
    d, h = {}, None
    for line in open(p):
        line = line.strip()
        if line.startswith(">"):
            h = line[1:].split()[0]
            d[h] = []
        elif line:
            d[h].append(line)
    return {k: "".join(v).rstrip("*") for k, v in d.items()}


def crit(r):
    return bool(r["pair_iptm"] >= 0.75 and r["contacts"] >= 10 and r["iface_pae"] is not None and r["iface_pae"] <= 5
                and r["iface_plddt"] >= 70)


def main():
    sel = pd.read_csv(W / "c1_selected.tsv", sep="\t")
    F1, F2 = W / "folds_c1", W / "folds_c1abc"
    rows = []
    for _, s in sel.iterrows():
        i = int(s.slot)
        r = dict(slot=i, instance=s.instance_id, phylum=s.phylum, ord=s.ord, genome=s.genome, lengths=s.lengths)
        for tag, fid in (("cog", f"c1_cog_{i:02d}"), ("msa", f"c1_cogmsa_{i:02d}"), ("swB", f"c1_swB_{i:02d}"),
                         ("swA", f"c1_swA_{i:02d}")):
            d, m = pair_metrics(F1, fid)
            ab = d["AB"]
            r[f"{tag}_iptm"] = round(ab["pair_iptm"], 3)
            r[f"{tag}_pae"] = ab["iface_pae"] and round(ab["iface_pae"], 2)
            r[f"{tag}_contacts"] = ab["contacts"]
            r[f"{tag}_iplddt"] = ab["iface_plddt"] and round(ab["iface_plddt"], 1)
            r[f"{tag}_pass"] = crit(ab)
            if tag in ("cog", "msa"):
                r[f"{tag}_plddt"] = m["chain_plddt"]
        for side in ("B", "A"):
            fid = f"c1_sw{side}msa_{i:02d}"
            try:
                d, _ = pair_metrics(F1, fid)
                r[f"sw{side}msa_iptm"] = round(d["AB"]["pair_iptm"], 3)
            except KeyError:
                r[f"sw{side}msa_iptm"] = np.nan
        for ch in ("A", "B"):
            _, m = pair_metrics(F1, f"c1_mono{ch}_{i:02d}") if False else (None, None)
        ma = json.loads([l for l in open(F1 / "metrics.jsonl") if f'"c1_monoA_{i:02d}"' in l][0])
        mb = json.loads([l for l in open(F1 / "metrics.jsonl") if f'"c1_monoB_{i:02d}"' in l][0])
        r["monoA_plddt"], r["monoB_plddt"] = ma["chain_plddt"][0], mb["chain_plddt"][0]
        r["monoA_ptm"], r["monoB_ptm"] = round(ma["ptm"], 3), round(mb["ptm"], 3)
        for tag, fid in (("abc", f"c1_abc_{i:02d}"), ("abcmsa", f"c1_abcmsa_{i:02d}"), ("abX", f"c1_abX_{i:02d}")):
            d, m = pair_metrics(F2, fid)
            r[f"{tag}_iptm"] = round(m["iptm"], 3)
            for pr in ("AB", "AC", "BC"):
                r[f"{tag}_{pr}_iptm"] = round(d[pr]["pair_iptm"], 3)
                r[f"{tag}_{pr}_pae"] = d[pr]["iface_pae"] and round(d[pr]["iface_pae"], 2)
                r[f"{tag}_{pr}_contacts"] = d[pr]["contacts"]
                r[f"{tag}_{pr}_pass"] = crit(d[pr])
            r[f"{tag}_plddt"] = m["chain_plddt"]
        mc = json.loads([l for l in open(F2 / "metrics.jsonl") if f'"c1_monoC_{i:02d}"' in l][0])
        r["monoC_plddt"], r["monoC_ptm"] = mc["chain_plddt"][0], round(mc["ptm"], 3)
        rows.append(r)
    T = pd.DataFrame(rows)
    T["delta_min"] = T.cog_iptm - T[["swA_iptm", "swB_iptm"]].max(axis=1)
    T.to_csv(W / "c1_folds.tsv", sep="\t", index=False)
    pd.set_option("display.width", 320)
    pd.set_option("display.max_columns", 60)
    print(T[["slot", "phylum", "ord", "lengths", "cog_iptm", "cog_pae", "cog_contacts", "cog_iplddt", "cog_pass",
             "msa_iptm", "msa_pae", "msa_pass", "swA_iptm", "swB_iptm", "swAmsa_iptm", "swBmsa_iptm", "delta_min",
             "monoA_plddt", "monoB_plddt", "cog_plddt"]].to_string())
    print(T[["slot", "abc_iptm", "abc_AB_iptm", "abc_AC_iptm", "abc_AC_pae", "abc_AC_contacts", "abc_BC_iptm",
             "abc_BC_pae", "abc_BC_contacts", "abc_BC_pass", "abcmsa_iptm", "abcmsa_BC_pass", "abX_iptm", "abX_AC_iptm",
             "abX_BC_iptm", "monoC_plddt", "monoC_ptm"]].to_string())
    S = dict(
        n=len(T), phyla=T.phylum.nunique(), orders=T.ord.nunique(),
        cog_iptm=dict(median=float(T.cog_iptm.median()), min=float(T.cog_iptm.min()), max=float(T.cog_iptm.max()),
                      ge075=int((T.cog_iptm >= 0.75).sum()), pass_c123=int(T.cog_pass.sum())),
        msa_iptm=dict(median=float(T.msa_iptm.median()), min=float(T.msa_iptm.min()), ge075=int((T.msa_iptm >= 0.75).sum()),
                      pass_c123=int(T.msa_pass.sum())),
        swaps_single=dict(n=24, median=float(pd.concat([T.swA_iptm, T.swB_iptm]).median()),
                          max=float(pd.concat([T.swA_iptm, T.swB_iptm]).max()),
                          ge05=int((pd.concat([T.swA_iptm, T.swB_iptm]) >= 0.5).sum())),
        swaps_msa=dict(n=12, median=float(pd.concat([T.swAmsa_iptm, T.swBmsa_iptm]).dropna().median()),
                       max=float(pd.concat([T.swAmsa_iptm, T.swBmsa_iptm]).dropna().max())),
        delta_min=dict(min=float(T.delta_min.min()), median=float(T.delta_min.median()),
                       ge030=int((T.delta_min >= 0.30).sum())),
        cog_iface_pae_range=[float(T.cog_pae.min()), float(T.cog_pae.max())],
        mono_plddt=dict(A=float(T.monoA_plddt.median()), B=float(T.monoB_plddt.median()),
                        A_in_complex=float(np.median([p[0] for p in T.cog_plddt])),
                        B_in_complex=float(np.median([p[1] for p in T.cog_plddt]))),
        abc=dict(iptm_median=float(T.abc_iptm.median()), ge075=int((T.abc_iptm >= 0.75).sum()),
                 BC_pass=int(T.abc_BC_pass.sum()), AC_pass=int(T.abc_AC_pass.sum()), AB_pass=int(T.abc_AB_pass.sum()),
                 msa_iptm_median=float(T.abcmsa_iptm.median()), msa_BC_pass=int(T.abcmsa_BC_pass.sum())),
        abX=dict(iptm_median=float(T.abX_iptm.median()), max=float(T.abX_iptm.max()),
                 Cprime_pair_max=float(T[["abX_AC_iptm", "abX_BC_iptm"]].max(axis=1).max()),
                 Cprime_pair_median=float(T[["abX_AC_iptm", "abX_BC_iptm"]].max(axis=1).median())),
        monoC=dict(plddt_median=float(T.monoC_plddt.median()), ptm_median=float(T.monoC_ptm.median())),
    )
    # stoichiometry
    st = {}
    for line in open(F1 / "metrics.jsonl"):
        m = json.loads(line)
        if m["fold_id"].startswith("c1_st_"):
            d, _ = pair_metrics(F1, m["fold_id"])
            st[m["fold_id"]] = dict(iptm=round(m["iptm"], 3), pairs={k: (round(v["pair_iptm"], 2), v["contacts"],
                                                                         v["iface_pae"] and round(v["iface_pae"], 1),
                                                                         v["pass"]) for k, v in d.items()})
    S["stoichiometry"] = st
    # fused
    fz = {}
    for fd in (F1, W / "folds_c1ext"):
        for line in open(fd / "metrics.jsonl"):
            m = json.loads(line)
            if "fused" in m["fold_id"]:
                fz[m["fold_id"]] = dict(ptm=round(m["ptm"], 3), plddt=m["chain_plddt"][0], length=m["lengths"][0])
    S["fused"] = fz
    # ------------------------------------------------ interface conservation
    A = read_fasta(W / "aln/c1A_inst.aln.fa")
    B = read_fasta(W / "aln/c1B_inst.aln.fa")
    cons = {}
    for fam, aln in (("A", A), ("B", B)):
        rows_ = list(aln.values())
        L = len(rows_[0])
        cc = []
        for j in range(L):
            col = [r_[j] for r_ in rows_]
            occ = sum(c != "-" for c in col) / len(col)
            vals = pd.Series([c for c in col if c != "-"]).value_counts()
            cc.append(dict(fam=fam, col=j, occ=occ, top=vals.index[0] if len(vals) else "-",
                           cons=(vals.iloc[0] / len(col)) if len(vals) else 0.0))
        cons[fam] = pd.DataFrame(cc)
    from dd_pairs import read_pdb
    hits = {("A", j): 0 for j in range(len(next(iter(A.values()))))}
    hits.update({("B", j): 0 for j in range(len(next(iter(B.values()))))})
    for _, s in sel.iterrows():
        i = int(s.slot)
        S_ = read_pdb(F1 / "pdb" / f"c1_cogmsa_{i:02d}.pdb")
        xa, xb = S_["A"][0], S_["B"][0]
        d = np.sqrt(((xa[:, None, :] - xb[None, :, :]) ** 2).sum(-1)) <= 8
        for fam, aln, mask in (("A", A, d.any(1)), ("B", B, d.any(0))):
            row = aln[s.instance_id]
            ri = -1
            for j, c in enumerate(row):
                if c != "-":
                    ri += 1
                    if mask[ri]:
                        hits[(fam, j)] += 1
    out = []
    for fam in ("A", "B"):
        c = cons[fam].copy()
        c["iface_frac"] = [hits[(fam, j)] / len(sel) for j in c.col]
        out.append(c)
    C = pd.concat(out)
    C.to_csv(W / "c1_interface_columns.tsv", sep="\t", index=False)
    C70 = C[C.occ >= 0.7]
    ic = C70[C70.iface_frac >= 0.5]
    oc = C70[C70.iface_frac < 0.5]
    S["interface_conservation"] = dict(
        cols_occ70=len(C70), iface_cols=len(ic), iface_cons_median=float(ic.cons.median()),
        other_cons_median=float(oc.cons.median()), mw_p_greater=float(mannwhitneyu(ic.cons, oc.cons, alternative="greater").pvalue),
        top_conserved=[(r.fam, int(r.col), r.top, round(r.cons, 2), round(r.iface_frac, 2))
                       for r in C70.sort_values("cons", ascending=False).head(14).itertuples()])
    # ------------------------------------------------ sequence features
    Sq = read_fasta(W / "dd_seqs.faa")
    c1 = pd.read_csv(W / "c1_instances.tsv", sep="\t")
    sp = pd.read_csv(W / "sp/prediction_results.txt", sep="\t", skiprows=1)
    sp.columns = [c.strip("# ") for c in sp.columns]
    spd = dict(zip(sp.ID.astype(str), sp.Prediction))
    feats = []
    for gl in c1.gids:
        for fam, gid in zip("AB", gl.split(";")):
            s = Sq[gid]
            kd = [np.mean([KD.get(c, 0) for c in s[k:k + 19]]) for k in range(max(1, len(s) - 18))]
            feats.append(dict(fam=fam, gid=gid, len=len(s), charge=sum(c in "KR" for c in s) - sum(c in "DE" for c in s),
                              kdmax19=max(kd), hyd=sum(c in "ILVFMAWC" for c in s) / len(s), cys=s.count("C"),
                              his=s.count("H"), signalp=spd.get(gid, "NA")))
    Fe = pd.DataFrame(feats)
    S["sequence_features"] = {fam: dict(n=len(x), len_median=float(x.len.median()), charge_median=float(x.charge.median()),
                                        charge_q10_q90=[float(x.charge.quantile(.1)), float(x.charge.quantile(.9))],
                                        kdmax19_median=float(x.kdmax19.median()), kdmax19_ge16=int((x.kdmax19 >= 1.6).sum()),
                                        hyd_median=float(x.hyd.median()), cys0=int((x.cys == 0).sum()),
                                        cys_median=float(x.cys.median()), his_median=float(x.his.median()),
                                        signalp=x.signalp.value_counts().to_dict())
                              for fam, x in Fe.groupby("fam")}
    json.dump(S, open(W / "c1_summary.json", "w"), indent=1, default=str)
    print(json.dumps({k: v for k, v in S.items() if k != "stoichiometry"}, indent=1, default=str))
    for k, v in st.items():
        print(k, v)


if __name__ == "__main__":
    main()
