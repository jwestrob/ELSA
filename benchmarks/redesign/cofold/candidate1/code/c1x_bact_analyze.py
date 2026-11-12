"""Task 6 analysis (local, seconds): bacterial / non-DPANN A-B-C folds (work/bact/folds) against partner swaps.

Pair metrics: deepdive/dd_pairs.pair_metrics (pair ipTM, CB-CB <= 8 A contacts, interface PAE, interface pLDDT;
pass = ipTM >= 0.75, >= 10 contacts, PAE <= 5 A, pLDDT >= 70; screen-2 rule).
Active site: C's His and Cys located by global alignment of the bacterial C to the DPANN reference C (slot 1,
His 66 / Cys 100 = C alignment columns 76 / 135); B's column-48 residue located by alignment to the slot-1 B
(Gly 28). Measured as in c1x_struct.py: side-chain SASA of His / Cys in C cut out of the model vs in B+C (burial
by B), B residues <= 4.5 A from the His / Cys side chains, Cys SG to the nearest B backbone carbonyl carbon.
Chain roles: abc A/B/C; ab A/B; bc, bX, Xc: B-role = chain A, C-role = chain B; fc, fX: fused A|B = chain A,
C-role = chain B (the B segment is the fused chain's C-terminal half).
Writes work/bact/bact_folds.tsv and bact_summary.json.
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import Align
from Bio.Align import substitution_matrices
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent.parent
CO = HERE.parent
sys.path.insert(0, str(CO / "deepdive"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dd_pairs import pair_metrics  # noqa: E402
from c1x_struct import sc_sasa, sub, rf, BB  # noqa: E402

F = HERE / "work" / "bact" / "folds"
DD = CO / "deepdive" / "work"


def aligner():
    a = Align.PairwiseAligner(mode="global", open_gap_score=-10, extend_gap_score=-0.5)
    a.substitution_matrix = substitution_matrices.load("BLOSUM62")
    a.end_gap_score = 0
    return a


def map_pos(query, ref, ref_pos):
    """1-based residue of query aligned to 1-based ref_pos (None if gapped)"""
    al = aligner().align(query, ref)[0]
    for (qs, qe), (rs, re_) in zip(*al.aligned):
        if rs < ref_pos <= re_:
            return qs + (ref_pos - rs)
    return None


def main():
    sel = pd.read_csv(DD / "c1_selected.tsv", sep="\t")
    i1 = sel[sel.slot == 1].instance_id.iloc[0]
    refC = rf(DD / "c1_aln_out" / "C_inst.aln.fa")[i1].replace("-", "")
    refB = rf(DD / "aln" / "c1B_inst.aln.fa")[i1].replace("-", "")
    assert refC[65] == "H" and refC[99] == "C" and refB[27] == "G"
    met = [json.loads(l) for l in open(F / "metrics.jsonl")]
    P = PDBParser(QUIET=True)
    rows = []
    for m in met:
        fid = m["fold_id"]
        kind, locus = fid.split("_")[1], fid.split("_")[2]
        pm, _ = pair_metrics(F, fid)
        r = dict(fold_id=fid, kind=kind, locus=locus, iptm=m["iptm"], lengths=";".join(map(str, m["lengths"])))
        for k, v in pm.items():
            r[f"{k}_iptm"], r[f"{k}_pae"], r[f"{k}_contacts"], r[f"{k}_pass"] = (v["pair_iptm"], v["iface_pae"],
                                                                                 v["contacts"], v["pass"])
        roles = {"abc": ("B", "C"), "abX": ("B", "C"), "aXc": ("B", "C"), "bc": ("A", "B"), "bX": ("A", "B"),
                 "Xc": ("A", "B"), "fc": ("A", "B"), "fX": ("A", "B")}.get(kind)
        if roles and kind in ("abc", "bc", "fc"):
            st = P.get_structure(fid, F / "pdb" / f"{fid}.pdb")
            chB, chC = st[0][roles[0]], st[0][roles[1]]
            cseq = "".join(seq1(x.get_resname()) for x in chC)
            bseq = "".join(seq1(x.get_resname()) for x in chB)
            h, c = map_pos(cseq, refC, 66), map_pos(cseq, refC, 100)
            g48 = map_pos(bseq, refB, 28) if kind != "fc" else map_pos(bseq[len(bseq) // 2 - 15:], refB, 28)
            if kind == "fc" and g48:
                g48 += len(bseq) // 2 - 15
            r["C_his"] = f"{cseq[h - 1]}{h}" if h else None
            r["C_cys"] = f"{cseq[c - 1]}{c}" if c else None
            r["B_col48"] = f"{bseq[g48 - 1]}{g48}" if g48 else None
            if h and c and cseq[h - 1] == "H" and cseq[c - 1] == "C":
                H, Cy = chC[h], chC[c]
                r["hisCB_cysCB"] = round(float(np.linalg.norm(H["CB"].coord - Cy["CB"].coord)), 2)
                cut = sub(st, {roles[1]})
                both = sub(st, {roles[0], roles[1]})
                for nm, num in (("his", h), ("cys", c)):
                    a, _ = sc_sasa(cut, roles[1], num)
                    b, relb = sc_sasa(both, roles[1], num)
                    r[f"{nm}_sc_cut"], r[f"{nm}_sc_withB"] = round(a, 1), round(b, 1)
                    r[f"{nm}_buried_by_B"] = round(1 - b / a, 3) if a > 0 else None
                bat = [(rr, at) for rr in chB for at in rr if at.element != "H"]
                for nm, R in (("his", H), ("cys", Cy)):
                    sc = [at for at in R if at.get_id() not in BB]
                    close = sorted({rr.id[1] for rr, at in bat if min(np.linalg.norm(at.coord - x.coord) for x in sc) <= 4.5})
                    r[f"{nm}_Bcontacts"] = ";".join(f"{seq1(chB[k].get_resname())}{k}" for k in close)
                best = min(((np.linalg.norm(rr["C"].coord - Cy["SG"].coord), rr.id[1]) for rr in chB if "C" in rr))
                r["cysSG_Bcarbonyl"] = round(float(best[0]), 2)
                r["cysSG_Bcarbonyl_res"] = f"{seq1(chB[best[1]].get_resname())}{best[1]}"
                r["carbonyl_res_is_col48"] = (g48 == best[1])
        rows.append(r)
    D = pd.DataFrame(rows).sort_values(["locus", "kind"])
    D.to_csv(HERE / "work" / "bact" / "bact_folds.tsv", sep="\t", index=False)
    S = {}

    def summ(x, col):
        x = x[col].dropna()
        return dict(n=int(len(x)), median=round(float(x.median()), 3) if len(x) else None,
                    min=round(float(x.min()), 3) if len(x) else None, max=round(float(x.max()), 3) if len(x) else None)

    S["abc_trimer_iptm"] = summ(D[D.kind == "abc"], "iptm")
    S["abc_BC_iptm"] = summ(D[D.kind == "abc"], "BC_iptm")
    S["abc_BC_pass"] = int(D[D.kind == "abc"].BC_pass.sum())
    S["abc_AB_pass"] = int(D[D.kind == "abc"].AB_pass.sum())
    S["abc_AC_pass"] = int(D[D.kind == "abc"].AC_pass.sum())
    S["ab_dimer_iptm"] = summ(D[D.kind == "ab"], "iptm")
    S["ab_dimer_pass"] = int(D[D.kind == "ab"].AB_pass.sum())
    S["bc_dimer_iptm"] = summ(D[D.kind == "bc"], "iptm")
    S["bc_dimer_pass"] = int(D[D.kind == "bc"].AB_pass.sum())
    S["Cprime_trimer_iptm"] = summ(D[D.kind == "abX"], "iptm")
    S["Cprime_BCprime_iptm"] = summ(D[D.kind == "abX"], "BC_iptm")
    S["Cprime_BCprime_pass"] = int(D[D.kind == "abX"].BC_pass.sum())
    S["Bprime_trimer_iptm"] = summ(D[D.kind == "aXc"], "iptm")
    S["Bprime_BprimeC_iptm"] = summ(D[D.kind == "aXc"], "BC_iptm")
    S["Bprime_BprimeC_pass"] = int(D[D.kind == "aXc"].BC_pass.sum())
    for k in ("bX", "Xc", "fc", "fX"):
        S[f"{k}_iptm"] = D[D.kind == k][["locus", "iptm"]].to_dict("records")
    act = D[D.kind.isin(["abc", "bc", "fc"])]
    S["active_site"] = act[["fold_id", "C_his", "C_cys", "B_col48", "hisCB_cysCB", "his_buried_by_B", "cys_buried_by_B",
                            "his_Bcontacts", "cys_Bcontacts", "cysSG_Bcarbonyl", "cysSG_Bcarbonyl_res",
                            "carbonyl_res_is_col48"]].to_dict("records")
    json.dump(S, open(HERE / "work" / "bact" / "bact_summary.json", "w"), indent=1, default=str)
    pd.set_option("display.width", 300)
    pd.set_option("display.max_colwidth", 40)
    print(json.dumps({k: v for k, v in S.items() if k != "active_site"}, indent=1, default=str))
    print(act[["fold_id", "iptm", "C_his", "C_cys", "B_col48", "cys_buried_by_B", "his_buried_by_B", "cys_Bcontacts",
               "cysSG_Bcarbonyl", "cysSG_Bcarbonyl_res", "carbonyl_res_is_col48"]].to_string())


if __name__ == "__main__":
    main()
