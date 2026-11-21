#!/usr/bin/env python
"""lut_loci s06 (Mac): candidates.tsv from the neighborhood map (n02, n05, n06), the site test (s02), co-folds (s05),
Foldseek and US-align readouts, and SignalP 6.0 on the representatives. Verdicts by tests_rule.json."""
import json

import numpy as np
import pandas as pd

B = "/Users/jacob/Documents/Sandbox/elsa_test/ELSA/benchmarks/redesign/dpann_leads/lut_loci"
D = pd.read_csv(B + "/data/sites_apo.tsv", sep="\t")
sp = pd.read_csv(B + "/data/signalp_reps.txt", sep="\t", skiprows=2, header=None).iloc[:, :2]
sp.columns = ["name", "sp"]
D = D.merge(sp, on="name", how="left")
C = pd.read_csv(B + "/data/cofold_best.tsv", sep="\t")
ctx = json.load(open(B + "/data/context.json"))["results"]
toks = json.load(open(B + "/data/context.json"))["family_tokens"]
S = pd.read_csv(B + "/data/families_summary.tsv", sep="\t").set_index("fam")
U = pd.read_csv(B + "/data/usalign_out.tsv", sep="\t")
U["fam"] = U.model.str.split("__").str[1]
F = pd.read_csv(B + "/data/foldseek_pdb.tsv", sep="\t", header=None,
                names="query,target,evalue,bits,alntmscore,qtmscore,ttmscore,lddt,prob,fident,qstart,qend,tstart,tend,theader".split(","))
F["fam"] = F["query"].str.split("__").str[1]
R = pd.read_parquet(B + "/data/xoxf_window.parquet")
nx = R[R.off == 0].xoxf.nunique()
META = {  # family -> (window family id, context key, tokens, AM1 locus, literature status from the reference set)
    "orf6": (19, "fam19", [298295], "META1_1746 (orf6), 55% median identity",
             "required for Ln-dependent growth (Tn screen: 0.03 vs 0.16 h-1; Roszczenko-Jasinska 2020 PMC7391723); no biochemistry"),
    "orf7": (42, "fam42", [361272], "META1_1747 (orf7), 40% median identity",
             "required for Ln-dependent growth (Tn screen: 9 h lag, 0.08 h-1; PMC7391723); no biochemistry"),
    "DUF3280": (13, "DUF3280", [220448], "META1_1777 (flank of lutA)", "no experimental record in the reference set"),
    "lutC": (8, "lutC", [253719, 136200], "META1_1780 (lutC)", "no experimental record in the reference set"),
    "lutG": (5, "lutG", [92541], "META1_1784 (lutG)",
             "deletion: 12 h lag, 0.03 h-1 in Ln-dependent growth (PMC7391723); exported; no biochemistry"),
    "lutI": (25, "lutI", [327650], "META1_1787 (lutI)", "no experimental record in the reference set"),
}
rows = []
for fam, (wf, ck, tk, am1, lit) in META.items():
    d = D[D.family == fam]
    s = S.loc[wf]
    c = ctx[ck]
    mg = c["matched_genus"]
    cl = C[(C.family == fam) & (C.ion == "LA")]
    cc = C[(C.family == fam) & (C.ion == "CA")]
    fx = F[(F.fam == fam) & (F.evalue <= 1e-3)].sort_values("evalue")
    xs = R[R.token.isin(tk) & (R.off != 0)].xoxf.nunique()
    site_pass = int(d.PASS_site.sum())
    ctx_pass = (mg["mh_or"] is None or (mg["mh_or"] or 0) >= 5) and mg["perm_p"] <= 0.01 or xs / nx >= 0.2
    rows.append(dict(
        candidate=fam, tokens=";".join(map(str, tk)), am1_ortholog=am1,
        modal_slot_in_lut_windows=f"{s.modal_slot} ({s.modal_slot_share:.2f})",
        lut_window_genes=int(s.genes), lut_loci=int(s.instances), genomes=int(s.genomes), genera=int(s.genera),
        genome_wide_genes=toks[ck]["genome_wide_genes"], near_K23995=f"{xs}/{nx}",
        median_length=int(d.length.median()), signal_peptide=f"{int((d.sp != 'OTHER').sum())}/{len(d)}",
        pfam="none" if fam != "DUF3280" else "DUF3280 (PF11684) only",
        slot_call="no label (partners carry no Pfam row)", bonded_to_lut_member=round(float(s.bonded_to_member), 3),
        literature=lit,
        foldseek_pdb=(f"{fx.iloc[0].target} E={fx.iloc[0].evalue:.1e} ({fx.iloc[0].theader[17:70].strip()})" if len(fx) else "no hit E<=1e-3"),
        TM_max_to_Ln_protein_crystals=round(float(U[(U.fam == fam) & U.crystal.isin(["8FNS", "8DQ2", "9VY4", "6OC5", "9C8W"])].TM1.max()), 2),
        reps=len(d), mean_pLDDT_median=float(d.mean_plddt.median()),
        cons_acidic_per100_median=float(d.cons_acidic_per100.median()),
        score_cons=";".join(map(str, d.score_cons.astype(int))), PASS_site=f"{site_pass}/{len(d)}",
        best_conserved_cluster=d.sort_values(["score_cons", "score_struct"], ascending=False).best_res.iloc[0],
        cofold_LA_best=";".join(f"{int(r.protein_O)}O/{int(r.acid_res)}acid" for r in cl.itertuples()),
        cofold_CA_best=";".join(f"{int(r.protein_O)}O/{int(r.acid_res)}acid" for r in cc.itertuples()),
        PASS_cofold=f"{int(cl.PASS_cofold_rule.sum())}/{len(cl)}",
        context_lut=f"lut {c['all_lut']} vs neg {c['all_neg']}; same-genus MH OR {mg['mh_or'] if mg['mh_or'] is not None else 'inf (0 negatives)'} p {mg['perm_p']}",
        PASS_context=bool(ctx_pass),
        verdict="KILLED" if site_pass == 0 else ("SUPPORTED" if site_pass >= len(d) / 2 else "WEAK"),
    ))
T = pd.DataFrame(rows)
reason = {
    "orf6": "no conserved carboxylate center (score_cons <= 2 at pLDDT >= 89); co-fold ions 1-2 O donors; no PDB fold match",
    "orf7": "no conserved carboxylate center (score_cons 1); LolA/LprG-like lipoprotein-carrier fold; co-fold <= 4 O",
    "DUF3280": "no conserved carboxylate center (score_cons 1); TolB-N/LpoB/FlgT-like periplasmic fold; co-fold 2 O",
    "lutC": "conserved E-E-D triad only (score_cons 3 in 4/6, below 4); cupredoxin fold without His/Cys/Met Cu ligands",
    "lutG": "no conserved carboxylate center (score_cons 1); no PDB fold match; co-fold <= 3 O",
    "lutI": "no conserved carboxylate center; 2 genera; carriers in one genome group, so the embedding lists cannot bond it",
}
T["one_line_reason"] = T.candidate.map(reason)
T.to_csv(B + "/candidates.tsv", sep="\t", index=False)
print(T.T.to_string())
