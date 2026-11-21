#!/usr/bin/env python3
"""Step 19 (rung 4 input): domain crops of the ESMFold2 models and the pair list for US-align. Each lead domain is compared
with an experimental structure of its family (first chain of the PDB entry), with the DPANN positive control (protein
annotated with the family at the gathering threshold) and the negative control (DPANN protein of similar length from an
unrelated family) compared with the same references. Writes folds/crops/*.pdb, folds/pairs.tsv; the structures and
references go to the biotite login node, where usalign_run.py runs US-align."""
import os, shutil, pandas as pd
os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/..")
SP = os.environ.get("SP", "/private/tmp/claude-501/-Users-jacob-Documents-Sandbox-elsa-test-ELSA/4a153818-3d15-49e9-b3de-41cb21532c73/scratchpad")
os.makedirs("folds/crops", exist_ok=True); os.makedirs(f"{SP}/us/str", exist_ok=True)
def crop(pid, a, b, tag):
    out = f"folds/crops/{pid}__{tag}.pdb"
    with open(out, "w") as o:
        for l in open(f"folds/out/{pid}.pdb"):
            if l.startswith("ATOM") and a <= int(l[22:26]) <= b: o.write(l)
        o.write("END\n")
    return f"{pid}__{tag}"
L = lambda pid: sum(1 for l in open(f"folds/out/{pid}.pdb") if l.startswith("ATOM") and l[12:16] == " CA ")
pairs = []
def add(lead, kind, q, r):
    pairs.append(dict(lead=lead, kind=kind, query=q, ref=r))
# L1 NTase-kinase
for m in ["HII14378.1", "MBX4196761.1", "HLC90278.1"]:
    n = crop(m, 1, 235, "NTase"); k = crop(m, 236, L(m), "kinase")
    add("L1", "member_NTase_vs_LicC", n, "1JYL"); add("L1", "member_NTase_vs_posctrl", n, "MCX8158633.1")
    for r in ["4R78", "2CKQ", "1J7L"]: add("L1", f"member_kinase_vs_{r}", k, r)
    add("L1", "member_kinase_vs_posctrl", k, "MCH7850213.1")
add("L1", "posctrl_NTP_transf_3_vs_LicC", "MCX8158633.1", "1JYL")
for r in ["4R78", "2CKQ", "1J7L"]: add("L1", f"posctrl_Choline_kinase_vs_{r}", "MCH7850213.1", r)
for r in ["1JYL", "4R78", "2CKQ", "1J7L"]: add("L1", f"negctrl_Cpn60_vs_{r}", "MCD6403374.1", r)
# L2 Tmk-NTP_transf_5
for m in ["MBT7106118.1", "MBN2421503.1", "MDP3698733.1"]:
    n = crop(m, 1, 215, "NTase5"); t = crop(m, 216, L(m), "Tmk")
    for r in ["4TMK", "1E2D"]: add("L2", f"member_Tmk_vs_{r}", t, r)
    add("L2", "member_Tmk_vs_posctrl", t, "JAFCEQ010000014.1_7")
    add("L2", "member_NTase5_vs_posctrl", n, "MBU0761438.1")
for r in ["4TMK", "1E2D"]: add("L2", f"posctrl_Tmk_vs_{r}", "JAFCEQ010000014.1_7", r); add("L2", f"negctrl_Enolase_vs_{r}", "MEK6860048.1", r)
add("L2", "negctrl_Enolase_vs_NTase5_posctrl", "MEK6860048.1", "MBU0761438.1")
add("L2", "negctrl_Methyltransf_vs_NTase5_posctrl", "MAG61195.1", "MBU0761438.1")
# L3 bicupin
for m, (a, b) in {"MBS3154103.1": (106, 245), "MDO8634247.1": (140, 264)}.items():
    n = crop(m, 1, a - 1, "cupinN"); c = crop(m, a, b, "cupinC")
    for q in (n, c):
        add("L3", "member_cupin_vs_posctrl", q, "DAISQB010000011.1_40"); add("L3", "member_cupin_vs_1DZR", q, "1DZR")
    add("L3", "member_full_vs_1J58_bicupin", m, "1J58"); add("L3", "member_halves_N_vs_C", n, c)
add("L3", "posctrl_MannoseP_isomer_vs_1DZR", "DAISQB010000011.1_40", "1DZR")
add("L3", "negctrl_Methyltransf_vs_1DZR", "MAG61195.1", "1DZR"); add("L3", "negctrl_Methyltransf_vs_1J58", "MAG61195.1", "1J58")
add("L3", "negctrl_Methyltransf_vs_posctrl", "MAG61195.1", "DAISQB010000011.1_40")
# L4 KamA-like
for m in ["MBT6744359.1", "MFA5796516.1"]: add("L4", "member_vs_KamA_2A5H", m, "2A5H"); add("L4", "member_vs_posctrl_K01843", m, "SKHI01000023.1_3")
add("L4", "posctrl_K01843_vs_KamA_2A5H", "SKHI01000023.1_3", "2A5H"); add("L4", "negctrl_tRNA-synt_2_vs_KamA_2A5H", "MCR4368540.1", "2A5H")
P = pd.DataFrame(pairs); P.to_csv("folds/pairs.tsv", sep="\t", index=False)
for f in os.listdir("folds/crops"): shutil.copy(f"folds/crops/{f}", f"{SP}/us/str/{f}")
for f in os.listdir("folds/out"):
    if f.endswith(".pdb"): shutil.copy(f"folds/out/{f}", f"{SP}/us/str/{f}")
for p in ["1JYL", "4R78", "2CKQ", "1J7L", "4TMK", "1E2D", "2A5H", "1DZR", "1J58"]:
    shutil.copy(f"{SP}/ref/pdb/{p}.pdb", f"{SP}/us/str/{p}.pdb")
shutil.copy("folds/pairs.tsv", f"{SP}/us/pairs.tsv")
print(len(P), "pairs")
