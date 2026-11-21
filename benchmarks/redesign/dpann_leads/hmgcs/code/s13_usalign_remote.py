#!/usr/bin/env python3
"""Run on the biotite login node (single-threaded serial loop). Extract reference chains from experimental structures,
then US-align every single-chain model against every reference chain (-outfmt 0, raw text kept so the residue-level
alignment can be parsed), all model pairs, and the two lead homodimer co-folds against the E. faecalis HMGS dimer (1YSL
A+B, -mm 1). usage: s13_usalign_remote.py BASE_DIR"""
import os, subprocess, sys, itertools
B = sys.argv[1]
US = "/shared/software/bin/USalign"
REFS = {"1YSL_A": ("1YSL", "A"), "2P8U_A": ("2P8U", "A"), "6ESQ_I": ("6ESQ", "I"), "4YXT_A": ("4YXT", "A"),
        "1HNJ_A": ("1HNJ", "A"), "1DM3_A": ("1DM3", "A"), "2GFW_A": ("2GFW", "A"), "1BI5_A": ("1BI5", "A")}
os.makedirs(f"{B}/refchains", exist_ok=True)
def extract(pdb, chains, out):
    with open(f"{B}/pdb/{pdb}.pdb") as fh, open(out, "w") as o:
        for l in fh:
            if l.startswith("ENDMDL"):
                break
            if l.startswith(("ATOM", "HETATM")) and l[21] in chains:
                if l.startswith("HETATM") and l[17:20].strip() in ("HOH", "SO4", "GOL", "PO4", "PEG", "MES", "BTB", "K", "NA", "ZN", "TRS"):
                    continue
                o.write(l)
        o.write("END\n")
for k, (p, c) in REFS.items():
    extract(p, c, f"{B}/refchains/{k}.pdb")
extract("1YSL", "AB", f"{B}/refchains/1YSL_AB.pdb")
models = sorted(f[:-4] for f in os.listdir(f"{B}/models/single") if f.endswith(".pdb"))
os.makedirs(f"{B}/us_raw", exist_ok=True)
def run(a, b, tag, extra=()):
    r = subprocess.run([US, a, b, "-outfmt", "0", *extra], capture_output=True, text=True)
    open(f"{B}/us_raw/{tag}.txt", "w").write(r.stdout)
n = 0
for m in models:
    for k in REFS:
        run(f"{B}/models/single/{m}.pdb", f"{B}/refchains/{k}.pdb", f"{m}__{k}"); n += 1
for a, b in itertools.combinations(models, 2):
    run(f"{B}/models/single/{a}.pdb", f"{B}/models/single/{b}.pdb", f"{a}__{b}"); n += 1
for m in ["OJT94780.1", "JAJZNR010000018.1_8", "CAISET010000016.1_25"]:
    run(f"{B}/models/cofold/{m}.pdb", f"{B}/refchains/1YSL_AB.pdb", f"cofold_{m}__1YSL_AB", ("-mm", "1", "-ter", "1")); n += 1
print(n, "alignments")
