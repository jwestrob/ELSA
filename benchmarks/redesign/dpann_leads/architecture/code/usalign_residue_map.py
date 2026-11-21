#!/usr/bin/env python3
"""US-align residue correspondences (biotite login node): for each (query, reference, reference residue numbers) the
query residue superposed on each reference catalytic / ligand-contact residue (distance < 5 A pairs are marked ':' by
US-align, < 5 A in the superposition) is reported with its identity.
usage: usalign_residue_map.py STR_DIR OUT.tsv"""
import subprocess, sys
US = "/shared/software/bin/USalign"
sd, out = sys.argv[1:3]
JOBS = [  # query, reference, chain of reference, reference residues (author numbering) with roles
    ("HII14378.1__NTase", "1JYL", {11: "LicC CTP contact", 12: "LicC CTP contact", 13: "LicC CTP contact", 14: "LicC CTP contact", 90: "LicC S90", 107: "LicC D107 (2.2 A to CDP-choline/Mg)", 82: "LicC Y82 choline site", 85: "LicC Y85 choline site", 136: "LicC W136 choline site", 190: "LicC Y190 choline site"}),
    ("MBX4196761.1__NTase", "1JYL", {14: "LicC CTP contact", 107: "LicC D107", 82: "LicC Y82 choline site", 190: "LicC Y190 choline site"}),
    ("HII14378.1__kinase", "4R78", {181: "LicA N181 catalytic loop", 194: "LicA D194 Mg/DFG-like", 89: "LicA E89 adenine"}),
    ("HII14378.1__kinase", "1J7L", {188: "APH H188", 190: "APH D190 catalytic base", 195: "APH N195", 208: "APH D208 Mg"}),
    ("MBX4196761.1__kinase", "1J7L", {190: "APH D190 catalytic base", 195: "APH N195", 208: "APH D208 Mg"}),
    ("MBN2421503.1__Tmk", "1E2D", {18: "TMPK G18 P-loop", 19: "TMPK K19 P-loop", 45: "TMPK R45", 72: "TMPK F72 thymine stacking", 76: "TMPK R76", 97: "TMPK R97 DRY", 151: "TMPK Y151"}),
    ("MBT7106118.1__Tmk", "1E2D", {19: "TMPK K19 P-loop", 72: "TMPK F72 thymine stacking", 97: "TMPK R97 DRY"}),
    ("MBT6744359.1", "2A5H", {125: "KamA C125 [4Fe-4S]", 129: "KamA C129", 132: "KamA C132", 337: "KamA K337 PLP", 268: "KamA C268 Zn", 377: "KamA C377 Zn"}),
]
def ca_numbers(fn):
    nums, seen = [], set()
    for l in open(fn):
        if l.startswith("ENDMDL"): break
        if l.startswith(("ATOM", "HETATM")) and l[12:16] == " CA " and (l[17:20] != "HOH"):
            ch = l[21]
            if not nums: first = ch
            if ch != first: break
            key = (int(l[22:26]), l[26])
            if key not in seen and (l.startswith("ATOM") or l[17:20] == "MSE"): seen.add(key); nums.append(key[0])
    return nums
rows = []
for q, r, roles in JOBS:
    res = subprocess.run([US, f"{sd}/{q}.pdb", f"{sd}/{r}.pdb"], capture_output=True, text=True).stdout.splitlines()
    i = next(k for k, l in enumerate(res) if l.startswith('(":"'))
    s1, mk, s2 = res[i + 1], res[i + 2], res[i + 3]
    qn, rn = ca_numbers(f"{sd}/{q}.pdb"), ca_numbers(f"{sd}/{r}.pdb")
    a = b = 0; m = {}
    for c1, c, c2 in zip(s1, mk, s2):
        if c1 != "-" and c2 != "-": m[rn[b]] = (qn[a], c1, c2, c)
        if c1 != "-": a += 1
        if c2 != "-": b += 1
    for num, role in roles.items():
        if num in m:
            qa, c1, c2, c = m[num]
            rows.append([q, r, role, f"{c2}{num}", f"{c1}{qa}", "superposed<5A" if c in ":." else "aligned>5A", c1 == c2])
        else:
            rows.append([q, r, role, str(num), "-", "unaligned", False])
with open(out, "w") as o:
    o.write("query\tref\trole\tref_res\tquery_res\tstatus\tidentical\n")
    for x in rows: o.write("\t".join(map(str, x)) + "\n")
print(len(rows), "residue correspondences")
