#!/usr/bin/env python
"""Biotite login node, serial: US-align rotation matrices that move 2VVP chain A onto each monomer model and onto chain A
of each dimer model (for transplanting the crystallographic R5P into the model frame). Run as a shell loop on 2026-11-21
(same commands, recorded here in Python because .sh is outside the allowed file types).
usage: 16_matrix_remote.py WORKDIR   (WORKDIR holds monomers/, dimers/, ref/2VVP_A.pdb; writes mats/)"""
import glob, os, subprocess, sys
US = "/shared/software/bin/USalign"
os.chdir(sys.argv[1]); os.makedirs("mats", exist_ok=True)
for f in sorted(glob.glob("monomers/*.pdb")):
    b = os.path.basename(f)[:-4]
    subprocess.run([US, "ref/2VVP_A.pdb", f, "-m", f"mats/mono_{b}.txt"], capture_output=True)
for f in sorted(glob.glob("dimers/*.pdb")):
    b = os.path.basename(f)[:-4]; tmp = f"dimers_chainA_{b}.tmp"
    with open(tmp, "w") as o:
        o.writelines(l for l in open(f) if l.startswith("ATOM") and l[21] == "A"); o.write("END\n")
    subprocess.run([US, "ref/2VVP_A.pdb", tmp, "-m", f"mats/dimer_{b}.txt"], capture_output=True); os.remove(tmp)
print(len(os.listdir("mats")), "matrices")
