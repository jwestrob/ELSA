#!/usr/bin/env python3
"""Copy the Pfam profiles Thy1 (PF02511) and Thymidylat_synt (PF00303) out of the Aksha Pfam store into
data/hmm/picked_hmm.txt (input for `aksha search --hmm_in`). Reads the store; writes only inside the thy folder.
usage: python code/02_hmm_extract.py"""
import os
SRC = os.path.expanduser("~/.config/Astra/PFAM/Pfam-A.hmm")
WANT = {"Thy1", "Thymidylat_synt"}
os.makedirs("data/hmm", exist_ok=True)
out, buf, name, got = open("data/hmm/picked_hmm.txt", "w"), [], None, []
with open(SRC) as f:
    for line in f:
        buf.append(line)
        if line.startswith("NAME "):
            name = line.split()[1]
        elif line.startswith("//"):
            if name in WANT:
                out.write("".join(buf)); got.append(name)
            buf, name = [], None
out.close()
print("extracted", got)
for l in open("data/hmm/picked_hmm.txt"):
    if l.startswith(("NAME", "ACC", "LENG", "GA ", "TC ", "NC ")):
        print(l.rstrip())
