"""Part A.4 (biotite, login node, one thread): contig sequences for the rank-27 and rank-106 block
instances from the DPANN genome FASTAs (Sharur data/dpann/genomes_fna or genomes_fna_new, named
<genome>.fna). Writes the full contig of every instance plus its SHA-256 (checked against the Sharur
assembly_contig_provenance table on the Mac). Usage: python overlap_extract_nt.py instances.tsv out.fna"""
import csv
import hashlib
import os
import sys

BASES = ["/groups/banfield/users/jwestrob/bin/Sharur/data/dpann/genomes_fna",
         "/groups/banfield/users/jwestrob/bin/Sharur/data/dpann/genomes_fna_new"]
want = {}
for r in csv.DictReader(open(sys.argv[1]), delimiter="\t"):
    want.setdefault(r["genome"], set()).add(r["contig"])
out = open(sys.argv[2], "w")
for genome, contigs in sorted(want.items()):
    path = next((os.path.join(b, genome + ".fna") for b in BASES if os.path.exists(os.path.join(b, genome + ".fna"))), None)
    if path is None:
        print("missing", genome, file=sys.stderr)
        continue
    name, seq = None, []
    def flush():
        if name in contigs:
            s = "".join(seq).upper()
            out.write(f">{genome}|{name}|{hashlib.sha256(s.encode()).hexdigest()}|{path}\n{s}\n")
    for line in open(path):
        if line.startswith(">"):
            flush()
            name, seq = line[1:].split()[0], []
        else:
            seq.append(line.strip())
    flush()
out.close()
