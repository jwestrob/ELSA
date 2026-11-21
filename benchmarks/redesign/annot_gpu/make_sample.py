#!/usr/bin/env python3
"""Draw a uniform random protein sample from a FASTA for the GPU/CPU validation gate.

Usage: make_sample.py IN.faa N SEED OUT.faa OUT.ids
Records keep their input order; headers and sequences are copied byte for byte.
Single process (login node).
"""
import random
import sys


def records(path):
    header, seq = None, []
    with open(path) as fh:
        for line in fh:
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(seq)
                header, seq = line.rstrip("\n"), []
            else:
                seq.append(line.rstrip("\n"))
    if header is not None:
        yield header, "".join(seq)


def main():
    src, n, seed, out_faa, out_ids = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5]
    total = 0
    with open(src) as fh:
        for line in fh:
            if line.startswith(">"):
                total += 1
    chosen = set(random.Random(seed).sample(range(total), n))
    written = 0
    residues = 0
    with open(out_faa, "w") as fo, open(out_ids, "w") as fi:
        for i, (h, s) in enumerate(records(src)):
            if i in chosen:
                fo.write(f"{h}\n{s}\n")
                fi.write(h[1:].split()[0] + "\n")
                written += 1
                residues += len(s)
    print(f"source_records {total} sampled {written} residues {residues} seed {seed}")


if __name__ == "__main__":
    main()
