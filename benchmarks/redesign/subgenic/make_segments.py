#!/usr/bin/env python3
"""Inputs for the internal-repeat (self-alignment) measurement.

Parents: giants (g<uid>), length controls of 1,000-2,045 aa (c<uid>), and giants with residues
shuffled inside consecutive 25-aa windows (s<uid>; keeps local composition, destroys repeat order).
Each parent is cut into 200-aa segments at step 100 (last segment flush with the C-terminus).
Every segment is paired with its own parent only, through a fake prefilter DB, and aligned with
--alt-ali: the first alignment is the segment's own location, which alt-ali then masks, so the
following alignments are internal repeats.
Keys: createdb --shuffle 0 assigns keys in input order (asserted by the wrapper through .lookup).
"""
import argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--work", required=True)
ap.add_argument("--win", type=int, default=200)
ap.add_argument("--step", type=int, default=100)
ap.add_argument("--shuffle-win", type=int, default=25)
ap.add_argument("--seed", type=int, default=17)
a = ap.parse_args()
W = a.work
giants = set(np.loadtxt(f"{W}/lists/giants.uid", dtype=np.int64).tolist())
lenctl = set(np.loadtxt(f"{W}/lists/lenctl.uid", dtype=np.int64).tolist())
need = giants | lenctl
seqs = {}
with open(f"{W}/uniq.faa") as fh:
    for h, s in zip(fh, fh):
        u = int(h[1:])
        if u in need:
            seqs[u] = s.strip()
rng = np.random.default_rng(a.seed)

def wshuffle(s):
    arr = np.frombuffer(s.encode(), dtype=np.uint8).copy()
    for i in range(0, len(arr), a.shuffle_win):
        seg = arr[i:i + a.shuffle_win]; rng.shuffle(seg); arr[i:i + a.shuffle_win] = seg
    return arr.tobytes().decode()

parents = [(f"g{u}", seqs[u]) for u in sorted(giants)]
parents += [(f"c{u}", seqs[u]) for u in sorted(lenctl)]
parents += [(f"s{u}", wshuffle(seqs[u])) for u in sorted(giants)]
D = f"{W}/rep"
nseg = 0
with open(f"{D}/parents.faa", "w") as pf, open(f"{D}/segments.faa", "w") as sf, open(f"{D}/pref.tsv", "w") as tf, \
        open(f"{D}/parents.len", "w") as lf:
    for pk, (name, s) in enumerate(parents):
        pf.write(f">{name}\n{s}\n"); lf.write(f"{name}\t{len(s)}\n")
        L = len(s)
        starts = list(range(0, max(L - a.win, 0) + 1, a.step))
        if starts[-1] + a.win < L:
            starts.append(L - a.win)
        for st in starts:
            sf.write(f">{name}_{st}\n{s[st:st + a.win]}\n")
            tf.write(f"{nseg}\t{pk}\t0\t0\n")
            nseg += 1
print(f"parents {len(parents)} segments {nseg}")
