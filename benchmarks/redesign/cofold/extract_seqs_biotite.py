"""Single pass over $RD/ws_dpann/work/proteins_gid.faa (header = gid) keeping the gids listed in argv[1].
Runs on the biotite login node (one thread, one read of the file)."""
import sys
want = set(int(x) for x in open(sys.argv[1]) if x.strip())
out = open(sys.argv[3], "w")
keep, n = False, 0
with open(sys.argv[2]) as f:
    for line in f:
        if line[0] == ">":
            keep = int(line[1:]) in want
            n += keep
        if keep:
            out.write(line)
out.close()
print(f"kept {n} of {len(want)}", file=sys.stderr)
