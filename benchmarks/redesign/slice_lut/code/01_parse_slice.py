#!/usr/bin/env python3
"""Parse GTDB r232 species-representative proteomes of one taxon into Parquet parts (workstream slice_lut).

usage: 01_parse_slice.py SP_CLUSTERS PROTEOME_DIR RANK TAXON OUT_DIR WORKERS [EXCLUDE_RANK EXCLUDE_TAXON]
  RANK   one of phylum class order family genus (GTDB rank of TAXON, e.g. order Rhizobiales), or list
  TAXON  bare GTDB name, e.g. Rhizobiales; with RANK list, a file of GTDB representative IDs (RS_/GB_ prefixed)
  EXCLUDE_RANK EXCLUDE_TAXON  optional: drop representatives in this taxon (e.g. order Rhizobiales)
Writes OUT_DIR/parts/part_NNNN.parquet (one row per protein) and OUT_DIR/genomes_src.tsv.
Columns: gtdb_id (RS_/GB_ prefixed), genome (bare accession), prodigal_id, contig, prodigal_n, file_index (0-based order in
the GTDB file), start, end, strand (+1/-1), partial, start_type, rbs_motif, gc_cont, length (aa, trailing * stripped),
internal_stops, seq.
"""
import gzip, os, sys, time
from multiprocessing import Pool
import pyarrow as pa, pyarrow.parquet as pq

SPC, PDIR, RANK, TAXON, OUT, NW = sys.argv[1:7]
NW = int(NW)
RK = dict(phylum=1, class_=2, order=3, family=4, genus=5)
LIST = set(l.strip() for l in open(TAXON) if l.strip()) if RANK == "list" else None
IDX = None if LIST is not None else RK[RANK if RANK != "class" else "class_"]
PFX = None if LIST is not None else "pcofg"[IDX - 1] + "__"
EX = None
if len(sys.argv) > 8:
    EI = RK[sys.argv[7] if sys.argv[7] != "class" else "class_"]
    EX = (EI, "pcofg"[EI - 1] + "__" + sys.argv[8])
os.makedirs(f"{OUT}/parts", exist_ok=True)

reps = []
for line in open(SPC):
    if line.startswith("Representative"):
        continue
    f = line.rstrip("\n").split("\t")
    tax = f[2].split(";")
    if tax[0] != "d__Bacteria":
        continue
    if EX is not None and tax[EX[0]] == EX[1]:
        continue
    if (LIST is not None and f[0] in LIST) or (LIST is None and tax[IDX] == PFX + TAXON):
        reps.append((f[0], f[2] + ";" + f[1]))
reps.sort()
print(f"{len(reps)} species representatives in {TAXON if LIST is None else 'list'} (exclude {EX})", flush=True)


def parse(job):
    k, sub = job
    cols = {c: [] for c in ["gtdb_id", "genome", "prodigal_id", "contig", "prodigal_n", "file_index", "start", "end",
                            "strand", "partial", "start_type", "rbs_motif", "gc_cont", "length", "internal_stops", "seq"]}
    stats = []
    for gid, _tax in sub:
        path = f"{PDIR}/{gid}_protein.faa.gz"
        genome = gid[3:]
        n = 0
        hdr, chunks = None, []

        def flush():
            s = "".join(chunks)
            if s.endswith("*"):
                s = s[:-1]
            f = [x.strip() for x in hdr.split("#")]
            pid = f[0]
            contig, pn = pid.rsplit("_", 1)
            kv = dict(x.split("=", 1) for x in f[4].split(";") if "=" in x)
            cols["gtdb_id"].append(gid); cols["genome"].append(genome); cols["prodigal_id"].append(pid)
            cols["contig"].append(contig); cols["prodigal_n"].append(int(pn)); cols["file_index"].append(len(cols["seq"]) - base)
            cols["start"].append(int(f[1])); cols["end"].append(int(f[2])); cols["strand"].append(int(f[3]))
            cols["partial"].append(kv.get("partial", "")); cols["start_type"].append(kv.get("start_type", ""))
            cols["rbs_motif"].append(kv.get("rbs_motif", "")); cols["gc_cont"].append(float(kv.get("gc_cont", "nan")))
            cols["length"].append(len(s)); cols["internal_stops"].append(s.count("*")); cols["seq"].append(s)

        base = len(cols["seq"])
        with gzip.open(path, "rt") as fh:
            for line in fh:
                if line.startswith(">"):
                    if hdr is not None:
                        flush(); n += 1
                    hdr, chunks = line[1:].rstrip("\n"), []
                else:
                    chunks.append(line.strip())
            if hdr is not None:
                flush(); n += 1
        stats.append((gid, n))
    t = pa.table({c: v for c, v in cols.items()})
    t = t.cast(pa.schema([("gtdb_id", pa.string()), ("genome", pa.string()), ("prodigal_id", pa.string()),
                          ("contig", pa.string()), ("prodigal_n", pa.int32()), ("file_index", pa.int32()),
                          ("start", pa.int64()), ("end", pa.int64()), ("strand", pa.int8()), ("partial", pa.string()),
                          ("start_type", pa.string()), ("rbs_motif", pa.string()), ("gc_cont", pa.float32()),
                          ("length", pa.int32()), ("internal_stops", pa.int32()), ("seq", pa.string())]))
    pq.write_table(t, f"{OUT}/parts/part_{k:04d}.parquet", compression="zstd")
    return stats


t0 = time.time()
B = max(1, len(reps) // (NW * 4))
jobs = [(k, reps[i:i + B]) for k, i in enumerate(range(0, len(reps), B))]
with Pool(NW) as pool:
    res = pool.map(parse, jobs, chunksize=1)
with open(f"{OUT}/genomes_src.tsv", "w") as f:
    f.write("gtdb_id\tgenome\ttax\tn_proteins\n")
    tax = dict(reps)
    for st in res:
        for gid, n in st:
            f.write(f"{gid}\t{gid[3:]}\t{tax[gid]}\t{n}\n")
print(f"parsed {sum(len(s) for s in res)} genomes, {sum(n for s in res for _, n in s)} proteins, "
      f"{len(jobs)} parts, {time.time() - t0:.0f} s", flush=True)
