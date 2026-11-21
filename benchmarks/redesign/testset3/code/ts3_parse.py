#!/usr/bin/env python3
"""Parse the GTDB r232 Prodigal proteomes of the testset3 selection into Parquet parts (from slice_lut 01_parse_slice.py).

usage: ts3_parse.py SELECTION_TSV OUT_DIR WORKERS
Columns per protein: gtdb_id, genome, domain, prodigal_id, contig, prodigal_n, file_index, start, end, strand, partial,
start_type, rbs_motif, gc_cont, length (aa, trailing * stripped), internal_stops, seq. Writes OUT_DIR/parts/*.parquet and
OUT_DIR/genomes_src.tsv (gtdb_id, genome, domain, n_proteins).
"""
import gzip, os, sys, time
from multiprocessing import Pool
import pandas as pd, pyarrow as pa, pyarrow.parquet as pq

SEL, OUT, NW = sys.argv[1], sys.argv[2], int(sys.argv[3])
G = "/groups/banfield/projects/environmental/sr/srvp2020/Jacob/giant_protein_folding/rp16_phylogeny_2026_06/gtdb_giant_scan/reps/protein_faa_reps"
PDIR = {"Bacteria": f"{G}/bacteria", "Archaea": f"{G}/archaea"}
os.makedirs(f"{OUT}/parts", exist_ok=True)
S = pd.read_csv(SEL, sep="\t", usecols=["accession", "domain"]).sort_values("accession")
jobs_in = list(zip(S.accession, S.domain))
SCHEMA = pa.schema([("gtdb_id", pa.string()), ("genome", pa.string()), ("domain", pa.string()), ("prodigal_id", pa.string()),
                    ("contig", pa.string()), ("prodigal_n", pa.int32()), ("file_index", pa.int32()), ("start", pa.int64()),
                    ("end", pa.int64()), ("strand", pa.int8()), ("partial", pa.string()), ("start_type", pa.string()),
                    ("rbs_motif", pa.string()), ("gc_cont", pa.float32()), ("length", pa.int32()), ("internal_stops", pa.int32()),
                    ("seq", pa.string())])


def parse(job):
    k, sub = job
    cols = {f.name: [] for f in SCHEMA}
    stats = []
    for gid, dom in sub:
        n = 0
        with gzip.open(f"{PDIR[dom]}/{gid}_protein.faa.gz", "rt") as fh:
            hdr, chunks = None, []
            def flush():
                s = "".join(chunks)
                if s.endswith("*"):
                    s = s[:-1]
                f = [x.strip() for x in hdr.split("#")]
                contig, pn = f[0].rsplit("_", 1)
                kv = dict(x.split("=", 1) for x in f[4].split(";") if "=" in x)
                for c, v in [("gtdb_id", gid), ("genome", gid[3:]), ("domain", dom), ("prodigal_id", f[0]), ("contig", contig),
                             ("prodigal_n", int(pn)), ("file_index", n), ("start", int(f[1])), ("end", int(f[2])),
                             ("strand", int(f[3])), ("partial", kv.get("partial", "")), ("start_type", kv.get("start_type", "")),
                             ("rbs_motif", kv.get("rbs_motif", "")), ("gc_cont", float(kv.get("gc_cont", "nan"))),
                             ("length", len(s)), ("internal_stops", s.count("*")), ("seq", s)]:
                    cols[c].append(v)
            for line in fh:
                if line.startswith(">"):
                    if hdr is not None:
                        flush(); n += 1
                    hdr, chunks = line[1:].rstrip("\n"), []
                else:
                    chunks.append(line.strip())
            if hdr is not None:
                flush(); n += 1
        stats.append((gid, gid[3:], dom, n))
    pq.write_table(pa.table(cols, schema=SCHEMA), f"{OUT}/parts/part_{k:04d}.parquet", compression="zstd")
    return stats


t0 = time.time()
B = max(1, len(jobs_in) // (NW * 4))
jobs = [(k, jobs_in[i:i + B]) for k, i in enumerate(range(0, len(jobs_in), B))]
with Pool(NW) as pool:
    res = pool.map(parse, jobs, chunksize=1)
pd.DataFrame([r for st in res for r in st], columns=["gtdb_id", "genome", "domain", "n_proteins"]).to_csv(
    f"{OUT}/genomes_src.tsv", sep="\t", index=False)
print(f"parsed {sum(len(s) for s in res)} genomes, {sum(r[3] for s in res for r in s)} proteins, {len(jobs)} parts, "
      f"{time.time() - t0:.0f} s", flush=True)
