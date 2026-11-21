"""Shared paths and the token-file writer for the ws_tokens workstream."""
import datetime as dt
import fcntl
import hashlib
import json
import os
import resource
import socket
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/spacedust")
RD = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111")
S = ROOT / "seq_invisible_20261111"
SCALE = ROOT / "model_scaling_20260913"
N = ROOT / "native_run_20260910"
WS = RD / "ws_tokens"
TOK = RD / "tokens"
UNIVERSE = RD / "universe/genes.parquet"
NREF = 4190383
NQUERY = 67148
NGENES = NREF + NQUERY
CODE = Path(__file__).resolve().parent


def code_hash(files=None):
    """sha256 over the workstream code files (sorted), so a manifest pins the code.
    The files are also copied to $RD/ws_tokens/code_snapshots/<hash>/ the first time."""
    files = sorted(files or list(CODE.glob("*.py")) + list(CODE.glob("*.sh")) + list(CODE.glob("*.sbatch")))
    h = hashlib.sha256()
    for f in files:
        h.update(f.name.encode())
        h.update(f.read_bytes())
    d = h.hexdigest()
    snap = WS / "code_snapshots" / d
    if not snap.exists():
        tmp = WS / "code_snapshots" / f".{d}.tmp{os.getpid()}"
        tmp.mkdir(parents=True, exist_ok=True)
        for f in files:
            (tmp / f.name).write_bytes(f.read_bytes())
        try:
            os.replace(tmp, snap)
        except OSError:
            pass
    return d


def peak_rss_gb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    c = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    return round(max(r, c) / 1024 / 1024, 2)  # Linux reports KiB


def write_tokens(arm, tokens, manifest, seconds):
    """Write $RD/tokens/<arm>.parquet (gid, token) atomically, its manifest, and an INDEX line."""
    tokens = np.asarray(tokens, dtype=np.int64)
    assert tokens.shape == (NGENES,), tokens.shape
    assert tokens.min() >= -1
    TOK.mkdir(parents=True, exist_ok=True)
    gid = np.arange(NGENES, dtype=np.int64)
    tab = pa.table({"gid": gid, "token": tokens})
    final = TOK / f"{arm}.parquet"
    tmp = TOK / f".{arm}.parquet.tmp{os.getpid()}"
    pq.write_table(tab, tmp, compression="zstd")
    back = pq.read_table(tmp)
    assert back.num_rows == NGENES and np.array_equal(back.column("token").to_numpy(), tokens)
    os.replace(tmp, final)
    n_assigned = int((tokens >= 0).sum())
    n_tokens = int(np.unique(tokens[tokens >= 0]).size)
    man = dict(manifest)
    man.update(arm=arm, rows=NGENES, n_assigned=n_assigned, n_unassigned=NGENES - n_assigned,
               n_tokens=n_tokens, seconds=round(seconds, 1), peak_rss_gb=peak_rss_gb(),
               cpus=int(os.environ.get("SLURM_CPUS_ON_NODE", "0")), host=socket.gethostname(),
               slurm_job=os.environ.get("SLURM_JOB_ID"), code_sha256=code_hash(),
               written=dt.datetime.now().isoformat(timespec="seconds"),
               token_file_sha256=hashlib.sha256(final.read_bytes()).hexdigest())
    jt = TOK / f".{arm}.json.tmp{os.getpid()}"
    with open(jt, "w") as fh:
        json.dump(man, fh, indent=1, default=str)
    os.replace(jt, TOK / f"{arm}.json")
    line = f"{arm}\t{n_tokens}\t{n_assigned}\t{round(seconds, 1)}\t{dt.date.today().isoformat()}\n"
    idx = TOK / "INDEX.tsv"
    with open(idx, "a+") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        fh.seek(0)
        if fh.read() == "":
            fh.write("arm\tn_tokens\tn_assigned\tseconds\tdate\n")
        fh.write(line)
        fcntl.flock(fh, fcntl.LOCK_UN)
    print(f"WROTE {arm}: tokens={n_tokens} assigned={n_assigned} seconds={seconds:.1f}", flush=True)
    return man


def stamp(*a):
    print(dt.datetime.now().isoformat(timespec="seconds"), *a, flush=True)
