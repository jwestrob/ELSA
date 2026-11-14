#!/usr/bin/env python
"""embed_queue: a resumable protein-embedding queue served by independent Slurm workers.

A run directory holds length-sorted shards of unique sequences and a filesystem queue.
Any number of workers (any GPU type, any start time) claim shards by atomic rename, embed
them and write one matrix per shard. A worker that dies leaves a claim whose heartbeat goes
stale; the next worker returns it to the queue. `finalize` assembles the row-aligned matrix.

Subcommands
  prep      FASTA -> run directory (normalize, deduplicate, length-sort, shard, queue)
  work      claim shards and embed until the queue is drained or the wall limit is near
  status    queue counts, throughput, workers
  reap      return stale claims to the queue
  finalize  assemble, QC, assembly check, aggregate the per-shard audits, manifest
  compare   row-wise cosine between the unique-sequence matrices of two runs

Pooling convention (same as the ELSA redesign vectors): final hidden state after the last
layer norm, arithmetic mean over residue positions only (<cls>, <eos> and padding excluded),
accumulated in float32. Shards store the raw mean; `finalize` also writes the L2-normalized
row-aligned matrix.

Layout of a run directory
  run.json                       identity of the source, model and policy
  shards/shard_00000.faa         >uid / sequence, ascending length
  shards/shard_00000.uids.npy    unique-sequence ids in shard order
  shards.tsv                     shard, n, tokens, min_len, max_len
  uid_of_row.npy                 int32, unique id of each input row (-1 = empty sequence)
  length_of_uid.npy              int32
  selfcheck.faa                  mixed-length proteins for the precision / padding check
  queue/{pending,claimed,done}/  one token file per shard
  out/shard_00000.npy (+ .json)  float32 [n, D] raw residue means and the receipt
  workers/<worker>.json          heartbeat and progress
  final/                         products of `finalize`
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ALPHABET = b"ACDEFGHIKLMNPQRSTVWYXBUZO"  # the 25 residue letters of the ESM-C / ESM++ vocabulary
_TABLE = bytearray(b"X" * 256)
for _c in ALPHABET:
    _TABLE[_c] = _c
    _TABLE[ord(chr(_c).lower())] = _c
_TABLE = bytes(_TABLE)
_ALLOWED_BOTH = ALPHABET + ALPHABET.lower()


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def log(*a) -> None:
    print(f"[{now()}]", *a, flush=True)


def write_json(path: Path, obj) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str) + "\n")
    os.replace(tmp, path)


def sha256_file(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def shard_name(i: int) -> str:
    return f"shard_{i:05d}"


# --------------------------------------------------------------------------------------
# prep
# --------------------------------------------------------------------------------------
def read_fasta(path: Path):
    ids, seqs, parts, started = [], [], [], False
    with open(path, "rb") as f:
        for line in f:
            if line[:1] == b">":
                if started:
                    seqs.append(b"".join(parts))
                ids.append(line[1:].split(None, 1)[0].decode() if len(line) > 2 else "")
                parts, started = [], True
            elif started:
                parts.append(line.strip())
    if started:
        seqs.append(b"".join(parts))
    return ids, seqs


def normalize(seq: bytes, subs: Counter) -> bytes:
    """Uppercase, drop terminal stop characters, map every symbol outside the vocabulary to X."""
    seq = seq.rstrip(b"*")
    odd = seq.translate(None, _ALLOWED_BOTH)
    if odd:
        subs.update(odd.decode("latin-1"))
    return seq.translate(_TABLE)


def cmd_prep(a) -> None:
    run = Path(a.run)
    if (run / "run.json").exists():
        sys.exit(f"{run}/run.json exists; a prepared run is never rebuilt in place")
    t0 = time.time()
    ids, raw = read_fasta(Path(a.fasta))
    log(f"read {len(raw):,} records from {a.fasta} in {time.time() - t0:.0f}s")
    ids_are_row_index = all(s == str(i) for i, s in enumerate(ids))
    keep = np.arange(len(raw))
    if a.sample:
        rng = np.random.default_rng(a.seed)
        ok = np.array([0 < len(s.rstrip(b"*")) <= a.sample_max_len for s in raw])
        keep = np.sort(rng.choice(np.flatnonzero(ok), size=min(a.sample, int(ok.sum())), replace=False))
        if a.sample_duplicates:  # exercise the duplicate-sequence path
            keep = np.concatenate([keep, keep[: a.sample_duplicates]])
    subs: Counter = Counter()
    uid_of: dict = {}
    uniq: list = []
    uid_of_row = np.full(len(keep), -1, dtype=np.int32)
    n_empty = 0
    for r, i in enumerate(keep):
        s = normalize(raw[i], subs)
        if not s:
            n_empty += 1
            continue
        u = uid_of.get(s)
        if u is None:
            u = len(uniq)
            uid_of[s] = u
            uniq.append(s)
        uid_of_row[r] = u
    del uid_of, raw
    L = np.fromiter((len(s) for s in uniq), dtype=np.int32, count=len(uniq))
    order = np.argsort(L, kind="stable")
    log(f"{len(keep):,} rows, {len(uniq):,} unique sequences, {int(L.sum()):,} unique residues, {n_empty} empty")

    for d in ("shards", "queue/pending", "queue/claimed", "queue/done", "out", "workers", "logs", "final"):
        (run / d).mkdir(parents=True, exist_ok=True)
    rows, start, tok, i = [], 0, 0, 0
    bounds = []
    for j, u in enumerate(order):
        t = int(L[u]) + 2
        if j > start and tok + t > a.shard_tokens:
            bounds.append((start, j, tok))
            start, tok = j, 0
        tok += t
    if len(order) > start:
        bounds.append((start, len(order), tok))
    for i, (s, e, tok) in enumerate(bounds):
        us = order[s:e]
        with open(run / "shards" / f"{shard_name(i)}.faa", "wb") as f:
            f.write(b"".join(b">%d\n%s\n" % (u, uniq[u]) for u in us))
        np.save(run / "shards" / f"{shard_name(i)}.uids.npy", us.astype(np.int32))
        rows.append((shard_name(i), len(us), tok, int(L[us[0]]), int(L[us[-1]])))
    with open(run / "shards.tsv", "w") as f:
        f.write("shard\tn\ttokens\tmin_len\tmax_len\n")
        f.writelines("\t".join(map(str, r)) + "\n" for r in rows)
    np.save(run / "uid_of_row.npy", uid_of_row)
    np.save(run / "length_of_uid.npy", L)
    if not ids_are_row_index or a.sample:
        (run / "row_ids.txt").write_text("\n".join(ids[i] for i in keep) + "\n")
    # self-check set: unique sequences at evenly spaced length quantiles, bounded length
    cand = order[L[order] <= a.selfcheck_max_len]
    pick = cand[np.unique((np.linspace(0.02, 0.995, a.selfcheck_n) * (len(cand) - 1)).astype(int))] if len(cand) else []
    with open(run / "selfcheck.faa", "wb") as f:
        f.write(b"".join(b">%d\n%s\n" % (u, uniq[u]) for u in pick))
    q = [0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999, 1]
    meta = dict(
        name=run.name, prepared=now(), source_fasta=str(Path(a.fasta).resolve()),
        source_sha256=sha256_file(Path(a.fasta)), source_records=len(ids), ids_are_row_index=ids_are_row_index,
        sample=a.sample or None, seed=a.seed if a.sample else None,
        rows=int(len(keep)), empty_rows=n_empty, unique_sequences=len(uniq),
        duplicate_rows=int(len(keep) - n_empty - len(uniq)),
        residues_rows=int(L[uid_of_row[uid_of_row >= 0]].sum()), residues_unique=int(L.sum()),
        tokens_unique=int(L.sum() + 2 * len(L)), shards=len(rows), shard_tokens=a.shard_tokens,
        length_quantiles={str(x): int(np.quantile(L, x)) for x in q} if len(L) else {},
        unique_longer_than_2046=int((L > 2046).sum()),
        substituted_symbols=dict(subs), substituted_total=int(sum(subs.values())),
        normalization="uppercase; terminal '*' removed; every symbol outside ACDEFGHIKLMNPQRSTVWYXBUZO -> X",
        backend=a.backend, model_dir=a.model_dir, model_id=a.model_id, model_revision=a.model_revision,
        dtype=a.dtype,
        pooling="final hidden state (after the last layer norm); arithmetic mean over residue positions only; "
                "float32 accumulation; shards store the raw mean",
        long_sequences="whole protein in a single pass, no truncation or windowing",
        selfcheck_proteins=int(len(pick)),
    )
    if a.model_dir and Path(a.model_dir).is_dir():
        meta["model_files_sha256"] = {
            p.name: sha256_file(p) for p in sorted(Path(a.model_dir).iterdir())
            if p.is_file() and p.suffix in (".json", ".py", ".safetensors")
        }
    for r in rows:
        (run / "queue" / "pending" / r[0]).touch()
    write_json(run / "run.json", meta)
    log(f"prepared {run}: {len(rows)} shards in {time.time() - t0:.0f}s")
    print(json.dumps({k: meta[k] for k in ("rows", "unique_sequences", "duplicate_rows", "residues_unique", "shards",
                                           "length_quantiles", "unique_longer_than_2046", "substituted_symbols")}, indent=2))


# --------------------------------------------------------------------------------------
# backends
# --------------------------------------------------------------------------------------
class FakeBackend:
    """Residue composition (25-d). Batch-invariant by construction; exercises the queue with no model."""
    dim = 25
    device = "cpu"
    info = dict(backend="fake", device="cpu", gpu=None, dtype="fp32")
    token_budget, square_budget = 20_000, 10 ** 12

    def __init__(self, delay: float = 0.0):
        self.delay = delay
        self.lut = np.full(256, -1, dtype=np.int64)
        for k, c in enumerate(ALPHABET):
            self.lut[c] = k

    def embed(self, seqs, reference: bool = False) -> np.ndarray:
        if self.delay:
            time.sleep(self.delay)
        out = np.zeros((len(seqs), self.dim), dtype=np.float32)
        for i, s in enumerate(seqs):
            out[i] = np.bincount(self.lut[np.frombuffer(s, dtype=np.uint8)], minlength=self.dim) / len(s)
        return out

    @staticmethod
    def is_oom(e) -> bool:
        return False


class TorchBackend:
    """ESM++ (Synthyra remote code) or native transformers ESM-C; ids from a byte lookup table."""

    def __init__(self, meta: dict, device: str, threads: int, backend: str | None = None, model_dir: str | None = None,
                 dtype: str | None = None, token_budget="auto", square_budget="auto"):
        import torch
        import transformers
        self.torch = torch
        torch.set_num_threads(max(1, threads))
        backend = backend or meta["backend"]
        model_dir = model_dir or meta["model_dir"]
        self.policy = dtype or meta["dtype"]
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        t0 = time.time()
        if backend == "esmpp":
            model = transformers.AutoModel.from_pretrained(model_dir, trust_remote_code=True, local_files_only=True)
            tok = model.tokenizer
        elif backend == "esmc_native":
            cls = getattr(transformers, "ESMCModel", None) or getattr(transformers, "EsmcModel", None) or transformers.AutoModel
            tcls = getattr(transformers, "ESMCTokenizer", None) or transformers.AutoTokenizer
            model = cls.from_pretrained(model_dir, local_files_only=True)
            tok = tcls.from_pretrained(model_dir, local_files_only=True)
        else:
            raise SystemExit(f"unknown backend {backend}")
        # The ESM++ module switches TF32 on at import. Keep float32 passes at full precision.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        self.model = model.eval().to(device)
        self.dim = int(getattr(model.config, "hidden_size", 0) or getattr(model.config, "d_model"))
        self.cls, self.eos, self.pad = int(tok.cls_token_id), int(tok.eos_token_id), int(tok.pad_token_id)
        self.lut = np.full(256, -1, dtype=np.int64)
        unk = tok.unk_token_id
        for c in ALPHABET:
            k = tok.convert_tokens_to_ids(chr(c))
            assert k is not None and k != unk and k not in (self.cls, self.eos, self.pad), f"letter {chr(c)} has no token"
            self.lut[c] = k
        for probe in (ALPHABET, b"MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQXBUZO", b"M"):
            want = tok(probe.decode())["input_ids"]
            got, _, _ = self._encode([probe])
            assert list(got[0]) == list(want), f"lookup-table ids differ from the tokenizer for {probe!r}"
        gpu = None
        self.autocast = False
        total_gb = 0.0
        if device == "cuda":
            p = torch.cuda.get_device_properties(0)
            total_gb = p.total_memory / 2 ** 30
            pci = subprocess.run(["nvidia-smi", "--query-gpu=pci.bus_id", "--format=csv,noheader"],
                                 capture_output=True, text=True).stdout.strip().splitlines()
            gpu = dict(name=p.name, memory_gb=round(total_gb, 1), pci=pci[0].strip() if pci else None)
            self.autocast = self.policy == "bf16"
        self.token_budget = int(token_budget) if token_budget != "auto" else (
            min(250_000, int(total_gb * 4000)) if device == "cuda" else 8_000)
        self.square_budget = int(square_budget) if square_budget != "auto" else (
            int(total_gb * 1.4e7) if device == "cuda" else 60_000_000)
        self.info = dict(backend=backend, model_dir=model_dir, device=device, gpu=gpu,
                         dtype=("bf16" if self.autocast else "fp32"), policy=self.policy, torch=torch.__version__,
                         transformers=transformers.__version__, threads=threads, host=socket.gethostname(),
                         token_budget=self.token_budget, square_budget=self.square_budget,
                         model_load_seconds=round(time.time() - t0, 1), dim=self.dim)

    def _encode(self, seqs):
        L = np.fromiter((len(s) for s in seqs), dtype=np.int64, count=len(seqs))
        B, M = len(seqs), int(L.max())
        tok = self.lut[np.frombuffer(b"".join(seqs), dtype=np.uint8)]
        assert tok.min() >= 0, "sequence contains a symbol outside the vocabulary (run prep normalization)"
        ids = np.full((B, M + 2), self.pad, dtype=np.int64)
        ids[:, 0] = self.cls
        ids[:, 1:M + 1][np.arange(M)[None, :] < L[:, None]] = tok
        ids[np.arange(B), L + 1] = self.eos
        att = np.arange(M + 2)[None, :] < (L + 2)[:, None]
        return ids, att, L

    def embed(self, seqs, reference: bool = False) -> np.ndarray:
        """Residue-only mean, float32. reference=True forces float32 inference."""
        torch = self.torch
        ids, att, L = self._encode(seqs)
        ids_t = torch.from_numpy(ids).to(self.device, non_blocking=True)
        att_t = torch.from_numpy(att).to(self.device, non_blocking=True)
        L_t = torch.from_numpy(L).to(self.device)
        with torch.inference_mode():
            if self.autocast and not reference:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    h = self.model(input_ids=ids_t, attention_mask=att_t).last_hidden_state
            else:
                h = self.model(input_ids=ids_t, attention_mask=att_t).last_hidden_state
            pos = torch.arange(ids_t.shape[1], device=self.device)[None, :]
            keep = (pos >= 1) & (pos <= L_t[:, None])
            assert bool((keep.sum(1) == L_t).all())
            v = (h.float() * keep[..., None]).sum(1) / L_t[:, None].float()
        return v.cpu().numpy().astype(np.float32, copy=False)

    def is_oom(self, e) -> bool:
        torch = self.torch
        kinds = tuple(k for k in (getattr(torch, "OutOfMemoryError", None), getattr(torch.cuda, "OutOfMemoryError", None)) if k)
        if isinstance(e, kinds) or (isinstance(e, RuntimeError) and "out of memory" in str(e).lower()):
            if self.device == "cuda":
                torch.cuda.empty_cache()
            return True
        return False


def make_backend(meta, a, **kw):
    backend = getattr(a, "backend", None) or meta["backend"]
    if backend == "fake":
        return FakeBackend(delay=getattr(a, "fake_delay", 0.0))
    return TorchBackend(meta, device=a.device, threads=a.threads, backend=backend, **kw)


def read_shard(path: Path):
    uids, seqs = [], []
    with open(path, "rb") as f:
        for line in f:
            if line[:1] == b">":
                uids.append(int(line[1:]))
            else:
                seqs.append(line.rstrip(b"\n"))
    return np.asarray(uids, dtype=np.int64), seqs


def cos_rows(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A = A.astype(np.float64)
    B = B.astype(np.float64)
    return (A * B).sum(1) / (np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1))


def batches(lengths, token_budget: int, square_budget: int):
    """Consecutive batches over ascending lengths, bounded in padded tokens and in B * T^2."""
    b, n = 0, len(lengths)
    while b < n:
        e = b + 1
        while e < n:
            t = int(lengths[e]) + 2
            if (e - b + 1) * t > token_budget or (e - b + 1) * t * t > square_budget:
                break
            e += 1
        yield b, e
        b = e


def selfcheck(backend, run: Path, min_cos: float) -> dict:
    """Float32 single-protein embeddings against one padded mixed-length batch in the run dtype."""
    _, seqs = read_shard(run / "selfcheck.faa")
    if not seqs:
        return dict(n=0)
    t0 = time.time()
    ref = np.vstack([backend.embed([s], reference=True) for s in seqs])
    got = np.vstack([backend.embed(seqs[b:e]) for b, e in ((i, min(i + 16, len(seqs))) for i in range(0, len(seqs), 16))])
    c = cos_rows(ref, got)
    rel = np.linalg.norm(ref - got, axis=1) / np.linalg.norm(ref, axis=1)
    res = dict(n=len(seqs), lengths=[min(map(len, seqs)), max(map(len, seqs))], min_cosine=float(c.min()),
               median_cosine=float(np.median(c)), max_relative_l2=float(rel.max()), gate=min_cos,
               passed=bool(c.min() >= min_cos), seconds=round(time.time() - t0, 1),
               compared="float32 single protein vs padded batches of 16 in the run dtype")
    return res


# --------------------------------------------------------------------------------------
# queue
# --------------------------------------------------------------------------------------
class Queue:
    def __init__(self, run: Path, wid: str):
        self.run, self.wid = run, wid
        self.pending, self.claimed, self.done = (run / "queue" / d for d in ("pending", "claimed", "done"))

    def claim(self, order: str):
        names = sorted(os.listdir(self.pending), reverse=(order == "desc"))
        for nm in names:
            if not nm.startswith("shard_"):
                continue
            dst = self.claimed / f"{nm}@{self.wid}"
            try:
                os.rename(self.pending / nm, dst)
            except FileNotFoundError:
                if not dst.exists():  # another worker took it
                    continue
            os.utime(dst, None)
            return nm, dst
        return None, None

    def release(self, nm: str, claim: Path) -> None:
        try:
            os.rename(claim, self.pending / nm)
        except FileNotFoundError:
            pass

    def finish(self, nm: str, claim: Path) -> None:
        for src in (claim, self.pending / nm):
            try:
                os.rename(src, self.done / nm)
                return
            except FileNotFoundError:
                continue
        if not (self.done / nm).exists():  # reaped and re-claimed elsewhere; the output is already complete
            (self.done / nm).touch()

    def reap(self, stale: float) -> list:
        back = []
        for f in os.listdir(self.claimed):
            nm, _, owner = f.partition("@")
            p = self.claimed / f
            try:
                age = time.time() - p.stat().st_mtime
            except FileNotFoundError:
                continue
            if owner == self.wid or age < stale:
                continue
            job = owner.split("_")[0]
            if job.isdigit() and age < 3 * stale:
                r = subprocess.run(["squeue", "-h", "-j", job, "-o", "%T"], capture_output=True, text=True)
                if r.returncode == 0 and r.stdout.strip() == "RUNNING":
                    continue
            try:
                os.rename(p, self.pending / nm)
                back.append((nm, owner, round(age)))
            except FileNotFoundError:
                pass
        return back

    def counts(self) -> dict:
        return {d.name: sum(1 for x in os.listdir(d) if x.startswith("shard_")) for d in (self.pending, self.claimed, self.done)}


def cmd_work(a) -> None:
    run = Path(a.run)
    meta = json.loads((run / "run.json").read_text())
    wid = f"{os.environ.get('SLURM_JOB_ID', 'local')}_{socket.gethostname().split('.')[0]}_{os.getpid()}"
    q = Queue(run, wid)
    stop = threading.Event()
    for s in (signal.SIGTERM, signal.SIGINT):
        signal.signal(s, lambda *_: stop.set())
    end = float(os.environ.get("SLURM_JOB_END_TIME", 0) or 0)
    backend = make_backend(meta, a, token_budget=a.token_budget, square_budget=a.square_budget)
    log(f"worker {wid} backend {json.dumps(backend.info)}")
    state = dict(worker=wid, started=now(), backend=backend.info, shards_done=0, residues=0, seconds=0.0,
                 current=None, order=a.order, selfcheck=None, finished=None, exit=None)
    wfile = run / "workers" / f"{wid}.json"
    if a.selfcheck:
        sc = selfcheck(backend, run, a.selfcheck_min_cos)
        state["selfcheck"] = sc
        log(f"self-check {json.dumps(sc)}")
        if sc.get("n") and not sc["passed"]:
            state.update(finished=now(), exit="selfcheck_failed")
            write_json(wfile, state)
            sys.exit(3)
    write_json(wfile, state)
    current = {"claim": None}

    def beat():
        while not stop.wait(20):
            c = current["claim"]
            if c is not None:
                try:
                    os.utime(c, None)
                except FileNotFoundError:
                    pass

    threading.Thread(target=beat, daemon=True).start()
    log(f"reaped at start: {q.reap(a.stale_seconds)}")
    reason = "queue_drained"
    while True:
        if stop.is_set():
            reason = "signal"
            break
        if (run / "STOP").exists():
            reason = "stop_file"
            break
        if end and end - time.time() < a.margin_seconds:
            reason = "wall_margin"
            break
        if a.max_shards and state["shards_done"] >= a.max_shards:
            reason = "max_shards"
            break
        nm, claim = q.claim(a.order)
        if nm is None:
            back = q.reap(a.stale_seconds)
            if back:
                log(f"reaped {back}")
                continue
            break
        current["claim"] = claim
        out, rec = run / "out" / f"{nm}.npy", run / "out" / f"{nm}.json"
        if out.exists() and rec.exists():  # finished earlier, the done marker was lost
            q.finish(nm, claim)
            current["claim"] = None
            continue
        uids, seqs = read_shard(run / "shards" / f"{nm}.faa")
        L = np.fromiter((len(s) for s in seqs), dtype=np.int64, count=len(seqs))
        V = np.full((len(seqs), backend.dim), np.nan, dtype=np.float32)
        t0, nb, splits, failed = time.time(), 0, 0, []
        state["current"] = nm
        todo = list(batches(L, backend.token_budget, backend.square_budget))
        interrupted = False
        while todo:
            if stop.is_set():
                interrupted = True
                break
            b, e = todo.pop(0)
            try:
                V[b:e] = backend.embed(seqs[b:e])
                nb += 1
            except Exception as ex:  # noqa: BLE001
                if not backend.is_oom(ex):
                    q.release(nm, claim)
                    raise
                splits += 1
                if e - b == 1:
                    failed.append(dict(uid=int(uids[b]), length=int(L[b]), reason="out_of_memory"))
                    log(f"{nm}: uid {int(uids[b])} length {int(L[b])} exceeds this device's memory")
                else:
                    m = (b + e) // 2
                    todo[:0] = [(b, m), (m, e)]
        if interrupted:
            q.release(nm, claim)
            current["claim"] = None
            reason = "signal"
            break
        dt = time.time() - t0
        # audit on the embedding device: a few proteins of this shard again, one per pass, in float32
        aud, ta = [], time.time()
        okrows = np.flatnonzero(np.isfinite(V).all(1))
        rs = np.random.default_rng(1000 + int(nm.split("_")[1]))
        for i in rs.choice(okrows, size=min(a.audit_per_shard, len(okrows)), replace=False):
            try:
                c = float(cos_rows(backend.embed([seqs[i]], reference=True), V[i:i + 1])[0])
                aud.append(dict(uid=int(uids[i]), length=int(L[i]), cosine=c))
            except Exception as ex:  # noqa: BLE001
                if not backend.is_oom(ex):
                    q.release(nm, claim)
                    raise
                aud.append(dict(uid=int(uids[i]), length=int(L[i]), cosine=None, reason="out_of_memory in float32"))
        ta = time.time() - ta
        tmp = run / "out" / f".{nm}.{wid}.tmp.npy"
        np.save(tmp, V)
        digest = sha256_file(tmp)
        os.replace(tmp, out)
        bad = int(np.isnan(V).any(1).sum())
        write_json(rec, dict(shard=nm, n=len(seqs), residues=int(L.sum()), tokens=int(L.sum() + 2 * len(L)),
                             min_len=int(L.min()), max_len=int(L.max()), seconds=round(dt, 3),
                             residues_per_second=round(L.sum() / dt, 1), batches=nb, oom_splits=splits,
                             failed=failed, nan_rows=bad, audit=aud, audit_seconds=round(ta, 2), worker=wid, host=backend.info.get("host"),
                             gpu=backend.info.get("gpu"), dtype=backend.info.get("dtype"), sha256=digest, finished=now()))
        q.finish(nm, claim)
        current["claim"] = None
        state["shards_done"] += 1
        state["residues"] += int(L.sum())
        state["seconds"] += dt
        state["current"] = None
        state["residues_per_second"] = round(state["residues"] / max(state["seconds"], 1e-9), 1)
        write_json(wfile, state)
        log(f"{nm} n={len(seqs):,} len {int(L.min())}-{int(L.max())} {dt:.1f}s {L.sum() / dt:,.0f} res/s "
            f"batches {nb} splits {splits} failed {len(failed)} audit min cos "
            f"{min([x['cosine'] for x in aud if x['cosine'] is not None], default=float('nan')):.6f} | total {state['shards_done']} shards "
            f"{state['residues_per_second']:,.0f} res/s | queue {q.counts()}")
    state.update(finished=now(), exit=reason, queue=q.counts())
    write_json(wfile, state)
    log(f"exit: {reason}; {state['shards_done']} shards, {state['residues']:,} residues, queue {q.counts()}")
    if reason == "wall_margin":
        sys.exit(75)  # the wrapper submits a successor


# --------------------------------------------------------------------------------------
# status / reap
# --------------------------------------------------------------------------------------
def run_state(run: Path) -> dict:
    meta = json.loads((run / "run.json").read_text())
    q = Queue(run, "status")
    c = q.counts()
    recs = [json.loads(p.read_text()) for p in (run / "out").glob("shard_*.json")]
    by = Counter()
    sec = Counter()
    for r in recs:
        g = (r.get("gpu") or {}).get("name") or r.get("host") or "cpu"
        by[g] += r["residues"]
        sec[g] += r["seconds"]
    res_done = sum(r["residues"] for r in recs)
    return dict(run=str(run), shards=meta["shards"], **c, all_done=(c["done"] == meta["shards"]),
                residues_done=res_done, residues_total=meta["residues_unique"],
                fraction=round(res_done / max(meta["residues_unique"], 1), 4),
                failed_proteins=sum(len(r["failed"]) for r in recs),
                by_device={g: dict(residues=by[g], seconds=round(sec[g], 1), residues_per_second=round(by[g] / max(sec[g], 1e-9)))
                           for g in by},
                claims=sorted(os.listdir(q.claimed)))


def cmd_status(a) -> None:
    st = run_state(Path(a.run))
    if a.check_done:
        sys.exit(0 if st["all_done"] else 1)
    print(json.dumps(st, indent=2))


def cmd_reap(a) -> None:
    print(json.dumps(Queue(Path(a.run), "reaper").reap(a.stale_seconds)))


# --------------------------------------------------------------------------------------
# finalize / compare
# --------------------------------------------------------------------------------------
def cmd_finalize(a) -> None:
    run = Path(a.run)
    meta = json.loads((run / "run.json").read_text())
    st = run_state(run)
    if not st["all_done"]:
        sys.exit(f"queue is not drained: {json.dumps({k: st[k] for k in ('shards', 'pending', 'claimed', 'done')})}")
    t0 = time.time()
    L = np.load(run / "length_of_uid.npy")
    uid_of_row = np.load(run / "uid_of_row.npy")
    names = [ln.split("\t")[0] for ln in (run / "shards.tsv").read_text().splitlines()[1:]]
    recs = {n: json.loads((run / "out" / f"{n}.json").read_text()) for n in names}
    D = int(np.load(run / "out" / f"{names[0]}.npy", mmap_mode="r").shape[1])
    fin = run / "final"
    raw = np.lib.format.open_memmap(fin / ".mean_raw_f32__uid.tmp.npy", mode="w+", dtype=np.float32, shape=(len(L), D))
    seen = np.zeros(len(L), dtype=np.uint8)

    def load(n):
        V = np.load(run / "out" / f"{n}.npy")
        u = np.load(run / "shards" / f"{n}.uids.npy")
        assert V.shape == (len(u), D) and len(u) == recs[n]["n"], f"{n}: shape differs from the shard"
        raw[u] = V
        seen[u] += 1
        return n

    with ThreadPoolExecutor(max(1, a.threads)) as ex:
        list(ex.map(load, names))
    assert (seen == 1).all(), f"{int((seen != 1).sum())} unique sequences are not covered exactly once"
    nan_uid = np.zeros(len(L), dtype=bool)
    norms = np.empty(len(L), dtype=np.float32)
    step = 200_000
    for s in range(0, len(L), step):
        blk = raw[s:s + step]
        nan_uid[s:s + step] = ~np.isfinite(blk).all(1)
        norms[s:s + step] = np.linalg.norm(blk, axis=1)
    status = np.where(nan_uid, 2, 1).astype(np.uint8)
    raw.flush()
    os.replace(fin / ".mean_raw_f32__uid.tmp.npy", fin / "mean_raw_f32__uid.npy")
    np.save(fin / "status__uid.npy", status)
    row = np.lib.format.open_memmap(fin / ".l2_f32__row.tmp.npy", mode="w+", dtype=np.float32, shape=(len(uid_of_row), D))

    def fill(s):
        u = uid_of_row[s:s + step]
        blk = np.full((len(u), D), np.nan, dtype=np.float32)
        ok = u >= 0
        v = raw[u[ok]]
        blk[ok] = v / np.linalg.norm(v, axis=1, keepdims=True)
        row[s:s + step] = blk
        return int((~np.isfinite(blk).all(1)).sum())

    with ThreadPoolExecutor(max(1, a.threads)) as ex:
        nan_rows = sum(ex.map(fill, range(0, len(uid_of_row), step)))
    row.flush()
    os.replace(fin / ".l2_f32__row.tmp.npy", fin / "l2_f32__row.npy")
    log(f"assembled {len(L):,} unique and {len(uid_of_row):,} rows in {time.time() - t0:.0f}s")

    # assembly check, independent of the .uids.npy files: uids parsed from the shard FASTA headers
    rng0 = np.random.default_rng(a.seed)
    mism = 0
    for n in rng0.choice(names, size=min(12, len(names)), replace=False):
        us, _ = read_shard(run / "shards" / f"{n}.faa")
        mism += int((~np.isclose(np.asarray(raw[us]), np.load(run / "out" / f"{n}.npy"), equal_nan=True)).any(1).sum())
    rr = np.sort(rng0.choice(np.flatnonzero(uid_of_row >= 0), size=min(20000, int((uid_of_row >= 0).sum())), replace=False))
    vv = np.asarray(raw[uid_of_row[rr]], dtype=np.float32)
    vv /= np.linalg.norm(vv, axis=1, keepdims=True)
    mism_rows = int((~np.isclose(vv, np.asarray(row[rr]), equal_nan=True, atol=1e-6)).any(1).sum())
    assembly = dict(shards_checked=min(12, len(names)), unique_rows_differing=mism, rows_checked=int(len(rr)),
                    rows_differing=mism_rows, passed=bool(mism == 0 and mism_rows == 0))
    cs = np.array([x["cosine"] for r in recs.values() for x in r.get("audit", []) if x["cosine"] is not None])
    shard_audit = dict(
        n=int(len(cs)), skipped=int(sum(1 for r in recs.values() for x in r.get("audit", []) if x["cosine"] is None)),
        compared="float32 single-protein pass against the stored batched vector, on the embedding device",
        min_cosine=float(cs.min()) if len(cs) else None, p01_cosine=float(np.quantile(cs, 0.01)) if len(cs) else None,
        median_cosine=float(np.median(cs)) if len(cs) else None, gate=a.audit_min_cos,
        passed=bool(len(cs) == 0 or cs.min() >= a.audit_min_cos),
        lowest=sorted((x for r in recs.values() for x in r.get("audit", []) if x["cosine"] is not None),
                      key=lambda x: x["cosine"])[:5])
    log(f"assembly {json.dumps(assembly)}")
    log(f"per-shard audit {json.dumps(shard_audit)}")
    audit = None
    if a.audit:
        import types
        args = types.SimpleNamespace(device="cpu", threads=a.threads, backend=None)
        backend = make_backend(meta, args, dtype="fp32")
        rng = np.random.default_rng(a.seed)
        ok = np.flatnonzero((status == 1) & (L <= a.audit_max_len))
        pick = np.sort(rng.choice(ok, size=min(a.audit, len(ok)), replace=False))
        want = set(int(u) for u in pick)
        seqs = {}
        for n in names:
            u = np.load(run / "shards" / f"{n}.uids.npy")
            if want.intersection(u.tolist()):
                us, ss = read_shard(run / "shards" / f"{n}.faa")
                seqs.update({int(x): s for x, s in zip(us, ss) if int(x) in want})
        t1 = time.time()
        ref = np.vstack([backend.embed([seqs[int(u)]], reference=True) for u in pick])
        c = cos_rows(ref, np.asarray(raw[pick]))
        rowpick = np.flatnonzero(np.isin(uid_of_row, pick))[: len(pick)]
        c_row = cos_rows(ref[np.searchsorted(pick, uid_of_row[rowpick])], np.asarray(row[rowpick]))
        audit = dict(n=int(len(pick)), max_len=a.audit_max_len, device="cpu float32, one protein per pass",
                     min_cosine=float(c.min()), p01_cosine=float(np.quantile(c, 0.01)), median_cosine=float(np.median(c)),
                     row_matrix_min_cosine=float(c_row.min()), gate=a.audit_min_cos, passed=bool(min(c.min(), c_row.min()) >= a.audit_min_cos),
                     seconds=round(time.time() - t1, 1), lengths=[int(L[pick].min()), int(L[pick].max())])
        log(f"audit {json.dumps(audit)}")
    workers = [json.loads(p.read_text()) for p in sorted((run / "workers").glob("*.json"))]
    man = dict(
        run=meta, finalized=now(), dim=D, unique=int(len(L)), rows=int(len(uid_of_row)),
        unique_failed=int(nan_uid.sum()), failed=[f for r in recs.values() for f in r["failed"]],
        rows_without_vector=int(nan_rows), rows_empty_sequence=int((uid_of_row < 0).sum()),
        raw_norm=dict(min=float(np.nanmin(norms)), median=float(np.nanmedian(norms)), max=float(np.nanmax(norms))),
        by_device=st["by_device"], embedding_seconds=round(sum(r["seconds"] for r in recs.values()), 1),
        oom_splits=int(sum(r["oom_splits"] for r in recs.values())),
        selfchecks=[dict(worker=w["worker"], gpu=(w["backend"].get("gpu") or {}).get("name"), dtype=w["backend"].get("dtype"),
                         **(w.get("selfcheck") or {})) for w in workers],
        shard_audit=shard_audit, assembly=assembly, cpu_model_audit=audit,
        files={p.name: dict(bytes=p.stat().st_size, sha256=sha256_file(p)) for p in
               (fin / "mean_raw_f32__uid.npy", fin / "l2_f32__row.npy", fin / "status__uid.npy",
                run / "uid_of_row.npy", run / "length_of_uid.npy")},
        products={"final/l2_f32__row.npy": "float32 [rows, D], L2-normalized, aligned to the input row order (NaN = no vector)",
                  "final/mean_raw_f32__uid.npy": "float32 [unique, D], raw residue means of the unique sequences",
                  "uid_of_row.npy": "int32 [rows], unique-sequence id of each row (-1 = empty sequence)",
                  "final/status__uid.npy": "uint8 [unique], 1 = embedded, 2 = failed"},
    )
    write_json(fin / "MANIFEST.json", man)
    log(f"finalized in {time.time() - t0:.0f}s")
    print(json.dumps({k: man[k] for k in ("dim", "unique", "rows", "unique_failed", "rows_without_vector", "raw_norm",
                                          "by_device", "embedding_seconds", "selfchecks", "shard_audit", "assembly")}, indent=2))
    if not (shard_audit["passed"] and assembly["passed"]) or (audit and not audit["passed"]):
        sys.exit(4)


def cmd_compare(a) -> None:
    A = np.load(Path(a.a) / "final" / "mean_raw_f32__uid.npy", mmap_mode="r")
    B = np.load(Path(a.b) / "final" / "mean_raw_f32__uid.npy", mmap_mode="r")
    assert A.shape == B.shape, f"shapes differ: {A.shape} vs {B.shape}"
    la, lb = np.load(Path(a.a) / "length_of_uid.npy"), np.load(Path(a.b) / "length_of_uid.npy")
    assert (la == lb).all(), "the two runs hold different sequences"
    c = cos_rows(np.asarray(A), np.asarray(B))
    res = dict(a=a.a, b=a.b, n=int(len(c)), min_cosine=float(np.nanmin(c)), p01_cosine=float(np.nanquantile(c, 0.01)),
               median_cosine=float(np.nanmedian(c)), max_abs_difference=float(np.nanmax(np.abs(np.asarray(A) - np.asarray(B)))))
    write_json(Path(a.a) / "final" / f"compare__{Path(a.b).name}.json", res)
    print(json.dumps(res, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep")
    p.add_argument("--fasta", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--backend", default="esmpp", choices=("esmpp", "esmc_native", "fake"))
    p.add_argument("--model-dir")
    p.add_argument("--model-id")
    p.add_argument("--model-revision")
    p.add_argument("--dtype", default="bf16", choices=("bf16", "fp32"))
    p.add_argument("--shard-tokens", type=int, default=4_000_000)
    p.add_argument("--sample", type=int, default=0)
    p.add_argument("--sample-max-len", type=int, default=6000)
    p.add_argument("--sample-duplicates", type=int, default=0)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--selfcheck-n", type=int, default=32)
    p.add_argument("--selfcheck-max-len", type=int, default=2500)
    p.set_defaults(fn=cmd_prep)
    p = sub.add_parser("work")
    p.add_argument("--run", required=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--threads", type=int, required=True)
    p.add_argument("--order", default="asc", choices=("asc", "desc"))
    p.add_argument("--backend")
    p.add_argument("--token-budget", default="auto")
    p.add_argument("--square-budget", default="auto")
    p.add_argument("--selfcheck", type=int, default=1)
    p.add_argument("--selfcheck-min-cos", type=float, default=0.999)
    p.add_argument("--stale-seconds", type=float, default=600)
    p.add_argument("--margin-seconds", type=float, default=900)
    p.add_argument("--max-shards", type=int, default=0)
    p.add_argument("--audit-per-shard", type=int, default=3)
    p.add_argument("--fake-delay", type=float, default=0.0)
    p.set_defaults(fn=cmd_work)
    p = sub.add_parser("status")
    p.add_argument("--run", required=True)
    p.add_argument("--check-done", action="store_true")
    p.set_defaults(fn=cmd_status)
    p = sub.add_parser("reap")
    p.add_argument("--run", required=True)
    p.add_argument("--stale-seconds", type=float, default=600)
    p.set_defaults(fn=cmd_reap)
    p = sub.add_parser("finalize")
    p.add_argument("--run", required=True)
    p.add_argument("--threads", type=int, required=True)
    p.add_argument("--audit", type=int, default=0, help="extra audit with the model on CPU (off by default)")
    p.add_argument("--audit-max-len", type=int, default=2500)
    p.add_argument("--audit-min-cos", type=float, default=0.999)
    p.add_argument("--seed", type=int, default=23)
    p.set_defaults(fn=cmd_finalize)
    p = sub.add_parser("compare")
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    p.set_defaults(fn=cmd_compare)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
