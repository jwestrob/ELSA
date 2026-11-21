"""Shared paths and helpers for the ws_chain workstream (ELSA redesign, 2026-11-11)."""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/spacedust")
SCALE = ROOT / "model_scaling_20260913"
S = ROOT / "seq_invisible_20261111"
N = ROOT / "native_run_20260910"
RD = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111")
W = RD / "ws_chain"
CODE = W / "code"
PROD = CODE / "production"          # private copy of $SCALE/production
SCALE_CODE = CODE / "scale"         # private copy of $SCALE/code
MODELS = ["esm2_8m", "esm2_650m", "esmc600m"]
PERM_SEEDS = (17, 18, 19)
TAGS = ("observed",) + tuple(f"order_seed{s}" for s in PERM_SEEDS)
SPAN = ["query_genome", "query_contig", "query_start", "query_end"]
HELDOUT = ("Micromonospora sp. M42", "Streptomyces himastatinicus ATCC 53653", "Streptomyces sp. C")
SAVED_V0 = {  # observed blocks, then permuted totals for seeds 17/18/19 (tonight's panels)
    "esm2_8m": (534990, [495161, 495380, 495342]),
}
SCORE_KINDS = ("raw", "top10", "zlist", "zq", "z2")
QSTAT_COLS = ("top10", "mean_list", "sd_list", "mu_bg", "sd_bg")
P_CLIP = (0.005, 0.995)

for p in (str(CODE), str(SCALE_CODE)):
    if p not in sys.path:
        sys.path.insert(0, p)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=1, default=float) + "\n")
    os.replace(tmp, path)


def ordered_metadata(metadata, seed):
    """Identical to permuted_panels.ordered_metadata (September rule)."""
    result = metadata.copy()
    for (sample, contig), frame in metadata.groupby(["sample_id", "contig_id"], observed=True, sort=True):
        rng = np.random.default_rng(int.from_bytes(
            hashlib.sha256(f"{seed}:{sample}:{contig}".encode()).digest()[:8], "big"))
        result.loc[frame.index, "position_index"] = rng.permutation(len(frame))[
            frame.position_index.to_numpy(dtype=np.int64)]
    return result


def cut_for(obs_m, perm_m, n_perm, target):
    """Identical to permuted_panels.cut_for."""
    if len(obs_m) == 0:
        return None
    obs_sorted, perm_sorted = np.sort(obs_m), np.sort(perm_m)
    for t in np.unique(np.quantile(obs_m, np.linspace(0, 1, 401))):
        o = len(obs_sorted) - np.searchsorted(obs_sorted, t, side="left")
        p = (len(perm_sorted) - np.searchsorted(perm_sorted, t, side="left")) / n_perm
        if o > 0 and p / o <= target:
            return float(t)
    return None


def size_bin(n):
    n = np.asarray(n)
    return np.minimum(n, 6)


def qvalues(obs_stat, perm_stat, n_perm):
    """Empirical q-value per observed block: min over t <= S of (#perm>=t / n_perm) / (#obs>=t)."""
    obs_stat = np.asarray(obs_stat, dtype=np.float64)
    if len(obs_stat) == 0:
        return np.empty(0)
    perm_sorted = np.sort(np.asarray(perm_stat, dtype=np.float64))
    order = np.argsort(-obs_stat, kind="stable")
    s = obs_stat[order]
    # ties: #obs >= s must count every tied block
    last_of_tie = np.searchsorted(-s, -s, side="right")
    n_obs_ge = last_of_tie.astype(np.float64)
    n_perm_ge = (len(perm_sorted) - np.searchsorted(perm_sorted, s, side="left")) / n_perm
    fdr = n_perm_ge / n_obs_ge
    # q(S) = min over t <= S of fdr(t): running minimum from the lowest score upward
    q_sorted = np.minimum.accumulate(fdr[::-1])[::-1]
    out = np.empty(len(obs_stat))
    out[order] = q_sorted
    return out


def base(model):
    return SCALE / "results" / model / "analysis/full_workflow"


def array_paths(model):
    from spacedust_prepared_extension_executor import ARRAY_NAMES
    b = base(model)
    proto = json.load(open(b / "protocol.json"))
    paths = {name: b / "arrays" / f"{name}.npy" for name in ARRAY_NAMES if not name.endswith("_vectors")}
    paths["query_vectors"] = Path(proto["vectors"]["query"]["path"])
    paths["reference_vectors"] = Path(proto["vectors"]["reference"]["path"])
    for p in paths.values():
        assert Path(p).exists(), p
    return paths


def load_meta():
    import run_spacedust_scaling as rss
    qmeta = rss.complete_metadata(pd.read_parquet(ROOT / "bgc_queries/metadata.parquet"), rss.QUERY_ROWS)
    rmeta = rss.complete_metadata(
        pd.read_parquet(ROOT / "results/esm2_t6/shard_000/export/metadata.parquet"), rss.REFERENCE_ROWS)
    return qmeta, rmeta


def load_published():
    import run_spacedust_scaling as rss
    return rss.load_published(PROD)


def model_dirs(model):
    work = W / "work" / model
    runs = W / "runs" / model
    work.mkdir(parents=True, exist_ok=True)
    runs.mkdir(parents=True, exist_ok=True)
    return work, runs


def load_map(model):
    m = json.load(open(W / "work" / model / "map.json"))
    return m["kind"], np.asarray(m["x"], dtype=np.float64), np.asarray(m["y"], dtype=np.float64)


def pair_score(kind, qrows, trows, cos, qstats, rstats):
    """Order-free pair score. qstats columns follow QSTAT_COLS; rstats = (mu_bg, sd_bg)."""
    cos = np.asarray(cos, dtype=np.float64)
    if kind == "raw":
        return cos
    q = qstats[np.asarray(qrows, dtype=np.int64)]
    if kind == "top10":
        return cos - q[:, 0]
    if kind == "zlist":
        return (cos - q[:, 1]) / np.maximum(q[:, 2], 1e-4)
    if kind == "zq":
        return (cos - q[:, 3]) / np.maximum(q[:, 4], 1e-6)
    if kind == "z2":
        r = rstats[np.asarray(trows, dtype=np.int64)].astype(np.float64)
        return 0.5 * ((cos - q[:, 3]) / np.maximum(q[:, 4], 1e-6) + (cos - r[:, 0]) / np.maximum(r[:, 1], 1e-6))
    raise ValueError(kind)


def evidence(s, x, y):
    """Calibrated log-odds from the isotonic map (piecewise linear, clamped at the ends)."""
    p = np.interp(np.asarray(s, dtype=np.float64), x, y)
    p = np.clip(p, *P_CLIP)
    out = np.log(p) - np.log1p(-p)
    return np.where(np.isfinite(s), out, -np.log(P_CLIP[1] / P_CLIP[0]))


def load_stats(model):
    work = W / "work" / model
    return (np.load(work / "query_stats.npy", mmap_mode="r"),
            np.load(work / "reference_stats.npy", mmap_mode="r"))


def universe_keys():
    """row -> mmseqs key for each side, and genome stratum per ELSA reference sample_id."""
    u = pd.read_parquet(RD / "universe/genes.parquet", columns=["side", "row_index", "mmseqs_key", "genome"])
    q = u[u.side == "query"].sort_values("row_index")
    r = u[u.side == "ref"].sort_values("row_index")
    assert np.array_equal(q.row_index.to_numpy(), np.arange(len(q)))
    assert np.array_equal(r.row_index.to_numpy(), np.arange(len(r)))
    return q.mmseqs_key.to_numpy(np.int64), r.mmseqs_key.to_numpy(np.int64)
