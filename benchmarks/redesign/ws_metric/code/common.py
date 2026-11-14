"""Shared paths, loaders and transforms for the ws_metric workstream."""
import hashlib
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/spacedust")
SCALE = ROOT / "model_scaling_20260913"
S = ROOT / "seq_invisible_20261111"
RD = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111")
U = RD / "universe"
W = RD / "ws_metric"
SETS = W / "sets"
MODELS = ["esm2_8m", "esm2_650m", "esmc600m", "esmc6b"]
N_REF = 4_190_383
N_QUERY = 67_148
N_GENES = N_REF + N_QUERY
ACTINO_STRATA = ("0_genus_Streptomyces", "1_same_class_other_genus", "2_same_phylum_other_class")
OTHER_STRATUM = "3_other_phylum"
STREP_STRATUM = "0_genus_Streptomyces"
K_RETRIEVE = 60


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def code_hash(*paths):
    h = hashlib.sha256()
    for p in paths:
        h.update(Path(p).read_bytes())
    return h.hexdigest()[:16]


def vector_paths(model):
    proto = json.load(open(SCALE / "results" / model / "analysis/full_workflow/protocol.json"))
    return proto["vectors"]["query"]["path"], proto["vectors"]["reference"]["path"]


class RowReader:
    """Reads row blocks of a 2-D .npy with large sequential reads (np.fromfile at an
    offset). Faster and steadier than mmap page faults on the VAST mount: an mmap pass
    over the ESM2-650M reference stalled at ~2 MB/s on one node (job 76309), while dd of
    the same file reads ~480 MB/s."""

    def __init__(self, path):
        self.path = str(path)
        with open(self.path, "rb") as f:
            version = np.lib.format.read_magic(f)
            shape, fortran, dtype = np.lib.format._read_array_header(f, version)
            self.offset = f.tell()
        assert not fortran and len(shape) == 2
        self.shape, self.dtype = shape, np.dtype(dtype)
        self.rowbytes = shape[1] * self.dtype.itemsize

    def __len__(self):
        return self.shape[0]

    def read(self, c0, c1):
        n = max(0, min(c1, self.shape[0]) - c0)
        with open(self.path, "rb") as f:
            f.seek(self.offset + c0 * self.rowbytes)
            buf = np.fromfile(f, dtype=self.dtype, count=n * self.shape[1])
        assert buf.size == n * self.shape[1], (self.path, c0, c1, buf.size)
        return buf.reshape(n, self.shape[1]).astype(np.float32, copy=False)

    def rows(self, idx, chunk=262_144):
        """Rows at sorted-or-unsorted indices idx, read block-wise; returns in idx order."""
        idx = np.asarray(idx, dtype=np.int64)
        out = np.empty((len(idx), self.shape[1]), np.float32)
        if not len(idx):
            return out
        order = np.argsort(idx, kind="stable")
        s = idx[order]
        lo = 0
        while lo < len(s):
            c0 = int(s[lo])
            c1 = min(c0 + chunk, self.shape[0])
            hi = int(np.searchsorted(s, c1, side="left"))
            block = self.read(c0, int(s[hi - 1]) + 1)
            out[order[lo:hi]] = block[s[lo:hi] - c0]
            lo = hi
        return out


def load_vectors(model):
    qp, rp = vector_paths(model)
    return np.load(qp, mmap_mode="r"), np.load(rp, mmap_mode="r")


def readers(model):
    qp, rp = vector_paths(model)
    return RowReader(qp), RowReader(rp)


def heldout():
    return json.load(open(SETS / "heldout.json"))


def ref_tables():
    """Per reference row: genome_code, stratum flags. Order = row_index = gid."""
    import pandas as pd
    g = pd.read_parquet(U / "genes.parquet", columns=["gid", "side", "row_index", "genome_code"])
    gn = pd.read_parquet(U / "genomes.parquet", columns=["genome_code", "stratum", "phylum"])
    r = g[g.side == "ref"].sort_values("row_index")
    assert (r.row_index.to_numpy() == np.arange(N_REF)).all()
    assert (r.gid.to_numpy() == np.arange(N_REF)).all()
    q = g[g.side == "query"].sort_values("row_index")
    assert (q.row_index.to_numpy() == np.arange(N_QUERY)).all()
    assert (q.gid.to_numpy() == N_REF + np.arange(N_QUERY)).all()
    strat = dict(zip(gn.genome_code, gn.stratum))
    rcodes = r.genome_code.to_numpy().astype(np.int64)
    rstrat = np.array([strat[c] for c in range(int(gn.genome_code.max()) + 1)], dtype=object)[rcodes]
    return dict(ref_codes=rcodes, query_codes=q.genome_code.to_numpy().astype(np.int64),
                ref_other=(rstrat == OTHER_STRATUM), ref_actino=np.isin(rstrat, ACTINO_STRATA),
                ref_strep=(rstrat == STREP_STRATUM), n_codes=int(gn.genome_code.max()) + 1)


# --------------------------------------------------------------------------------------
# Heads
# --------------------------------------------------------------------------------------

def make_head(kind, d_in, d_out):
    import torch
    if kind == "lin":
        return torch.nn.Linear(d_in, d_out, bias=False)
    if kind == "mlp":
        return torch.nn.Sequential(torch.nn.Linear(d_in, 1024), torch.nn.GELU(),
                                   torch.nn.Linear(1024, d_out))
    raise ValueError(kind)


# Learned-arm grid: one factor at a time around the default.
DEFAULT = dict(kind="lin", d_out=256, temp=0.05, pos="ge6", neg="inb")
LEARNED = {
    "lin256_t05_ge6_inb": dict(DEFAULT),
    "lin64_t05_ge6_inb": dict(DEFAULT, d_out=64),
    "lin128_t05_ge6_inb": dict(DEFAULT, d_out=128),
    "lin512_t05_ge6_inb": dict(DEFAULT, d_out=512),
    "mlp256_t05_ge6_inb": dict(DEFAULT, kind="mlp"),
    "lin256_t03_ge6_inb": dict(DEFAULT, temp=0.03),
    "lin256_t10_ge6_inb": dict(DEFAULT, temp=0.10),
    "lin256_t05_acc05_inb": dict(DEFAULT, pos="acc05"),
    "lin256_t05_vis_inb": dict(DEFAULT, pos="vis"),
    "lin256_t05_ge6_hard": dict(DEFAULT, neg="hard"),
}
# Round 2 (chosen after the esm2_8m round-1 results; exploratory, flagged as such in reports):
# combine the two best single factors, stack a head on the soft-ZCA input, extend the
# temperature axis, and one seed replicate for training variance.
LEARNED_R2 = {
    "lin256_t10_vis_inb": dict(DEFAULT, temp=0.10, pos="vis"),
    "zlin256_t10_ge6_inb": dict(DEFAULT, temp=0.10, inp="zca10"),
    "zlin256_t10_vis_inb": dict(DEFAULT, temp=0.10, pos="vis", inp="zca10"),
    "lin256_t20_ge6_inb": dict(DEFAULT, temp=0.20),
    "lin256_t10_ge6_inb_s18": dict(DEFAULT, temp=0.10, seed=18),
}
LEARNED.update(LEARNED_R2)
TRAIN = dict(steps=3000, batch=2048, lr=1e-3, n_hard=8, val_frac=0.05, seed=17)

UNTRAINED = ["raw", "gcenter", "genomecenter", "csls2", "abtt2", "abtt10", "abtt30", "zca10", "pca256"]


# --------------------------------------------------------------------------------------
# Transforms (torch, float32). Every transform returns L2-normalized rows except csls2,
# whose reference side is the augmented [t, r_t] and query side [2q, -1].
# --------------------------------------------------------------------------------------

class Transforms:
    def __init__(self, stats_dir, heads_dir=None, arms=None):
        import torch
        st = np.load(stats_dir / "stats.npz")
        self.mu = torch.from_numpy(st["mu"].astype(np.float32))
        self.gmeans = torch.from_numpy(st["gmeans"].astype(np.float32))
        evecs = st["evecs"].astype(np.float32)          # columns, descending eigenvalue
        evals = st["evals"].astype(np.float64)
        self.U = torch.from_numpy(np.ascontiguousarray(evecs))
        d = len(evals)
        alpha = 0.1
        lam = (1 - alpha) * evals + alpha * evals.sum() / d
        lam = np.maximum(lam, 1e-12)
        Wz = (evecs.astype(np.float64) * lam ** -0.5) @ evecs.T.astype(np.float64)
        self.Wz = torch.from_numpy(Wz.astype(np.float32))
        self.r_t = torch.from_numpy(np.load(stats_dir / "r_t.npy").astype(np.float32))
        self.heads = {}
        if heads_dir is not None:
            for name, cfg in LEARNED.items():
                if arms is not None and name not in arms:
                    continue
                p = heads_dir / f"{name}.pt"
                if not p.exists():
                    continue
                h = make_head(cfg["kind"], d, cfg["d_out"])
                h.load_state_dict(torch.load(p, map_location="cpu"))
                h.eval()
                self.heads[name] = h

    @staticmethod
    def l2n(Y):
        return Y / Y.norm(dim=1, keepdim=True).clamp_min(1e-12)

    def arms(self):
        return UNTRAINED + list(self.heads)

    def apply(self, arm, X, codes, side, ref_rows=None):
        """X: (n, d) float32 torch; codes: genome_code per row (torch long);
        side: 'q' or 't'; ref_rows: reference row ids (torch long) for csls2 targets."""
        import torch
        l2n = self.l2n
        if arm == "raw":
            return l2n(X)
        if arm == "gcenter":
            return l2n(X - self.mu)
        if arm == "genomecenter":
            return l2n(X - self.gmeans[codes])
        if arm.startswith("abtt"):
            k = int(arm[4:])
            C = X - self.mu
            Uk = self.U[:, :k]
            return l2n(C - (C @ Uk) @ Uk.T)
        if arm == "pca256":
            return l2n((X - self.mu) @ self.U[:, :256])
        if arm == "zca10":
            return l2n((X - self.mu) @ self.Wz)
        if arm == "csls2":
            Y = l2n(X)
            if side == "q":
                return torch.cat([2 * Y, -torch.ones(len(Y), 1)], dim=1)
            return torch.cat([Y, self.r_t[ref_rows][:, None]], dim=1)
        if arm in self.heads:
            with torch.no_grad():
                return l2n(self.heads[arm](self.head_input(LEARNED[arm].get("inp", "gcenter"), X, codes)))
        raise ValueError(arm)

    def head_input(self, inp, X, codes):
        """Input preprocessing of a learned head (default: subtract the global mean)."""
        if inp == "gcenter":
            return X - self.mu
        if inp == "zca10":
            return (X - self.mu) @ self.Wz
        if inp == "genomecenter":
            return X - self.gmeans[codes]
        raise ValueError(inp)
