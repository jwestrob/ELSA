"""Embedding post-processing transforms, as a fit/apply ladder.

Mean-pooled protein language model embeddings are strongly anisotropic. On the
DPANN corpus the mean of the unit vectors has norm 0.932, meaning 87% of every
embedding is one shared direction, so a raw cosine largely reports a constant:
two *unrelated* proteins sit at 0.935 while anchored pairs sit at 0.996.

The literature treats correcting this as a dose-response question rather than a
binary one. Removing a single direction is reliably a small win, while full-rank
whitening degrades every model tested in the surveyed panel, because it amplifies
noise in a heavy-tailed eigenspectrum. This module therefore implements the whole
ladder — including a random-direction control that separates "removing *the* mean
direction" from "removing *any* direction" — so the choice can be measured
instead of assumed.

Every transform fits on one matrix and applies to another, so a transform can be
fitted on held-out genomes or an external corpus and applied without leakage.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


def l2_normalize(matrix: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, eps)


def _as_float64(matrix: np.ndarray) -> np.ndarray:
    """Materialize a (possibly memory-mapped) matrix as float64."""
    return np.asarray(matrix, dtype=np.float64)


def sample_rows(
    matrix: np.ndarray,
    n: int,
    seed: int = 17,
) -> np.ndarray:
    """Draw a deterministic row sample; fitting on 3.4M vectors is unnecessary."""
    total = matrix.shape[0]
    if total <= n:
        return _as_float64(matrix[:])
    rng = np.random.default_rng(seed)
    rows = np.sort(rng.choice(total, n, replace=False))
    return _as_float64(matrix[rows])


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def mean_norm(matrix: np.ndarray) -> float:
    """‖mean of the unit vectors‖ — 0 is isotropic, 1 is fully collapsed.

    The single most predictive statistic for whether a correction will help: in
    the surveyed panel, benefit tracked this at r = +0.72.
    """
    unit = l2_normalize(_as_float64(matrix))
    return float(np.linalg.norm(unit.mean(axis=0)))


def isotropy_score(matrix: np.ndarray, seed: int = 17) -> float:
    """Fraction of dimensions effectively occupied, via normalized covariance spectrum.

    An IsoScore-style summary: 1.0 when variance is spread evenly over all
    directions, near 0 when it collapses onto a few.
    """
    unit = l2_normalize(_as_float64(matrix))
    centred = unit - unit.mean(axis=0)
    eigenvalues = np.linalg.eigvalsh(np.cov(centred, rowvar=False))
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    total = eigenvalues.sum()
    if total <= 0:
        return 0.0
    p = eigenvalues / total
    # exp(entropy) / dim: the participation ratio of the spectrum
    nonzero = p[p > 0]
    return float(np.exp(-(nonzero * np.log(nonzero)).sum()) / len(eigenvalues))


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

@dataclass
class EmbeddingTransform:
    """A fitted transform: state plus the recipe that produced it.

    ``kind`` selects the behaviour; ``state`` holds whatever arrays that kind
    needs. Keeping them together in one serializable object means a store or a
    run can record exactly which treatment produced it.
    """

    kind: str
    state: Dict[str, Any] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)
    renormalize: bool = True

    # -- application -------------------------------------------------------
    def apply(self, matrix: np.ndarray) -> np.ndarray:
        out = self._apply_raw(_as_float64(matrix))
        return l2_normalize(out) if self.renormalize else out

    def _apply_raw(self, X: np.ndarray) -> np.ndarray:
        kind = self.kind
        if kind == "raw":
            return X
        if kind == "r1_center":
            return X - self.state["mean"]
        if kind in ("r2_project", "random_direction"):
            d = self.state["direction"]
            return X - np.outer(X @ d, d)
        if kind == "abtt":
            centred = X - self.state["mean"]
            components = self.state["components"]          # (k, dim)
            return centred - (centred @ components.T) @ components
        if kind == "pca_whiten":
            centred = X - self.state["mean"]
            return (centred @ self.state["components"].T) / self.state["scale"]
        if kind == "soft_zca":
            return (X - self.state["mean"]) @ self.state["W"]
        raise ValueError(f"unknown transform kind: {kind!r}")

    # -- persistence -------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            __kind__=np.array(self.kind),
            __params__=np.array(json.dumps(self.params)),
            __renormalize__=np.array(self.renormalize),
            **{k: np.asarray(v) for k, v in self.state.items()},
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "EmbeddingTransform":
        with np.load(Path(path), allow_pickle=False) as data:
            kind = str(data["__kind__"])
            params = json.loads(str(data["__params__"]))
            renormalize = bool(data["__renormalize__"])
            state = {
                key: data[key] for key in data.files
                if not key.startswith("__")
            }
        return cls(kind=kind, state=state, params=params, renormalize=renormalize)

    def describe(self) -> str:
        detail = ", ".join(f"{k}={v}" for k, v in sorted(self.params.items()))
        return f"{self.kind}({detail})" if detail else self.kind

    @property
    def dim(self) -> Optional[int]:
        """Input dimension this transform was fitted for, or None if it is free.

        Only ``raw`` is dimension-free. Every other kind carries a fitted array
        whose trailing axis pins the input width, and applying it to vectors of
        another width is a mistake worth catching at the boundary rather than in
        a matmul several frames down.
        """
        for key in ("mean", "direction"):
            if key in self.state:
                return int(np.asarray(self.state[key]).shape[-1])
        return None


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

def fit_transform(
    kind: str,
    matrix: np.ndarray,
    *,
    k: int = 1,
    alpha: float = 0.1,
    n_components: Optional[int] = None,
    seed: int = 17,
    sample: int = 200_000,
) -> EmbeddingTransform:
    """Fit one rung of the ladder.

    ``matrix`` is the *fitting* corpus, which should not be the evaluation
    corpus: fitting a mean on the data you score is mild, fitting a covariance
    on it is not.
    """
    X = l2_normalize(sample_rows(matrix, sample, seed=seed))
    mu = X.mean(axis=0)
    dim = X.shape[1]

    if kind == "raw":
        return EmbeddingTransform("raw", {}, {})

    if kind == "r1_center":
        # Subtract the mean vector itself. Retains any parallel error in the
        # mean estimate, which is why r2 is expected to edge it out.
        return EmbeddingTransform("r1_center", {"mean": mu}, {})

    if kind == "r2_project":
        direction = mu / max(np.linalg.norm(mu), 1e-12)
        return EmbeddingTransform("r2_project", {"direction": direction}, {})

    if kind == "random_direction":
        # Negative control. If this scores like the r2 arm, the effect is not
        # about the mean direction and the whole result is void.
        rng = np.random.default_rng(seed + 1)
        direction = rng.normal(size=dim)
        direction /= np.linalg.norm(direction)
        return EmbeddingTransform("random_direction", {"direction": direction},
                                  {"seed": seed + 1})

    centred = X - mu

    if kind == "abtt":
        components = _top_components(centred, k)
        return EmbeddingTransform("abtt", {"mean": mu, "components": components},
                                  {"k": k})

    if kind == "pca_whiten":
        n = n_components or dim
        components, variances = _pca(centred, n)
        # Drop directions whose variance is negligible relative to the leading
        # one rather than dividing by a floor. This spectrum spans five orders of
        # magnitude, and rescaling a near-null direction to unit variance is
        # exactly the noise amplification that makes full whitening lose.
        keep = variances > variances.max() * 1e-8
        components, variances = components[keep], variances[keep]
        return EmbeddingTransform(
            "pca_whiten",
            {"mean": mu, "components": components,
             "scale": np.sqrt(variances)},
            {"n_components": int(len(variances)), "requested": int(n)},
        )

    if kind == "soft_zca":
        # Shrinkage toward the identity: alpha=0 is full ZCA whitening, alpha=1
        # leaves the covariance untouched. The regularised middle is what the
        # code-search work found best for retrieval.
        cov = np.cov(centred, rowvar=False)
        cov = (1.0 - alpha) * cov + alpha * np.eye(dim) * np.trace(cov) / dim
        values, vectors = np.linalg.eigh(cov)
        values = np.maximum(values, 1e-12)
        W = vectors @ np.diag(values ** -0.5) @ vectors.T
        return EmbeddingTransform("soft_zca", {"mean": mu, "W": W},
                                  {"alpha": float(alpha)})

    raise ValueError(f"unknown transform kind: {kind!r}")


def _pca(centred: np.ndarray, n_components: int) -> tuple[np.ndarray, np.ndarray]:
    _, singular, vt = np.linalg.svd(centred, full_matrices=False)
    variances = (singular ** 2) / max(len(centred) - 1, 1)
    return vt[:n_components], variances[:n_components]


def _top_components(centred: np.ndarray, k: int) -> np.ndarray:
    components, _ = _pca(centred, max(1, k))
    return components


#: The ladder, in dose order. Names are stable identifiers for result tables.
LADDER: Dict[str, Dict[str, Any]] = {
    "raw":              {"kind": "raw"},
    "r1_center":        {"kind": "r1_center"},
    "r2_project":       {"kind": "r2_project"},
    "random_direction": {"kind": "random_direction"},
    "abtt_k1":          {"kind": "abtt", "k": 1},
    "abtt_k2":          {"kind": "abtt", "k": 2},
    "abtt_k3":          {"kind": "abtt", "k": 3},
    "soft_zca_a50":     {"kind": "soft_zca", "alpha": 0.50},
    "soft_zca_a10":     {"kind": "soft_zca", "alpha": 0.10},
    "soft_zca_a01":     {"kind": "soft_zca", "alpha": 0.01},
    "pca_whiten":       {"kind": "pca_whiten"},
}


def fit_ladder(
    matrix: np.ndarray,
    names: Optional[list[str]] = None,
    *,
    seed: int = 17,
    sample: int = 200_000,
) -> Dict[str, EmbeddingTransform]:
    """Fit every named rung against one fitting corpus."""
    chosen = names or list(LADDER)
    out: Dict[str, EmbeddingTransform] = {}
    for name in chosen:
        spec = dict(LADDER[name])
        kind = spec.pop("kind")
        out[name] = fit_transform(kind, matrix, seed=seed, sample=sample, **spec)
    return out


__all__ = [
    "EmbeddingTransform",
    "LADDER",
    "fit_ladder",
    "fit_transform",
    "isotropy_score",
    "l2_normalize",
    "mean_norm",
    "sample_rows",
]
