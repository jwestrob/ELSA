"""The anchor path swaps strings for integer codes. Pin what that relies on.

Three substitutions make `find_cross_genome_anchors` faster, and each is only
safe because of a property that is invisible at the call site:

* the cross-genome test compares category codes instead of genome strings,
* the canonical query/target swap orders by code instead of by string,
* the pair dedup sorts one packed int64 instead of a two-field structured dtype.

A change to `_category_codes` that stopped sorting categories, or a corpus large
enough to overflow the packed key, would break these silently — the pipeline
would still produce anchors, just the wrong ones.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from elsa.seed import _category_codes


def test_codes_order_exactly_as_the_strings_do():
    """The canonical swap uses `>` on codes where it used to use `>` on names."""
    rng = np.random.default_rng(0)
    names = np.array([f"GC{'AF'[i % 2]}_{rng.integers(10**8):09d}.{i % 3}"
                      for i in range(500)])
    codes, categories = _category_codes(names)

    np.testing.assert_array_equal(categories[codes], names)
    a, b = rng.integers(0, len(names), 4000), rng.integers(0, len(names), 4000)
    np.testing.assert_array_equal(codes[a] > codes[b], names[a] > names[b])
    np.testing.assert_array_equal(codes[a] != codes[b], names[a] != names[b])


def test_codes_order_matches_strings_for_categorical_input():
    """Store-backed runs arrive with the column already categorical, and pandas
    does not promise those categories are sorted."""
    values = pd.Categorical(["gamma", "alpha", "beta", "alpha"],
                            categories=["gamma", "beta", "alpha"])
    codes, categories = _category_codes(values)

    np.testing.assert_array_equal(categories, ["alpha", "beta", "gamma"])
    np.testing.assert_array_equal(categories[codes],
                                  ["gamma", "alpha", "beta", "alpha"])
    assert codes[1] < codes[2] < codes[0]


def test_from_codes_reproduces_the_string_built_column():
    names = np.array(["b", "a", "c", "a", "b"])
    codes, categories = _category_codes(names)
    rows = np.array([4, 0, 2, 1])

    built = pd.Categorical.from_codes(codes[rows], categories=categories,
                                      ordered=True)
    expected = pd.Categorical(names[rows], categories=sorted(np.unique(names)),
                              ordered=True)
    assert built.equals(expected)


def test_packed_pair_key_deduplicates_like_the_structured_dtype():
    rng = np.random.default_rng(17)
    n = 40_000
    lo = rng.integers(0, n, 200_000).astype(np.int32)
    hi = rng.integers(0, n, 200_000).astype(np.int32)
    lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)

    structured = np.empty(len(lo), dtype=[("lo", np.int32), ("hi", np.int32)])
    structured["lo"], structured["hi"] = lo, hi
    _, by_struct = np.unique(structured, return_index=True)

    packed = (lo.astype(np.int64) << np.int64(32)) | hi.astype(np.int64)
    _, by_packed = np.unique(packed, return_index=True)

    np.testing.assert_array_equal(np.sort(by_struct), np.sort(by_packed))


def test_packed_key_stays_positive_at_the_guarded_limit():
    """The pack is used while n < 2**31. At that ceiling the shifted high half
    must stay inside a signed int64, or dedup silently groups unrelated pairs."""
    limit = 2_147_483_648
    lo = np.array([limit - 2, 0, 12345], dtype=np.int64)
    hi = np.array([limit - 1, 1, 999_999], dtype=np.int64)

    packed = (lo << np.int64(32)) | hi
    assert (packed > 0).all()
    np.testing.assert_array_equal(packed >> np.int64(32), lo)
    np.testing.assert_array_equal(packed & np.int64(0xFFFFFFFF), hi)


# ---------------------------------------------------------------------------
# Backend selection: 'auto' used to mean "spend 28% of the run on a graph"
# ---------------------------------------------------------------------------

def test_auto_prefers_ivfflat_at_scale():
    """At DPANN scale an HNSW graph costs 253s against 9.1s for IVF-Flat, and
    the two retrieve equivalently (recall@60 0.9916 vs 0.9913)."""
    from elsa.index import _resolve_backend

    assert _resolve_backend("auto", 3_395_472) == "faiss_ivfflat"
    assert _resolve_backend("auto", 200_000) == "faiss_ivfflat"


def test_auto_prefers_exact_search_on_small_corpora():
    """Below the threshold an exact index builds faster *and* is exact, so
    there is nothing an approximate one can win."""
    from elsa.index import _resolve_backend

    assert _resolve_backend("auto", 1_000) == "faiss_flat"
    assert _resolve_backend("auto", 49_999) == "faiss_flat"


def test_explicit_backend_is_never_overridden():
    from elsa.index import _resolve_backend

    for backend in ("hnsw", "faiss_flat", "faiss_ivfpq", "sklearn"):
        assert _resolve_backend(backend, 5_000_000) == backend


def test_pipeline_default_matches_the_validated_config():
    """`ChainConfig` here is what the Python API and the benchmark scripts
    construct; it drifted from params.py and silently selected hnswlib."""
    from elsa.analyze.pipeline import ChainConfig
    from elsa.params import ChainConfig as ValidatedChainConfig

    assert ChainConfig().index_backend == ValidatedChainConfig().index_backend
    assert ChainConfig().index_backend == "faiss_ivfflat"
