import numpy as np
import pytest

from elsa.analyze.shingles import icws_sample_indices, block_shingles


def test_icws_determinism_same_seed_same_output():
    band_ids = np.array([10, 20, 30, 40], dtype=np.uint64)
    weights = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    r = 5
    seed = 12345
    a = icws_sample_indices(band_ids, weights, r, seed)
    b = icws_sample_indices(band_ids, weights, r, seed)
    assert a.dtype == band_ids.dtype
    assert b.dtype == band_ids.dtype
    assert np.array_equal(a, b)


def test_icws_different_seeds_different_outputs():
    band_ids = np.array([1, 2, 3, 4, 5, 6], dtype=np.uint64)
    weights = np.ones_like(band_ids, dtype=np.float64)
    r = 4
    a = icws_sample_indices(band_ids, weights, r, 1)
    c = icws_sample_indices(band_ids, weights, r, 999)
    # With high probability, different seeds yield different tuples
    assert not np.array_equal(a, c)


def test_skipgram_count_matches_offsets():
    # Build simple window_tokens: 10 windows, each with 2 bands
    windows = [[i, i + 1000] for i in range(10)]
    # ICWS path with skip-gram offsets (0,2,5)
    offs = (0, 2, 5)
    sh = block_shingles(
        windows,
        method="icws",
        icws_r=3,
        icws_bbit=0,
        seed=42,
        skipgram_offsets=offs,
    )
    assert len(sh) == max(0, len(windows) - offs[2])


def test_contiguous_vs_skipgram_shape_differs():
    windows = [[i, i + 1000] for i in range(10)]
    sh_contig = block_shingles(windows, k=3, method="xor")
    sh_skip = block_shingles(windows, method="xor", skipgram_offsets=(0, 2, 5))
    # Different construction; sizes should differ on this synthetic input
    assert len(sh_contig) != len(sh_skip)


def test_strand_canonicalization_matches_forward_and_reverse():
    # Build a simple token stream and its reverse
    windows = [[10 + i, 100 + i] for i in range(6)]
    sh_fwd = block_shingles(windows, k=3, method="xor", strand_canonical_shingles=True)
    # Reverse window order simulating opposite strand
    sh_rev = block_shingles(list(reversed(windows)), k=3, method="xor", strand_canonical_shingles=True)
    assert sh_fwd == sh_rev
