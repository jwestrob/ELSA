"""
Utilities to discretize window embeddings to tokens and build order-aware shingles per block.

Adds an ICWS-r (Consistent Weighted Sampling) per-window combiner and
optional skip-gram shingling pattern while preserving legacy XOR behavior.
"""

import numpy as np
import hashlib
from typing import List, Set, Dict, Iterable, Tuple, Sequence


def _hash_u64_bytes(value: int) -> bytes:
    return int(value).to_bytes(8, byteorder="big", signed=False)


def _derive_seed(base_seed: int, window_index: int, band_tokens: Sequence[int]) -> int:
    """Derive a deterministic per-window seed from base seed, index, and band tokens."""
    h = hashlib.blake2b(digest_size=8)
    h.update(int(base_seed).to_bytes(8, "big", signed=False))
    h.update(int(window_index).to_bytes(8, "big", signed=False))
    for t in band_tokens:
        h.update(_hash_u64_bytes(int(t)))
    return int(np.frombuffer(h.digest(), dtype="<u8")[0])


def icws_sample_indices(
    band_ids: np.ndarray,
    band_weights: np.ndarray,
    r: int,
    seed: int,
) -> np.ndarray:
    """
    Consistent Weighted Sampling (Ioffe) to select r band-ids with probability
    proportional to their weights. Returns an ordered tuple of r selected band_ids.

    Preconditions: len(band_ids) == len(band_weights) > 0, weights > 0.
    Deterministic via numpy.random.Generator seeded from `seed` and sample index.
    """
    band_ids = np.asarray(band_ids)
    band_weights = np.asarray(band_weights, dtype=np.float64)
    if band_ids.ndim != 1 or band_weights.ndim != 1:
        raise ValueError("band_ids and band_weights must be 1D arrays")
    if band_ids.size == 0:
        return np.empty((0,), dtype=band_ids.dtype)
    if band_ids.size != band_weights.size:
        raise ValueError("band_ids and band_weights must have same length")
    if np.any(band_weights <= 0):
        raise ValueError("band_weights must be positive")

    out = np.empty((r,), dtype=band_ids.dtype)
    # Use PCG64 via default_rng with deterministic seeds per sample
    INV_GOLDEN = 0x9E3779B97F4A7C15
    logw = np.log(band_weights)
    for l in range(r):
        sub_seed = (int(seed) ^ ((l + 1) * INV_GOLDEN)) & 0xFFFFFFFFFFFFFFFF
        rng = np.random.default_rng(sub_seed)
        a = rng.gamma(shape=2.0, scale=1.0, size=band_ids.shape[0])
        b = rng.random(band_ids.shape[0])
        c = rng.gamma(shape=2.0, scale=1.0, size=band_ids.shape[0])
        t = np.floor(logw / a + b)
        y = np.exp(a * (t - b))
        key = c / (y * np.exp(a))
        out[l] = band_ids[int(np.argmin(key))]
    return out


_SRP_CACHE: Dict[Tuple[int, int, int], np.ndarray] = {}


def _get_srp_matrix(d: int, n_bits: int, seed: int) -> np.ndarray:
    """Memoize SRP projection matrix by (d, n_bits, seed) to avoid recomputation per block."""
    key = (int(d), int(n_bits), int(seed))
    R = _SRP_CACHE.get(key)
    if R is None:
        rng = np.random.RandomState(seed)
        R = rng.normal(0, 1, size=(d, n_bits))
        R = R / np.linalg.norm(R, axis=0, keepdims=True)
        _SRP_CACHE[key] = R
    return R


def srp_tokens(emb: np.ndarray, *, n_bits: int = 256, n_bands: int = 32, band_bits: int = 8, seed: int = 1337) -> List[List[int]]:
    """
    Generate SRP (Signed Random Projection) tokens from window embeddings.
    
    Args:
        emb: Window embeddings array of shape (n_windows, d)
        n_bits: Total number of projection bits (default 256)
        n_bands: Number of bands to split bits into (default 32)
        band_bits: Bits per band (default 8)
        seed: Random seed for deterministic projection
        
    Returns:
        List of per-window token lists, each containing n_bands tokens
    """
    if emb.ndim != 2:
        raise ValueError(f"Expected 2D array, got {emb.ndim}D")
    
    n_windows, d = emb.shape
    
    if n_bands * band_bits != n_bits:
        raise ValueError(f"n_bands * band_bits ({n_bands} * {band_bits}) must equal n_bits ({n_bits})")
    
    # Get memoized random projection matrix R (d x n_bits) with fixed seed
    R = _get_srp_matrix(d, n_bits, seed)
    
    # Compute sign bits: (emb @ R >= 0) -> bool matrix (n_windows x n_bits)
    bits = (emb @ R >= 0)
    
    # Split into bands and hash each band to 64-bit tokens
    window_tokens = []
    for window_idx in range(n_windows):
        band_tokens = []
        for band_idx in range(n_bands):
            start_bit = band_idx * band_bits
            end_bit = start_bit + band_bits
            
            # Extract band bits for this window
            band_bits_array = bits[window_idx, start_bit:end_bit]
            
            # Convert bool array to bytes for hashing
            band_bytes = np.packbits(band_bits_array, bitorder='big').tobytes()
            
            # Hash to 64-bit int using blake2b
            hash_obj = hashlib.blake2b(band_bytes, digest_size=8)
            token = np.frombuffer(hash_obj.digest(), dtype='<u8')[0]
            band_tokens.append(int(token))
        
        window_tokens.append(band_tokens)
    
    return window_tokens


def block_shingles(
    window_tokens: List[List[int]],
    k: int = 3,
    *,
    method: str = "xor",
    bands_per_window: int = 4,
    band_stride: int = 7,
    fixed_bands: List[int] | None = None,
    # ICWS parameters
    icws_r: int = 8,
    icws_bbit: int = 0,
    icws_weighting: str = "uniform",
    seed: int = 1337,
    # Skip-gram pattern (e.g., (0,2,5)); if None, use contiguous k-grams
    skipgram_offsets: Tuple[int, int, int] | None = None,
    # Orientation handling
    strand_canonical_shingles: bool = False,
) -> Set[int]:
    """
    Build order-aware k-gram shingles from per-window band-token lists.

    Args:
        window_tokens: List of per-window token lists (each length = n_bands)
        k: Shingle size (number of consecutive windows)
        method: How to derive a single token per window before shingling.
            - 'xor' (default): XOR all band tokens in the window (legacy behavior)
            - 'subset': Hash a deterministic subset of bands per window for stability
        bands_per_window: When method='subset', number of bands to include per window
        band_stride: When method='subset', stride to rotate band selection across windows

    Returns:
        Set of shingle hash IDs (or band tokens if method='bandset')
    """
    if not window_tokens or k <= 0 and method != 'bandset':
        return set()

    n_windows = len(window_tokens)
    if method != 'bandset' and n_windows < k:
        return set()

    # Derive one token per window according to method. Tokens can be ints or bytes.
    window_sequence: List[object] = []
    n_bands = len(window_tokens[0]) if window_tokens[0] else 0

    if method == "bandset" and n_bands > 0:
        # Return the union of all band tokens across all windows (order-agnostic)
        bandset = set()
        for band_tokens in window_tokens:
            for t in band_tokens:
                bandset.add(int(t))
        return bandset
    elif method == "fixed_subset" and n_bands > 0:
        # Use a fixed set of band indices for every window to build a stable per-window token
        idxs = fixed_bands if fixed_bands is not None else []
        if not idxs:
            # Default to 4 evenly spaced bands
            idxs = [0, n_bands // 4, n_bands // 2, (3 * n_bands) // 4]
        # Clamp and unique
        idxs = sorted({int(i) % n_bands for i in idxs})
        for i, band_tokens in enumerate(window_tokens):
            if not band_tokens or len(band_tokens) != n_bands:
                window_sequence.append(0)
                continue
            sel = [band_tokens[j] for j in idxs]
            payload = b"".join(int(t).to_bytes(8, byteorder="big") for t in sel)
            h = hashlib.blake2b(payload, digest_size=8).digest()
            token = int(np.frombuffer(h, dtype="<u8")[0])
            window_sequence.append(token)
    elif method == "subset" and n_bands > 0 and bands_per_window > 0:
        # Deterministic rotating subset of bands per window index
        bpw = max(1, min(bands_per_window, n_bands))
        stride = max(1, band_stride)
        for i, band_tokens in enumerate(window_tokens):
            if not band_tokens or len(band_tokens) != n_bands:
                window_sequence.append(0)
                continue
            # Choose bpw bands starting at offset (i*stride) mod n_bands
            start = (i * stride) % n_bands
            idxs = [(start + j) % n_bands for j in range(bpw)]
            sel = [band_tokens[j] for j in idxs]
            # Hash the selected band tokens into one 64-bit token
            payload = b"".join(int(t).to_bytes(8, byteorder="big") for t in sel)
            h = hashlib.blake2b(payload, digest_size=8).digest()
            token = int(np.frombuffer(h, dtype="<u8")[0])
            window_sequence.append(token)
    elif method == "icws" and n_bands > 0 and icws_r > 0:
        # ICWS-r ordered tuple per window, optionally b-bit packed
        r = int(icws_r)
        mask = (1 << int(icws_bbit)) - 1 if int(icws_bbit) > 0 else None
        for i, band_tokens in enumerate(window_tokens):
            if not band_tokens or len(band_tokens) != n_bands:
                window_sequence.append(b"\x00" * 8)
                continue
            band_arr = np.asarray(band_tokens, dtype=np.uint64)
            if icws_weighting == "uniform":
                weights = np.ones_like(band_arr, dtype=np.float64)
            else:
                # Fallback to uniform for unknown weighting modes
                weights = np.ones_like(band_arr, dtype=np.float64)
            w_seed = _derive_seed(seed, i, band_tokens)
            samples = icws_sample_indices(band_arr, weights, r, w_seed)
            if mask is not None:
                samples = samples & np.uint64(mask)
            # Serialize ordered r-tuple deterministically as bytes
            payload = b"".join(_hash_u64_bytes(int(x)) for x in samples)
            # Store bytes to preserve order; shingles will hash these bytes
            window_sequence.append(payload)
    else:
        # Legacy: XOR all band tokens per window
        for band_tokens in window_tokens:
            if not band_tokens:
                window_sequence.append(0)
                continue
            token = 0
            for t in band_tokens:
                token ^= int(t)
            window_sequence.append(token)

    # Build shingles over the ordered token sequence
    shingles: Set[int] = set()
    if skipgram_offsets is not None:
        # Validate offsets: 3-tuple with first == 0, nondecreasing
        if not (isinstance(skipgram_offsets, tuple) and len(skipgram_offsets) == 3):
            raise ValueError("skipgram_offsets must be a 3-tuple (o0,o1,o2)")
        o0, o1, o2 = skipgram_offsets
        if o0 != 0 or o1 < o0 or o2 < o1:
            raise ValueError("skipgram offsets must satisfy o0==0 and o0<=o1<=o2")
        max_off = o2
        if n_windows >= (max_off + 1):
            for i in range(0, n_windows - max_off):
                idxs = (i + o0, i + o1, i + o2)
                toks = window_sequence[idxs[0]], window_sequence[idxs[1]], window_sequence[idxs[2]]
                # Serialize: handle int and bytes
                parts: List[bytes] = []
                for tok in toks:
                    if isinstance(tok, (bytes, bytearray)):
                        parts.append(bytes(tok))
                    else:
                        parts.append(_hash_u64_bytes(int(tok)))
                shingle_bytes = b"".join(parts)
                if strand_canonical_shingles:
                    # Reverse the tuple orientation and take canonical bytes
                    rev_bytes = b"".join(reversed(parts))
                    if rev_bytes < shingle_bytes:
                        shingle_bytes = rev_bytes
                h = hashlib.blake2b(shingle_bytes, digest_size=8).digest()
                shingle_id = int(np.frombuffer(h, dtype="<u8")[0])
                shingles.add(shingle_id)
    else:
        # Contiguous k-grams over ints/bytes
        if method != 'bandset' and n_windows >= k:
            for i in range(n_windows - k + 1):
                shingle_tuple = tuple(window_sequence[i:i + k])
                parts: List[bytes] = []
                for tok in shingle_tuple:
                    if isinstance(tok, (bytes, bytearray)):
                        parts.append(bytes(tok))
                    else:
                        parts.append(_hash_u64_bytes(int(tok)))
                shingle_bytes = b"".join(parts)
                if strand_canonical_shingles:
                    rev_bytes = b"".join(reversed(parts))
                    if rev_bytes < shingle_bytes:
                        shingle_bytes = rev_bytes
                hash_obj = hashlib.blake2b(shingle_bytes, digest_size=8)
                shingle_id = int(np.frombuffer(hash_obj.digest(), dtype="<u8")[0])
                shingles.add(shingle_id)

    return shingles


def df_filter(shingle_df: Dict[int, int], df_max: int, S: Set[int]) -> Set[int]:
    """
    Filter out high document frequency shingles from a shingle set.
    
    Args:
        shingle_df: Dictionary mapping shingle_id -> document frequency
        df_max: Maximum allowed document frequency
        S: Set of shingle IDs to filter
        
    Returns:
        Filtered set with high-DF shingles removed
    """
    return {s for s in S if shingle_df.get(s, 0) <= df_max}


def jaccard(A: Set[int], B: Set[int]) -> float:
    """
    Compute Jaccard similarity between two sets.
    
    Args:
        A, B: Sets to compare
        
    Returns:
        Jaccard similarity J(A,B) = |A ∩ B| / |A ∪ B|
    """
    if not A and not B:
        return 1.0
    
    intersection = len(A & B)
    union = len(A | B)
    
    if union == 0:
        return 0.0
    
    return intersection / union
