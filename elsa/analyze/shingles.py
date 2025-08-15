"""
Utilities to discretize window embeddings to tokens and build order-aware shingles per block.
"""

import numpy as np
import hashlib
from typing import List, Set, Dict


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
    
    # Generate random projection matrix R (d x n_bits) with fixed seed
    rng = np.random.RandomState(seed)
    R = rng.normal(0, 1, size=(d, n_bits))
    
    # L2-normalize columns
    R = R / np.linalg.norm(R, axis=0, keepdims=True)
    
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


def block_shingles(window_tokens: List[List[int]], k: int = 3) -> Set[int]:
    """
    Build order-aware k-gram shingles from per-window token lists.
    
    Args:
        window_tokens: List of per-window token lists (each length = n_bands)
        k: Shingle size (number of consecutive windows)
        
    Returns:
        Set of shingle hash IDs
    """
    if not window_tokens or k <= 0:
        return set()
    
    n_windows = len(window_tokens)
    if n_windows < k:
        return set()
    
    # Derive one stable token per window by XORing all band tokens
    window_sequence = []
    for band_tokens in window_tokens:
        if not band_tokens:
            # If no tokens for this window, use 0
            window_token = 0
        else:
            # XOR all band tokens to get one token per window
            window_token = 0
            for token in band_tokens:
                window_token ^= token
        window_sequence.append(window_token)
    
    # Build k-gram shingles over the ordered token sequence
    shingles = set()
    for i in range(n_windows - k + 1):
        # Create k-gram tuple
        shingle_tuple = tuple(window_sequence[i:i+k])
        
        # Hash the tuple to 64-bit int using blake2b
        shingle_bytes = b''.join(token.to_bytes(8, byteorder='big') for token in shingle_tuple)
        hash_obj = hashlib.blake2b(shingle_bytes, digest_size=8)
        shingle_id = np.frombuffer(hash_obj.digest(), dtype='<u8')[0]
        
        shingles.add(int(shingle_id))
    
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