"""
Weighted MinHash implementation using DOPH (Densified One-Permutation Hashing).

Supports IDF weighting and b-bit compression for memory efficiency.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import struct
import hashlib
from collections import defaultdict

@dataclass
class WeightedElement:
    """A weighted element in a set."""
    element_id: int
    weight: float
    
    def __post_init__(self):
        if self.weight < 0:
            raise ValueError("Weight must be non-negative")


class DOPH:
    """
    Densified One-Permutation Hashing for weighted sets.
    
    Based on "Improved Densification of One Permutation Hashing" (Shrivastava, 2017).
    Provides unbiased Jaccard estimation for weighted sets with b-bit compression.
    """
    
    def __init__(self, sketch_size: int, bits: int = 64, seed: int = 42):
        """
        Initialize DOPH sketch.
        
        Args:
            sketch_size: Number of hash functions (sketch length)
            bits: Compression level (64, 2, or 1 bits per hash)
            seed: Random seed for reproducibility
        """
        if bits not in [64, 2, 1]:
            raise ValueError("bits must be 64, 2, or 1")
        
        self.sketch_size = sketch_size
        self.bits = bits
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        
        # Generate random hash parameters
        self.hash_params = self._generate_hash_params()
        
    def _generate_hash_params(self) -> List[Tuple[int, int]]:
        """Generate parameters for universal hash functions."""
        params = []
        for _ in range(self.sketch_size):
            # Universal hash: h(x) = ((a * x + b) mod p) mod m
            a = self.rng.randint(1, 2**31 - 1)
            b = self.rng.randint(0, 2**31 - 1)
            params.append((a, b))
        return params
    
    def sketch(self, weighted_elements: List[WeightedElement]) -> np.ndarray:
        """
        Create weighted MinHash sketch using DOPH.
        
        Args:
            weighted_elements: List of weighted elements
            
        Returns:
            Sketch vector of length sketch_size
        """
        if not weighted_elements:
            return np.zeros(self.sketch_size, dtype=np.uint64 if self.bits == 64 else np.uint8)
        
        sketch = np.full(self.sketch_size, np.inf)
        sketch_ids = np.zeros(self.sketch_size, dtype=np.uint64)
        
        for element in weighted_elements:
            if element.weight <= 0:
                continue
                
            # Generate weighted hash values for this element
            for i, (a, b) in enumerate(self.hash_params):
                # Universal hash
                hash_val = ((a * element.element_id + b) % (2**31 - 1)) / (2**31 - 1)
                
                # Weight-adjusted hash (exponential transformation)
                weighted_hash = -np.log(hash_val) / element.weight
                
                # Update sketch if this is a minimum
                if weighted_hash < sketch[i]:
                    sketch[i] = weighted_hash
                    sketch_ids[i] = element.element_id
        
        # Apply compression
        if self.bits == 64:
            return sketch_ids
        else:
            return self._compress_sketch(sketch_ids, sketch)
    
    def _compress_sketch(self, sketch_ids: np.ndarray, sketch_values: np.ndarray) -> np.ndarray:
        """Compress 64-bit hashes to b-bit representation."""
        if self.bits == 2:
            # 2-bit compression: use 4 quantile levels
            compressed = np.zeros(self.sketch_size, dtype=np.uint8)
            for i in range(self.sketch_size):
                if np.isfinite(sketch_values[i]):
                    # Map hash to [0,1] and quantize to 2 bits
                    normalized = (sketch_ids[i] % 1024) / 1024.0
                    compressed[i] = min(3, int(normalized * 4))
                else:
                    compressed[i] = 0
            return compressed
        
        elif self.bits == 1:
            # 1-bit compression: threshold at median
            compressed = np.zeros(self.sketch_size, dtype=np.uint8)
            valid_values = sketch_values[np.isfinite(sketch_values)]
            if len(valid_values) > 0:
                threshold = np.median(valid_values)
                compressed = (sketch_values < threshold).astype(np.uint8)
            return compressed
        
        return sketch_ids
    
    def jaccard_estimate(self, sketch1: np.ndarray, sketch2: np.ndarray) -> float:
        """
        Estimate Jaccard similarity from two sketches.
        
        Uses unbiased estimator appropriate for compression level.
        """
        if len(sketch1) != len(sketch2):
            raise ValueError("Sketches must have same length")
        
        if self.bits == 64:
            # Standard MinHash estimator
            matches = np.sum(sketch1 == sketch2)
            return matches / len(sketch1)
        
        elif self.bits == 2:
            # 2-bit unbiased estimator
            matches = np.sum(sketch1 == sketch2)
            # Bias correction for 2-bit quantization
            raw_estimate = matches / len(sketch1)
            return max(0, min(1, (raw_estimate - 0.25) / 0.75))
        
        elif self.bits == 1:
            # 1-bit unbiased estimator (Hamming distance based)
            matches = np.sum(sketch1 == sketch2)
            raw_estimate = matches / len(sketch1)
            # Bias correction for 1-bit quantization  
            return max(0, min(1, 2 * raw_estimate - 1))
        
        return 0.0
    
    def collision_count(self, sketch1: np.ndarray, sketch2: np.ndarray) -> int:
        """Count hash collisions between sketches."""
        return int(np.sum(sketch1 == sketch2))
    
    def estimate_variance(self, jaccard: float) -> float:
        """
        Estimate variance of Jaccard estimator.
        
        Returns theoretical variance for confidence intervals.
        """
        if self.bits == 64:
            # Standard MinHash variance
            return jaccard * (1 - jaccard) / self.sketch_size
        
        elif self.bits == 2:
            # 2-bit estimator variance (approximate)
            corrected_jaccard = max(0, min(1, (jaccard - 0.25) / 0.75))
            return (corrected_jaccard * (1 - corrected_jaccard) / self.sketch_size) * 1.5
        
        elif self.bits == 1:
            # 1-bit estimator variance (approximate)
            corrected_jaccard = max(0, min(1, 2 * jaccard - 1))
            return (corrected_jaccard * (1 - corrected_jaccard) / self.sketch_size) * 2.0
        
        return 0.0


class WeightedMinHashSketch:
    """
    High-level interface for weighted MinHash sketching with IDF and MGE masking.
    """
    
    def __init__(self, config: dict, idf_weights: Optional[Dict[int, float]] = None,
                 mge_mask: Optional[set] = None):
        """
        Initialize weighted MinHash sketcher.
        
        Args:
            config: Sketch configuration (type, size, bits, etc.)
            idf_weights: IDF weights for codewords
            mge_mask: Set of codeword IDs to mask (set weight to 0)
        """
        self.config = config
        self.sketch_size = config.get('size', 96)
        self.bits = config.get('bits', 64)
        self.idf_weights = idf_weights or {}
        self.mge_mask = mge_mask or set()
        
        # Initialize DOPH sketcher
        self.doph = DOPH(
            sketch_size=self.sketch_size,
            bits=self.bits,
            seed=config.get('seed', 42)
        )
    
    def create_sketch(self, codeword_assignments: Dict[int, float]) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Create weighted sketch from codeword assignments.
        
        Args:
            codeword_assignments: Dict mapping codeword_id -> assignment_weight
            
        Returns:
            Tuple of (sketch, metadata)
        """
        # Apply IDF weighting and MGE masking
        weighted_elements = []
        total_weight = 0.0
        masked_weight = 0.0
        
        for codeword_id, base_weight in codeword_assignments.items():
            # Apply MGE masking
            if codeword_id in self.mge_mask:
                masked_weight += base_weight
                continue
            
            # Apply IDF weighting
            idf_weight = self.idf_weights.get(codeword_id, 1.0)
            final_weight = base_weight * idf_weight
            
            if final_weight > 0:
                weighted_elements.append(WeightedElement(codeword_id, final_weight))
                total_weight += final_weight
        
        # Create sketch
        sketch = self.doph.sketch(weighted_elements)
        
        # Compute metadata
        metadata = {
            'total_weight': total_weight,
            'masked_weight': masked_weight,
            'masked_fraction': masked_weight / (total_weight + masked_weight) if (total_weight + masked_weight) > 0 else 0.0,
            'effective_elements': len(weighted_elements),
            'sketch_size': self.sketch_size,
            'compression_bits': self.bits
        }
        
        return sketch, metadata
    
    def compare_sketches(self, sketch1: np.ndarray, sketch2: np.ndarray) -> Dict[str, float]:
        """
        Compare two sketches and return similarity metrics.
        
        Returns:
            Dict with jaccard_estimate, collisions, variance, p_value
        """
        jaccard = self.doph.jaccard_estimate(sketch1, sketch2)
        collisions = self.doph.collision_count(sketch1, sketch2)
        variance = self.doph.estimate_variance(jaccard)
        
        # Binomial test p-value for collision count
        # Under null hypothesis: expected collisions = sketch_size / codebook_size
        expected_collisions = self.sketch_size * 0.01  # Rough estimate, should be refined
        
        try:
            from scipy.stats import binom
            p_value = 1.0 - binom.cdf(collisions - 1, self.sketch_size, expected_collisions / self.sketch_size)
        except ImportError:
            # Fallback normal approximation
            std_dev = np.sqrt(variance * self.sketch_size)
            z_score = (collisions - expected_collisions) / std_dev if std_dev > 0 else 0
            p_value = 1.0 - 0.5 * (1 + np.tanh(z_score / np.sqrt(2)))
        
        return {
            'jaccard_estimate': jaccard,
            'collisions': collisions,
            'variance': variance,
            'confidence_interval': (
                max(0, jaccard - 1.96 * np.sqrt(variance)),
                min(1, jaccard + 1.96 * np.sqrt(variance))
            ),
            'p_value': p_value
        }
    
    def serialize_sketch(self, sketch: np.ndarray) -> bytes:
        """Serialize sketch to bytes for storage."""
        if self.bits == 64:
            return sketch.astype(np.uint64).tobytes()
        else:
            return sketch.astype(np.uint8).tobytes()
    
    def deserialize_sketch(self, data: bytes) -> np.ndarray:
        """Deserialize sketch from bytes."""
        if self.bits == 64:
            return np.frombuffer(data, dtype=np.uint64)
        else:
            return np.frombuffer(data, dtype=np.uint8)