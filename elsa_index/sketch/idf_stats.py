"""
IDF (Inverse Document Frequency) statistics computation for weighted sketching.

Computes IDF weights from codeword document frequencies to down-weight common patterns.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import Counter, defaultdict
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)


class IDFStatistics:
    """
    Computes and manages IDF statistics for codewords across a corpus.
    """
    
    def __init__(self, max_idf: float = 10.0, smoothing: float = 1.0):
        """
        Initialize IDF statistics.
        
        Args:
            max_idf: Maximum IDF value (clamps very rare terms)
            smoothing: Smoothing factor for IDF computation
        """
        self.max_idf = max_idf
        self.smoothing = smoothing
        
        # Statistics
        self.document_count = 0
        self.codeword_df = Counter()  # Document frequency per codeword
        self.idf_weights = {}
        
    def add_document(self, codeword_assignments: Dict[int, float]) -> None:
        """
        Add a document (window) to the corpus for IDF computation.
        
        Args:
            codeword_assignments: Dict mapping codeword_id -> assignment_weight
        """
        self.document_count += 1
        
        # Count each codeword once per document (regardless of weight)
        for codeword_id in codeword_assignments.keys():
            self.codeword_df[codeword_id] += 1
    
    def compute_idf_weights(self) -> Dict[int, float]:
        """
        Compute IDF weights from collected document frequencies.
        
        Returns:
            Dict mapping codeword_id -> idf_weight
        """
        if self.document_count == 0:
            logger.warning("No documents added for IDF computation")
            return {}
        
        idf_weights = {}
        
        for codeword_id, df in self.codeword_df.items():
            # IDF formula: log((N + smoothing) / (df + smoothing))
            idf = np.log((self.document_count + self.smoothing) / (df + self.smoothing))
            
            # Clamp to maximum value
            idf = min(idf, self.max_idf)
            
            idf_weights[codeword_id] = idf
        
        self.idf_weights = idf_weights
        
        logger.info(f"Computed IDF weights for {len(idf_weights)} codewords")
        logger.info(f"IDF range: {min(idf_weights.values()):.3f} - {max(idf_weights.values()):.3f}")
        
        return idf_weights
    
    def get_statistics(self) -> Dict[str, float]:
        """Get summary statistics about IDF computation."""
        if not self.idf_weights:
            return {}
        
        idf_values = list(self.idf_weights.values())
        df_values = list(self.codeword_df.values())
        
        return {
            'total_documents': self.document_count,
            'unique_codewords': len(self.codeword_df),
            'idf_mean': np.mean(idf_values),
            'idf_std': np.std(idf_values),
            'idf_min': np.min(idf_values),
            'idf_max': np.max(idf_values),
            'df_mean': np.mean(df_values),
            'df_median': np.median(df_values),
            'df_max': np.max(df_values),
            'rare_codewords': sum(1 for df in df_values if df == 1),
            'common_codewords': sum(1 for df in df_values if df > self.document_count * 0.05)
        }
    
    def save(self, filepath: Path) -> None:
        """Save IDF statistics to JSON file."""
        data = {
            'document_count': self.document_count,
            'max_idf': self.max_idf,
            'smoothing': self.smoothing,
            'codeword_df': dict(self.codeword_df),
            'idf_weights': self.idf_weights,
            'statistics': self.get_statistics()
        }
        
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved IDF statistics to {filepath}")
    
    @classmethod
    def load(cls, filepath: Path) -> 'IDFStatistics':
        """Load IDF statistics from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        instance = cls(
            max_idf=data['max_idf'],
            smoothing=data['smoothing']
        )
        
        instance.document_count = data['document_count']
        instance.codeword_df = Counter({int(k): v for k, v in data['codeword_df'].items()})
        instance.idf_weights = {int(k): v for k, v in data['idf_weights'].items()}
        
        logger.info(f"Loaded IDF statistics from {filepath}")
        logger.info(f"  Documents: {instance.document_count}")
        logger.info(f"  Codewords: {len(instance.idf_weights)}")
        
        return instance


def compute_idf_weights(codeword_documents: List[Dict[int, float]], 
                       max_idf: float = 10.0,
                       smoothing: float = 1.0) -> Tuple[Dict[int, float], Dict[str, float]]:
    """
    Convenience function to compute IDF weights from a list of documents.
    
    Args:
        codeword_documents: List of documents, each is Dict[codeword_id -> weight]
        max_idf: Maximum IDF value
        smoothing: Smoothing factor
        
    Returns:
        Tuple of (idf_weights, statistics)
    """
    idf_stats = IDFStatistics(max_idf=max_idf, smoothing=smoothing)
    
    # Process all documents
    for doc in codeword_documents:
        idf_stats.add_document(doc)
    
    # Compute weights
    idf_weights = idf_stats.compute_idf_weights()
    statistics = idf_stats.get_statistics()
    
    return idf_weights, statistics


def filter_by_frequency(codeword_documents: List[Dict[int, float]],
                       min_df: int = 2,
                       max_df_fraction: float = 0.05) -> Tuple[set, Dict[str, int]]:
    """
    Filter codewords by document frequency to remove very rare and very common terms.
    
    Args:
        codeword_documents: List of documents
        min_df: Minimum document frequency (absolute count)
        max_df_fraction: Maximum document frequency (fraction of total documents)
        
    Returns:
        Tuple of (filtered_codeword_ids, filter_statistics)
    """
    # Count document frequencies
    df_counter = Counter()
    total_docs = len(codeword_documents)
    
    for doc in codeword_documents:
        for codeword_id in doc.keys():
            df_counter[codeword_id] += 1
    
    # Apply frequency filters
    max_df = int(total_docs * max_df_fraction)
    filtered_codewords = set()
    
    too_rare = 0
    too_common = 0
    kept = 0
    
    for codeword_id, df in df_counter.items():
        if df < min_df:
            too_rare += 1
        elif df > max_df:
            too_common += 1
        else:
            filtered_codewords.add(codeword_id)
            kept += 1
    
    statistics = {
        'total_codewords': len(df_counter),
        'too_rare': too_rare,
        'too_common': too_common,
        'kept': kept,
        'min_df_threshold': min_df,
        'max_df_threshold': max_df
    }
    
    logger.info(f"Frequency filtering: kept {kept}/{len(df_counter)} codewords")
    logger.info(f"  Too rare (df < {min_df}): {too_rare}")
    logger.info(f"  Too common (df > {max_df}): {too_common}")
    
    return filtered_codewords, statistics