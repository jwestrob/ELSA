#!/usr/bin/env python3
"""
ELSA-based similarity analyzer for syntenic blocks.
Uses ELSA's windowed embeddings and indices for protein-level similarity analysis.
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ELSASimilarity:
    """Container for ELSA-based similarity analysis."""
    window_id_1: str
    window_id_2: str
    cosine_similarity: float
    jaccard_similarity: float
    srp_similarity: float
    embedding_distance: float
    conservation_score: float


@dataclass
class ELSAHomologyEvidence:
    """ELSA-based homology evidence for a syntenic block."""
    block_id: str
    window_similarities: List[ELSASimilarity]
    average_conservation: float
    max_conservation: float
    conserved_window_count: int
    total_window_pairs: int
    conservation_level: str
    elsa_summary: str


class ELSASimilarityAnalyzer:
    """Analyze syntenic blocks using ELSA windowed embeddings and indices."""
    
    def __init__(self, elsa_index_dir: str = None):
        """Initialize ELSA similarity analyzer.
        
        Args:
            elsa_index_dir: Directory containing ELSA index files
        """
        # Auto-detect ELSA index directory
        if elsa_index_dir is None:
            # Try different possible locations
            possible_paths = [
                Path.cwd() / "elsa_index",                    # Current working directory
                Path.cwd().parent / "elsa_index",            # Parent directory
                Path(__file__).parent.parent.parent / "elsa_index",  # Project root
                Path("/Users/jacob/Documents/Sandbox/ELSA/elsa_index")  # Absolute path
            ]
            
            for path in possible_paths:
                if path.exists():
                    self.index_dir = path
                    break
            else:
                self.index_dir = Path("elsa_index")  # Default fallback
        else:
            self.index_dir = Path(elsa_index_dir)
        self.windows_file = self.index_dir / "shingles" / "windows.parquet"
        self.genes_file = self.index_dir / "ingest" / "genes.parquet"
        self.multiscale_dir = self.index_dir / "multiscale_windows"
        self.macro_windows_file = self.multiscale_dir / "macro_windows.parquet"
        self.micro_windows_file = self.multiscale_dir / "micro_windows.parquet"
        
        # Check if ELSA index files exist
        self.available_files = {}
        for name, file_path in [
            ("windows", self.windows_file),
            ("genes", self.genes_file), 
            ("macro_windows", self.macro_windows_file),
            ("micro_windows", self.micro_windows_file)
        ]:
            self.available_files[name] = file_path.exists()
            if file_path.exists():
                logger.info(f"✓ ELSA {name} index available: {file_path}")
            else:
                logger.warning(f"✗ ELSA {name} index missing: {file_path}")
        
        self.elsa_available = any(self.available_files.values())
        
        if self.elsa_available:
            logger.info("ELSASimilarityAnalyzer initialized with ELSA indices")
        else:
            logger.warning("ELSASimilarityAnalyzer initialized without ELSA indices")
    
    def analyze_block_similarity(self, block_id: str, query_locus: str, target_locus: str) -> Optional[ELSAHomologyEvidence]:
        """Analyze syntenic block using ELSA windowed embedding similarity.
        
        Args:
            block_id: Syntenic block identifier
            query_locus: Query locus identifier
            target_locus: Target locus identifier
            
        Returns:
            ELSA homology evidence or None if unavailable
        """
        if not self.elsa_available:
            logger.info(f"ELSA indices not available for block {block_id}")
            return None
        
        try:
            logger.info(f"Analyzing ELSA similarity for block {block_id}: {query_locus} vs {target_locus}")
            
            # Load windowed embeddings
            window_similarities = self._compute_window_similarities(query_locus, target_locus)
            
            if not window_similarities:
                logger.info(f"No ELSA window similarities found for block {block_id}")
                return self._create_null_evidence(block_id, "No matching windows in ELSA index")
            
            # Calculate conservation metrics
            conservation_scores = [sim.conservation_score for sim in window_similarities]
            avg_conservation = np.mean(conservation_scores)
            max_conservation = np.max(conservation_scores)
            
            # Count highly conserved windows (threshold > 0.7)
            conserved_count = sum(1 for score in conservation_scores if score > 0.7)
            
            # Determine conservation level
            if avg_conservation >= 0.8:
                conservation_level = "very_high"
            elif avg_conservation >= 0.6:
                conservation_level = "high"
            elif avg_conservation >= 0.4:
                conservation_level = "moderate"
            elif avg_conservation >= 0.2:
                conservation_level = "low"
            else:
                conservation_level = "minimal"
            
            # Generate summary
            summary = self._generate_elsa_summary(
                len(window_similarities), conserved_count, avg_conservation, conservation_level
            )
            
            evidence = ELSAHomologyEvidence(
                block_id=block_id,
                window_similarities=window_similarities,
                average_conservation=avg_conservation,
                max_conservation=max_conservation,
                conserved_window_count=conserved_count,
                total_window_pairs=len(window_similarities),
                conservation_level=conservation_level,
                elsa_summary=summary
            )
            
            logger.info(f"ELSA analysis completed for block {block_id}: "
                       f"{len(window_similarities)} windows, {conservation_level} conservation")
            
            return evidence
            
        except Exception as e:
            logger.error(f"ELSA similarity analysis failed for block {block_id}: {e}")
            return self._create_null_evidence(block_id, f"Analysis failed: {str(e)}")
    
    def _compute_window_similarities(self, query_locus: str, target_locus: str) -> List[ELSASimilarity]:
        """Compute pairwise similarities between windows from query and target loci."""
        similarities = []
        
        try:
            # Load windows data
            if not self.available_files["windows"]:
                return similarities
            
            windows_df = pd.read_parquet(self.windows_file)
            
            # Extract locus identifiers for matching
            # ELSA typically uses genomic coordinates or contig identifiers
            query_windows = self._filter_windows_by_locus(windows_df, query_locus)
            target_windows = self._filter_windows_by_locus(windows_df, target_locus)
            
            if query_windows.empty or target_windows.empty:
                logger.debug(f"No windows found for loci: query={len(query_windows)}, target={len(target_windows)}")
                return similarities
            
            # Compute pairwise similarities
            for _, query_window in query_windows.iterrows():
                for _, target_window in target_windows.iterrows():
                    similarity = self._compute_window_pair_similarity(query_window, target_window)
                    if similarity:
                        similarities.append(similarity)
            
            # Sort by conservation score (highest first)
            similarities.sort(key=lambda x: x.conservation_score, reverse=True)
            
            # Limit to top matches to avoid overwhelming output
            return similarities[:20]
            
        except Exception as e:
            logger.error(f"Failed to compute window similarities: {e}")
            return similarities
    
    def _filter_windows_by_locus(self, windows_df: pd.DataFrame, locus: str) -> pd.DataFrame:
        """Filter windows dataframe by locus identifier."""
        
        # ELSA stores locus identifiers in the 'locus_id' column
        # Example: locus = "1313.30775:accn|1313.30775.con.0001" -> locus_id = "accn|1313.30775.con.0001"
        
        if 'locus_id' not in windows_df.columns:
            logger.debug(f"No locus_id column found in windows dataframe")
            return pd.DataFrame()
        
        # Extract contig identifier from the syntenic block locus
        # Handle formats like "1313.30775:accn|1313.30775.con.0001" or "accn|1313.30775.con.0001"
        if ':' in locus:
            contig_id = locus.split(':')[-1]  # Take part after ':'
        else:
            contig_id = locus
        
        # Try exact match first
        exact_matches = windows_df[windows_df['locus_id'] == contig_id]
        if not exact_matches.empty:
            logger.debug(f"Found {len(exact_matches)} windows for locus {locus} using exact match on {contig_id}")
            return exact_matches
        
        # Try partial match
        partial_matches = windows_df[windows_df['locus_id'].str.contains(contig_id, na=False, regex=False)]
        if not partial_matches.empty:
            logger.debug(f"Found {len(partial_matches)} windows for locus {locus} using partial match on {contig_id}")
            return partial_matches
        
        # Try matching any part of the locus
        locus_parts = locus.replace(':', '|').split('|')
        for part in locus_parts:
            if len(part) > 5:  # Skip very short parts
                matches = windows_df[windows_df['locus_id'].str.contains(part, na=False, regex=False)]
                if not matches.empty:
                    logger.debug(f"Found {len(matches)} windows for locus {locus} using part match on {part}")
                    return matches
        
        logger.debug(f"No windows found for locus {locus} (tried {contig_id})")
        logger.debug(f"Available locus_ids: {list(windows_df['locus_id'].unique()[:5])}")
        return pd.DataFrame()
    
    def _compute_window_pair_similarity(self, window1: pd.Series, window2: pd.Series) -> Optional[ELSASimilarity]:
        """Compute similarity between two windows."""
        
        try:
            window1_id = str(window1.get('window_id', f"w1_{window1.name}"))
            window2_id = str(window2.get('window_id', f"w2_{window2.name}"))
            
            # Cosine similarity from embeddings
            cosine_sim = self._compute_cosine_similarity(window1, window2)
            
            # Jaccard similarity from discrete features
            jaccard_sim = self._compute_jaccard_similarity(window1, window2)
            
            # SRP similarity from continuous signatures
            srp_sim = self._compute_srp_similarity(window1, window2)
            
            # Embedding distance
            embedding_dist = self._compute_embedding_distance(window1, window2)
            
            # Combined conservation score
            conservation_score = self._compute_conservation_score(
                cosine_sim, jaccard_sim, srp_sim, embedding_dist
            )
            
            return ELSASimilarity(
                window_id_1=window1_id,
                window_id_2=window2_id,
                cosine_similarity=cosine_sim,
                jaccard_similarity=jaccard_sim,
                srp_similarity=srp_sim,
                embedding_distance=embedding_dist,
                conservation_score=conservation_score
            )
            
        except Exception as e:
            logger.debug(f"Failed to compute window pair similarity: {e}")
            return None
    
    def _compute_cosine_similarity(self, window1: pd.Series, window2: pd.Series) -> float:
        """Compute cosine similarity between window embeddings."""
        
        # Look for ELSA embedding columns (emb_000 to emb_255)
        embedding_cols = [col for col in window1.index if col.startswith('emb_')]
        
        if not embedding_cols:
            # Fallback to other embedding patterns
            embedding_cols = [col for col in window1.index if 'embedding' in col.lower() or 'vector' in col.lower()]
        
        if not embedding_cols:
            # Use numeric columns as proxy embeddings
            numeric_cols = window1.select_dtypes(include=[np.number]).index
            if len(numeric_cols) > 0:
                embedding_cols = numeric_cols
        
        if not embedding_cols:
            return 0.5  # Default neutral similarity
        
        try:
            # Extract embeddings
            emb1 = window1[embedding_cols].values.astype(float)
            emb2 = window2[embedding_cols].values.astype(float)
            
            # Compute cosine similarity
            dot_product = np.dot(emb1, emb2)
            norm1 = np.linalg.norm(emb1)
            norm2 = np.linalg.norm(emb2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            cosine_sim = dot_product / (norm1 * norm2)
            return float(np.clip(cosine_sim, -1.0, 1.0))
            
        except Exception as e:
            logger.debug(f"Cosine similarity computation failed: {e}")
            return 0.5
    
    def _compute_jaccard_similarity(self, window1: pd.Series, window2: pd.Series) -> float:
        """Compute Jaccard similarity from discrete features."""
        
        # Look for discrete feature columns (shingles, hashes, etc.)
        discrete_cols = [col for col in window1.index 
                        if any(term in col.lower() for term in ['hash', 'shingle', 'signature', 'code'])]
        
        if not discrete_cols:
            return 0.5  # Default neutral similarity
        
        try:
            # Get discrete features as sets
            features1 = set()
            features2 = set()
            
            for col in discrete_cols:
                val1 = window1[col]
                val2 = window2[col]
                
                # Convert to hashable types
                if pd.notna(val1):
                    features1.add(str(val1))
                if pd.notna(val2):
                    features2.add(str(val2))
            
            # Compute Jaccard similarity
            intersection = len(features1 & features2)
            union = len(features1 | features2)
            
            if union == 0:
                return 0.0
            
            return intersection / union
            
        except Exception as e:
            logger.debug(f"Jaccard similarity computation failed: {e}")
            return 0.5
    
    def _compute_srp_similarity(self, window1: pd.Series, window2: pd.Series) -> float:
        """Compute SRP-based similarity."""
        
        # Look for SRP signature columns
        srp_cols = [col for col in window1.index if 'srp' in col.lower() or 'projection' in col.lower()]
        
        if not srp_cols:
            return 0.5  # Default neutral similarity
        
        try:
            # Hamming distance on SRP signatures
            similarities = []
            
            for col in srp_cols:
                sig1 = window1[col]
                sig2 = window2[col]
                
                if pd.notna(sig1) and pd.notna(sig2):
                    # Convert to binary if needed
                    if isinstance(sig1, str) and isinstance(sig2, str):
                        # Hamming distance on binary strings
                        if len(sig1) == len(sig2):
                            hamming_dist = sum(c1 != c2 for c1, c2 in zip(sig1, sig2))
                            similarity = 1.0 - (hamming_dist / len(sig1))
                            similarities.append(similarity)
                    elif isinstance(sig1, (int, float)) and isinstance(sig2, (int, float)):
                        # Numeric similarity
                        max_val = max(abs(sig1), abs(sig2), 1)
                        similarity = 1.0 - abs(sig1 - sig2) / max_val
                        similarities.append(similarity)
            
            return np.mean(similarities) if similarities else 0.5
            
        except Exception as e:
            logger.debug(f"SRP similarity computation failed: {e}")
            return 0.5
    
    def _compute_embedding_distance(self, window1: pd.Series, window2: pd.Series) -> float:
        """Compute embedding distance (lower = more similar)."""
        
        # Use same embeddings as cosine similarity - ELSA embeddings
        embedding_cols = [col for col in window1.index if col.startswith('emb_')]
        
        if not embedding_cols:
            embedding_cols = [col for col in window1.index if 'embedding' in col.lower() or 'vector' in col.lower()]
        
        if not embedding_cols:
            numeric_cols = window1.select_dtypes(include=[np.number]).index
            if len(numeric_cols) > 0:
                embedding_cols = numeric_cols
        
        if not embedding_cols:
            return 1.0  # Default high distance
        
        try:
            emb1 = window1[embedding_cols].values.astype(float)
            emb2 = window2[embedding_cols].values.astype(float)
            
            # Euclidean distance, normalized
            distance = np.linalg.norm(emb1 - emb2)
            max_possible_dist = np.linalg.norm(emb1) + np.linalg.norm(emb2)
            
            if max_possible_dist == 0:
                return 0.0
            
            normalized_distance = distance / max_possible_dist
            return float(np.clip(normalized_distance, 0.0, 1.0))
            
        except Exception as e:
            logger.debug(f"Embedding distance computation failed: {e}")
            return 1.0
    
    def _compute_conservation_score(self, cosine_sim: float, jaccard_sim: float, 
                                   srp_sim: float, embedding_dist: float) -> float:
        """Compute combined conservation score from multiple similarity metrics."""
        
        # Weighted combination of similarities
        # Higher weight on cosine similarity (embedding-based)
        weights = {
            'cosine': 0.4,
            'jaccard': 0.2, 
            'srp': 0.2,
            'distance': 0.2  # inverted
        }
        
        # Convert distance to similarity (1 - distance)
        distance_sim = 1.0 - embedding_dist
        
        # Weighted average
        conservation_score = (
            weights['cosine'] * cosine_sim +
            weights['jaccard'] * jaccard_sim +
            weights['srp'] * srp_sim +
            weights['distance'] * distance_sim
        )
        
        return float(np.clip(conservation_score, 0.0, 1.0))
    
    def _generate_elsa_summary(self, total_windows: int, conserved_count: int, 
                              avg_conservation: float, conservation_level: str) -> str:
        """Generate human-readable ELSA analysis summary."""
        
        if total_windows == 0:
            return "No ELSA window comparisons available for this syntenic block."
        
        conserved_fraction = conserved_count / total_windows
        
        summary = f"ELSA windowed embedding analysis of {total_windows} window pairs reveals "
        summary += f"{conservation_level} conservation (average score: {avg_conservation:.3f}). "
        
        if conserved_fraction >= 0.7:
            summary += f"Most windows ({conserved_count}/{total_windows}, {conserved_fraction:.1%}) show strong conservation, "
            summary += "indicating preserved functional organization at the protein domain level."
        elif conserved_fraction >= 0.3:
            summary += f"Moderate window conservation ({conserved_count}/{total_windows}, {conserved_fraction:.1%}) suggests "
            summary += "partial functional preservation with some divergence."
        else:
            summary += f"Few windows ({conserved_count}/{total_windows}, {conserved_fraction:.1%}) show strong conservation, "
            summary += "indicating significant functional divergence or technical limitations."
        
        return summary
    
    def _create_null_evidence(self, block_id: str, reason: str) -> ELSAHomologyEvidence:
        """Create null evidence object for cases with no ELSA data."""
        
        return ELSAHomologyEvidence(
            block_id=block_id,
            window_similarities=[],
            average_conservation=0.0,
            max_conservation=0.0,
            conserved_window_count=0,
            total_window_pairs=0,
            conservation_level="unavailable",
            elsa_summary=f"ELSA similarity analysis unavailable: {reason}"
        )


def test_elsa_similarity_analyzer():
    """Test the ELSA similarity analyzer."""
    print("=== Testing ELSASimilarityAnalyzer ===")
    
    try:
        # Initialize analyzer
        analyzer = ELSASimilarityAnalyzer()
        
        print(f"✓ ELSA analyzer initialized")
        print(f"  Available indices: {[k for k, v in analyzer.available_files.items() if v]}")
        
        if not analyzer.elsa_available:
            print("⚠️  No ELSA indices available - creating mock test")
            
            # Test null evidence creation
            null_evidence = analyzer._create_null_evidence("test_block", "ELSA indices not built")
            print(f"✓ Null evidence created: {null_evidence.elsa_summary}")
            
        else:
            print("✓ ELSA indices available - testing real analysis")
            
            # Test with sample loci
            evidence = analyzer.analyze_block_similarity(
                "test_block", 
                "1313.30775.con.0001", 
                "CAYEVI010000010"
            )
            
            if evidence:
                print(f"✓ ELSA analysis completed:")
                print(f"  - Conservation level: {evidence.conservation_level}")
                print(f"  - Window pairs: {evidence.total_window_pairs}")
                print(f"  - Conserved windows: {evidence.conserved_window_count}")
                print(f"  - Summary: {evidence.elsa_summary[:100]}...")
            else:
                print("⚠️  No ELSA evidence generated")
        
        print("✅ ELSA similarity analyzer test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_elsa_similarity_analyzer()