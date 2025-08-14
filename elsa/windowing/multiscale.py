"""
Multi-scale windowing for ELSA Phase-2.

Implements macro→micro windowing strategy:
1. Macro windows (8-20 genes) for candidate filtering
2. Micro windows (3-5 genes) for refinement
3. Efficient mapping between scales
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
import logging
from collections import defaultdict

from ..params import ELSAConfig

logger = logging.getLogger(__name__)


@dataclass
class WindowMapping:
    """Mapping between macro and micro windows."""
    macro_window_id: str
    micro_window_ids: List[str]
    overlap_fraction: float
    gene_coverage: Tuple[int, int]  # (start_gene_idx, end_gene_idx)


@dataclass
class MultiScaleWindow:
    """A window at a specific scale."""
    window_id: str
    sample_id: str
    locus_id: str
    scale: str  # 'macro' or 'micro'
    window_idx: int
    start_gene_idx: int
    end_gene_idx: int
    gene_count: int
    embedding: np.ndarray
    strand_composition: Dict[str, int]  # '+': count, '-': count


class MultiScaleWindowGenerator:
    """
    Generates windows at multiple scales for hierarchical filtering.
    """
    
    def __init__(self, config: ELSAConfig):
        self.config = config
        self.window_config = config.window if hasattr(config, 'window') else None
        
        # Default parameters if not configured
        if self.window_config:
            self.macro_size = self.window_config.macro.get('size', 12)
            self.macro_stride = self.window_config.macro.get('stride', 3)
            self.micro_size = self.window_config.micro.get('size', 5)
            self.micro_stride = self.window_config.micro.get('stride', 1)
        else:
            # Fallback to current shingle config scaled up/down
            base_size = config.shingles.n
            self.macro_size = max(8, base_size * 2)
            self.macro_stride = max(2, base_size // 2)
            self.micro_size = max(3, base_size - 1)
            self.micro_stride = 1
        
        logger.info(f"Multi-scale windowing: macro {self.macro_size}×{self.macro_stride}, micro {self.micro_size}×{self.micro_stride}")
    
    def generate_multiscale_windows(self, projected_proteins: List) -> Tuple[List[MultiScaleWindow], List[WindowMapping]]:
        """
        Generate both macro and micro windows from projected proteins.
        
        Args:
            projected_proteins: List of ProteinEmbedding objects
            
        Returns:
            Tuple of (all_windows, mappings) where mappings connect macro to micro windows
        """
        all_windows = []
        mappings = []
        
        # Group proteins by sample and contig (locus = contig for projected proteins)
        sample_loci = defaultdict(lambda: defaultdict(list))
        for protein in projected_proteins:
            # Use contig_id as locus_id for compatibility
            locus_id = protein.contig_id
            sample_loci[protein.sample_id][locus_id].append(protein)
        
        for sample_id, loci in sample_loci.items():
            for locus_id, proteins in loci.items():
                # Sort proteins by genomic position
                proteins.sort(key=lambda p: p.start)
                
                if len(proteins) < self.micro_size:
                    logger.debug(f"Skipping {sample_id}:{locus_id} - too few genes ({len(proteins)})")
                    continue
                
                # Generate macro windows
                macro_windows = self._generate_scale_windows(
                    proteins, sample_id, locus_id, 'macro', 
                    self.macro_size, self.macro_stride
                )
                
                # Generate micro windows  
                micro_windows = self._generate_scale_windows(
                    proteins, sample_id, locus_id, 'micro',
                    self.micro_size, self.micro_stride
                )
                
                # Create mappings between macro and micro windows
                locus_mappings = self._create_window_mappings(macro_windows, micro_windows)
                
                all_windows.extend(macro_windows)
                all_windows.extend(micro_windows)
                mappings.extend(locus_mappings)
        
        logger.info(f"Generated {len(all_windows)} multi-scale windows ({len([w for w in all_windows if w.scale == 'macro'])} macro, {len([w for w in all_windows if w.scale == 'micro'])} micro)")
        logger.info(f"Created {len(mappings)} macro→micro mappings")
        
        return all_windows, mappings
    
    def _generate_scale_windows(self, proteins: List, sample_id: str, locus_id: str, 
                               scale: str, window_size: int, stride: int) -> List[MultiScaleWindow]:
        """Generate windows at a specific scale."""
        windows = []
        
        for window_idx, start_idx in enumerate(range(0, len(proteins) - window_size + 1, stride)):
            end_idx = start_idx + window_size
            window_proteins = proteins[start_idx:end_idx]
            
            # Create window embedding by pooling protein embeddings
            embeddings_matrix = np.array([p.embedding for p in window_proteins])
            
            # Use mean pooling for window embedding
            window_embedding = np.mean(embeddings_matrix, axis=0)
            
            # Analyze strand composition
            strand_counts = {'+': 0, '-': 0}
            for protein in window_proteins:
                # Convert strand integer to string
                if hasattr(protein, 'strand'):
                    strand = '+' if protein.strand >= 0 else '-'
                else:
                    strand = '+'
                strand_counts[strand] = strand_counts.get(strand, 0) + 1
            
            # Create window
            window_id = f"{sample_id}_{locus_id}_{scale}_{window_idx}"
            window = MultiScaleWindow(
                window_id=window_id,
                sample_id=sample_id,
                locus_id=locus_id,
                scale=scale,
                window_idx=window_idx,
                start_gene_idx=start_idx,
                end_gene_idx=end_idx - 1,
                gene_count=window_size,
                embedding=window_embedding,
                strand_composition=strand_counts
            )
            
            windows.append(window)
        
        return windows
    
    def _create_window_mappings(self, macro_windows: List[MultiScaleWindow], 
                               micro_windows: List[MultiScaleWindow]) -> List[WindowMapping]:
        """Create mappings between macro and micro windows based on overlap."""
        mappings = []
        
        for macro_window in macro_windows:
            overlapping_micros = []
            
            for micro_window in micro_windows:
                # Check if micro window overlaps with macro window
                overlap_start = max(macro_window.start_gene_idx, micro_window.start_gene_idx)
                overlap_end = min(macro_window.end_gene_idx, micro_window.end_gene_idx)
                
                if overlap_start <= overlap_end:
                    overlap_genes = overlap_end - overlap_start + 1
                    overlap_fraction = overlap_genes / micro_window.gene_count
                    
                    # Require significant overlap (at least 50% of micro window)
                    if overlap_fraction >= 0.5:
                        overlapping_micros.append(micro_window.window_id)
            
            if overlapping_micros:
                mapping = WindowMapping(
                    macro_window_id=macro_window.window_id,
                    micro_window_ids=overlapping_micros,
                    overlap_fraction=len(overlapping_micros) / len(micro_windows),
                    gene_coverage=(macro_window.start_gene_idx, macro_window.end_gene_idx)
                )
                mappings.append(mapping)
        
        return mappings
    
    def save_multiscale_windows(self, windows: List[MultiScaleWindow], 
                               mappings: List[WindowMapping], output_dir: Path):
        """Save multi-scale windows and mappings to disk."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert windows to DataFrame format
        macro_rows = []
        micro_rows = []
        
        for window in windows:
            row_data = {
                'window_id': window.window_id,
                'sample_id': window.sample_id,
                'locus_id': window.locus_id,
                'window_idx': window.window_idx,
                'start_gene_idx': window.start_gene_idx,
                'end_gene_idx': window.end_gene_idx,
                'gene_count': window.gene_count,
                'strand_plus': window.strand_composition.get('+', 0),
                'strand_minus': window.strand_composition.get('-', 0)
            }
            
            # Add embedding columns
            for i, val in enumerate(window.embedding):
                row_data[f'emb_{i:03d}'] = val
            
            if window.scale == 'macro':
                macro_rows.append(row_data)
            else:
                micro_rows.append(row_data)
        
        # Save windows
        if macro_rows:
            macro_df = pd.DataFrame(macro_rows)
            macro_path = output_dir / 'macro_windows.parquet'
            macro_df.to_parquet(macro_path, index=False)
            logger.info(f"Saved {len(macro_rows)} macro windows to {macro_path}")
        
        if micro_rows:
            micro_df = pd.DataFrame(micro_rows)
            micro_path = output_dir / 'micro_windows.parquet'
            micro_df.to_parquet(micro_path, index=False)
            logger.info(f"Saved {len(micro_rows)} micro windows to {micro_path}")
        
        # Save mappings
        if mappings:
            mapping_data = []
            for mapping in mappings:
                mapping_data.append({
                    'macro_window_id': mapping.macro_window_id,
                    'micro_window_ids': ','.join(mapping.micro_window_ids),
                    'overlap_fraction': mapping.overlap_fraction,
                    'gene_start': mapping.gene_coverage[0],
                    'gene_end': mapping.gene_coverage[1]
                })
            
            mapping_df = pd.DataFrame(mapping_data)
            mapping_path = output_dir / 'window_mappings.parquet'
            mapping_df.to_parquet(mapping_path, index=False)
            logger.info(f"Saved {len(mappings)} window mappings to {mapping_path}")
        
        # Save metadata
        metadata = {
            'macro_size': self.macro_size,
            'macro_stride': self.macro_stride,
            'micro_size': self.micro_size,
            'micro_stride': self.micro_stride,
            'total_windows': len(windows),
            'macro_windows': len([w for w in windows if w.scale == 'macro']),
            'micro_windows': len([w for w in windows if w.scale == 'micro']),
            'total_mappings': len(mappings)
        }
        
        import json
        with open(output_dir / 'multiscale_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)


class MultiScaleSearchEngine:
    """
    Search engine that uses hierarchical macro→micro filtering.
    """
    
    def __init__(self, macro_windows_path: Path, micro_windows_path: Path, 
                 mappings_path: Path):
        self.macro_df = pd.read_parquet(macro_windows_path)
        self.micro_df = pd.read_parquet(micro_windows_path)
        
        # Load mappings
        mapping_df = pd.read_parquet(mappings_path)
        self.mappings = {}
        for _, row in mapping_df.iterrows():
            micro_ids = row['micro_window_ids'].split(',') if row['micro_window_ids'] else []
            self.mappings[row['macro_window_id']] = micro_ids
        
        logger.info(f"Loaded {len(self.macro_df)} macro windows, {len(self.micro_df)} micro windows, {len(self.mappings)} mappings")
    
    def hierarchical_search(self, query_embedding: np.ndarray, 
                           macro_candidates: int = 100, 
                           final_results: int = 50) -> List[Dict]:
        """
        Perform hierarchical search: macro filtering → micro refinement.
        
        Args:
            query_embedding: Query window embedding
            macro_candidates: Number of macro candidates to consider
            final_results: Number of final micro results to return
            
        Returns:
            List of top micro windows with scores
        """
        # Stage 1: Macro window filtering
        macro_emb_cols = [col for col in self.macro_df.columns if col.startswith('emb_')]
        macro_embeddings = self.macro_df[macro_emb_cols].values
        
        # Compute cosine similarities with macro windows
        query_norm = np.linalg.norm(query_embedding)
        macro_norms = np.linalg.norm(macro_embeddings, axis=1)
        
        cosine_sims = np.dot(macro_embeddings, query_embedding) / (macro_norms * query_norm + 1e-10)
        
        # Get top macro candidates
        top_macro_indices = np.argsort(cosine_sims)[::-1][:macro_candidates]
        top_macro_ids = self.macro_df.iloc[top_macro_indices]['window_id'].values
        
        # Stage 2: Micro window refinement
        candidate_micro_ids = set()
        for macro_id in top_macro_ids:
            if macro_id in self.mappings:
                candidate_micro_ids.update(self.mappings[macro_id])
        
        if not candidate_micro_ids:
            logger.warning("No micro windows found for macro candidates")
            return []
        
        # Filter micro dataframe to candidates only
        candidate_micro_df = self.micro_df[self.micro_df['window_id'].isin(candidate_micro_ids)]
        
        if len(candidate_micro_df) == 0:
            return []
        
        # Compute micro window similarities
        micro_emb_cols = [col for col in candidate_micro_df.columns if col.startswith('emb_')]
        micro_embeddings = candidate_micro_df[micro_emb_cols].values
        micro_norms = np.linalg.norm(micro_embeddings, axis=1)
        
        micro_cosine_sims = np.dot(micro_embeddings, query_embedding) / (micro_norms * query_norm + 1e-10)
        
        # Get top micro results
        top_micro_indices = np.argsort(micro_cosine_sims)[::-1][:final_results]
        
        results = []
        for idx in top_micro_indices:
            row = candidate_micro_df.iloc[idx]
            results.append({
                'window_id': row['window_id'],
                'sample_id': row['sample_id'],
                'locus_id': row['locus_id'],
                'cosine_similarity': float(micro_cosine_sims[idx]),
                'gene_start': int(row['start_gene_idx']),
                'gene_end': int(row['end_gene_idx']),
                'gene_count': int(row['gene_count'])
            })
        
        logger.debug(f"Hierarchical search: {len(top_macro_ids)} macro → {len(candidate_micro_ids)} micro candidates → {len(results)} final results")
        
        return results
    
    def get_search_statistics(self, query_embedding: np.ndarray, 
                             macro_candidates: int = 100) -> Dict:
        """Get statistics about the hierarchical search process."""
        macro_emb_cols = [col for col in self.macro_df.columns if col.startswith('emb_')]
        macro_embeddings = self.macro_df[macro_emb_cols].values
        
        query_norm = np.linalg.norm(query_embedding)
        macro_norms = np.linalg.norm(macro_embeddings, axis=1)
        cosine_sims = np.dot(macro_embeddings, query_embedding) / (macro_norms * query_norm + 1e-10)
        
        top_macro_indices = np.argsort(cosine_sims)[::-1][:macro_candidates]
        top_macro_ids = self.macro_df.iloc[top_macro_indices]['window_id'].values
        
        candidate_micro_ids = set()
        for macro_id in top_macro_ids:
            if macro_id in self.mappings:
                candidate_micro_ids.update(self.mappings[macro_id])
        
        return {
            'total_macro_windows': len(self.macro_df),
            'total_micro_windows': len(self.micro_df),
            'macro_candidates': len(top_macro_ids),
            'micro_candidates': len(candidate_micro_ids),
            'filtering_ratio': len(candidate_micro_ids) / len(self.micro_df) if len(self.micro_df) > 0 else 0,
            'macro_similarity_range': (float(cosine_sims.min()), float(cosine_sims.max())),
            'top_macro_similarities': [float(cosine_sims[i]) for i in top_macro_indices[:10]]
        }


if __name__ == "__main__":
    # Example usage and testing
    from ..params import ELSAConfig
    
    # Create test configuration
    config = ELSAConfig()
    
    # Enable multi-scale windowing
    if hasattr(config, 'phase2'):
        config.phase2.multiscale = True
    
    generator = MultiScaleWindowGenerator(config)
    print(f"Multi-scale configuration:")
    print(f"  Macro: {generator.macro_size} genes, stride {generator.macro_stride}")
    print(f"  Micro: {generator.micro_size} genes, stride {generator.micro_stride}")