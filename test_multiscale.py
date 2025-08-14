#!/usr/bin/env python3
"""
Quick test script for multi-scale windowing functionality.
"""

import sys
from pathlib import Path

# Add ELSA to path
sys.path.insert(0, str(Path(__file__).parent))

from elsa.params import ELSAConfig
from elsa.windowing.multiscale import MultiScaleWindowGenerator, MultiScaleWindow
import numpy as np


# Mock protein class for testing
class MockProtein:
    def __init__(self, sample_id, locus_id, gene_id, embedding, start=0, end=100, strand='+', contig_id='contig1'):
        self.sample_id = sample_id
        self.locus_id = locus_id
        self.gene_id = gene_id
        self.embedding = embedding
        self.start = start
        self.end = end
        self.strand = strand
        self.contig_id = contig_id


def test_multiscale_windowing():
    """Test basic multi-scale windowing functionality."""
    print("🧪 Testing Multi-Scale Windowing")
    
    # Create test configuration
    config = ELSAConfig()
    config.phase2.enable = True
    config.phase2.multiscale = True
    
    # Set up window parameters
    config.window.macro = {'size': 8, 'stride': 3}
    config.window.micro = {'size': 4, 'stride': 1}
    
    # Create generator
    generator = MultiScaleWindowGenerator(config)
    print(f"✓ Generator created with macro {generator.macro_size}×{generator.macro_stride}, micro {generator.micro_size}×{generator.micro_stride}")
    
    # Create mock protein data
    embedding_dim = 256
    proteins = []
    
    # Sample 1, Locus 1: 15 proteins
    for i in range(15):
        embedding = np.random.randn(embedding_dim)
        protein = MockProtein(
            sample_id='sample1',
            locus_id='locus1', 
            gene_id=f'gene_{i}',
            embedding=embedding,
            start=i*100,
            end=(i+1)*100,
            strand='+' if i % 3 != 0 else '-'
        )
        proteins.append(protein)
    
    # Sample 2, Locus 1: 12 proteins  
    for i in range(12):
        embedding = np.random.randn(embedding_dim)
        protein = MockProtein(
            sample_id='sample2',
            locus_id='locus1',
            gene_id=f'gene_{i}',
            embedding=embedding,
            start=i*100,
            end=(i+1)*100,
            strand='+'
        )
        proteins.append(protein)
    
    print(f"✓ Created {len(proteins)} mock proteins across 2 samples")
    
    # Generate multi-scale windows
    windows, mappings = generator.generate_multiscale_windows(proteins)
    
    macro_windows = [w for w in windows if w.scale == 'macro']
    micro_windows = [w for w in windows if w.scale == 'micro']
    
    print(f"✓ Generated {len(macro_windows)} macro windows, {len(micro_windows)} micro windows")
    print(f"✓ Created {len(mappings)} macro→micro mappings")
    
    # Test window properties
    if macro_windows:
        w = macro_windows[0]
        print(f"✓ Macro window example: {w.window_id}, genes {w.start_gene_idx}-{w.end_gene_idx}, embedding shape {w.embedding.shape}")
    
    if micro_windows:
        w = micro_windows[0]
        print(f"✓ Micro window example: {w.window_id}, genes {w.start_gene_idx}-{w.end_gene_idx}, embedding shape {w.embedding.shape}")
    
    # Test mappings
    if mappings:
        mapping = mappings[0]
        print(f"✓ Mapping example: {mapping.macro_window_id} → {len(mapping.micro_window_ids)} micro windows")
        print(f"  Overlap fraction: {mapping.overlap_fraction:.3f}")
        print(f"  Gene coverage: {mapping.gene_coverage}")
    
    # Test saving functionality
    output_dir = Path("test_multiscale_output")
    generator.save_multiscale_windows(windows, mappings, output_dir)
    print(f"✓ Saved windows to {output_dir}")
    
    # Verify files were created
    expected_files = ['macro_windows.parquet', 'micro_windows.parquet', 'window_mappings.parquet', 'multiscale_metadata.json']
    for filename in expected_files:
        filepath = output_dir / filename
        if filepath.exists():
            print(f"  ✓ {filename} created")
        else:
            print(f"  ✗ {filename} missing")
    
    print("\n🎉 Multi-scale windowing test completed!")
    return True


def test_hierarchical_search():
    """Test hierarchical search if windows exist."""
    output_dir = Path("test_multiscale_output")
    
    macro_path = output_dir / "macro_windows.parquet"
    micro_path = output_dir / "micro_windows.parquet"
    mappings_path = output_dir / "window_mappings.parquet"
    
    if not all(p.exists() for p in [macro_path, micro_path, mappings_path]):
        print("⏭️  Skipping hierarchical search test (no window files)")
        return
    
    print("\n🔍 Testing Hierarchical Search")
    
    from elsa.windowing.multiscale import MultiScaleSearchEngine
    
    # Create search engine
    search_engine = MultiScaleSearchEngine(macro_path, micro_path, mappings_path)
    print(f"✓ Search engine loaded")
    
    # Create a random query
    query_embedding = np.random.randn(256)
    
    # Get search statistics
    stats = search_engine.get_search_statistics(query_embedding, macro_candidates=10)
    print(f"✓ Search statistics:")
    print(f"  Total macro windows: {stats['total_macro_windows']}")
    print(f"  Total micro windows: {stats['total_micro_windows']}")
    print(f"  Filtering ratio: {stats['filtering_ratio']:.3f}")
    print(f"  Similarity range: [{stats['macro_similarity_range'][0]:.3f}, {stats['macro_similarity_range'][1]:.3f}]")
    
    # Perform hierarchical search
    results = search_engine.hierarchical_search(query_embedding, macro_candidates=10, final_results=5)
    print(f"✓ Found {len(results)} hierarchical search results")
    
    if results:
        top_result = results[0]
        print(f"  Top result: {top_result['window_id']} (similarity: {top_result['cosine_similarity']:.3f})")
    
    print("🎉 Hierarchical search test completed!")


if __name__ == "__main__":
    try:
        test_multiscale_windowing()
        test_hierarchical_search()
        
        # Clean up
        import shutil
        output_dir = Path("test_multiscale_output")
        if output_dir.exists():
            shutil.rmtree(output_dir)
            print(f"✓ Cleaned up {output_dir}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)