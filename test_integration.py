#!/usr/bin/env python3
"""
Test multi-scale windowing integration with real ELSA data.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add ELSA to path
sys.path.insert(0, str(Path(__file__).parent))

from elsa.params import ELSAConfig, load_config
from elsa.projection import ProjectedProtein
from elsa.shingling import ShingleSystem
from elsa.manifest import ELSAManifest


def test_multiscale_integration():
    """Test multi-scale windowing with real ELSA projected proteins."""
    print("🧪 Testing Multi-Scale Integration with Real Data")
    
    # Load configuration
    config = load_config("elsa.config.yaml")
    print(f"✓ Configuration loaded (multiscale: {config.phase2.multiscale})")
    
    # Check if projected proteins exist
    genes_path = Path("elsa_index/ingest/genes.parquet")
    if not genes_path.exists():
        print("❌ No projected proteins found. Run 'elsa embed' first.")
        return False
    
    print(f"✓ Found projected proteins: {genes_path}")
    
    # Load projected proteins from parquet
    genes_df = pd.read_parquet(genes_path)
    print(f"✓ Loaded {len(genes_df)} projected proteins")
    
    # Convert to ProjectedProtein objects
    projected_proteins = []
    for _, row in genes_df.iterrows():
        # Extract embedding
        emb_cols = [col for col in genes_df.columns if col.startswith('emb_')]
        embedding = np.array([row[col] for col in emb_cols])
        
        protein = ProjectedProtein(
            sample_id=row['sample_id'],
            contig_id=row['contig_id'],
            gene_id=row['gene_id'],
            start=row['start'],
            end=row['end'],
            strand=row['strand'],
            embedding=embedding,
            original_length=row['original_length']
        )
        projected_proteins.append(protein)
    
    print(f"✓ Converted to {len(projected_proteins)} ProjectedProtein objects")
    
    # Test multi-scale windowing
    manifest = ELSAManifest(config.data.work_dir)
    shingle_system = ShingleSystem(config.shingles, config.data.work_dir, manifest, config)
    
    print("🔄 Testing multi-scale windowing...")
    
    # Test just a subset to avoid timeout
    test_proteins = projected_proteins[:1000]  # Test with 1000 proteins
    
    try:
        windows = shingle_system._process_multiscale_windows(test_proteins)
        print(f"✅ Multi-scale windowing successful! Generated {len(windows)} windows")
        
        # Check if multiscale directory was created
        multiscale_dir = Path("elsa_index/multiscale_windows")
        if multiscale_dir.exists():
            files = list(multiscale_dir.glob("*.parquet"))
            print(f"✓ Multi-scale files created: {[f.name for f in files]}")
            
            # Check file contents
            if (multiscale_dir / "macro_windows.parquet").exists():
                macro_df = pd.read_parquet(multiscale_dir / "macro_windows.parquet")
                print(f"  - Macro windows: {len(macro_df)}")
                
            if (multiscale_dir / "micro_windows.parquet").exists():
                micro_df = pd.read_parquet(multiscale_dir / "micro_windows.parquet")
                print(f"  - Micro windows: {len(micro_df)}")
                
            if (multiscale_dir / "window_mappings.parquet").exists():
                mappings_df = pd.read_parquet(multiscale_dir / "window_mappings.parquet")
                print(f"  - Window mappings: {len(mappings_df)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Multi-scale windowing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_multiscale_integration()
    if success:
        print("\n🎉 Multi-scale windowing integration test passed!")
    else:
        print("\n❌ Multi-scale windowing integration test failed!")
        sys.exit(1)