#!/usr/bin/env python3
"""
Debug homology integration issues in the genome browser.
"""

import logging
from pathlib import Path
import sys

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def debug_homology_data(block_id):
    """Debug homology data collection and formatting for a specific block."""
    print(f"=== Debugging Homology Integration for Block {block_id} ===")
    
    try:
        # Import components
        from genome_browser.analysis.data_collector import SyntenicDataCollector
        from genome_browser.analysis.gpt5_analyzer import GPT5SyntenicAnalyzer
        
        print("✓ Components imported successfully")
        
        # Initialize data collector
        collector = SyntenicDataCollector()
        print("✓ Data collector initialized")
        
        # Collect block data (this should include homology analysis)
        print(f"\nCollecting data for block {block_id}...")
        block_data = collector.collect_block_data(block_id)
        
        if not block_data:
            print(f"❌ No data collected for block {block_id}")
            return False
        
        print(f"✓ Block data collected:")
        print(f"  - Block ID: {block_data.block_id}")
        print(f"  - Genes: {len(block_data.genes)}")
        print(f"  - Genomes: {len(block_data.genomes)}")
        print(f"  - PFAM domains: {len(block_data.pfam_domains)}")
        
        # Check homology analysis data
        print(f"\nChecking homology analysis...")
        if block_data.homology_analysis:
            homology = block_data.homology_analysis
            print(f"✓ Homology analysis present:")
            print(f"  - Ortholog pairs: {len(homology.get('ortholog_pairs', []))}")
            print(f"  - Functional groups: {len(homology.get('functional_groups', []))}")
            print(f"  - Conservation level: {homology.get('conservation_summary', {}).get('conservation_level', 'unknown')}")
            
            # Show some ortholog details
            orthologs = homology.get('ortholog_pairs', [])
            if orthologs:
                print(f"\nSample ortholog pairs:")
                for i, pair in enumerate(orthologs[:3]):
                    print(f"  {i+1}. {pair.get('query_id', 'unknown')} ↔ {pair.get('target_id', 'unknown')}")
                    print(f"     Identity: {pair.get('identity', 0):.3f}, Relationship: {pair.get('relationship', 'unknown')}")
            else:
                print("❌ No ortholog pairs found!")
        else:
            print("❌ No homology analysis data!")
            return False
        
        # Test GPT-5 analyzer formatting
        print(f"\nTesting GPT-5 data formatting...")
        analyzer = GPT5SyntenicAnalyzer()
        
        # Format homology data
        homology_formatted = analyzer._format_homology_data(block_data)
        print(f"✓ Homology data formatted ({len(homology_formatted)} chars)")
        
        print(f"\nFormatted homology data preview:")
        print("=" * 50)
        print(homology_formatted[:500] + "..." if len(homology_formatted) > 500 else homology_formatted)
        print("=" * 50)
        
        # Check if output directory exists for this block
        output_dir = Path("homology_analysis") / f"block_{block_id}"
        if output_dir.exists():
            print(f"✓ Output directory exists: {output_dir}")
            files = list(output_dir.glob("*"))
            for file in files:
                print(f"  - {file.name}: {file.stat().st_size} bytes")
        else:
            print(f"❌ Output directory missing: {output_dir}")
        
        return True
        
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Debug multiple blocks to find the issue."""
    
    # Test a few different blocks
    test_blocks = [0, 47, 263, 281]
    
    for block_id in test_blocks:
        print(f"\n{'='*60}")
        success = debug_homology_data(block_id)
        if success:
            print(f"✅ Block {block_id} processed successfully")
        else:
            print(f"❌ Block {block_id} failed")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()