#!/usr/bin/env python3
"""
Test script for GPT-5 analyzer to debug issues.
"""

import sys
from pathlib import Path
import logging

# Setup logging to see debug info
logging.basicConfig(level=logging.INFO)

from gpt5_analyzer import GPT5Analyzer, analyze_syntenic_block

def test_analyzer():
    """Test the GPT-5 analyzer step by step."""
    
    # Test block extraction
    print("Testing GPT-5 Analyzer...")
    
    db_path = Path("genome_browser.db")
    analyzer = GPT5Analyzer(db_path)
    
    # Try to get a block analysis (use block 0 as a test)
    test_block_id = 0
    print(f"\n1. Testing block {test_block_id} extraction...")
    
    try:
        analysis = analyzer.get_syntenic_block_analysis(test_block_id)
        if analysis:
            print(f"✅ Block extraction successful!")
            print(f"   Block ID: {analysis.block_id}")
            print(f"   Query: {analysis.query_locus.organism_name} ({analysis.query_locus.gene_count} genes)")
            print(f"   Target: {analysis.target_locus.organism_name} ({analysis.target_locus.gene_count} genes)")
            print(f"   Identity: {analysis.identity:.1%}")
            
            # Test a few genes from each locus
            print(f"\n   Query genes sample:")
            for i, gene in enumerate(analysis.query_locus.genes[:3]):
                print(f"     {i+1}. {gene.gene_id} ({gene.start}-{gene.end}, PFAM: {gene.pfam_domains[:50]}...)")
            
            print(f"\n   Target genes sample:")
            for i, gene in enumerate(analysis.target_locus.genes[:3]):
                print(f"     {i+1}. {gene.gene_id} ({gene.start}-{gene.end}, PFAM: {gene.pfam_domains[:50]}...)")
            
            return analysis
        else:
            print(f"❌ Block extraction failed")
            return None
    except Exception as e:
        print(f"❌ Block extraction error: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_gpt_call(analysis):
    """Test the GPT-5 call separately."""
    print(f"\n2. Testing GPT-5 call...")
    
    if not analysis:
        print("❌ No analysis data to test with")
        return
    
    try:
        db_path = Path("genome_browser.db")
        analyzer = GPT5Analyzer(db_path)
        
        # Test GPT analysis
        report = analyzer.generate_gpt_analysis(analysis)
        if report and "GPT-5 analysis failed" not in report:
            print(f"✅ GPT-5 analysis successful!")
            print(f"   Report length: {len(report)} characters")
            print(f"   First 200 chars: {report[:200]}...")
            return True
        else:
            print(f"❌ GPT-5 analysis failed: {report}")
            return False
    except Exception as e:
        print(f"❌ GPT-5 call error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("GPT-5 Analyzer Debug Test")
    print("=" * 60)
    
    # Test step by step
    analysis = test_analyzer()
    
    if analysis:
        test_gpt_call(analysis)
    
    print("\n" + "=" * 60)
    print("Test completed")