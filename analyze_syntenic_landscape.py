#!/usr/bin/env python3
"""
Syntenic Landscape Analysis Script

Analyzes the syntenic_landscape.json output from ELSA pipeline.
Uses streaming JSON processing to handle large files (44GB+) without memory issues.
"""

import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict, Counter
import ijson
import sys

def stream_landscape_summary(file_path):
    """Extract summary statistics from landscape file via streaming"""
    print(f"Streaming analysis of: {file_path}")
    print("This may take a while for large files...\n")
    
    summary_stats = {}
    
    try:
        with open(file_path, 'rb') as file:
            # Parse the summary section first
            parser = ijson.parse(file)
            
            current_path = []
            for prefix, event, value in parser:
                if prefix == 'landscape.total_loci' and event == 'number':
                    summary_stats['total_loci'] = value
                elif prefix == 'landscape.total_blocks' and event == 'number':
                    summary_stats['total_blocks'] = value
                elif prefix == 'landscape.total_clusters' and event == 'number':
                    summary_stats['total_clusters'] = value
                elif prefix == 'landscape.blocks' and event == 'start_array':
                    # We've reached the blocks array, stop parsing summary
                    break
                    
    except Exception as e:
        print(f"Error parsing landscape file: {e}")
        return None
    
    return summary_stats

def count_blocks_per_genome_streaming(file_path, max_blocks=10000):
    """Count blocks per genome pair using streaming, limited for memory"""
    print(f"Counting genome pair relationships (processing up to {max_blocks:,} blocks)...")
    
    genome_pairs = Counter()
    query_genomes = Counter()
    target_genomes = Counter()
    block_count = 0
    
    try:
        with open(file_path, 'rb') as file:
            # Stream parse the blocks array
            blocks = ijson.items(file, 'landscape.blocks.item')
            
            for block in blocks:
                if block_count >= max_blocks:
                    print(f"Reached limit of {max_blocks:,} blocks")
                    break
                    
                query_locus = block.get('query_locus', '')
                target_locus = block.get('target_locus', '')
                
                # Extract genome IDs
                query_genome = query_locus.split(':')[0] if ':' in query_locus else query_locus
                target_genome = target_locus.split(':')[0] if ':' in target_locus else target_locus
                
                # Count relationships
                genome_pairs[(query_genome, target_genome)] += 1
                query_genomes[query_genome] += 1
                target_genomes[target_genome] += 1
                
                block_count += 1
                
                if block_count % 1000 == 0:
                    print(f"Processed {block_count:,} blocks...", end='\r')
    
    except Exception as e:
        print(f"Error processing blocks: {e}")
        return None, None, None, 0
    
    print(f"\nCompleted processing {block_count:,} blocks")
    return genome_pairs, query_genomes, target_genomes, block_count

def analyze_window_patterns_streaming(file_path, max_blocks=5000):
    """Analyze syntenic window patterns using streaming"""
    print(f"Analyzing window patterns (processing up to {max_blocks:,} blocks)...")
    
    window_counts = []
    window_lengths = []
    identity_scores = []
    block_count = 0
    
    try:
        with open(file_path, 'rb') as file:
            blocks = ijson.items(file, 'landscape.blocks.item')
            
            for block in blocks:
                if block_count >= max_blocks:
                    break
                    
                # Extract window information
                query_windows = block.get('query_windows', [])
                target_windows = block.get('target_windows', [])
                
                if query_windows and target_windows:
                    window_counts.append(len(query_windows))
                    
                    # Calculate approximate block length from windows
                    if len(query_windows) > 1:
                        first_start = query_windows[0].get('start', 0)
                        last_end = query_windows[-1].get('end', 0)
                        if last_end > first_start:
                            window_lengths.append(last_end - first_start)
                    
                    # Extract identity if available
                    identity = block.get('identity', None)
                    if identity is not None:
                        identity_scores.append(identity)
                
                block_count += 1
                
                if block_count % 500 == 0:
                    print(f"Processed {block_count:,} blocks...", end='\r')
    
    except Exception as e:
        print(f"Error analyzing windows: {e}")
        return None, None, None
    
    print(f"\nAnalyzed {block_count:,} blocks")
    return window_counts, window_lengths, identity_scores

def generate_landscape_report(summary_stats, genome_pairs, query_genomes, target_genomes, 
                             window_counts, window_lengths, identity_scores, output_dir):
    """Generate comprehensive landscape analysis report"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    report_lines = []
    report_lines.append("# SYNTENIC LANDSCAPE ANALYSIS REPORT\n")
    
    # Summary statistics
    if summary_stats:
        report_lines.append("## Summary Statistics\n")
        for key, value in summary_stats.items():
            report_lines.append(f"- **{key.replace('_', ' ').title()}**: {value:,}")
        report_lines.append("\n")
    
    # Genome pair analysis
    if genome_pairs:
        report_lines.append("## Genome Pair Analysis\n")
        report_lines.append(f"Total unique genome pairs analyzed: {len(genome_pairs):,}\n")
        
        report_lines.append("### Top 20 Most Connected Genome Pairs:\n")
        for (query, target), count in genome_pairs.most_common(20):
            report_lines.append(f"- {query} ↔ {target}: {count:,} syntenic blocks")
        report_lines.append("\n")
    
    # Individual genome analysis
    if query_genomes and target_genomes:
        report_lines.append("## Individual Genome Analysis\n")
        
        report_lines.append("### Top 10 Query Genomes (most blocks as query):\n")
        for genome, count in query_genomes.most_common(10):
            report_lines.append(f"- {genome}: {count:,} blocks")
        report_lines.append("\n")
        
        report_lines.append("### Top 10 Target Genomes (most blocks as target):\n")
        for genome, count in target_genomes.most_common(10):
            report_lines.append(f"- {genome}: {count:,} blocks")
        report_lines.append("\n")
    
    # Window pattern analysis
    if window_counts:
        report_lines.append("## Window Pattern Analysis\n")
        report_lines.append(f"Blocks with window data analyzed: {len(window_counts):,}\n")
        
        if window_counts:
            report_lines.append(f"- Mean windows per block: {np.mean(window_counts):.1f}")
            report_lines.append(f"- Median windows per block: {np.median(window_counts):.0f}")
            report_lines.append(f"- Min windows per block: {min(window_counts):,}")
            report_lines.append(f"- Max windows per block: {max(window_counts):,}")
        
        if window_lengths:
            report_lines.append(f"\n### Block Length Analysis (from windows):")
            report_lines.append(f"- Mean block length: {np.mean(window_lengths):,.0f} bp")
            report_lines.append(f"- Median block length: {np.median(window_lengths):,.0f} bp")
            report_lines.append(f"- Min block length: {min(window_lengths):,} bp")
            report_lines.append(f"- Max block length: {max(window_lengths):,} bp")
        
        if identity_scores:
            report_lines.append(f"\n### Identity Score Analysis:")
            report_lines.append(f"- Mean identity: {np.mean(identity_scores):.4f}")
            report_lines.append(f"- Median identity: {np.median(identity_scores):.4f}")
            report_lines.append(f"- Min identity: {min(identity_scores):.4f}")
            report_lines.append(f"- Max identity: {max(identity_scores):.4f}")
        
        report_lines.append("\n")
    
    # Write report
    with open(output_path / 'landscape_analysis_report.md', 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"\nLandscape analysis report saved to: {output_path / 'landscape_analysis_report.md'}")

def create_landscape_plots(genome_pairs, window_counts, window_lengths, identity_scores, output_dir):
    """Create visualization plots for landscape analysis"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    plt.style.use('default')
    
    # 1. Top genome pairs plot
    if genome_pairs:
        top_pairs = dict(genome_pairs.most_common(15))
        
        plt.figure(figsize=(12, 8))
        pairs_labels = [f"{q}→{t}" for (q, t) in top_pairs.keys()]
        pairs_counts = list(top_pairs.values())
        
        plt.barh(range(len(pairs_labels)), pairs_counts)
        plt.yticks(range(len(pairs_labels)), pairs_labels)
        plt.xlabel('Number of Syntenic Blocks')
        plt.title('Top 15 Genome Pairs by Syntenic Block Count')
        plt.tight_layout()
        plt.savefig(output_path / 'top_genome_pairs.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 2. Windows per block distribution
    if window_counts:
        plt.figure(figsize=(10, 6))
        plt.hist(window_counts, bins=50, alpha=0.7, log=True)
        plt.xlabel('Number of Windows per Block')
        plt.ylabel('Count (log scale)')
        plt.title('Distribution of Windows per Syntenic Block')
        plt.savefig(output_path / 'windows_per_block_histogram.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. Block length distribution from windows
    if window_lengths:
        plt.figure(figsize=(10, 6))
        plt.hist(window_lengths, bins=50, alpha=0.7, log=True)
        plt.xlabel('Block Length (bp)')
        plt.ylabel('Count (log scale)')
        plt.title('Distribution of Syntenic Block Lengths (from Windows)')
        plt.savefig(output_path / 'block_lengths_from_windows.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 4. Identity scores distribution
    if identity_scores:
        plt.figure(figsize=(10, 6))
        plt.hist(identity_scores, bins=50, alpha=0.7, color='green')
        plt.xlabel('Identity Score')
        plt.ylabel('Count')
        plt.title('Distribution of Identity Scores (from Landscape)')
        plt.savefig(output_path / 'identity_scores_landscape.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Landscape plots saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Analyze ELSA syntenic landscape output (streaming for large files)')
    parser.add_argument('--input', '-i', default='syntenic_analysis/syntenic_landscape.json',
                       help='Input syntenic landscape JSON file')
    parser.add_argument('--output', '-o', default='landscape_analysis_output',
                       help='Output directory for results')
    parser.add_argument('--max-blocks', type=int, default=10000,
                       help='Maximum blocks to process for genome analysis (default: 10000)')
    parser.add_argument('--max-windows', type=int, default=5000,
                       help='Maximum blocks to process for window analysis (default: 5000)')
    parser.add_argument('--plots', action='store_true', help='Generate visualization plots')
    parser.add_argument('--report', action='store_true', help='Generate detailed analysis report')
    
    args = parser.parse_args()
    
    # Check if file exists
    if not Path(args.input).exists():
        print(f"Error: Input file {args.input} not found")
        sys.exit(1)
    
    print(f"Analyzing syntenic landscape from: {args.input}")
    print(f"File size: {Path(args.input).stat().st_size / 1e9:.1f} GB")
    print("Using streaming processing to handle large file...\n")
    
    # Stream analysis
    summary_stats = stream_landscape_summary(args.input)
    if summary_stats:
        print("=== LANDSCAPE SUMMARY ===")
        for key, value in summary_stats.items():
            print(f"{key.replace('_', ' ').title()}: {value:,}")
        print()
    
    # Genome pair analysis
    genome_pairs, query_genomes, target_genomes, processed_blocks = count_blocks_per_genome_streaming(
        args.input, args.max_blocks)
    
    if genome_pairs:
        print(f"\n=== TOP 10 GENOME PAIRS ===")
        for (query, target), count in genome_pairs.most_common(10):
            print(f"{query} ↔ {target}: {count:,} blocks")
    
    # Window pattern analysis
    window_counts, window_lengths, identity_scores = analyze_window_patterns_streaming(
        args.input, args.max_windows)
    
    # Generate outputs
    if args.report:
        generate_landscape_report(summary_stats, genome_pairs, query_genomes, target_genomes,
                                window_counts, window_lengths, identity_scores, args.output)
    
    if args.plots:
        create_landscape_plots(genome_pairs, window_counts, window_lengths, identity_scores, args.output)

if __name__ == '__main__':
    main()