#!/usr/bin/env python3
"""
Syntenic Blocks Analysis Script

Analyzes the syntenic_blocks.csv output from ELSA pipeline.
Handles large datasets efficiently with pandas chunking and streaming analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import sys
from collections import defaultdict

def load_blocks_data(file_path, chunksize=10000):
    """Load syntenic blocks data efficiently"""
    try:
        # Try loading full dataset first for smaller files
        df = pd.read_csv(file_path)
        print(f"Loaded {len(df):,} syntenic blocks")
        return df
    except MemoryError:
        print(f"File too large, using chunked processing...")
        return pd.read_csv(file_path, chunksize=chunksize)

def basic_statistics(df):
    """Calculate basic statistics for syntenic blocks"""
    print("\n=== SYNTENIC BLOCKS STATISTICS ===")
    print(f"Total blocks: {len(df):,}")
    print(f"Unique query loci: {df['query_locus'].nunique():,}")
    print(f"Unique target loci: {df['target_locus'].nunique():,}")
    
    print(f"\nBlock lengths:")
    print(f"  Mean: {df['length'].mean():.0f} bp")
    print(f"  Median: {df['length'].median():.0f} bp")
    print(f"  Min: {df['length'].min():,} bp")
    print(f"  Max: {df['length'].max():,} bp")
    
    print(f"\nIdentity scores:")
    print(f"  Mean: {df['identity'].mean():.4f}")
    print(f"  Median: {df['identity'].median():.4f}")
    print(f"  Min: {df['identity'].min():.4f}")
    print(f"  Max: {df['identity'].max():.4f}")
    
    print(f"\nAlignment scores:")
    print(f"  Mean: {df['score'].mean():.0f}")
    print(f"  Median: {df['score'].median():.0f}")
    print(f"  Min: {df['score'].min():.0f}")
    print(f"  Max: {df['score'].max():.0f}")

def analyze_genome_pairs(df):
    """Analyze syntenic relationships between genome pairs"""
    print("\n=== GENOME PAIR ANALYSIS ===")
    
    # Extract genome IDs from loci
    df['query_genome'] = df['query_locus'].str.split(':').str[0]
    df['target_genome'] = df['target_locus'].str.split(':').str[0]
    
    # Count blocks per genome pair
    pair_counts = df.groupby(['query_genome', 'target_genome']).size().reset_index(name='block_count')
    pair_counts = pair_counts.sort_values('block_count', ascending=False)
    
    print(f"Top 10 genome pairs by syntenic block count:")
    for i, row in pair_counts.head(10).iterrows():
        print(f"  {row['query_genome']} ↔ {row['target_genome']}: {row['block_count']:,} blocks")
    
    return pair_counts

def length_distribution_analysis(df):
    """Analyze distribution of syntenic block lengths"""
    print("\n=== BLOCK LENGTH DISTRIBUTION ===")
    
    # Length percentiles
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    length_stats = df['length'].quantile([p/100 for p in percentiles])
    
    for p in percentiles:
        print(f"  {p}th percentile: {length_stats[p/100]:,.0f} bp")
    
    # Length categories
    bins = [0, 1000, 5000, 10000, 25000, 50000, float('inf')]
    labels = ['<1kb', '1-5kb', '5-10kb', '10-25kb', '25-50kb', '>50kb']
    df['length_category'] = pd.cut(df['length'], bins=bins, labels=labels, include_lowest=True)
    
    length_dist = df['length_category'].value_counts().sort_index()
    print(f"\nBlock length distribution:")
    for cat, count in length_dist.items():
        pct = count / len(df) * 100
        print(f"  {cat}: {count:,} blocks ({pct:.1f}%)")

def identity_vs_length_analysis(df):
    """Analyze relationship between identity and block length"""
    print("\n=== IDENTITY vs LENGTH ANALYSIS ===")
    
    # Correlation
    correlation = df['identity'].corr(df['length'])
    print(f"Identity-Length correlation: {correlation:.4f}")
    
    # Identity by length category
    if 'length_category' in df.columns:
        identity_by_length = df.groupby('length_category')['identity'].agg(['mean', 'std', 'count'])
        print(f"\nMean identity by length category:")
        for cat, stats in identity_by_length.iterrows():
            print(f"  {cat}: {stats['mean']:.4f} ± {stats['std']:.4f} (n={stats['count']:,})")

def create_summary_plots(df, output_dir):
    """Create summary visualization plots"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Set up plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Block length histogram (log scale)
    plt.figure(figsize=(10, 6))
    plt.hist(df['length'], bins=50, alpha=0.7, log=True)
    plt.xlabel('Block Length (bp)')
    plt.ylabel('Count (log scale)')
    plt.title('Distribution of Syntenic Block Lengths')
    plt.savefig(output_path / 'block_lengths_histogram.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Identity distribution
    plt.figure(figsize=(10, 6))
    plt.hist(df['identity'], bins=50, alpha=0.7, color='orange')
    plt.xlabel('Identity Score')
    plt.ylabel('Count')
    plt.title('Distribution of Syntenic Block Identity Scores')
    plt.savefig(output_path / 'identity_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Identity vs Length scatter (sampled for performance)
    sample_size = min(10000, len(df))
    df_sample = df.sample(sample_size) if len(df) > sample_size else df
    
    plt.figure(figsize=(10, 6))
    plt.scatter(df_sample['length'], df_sample['identity'], alpha=0.6, s=1)
    plt.xlabel('Block Length (bp)')
    plt.ylabel('Identity Score')
    plt.title(f'Identity vs Length (n={sample_size:,} blocks)')
    plt.xscale('log')
    plt.savefig(output_path / 'identity_vs_length_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nPlots saved to: {output_path}")

def export_filtered_datasets(df, output_dir):
    """Export filtered subsets for detailed analysis"""
    output_path = Path(output_dir)
    
    # High-quality blocks (high identity, reasonable length)
    high_quality = df[(df['identity'] > 0.9) & (df['length'] > 5000)]
    high_quality.to_csv(output_path / 'high_quality_blocks.csv', index=False)
    print(f"\nExported {len(high_quality):,} high-quality blocks")
    
    # Large blocks
    large_blocks = df[df['length'] > df['length'].quantile(0.95)]
    large_blocks.to_csv(output_path / 'large_blocks.csv', index=False)
    print(f"Exported {len(large_blocks):,} large blocks (>95th percentile)")
    
    # Top scoring blocks
    top_scoring = df.nlargest(1000, 'score')
    top_scoring.to_csv(output_path / 'top_scoring_blocks.csv', index=False)
    print(f"Exported top 1000 scoring blocks")

def main():
    parser = argparse.ArgumentParser(description='Analyze ELSA syntenic blocks output')
    parser.add_argument('--input', '-i', default='syntenic_analysis/syntenic_blocks.csv',
                       help='Input syntenic blocks CSV file')
    parser.add_argument('--output', '-o', default='analysis_output',
                       help='Output directory for results')
    parser.add_argument('--plots', action='store_true', help='Generate visualization plots')
    parser.add_argument('--export', action='store_true', help='Export filtered datasets')
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading syntenic blocks from: {args.input}")
    df = load_blocks_data(args.input)
    
    # Perform analyses
    basic_statistics(df)
    genome_pairs = analyze_genome_pairs(df)
    length_distribution_analysis(df)
    identity_vs_length_analysis(df)
    
    # Optional outputs
    if args.plots:
        create_summary_plots(df, args.output)
    
    if args.export:
        export_filtered_datasets(df, args.output)

if __name__ == '__main__':
    main()