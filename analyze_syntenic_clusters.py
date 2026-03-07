#!/usr/bin/env python3
"""
Syntenic Clusters Analysis Script

Analyzes the syntenic_clusters.csv output from ELSA pipeline.
Focuses on cluster characteristics and diversity patterns.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse

def load_clusters_data(file_path):
    """Load syntenic clusters data"""
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df):,} syntenic clusters")
    return df

def basic_cluster_statistics(df):
    """Calculate basic statistics for syntenic clusters"""
    print("\n=== SYNTENIC CLUSTERS STATISTICS ===")
    print(f"Total clusters: {len(df):,}")
    
    print(f"\nCluster sizes:")
    print(f"  Mean: {df['size'].mean():.0f} blocks")
    print(f"  Median: {df['size'].median():.0f} blocks")
    print(f"  Min: {df['size'].min():,} blocks")
    print(f"  Max: {df['size'].max():,} blocks")
    print(f"  Total blocks in clusters: {df['size'].sum():,}")
    
    print(f"\nConsensus lengths:")
    print(f"  Mean: {df['consensus_length'].mean():.0f} bp")
    print(f"  Median: {df['consensus_length'].median():.0f} bp")
    print(f"  Min: {df['consensus_length'].min():,} bp")
    print(f"  Max: {df['consensus_length'].max():,} bp")
    
    print(f"\nConsensus scores:")
    print(f"  Mean: {df['consensus_score'].mean():.0f}")
    print(f"  Median: {df['consensus_score'].median():.0f}")
    print(f"  Min: {df['consensus_score'].min():.0f}")
    print(f"  Max: {df['consensus_score'].max():.0f}")
    
    print(f"\nDiversity scores:")
    print(f"  Mean: {df['diversity'].mean():.4f}")
    print(f"  Median: {df['diversity'].median():.4f}")
    print(f"  Min: {df['diversity'].min():.4f}")
    print(f"  Max: {df['diversity'].max():.4f}")

def cluster_size_analysis(df):
    """Analyze cluster size distribution"""
    print("\n=== CLUSTER SIZE ANALYSIS ===")
    
    # Size percentiles
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    size_stats = df['size'].quantile([p/100 for p in percentiles])
    
    print("Cluster size percentiles:")
    for p in percentiles:
        print(f"  {p}th percentile: {size_stats[p/100]:,.0f} blocks")
    
    # Size categories
    bins = [0, 5, 10, 50, 100, 1000, float('inf')]
    labels = ['1-5', '6-10', '11-50', '51-100', '101-1000', '>1000']
    df['size_category'] = pd.cut(df['size'], bins=bins, labels=labels, include_lowest=True)
    
    size_dist = df['size_category'].value_counts().sort_index()
    print(f"\nCluster size distribution:")
    for cat, count in size_dist.items():
        pct = count / len(df) * 100
        total_blocks = df[df['size_category'] == cat]['size'].sum()
        print(f"  {cat} blocks: {count} clusters ({pct:.1f}%), {total_blocks:,} total blocks")

def diversity_analysis(df):
    """Analyze cluster diversity patterns"""
    print("\n=== DIVERSITY ANALYSIS ===")
    
    # Diversity vs size correlation
    diversity_size_corr = df['diversity'].corr(df['size'])
    print(f"Diversity-Size correlation: {diversity_size_corr:.4f}")
    
    # Diversity vs consensus score correlation
    diversity_score_corr = df['diversity'].corr(df['consensus_score'])
    print(f"Diversity-Consensus Score correlation: {diversity_score_corr:.4f}")
    
    # High/low diversity clusters
    high_div_threshold = df['diversity'].quantile(0.75)
    low_div_threshold = df['diversity'].quantile(0.25)
    
    high_div = df[df['diversity'] >= high_div_threshold]
    low_div = df[df['diversity'] <= low_div_threshold]
    
    print(f"\nHigh diversity clusters (>75th percentile, diversity ≥ {high_div_threshold:.4f}):")
    print(f"  Count: {len(high_div)} clusters")
    print(f"  Mean size: {high_div['size'].mean():.0f} blocks")
    print(f"  Mean consensus score: {high_div['consensus_score'].mean():.0f}")
    
    print(f"\nLow diversity clusters (≤25th percentile, diversity ≤ {low_div_threshold:.4f}):")
    print(f"  Count: {len(low_div)} clusters")
    print(f"  Mean size: {low_div['size'].mean():.0f} blocks")
    print(f"  Mean consensus score: {low_div['consensus_score'].mean():.0f}")

def representative_analysis(df):
    """Analyze representative sequences in clusters"""
    print("\n=== REPRESENTATIVE SEQUENCES ANALYSIS ===")
    
    # Extract genome IDs from representatives
    df['rep_query_genome'] = df['representative_query'].str.split(':').str[0]
    df['rep_target_genome'] = df['representative_target'].str.split(':').str[0]
    
    print("Most frequent representative query genomes:")
    query_counts = df['rep_query_genome'].value_counts().head(10)
    for genome, count in query_counts.items():
        print(f"  {genome}: {count} clusters")
    
    print("\nMost frequent representative target genomes:")
    target_counts = df['rep_target_genome'].value_counts().head(10)
    for genome, count in target_counts.items():
        print(f"  {genome}: {count} clusters")
    
    # Self-comparisons vs cross-comparisons
    self_comparisons = df[df['rep_query_genome'] == df['rep_target_genome']]
    cross_comparisons = df[df['rep_query_genome'] != df['rep_target_genome']]
    
    print(f"\nComparison types:")
    print(f"  Self-comparisons: {len(self_comparisons)} clusters ({len(self_comparisons)/len(df)*100:.1f}%)")
    print(f"  Cross-comparisons: {len(cross_comparisons)} clusters ({len(cross_comparisons)/len(df)*100:.1f}%)")

def detailed_cluster_report(df, output_dir):
    """Generate detailed report for each cluster"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Sort clusters by size (largest first)
    df_sorted = df.sort_values('size', ascending=False)
    
    report_lines = []
    report_lines.append("# DETAILED CLUSTER REPORT\n")
    report_lines.append(f"Generated from {len(df)} syntenic clusters\n\n")
    
    for i, row in df_sorted.iterrows():
        report_lines.append(f"## Cluster {row['cluster_id']} (Rank #{i+1} by size)\n")
        report_lines.append(f"- **Size**: {row['size']:,} syntenic blocks")
        report_lines.append(f"- **Consensus Length**: {row['consensus_length']:,} bp")
        report_lines.append(f"- **Consensus Score**: {row['consensus_score']:.1f}")
        report_lines.append(f"- **Diversity**: {row['diversity']:.6f}")
        report_lines.append(f"- **Representative Query**: {row['representative_query']}")
        report_lines.append(f"- **Representative Target**: {row['representative_target']}")
        
        # Add interpretation
        if row['size'] > 1000:
            report_lines.append(f"- *Very large cluster - may represent highly conserved syntenic region*")
        elif row['size'] < 5:
            report_lines.append(f"- *Small cluster - may represent specific or rare syntenic pattern*")
        
        if row['diversity'] < 0.01:
            report_lines.append(f"- *Low diversity - highly conserved syntenic block*")
        elif row['diversity'] > 1.0:
            report_lines.append(f"- *High diversity - variable syntenic relationships*")
        
        report_lines.append("\n")
    
    # Write report
    with open(output_path / 'cluster_detailed_report.md', 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"\nDetailed cluster report saved to: {output_path / 'cluster_detailed_report.md'}")

def create_cluster_plots(df, output_dir):
    """Create visualization plots for cluster analysis"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Set up plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Cluster size distribution (log scale)
    plt.figure(figsize=(10, 6))
    plt.hist(df['size'], bins=20, alpha=0.7, log=True)
    plt.xlabel('Cluster Size (number of blocks)')
    plt.ylabel('Count (log scale)')
    plt.title('Distribution of Syntenic Cluster Sizes')
    plt.savefig(output_path / 'cluster_sizes_histogram.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Diversity vs Size scatter
    plt.figure(figsize=(10, 6))
    plt.scatter(df['size'], df['diversity'], alpha=0.7, s=50)
    plt.xlabel('Cluster Size (number of blocks)')
    plt.ylabel('Diversity Score')
    plt.title('Cluster Diversity vs Size')
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig(output_path / 'diversity_vs_size_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Consensus score vs Size
    plt.figure(figsize=(10, 6))
    plt.scatter(df['size'], df['consensus_score'], alpha=0.7, s=50, color='orange')
    plt.xlabel('Cluster Size (number of blocks)')
    plt.ylabel('Consensus Score')
    plt.title('Consensus Score vs Cluster Size')
    plt.xscale('log')
    plt.savefig(output_path / 'consensus_score_vs_size_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Cluster characteristics heatmap (if clusters > 1)
    if len(df) > 1:
        # Normalize data for heatmap
        heatmap_data = df[['size', 'consensus_length', 'consensus_score', 'diversity']].copy()
        heatmap_data_norm = (heatmap_data - heatmap_data.mean()) / heatmap_data.std()
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(heatmap_data_norm.T, 
                   xticklabels=[f"Cluster {i}" for i in df['cluster_id']], 
                   yticklabels=['Size', 'Consensus Length', 'Consensus Score', 'Diversity'],
                   cmap='RdYlBu_r', center=0, annot=True, fmt='.2f')
        plt.title('Cluster Characteristics (Standardized)')
        plt.tight_layout()
        plt.savefig(output_path / 'cluster_characteristics_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"\nCluster plots saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Analyze ELSA syntenic clusters output')
    parser.add_argument('--input', '-i', default='syntenic_analysis/syntenic_clusters.csv',
                       help='Input syntenic clusters CSV file')
    parser.add_argument('--output', '-o', default='cluster_analysis_output',
                       help='Output directory for results')
    parser.add_argument('--plots', action='store_true', help='Generate visualization plots')
    parser.add_argument('--report', action='store_true', help='Generate detailed cluster report')
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading syntenic clusters from: {args.input}")
    df = load_clusters_data(args.input)
    
    # Perform analyses
    basic_cluster_statistics(df)
    cluster_size_analysis(df)
    diversity_analysis(df)
    representative_analysis(df)
    
    # Optional outputs
    if args.plots:
        create_cluster_plots(df, args.output)
    
    if args.report:
        detailed_cluster_report(df, args.output)

if __name__ == '__main__':
    main()