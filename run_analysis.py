#!/usr/bin/env python3
"""
Master Analysis Runner

Executes all syntenic analysis scripts with appropriate parameters.
Provides a single entry point for comprehensive analysis of ELSA results.
"""

import subprocess
import sys
import argparse
from pathlib import Path
import time

def run_command(cmd, description):
    """Run a command and report results"""
    print(f"\n{'='*60}")
    print(f"RUNNING: {description}")
    print(f"COMMAND: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    end_time = time.time()
    
    if result.returncode == 0:
        print(result.stdout)
        print(f"\n✅ SUCCESS ({end_time - start_time:.1f}s)")
    else:
        print(f"❌ ERROR ({end_time - start_time:.1f}s)")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
    
    return result.returncode == 0

def check_dependencies():
    """Check if required Python packages are available"""
    required_packages = ['pandas', 'matplotlib', 'seaborn', 'numpy', 'ijson']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing required packages: {', '.join(missing_packages)}")
        print("Install with: pip install " + " ".join(missing_packages))
        return False
    
    print("✅ All required packages available")
    return True

def main():
    parser = argparse.ArgumentParser(description='Run comprehensive ELSA syntenic analysis')
    parser.add_argument('--data-dir', default='syntenic_analysis',
                       help='Directory containing syntenic analysis output files')
    parser.add_argument('--output-dir', default='analysis_results',
                       help='Directory for analysis outputs')
    parser.add_argument('--blocks-only', action='store_true',
                       help='Only analyze syntenic blocks (skip clusters and landscape)')
    parser.add_argument('--clusters-only', action='store_true',
                       help='Only analyze syntenic clusters (skip blocks and landscape)')
    parser.add_argument('--landscape-only', action='store_true',
                       help='Only analyze syntenic landscape (skip blocks and clusters)')
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip generating plots (faster execution)')
    parser.add_argument('--max-landscape-blocks', type=int, default=10000,
                       help='Maximum blocks to process for landscape analysis')
    parser.add_argument('--skip-deps-check', action='store_true',
                       help='Skip dependency checking')
    
    args = parser.parse_args()
    
    # Check dependencies
    if not args.skip_deps_check and not check_dependencies():
        sys.exit(1)
    
    # Check input files
    data_path = Path(args.data_dir)
    blocks_file = data_path / 'syntenic_blocks.csv'
    clusters_file = data_path / 'syntenic_clusters.csv'
    landscape_file = data_path / 'syntenic_landscape.json'
    
    if not data_path.exists():
        print(f"❌ Data directory not found: {data_path}")
        sys.exit(1)
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(exist_ok=True)
    
    print(f"🚀 Starting comprehensive ELSA analysis")
    print(f"Data directory: {data_path}")
    print(f"Output directory: {output_path}")
    
    success_count = 0
    total_count = 0
    
    # Syntenic blocks analysis
    if not args.clusters_only and not args.landscape_only:
        if blocks_file.exists():
            total_count += 1
            cmd = [
                sys.executable, 'analyze_syntenic_blocks.py',
                '--input', str(blocks_file),
                '--output', str(output_path / 'blocks_analysis'),
                '--export'
            ]
            if not args.no_plots:
                cmd.append('--plots')
            
            if run_command(cmd, "Syntenic Blocks Analysis"):
                success_count += 1
        else:
            print(f"⚠️  Blocks file not found: {blocks_file}")
    
    # Syntenic clusters analysis
    if not args.blocks_only and not args.landscape_only:
        if clusters_file.exists():
            total_count += 1
            cmd = [
                sys.executable, 'analyze_syntenic_clusters.py',
                '--input', str(clusters_file),
                '--output', str(output_path / 'clusters_analysis'),
                '--report'
            ]
            if not args.no_plots:
                cmd.append('--plots')
            
            if run_command(cmd, "Syntenic Clusters Analysis"):
                success_count += 1
        else:
            print(f"⚠️  Clusters file not found: {clusters_file}")
    
    # Syntenic landscape analysis
    if not args.blocks_only and not args.clusters_only:
        if landscape_file.exists():
            total_count += 1
            cmd = [
                sys.executable, 'analyze_syntenic_landscape.py',
                '--input', str(landscape_file),
                '--output', str(output_path / 'landscape_analysis'),
                '--max-blocks', str(args.max_landscape_blocks),
                '--report'
            ]
            if not args.no_plots:
                cmd.append('--plots')
            
            if run_command(cmd, "Syntenic Landscape Analysis (Streaming)"):
                success_count += 1
        else:
            print(f"⚠️  Landscape file not found: {landscape_file}")
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"✅ Successful analyses: {success_count}/{total_count}")
    
    if success_count > 0:
        print(f"\n📊 Results saved to: {output_path}")
        print(f"\nGenerated outputs:")
        for result_dir in output_path.iterdir():
            if result_dir.is_dir():
                print(f"  📁 {result_dir.name}/")
                for file in sorted(result_dir.iterdir()):
                    if file.is_file():
                        size = file.stat().st_size
                        if size < 1024:
                            size_str = f"{size} B"
                        elif size < 1024*1024:
                            size_str = f"{size/1024:.1f} KB"
                        else:
                            size_str = f"{size/(1024*1024):.1f} MB"
                        print(f"    📄 {file.name} ({size_str})")
    
    if success_count < total_count:
        print(f"\n⚠️  Some analyses failed. Check error messages above.")
        sys.exit(1)

if __name__ == '__main__':
    main()