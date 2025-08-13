#!/usr/bin/env python3
"""
Complete setup script for ELSA Genome Browser.
Sets up database, runs PFAM annotations, and ingests all data.
"""

import argparse
import logging
import sys
from pathlib import Path
import subprocess
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_command(cmd, description, cwd=None):
    """Run a command and handle errors."""
    logger.info(f"Running: {description}")
    logger.info(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, 
                              text=True, cwd=cwd)
        logger.info(f"✓ {description} completed successfully")
        return result
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ {description} failed:")
        logger.error(f"Return code: {e.returncode}")
        logger.error(f"STDOUT: {e.stdout}")
        logger.error(f"STDERR: {e.stderr}")
        raise

def check_requirements():
    """Check if required tools and files are available."""
    logger.info("Checking requirements...")
    
    # Check Python packages
    required_packages = ['streamlit', 'pandas', 'plotly', 'Bio']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"Missing Python packages: {missing_packages}")
        logger.error("Install with: pip install -r requirements.txt")
        return False
    
    # Check astra installation
    try:
        result = subprocess.run(['astra', '--help'], capture_output=True, timeout=10)
        if result.returncode != 0:
            logger.warning("Astra not found or not working properly")
            logger.warning("PFAM annotation will be skipped")
            return "no_astra"
    except (subprocess.TimeoutExpired, FileNotFoundError):
        logger.warning("Astra not found in PATH")
        logger.warning("PFAM annotation will be skipped")
        return "no_astra"
    
    logger.info("✓ All requirements satisfied")
    return True

def setup_database(db_path):
    """Set up the SQLite database."""
    logger.info("Setting up database...")
    
    setup_script = Path(__file__).parent / "database" / "setup_db.py"
    cmd = [sys.executable, str(setup_script), "--db-path", str(db_path), "--force"]
    
    run_command(cmd, "Database setup")
    return True

def run_pfam_annotation(genome_dir, output_dir, threads=4, max_workers=2):
    """Run PFAM annotation pipeline."""
    logger.info("Running PFAM annotation...")
    
    annotation_script = Path(__file__).parent / "annotation" / "pfam_processor.py"
    cmd = [
        sys.executable, str(annotation_script),
        "--protein-dir", str(genome_dir),
        "--output-dir", str(output_dir),
        "--threads", str(threads),
        "--max-workers", str(max_workers)
    ]
    
    try:
        run_command(cmd, "PFAM annotation")
        return output_dir / "pfam_annotation_results.json"
    except subprocess.CalledProcessError:
        logger.warning("PFAM annotation failed - continuing without annotations")
        return None

def ingest_data(db_path, genome_data_paths, blocks_file, clusters_file, pfam_results=None, landscape_file=None):
    """Ingest all data into the database."""
    logger.info("Ingesting data into database...")
    
    ingest_script = Path(__file__).parent / "database" / "populate_db.py"
    cmd = [
        sys.executable, str(ingest_script),
        "--db-path", str(db_path),
        "--sequences-dir", str(genome_data_paths['sequences_dir']),
        "--proteins-dir", str(genome_data_paths['proteins_dir']),
        "--blocks-file", str(blocks_file),
        "--clusters-file", str(clusters_file)
    ]
    
    # Only add annotations-dir if it exists (for legacy format)
    if genome_data_paths.get('annotations_dir'):
        cmd.extend(["--annotations-dir", str(genome_data_paths['annotations_dir'])])
    
    if pfam_results:
        cmd.extend(["--pfam-results", str(pfam_results)])
    
    if landscape_file:
        cmd.extend(["--landscape-file", str(landscape_file)])
    
    run_command(cmd, "Data ingestion")
    return True

def verify_setup(db_path):
    """Verify the setup was successful."""
    logger.info("Verifying setup...")
    
    setup_script = Path(__file__).parent / "database" / "setup_db.py"
    cmd = [sys.executable, str(setup_script), "--db-path", str(db_path), "--info"]
    
    result = run_command(cmd, "Database verification")
    logger.info("Database contents:")
    logger.info(result.stdout)
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Set up ELSA Genome Browser")
    parser.add_argument("--genome-dir", type=Path,
                       help="Directory containing genome files (fna, gff, faa) - legacy format")
    parser.add_argument("--sequences-dir", type=Path,
                       help="Directory containing nucleotide sequences (.fna)")
    parser.add_argument("--proteins-dir", type=Path,
                       help="Directory containing proteins (.faa)")
    parser.add_argument("--blocks-file", type=Path, required=True,
                       help="Syntenic blocks CSV file")
    parser.add_argument("--clusters-file", type=Path, required=True,  
                       help="Syntenic clusters CSV file")
    parser.add_argument("--landscape-file", type=Path,
                       help="Syntenic landscape JSON file with detailed window information")
    parser.add_argument("--db-path", type=Path, default="genome_browser.db",
                       help="SQLite database path")
    parser.add_argument("--pfam-output", type=Path, default="pfam_annotations",
                       help="PFAM annotation output directory")
    parser.add_argument("--threads", type=int, default=4,
                       help="Threads for PFAM annotation")
    parser.add_argument("--max-workers", type=int, default=2,
                       help="Max concurrent PFAM processes")
    parser.add_argument("--skip-pfam", action="store_true",
                       help="Skip PFAM annotation")
    parser.add_argument("--force", action="store_true",
                       help="Force recreation of database")
    
    args = parser.parse_args()
    
    # Handle directory structure - support both old and new formats
    if args.genome_dir:
        # Legacy format: all files in one directory
        genome_data_paths = {
            'sequences_dir': args.genome_dir,
            'annotations_dir': args.genome_dir, 
            'proteins_dir': args.genome_dir
        }
        logger.info("Using legacy directory structure (all files in genome-dir)")
    else:
        # New organized format: separate directories
        if not all([args.sequences_dir, args.proteins_dir]):
            logger.error("Either --genome-dir OR both (--sequences-dir, --proteins-dir) must be provided")
            sys.exit(1)
        
        genome_data_paths = {
            'sequences_dir': args.sequences_dir,
            'annotations_dir': None,  # Not needed anymore
            'proteins_dir': args.proteins_dir
        }
        logger.info("Using organized directory structure")
    
    # Validate directories exist (skip None values)
    for dir_type, dir_path in genome_data_paths.items():
        if dir_path and not dir_path.exists():
            logger.error(f"{dir_type.replace('_', ' ').title()} not found: {dir_path}")
            sys.exit(1)
    
    if not args.blocks_file.exists():
        logger.error(f"Blocks file not found: {args.blocks_file}")
        sys.exit(1)
    
    if not args.clusters_file.exists():
        logger.error(f"Clusters file not found: {args.clusters_file}")
        sys.exit(1)
    
    logger.info("🚀 Starting ELSA Genome Browser setup")
    logger.info(f"Genome directory: {args.genome_dir}")
    logger.info(f"Blocks file: {args.blocks_file}")
    logger.info(f"Clusters file: {args.clusters_file}")
    logger.info(f"Database: {args.db_path}")
    
    try:
        # Check requirements
        req_status = check_requirements()
        if req_status is False:
            sys.exit(1)
        
        # Setup database
        setup_database(args.db_path)
        
        # Run PFAM annotation if requested and available
        pfam_results = None
        if not args.skip_pfam and req_status != "no_astra":
            # First check if ELSA already generated PFAM results
            elsa_pfam_results = Path("../elsa_index/pfam_annotations/pfam_annotation_results.json")
            pfam_results_file = args.pfam_output / "pfam_annotation_results.json"
            
            if elsa_pfam_results.exists():
                logger.info(f"Using existing ELSA PFAM results: {elsa_pfam_results}")
                logger.info("Copying ELSA PFAM annotations to genome browser...")
                
                # Copy the entire PFAM annotation directory
                elsa_pfam_dir = Path("../elsa_index/pfam_annotations")
                if elsa_pfam_dir.exists():
                    import shutil
                    if args.pfam_output.exists():
                        shutil.rmtree(args.pfam_output)
                    shutil.copytree(elsa_pfam_dir, args.pfam_output)
                    pfam_results = args.pfam_output / "pfam_annotation_results.json"
                    logger.info("✓ Successfully copied ELSA PFAM annotations")
                else:
                    logger.warning("ELSA PFAM directory not found, will regenerate")
                    pfam_results = None
            elif pfam_results_file.exists() and not args.force:
                logger.info(f"PFAM annotation results already exist: {pfam_results_file}")
                logger.info("Use --force to regenerate PFAM annotations")
                pfam_results = pfam_results_file
            else:
                if args.force and (pfam_results_file.exists() or elsa_pfam_results.exists()):
                    logger.info("--force specified, regenerating PFAM annotations")
                elif not elsa_pfam_results.exists():
                    logger.info("No existing PFAM results found, generating new annotations")
                
                pfam_results = run_pfam_annotation(
                    genome_data_paths['proteins_dir'], args.pfam_output, 
                    args.threads, args.max_workers
                )
        elif args.skip_pfam:
            logger.info("Skipping PFAM annotation (--skip-pfam flag)")
        else:
            logger.info("Skipping PFAM annotation (astra not available)")
        
        # Ingest all data
        ingest_data(args.db_path, genome_data_paths, args.blocks_file, 
                   args.clusters_file, pfam_results, args.landscape_file)
        
        # Verify setup
        verify_setup(args.db_path)
        
        logger.info("🎉 Setup completed successfully!")
        logger.info(f"\nTo start the genome browser:")
        logger.info(f"  cd {Path(__file__).parent}")
        logger.info(f"  streamlit run app.py")
        
    except Exception as e:
        logger.error(f"❌ Setup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()